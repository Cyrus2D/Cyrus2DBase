/*
    Copyright:
    Cyrus2D
    Modified by Aref Sayareh, Nader Zare, Omid Amini
*/

#ifdef HAVE_CONFIG_H

#include <config.h>

#endif

#include "strategy.h"
#include "bhv_unmark.h"
#include "intention_receive.h"
#include "planner/field_analyzer.h"
#include <vector>

#include <rcsc/player/say_message_builder.h>
#include "basic_actions/basic_actions.h"
#include "basic_actions/body_go_to_point.h"
#include "basic_actions/neck_turn_to_ball_or_scan.h"
#include <rcsc/math_util.h>
#include <rcsc/player/player_agent.h>
#include <rcsc/player/intercept_table.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/geom/ray_2d.h>

#include "data_extractor/offensive_data_extractor.h"
#include "data_extractor/DEState.h"

using namespace std;
using namespace rcsc;


// static bool debug = false;
Bhv_Unmark::UnmarkPosition Bhv_Unmark::last_unmark_position = UnmarkPosition();
DeepNueralNetwork * Bhv_Unmark::pass_prediction = new DeepNueralNetwork();


bool Bhv_Unmark::execute(PlayerAgent *agent) {
    const WorldModel &wm = agent->world();
    int passer = find_passer_dnn(wm,agent);
    if (!can_unmarking(wm))
        return false;

    if (last_unmark_position.target.isValid()
        && last_unmark_position.last_run_cycle == wm.time().cycle() - 1
        && last_unmark_position.end_valid_cycle > wm.time().cycle()
        && last_unmark_position.target.x < wm.offsideLineX()) {
        dlog.addText(Logger::POSITIONING, "run last unmarking to (%.1f, %.1f)",
                     last_unmark_position.target.x, last_unmark_position.target.y);
        last_unmark_position.last_run_cycle = wm.time().cycle();
        if (run(agent, last_unmark_position)) {
            agent->debugClient().addMessage("Unmarking to (%.1f, %.1f)", last_unmark_position.target.x,
                                            last_unmark_position.target.y);
            return true;
        }
    }

    // int passer = passer_finder(agent);

    // std::cout << "st_____" <<  wm.time().cycle() << " " << wm.self().unum() << std::endl;

    // DEState state = DEState(wm);
    // int fastest_tm = 0;
    // if (wm.interceptTable()->fastestTeammate() != nullptr)
    //     fastest_tm = wm.interceptTable()->fastestTeammate()->unum();
    // int tm_reach_cycle = wm.interceptTable()->teammateReachCycle();
    // if (fastest_tm != 0 && !state.updateKicker(fastest_tm, wm.ball().inertiaPoint(tm_reach_cycle)))
    //     fastest_tm = 0;
    // if (fastest_tm != 0){
    //     auto features = OffensiveDataExtractor::i().get_data(state);
    //     vector<int> ignored_player;
    //     ignored_player.push_back(5);
    //     auto passes = predict_pass_dnn(features, ignored_player, fastest_tm);
    // }

    // std::cout << wm.time().cycle()  << "ed___ " << wm.self().unum() << " __ "  << passer << std::endl;
    // passer = passer_finder(agent);

    dlog.addText(Logger::POSITIONING, "Should unmarking for %d", passer);
    if (passer == 0)
        return false;
    vector<Bhv_Unmark::UnmarkPosition> unmark_positions;
    simulate_dash(agent, passer, unmark_positions);

    if (unmark_positions.empty())
        return false;

    double max_eval = 0;
    int best = -1; //-1=not 0=last other=other
    for (size_t i = 0; i < unmark_positions.size(); i++) {
        double ev = unmark_positions[i].eval;
        if (ev > max_eval) {
            best = i;
            max_eval = ev;
        }
    }
    if (best == -1)
        return false;

    last_unmark_position = unmark_positions[best];
    last_unmark_position.last_run_cycle = wm.time().cycle();
    last_unmark_position.end_valid_cycle = wm.time().cycle() + 5;
    if (run(agent, unmark_positions[best])) {
        agent->debugClient().addMessage("Unmarking to (%.1f, %.1f)", unmark_positions[best].target.x,
                                        unmark_positions[best].target.y);
        return true;
    }
    return false;
}

bool Bhv_Unmark::can_unmarking(const WorldModel &wm) {
    int mate_min = wm.interceptTable().teammateStep();
    int opp_min = wm.interceptTable().opponentStep();
    int unum = wm.self().unum();
    double stamina = wm.self().stamina();
    double dist2target = Strategy::instance().getPosition(unum).dist(wm.self().pos());
    int min_stamina_limit = 3500;
    if (wm.self().unum() >= 9) {
        if (wm.ball().pos().x > 30)
            min_stamina_limit = 2700;
        else if (wm.ball().pos().x > 10)
            min_stamina_limit = 3500;
        else if (wm.ball().pos().x > -30)
            min_stamina_limit = 5000;
        else if (wm.ball().pos().x > -55)
            min_stamina_limit = 6000;
    } else if (wm.self().unum() >= 6) {
        if (wm.ball().pos().x > 30)
            min_stamina_limit = 3000;
        else if (wm.ball().pos().x > 10)
            min_stamina_limit = 4000;
        else if (wm.ball().pos().x > -30)
            min_stamina_limit = 5000;
        else if (wm.ball().pos().x > -55)
            min_stamina_limit = 6000;
    } else {
        if (wm.ball().pos().x > 30)
            min_stamina_limit = 6000;
        else if (wm.ball().pos().x > 10)
            min_stamina_limit = 4000;
        else if (wm.ball().pos().x > -30)
            min_stamina_limit = 3500;
        else if (wm.ball().pos().x > -55)
            min_stamina_limit = 2500;
    }

    if (opp_min < mate_min || stamina < min_stamina_limit || dist2target > 10) {
        dlog.addText(Logger::POSITIONING,
                     "can not for opp cycle or stamina or dist");
        return false;
    }

    if (opp_min == mate_min)
        if (unum < 6) {
            dlog.addText(Logger::POSITIONING,
                         "can not for opp cycle or stamina or dist def");
            return false;
        }

    if (wm.self().isFrozen()) {
        dlog.addText(Logger::POSITIONING, "can not for frozen");
        return false;
    }
    if (wm.ball().inertiaPoint(mate_min).dist(wm.self().pos()) > 35)
        return false;
    Vector2D home_pos = Strategy::instance().getPosition(wm.self().unum());
    if (wm.ball().inertiaPoint(mate_min).x < 30
        && home_pos.x  > wm.offsideLineX() - 10)
        return false;
    return true;
}

int Bhv_Unmark::passer_finder(rcsc::PlayerAgent *agent) {
    const WorldModel &wm = agent->world();
    auto tm = wm.interceptTable().firstTeammate();
    if (tm != nullptr && tm->unum() > 0)
        return tm->unum();
    return 0;
}

void Bhv_Unmark::simulate_dash(rcsc::PlayerAgent *agent, int tm,
                               vector<Bhv_Unmark::UnmarkPosition> &unmark_positions) {
    const WorldModel &wm = agent->world();
    const AbstractPlayerObject *passer = wm.ourPlayer(tm);
    int mate_min = wm.interceptTable().teammateStep();

    Vector2D ball_pos = wm.ball().inertiaPoint(mate_min);
    Vector2D self_pos = wm.self().inertiaFinalPoint();
    Vector2D home_pos = Strategy::instance().getPosition(wm.self().unum());
    Vector2D passer_pos = passer->pos();
    // Vector2D self_vel = wm.self().vel();
    // AngleDeg self_body = wm.self().body().degree();

    // const PlayerType *self_type = &(wm.self().playerType());
    // double self_max_speed = self_type->realSpeedMax();
    // double self_speed = self_vel.r();
    double offside_lineX = wm.offsideLineX();

    vector<Vector2D> positions;
    if (self_pos.dist(home_pos) < 5){
        for (double dist = 2.0; dist <= 7.0; dist += 1.0){
            for (double angle = -180; angle < 180; angle += 20){
                Vector2D position = self_pos + Vector2D::polar2vector(dist, angle);
                positions.push_back(position);
            }
        }
    }else{
        for (double dist = 3.0; dist <= 8.0; dist += 1){
            double center_angle = (home_pos - self_pos).th().degree();
            for (double angle = -30; angle < 30; angle += 10){
                Vector2D position = self_pos + Vector2D::polar2vector(dist, angle + center_angle);
                positions.push_back(position);
            }
        }
    }
    int position_id = 0;
    for (auto target: positions){
        position_id += 1;
        dlog.addText(Logger::POSITIONING, "# %d ##### (%.1f,%.1f)", position_id, target.x, target.y);
        char num[8];
        snprintf(num, 8, "%d", position_id);
        dlog.addMessage(Logger::POSITIONING, target + Vector2D(0, 0), num);
        if (target.x > offside_lineX) {
            dlog.addCircle(Logger::POSITIONING, target, 0.5, 255, 0, 0);
            dlog.addText(Logger::POSITIONING, "---- more than offside");
            continue;
        }

        double home_max_dist = 7;

        if (target.dist(home_pos) > home_max_dist) {
            dlog.addCircle(Logger::POSITIONING, target, 0.5, 255, 0, 0);
            dlog.addText(Logger::POSITIONING, "---- far to home pos");
            continue;
        }

        double min_tm_dist =
                ServerParam::i().theirPenaltyArea().contains(target) ?
                5 : 8;
        if (nearest_tm_dist_to(wm, target) < min_tm_dist) {
            dlog.addCircle(Logger::POSITIONING, target, 0.5, 255, 0, 0);
            dlog.addText(Logger::POSITIONING, "---- close to tm");
            continue;
        }
        if (target.absX() > 52 || target.absY() > 31.5) {
            dlog.addCircle(Logger::POSITIONING, target, 0.5, 255, 0, 0);
            dlog.addText(Logger::POSITIONING, "---- out of field");
            continue;
        }

        vector<UnmakingPass> passes;
        lead_pass_simulator(wm, passer_pos, target, //0,
         passes);

        if (!passes.empty()) {
            double pos_eval = 0;
            UnmarkPosition new_pos(position_id, ball_pos, target, pos_eval, passes);
            pos_eval = evaluate_position(wm, new_pos);
            new_pos.eval = pos_eval;
            dlog.addCircle(Logger::POSITIONING, target, 0.5, 0, 0, 255);
            dlog.addText(Logger::POSITIONING, "---- OK (%.1f, %.1f) passes: %d eval: %.1f", target.x,
                         target.y, passes.size(), pos_eval);
            unmark_positions.push_back(new_pos);
        } else {
            dlog.addText(Logger::POSITIONING, "---- NOK no pass");
            dlog.addCircle(Logger::POSITIONING, target, 0.5, 0, 0, 0);
        }
    }
}

double Bhv_Unmark::nearest_tm_dist_to(const WorldModel &wm, Vector2D point) {

    double dist = 1000;
    for (auto &tm: wm.teammatesFromSelf()) {
        if (tm != nullptr && tm->unum() > 0) {
            if (!tm->pos().isValid())
                continue;
            if (dist > tm->pos().dist(point))
                dist = tm->pos().dist(point);
        }
    }
    return dist;
}

double passSpeed(double dist_ball_to_unmark_target, double dist_unmark_to_pass_target){
    double pass_speed = 1.5;
    if (dist_ball_to_unmark_target >= 20.0)
        pass_speed = 3.0;
    else if (dist_ball_to_unmark_target >= 8.0)
        pass_speed = 2.6;
    else if (dist_ball_to_unmark_target >= 5.0)
        pass_speed = 2.0;
    if (dist_unmark_to_pass_target < 0.1)
        pass_speed += 0.5;
    pass_speed = std::min(3.0, pass_speed);
    return pass_speed;
}
void Bhv_Unmark::lead_pass_simulator(const WorldModel &wm, Vector2D passer_pos,
                                     Vector2D unmark_target, //int n_step,
                                      vector<UnmakingPass> &passes) {

    int mate_min = wm.interceptTable().teammateStep();
    Vector2D pass_start = wm.ball().inertiaPoint(mate_min);
    // Vector2D current_self_pos = wm.self().pos();

    vector<Vector2D> pass_targets;
    for (double dist = 0; dist <= 3; dist += 3.0){
        for (double angle = -180; angle < 180; angle += 90){
            pass_targets.push_back(unmark_target + Vector2D::polar2vector(dist, angle));
            if (dist == 0)
                break;
        }
    }
    for (auto & pass_target: pass_targets){
        double pass_speed = passSpeed(passer_pos.dist(unmark_target), unmark_target.dist(pass_target));
        int pass_cycle = pass_travel_cycle(pass_start, pass_speed, pass_target);
        int min_opp_cut_cycle = opponents_cycle_intercept(wm, pass_start, pass_speed,
                                                          pass_target, pass_cycle);


        if (pass_cycle < min_opp_cut_cycle) {
            dlog.addText(Logger::POSITIONING,
                         "------pass_start(%.1f,%.1f), pass_target(%.1f,%.1f), self_cycle(%d), opp_cycle(%d) OK",
                         pass_start.x, pass_start.y, pass_target.x, pass_target.y,
                         pass_cycle, min_opp_cut_cycle);
            double pass_eval = pass_target.x + std::max(0.0, 40.0 - pass_target.dist(Vector2D(50.0, 0)));

            UnmakingPass pass_obj = UnmakingPass(pass_target, pass_speed,
                                                 pass_eval, pass_cycle);

            passes.push_back(pass_obj);
            dlog.addCircle(Logger::POSITIONING, pass_target, 0.1, 0, 0, 200);
        } else {
            dlog.addCircle(Logger::POSITIONING, pass_target, 0.1, 255, 0, 0);
            dlog.addText(Logger::POSITIONING,
                         "------pass_start(%.1f,%.1f), pass_target(%.1f,%.1f), self_cycle(%d), opp_cycle(%d) NOT OK",
                         pass_start.x, pass_start.y, pass_target.x, pass_target.y,
                         pass_cycle, min_opp_cut_cycle);
        }
    }

}

int Bhv_Unmark::pass_travel_cycle(Vector2D pass_start, double pass_speed, Vector2D &pass_target) {
    const ServerParam &SP = ServerParam::i();
    double cycle = -(pass_start.dist(pass_target) / pass_speed * ( 1 - SP.ballDecay()) - 1);
    cycle = std::log(cycle) / std::log(SP.ballDecay());
    return static_cast<int>(cycle);
}

int Bhv_Unmark::opponents_cycle_intercept(const WorldModel &wm,
                                          Vector2D pass_start, double pass_speed,
                                          Vector2D pass_target,
                                          int pass_cycle) {
    int min_cycle = 1000;
    for (auto &opp: wm.opponentsFromSelf()) {
        if (opp == nullptr)
            continue;
        int opp_cycle = opponent_cycle_intercept(opp, pass_start, pass_speed, pass_target, pass_cycle);
        if (min_cycle > opp_cycle)
            min_cycle = opp_cycle;
        if (min_cycle <= pass_cycle)
            break;
    }
    return min_cycle;

}

int Bhv_Unmark::opponent_cycle_intercept(const AbstractPlayerObject *opp, Vector2D pass_start, double pass_speed,
                                         Vector2D pass_target, int pass_cycle) {

    const ServerParam &SP = ServerParam::i();

    AngleDeg pass_angle = (pass_target - pass_start).th();

    Vector2D pass_start_vel = Vector2D::polar2vector(pass_speed, pass_angle);
    Vector2D opp_pos = (*opp).pos();

    const PlayerType *opp_type = (*opp).playerTypePtr();

    for (int cycle = 1; cycle <= pass_cycle; cycle++) {
        const Vector2D ball_pos = inertia_n_step_point(pass_start,
                                                       pass_start_vel, cycle, SP.ballDecay());

        double dash_dist = ball_pos.dist(opp_pos);
        dash_dist -= 0.5;

        int n_dash = opp_type->cyclesToReachDistance(dash_dist);
        int n_step = n_dash;
        if (n_step <= cycle) {
            return cycle;
        }
    }
    return 1000;
}

double Bhv_Unmark::evaluate_position(const WorldModel &wm, const UnmarkPosition &unmark_position) {
    double sum_eval = 0;
    double best_pass_eval = 0;
    double opp_eval = 10;
    for (auto &i: unmark_position.pass_list) {
        if (best_pass_eval < i.pass_eval)
            best_pass_eval = i.pass_eval;

        sum_eval += i.pass_eval;
    }

    for (auto &opp: wm.theirPlayers()) {
        if (opp != nullptr && opp->unum() > 0) {
            double opp_dist = opp->pos().dist(unmark_position.target);
            if (opp_dist < opp_eval)
                opp_eval = opp_dist;

        }
    }

    bool have_turn =
            ((unmark_position.target - wm.self().pos()).th() - wm.self().body()).abs() >= 15;
    bool up_pos =
            wm.self().unum() >= 6
            && (unmark_position.target - wm.self().pos()).th().abs() < 60;
    sum_eval /= unmark_position.pass_list.size();
    sum_eval += (sum_eval * unmark_position.pass_list.size() / 10);
    sum_eval += best_pass_eval;
    sum_eval += opp_eval;
    (!have_turn) ? sum_eval += 10 : sum_eval += 0;
    (up_pos) ? sum_eval += 10 : sum_eval += 0;
    return sum_eval;
}

bool Bhv_Unmark::run(PlayerAgent *agent, const UnmarkPosition &unmark_position) {

    const WorldModel &wm = agent->world();
    Vector2D target = unmark_position.target;
    Vector2D ball_pos = unmark_position.ball_pos;
    // Vector2D me = wm.self().pos();
    // Vector2D homePos = Strategy::i().getPosition(wm.self().unum());
    // const int self_min = wm.interceptTable()->selfReachCycle();
    const int mate_min = wm.interceptTable().teammateStep();
    // const int opp_min = wm.interceptTable()->opponentReachCycle();

    double thr = 0.5;
    if (agent->world().self().inertiaPoint(1).dist(unmark_position.target) < thr) {
        AngleDeg bestAngle = (ball_pos - unmark_position.target).th() + 80;
        if (abs(bestAngle.degree()) > 90)
            bestAngle = (ball_pos - unmark_position.target).th() - 80;
        Body_TurnToAngle(bestAngle).execute(agent);
        agent->setNeckAction(new Neck_TurnToBallOrScan(0));
        return true;
    }
    dlog.addCircle(Logger::POSITIONING, target, 0.5, 0, 0, 255, true);
    double dash_power = (
            ball_pos.x > 30 && wm.self().stamina() > 6000 && wm.self().unum() > 6 ?
            100 : Strategy::get_normal_dash_power(agent->world()));
    if (!Body_GoToPoint(unmark_position.target, thr, dash_power).execute(agent)){
        Body_TurnToBall().execute( agent );
    }
    if (mate_min <= 3)
        agent->setNeckAction(new Neck_TurnToBallOrScan(0));
    else
        agent->setNeckAction(new Neck_TurnToBallOrScan(1));
    return true;
}

void Bhv_Unmark::load_dnn(){
    static bool load_dnn = false;
    if(!load_dnn){
        load_dnn = true; 
        pass_prediction->ReadFromKeras("./unmark_dnn_weights.txt");
    }
}

vector<pass_prob> Bhv_Unmark::predict_pass_dnn(vector<double> & features, vector<int> ignored_player, int kicker){
    load_dnn();
    MatrixXd input(290,1); // 290 12
    for (int i = 1; i <= 290; i += 1){
        input(i - 1,0) = features[i];
    }
    pass_prediction->Calculate(input);
    vector<pass_prob> predict;
    for (int i = 0; i < 12; i++){

        if (i == 0){
            dlog.addText(Logger::POSITIONING, "##### Pass from %d to %d : %.6f NOK(0)", kicker, i, pass_prediction->mOutput(i));
        }else if(std::find(ignored_player.begin(), ignored_player.end(), i) == std::end(ignored_player)){
            dlog.addText(Logger::POSITIONING, "##### Pass from %d to %d : %.6f OKKKK", kicker, i, pass_prediction->mOutput(i));
            predict.push_back(pass_prob(pass_prediction->mOutput(i), kicker, i));
        }else{
            dlog.addText(Logger::POSITIONING, "##### Pass from %d to %d : %.6f NOK(ignored)", kicker, i, pass_prediction->mOutput(i));
        }
    }
    std::sort(predict.begin(), predict.end(),pass_prob::ProbCmp);
    return predict;
}


int Bhv_Unmark::find_passer_dnn(const WorldModel & wm, PlayerAgent * agent){
    dlog.addText(Logger::POSITIONING, "############### Start Update Passer DNN ###########");
    DEState state = DEState(wm);

    int fastest_tm = 0;
    if (wm.interceptTable().firstTeammate() != nullptr)
        fastest_tm = wm.interceptTable().firstTeammate()->unum();
    if (fastest_tm < 1)
        return 0;
    int tm_reach_cycle = wm.interceptTable().teammateStep();
    if (!state.updateKicker(fastest_tm, wm.ball().inertiaPoint(tm_reach_cycle)))
        return 0;

    vector<int> ignored_player;
    string ignored = "";
    for (int i = 1; i <= 11; i++){
        if (wm.ourPlayer(i) == nullptr || wm.ourPlayer(i)->unum() < 1 || not wm.ourPlayer(i)->pos().isValid()){
            ignored_player.push_back(i);
            ignored += std::to_string(i) + ",";
        }
    }
    dlog.addText(Logger::POSITIONING, "ignored: %s", ignored.c_str());
    vector<pass_prob> best_passes;
    vector<pass_prob> all_passes;
    all_passes.push_back(pass_prob(100.0, 0, fastest_tm));

    for (int processed_player = 0; processed_player < 6 && all_passes.size() > 0; processed_player++){
        std::sort(all_passes.begin(), all_passes.end(),pass_prob::ProbCmp);
        auto best_pass = all_passes.back();
        all_passes.pop_back();

        dlog.addText(Logger::POSITIONING, "###selected best pass: %d to %d, %.5f", best_pass.pass_sender, best_pass.pass_getter, best_pass.prob);
        if (std::find(ignored_player.begin(), ignored_player.end(), best_pass.pass_getter) != ignored_player.end()){
            dlog.addText(Logger::POSITIONING, "######is in ignored");
            continue;
        }
        if (best_pass.prob < 0.01){
            dlog.addText(Logger::POSITIONING, "######is not valuable");
            continue;
        }

        if (best_pass.pass_sender != 0)
            best_passes.push_back(best_pass);
        ignored_player.push_back(best_pass.pass_getter);

        if (state.updateKicker(best_pass.pass_getter)){
            auto features = OffensiveDataExtractor::i().get_data(state);
            auto passes = predict_pass_dnn(features, ignored_player, best_pass.pass_getter);
            int max_pass = 2;
            for (int p = passes.size() - 1; p >= 0; p--){
                if (max_pass == 0)
                    break;
                all_passes.push_back(passes[p]);
                max_pass -= 1;
            }
        }
    }

    vector<unmark_passer> res;
    for (auto &p : best_passes)
    {
        Vector2D kicker_pos = wm.ourPlayer(p.pass_sender)->pos();
        Vector2D target_pos = wm.ourPlayer(p.pass_getter)->pos();
        dlog.addLine(Logger::POSITIONING,kicker_pos - Vector2D(-0.2, 0), target_pos - Vector2D(-0.2, 0));
        dlog.addLine(Logger::POSITIONING,kicker_pos - Vector2D(-0.1, 0), target_pos - Vector2D(-0.1, 0));
        dlog.addLine(Logger::POSITIONING,kicker_pos, target_pos);
        dlog.addLine(Logger::POSITIONING,kicker_pos - Vector2D(0.2, 0), target_pos - Vector2D(0.2, 0));
        dlog.addLine(Logger::POSITIONING,kicker_pos - Vector2D(0.1, 0), target_pos - Vector2D(0.1, 0));
        dlog.addCircle(Logger::POSITIONING, target_pos, 2);

        if (p.pass_getter == wm.self().unum())
            res.push_back(unmark_passer(p.pass_sender, kicker_pos, wm.interceptTable().opponentStep()));
    }
    if (!res.empty()){
        return res[0].unum;
    }
    return 0;
}
// */