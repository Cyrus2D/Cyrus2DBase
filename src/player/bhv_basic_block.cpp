// -*-c++-*-

/*
    Cyrus2D
    Modified by Nader Zare, Omid Amini.
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif
#include "bhv_basic_block.h"
#include "strategy.h"
#include "bhv_basic_tackle.h"
#include "neck_offensive_intercept_neck.h"


#include "basic_actions/basic_actions.h"
#include "basic_actions/body_go_to_point.h"
#include "basic_actions/body_turn_to_point.h"
#include "basic_actions/neck_turn_to_ball_or_scan.h"
#include <rcsc/player/player_agent.h>
#include <rcsc/player/intercept_table.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include "basic_actions/body_intercept.h"


#define DEBUG_BLOCK
using namespace rcsc;

int Bhv_BasicBlock::last_block_cycle = -1;
Vector2D Bhv_BasicBlock::last_block_pos = Vector2D::INVALIDATED;

bool Bhv_BasicBlock::execute(PlayerAgent *agent)
{
    const WorldModel &wm = agent->world();
    const int self_min = wm.interceptTable().selfStep();
    const int mate_min = wm.interceptTable().teammateStep();
    const int opp_min = wm.interceptTable().opponentStep();
    if ( ! wm.kickableTeammate()
         && ( self_min <= 3
              || ( self_min <= mate_min
                   && self_min < opp_min + 3 ) // pressing
         )
            )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": intercept" );
        Body_Intercept().execute( agent );
        agent->setNeckAction( new Neck_OffensiveInterceptNeck() );

        return true;
    }

    auto tm_blockers = get_blockers(agent);

    int self_unum = wm.self().unum();
    if (tm_blockers.empty() || std::find(tm_blockers.begin(), tm_blockers.end(), self_unum) == tm_blockers.end())
    {
        last_block_cycle = -1;
        return false;
    }

    std::pair<int, Vector2D> best_blocker_target = get_best_blocker(agent, tm_blockers);
    if (best_blocker_target.first != self_unum)
    {
        last_block_cycle = -1;
        return false;
    }
    Vector2D target_point = best_blocker_target.second;
    double safe_dist = 2;
    if (wm.self().pos().dist(target_point) > 15)
        safe_dist = 5;
    if (last_block_pos.isValid() && last_block_cycle > wm.time().cycle() - 5 && target_point.dist(last_block_pos) < safe_dist)
    {
        target_point = last_block_pos;
    }else{
        last_block_cycle = wm.time().cycle();
        last_block_pos = target_point;
    }

    dlog.addText(Logger::TEAM,
                 __FILE__ ": Bhv_BasicBlock target=(%.1f %.1f)",
                 target_point.x, target_point.y);

    agent->debugClient().addMessage("BasicBlock%.0f", 100.0);
    agent->debugClient().setTarget(target_point);

    if (!Body_GoToPoint(target_point,
                        0.5,
                        100,
                        -1,
                        100,
                        false,
                        25,
                        1.0,
                        false)
             .execute(agent))
    {
        Body_TurnToPoint(target_point).execute(agent);
    }

    if (wm.kickableOpponent() && wm.ball().distFromSelf() < 18.0)
    {
        agent->setNeckAction(new Neck_TurnToBall());
    }
    else
    {
        agent->setNeckAction(new Neck_TurnToBallOrScan(0));
    }

    return true;
}

std::vector<int> Bhv_BasicBlock::get_blockers(const PlayerAgent *agent)
{
    const WorldModel &wm = agent->world();
    int opp_min = wm.interceptTable().opponentStep();
    Vector2D ball_inertia = wm.ball().inertiaPoint(opp_min);
    std::vector<int> tm_blockers;
    for (auto tm : wm.ourPlayers())
    {
        if (tm->isGhost())
            continue;
        if (tm->goalie())
            continue;
        if (tm->isTackling())
            continue;
        if (tm->pos().dist(ball_inertia) > 40)
            continue;
        Vector2D tm_home_pos = Strategy::i().getPosition(tm->unum());
        if (tm_home_pos.dist(ball_inertia) > 40)
            continue;
        if (tm->unum() <= 5){
            double tm_defense_line_x = 0;
            for (int i = 2; i <= 5; i++){
                auto defensive_tm = wm.ourPlayer(i);
                if (defensive_tm != nullptr && defensive_tm->unum() > 0){
                    if (defensive_tm->pos().x < tm_defense_line_x){
                        tm_defense_line_x = defensive_tm->pos().x;
                    }
                }
            }
            if (ball_inertia.x > -30 && ball_inertia.x > tm_home_pos.x + 10 && ball_inertia.x > tm_defense_line_x + 10)
                continue;
        }

        tm_blockers.push_back(tm->unum());
    }

    #ifdef DEBUG_BLOCK
    for (auto & blocker: tm_blockers)
        dlog.addText(Logger::BLOCK, "- tm %d is add as blocker", blocker);
    dlog.addText(Logger::BLOCK, "================================");
    #endif
    return tm_blockers;
}

std::pair<int, Vector2D> Bhv_BasicBlock::get_best_blocker(const PlayerAgent *agent, std::vector<int> &tm_blockers)
{
    const WorldModel &wm = agent->world();
    int opp_min = wm.interceptTable().opponentStep();
    Vector2D ball_inertia = wm.ball().inertiaPoint(opp_min);
    double dribble_speed = 0.7;
    #ifdef DEBUG_BLOCK
    dlog.addText(Logger::BLOCK, "=====get best blocker=====");
    #endif
    for (int cycle = opp_min + 1; cycle <= opp_min + 40; cycle += 1)
    {
        AngleDeg dribble_dir = dribble_direction_detector(ball_inertia);
        ball_inertia += Vector2D::polar2vector(dribble_speed, dribble_dir);
        #ifdef DEBUG_BLOCK
        dlog.addText(Logger::BLOCK, "## id=%d, ball_pos=(%.1f, %.1f)", cycle, ball_inertia.x, ball_inertia.y);
        dlog.addCircle(Logger::BLOCK, ball_inertia, 0.5, 255, 0, 0, false);
        char num[8];
        snprintf( num, 8, "%d", cycle );
        dlog.addMessage(Logger::BLOCK, ball_inertia + Vector2D(0, 1), num);
        #endif
        for (auto &tm_unum : tm_blockers)
        {
            auto tm = wm.ourPlayer(tm_unum);
            Vector2D tm_pos = tm->pos() + tm->vel(); //tm->playerTypePtr()->inertiaPoint(tm_pos, tm->vel(), cycle);
            double dist = ball_inertia.dist(tm_pos);
            int dash_step = tm->playerTypePtr()->cyclesToReachDistance(dist);
            if (dash_step <= cycle)
            {
                #ifdef DEBUG_BLOCK
                dlog.addText(Logger::BLOCK, "$$$$ tm=%d, block_step=%d, can block", tm->unum(), dash_step);
                dlog.addCircle(Logger::BLOCK, ball_inertia, 0.5, 0, 0, 255, false);
                dlog.addLine(Logger::BLOCK, ball_inertia, tm_pos);
                #endif
                return std::make_pair(tm_unum, ball_inertia);
            }
            #ifdef DEBUG_BLOCK
            else
            {
                dlog.addText(Logger::BLOCK, "$$$$ tm=%d, block_step=%d, can not block", tm->unum(), dash_step);
            }
            #endif
        }
    }

    return std::make_pair(0, Vector2D::INVALIDATED);
}

AngleDeg Bhv_BasicBlock::dribble_direction_detector(Vector2D dribble_pos)
{
    AngleDeg best_dir(-180);
    double best_score = -1e9;
    double dist = 10;
    for (double dir = -180; dir < 180; dir += 10)
    {
        Vector2D target = dribble_pos + Vector2D::polar2vector(dist, AngleDeg(dir));
        if (target.absX() > ServerParam::i().pitchHalfLength())
            continue;
        if (target.absY() > ServerParam::i().pitchHalfWidth())
            continue;
        double score = -target.x + std::max(0.0, 40 - target.dist(Vector2D(-50, 0)));
        if (score > best_score)
        {
            best_score = score;
            best_dir = AngleDeg(dir);
        }
    }
    return best_dir;
}