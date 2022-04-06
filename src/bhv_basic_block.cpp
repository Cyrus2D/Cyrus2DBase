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

#include <rcsc/action/basic_actions.h>
#include <rcsc/action/body_go_to_point.h>
#include <rcsc//action/body_turn_to_point.h>
#include <rcsc/action/body_intercept.h>
#include <rcsc/action/neck_turn_to_ball_or_scan.h>
#include <rcsc/action/neck_turn_to_low_conf_teammate.h>

#include <rcsc/player/player_agent.h>
#include <rcsc/player/debug_client.h>
#include <rcsc/player/intercept_table.h>

#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

#include "neck_offensive_intercept_neck.h"

#define DEBUG_BLOCK
using namespace rcsc;
int Bhv_BasicBlock::last_block_cycle = -1;
rcsc::Vector2D Bhv_BasicBlock::last_block_pos = Vector2D::INVALIDATED;


bool Bhv_BasicBlock::execute(rcsc::PlayerAgent *agent) {
    auto tm_blockers = get_blockers(agent);
    const WorldModel & wm = agent->world();
    int self_unum = wm.self().unum();
    if (tm_blockers.empty() || std::find(tm_blockers.begin(), tm_blockers.end(), self_unum) == tm_blockers.end()){
        last_block_cycle = -1;
        return false;
    }

    std::pair<int, Vector2D> best_blocker_target = get_best_blocker(wm, tm_blockers);
    if (best_blocker_target.first != self_unum){
        last_block_cycle = -1;
        return false;
    }
    Vector2D target_point = best_blocker_target.second;
    double safe_dist = 2;
    if (wm.self().pos().dist(target_point) > 15)
        safe_dist = 5;
    if (last_block_pos.isValid()
        && last_block_cycle > wm.time().cycle() - 5
        && target_point.dist(last_block_pos) < safe_dist){
        target_point = last_block_pos;
    }

    dlog.addText( Logger::TEAM,
                  __FILE__": Bhv_BasicBlock target=(%.1f %.1f)",
                  target_point.x, target_point.y);

    agent->debugClient().addMessage( "BasicBlock%.0f", 100.0 );
    agent->debugClient().setTarget( target_point );

    if ( ! Body_GoToPoint(target_point,
                          0.5,
                          100,
                          -1,
                          100,
                          false,
                          25,
                          1.0,
                          false).execute( agent ) )
    {
        Body_TurnToPoint(target_point).execute(agent);
    }

    if ( wm.kickableOpponent()
         && wm.ball().distFromSelf() < 18.0 )
    {
        agent->setNeckAction( new Neck_TurnToBall() );
    }
    else
    {
        agent->setNeckAction( new Neck_TurnToBallOrScan( 0 ) );
    }

    return true;
}

bool Bhv_BasicBlock::get_blockers( const rcsc::WorldModel & wm ){
    int opp_min = wm.interceptTable()->opponentReachCycle();
    Vector2D ball_inertia = wm.ball().inertiaPoint(opp_min);
    std::vector<int> tm_blockers;
    for (auto tm: wm.ourPlayers()){
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
        if (tm->unum() <= 5 && ball_inertia.x > -30 && ball_inertia.x > tm_home_pos.x + 5)
            continue;
        #ifdef DEBUG_BLOCK
        dlog.addText(Logger::BLOCK, "- tm %d is add as blocker", tm->unum());
        #endif
        tm_blockers.push_back(tm->unum());
    }
    return tm_blockers;
}

std::pair<int, Vector2D> Bhv_BasicBlock::get_best_blocker( const rcsc::WorldModel & wm, std::vector<int> & tm_blockers ){
    int opp_min = wm.interceptTable()->opponentReachCycle();
    Vector2D ball_inertia = wm.ball().inertiaPoint(opp_min);
    double dribble_speed = 0.7;
    #ifdef DEBUG_BLOCK
    dlog.addText(Logger::BLOCK, "=====get best blocker=====");
    #endif
    for (int cycle = opp_min + 1; cycle <= opp_min + 40; cycle += 1){
        AngleDeg dribble_dir = dribble_direction_detector(ball_inertia);
        ball_inertia += Vector2D::polar2vector(dribble_speed, dribble_dir);
        #ifdef DEBUG_BLOCK
        dlog.addText(Logger::BLOCK, "## id=%d, ball_pos=(%.1f, %.1f)", cycle, ball_inertia.x, ball_inertia.y);
        dlog.addCircle(Logger::BLOCK, ball_inertia, 0.5, 255, 0, 0, false);
        dlog.addMessage(Logger::BLOCK, ball_inertia + Vector2D(0, 1), "%d", cycle);
        #endif
        for (auto & tm_unum: tm_blockers){
            auto tm = wm.ourPlayer(tm_unum);
            Vector2D tm_pos = tm->playerTypePtr()->inertiaPoint(tm_pos, tm->vel(), cycle);
            double dist = ball_inertia.dist(tm_pos);
            int dash_step = tm->playerTypePtr()->cyclesToReachDistance(dist);
            #ifdef DEBUG_BLOCK
            dlog.addText(Logger::BLOCK, "$$$$ tm=%d, block_step=%d", tm->unum(), dash_step);
            #endif
            if (dash_step <= cycle){
                #ifdef DEBUG_BLOCK
                dlog.addCircle(Logger::BLOCK, ball_inertia, 0.5, 0, 0, 255, false);
                dlog.addLine(Logger::BLOCK, ball_inertia, tm_pos);
                dlog.addText(Logger::BLOCK, "=====tm %d can block=====", tm_unum);
                #endif
                return std::make_pair(tm_unum, ball_inertia);
            }
        }
    }
    #ifdef DEBUG_BLOCK
    dlog.addText(Logger::BLOCK, "=====tms can't block=====", tm_unum);
    #endif
    return std::make_pair(0, Vector2D::INVALIDATED);
}

rcsc::AngleDeg Bhv_BasicBlock::dribble_direction_detector ( Vector2D dribble_pos ){
    AngleDeg best_dir(-180);
    double best_score = INT_MIN;
    double dist = 10;
    for (double dir = -180; dir < 180; dir += 10){
        Vector2D target = dribble_pos + Vector2D::polar2vector(dist, AngleDeg(dir));
        if (target.absX() > ServerParam::i().pitchHalfLength())
            continue;
        if (target.absY() > ServerParam::i().pitchHalfWidth())
            continue;
        double score = -target.x + std::max(0, 40 - target.dist(Vector2D(-50, 0)));
        if (score > best_score){
            best_score = score;
            best_dir = AngleDeg(dir);
        }
    }
    return best_dir;
}