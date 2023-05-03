// -*-c++-*-

/*!
  \file body_intercept2009.cpp
  \brief ball chasing action including smart planning.
*/

/*
 *Copyright:

 Copyright (C) Hidehisa AKIYAMA

 This code is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 3 of the License, or (at your option) any later version.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

 *EndCopyright:
 */

/////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "body_intercept2009.h"

#include "basic_actions.h"
#include "body_go_to_point.h"

#include <rcsc/player/intercept_table.h>
#include <rcsc/player/player_agent.h>
#include <rcsc/player/debug_client.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/soccer_math.h>
#include <rcsc/math_util.h>

using namespace rcsc;

// #define DEBUG_PRINT
// #define DEBUG_PRINT_INTERCEPT_LIST

#define USE_GOALIE_MODE

/*-------------------------------------------------------------------*/
/*!

*/
bool
Body_Intercept2009::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::INTERCEPT,
                  __FILE__": Body_Intercept2009" );

    const WorldModel & wm = agent->world();

    /////////////////////////////////////////////
    if ( doKickableOpponentCheck( agent ) )
    {
        return true;;
    }

    const InterceptTable * table = wm.interceptTable();

    /////////////////////////////////////////////
    if ( table->selfReachCycle() > 100 )
    {
        Vector2D final_point = wm.ball().inertiaFinalPoint();
        agent->debugClient().setTarget( final_point );

        dlog.addText( Logger::INTERCEPT,
                      __FILE__": no solution... Just go to ball end point (%.2f %.2f)",
                      final_point.x, final_point.y );
        agent->debugClient().addMessage( "InterceptNoSolution" );
        Body_GoToPoint( final_point,
                        2.0,
                        ServerParam::i().maxDashPower()
                        ).execute( agent );
        return true;
    }

    /////////////////////////////////////////////
    InterceptInfo best_intercept = getBestIntercept( wm, table );
    //InterceptInfo best_intercept_test = getBestIntercept( wm, table );

    dlog.addText( Logger::INTERCEPT,
                  __FILE__": solution size= %d. selected best cycle is %d"
                  " (turn:%d + dash:%d) power=%.1f dir=%.1f",
                  table->selfCache().size(),
                  best_intercept.reachCycle(),
                  best_intercept.turnCycle(), best_intercept.dashCycle(),
                  best_intercept.dashPower(), best_intercept.dashDir() );

    Vector2D target_point = wm.ball().inertiaPoint( best_intercept.reachCycle() );
    agent->debugClient().setTarget( target_point );

    if ( best_intercept.dashCycle() == 0 )
    {
        dlog.addText( Logger::INTERCEPT,
                      __FILE__": can get the ball only by inertia move. Turn!" );

        Vector2D face_point = M_face_point;
        if ( ! face_point.isValid() )
        {
            face_point.assign( 50.5, wm.self().pos().y * 0.75 );
        }

        agent->debugClient().addMessage( "InterceptTurnOnly" );
        Body_TurnToPoint( face_point,
                          best_intercept.reachCycle() ).execute( agent );
        return true;
    }

    /////////////////////////////////////////////
    if ( best_intercept.turnCycle() > 0 )
    {
        Vector2D my_inertia = wm.self().inertiaPoint( best_intercept.reachCycle() );
        AngleDeg target_angle = ( target_point - my_inertia ).th();
        if ( best_intercept.dashPower() < 0.0 )
        {
            // back dash
            target_angle -= 180.0;
        }

        dlog.addText( Logger::INTERCEPT,
                      __FILE__": turn.first.%s target_body_angle = %.1f",
                      ( best_intercept.dashPower() < 0.0 ? "BackMode" : "" ),
                      target_angle.degree() );
        agent->debugClient().addMessage( "InterceptTurn%d(%d/%d)",
                                         best_intercept.reachCycle(),
                                         best_intercept.turnCycle(),
                                         best_intercept.dashCycle() );

        return agent->doTurn( target_angle - wm.self().body() );
    }

    /////////////////////////////////////////////
    dlog.addText( Logger::INTERCEPT,
                  __FILE__": try dash. power=%.1f  target_point=(%.2f, %.2f)",
                  best_intercept.dashPower(),
                  target_point.x, target_point.y );

    if ( doWaitTurn( agent, target_point, best_intercept ) )
    {
        return true;
    }

    if ( M_save_recovery
         && ! wm.self().staminaModel().capacityIsEmpty() )
    {
        double consumed_stamina = best_intercept.dashPower();
        if ( best_intercept.dashPower() < 0.0 ) consumed_stamina *= -2.0;

        if ( wm.self().stamina() - consumed_stamina
             < ServerParam::i().recoverDecThrValue() + 1.0 )
        {
            dlog.addText( Logger::INTERCEPT,
                          __FILE__": insufficient stamina" );
            agent->debugClient().addMessage( "InterceptRecover" );
            agent->doTurn( 0.0 );
            return false;
        }

    }

    return doInertiaDash( agent,
                          target_point,
                          best_intercept );
}

/*-------------------------------------------------------------------*/
/*!

*/
bool
Body_Intercept2009::doKickableOpponentCheck( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();
    if ( wm.ball().distFromSelf() < 2.0
         && wm.kickableOpponent() )
    {
        const PlayerObject * opp = wm.opponentsFromBall().front();
        if ( opp )
        {
            Vector2D goal_pos( -ServerParam::i().pitchHalfLength(), 0.0 );
            Vector2D my_next = wm.self().pos() + wm.self().vel();
            Vector2D attack_pos = opp->pos() + opp->vel();

            if ( attack_pos.dist2( goal_pos ) > my_next.dist2( goal_pos ) )
            {
                dlog.addText( Logger::INTERCEPT,
                              __FILE__": attack to opponent" );

                Body_GoToPoint( attack_pos,
                                0.1,
                                ServerParam::i().maxDashPower(),
                                -1.0, // dash speed
                                1, // cycle
                                true, // save recovery
                                15.0  // dir thr
                                ).execute( agent );
                return true;
            }
        }
    }
    return false;
}

/*-------------------------------------------------------------------*/
/*!

*/
InterceptInfo
Body_Intercept2009::getBestIntercept( const WorldModel & wm,
                                      const InterceptTable * table ) const
{
    const ServerParam & SP = ServerParam::i();
    const std::vector< InterceptInfo > & cache = table->selfCache();

    if ( cache.empty() )
    {
        return InterceptInfo();
    }

#ifdef DEBUG_PRINT
    dlog.addText( Logger::INTERCEPT,
                  "===== getBestIntercept =====");
#endif

    const Vector2D goal_pos( 65.0, 0.0 );
    const Vector2D our_goal_pos( -SP.pitchHalfLength(), 0.0 );
    const double max_pitch_x = ( SP.keepawayMode()
                                 ? SP.keepawayLength() * 0.5 - 1.0
                                 : SP.pitchHalfLength() - 1.0 );
    const double max_pitch_y = ( SP.keepawayMode()
                                 ? SP.keepawayWidth() * 0.5 - 1.0
                                 : SP.pitchHalfWidth() - 1.0 );
    const double penalty_x = SP.ourPenaltyAreaLineX();
    const double penalty_y = SP.penaltyAreaHalfWidth();
    const double speed_max = wm.self().playerType().realSpeedMax() * 0.9;
    const int opp_min = table->opponentReachCycle();
    const int mate_min = table->teammateReachCycle();
    //const PlayerObject * fastest_opponent = table->fastestOpponent();

    const InterceptInfo * attacker_best = nullptr;
    double attacker_score = 0.0;

    const InterceptInfo * forward_best = nullptr;
    double forward_score = 0.0;

    const InterceptInfo * noturn_best = nullptr;
    double noturn_score = 10000.0;

    const InterceptInfo * nearest_best = nullptr;
    double nearest_score = 10000.0;

#ifdef USE_GOALIE_MODE
    const InterceptInfo * goalie_best = nullptr;
    double goalie_score = -10000.0;

    const InterceptInfo * goalie_aggressive_best = nullptr;
    double goalie_aggressive_score = -10000.0;
#endif

    const std::size_t MAX = cache.size();
    for ( std::size_t i = 0; i < MAX; ++i )
    {
        if ( M_save_recovery
             && cache[i].staminaType() != InterceptInfo::NORMAL )
        {
            continue;
        }

        const int cycle = cache[i].reachCycle();
        const Vector2D self_pos = wm.self().inertiaPoint( cycle );
        const Vector2D ball_pos = wm.ball().inertiaPoint( cycle );
        const Vector2D ball_vel = wm.ball().vel() * std::pow( SP.ballDecay(), cycle );

#ifdef DEBUG_PRINT_INTERCEPT_LIST
        dlog.addText( Logger::INTERCEPT,
                      "intercept %d: cycle=%d t=%d d=%d pos=(%.2f %.2f) vel=(%.2f %.1f) trap_ball_dist=%f",
                      i,  cycle, cache[i].turnCycle(), cache[i].dashCycle(),
                      ball_pos.x, ball_pos.y,
                      ball_vel.x, ball_vel.y,
                      cache[i].ballDist() );
#endif

        if ( ball_pos.absX() > max_pitch_x
             || ball_pos.absY() > max_pitch_y )
        {
            continue;
        }

#ifdef USE_GOALIE_MODE
        if ( wm.self().goalie()
             && wm.lastKickerSide() != wm.ourSide()
             && ball_pos.x < penalty_x - 1.0
             && ball_pos.absY() < penalty_y - 1.0
             && cycle < opp_min - 1 )
        {
            if ( ( cache[i].turnCycle() == 0
                   && cache[i].ballDist() < SP.catchableArea() * 0.5 )
                 || cache[i].ballDist() < 0.01 )
            {
                double d = ball_pos.dist2( our_goal_pos );
                if ( d> goalie_score )
                {
                    goalie_score = d;
                    goalie_best = &cache[i];
#ifdef DEBUG_PRINT
                    dlog.addText( Logger::INTERCEPT,
                                  "___ %d updated goalie_best score=%f  trap_ball_dist=%f",
                                  i, goalie_score, cache[i].ballDist() );
#endif
                }
            }
        }

        if ( wm.self().goalie()
             && wm.lastKickerSide() != wm.ourSide()
             && cycle < mate_min - 3
             && cycle < opp_min - 5
             && ( ball_pos.x > penalty_x - 1.0
                  || ball_pos.absY() > penalty_y - 1.0 ) )
        {
            if ( ( cache[i].turnCycle() == 0
                   && cache[i].ballDist() < wm.self().playerType().kickableArea() * 0.5 )
                 || cache[i].ballDist() < 0.01 )
            {
                if ( ball_pos.x > goalie_aggressive_score )
                {
                    goalie_aggressive_score = ball_pos.x;
                    goalie_aggressive_best = &cache[i];
#ifdef DEBUG_PRINT
                    dlog.addText( Logger::INTERCEPT,
                                  "___ %d updated goalie_aggressive_best score=%f  trap_ball_dist=%f",
                                  i, goalie_aggressive_score, cache[i].ballDist() );
#endif
                }
            }
        }
#endif

        bool attacker = false;
        if ( ball_vel.x > 0.5
             && ball_vel.r2() > std::pow( speed_max, 2 )
             && cache[i].dashPower() >= 0.0
             && ball_pos.x < 47.0
             //&& std::fabs( ball_pos.y - wm.self().pos().y ) < 10.0
             && ( ball_pos.x > 35.0
                  || ball_pos.x > wm.offsideLineX() )
             )
        {
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "___ %d attacker", i );
#endif
            attacker = true;
        }

        const double opp_rate = ( attacker ? 0.95 : 0.7 );
#if 0
        if ( attacker
             && opp_min <= cycle - 5
             && ball_vel.r2() > std::pow( 1.2, 2 ) )
        {
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "___ %d attacker ignores opponent cycle=%d: ball_vel=(%.1f %.1f)",
                          i,
                          opp_min,
                          ball_vel.x, ball_vel.y );
#endif
        }
        else
#endif
            if ( cycle >= opp_min * opp_rate )
        {
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "___ %d failed: cycle=%d pos=(%.1f %.1f) turn=%d dash=%d  opp_min=%d rated=%.2f",
                          i, cycle,
                          ball_pos.x, ball_pos.y,
                          cache[i].turnCycle(), cache[i].dashCycle(),
                          opp_min, opp_min * opp_rate );
#endif
            continue;
        }

        // attacker type

        if ( attacker )
        {
            double goal_dist = 100.0 - std::min( 100.0, ball_pos.dist( goal_pos ) );
            double x_diff = 47.0 - ball_pos.x;

            double score
                = ( goal_dist / 100.0 )
                * std::exp( - ( x_diff * x_diff ) / ( 2.0 * 100.0 ) );
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "___ %d attacker cycle=%d pos=(%.1f %.1f) turn=%d dash=%d score=%f",
                          i, cycle,
                          ball_pos.x, ball_pos.y,
                          cache[i].turnCycle(), cache[i].dashCycle(),
                          score );
#endif
            if ( score > attacker_score )
            {
                attacker_best = &cache[i];
                attacker_score = score;
#ifdef DEBUG_PRINT
                dlog.addText( Logger::INTERCEPT,
                              "___ %d updated attacker_best score=%f",
                              i, score );
#endif
            }

            continue;
        }

        // no turn type

        if ( cache[i].turnCycle() == 0 )
        {
            //double score = ball_pos.x;
            //double score = wm.self().pos().dist2( ball_pos );
            double score = cycle;
            //if ( ball_vel.x > 0.0 )
            //{
            //    score *= std::exp( - std::pow( ball_vel.r() - 1.0, 2.0 )
            //                       / ( 2.0 * 1.0 ) );
            //}
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "___ %d noturn cycle=%d pos=(%.1f %.1f) turn=%d dash=%d score=%f",
                          i, cycle,
                          ball_pos.x, ball_pos.y,
                          cache[i].turnCycle(), cache[i].dashCycle(),
                          score );
#endif
            if ( score < noturn_score )
            {
                noturn_best = &cache[i];
                noturn_score = score;
#ifdef DEBUG_PRINT
                dlog.addText( Logger::INTERCEPT,
                              "___ %d updated noturn_best score=%f",
                              i, score );
#endif
            }

            continue;
        }

        // forward type

//         if ( ball_vel.x > 0.5
//              && ball_pos.x > wm.offsideLineX() - 15.0
//              && ball_vel.r() > speed_max * 0.98
//              && cycle <= opp_min - 5 )
        if ( ball_vel.x > 0.1
             && cycle <= opp_min - 5
             && ball_vel.r2() > std::pow( 0.6, 2 ) )
        {
            double score
                = ( 100.0 * 100.0 )
                - std::min( 100.0 * 100.0, ball_pos.dist2( goal_pos ) );
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "___ %d forward cycle=%d pos=(%.1f %.1f) turn=%d dash=%d score=%f",
                          i, cycle,
                          ball_pos.x, ball_pos.y,
                          cache[i].turnCycle(), cache[i].dashCycle(),
                          score );
#endif
            if ( score > forward_score )
            {
                forward_best = &cache[i];
                forward_score = score;
#ifdef DEBUG_PRINT
                dlog.addText( Logger::INTERCEPT,
                              "___ %d updated forward_best score=%f",
                              i, score );
#endif
            }

            continue;
        }

        // other: select nearest one

        {
            //double d = wm.self().pos().dist2( ball_pos );
            double d = self_pos.dist2( ball_pos );
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "___ %d other cycle=%d pos=(%.1f %.1f) turn=%d dash=%d dist2=%.2f",
                          i, cycle,
                          ball_pos.x, ball_pos.y,
                          cache[i].turnCycle(), cache[i].dashCycle(),
                          d );
#endif
            if ( d < nearest_score )
            {
                nearest_best = &cache[i];
                nearest_score = d;
#ifdef DEBUG_PRINT
                dlog.addText( Logger::INTERCEPT,
                              "___ %d updated nearest_best score=%f",
                              i, nearest_score );
#endif
            }
        }

    }

#ifdef USE_GOALIE_MODE
    if ( goalie_aggressive_best )
    {
        dlog.addText( Logger::INTERCEPT,
                      "<--- goalie aggressive_best: cycle=%d(t=%d,d=%d) ball_dist=%.3f score=%f",
                      goalie_aggressive_best->reachCycle(),
                      goalie_aggressive_best->turnCycle(), goalie_aggressive_best->dashCycle(),
                      goalie_aggressive_best->ballDist(),
                      goalie_aggressive_score );
        return *goalie_aggressive_best;
    }

    if ( goalie_best )
    {
        dlog.addText( Logger::INTERCEPT,
                      "<--- goalie best: cycle=%d(t=%d,d=%d) ball_dist=%.3f score=%f",
                      goalie_best->reachCycle(),
                      goalie_best->turnCycle(), goalie_best->dashCycle(),
                      goalie_best->ballDist(),
                      goalie_score );
        return *goalie_best;
    }
#endif

    if ( attacker_best )
    {
        dlog.addText( Logger::INTERCEPT,
                      "<--- attacker best: cycle=%d(t=%d,d=%d) score=%f",
                      attacker_best->reachCycle(),
                      attacker_best->turnCycle(), attacker_best->dashCycle(),
                      attacker_score );

        return *attacker_best;
    }

    if ( noturn_best && forward_best )
    {
        //const Vector2D forward_ball_pos = wm.ball().inertiaPoint( forward_best->reachCycle() );
        //const Vector2D forward_ball_vel
        //    = wm.ball().vel()
        //    * std::pow( SP.ballDecay(), forward_best->reachCycle() );

        if ( forward_best->reachCycle() >= 5 )
        {
            dlog.addText( Logger::INTERCEPT,
                          "<--- forward best(1): cycle=%d(t=%d,d=%d) score=%f",
                          forward_best->reachCycle(),
                          forward_best->turnCycle(), forward_best->dashCycle(),
                          forward_score );
        }

        const Vector2D noturn_ball_vel
            = wm.ball().vel()
            * std::pow( SP.ballDecay(), noturn_best->reachCycle() );
        const double noturn_ball_speed = noturn_ball_vel.r();
        if ( noturn_ball_vel.x > 0.1
             && ( noturn_ball_speed > speed_max
                  || noturn_best->reachCycle() <= forward_best->reachCycle() + 2 )
             )
        {
            dlog.addText( Logger::INTERCEPT,
                              "<--- noturn best(1): cycle=%d(t=%d,d=%d) score=%f",
                          noturn_best->reachCycle(),
                          noturn_best->turnCycle(), noturn_best->dashCycle(),
                          noturn_score );
            return *noturn_best;
        }
    }

    if ( forward_best )
    {
        dlog.addText( Logger::INTERCEPT,
                      "<--- forward best(2): cycle=%d(t=%d,d=%d) score=%f",
                      forward_best->reachCycle(),
                      forward_best->turnCycle(), forward_best->dashCycle(),
                      forward_score );

        return *forward_best;
    }

    const Vector2D fastest_pos = wm.ball().inertiaPoint( cache[0].reachCycle() );
    const Vector2D fastest_vel = wm.ball().vel() * std::pow( SP.ballDecay(),
                                                             cache[0].reachCycle() );
    if ( ( fastest_pos.x > -33.0
           || fastest_pos.absY() > 20.0 )
         && ( cache[0].reachCycle() >= 10
             //|| wm.ball().vel().r() < 1.5 ) )
             || fastest_vel.r() < 1.2 ) )
    {
        dlog.addText( Logger::INTERCEPT,
                      "<--- fastest best: cycle=%d(t=%d,d=%d)",
                      cache[0].reachCycle(),
                      cache[0].turnCycle(), cache[0].dashCycle() );
        return cache[0];
    }

    if ( noturn_best && nearest_best )
    {
        const Vector2D noturn_self_pos = wm.self().inertiaPoint( noturn_best->reachCycle() );
        const Vector2D noturn_ball_pos = wm.ball().inertiaPoint( noturn_best->reachCycle() );
        const Vector2D nearest_self_pos = wm.self().inertiaPoint( nearest_best->reachCycle() );
        const Vector2D nearest_ball_pos = wm.ball().inertiaPoint( nearest_best->reachCycle() );

//         if ( wm.self().pos().dist2( noturn_ball_pos )
//              < wm.self().pos().dist2( nearest_ball_pos ) )
        if ( noturn_self_pos.dist2( noturn_ball_pos )
             < nearest_self_pos.dist2( nearest_ball_pos ) )
        {
            dlog.addText( Logger::INTERCEPT,
                          "<--- noturn best(2): cycle=%d(t=%d,d=%d) score=%f",
                          noturn_best->reachCycle(),
                          noturn_best->turnCycle(), noturn_best->dashCycle(),
                          noturn_score );

            return *noturn_best;
        }

        if ( nearest_best->reachCycle() <= noturn_best->reachCycle() + 2 )
        {
            const Vector2D nearest_ball_vel
                = wm.ball().vel()
                * std::pow( SP.ballDecay(), nearest_best->reachCycle() );
            const double nearest_ball_speed = nearest_ball_vel.r();
            if ( nearest_ball_speed < 0.7 )
            {
                dlog.addText( Logger::INTERCEPT,
                              "<--- nearest best(2): cycle=%d(t=%d,d=%d) score=%f",
                              nearest_best->reachCycle(),
                              nearest_best->turnCycle(), nearest_best->dashCycle(),
                              nearest_score );
                return *nearest_best;
            }

            const Vector2D noturn_ball_vel
                = wm.ball().vel()
                * std::pow( SP.ballDecay(), noturn_best->reachCycle() );

            if ( nearest_best->ballDist() < wm.self().playerType().kickableArea() - 0.4
                 && nearest_best->ballDist() < noturn_best->ballDist()
                 && noturn_ball_vel.x < 0.5
                 && noturn_ball_vel.r2() > std::pow( 1.0, 2 )
                 && noturn_ball_pos.x > nearest_ball_pos.x )
            {
                dlog.addText( Logger::INTERCEPT,
                              "<--- nearest best(3): cycle=%d(t=%d,d=%d) score=%f",
                              nearest_best->reachCycle(),
                              nearest_best->turnCycle(), nearest_best->dashCycle(),
                              nearest_score );
                return *nearest_best;
            }

            Vector2D nearest_self_pos = wm.self().inertiaPoint( nearest_best->reachCycle() );
            if ( nearest_ball_speed > 0.7
                //&& wm.self().pos().dist( nearest_ball_pos ) < wm.self().playerType().kickableArea() )
                 && nearest_self_pos.dist( nearest_ball_pos ) < wm.self().playerType().kickableArea() )
            {
                dlog.addText( Logger::INTERCEPT,
                              "<--- nearest best(4): cycle=%d(t=%d,d=%d) score=%f",
                              nearest_best->reachCycle(),
                              nearest_best->turnCycle(), nearest_best->dashCycle(),
                              nearest_score );
                return *nearest_best;
            }
        }

        dlog.addText( Logger::INTERCEPT,
                          "<--- noturn best(3): cycle=%d(t=%d,d=%d) score=%f",
                      noturn_best->reachCycle(),
                      noturn_best->turnCycle(), noturn_best->dashCycle(),
                      noturn_score );

        return *noturn_best;
    }

    if ( noturn_best )
    {
        dlog.addText( Logger::INTERCEPT,
                      "<--- noturn best only: cycle=%d(t=%d,d=%d) score=%f",
                      noturn_best->reachCycle(),
                      noturn_best->turnCycle(), noturn_best->dashCycle(),
                      noturn_score );

        return *noturn_best;
    }

    if ( nearest_best )
    {
        dlog.addText( Logger::INTERCEPT,
                      "<--- nearest best only: cycle=%d(t=%d,d=%d) score=%f",
                      nearest_best->reachCycle(),
                      nearest_best->turnCycle(), nearest_best->dashCycle(),
                      nearest_score );

        return *nearest_best;
    }



    if ( wm.self().pos().x > 40.0
         && wm.ball().vel().r() > 1.8
         && wm.ball().vel().th().abs() < 100.0
         && cache[0].reachCycle() > 1 )
    {
        const InterceptInfo * chance_best = nullptr;
        for ( std::size_t i = 0; i < MAX; ++i )
        {
            if ( cache[i].reachCycle() <= cache[0].reachCycle() + 3
                 && cache[i].reachCycle() <= opp_min - 2 )
            {
                chance_best = &cache[i];
            }
        }

        if ( chance_best )
        {
            dlog.addText( Logger::INTERCEPT,
                          "<--- chance best only: cycle=%d(t=%d,d=%d)",
                          chance_best->reachCycle(),
                          chance_best->turnCycle(), chance_best->dashCycle() );
            return *chance_best;
        }
    }

    return cache[0];

}

/*-------------------------------------------------------------------*/
/*!

*/
InterceptInfo
Body_Intercept2009::getBestIntercept_Test( const WorldModel & /*wm*/,
                                           const InterceptTable * /*table*/ ) const
{
#if 0
    const std::vector< InterceptInfo > & cache = table->selfCache();

    if ( cache.empty() )
    {
        return InterceptInfo();
    }

    const ServerParam & SP = ServerParam::i();
    const int our_min = table->teammateReachCycle();
    const int opp_min = table->opponentReachCycle();
    const PlayerObject * opponent = table->fastestOpponent();

    const std::size_t MAX = cache.size();

    for ( std::size_t i = 0; i < MAX; ++i )
    {
        const InterceptInfo & info = cache[i];

        if ( M_save_recovery
             && info.mode() != InterceptInfo::NORMAL )
        {
            continue;
        }

        const int reach_cycle = info.reachCycle();
        const Vector2D ball_pos = wm.ball().inertiaPoint( reach_cycle );
        const Vector2D ball_vel = wm.ball().vel() * std::pow( SP.ballDecay(), reach_cycle );



    }


#if 0
    dlog.addText( Logger::INTERCEPT,
                  "____ test best cycle=%d"
                  " (turn:%d + dash:%d) power=%.1f dir=%.1f pos=(%.1f %.1f) stamina=%.1f %.1f",
                  table->selfCache().size(),
                  best_intercept_test.reachCycle(),
                  best_intercept_test.turnCycle(), best_intercept_test.dashCycle(),
                  best_intercept_test.dashPower(), best_intercept_test.dashAngle().degree(),
                  best_intercept_test.selfPos().x, best_intercept_test.selfPos().y,
                  best_intercept_test.stamina() );
#endif
#endif
    return InterceptInfo();
}

/*-------------------------------------------------------------------*/
/*!

*/
bool
Body_Intercept2009::doWaitTurn( PlayerAgent * agent,
                                const Vector2D & target_point,
                                const InterceptInfo & info )
{
    const WorldModel & wm = agent->world();

    {
        const PlayerObject * opp = wm.getOpponentNearestToSelf( 5 );
        if ( opp && opp->distFromSelf() < 3.0 )
        {
            dlog.addText( Logger::INTERCEPT,
                          __FILE__": doWaitTurn. exist near opponent, cancel." );
            return false;
        }

        int opp_min = wm.interceptTable()->opponentReachCycle();
        if ( info.reachCycle() > opp_min - 5 )
        {
            dlog.addText( Logger::INTERCEPT,
                          __FILE__": doWaitTurn. exist opponent intercepter, cancel." );
            return false;
        }
    }

    const Vector2D my_inertia = wm.self().inertiaPoint( info.reachCycle() );
    const Vector2D target_rel = ( target_point - my_inertia ).rotatedVector( - wm.self().body() );
    const double target_dist = target_rel.r();

    const double ball_travel
        = inertia_n_step_distance( wm.ball().vel().r(),
                                   info.reachCycle(),
                                   ServerParam::i().ballDecay() );
    const double ball_noise = ball_travel * ServerParam::i().ballRand();

    if ( info.reachCycle() == 1
         && info.turnCycle() == 1 )
    {
        Vector2D face_point = M_face_point;
        if ( ! face_point.isValid() )
        {
            face_point.assign( 50.5, wm.self().pos().y * 0.9 );
        }
        Body_TurnToPoint( face_point ).execute( agent );
        dlog.addText( Logger::INTERCEPT,
                      __FILE__": doWaitTurn. 1 step inertia_ball_dist=%.2f",
                      target_dist  );
        agent->debugClient().addMessage( "WaitTurn1" );
        return true;
    }

    double extra_buf = 0.1 * bound( 0, info.reachCycle() - 1, 4 );
    {
        double angle_diff = ( wm.ball().vel().th() - wm.self().body() ).abs();
        if ( angle_diff < 10.0
             || 170.0 < angle_diff )
        {
            extra_buf = 0.0;
        }
    }

    double dist_buf = wm.self().playerType().kickableArea() - 0.3 + extra_buf;
    dist_buf -= 0.1 * wm.ball().seenPosCount();

    dlog.addText( Logger::INTERCEPT,
                  __FILE__": doWaitTurn. inertia_ball_dist=%.2f buf=%.2f extra=%.2f ball_noise=%.3f",
                  target_dist,
                  dist_buf, extra_buf, ball_noise );

    if ( target_dist > dist_buf )
    {
        return false;
    }

    Vector2D face_point = M_face_point;
    if ( info.reachCycle() > 2 )
    {
        face_point = my_inertia
            + ( wm.ball().pos() - my_inertia ).rotatedVector( 90.0 );
        if ( face_point.x < my_inertia.x )
        {
            face_point = my_inertia
                + ( wm.ball().pos() - my_inertia ).rotatedVector( -90.0 );
        }
    }

    if ( ! face_point.isValid() )
    {
        face_point.assign( 50.5, wm.self().pos().y * 0.9 );
    }

    Vector2D face_rel = face_point - my_inertia;
    AngleDeg face_angle = face_rel.th();

    Vector2D faced_rel = target_point - my_inertia;
    faced_rel.rotate( face_angle );
    if ( faced_rel.absY() > wm.self().playerType().kickableArea() - ball_noise - 0.2 )
    {
        dlog.addText( Logger::INTERCEPT,
                      __FILE__": doWaitTurn. inertia_y_diff %.2f  ball_noise=%.2f",
                      faced_rel.y, ball_noise );
        return false;
    }

    Body_TurnToPoint( face_point ).execute( agent );
    agent->debugClient().addMessage( "WaitTurn%d", info.reachCycle() );

    return true;
}

/*-------------------------------------------------------------------*/
/*!

*/
bool
Body_Intercept2009::doInertiaDash( PlayerAgent * agent,
                                   const Vector2D & target_point,
                                   const InterceptInfo & info )
{
    const WorldModel & wm = agent->world();
    const PlayerType & ptype = wm.self().playerType();

    if ( info.reachCycle() == 1 )
    {
        agent->debugClient().addMessage( "Intercept1Dash%.0f|%.0f",
                                         info.dashPower(), info.dashDir() );
        agent->doDash( info.dashPower(), info.dashDir() );
#ifdef DEBUG_PRINT
        if ( info.dashAngle().abs() > 1.0 )
        {
            std::cerr << wm.time()
                      << ' ' << wm.self().unum()
                      << " intercept omnidash "
                      << info.dashAngle()
                      << std::endl;
        }
#endif
        return true;
    }

    Vector2D target_rel = target_point - wm.self().pos();
    target_rel.rotate( - wm.self().body() );

    AngleDeg accel_angle = wm.self().body();
    if ( info.dashPower() < 0.0 ) accel_angle += 180.0;

    Vector2D ball_vel = wm.ball().vel() * std::pow( ServerParam::i().ballDecay(),
                                                    info.reachCycle() );

    if ( ( ! wm.self().goalie()
           || wm.lastKickerSide() == wm.ourSide() )
         && wm.self().body().abs() < 50.0 )
    {
        double buf = 0.3;
        if ( info.reachCycle() >= 8 )
        {
            buf = 0.0;
        }
        else if ( target_rel.absY() > wm.self().playerType().kickableArea() - 0.25 )
        {
            buf = 0.0;
        }
        else if ( target_rel.x < 0.0 )
        {
            if ( info.reachCycle() >= 3 ) buf = 0.5;
        }
        else if ( target_rel.x < 0.3 )
        {
            if ( info.reachCycle() >= 3 ) buf = 0.5;
        }
        else if ( target_rel.absY() < 0.5 )
        {
            if ( info.reachCycle() >= 3 ) buf = 0.5;
            if ( info.reachCycle() == 2 ) buf = std::min( target_rel.x, 0.5 );
        }
        else if ( ball_vel.r() < 1.6 )
        {
            buf = 0.4;
        }
        else
        {
            if ( info.reachCycle() >= 4 ) buf = 0.3;
            else if ( info.reachCycle() == 3 ) buf = 0.3;
            else if ( info.reachCycle() == 2 ) buf = std::min( target_rel.x, 0.3 );
        }

        target_rel.x -= buf;

        dlog.addText( Logger::INTERCEPT,
                      __FILE__": doInertiaDash. slightly back to wait. buf=%.3f",
                      buf );
    }

    double used_power = info.dashPower();

    if ( wm.ball().seenPosCount() <= 2
         && wm.ball().vel().r() * std::pow( ServerParam::i().ballDecay(), info.reachCycle() ) < ptype.kickableArea() * 1.5
         && std::fabs( info.dashDir() ) < 5.0
         && target_rel.absX() < ( ptype.kickableArea()
                                  + ptype.dashRate( wm.self().effort() )
                                  * ServerParam::i().maxDashPower()
                                  * 0.8 ) )
    {
        double first_speed
            = calc_first_term_geom_series( target_rel.x,
                                           wm.self().playerType().playerDecay(),
                                           info.reachCycle() );

        first_speed = min_max( - wm.self().playerType().playerSpeedMax(),
                               first_speed,
                               wm.self().playerType().playerSpeedMax() );
        Vector2D rel_vel = wm.self().vel().rotatedVector( - wm.self().body() );
        double required_accel = first_speed - rel_vel.x;
        used_power = required_accel / wm.self().dashRate();
        used_power /= ServerParam::i().dashDirRate( info.dashDir() );

        //if ( info.dashPower() < 0.0 ) used_power = -used_power;

        used_power = ServerParam::i().normalizeDashPower( used_power );
        if ( M_save_recovery )
        {
            used_power = wm.self().getSafetyDashPower( used_power );
        }

        agent->debugClient().addMessage( "InterceptInertiaDash%d:%.0f|%.0f",
                                         info.reachCycle(), used_power, info.dashDir() );
        dlog.addText( Logger::INTERCEPT,
                      __FILE__": doInertiaDash. x_diff=%.2f first_speed=%.2f"
                      " accel=%.2f power=%.1f",
                      target_rel.x, first_speed, required_accel, used_power );

    }
    else
    {
        agent->debugClient().addMessage( "InterceptDash%d:%.0f|%.0f",
                                         info.reachCycle(), used_power, info.dashDir() );
        dlog.addText( Logger::INTERCEPT,
                      __FILE__": doInertiaDash. normal dash. x_diff=%.2f ",
                      target_rel.x );
    }


    if ( info.reachCycle() >= 4
         && ( target_rel.absX() < 0.5
              || std::fabs( used_power ) < 5.0 )
         )
    {
        agent->debugClient().addMessage( "LookBall" );

        Vector2D my_inertia = wm.self().inertiaPoint( info.reachCycle() );
        Vector2D face_point = M_face_point;
        if ( ! M_face_point.isValid() )
        {
            face_point.assign( 50.5, wm.self().pos().y * 0.75 );
        }
        AngleDeg face_angle = ( face_point - my_inertia ).th();

        Vector2D ball_next = wm.ball().pos() + wm.ball().vel();
        AngleDeg ball_angle = ( ball_next - my_inertia ).th();
        double normal_half_width = ViewWidth::width( ViewWidth::NORMAL );

        if ( ( ball_angle - face_angle ).abs()
             > ( ServerParam::i().maxNeckAngle()
                 + normal_half_width
                 - 10.0 )
             )
        {
            face_point.x = my_inertia.x;
            if ( ball_next.y > my_inertia.y + 1.0 ) face_point.y = 50.0;
            else if ( ball_next.y < my_inertia.y - 1.0 ) face_point.y = -50.0;
            else  face_point = ball_next;
            dlog.addText( Logger::INTERCEPT,
                          __FILE__": doInertiaDash. check ball with turn."
                          " face to (%.1f %.1f)",
                          face_point.x, face_point.y );
        }
        else
        {
            dlog.addText( Logger::INTERCEPT,
                          __FILE__": doInertiaDash. can check ball without turn"
                          " face to (%.1f %.1f)",
                          face_point.x, face_point.y );
        }
        Body_TurnToPoint( face_point ).execute( agent );
        return true;
    }

    agent->doDash( used_power, info.dashDir() );
#ifdef DEBUG_PRINT
        if ( info.dashAngle().abs() > 1.0 )
        {
            std::cerr << wm.time()
                      << ' ' << wm.self().unum()
                      << " intercept omnidash "
                      << info.dashAngle()
                      << std::endl;
        }
#endif
    return true;
}
