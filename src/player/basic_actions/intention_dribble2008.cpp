// -*-c++-*-

/*!
  \file intention_dribble2008.cpp
  \brief queued dribble behavior
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

#include "intention_dribble2008.h"

#include "basic_actions.h"
#include "neck_scan_field.h"
#include "neck_turn_to_goalie_or_scan.h"
#include "view_synch.h"

#include <rcsc/player/audio_sensor.h>
#include <rcsc/player/player_agent.h>
#include <rcsc/player/debug_client.h>
#include <rcsc/player/say_message_builder.h>

#include <rcsc/common/audio_memory.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/geom/line_2d.h>

using namespace rcsc;

#define USE_CHANGE_VIEW

/*-------------------------------------------------------------------*/
/*!

*/
bool
IntentionDribble2008::finished( const PlayerAgent* agent )
{
    if ( M_turn_step + M_dash_step == 0 )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": finished() check finished. empty queue" );
        return true;
    }

    // NOTE: use operator+  operator- has bugs
    if ( M_last_execute_time.cycle() + 1 != agent->world().time().cycle() )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": finished(). last execute time is not match" );
        return true;
    }

    if ( agent->world().ball().pos().dist2( M_target_point ) < 2.0 * 2.0
         && agent->world().self().pos().dist2( M_target_point ) < 2.0 * 2.0 )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": finished(). reached target point" );
        return true;
    }

    if ( agent->world().audioMemory().passRequestTime() == agent->world().time() )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": finished(). heard pass request." );
        return true;
    }

    // playmode is checked in PlayerAgent::parse()
    // and intention queue is totally managed at there.

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": finished(). not finished yet." );

    return false;
}

/*-------------------------------------------------------------------*/
/*!

*/
bool
IntentionDribble2008::execute( PlayerAgent * agent )
{
    if ( M_turn_step + M_dash_step == 0 )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": execute. empty queue." );
        return false;
    }

    const WorldModel & wm = agent->world();

    const Vector2D ball_next = wm.ball().pos() + wm.ball().vel();
    if ( ball_next.absX() > ServerParam::i().pitchHalfLength() - 0.5
         || ball_next.absY() > ServerParam::i().pitchHalfWidth() - 0.5 )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": execute(). ball will be out of pitch. stop intention." );
        return false;
    }

    if ( checkOpponent( wm ) )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": execute(). but exist interfere opponent. cancel intention." );
        return false;
    }


    /*--------------------------------------------------------*/
    // execute action
    if ( M_turn_step > 0 )
    {
        if ( ! doTurn( agent ) )
        {
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": exuecute() failed to. turn. clear intention" );
            this->clear();
            return false;
        }
    }
    else if ( M_dash_step > 0 )
    {
        if ( ! doDash( agent ) )
        {
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": execute() failed to. dash.  clear intention" );
            this->clear();
            return false;
        }
    }
    else
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": execute(). No command queue" );
        this->clear();
        return false;
    }

    if ( wm.gameMode().type() == GameMode::PenaltyTaken_ )
    {
        const AbstractPlayerObject * opp_goalie = wm.getTheirGoalie();
        if ( opp_goalie )
        {
            agent->setNeckAction( new Neck_TurnToPoint( opp_goalie->pos() ) );
        }
        else
        {
            agent->setNeckAction( new Neck_ScanField() );
        }
    }
    else
    {
        if ( wm.self().pos().x > 36.0
             && wm.self().pos().absY() < 20.0 )
        {
            agent->setNeckAction( new Neck_TurnToGoalieOrScan( 2 ) );
        }
        else
        {
            agent->setNeckAction( new Neck_ScanField() );
        }
    }

#ifdef USE_CHANGE_VIEW
    if ( wm.gameMode().type() != GameMode::PlayOn
         || M_turn_step + M_dash_step <= 1 )
    {
        agent->setViewAction( new View_Synch() );
    }
    else
    {
        agent->setViewAction( new View_Normal() );
    }
#endif

    M_last_execute_time = wm.time();

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": execute(). done" );
    agent->debugClient().addMessage( "DribbleQ" );
    agent->debugClient().setTarget( M_target_point );

#if 0
    if ( M_turn_step + M_dash_step > 0
         && agent->config().useCommunication() )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__":  set dribble communication." );
        agent->debugClient().addMessage( "Say_D" );
        agent->addSayMessage( new DribbleMessage( M_target_point,
                                                  M_turn_step + M_dash_step ) );
    }
#endif
    return true;
}

/*-------------------------------------------------------------------*/
/*!

*/
bool
IntentionDribble2008::checkOpponent( const WorldModel & world )
{
    const Vector2D ball_next = world.ball().pos() + world.ball().vel();

    /*--------------------------------------------------------*/
    // exist near opponent goalie in NEXT cycle
    if ( ball_next.x > ServerParam::i().theirPenaltyAreaLineX()
         && ball_next.absY() < ServerParam::i().penaltyAreaHalfWidth() )
    {
        const AbstractPlayerObject * opp_goalie = world.getTheirGoalie();
        if ( opp_goalie
             && opp_goalie->distFromBall() < ( ServerParam::i().catchableArea()
                                               + ServerParam::i().defaultPlayerSpeedMax() )
             )
        {
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": existOpponent(). but exist near opponent goalie" );
            this->clear();
            return true;
        }
    }

    const PlayerObject * nearest_opp = world.getOpponentNearestToSelf( 5 );

    if ( ! nearest_opp )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": existOppnent(). No opponent" );
        return false;
    }

    /*--------------------------------------------------------*/
    // exist very close opponent in CURRENT cycle
    if (  nearest_opp->distFromBall() < ServerParam::i().defaultKickableArea() + 0.2 )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": existOpponent(). but exist kickable opp" );
        this->clear();
        return true;
    }

    /*--------------------------------------------------------*/
    // exist near opponent in NEXT cycle
    const double opp2ball_dist_next = nearest_opp->pos().dist( ball_next );
    if ( opp2ball_dist_next < ( ServerParam::i().defaultPlayerSpeedMax()
                                + ServerParam::i().defaultKickableArea() + 0.3 ) )
    {
        const Vector2D opp_next = nearest_opp->pos() + nearest_opp->vel();
        // oppopnent angle is known
        if ( nearest_opp->bodyCount() == 0
             || nearest_opp->vel().r() > 0.2 )
        {
            Line2D opp_line( opp_next,
                             ( nearest_opp->bodyCount() == 0
                               ? nearest_opp->body()
                               : nearest_opp->vel().th() ) );
            if ( opp_line.dist( ball_next ) > 1.2 )
            {
                // never reach
                dlog.addText( Logger::DRIBBLE,
                              __FILE__": existOpponent(). opp never reach." );
            }
            else if ( opp_next.dist( ball_next ) < 0.6 + 1.2 )
            {
                dlog.addText( Logger::DRIBBLE,
                              __FILE__": existOpponent(). but opp may reachable(1)." );
                this->clear();
                return true;
            }

            dlog.addText( Logger::DRIBBLE,
                          __FILE__": existOpponent(). opp angle is known. opp may not be reached." );
        }
        // opponent angle is not known
        else
        {
            if ( opp_next.dist( ball_next ) < 1.2 + 1.2 ) //0.6 + 1.2 )
            {
                dlog.addText( Logger::DRIBBLE,
                              __FILE__": existOpponent(). but opp may reachable(2)." );
                this->clear();
                return true;
            }
        }

        dlog.addText( Logger::DRIBBLE,
                      __FILE__": existOpponent(). exist near opp. but avoidable?" );
    }

    return false;
}

/*-------------------------------------------------------------------*/
/*!

*/
bool
IntentionDribble2008::doTurn( PlayerAgent * agent )
{
    if ( M_turn_step <=0 )
    {
        return false;
    }

    const WorldModel & wm = agent->world();

    --M_turn_step;

    Vector2D my_final = wm.self().inertiaFinalPoint();
    AngleDeg target_angle = ( M_target_point - my_final ).th();
    AngleDeg angle_diff = target_angle - wm.self().body();
    if ( M_back_dash_mode )
    {
        angle_diff -= 180.0;
    }

    double target_dist = ( M_target_point - my_final ).r();
    double angle_margin
        = std::max( 15.0,
                    std::fabs( AngleDeg::atan2_deg( M_dist_thr,
                                                    target_dist ) ) );

    if ( angle_diff.abs() < angle_margin )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doTurn.  but already facing. diff = %.1f  margin=%.1f",
                      angle_diff.degree(), angle_margin );
        this->clear();
        return false;
    }
    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doTurn. turn to (%.2f, %.2f)",
                  M_target_point.x, M_target_point.y );

    agent->doTurn( angle_diff );

    return true;
}

/*-------------------------------------------------------------------*/
/*!

*/
bool
IntentionDribble2008::doDash( PlayerAgent * agent )
{
    if ( M_dash_step <= 0 )
    {
        return false;
    }

    const WorldModel & wm = agent->world();

    --M_dash_step;

    AngleDeg accel_angle = ( M_back_dash_mode
                             ? wm.self().body() - 180.0
                             : wm.self().body() );
    double used_power
        = wm.self().getSafetyDashPower( M_dash_power_abs
                                        * ( M_back_dash_mode ? -1.0 : 1.0 ) );
    const double max_accel_mag = std::fabs( used_power )
        * wm.self().playerType().dashPowerRate()
        * wm.self().effort();
    double accel_mag = max_accel_mag;
    if ( wm.self().playerType().normalizeAccel( wm.self().vel(),
                                                accel_angle,
                                                &accel_mag ) )
    {
        used_power *=  accel_mag / max_accel_mag;
    }

    Vector2D dash_accel = Vector2D::polar2vector( accel_mag, accel_angle );

    Vector2D my_next = wm.self().pos() + wm.self().vel() + dash_accel;
    Vector2D ball_next = wm.ball().pos() + wm.ball().vel();
    Vector2D ball_next_rel = ( ball_next - my_next ).rotatedVector( - accel_angle );
    double ball_next_dist = ball_next_rel.r();

    if ( ball_next_dist < ( wm.self().playerType().playerSize()
                            + ServerParam::i().ballSize() )//+ 0.1 )
         )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doDash() collision may occur. ball_dist = %f",
                      ball_next_dist );
        this->clear();
        return false;
    }

    if ( ball_next_rel.absY() > wm.self().playerType().kickableArea() - 0.1 )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doDash() next Y difference is over. y_diff = %f",
                      ball_next_rel.absY() );
        this->clear();
        return false;
    }

    // this dash is the last of queue
    // but at next cycle, ball is NOT kickable
    if ( M_dash_step <= 0 )
    {
        if ( ball_next_dist > wm.self().playerType().kickableArea() - 0.15 )
        {
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": doDash() last dash. but not kickable at next. ball_dist=%f",
                          ball_next_dist );
            this->clear();
            return false;
        }
    }

    if ( M_dash_step > 0 )
    {
        // remain dash step. but next dash cause over run.
        AngleDeg ball_next_angle = ( ball_next - my_next ).th();
        if ( ( accel_angle - ball_next_angle ).abs() > 90.0
             && ball_next_dist > wm.self().playerType().kickableArea() - 0.2 )
        {
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": doDash() dash. but run over. ball_dist=%f",
                          ball_next_dist );
            this->clear();
            return false;
        }
    }

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doDash() power=%.1f  accel_mag=%.2f",
                  used_power, accel_mag );
    agent->doDash( used_power );

    return true;
}
