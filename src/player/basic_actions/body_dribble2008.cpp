// -*-c++-*-

/*!
  \file body_dribble2008.cpp
  \brief advanced dribble action. player agent can avoid opponent
  players.
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

#include "body_dribble2008.h"
#include "intention_dribble2008.h"

#include "body_intercept.h"
#include "body_hold_ball.h"

#include "basic_actions.h"
#include "body_kick_to_relative.h"
#include "body_stop_ball.h"
#include "neck_scan_field.h"

#include <rcsc/player/audio_sensor.h>
#include <rcsc/player/player_agent.h>
#include <rcsc/player/debug_client.h>
#include <rcsc/player/say_message_builder.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/geom/ray_2d.h>
#include <rcsc/geom/rect_2d.h>
#include <rcsc/geom/sector_2d.h>
#include <rcsc/soccer_math.h>
#include <rcsc/math_util.h>
#include <rcsc/timer.h>

#include <list>
#include <functional>

using namespace rcsc;

#define USE_CHANGE_VIEW

/*-------------------------------------------------------------------*/
/*!
  execute action
*/
bool
Body_Dribble2008::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::DRIBBLE,
                  __FILE__": Body_Dribble. to(%.1f, %.1f) dash_power=%.1f dash_count=%d",
                  M_target_point.x, M_target_point.y,
                  M_dash_power, M_dash_count );

    if ( ! agent->world().self().isKickable() )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": Body_Dribble. not kickable" );
        return Body_Intercept().execute( agent );
    }

    if ( ! agent->world().ball().velValid() )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": Body_Dribble. invalid ball vel" );
        return Body_StopBall().execute( agent );
    }

    M_dash_power = agent->world().self().getSafetyDashPower( M_dash_power );

    doAction( agent,
              M_target_point,
              M_dash_power,
              M_dash_count,
              M_dodge_mode );  // dodge mode;
    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
void
Body_Dribble2008::say( PlayerAgent * agent,
                       const Vector2D & target_point,
                       const int queue_count )
{
    //if ( agent->config().useCommunication() )
    if ( 0 )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__":  set dribble target communication." );
        agent->debugClient().addMessage( "Say_D" );
        agent->addSayMessage( new DribbleMessage( target_point, queue_count ) );
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Dribble2008::doAction( PlayerAgent * agent,
                            const Vector2D & target_point,
                            const double & dash_power,
                            const int dash_count,
                            const bool dodge,
                            const bool dodge_mode )
{
    // try to create the action queue.
    // kick -> dash -> dash -> ...
    // the number of dash is specified by M_dash_count

    /*--------------------------------------------------------*/
    // do dodge dribble
    if ( dodge
         && isDodgeSituation( agent, target_point ) )
    {
        agent->debugClient().addMessage( "DribDodge" );
        return doDodge( agent, target_point );
    }

    /*--------------------------------------------------------*/
    // normal dribble

    const WorldModel & wm = agent->world();
    const Vector2D my_final_pos= wm.self().inertiaFinalPoint();
    const Vector2D target_rel = target_point - my_final_pos;
    const double target_dist = target_rel.r();

    // already reach the target point
    if ( target_dist < M_dist_thr )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doAction() already there. hold" );
        return Body_HoldBall2008().execute( agent );
    }

    agent->debugClient().setTarget( M_target_point );


    /*--------------------------------------------------------*/
    // decide dribble angle & dist

    if ( doTurn( agent, target_point, dash_power, dash_count, dodge ) )
    {
        return true;
    }

    /*--------------------------------------------------------*/
    // after one dash, ball will kickable
    double used_dash_power = dash_power;
    if ( canKickAfterDash( agent, &used_dash_power ) )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doAction() next kickable. after dash. dash_power=%.1f",
                      used_dash_power );
        return agent->doDash( used_dash_power );
    }

    /*--------------------------------------------------------*/
    // do kick first

    if ( ( ! dodge_mode || dash_count >= 2 ) &&
         doKickDashesWithBall( agent, target_point, dash_power, dash_count,
                               dodge_mode ) )
    {
        return true;
    }

    return doKickDashes( agent, target_point, dash_power, dash_count );
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Dribble2008::doTurn( PlayerAgent * agent,
                          const Vector2D & target_point,
                          const double & dash_power,
                          const int /*dash_count*/,
                          const bool /*dodge*/ )
{
    const WorldModel & wm = agent->world();

    const Vector2D my_final_pos = wm.self().inertiaFinalPoint();
    const Vector2D target_rel = target_point - my_final_pos;
    const AngleDeg target_angle = target_rel.th();
    const double dir_diff
        = ( dash_power > 0.0
            ? ( target_angle - wm.self().body() ).degree()
            : ( target_angle - wm.self().body() - 180.0 ).degree() );
    const double dir_margin_abs
        = std::max( 15.0,
                    std::fabs( AngleDeg::atan2_deg( M_dist_thr, target_rel.r() ) ) );

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doTurn() dir_diff=%.1f dir_margin=%.1f",
                  dir_diff, dir_margin_abs );

    /*--------------------------------------------------------*/
    // already facing to the target
    if ( std::fabs( dir_diff ) < dir_margin_abs )
    {
        return false;
    }

    if ( doTurnOnly( agent, target_point, dash_power,
                     dir_diff ) )
    {
        return true;
    }

    if ( doKickTurnsDash( agent, target_point, dash_power,
                          dir_diff, dir_margin_abs ) )
    {
        return true;
    }

    // just stop the ball

    double kick_power = wm.ball().vel().r() / wm.self().kickRate();
    AngleDeg kick_dir = ( wm.ball().vel().th() - 180.0 ) - wm.self().body();

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doTurn() just stop the ball." );
    agent->doKick( kick_power, kick_dir );
    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Dribble2008::doTurnOnly( PlayerAgent * agent,
                              const Vector2D & /*target_point*/,
                              const double & /*dash_power*/,
                              const double & dir_diff )
{
    const WorldModel & wm = agent->world();

    //---------------------------------------------------------//
    // check opponent
    if ( wm.interceptTable().opponentStep() <= 1 )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doTurnOnly()  exist near opponent" );
        // TODO:
        //   emergent avoidance action
        return false;
    }

    const ServerParam & SP = ServerParam::i();

    const PlayerType & ptype = wm.self().playerType();
    const double my_speed = wm.self().vel().r();
    const double ball_speed = wm.ball().vel().r();

    //---------------------------------------------------------//
    // check next ball dist after turn
    Vector2D my_next = wm.self().pos() + wm.self().vel();
    Vector2D ball_next = wm.ball().pos() + wm.ball().vel();
    const double ball_next_dist = my_next.dist( ball_next );

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doTurnOnly() next_ball_dist=%.2f",
                  ball_next_dist );

    // not kickable at next cycle, if do turn at current cycle.
    if ( ball_next_dist > ( ptype.kickableArea()
                            - ball_speed * SP.ballRand()
                            - my_speed * SP.playerRand()
                            - 0.15 ) )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doTurnOnly  not kickable at next. next_ball_dist=%f",
                      ball_next_dist );
        return false;
    }

    //---------------------------------------------------------//
    // check required turn step.
    //if ( ! wm.self().canTurn( dir_diff ) )
    if ( dir_diff <= ptype.effectiveTurn( SP.maxMoment(), my_speed ) )
    {
        // it is necessary to turn more than one step.
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doTurnOnly() cannot turn by 1 step. angle_diff = %.1f",
                      dir_diff );
        Vector2D my_next2 = wm.self().inertiaPoint( 2 );
        Vector2D ball_next2 = wm.ball().inertiaPoint( 2 );
        double ball_dist_next2 = my_next2.dist( ball_next2 );
        if ( ball_dist_next2 > ( ptype.kickableArea()
                                 - ball_speed * SP.ballRand()
                                 - my_speed * SP.playerRand()
                                 - 0.15 ) )
        {
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": doTurnOnly  not kickable at 2 cycles later. next2_ball_dsit=%f",
                          ball_dist_next2 );
            return false;
        }
    }

    //
    // check opponent
    //
    const PlayerObject * nearest_opp = wm.getOpponentNearestToBall( 5 );
    if ( nearest_opp )
    {
        Vector2D opp_next = nearest_opp->pos() + nearest_opp->vel();
        AngleDeg opp_angle = ( nearest_opp->bodyCount() == 0
                               ? nearest_opp->body()
                               : nearest_opp->vel().th() );
        Line2D opp_line( opp_next, opp_angle );
        if ( opp_line.dist( ball_next ) < 1.1
             && nearest_opp->pos().dist( ball_next ) < 2.0 )
        {
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": doTurnOnly  opponent maybe reach the ball" );
            return false;
        }
    }

    //---------------------------------------------------------//
    // turn only
    agent->debugClient().addMessage( "TurnOnly" );
    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doTurnOnly  done. required_moment = %.1f",
                  dir_diff );

    agent->doTurn( dir_diff );
    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Dribble2008::doCollideWithBall( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    Vector2D required_accel = wm.self().vel(); // target relative pos
    required_accel -= wm.ball().rpos(); // required vel
    required_accel -= wm.ball().vel(); // ball next rpos

    double required_power = required_accel.r()/ wm.self().kickRate();
    if ( required_power > ServerParam::i().maxPower() * 1.1 )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doCollideWithBall.  over max power(%.1f). never collide",
                      required_power );
        return false;
    }


    agent->doKick( std::min( required_power, ServerParam::i().maxPower() ),
                   required_accel.th() - wm.self().body() );
    return true;

}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Dribble2008::doCollideForTurn( PlayerAgent * agent,
                                    const double & dir_diff_abs,
                                    const bool kick_first )
{
    const WorldModel & wm = agent->world();
    double my_speed = wm.self().vel().r();

    if ( kick_first )
    {
        my_speed *= wm.self().playerType().playerDecay();
    }

    const double max_turn_moment
        = wm.self().playerType().effectiveTurn( ServerParam::i().maxMoment(),
                                                my_speed );

    if ( max_turn_moment > dir_diff_abs * 0.9 )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doCollideForTurn.  can face to target by next turn" );
        return false;
    }

    if ( doCollideWithBall( agent ) )
    {
        agent->debugClient().addMessage( "CollideForTurn" );
        return true;
    }

    return false;
}

/*-------------------------------------------------------------------*/
/*!
  if back dash mode, dash_power is negative value.
*/
bool
Body_Dribble2008::doKickTurnsDash( PlayerAgent * agent,
                                   const Vector2D & target_point,
                                   const double & dash_power,
                                   const double & dir_diff,
                                   const double & /*dir_margin_abs*/ )
{
    // try to create these action queue
    // kick -> turn -> turn -> ... -> one dash -> normal dribble kick

    // assume that ball kickable and require to turn to target.

    const WorldModel & wm = agent->world();

    const Vector2D my_final_pos = wm.self().inertiaFinalPoint();
    const Vector2D target_rel = target_point - my_final_pos;
    const AngleDeg target_angle = target_rel.th();

    // simulate kick - turn - dash

    // first step is kick
    double my_speed
        = wm.self().vel().r()
        * wm.self().playerType().playerDecay();
    int n_turn = 0;
    double dir_diff_abs = std::fabs( dir_diff );

    while ( dir_diff_abs > 0.0 )
    {
        // !!! it is necessary to consider about self inertia moment

        double moment_abs = effective_turn( ServerParam::i().maxMoment(),
                                            my_speed,
                                            wm.self().playerType().inertiaMoment() );
        moment_abs = std::min( dir_diff_abs, moment_abs );
        dir_diff_abs -= moment_abs;
        my_speed *= wm.self().playerType().playerDecay();
        ++n_turn;
    }

    if ( n_turn <= 2
         //&& wm.dirCount( target_angle ) <= 5
         && wm.ball().pos().x > 0.0
         && doKickTurnsDashes( agent, target_point, dash_power, n_turn ) )
    {
        return true;
    }

    AngleDeg keep_global_angle;
    bool exist_opp = existCloseOpponent( agent, &keep_global_angle );

    if ( ! exist_opp
         && doCollideForTurn( agent, std::fabs( dir_diff ), true ) )
    {
        // several turns are required after kick.
        // try to collide with ball.
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doKickTurnsDash() no opp. collide with ball" );
        return true;
    }

    const Vector2D my_pos
        = inertia_n_step_point
        ( Vector2D( 0.0, 0.0 ),
          wm.self().vel(),
          1 + n_turn, // kick + turns
          wm.self().playerType().playerDecay() );

    const double control_dist = wm.self().playerType().kickableArea() * 0.7;

    const PlayerObject * nearest_opp = wm.getOpponentNearestToBall( 5 );
    if ( nearest_opp
         && nearest_opp->distFromBall() < 5.0 )
    {
        double best_angle = 0.0;
        double min_dist2 = 100000.0;
        for ( double angle = -90.0; angle <= 91.0; angle += 10.0 )
        {
            Vector2D keep_pos = my_pos + Vector2D::from_polar( control_dist, target_angle + angle );
            double d2 = nearest_opp->pos().dist2( keep_pos );
            if ( d2 < min_dist2 )
            {
                best_angle = angle;
                min_dist2 = d2;
            }
        }
        keep_global_angle = target_angle + best_angle;
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doKickTurnsDash. target_angle = %.0f best_keep_angle = %.0f  rel = %.0f",
                      target_angle.degree(),
                      keep_global_angle.degree(), best_angle );
    }
    else if ( ! exist_opp )
    {
        if ( target_angle.isLeftOf( wm.ball().angleFromSelf() ) )
        {
            keep_global_angle = target_angle + 35.0;
        }
        else
        {
            keep_global_angle = target_angle - 35.0;
        }
    }

    // relative to current my pos, angle is global
    Vector2D required_ball_rel_pos
        = my_pos
        + Vector2D::polar2vector( control_dist, keep_global_angle );


    // travel = firstvel * (1 + dec + dec^2 + ...)
    // firstvel = travel / (1 + dec + dec^2 + ...)
    const double term
        = ( 1.0 - std::pow( ServerParam::i().ballDecay(), n_turn + 2 ) )
        / ( 1.0 - ServerParam::i().ballDecay() );
    const Vector2D required_first_vel
        = ( required_ball_rel_pos - wm.ball().rpos() ) / term;
    const Vector2D required_accel
        = required_first_vel
        - wm.ball().vel();

    // check power overflow
    const double required_kick_power
        = required_accel.r() / wm.self().kickRate();

    // cannot get the required accel using only one kick
    if ( required_kick_power > ServerParam::i().maxPower()
         || required_first_vel.r() > ServerParam::i().ballSpeedMax() )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doKickTurnsDash. kick power over= %.2f or required speed over= %.2f",
                      required_kick_power, required_first_vel.r() );

        Vector2D ball_next
            = wm.self().pos() + wm.self().vel()
            + Vector2D::polar2vector( control_dist, keep_global_angle );
        if ( ball_next.absX() > ServerParam::i().pitchHalfLength() - 0.5
             || ball_next.absY() > ServerParam::i().pitchHalfLength() - 0.5 )
        {
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": doKickTurnsDash. maybe out of pitch. keep_pos=(%.1f %.1f)",
                          ball_next.x, ball_next.y );
            return false;
        }

        agent->debugClient().addMessage( "DribRotPowerOver" );
        return Body_KickToRelative( control_dist,
                                    keep_global_angle - wm.self().body(),
                                    false // not need to stop
                                    ).execute( agent );
    }

    // check collision
    if ( 0 ) // 2008-04-30
    {
        Vector2D tmp_my_pos( 0.0, 0.0 );
        Vector2D my_vel = wm.self().vel();
        Vector2D ball_pos = wm.ball().rpos();
        Vector2D ball_vel = required_first_vel;
        const double collide_dist2
            = square( wm.self().playerType().playerSize()
                      + ServerParam::i().ballSize() );
        for ( int i = 1; i < n_turn + 1; i++ )
        {
            tmp_my_pos += my_vel;
            ball_pos += ball_vel;

            if ( tmp_my_pos.dist2( ball_pos ) < collide_dist2 )
            {
                dlog.addText( Logger::DRIBBLE,
                              __FILE__": doKickTurnsDash. maybe cause collision. keep_angle=%.0f",
                              keep_global_angle.degree() );

                Vector2D ball_next
                    = wm.self().pos() + wm.self().vel()
                    + Vector2D::polar2vector( control_dist, keep_global_angle );
                if ( ball_next.absX() > ServerParam::i().pitchHalfLength() - 0.5
                     || ball_next.absY() > ServerParam::i().pitchHalfLength() - 0.5 )
                {
                    dlog.addText( Logger::DRIBBLE,
                                  __FILE__": doKickTurnsDash. maybe out of pitch. keep_pos=(%.1f %.1f)",
                                  ball_next.x, ball_next.y );
                    return false;
                }

                agent->debugClient().addMessage( "DribRotNoColKick" );
                return Body_KickToRelative( control_dist,
                                            keep_global_angle - wm.self().body(),
                                            false // not need to stop
                                            ).execute( agent );
            }

            my_vel *= wm.self().playerType().playerDecay();
            ball_vel *= ServerParam::i().ballDecay();
        }
    }

    // can archieve required vel

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doKickTurnsDash() kick -> turn[%d]",
                  n_turn );
    agent->debugClient().addMessage( "DribKT%dD", n_turn );

    //////////////////////////////////////////////////////////
    // register intention
    agent->setIntention
        ( new IntentionDribble2008( target_point,
                                    M_dist_thr,
                                    n_turn,
                                    1, // one dash
                                    std::fabs( dash_power ),
                                    ( dash_power < 0.0 ), // back_dash
                                    wm.time() ) );
    say( agent, target_point, n_turn + 1 );
#ifdef USE_CHANGE_VIEW
    if ( wm.gameMode().type() == GameMode::PlayOn
         && n_turn + 1 >= 3 )
    {
        agent->setViewAction( new View_Normal() );
    }
#endif

    // execute first kick
    return agent->doKick( required_kick_power,
                          required_accel.th() - wm.self().body() );
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Dribble2008::doKickTurnsDashes( PlayerAgent * agent,
                                     const Vector2D & target_point,
                                     const double & dash_power,
                                     const int n_turn )
{
    static std::vector< Vector2D > self_cache;

    const int max_dash = 5;

    const WorldModel & wm = agent->world();

    createSelfCache( agent,
                     target_point, dash_power,
                     n_turn, max_dash, self_cache );

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doKickTurnsDashes() target=(%.1f %.1f) dash_power=%.1f n_turn=%d",
                  target_point.x, target_point.y,
                  dash_power,
                  n_turn );

    const double max_moment = ServerParam::i().maxMoment();
    const AngleDeg accel_angle = ( target_point - self_cache[n_turn] ).th();

    const Vector2D trap_rel
        = Vector2D::polar2vector( ( wm.self().playerType().playerSize()
                                    + wm.self().playerType().kickableMargin() * 0.2
                                    + ServerParam::i().ballSize() ),
                                  accel_angle );

    for ( int n_dash = max_dash; n_dash >= 2; --n_dash )
    {
        const Vector2D ball_trap_pos = self_cache[n_turn + n_dash] + trap_rel;

        dlog.addText( Logger::DRIBBLE,
                      "_ n_turn=%d n_dash=%d ball_trap=(%.1f %.1f)",
                      n_turn, n_dash,
                      ball_trap_pos.x, ball_trap_pos.y );

        if ( ball_trap_pos.absX() > ServerParam::i().pitchHalfLength() - 0.5
             || ball_trap_pos.absY() > ServerParam::i().pitchHalfWidth() - 0.5 )
        {
            dlog.addText( Logger::DRIBBLE,
                          "__xx out of pitch" );
            continue;
        }

        const double term
            = ( 1.0 - std::pow( ServerParam::i().ballDecay(), 1 + n_turn + n_dash ) )
            / ( 1.0 - ServerParam::i().ballDecay() );
        const Vector2D first_vel = ( ball_trap_pos - wm.ball().pos() ) / term;
        const Vector2D kick_accel = first_vel - wm.ball().vel();
        const double kick_power = kick_accel.r() / wm.self().kickRate();

        if ( kick_power > ServerParam::i().maxPower()
             || kick_accel.r() > ServerParam::i().ballAccelMax()
             || first_vel.r() > ServerParam::i().ballSpeedMax() )
        {
            dlog.addText( Logger::DRIBBLE,
                          "__xx cannot kick. first_vel=(%.1f %.1f)r%.2f accel=(%.1f %.1f)r%.2f power=%.1f",
                          first_vel.x, first_vel.y, first_vel.r(),
                          kick_accel.x, kick_accel.y, kick_accel.r(),
                          kick_power );
            continue;
        }

        if ( ( wm.ball().pos() + first_vel ).dist( self_cache[0] )
             < wm.self().playerType().playerSize() + ServerParam::i().ballSize() + 0.1 )
        {
            dlog.addText( Logger::DRIBBLE,
                          "__xx maybe collision. first_vel=(%.1f %.1f)r%.2f accel=(%.1f %.1f)r%.2f power=%.1f",
                          first_vel.x, first_vel.y, first_vel.r(),
                          kick_accel.x, kick_accel.y, kick_accel.r(),
                          kick_power );
            continue;
        }

        bool failed = false;

        const int dribble_step = 1 + n_turn + n_dash;

        for ( const PlayerObject * o : wm.opponentsFromSelf() )
        {
            if ( o->distFromSelf() > 30.0 ) break;

            double control_area = o->playerTypePtr()->kickableArea();
            if ( o->goalie()
                 && ball_trap_pos.x > ServerParam::i().theirPenaltyAreaLineX()
                 && ball_trap_pos.absY() < ServerParam::i().penaltyAreaHalfWidth() )
            {
                control_area = ServerParam::i().catchableArea();
            }

            const Vector2D & opos = ( o->seenPosCount() <= o->posCount()
                                      ? o->seenPos()
                                      : o->pos() );
            const int vel_count = std::min( o->seenVelCount(), o->velCount() );
            const Vector2D & ovel = ( o->seenVelCount() <= o->velCount()
                                      ? o->seenVel()
                                      : o->vel() );

            Vector2D opp_pos = ( o->velCount() <= 1
                                 ? inertia_n_step_point( opos, ovel, dribble_step,
                                                         o->playerTypePtr()->playerDecay() )
                                 : opos + ovel );
            Vector2D opp_to_pos = ball_trap_pos - opp_pos;

            double opp_dist = opp_to_pos.r();
            int opp_turn_step = 0;

            if ( o->bodyCount() <= 5
                 || vel_count <= 5 )
            {
                double angle_diff = ( o->bodyCount() <= 1
                                      ? ( opp_to_pos.th() - o->body() ).abs()
                                      : ( opp_to_pos.th() - ovel.th() ).abs() );

                double turn_margin = 180.0;
                if ( control_area < opp_dist )
                {
                    turn_margin = AngleDeg::asin_deg( control_area / opp_dist );
                }
                turn_margin = std::max( turn_margin, 15.0 );

                double opp_speed = ovel.r();
                while ( angle_diff > turn_margin )
                {
                    double max_turn = o->playerTypePtr()->effectiveTurn( max_moment, opp_speed );
                    angle_diff -= max_turn;
                    opp_speed *= o->playerTypePtr()->playerDecay();
                    ++opp_turn_step;
                }
            }

            opp_dist -= control_area;
            opp_dist -= 0.2;
            //opp_dist -= o->distFromSelf() * 0.05;

            if ( opp_dist < 0.0 )
            {
                dlog.addText( Logger::DRIBBLE,
                              "__xx step=%d opponent %d(%.1f %.1f) is already at receive point",
                              dribble_step,
                              o->unum(),
                              o->pos().x, o->pos().y );
                failed = true;
                break;
            }

            int opp_reach_step = o->playerTypePtr()->cyclesToReachDistance( opp_dist );
            opp_reach_step += opp_turn_step;
            opp_reach_step -= bound( 0, o->posCount(), 10 );

            if ( opp_reach_step <= dribble_step )
            {
                dlog.addText( Logger::DRIBBLE,
                              "__xx step=%d opponent %d (%.1f %.1f) can reach faster then self."
                              " opp_step=%d(turn=%d)",
                              dribble_step,
                              o->unum(),
                              o->pos().x, o->pos().y,
                              opp_reach_step,
                              opp_turn_step );
                failed = true;
                break;
            }

            dlog.addText( Logger::DRIBBLE,
                          "__ok step=%d opponent %d (%.1f %.1f)"
                          " opp_step=%d(turn=%d)",
                          dribble_step,
                          o->unum(),
                          o->pos().x, o->pos().y,
                          opp_reach_step,
                          opp_turn_step );
        }

        if ( failed ) continue;

        agent->debugClient().addMessage( "DribKT%dD%d:%.0f",
                                         n_turn, n_dash, dash_power );
        agent->debugClient().addLine( wm.self().pos(), self_cache[n_turn + n_dash] );
        agent->debugClient().addCircle( ball_trap_pos, 0.15 );

        dlog.addText( Logger::DRIBBLE,
                      "<<<<< turn=%d dash=%d. first_vel=(%.1f %.1f) accel=(%.1f %.1f) power=%.1f",
                      n_turn, n_dash,
                      first_vel.x, first_vel.y,
                      kick_accel.x, kick_accel.y,
                      kick_power );

        agent->doKick( kick_power, kick_accel.th() - wm.self().body() );

        agent->setIntention
            ( new IntentionDribble2008( target_point,
                                        M_dist_thr,
                                        n_turn,
                                        n_dash,
                                        std::fabs( dash_power ),
                                        ( dash_power < 0.0 ), // back_dash
                                        wm.time() ) );
        say( agent, target_point, n_dash + n_turn );
#ifdef USE_CHANGE_VIEW
        if ( n_turn + n_dash >= 3 )
        {
            agent->setViewAction( new View_Normal() );
        }
#endif
        return true;
    }

    return false;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Dribble2008::doKickDashes( PlayerAgent * agent,
                                const Vector2D & target_point,
                                const double & dash_power,
                                const int dash_count )
{
    static std::vector< Vector2D > self_cache;

    // do dribble kick. simulate next action queue.
    // kick -> dash -> dash -> ...

    const WorldModel & wm = agent->world();

    ////////////////////////////////////////////////////////
    // simulate my pos after one kick & dashes
    createSelfCache( agent,
                     target_point, dash_power,
                     0, dash_count, // no turn
                     self_cache );

    // my moved position after 1 kick and n dashes
    const Vector2D my_pos = self_cache.back() - wm.self().pos();
    const double my_move_dist = my_pos.r();
    // my move direction
    const AngleDeg my_move_dir = my_pos.th();

    const AngleDeg accel_angle = ( dash_power > 0.0
                                   ? wm.self().body()
                                   : wm.self().body() - 180.0 );

    ////////////////////////////////////////////////////////
    // estimate required kick param
    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doKickDashes() my move dist = %.3f  dir = %.1f  accel_angle=%.1f",
                  my_move_dist, my_move_dir.degree(),
                  accel_angle.degree() );


    // decide next ball control point

    AngleDeg keep_global_angle;
    bool exist_close_opp = existCloseOpponent( agent, &keep_global_angle );

    double control_dist;
    double add_angle_abs;
    {
        double y_dist
            = wm.self().playerType().playerSize()
            + ServerParam::i().ballSize()
            + 0.2;
        Vector2D cur_ball_rel = wm.ball().rpos().rotatedVector( - my_move_dir );
        if ( cur_ball_rel.absY() < y_dist )
        {
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": doKickDashes() y_dist(%.2f) is inner from keep Y(%2.f). correct.",
                          cur_ball_rel.absY(), y_dist );
            //y_dist = ( y_dist + cur_ball_rel.absY() ) * 0.5;
            y_dist += 0.1;
            y_dist = std::min( y_dist, cur_ball_rel.absY() );
        }
        double x_dist
            = std::sqrt( std::pow( wm.self().playerType().kickableArea(), 2 )
                         - std::pow( y_dist, 2 ) )
            - 0.2 - std::min( 0.6, my_move_dist * 0.05 );
        control_dist = std::sqrt( std::pow( x_dist, 2 ) + std::pow( y_dist, 2 ) );
        add_angle_abs = std::fabs( AngleDeg::atan2_deg( y_dist, x_dist ) );
    }

    if ( exist_close_opp )
    {
        //         if ( my_move_dir.isLeftOf( keep_global_angle ) )
        //         {
        //             keep_global_angle = my_move_dir + add_angle_abs;
        //             dlog.addText( Logger::DRIBBLE,
        //                           __FILE__": doKickDashes() avoid. keep right" );
        //         }
        //         else
        //         {
        //             keep_global_angle = my_move_dir - add_angle_abs;
        //             dlog.addText( Logger::DRIBBLE,
        //                           __FILE__": doKickDashes() avoid. keep left" );
        //         }
    }
    else
    {
        if ( my_move_dir.isLeftOf( wm.ball().angleFromSelf() ) )
        {
            keep_global_angle = my_move_dir + add_angle_abs;
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": doKickDashes() keep right."
                          " accel_angle= %.1f < ball_angle=%.1f",
                          accel_angle.degree(),
                          wm.ball().angleFromSelf().degree() );
        }
        else
        {
            keep_global_angle = my_move_dir - add_angle_abs;
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": doKickDashes() keep left."
                          " accel_angle= %.1f > ball_angle=%.1f",
                          accel_angle.degree(),
                          wm.ball().angleFromSelf().degree() );
        }
    }

    const Vector2D next_ball_rel
        = Vector2D::polar2vector( control_dist, keep_global_angle );
    Vector2D next_ctrl_ball_pos = wm.self().pos() + my_pos + next_ball_rel;

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doKickDashes() next_ball_rel=(%.2f, %.2f) global(%.2f %.2f)"
                  " ctrl_dist= %.2f, keep_anggle=%.1f",
                  next_ball_rel.x, next_ball_rel.y,
                  next_ctrl_ball_pos.x, next_ctrl_ball_pos.y,
                  control_dist, keep_global_angle.degree() );

    // calculate required kick param

    // relative to current my pos
    const Vector2D required_ball_pos = my_pos + next_ball_rel;
    const double term
        = ( 1.0 - std::pow( ServerParam::i().ballDecay(), dash_count + 1 ) )
        / ( 1.0 - ServerParam::i().ballDecay() );
    const Vector2D required_first_vel
        = (required_ball_pos - wm.ball().rpos()) / term;
    const Vector2D required_accel
        = required_first_vel
        - wm.ball().vel();
    const double required_kick_power
        = required_accel.r() / wm.self().kickRate();

    ////////////////////////////////////////////////////////
    // never kickable
    if ( required_kick_power > ServerParam::i().maxPower()
         || required_first_vel.r() > ServerParam::i().ballSpeedMax() )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doKickDashes() never reach. rotate." );
        agent->debugClient().addMessage( "DribKDFail" );

        return Body_KickToRelative( wm.self().playerType().kickableArea() * 0.7,
                                    keep_global_angle - wm.self().body(),
                                    false  // not need to stop
                                    ).execute( agent );
    }


    ////////////////////////////////////////////////////////
    // check next collision

    const double collide_dist2
        = std::pow( wm.self().playerType().playerSize()
                    + ServerParam::i().ballSize()
                    + 0.15,
                    2 );
    if ( ( wm.ball().rpos()
           + required_first_vel - wm.self().vel() ).r2() // next rel pos
         < collide_dist2 )
    {
        AngleDeg rotate_global_angle = keep_global_angle;
        if ( ( wm.ball().angleFromSelf() - my_move_dir ).abs() > 90.0 )
        {
            if ( keep_global_angle.isLeftOf( my_move_dir ) )
            {
                rotate_global_angle = my_move_dir + 90.0;
            }
            else
            {
                rotate_global_angle = my_move_dir + 90.0;
            }
        }
        AngleDeg rotate_rel_angle = rotate_global_angle - wm.self().body();
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doKickDashes() maybe collision. rotate. rel_angle=%.1f",
                      rotate_rel_angle.degree() );
        agent->debugClient().addMessage( "DribKDCol" );
        return Body_KickToRelative( wm.self().playerType().kickableArea() * 0.7,
                                    rotate_rel_angle,
                                    false  // not need to stop
                                    ).execute( agent );
    }

    agent->debugClient().addMessage( "DribKD%d:%.0f", dash_count, dash_power );
    agent->debugClient().addLine( wm.self().pos(), wm.self().pos() + my_pos );
    //////////////////////////////////////////////////////////
    // register intention
    agent->setIntention
        ( new IntentionDribble2008( target_point,
                                    M_dist_thr,
                                    0, // zero turn
                                    dash_count,
                                    std::fabs( dash_power ),
                                    ( dash_power < 0.0 ), // back_dash
                                    wm.time() ) );
    say( agent, target_point, dash_count );
#ifdef USE_CHANGE_VIEW
    if ( dash_count >= 2 )
    {
        agent->setViewAction( new View_Normal() );
    }
#endif

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doKickDashes() register intention. dash_count=%d",
                  dash_count );

#if 0
    {
        char msg[16];
        int count = 0;
        int r = 255, g = 0, b = 0;
        Vector2D bpos = wm.ball().pos() + required_first_vel;
        Vector2D bvel = required_first_vel;
        for ( std::vector< Vector2D >::iterator p = self_cache.begin();
              p != self_cache.end();
              ++p, ++count )
        {
            snprintf( msg, 16, "d%d", count );
            dlog.addCircle( Logger::DRIBBLE,
                            *p, 0.1, r, g, b );
            dlog.addMessage( Logger::DRIBBLE,
                             p->x, p->y - 0.1, msg, r, g, b );
            b += 16;

            dlog.addCircle( Logger::DRIBBLE,
                            *p, 0.1, r, 255, b );
            bvel *= ServerParam::i().ballDecay();
            bpos += bvel;
        }
    }

#endif

    // execute first kick
    return agent->doKick( required_kick_power,
                          required_accel.th() - wm.self().body() );
}


/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Dribble2008::doKickDashesWithBall( PlayerAgent * agent,
                                        const Vector2D & target_point,
                                        const double & dash_power,
                                        const int dash_count,
                                        const bool dodge_mode )
{
    static std::vector< Vector2D > my_state;
    static std::vector< KeepDribbleInfo > dribble_info;

    my_state.clear();
    dribble_info.clear();

    // do dribble kick. simulate next action queue.
    // kick -> dash -> dash -> ...
    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doKickDashesWithBall." );

    const WorldModel & wm = agent->world();

    Timer timer;

    // estimate my move positions
    createSelfCache( agent,
                     target_point, dash_power,
                     0, // no turn
                     std::max( 12, dash_count ),
                     my_state );

    const AngleDeg accel_angle = ( dash_power > 0.0
                                   ? wm.self().body()
                                   : wm.self().body() - 180.0 );

    const int DIST_DIVS = 10;

    const double max_dist = wm.self().playerType().kickableArea() + 0.2;
    double first_ball_dist = ( wm.self().playerType().playerSize()
                               + ServerParam::i().ballSize()
                               + 0.15 );
    const double dist_step = ( max_dist - first_ball_dist ) / ( DIST_DIVS - 1 );

    const double angle_range = 240.0;
    const double angle_range_forward = 160.0;
    const double arc_dist_step = 0.1;

    int total_loop_count = 0;

    for ( int dist_loop = 0;
          dist_loop < DIST_DIVS;
          ++dist_loop, first_ball_dist += dist_step )
    {
        const double angle_step
            = ( arc_dist_step * 360.0 )
            / ( 2.0 * first_ball_dist * M_PI );
        const int ANGLE_DIVS
            = ( first_ball_dist < wm.self().playerType().kickableArea() - 0.1
                ? static_cast< int >( std::ceil( angle_range / angle_step ) ) + 1
                : static_cast< int >( std::ceil( angle_range_forward / angle_step ) ) + 1 );

        AngleDeg first_ball_angle = accel_angle - angle_step * ( ANGLE_DIVS/2 );

        // angle loop
        for ( int angle_loop = 0;
              angle_loop < ANGLE_DIVS;
              ++angle_loop, first_ball_angle += angle_step )
        {
            ++total_loop_count;

            const Vector2D first_ball_pos
                = my_state.front()
                + Vector2D::polar2vector( first_ball_dist, first_ball_angle );
            const Vector2D first_ball_vel = first_ball_pos - wm.ball().pos();

            KeepDribbleInfo info;
            if ( simulateKickDashes( wm,
                                     my_state,
                                     dash_count,
                                     accel_angle,
                                     first_ball_pos,
                                     first_ball_vel,
                                     &info ) )
            {
                dribble_info.push_back( info );
                dlog.addText( Logger::DRIBBLE,
                              "_____ add bdist=%.2f bangle=%.1f"
                              " vel=(%.1f %.1f) dash_step=%d opp_dist=%.1f",
                              first_ball_dist,
                              first_ball_angle.degree(),
                              info.first_ball_vel_.x, info.first_ball_vel_.y,
                              info.dash_count_,
                              info.min_opp_dist_ );
            }
        }
    }

    dlog.addText( Logger::DRIBBLE,
                  "___ total loop=%d, solution size=%d, elapsed %.3f [ms]",
                  total_loop_count, dribble_info.size(), timer.elapsedReal() );

    if ( dribble_info.empty() )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doKickDashesWithBall() no solution" );

        return false;
    }

    std::vector< KeepDribbleInfo >::const_iterator dribble
        = std::min_element( dribble_info.begin(),
                            dribble_info.end(),
                            []( const KeepDribbleInfo & lhs, const KeepDribbleInfo & rhs )
                              {
                                  if ( lhs.dash_count_ > rhs.dash_count_ ) return true;
                                  if ( lhs.dash_count_ == rhs.dash_count_ )
                                  {
                                       if ( lhs.min_opp_dist_ > 5.0
                                            && rhs.min_opp_dist_ > 5.0 )
                                       {
                                           return lhs.ball_forward_travel_ > rhs.ball_forward_travel_;
                                           //return lhs.ball_forward_travel_ < rhs.ball_forward_travel_;
                                           ///return lhs.last_ball_rel_.absX() < rhs.last_ball_rel_.absX();
                                       }
                                       return lhs.min_opp_dist_ > rhs.min_opp_dist_;
                                  }
                                  return false;
                              });

    if ( dodge_mode
         && dash_count > dribble->dash_count_ )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doKickDashesWithBall() dodge mode. but not found. required_dash=%d found_dash=%d",
                      dash_count, dribble->dash_count_ );
        return false;
    }

    agent->debugClient().addMessage( "DribKDKeep%d:%.0f",
                                     dribble->dash_count_,
                                     dash_power );

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doKickDashesWithBall() dash_count=%d, ball_vel=(%.1f %.1f) ball_travel_x=%.1f",
                  dribble->dash_count_,
                  dribble->first_ball_vel_.x,
                  dribble->first_ball_vel_.y,
                  dribble->ball_forward_travel_ );

#if 1
    {
        Vector2D ball_pos = wm.ball().pos();
        Vector2D ball_vel = dribble->first_ball_vel_;
        const int DASH = dribble->dash_count_ + 1;
        for ( int i = 0; i < DASH; ++i )
        {
            ball_pos += ball_vel;
            ball_vel *= ServerParam::i().ballDecay();
            dlog.addCircle( Logger::DRIBBLE,
                            ball_pos, 0.05, "#0000FF" );
            dlog.addCircle( Logger::DRIBBLE,
                            my_state[i], wm.self().playerType().kickableArea(), "#FF00FF" );
            //agent->debugClient().addCircle( ball_pos, 0.05 );
            //agent->debugClient().addCircle( my_state[i], wm.self().kickableArea() );
        }
    }
#endif

    Vector2D kick_accel = dribble->first_ball_vel_ - wm.ball().vel();

    // execute first kick
    agent->doKick( kick_accel.r() / wm.self().kickRate(),
                   kick_accel.th() - wm.self().body() );

    //
    // register intention
    //
    agent->setIntention
        ( new IntentionDribble2008( target_point,
                                    M_dist_thr,
                                    0, // zero turn
                                    std::min( dribble->dash_count_, dash_count ),
                                    std::fabs( dash_power ),
                                    ( dash_power < 0.0 ), // back_dash
                                    wm.time() ) );
    say( agent, target_point, std::min( dribble->dash_count_, dash_count ) );
#ifdef USE_CHANGE_VIEW
    if ( std::min( dribble->dash_count_, dash_count ) >= 2 )
    {
        agent->setViewAction( new View_Normal() );
    }
#endif

    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
void
Body_Dribble2008::createSelfCache( PlayerAgent * agent,
                                   const Vector2D & target_point,
                                   const double & dash_power,
                                   const int turn_count,
                                   const int dash_count,
                                   std::vector< Vector2D > & self_cache )
{
    const WorldModel & wm = agent->world();

    self_cache.clear();
    self_cache.reserve( turn_count + dash_count + 1 );

    StaminaModel stamina_model = wm.self().staminaModel();

    Vector2D my_pos = wm.self().pos();
    Vector2D my_vel = wm.self().vel();

    my_pos += my_vel;
    my_vel *= wm.self().playerType().playerDecay();

    self_cache.push_back( my_pos ); // first element is next cycle just after kick

    for ( int i = 0; i < turn_count; ++i )
    {
        my_pos += my_vel;
        my_vel *= wm.self().playerType().playerDecay();
        self_cache.push_back( my_pos );
    }

    stamina_model.simulateWaits( wm.self().playerType(), 1 + turn_count );

    AngleDeg accel_angle;
    if ( turn_count == 0 )
    {
        accel_angle = ( dash_power > 0.0
                        ? wm.self().body()
                        : wm.self().body() - 180.0 );
    }
    else
    {
        accel_angle = ( target_point - wm.self().inertiaFinalPoint() ).th();
    }

    for ( int i = 0; i < dash_count; ++i )
    {
        double available_stamina
            =  std::max( 0.0,
                         stamina_model.stamina()
                         - ServerParam::i().recoverDecThrValue()
                         - 300.0 );
        double consumed_stamina = ( dash_power > 0.0
                                    ? dash_power
                                    : dash_power * -2.0 );
        consumed_stamina = std::min( available_stamina,
                                     consumed_stamina );
        double used_power = ( dash_power > 0.0
                              ? consumed_stamina
                              : consumed_stamina * -0.5 );
        double max_accel_mag = ( std::fabs( used_power )
                                 * wm.self().playerType().dashPowerRate()
                                 * stamina_model.effort() );
        double accel_mag = max_accel_mag;
        if ( wm.self().playerType().normalizeAccel( my_vel,
                                                    accel_angle,
                                                    &accel_mag ) )
        {
            used_power *= accel_mag / max_accel_mag;
        }

        Vector2D dash_accel
            = Vector2D::polar2vector( std::fabs( used_power )
                                      * stamina_model.effort()
                                      * wm.self().playerType().dashPowerRate(),
                                      accel_angle );
        my_vel += dash_accel;
        my_pos += my_vel;

        self_cache.push_back( my_pos );

        my_vel *= wm.self().playerType().playerDecay();

        stamina_model.simulateDash( wm.self().playerType(), used_power );
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Dribble2008::simulateKickDashes( const WorldModel & wm,
                                      const std::vector< Vector2D > & self_cache,
                                      const int dash_count,
                                      const AngleDeg & accel_angle,
                                      const Vector2D & first_ball_pos,
                                      const Vector2D & first_ball_vel,
                                      KeepDribbleInfo * dribble_info )
{
    static const Rect2D pitch_rect( Vector2D( - ServerParam::i().pitchHalfLength() + 0.2,
                                              - ServerParam::i().pitchHalfWidth() + 0.2 ),
                                    Size2D( ServerParam::i().pitchLength() - 0.4,
                                            ServerParam::i().pitchWidth() - 0.4 ) );

    if ( ! pitch_rect.contains( first_ball_pos ) )
    {
        return false;
    }

    const ServerParam & param = ServerParam::i();
    const double collide_dist = ( wm.self().playerType().playerSize()
                                  + param.ballSize() );
    const double kickable_area = wm.self().playerType().kickableArea();

    const Vector2D first_ball_accel = first_ball_vel - wm.ball().vel();
    const double first_ball_accel_r = first_ball_accel.r();

    if ( first_ball_vel.r() > param.ballSpeedMax()
         || first_ball_accel_r > param.ballAccelMax()
         || ( first_ball_accel_r > wm.self().kickRate() * param.maxPower() )
         )
    {
        // cannot acccelerate to the desired speed
        return false;
    }

    double min_opp_dist = 1000.0;

    if ( existKickableOpponent( wm, first_ball_pos, &min_opp_dist ) )
    {
        return false;
    }

    Vector2D ball_pos = first_ball_pos;
    Vector2D ball_vel = first_ball_vel;
    ball_vel *= param.ballDecay();

    int tmp_dash_count = 0;
    Vector2D total_ball_move( 0.0, 0.0 );
    Vector2D last_ball_rel( 0.0, 0.0 );

    // future state loop
    for ( std::vector< Vector2D >::const_iterator my_pos = self_cache.begin() + 1, end = self_cache.end();
          my_pos != end;
          ++my_pos )
    {
        ball_pos += ball_vel;

        // out of pitch
        if ( ! pitch_rect.contains( ball_pos ) ) break;

        const Vector2D ball_rel = ( ball_pos - *my_pos ).rotatedVector( - accel_angle );
        const double new_ball_dist = ball_rel.r();

        const double ball_travel = ball_pos.dist( wm.ball().pos() );
        const double my_travel = my_pos->dist( wm.self().pos() );

        // check collision
        //double dist_buf = std::min( 0.01 * ball_travel + 0.02 * my_travel,
        //                            0.3 );
        //if ( new_ball_dist < collide_dist - dist_buf + 0.1 ) break;
        //double dist_buf = std::min( 0.02 * ball_travel + 0.03 * my_travel,
        //                            0.3 );
        //if ( new_ball_dist < collide_dist - dist_buf + 0.15 ) break;
        //double dist_buf = std::min( 0.02 * ball_travel + 0.03 * my_travel,
        //                            0.1 );
        //if ( new_ball_dist < collide_dist - dist_buf + 0.15 ) break;
        double dist_buf = std::min( 0.02 * ball_travel + 0.03 * my_travel,
                                    0.1 );
        if ( new_ball_dist < collide_dist - dist_buf + 0.2 ) break;

        // check kickable

        if ( tmp_dash_count == dash_count - 1
             && ball_rel.x > 0.0
             && new_ball_dist > kickable_area - 0.25 ) break;
        //         if ( tmp_dash_count >= dash_count
        //              && ball_rel.x > 0.0
        //              && new_ball_dist > kickable_area - 0.25 ) break;

        //dist_buf = std::min( 0.025 * ball_travel + 0.05 * my_travel,
        //                     0.2 );
        //if ( new_ball_dist > kickable_area + dist_buf ) break;
        //if ( new_ball_dist > kickable_area - 0.2 * std::pow( 0.8, tmp_dash_count ) ) break;
        if ( new_ball_dist > kickable_area - 0.2 ) break;

        // front x buffer
        dist_buf = std::min( 0.02 * ball_travel + 0.04 * my_travel,
                             0.2 );
        if ( ball_rel.x > kickable_area - dist_buf - 0.2 ) break;

        // side y buffer
        //dist_buf = std::min( 0.02 * ball_travel + 0.04 + my_travel,
        //                     0.2 );
        //if ( ball_rel.absY() > kickable_area - dist_buf - 0.25 ) break;
        dist_buf = std::min( 0.02 * ball_travel + 0.055 + my_travel,
                             0.35 );
        if ( ball_rel.absY() > kickable_area - dist_buf - 0.15 ) break;

        // check opponent kickable possibility
        if ( existKickableOpponent( wm, ball_pos, &min_opp_dist ) )
        {
            break;
        }

        total_ball_move = ball_pos - wm.ball().pos();
        ++tmp_dash_count;
        last_ball_rel = ball_rel;
        ball_vel *= param.ballDecay();
    }

    if ( tmp_dash_count > 0 )
    {
        dribble_info->first_ball_vel_ = first_ball_vel;
        dribble_info->last_ball_rel_ = last_ball_rel;
        dribble_info->ball_forward_travel_ = total_ball_move.rotate( - accel_angle ).x;
        dribble_info->dash_count_ = tmp_dash_count;
        dribble_info->min_opp_dist_ = min_opp_dist;
    }

    return ( tmp_dash_count > 0 );
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Dribble2008::existKickableOpponent( const WorldModel & wm,
                                         const Vector2D & ball_pos,
                                         double * min_opp_dist ) const
{
    static const double kickable_area = ServerParam::i().defaultKickableArea() + 0.2;

    for ( const PlayerObject * o : wm.opponentsFromSelf() )
    {
        if ( o->posCount() > 5 )
        {
            continue;
        }

        if ( o->distFromSelf() > 30.0 )
        {
            break;
        }

        // goalie's catchable check
        if ( o->goalie() )
        {
            if ( ball_pos.x > ServerParam::i().theirPenaltyAreaLineX()
                 && ball_pos.absY() < ServerParam::i().penaltyAreaHalfWidth() )
            {
                double d = o->pos().dist( ball_pos );
                if ( d < ServerParam::i().catchableArea() )
                {
                    return true;
                }

                d -= ServerParam::i().catchableArea();
                if ( *min_opp_dist > d )
                {
                    *min_opp_dist = d;
                }
            }
        }

        // normal kickable check
        double d = o->pos().dist( ball_pos );
        if ( d < kickable_area )
        {
            return true;
        }

        if ( *min_opp_dist > d )
        {
            *min_opp_dist = d;
        }
    }

    return false;
}


/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Dribble2008::doDodge( PlayerAgent * agent,
                           const Vector2D & target_point )
{
    const WorldModel & wm = agent->world();

    const double new_target_dist = 6.0;

    AngleDeg avoid_angle
        = getAvoidAngle( agent,
                         ( target_point - wm.self().pos() ).th() );

    const Vector2D new_target_rel
        = Vector2D::polar2vector( new_target_dist, avoid_angle );
    Vector2D new_target
        = wm.self().pos()
        + new_target_rel;

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doDodge. avoid_angle=%.1f",
                  avoid_angle.degree() );
    agent->debugClient().addCircle( new_target, 0.7 );

    const PlayerObject::Cont & opponents = wm.opponentsFromSelf();

    for ( const PlayerObject * o : opponents )
    {
        if ( o->posCount() >= 5 ) break;
        if ( o->isGhost() ) break;
        if ( o->distFromSelf() > 3.0 ) break;

        if ( ( o->angleFromSelf() - avoid_angle ).abs() > 90.0 )
        {
            continue;
        }

        if ( o->distFromSelf()
             < ServerParam::i().defaultKickableArea() + 0.3 )
        {
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": doDodge. emergency avoidance" );
            return doAvoidKick( agent, avoid_angle );
        }
    }

    double min_opp_dist = ( opponents.empty()
                            ? 100.0
                            : opponents.front()->distFromSelf() );

    double dir_diff_abs = ( avoid_angle - wm.self().body() ).abs();
    double avoid_dash_power;
    if ( min_opp_dist > 3.0
         || dir_diff_abs < 120.0
         || agent->world().self().stamina() < ServerParam::i().staminaMax() * 0.5
         )
    {
        avoid_dash_power
            = agent->world().self().getSafetyDashPower( ServerParam::i().maxDashPower() );
    }
    else
    {
        // backward
        avoid_dash_power
            = agent->world().self().getSafetyDashPower( ServerParam::i().minDashPower() );

    }

    const double pitch_buffer = 1.0;
    if ( new_target.absX() > ServerParam::i().pitchHalfLength() - pitch_buffer )
    {
        double diff
            = new_target.absX()
            - (ServerParam::i().pitchHalfLength() - pitch_buffer);
        double rate = 1.0 - diff / new_target_rel.absX();
        new_target
            = wm.self().pos()
            + Vector2D::polar2vector( new_target_dist * rate,
                                      avoid_angle );
    }
    if ( new_target.absY() > ServerParam::i().pitchHalfWidth() - pitch_buffer )
    {
        double diff
            = new_target.absY()
            - (ServerParam::i().pitchHalfWidth() - pitch_buffer);
        double rate = 1.0 - diff / new_target_rel.absY();
        new_target
            = wm.self().pos()
            + Vector2D::polar2vector( new_target_dist * rate,
                                      avoid_angle );
    }

    int n_dash = 2;

    if ( avoid_dash_power > 0.0
         && wm.self().pos().x > -20.0
         && new_target.absY() > 15.0 )
    {
        double dist_to_target = wm.self().pos().dist( new_target );
        n_dash = wm.self().playerType().cyclesToReachDistance( dist_to_target );
        n_dash = std::min( 3, n_dash );

        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doDodge. dash step = %d",
                      n_dash );
    }

    {
        Ray2D drib_ray( wm.self().pos(), avoid_angle );
        Rect2D pitch_rect( Vector2D( - ServerParam::i().pitchHalfLength() + 0.5,
                                     - ServerParam::i().pitchHalfWidth() + 0.5 ),
                           Size2D( ServerParam::i().pitchLength() - 1.0,
                                   ServerParam::i().pitchWidth() - 1.0 ) );
        Vector2D pitch_intersect;
        if ( pitch_rect.intersection( drib_ray, &pitch_intersect, NULL ) == 1 )
        {
            if ( wm.self().pos().dist( pitch_intersect ) < 7.0 )
            {
                dlog.addText( Logger::DRIBBLE,
                              __FILE__": doDodge. pitch intersection near."
                              " enforce 1 dash step" );
                n_dash = 1;
            }
        }
    }

    return doAction( agent, new_target, avoid_dash_power,
                     n_dash, false, true ); // no dodge & dodge_mode flag
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Dribble2008::doAvoidKick( PlayerAgent * agent,
                               const AngleDeg & avoid_angle )
{
    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doAvoidKick" );

    const WorldModel & wm = agent->world();

    const double ball_move_radius = 2.0;

    const Vector2D target_rel_point
        = Vector2D::polar2vector( ball_move_radius, avoid_angle );

    // my max turnable moment with current speed
    const double next_turnable
        = ServerParam::i().maxMoment()
        / ( 1.0
            + wm.self().playerType().inertiaMoment()
            * (wm.self().vel().r()
               * wm.self().playerType().playerDecay()) );
    // my inernia move vector
    const Vector2D my_final_rel_pos
        = wm.self().vel()
        / ( 1.0 - wm.self().playerType().playerDecay() );

    AngleDeg target_angle = (target_rel_point - my_final_rel_pos).th();
    double dir_diff_abs = (target_angle - wm.self().body()).abs();
    double dir_margin_abs
        = std::max( 12.0,
                    std::fabs( AngleDeg::atan2_deg( wm.self().playerType().kickableArea() * 0.8,
                                                    ball_move_radius ) ) );

    double ball_first_speed;
    // kick -> dash -> dash -> dash -> ...
    if ( dir_diff_abs < dir_margin_abs
         || dir_diff_abs > 180.0 - dir_margin_abs ) // backward dash
    {
        ball_first_speed = 0.7;
    }
    // kick -> turn -> dash -> dash -> ...
    else if ( dir_diff_abs < next_turnable
              || dir_diff_abs > 180.0 - next_turnable )
    {
        ball_first_speed = 0.5;
    }
    // kick -> turn -> turn -> dash -> ...
    else
    {
        ball_first_speed = 0.3;
    }

    Vector2D required_first_vel
        = Vector2D::polar2vector( ball_first_speed,
                                  ( target_rel_point - wm.ball().rpos() ).th() );
    Vector2D required_accel = required_first_vel - wm.ball().vel();
    double required_kick_power = required_accel.r() / wm.self().kickRate();

    // over max power
    if ( required_kick_power > ServerParam::i().maxPower() )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doAvoidKick. power over. hold" );
        Vector2D face_point
            = wm.self().pos()
            + Vector2D::polar2vector( 20.0, target_angle );
        return Body_HoldBall2008( true, face_point ).execute( agent );
    }

    // check collision
    if ( ( wm.ball().rpos() + required_first_vel ).dist( wm.self().vel() )
         < wm.self().playerType().playerSize() + ServerParam::i().ballSize() )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": doAvoidKick. detect collision. hold" );
        Vector2D face_point
            = wm.self().pos()
            + Vector2D::polar2vector(20.0, target_angle);
        return Body_HoldBall2008( true, face_point ).execute( agent );
    }

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": doAvoidKick. done" );
    agent->debugClient().addMessage( "AvoidKick" );

    return agent->doKick( required_kick_power,
                          required_accel.th() - wm.self().body() );
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Dribble2008::isDodgeSituation( const PlayerAgent * agent,
                                    const Vector2D & target_point )
{
    const WorldModel & wm = agent->world();

    //////////////////////////////////////////////////////

    const AngleDeg target_angle = (target_point - wm.self().pos()).th();
    // check if opponent on target dir
    const Sector2D sector( wm.self().pos(),
                           0.6,
                           std::min( 4, M_dash_count )
                           * wm.self().playerType().realSpeedMax(),
                           target_angle - 20.0, target_angle + 20.0 );
    const double base_safety_dir_diff = 60.0;
    double dodge_consider_dist
        = ( ServerParam::i().defaultPlayerSpeedMax() * M_dash_count * 1.5 );
    //     double dodge_consider_dist
    //         = ( ServerParam::i().defaultPlayerSpeedMax() * M_dash_count * 2.0 )
    //         + 4.0;

    if ( dodge_consider_dist > 10.0 )
    {
        dodge_consider_dist = 10.0;
    }

    for ( const PlayerObject * o : wm.opponentsFromSelf() )
    {
        if ( o->posCount() >= 10 ) continue;
        if ( o->isGhost() ) continue;

        if ( sector.contains( o->pos() ) )
        {
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": exist opp on dir" );
            return true;
        }

        const double dir_diff = ( o->angleFromSelf() - target_angle ).abs();
        double add_buf = 0.0;
        if ( o->distFromSelf() < dodge_consider_dist
             && o->distFromSelf() > 3.0 )
        {
            add_buf = 30.0 / o->distFromSelf();
        }

        if ( o->distFromSelf() < 1.0
             || ( o->distFromSelf() < 1.5 && dir_diff < 120.0 )
             || ( o->distFromSelf() < dodge_consider_dist
                  && dir_diff < base_safety_dir_diff + add_buf ) )
        {
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": exist obstacle (%.1f, %.1f) dist=%.2f"
                          " dir_diff=%.1f dir_buf=%.1f",
                          o->pos().x, o->pos().y,
                          o->distFromSelf(),
                          dir_diff, base_safety_dir_diff + add_buf );
            return true;
        }
    }

    return false;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Dribble2008::canKickAfterDash( const PlayerAgent * agent,
                                    double * dash_power )
{
    static const Rect2D penalty_area( Vector2D( ServerParam::i().theirPenaltyAreaLineX(),
                                                - ServerParam::i().penaltyAreaHalfWidth() ),
                                      Size2D( ServerParam::i().penaltyAreaLength(),
                                              ServerParam::i().penaltyAreaWidth() ) );


    const WorldModel & wm = agent->world();

    if ( wm.interceptTable().opponentStep() <= 1 )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": canKickAfterDash..exist reachable opponent" );
        return false;
    }

    const Vector2D ball_next = wm.ball().pos() + wm.ball().vel();
    if ( ball_next.absX() > ServerParam::i().pitchHalfLength() - 0.2
         || ball_next.absY() > ServerParam::i().pitchHalfWidth() - 0.2 )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": canKickAfterDash..next ball pos is out of pitch" );
        return false;
    }

    // check dash possibility
    {
        AngleDeg accel_angle = ( *dash_power < 0.0
                                 ? wm.self().body() - 180.0
                                 : wm.self().body() );
        Vector2D my_pos = wm.self().pos();
        Vector2D my_vel = wm.self().vel();

        const double max_accel_mag = ( std::fabs( *dash_power )
                                       * wm.self().playerType().dashPowerRate()
                                       * wm.self().effort() );
        double accel_mag = max_accel_mag;
        if ( wm.self().playerType().normalizeAccel( wm.self().vel(),
                                                    accel_angle,
                                                    &accel_mag ) )
        {
            *dash_power *= accel_mag / max_accel_mag;
        }

        Vector2D dash_accel = Vector2D::polar2vector( accel_mag, accel_angle );

        my_vel += dash_accel;
        my_pos += my_vel;

        double ball_dist = my_pos.dist( ball_next );
        double noise_buf
            = my_vel.r() * ServerParam::i().playerRand() * 0.5
            + wm.ball().vel().r() * ServerParam::i().ballRand() * 0.5;

        dlog.addText( Logger::DRIBBLE,
                      __FILE__": canKickAfterDash. ball_dist= %.2f, noise= %.2f",
                      ball_dist, noise_buf );

        if ( ( ( ball_next - my_pos ).th() - accel_angle ).abs() < 150.0
             && ball_dist < wm.self().playerType().kickableArea() - noise_buf - 0.2
             && ( ball_dist - noise_buf > ( wm.self().playerType().playerSize()
                                            + ServerParam::i().ballSize() ) ) )
        {
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": canKickAfterDash. kickable after one dash" );
        }
        else
        {
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": canKickAfterDash. no kickable after one dash." );
            return false;
        }
    }

    for ( const PlayerObject * o : wm.opponentsFromBall() )
    {
        if ( o->distFromSelf() > 8.0 ) break;
        if ( o->posCount() > 5 ) continue;
        if ( o->isGhost() ) continue;
        if ( o->isTackling() ) continue;

        const PlayerType * player_type = o->playerTypePtr();
        const double control_area = ( ( o->goalie()
                                        && penalty_area.contains( o->pos() )
                                        && penalty_area.contains( ball_next ) )
                                      ? ServerParam::i().catchableArea()
                                      : player_type->kickableArea() );
        const Vector2D opp_next = o->pos() + o->vel();
        const AngleDeg opp_body =  ( o->bodyCount() <= 1
                                     ? o->body()
                                     : ( ball_next - opp_next ).th() );
        const Vector2D opp_2_ball = ( ball_next - opp_next ).rotatedVector( - opp_body );

        double tackle_dist = ( opp_2_ball.x > 0.0
                               ? ServerParam::i().tackleDist()
                               : ServerParam::i().tackleBackDist() );
        if ( tackle_dist > 1.0e-5 )
        {
            double tackle_prob = ( std::pow( opp_2_ball.absX() / tackle_dist,
                                             ServerParam::i().tackleExponent() )
                                   + std::pow( opp_2_ball.absY() / ServerParam::i().tackleWidth(),
                                               ServerParam::i().tackleExponent() ) );
            if ( tackle_prob < 1.0
                 && 1.0 - tackle_prob > 0.6 ) // success probability
            {
                dlog.addText( Logger::DRIBBLE,
                              __FILE__": canKickAfterDash. exist tackle opp %d(%.1f %.1f)",
                              o->unum(),
                              o->pos().x, o->pos().y );
                return false;
            }
        }

        const double max_accel = ( ServerParam::i().maxDashPower()
                                   * player_type->dashPowerRate()
                                   * player_type->effortMax() );

        if ( opp_2_ball.absY() < control_area + 0.1
             && ( opp_2_ball.absX() < max_accel
                  || ( opp_2_ball + Vector2D( max_accel, 0.0 ) ).r() < control_area + 0.1
                  || ( opp_2_ball - Vector2D( max_accel, 0.0 ) ).r() < control_area + 0.1 )
             )
        {
            dlog.addText( Logger::DRIBBLE,
                          __FILE__": canKickAfterDash. exist kickable opp after dash %d(%.1f %.1f)",
                          o->unum(),
                          o->pos().x, o->pos().y );
            return false;
        }
        else if ( opp_2_ball.absY() < ServerParam::i().tackleWidth()
                  && opp_2_ball.x > 0.0
                  && opp_2_ball.x - max_accel < ServerParam::i().tackleDist() + 0.1 )
        {

            dlog.addText( Logger::DRIBBLE,
                          __FILE__": canKickAfterDash. exist tackle opp after dash %d(%.1f %.1f)",
                          o->unum(),
                          o->pos().x, o->pos().y );
            return false;
        }

    }

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": canKickAfterDash. ok. no interfere." );
    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Dribble2008::existCloseOpponent( const PlayerAgent * agent,
                                      AngleDeg * keep_angle )
{
    const WorldModel & wm = agent->world();

    const PlayerObject * opp = wm.getOpponentNearestToBall( 5 );
    if ( ! opp )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": existCloseOppnent. No opponent." );
        return false;
    }

    if ( opp->distFromBall()
         > ( ServerParam::i().defaultPlayerSpeedMax()
             + ServerParam::i().tackleDist() )
         )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": existCloseOpponent. No dangerous opponent" );
        return false;
    }


    // opp is in dangerous range

    Vector2D my_next = wm.self().pos() + wm.self().vel();
    Vector2D opp_next = opp->pos() + opp->vel();

    if ( opp->bodyCount() == 0
         || ( opp->velCount() <= 1 && opp->vel().r() > 0.2 ) )
    {
        Line2D opp_line( opp_next,
                         ( opp->bodyCount() == 0
                           ? opp->body()
                           : opp->vel().th() ) );
        Vector2D proj_pos = opp_line.projection( my_next );

        *keep_angle = ( my_next - proj_pos ).th();

        dlog.addText( Logger::DRIBBLE,
                      __FILE__"d: existCloseOpponent  found interfere opponent (%.1f %.1f). avoid line."
                      " keep_angle=%.1f",
                      opp_next.x, opp_next.y,
                      keep_angle->degree() );

        return true;
    }

    *keep_angle = ( my_next - opp_next ).th();

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": existCloseOpponent  found interfere opponent (%.1f %.1f). opposite side."
                  " keep_angle=%.1f",
                  opp_next.x, opp_next.y,
                  keep_angle->degree() );
    return true;
}

/*-------------------------------------------------------------------*/
/*!
  TODO: angles which agent can turn by 1 step should have the priority
*/
AngleDeg
Body_Dribble2008::getAvoidAngle( const PlayerAgent * agent,
                                 const AngleDeg & target_angle )
{
    const WorldModel & wm = agent->world();

    if ( wm.opponentsFromSelf().empty() )
    {
        return target_angle;
    }

    const double avoid_radius = 4.4; //5.0; 2008-04-30
    const double safety_opp_dist = 5.0;
    const double safety_space_body_ang_radius2 = 3.0 * 3.0;


    const PlayerObject::Cont & opponents = wm.opponentsFromSelf();

    // at first, check my body dir and opposite dir of my body
    if ( ! opponents.empty()
         && opponents.front()->distFromSelf() < 3.0 )
    {
        dlog.addText( Logger::DRIBBLE,
                      __FILE__": getAvoidAngle. check body line. base_target_angle=%.0f",
                      target_angle.degree() );

        AngleDeg new_target_angle = wm.self().body();
        for ( int i = 0; i < 2; i++ )
        {
            const Vector2D sub_target
                = wm.self().pos()
                + Vector2D::polar2vector( avoid_radius,
                                          new_target_angle );

            if ( sub_target.absX() > ServerParam::i().pitchHalfLength() - 1.8
                 || sub_target.absY() > ServerParam::i().pitchHalfWidth() - 1.8 )
            {
                // out of pitch
                continue;
            }

            bool success = true;
            for ( const PlayerObject * o : opponents )
            {
                if ( o->posCount() >= 10 ) continue;

                if ( o->distFromSelf() > 20.0 )
                {
                    break;
                }

                if ( o->distFromSelf() < safety_opp_dist
                     && (o->angleFromSelf() - new_target_angle).abs() < 30.0 )
                {
                    dlog.addText( Logger::DRIBBLE,
                                  "____ body line dir=%.1f"
                                  " exist near opp(%.1f, %.1f)",
                                  new_target_angle.degree(),
                                  o->pos().x, o->pos().y );
                    success = false;
                    break;
                }

                if ( sub_target.dist2( o->pos() )
                     < safety_space_body_ang_radius2 )
                {
                    dlog.addText( Logger::DRIBBLE,
                                  "____ body line dir=%.1f"
                                  " exist opp(%.1f, %.1f) "
                                  "close to subtarget(%.1f, %.1f)",
                                  new_target_angle.degree(),
                                  o->pos().x, o->pos().y,
                                  sub_target.x, sub_target.y );
                    success = false;
                    break;
                }
            }

            if ( success )
            {
                dlog.addText( Logger::DRIBBLE,
                              "---> avoid to body line. angle=%.1f",
                              new_target_angle.degree() );
                return new_target_angle;
            }

            new_target_angle -= 180.0;
        }
    }

    // search divisions

    const int search_divs = 10;
    const double div_dir = 360.0 / static_cast< double >( search_divs );
    const double safety_angle = 60.0;
    const double safety_space_radius2 = avoid_radius * avoid_radius;

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": getAvoidAngle. search angles. base_target_angle=%.0f",
                  target_angle.degree() );

    double angle_sign = 1.0;
    if ( agent->world().self().pos().y < 0.0 ) angle_sign = -1.0;

    for ( int i = 1; i < search_divs; ++i, angle_sign *= -1.0 )
    {
        const AngleDeg new_target_angle
            = target_angle
            + ( angle_sign * div_dir * ( (i+1)/2 ) );

        const Vector2D sub_target
            = wm.self().pos()
            + Vector2D::polar2vector( avoid_radius,
                                      new_target_angle );

        if ( sub_target.absX()
             > ServerParam::i().pitchHalfLength() - wm.self().playerType().kickableArea() - 0.2
             || sub_target.absY()
             > ServerParam::i().pitchHalfWidth() - wm.self().playerType().kickableArea() - 0.2 )
        {
            dlog.addText( Logger::DRIBBLE,
                          "avoid angle. out of pitch. angle=%.0f pos=(%.1f %.1f)",
                          new_target_angle.degree(),
                          sub_target.x, sub_target.y );
            // out of pitch
            continue;
        }

        if ( sub_target.x < 30.0
             && sub_target.x < wm.self().pos().x - 2.0 )
        {
            dlog.addText( Logger::DRIBBLE,
                          "avoid angle. backword.. angle=%.0f pos=(%.1f %.1f)",
                          new_target_angle.degree(),
                          sub_target.x, sub_target.y );
            continue;
        }

        dlog.addText( Logger::DRIBBLE,
                      "avoid angle=%.0f pos=(%.1f %.1f)",
                      new_target_angle.degree(),
                      sub_target.x, sub_target.y );

        bool success = true;
        for ( const PlayerObject * o : opponents )
        {
            if ( o->posCount() >= 10 ) continue;

            if ( o->distFromSelf() > 20.0 ) break;

            double add_dir = 5.8 / o->distFromSelf();
            add_dir = std::min( 180.0 - safety_angle, add_dir );
            if ( o->distFromSelf() < safety_opp_dist
                 && ( ( o->angleFromSelf() - new_target_angle ).abs()
                      < safety_angle + add_dir ) )
            {
                dlog.addText( Logger::DRIBBLE,
                              "____ opp angle close. cannot avoid to %.1f",
                              new_target_angle.degree() );
                success = false;
                break;
            }

            if ( sub_target.dist2( o->pos() ) < safety_space_radius2 )
            {
                dlog.addText( Logger::DRIBBLE,
                              "____ opp dist close. cannot avoid to %.1f",
                              new_target_angle.degree() );
                success = false;
                break;
            }
        }

        if ( success )
        {
            dlog.addText( Logger::DRIBBLE,
                          "---> avoid to angle= %.1f",
                          new_target_angle.degree() );
            return new_target_angle;
        }

    }


    // Best angle is not found.
    // go to the least congestion point

    dlog.addText( Logger::DRIBBLE,
                  __FILE__": getAvoidAngle. search least congestion point." );

    Rect2D target_rect( Vector2D( wm.self().pos().x - 4.0,
                                  wm.self().pos().y - 4.0 ),
                        Size2D( 8.0, 8.0 ) );

    double best_score = 10000.0;
    Vector2D best_target = wm.self().pos();

    double x_i = 30.0 - wm.self().pos().x;
    if ( x_i > 0.0 ) x_i = 0.0;
    if ( x_i < -8.0 ) x_i = -8.0;

    for ( ; x_i < 8.5; x_i += 1.0 )
    {
        for ( double y_i = - 8.0; y_i < 8.5; y_i += 1.0 )
        {
            Vector2D candidate = wm.self().pos();
            candidate.add( x_i, y_i );

            if ( candidate.absX() > ServerParam::i().pitchHalfLength() - 2.0
                 || candidate.absY() > ServerParam::i().pitchHalfWidth() - 2.0 )
            {
                continue;
            }

            double tmp_score = 0.0;
            for ( const PlayerObject * o : opponents )
            {
                double d2 = o->pos().dist2( candidate );

                if ( d2 > 15.0 * 15.0 ) continue;

                tmp_score += 1.0 / d2;
            }

            if ( tmp_score < best_score )
            {
                dlog.addText( Logger::DRIBBLE,
                              "    update least congestion point to"
                              " (%.2f, %.2f) score=%.4f",
                              candidate.x, candidate.y, tmp_score );
                best_target = candidate;
                best_score = tmp_score;
            }
        }
    }

    dlog.addText( Logger::DRIBBLE,
                  "  avoid to point (%.2f, %.2f)",
                  best_target.x, best_target.y );

    return ( best_target - wm.self().pos() ).th();
}
