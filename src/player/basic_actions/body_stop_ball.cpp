// -*-c++-*-

/*!
  \file body_stop_ball.cpp
  \brief kick the ball to keep a current positional relation.
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

#include "body_stop_ball.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/player/debug_client.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

#include <algorithm>

using namespace rcsc;

/*-------------------------------------------------------------------*/
/*!

*/
bool
Body_StopBall::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::ACTION,
                  "%s:%d: Body_StopBall"
                  ,__FILE__, __LINE__ );

    const WorldModel & wm = agent->world();

    if ( ! wm.self().isKickable() )
    {
        std::cerr << __FILE__ << ": " << __LINE__
                  << " not ball kickable!"
                  << std::endl;
        dlog.addText( Logger::ACTION,
                      __FILE__":  not kickable" );
        return false;
    }

    if ( ! wm.ball().velValid() )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": (execute) invalid ball velocity. kick to self" );
        agent->debugClient().addMessage( "StopBall:InvalidBallVel" );
        // keep current relative position
        Vector2D required_accel = wm.self().vel() - wm.ball().rpos();
        double kick_power = required_accel.r() / wm.self().kickRate();
        kick_power *= 0.5;
        agent->doKick( std::min( kick_power, ServerParam::i().maxPower() ),
                       required_accel.th() - wm.self().body() );
        return true;
    }

    double accel_radius = 0.0;
    AngleDeg accel_angle;
    calcAccel( agent, &accel_radius, &accel_angle );

    if ( accel_radius < 0.02 )
    {
        agent->doTurn( 0.0 );
        return false;
    }

    double kick_power = accel_radius / wm.self().kickRate();
    kick_power = std::min( kick_power, ServerParam::i().maxPower() );

    agent->debugClient().addMessage( "StopBall%.0f", kick_power );

    return agent->doKick( kick_power,
                          accel_angle - wm.self().body() );
}

/*-------------------------------------------------------------------*/
/*!

*/
void
Body_StopBall::calcAccel( PlayerAgent * agent,
                          double * accel_radius,
                          AngleDeg * accel_angle )
{
    const WorldModel & wm = agent->world();

    const double safety_dist
        = wm.self().playerType().playerSize()
        + ServerParam::i().ballSize()
        + 0.1;

    double target_dist = wm.ball().distFromSelf();
    if ( target_dist < safety_dist )
    {
        target_dist = safety_dist;
    }
    if ( target_dist > wm.self().playerType().kickableArea() - 0.1 )
    {
        target_dist = wm.self().playerType().kickableArea() - 0.1;
    }

    Vector2D target_rel = wm.ball().rpos();
    target_rel.setLength( target_dist );

    //Vector2D next_pos = target_rel + world().self().vel();
    //Vector2D required_vel = next_pos - world.ball().rpos();
    //Vector2D required_accel = required_vel - world.ball().vel();

    Vector2D required_accel = wm.self().vel();
    required_accel += target_rel; // target relative to current
    required_accel -= wm.ball().rpos(); // required vel = pos diff
    required_accel -= wm.ball().vel();  // required accel

    *accel_radius = required_accel.r();

    if ( *accel_radius < 0.01 )
    {
        return;
    }

    // check max accel with player's kick rate

    const double max_accel
        = ServerParam::i().maxPower()
        * wm.self().kickRate();
    if ( max_accel > *accel_radius )
    {
        // can accelerate --> can stop ball successfully
        *accel_angle = required_accel.th();
        return;
    }

    ////////////////////////////////////////////////////////////////////
    // keep the ball as much as possible near the best point

    //Vector2D my_next = world.self().vel();
    //Vector2D ball_next = world.ball().rpos() + world.ball().vel();
    //Vector2D next_ball_to_self = my_next - ball_next;
    Vector2D next_ball_to_self = wm.self().vel();
    next_ball_to_self -= wm.ball().rpos();
    next_ball_to_self -= wm.ball().vel();

    double keep_dist
        = wm.self().playerType().playerSize()
        + wm.self().playerType().kickableMargin() * 0.4;

    *accel_radius = std::min( max_accel, next_ball_to_self.r() - keep_dist );
    *accel_angle = next_ball_to_self.th();

    if ( *accel_radius < 0.0 ) // == next_ball_dist < keep_dist
    {
        // next ball dist will be closer than keep dist.
        //  --> kick angle must be reversed.
        *accel_radius *= -1.0;
        *accel_radius = std::min( *accel_radius, max_accel );
        *accel_angle -= 180.0;
    }
}
