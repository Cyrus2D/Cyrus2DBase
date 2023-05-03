// -*-c++-*-

/*!
  \file body_kick_one_step.cpp
  \brief one step kick behavior.
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

#include "body_kick_one_step.h"

#include "body_stop_ball.h"
//#include "body_kick_to_relative.h"
#include "body_hold_ball2008.h"
#include "kick_table.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/geom/circle_2d.h>
#include <rcsc/geom/ray_2d.h>

#include <algorithm>

using namespace rcsc;

/*-------------------------------------------------------------------*/
/*!

*/
bool
Body_KickOneStep::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::KICK,
                  __FILE__": Body_KickOneStep" );

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

    Vector2D ball_vel = wm.ball().vel();

    if ( ! wm.ball().velValid() )
    {
        if ( ! M_force_mode )
        {
            dlog.addText( Logger::KICK,
                          __FILE__". unknown ball vel" );
            return Body_StopBall().execute( agent );
        }

        ball_vel.assign( 0.0, 0.0 );
    }

    M_first_speed = std::min( M_first_speed, ServerParam::i().ballSpeedMax() );

    const AngleDeg target_angle = ( M_target_point - wm.ball().pos() ).th();

    // Vector2D first_vel = get_max_possible_vel( target_angle,
    //                                            wm.self().kickRate(),
    //                                            ball_vel );
    Vector2D first_vel = KickTable::calc_max_velocity( target_angle,
                                                       wm.self().kickRate(),
                                                       ball_vel );
    double first_speed = first_vel.r();

    if ( first_speed > M_first_speed )
    {
        first_vel.setLength( M_first_speed );
        first_speed = M_first_speed;
    }
    else
    {
        dlog.addText( Logger::KICK,
                      __FILE__": cannot get required vel. only angle adjusted" );
    }

    // first_vel.r() may be less than M_first_speed ...


    const Vector2D kick_accel = first_vel - ball_vel;

    double kick_power = kick_accel.r() / wm.self().kickRate();
    const AngleDeg kick_dir = kick_accel.th() - wm.self().body();

    if ( kick_power > ServerParam::i().maxPower() + 0.01 )
    {
        // std::cerr << agent->config().teamName() << ' '
        //           << wm.self().unum() << ": "
        //           << wm.time() << " Body_KickOneStep unexpected reach."
        //           << " kick_power=" << kick_power << std::endl;
        if ( first_speed < 0.001 )
        {
            dlog.addText( Logger::KICK,
                          __FILE__": could not stop the ball completely, but try to stop. kick_power=%f",
                          kick_power );
            return Body_StopBall().execute( agent );
        }

        if ( ! M_force_mode )
        {
            dlog.addText( Logger::KICK,
                          __FILE__": could not stop the ball completely. hold ball. kick_power=%f" ,
                          kick_power );
            return Body_HoldBall2008( true,
                                      M_target_point,
                                      M_target_point
                                      ).execute( agent );
        }

        kick_power = ServerParam::i().maxPower();
    }

    dlog.addText( Logger::KICK,
                  __FILE__": first_speed=%.3f, angle=%.1f, power=%.1f, dir=%.1f ",
                  first_vel.r(), first_vel.th().degree(),
                  kick_power, kick_dir.degree() );

    M_ball_result_pos = wm.ball().pos() + first_vel;
    M_ball_result_vel = first_vel * ServerParam::i().ballDecay();
    M_kick_step = 1;

    return agent->doKick( kick_power, kick_dir );
}

#if 0
/*-------------------------------------------------------------------*/
/*!

*/
Vector2D
Body_KickOneStep::get_max_possible_vel( const AngleDeg & target_angle,
                                        const double & krate,
                                        const Vector2D & ball_vel )
{
    // ball info may be the estimated value..

    const double max_accel
        = std::min( ServerParam::i().maxPower() * krate,
                    ServerParam::i().ballAccelMax() );

    // origin point is current ball pos
    Ray2D desired_ray( Vector2D( 0.0, 0.0 ), target_angle );
    // center point is next ball pos relative to current ball pos
    Circle2D next_reachable_circle( ball_vel, max_accel );


    // sol is ball vel (= current ball vel + accel)
    Vector2D sol1, sol2; // rel to current ball pos
    int num = next_reachable_circle.intersection( desired_ray, &sol1, &sol2 );

    if ( num == 0 )
    {
        dlog.addText( Logger::KICK,
                      __FILE__": (get_max_possible_vel) angle=%.1f. No solution. try to stop the ball",
                      target_angle.degree() );
        //return Vector2D( 0.0, 0.0 );
        Vector2D accel = -ball_vel;
        double accel_r = accel.r();
        if ( accel_r > max_accel )
        {
            accel *= max_accel / accel_r;
        }
        return accel; // stop the ball
    }

    if ( num == 1 )
    {
        if ( sol1.r() > ServerParam::i().ballSpeedMax() )
        {
            // next inertia ball point is within reachable circle.
            if ( next_reachable_circle.contains( Vector2D( 0.0, 0.0 ) ) )
            {
                // can adjust angle at least
                sol1.setLength( ServerParam::i().ballSpeedMax() );

                dlog.addText( Logger::KICK,
                              __FILE__": (get_max_possible_vel) angle=%.1f."
                              " 1 solution  adjust.",
                              target_angle.degree() );
            }
            else
            {
                // failed
                sol1.assign( 0.0, 0.0 );

                dlog.addText( Logger::KICK,
                              __FILE__": (get_max_possible_vel) angle=%.1f."
                              " 1 solution. failed.",
                              target_angle.degree() );
            }
        }
#ifdef DEBUG
        dlog.addText( Logger::KICK,
                      "one kick -----> angle=%.1f max_vel=(%.2f, %.2f)r=%.2f",
                      target_angle.degree(),
                      sol1.x, sol1.y, sol1.r() );
#endif
        return sol1;
    }


    // num == 2

    double length1 = sol1.r();
    double length2 = sol2.r();

    if ( length1 < length2 )
    {
        std::swap( sol1, sol2 );
        std::swap( length1, length2 );
        dlog.addText( Logger::KICK,
                      __FILE__": (get_max_possible_vel) swap" );
    }

    if ( length1 > ServerParam::i().ballSpeedMax() )
    {
        if ( length2 > ServerParam::i().ballSpeedMax() )
        {
            sol1.assign( 0.0, 0.0 );

            dlog.addText( Logger::KICK,
                          __FILE__": (get_max_possible_vel) angle=%.1f."
                          " 2 solutions. but never reach",
                          target_angle.degree() );
        }
        else
        {
            sol1.setLength( ServerParam::i().ballSpeedMax() );

            dlog.addText( Logger::KICK,
                          __FILE__": (get_max_possible_vel) angle=%.1f."
                          " 2 solutions. adjust to ballSpeedMax",
                          target_angle.degree() );
        }
    }

    dlog.addText( Logger::KICK,
                  __FILE__": (get_max_possible_vel) 2 solutions: angle=%.1f max_vel=(%.2f, %.2f)r=%.2f",
                  target_angle.degree(),
                  sol1.x, sol1.y, sol1.r() );

    return sol1;
}
#endif
