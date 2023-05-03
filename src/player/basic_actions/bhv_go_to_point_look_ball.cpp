// -*-c++-*-

/*!
  \file bhv_go_to_point_look_ball.cpp
  \brief run behavior which player always looks the ball.
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

#include "bhv_go_to_point_look_ball.h"

#include "basic_actions.h"
#include "body_go_to_point.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/common/server_param.h>
#include <rcsc/math_util.h>

using namespace rcsc;

/*-------------------------------------------------------------------*/
/*!

*/
bool
Bhv_GoToPointLookBall::execute( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    const Vector2D my_final_pos = wm.self().inertiaFinalPoint();

    // already there
    if ( my_final_pos.isValid()
         && my_final_pos.dist2( M_target_point ) < square( M_dist_thr ) )
    {
        if ( wm.ball().posCount() > 5 )
        {
            return Bhv_BodyNeckToBall().execute( agent );
        }
        else
        {
            agent->doTurn( 0.0 );
            agent->setNeckAction( new Neck_TurnToBall() );
            return true;
        }
    }

    const Vector2D my_next = wm.self().pos() + wm.self().vel();
    const Vector2D ball_next = wm.ball().pos() + wm.ball().vel();

    // check angle difference between target and ball
    // if defference angle is out of my view cone width
    //   -> back dash is required
    const AngleDeg ball_target_angle_diff
        = ( M_target_point - my_next ).th()  - ( ball_next - my_next ).th();

    /*--------------------------------------------------------*/
    // normal move
    if ( ball_target_angle_diff.abs()
         < ServerParam::i().maxNeckAngle() - 5.0 )
    {
        if ( Body_GoToPoint( M_target_point,
                             M_dist_thr,
                             M_dash_power ).execute( agent ) )
        {
            agent->setNeckAction( new Neck_TurnToBall() );
            return true;
        }

        Bhv_NeckBodyToBall().execute( agent );
        return false;
    }

    /*--------------------------------------------------------*/
    // back dash
    {
        const AngleDeg target_reverse_angle
            = ( my_next - M_target_point ).th();
        const double dir_diff_abs
            = ( target_reverse_angle - wm.self().body() ).abs();

        const double buf = ( wm.self().goalie()
                             ? ServerParam::i().catchableArea()
                             : wm.self().playerType().kickableArea() );

        const double dir_margin
            = std::max( 6.0,
                        std::fabs( AngleDeg::atan2_deg
                                   (buf,
                                    ( M_target_point - my_next ).r() ) ) );

        // turn
        if ( dir_diff_abs > dir_margin )
        {
            // reverse target based on my pos
            Body_TurnToAngle( target_reverse_angle ).execute( agent );
            agent->setNeckAction( new Neck_TurnToBall() );
        }
        // dash
        else
        {
            // back dash
            double dash_power
                = wm.self().getSafetyDashPower( wm.self().goalie()
                                                ? -M_dash_power
                                                : -M_dash_power * M_back_power_rate );
            agent->doDash( dash_power );
            agent->setNeckAction( new Neck_TurnToBall() );
        }
    }

    return true;
}
