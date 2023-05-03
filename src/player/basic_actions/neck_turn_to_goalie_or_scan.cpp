// -*-c++-*-

/*!
  \file neck_turn_to_goalie_or_scan.cpp
  \brief check opponent goalie or scan field with neck evenly
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

#include "neck_turn_to_goalie_or_scan.h"

#include "basic_actions.h"
#include "bhv_scan_field.h"
#include "neck_scan_field.h"
#include "neck_turn_to_ball_or_scan.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

using namespace rcsc;

/*-------------------------------------------------------------------*/
/*!

*/
bool
Neck_TurnToGoalieOrScan::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::ACTION,
                  __FILE__": Neck_TurnToGoalieOrScan" );

    const WorldModel & wm = agent->world();

    const AbstractPlayerObject * opp_goalie = wm.getTheirGoalie();

    if ( ! opp_goalie
         || opp_goalie->posCount() <= M_count_thr )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": goalie not found, or already valid" );
        return Neck_TurnToBallOrScan( 0 ).execute( agent );
    }

    if ( opp_goalie->isGhost()
         && opp_goalie->posCount() >= 3 )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": detect ghost goalie" );
        return Neck_TurnToBallOrScan( 0 ).execute( agent );
    }

    const ServerParam & param = ServerParam::i();

    const ViewWidth vwidth = agent->effector().queuedNextViewWidth();
    const double next_view_width_half = vwidth.width() * 0.5;
    const Vector2D my_next = agent->effector().queuedNextSelfPos();
    const AngleDeg my_next_body = agent->effector().queuedNextSelfBody();

    const Vector2D goalie_next = opp_goalie->pos() + opp_goalie->vel();
    const AngleDeg goalie_angle = ( goalie_next - my_next ).th();

    if ( goalie_next.dist( my_next ) < param.visibleDistance() - 1.0
         || ( ( goalie_angle - my_next_body ).abs()
              > param.maxNeckAngle() + next_view_width_half - 7.0 ) )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": never face to goalie. next_width_half=%.0f",
                      next_view_width_half );
        return Neck_TurnToBallOrScan( 0 ).execute( agent );
    }

    if ( opp_goalie->posCount() == 0 )
    {
        AngleDeg target_angle = goalie_angle;
        if ( goalie_next.y < my_next.y )
        {
            target_angle -= next_view_width_half - 7.0;
        }
        else
        {
            target_angle += next_view_width_half - 7.0;
        }

        if ( ( target_angle - my_next_body ).abs() < param.maxNeckAngle() )
        {
            dlog.addText( Logger::ACTION,
                          __FILE__": neck to goalie and other area. angle=%.0f",
                          target_angle.degree() );
            target_angle -= my_next_body;
            target_angle -= wm.self().neck();
            return agent->doTurnNeck( target_angle );
        }

        Vector2D goal( param.pitchHalfLength() - 1.0, 0.0 );
        goal.y = ( goalie_next.y < my_next.y
                   ? - param.goalHalfWidth()
                   : + param.goalHalfWidth() );
        AngleDeg goal_angle = ( goal - my_next ).th();
        if ( ( goal_angle - my_next_body ).abs() < param.maxNeckAngle() + next_view_width_half
             && ( goal_angle - goalie_angle ).abs() < next_view_width_half * 2.0 - 10.0 )
        {
            target_angle = ( goal_angle.isLeftOf( goalie_angle )
                             ? AngleDeg::bisect( goal_angle, goalie_angle )
                             : AngleDeg::bisect( goalie_angle, goal_angle ) );
            dlog.addText( Logger::ACTION,
                          __FILE__": neck to goalie and goal. angle=%.0f",
                          target_angle.degree() );
            target_angle -= my_next_body;
            target_angle -= wm.self().neck();
            return agent->doTurnNeck( target_angle );
        }
    }

    dlog.addText( Logger::ACTION,
                  __FILE__": neck to goalie. pos=(%.1f %.1f)",
                  goalie_next.x, goalie_next.y );
    return Neck_TurnToPoint( goalie_next ).execute( agent );
}
