// -*-c++-*-

/*!
  \file neck_turn_to_ball_or_scan.cpp
  \brief check the ball or scan field with neck evenly
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

#include "neck_turn_to_ball_or_scan.h"

#include "basic_actions.h"
#include "bhv_scan_field.h"
#include "neck_scan_field.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

using namespace rcsc;

/*-------------------------------------------------------------------*/
/*!

*/
bool
Neck_TurnToBallOrScan::execute( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    dlog.addText( Logger::ACTION,
                  __FILE__": ball_count(%d) count_thr(%d)",
                  wm.ball().posCount(), M_count_thr );

    if ( wm.ball().posCount() <= M_count_thr )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": ball_count < count_thr, ScanField" );
        return Neck_ScanField().execute( agent );
    }

    const ServerParam & SP = ServerParam::i();

    const Vector2D ball_next = agent->effector().queuedNextBallPos();
    const Vector2D my_next = agent->effector().queuedNextSelfPos();

    if ( wm.ball().posCount() <= 0
         && ! wm.kickableOpponent()
         && ! wm.kickableTeammate()
         && my_next.dist( ball_next ) < SP.visibleDistance() - 0.2 )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": in visible distance." );
        return Neck_ScanField().execute( agent );
    }

    const AngleDeg my_next_body = agent->effector().queuedNextSelfBody();
    const double next_view_width = agent->effector().queuedNextViewWidth().width();

    if ( ( ( ball_next - my_next ).th() - my_next_body ).abs()
         > SP.maxNeckAngle() + next_view_width * 0.5 + 2.0 )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": never face to ball" );
        return Neck_ScanField().execute( agent );
    }

    dlog.addText( Logger::ACTION,
                  __FILE__": turn neck to ball" );
    return Neck_TurnToBall().execute( agent );
}
