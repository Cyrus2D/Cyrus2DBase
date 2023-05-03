// -*-c++-*-

/*!
  \file neck_turn_to_player_or_scan.cpp
  \brief check the player or scan field with neck evenly
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

#include "neck_turn_to_player_or_scan.h"

#include "basic_actions.h"
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
Neck_TurnToPlayerOrScan::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::ACTION,
                  __FILE__": Neck_TurnToPlayerOrScan" );

    if ( ! M_target_player
         || M_target_player->posCount() <= M_count_thr )
    {
        return Neck_ScanField().execute( agent );
    }

    if ( M_target_player->isSelf()
         || M_target_player->isGhost() )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": target player is ghost" );
        int count_thr = ( agent->world().kickableOpponent()
                          ? -1
                          : 0 );
        return Neck_TurnToBallOrScan( count_thr ).execute( agent );
    }

    const Vector2D my_next = agent->effector().queuedNextSelfPos();
    const AngleDeg my_next_body = agent->effector().queuedNextSelfBody();

    const Vector2D player_next = M_target_player->pos() + M_target_player->vel();

    const double next_view_width = agent->effector().queuedNextViewWidth().width();

    bool can_face = false;

    if ( ( (player_next - my_next).th() - my_next_body ).abs()
         < ServerParam::i().maxNeckAngle() + next_view_width * 0.5 - 2.0 )
    {
        can_face = true;
    }

    if ( ! can_face )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": never face to player %d (%.1f %.1f)",
                      M_target_player->unum(),
                      player_next.x, player_next.y );
        int count_thr = ( agent->world().kickableOpponent()
                          ? -1
                          : 0 );
        return Neck_TurnToBallOrScan( count_thr ).execute( agent );
    }

    dlog.addText( Logger::ACTION,
                  __FILE__": face to player %d (%.1f %.1f)",
                  M_target_player->unum(),
                  player_next.x, player_next.y );

    return Neck_TurnToPoint( player_next ).execute( agent );
}
