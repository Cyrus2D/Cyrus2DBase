// -*-c++-*-

/*
 *Copyright:

 Copyright (C) Hidehisa AKIYAMA

 This code is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3, or (at your option)
 any later version.

 This code is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this code; see the file COPYING.  If not, write to
 the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.

 *EndCopyright:
 */

/////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "role_keepaway_keeper.h"

#include "strategy.h"

#include "planner/bhv_planned_action.h"

#include "basic_actions/body_go_to_point.h"
#include "basic_actions/body_intercept.h"
#include "basic_actions/body_turn_to_ball.h"

#include "basic_actions/neck_turn_to_ball_or_scan.h"
#include "basic_actions/neck_scan_field.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/player/intercept_table.h>

#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

using namespace rcsc;

const std::string RoleKeepawayKeeper::NAME( "KeepawayKeeper" );

/*-------------------------------------------------------------------*/
/*!

 */
namespace {
rcss::RegHolder role = SoccerRole::creators().autoReg( &RoleKeepawayKeeper::create,
                                                       RoleKeepawayKeeper::NAME );
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
RoleKeepawayKeeper::execute( PlayerAgent * agent )
{

    bool kickable = agent->world().self().isKickable();
    if ( agent->world().kickableTeammate()
         && agent->world().teammatesFromBall().front()->distFromBall()
         < agent->world().ball().distFromSelf() )
    {
        kickable = false;
    }

    if ( kickable )
    {
        doKick( agent );
    }
    else
    {
        doMove( agent );
    }

    return true;
}


/*-------------------------------------------------------------------*/
/*!

 */
void
RoleKeepawayKeeper::doKick( PlayerAgent * agent )
{
    if ( Bhv_PlannedAction().execute( agent ) )
    {
        dlog.addText( Logger::TEAM,
                      __FILE__": (execute) do planned action" );
        agent->debugClient().addMessage( "PlannedAction" );
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void
RoleKeepawayKeeper::doMove( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    const int self_min = wm.interceptTable().selfStep();
    const int mate_min = wm.interceptTable().teammateStep();

    if ( ! wm.kickableTeammate()
         && self_min <= mate_min + 1 )
    {
        Vector2D face_point( 0.0, 0.0 );
        Body_Intercept( true, face_point ).execute( agent );
        agent->setNeckAction( new Neck_TurnToBallOrScan( 0 ) );
        return;
    }

    const Vector2D home_pos = Strategy::i().getPosition( wm.self().unum() );


    if ( ! Body_GoToPoint( home_pos,
                           0.5,
                           ServerParam::i().maxDashPower() ).execute( agent ) )
    {
        Body_TurnToBall().execute( agent );
    }

    agent->setNeckAction( new Neck_TurnToBallOrScan( 0 ) );
}
