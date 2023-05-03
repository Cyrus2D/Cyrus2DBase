// -*-c++-*-

/*!
  \file bhv_scan_field.cpp
  \brief behavior to scan the field evenly.
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

#include "bhv_scan_field.h"

#include "basic_actions.h"
#include "neck_scan_field.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

using namespace rcsc;

/*-------------------------------------------------------------------*/
/*!

*/
bool
Bhv_ScanField::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::ACTION,
                  __FILE__": Bhv_ScanField" );

    const WorldModel & wm = agent->world();

    if ( ! wm.self().posValid() )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": invalid my pos" );
        agent->doTurn( 60.0 );
        agent->setNeckAction( new Neck_TurnToRelative( 0.0 ) );
        return true;
    }

    if ( wm.ball().posValid() )
    {
//         agent->doTurn( 90.0 );
//         agent->setNeckAction( new Neck_ScanField() );
        scanAllField( agent );
        return true;
    }

    findBall( agent );
    return true;
}

/*-------------------------------------------------------------------*/
/*!

*/
void
Bhv_ScanField::findBall( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    if ( wm.seeTimeStamp() < wm.decisionTimeStamp() )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": (findBall) no see info after the previous decision" );
        agent->doTurn( 0.0 );
        agent->setNeckAction( new Neck_TurnToRelative( wm.self().neck() ) );
        return;
    }

    // change to WIDE view mode
    // and only face to last ball seen pos or reverse side
    if ( agent->effector().queuedNextViewWidth().type() != ViewWidth::WIDE )
    {
        //agent->doChangeView( ViewWidth::WIDE );
        agent->setViewAction( new View_Wide() );
    }

    Vector2D mynext
        = wm.self().pos()
        + wm.self().vel();
    AngleDeg face_angle
        = ( wm.ball().seenPos().isValid()
            ? ( wm.ball().seenPos() - mynext ).th()
            : ( -mynext ).th()  // face to field center
            );

    int search_flag = wm.ball().lostCount() / 3;
//         = ( agent->config().synchSee()
//             ? wm.ball().lostCount() / 3
//            : wm.ball().lostCount() / 2 );

    if ( search_flag % 2 )
    {
        face_angle += 180.0;
    }

    Vector2D face_point
        = mynext
        + Vector2D::polar2vector( 10.0, face_angle );

    dlog.addText( Logger::ACTION,
                  __FILE__": (findBall) lost_count=%d, search_flag=%d, angle=%.1f",
                  wm.ball().lostCount(),
                  search_flag,
                  face_angle.degree() );
    Bhv_NeckBodyToPoint( face_point ).execute( agent );
}

/*-------------------------------------------------------------------*/
/*!

*/
void
Bhv_ScanField::scanAllField( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    if ( wm.seeTimeStamp() < wm.decisionTimeStamp() )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__":scanAllField) no see info after the previous decision" );
        agent->doTurn( 0.0 );
        agent->setNeckAction( new Neck_TurnToRelative( wm.self().neck() ) );
        return;
    }

    if ( agent->effector().queuedNextViewWidth().type() != ViewWidth::WIDE )
    {
        agent->setViewAction( new View_Wide() );

        dlog.addText( Logger::ACTION,
                      __FILE__": (scanAllField)" );
    }

    AngleDeg turn_moment
        = ( wm.self().viewWidth().width()
            + agent->effector().queuedNextViewWidth().width() )
        * 0.5;

    agent->doTurn( turn_moment ); // current width
    agent->setNeckAction( new Neck_TurnToRelative( wm.self().neck() ) );
}
