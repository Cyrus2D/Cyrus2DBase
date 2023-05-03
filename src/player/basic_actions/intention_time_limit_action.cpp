// -*-c++-*-

/*!
  \file intention_time_limit_action.cpp
  \brief generic queued behavior that has a time limit.
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

#include "intention_time_limit_action.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/player/soccer_action.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

using namespace rcsc;

/*-------------------------------------------------------------------*/
/*!

*/
IntentionTimeLimitAction::
IntentionTimeLimitAction( BodyAction * body_action,
                          NeckAction * neck_action,
                          ViewAction * view_action,
                          const long & max_step,
                          const GameTime & start_time )
    : M_body_action( body_action )
    , M_neck_action( neck_action )
    , M_view_action( view_action )
    , M_timer_count( max_step )
    , M_last_execute_time( start_time )
{

}

/*-------------------------------------------------------------------*/
/*!

*/
bool
IntentionTimeLimitAction::finished( const PlayerAgent * agent )
{
    if ( M_timer_count <= 0 )
    {
        dlog.addText( Logger::KICK,
                      "%s:%d: time limit intention finished"
                      ,__FILE__, __LINE__ );
        return true;
    }

    if ( M_last_execute_time.cycle() + 1 != agent->world().time().cycle() )
    {
        return true;
    }

    return false;
}

/*-------------------------------------------------------------------*/
/*!

*/
bool
IntentionTimeLimitAction::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::ACTION,
                  "%s:%d: execute. timer_count = %ld"
                  ,__FILE__, __LINE__,
                  M_timer_count );

    if ( M_timer_count <= 0 )
    {
        dlog.addText( Logger::ACTION,
                      "%s:%d: error. time is over "
                      ,__FILE__, __LINE__ );
        return false;
    }

    M_timer_count -= 1;
    M_last_execute_time = agent->world().time();

    bool done_body = false;
    bool done_neck = false;

    if ( M_body_action )
    {
        done_body = M_body_action->execute( agent );
    }

    if ( M_view_action )
    {
        M_view_action->execute( agent );
    }

    if ( M_neck_action )
    {
        done_neck = M_neck_action->execute( agent );
    }

    if ( ! done_body )
    {
        std::cerr << agent->config().teamName()
                  << ' ' << agent->world().self().unum()
                  << " IntentionTimeLimitAction. no body action "
                  << std::endl;
        dlog.addText( Logger::ACTION,
                      "%s:%d: error. no body action "
                      ,__FILE__, __LINE__ );
        return false;
    }

    if ( ! done_neck )
    {
        std::cerr << agent->config().teamName()
                  << ' ' << agent->world().self().unum()
                  << " IntentionTimeLimitAction. no neck action "
                  << std::endl;
        dlog.addText( Logger::ACTION,
                      "%s:%d: error. no neck action "
                      ,__FILE__, __LINE__ );
    }

    return true;
}
