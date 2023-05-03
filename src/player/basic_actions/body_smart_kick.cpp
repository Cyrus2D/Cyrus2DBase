// -*-c++-*-

/*!
  \file body_smart_kick.cpp
  \brief smart kick action class source file.
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

#include "body_smart_kick.h"

#include "kick_table.h"

#include "body_stop_ball.h"
#include "body_hold_ball2008.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

#include <algorithm>

using namespace rcsc;

// #define DEBUG_PRINT

/*-------------------------------------------------------------------*/
/*!

*/
bool
Body_SmartKick::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::KICK,
                  __FILE__": Body_SmartKick" );

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
        dlog.addText( Logger::KICK,
                      __FILE__". unknown ball vel" );
        return Body_StopBall().execute( agent );
    }

    double first_speed = bound( 0.001, M_first_speed, ServerParam::i().ballSpeedMax() );
    double first_speed_thr = std::max( 0.0, M_first_speed_thr );
    int max_step = std::max( 1, M_max_step );

    if ( KickTable::instance().simulate( wm,
                                         M_target_point,
                                         first_speed,
                                         first_speed_thr,
                                         max_step,
                                         M_sequence )
         || M_sequence.speed_ >= first_speed_thr )
    {
        agent->debugClient().addMessage( "SmartKick%d", (int)M_sequence.pos_list_.size() );
#ifdef DEBUG_PRINT
        for ( const Vector2D & p : M_sequence.pos_list_ )
        {
            agent->debugClient().addCircle( p, 0.05 );
        }
#endif
        dlog.addText( Logger::KICK,
                      "(Body_SmartKick) Success! target=(%.2f %.2f) speed=%.3f speed_thr=%.3f max_step=%d",
                      M_target_point.x, M_target_point.y,
                      first_speed, first_speed_thr, max_step );

        dlog.addText( Logger::KICK,
                      "(Body_SmartKick) -> achieved_speed=%.3f power=%.2f actual_step=%d",
                      M_sequence.speed_,
                      M_sequence.power_,
                      (int)M_sequence.pos_list_.size() );

        Vector2D vel = M_sequence.pos_list_.front() - wm.ball().pos();
        Vector2D kick_accel = vel - wm.ball().vel();
        agent->doKick( kick_accel.r() / wm.self().kickRate(),
                       kick_accel.th() - wm.self().body() );
        return true;
    }


    //
    // TODO: force mode
    //


    // failed to search the kick sequence

    agent->debugClient().addMessage( "SmartKick.Hold" );
    dlog.addText( Logger::KICK,
                  "(Body_SmartKick) Failure! target=(%.2f %.2f) speed=%.3f speed_thr=%.3f max_step=%d",
                  M_target_point.x, M_target_point.y,
                  first_speed,
                  first_speed_thr, max_step );
    dlog.addText( Logger::KICK,
                  "(Body_SmartKick) -> speed=%.3f power=%.2f step=%d",
                  M_sequence.speed_,
                  M_sequence.power_,
                  (int)M_sequence.pos_list_.size() );

    Body_HoldBall2008( false, M_target_point, M_target_point ).execute( agent );
    return false;
}
