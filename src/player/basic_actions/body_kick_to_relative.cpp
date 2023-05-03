// -*-c++-*-

/*!
  \file body_kick_to_relative.cpp
  \brief kick the ball to the position relative to player agent.
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

#include "body_kick_to_relative.h"

#include "body_stop_ball.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/soccer_math.h>
#include <rcsc/math_util.h>

using namespace rcsc;

// #define DEBUG_PRINT
// #define CHECK_OPPONENT

namespace {

const size_t DEFAULT_KICK_QUEUE_MAX = 5;

#ifdef CHECK_OPPONENT
bool
exist_opponent( const WorldModel & wm,
                const Vector2D & ball_pos )
{
#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION,
                  __FILE__":(exit_opponent) ball_pos=(%.2f %.2f)",
                  ball_pos.x, ball_pos.y );
#endif
    for ( const PlayerObject * o : wm.opponentsFromSelf() )
    {
        if ( o->posCount() >= 4 ) continue;
        if ( o->distFromSelf() > 2.0 ) break;

        const PlayerType * ptype = o->playerTypePtr();
        const double control_area = ( o->goalie()
                                      && ball_pos.x > ServerParam::i().theirPenaltyAreaLineX()
                                      && ball_pos.absY() < ServerParam::i().penaltyAreaHalfWidth()
                                      ? ptype->maxCatchableDist() + 0.15
                                      : ptype->kickableArea() + 0.15 );
        const Vector2D opponent_pos = o->pos() + o->vel();

        if ( opponent_pos.dist2( ball_pos ) < std::pow( control_area, 2 ) )
        {
#ifdef DEBUG_PRINT
            dlog.addText( Logger::ACTION,
                          __FILE__":(exit_opponent) found opponent[%d](%.2f %.2f)",
                          o->unum(), o->pos().x, o->pos().y );
#endif
            return false;
        }
    }

    return true;
}
#endif

}

/*-------------------------------------------------------------------*/
/*!

*/
bool
Body_KickToRelative::execute( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    dlog.addText( Logger::ACTION,
                  __FILE__":(execute) dist=%.2f  rel_angle=%.1f",
                  M_target_dist, M_target_angle_relative.degree() );

    if ( ! wm.self().isKickable() )
    {
        std::cerr << __FILE__ << ": " << __LINE__
                  << " not ball kickable!"
                  << std::endl;
        dlog.addText( Logger::ACTION,
                      __FILE__":(execute)  not kickable" );
        return false;
    }

    if ( M_stop_ball )
    {
        if ( std::fabs( M_target_dist - wm.ball().distFromSelf() ) <= 0.1 )
        {
            AngleDeg ball_angle = wm.ball().angleFromSelf() - wm.self().body();
            if ( ( M_target_angle_relative - ball_angle ).abs() < 4.0 )
            {
                dlog.addText( Logger::ACTION,
                              __FILE__":(execute) already there. stop the ball" );
                return Body_StopBall().execute( agent );
            }
        }
    }

    Vector2D required_accel( 0.0, 0.0 );

    // simulate kick queue
    if ( ! simulate( wm, false, &required_accel ) // simulate near side rotation
         && ! simulate( wm, true, &required_accel ) ) // simulate far side rotation
    {
        dlog.addText( Logger::ACTION,
                      __FILE__":(execute) failed. stop the ball" );
        return Body_StopBall().execute( agent );
    }

    const double accel_radius = required_accel.r();
    const AngleDeg accel_angle = required_accel.th();

    const double kick_power = std::min( accel_radius / wm.self().kickRate(),
                                        ServerParam::i().maxPower() );
    dlog.addText( Logger::ACTION,
                  __FILE__": (execute) accel=(%.2f, %.2f) kick_power=%.3f kick_angle=%.1f",
                  required_accel.x, required_accel.y,
                  kick_power, accel_angle.degree() );

    if ( accel_radius < 0.02 )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__":(execute) accel is very small. not needed to kick." );
    }

    return agent->doKick( kick_power, accel_angle - wm.self().body() );
}

/*-------------------------------------------------------------------*/
/*!

*/
bool
Body_KickToRelative::simulate( const WorldModel & wm,
                               const bool reverse,
                               Vector2D * required_accel )
{
    // now, we do NOT consider about opponent players
    // it may be necessary to check opponents in high level decisions.

    const PlayerType & ptype = wm.self().playerType();

    const double ROTATE_DIST = ptype.playerSize()
        + ptype.kickableMargin() * 0.5
        + ServerParam::i().ballSize();

    std::list< Vector2D > subtarget_rpos;
    // subtarget_rpos.front() must be current ball relative pos (angle is global)
    // subtarget_rpos.back()  must be always target relative pos (angle is global)

    subtarget_rpos.push_back( ( wm.ball().rposCount() == 0
                                ? wm.ball().rpos()
                                : wm.ball().pos() - wm.self().pos() ) );
    subtarget_rpos.push_back( Vector2D::polar2vector( M_target_dist,
                                                      M_target_angle_relative
                                                      + wm.self().body() ) );


    /*-------------------------------------------------------------------*/
    // main loop
    bool success = false;
    do
    {
        /////////////////////////////////////////////////////////////
        // kick simulateion
        // consider relative environment. set my current pos to (0, 0)
        Vector2D ball_rpos = subtarget_rpos.front();
        Vector2D ball_vel = wm.ball().vel();
        Vector2D my_pos( 0.0, 0.0 ); // relative to current position
        Vector2D my_vel = wm.self().vel();
#ifdef CHECK_OPPONENT
        bool first = true;
#endif
        std::list< Vector2D >::iterator it = subtarget_rpos.begin();
        ++it;
        while ( it != subtarget_rpos.end() )
        {
            // current iterator's value is next sub-target.

            // TODO: opponent check

            Vector2D ball_pos = my_pos + ball_rpos;

            my_pos += my_vel;
            my_vel *= ptype.playerDecay();

            Vector2D next_ball_pos = my_pos + *it;

            Vector2D new_ball_vel = next_ball_pos - ball_pos;
            Vector2D accel = new_ball_vel - ball_vel;
            double ball_rel_dir = ( ball_rpos.th() - wm.self().body() ).degree();
            double krate = ptype.kickRate( ball_rpos.r(), ball_rel_dir );

            // cannot reach to sub-target
            if ( accel.r2() > std::pow( ServerParam::i().maxPower() * krate, 2 ) )
            {
                break;
            }
#ifdef CHECK_OPPONENT
            if ( first
                 && exist_opponent( wm, wm.self().pos() + next_ball_pos ) )
            {
                break;
            }
            first = false;
#endif
            ball_vel += accel;
            ball_vel *= ServerParam::i().ballDecay();
            ball_rpos = *it;
            ++it;
        }

        /////////////////////////////////////////////////////////////
        // can reach the final target point
        if ( it == subtarget_rpos.end() )
        {
            if ( M_stop_ball )
            {
                // check if I can stop the ball.
                double krate = ptype.kickRate( M_target_dist,
                                               M_target_angle_relative.degree() );
                double ball_speed = ball_vel.r();
                if ( ball_speed < ServerParam::i().maxPower() * krate )
                {
                    dlog.addText( Logger::ACTION,
                                  __FILE__": (simulate) success to search the stop kick."
                                  " subtarget size =%d",
                                  subtarget_rpos.size() - 2 );
                    success = true;
                    break;
                }
            }
            else
            {
                dlog.addText( Logger::ACTION,
                              __FILE__": (simulate) success to search rotate kick."
                              " subtarget size = %d",
                              subtarget_rpos.size() - 2 );
                success = true;
                break;
            }
        }

        /////////////////////////////////////////////////////////////
        // generate intermediate sub-target
        // iterator "it" points the failed sub-target.
        {
            std::list< Vector2D >::iterator prev_it = it;
            --prev_it;
            AngleDeg new_target_rel_angle = prev_it->th();
            double dir_diff = ( it->th() - new_target_rel_angle ).degree();
            if ( std::fabs( dir_diff ) < 2.0 )
            {
                // failure
                break;
            }
            new_target_rel_angle += dir_diff * 0.5;
            if ( reverse // this is reverse side search
                 && subtarget_rpos.size() == 2 ) // this is first search
            {
                new_target_rel_angle += 180.0;
            }

            subtarget_rpos.insert( it,
                                   Vector2D::polar2vector( ROTATE_DIST,
                                                           new_target_rel_angle ) );
        }

    }
    while ( subtarget_rpos.size() < DEFAULT_KICK_QUEUE_MAX );


    if ( success )
    {
        if ( required_accel )
        {
            std::list< Vector2D >::iterator second_it = subtarget_rpos.begin();
            ++second_it;
            *required_accel = ( (*second_it + wm.self().vel()) // rel to current
                                - subtarget_rpos.front()
                                - wm.ball().vel() ); // rel to current
        }
        return true;
    }
    return false;
}
