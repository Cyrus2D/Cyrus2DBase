// -*-c++-*-

/*!
  \file body_clear_ball2009.cpp
  \brief kick the ball to escape from dangerous situation
*/

/*
 *Copyright:

 Copyright (C) Hiroki SHIMORA, Hidehisa AKIYAMA

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

#include "body_clear_ball2009.h"

#include "body_smart_kick.h"
#include "body_kick_one_step.h"

#include <rcsc/player/player_evaluator.h>
#include <rcsc/player/player_agent.h>
#include <rcsc/player/player_predicate.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/math_util.h>
#include <rcsc/timer.h>

#include <cmath>
#include <limits>

using namespace rcsc;

// #define DEBUG_PROFILE
// #define DEBUG_PRINT_RECURSIVE

namespace {

/*-------------------------------------------------------------------*/
/*!

 */
double
get_minimum_evaluation( const AbstractPlayerObject::Cont & container,
                        const PlayerEvaluator * evaluator )
{
    double min_value = std::numeric_limits< double >::max();

    for ( const AbstractPlayerObject * p : container )
    {
        double value = (*evaluator)( *p );

        if ( value < min_value )
        {
            min_value = value;
        }
    }

    delete evaluator;
    return min_value;
}

/*-------------------------------------------------------------------*/
/*!

 */
double
get_free_angle( const WorldModel & wm,
                const AngleDeg & angle )
{
    // return std::min( get_minimum_evaluation( wm.getPlayerCont( new OpponentOrUnknownPlayerPredicate( wm ) ),
    //                                          new AbsAngleDiffPlayerEvaluator( wm.ball().pos(), angle ) ),
    //                  360.0 );

    double min_diff = 360.0;
    for ( const AbstractPlayerObject * p : wm.theirPlayers() )
    {
        if ( p->distFromBall() > 35.0 ) continue;

        double angle_diff = ( angle - p->angleFromBall() ).abs();
        if ( angle_diff < min_diff )
        {
            min_diff = angle_diff;
        }
    }

    return min_diff;
}

/*-------------------------------------------------------------------*/
/*!

 */
AngleDeg
get_clear_course_recursive( const WorldModel & wm,
                            double safe_angle,
                            int recursive_count )

{
    const ServerParam & param = ServerParam::i();

    Vector2D clear_point;
    AngleDeg clear_dir;

    for ( double x = 50.0; x > 11.0; x -= 10.0 )
    {
        double y_sign = sign( wm.ball().pos().y );

        for ( int y = 0; y < 2; ++y )
        {
            clear_point.assign( wm.self().pos().x + x,
                                y_sign * ( param.pitchHalfWidth() - 2.0 ) );

            if ( wm.self().pos().dist2( clear_point ) >= 10.0 * 10.0 )
            {
                clear_dir = ( clear_point - wm.ball().pos() ).th();
#ifdef DEBUG_PRINT_RECURSIVE
                dlog.addText( Logger::CLEAR,
                              "clear recursive %d (1) angle=%.1f",
                              recursive_count, clear_dir.degree() );
#endif
                if ( get_free_angle( wm, clear_dir ) >= safe_angle )
                {
#ifdef DEBUG_PRINT_RECURSIVE
                    dlog.addLine( Logger::CLEAR,
                                  wm.self().pos(), clear_point, "#00F" );
                    dlog.addText( Logger::CLEAR,
                                  __FILE__" (get_clear_course_recursive) recursive %d safe_angle=%.1f point=(%.1f %.1f).angle=%.1f",
                                  recursive_count,
                                  safe_angle,
                                  clear_point.x, clear_point.y,
                                  clear_dir.degree() );
#endif
                    return clear_dir;
                }
            }

            if ( wm.ball().pos().absY() > 3.0 )
            {
                break;
            }

            y_sign *= -1.0;
        }
    }

    // search from 30 degree to 75 degree
    for ( double dir = 30.0; dir <= 75.0 + EPS; dir += 1.0 )
    {
        double y_sign = sign( wm.ball().pos().y );

        for ( int y = 0; y < 2; ++y )
        {
#ifdef DEBUG_PRINT_RECURSIVE
            dlog.addText( Logger::CLEAR,
                          "clear recursive %d (2) angle=%.1f",
                          recursive_count, y_sign * dir );
#endif
            if ( get_free_angle( wm, y_sign * dir ) >= safe_angle )
            {
#ifdef DEBUG_PRINT_RECURSIVE
                dlog.addLine( Logger::CLEAR,
                              wm.self().pos(),
                              Vector2D::from_polar( 20.0, AngleDeg( y_sign * dir ) ),
                              "#0F0" );
                dlog.addText( Logger::CLEAR,
                              __FILE__" (get_clear_course_recursive) recursive %d safe_angle=%.1f angle=%.1f",
                              recursive_count,
                              safe_angle,
                              y_sign * dir );
#endif
                return y_sign * dir;
            }

            if ( wm.ball().pos().absY() > 3.0 )
            {
                break;
            }

            y_sign *= -1.0;
        }
    }

    const double y_sign = sign( wm.ball().pos().y );

    // search from 30 degree to -30 degree
    for ( double dir = 30.0; dir >= -30.0 - EPS; dir -= 1.0 )
    {
#ifdef DEBUG_PRINT_RECURSIVE
        dlog.addText( Logger::CLEAR,
                      "clear recursive %d (3) angle=%.1f",
                      recursive_count, y_sign * dir );
#endif
        if ( get_free_angle( wm, y_sign * dir ) >= safe_angle )
        {
#ifdef DEBUG_PRINT_RECURSIVE
            dlog.addLine( Logger::CLEAR,
                          wm.self().pos(),
                          Vector2D::from_polar( 20.0, AngleDeg( y_sign * dir ) ),
                          "#0F0" );
            dlog.addText( Logger::CLEAR,
                          __FILE__" (get_clear_course_recursive) recursive %d safe_angle=%.1f angle=%.1f",
                          recursive_count,
                          safe_angle,
                          y_sign * dir );
#endif
            return y_sign * dir;
        }
    }

    if ( recursive_count > 0 )
    {
#ifdef DEBUG_PRINT_RECURSIVE
        dlog.addText( Logger::CLEAR,
                      __FILE__" (get_clear_course_recursive) recursive: %d",
                      recursive_count );
#endif
        return get_clear_course_recursive( wm,
                                           safe_angle * 0.7,
                                           recursive_count - 1 );
    }
    else
    {
        AngleDeg goal_away = ( wm.ball().pos() - param.ourTeamGoalPos() ).th();

        if ( std::fabs( wm.ball().pos().y + goal_away.sin() * 30.0 )
             < param.pitchHalfWidth() )
        {
            // goal away
#ifdef DEBUG_PRINT_RECURSIVE
            dlog.addLine( Logger::CLEAR,
                          wm.self().pos(),
                          Vector2D::from_polar( 20.0, goal_away ),
                          "#0FF" );
            dlog.addText( Logger::CLEAR,
                          __FILE__" (get_clear_course_recursive) goal_away" );
#endif
            return goal_away;
        }
        else
        {
            // beside line
#ifdef DEBUG_PRINT_RECURSIVE
            dlog.addText( Logger::CLEAR,
                          __FILE__" (get_clear_course_recursive) beside line" );
#endif
            if ( wm.self().pos().absY() <= 25.0 )
            {
#ifdef DEBUG_PRINT_RECURSIVE
                dlog.addLine( Logger::CLEAR,
                              wm.self().pos(),
                              Vector2D::from_polar( 20.0, AngleDeg( +y_sign * 3.0 ) ),
                              "#FF0" );
#endif
                return + y_sign * 3.0;
            }
            else
            {
#ifdef DEBUG_PRINT_RECURSIVE
                dlog.addLine( Logger::CLEAR,
                              wm.self().pos(),
                              Vector2D::from_polar( 20.0, AngleDeg( -y_sign * 3.0 ) ),
                              "#FF0" );
#endif
                return - y_sign * 3.0;
            }
        }
    }

    return 0.0;
}


/*-------------------------------------------------------------------*/
/*!

 */
AngleDeg
get_clear_course( const WorldModel & wm )
{
    static GameTime s_update_time( 0, 0 );
    static AngleDeg s_last_angle = 0.0;

    if ( s_update_time == wm.time() )
    {
        return s_last_angle;
    }
    s_update_time = wm.time();

#ifdef DEBUG_PROFILE
    Timer timer;
#endif
    s_last_angle = get_clear_course_recursive( wm,
                                               25.0, /* safe angle */
                                               4 /* recursive count */ );
#ifdef DEBUG_PROFILE
    dlog.addText( Logger::CLEAR,
                  __FILE__" (get_clear_course) elapsed %.3f [ms]",
                  timer.elapsedReal() );
#endif

    return s_last_angle;
}

}

/*-------------------------------------------------------------------*/
/*!

 */
Body_ClearBall2009::Body_ClearBall2009()
{

}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_ClearBall2009::execute( PlayerAgent * agent )
{
    // TODO:
    // . if result free cycles will be same steps, choice by free angle width
    // . check tackle

    const WorldModel & wm = agent->world();
    const ServerParam & param = ServerParam::i();


    // enforce kick
    if ( wm.kickableOpponent() )
    {
        agent->debugClient().addMessage( "ClearEnforce" );
        dlog.addText( Logger::CLEAR,
                      __FILE__" (execute) exist kickable opponent" );
        return Body_KickOneStep( wm.ball().pos() + Vector2D( 10.0, 0.0 ),
                                 param.ballSpeedMax() ).execute( agent );
    }

    AngleDeg clear_angle = get_clear_course( wm );
    Vector2D kick_target = wm.ball().pos() + Vector2D::polar2vector( 30.0, clear_angle );

    agent->debugClient().addLine( wm.ball().pos(), kick_target );

    if ( wm.gameMode().type() != GameMode::PlayOn
         || get_minimum_evaluation( wm.getPlayers( new OpponentOrUnknownPlayerPredicate( wm ) ),
                                    new DistFromPosPlayerEvaluator( wm.ball().pos() ) ) < 1.5 )
    {
        agent->debugClient().addMessage( "Clear1" );
        dlog.addText( Logger::CLEAR,
                      __FILE__" (execute) Clear 1 step kick. target=(%.1f %.1f)",
                      kick_target.x, kick_target.y );
        return Body_KickOneStep( kick_target,
                                 param.ballSpeedMax() ).execute( agent );
    }
    else
    {
        agent->debugClient().addMessage( "ClearS" );
        dlog.addText( Logger::CLEAR,
                      __FILE__" (execute) Clear smart kick. target=(%.1f %.1f)",
                      kick_target.x, kick_target.y );
        return Body_SmartKick( kick_target,
                               param.ballSpeedMax(),
                               std::max( 2.5, param.ballSpeedMax() * 0.85 ),
                               2 ).execute( agent );
    }
}
