// -*-c++-*-

/*!
  \file body_advance_ball2009.cpp
  \brief kick the ball to a forward direction
*/

/*
 *Copyright:

 Copyright (C) Hidehisa AKIYAMA

 This code is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.

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

#include "body_advance_ball2009.h"

#include "kick_table.h"

#include "body_smart_kick.h"
#include "body_kick_one_step.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/player/debug_client.h>

#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

using namespace rcsc;

GameTime Body_AdvanceBall2009::S_last_calc_time( 0, 0 );
AngleDeg Body_AdvanceBall2009::S_cached_best_angle = 0.0;


namespace {

/*-------------------------------------------------------------------*/
/*!

*/
double
calc_score( const PlayerAgent * agent,
            const AngleDeg & target_angle )
{
    static const double kickable_area = 1.2;

    double score = 1000.0;

    const Line2D angle_line( agent->world().self().pos(),
                             target_angle );

    const AngleDeg target_left_angle = target_angle - 30.0;
    const AngleDeg target_right_angle = target_angle + 30.0;

    for ( const PlayerObject * o : agent->world().opponentsFromSelf() )
    {
        if ( o->distFromBall() > 40.0 ) continue;

        if ( o->angleFromSelf().isWithin( target_left_angle, target_right_angle ) )
        {
            Vector2D project_point = angle_line.projection( o->pos() );
            double width = std::max( 0.0,
                                     angle_line.dist( o->pos() ) - kickable_area );
            double dist = agent->world().self().pos().dist( project_point );
            score *= width / dist;
        }
    }

    return score;
}

/*-------------------------------------------------------------------*/
/*!
  lower < uppeer
*/
AngleDeg
get_best_advance_angle( const PlayerAgent * agent,
                        const double & lower_angle,
                        const double & upper_angle,
                        const bool clear_mode )
{
    const double angle_range = upper_angle - lower_angle;
    // defalut angle step is 8.0
    const int ANGLE_DIVS
        = static_cast< int >( rint( angle_range / 8.0 ) ) + 1;
    const double angle_step = angle_range / ( ANGLE_DIVS - 1 );

    AngleDeg best_angle = 0.0;
    double best_score = 0.0;

    AngleDeg angle = lower_angle;

    for ( int i = 0;
          i < ANGLE_DIVS;
          ++i, angle += angle_step )
    {
        double score = calc_score( agent, angle );

        if ( clear_mode )
        {
            score *= ( 0.5
                       * ( AngleDeg::sin_deg( 1.5 * angle.abs() + 45.0 )
                           + 1.0) );
        }

        score *= std::pow( 0.95,
                           std::max( 0, agent->world().dirCount( angle ) - 3 ) );

        if ( score > best_score )
        {
            best_angle = angle;
            best_score = score;
        }

        dlog.addText( Logger::CLEAR,
                      __FILE__": get_best_angle. search_angle=%.0f, score=%f",
                      angle.degree(), score );
    }

    return best_angle;
}

}


/*-------------------------------------------------------------------*/
/*!
  execute action
*/
bool
Body_AdvanceBall2009::execute( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    dlog.addText( Logger::ACTION,
                  __FILE__": Body_AdvanceBall" );

    if ( ! wm.self().isKickable() )
    {
        std::cerr << __FILE__ << ": " << __LINE__
                  << " not ball kickable!"
                  << std::endl;
        dlog.addText( Logger::ACTION,
                      __FILE__": Body_AdvanceBall. not kickable" );
        return false;
    }

    if ( S_last_calc_time != wm.time() )
    {
        dlog.addText( Logger::CLEAR,
                      __FILE__": update" );
        S_cached_best_angle = getBestAngle( agent );
        S_last_calc_time = wm.time();
    }


    const Vector2D target_point
        = wm.self().pos()
        + Vector2D::polar2vector( 30.0, S_cached_best_angle );

    dlog.addText( Logger::CLEAR,
                  __FILE__": target_angle=%.1f",
                  S_cached_best_angle.degree() );
    agent->debugClient().setTarget( target_point );
    agent->debugClient().addLine( wm.ball().pos(), target_point );

    if ( wm.gameMode().type() != GameMode::PlayOn )
    {
        agent->debugClient().addMessage( "Advance" );
        // enforce one step kick
        Body_KickOneStep( target_point,
                          ServerParam::i().ballSpeedMax()
                          ).execute( agent );
        return true;
    }

    Vector2D one_step_vel
        // = Body_KickOneStep::get_max_possible_vel
        = KickTable::calc_max_velocity( ( target_point - wm.ball().pos() ).th(),
                                        wm.self().kickRate(),
                                        wm.ball().vel() );

    if ( one_step_vel.r() > ServerParam::i().ballSpeedMax() * 0.7 ) //if ( one_step_vel.r() > 2.1 )
    {
        Body_KickOneStep( target_point,
                          ServerParam::i().ballSpeedMax()
                          ).execute( agent );
        agent->debugClient().addMessage( "Advance1K" );
        return true;
    }

    agent->debugClient().addMessage( "Advance3K" );
    Body_SmartKick( target_point,
                    ServerParam::i().ballSpeedMax(),
                    ServerParam::i().ballSpeedMax() * 0.9,
                    3 ).execute( agent );
    return true;
}

/*-------------------------------------------------------------------*/
/*!

*/
AngleDeg
Body_AdvanceBall2009::getBestAngle( const PlayerAgent * agent )
{
    const BallObject & ball = agent->world().ball();

    const double max_dist = 30.0;

    Vector2D left_limit( 0.0,
                         -ServerParam::i().pitchHalfWidth() + 3.0 );
    double ydiff = std::fabs( left_limit.y - ball.pos().y );
    if ( ydiff < max_dist )
    {
        left_limit.x = ball.pos().x
            + std::sqrt( max_dist * max_dist - ydiff * ydiff );
    }

    if ( left_limit.x > 50.0 )
    {
        left_limit.x = 50.0;
    }

    Vector2D right_limit( 0.0,
                          ServerParam::i().pitchHalfWidth() - 5.0 );
    ydiff = std::fabs( right_limit.y - ball.pos().y );
    if ( ydiff < max_dist )
    {
        right_limit.x = ball.pos().x
            + std::sqrt( max_dist * max_dist - ydiff * ydiff );
    }
    if ( right_limit.x > 50.0 )
    {
        right_limit.x = 50.0;
    }

    double lower_angle = ( left_limit - ball.pos() ).th().degree();
    if ( lower_angle < -65.0 ) lower_angle = -65.0;

    double upper_angle = ( right_limit - ball.pos() ).th().degree();
    if ( upper_angle > 65.0 ) upper_angle = 65.0;

    if ( lower_angle > upper_angle )
    {
        dlog.addText( Logger::CLEAR,
                      __FILE__": getBestAngle. angle_error. lower=%.1f, uppser=%.1f",
                      lower_angle, upper_angle );
        return AngleDeg( 0.0 );
    }

    dlog.addText( Logger::CLEAR,
                  __FILE__": getBestAngle. left(%.1f %.1f)lower_angle=%.1f right(%.1f %.1f)upper_angle=%.1f",
                  left_limit.x, left_limit.y, lower_angle,
                  right_limit.x, right_limit.y, upper_angle );

    return get_best_advance_angle( agent,
                                   lower_angle, upper_angle,
                                   false );
}
