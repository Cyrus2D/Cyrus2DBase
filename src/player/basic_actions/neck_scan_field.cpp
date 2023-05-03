// -*-c++-*-

/*!
  \file neck_scan_field.cpp
  \brief scan field with neck evenly
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

#include "neck_scan_field.h"

#include "basic_actions.h"
#include "neck_scan_players.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/geom/rect_2d.h>

#include <algorithm>
#include <numeric>
#include <deque>

using namespace rcsc;

// #define DEBUG_PRINT

const double Neck_ScanField::INVALID_ANGLE = -360.0;

/*-------------------------------------------------------------------*/
/*!

*/
bool
Neck_ScanField::execute( PlayerAgent * agent )
{
    static GameTime s_last_calc_time( 0, 0 );
    static ViewWidth s_last_calc_view_width = ViewWidth::NORMAL;
    static AngleDeg s_cached_target_angle = 0.0;

    const WorldModel & wm = agent->world();

    if ( s_last_calc_time == wm.time()
         && s_last_calc_view_width != agent->effector().queuedNextViewWidth() )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": (execute) cached angle=%.1f",
                      s_cached_target_angle.degree() );
        return agent->doTurnNeck( s_cached_target_angle
                                  - agent->effector().queuedNextSelfBody()
                                  - agent->world().self().neck() );


    }

    s_last_calc_time = agent->world().time();
    s_last_calc_view_width = agent->effector().queuedNextViewWidth();

    //
    // for wide mode
    //
    double angle = calcAngleForWidePitchEdge( agent );

    if ( angle != INVALID_ANGLE )
    {
        s_cached_target_angle = angle;

        dlog.addText( Logger::ACTION,
                      __FILE__": (execute) wide mode scan " );
        agent->debugClient().addMessage( "NeckScan:Wide" );

        agent->doTurnNeck( s_cached_target_angle
                           - agent->effector().queuedNextSelfBody()
                           - wm.self().neck() );
        return true;
    }

    //
    // try to maximize player accuracy
    //
    bool exist_ghost = false;
    for ( const AbstractPlayerObject * p : wm.allPlayers() )
    {
        if ( p->isGhost()
             && p->distFromSelf() < 30.0 )
        {
            dlog.addText( Logger::ACTION,
                          __FILE__": (execute) detect ghost player %c %d",
                          side_char( p->side() ), p->unum() );
            exist_ghost = true;
            break;
        }
    }

    if ( ! exist_ghost )
    {
        angle = Neck_ScanPlayers::get_best_angle( agent );

        if ( angle != INVALID_ANGLE )
        {
            s_cached_target_angle = angle;

            dlog.addText( Logger::ACTION,
                          __FILE__": (execute) scan players. target_angle=%.1f", angle );
            agent->debugClient().addMessage( "NeckScan:Pl" );

            agent->doTurnNeck( s_cached_target_angle
                               - agent->effector().queuedNextSelfBody()
                               - agent->world().self().neck() );
            return true;
        }
    }

    //
    // try to look the lowest accuracy area
    //
    const bool consider_pitch = ( wm.gameMode().type() == GameMode::PlayOn
                                  || ( wm.gameMode().type() != GameMode::IndFreeKick_
                                       && wm.gameMode().type() != GameMode::BackPass_
                                       && wm.ball().distFromSelf() < wm.self().playerType().playerSize() + 0.15 )
                                  );
    angle = calcAngleDefault( agent, consider_pitch );

    if ( consider_pitch
         && ( AngleDeg( angle ) - wm.self().face() ).abs() < 5.0 )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (calcAngle) face angle not changed. re-calculate without pitch" );
#endif
        angle = calcAngleDefault( agent, false );
    }

    s_cached_target_angle = angle;

    dlog.addText( Logger::ACTION,
                  __FILE__": (execute) target_angle=%.1f",
                  s_cached_target_angle.degree() );
    agent->debugClient().addMessage( "NeckScan" );

    agent->doTurnNeck( s_cached_target_angle
                       - agent->effector().queuedNextSelfBody()
                       - agent->world().self().neck() );
    return true;
}

/*-------------------------------------------------------------------*/
/*!

*/
double
Neck_ScanField::calcAngleDefault( const PlayerAgent * agent,
                                  const bool consider_pitch )
{
    const ServerParam & SP = ServerParam::i();
    const Rect2D pitch_rect
        = SP.keepawayMode()
        ? Rect2D( Vector2D( - SP.keepawayLength() * 0.5,
                            - SP.keepawayWidth() * 0.5 ),
                  Size2D( SP.keepawayLength(),
                          SP.keepawayWidth() ) )
        : Rect2D( Vector2D( - SP.pitchHalfLength(),
                            - SP.pitchHalfWidth() ),
                  Size2D( SP.pitchLength(),
                          SP.pitchWidth() ) );
    const Rect2D expand_pitch_rect
        = SP.keepawayMode()
        ? Rect2D( Vector2D( - SP.keepawayLength() * 0.5,
                            - SP.keepawayWidth() * 0.5 ),
                  Size2D( SP.keepawayLength(),
                          SP.keepawayWidth() ) )
        : Rect2D( Vector2D( - SP.pitchHalfLength() - 3.0,
                            - SP.pitchHalfWidth() - 3.0 ),
                  Size2D( SP.pitchLength() + 6.0,
                          SP.pitchWidth() + 6.0 ) );
    const Rect2D goalie_rect
        = SP.keepawayMode()
        ? Rect2D( Vector2D( 10000.0, 10000.0 ),
                  Size2D( 0.1, 0.1 ) ) // dummy rect
        : Rect2D( Vector2D( SP.pitchHalfLength() - 3.0, -15.0 ),
                  Size2D( 10.0, 30.0 ) );

    const WorldModel & wm = agent->world();

    const double next_view_width = agent->effector().queuedNextViewWidth().width();

    const AngleDeg left_start
        = agent->effector().queuedNextSelfBody()
        + ( ServerParam::i().minNeckAngle() - ( next_view_width * 0.5 ) );
    const double scan_range
        = ( ( ServerParam::i().maxNeckAngle() - ServerParam::i().minNeckAngle() )
            + next_view_width );

#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION,
                  __FILE__": (calcAngleDefault) next_left_limit=%.0f, next_neck_range=%.0f",
                  left_start.degree(), scan_range );
#endif

    //const double shrinked_scan_range = scan_range - WorldModel::DIR_STEP * 1.5;
    const double shrinked_next_view_width = next_view_width - WorldModel::DIR_STEP * 1.5;

    AngleDeg sol_angle = left_start + scan_range * 0.5;

    if ( scan_range < shrinked_next_view_width )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (calcAngleDefault) scan reange is smaller than next view width." );
#endif
        return sol_angle.degree();
    }


    AngleDeg tmp_angle = left_start;

    const std::size_t size_of_view_width
        = static_cast< std::size_t >
        ( rint( shrinked_next_view_width / WorldModel::DIR_STEP ) );

    std::deque< int > dir_counts( size_of_view_width );

    // generate first visible cone score list
    {

        for ( std::deque< int >::iterator it = dir_counts.begin(), end = dir_counts.end();
              it != end;
              ++it )
        {
            *it = wm.dirCount( tmp_angle );
            tmp_angle += WorldModel::DIR_STEP;
        }
    }

    int max_count_sum = 0;
    double add_dir = shrinked_next_view_width;

#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION,
                  __FILE__": (calcAngleDefault) loop start. left_start=%.0f shrinked_scan_range=%.0f",
                  left_start.degree(),
                  shrinked_scan_range );
#endif

    const Vector2D my_next = agent->effector().queuedNextSelfPos();

    do
    {
        int tmp_count_sum = std::accumulate( dir_counts.begin(), dir_counts.end(), 0 );

        AngleDeg angle = tmp_angle - shrinked_next_view_width * 0.5;
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      "%.0f: count_sum=%d",
                      angle.degree(),
                      tmp_count_sum );
#endif
        if ( tmp_count_sum > max_count_sum )
        {
            bool update = true;
            if ( consider_pitch )
            {
                {
                    Vector2D face_point
                        = my_next
                        + Vector2D::polar2vector( 20.0, angle );
                    if ( ! pitch_rect.contains( face_point )
                         && ! goalie_rect.contains( face_point ) )
                    {
                        update = false;
                    }
                }

                if ( update )
                {
                    Vector2D left_face_point
                        = my_next
                        + Vector2D::polar2vector( 20.0, angle - next_view_width*0.5 );
                    if ( ! expand_pitch_rect.contains( left_face_point )
                         && ! goalie_rect.contains( left_face_point ) )
                    {
                        update = false;
                    }
                }

                if ( update )
                {
                    Vector2D right_face_point
                        = my_next
                        + Vector2D::polar2vector( 20.0, angle + next_view_width*0.5 );
                    if ( ! expand_pitch_rect.contains( right_face_point )
                         && ! goalie_rect.contains( right_face_point ) )
                    {
                        update = false;
                    }
                }
            }

            if ( update )
            {
#ifdef DEBUG_PRINT
                dlog.addText( Logger::ACTION,
                              "--> updated" );
#endif
                sol_angle = angle;
                max_count_sum = tmp_count_sum;
            }
        }

        dir_counts.pop_front();
        add_dir += WorldModel::DIR_STEP;
        tmp_angle += WorldModel::DIR_STEP;
        dir_counts.push_back( wm.dirCount( tmp_angle ) );
    }
    while ( add_dir <= scan_range );

#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION,
                  __FILE__": (calcAngleDefault) best_angle=%.0f",
                  sol_angle.degree() );
#endif
    return sol_angle.degree();
}

/*-------------------------------------------------------------------*/
/*!

*/
double
Neck_ScanField::calcAngleForWidePitchEdge( const PlayerAgent * agent )
{
    if ( agent->effector().queuedNextViewWidth().type() != ViewWidth::WIDE )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (calcAngleForWidePitchEdge) no wide view" );
#endif
        return INVALID_ANGLE;
    }

    const WorldModel & wm = agent->world();

    if ( wm.gameMode().type() != GameMode::PlayOn
         && wm.gameMode().type() != GameMode::GoalKick_
         && wm.ball().distFromSelf() > 2.0 )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (calcAngleForWidePitchEdge) set play" );
#endif
        return INVALID_ANGLE;
    }

    const ServerParam & SP = ServerParam::i();

    if ( SP.keepawayMode() )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (calcAngleForWidePitchEdge) keepaway" );
#endif
        return INVALID_ANGLE;
    }

    const Vector2D next_self_pos = wm.self().pos() + wm.self().vel();
    const double pitch_x_thr = SP.pitchHalfLength() - 15.0;
    const double pitch_y_thr = SP.pitchHalfLength() - 10.0;

    double target_angle = INVALID_ANGLE;

    if ( next_self_pos.absY() > pitch_y_thr )
    {
        Vector2D target_pos( SP.pitchHalfLength() - 7.0, 0.0 );
        target_pos.x = std::min( target_pos.x, target_pos.x * 0.7 * next_self_pos.x * 0.3 );

        if ( next_self_pos.y > +pitch_y_thr )
        {
            target_angle = ( target_pos - next_self_pos ).th().degree();
#ifdef DEBUG_PRINT
            dlog.addText( Logger::ACTION,
                          __FILE__": (calcAngleForWidePitchEdge) y+ point=(%.1f %.1f) dir=%.1f",
                          target_pos.x, target_pos.y,
                          target_angle );
#endif
        }

        if ( next_self_pos.y < -pitch_y_thr )
        {
            target_angle = ( target_pos - next_self_pos ).th().degree();
#ifdef DEBUG_PRINT
            dlog.addText( Logger::ACTION,
                          __FILE__": (calcAngleForWidePitchEdge) y- point=(%.1f %.1f) dir=%.1f",
                          target_pos.x, target_pos.y,
                          target_angle );
#endif
        }
    }

    if ( next_self_pos.absX() > pitch_x_thr )
    {
        Vector2D target_pos( SP.pitchHalfLength() * 0.5, 0.0 );

        if ( next_self_pos.x > + pitch_x_thr )
        {
            target_angle = ( target_pos - next_self_pos ).th().degree();
#ifdef DEBUG_PRINT
            dlog.addText( Logger::ACTION,
                          __FILE__": (calcAngleForWidePitchEdge) x+ point=(%.1f %.1f) dir=%.1f",
                          target_pos.x, target_pos.y,
                          target_angle );
#endif
        }

        if ( next_self_pos.x < - pitch_x_thr )
        {
            target_angle = ( target_pos - next_self_pos ).th().degree();
#ifdef DEBUG_PRINT
            dlog.addText( Logger::ACTION,
                          __FILE__": (calcAngleForWidePitchEdge) x- point=(%.1f %.1f) dir=%.1f",
                          target_pos.x, target_pos.y,
                          target_angle );
#endif
        }
    }

#ifdef DEBUG_PRINT
    if ( target_angle == INVALID_ANGLE )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": (calcAngleForWidePitchEdge) no target" );
    }
    else
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": (calcAngleForWidePitchEdge) best_angle=%.1f",
                      target_angle );
    }
#endif

    return target_angle;
}
