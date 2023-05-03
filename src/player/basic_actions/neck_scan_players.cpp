// -*-c++-*-

/*!
  \file neck_scan_players.cpp
  \brief scan players only by turn_neck
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

#include "neck_scan_players.h"

#include "basic_actions.h"
#include "neck_scan_field.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/player/intercept_table.h>
#include <rcsc/player/view_mode.h>
#include <rcsc/common/server_param.h>
#include <rcsc/common/logger.h>
#include <rcsc/geom/rect_2d.h>
#include <rcsc/geom/angle_deg.h>
#include <rcsc/geom/vector_2d.h>
#include <rcsc/game_time.h>

#include <algorithm>
#include <limits>
#include <cstdio>

using namespace rcsc;

// #define DEBUG_PRINT

//! invalid angle value
const double Neck_ScanPlayers::INVALID_ANGLE = -360.0;


/*-------------------------------------------------------------------*/
/*!

*/
bool
Neck_ScanPlayers::execute( PlayerAgent * agent )
{
    static GameTime s_last_calc_time( 0, 0 );
    static ViewWidth s_last_calc_view_width = ViewWidth::NORMAL;
    static double s_last_calc_min_neck_angle = 0.0;
    static double s_last_calc_max_neck_angle = 0.0;
    static double s_cached_target_angle = 0.0;

    if ( s_last_calc_time != agent->world().time()
         || s_last_calc_view_width != agent->effector().queuedNextViewWidth()
         || std::fabs( s_last_calc_min_neck_angle - M_min_neck_angle ) > 1.0e-3
         || std::fabs( s_last_calc_max_neck_angle - M_max_neck_angle ) > 1.0e-3 )
    {
        s_last_calc_time = agent->world().time();
        s_last_calc_view_width = agent->effector().queuedNextViewWidth();
        s_last_calc_min_neck_angle = M_min_neck_angle;
        s_last_calc_max_neck_angle = M_max_neck_angle;

#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (execute) call calcAngle()" );
#endif
        s_cached_target_angle = get_best_angle( agent,
                                                M_min_neck_angle,
                                                M_max_neck_angle );
    }

    if ( s_cached_target_angle == INVALID_ANGLE )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": (execute) envalid angle" );
        return Neck_ScanField().execute( agent );
    }

    AngleDeg target_angle = s_cached_target_angle;

    dlog.addText( Logger::ACTION,
                  __FILE__": (execute) target_angle=%.1f cached_value=%.1f",
                  target_angle.degree(), s_cached_target_angle );
    agent->debugClient().addMessage( "NeckScanPl" );

    agent->doTurnNeck( target_angle
                       - agent->effector().queuedNextSelfBody()
                       - agent->world().self().neck() );
    return true;
}

/*-------------------------------------------------------------------*/
/*!

*/
double
Neck_ScanPlayers::get_best_angle( const PlayerAgent * agent,
                                  const double & min_neck_angle,
                                  const double & max_neck_angle )
{
    const WorldModel & wm = agent->world();

    if ( wm.allPlayers().size() < 22 )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": player_size=%d < 22",
                      wm.allPlayers().size() );
#endif
        return INVALID_ANGLE;
    }

    const ServerParam & SP = ServerParam::i();

    const Vector2D next_self_pos = agent->effector().queuedNextSelfPos();
    const AngleDeg next_self_body = agent->effector().queuedNextSelfBody();
    const double view_width = agent->effector().queuedNextViewWidth().width();
    const double view_half_width = view_width * 0.5;
    const double neck_min = ( min_neck_angle == INVALID_ANGLE
                              ? SP.minNeckAngle()
                              : bound( SP.minNeckAngle(),
                                       min_neck_angle,
                                       SP.maxNeckAngle() ) );
    const double neck_max = ( max_neck_angle == INVALID_ANGLE
                              ? SP.maxNeckAngle()
                              : bound( SP.minNeckAngle(),
                                       max_neck_angle,
                                       SP.maxNeckAngle() ) );
    const double neck_step = std::max( 1.0, ( neck_max - neck_min ) / 36.0 );


#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION,
                  __FILE__": (get_best_angle) args min_neck_angle=%.1f max_neck_angle=%.1f",
                  min_neck_angle, max_neck_angle );
    dlog.addText( Logger::ACTION,
                  __FILE__": (get_best_angle) next_body=%.1f min_neck=%.1f max_neck=%.1f neck_step=%.1f",
                  next_self_body.degree(),
                  neck_min, neck_max,
                  neck_step );
#endif

    double best_dir = INVALID_ANGLE;
    double best_score = 0.0; //-std::numeric_limits< double >::max();

    for ( double dir = neck_min; dir < neck_max + 0.5; dir += neck_step )
    {
        const AngleDeg left_angle = next_self_body + ( dir - ( view_half_width - 0.01 ) );
        const AngleDeg right_angle = next_self_body + ( dir + ( view_half_width - 0.01 ) );

#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      "@ angle=%.0f (dir=%.0f): left=%.1f right=%.1f",
                      ( next_self_body + dir ).degree(),
                      dir,
                      left_angle.degree(), right_angle.degree() );
#endif

        double score = calculate_score( wm, next_self_pos, left_angle, right_angle );

        if ( score > best_score )
        {
            best_dir = dir;
            best_score = score;
#ifdef DEBUG_PRINT
            dlog.addText( Logger::ACTION,
                          ">>>> updated" );
#endif
        }

#ifdef DEBUG_PRINT
        Vector2D face_point = next_self_pos + Vector2D::from_polar( 20.0, next_self_body + dir );
        dlog.addLine( Logger::ACTION,
                      next_self_pos, face_point,
                      "#0000ff" );
        char buf[32];
        snprintf( buf, 32, "%f", score );
        dlog.addMessage( Logger::ACTION,
                         face_point,
                         buf );
#endif
    }

    if ( best_dir == INVALID_ANGLE
         || std::fabs( best_score ) < 1.0e-5 )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (get_best_angle) not found" );
#endif
        return INVALID_ANGLE;
    }


    AngleDeg angle = next_self_body + best_dir;
#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION,
                  __FILE__": (get_best_angle)  best_rel_dir=%.0f  angle=%.0f",
                  best_dir, angle.degree() );
    Vector2D face_point = next_self_pos + Vector2D::from_polar( 30.0, angle );
    dlog.addLine( Logger::ACTION,
                  next_self_pos, face_point,
                  "#ff0000" );
#endif
    return angle.degree();
}


/*-------------------------------------------------------------------*/
/*!

*/
double
Neck_ScanPlayers::calculate_score( const WorldModel & wm,
                                   const Vector2D & next_self_pos,
                                   const AngleDeg & left_angle,
                                   const AngleDeg & right_angle )
{
    double score = 0.0;
    double view_buffer = 90.0;

    const int our_min = std::min( wm.interceptTable().selfStep(),
                                  wm.interceptTable().teammateStep() );
    const int opp_min = wm.interceptTable().opponentStep();
    const bool our_ball = ( our_min <= opp_min );

    const AngleDeg reduced_left_angle = left_angle + 5.0;
    const AngleDeg reduced_right_angle = right_angle - 5.0;

    for ( const AbstractPlayerObject * p : wm.allPlayers() )
    {
        if ( p->isSelf() ) continue;

        Vector2D pos = p->pos() + p->vel();
        AngleDeg angle = ( pos - next_self_pos ).th();

        if ( ! angle.isRightOf( reduced_left_angle )
             || ! angle.isLeftOf( reduced_right_angle ) )
        {
            continue;
        }

        if ( p->ghostCount() >= 5 ) continue;

        double pos_count = p->seenPosCount();
        if ( p->isGhost()
             && p->ghostCount() % 2 == 1 )
        {
            pos_count = std::min( 2.0, pos_count );
        }
        pos_count += 1.0;

        if ( our_ball )
        {
            if ( p->side() == wm.ourSide()
                 && ( p->pos().x > wm.ball().pos().x - 10.0
                      || p->pos().x > 30.0 ) )
            {
                pos_count *= 2.0;
            }
        }

        double base_val = std::pow( pos_count, 2 );
        double rate = std::exp( - std::pow( p->distFromSelf(), 2 )
                                / ( 2.0 * std::pow( 20.0, 2 ) ) ); // Magic Number
        score += base_val * rate;

        double buf = std::min( ( angle - left_angle ).abs(),
                               ( angle - right_angle ).abs() );
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      "__ %c_%d (%.2f %.2f) count=%d base=%f rate=%f +%f buf=%.1f",
                      p->side() == LEFT ? 'L' : p->side() == RIGHT ? 'R' : 'N',
                      p->unum(),
                      p->pos().x, p->pos().y,
                      p->posCount(),
                      base_val, rate, base_val * rate,
                      buf );
#endif

        if ( buf < view_buffer )
        {
            view_buffer = buf;
        }
    }

    // The bigger view buffer, the bigger rate
    // range: [1.0:2.0]
    // double rate = 2.0 - std::exp( - std::pow( view_buffer, 2 )
    //                               / ( 2.0 * std::pow( 180.0, 2 ) ) ); // Magic Number
    double rate = 1.0 + view_buffer / 90.0;
#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION,
                  "base_score=%.1f view_buf=%.1f rate=%f -> %f",
                  score,
                  view_buffer,
                  rate,
                  score * rate );
#endif

    score *= rate;

    return score;

}
