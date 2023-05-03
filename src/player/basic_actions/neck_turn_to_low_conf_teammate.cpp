// -*-c++-*-

/*!
  \file neck_turn_to_low_conf_teammate.cpp
  \brief check teammate player that has low confidence value
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

#include "neck_turn_to_low_conf_teammate.h"

#include "basic_actions.h"
#include "neck_scan_field.h"
#include "neck_turn_to_ball_or_scan.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/player/debug_client.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

using namespace rcsc;

//#define DEBUG

/*-------------------------------------------------------------------*/
/*!

 */
bool
Neck_TurnToLowConfTeammate::execute( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();
    const PlayerObject::Cont & teammates = wm.teammatesFromSelf();

    if ( teammates.empty() )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": no teammate" );
        return Neck_ScanField().execute( agent );
    }

    const double next_view_half_width = agent->effector().queuedNextViewWidth().width() * 0.5;
    const double next_neck_half_range = ServerParam::i().maxNeckAngle() + next_view_half_width;
    const Vector2D next_self_pos = agent->effector().queuedNextSelfPos();
    const AngleDeg next_self_body = agent->effector().queuedNextSelfBody();

    int pos_count = 1;
    Vector2D candidate_point( -52.5, 0.0 );
    int candidate_unum = 0;

    const double max_dist = 40.0;

    for ( const PlayerObject * t : teammates )
    {
        if ( t->isGhost() )
        {
            continue;
        }

        if ( t->distFromSelf() > max_dist )
        {
            break;
        }

        if ( t->posCount() >= pos_count
             && candidate_point.x < t->pos().x )
        {
            Vector2D pos = t->pos() + t->vel();
            AngleDeg angle = ( pos - next_self_pos ).th();

            if ( ( angle - next_self_body ).abs() < next_neck_half_range - 5.0 )
            {
                // can face
                candidate_unum = t->unum();
                candidate_point = pos;
                pos_count = t->posCount();
            }
        }
    }

    if ( pos_count <= 1 )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": no candidate. pos_count=%d",
                      pos_count );
        return Neck_TurnToBallOrScan( 0 ).execute( agent );
    }

    dlog.addText( Logger::ACTION,
                  __FILE__": look teammate %d (%.1f, %.1f). pos_count=%d",
                  candidate_unum,
                  candidate_point.x, candidate_point.y,
                  pos_count );

    //
    // search the best angle
    //

    AngleDeg player_angle = ( candidate_point - next_self_pos ).th();
    AngleDeg left_max_angle = player_angle - ( next_view_half_width - 15.0 );
    AngleDeg right_max_angle = player_angle + ( next_view_half_width - 15.0 );

    if ( left_max_angle.isLeftOf( next_self_body + ServerParam::i().minNeckAngle() ) )
    {
        left_max_angle = next_self_body + ServerParam::i().minNeckAngle();
    }

    if ( right_max_angle.isRightOf( next_self_body + ServerParam::i().maxNeckAngle() ) )
    {
        right_max_angle = next_self_body + ServerParam::i().maxNeckAngle();
    }

    dlog.addText( Logger::ACTION,
                  __FILE__": player_angle=%.0f  left_angle=%.0f  right_angle=%.0f",
                  player_angle.degree(),
                  left_max_angle.degree(),
                  right_max_angle.degree() );

    // count players in view cone

    const double angle_range = ( right_max_angle - left_max_angle ).abs();
    const double neck_step = angle_range / 10.0;

    double best_dir = -360.0;
    double best_score = -1.0;
    double best_angle_diff = 0.0;


    if ( wm.opponentsFromSelf().size() >= 11 )
    {
        for ( double dir = 0.0; dir < angle_range + 0.001; dir += neck_step )
        {
            AngleDeg target_angle = left_max_angle + dir;
            AngleDeg left = target_angle - next_view_half_width;
            AngleDeg right = target_angle + next_view_half_width;

            double score = 0.0;
            double angle_diff = 180.0;

            for ( const AbstractPlayerObject * p : wm.allPlayers() )
            {
                if ( p->isSelf() ) continue;
                //if ( p->isGhost() ) continue;
                Vector2D pos = p->pos() + p->vel();
                if ( pos.dist( next_self_pos ) > 35.0 ) continue; // XXX magic number XXX

                AngleDeg angle = ( pos - next_self_pos ).th();
                if ( angle.isRightOf( left )
                     && angle.isLeftOf( right ) )
                {
#if 1

                    int pos_count = p->seenPosCount();
                    if ( p->isGhost() ) pos_count = std::min( 2, pos_count );

                    double base_val = ( pos_count + 1 ) * 2.0;
                    //double rate = std::exp( - std::pow( p->distFromBall(), 2 )
                    double rate = std::exp( - std::pow( p->distFromSelf(), 2 )
                                            / ( 2.0 * std::pow( 40.0, 2 ) ) ); // Magic Number: variance = 40.0
                    score += base_val * rate;


#else
                    if ( p->goalie()
                         && next_self_pos.x > 33.0 )
                    {
                        score += p->posCount() * 2 + 20;
                    }
                    else if ( pos.x > next_self_pos.x - 5.0 )
                    {
                        score += p->posCount() * 2 + 1;
                    }
                    else
                    {
                        score += p->posCount();
                    }
#endif

                    double tmp_diff = std::min( ( angle - left ).abs(),
                                                ( angle - right ).abs() );
                    if ( tmp_diff < angle_diff )
                    {
                        angle_diff = tmp_diff;
                    }
                }
            }
#ifdef DEBUG
            dlog.addText( Logger::ACTION,
                          "__ dir=%.0f  score=%.1f angle_diff=%.1f",
                          target_angle.degree(),
                          score,
                          angle_diff );
#endif
            if ( score > best_score )
            {
                best_dir = target_angle.degree();
                best_score = score;
                best_angle_diff = angle_diff;
#ifdef DEBUG
                dlog.addText( Logger::ACTION,
                              "__ update best dir (1)" );
#endif
            }
            else if ( score == best_score )
            {
                if ( best_angle_diff < angle_diff )
                {
                    best_dir = target_angle.degree();
                    best_score = score;
                    best_angle_diff = angle_diff;
#ifdef DEBUG
                    dlog.addText( Logger::ACTION,
                                  "__ update best dir (2)" );
#endif
                }
            }
        }
    }

    AngleDeg best_angle = player_angle;
    if ( best_dir != -360.0 )
    {
        dlog.addText( Logger::ACTION,
                      __FILE__": changed target angle to max player count. player_angle=%.0f -> %.0f",
                      player_angle.degree(),
                      best_dir );
        best_angle = best_dir;
    }
    else
    {
        int left_sum_count = 0;
        int right_sum_count = 0;
        wm.dirRangeCount( left_max_angle,
                          next_view_half_width * 2.0,
                          NULL, &left_sum_count, NULL );
        wm.dirRangeCount( right_max_angle,
                          next_view_half_width * 2.0,
                          NULL, &right_sum_count, NULL );
        if ( left_sum_count < right_sum_count )
        {
            best_angle = left_max_angle;
            dlog.addText( Logger::ACTION,
                          __FILE__": changed target angle to left max. player_angle=%.0f -> %.0f",
                          player_angle.degree(),
                          left_max_angle.degree() );
        }
        else if ( right_sum_count < left_sum_count )
        {
            best_angle = right_max_angle;
            dlog.addText( Logger::ACTION,
                          __FILE__": changed target angle to right max. player_angle=%.0f -> %.0f",
                          player_angle.degree(),
                          right_max_angle.degree() );
        }
    }

    AngleDeg neck_moment = best_angle;
    neck_moment -= next_self_body; // relative to body
    neck_moment -= wm.self().neck(); // relative to face
    neck_moment = ServerParam::i().normalizeNeckMoment( neck_moment.degree() );

    agent->debugClient().addMessage( "LookOur%d", candidate_unum );
    return agent->doTurnNeck( neck_moment );
}
