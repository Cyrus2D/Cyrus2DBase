// -*-c++-*-

/*!
  \file body_go_to_point.cpp
  \brief run behavior which has target point.
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

#include "body_go_to_point.h"

#include "basic_actions.h"
#include "body_stop_dash.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/geom/circle_2d.h>
#include <rcsc/geom/segment_2d.h>
#include <rcsc/geom/matrix_2d.h>
#include <rcsc/soccer_math.h>
#include <rcsc/math_util.h>

using namespace rcsc;

// #define USE_OMNI_DASH_2012

#define DEBUG_PRINT

namespace {
const double ILLEGAL_POWER = -65535.0;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_GoToPoint::execute( PlayerAgent * agent )
{
#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION,
                  __FILE__": (execute) target=(%.2f %.2f) max_power=%.3f speed=%.3f dist_thr=%.3f",
                  M_target_point.x, M_target_point.y,
                  M_max_dash_power, M_dash_speed,
                  M_dist_thr );
#endif

    if ( std::fabs( M_max_dash_power ) < 0.1
         || std::fabs( M_dash_speed ) < 0.001 )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": max_dash_power=%f dash_speed=%f, turn only",
                      M_max_dash_power, M_dash_speed );
#endif
        agent->doTurn( 0.0 );
        return false;
    }

    const WorldModel & wm = agent->world();

    const Vector2D inertia_point = wm.self().inertiaPoint( M_cycle );
    Vector2D target_rel = M_target_point - inertia_point;

    //
    // already there
    //
    double target_dist = target_rel.r();
    if ( target_dist < M_dist_thr )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": already there. inertia_point_dist=%.3f < dist_thr=%.3f",
                      target_dist, M_dist_thr );
#endif
        agent->doTurn( 0.0 ); // dumy action
        return false;
    }

    //
    // if necessary, change the target point to avoid goal post
    //
    checkGoalPost( agent );

    //
    // omnidir dash
    //
    if ( doOmniDash( agent ) )
    {
        return true;
    }

    //
    // turn
    //
    if ( doTurn( agent ) )
    {
        return true;
    }

    //
    // dash
    //
    if ( doDash( agent ) )
    {
        return true;
    }

    agent->doTurn( 0.0 ); // dummy action
    return false;
}

/*-------------------------------------------------------------------*/
/*!

 */
void
Body_GoToPoint::checkGoalPost( const PlayerAgent * agent )
{
    const ServerParam & SP = ServerParam::i();
    const WorldModel & wm = agent->world();

    const double collision_dist = wm.self().playerType().playerSize() + SP.goalPostRadius() + 0.2;

    const Vector2D goal_post_l( -SP.pitchHalfLength() + SP.goalPostRadius(),
                                -SP.goalHalfWidth() - SP.goalPostRadius() );
    const Vector2D goal_post_r( -SP.pitchHalfLength() + SP.goalPostRadius(),
                                +SP.goalHalfWidth() + SP.goalPostRadius() );

    const double dist_post_l2 = wm.self().pos().dist2( goal_post_l );
    const double dist_post_r2 = wm.self().pos().dist2( goal_post_r );

    const Vector2D nearest_post = ( dist_post_l2 < dist_post_r2
                                    ? goal_post_l
                                    : goal_post_r );
    const double dist_post = std::sqrt( std::min( dist_post_l2, dist_post_r2 ) );

#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION,
                  __FILE__": (checkGoalPost) collision_dist=%f dist_post=%f",
                  collision_dist, dist_post );
#endif

    if ( dist_post > collision_dist + wm.self().playerType().realSpeedMax() + 0.5 )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (checkGoalPost) far from goal post" );
#endif
        return;
    }

    const Circle2D post_circle( nearest_post, collision_dist );
    const Segment2D move_line( wm.self().pos(), M_target_point );

    if ( post_circle.intersection( move_line, NULL, NULL ) == 0 )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (checkGoalPost) no intersection" );
#endif
        return;
    }

    const AngleDeg post_angle = ( nearest_post - wm.self().pos() ).th();
    const AngleDeg target_angle = ( M_target_point - wm.self().pos() ).th();

    Vector2D new_target = nearest_post;

    if ( post_angle.isLeftOf( target_angle ) )
    {
        new_target += Vector2D::from_polar( collision_dist + 5.0, post_angle + 90.0 );
    }
    else
    {
        new_target += Vector2D::from_polar( collision_dist + 5.0, post_angle - 90.0 );
    }

#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION,
                  __FILE__": adjust to avoid goal post. (%.2f %.2f) -> (%.2f %.2f)",
                  M_target_point.x, M_target_point.y,
                  new_target.x, new_target.y );
    dlog.addRect( Logger::ACTION,
                  new_target.x - 0.1, new_target.y - 0.1,
                  0.2, 0.2,
                  "#ff0000", true );
#endif

    M_target_point = new_target;
}

#ifdef USE_OMNI_DASH_2012
/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_GoToPoint::doOmniDash( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    const Vector2D inertia_point = wm.self().inertiaPoint( M_cycle );

    if ( inertia_point.dist2( M_target_point ) > std::pow( M_dist_thr + M_omni_dash_dist_thr, 2 ) )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (doOmniDash) over adjustable distance. %f",
                      inertia_point.dist( M_target_point ) );
#endif
        return false;
    }

    const Matrix2D rotate_matrix = Matrix2D::make_rotation( -wm.self().body() );

    const Vector2D target_rel = rotate_matrix.transform( M_target_point - inertia_point );

    if ( target_rel.absY() < M_dist_thr )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (doOmniDash) target_y_diff=%.3f, omni dash is not required.",
                      target_rel.y );
#endif
        return false;
    }

    const AngleDeg target_angle = target_rel.th();

    if ( target_angle.abs() < M_dir_thr )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (doOmniDash) target_angle=%.3f dir_thr=%.3f omni dash is not required.",
                      target_angle.degree(), M_dir_thr );
#endif
        return false;
    }

    const ServerParam & SP = ServerParam::i();
    const PlayerType & ptype = wm.self().playerType();

    const double dash_angle_step = std::max( 15.0, SP.dashAngleStep() );
    const double min_dash_angle = ( -180.0 < SP.minDashAngle() && SP.maxDashAngle() < 180.0
                                    ? SP.minDashAngle()
                                    : dash_angle_step * static_cast< int >( -180.0 / dash_angle_step ) );
    const double max_dash_angle = ( -180.0 < SP.minDashAngle() && SP.maxDashAngle() < 180.0
                                    ? SP.maxDashAngle() + dash_angle_step * 0.5
                                    : dash_angle_step * static_cast< int >( 180.0 / dash_angle_step ) - 1.0 );

    const int max_step = std::min( 3, M_cycle );


    Vector2D self_pos( 0.0, 0.0 );
    Vector2D self_vel = rotate_matrix.transform( wm.self().vel() );
    StaminaModel stamina_model = wm.self().staminaModel();

    std::vector< double > result_dash_powers;
    result_dash_powers.reserve( 4 );
    std::vector< double > result_dash_dirs;
    result_dash_dirs.reserve( 4 );
#ifdef DEBUG_PRINT
    std::vector< Vector2D > result_self_pos;
    result_self_pos.reserve( 4 );
#endif

    for ( int step = 0; step < max_step; ++step )
    {
        if ( std::fabs( target_rel.y - self_pos.y ) < 0.001 )
        {
            break;
        }

        const Vector2D required_vel = ( target_rel - self_pos )
            * ( ( 1.0 - ptype.playerDecay() )
                / ( 1.0 - std::pow( ptype.playerDecay(), M_cycle - step ) ) );
        const Vector2D required_accel = required_vel - self_vel;

        const double max_dash_power = ( M_save_recovery
                                        ? stamina_model.getSafetyDashPower( ptype, M_max_dash_power )
                                        : std::fabs( M_max_dash_power ) );

#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      "(doOmniDash_Test) step=%d self_pos=(%.2f %.2f) vel=(%.2f %.2f)",
                      step, self_pos.x, self_pos.y, self_vel.x, self_vel.y );
        dlog.addText( Logger::ACTION,
                      "required_vel=(%.2f %.2f) accel=(%.2f %.2f) max_power=%.2f",
                      step,
                      required_vel.x, required_vel.y,
                      required_accel.x, required_accel.y,
                      max_dash_power );
#endif

        double min_dist2 = 1000000.0;
        Vector2D best_self_pos = Vector2D::INVALIDATED;
        Vector2D best_self_vel;
        double best_dash_power = 0.0;
        double best_dash_dir = 0.0;
        for ( double dir = min_dash_angle;
              dir < max_dash_angle;
              dir += dash_angle_step )
        {
            if ( std::fabs( dir ) > 100.0 ) continue; // Magic Number

            const AngleDeg dash_angle = SP.discretizeDashAngle( dir );
            const double dash_rate = ptype.dashPowerRate() * stamina_model.effort() * SP.dashDirRate( dir );
            const Vector2D required_accel_rel = required_accel.rotatedVector( -dash_angle );

            const double dash_power = bound( 0.0, required_accel_rel.x / dash_rate, max_dash_power );
            const Vector2D dash_accel = Vector2D::polar2vector( dash_rate * dash_power, dash_angle );

            const Vector2D tmp_self_vel = self_vel + dash_accel;
            const Vector2D tmp_self_pos = self_pos + tmp_self_vel;
            const double d2 = tmp_self_pos.dist2( target_rel );

#ifdef DEBUG_PRINT
            dlog.addText( Logger::ACTION,
                          "__ %d: dir=%.1f drate=%f dpower=%.2f self=(%.2f %.2f) move_dist=%.3f",
                          step, dir, dash_rate, dash_power, tmp_self_pos.x, tmp_self_pos.y, std::sqrt( d2 ) );
#endif
            if ( d2 < min_dist2 )
            {
#ifdef DEBUG_PRINT
                dlog.addText( Logger::ACTION,
                              "== updated" );
#endif
                min_dist2 = d2;
                best_self_pos = tmp_self_pos;
                best_self_vel = tmp_self_vel;
                best_dash_power = dash_power;
                best_dash_dir = dir;
            }
        }

        self_pos = best_self_pos;
        self_vel = best_self_vel;
        self_vel *= ptype.playerDecay();
        stamina_model.simulateDash( ptype, best_dash_power );

        result_dash_powers.push_back( best_dash_power );
        result_dash_dirs.push_back( best_dash_dir );
#ifdef DEBUG_PRINT
        result_self_pos.push_back( self_pos );
#endif
    }

    if ( ! result_dash_powers.empty() )
    {
        AngleDeg dash_dir = SP.discretizeDashAngle( result_dash_dirs.front() );
#ifdef DEBUG_PRINT
        agent->debugClient().addMessage( "OmniDash%.0f", dash_dir.degree() );
        dlog.addText( Logger::ACTION,
                      __FILE__": (doOmniDash) power=%.3f dir=%.1f",
                      result_dash_powers.front(), dash_dir.degree() );
        for ( size_t i = 0; i < result_self_pos.size(); ++i )
        {
            Vector2D pos = inertia_point + result_self_pos[i].rotate( wm.self().body() );
            char msg[8]; snprintf( msg, 8, "%zd", i + 1 );
            dlog.addCircle( Logger::ACTION,
                            pos, 0.3, "#00F", false );
            dlog.addMessage( Logger::ACTION,
                             pos, msg, "#00F" );
            dlog.addText( Logger::ACTION,
                          "result : step=%zd power=%.3f dir=%.1f",
                          i + 1, result_dash_powers[i], result_dash_dirs[i] );

        }
#endif
        return agent->doDash( result_dash_powers.front(), dash_dir );
    }

    return false;
}

#else

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_GoToPoint::doOmniDash( PlayerAgent * agent )
{
    const ServerParam & SP = ServerParam::i();
    const WorldModel & wm = agent->world();

    const Vector2D inertia_point = wm.self().inertiaPoint( M_cycle );
    Vector2D target_rel = M_target_point - inertia_point;

    if ( target_rel.r2() > std::pow( M_dist_thr + M_omni_dash_dist_thr, 2 ) )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (doOmniDash) over adjustable distance. %.3f  dist_thr=%.3f omni_thr=%.3f",
                      target_rel.r(), M_dist_thr, M_omni_dash_dist_thr );
#endif
        return false;
    }

    target_rel.rotate( - wm.self().body() );

    if ( target_rel.absY() < M_dist_thr )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (doOmniDash) target_y_diff=%.3f, omni dash is not required.",
                      target_rel.y );
#endif
        return false;
    }

    const AngleDeg target_angle = target_rel.th();

    if ( target_angle.abs() < M_dir_thr )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (doOmniDash) target_angle=%.3f dir_thr=%.3f omni dash is not required.",
                      target_angle.degree(), M_dir_thr );
#endif
        return false;
    }

    const Vector2D rel_vel = wm.self().vel().rotatedVector( - wm.self().body() );

    const double dash_angle_step = std::max( 15.0, SP.dashAngleStep() );
    const double min_dash_angle = ( -180.0 < SP.minDashAngle() && SP.maxDashAngle() < 180.0
                                    ? SP.minDashAngle()
                                    : dash_angle_step * static_cast< int >( -180.0 / dash_angle_step ) );
    const double max_dash_angle = ( -180.0 < SP.minDashAngle() && SP.maxDashAngle() < 180.0
                                    ? SP.maxDashAngle() + dash_angle_step * 0.5
                                    : dash_angle_step * static_cast< int >( 180.0 / dash_angle_step ) - 1.0 );

    double best_dir = -360.0;
    double best_dist = 1000000.0;
    int best_cycle = 1000;
    double best_dash_power = 0.0;
    double best_stamina = 0.0;

    for ( double dir = min_dash_angle;
          dir < max_dash_angle;
          dir += dash_angle_step )
    {
        if ( std::fabs( dir ) < 0.5 ) continue;
        if ( std::fabs( dir ) > 100.0 ) continue; // Magic Number

        const AngleDeg dash_angle = SP.discretizeDashAngle( dir );

        if ( ( dash_angle - target_angle ).abs() > 90.0 )
        {
#ifdef DEBUG_PRINT
            dlog.addText( Logger::ACTION,
                          "__ dir=%.1f, invalid direction. continue",
                          dash_angle.degree() );
#endif
            continue;
        }

        const Matrix2D rotate_matrix = Matrix2D::make_rotation( -dash_angle );
        const double dash_rate = wm.self().dashRate() * SP.dashDirRate( dir );

#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      "__ dir=%.1f, dash_rate=%f",
                      dash_angle.degree(), dash_rate );
#endif
        //
        // check if player can adjust y diff with few dashes.
        //

        const int max_cycle = std::min( 3, M_cycle );

        Vector2D my_pos( 0.0, 0.0 );
        Vector2D my_vel = rel_vel;
        StaminaModel stamina_model = wm.self().staminaModel();
        double first_dash_power = ILLEGAL_POWER;

        my_pos = wm.self().playerType().inertiaPoint( my_pos, rel_vel, M_cycle );

        int cycle = 0;
        for ( ; cycle < max_cycle; ++cycle )
        {
            Vector2D required_move = rotate_matrix.transform( target_rel - my_pos );
            double required_x_accel
                = calc_first_term_geom_series( required_move.x,
                                               wm.self().playerType().playerDecay(),
                                               M_cycle - cycle );

            if ( required_x_accel < 0.01 )
            {
                break;
            }

            double required_dash_power = required_x_accel / dash_rate;
            double available_stamina = ( M_save_recovery
                                         ? std::max( 0.0, stamina_model.stamina() - SP.recoverDecThrValue() - 1.0 )
                                         : stamina_model.stamina() + wm.self().playerType().extraStamina() );

            double dash_power = std::min( available_stamina, M_max_dash_power );
            dash_power = std::min( dash_power, required_dash_power );
            dash_power = std::min( dash_power, SP.maxDashPower() );

            if ( cycle == 0 )
            {
                first_dash_power = dash_power;
            }

            Vector2D accel = Vector2D::polar2vector( dash_power * dash_rate,
                                                     dash_angle );

#ifdef DEBUG_PRINT
            dlog.addText( Logger::ACTION,
                          "____ cycle=%d requied_accel=%.3f required_power=%.3f available_stamina=%.1f dash_power=%.1f",
                          cycle, required_x_accel, required_dash_power,
                          available_stamina, dash_power );
#endif
            my_vel += accel;
            my_pos += my_vel;
            my_vel *= wm.self().playerType().playerDecay();

            stamina_model.simulateDash( wm.self().playerType(), dash_power );
        }

        if ( first_dash_power == ILLEGAL_POWER )
        {
            continue;
        }

        double last_dist = my_pos.dist( target_rel );

#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      "__ dir=%.1f last_dist=%.3f cycle=%d dash_power=%.1f stamina=%.1f",
                      dir, last_dist,
                      cycle, first_dash_power, stamina_model.stamina() );
#endif

        if ( last_dist < M_dist_thr )
        {
            bool update = false;
            if ( last_dist < 0.1
                 || std::fabs( last_dist - best_dist ) < 0.001 )
            {
                if ( cycle < best_cycle
                     || stamina_model.stamina() > best_stamina )
                {
                    update = true;
                }
            }
            else if ( last_dist < best_dist )
            {
                update = true;
            }

            if ( update )
            {
                best_dir = dir;
                best_dist = last_dist;
                best_cycle = cycle;
                best_dash_power = first_dash_power;
                best_stamina = stamina_model.stamina();
#ifdef DEBUG_PRINT
                dlog.addText( Logger::ACTION,
                              "__ update dir=%.1f dist=%.3f cycle=%d power=%.1f stamina=%.1f",
                              best_dir, best_dist, best_cycle, best_dash_power, best_stamina );
#endif
            }
        }
    }

    if ( best_dir != -360.0 )
    {
        AngleDeg dash_angle = SP.discretizeDashAngle( best_dir );
#ifdef DEBUG_PRINT
        agent->debugClient().addMessage( "OmniDash%.0f", best_dir );
        dlog.addText( Logger::ACTION,
                      __FILE__": (doOmniDash) power=%.3f dir=%.1f",
                      best_dash_power, dash_angle.degree() );
#endif
        return agent->doDash( best_dash_power, dash_angle );
    }

    return false;
}
#endif
/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_GoToPoint::doTurn( PlayerAgent * agent )
{
    const ServerParam & SP = ServerParam::i();
    const WorldModel & wm = agent->world();

    const Vector2D inertia_pos = wm.self().inertiaPoint( M_cycle );
    Vector2D target_rel = M_target_point - inertia_pos;

    const double target_dist = target_rel.r();
    const double max_turn = wm.self().playerType().effectiveTurn( SP.maxMoment(),
                                                                  wm.self().vel().r() );

    AngleDeg turn_moment = target_rel.th() - wm.self().body();

#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION,
                  __FILE__": (doTurn) inertia_pos=(%.1f %.1f ) target_rel=(%.1f %.1f) dist=%.3f turn_moment=%.1f",
                  inertia_pos.x, inertia_pos.y,
                  target_rel.x, target_rel.y,
                  target_dist,
                  turn_moment.degree() );
#endif

    // if target is very near && turn_angle is big && agent has enough stamina,
    // it is useful to reverse accel angle.
    if ( M_use_back_dash
         && turn_moment.abs() > max_turn
         && turn_moment.abs() > 90.0
         && target_dist < 2.0
         && wm.self().stamina() > SP.recoverDecThrValue() + 500.0 )
    {
        double effective_power = SP.maxDashPower() * wm.self().dashRate();
        double effective_back_power = SP.minDashPower() * wm.self().dashRate();
        if ( std::fabs( effective_back_power ) > std::fabs( effective_power ) * 0.75 )
        {
            M_back_mode = true;
            turn_moment += 180.0;
#ifdef DEBUG_PRINT
            dlog.addText( Logger::ACTION,
                          __FILE__": (doTurn) back mode. turn_moment=%.1f",
                          turn_moment.degree() );
#endif
        }
    }

    double turn_thr = 180.0;
// #ifdef USE_ADJUST_DASH
//     if ( M_dist_thr + adjustable_dist < target_dist )
//     {
//         turn_thr = AngleDeg::asin_deg( std::min( 1.0, ( M_dist_thr + adjustable_dist ) / target_dist ) );
//     }
// #else
    if ( M_dist_thr < target_dist )
    {
        turn_thr = AngleDeg::asin_deg( M_dist_thr / target_dist );
    }
// #endif

    turn_thr = std::max( M_dir_thr, turn_thr );

#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION,
                  __FILE__": (doTurn) turn_thr=%.1f",
                  turn_thr );
#endif

    //
    // it is not necessary to perform turn action.
    //
    if ( turn_moment.abs() < turn_thr )
    {
        return false;
    }

    //
    // register turn command
    //
#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION,
                  __FILE__": (doTurn) turn to point. angle=%.1f",
                  turn_moment.degree() );
#endif
    return agent->doTurn( turn_moment );
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_GoToPoint::doDash( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    const Vector2D inertia_pos = wm.self().inertiaPoint( M_cycle );
    Vector2D target_rel = M_target_point - inertia_pos;

    AngleDeg accel_angle = wm.self().body();
    if ( M_back_mode )
    {
        accel_angle += 180.0;
    }

    target_rel.rotate( -accel_angle ); // required_dash_dist == target_rel.x

    // consider inertia travel
    double first_speed
        = calc_first_term_geom_series( target_rel.x,
                                       wm.self().playerType().playerDecay(),
                                       M_cycle );
    first_speed = bound( - wm.self().playerType().playerSpeedMax(),
                         first_speed,
                         wm.self().playerType().playerSpeedMax() );
    if ( M_dash_speed > 0.0 )
    {
        if ( first_speed > 0.0 )
        {
            first_speed = std::min( first_speed, +M_dash_speed );
        }
        else
        {
            first_speed = std::max( first_speed, -M_dash_speed );
        }
    }

    Vector2D rel_vel = wm.self().vel();
    rel_vel.rotate( -accel_angle );

    double required_accel = first_speed - rel_vel.x;

#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION,
                  __FILE__": (doDash) target_rel=(%.2f %.2f) first_speed=%.3f accel=%.3f",
                  target_rel.x, target_rel.y, first_speed, required_accel );
#endif

    if ( std::fabs( required_accel ) < 0.05 )
    {
        // ------- no action -------
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (doDash) required_accel=%.3f, too small. no dash",
                      required_accel );
#endif
        return false;
    }

    double dash_power = required_accel / wm.self().dashRate();
    dash_power = std::min( dash_power, M_max_dash_power );
    if ( M_back_mode )
    {
        dash_power = -dash_power;
    }
    dash_power = ServerParam::i().normalizeDashPower( dash_power );

#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION,
                  __FILE__": (doDash) required dash power = %.3f",
                  dash_power );
#endif

    if ( M_save_recovery )
    {
        dash_power = wm.self().getSafetyDashPower( dash_power );
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION,
                      __FILE__": (doDash) set recoverry save dash power=%.3f",
                      dash_power );
#endif
    }

    return agent->doDash( dash_power );
}
