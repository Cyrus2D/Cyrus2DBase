// -*-c++-*-

/*
 *Copyright:

 Copyright (C) Hidehisa Akiyama

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
#include <fstream>

#include "intercept_evaluator_sample.h"

#include <rcsc/player/intercept_table.h>
#include <rcsc/player/world_model.h>
#include <rcsc/common/server_param.h>
#include <rcsc/common/logger.h>

#include <iostream>
#include <limits>

// #define DEBUG_PRINT

using namespace rcsc;

namespace {
rcss::RegHolder sample = InterceptEvaluator::creators()
    .autoReg( [](){
                  return InterceptEvaluator::Ptr( new InterceptEvaluatorSample() );
              },
        "Sample" );

rcss::RegHolder def = InterceptEvaluator::creators()
    .autoReg( [](){
                  return InterceptEvaluator::Ptr( new InterceptEvaluatorSample() );
              },
        "Default" );

}


/*-------------------------------------------------------------------*/
/*!

 */
InterceptEvaluatorSample::InterceptEvaluatorSample()
    : M_count( 0 )
{

}

/*-------------------------------------------------------------------*/
/*!

 */
InterceptEvaluatorSample::~InterceptEvaluatorSample()
{
#ifdef DEBUG_PRINT
    dlog.addText( Logger::INTERCEPT,
                  "(InterceptEvalSample) delete InterceptEvaluatorSample" );
#endif

}

/*-------------------------------------------------------------------*/
/*!

 */
double
InterceptEvaluatorSample::evaluate( const WorldModel & wm,
                                    const Intercept & action,
                                    const bool save_recovery )
{
    ++M_count;

#ifdef DEBUG_PRINT
    dlog.addText( Logger::INTERCEPT,
                  "------------------------------------------------------------------------------------" );
#endif
    if ( save_recovery
         && action.staminaType() != Intercept::NORMAL )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::INTERCEPT,
                      "(InterceptEvalSample) %d: save recovery",
                      M_count );
#endif
        return -std::numeric_limits< double >::max();
    }


    const ServerParam & SP = ServerParam::i();

    const double max_x = ( SP.keepawayMode()
                           ? SP.keepawayLength() * 0.5 - 1.0
                           : SP.pitchHalfLength() - 1.0 );
    const double max_y = ( SP.keepawayMode()
                           ? SP.keepawayWidth() * 0.5 - 1.0
                           : SP.pitchHalfWidth() - 1.0 );

    const Vector2D ball_pos = wm.ball().inertiaPoint( action.reachStep() );

#ifdef DEBUG_PRINT
    dlog.addText( Logger::INTERCEPT,
                  "%d: intercept step=%d(t:%d d:%d)  ( %lf, %lf )", M_count,
                  action.reachStep(), action.turnStep(), action.dashStep(),
                  ball_pos.x, ball_pos.y );
#endif

    double value = 0.0;

    if ( ball_pos.absX() > max_x
         || ball_pos.absY() > max_y )
    {
        value = -1000.0 - action.reachStep();
#ifdef DEBUG_PRINT
        dlog.addText( Logger::INTERCEPT,
                      "%d: out of pitch = %.3f",
                      M_count, value );
#endif
        return value;
    }

    addShootSpotValue( ball_pos, &value );
    addOpponentStepValue( wm, ball_pos, action, &value );
    addTeammateStepValue( wm, ball_pos, action, &value );
    addTurnPenalty( wm, action, &value );
    addMoveDistPenalty( wm, ball_pos, action, &value );
    addBallDistPenalty( wm, action, &value );
    addStaminaPenalty( wm, action, &value );
    addBallSpeedPenalty( wm, ball_pos, action, &value );

    // 2021-06-18
    if ( wm.kickableOpponent()
         && ( action.actionType() == Intercept::OMNI_DASH
              || action.dashPower() < 0.0 ) )
    {
        const double angle_diff = ( wm.ball().angleFromSelf() - wm.self().body() ).abs();
        if ( angle_diff > 40.0 )
        {
            double tmp_val = -100.0;
            value += tmp_val;
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "%d: kickable_opp value= %.3f (angle_diff = %.1f) (value_sum = %.3f)",
                          M_count, tmp_val, angle_diff, value );
#endif
        }
    }

    return value;
}

/*-------------------------------------------------------------------*/
/*!

 */

void
InterceptEvaluatorSample::addShootSpotValue( const Vector2D & ball_pos,
                                             double * value )
{

    double spot_x_dist = std::fabs( ball_pos.x - 44.0 );
    double spot_y_dist = ball_pos.absY() * 0.5;
    double spot_dist = std::sqrt( std::pow( spot_x_dist, 2 ) + std::pow( spot_y_dist, 2 ) );
    double tmp_val = 0.0;

    tmp_val = spot_dist * -0.2;
    *value += tmp_val;
#ifdef DEBUG_PRINT
    dlog.addText( Logger::INTERCEPT,
                  "%d: (intercept eval) shoot spot dist = %.3f (%.3f)",
                  M_count, tmp_val, *value );
#endif

}
/*-------------------------------------------------------------------*/
/*!

 */

void
InterceptEvaluatorSample::addOpponentStepValue(  const WorldModel & wm,
                                                 const Vector2D & ball_pos,
                                                 const Intercept & action,
                                                 double * value )
{
    const ServerParam & SP = ServerParam::i();
    double tmp_val = 0.0;

    if ( wm.gameMode().type() == GameMode::GoalKick_
         && wm.gameMode().side() == wm.ourSide()
         && ball_pos.x < SP.ourPenaltyAreaLineX() - 2.0
         && ball_pos.absY() < SP.penaltyAreaHalfWidth() - 2.0 )
    {
        // no penalty
    }
    else
    {
        const int opponent_step = wm.interceptTable().opponentStep();
        const int bonus_step = ( wm.interceptTable().firstOpponent()
                                 ? std::min( 5, wm.interceptTable().firstOpponent()->posCount() )
                                 : 0 );
        const int evaluate_step = opponent_step - bonus_step;
        if ( evaluate_step <= action.reachStep() + 3 )
        {
            tmp_val = ( evaluate_step - ( action.reachStep() + 3 ) ) * 5.0;
        }
    }
    *value += tmp_val;

#ifdef DEBUG_PRINT
    dlog.addText( Logger::INTERCEPT,
                  "%d: (intercept eval) opponent diff = %.3f (%.3f)",
                  M_count, tmp_val, *value );
#endif

}

/*-------------------------------------------------------------------*/
/*!

 */

void
InterceptEvaluatorSample::addTeammateStepValue( const WorldModel & wm,
                                                const Vector2D & /*ball_pos*/,
                                                const Intercept & action,
                                                double * value )
{
    const int teammate_step = wm.interceptTable().teammateStep();
    double tmp_val = 0.0;

    if ( teammate_step <= action.reachStep() + 3 )
    {
        tmp_val = ( teammate_step - ( action.reachStep() + 3 ) ) * 0.5;
    }

    *value += tmp_val;

#ifdef DEBUG_PRINT
    dlog.addText( Logger::INTERCEPT,
                  "%d: (intercept eval) teammate diff = %.3f (%.3f)",
                  M_count, tmp_val, *value );
#endif

}
/*-------------------------------------------------------------------*/
/*!

 */

void
InterceptEvaluatorSample::addTurnPenalty( const WorldModel & wm,
                                          const Intercept & action,
                                          double * value )
{
    const double kickable_area = wm.self().playerType().kickableArea();
    const AngleDeg body_angle = wm.self().body() + action.turnAngle();
    double tmp_val = 0.0;

    if ( action.actionType() == Intercept::TURN_FORWARD_DASH
         || action.actionType() == Intercept::TURN_BACK_DASH )
    {
        if ( action.ballDist() > kickable_area - 0.3
             && action.turnStep() > 0 )
        {
            tmp_val = std::fabs( action.turnAngle() ) * -0.025;
            *value += tmp_val;
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "%d: (intercept eval) turn penalty = %.3f (angle=%.1f) (%.3f)",
                          M_count, tmp_val, action.turnAngle(), *value );
#endif
        }
    }

    if ( action.turnStep() > 0 )
    {
        tmp_val = -0.01;
        *value += tmp_val;
#ifdef DEBUG_PRINT
        dlog.addText( Logger::INTERCEPT,
                      "%d: (intercept eval) turn penalty(2) = %.3f (%.3f)",
                      M_count, tmp_val, *value );
#endif
    }

    if ( action.turnStep() > 0
         && action.dashPower() < 0.0 )
    {
        tmp_val = std::max( 0.0, body_angle.abs() - 90.0 ) * -0.1;
        *value += tmp_val;
#ifdef DEBUG_PRINT
        dlog.addText( Logger::INTERCEPT,
                      "%d: (intercept eval) turn back penalty = %.3f (%.3f)",
                      M_count, tmp_val, *value );
#endif
        if ( ( wm.self().body() - body_angle ).abs() > 90.0 )
        {
            tmp_val = -0.001;
            *value += tmp_val;
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "%d: (intercept eval) turn back penalty(2) = %.3f (%.3f)",
                          M_count, tmp_val, *value );
#endif
        }
    }

}

/*-------------------------------------------------------------------*/
/*!

 */

void
InterceptEvaluatorSample::addMoveDistPenalty( const WorldModel & wm,
                                              const Vector2D & ball_pos,
                                              const Intercept & action,
                                              double * value )
{
    const double kickable_area = wm.self().playerType().kickableArea();
    double move_dist = action.selfPos().dist( wm.self().pos() );
    double tmp_val = 0.0;

    if ( ball_pos.x < wm.offsideLineX() )
    {
        tmp_val = move_dist * -0.3; //-0.1;
    }

    if ( action.ballDist() > kickable_area - 0.3 )
    {
        tmp_val += move_dist * -0.5; //-0.2;
    }

    *value += tmp_val;

#ifdef DEBUG_PRINT
    dlog.addText( Logger::INTERCEPT,
                  "%d: (intercept eval) move dist penalty = %.3f (%.3f)",
                  M_count, tmp_val, *value );
#endif
}

/*-------------------------------------------------------------------*/
/*!

 */

void
InterceptEvaluatorSample::addBallDistPenalty( const WorldModel & wm,
                                              const Intercept & action,
                                              double * value )
{
    const double kickable_area = wm.self().playerType().kickableArea();
    double tmp_val = 0.0;

    if ( action.ballDist() > kickable_area - 0.4 )
    {
        //tmp_val = ( action.ballDist() - ( kickable_area - 0.4 ) ) * -5.0;
        tmp_val = ( action.ballDist() - ( kickable_area - 0.4 ) ) * -3.0 - 0.5;
        *value += tmp_val;
#ifdef DEBUG_PRINT
        dlog.addText( Logger::INTERCEPT,
                      "%d: (intercept eval) ball dist penalty(1) = %.3f (%.3f)",
                      M_count, tmp_val, *value );
#endif
    }
}
/*-------------------------------------------------------------------*/
/*!

 */
void
InterceptEvaluatorSample::addStaminaPenalty( const WorldModel & wm,
                                             const Intercept & action,
                                             double * value )
{
    double tmp_val = ( action.stamina() - wm.self().stamina() ) * 0.0001;
    *value += tmp_val;
#ifdef DEBUG_PRINT
    dlog.addText( Logger::INTERCEPT,
                  "%d: (intercept eval) stamina penalty = %.3f (%.3f)",
                  M_count, tmp_val, *value );
#endif

}


/*-------------------------------------------------------------------*/
/*!

 */

void
InterceptEvaluatorSample::addBallSpeedPenalty( const WorldModel & wm,
                                               const Vector2D & ball_pos,
                                               const Intercept & action,
                                               double * value )
{
    const ServerParam & SP = ServerParam::i();

    const double kickable_area = wm.self().playerType().kickableArea();
    const double fast_speed_thr = wm.self().playerType().realSpeedMax() * 0.8;
    const double slow_ball_speed_thr = 0.55;
    const double first_ball_speed = wm.ball().vel().r();
    const AngleDeg ball_vel_angle = wm.ball().vel().th();

    const AngleDeg body_angle = wm.self().body() + action.turnAngle();
    const double ball_speed = first_ball_speed * std::pow( SP.ballDecay(), action.reachStep() );
    double tmp_val = 0.0;

    if ( ( body_angle - ball_vel_angle ).abs() < 30.0 //aligned_body_angle_to_ball_move_line
         && Segment2D( wm.ball().pos(), ball_pos ).dist( wm.self().pos() ) < kickable_area - 0.3 )
    {
        if ( ball_speed < fast_speed_thr )
        {
            tmp_val = ( fast_speed_thr - ball_speed ) * -20.0;
            *value += tmp_val;
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "%d: (intercept eval) ball speed penalty(1) = %.3f (%.3f)",
                          M_count, tmp_val, *value );
#endif
        }
    }
    else
    {
        if ( ball_speed < slow_ball_speed_thr ) // magic number
        {
            //tmp_val = ( slow_ball_speed_thr - ball_speed ) * -20.0;
            tmp_val = std::pow( slow_ball_speed_thr - ball_speed, 2 ) * -70.0;
            *value += tmp_val;
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "%d: (intercept eval) ball speed penalty(2) = %.3f (%.3f)",
                          M_count, tmp_val, *value );
#endif
        }
    }

}

/*-------------------------------------------------------------------*/
/*!

 */
