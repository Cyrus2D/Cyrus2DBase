// -*-c++-*-

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

#include "body_intercept2018.h"

#include "intercept_evaluator.h"

#include "basic_actions/basic_actions.h"
#include "basic_actions/body_go_to_point.h"

#include <rcsc/player/intercept_table.h>
#include <rcsc/player/player_agent.h>
#include <rcsc/player/debug_client.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/soccer_math.h>
#include <rcsc/math_util.h>

#include <string>
#include <sstream>
#include <iostream>

// #define DEBUG_PRINT
// #define DEBUG_PRINT_FACE_POINT

using namespace rcsc;

namespace {

#ifdef DEBUG_PRINT
char
type_char( const InterceptInfo::ActionType t )
{
    switch ( t ) {
    case InterceptInfo::OMNI_DASH:
        return 'o';
    case InterceptInfo::TURN_FORWARD_DASH:
        return 'f';
    case InterceptInfo::TURN_BACK_DASH:
        return 'b';
    default:
        break;
    }
    return 'u';
}
#endif


struct InterceptCandidate {
    const rcsc::Intercept * info_;
    double value_;
    int num_;
    int label_;

    explicit
    InterceptCandidate( const rcsc::Intercept * info,
                        const double value,
                        const int num,
                        const int label )
        : info_( info ),
          value_( value ),
          num_( num ),
          label_( label )
    { }
};

}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Intercept2018::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::TEAM,
                  __FILE__": Body_Intercept2018" );

    const WorldModel & wm = agent->world();

    if ( wm.interceptTable().selfStep() > 100 )
    {
        doGoToFinalPoint( agent );
        return true;
    }

    Intercept best_intercept = get_best_intercept( wm, M_save_recovery );

    // debug output
    {
        Vector2D ball_pos = wm.ball().inertiaPoint( best_intercept.reachStep() );
        agent->debugClient().setTarget( ball_pos );

        dlog.addText( Logger::INTERCEPT,
                      __FILE__": solution size=%d. best_cycle=%d"
                      "(turn:%d dash:%d) (dash power=%.1f dir=%.1f)",
                      wm.interceptTable().selfResults().size(),
                      best_intercept.reachStep(),
                      best_intercept.turnStep(), best_intercept.dashStep(),
                      best_intercept.dashPower(), best_intercept.dashDir() );
        dlog.addText( Logger::INTERCEPT,
                      __FILE__": ball pos=(%.2f %.2f)",
                      ball_pos.x, ball_pos.y );
    }

    if ( doOneStepTurn( agent, best_intercept ) )
    {
        return true;
    }

    if ( doFirstTurn( agent, best_intercept ) )
    {
        return true;
    }

    if ( doSaveRecovery( agent, best_intercept ) )
    {
        return true;
    }

    doDash( agent, best_intercept );
    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Intercept2018::doGoToFinalPoint( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    Vector2D final_point = wm.ball().inertiaFinalPoint();
    agent->debugClient().setTarget( final_point );

    dlog.addText( Logger::INTERCEPT,
                  __FILE__": (doGoToFinalPoint) no solution... go to (%.2f %.2f)",
                  final_point.x, final_point.y );
    agent->debugClient().addMessage( "InterceptNoSolution" );

    Body_GoToPoint( final_point,
                    wm.self().playerType().kickableArea() - 0.2,
                    ServerParam::i().maxDashPower() ).execute( agent );

    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Intercept2018::doOneStepTurn( PlayerAgent * agent,
                                   const Intercept & info )
{
    if ( info.reachStep() > 1
         || info.dashStep() > 0 )
    {
        return false;
    }

    const WorldModel & wm = agent->world();

    Vector2D face_point = M_face_point;
    if ( ! face_point.isValid() )
    {
        face_point.assign( 44.0, wm.self().pos().y * 0.75 );
    }

    dlog.addText( Logger::INTERCEPT,
                  __FILE__": (doOneStepTurn) true: face point=(%.2f %.2f)",
                  face_point.x, face_point.y );
    agent->debugClient().addMessage( "InterceptTurnOnly" );

    Body_TurnToPoint( face_point, 100 ).execute( agent );

    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Intercept2018::doFirstTurn( PlayerAgent * agent,
                                 const Intercept & info )
{
    if ( info.turnStep() == 0 )
    {
        return false;
    }

    const WorldModel & wm = agent->world();

    const Vector2D self_pos = wm.self().inertiaPoint( info.reachStep() );
    const Vector2D ball_pos = wm.ball().inertiaPoint( info.reachStep() );

    AngleDeg ball_angle = ( ball_pos - self_pos ).th();
    if ( info.dashPower() < 0.0 )
    {
        ball_angle -= 180.0;
    }

    dlog.addText( Logger::INTERCEPT,
                  __FILE__": (doFirstTurn) [%s] target_body_angle = %.1f",
                  ( info.dashPower() < 0.0 ? "back" : "forward" ),
                  ball_angle.degree() );
    agent->debugClient().addMessage( "InterceptTurn%d(%d:%d)",
                                     info.reachStep(), info.turnStep(), info.dashStep() );

    agent->doTurn( ball_angle - wm.self().body() );
    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Intercept2018::doSaveRecovery( PlayerAgent * agent,
                                    const Intercept & info )
{
    const WorldModel & wm = agent->world();

    if ( M_save_recovery
         && ! wm.self().staminaModel().capacityIsEmpty() )
    {
        double consumed_stamina = info.dashPower();
        if ( info.dashPower() < 0.0 ) consumed_stamina *= -2.0;

        if ( wm.self().stamina() - consumed_stamina < ServerParam::i().recoverDecThrValue() + 1.0 )
        {
            dlog.addText( Logger::INTERCEPT,
                          __FILE__": (doSaveRecovery) insufficient stamina" );
            agent->debugClient().addMessage( "InterceptRecover" );
            agent->doTurn( 0.0 );
            return true;
        }
    }

    return false;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Intercept2018::doDashWait( PlayerAgent * agent,
                                const Intercept & info )
{
    const WorldModel & wm = agent->world();

    const Vector2D self_pos = wm.self().inertiaPoint( info.reachStep() );
    const Vector2D ball_pos = wm.ball().inertiaPoint( info.reachStep() );
    const bool goalie_mode = ( wm.self().goalie()
                               && wm.lastKickerSide() != wm.ourSide()
                               && ball_pos.x < ServerParam::i().ourPenaltyAreaLineX()
                               && ball_pos.absY() < ServerParam::i().penaltyAreaHalfWidth() );
    const double control_area = ( goalie_mode
                                  ? wm.self().playerType().reliableCatchableDist()
                                  : wm.self().playerType().kickableArea() );

    if ( self_pos.dist2( ball_pos ) > std::pow( control_area - 0.3, 2 ) )
    {
        dlog.addText( Logger::INTERCEPT,
                      __FILE__": (doDashWait) still need dash" );

        return false;
    }

    Vector2D face_point = M_face_point;
    if ( ! M_face_point.isValid() )
    {
        face_point.assign( 50.5, wm.self().pos().y * 0.75 );
    }

    if ( info.reachStep() >= 2
         && wm.ball().seenPosCount() > 3 )
    {
        face_point = wm.ball().pos() + wm.ball().vel();
        dlog.addText( Logger::INTERCEPT,
                      __FILE__": (doDashWait) check ball. pos_count=%d",
                      wm.ball().seenPosCount() );
    }
    else if ( info.reachStep() >= 4 )
    {
        Vector2D my_inertia = wm.self().inertiaFinalPoint();
        AngleDeg face_angle = ( face_point - my_inertia ).th();
        Vector2D next_ball_pos = wm.ball().pos() + wm.ball().vel();
        AngleDeg next_ball_angle = ( next_ball_pos - my_inertia ).th();
        double normal_half_width = ViewWidth::width( ViewWidth::NORMAL );

        if ( ( next_ball_angle - face_angle ).abs()
             > ( ServerParam::i().maxNeckAngle() + normal_half_width - 10.0 ) )
        {
            face_point.x = my_inertia.x;
            if ( next_ball_pos.y > my_inertia.y + 1.0 )
            {
                face_point.y = 50.0;
                dlog.addText( Logger::INTERCEPT,
                              __FILE__": (doDashWait) check ball with turn. adjust(1)" );

            }
            else if ( next_ball_pos.y < my_inertia.y - 1.0 )
            {
                face_point.y = -50.0;
                dlog.addText( Logger::INTERCEPT,
                              __FILE__": (doDashWait) check ball with turn. adjust(2)" );
            }
            else
            {
                face_point = next_ball_pos;
                dlog.addText( Logger::INTERCEPT,
                              __FILE__": (doDashWait) check ball with turn. adjust(3)" );
            }
        }
        else
        {
            dlog.addText( Logger::INTERCEPT,
                          __FILE__": (doDashWait) check ball without turn" );
        }
    }

    agent->debugClient().addMessage( "InterceptDashWait%d", info.reachStep() );
    dlog.addText( Logger::INTERCEPT,
                  __FILE__": (doDashWait) turn to (%.2f %.2f)",
                  face_point.x, face_point.y );

    Body_TurnToPoint( face_point, 100 ).execute( agent );
    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Intercept2018::doOneAdjustDash( PlayerAgent * agent,
                                     const Intercept & info )
{
    const WorldModel & wm = agent->world();
    const PlayerType & ptype = wm.self().playerType();

    Vector2D self_next = wm.self().pos() + wm.self().vel();
    Vector2D ball_next = wm.ball().pos() + wm.ball().vel();

    double ball_dist = self_next.dist( ball_next );
    if ( ball_dist < ptype.kickableArea() - 0.3 )
    {
        double ball_speed = wm.ball().vel().r() * ServerParam::i().ballDecay();
        double kick_rate = ptype.kickRate( ball_dist, 180.0 );

        if ( ( ball_next.x > ServerParam::i().pitchHalfLength() - 4.0
               && ball_next.absY() < ServerParam::i().goalHalfWidth() + 1.0
               && wm.ball().vel().x > 0.0 )
             || ServerParam::i().maxPower() * kick_rate + 0.3 > ball_speed )
        {
            if ( doDashWait( agent, info ) )
            {
                return true;
            }
        }
    }

    agent->debugClient().addMessage( "Intercept1Dash%.0f:%.0f",
                                     info.dashPower(), info.dashDir() );
    dlog.addText( Logger::INTERCEPT,
                  __FILE__": (doOneAdjustDash) 1 dash" );
    agent->doDash( info.dashPower(), info.dashDir() );
    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_Intercept2018::doDash( PlayerAgent * agent,
                            const Intercept & info )
{
    if ( info.reachStep() == 1 )
    {
        doOneAdjustDash( agent, info );
        //agent->doDash( info.dashPower(), info.dashDir() );
        return true;
    }

    //
    //
    //
    if ( doDashWait( agent, info ) )
    {
        return true;
    }

    //
    //
    //
    agent->debugClient().addMessage( "InterceptDash%d:%.0f:%.0f",
                                     info.reachStep(),
                                     info.dashPower(), info.dashDir() );
    dlog.addText( Logger::INTERCEPT,
                  __FILE__": (doDash) power=%.3f dir=%.1f",
                  info.dashPower(), info.dashDir() );

    agent->doDash( info.dashPower(), info.dashDir() );
    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
Intercept
Body_Intercept2018::get_best_intercept( const WorldModel & wm,
                                        const bool save_recovery )
{
    if ( wm.self().goalie() )
    {
        return get_best_intercept_goalie( wm );
    }
    return get_best_intercept_player( wm, save_recovery );
}

/*-------------------------------------------------------------------*/
/*!

 */
Intercept
Body_Intercept2018::get_best_intercept_player( const WorldModel & wm,
                                               const bool save_recovery )
{
    static GameTime s_time;
    static Intercept s_cached_best;

    if ( s_time == wm.time() )
    {
        return s_cached_best;
    }
    s_time = wm.time();

    InterceptEvaluator::Ptr intercept_evaluator = InterceptEvaluator::create( "Default" );

    if ( ! intercept_evaluator )
    {
        std::cerr << wm.ourTeamName() << ' ' << wm.self().unum()
                  << ": ERROR: no intercept evaluator." << std::endl;
        if ( wm.interceptTable().selfResults().empty() )
        {
            return Intercept();
        }
        return wm.interceptTable().selfResults().front();
    }

#ifdef DEBUG_PRINT
    dlog.addText( Logger::INTERCEPT,
                  "===== getBestIntercept =====");
#endif

    int count = 0;

    std::vector< InterceptCandidate > candidates;
    candidates.reserve( wm.interceptTable().selfResults().size() );

    for ( const Intercept & it : wm.interceptTable().selfResults() )
    {
        ++count;
        candidates.push_back( InterceptCandidate( &it, 0.0, count, 0 ) );
        InterceptCandidate & candidate = candidates.back();
        candidate.value_ = intercept_evaluator->evaluate( wm, *candidate.info_, save_recovery );
    }



#ifdef DEBUG_PRINT
    dlog.addText( Logger::INTERCEPT, "==========" );
    count = 0;
    for ( const InterceptCandidate & v : candidates )
    {
        ++count;
        dlog.addText( Logger::INTERCEPT,
                      "%d: value=%.4f step=%d: type[%c] (t%d:d%d) turn=%.1f power=%.1f dir=%.1f",
                      count, v.value_, v.info_->reachStep(),
                      type_char( v.info_->actionType() ),
                      v.info_->turnStep(), v.info_->dashStep(),
                      v.info_->turnAngle(), v.info_->dashPower(), v.info_->dashDir() );
        const Vector2D ball_pos = wm.ball().inertiaPoint( v.info_->reachStep() );
        dlog.addCircle( Logger::INTERCEPT,
                        ball_pos.x, ball_pos.y, 0.2,
                        "#000000");

        // dlog.addCircle( Logger::INTERCEPT,
        //                 v.info_->selfPos().x,
        //                 v.info_->selfPos().y ,
        //                 0.5,
        //                 "yellow");

    }
#endif

    std::vector< InterceptCandidate >::const_iterator best = std::min_element( candidates.begin(),
                                                                               candidates.end(),
                                                                               []( const InterceptCandidate & lhs,
                                                                                   const InterceptCandidate & rhs )
                                                                               {
                                                                                   return lhs.value_ > rhs.value_;
                                                                               } );

    if ( best != candidates.end() )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::INTERCEPT, "==========" );

        dlog.addText( Logger::INTERCEPT,
                      "best[%zd]: step=%d type[%c](t%d:d%d) power=%.1f dir=%.1f -- value=%.4f",
                      best - candidates.begin() + 1, best->info_->reachStep(),
                      type_char( best->info_->actionType() ),
                      best->info_->turnStep(), best->info_->dashStep(),
                      best->info_->dashPower(), best->info_->dashDir(),
                      best->value_ );

        const Vector2D ball_pos = wm.ball().inertiaPoint( best->info_->reachStep() );
        dlog.addCircle( Logger::INTERCEPT,
                        ball_pos.x, ball_pos.y, 0.1,
                        "#000000",
                        true );

        dlog.addText( Logger::INTERCEPT, "==========" );
#endif
        s_cached_best = *(best->info_);
        return s_cached_best;
    }

    if ( candidates.empty() )
    {
        s_cached_best = Intercept();
    }
    else
    {
        s_cached_best = *(candidates.front().info_);
    }

    return s_cached_best;
}


/*-------------------------------------------------------------------*/
/*!

 */
Intercept
Body_Intercept2018::get_best_intercept_goalie( const WorldModel & wm )
{
    static GameTime s_time;
    static Intercept s_cached_best;

    if ( s_time == wm.time() )
    {
        return s_cached_best;
    }
    s_time = wm.time();

    dlog.addText( Logger::INTERCEPT,
                  __FILE__":(getBestInterceptGoalie)" );


    const ServerParam & SP = ServerParam::i();

    const double max_x = SP.pitchHalfLength();
    const double max_y = SP.pitchHalfWidth();

    const double kickable_area = wm.self().playerType().kickableArea();
    const double catchable_area = wm.self().playerType().reliableCatchableDist();

    const int opponent_step = wm.interceptTable().opponentStep();

    std::vector< InterceptCandidate > candidates;
    candidates.reserve( wm.interceptTable().selfResults().size() );

    int count = 0;
    for ( std::vector< Intercept >::const_iterator it = wm.interceptTable().selfResults().begin(),
              end = wm.interceptTable().selfResults().end();
          it != end;
          ++it, ++count )
    {
        const Vector2D ball_pos = wm.ball().inertiaPoint( it->reachStep() );
        const AngleDeg body_angle = wm.self().body() + it->turnAngle();
        const double control_area = ( ball_pos.x < SP.ourPenaltyAreaLineX() - 0.5
                                      && ball_pos.absY() < SP.penaltyAreaHalfWidth() - 0.5
                                      ? catchable_area
                                      : kickable_area );

#ifdef DEBUG_PRINT
        dlog.addText( Logger::INTERCEPT, "----------" );
        dlog.addText( Logger::INTERCEPT,
                      "%d: intercept type[%c] step=%d(t%d:d%d) turn=%.1f power=%.1f dir=%.1f",
                      count, type_char( it->actionType() ),
                      it->reachStep(), it->turnStep(), it->dashStep(),
                      it->turnAngle(), it->dashPower(), it->dashDir() );
        dlog.addText( Logger::INTERCEPT,
                      "%d: intercept bpos=(%.2f %.2f) bdist=%.3f stamina=%.1f",
                      count, ball_pos.x, ball_pos.y, it->ballDist(), it->stamina() );
#endif
        candidates.push_back( InterceptCandidate( &(*it), 0.0, count, 0 ) );
        InterceptCandidate & candidate = candidates.back();

        if ( it->staminaType() != Intercept::NORMAL )
        {
            candidate.value_ -= 1000.0;
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "%d: (intercept eval) recovery decay = %.3f",
                          count, candidate.value_ );
#endif
        }

        if ( ball_pos.absX() > max_x
             || ball_pos.absY() > max_y )
        {
            candidate.value_ -= ( 1000.0 + it->reachStep() );
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "%d: (intercept eval) out of pitch = %.3f",
                          count, candidate.value_ );
#endif
            continue;
        }

        double tmp_val;

        {
            tmp_val = ball_pos.x;
            candidate.value_ = tmp_val;
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "%d: (intercept eval) ball.x = %.3f (%.3f)",
                          count, tmp_val, candidate.value_ );
#endif
        }

        if ( opponent_step <= it->reachStep() + 3 )
        {
            tmp_val = ( opponent_step - ( it->reachStep() + 3 ) ) * 5.0;
            candidate.value_ += tmp_val;
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "%d: (intercept eval) opponent diff = %.3f (%.3f)",
                          count, tmp_val, candidate.value_ );
#endif
        }

        if ( it->actionType() == Intercept::TURN_FORWARD_DASH
             || it->actionType() == Intercept::TURN_BACK_DASH )
        {
            if ( it->ballDist() > control_area - 0.3
                 && it->turnStep() > 0 )
            {
                tmp_val = std::fabs( it->turnAngle() ) * -0.025;
                candidate.value_ += tmp_val;
#ifdef DEBUG_PRINT
                dlog.addText( Logger::INTERCEPT,
                              "%d: (intercept eval) turn penalty = %.3f (moment=%.1f) (%.3f)",
                              count, tmp_val, it->turnAngle(), candidate.value_ );
#endif
            }
        }

        {
            tmp_val = -1.0 * it->ballDist();
            tmp_val = ( it->ballDist() - ( kickable_area - 0.4 ) ) * -3.0 - 0.5;
            candidate.value_ += tmp_val;
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "%d: (intercept eval) ball dist penalty = %.3f (%.3f)",
                          count, tmp_val, candidate.value_ );
#endif
        }

        if ( ball_pos.x > SP.ourPenaltyAreaLineX() - 0.5
             || ball_pos.absY() >SP.penaltyAreaHalfWidth() )
        {
            tmp_val = -0.1 * body_angle.abs();
            candidate.value_ += tmp_val;
#ifdef DEBUG_PRINT
            dlog.addText( Logger::INTERCEPT,
                          "%d: (intercept eval) body angle penalty = %.3f (%.3f)",
                          count, tmp_val, candidate.value_ );
#endif
        }

    }


#ifdef DEBUG_PRINT
    dlog.addText( Logger::INTERCEPT, "==========" );
    count = 1;
    for ( std::vector< InterceptCandidate >::const_iterator it = candidates.begin();
          it != candidates.end();
          ++it, ++count )
    {
        dlog.addText( Logger::INTERCEPT,
                      "%d: step=%d: type[%c] (t%d:d%d) turn=%.1f power=%.1f dir=%.1f -- value=%.4f",
                      count, it->info_->reachStep(),
                      type_char( it->info_->actionType() ),
                      it->info_->turnStep(), it->info_->dashStep(),
                      it->info_->turnAngle(), it->info_->dashPower(), it->info_->dashDir(),
                      it->value_ );
    }
#endif

    std::vector< InterceptCandidate >::const_iterator best = std::min_element( candidates.begin(),
                                                                               candidates.end(),
                                                                               []( const InterceptCandidate & lhs,
                                                                                   const InterceptCandidate & rhs )
                                                                               {
                                                                                   return lhs.value_ > rhs.value_;
                                                                               } );
    if ( best != candidates.end() )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::INTERCEPT, "==========" );
        dlog.addText( Logger::INTERCEPT,
                      "best[%zd]: step=%d type[%c](t%d:d%d) power=%.1f dir=%.1f -- value=%.4f",
                      best - candidates.begin() + 1, best->info_->reachStep(),
                      type_char( best->info_->actionType() ),
                      best->info_->turnStep(), best->info_->dashStep(),
                      best->info_->dashPower(), best->info_->dashDir(),
                      best->value_ );
        dlog.addText( Logger::INTERCEPT, "==========" );
#endif
        s_cached_best = *(best->info_);
        return s_cached_best;
    }

    if ( candidates.empty() )
    {
        s_cached_best = Intercept();
    }
    else
    {
        s_cached_best = *(candidates.front().info_);
    }

    return s_cached_best;
}
