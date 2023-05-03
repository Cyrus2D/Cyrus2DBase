// -*-c++-*-

/*!
  \file body_hold_ball2008.cpp
  \brief stay there and keep the ball from opponent players.
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

#include "body_hold_ball2008.h"

#include "basic_actions.h"
#include "body_kick_to_relative.h"
#include "body_stop_ball.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/player/debug_client.h>

#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/geom/rect_2d.h>
#include <rcsc/geom/line_2d.h>

using namespace rcsc;

// #define DEBUG_CREATE
// #define DEBUG_EVAL
// #define DEBUG_PRINT_RESULTS

namespace {

struct KeepPointSorter {
    bool operator()( const Body_HoldBall2008::KeepPoint & lhs,
                     const Body_HoldBall2008::KeepPoint & rhs ) const
      {
          return lhs.score_ < rhs.score_;
      }
};

}

const double Body_HoldBall2008::DEFAULT_SCORE = 100.0;

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_HoldBall2008::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::ACTION,
                  __FILE__": Body_HoldBall2008" );

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
        return Body_StopBall().execute( agent );
    }

    if ( keepReverse( agent ) )
    {
        return true;
    }

    if ( turnToPoint( agent ) )
    {
        return true;
    }

    if ( keepFront( agent ) )
    {
        return true;
    }

    if ( avoidOpponent( agent ) )
    {
        return true;
    }

    dlog.addText( Logger::HOLD,
                  __FILE__":(execute) only stop the ball" );
    return Body_StopBall().execute( agent );
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_HoldBall2008::avoidOpponent( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    Vector2D point = searchKeepPoint( wm );

    if ( ! point.isValid() )
    {
        dlog.addText( Logger::HOLD,
                      __FILE__":(avoidOpponent) no candidate point" );
        return false;
    }

    Vector2D ball_move = point - wm.ball().pos();
    Vector2D kick_accel = ball_move - wm.ball().vel();
    double kick_accel_r = kick_accel.r();

    agent->debugClient().addMessage( "HoldAvoidOpp" );
    agent->debugClient().addCircle( point, 0.05 );

    dlog.addText( Logger::HOLD,
                  __FILE__":(avoidOpponent) pos=(%.2f %.2f) accel=(%.2f %.2f)%f",
                  point.x, point.y,
                  kick_accel.x, kick_accel.y,
                  kick_accel_r);

    agent->doKick( kick_accel_r / wm.self().kickRate(),
                   kick_accel.th() - wm.self().body() );
    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
Vector2D
Body_HoldBall2008::searchKeepPoint( const WorldModel & wm )
{
    static GameTime s_last_update_time( 0, 0 );
    static std::vector< KeepPoint > s_keep_points;
    static KeepPoint s_best_keep_point;

    if ( s_last_update_time != wm.time() )
    {
        s_best_keep_point.reset();

        createKeepPoints( wm, s_keep_points );
        evaluateKeepPoints( wm, s_keep_points );

        if ( ! s_keep_points.empty() )
        {
            s_best_keep_point = *std::max_element( s_keep_points.begin(),
                                                   s_keep_points.end(),
                                                   KeepPointSorter() );
        }
    }

    return s_best_keep_point.pos_;
}

/*-------------------------------------------------------------------*/
/*!

 */
void
Body_HoldBall2008::createKeepPoints( const WorldModel & wm,
                                     std::vector< KeepPoint > & candidates )
{
    const ServerParam & SP = ServerParam::i();

    const double max_pitch_x = ( SP.keepawayMode()
                                 ? SP.keepawayLength() * 0.5 - 0.2
                                 : SP.pitchHalfLength() - 0.2 );
    const double max_pitch_y = ( SP.keepawayMode()
                                 ? SP.keepawayWidth() * 0.5 - 0.2
                                 : SP.pitchHalfWidth() - 0.2 );

    const int dir_divs = 20;
    const double dir_step = 360.0 / dir_divs;

    const double near_dist = wm.self().playerType().playerSize()
        + SP.ballSize()
        + wm.self().playerType().kickableMargin() * 0.4;
    const double mid_dist = wm.self().playerType().playerSize()
        + SP.ballSize()
        + wm.self().playerType().kickableMargin() * 0.6;
    const double far_dist = wm.self().playerType().playerSize()
        + SP.ballSize()
        + wm.self().playerType().kickableMargin() * 0.75;

    candidates.clear();
    candidates.reserve( dir_divs * 2 );

#ifdef DEBUG_CREATE
    dlog.addText( Logger::HOLD,
                  __FILE__": createCandidatePoints() dir_divs=%d",
                  dir_divs );
#endif

    const Vector2D my_next = wm.self().pos() + wm.self().vel();

    const double my_noise = wm.self().vel().r() * SP.playerRand();
    const double current_dir_diff_rate
        = ( wm.ball().angleFromSelf() - wm.self().body() ).abs() / 180.0;
    const double current_dist_rate
        = ( wm.ball().distFromSelf()
            - wm.self().playerType().playerSize()
            - SP.ballSize() )
        / wm.self().playerType().kickableMargin();
    const double current_pos_rate
        = 0.5 + 0.25 * ( current_dir_diff_rate + current_dist_rate );
    const double current_speed_rate
        = 0.5 + 0.5 * ( wm.ball().vel().r()
                        / ( SP.ballSpeedMax() * SP.ballDecay() ) );

    // angle loop
    for ( double d = -180.0; d < 180.0; d += dir_step )
    {
        const AngleDeg angle = d;
        const double dir_diff = ( angle - wm.self().body() ).abs();
        const Vector2D unit_pos = Vector2D::polar2vector( 1.0, angle );

        // near side point
        {
        const Vector2D near_pos = my_next + unit_pos.setLengthVector( near_dist );
        if ( near_pos.absX() < max_pitch_x
             && near_pos.absY() < max_pitch_y )
        {
            Vector2D ball_move = near_pos - wm.ball().pos();
            Vector2D kick_accel = ball_move - wm.ball().vel();

            // can kick to the point by 1 step kick
            if ( kick_accel.r() < SP.maxPower() * wm.self().kickRate() )
            {
                double near_krate = wm.self().playerType().kickRate( near_dist, dir_diff );
                // can stop the ball by 1 step kick
                if ( ball_move.r() * SP.ballDecay() < SP.maxPower() * near_krate )
                {
#ifdef DEBUG_CREATE
                    dlog.addText( Logger::HOLD,
                                  "__add near point (%.2f %.2f) angle=%.0f dist=%.2f",
                                  near_pos.x, near_pos.y,
                                  d, near_dist );
#endif
                    candidates.emplace_back( near_pos,
                                             near_krate,
                                             DEFAULT_SCORE );
                }
#ifdef DEBUG_CREATE
                else
                {
                    dlog.addText( Logger::HOLD,
                                  "__cancel near point (%.2f %.2f) angle=%.0f dist=%.2f"
                                  " cannot stop ball"
                                  " ball_move=(%.3f %.3f)%.3f krate=%f",
                                  near_pos.x, near_pos.y,
                                  d, near_dist,
                                  ball_move.x, ball_move.y, ball_move.r(),
                                  near_krate );
                }
#endif
            }
#ifdef DEBUG_CREATE
            else
            {
                dlog.addText( Logger::HOLD,
                              "__cancel near point (%.2f %.2f) angle=%.0f dist=%.2f"
                              " cannot kick"
                              " required_accel=(%.3f %.3f)%.3f cur_krate=%f",
                              near_pos.x, near_pos.y,
                              d, near_dist,
                              kick_accel.x, kick_accel.y,
                              kick_accel.r(),
                              wm.self().kickRate() );
            }
#endif
        }
        }

        // middle point
        {
        const Vector2D mid_pos = my_next + unit_pos.setLengthVector( mid_dist );
        if ( mid_pos.absX() < max_pitch_x
             && mid_pos.absY() < max_pitch_y )
        {
            Vector2D ball_move = mid_pos - wm.ball().pos();
            Vector2D kick_accel = ball_move - wm.ball().vel();
            double kick_power = kick_accel.r() / wm.self().kickRate();

            // can kick to the point by 1 step kick
            if ( kick_power < SP.maxPower() )
            {
                // check move noise
                const double move_dist = ball_move.r();
                const double ball_noise = move_dist * SP.ballRand();
                const double max_kick_rand
                    = wm.self().playerType().kickRand()
                    * ( kick_power / SP.maxPower() )
                    * ( current_pos_rate + current_speed_rate );
                // move noise is small
                if ( ( my_noise + ball_noise + max_kick_rand ) * 0.95
                     < wm.self().playerType().kickableArea() - mid_dist - 0.1 )
                {
                    double mid_krate = wm.self().playerType().kickRate( mid_dist, dir_diff );
                    // can stop the ball by 1 step kick
                    if ( move_dist * SP.ballDecay()
                         < SP.maxPower() * mid_krate )
                    {
#ifdef DEBUG_CREATE
                        dlog.addText( Logger::HOLD,
                                      "__add mid point (%.2f %.2f) angle=%.0f dist=%.2f",
                                      mid_pos.x, mid_pos.y,
                                      d, mid_dist );
#endif
                        candidates.emplace_back( mid_pos,
                                                 mid_krate,
                                                 DEFAULT_SCORE );
                    }
#ifdef DEBUG_CREATE
                    else
                    {
                        dlog.addText( Logger::HOLD,
                                      "__cancel mid point (%.2f %.2f) angle=%.0f dist=%.2f"
                                      " cannot stop ball"
                                      " ball_move=(%.3f %.3f)%.3f krate=%f",
                                      mid_pos.x, mid_pos.y,
                                      d, mid_dist,
                                      ball_move.x, ball_move.y, ball_move.r(),
                                      mid_krate );
                    }
#endif
                }
#ifdef DEBUG_CREATE
                else
                {
                    dlog.addText( Logger::HOLD,
                                  "__cancel mid point (%.2f %.2f) angle=%.0f dist=%.2f"
                                  " big noise"
                                  " my=%.3f ball=%.3f kick=%.3f. total=%f > kickable_buf=%f",
                                  mid_pos.x, mid_pos.y,
                                  d, mid_dist,
                                  my_noise, ball_noise, max_kick_rand,
                                  ( my_noise + ball_noise + max_kick_rand ) * 0.95,
                                  wm.self().kickableArea() - mid_dist - 0.1 );
                }
#endif
            }
#ifdef DEBUG_CREATE
            else
            {
                dlog.addText( Logger::HOLD,
                              "__cancel mid point (%.2f %.2f) angle=%.0f dist=%.2f"
                              " cannot kick"
                              " required_accel=(%.3f %.3f)%.3f cur_krate=%f",
                              mid_pos.x, mid_pos.y,
                              d, mid_dist,
                              kick_accel.x, kick_accel.y,
                              kick_accel.r(),
                              wm.self().kickRate() );
            }
#endif
        }
        }

        // far side point
        {
        const Vector2D far_pos = my_next + unit_pos.setLengthVector( far_dist );
        if ( far_pos.absX() < max_pitch_x
             && far_pos.absY() < max_pitch_y )
        {
            Vector2D ball_move = far_pos - wm.ball().pos();
            Vector2D kick_accel = ball_move - wm.ball().vel();
            double kick_power = kick_accel.r() / wm.self().kickRate();

            // can kick to the point by 1 step kick
            if ( kick_power < SP.maxPower() )
            {
                // check move noise
                const double move_dist = ball_move.r();
                const double ball_noise = move_dist * SP.ballRand();
                const double max_kick_rand
                    = wm.self().playerType().kickRand()
                    * ( kick_power / SP.maxPower() )
                    * ( current_pos_rate + current_speed_rate );
                // move noise is small
                if ( ( my_noise + ball_noise + max_kick_rand ) * 0.95
                     < wm.self().playerType().kickableArea() - far_dist - 0.1 )
                {
                    double far_krate = wm.self().playerType().kickRate( far_dist, dir_diff );
                    // can stop the ball by 1 step kick
                    if ( move_dist * SP.ballDecay()
                         < SP.maxPower() * far_krate )
                    {
#ifdef DEBUG_CREATE
                        dlog.addText( Logger::HOLD,
                                      "__add far point (%.2f %.2f) angle=%.0f dist=%.2f",
                                      far_pos.x, far_pos.y,
                                      d, far_dist );
#endif
                        candidates.emplace_back( far_pos,
                                                 far_krate,
                                                 DEFAULT_SCORE );
                    }
#ifdef DEBUG_CREATE
                    else
                    {
                        dlog.addText( Logger::HOLD,
                                  "__cancel far point (%.2f %.2f) angle=%.0f dist=%.2f"
                                  " cannot stop ball"
                                  " ball_move=(%.3f %.3f)%.3f krate=%f",
                                  far_pos.x, far_pos.y,
                                  d, far_dist,
                                  ball_move.x, ball_move.y, ball_move.r(),
                                  far_krate );
                    }
#endif
                }
#ifdef DEBUG_CREATE
                else
                {
                    dlog.addText( Logger::HOLD,
                                  "__cancel far point (%.2f %.2f) angle=%.0f dist=%.2f"
                                  " big noise"
                                  " my=%.3f ball=%.3f kick=%.3f. total=%f > kickable_buf=%f",
                                  far_pos.x, far_pos.y,
                                  d, far_dist,
                                  my_noise, ball_noise, max_kick_rand,
                                  ( my_noise + ball_noise + max_kick_rand ) * 0.95,
                                  wm.self().kickableArea() - far_dist - 0.1 );
                }
#endif
            }
#ifdef DEBUG_CREATE
            else
            {
                dlog.addText( Logger::HOLD,
                              "__cancel far point (%.2f %.2f) angle=%.0f dist=%.2f"
                              " cannot kick"
                              " required_accel=(%.3f %.3f)%.3f cur_krate=%f",
                              far_pos.x, far_pos.y,
                              d, far_dist,
                              kick_accel.x, kick_accel.y,
                              kick_accel.r(),
                              wm.self().kickRate() );
            }
#endif
        }
        }
    }

    dlog.addText( Logger::HOLD,
                  __FILE__": createCandidatePoints() size=%d",
                  (int)candidates.size() );
}

/*-------------------------------------------------------------------*/
/*!

 */
void
Body_HoldBall2008::evaluateKeepPoints( const WorldModel & wm,
                                       std::vector< KeepPoint > & keep_points )
{
#ifdef DEBUG_EVAL
    int count = 0;
    dlog.addText( Logger::HOLD,
                  __FILE__"(evaluate) =========" );
#endif
    for ( KeepPoint & p : keep_points )
    {
#ifdef DEBUG_EVAL
        dlog.addText( Logger::HOLD,
                      "%d: (evaluate) (%.2f %.2f)",
                      ++count, p.pos_.x, p.pos_.y );
#endif
        p.score_ = evaluateKeepPoint( wm, p.pos_ );
        // if ( p.score_ < DEFAULT_SCORE - 1.0e-5 )
        // {
        //     p.score_ += p.pos_.dist( wm.ball().pos() ) * 0.001;
        // }
        // else
        {
            p.score_ += p.kick_rate_ * 1000.0;
        }
    }

#ifdef DEBUG_EVAL
    Vector2D self_next = wm.self().pos() + wm.self().vel();
    double k = wm.self().playerType().kickableArea();
    dlog.addCircle( Logger::HOLD,
                    self_next, 0.3, "#F00", true );
    dlog.addCircle( Logger::HOLD,
                    self_next, k, "#F00" );

    dlog.addText( Logger::HOLD,
                  __FILE__"(results) =========" );
    count = 0;
    for ( const KeepPoint & p : keep_points )
    {
        ++count;
        char score[16];
        snprintf( score, 16, "%d:%.3f", count, p.score_ );
        dlog.addText( Logger::HOLD,
                      "%d: (evaluate) (%.2f %.2f) score=%f",
                      count, p.pos_.x, p.pos_.y, p.score_ );
        dlog.addRect( Logger::HOLD,
                      p.pos_.x - 0.02, p.pos_.y - 0.02, 0.04, 0.04, "#0F0" );
        dlog.addMessage( Logger::HOLD,
                         p.pos_, score );
    }
    dlog.addText( Logger::HOLD,
                  __FILE__"(results) =========" );
#endif
}

/*-------------------------------------------------------------------*/
/*!

 */
double
Body_HoldBall2008::evaluateKeepPoint( const WorldModel & wm,
                                      const Vector2D & keep_point )
{
    static const Rect2D penalty_area( Vector2D( ServerParam::i().theirPenaltyAreaLineX(),
                                                - ServerParam::i().penaltyAreaHalfWidth() ),
                                      Size2D( ServerParam::i().penaltyAreaLength(),
                                              ServerParam::i().penaltyAreaWidth() ) );
    static const double consider_dist = ( ServerParam::i().tackleDist()
                                          + ServerParam::i().defaultPlayerSpeedMax()
                                          + 1.0 );
    const ServerParam & SP = ServerParam::i();

    const Vector2D my_next = wm.self().pos() + wm.self().vel();

    double score = DEFAULT_SCORE;

    for ( const PlayerObject * o : wm.opponentsFromBall() )
    {
        if ( o->distFromBall() > consider_dist ) break;

        if ( o->posCount() > 10 ) continue;
        if ( o->isGhost() ) continue;
        if ( o->isTackling() ) continue;

        const PlayerType * player_type = o->playerTypePtr();
        const Vector2D opp_next = o->pos() + o->vel();
        const double control_area = ( ( o->goalie()
                                        && penalty_area.contains( opp_next )
                                        && penalty_area.contains( keep_point ) )
                                      ? SP.catchableArea()
                                      : player_type->kickableArea() );
        const double opp_dist = opp_next.dist( keep_point );

        if ( opp_dist < control_area * 0.5 )
        {
            score -= 200.0;
#ifdef DEBUG_EVAL
            dlog.addText( Logger::HOLD,
                          "____ opp %d(%.1f %.1f) can control(1). score=%.3f",
                          o->unum(),
                          o->pos().x, o->pos().y, score );

#endif
        }
        else if ( opp_dist < control_area + 0.1 )
        {
            score -= 150.0;
#ifdef DEBUG_EVAL
            dlog.addText( Logger::HOLD,
                          "____ opp %d(%.1f %.1f) can control(2). score=%.3f",
                          o->unum(),
                          o->pos().x, o->pos().y, score );

#endif
        }
        else if ( opp_dist < SP.tackleDist() - 0.2 )
        {
            score -= 25.0;
#ifdef DEBUG_EVAL
            dlog.addText( Logger::HOLD,
                          "____ opp %d(%.1f %.1f) within tackle. score=%.3f",
                          o->unum(),
                          o->pos().x, o->pos().y, score );

#endif
        }

        AngleDeg opp_body;
        if ( o->bodyCount() == 0 )
        {
            opp_body = o->body();
        }
        else if ( o->velCount() <= 1
                  && o->vel().r() > 0.2 )
        {
            opp_body = o->vel().th();
        }
        else
        {
            opp_body = ( my_next - opp_next ).th();
        }

        //
        // check opponent body line
        //

        const Vector2D player_2_pos = ( keep_point - opp_next ).rotatedVector( -opp_body );

        if ( player_2_pos.absY() < control_area )
        {
            // if ( player_2_pos.absY() < control_area * 0.8 )
            // {
            //     score -= 20.0;
            // }
            // else
            // {
            //     score -= 10.0;
            // }

            score -= ( control_area - player_2_pos.absY() ) * 50.0;

#ifdef DEBUG_EVAL
            dlog.addText( Logger::HOLD,
                          "____ opp %d(%.1f %.1f) on body line. body=%.1f y=%.3f score=%.3f",
                          o->unum(),
                          o->pos().x, o->pos().y, opp_body.degree(),
                          player_2_pos.absY(),
                          score );

#endif
        }

        //
        // check tackle probability
        //
        {
            double tackle_dist = ( player_2_pos.x > 0.0
                                   ? SP.tackleDist()
                                   : SP.tackleBackDist() );
            if ( tackle_dist > 1.0e-5 )
            {
                double tackle_fail_prob = ( std::pow( player_2_pos.absX() / tackle_dist,
                                                      SP.foulExponent() )
                                            + std::pow( player_2_pos.absY() / SP.tackleWidth(),
                                                        SP.foulExponent() ) );
                if ( tackle_fail_prob < 1.0 )
                    //&& 1.0 - tackle_fail_prob > 0.7 ) // success probability
                {
                    //score -= 30.0;
                    score -= ( 1.0 - tackle_fail_prob ) * 50.0;
#ifdef DEBUG_EVAL
                    dlog.addText( Logger::HOLD,
                                  "____ tackle_prob=%.3f %d(%.1f %.1f) body=%.1f score=%.3f",
                                  1.0 - tackle_fail_prob,
                                  o->unum(),
                                  o->pos().x, o->pos().y,
                                  opp_body.degree(), score );
#endif
                }
            }
        }

        //
        // check kick or tackle possibility after dash
        //
        const double dash_angle_step = std::max( 15.0, SP.dashAngleStep() );
        const int dash_angle_divs
            = static_cast< int >( std::floor( ( SP.maxDashAngle() - SP.minDashAngle() )
                                              / dash_angle_step ) );

        double next_kick_penalty = 0.0;
        double next_tackle_penalty = 0.0;
        for ( int d = 0; d < dash_angle_divs; ++d )
        {
            const double dir = AngleDeg::normalize_angle( SP.minDashAngle() + ( dash_angle_step * d ) );
            const AngleDeg dash_angle = SP.discretizeDashAngle( dir );
            const double max_accel = ( SP.maxDashPower()
                                       * player_type->dashPowerRate()
                                       * player_type->effortMax()
                                       * SP.dashDirRate( dir ) );
            const Vector2D max_move = Vector2D::from_polar( max_accel, dash_angle );

            const Vector2D next_player_2_pos = player_2_pos - max_move;

            if ( next_player_2_pos.r2() < std::pow( control_area + 0.1, 2 ) )
            {
#ifdef DEBUG_EVAL
                dlog.addText( Logger::HOLD,
                              "____ next kickable %d opponent_body=%.1f dash_dir=%.0f max_accel=%.3f",
                              o->unum(), opp_body.degree(), dir, max_accel );
#endif
                //next_kick_penalty = -20.0;
                next_kick_penalty -= 20.0;
            }
            else if ( next_player_2_pos.absY() < SP.tackleWidth() + 0.1
                      && next_player_2_pos.x > 0.0
                      && next_player_2_pos.x < SP.tackleDist() + 0.1 )
            {
#ifdef DEBUG_EVAL
                dlog.addText( Logger::HOLD,
                              "____ next tackle %d opponent_body=%.1f dash_dir=%.0f max_accel=%.3f",
                              o->unum(), opp_body.degree(), dir, max_accel );
#endif
                //next_tackle_penalty = -10.0;
                next_tackle_penalty -= 10.0;
            }
        }

        score += next_kick_penalty;
        score += next_tackle_penalty;
#ifdef DEBUG_EVAL
        dlog.addText( Logger::HOLD,
                      "____ %d kick_penalty=%.1f tackle_penalty=%.1f score=%.3f",
                      o->unum(), next_kick_penalty, next_tackle_penalty, score );
#endif

    }

#if 1
    double ball_move_dist = ( keep_point - wm.ball().pos() ).r();
    if ( ball_move_dist > wm.self().playerType().kickableArea() * 1.6 )
    {
        double next_ball_dist = my_next.dist( keep_point );
        double threshold = wm.self().playerType().kickableArea() - 0.4;
        double rate = 1.0 - 0.5 * std::max( 0.0, ( next_ball_dist - threshold ) / 0.4 );
        score *= rate;

#ifdef DEBUG_EVAL
        dlog.addText( Logger::HOLD,
                      "__ applied keep distance threshold. ball_dist=%.3f thr=%.3f rate=%f",
                      next_ball_dist, threshold, rate );
#endif
    }
#endif

    return score;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_HoldBall2008::keepFront( PlayerAgent * agent )
{
    const ServerParam & SP = ServerParam::i();
    const double max_pitch_x = ( SP.keepawayMode()
                                 ? SP.keepawayLength() * 0.5 - 0.2
                                 : SP.pitchHalfLength() - 0.2 );
    const double max_pitch_y = ( SP.keepawayMode()
                                 ? SP.keepawayWidth() * 0.5 - 0.2
                                 : SP.pitchHalfWidth() - 0.2 );

    const WorldModel & wm = agent->world();
    const double front_keep_dist
        = wm.self().playerType().playerSize()
        + SP.ballSize() + 0.05;
    const Vector2D my_next = wm.self().pos() + wm.self().vel();

    Vector2D front_pos
        = my_next
        + Vector2D::polar2vector( front_keep_dist, wm.self().body() );

    if ( front_pos.absX() > max_pitch_x
         || front_pos.absY() > max_pitch_y )
    {
        dlog.addText( Logger::HOLD,
                      __FILE__":(keepFront) failed. out of pitch. point=(%.2f %.2f)",
                      front_pos.x, front_pos.y );
        return false;
    }

    Vector2D ball_move = front_pos - wm.ball().pos();
    Vector2D kick_accel = ball_move - wm.ball().vel();
    double kick_power = kick_accel.r() / wm.self().kickRate();

    // can kick to the point by 1 step kick
    if ( kick_power > SP.maxPower() )
    {
        dlog.addText( Logger::HOLD,
                      __FILE__":(keepFront) failed. cannot kick to front point (%.2f %.2f) by 1 step",
                      front_pos.x, front_pos.y );
        return false;
    }

    double score = evaluateKeepPoint( wm, front_pos );

    if ( score < DEFAULT_SCORE - 1.0e-5 )
    {
        dlog.addText( Logger::HOLD,
                      __FILE__":(keepFront) failed. front point (%.2f %.2f) is not safe.",
                      front_pos.x, front_pos.y );
        return false;
    }

    dlog.addText( Logger::HOLD,
                  __FILE__":(keepFront) ok. front point (%.2f %.2f) dist=%.2f score=%f",
                  front_pos.x, front_pos.y,
                  front_keep_dist,
                  score );
    agent->debugClient().addMessage( "HoldFront" );

    agent->doKick( kick_power,
                   kick_accel.th() - wm.self().body() );
    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_HoldBall2008::keepReverse( PlayerAgent * agent )
{
    if ( ! M_kick_target_point.isValid() )
    {
        return false;
    }

    const WorldModel & wm = agent->world();
    const ServerParam & SP = ServerParam::i();

    const double max_pitch_x = ( SP.keepawayMode()
                                 ? SP.keepawayLength() * 0.5 - 0.2
                                 : SP.pitchHalfLength() - 0.2 );
    const double max_pitch_y = ( SP.keepawayMode()
                                 ? SP.keepawayWidth() * 0.5 - 0.2
                                 : SP.pitchHalfWidth() - 0.2 );

    //const Vector2D my_inertia = wm.self().inertiaFinalPoint();
    const Vector2D my_inertia = wm.self().pos() + wm.self().vel();

    const double my_noise = wm.self().vel().r() * SP.playerRand();
    const double current_dir_diff_rate
        = ( wm.ball().angleFromSelf() - wm.self().body() ).abs() / 180.0;
    const double current_dist_rate
        = ( wm.ball().distFromSelf()
            - wm.self().playerType().playerSize()
            - SP.ballSize() )
        / wm.self().playerType().kickableMargin();
    const double current_pos_rate
        = 0.5 + 0.25 * ( current_dir_diff_rate + current_dist_rate );
    const double current_speed_rate
        = 0.5 + 0.5 * ( wm.ball().vel().r()
                        / ( SP.ballSpeedMax() * SP.ballDecay() ) );

    const AngleDeg keep_angle = ( my_inertia - M_kick_target_point ).th();
    const double dir_diff = ( keep_angle - wm.self().body() ).abs();
    const double min_dist = ( wm.self().playerType().playerSize()
                              + SP.ballSize()
                              + 0.2 );

    double keep_dist
        = wm.self().playerType().playerSize()
        + wm.self().playerType().kickableMargin() * 0.5
        + ServerParam::i().ballSize();

    const Vector2D unit_vec = Vector2D::polar2vector( 1.0, keep_angle );
    for ( ; keep_dist > min_dist; keep_dist -= 0.05 )
    {
        Vector2D keep_pos = my_inertia + unit_vec * keep_dist;

        if ( keep_pos.absX() > max_pitch_x
             || keep_pos.absY() > max_pitch_y )
        {
            continue;
        }

        Vector2D ball_move = keep_pos - wm.ball().pos();
        Vector2D kick_accel = ball_move - wm.ball().vel();
        double kick_power = kick_accel.r() / wm.self().kickRate();

        if ( kick_power > SP.maxPower() )
        {
            continue;
        }

        double move_dist = ball_move.r();
        double ball_noise = move_dist * SP.ballRand();
        double max_kick_rand
            = wm.self().playerType().kickRand()
            * ( kick_power / SP.maxPower() )
            * ( current_pos_rate + current_speed_rate );
        if ( ( my_noise + ball_noise + max_kick_rand )
             > wm.self().playerType().kickableArea() - keep_dist - 0.1 )
        {
            continue;
        }

        double new_krate = wm.self().playerType().kickRate( keep_dist, dir_diff );
        if ( move_dist * SP.ballDecay() > new_krate * SP.maxPower() )
        {
            continue;
        }

        double score = evaluateKeepPoint( wm, keep_pos );
        if ( score > DEFAULT_SCORE + 1.0e-5 )
        {
            dlog.addText( Logger::HOLD,
                          __FILE__":(keepReverse) kick_target=(%.1f %.1f) reverse_point=(%.2f %.2f) angle=%.0f dist=%.2f score=%f",
                          M_kick_target_point.x, M_kick_target_point.y,
                          keep_pos.x, keep_pos.y,
                          keep_angle.degree(), keep_dist,
                          score );
            agent->debugClient().addMessage( "HoldReverse" );
            agent->debugClient().addCircle( keep_pos, 0.05 );

            agent->doKick( kick_power,
                           kick_accel.th() - wm.self().body() );
            return true;
        }
    }

    dlog.addText( Logger::HOLD,
                  __FILE__":(keepReverse) failed" );

    return false;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
Body_HoldBall2008::turnToPoint( PlayerAgent * agent )
{
    const ServerParam & SP = ServerParam::i();
    const double max_pitch_x = ( SP.keepawayMode()
                                 ? SP.keepawayLength() * 0.5 - 0.2
                                 : SP.pitchHalfLength() - 0.2 );
    const double max_pitch_y = ( SP.keepawayMode()
                                 ? SP.keepawayWidth() * 0.5 - 0.2
                                 : SP.pitchHalfWidth() - 0.2 );

    const WorldModel & wm = agent->world();
    const Vector2D my_next = wm.self().pos() + wm.self().vel();
    const Vector2D ball_next = wm.ball().pos() + wm.ball().vel();

    if ( ball_next.absX() > max_pitch_x
         || ball_next.absY() > max_pitch_y )
    {
        dlog.addText( Logger::HOLD,
                      __FILE__":(turnToPoint) failed. out of pitch. ball_next=(%.2f %.2f)",
                      ball_next.x, ball_next.y );
        return false;
    }

    const double my_noise = wm.self().vel().r() * SP.playerRand();
    const double ball_noise = wm.ball().vel().r() * SP.ballRand();

    const double next_ball_dist = my_next.dist( ball_next );
    if ( next_ball_dist > ( wm.self().playerType().kickableArea()
                            - my_noise
                            - ball_noise
                            - 0.15 ) )
    {
        dlog.addText( Logger::HOLD,
                      __FILE__":(turnToPoint) no kickable at next cycle. ball_dist=%.3f",
                      next_ball_dist );
        return false;
    }

    Vector2D face_point( 0.0, 0.0 );
    if ( ! SP.keepawayMode() )
    {
        face_point.x =  SP.pitchHalfLength() - 5.0;
    }

    if ( M_do_turn )
    {
        face_point = M_turn_target_point;

        dlog.addText( Logger::HOLD,
                      __FILE__":(turnToPoint) face target=(%.1f, %.1f)",
                      face_point.x, face_point.y );
    }

    const Vector2D my_inertia = wm.self().inertiaFinalPoint();
    AngleDeg target_angle = ( face_point - my_inertia ).th();

    if ( ( wm.self().body() - target_angle ).abs() < 5.0 )
    {
        dlog.addText( Logger::HOLD,
                      __FILE__":(turnToPoint) already face to (%.1f %.1f).",
                      face_point.x, face_point.y );
        return false;
    }

    double score = evaluateKeepPoint( wm, ball_next );
    if ( score < DEFAULT_SCORE - 1.0e-5 )
    {
        dlog.addText( Logger::HOLD,
                      __FILE__":(turnToPoint) next_ball_pos(%.1f %.1f) is not safety",
                      ball_next.x, ball_next.y );
        return false;
    }

    dlog.addText( Logger::HOLD,
                  __FILE__":(turnToPoint) next_ball_dist=%.2f turn to (%.1f, %.1f) score=%f",
                  next_ball_dist,
                  face_point.x, face_point.y,
                  score );
    agent->debugClient().addMessage( "HoldTurn" );
    Body_TurnToPoint( face_point, 100 ).execute( agent );
    return true;
}
