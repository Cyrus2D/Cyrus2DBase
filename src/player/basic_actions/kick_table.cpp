// -*-c++-*-

/*!
  \file kick_table.cpp
  \brief kick planner and cache holder class Source File
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

#include "kick_table.h"

#include <rcsc/player/world_model.h>
#include <rcsc/geom/ray_2d.h>
#include <rcsc/geom/circle_2d.h>
#include <rcsc/geom/rect_2d.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/common/player_param.h>
#include <rcsc/common/player_type.h>
#include <rcsc/game_time.h>
#include <rcsc/math_util.h>
#include <rcsc/soccer_math.h>
#include <rcsc/timer.h>

#include <algorithm>
#include <functional>
#include <fstream>

using namespace rcsc;

// #define DEBUG_PROFILE
// #define DEBUG
// #define DEBUG_OPPONENT
// #define DEBUG_PRINT_STATE_CACHE
// #define DEBUG_PRINT_SEQUENCE
// #define DEBUG_EVALUATE
// #define DEBUG_ONE_STEP
// #define DEBUG_TWO_STEP
// #define DEBUG_THREE_STEP
// #define DEBUG_THREE_STEP_DETAIL

namespace  {

const double NEAR_SIDE_RATE = 0.3;
const double MID_RATE = 0.5;
const double FAR_SIDE_RATE = 0.7;

const int STATE_DIVS_NEAR = 12;
const int STATE_DIVS_MID = 15;
const int STATE_DIVS_FAR = 20;
const int NUM_STATE = STATE_DIVS_NEAR + STATE_DIVS_MID + STATE_DIVS_FAR;

const size_t MAX_TABLE_SIZE = 1024;


/*!
 \struct TableSorter
 \brief kick path evaluator
*/
struct PathSorter {
    /*!
      \brief compare operation function
      \param lhs left hand side variable
      \param rhs right hand side variable
      \return compared result
     */
    bool operator()( const KickTable::Path & lhs,
                     const KickTable::Path & rhs )
      {
          if ( std::fabs( lhs.max_speed_ - rhs.max_speed_ ) < 1.0e-3 )
          {
              return lhs.power_ < rhs.power_;
          }
          return lhs.max_speed_ > rhs.max_speed_;
      }
};


/*!
 \struct SequenceSorter
 \brief kick sequence evaluator
*/
struct SequenceSorter {
    /*!
      \brief compare operation function
      \param lhs left hand side variable
      \param rhs right hand side variable
      \return compared result
     */
    bool operator()( const KickTable::Sequence & lhs,
                     const KickTable::Sequence & rhs )
      {
          return lhs.score_ < rhs.score_;
      }
};

/*-------------------------------------------------------------------*/
/*!
  \brief caclulate the distance of near side sub-target
  \param player_type calculated PlayerType
  \return distance from the center of the player
*/
inline
double
calc_near_dist( const PlayerType & player_type,
                const double kick_margin_delta = 0.0 )
{
    //       0.3 + 0.6*0.3 + 0.085 = 0.565
    // near: 0.3 + 0.7*0.3 + 0.085 = 0.595
    //       0.3 + 0.8*0.3 + 0.085 = 0.625

    double kmargin = std::min( player_type.kickableMargin(),
                               player_type.kickableMargin() + kick_margin_delta );
    return bound( player_type.playerSize() + ServerParam::i().ballSize() + 0.1,
                  ( player_type.playerSize()
                    + ( kmargin * NEAR_SIDE_RATE )
                    + ServerParam::i().ballSize() ),
                  player_type.kickableArea() - 0.2 );
}

/*-------------------------------------------------------------------*/
/*!
  \brief caclulate the distance of middile distance sub-target
  \param player_type calculated PlayerType
  \return distance from the center of the player
*/
inline
double
calc_mid_dist( const PlayerType & player_type,
               const double kick_margin_delta = 0.0 )
{
    //      0.3 + 0.6*0.5 + 0.085 = 0.705
    // mid: 0.3 + 0.7*0.5 + 0.085 = 0.735
    //      0.3 + 0.8*0.5 + 0.085 = 0.765

    double kmargin = std::min( player_type.kickableMargin(),
                               player_type.kickableMargin() + kick_margin_delta );
    return bound( player_type.playerSize() + ServerParam::i().ballSize() + 0.1,
                  ( player_type.playerSize()
                    + ( kmargin * MID_RATE )
                    + ServerParam::i().ballSize() ),
                  player_type.kickableArea() - 0.2 );
}

/*-------------------------------------------------------------------*/
/*!
  \brief caclulate the distance of far side sub-target
  \param player_type calculated PlayerType
  \return distance from the center of the player
*/
inline
double
calc_far_dist( const PlayerType & player_type,
               const double kick_margin_delta = 0.0 )
{
    //      0.3 + 0.6*0.7 + 0.085 = 0.865 (=0.985-0.12 -> 0.785)
    // far: 0.3 + 0.7*0.7 + 0.085 = 0.875 (=1.085-0.21)
    //      0.3 + 0.8*0.7 + 0.085 = 0.945 (=1.185-0.24)

    //      0.3 + 0.6*0.68 + 0.085 = 0.793 (=0.985-0.192 -> 0.760)
    // far: 0.3 + 0.7*0.68 + 0.085 = 0.861 (=1.085-0.224 -> 0.860)
    //      0.3 + 0.8*0.68 + 0.085 = 0.929 (=1.185-0.256)

    //      0.3 + 0.6*0.675 + 0.085 = 0.79   (=0.985-0.195)
    // far: 0.3 + 0.7*0.675 + 0.085 = 0.8575 (=1.085-0.2275)
    //      0.3 + 0.8*0.675 + 0.085 = 0.925  (=1.185-0.26)

    double kmargin = std::min( player_type.kickableMargin(),
                               player_type.kickableMargin() + kick_margin_delta );
    return bound( player_type.playerSize() + ServerParam::i().ballSize() + 0.1,
                  ( player_type.playerSize()
                    + ( kmargin * FAR_SIDE_RATE )
                    + ServerParam::i().ballSize() ),
                  player_type.kickableArea() - 0.2 );
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
is_risky_flag( const int flag )
{
    return ( flag & KickTable::NEXT_TACKLABLE
             || flag & KickTable::NEXT_KICKABLE
             || flag & KickTable::TACKLABLE
             || flag & KickTable::MAYBE_RELEASE_INTERFERE
             || flag & KickTable::KICK_MISS_POSSIBILITY );
}

}

/*-------------------------------------------------------------------*/
/*!

 */
Vector2D
KickTable::calc_max_velocity( const AngleDeg & target_angle,
                              const double krate,
                              const Vector2D & ball_vel )
{
    const double ball_speed_max2 = std::pow( ServerParam::i().ballSpeedMax(), 2 );
    const double max_accel
        = std::min( ServerParam::i().maxPower() * krate,
                    ServerParam::i().ballAccelMax() );

    Ray2D desired_ray( Vector2D( 0.0, 0.0 ), target_angle );
    Circle2D next_reachable_circle( ball_vel, max_accel );

    Vector2D vel1, vel2;
    int num = next_reachable_circle.intersection( desired_ray, &vel1, &vel2 );

    if ( num == 0 )
    {
        return Vector2D( 0.0, 0.0 );
    }

    if ( num == 1 )
    {
        if ( vel1.r2() > ball_speed_max2 )
        {
            // next inertia ball point is within reachable circle.
            if ( next_reachable_circle.contains( Vector2D( 0.0, 0.0 ) ) )
            {
                // can adjust angle at least
                vel1.setLength( ServerParam::i().ballSpeedMax() );
            }
            else
            {
                // failed
                vel1.assign( 0.0, 0.0 );
            }
        }
        return vel1;
    }

    //
    // num == 2
    //   ball reachable circle does not contain the current ball pos.

    double length1 = vel1.r2();
    double length2 = vel2.r2();

    if ( length1 < length2 )
    {
        std::swap( vel1, vel2 );
        std::swap( length1, length2 );
    }

    if ( length1 > ball_speed_max2 )
    {
        if ( length2 > ball_speed_max2 )
        {
            // failed
            vel1.assign( 0.0, 0.0 );
        }
        else
        {
            vel1.setLength( ServerParam::i().ballSpeedMax() );
        }
    }

    return vel1;
}

/*-------------------------------------------------------------------*/
/*!

 */
KickTable &
KickTable::instance()
{
    static KickTable s_instance;
    return s_instance;
}

/*-------------------------------------------------------------------*/
/*!

 */
KickTable::KickTable()
    : M_player_size( 0.0 ),
      M_kickable_margin( 0.0 ),
      M_ball_size( 0.0 ),
      M_use_risky_node( false )
{
    for ( int i = 0; i < MAX_DEPTH; ++ i )
    {
        M_state_cache[i].reserve( NUM_STATE );
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
KickTable::createTables()
{
    const PlayerType player_type; // default type

    if ( std::fabs( M_player_size - player_type.playerSize() ) < rcsc::EPS
         && std::fabs( M_kickable_margin - player_type.kickableMargin() ) < rcsc::EPS
         && std::fabs( M_ball_size - ServerParam::i().ballSize() ) < rcsc::EPS )
    {
        return false;
    }

    //std::cerr << "createTables" << std::endl;

    M_player_size = player_type.playerSize();
    M_kickable_margin = player_type.kickableMargin();
    M_ball_size = ServerParam::i().ballSize();

    createStateList( player_type );

    Timer timer;

    const double angle_step = 360.0 / DEST_DIR_DIVS;
    AngleDeg angle = -180.0;

    for ( int i = 0; i < DEST_DIR_DIVS; ++i, angle += angle_step )
    {
        createTable( angle, M_tables[i] );
    }

    dlog.addText( Logger::KICK,
                  "(KickTable::createTables) elapsed %f [ms]",
                  timer.elapsedReal() );

#if 0
    const double kprate = ServerParam::i().kickPowerRate();
    for ( const State & s : M_state_list )
    {
        std::cout << "  state "
                  << " index=" << s.index_
                  << " angle="  << s.pos_.th()
                  << " dist="  << s.pos_.r()
                  << " pos=" << s.pos_
                  << " kick_rate=" << s.kick_rate_
                  << " (" << s.kick_rate_ / kprate * 100.0 << "%)"
                  << std::endl;
    }

    for ( int i = 0; i < DEST_DIR_DIVS; ++i, angle += angle_step )
    {
        std::cout << "create table " << i << " : angle="  << angle << std::endl;
        for ( const Path & p : M_tables[i] )
        {
            std::cout << "  table "
                      << " origin=" << p.origin_
                      << " dest=" << p.dest_
                      << " max_speed=" << p.max_speed_
                      << " power=" << p.power_
                      << std::endl;
        }
    }
#endif
    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
KickTable::read( const std::string & file_path )
{
    int n_line = 0;

    std::ifstream fin( file_path.c_str() );
    if ( ! fin.is_open() )
    {
        return false;
    }

    M_state_list.clear();
    M_state_list.reserve( NUM_STATE );

    for ( int dir = 0; dir < DEST_DIR_DIVS; ++dir )
    {
        M_tables[dir].clear();
        M_tables[dir].reserve( NUM_STATE * NUM_STATE );
    }

    std::string line_buf;

    //
    // read simulator paramters
    //
    double player_size = 0.0;
    double kickable_margin = 0.0;
    double ball_size = 0.0;

    ++n_line;
    if ( ! std::getline( fin, line_buf ) )
    {
        std::cerr << "read kick table ... failed. line=" << n_line
                  << std::endl;
        return false;
    }

    if ( std::sscanf( line_buf.c_str(),
                      " %lf %lf %lf ",
                      &player_size, &kickable_margin, &ball_size ) != 3 )
    {
        std::cerr << "read kick table ... failed. line=" << n_line
                  << std::endl;
        return false;
    }

    //
    // read state size
    //
    int state_size = 0;

    ++n_line;
    if ( ! std::getline( fin, line_buf ) )
    {
        std::cerr << "read kick table ... failed. line=" << n_line
                  << std::endl;
        return false;
    }

    if ( std::sscanf( line_buf.c_str(), " %d ", &state_size ) != 1 )
    {
        std::cerr << "read kick table ... failed. line=" << n_line
                  << std::endl;
        return false;
    }

    //
    // read state list
    //
    for ( int i = 0; i < state_size; ++i )
    {
        ++n_line;
        if ( ! std::getline( fin, line_buf ) )
        {
            std::cerr << "read kick table ... failed. line=" << n_line
                      << std::endl;
            return false;
        }

        State state;
        if ( std::sscanf( line_buf.c_str(),
                          " %d %lf %lf %lf ",
                          &state.index_,
                          &state.pos_.x, &state.pos_.y,
                          &state.kick_rate_ ) != 4 )
        {
            std::cerr << "read kick table ... failed. line=" << n_line
                      << std::endl;
            return false;
        }

        state.flag_ = SAFETY;
        M_state_list.push_back( state );

    }

    for ( int dir = 0; dir < DEST_DIR_DIVS; ++dir )
    {
        //
        // read path size
        //
        int path_size = 0;

        ++n_line;
        if ( ! std::getline( fin, line_buf ) )
        {
            std::cerr << "read kick table ... failed. line=" << n_line
                      << std::endl;
            return false;
        }

        if ( std::sscanf( line_buf.c_str(), " %d ", &path_size ) != 1 )
        {
            std::cerr << "read kick table ... failed. line=" << n_line
                      << std::endl;
            return false;
        }

        //
        // read path list
        //
        for ( int i = 0; i < path_size; ++i )
        {
            ++n_line;
            if ( ! std::getline( fin, line_buf ) )
            {
                std::cerr << "read kick table ... failed. line=" << n_line
                          << std::endl;
                return false;
            }

            Path path( 0, 0 );
            if ( std::sscanf( line_buf.c_str(),
                              " %d %d %lf %lf ",
                              &path.origin_,
                              &path.dest_,
                              &path.max_speed_,
                              &path.power_ ) != 4 )
            {
                return false;
            }

            M_tables[dir].push_back( path );
        }
    }

    M_player_size = player_size;
    M_kickable_margin = kickable_margin;
    M_ball_size = ball_size;

    std::cerr << "read kick table ... ok" << std::endl;

    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
KickTable::write( const std::string & file_path )
{
    std::ofstream fout( file_path.c_str() );
    if ( ! fout.is_open() )
    {
        return false;
    }

    //
    // write server parameters
    //
    fout << M_player_size << ' '
         << M_kickable_margin << ' '
         << M_ball_size << '\n';

    //
    // write state size
    //
    fout << M_state_list.size() << '\n';

    //
    // write state list
    //
    for ( const State & s : M_state_list )
    {
        fout << s.index_ << ' '
             << s.pos_.x << ' ' << s.pos_.y << ' '
             << s.kick_rate_ << '\n';
    }
    std::cerr << std::flush;

    for ( int dir = 0; dir < DEST_DIR_DIVS; ++dir )
    {
        fout << M_tables[dir].size() << '\n';

        for ( const Path & t : M_tables[dir] )
        {
            fout << t.origin_ << ' '
                 << t.dest_ << ' '
                 << t.max_speed_ << ' '
                 << t.power_ << '\n';
        }
    }

    fout.flush();
    fout.close();

    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
void
KickTable::createStateList( const PlayerType & player_type )
{
    const double near_dist = calc_near_dist( player_type, PlayerParam::i().kickableMarginDeltaMin() );
    const double mid_dist = calc_mid_dist( player_type, PlayerParam::i().kickableMarginDeltaMin() );
    const double far_dist = calc_far_dist( player_type, PlayerParam::i().kickableMarginDeltaMin() );

    const double near_angle_step = 360.0 / STATE_DIVS_NEAR;
    const double mid_angle_step = 360.0 / STATE_DIVS_MID;
    const double far_angle_step = 360.0 / STATE_DIVS_FAR;

#if 0
    std::cout << "createStateList"
              << "\n  near_dist=" << near_dist
              << " mid_dist=" << mid_dist
              << " far_dist=" << far_dist
              << "\n  near_angle_step=" << near_angle_step
              << " mid_angle_step=" << mid_angle_step
              << " far_angle_step=" << far_angle_step
              << std::endl;
#endif

    int index = 0;
    M_state_list.clear();
    M_state_list.reserve( NUM_STATE );

    for ( int near = 0; near < STATE_DIVS_NEAR; ++near )
    {
        AngleDeg angle = -180.0 + ( near_angle_step * near );
        Vector2D pos = Vector2D::polar2vector( near_dist, angle );
        double krate = player_type.kickRate( near_dist, angle.degree() );
        M_state_list.emplace_back( index, near_dist, pos, krate );
        ++index;
    }

    for ( int mid = 0; mid < STATE_DIVS_MID; ++mid )
    {
        AngleDeg angle = -180.0 + ( mid_angle_step * mid );
        Vector2D pos = Vector2D::polar2vector( mid_dist, angle );
        double krate = player_type.kickRate( mid_dist, angle.degree() );
        M_state_list.emplace_back( index, mid_dist, pos, krate );
        ++index;
    }

    for ( int far = 0; far < STATE_DIVS_FAR; ++far )
    {
        AngleDeg angle = -180.0 + ( far_angle_step * far );
        Vector2D pos = Vector2D::polar2vector( far_dist, angle );
        double krate = player_type.kickRate( far_dist, angle.degree() );
        M_state_list.emplace_back( index, far_dist, pos, krate );
        ++index;
    }

#if 0
    for ( const State & s : M_state_list )
    {
        std::cerr << s.index_ << ' '
                  << s.pos_.x << ' ' << s.pos_.y << ' '
                  << s.kick_rate_ << '\n';
    }
    std::cerr << std::flush;
#endif
}

/*-------------------------------------------------------------------*/
/*!

 */
void
KickTable::createTable( const AngleDeg & angle,
                        std::vector< Path > & table )
{
    const int max_combination = NUM_STATE * NUM_STATE;
    const int max_state = M_state_list.size();

    table.clear();
    table.reserve( max_combination );

    for ( int origin = 0; origin < max_state; ++origin )
    {
        for ( int dest = 0; dest < max_state; ++dest )
        {
            Vector2D vel = M_state_list[dest].pos_ - M_state_list[origin].pos_;
            Vector2D max_vel = calc_max_velocity( angle,
                                                  M_state_list[dest].kick_rate_,
                                                  vel );
            Vector2D accel = max_vel - vel;

            Path path( origin, dest );
            path.max_speed_ = max_vel.r();
            path.power_ = accel.r() / M_state_list[dest].kick_rate_;
            table.push_back( path );
        }
    }

    std::sort( table.begin(), table.end(), PathSorter() );

    if ( table.size() > MAX_TABLE_SIZE )
    {
        table.erase( table.begin() + MAX_TABLE_SIZE,
                     table.end() );
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void
KickTable::updateState( const WorldModel & world )
{
    static GameTime s_update_time( -1, 0 );

    if ( s_update_time == world.time() )
    {
        return;
    }

    s_update_time = world.time();

    //
    // update current state
    //
#ifdef DEBUG_PROFILE
    Timer timer;
#endif

    createStateCache( world );

#ifdef DEBUG_PROFILE
    dlog.addText( Logger::KICK,
                  "(KickTable::updateState) KickTable_elapsed %f [ms]",
                  timer.elapsedReal() );
#endif
}

/*-------------------------------------------------------------------*/
/*!

 */
void
KickTable::createStateCache( const WorldModel & world )
{
#ifdef DEBUG
    dlog.addText( Logger::KICK,
                  "(KickTable::createStateCache)" );
#endif

    const ServerParam & param = ServerParam::i();
    const Rect2D pitch
        = param.keepawayMode()
        ? Rect2D( Vector2D( - param.keepawayLength() * 0.5 + 0.2,
                            - param.keepawayWidth() * 0.5 + 0.2 ),
                  Size2D( param.keepawayLength() - 0.4,
                          param.keepawayWidth() - 0.4 ) )
        : Rect2D( Vector2D( - param.pitchHalfLength(),
                            - param.pitchHalfWidth() ),
                  Size2D( param.pitchLength(),
                          param.pitchWidth() ) );

    const PlayerType & self_type = world.self().playerType();
    const double near_dist = calc_near_dist( self_type );
    const double mid_dist = calc_mid_dist( self_type );
    const double far_dist = calc_far_dist( self_type );

    //
    // current state
    //
    {
        Vector2D rpos = world.ball().rpos();
        rpos.rotate( - world.self().body() );

        const double dist = rpos.r();
        const AngleDeg angle = rpos.th();

        const int dir_div = ( std::fabs( dist - near_dist ) < std::fabs( dist - far_dist )
                              ? STATE_DIVS_NEAR
                              : STATE_DIVS_FAR );

        M_current_state.index_ = static_cast< int >( rint( dir_div * rint( angle.degree() + 180.0 ) / 360.0 ) );
        if ( M_current_state.index_ >= dir_div ) M_current_state.index_ = 0;

        //M_current_state.pos_ = world.ball().rpos();
        M_current_state.pos_ = world.ball().pos();
        M_current_state.kick_rate_ = world.self().kickRate();
#ifdef DEBUG
        dlog.addText( Logger::KICK,
                      "__ current_state pos=(%.2f %.2f) kick_rate=%.3f",
                      world.ball().pos().x, world.ball().pos().y,
                      M_current_state.kick_rate_ );
#endif
        checkInterfereAt( world, 0, M_current_state );
    }

    //
    // create future state
    //

    Vector2D self_pos = world.self().pos();
    Vector2D self_vel = world.self().vel();

    for ( int i = 0; i < MAX_DEPTH; ++i )
    {
        M_state_cache[i].clear();

        self_pos += self_vel;
        self_vel *= self_type.playerDecay();

        int index = 0;
        for ( int near = 0; near < STATE_DIVS_NEAR; ++near )
        {
            Vector2D pos = M_state_list[index].pos_;
            double krate = self_type.kickRate( near_dist, pos.th().degree() );

            pos.rotate( world.self().body() );
            pos.setLength( near_dist );
            pos += self_pos;

            M_state_cache[i].emplace_back( index, near_dist, pos, krate );
            checkInterfereAt( world, i + 1, M_state_cache[i].back() );
            if ( ! pitch.contains( pos ) )
            {
                M_state_cache[i].back().flag_ |= OUT_OF_PITCH;
            }
#ifdef DEBUG
            dlog.addText( Logger::KICK,
                          "__ cache_near_%d index=%d pos=(%.2f %.2f) kick_rate=%f/%f",
                          i+1, index,
                          pos.x, pos.y,
                          krate, M_state_list[index].kick_rate_ );
#endif
            ++index;
        }

        for ( int mid = 0; mid < STATE_DIVS_MID; ++mid )
        {
            Vector2D pos = M_state_list[index].pos_;
            double krate = self_type.kickRate( mid_dist, pos.th().degree() );

            pos.rotate( world.self().body() );
            pos.setLength( mid_dist );
            pos += self_pos;

            M_state_cache[i].emplace_back( index, mid_dist, pos, krate );
            checkInterfereAt( world, i + 1, M_state_cache[i].back() );
            if ( ! pitch.contains( pos ) )
            {
                M_state_cache[i].back().flag_ |= OUT_OF_PITCH;
            }
#ifdef DEBUG
            dlog.addText( Logger::KICK,
                          "__ cache_mid_%d index=%d pos=(%.2f %.2f) kick_rate=%f/%f",
                          i+1, index,
                          pos.x, pos.y,
                          krate,  M_state_list[index].kick_rate_ );
#endif
            ++index;
        }

        for ( int far = 0; far < STATE_DIVS_FAR; ++far )
        {
            Vector2D pos = M_state_list[index].pos_;
            double krate = self_type.kickRate( far_dist, pos.th().degree() );

            pos.rotate( world.self().body() );
            pos.setLength( far_dist );
            pos += self_pos;

            M_state_cache[i].emplace_back( index, far_dist, pos, krate );
            checkInterfereAt( world, i + 1, M_state_cache[i].back() );
            if ( ! pitch.contains( pos ) )
            {
                M_state_cache[i].back().flag_ |= OUT_OF_PITCH;
            }
#ifdef DEBUG
            dlog.addText( Logger::KICK,
                          "__ cache_far_%d index=%d pos=(%.2f %.2f) kick_rate=%f/%f",
                          i+1, index,
                          pos.x, pos.y,
                          krate, M_state_list[index].kick_rate_ );
#endif
            ++index;
        }

    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void
KickTable::checkCollisionAfterRelease( const WorldModel & world,
                                       const Vector2D & target_point,
                                       const double first_speed )
{
#ifdef DEBUG
    dlog.addText( Logger::KICK,
                  "(KickTable::checkCollisionAfterRelease)" );
#endif

    const PlayerType & self_type = world.self().playerType();

    const double collide_dist2 = std::pow( self_type.playerSize() + ServerParam::i().ballSize(), 2 );

    Vector2D self_pos = world.self().pos();
    Vector2D self_vel = world.self().vel();

    // check the release kick from current state

    self_pos += self_vel;
    self_vel *= self_type.playerDecay();

    {
        Vector2D release_pos = ( target_point - M_current_state.pos_ );
        release_pos.setLength( first_speed );

        if ( self_pos.dist2( release_pos ) < collide_dist2 )
        {
#ifdef DEBUG
            dlog.addText( Logger::KICK,
                          "__ collision current_state self_pos=(%.2f %.2f) release_pos=(%.2f %.2f) dist=%.3f",
                          self_pos.x, self_pos.y,
                          release_pos.x, release_pos.y,
                          self_pos.dist( release_pos ) );
#endif
            M_current_state.flag_ |= SELF_COLLISION;
        }
        else
        {
#ifdef DEBUG
            dlog.addText( Logger::KICK,
                          "__ no collision with current_state" );
#endif
            M_current_state.flag_ &= ~SELF_COLLISION;
        }
    }

    // check the release kick from future state

    for ( int i = 0; i < MAX_DEPTH; ++ i )
    {
        self_pos += self_vel;
        self_vel *= self_type.playerDecay();

        for ( State & state : M_state_cache[i] )
        {
            Vector2D release_pos = ( target_point - state.pos_ );
            release_pos.setLength( first_speed );

            if ( self_pos.dist2( release_pos ) < collide_dist2 )
            {
#ifdef DEBUG
                dlog.addText( Logger::KICK,
                              "__ collision cached_state (%d) index=%d state_pos=(%.2f %.2f)"
                              " release_pos=(%.2f %.2f) dist=%.3f",
                              i + 1,
                              state.index_,
                              state.pos_.x, state.pos_.y,
                              release_pos.x, release_pos.y,
                              self_pos.dist( release_pos ) );
#endif
                state.flag_ |= SELF_COLLISION;
            }
            else
            {
#ifdef DEBUG
                dlog.addText( Logger::KICK,
                              "__ no collision cached_state (%d) index=%d (%.2f %.2f)",
                              i + 1,
                              state.index_,
                              state.pos_.x, state.pos_.y );
#endif
                state.flag_ &= ~SELF_COLLISION;
            }
        }
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void
KickTable::checkInterfereAt( const WorldModel & world,
                             const int step,
                             State & state )
{
    static const Rect2D penalty_area( Vector2D( ServerParam::i().theirPenaltyAreaLineX(),
                                                - ServerParam::i().penaltyAreaHalfWidth() ),
                                      Size2D( ServerParam::i().penaltyAreaLength(),
                                              ServerParam::i().penaltyAreaWidth() ) );
#ifndef DEBUG_OPPONENT
    (void)step;
#endif

    int flag = 0x0000;

    for ( const PlayerObject * o : world.opponentsFromBall() )
    {
        if ( o->posCount() >= 8 ) continue;
        if ( o->isGhost() ) continue;
        if ( o->distFromBall() > 10.0 ) break;

        const Vector2D opp_next = o->pos() + o->vel();
        const double opp_dist = opp_next.dist( state.pos_ );

        if ( o->isTackling() )
        {
            if ( opp_dist < ( o->playerTypePtr()->playerSize()
                              + ServerParam::i().ballSize() )
                 )
            {
                flag |= KICKABLE;
#ifdef DEBUG_OPPONENT
                dlog.addText( Logger::KICK,
                              "%d: state %d (%.2f %.2f) opp=%d(%.2f %.2f) is tackling but may collide",
                              step, state.index_,
                              state.pos_.x, state.pos_.y,
                              o->unum(),
                              o->pos().x, o->pos().y );
#endif
                break;
            }

            continue;
        }

        const double control_area = ( ( o->goalie()
                                        && penalty_area.contains( o->pos() )
                                        && penalty_area.contains( state.pos_ ) )
                                      ? ServerParam::i().catchableArea()
                                      : o->playerTypePtr()->kickableArea() );
        //
        // check kick possibility
        //

        if ( ! o->isGhost()
             && o->posCount() <= 2
             && opp_dist < control_area + 0.15 )
        {
            flag |= KICKABLE;
#ifdef DEBUG_OPPONENT
            dlog.addText( Logger::KICK,
                          "%d: state %d (%.2f %.2f) kickable opp %d(%.2f %.2f)",
                          step, state.index_,
                          state.pos_.x, state.pos_.y,
                          o->unum(),
                          o->pos().x, o->pos().y );
#endif
            break;
        }

        //
        //
        //
        const AngleDeg opp_body =  ( o->bodyCount() <= 1
                                     ? o->body()
                                     : ( state.pos_ - opp_next ).th() );
        Vector2D player_2_pos = state.pos_ - opp_next;
        player_2_pos.rotate( - opp_body );

        //
        // check tackle possibility
        //
        {
            double tackle_dist = ( player_2_pos.x > 0.0
                                   ? ServerParam::i().tackleDist()
                                   : ServerParam::i().tackleBackDist() );
            if ( tackle_dist > 1.0e-5 )
            {
                double tackle_prob = ( std::pow( player_2_pos.absX() / tackle_dist,
                                                 ServerParam::i().foulExponent() )
                                       + std::pow( player_2_pos.absY() / ServerParam::i().tackleWidth(),
                                                   ServerParam::i().foulExponent() ) );
                if ( tackle_prob < 1.0
                     && 1.0 - tackle_prob > 0.7 ) // success probability
                {
                    flag |= TACKLABLE;
#ifdef DEBUG_OPPONENT
                    dlog.addText( Logger::KICK,
                                  "%d: state %d (%.2f %.2f) tackle opp %d(%.1f %.1f)",
                                  step, state.index_,
                                  state.pos_.x, state.pos_.y,
                                  o->unum(),
                                  o->pos().x, o->pos().y );
#endif
                }
            }
        }

        // check kick or tackle possibility after dash

        const PlayerType * player_type = o->playerTypePtr();
        const double max_accel = ( ServerParam::i().maxDashPower()
                                   * player_type->dashPowerRate()
                                   * player_type->effortMax() );

        if ( player_2_pos.absY() < control_area
             && ( player_2_pos.absX() < max_accel
                  || ( player_2_pos + Vector2D( max_accel, 0.0 ) ).r() < control_area
                  || ( player_2_pos - Vector2D( max_accel, 0.0 ) ).r() < control_area )
             )
        {
            flag |= NEXT_KICKABLE;
#ifdef DEBUG_OPPONENT
            dlog.addText( Logger::KICK,
                          "%d: state %d (%.2f %.2f) next kickable opp %d(%.1f %.1f)",
                          step, state.index_,
                          state.pos_.x, state.pos_.y,
                          o->unum(),
                          o->pos().x, o->pos().y );
#endif
        }
        else if ( player_2_pos.absY() < ServerParam::i().tackleWidth() * 0.7
                  && player_2_pos.x > 0.0
                  && player_2_pos.x - max_accel < ServerParam::i().tackleDist() - 0.3 )
        {
#ifdef DEBUG_OPPONENT
            dlog.addText( Logger::KICK,
                          "%d: state %d (%.2f %.2f) next tackle opp %d(%.1f %.1f)",
                          step, state.index_,
                          state.pos_.x, state.pos_.y,
                          o->unum(),
                          o->pos().x, o->pos().y );
#endif
            flag |= NEXT_TACKLABLE;
        }
    }

    state.flag_ = flag;
}

/*-------------------------------------------------------------------*/
/*!

 */
void
KickTable::checkInterfereAfterRelease( const WorldModel & world,
                                       const Vector2D & target_point,
                                       const double first_speed )
{
    checkInterfereAfterRelease( world, target_point, first_speed, 1, M_current_state );

    for ( int i = 0; i < MAX_DEPTH; ++i )
    {
        for ( State & state : M_state_cache[i] )
        {
            state.flag_ &= ~RELEASE_INTERFERE;
            state.flag_ &= ~MAYBE_RELEASE_INTERFERE;

            checkInterfereAfterRelease( world, target_point, first_speed, i + 2, state );
        }
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void
KickTable::checkInterfereAfterRelease( const WorldModel & world,
                                       const Vector2D & target_point,
                                       const double first_speed,
                                       const int cycle,
                                       State & state )
{
    static const Rect2D penalty_area( Vector2D( ServerParam::i().theirPenaltyAreaLineX(),
                                                - ServerParam::i().penaltyAreaHalfWidth() ),
                                      Size2D( ServerParam::i().penaltyAreaLength(),
                                              ServerParam::i().penaltyAreaWidth() ) );

    Vector2D ball_pos = target_point - state.pos_;
    ball_pos.setLength( first_speed );
    ball_pos += state.pos_;

#ifdef DEBUG
    dlog.addText( Logger::KICK,
                  "____ state %d-%d (%.2f %.2f) check release interfere. bpos=(%.2f %.2f)",
                  cycle,
                  state.index_,
                  state.pos_.x, state.pos_.y,
                  ball_pos.x, ball_pos.y );
#endif

    for ( const PlayerObject * o : world.opponentsFromBall() )
    {
        if ( o->posCount() >= 8 ) continue;
        if ( o->isGhost() ) continue;
        if ( o->distFromBall() > 10.0 ) break;

        Vector2D opp_pos = o->inertiaPoint( cycle );
        if ( ! opp_pos.isValid() )
        {
            opp_pos = o->pos() + o->vel();
        }

        if ( o->isTackling() )
        {
            if ( opp_pos.dist( ball_pos ) < ( o->playerTypePtr()->playerSize()
                                              + ServerParam::i().ballSize() )
                 )
            {
                state.flag_ |= RELEASE_INTERFERE;
#ifdef DEBUG
                dlog.addText( Logger::KICK,
                              "____ state %d-%d (%.2f %.2f) opp=%d(%.1f %.1f)"
                              " is tackling but may be collided",
                              cycle,
                              state.index_,
                              state.pos_.x, state.pos_.y,
                              o->unum(),
                              o->pos().x, o->pos().y );
#endif
            }

            continue;
        }

        double control_area = ( ( o->goalie()
                                  && penalty_area.contains( opp_pos )
                                  && penalty_area.contains( ball_pos ) )
                                ? ServerParam::i().catchableArea()
                                : o->playerTypePtr()->kickableArea() );
        control_area += 0.1;
        double control_area2 = std::pow( control_area, 2 );

        if ( ball_pos.dist2( opp_pos ) < control_area2 )
        {
            if ( cycle <= 1 )
            {
                state.flag_ |= RELEASE_INTERFERE;
            }
            else
            {
                //state.flag_ |= MAYBE_RELEASE_INTERFERE;
                state.flag_ |= RELEASE_INTERFERE;
            }
#ifdef DEBUG
            if ( cycle <= 1 )
            dlog.addText( Logger::KICK,
                          "____ state %d-%d (%.2f %.2f) opp %d(%.1f %.1f) maybe interfere after release",
                          cycle,
                          state.index_,
                          state.pos_.x, state.pos_.y,
                          o->unum(),
                          o->pos().x, o->pos().y );
#endif
        }
#if 1
        else //if ( cycle <= 1 )
        {
            const AngleDeg opp_body =  ( o->bodyCount() <= 1
                                     ? o->body()
                                     : ( ball_pos - opp_pos ).th() );
            Vector2D player_2_pos = ball_pos - opp_pos;
            player_2_pos.rotate( - opp_body );

            {
                double tackle_dist = ( player_2_pos.x > 0.0
                                       ? ServerParam::i().tackleDist()
                                       : ServerParam::i().tackleBackDist() );
                if ( tackle_dist > 1.0e-5 )
                {
                    double tackle_prob = ( std::pow( player_2_pos.absX() / tackle_dist,
                                                     ServerParam::i().tackleExponent() )
                                           + std::pow( player_2_pos.absY() / ServerParam::i().tackleWidth(),
                                                       ServerParam::i().tackleExponent() ) );
                    if ( tackle_prob < 1.0
                         && 1.0 - tackle_prob > 0.8 ) // success probability
                    {
                        state.flag_ |= MAYBE_RELEASE_INTERFERE;
#ifdef DEBUG
                        dlog.addText( Logger::KICK,
                                      "____ state %d-%d (%.2f %.2f) opp %d(%.1f %.1f)"
                                      "maybe tackle after release",
                                      cycle,
                                      state.index_,
                                      state.pos_.x, state.pos_.y,
                                      o->unum(),
                                      o->pos().x, o->pos().y );
#endif
                    }
                }
            }

            //if ( o->bodyCount() <= 1 )
            {
                const PlayerType * player_type = o->playerTypePtr();
                const double max_accel
                    = ServerParam::i().maxDashPower()
                    * player_type->dashPowerRate()
                    * player_type->effortMax()
                    * 0.8;

                if ( player_2_pos.absY() < control_area - 0.1
                     && ( player_2_pos.absX() < max_accel
                          || ( player_2_pos + Vector2D( max_accel, 0.0 ) ).r() < control_area - 0.25
                          || ( player_2_pos - Vector2D( max_accel, 0.0 ) ).r() < control_area - 0.25 )
                     )
                {
                    state.flag_ |= MAYBE_RELEASE_INTERFERE;
#ifdef DEBUG
                    dlog.addText( Logger::KICK,
                                  "____ state %d-%d (%.2f %.2f) opp %d(%.1f %.1f)"
                                  "maybe kickable after release, opp dash",
                                  cycle,
                                  state.index_,
                                  state.pos_.x, state.pos_.y,
                                  o->unum(),
                                  o->pos().x, o->pos().y );
#endif
                }
                else if ( player_2_pos.absY() < ServerParam::i().tackleWidth() * 0.7
                          && player_2_pos.x - max_accel < ServerParam::i().tackleDist() - 0.5 )
                {
                    state.flag_ |= MAYBE_RELEASE_INTERFERE;
#ifdef DEBUG
                    dlog.addText( Logger::KICK,
                                  "____ state %d-%d (%.2f %.2f) opp %d(%.1f %.1f)"
                                  "maybe tackle after release, opp dash",
                                  cycle,
                                  state.index_,
                                  state.pos_.x, state.pos_.y,
                                  o->unum(),
                                  o->pos().x, o->pos().y );
#endif
                }
            }
        }
#endif
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
KickTable::simulateOneStep( const WorldModel & world,
                            const Vector2D & target_point,
                            const double first_speed )
{
    if ( M_current_state.flag_ & SELF_COLLISION )
    {
#ifdef DEBUG_ONE_STEP
        dlog.addText( Logger::KICK,
                      "xx__ 1 step: self collision" );
#endif
        return false;
    }

    if ( M_current_state.flag_ & RELEASE_INTERFERE )
    {
#ifdef DEBUG_ONE_STEP
        dlog.addText( Logger::KICK,
                      "xx__ 1 step: opponent can interfere after release" );
#endif
        return false;
    }

    const double current_max_accel = std::min( M_current_state.kick_rate_ * ServerParam::i().maxPower(),
                                               ServerParam::i().ballAccelMax() );
    Vector2D target_vel = ( target_point - world.ball().pos() );
    target_vel.setLength( first_speed );

    Vector2D accel = target_vel - world.ball().vel();
    double accel_r = accel.r();
    if ( accel_r > current_max_accel )
    {
#ifdef DEBUG_ONE_STEP
        dlog.addText( Logger::KICK,
                      "xx__ 1 step: failed. max_vel=required_accel=%f > max_accel=%f",
                      accel_r, current_max_accel );
#endif
        Vector2D max_vel = calc_max_velocity( target_vel.th(),
                                              M_current_state.kick_rate_,
                                              world.ball().vel() );
        accel = max_vel - world.ball().vel();
        M_candidates.push_back( Sequence() );
        M_candidates.back().index_ = 0;
        M_candidates.back().flag_ = M_current_state.flag_;
        M_candidates.back().pos_list_.push_back( world.ball().pos() + max_vel );
        M_candidates.back().speed_ = max_vel.r();
        M_candidates.back().power_ = accel.r() / M_current_state.kick_rate_;
        return false;
    }

    M_candidates.push_back( Sequence() );
    M_candidates.back().index_ = 0;
    M_candidates.back().flag_ = M_current_state.flag_;
    M_candidates.back().pos_list_.push_back( world.ball().pos() + target_vel );
    M_candidates.back().speed_ = first_speed;
    M_candidates.back().power_ = accel_r / M_current_state.kick_rate_;
#ifdef DEBUG_ONE_STEP
    dlog.addText( Logger::KICK,
                  "ok__ 1 step: target_vel=(%.2f %.2f)%.3f required_accel=%.3f < max_accel=%.3f"
                  " kick_rate=%f power=%.1f",
                  target_vel.x, target_vel.y,
                  first_speed,
                  accel_r,
                  current_max_accel,
                  M_current_state.kick_rate_,
                  M_candidates.back().power_ );
#endif
    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
KickTable::simulateTwoStep( const WorldModel & world,
                            const Vector2D & target_point,
                            const double first_speed )
{
    static const double max_power = ServerParam::i().maxPower();
    static const double accel_max = ServerParam::i().ballAccelMax();
    static const double ball_decay = ServerParam::i().ballDecay();

    const PlayerType & self_type = world.self().playerType();
    const double current_max_accel = std::min( M_current_state.kick_rate_ * max_power, accel_max );

    const ServerParam & param = ServerParam::i();
    const double my_kickable_area = self_type.kickableArea();

    const double my_noise = world.self().vel().r() * param.playerRand();
    const double current_dir_diff_rate
        = ( world.ball().angleFromSelf() - world.self().body() ).abs() / 180.0;
    const double current_dist_rate = ( ( world.ball().distFromSelf()
                                         - self_type.playerSize()
                                         - param.ballSize() )
                                       / self_type.kickableMargin() );
    const double current_pos_rate
        = 0.5 + 0.25 * ( current_dir_diff_rate + current_dist_rate );
    const double current_speed_rate
        = 0.5 + 0.5 * ( world.ball().vel().r()
                        / ( param.ballSpeedMax() * param.ballDecay() ) );

    int success_count = 0;
    double max_speed2 = 0.0;
    int count = 1;

    for ( int i = 0; i < NUM_STATE; ++i, ++count )
    {
        const State & state = M_state_cache[0][i];

        if ( state.flag_ & OUT_OF_PITCH )
        {
#ifdef DEBUG_TWO_STEP
            dlog.addText( Logger::KICK,
                          "%d: xx__ 2 step: skip. out of pitch. state_pos=(%.2f %.2f)",
                          count, state.pos_.x, state.pos_.y );
#endif
            continue;
        }

        if ( state.flag_ & KICKABLE )
        {
#ifdef DEBUG_TWO_STEP
            dlog.addText( Logger::KICK,
                          "%d: xx__ 2 step: skip. exist kicable opp. state_pos=(%.2f %.2f)",
                          count, state.pos_.x, state.pos_.y );
#endif
            continue;
        }

        if ( state.flag_ & SELF_COLLISION )
        {
#ifdef DEBUG_TWO_STEP
            dlog.addText( Logger::KICK,
                          "%d: xx__ 2 step: skip. self collision. state_pos=(%.2f %.2f)",
                          count, state.pos_.x, state.pos_.y );
#endif
            continue;
        }

        if ( state.flag_ & RELEASE_INTERFERE )
        {
#ifdef DEBUG_TWO_STEP
            dlog.addText( Logger::KICK,
                          "%d: xx__ 2 step: interfere after release. state_pos=(%.2f %.2f)",
                          count, state.pos_.x, state.pos_.y );
#endif
            //return false;
            continue;
        }

        if ( ! M_use_risky_node
             && is_risky_flag( state.flag_ ) )
        {
            continue;
        }

        int kick_miss_flag = SAFETY;
        const Vector2D target_vel = ( target_point - state.pos_ ).setLengthVector( first_speed );

        Vector2D vel = state.pos_ - world.ball().pos();
        Vector2D accel = vel - world.ball().vel();
        double accel_r = accel.r();

        if ( accel_r > current_max_accel )
        {
#ifdef DEBUG_TWO_STEP
            dlog.addText( Logger::KICK,
                          "%d: xx__ 2 step: failed(1) required_accel=%.3f > max_accel=%.3f",
                          count, accel_r, current_max_accel );
#endif
            continue;
        }
        {
            double kick_power = accel_r / world.self().kickRate();
            double ball_noise = vel.r() * param.ballRand();
            double max_kick_rand
                = self_type.kickRand()
                * ( kick_power / param.maxPower() )
                * ( current_pos_rate + current_speed_rate );
            if ( ( my_noise + ball_noise + max_kick_rand ) //* 0.9
                 > my_kickable_area - state.dist_ - 0.05 ) //0.1 )
            {
#ifdef DEBUG_TWO_STEP
                dlog.addText( Logger::KICK,
                              "%d: xx__ 2 step: failed. buffer is not safety. power=%.3f"
                              " my_kickable=%.3f state_dist=%.3f,"
                              " noise=%f(my_noise=%f ball_noise=%f kick_rand=%f)",
                              count, kick_power,
                              my_kickable_area, state.dist_,
                              ( my_noise + ball_noise + max_kick_rand ) * 0.9,
                              my_noise, ball_noise, max_kick_rand );
#endif
                kick_miss_flag |= KICK_MISS_POSSIBILITY;
                // if ( ! M_use_risky_node )
                // {
                //     continue;
                // }
            }
        }

        vel *= ball_decay;

        accel = target_vel - vel;
        accel_r = accel.r();

        if ( accel_r > std::min( state.kick_rate_ * max_power, accel_max ) )
        {
#ifdef DEBUG_TWO_STEP
            dlog.addText( Logger::KICK,
                          "%d: xx__ 2step: failed(2) required_accel=%.3f > max_accel=%.3f",
                          count, accel_r, std::min( state.kick_rate_ * max_power, accel_max ) );
#endif
            if ( success_count == 0 )
            {
                Vector2D max_vel = calc_max_velocity( target_vel.th(),
                                                      state.kick_rate_,
                                                      vel );
                double d2 = max_vel.r2();
                if ( max_speed2 < d2 )
                {
                    if ( max_speed2 == 0.0 )
                    {
                        M_candidates.push_back( Sequence() );
                    }
                    max_speed2 = d2;
                    accel = max_vel - vel;

                    M_candidates.back().index_ = 100 + count;
                    M_candidates.back().flag_ = ( ( M_current_state.flag_ & ~RELEASE_INTERFERE )
                                                  | state.flag_ );
                    M_candidates.back().pos_list_.clear();
                    M_candidates.back().pos_list_.push_back( state.pos_ );
                    M_candidates.back().pos_list_.push_back( state.pos_ + max_vel );
                    M_candidates.back().speed_ = std::sqrt( max_speed2 );
                    M_candidates.back().power_ = accel.r() / state.kick_rate_;
#ifdef DEBUG_TWO_STEP
                    dlog.addText( Logger::KICK,
                                  "%d: ____ update max vel (%.2f %.2f) %.3f",
                                  count, max_vel.x, max_vel.y,
                                  M_candidates.back().speed_ );
#endif
                }
            }
            continue;
        }

        M_candidates.push_back( Sequence() );
        M_candidates.back().index_ = 100 + count;
        M_candidates.back().flag_ = ( ( M_current_state.flag_ & ~RELEASE_INTERFERE )
                                      | state.flag_
                                      | kick_miss_flag );
        M_candidates.back().pos_list_.push_back( state.pos_ );
        M_candidates.back().pos_list_.push_back( state.pos_ + target_vel );
        M_candidates.back().speed_ = first_speed;
        M_candidates.back().power_ = accel_r / state.kick_rate_;
#ifdef DEBUG_TWO_STEP
        dlog.addText( Logger::KICK,
                      "%d: ok__ 2 step: last_power=%.2f subtarget=(%.2f %.2f)",
                      count, M_candidates.back().power_,
                      state.pos_.x, state.pos_.y );
#endif
    }

    return false;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool
KickTable::simulateThreeStep( const WorldModel & world,
                              const Vector2D & target_point,
                              const double first_speed )
{
    static const double max_power = ServerParam::i().maxPower();
    static const double accel_max = ServerParam::i().ballAccelMax();
    static const double ball_decay = ServerParam::i().ballDecay();

    const double current_max_accel = std::min( M_current_state.kick_rate_ * max_power,
                                               accel_max );
    const double current_max_accel2 = current_max_accel * current_max_accel;
#if 1
    const ServerParam & param = ServerParam::i();
    const PlayerType & self_type = world.self().playerType();

    const double my_kickable_area = self_type.kickableArea();

    const double my_noise1 = world.self().vel().r() * param.playerRand();
    //const double my_noise2 = my_noise1 * self_type.playerDecay();
    const double current_dir_diff_rate
        = ( world.ball().angleFromSelf() - world.self().body() ).abs() / 180.0;
    const double current_dist_rate = ( ( world.ball().distFromSelf()
                                         - self_type.playerSize()
                                         - param.ballSize() )
                                       / self_type.kickableMargin() );
    const double current_pos_rate
        = 0.5 + 0.25 * ( current_dir_diff_rate + current_dist_rate );
    const double current_speed_rate
        = 0.5 + 0.5 * ( world.ball().vel().r()
                        / ( param.ballSpeedMax() * param.ballDecay() ) );
#endif
    AngleDeg target_rel_angle = ( target_point - world.self().pos() ).th() - world.self().body();
    double angle_deg = target_rel_angle.degree() + 180.0;
    int target_angle_index = static_cast< int >( rint( DEST_DIR_DIVS * ( angle_deg / 360.0 ) ) );
    if ( target_angle_index >= DEST_DIR_DIVS ) target_angle_index = 0;

#ifdef DEBUG
    dlog.addText( Logger::KICK,
                  "(KickTable::simulateThreeStep) target angle index = %d ",
                  target_angle_index );
#endif

    const std::vector< Path > & table = M_tables[target_angle_index];

    int success_count = 0;
    double max_speed2 = 0.0;

    size_t count = 0;
    for ( std::vector< Path >::const_iterator it = table.begin(), end = table.end();
          it != end && count < MAX_TABLE_SIZE && success_count <= 10;
          ++it, ++count )
    {
        const State & state_1st = M_state_cache[0][it->origin_];
        const State & state_2nd = M_state_cache[1][it->dest_];

        if ( state_1st.flag_ & OUT_OF_PITCH )
        {
#ifdef DEBUG_THREE_STEP_DETAIL
            dlog.addText( Logger::KICK,
                          "%zd: xx__ 3 step: skip. out of pitch. state_1st pos=(%.2f %.2f)",
                          count, state_1st.pos_.x, state_1st.pos_.y );
#endif
            continue;
        }

        if ( state_2nd.flag_ & OUT_OF_PITCH )
        {
#ifdef DEBUG_THREE_STEP_DETAIL
            dlog.addText( Logger::KICK,
                          "%zd: xx__ 3 step: skip. out of pitch. state_2nd pos=(%.2f %.2f)",
                          count, state_2nd.pos_.x, state_2nd.pos_.y );
#endif
            continue;
        }

        if ( state_1st.flag_ & KICKABLE )
        {
#ifdef DEBUG_THREE_STEP_DETAIL
            dlog.addText( Logger::KICK,
                          "%zd: xx__ 3 step: skip. exist kicable opp. state_1st pos=(%.2f %.2f)",
                          count, state_1st.pos_.x, state_1st.pos_.y );
#endif
            continue;
        }

        if ( state_2nd.flag_ & KICKABLE )
        {
#ifdef DEBUG_THREE_STEP_DETAIL
            dlog.addText( Logger::KICK,
                          "%zd: xx__ 3 step: skip. exist kicable opp. state_2nd pos=(%.2f %.2f)",
                          count, state_2nd.pos_.x, state_2nd.pos_.y );
#endif
            continue;
        }

        if ( state_2nd.flag_ & SELF_COLLISION )
        {
#ifdef DEBUG_THREE_STEP_DETAIL
            dlog.addText( Logger::KICK,
                          "%zd: xx__ 3 step: skip. self collision. state_2nd_pos=(%.2f %.2f)",
                          count, state_2nd.pos_.x, state_2nd.pos_.y );
#endif
            continue;
        }

        if ( state_2nd.flag_ & RELEASE_INTERFERE )
        {
#ifdef DEBUG_THREE_STEP_DETAIL
            dlog.addText( Logger::KICK,
                          "%zd: xx__ 3 step: interfere after release. state_pos=(%.2f %.2f)",
                          count, state_2nd.pos_.x, state_2nd.pos_.y );
#endif
            //return false;
            continue;
        }

        if ( ! M_use_risky_node
             && ( is_risky_flag( state_1st.flag_ )
                  || is_risky_flag( state_2nd.flag_ ) )
             )
        {
            continue;
        }

        const Vector2D target_vel = ( target_point - state_2nd.pos_ ).setLengthVector( first_speed );

        int kick_miss_flag = SAFETY;

        Vector2D vel1 = state_1st.pos_ - world.ball().pos();
        Vector2D accel = vel1 - world.ball().vel();
        double accel_r2 = accel.r2();

        if ( accel_r2 > current_max_accel2 )
        {
#ifdef DEBUG_THREE_STEP_DETAIL
            dlog.addText( Logger::KICK,
                          "%zd: xx__ 3 step: failed(1) required_accel=%.3f > max_accel=%.3f",
                          count, std::sqrt( accel_r2 ), std::sqrt( current_max_accel2 ) );
#endif
            continue;
        }

#if 1
        {
            double kick_power = std::sqrt( accel_r2 ) / world.self().kickRate();
            double ball_noise = vel1.r() * param.ballRand();
            double max_kick_rand
                = self_type.kickRand()
                * ( kick_power / param.maxPower() )
                * ( current_pos_rate + current_speed_rate );
            if ( ( my_noise1 + ball_noise + max_kick_rand )
                 > my_kickable_area - state_1st.dist_ - 0.1 )
            {
#ifdef DEBUG_THREE_STEP
                dlog.addText( Logger::KICK,
                              "%zd: xx__ 3 step: 1st kick may cause unkickable. power=%.3f"
                              " my_kickable=%.3f state_dist=%.3f,"
                              " noise=%f(my_noise=%f ball_noise=%f kick_rand=%f)",
                              count,
                              kick_power,
                              my_kickable_area, state_1st.dist_,
                              ( my_noise1 + ball_noise + max_kick_rand ) * 0.9,
                              my_noise1, ball_noise, max_kick_rand );
#endif
                kick_miss_flag |= KICK_MISS_POSSIBILITY;
                // if ( ! M_use_risky_node )
                // {
                //     continue;
                // }
            }
        }
#endif

        vel1 *= ball_decay;

        Vector2D vel2 = state_2nd.pos_ - state_1st.pos_;
        accel = vel2 - vel1;
        accel_r2 = accel.r2();

        if ( accel_r2 > square( std::min( state_1st.kick_rate_ * max_power * 0.9, accel_max ) ) )
        {
#ifdef DEBUG_THREE_STEP_DETAIL
            dlog.addText( Logger::KICK,
                          "%zd: xx__ 3 step: failed(2) required_accel=%.3f > max_accel=%.3f",
                          count,
                          std::sqrt( accel_r2 ),
                          std::min( state_1st.kick_rate_ * max_power, accel_max ) );
#endif
            continue;
        }
        vel2 *= ball_decay;

        accel = target_vel - vel2;
        accel_r2 = accel.r2();
        if ( accel_r2 > square( std::min( state_2nd.kick_rate_ * max_power, accel_max ) ) )
        {
#ifdef DEBUG_THREE_STEP_DETAIL
            dlog.addText( Logger::KICK,
                          "xx__ 3 step: failed(3) required_accel=%.3f > max_accel=%.3f",
                          std::sqrt( accel_r2 ),
                          std::min( state_2nd.kick_rate_ * max_power, accel_max ) );
#endif
            if ( success_count == 0 )
            {
                Vector2D max_vel = calc_max_velocity( target_vel.th(),
                                                      state_2nd.kick_rate_,
                                                      vel2 );
                double d2 = max_vel.r2();
                if ( max_speed2 < d2 )
                {
                    if ( M_candidates.empty() ) // max_speed2 == 0.0
                    {
                        M_candidates.push_back( Sequence() );
                    }
                    max_speed2 = d2;
                    accel = max_vel - vel2;

                    M_candidates.back().index_ = 10000 + count;
                    M_candidates.back().flag_ = ( ( M_current_state.flag_ & ~RELEASE_INTERFERE )
                                                  | ( state_1st.flag_ & ~RELEASE_INTERFERE )
                                                  | state_2nd.flag_ );
                    M_candidates.back().pos_list_.clear();
                    M_candidates.back().pos_list_.push_back( state_1st.pos_ );
                    M_candidates.back().pos_list_.push_back( state_2nd.pos_ );
                    M_candidates.back().pos_list_.push_back( state_2nd.pos_ + max_vel );
                    M_candidates.back().speed_ = std::sqrt( max_speed2 );
                    M_candidates.back().power_ = accel.r() / state_2nd.kick_rate_;

#ifdef DEBUG_THREE_STEP
                    dlog.addText( Logger::KICK,
                                  "____ update max vel (%.2f %.2f) %.3f",
                                  max_vel.x, max_vel.y,
                                  M_candidates.back().speed_ );
#endif
                }
            }
            continue;
        }

        M_candidates.push_back( Sequence() );
        M_candidates.back().index_ = 10000 + count;
        M_candidates.back().flag_ = ( ( M_current_state.flag_ & ~RELEASE_INTERFERE )
                                      | ( state_1st.flag_ & ~RELEASE_INTERFERE )
                                      | state_2nd.flag_
                                      | kick_miss_flag );
        M_candidates.back().pos_list_.push_back( state_1st.pos_ );
        M_candidates.back().pos_list_.push_back( state_2nd.pos_ );
        M_candidates.back().pos_list_.push_back( state_2nd.pos_ + target_vel );
        M_candidates.back().speed_ = first_speed;
        M_candidates.back().power_ = std::sqrt( accel_r2 ) / state_2nd.kick_rate_;

#ifdef DEBUG_THREE_STEP
        dlog.addText( Logger::KICK,
                      "%zd: ok__ 3 step: last_power=%.2f sub1=(%.2f %.2f) sub2(%.2f %.2f)",
                      count,
                      M_candidates.back().power_,
                      state_1st.pos_.x, state_1st.pos_.y,
                      state_2nd.pos_.x, state_2nd.pos_.y );
#endif
        ++success_count;
    }

#ifdef DEBUG_THREE_STEP
    dlog.addText( Logger::KICK,
                  "simulateThreeKick() solution_size=%d",
                  success_count );
    if ( success_count == 0 )
    {
        dlog.addText( Logger::KICK,
                      "simulateThreeKick() max_speed=%.3f",
                      std::sqrt( max_speed2 ) );
    }
#endif

    return success_count > 0;
}

/*-------------------------------------------------------------------*/
/*!

 */
void
KickTable::evaluate( const WorldModel & wm,
                     const double first_speed,
                     const double allowable_speed )
{
    dlog.addText( Logger::KICK,
                  "(KickTable::evaluate) candidate size=%zd",
                  M_candidates.size() );

#ifndef DEBUG_PRINT_EVALUATE
    (void)wm;
#endif

    const double power_thr1 = ServerParam::i().maxPower() * 0.94;
    const double power_thr2 = ServerParam::i().maxPower() * 0.9;

    int count = 0;
    for ( Sequence & seq : M_candidates )
    {
        ++count;

        const int n_kick = seq.pos_list_.size();

        seq.score_ = 1000.0;

        if ( seq.speed_ < first_speed )
        {
            if ( n_kick > 1
                 || seq.speed_ < allowable_speed )
            {
                seq.score_ = -10000.0;
                seq.score_ -= ( first_speed - seq.speed_ ) * 100000.0;
            }
            else
            {
                seq.score_ -= 50.0;
            }
        }

        if ( seq.flag_ & TACKLABLE ) seq.score_ -= 500.0;

        if ( seq.flag_ & NEXT_TACKLABLE ) seq.score_ -= 300.0;

        if ( seq.flag_ & NEXT_KICKABLE ) seq.score_ -= 600.0;

        if ( seq.flag_ & MAYBE_RELEASE_INTERFERE )
        {
            if ( n_kick == 1 )
            {
                seq.score_ -= 250.0;
            }
            else
            {
                seq.score_ -= 200.0;
            }
        }

        if ( n_kick == 3 )
        {
            seq.score_ -= 200.0;
        }
        else if ( n_kick == 2 )
        {
            seq.score_ -= 50.0;
        }

        if ( n_kick > 1 )
        {
            if ( seq.power_ > power_thr1 )
            {
                seq.score_ -= 75.0;
            }
            else if ( seq.power_ > power_thr2 )
            {
                seq.score_ -= 25.0;
            }
        }

        seq.score_ -= seq.power_ * 0.5;

        if ( seq.flag_ & KICK_MISS_POSSIBILITY )
        {
            seq.score_ -= 30.0;
#ifdef DEBUG_EVALUATE
            dlog.addText( Logger::KICK,
                          "%d: (eval) %d maybe kick failure flag=%x n_kick=%d speed=%.3f last_kick_power=%f",
                          count, seq.index_, seq.flag_, n_kick, seq.speed_, seq.power_ );
#endif
        }
#ifdef DEBUG_EVALUATE
        dlog.addText( Logger::KICK,
                      "%d: (eval) %d score %.2f flag=%x n_kick=%d speed=%.3f last_kick_power=%f",
                      count, seq.index_, seq.score_, seq.flag_, n_kick, seq.speed_, seq.power_ );

        //if ( count == 4 ) debug_print_sequence( wm, *it );
#endif
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void
KickTable::debugPrintStateCache()
{
    for ( int i = 0; i < MAX_DEPTH; ++i )
    {
        for ( const State & s : M_state_cache[i] )
        {
            char buf[8];

            if ( s.flag_ & OUT_OF_PITCH ) snprintf( buf, 8, "#333" );
            else if  ( s.flag_ & SELF_COLLISION ) snprintf( buf, 8, "#FF0" );
            if ( s.flag_ & KICKABLE ) snprintf( buf, 8, "#F00" );
            else if ( s.flag_ & TACKLABLE ) snprintf( buf, 8, "#800" );
            else if ( s.flag_ & NEXT_KICKABLE ) snprintf( buf, 8, "#00F" );
            else if ( s.flag_ & NEXT_TACKLABLE ) snprintf( buf, 8, "#008" );
            else if ( s.flag_ & RELEASE_INTERFERE ) snprintf( buf, 8, "#0F0" );
            else if ( s.flag_ & MAYBE_RELEASE_INTERFERE ) snprintf( buf, 8, "#080" );
            else if ( s.flag_ & KICK_MISS_POSSIBILITY ) snprintf( buf, 8, "#0FF" );
            dlog.addRect( Logger::KICK,
                          s.pos_.x - 0.01, s.pos_.y - 0.01, 0.02, 0.02,
                          buf, true );

            snprintf( buf, 8, "%d", i+1 );
            dlog.addMessage( Logger::KICK, s.pos_, buf );
        }
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
void
KickTable::debugPrintSequence( const WorldModel & wm,
                               const KickTable::Sequence & seq )
{
    if ( ! seq.pos_list_.empty() )
    {
        dlog.addLine( Logger::KICK,
                      wm.ball().pos(), seq.pos_list_.front(),
                      "#F00" );
        dlog.addRect( Logger::KICK,
                      seq.pos_list_.front().x - 0.02,
                      seq.pos_list_.front().y - 0.02,
                      0.04, 0.04, "#F00" );

        for ( size_t i = 1; i < seq.pos_list_.size(); ++i )
        {
            dlog.addLine( Logger::KICK,
                          seq.pos_list_[i-1], seq.pos_list_[i],
                          std::max( 0, int( 255 - i * 80 ) ), 0, 0 );
            dlog.addRect( Logger::KICK,
                      seq.pos_list_[i].x - 0.02,
                      seq.pos_list_[i].y - 0.02,
                      0.04, 0.04, "#F00" );
        }
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
namespace {
bool
check_candidates_max_speed( const std::vector< KickTable::Sequence > & candidates,
                            const double speed_thr )
{
    for ( const KickTable::Sequence & seq : candidates )
    {
        if ( seq.speed_ > speed_thr )
        {
            // dlog.addText( Logger::KICK,
            //               "(KickTable::check_candidate) OK %d speed=%.3f  thr=%.3f",
            //               seq.index_, seq.speed_, speed_thr );
            dlog.addText( Logger::KICK,
                          "(KickTable::check_candidate) OK found" );
            return true;
        }
        // dlog.addText( Logger::KICK,
        //               "(KickTable::check_candidate) NG %d speed=%.3f  thr=%.3f",
        //               seq.index_, seq.speed_, speed_thr );
    }

    dlog.addText( Logger::KICK,
                  "(KickTable::check_candidate) NG not found" );
    return false;
}

}

/*-------------------------------------------------------------------*/
/*!

 */
bool
KickTable::simulate( const WorldModel & world,
                     const Vector2D & target_point,
                     const double first_speed,
                     const double allowable_speed,
                     const int max_step,
                     Sequence & sequence )
{
    if ( M_state_list.empty() )
    {
        dlog.addText( Logger::KICK,
                      "(KickTable::simulate) KickTable is not initialized!." );
        std::cerr << "KickTable has not been initialized! "
                  << "KickTable::instance().createTable() has to be called before using KickTable::simulate()."
                  << std::endl;
        return false;
    }

#ifdef DEBUG_PROFILE
    Timer timer;
#endif

    double target_speed = bound( 0.0,
                                 first_speed,
                                 ServerParam::i().ballSpeedMax() );
    double speed_thr = bound( 0.0,
                              allowable_speed,
                              target_speed );

    dlog.addText( Logger::KICK,
                  "(KickTable::simulate) start. target=(%.2f %.2f) speed=%.2f",
                  target_point.x, target_point.y,
                  target_speed );

    M_candidates.clear();

    updateState( world );

    checkCollisionAfterRelease( world,
                                target_point,
                                target_speed );
    checkInterfereAfterRelease( world,
                                target_point,
                                target_speed );

#ifdef DEBUG_PRINT_STATE_CACHE
    debugPrintStateCache();
#endif

    if ( max_step >= 1
         && simulateOneStep( world,
                             target_point,
                             target_speed ) )
    {
        dlog.addText( Logger::KICK,
                      "(KickTable::simulate) found 1 step" );
    }

    M_use_risky_node = false;

    if ( max_step >= 2
         && simulateTwoStep( world,
                             target_point,
                             target_speed ) )
    {
        dlog.addText( Logger::KICK,
                      "(KickTable::simulate) found 2 step" );
    }

    if ( max_step >= 3
         && simulateThreeStep( world,
                               target_point,
                               target_speed ) )
    {
        dlog.addText( Logger::KICK,
                      "(KickTable::simulate) found 3 step" );
    }

    // dlog.addText( Logger::KICK,
    //               "(KickTable::simulate) candidate size = %zd", M_candidates.size() );

    if ( ! check_candidates_max_speed( M_candidates, speed_thr ) )
    {
        M_use_risky_node = true;

        // dlog.addText( Logger::KICK,
        //               "(KickTable::simulate) try risky mode" );

        if ( max_step >= 2
             && simulateTwoStep( world,
                                 target_point,
                                 target_speed ) )
        {
            dlog.addText( Logger::KICK,
                          "(KickTable::simulate) found 2 step with risky node" );
        }

        if ( max_step >= 3
             && simulateThreeStep( world,
                                   target_point,
                                   target_speed ) )
        {
            dlog.addText( Logger::KICK,
                          "(KickTable::simulate) found 3 step with risky node" );
        }
    }

    // TODO:
    // 4 steps simulation

    if ( M_candidates.empty() )
    {
        dlog.addText( Logger::KICK,
                      "(KickTable::simulate) No candidate" );
        return false;
    }

    // TODO:
    // dynamic evaluator
    evaluate( world, target_speed, speed_thr );

    sequence = *std::max_element( M_candidates.begin(),
                                  M_candidates.end(),
                                  SequenceSorter() );

    dlog.addText( Logger::KICK,
                  "(KickTable::simulate) result next_pos=(%.2f %.2f) flag=%x n_kick=%d speed=%.2f power=%.2f score=%.2f",
                  sequence.pos_list_.front().x,
                  sequence.pos_list_.front().y,
                  sequence.flag_,
                  (int)sequence.pos_list_.size(),
                  sequence.speed_,
                  sequence.power_,
                  sequence.score_ );
#ifdef DEBUG_PRINT_SEQUENCE
    debugPrintSequence( world, sequence );
#endif

#ifdef DEBUG_PROFILE
    dlog.addText( Logger::KICK,
                  "(KickTable::simulate) KickTable_elapsed=%f [ms].",
                  timer.elapsedReal() );
#endif
    return sequence.speed_ >= target_speed - rcsc::EPS;
}
