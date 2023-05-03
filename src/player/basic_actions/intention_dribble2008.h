// -*-c++-*-

/*!
  \file intention_dribble2008.h
  \brief queued dribble behavior
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

#ifndef RCSC_ACTION_INTENTION_DRIBBLE_2008_H
#define RCSC_ACTION_INTENTION_DRIBBLE_2008_H

#include <rcsc/player/soccer_intention.h>
#include <rcsc/geom/vector_2d.h>
#include <rcsc/game_time.h>

namespace rcsc {
class WorldModel;
}

/*!
  \class IntentionDribble2008
  \brief queued dribble behavior
*/
class IntentionDribble2008
    : public rcsc::SoccerIntention {
private:
    const rcsc::Vector2D M_target_point; //!< dribble target point
    const double M_dist_thr; //!< distance threshold from the target point

    int M_turn_step; //!< remained turn step
    int M_dash_step; //!< remained dash step

    const double M_dash_power_abs; //!< absolute dash power
    const bool M_back_dash_mode; //!< if true, agent try to dribble backwards

    rcsc::GameTime M_last_execute_time; //!< last executed time

public:
    /*!
      \brief constructor
      \param target_point dribble target point
      \param dist_thr distance threshold from the target point
      \param turn_step scheduled turn step
      \param dash_step scheduled dash step
      \param dash_power_abs absolute dash power
      \param back_dash if true, agent try to dribble backwards.
      \param start_time dribble start time
     */
    IntentionDribble2008( const rcsc::Vector2D & target_point,
                          const double & dist_thr,
                          const int turn_step,
                          const int dash_step,
                          const double & dash_power_abs,
                          const bool back_dash,
                          const rcsc::GameTime & start_time )
        : M_target_point( target_point )
        , M_dist_thr( dist_thr )
        , M_turn_step( turn_step )
        , M_dash_step( dash_step )
        , M_dash_power_abs( dash_power_abs )
        , M_back_dash_mode( back_dash )
        , M_last_execute_time( start_time )
      { }

    /*!
      \brief check the status.
      \param agent pointer to the agent itself.
      \return true if action queue is finished.
     */
    bool finished( const rcsc::PlayerAgent* agent );

    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
    */
    bool execute( rcsc::PlayerAgent* agent );

private:

    /*!
      \brief clear the action queue
     */
    void clear()
      {
          M_turn_step = M_dash_step = 0;
      }

    /*!
      \brief check if opponent may interfare this dribble.
      \param world const reference to the world model.
      \return true if dangerous opponent exists.
     */
    bool checkOpponent( const rcsc::WorldModel & world );

    /*!
      \brief execute turn action
      \param agent pointer to the agent itself
      \return true if action is executed
     */
    bool doTurn( rcsc::PlayerAgent * agent );

    /*!
      \brief execute dash action
      \param agent pointer to the agent itself
      \return true if action is executed
     */
    bool doDash( rcsc::PlayerAgent * agent );

};

#endif
