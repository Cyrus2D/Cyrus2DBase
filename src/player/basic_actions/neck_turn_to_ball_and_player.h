// -*-c++-*-

/*!
  \file neck_turn_to_ball_and_player.h
  \brief turn only neck to ball
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

#ifndef RCSC_ACTION_NECK_TURN_TO_BALL_AND_PLAYER_H
#define RCSC_ACTION_NECK_TURN_TO_BALL_AND_PLAYER_H

#include <rcsc/player/soccer_action.h>

namespace rcsc {
class AbstractPlayerObject;
}

/*!
  \class Neck_TurnToBallAndPlayer
  \brief if possible, turn neck to ball and player
*/
class Neck_TurnToBallAndPlayer
    : public rcsc::NeckAction {
private:

    const rcsc::AbstractPlayerObject * M_target_player; //!< target player pointer
    const int M_count_thr; //!< count threshold value

public:
    /*!
      \brief accessible from global.
      \param target_player pointer to the target player
      \param count_thr if player's posCount is smaller than this value, agent simply turns neck to the ball.
     */
    explicit
    Neck_TurnToBallAndPlayer( const rcsc::AbstractPlayerObject * target_player,
                              const int count_thr )
        : M_target_player( target_player )
        , M_count_thr( count_thr )
      { }

    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
     */
    bool execute( rcsc::PlayerAgent * agent );

    /*!
      \brief create cloned object
      \return pointer to the cloned object
     */
    rcsc::NeckAction * clone() const
      {
          return new Neck_TurnToBallAndPlayer( M_target_player, M_count_thr );
      }
};

#endif
