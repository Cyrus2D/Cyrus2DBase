// -*-c++-*-

/*!
  \file neck_scan_players.h
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

#ifndef RCSC_ACTION_NECK_SCAN_PLAYERS_H
#define RCSC_ACTION_NECK_SCAN_PLAYERS_H

#include <rcsc/player/soccer_action.h>

namespace rcsc {
class AngleDeg;
class Vector2D;
class WorldModel;
}

/*!
  \class Neck_ScanPlayers
  \brief scan players
*/
class Neck_ScanPlayers
    : public rcsc::NeckAction {
public:

    //! invalid angle value
    static const double INVALID_ANGLE;

private:

    const double M_min_neck_angle; //!< search range: minimuam neck angle (relative to body)
    const double M_max_neck_angle; //!< search range: maximum neck angle (relative to body)

public:
    /*!
      \brief accessible from global.
      \param min_neck_angle search range: minimum neck angle (relative to body)
      \param max_neck_angle search range: minimum neck angle (relative to body)
    */
    Neck_ScanPlayers( const double & min_neck_angle = INVALID_ANGLE,
                      const double & max_neck_angle = INVALID_ANGLE )
        : M_min_neck_angle( min_neck_angle ),
          M_max_neck_angle( max_neck_angle )
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
          return new Neck_ScanPlayers;
      }

public:

    /*!
      \brief get the best target angle for all players
      \param agent agent object
      \param min_neck_angle search range: minimum neck angle (relative to body)
      \param max_neck_angle search range: minimum neck angle (relative to body)
      \return score value
     */
    static
    double get_best_angle( const rcsc::PlayerAgent * agent,
                           const double & min_neck_angle = INVALID_ANGLE,
                           const double & max_neck_angle = INVALID_ANGLE );

private:
    /*!
      \brief calculate score of range [left_angle, right_angle]
      \param wm world model
      \param next_self_pos next agent position
      \param left_angle search range: global angle
      \param right_angle search range: global angle
     */
    static
    double calculate_score( const rcsc::WorldModel & wm,
                            const rcsc::Vector2D & next_self_pos,
                            const rcsc::AngleDeg & left_angle,
                            const rcsc::AngleDeg & right_angle );

};

#endif
