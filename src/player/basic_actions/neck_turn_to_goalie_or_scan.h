// -*-c++-*-

/*!
  \file neck_turn_to_goalie_or_scan.h
  \brief check opponent goalie or scan field with neck evenly
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

#ifndef RCSC_ACTION_NECK_TURN_TO_GOALIE_OR_SCAN_H
#define RCSC_ACTION_NECK_TURN_TO_GOALIE_OR_SCAN_H

#include <rcsc/player/soccer_action.h>

/*!
  \class Neck_TurnToGoalieOrScan
  \brief check opponent goalie or scan field with neck evenly
  if next ball position is NOT over view width ,scan field
  else face to ball
*/
class Neck_TurnToGoalieOrScan
    : public rcsc::NeckAction {
private:

    //! if goalie accuracy count is more then this value, player will try look the goalie.
    int M_count_thr;

public:
    /*!
      \brief accessible from global.
      \param count_thr accuracy count threshold to determine whether the agent check opponent goalie or not
     */
    explicit
    Neck_TurnToGoalieOrScan( const int count_thr )
        : M_count_thr( count_thr )
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
          return new Neck_TurnToGoalieOrScan( M_count_thr );
      }
};

#endif
