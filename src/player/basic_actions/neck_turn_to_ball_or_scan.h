// -*-c++-*-

/*!
  \file neck_turn_to_ball_or_scan.h
  \brief check the ball or scan field with neck evenly
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

#ifndef RCSC_ACTION_NECK_TURN_TO_BALL_OR_SCAN_H
#define RCSC_ACTION_NECK_TURN_TO_BALL_OR_SCAN_H

#include <rcsc/player/soccer_action.h>

/*!
  \class Neck_TurnToBallOrScan
  \brief check the ball or scan field with neck evenly.
  if next ball position is NOT over view width ,scan field
  else face to ball
*/
class Neck_TurnToBallOrScan
    : public rcsc::NeckAction {
private:

    const int M_count_thr;

public:
    /*!
      \brief accessible from global.
      \param count_thr if ball::posCount() is less than or equal to
      this value, agent always perform field scan, otherwise try to
      look the ball. if this value is negative, agent always try to
      look the ball.
     */
    explicit
    Neck_TurnToBallOrScan( const int count_thr )
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
          return new Neck_TurnToBallOrScan( M_count_thr );
      }
};

#endif
