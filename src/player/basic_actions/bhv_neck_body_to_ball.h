// -*-c++-*-

/*!
  \file bhv_neck_body_to_ball.h
  \brief face to the ball with turn & turn_neck header file
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

#ifndef RCSC_ACTION_BHV_NECK_BODY_TO_BALL_H
#define RCSC_ACTION_BHV_NECK_BODY_TO_BALL_H

#include <rcsc/player/soccer_action.h>

/*!
  \class Bhv_NeckBodyToBall
  \brief face to ball. if possible, only turn neck
*/
class Bhv_NeckBodyToBall
    : public rcsc::SoccerBehavior {
private:
    //! angle buffer for view cone
    const double M_angle_buf;

public:
    /*!
      \brief accessible from global.
     */
    explicit
    Bhv_NeckBodyToBall( const double & angle_buf = 5.0 )
        : M_angle_buf( angle_buf )
      { }

    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
     */
    bool execute( rcsc::PlayerAgent * agent );
};

#endif
