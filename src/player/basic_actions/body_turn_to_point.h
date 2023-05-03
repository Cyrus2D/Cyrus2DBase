// -*-c++-*-

/*!
  \file body_turn_to_point.h
  \brief turn only body to point
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

#ifndef RCSC_ACTION_BODY_TURN_TO_POINT_H
#define RCSC_ACTION_BODY_TURN_TO_POINT_H

#include <rcsc/player/soccer_action.h>
#include <rcsc/geom/vector_2d.h>

/*!
  \class Body_TurnToPoint
  \brief turn only body to point
*/
class Body_TurnToPoint
    : public rcsc::BodyAction {
private:
    //! body facing target point
    const rcsc::Vector2D M_point;
    //! agent's body will face the target point after this cycles
    const int M_cycle;
public:
    /*!
      \brief constructor
      \param point target point
      \param cycle cycles to turn
     */
    explicit
    Body_TurnToPoint( const rcsc::Vector2D & point,
                      const int cycle = 1 )
        : M_point( point )
        , M_cycle( cycle )
      { }

    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
    */
    bool execute( rcsc::PlayerAgent * agent );
};

#endif
