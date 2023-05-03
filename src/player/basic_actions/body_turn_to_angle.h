// -*-c++-*-

/*!
  \file body_turn_to_angle.h
  \brief align body angle
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

#ifndef RCSC_ACTION_BODY_TURN_TO_ANGEL_H
#define RCSC_ACTION_BODY_TURN_TO_ANGEL_H

#include <rcsc/player/soccer_action.h>
#include <rcsc/geom/angle_deg.h>

/*!
  \class Body_TurnToAngle
  \brief align body angle to the target angle
*/
class Body_TurnToAngle
    : public rcsc::BodyAction {
private:
    //! body facing target angle
    const rcsc::AngleDeg M_angle;
public:
    /*!
      \brief construct wiht target angle
      \param angle facing target angle
     */
    explicit
    Body_TurnToAngle( const rcsc::AngleDeg & angle )
        : M_angle( angle )
      { }

    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
    */
    bool execute( rcsc::PlayerAgent * agent );
};

#endif
