// -*-c++-*-

/*!
  \file neck_turn_to_relative.h
  \brief turn neck to the angle relative to body
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

#ifndef RCSC_ACTION_NECK_TURN_TO_RELATIVE_H
#define RCSC_ACTION_NECK_TURN_TO_RELATIVE_H

#include <rcsc/player/soccer_action.h>
#include <rcsc/geom/angle_deg.h>

/*!
  \class Neck_TurnToRelative
  \brief turn neck to the angle relative to body
*/
class Neck_TurnToRelative
    : public rcsc::NeckAction {
private:
    //! target neck angle relative to body angle
    const rcsc::AngleDeg M_angle_rel_to_body;

public:
    /*!
      \brief constructor
      \param rel_angle relative angle from the body
     */
    explicit
    Neck_TurnToRelative( const rcsc::AngleDeg & rel_angle )
        : M_angle_rel_to_body( rel_angle )
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
          return new Neck_TurnToRelative( M_angle_rel_to_body );
      }
};

#endif
