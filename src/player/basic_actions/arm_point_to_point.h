// -*-c++-*-

/*!
  \file arm_point_to_point.h
  \brief point to point
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

#ifndef RCSC_ACTION_ARM_POINT_TO_POINT_H
#define RCSC_ACTION_ARM_POINT_TO_POINT_H

#include <rcsc/player/soccer_action.h>
#include <rcsc/geom/vector_2d.h>

/*!
  \class Arm_PointToPoint
  \brief point to point
*/
class Arm_PointToPoint
    : public rcsc::ArmAction {
private:
    //! body facing target point
    const rcsc::Vector2D M_point;
public:
    /*!
      \brief constructor
      \param point target point
     */
    explicit
    Arm_PointToPoint( const rcsc::Vector2D & point )
        : M_point( point )
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
    ArmAction * clone() const
      {
          return new Arm_PointToPoint( M_point );
      }
};

#endif
