// -*-c++-*-

/*!
  \file neck_turn_to_point.h
  \brief turn only neck to point
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

#ifndef RCSC_ACTION_NECK_TURN_TO_POINT_H
#define RCSC_ACTION_NECK_TURN_TO_POINT_H

#include <rcsc/player/soccer_action.h>
#include <rcsc/geom/vector_2d.h>

#include <vector>

/*!
  \class Neck_TurnToPoint
  \brief turn only neck to point
*/
class Neck_TurnToPoint
    : public rcsc::NeckAction {
private:
    //! facing target point container
    const std::vector< rcsc::Vector2D > M_points;
public:
    /*!
      \brief construct with one point
      \param point face target point
     */
    explicit
    Neck_TurnToPoint( const rcsc::Vector2D & point )
        : M_points( 1, point )
      { }

    /*!
      \brief construct with points container
      \param points container to the face target point
     */
    explicit
    Neck_TurnToPoint( const std::vector< rcsc::Vector2D > & points )
        : M_points( points )
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
          return new Neck_TurnToPoint( M_points );
      }
};

#endif
