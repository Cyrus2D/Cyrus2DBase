// -*-c++-*-

/*!
  \file body_tackle_to_point.h
  \brief tackle ball to point
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

#ifndef RCSC_ACTION_BODY_TACKLE_TO_POINT_H
#define RCSC_ACTION_BODY_TACKLE_TO_POINT_H

#include <rcsc/player/soccer_action.h>
#include <rcsc/geom/vector_2d.h>

/*!
  \class Body_TackleToPoint
  \brief tackle ball to point
*/
class Body_TackleToPoint
    : public rcsc::BodyAction {
private:
    //! body facing target point
    const rcsc::Vector2D M_point;
    //! tackle probabolity threshold
    const double M_min_prob;
    //! result speed threshold
    const double M_min_speed;
public:
    /*!
      \brief constructor
      \param point target point
      \param min_prob tackle probability threshold
      \param min_speed result speed threshold
     */
    explicit
    Body_TackleToPoint( const rcsc::Vector2D & point,
                        const double & min_prob = 0.5,
                        const double & min_speed = 0.0 )
        : M_point( point )
        , M_min_prob( min_prob )
        , M_min_speed( min_speed )
      { }

    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
    */
    bool execute( rcsc::PlayerAgent * agent );
};

#endif
