// -*-c++-*-

/*!
  \file body_go_to_point_dodge.h
  \brief sub behavior for Body_GoToPoint.
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

#ifndef RCSC_ACTION_GODT_GO_TO_POINT_DODGE_H
#define RCSC_ACTION_GODT_GO_TO_POINT_DODGE_H

#include <rcsc/player/soccer_action.h>
#include <rcsc/geom/vector_2d.h>

/*!
  \class Body_GoToPointDodge
  \brief sub behavior for Body_GoToPoint.
*/
class Body_GoToPointDodge
    : public rcsc::BodyAction {
private:
    //! target point to be reached
    const rcsc::Vector2D M_point;
    //! power parameter for dash command
    const double M_dash_power;

public:
    /*!
      \brief construct with all parameters
      \param point target point to be reached
      \param dash_power parameter for dash command
    */
    Body_GoToPointDodge( const rcsc::Vector2D & point,
                         const double & dash_power )
        : M_point( point )
        , M_dash_power( dash_power )
      { }


    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
    */
    bool execute( rcsc::PlayerAgent * agent );

public:
    /*!
      \brief calculate new sub target point
      \param agent const pointer to agent itself
      \param target original target point
      \param sol new sub target is stored to this
      \return true if agent should perform dodge action
    */
    static
    bool get_dodge_point( const rcsc::PlayerAgent * agent,
                          const rcsc::Vector2D & target,
                          rcsc::Vector2D * sol );
};

#endif
