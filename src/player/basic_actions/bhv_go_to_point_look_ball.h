// -*-c++-*-

/*!
  \file bhv_go_to_point_look_ball.h
  \brief run behavior which player always looks the ball. header file
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

#ifndef RCSC_ACTION_BODY_GO_TO_POINT_LOOK_BALL_H
#define RCSC_ACTION_BODY_GO_TO_POINT_LOOK_BALL_H

#include <rcsc/player/soccer_action.h>
#include <rcsc/geom/vector_2d.h>

/*!
  \class Bhv_GoToPointLookBall
  \brief run behavior which player always looks the ball.
*/
class Bhv_GoToPointLookBall
    : public rcsc::SoccerBehavior {
private:
    //! target point
    const rcsc::Vector2D M_target_point;
    //! distance threshold to the target point
    const double M_dist_thr;
    //! power parameter for dash command
    const double M_dash_power;
    //! power rate parameter for dash command with negative value
    const double M_back_power_rate;

public:
    /*!
      \brief construct with all parameters
      \param target_point target point
      \param dist_thr distance_threshold
      \param dash_power power parameter for dash commadn
      \param back_power_rate power rate parameter for back dash
    */
    Bhv_GoToPointLookBall( const rcsc::Vector2D & target_point,
                           const double & dist_thr,
                           const double & dash_power,
                           const double & back_power_rate = 0.7 )
        : M_target_point( target_point )
        , M_dist_thr( dist_thr )
        , M_dash_power( dash_power )
        , M_back_power_rate( back_power_rate )
      { }

    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
    */
    bool execute( rcsc::PlayerAgent * agent );
};

#endif
