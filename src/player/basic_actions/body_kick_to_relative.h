// -*-c++-*-

/*!
  \file body_kick_to_relative.h
  \brief kick the ball to the position relative to player agent.
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

#ifndef RCSC_ACTION_BODY_KICK_TO_RELATIVE_H
#define RCSC_ACTION_BODY_KICK_TO_RELATIVE_H

#include <rcsc/player/soccer_action.h>
#include <rcsc/geom/angle_deg.h>

namespace rcsc {
class Vector2D;
class WorldModel;
}

/*!
  \class Body_KickToRelative
  \brief kick the ball to relative pos(target_dist, target_angle_relative)
  if stop_ball == true, then stop the ball at that pos.
  estimate suitable kick queue, but execute only one kick, *not* queued.
  this action is assumed to use for ball holding.
*/
class Body_KickToRelative
    : public rcsc::BodyAction {
private:

    //! target point distance from self
    const double M_target_dist;
    //! target point angle relative to body
    const rcsc::AngleDeg M_target_angle_relative;
    //! if true, agent should stop the ball at target point
    const bool M_stop_ball;

public:
    /*!
      \brief construct with all parameters
      \param target_dist target point distance from self
      \param target_angle_relative target point angle relative to body
      \param stop if this valus is true, agent should stop the ball at target point
    */
    Body_KickToRelative( const double & target_dist,
                         const rcsc::AngleDeg & target_angle_relative,
                         const bool stop )
        : M_target_dist( target_dist ),
          M_target_angle_relative( target_angle_relative ),
          M_stop_ball( stop )
      { }

    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
    */
    bool execute( rcsc::PlayerAgent * agent );

private:
    /*!
      \brief internal simulation.
      kick the ball to relative pos(target_dist, target_angle_relative)
      if M_stop_ball == true, then stop the ball at that pos.
      \param wm const reference to the WorldModel instance
      \param reverse if true, far side rotation is simulated.
      \param required_accel required first accel value is stored to this
      \return true if agent can move the ball to the target point
    */
    bool simulate( const rcsc::WorldModel & wm,
                   const bool reverse,
                   rcsc::Vector2D * required_accel );

};

#endif
