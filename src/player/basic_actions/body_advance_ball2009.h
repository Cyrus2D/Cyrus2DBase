// -*-c++-*-

/*!
  \file body_advance_ball2009.h
  \brief kick the ball to a forward direction
*/

/*
 *Copyright:

 Copyright (C) Hidehisa AKIYAMA

 This code is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.

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

#ifndef RCSC_ACTION_BODY_ADVANCE_BALL_2009_H
#define RCSC_ACTION_BODY_ADVANCE_BALL_2009_H

#include <rcsc/player/soccer_action.h>
#include <rcsc/geom/angle_deg.h>
#include <rcsc/game_time.h>

/*!
  \class Body_AdvanceBall2009
  \brief kick the ball to a forward direction
 */
class Body_AdvanceBall2009
    : public rcsc::BodyAction {
private:
    //! last game time when calcuration is done.
    static rcsc::GameTime S_last_calc_time;
    //! last calculated result
    static rcsc::AngleDeg S_cached_best_angle;

public:
    /*!
      \brief accessible from global.
     */
    Body_AdvanceBall2009()
      { }

    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
     */
    bool execute( rcsc::PlayerAgent * agent );

private:

    /*!
      \brief calculate best kick angle
      \param agent const pointer to the agent itself
      \return calculated result
     */
    rcsc::AngleDeg getBestAngle( const rcsc::PlayerAgent * agent );
};

#endif
