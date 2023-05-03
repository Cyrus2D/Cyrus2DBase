// -*-c++-*-

/*!
  \file body_smart_kick.h
  \brief smart kick action class header file.
*/

/*
 *Copyright:

 Copyright (C) Hidehisa AKIYAMA

 This code is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2, or (at your option)
 any later version.

 This code is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this code; see the file COPYING.  If not, write to
 the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.

 *EndCopyright:
 */

/////////////////////////////////////////////////////////////////////

#ifndef RCSC_ACTION_BODY_SMARTKICK_H
#define RCSC_ACTION_BODY_SMARTKICK_H

#include "basic_actions/kick_table.h"

#include <rcsc/player/soccer_action.h>
#include <rcsc/geom/vector_2d.h>

/*!
  \class Body_SmartKick
  \brief smart kick action
 */
class Body_SmartKick
    : public rcsc::BodyAction {
private:
    //! target point where the ball should move to
    const rcsc::Vector2D M_target_point;
    //! desired ball first speed
    const double M_first_speed;
    //! threshold value for the ball first speed
    const double M_first_speed_thr;
    //! maximum number of kick steps
    const int M_max_step;

    //! result kick sequence holder
    KickTable::Sequence M_sequence;

public:
    /*!
      \brief construct with all parameters
      \param target_point target point where the ball should move to
      \param first_speed desired ball first speed
      \param first_speed_thr threshold value for the ball first speed
      \param max_step maximum number of kick steps
    */
    Body_SmartKick( const rcsc::Vector2D & target_point,
                    const double & first_speed,
                    const double & first_speed_thr,
                    const int max_step )
        : M_target_point( target_point )
        , M_first_speed( first_speed )
        , M_first_speed_thr( first_speed_thr )
        , M_max_step( max_step )
      { }

    bool execute( rcsc::PlayerAgent * agent );

    /*!
      \brief get the result kick sequence
      \return kick sequence object
     */
    const
    KickTable::Sequence & sequence() const
      {
          return M_sequence;
      }

};

#endif
