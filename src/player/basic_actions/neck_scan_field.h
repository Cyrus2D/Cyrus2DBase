// -*-c++-*-

/*!
  \file neck_scan_field.h
  \brief scan field with neck evenly
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

#ifndef RCSC_ACTION_NECK_SCAN_FIELD_H
#define RCSC_ACTION_NECK_SCAN_FIELD_H

#include <rcsc/player/soccer_action.h>
#include <rcsc/player/view_mode.h>
#include <rcsc/geom/vector_2d.h>
#include <rcsc/geom/angle_deg.h>
#include <rcsc/game_time.h>

namespace rcsc {
class WorldModel;
}

/*!
  \class Neck_ScanField
  \brief scan field with neck evenly
*/
class Neck_ScanField
    : public rcsc::NeckAction {
private:

    //! invalid angle value
    static const double INVALID_ANGLE;

public:
    /*!
      \brief accessible from global.
    */
    Neck_ScanField()
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
          return new Neck_ScanField;
      }

private:

    double calcAngleDefault( const rcsc::PlayerAgent * agent,
                             const bool consider_pitch );

    /*!
      \retval -360.0 if no solution.
     */
    double calcAngleForWidePitchEdge( const rcsc::PlayerAgent * agent );

};

#endif
