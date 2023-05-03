// -*-c++-*-

/*!
  \file bhv_scan_field.h
  \brief behavior to scan the field evenly header file
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

#ifndef RCSC_ACTION_BHV_SCAN_FIELD_H
#define RCSC_ACTION_BHV_SCAN_FIELD_H

#include <rcsc/player/soccer_action.h>

/*!
  \class Bhv_ScanField
  \brief scan the field evenly
 */
class Bhv_ScanField
    : public rcsc::SoccerBehavior {
private:

public:
    /*!
      \brief accessible from global.
     */
    Bhv_ScanField()
      { }

    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
     */
    bool execute( rcsc::PlayerAgent * agent );

private:

    void findBall( rcsc::PlayerAgent * agent );
    void scanAllField( rcsc::PlayerAgent * agent );

};

#endif
