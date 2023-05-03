// -*-c++-*-

/*!
  \file bhv_emergency.h
  \brief behavior for emergency situation header file
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

#ifndef RCSC_ACTION_BHV_EMERGENCY_H
#define RCSC_ACTION_BHV_EMERGENCY_H

#include <rcsc/player/soccer_action.h>

/*!
  \class Bhv_Emergency
  \brief behavior for emergency situations. just scan field.
 */
class Bhv_Emergency
    : public rcsc::SoccerBehavior {
private:

public:
    /*!
      \brief accessible from global.
     */
    Bhv_Emergency()
      { }

    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
     */
    bool execute( rcsc::PlayerAgent* agent );
};

#endif
