// -*-c++-*-

/*!
  \file view_synch.h
  \brief change view width to synchronize see timing with server
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

#ifndef RCSC_ACTION_VIEW_SYNCH_H
#define RCSC_ACTION_VIEW_SYNCH_H

#include <rcsc/player/soccer_action.h>

/*!
  \class View_Synch
  \brief change view width to synchronize see timing with server
*/
class View_Synch
    : public rcsc::ViewAction {
private:

public:
    /*!
      \brief accessible from global.
     */
    View_Synch()
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
    rcsc::ViewAction * clone() const
      {
          return new View_Synch;
      }

private:
    bool doTimerSynchView( rcsc::PlayerAgent * agent );
};

#endif
