// -*-c++-*-

/*!
  \file view_wide.h
  \brief change view width to wide
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

#ifndef RCSC_ACTION_VIEW_WIDE_H
#define RCSC_ACTION_VIEW_WIDE_H

#include <rcsc/player/soccer_action.h>

/*!
  \class View_Wide
  \brief change view width to wide
*/
class View_Wide
    : public rcsc::ViewAction {
private:

public:
    /*!
      \brief accessible from global.
     */
    View_Wide()
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
          return new View_Wide;
      }
};

#endif
