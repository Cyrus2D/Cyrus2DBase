// -*-c++-*-

/*!
  \file view_change_width.h
  \brief change view width to normal
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

#ifndef RCSC_ACTION_VIEW_CHANGE_WIDTH_H
#define RCSC_ACTION_VIEW_CHANGE_WIDTH_H

#include <rcsc/player/soccer_action.h>
#include <rcsc/player/view_mode.h>

/*!
  \class View_ChangeWidth
  \brief change view width to the specified type
*/
class View_ChangeWidth
    : public rcsc::ViewAction {
private:
    const rcsc::ViewWidth M_width;
public:
    /*!
      \brief construct with view width
      \param width new view width
     */
    explicit
    View_ChangeWidth( const rcsc::ViewWidth & width )
        : M_width( width )
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
          return new View_ChangeWidth( M_width );
      }
};

#endif
