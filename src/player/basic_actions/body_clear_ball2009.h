// -*-c++-*-

/*!
  \file body_clear_ball2009.h
  \brief kick the ball to escape from dangerous situation
*/

/*
 *Copyright:

 Copyright (C) Hiroki SHIMORA, Hidehisa AKIYAMA

 This code is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3, or (at your option)
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

#ifndef BODY_CLEAR_BALL_2009_H
#define BODY_CLEAR_BALL_2009_H

#include <rcsc/player/soccer_action.h>

/*!
  \class Body_ClearBall2009
  \brief kick the ball to escape from dangerous situation
*/
class Body_ClearBall2009
    : public rcsc::BodyAction {

public:
    /*!
      \brief constructor
     */
    Body_ClearBall2009();

    /*!
      \brief do clear ball
      \param agent agent pointer to the agent itself
      \return true with action, false if can't do clear
     */
    bool execute( rcsc::PlayerAgent * agent );
};

#endif
