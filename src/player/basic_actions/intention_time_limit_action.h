// -*-c++-*-

/*!
  \file intention_time_limit_action.h
  \brief generic queued behavior that has a time limit.
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

#ifndef RCSC_ACTION_INTENAION_TIME_LIMIT_ACTION_H
#define RCSC_ACTION_INTENAION_TIME_LIMIT_ACTION_H

#include <rcsc/player/soccer_intention.h>
#include <rcsc/game_time.h>

namespace rcsc {
class BodyAction;
class NeckAction;
class ViewAction;
}

/*!
  \class IntentionTimeLimitAction
  \brief generic queued behavior that has a time limit.
*/

class IntentionTimeLimitAction
    : public rcsc::SoccerIntention {
private:
    std::shared_ptr< rcsc::BodyAction > M_body_action; //!< intended body action

    std::shared_ptr< rcsc::NeckAction > M_neck_action; //!< intended neck action

    std::shared_ptr< rcsc::ViewAction > M_view_action; //!< intended view action

    long M_timer_count; //!< the size of remained queue

    rcsc::GameTime M_last_execute_time; //!< last executed time

public:
    /*!
      \brief constructor
      \param body_action dynamically allocated action object
      \param neck_action dynamically allocated action object
      \param view_action dynamically allocated action object
      \param max_step the size of queue
      \param start_time action started time
     */
    IntentionTimeLimitAction( rcsc::BodyAction * body_action,
                              rcsc::NeckAction * neck_action,
                              rcsc::ViewAction * view_action,
                              const long & max_step,
                              const rcsc::GameTime & start_time );

    /*!
      \brief check the status.
      \param agent pointer to the agent itself.
      \return true if action queue is finished.
     */
    bool finished( const rcsc::PlayerAgent * agent );

    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
    */
    bool execute( rcsc::PlayerAgent * agent );
};

#endif
