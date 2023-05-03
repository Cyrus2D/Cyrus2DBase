// -*-c++-*-

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

#ifndef RCSC_ACTION_BODY_INTERCEPT_2018_H
#define RCSC_ACTION_BODY_INTERCEPT_2018_H

#include "intercept_evaluator.h"

#include <rcsc/player/soccer_action.h>
#include <rcsc/player/intercept.h>
#include <rcsc/common/player_type.h>
#include <rcsc/geom/vector_2d.h>

/*!
  \class Body_Intercept2018
  \brief ball chasing action including smart planning.
*/
class Body_Intercept2018
    : public rcsc::BodyAction {
private:
    //! if true, agent must save recovery parameter
    const bool M_save_recovery;
    //! facing target target point. if specified this, plaeyr try to turn neck to this point.
    const rcsc::Vector2D M_face_point;

public:
    /*!
      \brief construct with all parameters
      \param save_recovery if true, agent must save recovery parameter
      \param face_point desired body facing point
    */
    explicit
    Body_Intercept2018( const bool save_recovery = true,
                        const rcsc::Vector2D & face_point = rcsc::Vector2D::INVALIDATED )
        : M_save_recovery( save_recovery ),
          M_face_point( face_point )
    { }

    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
    */
    bool execute( rcsc::PlayerAgent * agent );

    /*!
      \brief calculate best interception point using cached table
      \param wm const refefence to the WorldModel
      \param save_recovery if this value is false, agent may lose its recovery
      \return interception object
    */
    static
    rcsc::Intercept get_best_intercept( const rcsc::WorldModel & wm,
                                        const bool save_recovery );

private:

    bool doGoToFinalPoint( rcsc::PlayerAgent * agent );

    bool doOneStepTurn( rcsc::PlayerAgent * agent,
                        const rcsc::Intercept & info );

    bool doFirstTurn( rcsc::PlayerAgent * agent,
                      const rcsc::Intercept & info );

    bool doSaveRecovery( rcsc::PlayerAgent * agent,
                         const rcsc::Intercept & info );

    bool doDashWait( rcsc::PlayerAgent * agent,
                     const rcsc::Intercept & info );
    bool doOneAdjustDash( rcsc::PlayerAgent * agent,
                          const rcsc::Intercept & info );
    bool doDash( rcsc::PlayerAgent * agent,
                 const rcsc::Intercept & info );

    static
    rcsc::Intercept get_best_intercept_player( const rcsc::WorldModel & wm,
                                               const bool save_recovery );

    static
    rcsc::Intercept get_best_intercept_goalie( const rcsc::WorldModel & wm );
};


#endif
