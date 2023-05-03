// -*-c++-*-

/*!
  \file body_intercept2009.h
  \brief ball chasing action including smart planning.
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

#ifndef RCSC_ACTION_BODY_INTERCEPT_2009_H
#define RCSC_ACTION_BODY_INTERCEPT_2009_H

#include <rcsc/player/soccer_action.h>
#include <rcsc/player/intercept_table.h>
#include <rcsc/common/player_type.h>
#include <rcsc/geom/vector_2d.h>

/*!
  \class Body_Intercept2009
  \brief ball chasing action including smart planning.
*/
class Body_Intercept2009
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
    Body_Intercept2009( const bool save_recovery = true,
                        const rcsc::Vector2D & face_point = rcsc::Vector2D::INVALIDATED )
        : M_save_recovery( save_recovery )
        , M_face_point( face_point )
      { }

    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
    */
    bool execute( rcsc::PlayerAgent * agent );


private:

    /*!
      \brief if kickable opponent exists, perform special action
      \param agent pointer to the agent itself
      \return true if action is performed
    */
    bool doKickableOpponentCheck( rcsc::PlayerAgent * agent );

    /*!
      \brief calculate best interception point using cached table
      \param wm const refefence to the WorldModel
      \param table const pointer to the cached table
      \return interception info object
    */
    rcsc::InterceptInfo getBestIntercept( const rcsc::WorldModel & wm,
                                          const rcsc::InterceptTable * table ) const;

    rcsc::InterceptInfo getBestIntercept_Test( const rcsc::WorldModel & wm,
                                               const rcsc::InterceptTable * table ) const;

    /*!
      \brief try to perform ball wait action
      \param agent pointer to the agent itself
      \param target_point intercept target ball point
      \param info interception info that is considered now
      \return true if action is performed
    */
    bool doWaitTurn( rcsc::PlayerAgent * agent,
                     const rcsc::Vector2D & target_point,
                     const rcsc::InterceptInfo & info );

    /*!
      \brief adjutment dash action. if possible try to perform turn action.
      \param agent pointer to agent itself
      \param target_point intercept target ball point
      \param info interception info that is considered now
      \return true if action is performed
    */
    bool doInertiaDash( rcsc::PlayerAgent * agent,
                        const rcsc::Vector2D & target_point,
                        const rcsc::InterceptInfo & info );

};

#endif
