// -*-c++-*-

/*!
  \file body_hold_ball2008.h
  \brief stay there and keep the ball from opponent players.
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

#ifndef RCSC_ACTION_BODY_HOLD_BALL_2008_H
#define RCSC_ACTION_BODY_HOLD_BALL_2008_H

#include <rcsc/player/soccer_action.h>
#include <rcsc/geom/vector_2d.h>

#include <vector>

namespace rcsc {
class PlayerObject;
class WorldModel;
}

/*!
  \class Body_HoldBall2008
  \brief stay there and keep the ball from opponent players.
*/
class Body_HoldBall2008
    : public rcsc::BodyAction {
public:

    //! default score value
    static const double DEFAULT_SCORE;

    /*!
      \struct KeepPoint
      \brief keep point info
     */
    struct KeepPoint {
        rcsc::Vector2D pos_; //!< global position
        double kick_rate_; //!< kick rate at the position
        double score_; //!< evaluated score

        /*!
          \brief construct an illegal object
         */
        KeepPoint()
            : pos_( rcsc::Vector2D::INVALIDATED )
            , kick_rate_( 0.0 )
            , score_( -100000.0 )
          { }

        /*!
          \brief construct with arguments
          \param pos global position
          \param krate kick rate at the position
          \param score initial score
         */
        KeepPoint( const rcsc::Vector2D & pos,
                   const double & krate,
                   const double & score )
            : pos_( pos )
            , kick_rate_( krate )
            , score_( score )
          { }

        /*!
          \brief reset object value
         */
        void reset()
          {
              pos_.invalidate();
              kick_rate_ = 0.0;
              score_ = -100000.0;
          }
    };


private:
    //! if true, agent will try to face to the target point
    const bool M_do_turn;
    //! face target point
    const rcsc::Vector2D M_turn_target_point;
    //! next kick target point (if exist)
    const rcsc::Vector2D M_kick_target_point;
public:
    /*!
      \brief construct with all parameters
      \param do_turn if true, agent will try to face to the target point
      \param turn_target_point face target point
      \param kick_target_point intended next kick target point
    */
    Body_HoldBall2008( const bool do_turn = false,
                       const rcsc::Vector2D & turn_target_point = rcsc::Vector2D( 0.0, 0.0 ),
                       const rcsc::Vector2D & kick_target_point = rcsc::Vector2D::INVALIDATED )
        : M_do_turn( do_turn )
        , M_turn_target_point( turn_target_point )
        , M_kick_target_point( kick_target_point )
      { }

    /*!
      \brief execute action
      \param agent pointer to the agent itself
      \return true if action is performed
    */
    bool execute( rcsc::PlayerAgent * agent );


private:

    /*!
      \brief kick the ball to avoid opponent
      \param agent pointer to agent itself
      \return true if action is performed
    */
    bool avoidOpponent( rcsc::PlayerAgent * agent );

    /*!
      \brief search the best keep point
      \param wm const reference to the WorldModel instance
      \return estimated best keep point. if no point, INVALIDATED is returned.
     */
    rcsc::Vector2D searchKeepPoint( const rcsc::WorldModel & wm );

    /*!
      \brief create candidate keep points
      \param wm const reference to the WorldModel instance
      \param keep_points reference to the variable container
     */
    void createKeepPoints( const rcsc::WorldModel & wm,
                           std::vector< KeepPoint > & keep_points );

    /*!
      \brief evaluate candidate keep points
      \param wm const reference to the WorldModel instance
      \param keep_points reference to the variable container
     */
    void evaluateKeepPoints( const rcsc::WorldModel & wm,
                             std::vector< KeepPoint > & keep_points );

    /*!
      \brief evaluate the keep point
      \param wm const reference to the WorldModel instance
      \param keep_point keep point value
     */
    double evaluateKeepPoint( const rcsc::WorldModel & wm,
                              const rcsc::Vector2D & keep_point );

    /*!
      \brief if possible, turn to face target point
      \param agent agent pointer to agent itself
      \return true if action is performed
    */
    bool turnToPoint( rcsc::PlayerAgent * agent );

    /*!
      \brief keep the ball at body front
      \param agent pointer to agent itself
      \return true if action is performed
    */
    bool keepFront( rcsc::PlayerAgent * agent );

    /*!
      \brief keep the ball at reverse point from the kick target point
      \param agent pointer to agent itself
     */
    bool keepReverse( rcsc::PlayerAgent * agent );

};

#endif
