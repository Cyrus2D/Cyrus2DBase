// -*-c++-*-

/*!
  \file body_go_to_point_dodge.cpp
  \brief sub behavior for Body_GoToPoint.
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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "body_go_to_point_dodge.h"

#include "body_go_to_point.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/geom/line_2d.h>
#include <rcsc/math_util.h>

using namespace rcsc;

/*-------------------------------------------------------------------*/
/*!

*/
bool
Body_GoToPointDodge::execute( PlayerAgent * agent )
{
    dlog.addText( Logger::ACTION,
                  "%s:%d: Body_GoToPointDodge"
                  ,__FILE__, __LINE__ );

    Vector2D dodge_pos;
    if ( ! get_dodge_point( agent, M_point, &dodge_pos ) )
    {
        return Body_GoToPoint( M_point,
                               0.1,
                               M_dash_power,
                               -1.0, // dash speed
                               3 ).execute( agent );
    }

    AngleDeg target_angle = ( M_point - agent->world().self().pos() ).th();
    Vector2D dodge_rel = dodge_pos - agent->world().self().pos();

    // difference angle between target and this player to dodge
    // if diff is positive this player is left from target angle.
    double ang_diff = ( target_angle - dodge_rel.th() ).degree();
    if ( std::fabs( ang_diff ) > 90.0 )
    {
        return Body_GoToPoint( M_point,
                               0.1,
                               M_dash_power,
                               -1.0, // dash speed
                               3 ).execute( agent );
    }

    Vector2D new_target;
    if ( ang_diff > 0.0 )
    { // obstacle is left from target dir -> new target is right from old
        new_target
            = dodge_pos
            + Vector2D::polar2vector( ServerParam::i().defaultPlayerSize() * 4.0,
                                      target_angle + 90.0 );
    }
    else
    { // obstacle is right from target dir -> new target is left from old
        new_target
            = dodge_pos
            + Vector2D::polar2vector( ServerParam::i().defaultPlayerSize() * 4.0,
                                      target_angle - 90.0 );
    }

    dlog.addText( Logger::ACTION,
                  "%s:%d: dodge(%f, %f) sub-target(%f, %f)"
                  ,__FILE__, __LINE__,
                  dodge_pos.x, dodge_pos.y, new_target.x, new_target.y );
    // never do dodge agen
    return Body_GoToPoint( new_target,
                           0.1,
                           M_dash_power,
                           -1.0, // dash speed
                           3 ).execute( agent );
}

/*-------------------------------------------------------------------*/
/*!

*/
bool
Body_GoToPointDodge::get_dodge_point( const PlayerAgent * agent,
                                      const Vector2D & target,
                                      Vector2D * sol )
{
    const WorldModel & wm = agent->world();

    if ( wm.self().pos().dist( target )
         < ServerParam::i().defaultPlayerSize() * 2.0 )
    {
        return false;
    }

    const Line2D line2target( wm.self().pos(), target );

    /*--------------------------------------------------------*/
    // check players
    const AngleDeg target_angle = ( target - wm.self().pos() ).th();
    const double dodge_perpend_dist2
        = square( ServerParam::i().defaultPlayerSize() * 2.0 + 0.1 );

    const double consider_dist_max = 1.5;

    double mindist = 100.0;
    bool found = false;

    for ( const PlayerObject * t : wm.teammatesFromSelf() )
    {
        if ( t->distFromSelf() > consider_dist_max )
        {
            break;
        }
        // this player is near to target trajectry
        if ( mindist > t->distFromSelf()
             && line2target.dist2( t->pos() ) < dodge_perpend_dist2
             && (target_angle - t->angleFromSelf()).abs() < 90.0 )
        {
            mindist = t->distFromSelf();
            *sol = t->pos();
            found = true;
        }
    }

    for ( const PlayerObject * o : wm.opponentsFromSelf() )
    {
        if ( o->distFromSelf() > consider_dist_max )
        {
            break;
        }
        // this player is near to target trajectry
        if ( mindist > o->distFromSelf()
             && line2target.dist2( o->pos() ) < dodge_perpend_dist2
             && ( target_angle - o->angleFromSelf() ).abs() < 90.0 )
        {
            mindist = o->distFromSelf();
            *sol = o->pos();
            found = true;
        }
    }

    /*--------------------------------------------------------*/
    // check ball
    if ( wm.gameMode().type() != GameMode::PlayOn
         && wm.ball().posValid() )
    {
        if ( wm.ball().distFromSelf() < consider_dist_max
             && line2target.dist( wm.ball().pos() ) < 1.5
             && ( ( target_angle - ( wm.ball().pos() - wm.self().pos() ).th() ).abs() < 90.0 ) )
        {
            if ( mindist > wm.ball().distFromSelf() )
            {
                *sol = wm.ball().pos();
                found = true;
            }
        }
    }

    return found;
}
