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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "focus_move_to_point.h"

#include <rcsc/player/player_agent.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>

namespace rcsc {

constexpr double MAX_FOCUS_DIST = 40.0;

/*-------------------------------------------------------------------*/
/*!

*/
bool
Focus_MoveToPoint::execute( PlayerAgent * agent )
{
    const WorldModel & wm = agent->world();

    const Vector2D self_next = agent->effector().queuedNextSelfPos();
    const AngleDeg next_body = agent->effector().queuedNextSelfBody();

    const double next_neck = rcsc::bound( ServerParam::i().minNeckAngle(),
                                          wm.self().neck().degree() + agent->effector().getTurnNeckMoment(),
                                          ServerParam::i().maxNeckAngle() );
    const AngleDeg next_face = next_body + next_neck;
    //const Vector2D next_focus_point = self_next + Vector2D::from_polar( wm.self().focusDist(), next_face + wm.self().focusDir() );
    const Vector2D target_rel = M_point - self_next;
    AngleDeg target_angle = target_rel.th();
    double target_dist = target_rel.r();

    dlog.addText( Logger::ACTION,
                  __FILE__": focus pos=(%.1f %.1f) rel_dist=%.1f rel_dir=%.1f",
                  M_point.x, M_point.y,
                  target_dist,
                  ( target_angle - next_face ).degree() );

    const double next_view_half = agent->effector().queuedNextViewWidth().width() * 0.5;
    const AngleDeg next_view_left = next_face - next_view_half + 0.5;
    const AngleDeg next_view_right = next_face + next_view_half - 0.5;

    if ( target_angle.isWithin( next_view_left, next_view_right ) )
    {
        // focus to the target point directly, or to the nearest point on the arc
        if ( target_dist > 40.0 )
        {
            target_dist = 40.0;
            dlog.addText( Logger::ACTION,
                          __FILE__": inside. but over the range" );
        }
    }
    else
    {
        // find the nearest point
        const Vector2D left_nearest = Segment2D( self_next, 40.0, next_view_left ).nearestPoint( M_point );
        const Vector2D right_nearest = Segment2D( self_next, 40.0, next_view_right ).nearestPoint( M_point );

        const Vector2D new_target = ( left_nearest.dist2( M_point ) < right_nearest.dist2( M_point )
                                      ? left_nearest
                                      : right_nearest );

        target_dist = ( new_target - self_next ).r();
        target_angle = ( new_target - self_next ).th();

        dlog.addText( Logger::ACTION,
                      __FILE__": outside. new_target=(%.1f %.1f) dist=%.1f angle=%.1f",
                      new_target.x, new_target.y, target_dist, target_angle.degree() );

    }

    const double moment_dist = target_dist - wm.self().focusDist();
    const AngleDeg moment_dir = ( target_dist < 1.0e-5
                                  ? -wm.self().focusDir()
                                  : target_angle - ( next_face + wm.self().focusDir() ) );

    dlog.addText( Logger::ACTION,
                  __FILE__": current focus_dist=%.1f focus_dir=%.1f",
                  wm.self().focusDist(), wm.self().focusDir() );
    dlog.addText( Logger::ACTION,
                  __FILE__": target dist=%.1f angle=%.1f (next_face=%.1f)",
                  target_dist, target_angle.degree(), next_face.degree() );

    dlog.addText( Logger::ACTION,
                  __FILE__": moment dist=%.1f dir=%.1f",
                  moment_dist, moment_dir.degree() );

    agent->doChangeFocus( moment_dist, moment_dir );
    return true;
}

}
