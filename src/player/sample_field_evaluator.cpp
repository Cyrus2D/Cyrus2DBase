// -*-c++-*-

/*
 *Copyright:

 Cyrus2D
 Modified by Omid Amini, Nader Zare
 
 Gliders2d
 Modified by Mikhail Prokopenko, Peter Wang

 Copyright (C) Hiroki SHIMORA

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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "sample_field_evaluator.h"

#include "field_analyzer.h"
#include "simple_pass_checker.h"

#include <rcsc/player/player_evaluator.h>
#include <rcsc/common/server_param.h>
#include <rcsc/common/logger.h>
#include <rcsc/math_util.h>

#include <rcsc/player/world_model.h>

#include <rcsc/geom/voronoi_diagram.h>

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cfloat>

// #define DEBUG_PRINT

using namespace rcsc;

static const int VALID_PLAYER_THRESHOLD = 8;


/*-------------------------------------------------------------------*/
/*!

 */
static double evaluate_state( const PredictState & state , const rcsc::WorldModel & wm );


/*-------------------------------------------------------------------*/
/*!

 */
SampleFieldEvaluator::SampleFieldEvaluator()
{

}

/*-------------------------------------------------------------------*/
/*!

 */
SampleFieldEvaluator::~SampleFieldEvaluator()
{

}

/*-------------------------------------------------------------------*/
/*!

 */
double
SampleFieldEvaluator::operator()(const PredictState &state,
                                 const std::vector<ActionStatePair> & /*path*/,
                                 const rcsc::WorldModel &wm) const
{
    const double final_state_evaluation = evaluate_state( state , wm);

    //
    // ???
    //

    double result = final_state_evaluation;

    return result;
}


/*-------------------------------------------------------------------*/
/*!

 */
static
double
evaluate_state( const PredictState & state, const rcsc::WorldModel & wm )
{
    const ServerParam & SP = ServerParam::i();

    const AbstractPlayerObject * holder = state.ballHolder();

#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION_CHAIN,
                  "========= (evaluate_state) ==========" );
#endif

    //
    // if holder is invalid, return bad evaluation
    //
    if ( ! holder )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION_CHAIN,
                      "(eval) XXX null holder" );
#endif
        return - DBL_MAX / 2.0;
    }

    const int holder_unum = holder->unum();


    //
    // ball is in opponent goal
    //
    if ( state.ball().pos().x > + ( SP.pitchHalfLength() - 0.1 )
         && state.ball().pos().absY() < SP.goalHalfWidth() + 2.0 )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION_CHAIN,
                      "(eval) *** in opponent goal" );
#endif
        return +1.0e+7;
    }

    //
    // ball is in our goal
    //
    if ( state.ball().pos().x < - ( SP.pitchHalfLength() - 0.1 )
         && state.ball().pos().absY() < SP.goalHalfWidth() )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION_CHAIN,
                      "(eval) XXX in our goal" );
#endif

        return -1.0e+7;
    }


    //
    // out of pitch
    //
    if ( state.ball().pos().absX() > SP.pitchHalfLength()
         || state.ball().pos().absY() > SP.pitchHalfWidth() )
    {
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION_CHAIN,
                      "(eval) XXX out of pitch" );
#endif

        return - DBL_MAX / 2.0;
    }


    //
    // set basic evaluation
    //

    // G2d: to retrieve opp team name 
    // C2D: Helios 18 Tune removed -> replace with BNN
    // bool heliosbase = false;
    // if (wm.opponentTeamName().find("HELIOS_base") != std::string::npos)
    //     heliosbase = true;

    // G2d: number of direct opponents
        int opp_forward = 0;

        Vector2D egl (52.5, -8.0);
        Vector2D egr (52.5, 8.0);
        Vector2D left = egl - wm.self().pos();
        Vector2D right = egr - wm.self().pos();

        Sector2D sector(wm.self().pos(), 0.0, 10000.0, left.th(), right.th());

        for (auto of = wm.opponentsFromSelf().begin();
             of != wm.opponentsFromSelf().end();
             ++of)
        {
                if ( sector.contains( (*of)->pos() ) && !((*of)->goalie()) )
                opp_forward++;
        }

        double weight = 1.0;
        if (wm.ball().pos().x > 35.0)
            weight = 0.3;

	double depth = 10.0;
    // C2D: Helios 18 Tune removed -> replace with BNN
	// if (heliosbase)
	// 	depth = 0.0;

    double point = state.ball().pos().x * weight;

        Vector2D best_point = ServerParam::i().theirTeamGoalPos();

    // G2d: new eval function
    // C2D: replace PlayerPtrCont::const_iterator with auto
        if ( wm.ball().pos().x < depth || opp_forward == 0 )
	{
		// stay with best point = opp goal
	}
	else
	{

		if ( wm.ball().pos().x < 35.0 &&  state.ball().pos().x > 5.0 )
		{
                       VoronoiDiagram vd;

                        std::vector<Vector2D> vd_cont;
                        for ( auto o = wm.opponentsFromSelf().begin();
                                o != wm.opponentsFromSelf().end();
                                ++o )
                        {
                                   vd.addPoint((*o)->pos());
                        }

                        vd.compute();


			    double max_dist = -1000.0;

                            for ( VoronoiDiagram::Vector2DCont::const_iterator p = vd.vertices().begin(),
                                      end = vd.vertices().end();
                                          p != end;
                                          ++p )
                            {
						if ( (*p).x < state.ball().pos().x - 5.0 || (*p).x > 52.5 || fabs((*p).y) > 34.0 )
							continue;

						if ( ( (*p) - state.ball().pos() ).length() > 34.0 )
							continue;

						double min_dist = 1000.0;
						double our_dist = 1000.0;

                                                for ( auto of = wm.opponentsFromSelf().begin();
                                                        of != wm.opponentsFromSelf().end();
                                                        ++of )
                                                {
                                                        Vector2D tmp = (*of)->pos() - (*p);
                                                        if ( min_dist > tmp.length() )
                                                                min_dist = tmp.length();
                                                }

                                                for ( auto of = wm.teammatesFromSelf().begin();
                                                        of != wm.teammatesFromSelf().end();
                                                        ++of )
                                                {
							if ((*of)->pos().x > wm.offsideLineX() + 1.0) continue;
                                                        Vector2D tmp = (*of)->pos() - (*p);
                                                        if ( our_dist > tmp.length() )
                                                                our_dist = tmp.length();
                                                }

						Vector2D tmp = wm.self().pos() - (*p); 
						if ( wm.self().pos().x < (*p).x && tmp.length() > 7.0 && our_dist > tmp.length() )
                                                        our_dist = tmp.length();

					if (max_dist < min_dist - our_dist )
					{
						max_dist = min_dist - our_dist;
						best_point = (*p);
					}

			    }

                        std::vector<Vector2D> OffsideSegm_cont;
                        std::vector<Vector2D> OffsideSegm_tmpcont;

                        Vector2D y1( wm.offsideLineX(), -34.0);
                        Vector2D y2( wm.offsideLineX(), 34.0);

                        Vector2D z1( wm.offsideLineX(), -34.0);
                        Vector2D z2( wm.offsideLineX(), 34.0);

                        if (wm.ball().pos().x > 25.0)
                        {
                                if (wm.ball().pos().y < 0.0)
                                        y2.y = 20.0;
                                if (wm.ball().pos().y > 0.0)
                                        y1.y = -20.0;
                        }
                        if (wm.ball().pos().x > 36.0)
                        {
                                if (wm.ball().pos().y < 0.0)
                                        y2.y = 8.0;
                                if (wm.ball().pos().y > 0.0)
                                        y1.y = -8.0;
                        }

                                z1.x = y1.x + 6.0;
				if (z1.x > 52.5)
					z1.x = 52.0;

                                z2.x = y2.x + 6.0;
				if (z2.x > 52.5)
					z2.x = 52.0;

                                z1.y = y1.y;
                                z2.y = y2.y;


                        Line2D offsideLine (y1, y2);
                        Line2D forwardLine (z1, z2);

                            for ( VoronoiDiagram::Segment2DCont::const_iterator p = vd.segments().begin(),
                                      end = vd.segments().end();
                                          p != end;
                                          ++p )
                            {
                                Vector2D si = (*p).intersection( offsideLine );
                                Vector2D fi = (*p).intersection( forwardLine );
                                if (si.isValid() && fabs(si.y) < 34.0 && fabs(si.x) < 52.5)
                                {
                                        OffsideSegm_tmpcont.push_back(si);
                                }
                                if (fi.isValid() && fabs(fi.y) < 34.0 && fabs(fi.x) < 52.5 && wm.ball().pos().x < 37.0)
                                {
                                        OffsideSegm_tmpcont.push_back(fi);
                                }
                            }

                            for ( std::vector<Vector2D>::iterator p = OffsideSegm_tmpcont.begin(),
                                      end = OffsideSegm_tmpcont.end();
                                          p != end;
                                          ++p )
                            {

						if ( (*p).x < state.ball().pos().x - 25.0 || (*p).x > 52.5 || fabs((*p).y) > 34.0 )
							continue;

						if ( ( (*p) - state.ball().pos() ).length() > 34.0 )
							continue;


						double min_dist = 1000.0;
						double our_dist = 1000.0;

                                                for ( auto of = wm.opponentsFromSelf().begin();
                                                        of != wm.opponentsFromSelf().end();
                                                        ++of )
                                                {
                                                        Vector2D tmp = (*of)->pos() - (*p);
                                                        if ( min_dist > tmp.length() )
                                                                min_dist = tmp.length();
                                                }

                                                for ( auto  of = wm.teammatesFromSelf().begin();
                                                        of != wm.teammatesFromSelf().end();
                                                        ++of )
                                                {
							if ((*of)->pos().x > wm.offsideLineX() + 1.0) continue;
                                                        Vector2D tmp = (*of)->pos() - (*p);
                                                        if ( our_dist > tmp.length() )
                                                                our_dist = tmp.length();
                                                }

						Vector2D tmp = wm.self().pos() - (*p);
						if ( wm.self().pos().x < (*p).x && tmp.length() > 7.0 && our_dist > tmp.length() )
                                                        our_dist = tmp.length();


					if (max_dist < min_dist - our_dist ) 
					{
						max_dist = min_dist - our_dist;
						best_point = (*p);
					}
			    }
		}
	}


    dlog.addText( Logger::TEAM, __FILE__": best point=(%.1f %.1f)", best_point.x, best_point.y);


    point += std::max( 0.0, 40.0 - best_point.dist( state.ball().pos() ) );

//  point += std::max( 0.0, 40.0 - ServerParam::i().theirTeamGoalPos().dist( state.ball().pos() ) );

#ifdef DEBUG_PRINT
    dlog.addText( Logger::ACTION_CHAIN,
                  "(eval) eval-center (%d) state ball pos (%f, %f)",
                  evalcenter, state.ball().pos().x, state.ball().pos().y );

    dlog.addText( Logger::ACTION_CHAIN,
                  "(eval) initial value (%f)", point );
#endif

    //
    // add bonus for goal, free situation near offside line
    //
    if ( FieldAnalyzer::can_shoot_from( holder->unum() == state.self().unum(),
                                        holder->pos(),
                                        state.getPlayers( new OpponentOrUnknownPlayerPredicate( state.ourSide() ) ),
                                        VALID_PLAYER_THRESHOLD ) )
    {
        point += 1.0e+6;
#ifdef DEBUG_PRINT
        dlog.addText( Logger::ACTION_CHAIN,
                      "(eval) bonus for goal %f (%f)", 1.0e+6, point );
#endif

        if ( holder_unum == state.self().unum() )
        {
            point += 5.0e+5;
#ifdef DEBUG_PRINT
            dlog.addText( Logger::ACTION_CHAIN,
                          "(eval) bonus for goal self %f (%f)", 5.0e+5, point );
#endif
        }
    }

    return point;
}
