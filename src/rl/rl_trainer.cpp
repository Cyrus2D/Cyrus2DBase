// -*-c++-*-

/*
 *Copyright:

 Copyright (C) Hidehisa AKIYAMA

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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "rl_trainer.h"

#include <rcsc/trainer/trainer_command.h>
#include <rcsc/trainer/trainer_config.h>
#include <rcsc/coach/coach_world_model.h>
#include <rcsc/common/abstract_client.h>
#include <rcsc/common/player_param.h>
#include <rcsc/common/player_type.h>
#include <rcsc/common/server_param.h>
#include <rcsc/param/param_map.h>
#include <rcsc/param/cmd_line_parser.h>
#include <rcsc/random.h>
#include <stdexcept>

using namespace rcsc;

/*-------------------------------------------------------------------*/
/*!

 */
RLTrainer::RLTrainer()
        : TrainerAgent()
{
    M_counter = 0;
    M_max_episode_length = 200;
}

/*-------------------------------------------------------------------*/
/*!

 */
RLTrainer::~RLTrainer()
{

}

/*-------------------------------------------------------------------*/
/*!

 */
bool
RLTrainer::initImpl( CmdLineParser & cmd_parser )
{
    bool result = TrainerAgent::initImpl( cmd_parser );

#if 0
    ParamMap my_params;

    std::string formation_conf;
    my_map.add()
        ( &conf_path, "fconf" )
        ;

    cmd_parser.parse( my_params );
#endif

    if ( cmd_parser.failed() )
    {
        std::cerr << "coach: ***WARNING*** detected unsupported options: ";
        cmd_parser.print( std::cerr );
        std::cerr << std::endl;
    }

    if ( ! result )
    {
        return false;
    }

    //////////////////////////////////////////////////////////////////
    // Add your code here.
    //////////////////////////////////////////////////////////////////

    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
void
RLTrainer::actionImpl()
{
    if ( world().teamNameLeft().empty() )
    {
        doTeamNames();
        return;
    }

    //////////////////////////////////////////////////////////////////
    // Add your code here.

    //sampleAction();
    //recoverForever();
    //doSubstitute();
//    doKeepaway();
    doRL();
}

/*-------------------------------------------------------------------*/
/*!

*/
void
RLTrainer::handleInitMessage()
{

}

/*-------------------------------------------------------------------*/
/*!

*/
void
RLTrainer::handleServerParam()
{

}

/*-------------------------------------------------------------------*/
/*!

*/
void
RLTrainer::handlePlayerParam()
{

}

/*-------------------------------------------------------------------*/
/*!

*/
void
RLTrainer::handlePlayerType()
{

}

/*-------------------------------------------------------------------*/
/*!

 */

void
RLTrainer::doRL()
{
    const CoachPlayerObject * player = world().teammate(1);
    if (player == nullptr || player->unum() != 1){
        return;
    }
    RLClient::i(this->config().port());
    if (!M_start_sent)
    {
        sendHi();
        doReset();
        M_start_sent = true;
        return;
    }
    auto status_rewards_done = calcRewards();
    auto status_rewards = status_rewards_done.first;
    bool done = status_rewards_done.second;
    M_counter += 1;
    RLClient::i()->send_trainer_status_reward(0, world().time().cycle(), status_rewards);
//    RLClient::i()->wait_for_python(0, world().time().cycle(), status_rewards);
    auto resp = RLClient::i()->get_message(0, world().time().cycle());
    if (!resp.is_string)
        throw std::runtime_error("Error: the trainer did not receive string from python.");
    if (resp.string_message != string ("OK"))
    {
        throw std::runtime_error(string ("Error: doRL: the trainer did not receive \"OK\" from python. Message:") + resp.string_message);
    }

    if (done)
    {
        doReset();
        return;
    }
}

/*-------------------------------------------------------------------*/
/*!

 */

pair<vector<double>, bool>
RLTrainer::calcRewards()
{
    double goal_reward = 2.0;
    double move_reward = -0.01;
    double move_out_reward = -2.0;
    vector<double> res;

    rcsc::Vector2D ball_pos = world().ball().pos();
    const CoachPlayerObject * player = world().teammate(1);
    rcsc::Vector2D player_pos = player->pos();
    Vector2D target_pos = Vector2D(0,0);//world().ball().pos();
    double diff_dist = 0.0;
    if (M_last_pos.isValid())
        diff_dist = M_last_pos.dist(target_pos) - player_pos.dist(target_pos);
    double diff_dist_reward = diff_dist / 100.0;
    M_last_pos = player_pos;

    double target_r = 10.0;
//    0 start
//    1 normal
//    2 end goal
//    3 end out
//    4 end time
    if (player_pos.dist(target_pos) < target_r)
    {
        res.push_back(2);  //end goal
        res.push_back(goal_reward);
        return make_pair(res, true);
    }

    if (player_pos.absX() > 52.5 || player_pos.absY() > 34.0)
    {
        res.push_back(3);  //end out
        res.push_back(move_out_reward);
        return make_pair(res, true);
    }


    if (counter() > max_episode_length())
    {
        res.push_back(4);  //end time
        res.push_back(move_reward + diff_dist_reward);
        return make_pair(res, true);
    }
    res.push_back(M_counter == -1 ? 0 : 1);  // normal or start
    res.push_back(move_reward + diff_dist_reward);
    return make_pair(res, false);
}

/*-------------------------------------------------------------------*/
/*!

 */
void
RLTrainer::sendHi()
{
    RLClient::i()->send_hi(0, to_string(world().time().cycle()));
    auto resp = RLClient::i()->get_message(0);
    if (!resp.is_string || resp.string_message != string ("OK"))
        throw std::runtime_error(string ("Error: doRL: the trainer did not receive \"OK\" from python. Message:") + resp.string_message);
}

/*-------------------------------------------------------------------*/
/*!

 */
void
RLTrainer::doReset()
{
    doRecover();
    // move ball to center
//    doMoveBall( Vector2D( -10.0, -30.0 ),
//                Vector2D( 0.0, 0.0 ) );
    // change playmode to play_on
    doChangeMode( PM_PlayOn );
    {
        // move player to random point
        UniformReal uni01( 0.0, 1.0 );
        UniformReal uni02( 0.0, 1.0 );
        UniformReal uni03( 0.0, 1.0 );
        Vector2D move_pos (uni01() * 105.0 - 52.5, uni02() * 68.0 - 34.0);
        doMovePlayer( config().teamName(),
                      1, // uniform number
                      move_pos,
                      uni03() * 360.0 - 180.0 );
        UniformReal uni04( 0.0, 1.0 );
        UniformReal uni05( 0.0, 1.0 );
        doMoveBall(Vector2D(uni01() * 105.0 - 52.5, uni02() * 68.0 - 34.0),
                   Vector2D(0, 0));
    }
    M_counter = -1;
}
