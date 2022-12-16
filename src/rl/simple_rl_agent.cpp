//
// Created by nader on 2022-11-30.
//

#include "simple_rl_agent.h"
#include "rl_feature_gen.h"
#include <stdexcept>
#include <random>

SimpleRLAgent * SimpleRLAgent::inst = nullptr;

void SimpleRLAgent::do_action(rcsc::PlayerAgent * agent){
    const rcsc::WorldModel & wm = agent->world();
    auto features = RLFeatureGen().feature_generator(wm);
    RLClient::i()->send_player_request(wm.self().unum(), wm.time().cycle(), std::move(features));
    auto resp = RLClient::i()->get_message(wm.self().unum(), wm.time().cycle(), 100);
    if (resp.is_vector){
        auto action = resp.vector_message;
        if (action.empty())
        {
            agent->doTurn(0);
        }
        else
        {
            agent->doDash(100, action[0] * 180.0);
//            double turn_prob = action[0];
//            double dash_prob = action[1];
//            double turn_angle = action[2];
//            double dash_angle = action[3];
//            if (turn_prob + dash_prob > 0)
//            {
//                turn_prob /= (turn_prob + dash_prob);
//                dash_prob /= (turn_prob + dash_prob);
//            }
//            else
//            {
//                turn_prob = 0.5;
//                dash_prob = 0.5;
//            }
//
//            std::random_device rd;
//            std::mt19937 mt(rd());
//            std::uniform_real_distribution<double> r(0.0, 1.0);
//
//            if (r(mt) < turn_prob)
//            {
//                agent->doTurn(turn_angle * 360.0 - 180.0);
//            }
//            else
//            {
//                agent->doDash(100, dash_angle * 360.0 - 180.0);
//            }
        }
    }
    else if (resp.is_string)
    {
        if (resp.string_message != string("OK"))
            throw std::runtime_error("The player receives string message except \"OK\" message.");
    }
    else if (resp.is_number)
    {
        throw std::runtime_error("The response to player can not be a number.");
    }
    else
    {
    }
}

SimpleRLAgent * SimpleRLAgent::i(){
    if (SimpleRLAgent::inst == nullptr){
        SimpleRLAgent::inst = new SimpleRLAgent();
    }
    return SimpleRLAgent::inst;
}