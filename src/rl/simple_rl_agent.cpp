//
// Created by nader on 2022-11-30.
//

#include "simple_rl_agent.h"
#include "rl_feature_gen.h"
#include <stdexcept>

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