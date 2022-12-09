//
// Created by nader on 2022-11-30.
//

#include "simple_rl_agent.h"
#include "rl_feature_gen.h"
SimpleRLAgent * SimpleRLAgent::inst = nullptr;

void SimpleRLAgent::do_action(rcsc::PlayerAgent * agent){
    const rcsc::WorldModel & wm = agent->world();
    auto features = RLFeatureGen().feature_generator(wm);
    auto action = RLClient::i()->player_send_request_and_get_response(1, wm.time().cycle(), features);
    if (action.empty())
    {
        agent->doTurn(0);
    }
    else
    {
        agent->doDash(100, action[0] * 360.0 - 180.0);
    }
}

SimpleRLAgent * SimpleRLAgent::i(){
    if (SimpleRLAgent::inst == nullptr){
        SimpleRLAgent::inst = new SimpleRLAgent();
    }
    return SimpleRLAgent::inst;
}