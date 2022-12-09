//
// Created by nader on 2022-11-30.
//

#ifndef HELIOS_BASE_SIMPLE_RL_AGENT_H
#define HELIOS_BASE_SIMPLE_RL_AGENT_H

#include <rcsc/player/player_agent.h>
#include "rl_by_redis.h"
class SimpleRLAgent {
public:
    static SimpleRLAgent * inst;
    SimpleRLAgent()
    {
    }
    void do_action(rcsc::PlayerAgent * agent);
    static SimpleRLAgent * i();

};


#endif //HELIOS_BASE_SIMPLE_RL_AGENT_H
