//
// Created by nader on 2022-11-29.
//

#ifndef HELIOS_BASE_RL_BY_REDIS_H
#define HELIOS_BASE_RL_BY_REDIS_H

#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <sw/redis++/redis++.h>
#include <rcsc/timer.h>
#include <unistd.h>

using namespace std;
using namespace sw::redis;
class RLClient {
public:
    std::string M_request_pre_pose = "req";
    std::string M_response_pre_pose = "resp";
    Redis * M_redis_client;
    static RLClient * instant;
    RLClient();
    static RLClient * i();
    std::vector<double> player_send_request_and_get_response(int num, long cycle, vector<double> features=vector<double>{-1});
    void send_player_request(int num, long cycle, vector<double> features) const;
    std::vector<double> get_player_response(int num, long cycle) const;
    void send_trainer_status_reward(int num, long cycle, vector<double> status_rewards) const;
    void wait_for_python(int num, long cycle) const;
};
#endif //HELIOS_BASE_RL_BY_REDIS_H
