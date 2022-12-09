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
    std::string M_player_request_pre_pose = "req";
    std::string M_response_pre_pose = "resp";
    Redis * M_redis_client;
    static RLClient * instant;
    RLClient()
    {
        M_redis_client = new Redis("tcp://127.0.0.1:6379");
    }

    static RLClient * i(){
        if (instant == nullptr)
            instant = new RLClient();
        return instant;
    }
    std::vector<double> player_send_request_and_get_response(int num, int cycle, vector<double> features=vector<double>{-1})
    {
        send_player_request(num, cycle, std::move(features));
        return get_player_response(num, cycle);
    }
    void send_player_request(int num, int cycle, vector<double> features)
    {
        string key = M_player_request_pre_pose + "_" + to_string(num) + "_" + to_string(cycle);
        M_redis_client->rpush(key, features.begin(), features.end());
    }

    std::vector<double> get_player_response(int num, int cycle)
    {
        string key = M_response_pre_pose + "_" + to_string(num) + "_" + to_string(cycle);
        rcsc::Timer start_time = rcsc::Timer();
        std::vector<double> res;
        while (true)
        {
            double waited_time = start_time.elapsedReal(rcsc::Timer::Sec);
            std::cout<<waited_time<<""<<std::endl;
            if (waited_time > 100 ){
                vector<string > keys;
                M_redis_client->keys("*", std::back_inserter(keys));
                for (auto &k:keys){
                    if(k.find("resp_1") != std::string::npos)
                    {
                        M_redis_client->del(key);
                    }
                }
                break;
            }

            std::vector<std::string> vec;

            M_redis_client->lrange(key, 0, -1, std::back_inserter(vec));
            if (!vec.empty()) {
                for (const auto& v : vec)
                    res.push_back(std::stof(v));
                M_redis_client->del(key);
                return res;
            }
            usleep(1000);
        }
        return res;
    }

    void send_trainer_reward(int num, int cycle, vector<double> done_rewards) const
    {
        string key = M_player_request_pre_pose + "_" + to_string(num) + "_" + to_string(cycle);
        M_redis_client->rpush(key, done_rewards.begin(), done_rewards.end());
    }

    void wait_for_python(int num, int cycle) const
    {
        rcsc::Timer start_time = rcsc::Timer();
        string key = M_response_pre_pose + "_" + to_string(num) + "_" + to_string(cycle);
        while (true)
        {
            std::vector<std::string> vec;
            M_redis_client->lrange(key, 0, -1, std::back_inserter(vec));
            if (!vec.empty()){
                M_redis_client->del(key);
                break;
            }
            usleep(1000);
        }
    }
};
#endif //HELIOS_BASE_RL_BY_REDIS_H
