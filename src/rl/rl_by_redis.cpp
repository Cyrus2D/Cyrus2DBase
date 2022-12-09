//
// Created by nader on 2022-12-04.
//

#include "rl_by_redis.h"

RLClient * RLClient::instant = nullptr;

RLClient::RLClient()
{
    M_redis_client = new Redis("tcp://127.0.0.1:6379");
}

RLClient * RLClient::i(){
    if (instant == nullptr)
        instant = new RLClient();
    return instant;
}

std::vector<double> RLClient::player_send_request_and_get_response(int num, long cycle, vector<double> features)
{
    send_player_request(num, cycle, std::move(features));
    return get_player_response(num, cycle);
}

void RLClient::send_player_request(int num, long cycle, vector<double> features)
{
    string key = M_player_request_pre_pose + "_" + to_string(num) + "_" + to_string(cycle);
    M_redis_client->rpush(key, features.begin(), features.end());
}

std::vector<double> RLClient::get_player_response(int num, long cycle)
{
    string key = M_response_pre_pose + "_" + to_string(num) + "_" + to_string(cycle);
    rcsc::Timer start_time = rcsc::Timer();
    std::vector<double> res;
    while (true)
    {
        double waited_time = start_time.elapsedReal(rcsc::Timer::Sec);
        if (waited_time > 100 ){
            vector<string > keys;
            M_redis_client->keys("*", std::back_inserter(keys));
            for (auto &k:keys){
                if(k.find("resp_" + to_string(num)) != std::string::npos)
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

void RLClient::send_trainer_reward(int num, long cycle, vector<double> done_rewards) const
{
    string key = M_player_request_pre_pose + "_" + to_string(num) + "_" + to_string(cycle);
    M_redis_client->rpush(key, done_rewards.begin(), done_rewards.end());
}

void RLClient::wait_for_python(int num, long cycle) const
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
