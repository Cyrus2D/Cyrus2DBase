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

std::vector<double> RLClient::player_send_request_and_get_response(int num, long cycle, unsigned long msg_size, vector<double> features) const
{
    send_player_request(num, cycle, std::move(features));
    return get_player_response(num, cycle, msg_size);
}

void RLClient::send_player_request(int num, long cycle, vector<double> features) const
{
    string key = M_request_pre_pose + "_" + to_string(num) + "_" + to_string(cycle);
    long long res = 0;
    while (res <= 0){
        res = M_redis_client->rpush(key, features.begin(), features.end());
        if (res <= 0){
            M_redis_client->del(key);
        }
    }
}

std::vector<double> RLClient::get_player_response(int num, long cycle, unsigned long msg_size) const
{
    string key = M_response_pre_pose + "_" + to_string(num) + "_" + to_string(cycle);
    rcsc::Timer start_time = rcsc::Timer();
    std::vector<double> res;
    while (true)
    {
        double waited_time = start_time.elapsedReal(rcsc::Timer::Sec);
        if (waited_time > 100 ){
            vector<string> keys;
            M_redis_client->keys("*", std::back_inserter(keys));
            for (auto &k:keys){
                if(k.find(M_response_pre_pose + "_" + to_string(num)) != std::string::npos)
                {
                    M_redis_client->del(key);
                }
            }
            break;
        }

        std::vector<std::string> vec;

        M_redis_client->lrange(key, 0, -1, std::back_inserter(vec));
        if (!vec.empty() && vec.size() == msg_size) {
            for (const auto& v : vec)
                res.push_back(std::stof(v));
            M_redis_client->del(key);
            return res;
        }
        usleep(1000);
    }
    return res;
}

void RLClient::send_trainer_status_reward(int num, long cycle, vector<double> status_rewards) const
{
    string key = M_request_pre_pose + "_" + to_string(num) + "_" + to_string(cycle);
    long long res = 0;
    while (res <= 0){
        res = M_redis_client->rpush(key, status_rewards.begin(), status_rewards.end());
        if (res <= 0){
            M_redis_client->del(key);
        }
    }
}

void RLClient::wait_for_python(int num, long cycle, vector<double> status_rewards) const
{
    rcsc::Timer start_time = rcsc::Timer();
    string key = M_response_pre_pose + "_" + to_string(num) + "_" + to_string(cycle);
    int i = 0;
    while (true)
    {
        if (i > 300){
            i = 0;
            string sent_key = M_request_pre_pose + "_" + to_string(num) + "_" + to_string(cycle);
            M_redis_client->del(sent_key);
            RLClient::i()->send_trainer_status_reward(0, cycle, status_rewards);

        }
        std::vector<std::string> vec;
        M_redis_client->lrange(key, 0, -1, std::back_inserter(vec));
        if (!vec.empty()){
            M_redis_client->del(key);
            break;
        }
        usleep(1000);
        i += 1;
    }
}
