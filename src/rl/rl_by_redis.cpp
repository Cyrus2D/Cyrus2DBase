//
// Created by nader on 2022-12-04.
//

#include "rl_by_redis.h"

RLClient * RLClient::instant = nullptr;

RLClient::RLClient(int port)
{
    int db = redis_db(port);
    M_redis_client = new Redis("tcp://127.0.0.1:6379/" + to_string(db));
}

RLClient * RLClient::i(int port){
    if (instant == nullptr)
        instant = new RLClient(port);
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

void RLClient::send_player_fake_message(int num, long cycle, string message) const
{
    string key = M_request_pre_pose + "_" + to_string(num) + "_" + to_string(cycle);
    while (true){
        bool res = M_redis_client->set(key, message);
        if (res){
            break;
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

ReceivedMessage RLClient::get_message(int num, long cycle, int wait_time)
{
    rcsc::Timer start_time = rcsc::Timer();
    string key = M_response_pre_pose + "_" + to_string(num);
    if (cycle != -1)
    {
        key += "_";
        key += to_string(cycle);
    }
    while (true)
    {
        if (M_redis_client->exists(key))
        {
            if (M_redis_client->type(key) == string("list"))
            {
                std::vector<std::string> vec;
                M_redis_client->lrange(key, 0, -1, std::back_inserter(vec));
                if (!vec.empty()){
                    vector<double> vec_double;
                    vec_double.reserve(vec.size());
                    for (auto & v: vec)
                        vec_double.push_back(stod(v));
                    M_redis_client->del(key);
                    return ReceivedMessage(vec_double);
                }
            }
            else if (M_redis_client->type(key) == string("string"))
            {
                auto msg = M_redis_client->get(key).value();
                M_redis_client->del(key);
                return ReceivedMessage(msg);
            }
        }
        double waited_time = start_time.elapsedReal(rcsc::Timer::Sec);
        if (wait_time != -1 && waited_time > wait_time ){ //100 for player
            return ReceivedMessage();
        }
        usleep(1000);
    }
    return ReceivedMessage();
}

void
RLClient::send_hi(int num, const string& msg) const
{
    string key = M_request_pre_pose + "_" + to_string(num);
    long long res = 0;
    while (res <= 0){
        res = M_redis_client->set(key, msg);
    }
}

int RLClient::redis_db(int port){
    port = port - 6000;
    return int(port / 10) + 1;
}
