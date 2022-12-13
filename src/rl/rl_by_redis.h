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
#include <type_traits>
using namespace std;
using namespace sw::redis;

class ReceivedMessage {
public:
    bool is_string = false;
    bool is_vector = false;
    bool is_number = false;
    vector<double> vector_message;
    string string_message;
    double double_message;

    ReceivedMessage() {

    }

    ReceivedMessage(vector<double> message_) {
        is_vector = true;
        vector_message = message_;
    }
    ReceivedMessage(double message_) {
        is_number = true;
        double_message = message_;
    }
    ReceivedMessage(int message_) {
        is_number = true;
        double_message = message_;
    }
    ReceivedMessage(string message_) {
        is_string = true;
        string_message = message_;
    }

    template<class T> T message()
    {
        if (std::is_same<T, vector<double> >::value)
        {
            return vector_message;
        }
        if (std::is_same<T, double >::value)
        {
            return double_message;
        }
        if (std::is_same<T, int >::value)
        {
            return double_message;
        }
        if (std::is_same<T, string >::value)
        {
            return string_message;
        }
    }
};
class RLClient {
public:
    std::string M_request_pre_pose = "req";
    std::string M_response_pre_pose = "resp";
    Redis * M_redis_client;
    static RLClient * instant;
    RLClient(int port);
    static RLClient * i(int port=6000);
    std::vector<double> player_send_request_and_get_response(int num, long cycle, unsigned long msg_size, vector<double> features=vector<double>{-1}) const;
    void send_player_fake_message(int num, long cycle, string message) const;
    void send_player_request(int num, long cycle, vector<double> features) const;
    std::vector<double> get_player_response(int num, long cycle, unsigned long msg_size) const;
    void send_trainer_status_reward(int num, long cycle, vector<double> status_rewards) const;
    void wait_for_python(int num, long cycle, vector<double> status_rewards) const;

    ReceivedMessage get_message(int num, long cycle=-1, int wait_time=-1);

    void send_hi(int num, const string& msg) const;
    int redis_db(int port);
};
#endif //HELIOS_BASE_RL_BY_REDIS_H
