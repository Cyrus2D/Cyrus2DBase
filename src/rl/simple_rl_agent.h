//
// Created by nader on 2022-11-30.
//

#ifndef HELIOS_BASE_SIMPLE_RL_AGENT_H
#define HELIOS_BASE_SIMPLE_RL_AGENT_H

#include <rcsc/player/player_agent.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include <grpcpp/grpcpp.h>
#include "../../build/src/rl/cyrus.grpc.pb.h"
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using cyrus::SampleService;
using cyrus::State;
using cyrus::Action;

class SimpleRLAgent {
public:
    static SimpleRLAgent * inst;
    std::shared_ptr<Channel> channel;
    std::unique_ptr<SampleService::Stub> stub_;
    SimpleRLAgent()
    {
        channel = grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials());
        stub_ = SampleService::NewStub(channel);
    }
    void do_action(rcsc::PlayerAgent * agent);
    static SimpleRLAgent * i();

};


#endif //HELIOS_BASE_SIMPLE_RL_AGENT_H
