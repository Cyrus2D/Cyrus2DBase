// -*-c++-*-

/*
 *Copyright:

 Copyright (C) Hidehisa AKIYAMA

 This code is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3, or (at your option)
 any later version.

 This code is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this code; see the file COPYING.  If not, write to
 the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.

 *EndCopyright:
 */

/////////////////////////////////////////////////////////////////////

#ifndef RL_TRAINER_H
#define RL_TRAINER_H

#include <rcsc/trainer/trainer_agent.h>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include <grpcpp/grpcpp.h>
#include "../../build/src/cyrus.grpc.pb.h"
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using cyrus::Reward;
using cyrus::OK;
using cyrus::SampleService;


using namespace std;
class RLTrainer
        : public rcsc::TrainerAgent {
private:

public:
    std::shared_ptr<Channel> channel;
    std::unique_ptr<SampleService::Stub> stub_;
    RLTrainer();

    virtual
    ~RLTrainer();

protected:

    /*!
      You can override this method.
      But you must call TrainerAgent::doInit() in this method.
    */
    virtual
    bool initImpl( rcsc::CmdLineParser & cmd_parser );

    //! main decision
    virtual
    void actionImpl();

    virtual
    void handleInitMessage();
    virtual
    void handleServerParam();
    virtual
    void handlePlayerParam();
    virtual
    void handlePlayerType();

private:
    int M_counter;
    int M_max_episode_length;
    bool M_start_sent = false;
    rcsc::Vector2D M_last_pos = rcsc::Vector2D::INVALIDATED;
    int counter()
    {
        return M_counter;
    }
    int max_episode_length()
    {
        return M_max_episode_length;
    }

    void doRL();
    pair<vector<double>, bool> calcRewards();
    void doReset();
    void sendHi();
    void sendResetMessage();
};

#endif
