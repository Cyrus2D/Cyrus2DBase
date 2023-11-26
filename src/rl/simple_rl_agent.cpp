//
// Created by nader on 2022-11-30.
//

#include "simple_rl_agent.h"
#include "rl_feature_gen.h"
#include <stdexcept>

SimpleRLAgent * SimpleRLAgent::inst = nullptr;

void SimpleRLAgent::do_action(rcsc::PlayerAgent * agent){
    std::cout<<"do_action called"<<std::endl;
    const rcsc::WorldModel & wm = agent->world();
    // auto features = RLFeatureGen().feature_generator(wm);
    State state;
    state.set_cycle(wm.time().cycle() );
    auto position = state.mutable_position();
    position->set_x(wm.self().pos().x);
    position->set_y(wm.self().pos().y);
    auto body = state.mutable_body();
    body->set_angle(wm.self().body().degree());
    ClientContext context;
    Action action;
    stub_->GetBestAction(&context, state, &action);

    std::cout << "Action received: " << action.DebugString() << std::endl;

    if (action.action_case() == Action::kTurn)
    {
        agent->doTurn(0);
    }
    else if (action.action_case() == Action::kDash)
    {
        agent->doDash(100, action.dash().dir().angle());
    }
    else
    {
        throw std::runtime_error("The response to player can not be a number.");
    }
}

SimpleRLAgent * SimpleRLAgent::i(){
    if (SimpleRLAgent::inst == nullptr){
        SimpleRLAgent::inst = new SimpleRLAgent();
    }
    return SimpleRLAgent::inst;
}