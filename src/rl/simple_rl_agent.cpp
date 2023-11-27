//
// Created by nader on 2022-11-30.
//

#include "simple_rl_agent.h"
#include "rl_feature_gen.h"
#include <stdexcept>
#include <rcsc/geom/vector_2d.h>

SimpleRLAgent * SimpleRLAgent::inst = nullptr;

void SimpleRLAgent::do_action(rcsc::PlayerAgent * agent){
    std::cout<<"do_action called"<<std::endl;
    const rcsc::WorldModel & wm = agent->world();
    StateMessage state;
    state.set_cycle(wm.time().cycle());
    // auto rawState = state.mutable_rawist();
    
    // message RawListMessage {
    //     repeated float Value = 1;
    // }
    auto rawState = state.mutable_rawlist();
    rawState -> add_value((wm.self().body() - (wm.ball().pos() - wm.self().pos()).th()).degree());

    // auto goto_ball_state = state.mutable_gotoballstate();
    // auto ball = goto_ball_state->mutable_ballposition();
    // ball->set_x(wm.ball().pos().x);
    // ball->set_y(wm.ball().pos().y);
    // auto selfPos = goto_ball_state->mutable_position();
    // selfPos->set_x(wm.self().pos().x);
    // selfPos->set_y(wm.self().pos().y);
    // goto_ball_state->set_body(wm.self().body().degree());
    // goto_ball_state->set_bodydiff((wm.self().body() - (wm.ball().pos() - wm.self().pos()).th()).degree());
    ClientContext context;
    Action action;
    stub_->GetBestAction(&context, state, &action);

    std::cout << "Action received: " << action.DebugString() << std::endl;

    if (action.action_case() == Action::kTurn)
    {
        agent->doTurn(action.turn().dir());
    }
    else if (action.action_case() == Action::kDash)
    {
        agent->doDash(100, action.dash().dir());
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