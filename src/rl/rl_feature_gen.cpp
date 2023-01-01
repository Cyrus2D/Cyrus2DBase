//
// Created by nader on 2022-11-30.
//

#include "rl_feature_gen.h"
std::vector<double> RLFeatureGen::feature_generator(const rcsc::WorldModel & wm){
    std::vector<double> res;
    res.push_back(wm.self().pos().r() / 150.0);
    res.push_back(wm.self().pos().th().degree() / 180.0);
    res.push_back(wm.self().body().degree() / 180.0);
    res.push_back(wm.self().pos().x / 52.5);
    res.push_back(wm.self().pos().y / 34);
    res.push_back(wm.ball().pos().r() / 150.0);
    res.push_back(wm.ball().pos().th().degree() / 180.0);
    res.push_back(wm.ball().pos().x / 52.5);
    res.push_back(wm.ball().pos().y / 34);
    res.push_back((wm.ball().pos() - wm.self().pos()).th().degree() / 180.0);
    res.push_back((wm.ball().pos().dist(wm.self().pos())) / 150.0);
    res.push_back(((wm.ball().pos() - wm.self().pos()).th() - wm.self().body()).degree() / 180.0);
    return res;
}
