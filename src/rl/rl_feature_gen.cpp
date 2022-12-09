//
// Created by nader on 2022-11-30.
//

#include "rl_feature_gen.h"
std::vector<double> RLFeatureGen::feature_generator(const rcsc::WorldModel & wm){
    std::vector<double> res;
    res.push_back(wm.self().pos().r() / 150.0);
    res.push_back(wm.self().pos().th().degree() / 180.0);
    res.push_back(wm.self().body().degree() / 180.0);
    return res;
}