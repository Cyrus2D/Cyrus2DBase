//
// Created by nader on 2022-11-30.
//

#ifndef HELIOS_BASE_RL_FEATURE_GEN_H
#define HELIOS_BASE_RL_FEATURE_GEN_H

#include <rcsc/player/world_model.h>
#include <vector>
class RLFeatureGen {
public:
    RLFeatureGen() = default;

    std::vector<double> feature_generator(const rcsc::WorldModel & wm);
};


#endif //HELIOS_BASE_RL_FEATURE_GEN_H
