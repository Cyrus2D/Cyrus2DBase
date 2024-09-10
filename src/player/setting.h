//
// Created by nader on 6/15/24.
//

#ifndef HELIOS_BASE_SETTING_H
#define HELIOS_BASE_SETTING_H

#include "json.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using json = nlohmann::json;

class Setting
{
public:
    static Setting &i();

    int version = 0;
    vector<string> formation_options = {"433", "442", "523"};

    string formation_name = "433";
    string winner_formation_name = "433";
    string loser_formation_name = "433";
    // options: 433, 442, 523

    bool offensive_kick_planner_use_direct_pass = true;
    bool offensive_kick_planner_use_lead_pass = true;
    bool offensive_kick_planner_use_through_pass = true;
    bool offensive_kick_planner_use_cross_pass = true;
    bool offensive_kick_planner_use_short_dribble = true;
    bool offensive_kick_planner_use_long_dribble = true;
    bool offensive_kick_planner_use_sample_shot = true;
    bool offensive_kick_planner_use_sample_pass = true;
    bool offensive_kick_planner_use_sample_dribble = true;

    double moving_save_energy = 60;    // 0 to 100
    double pressing = 0; // 0 or 1
    int offside_trap = 0;
    double aggressiveness = 0.5; // 0 to 1

    void find_version(const string &json_string);

    void load_from_json_string(const string &json_string, const string &encoding = "json");

    void read_from_file(string file_path, const string &encoding = "json");

    string decode(const string &encoding, const string &encoded_string);

    void print() const;

    void read_from_arguments(int argc, char *argv[]);
};

#endif // HELIOS_BASE_SETTING_H
