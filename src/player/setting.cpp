//
// Created by nader on 6/15/24.
//

#include "setting.h"

Setting &Setting::i()
{
    static Setting instance;
    return instance;
}

void Setting::find_version(const string &json_str)
{
    try
    {
        json j = json::parse(json_str);

        if (j.contains("version"))
            version = j.at("version").get<int>();
        std::cout << "version: " << version << std::endl;
    }
    catch (exception &e)
    {
        std::cout << "Error in parsing json string: " << e.what() << std::endl;
    }
}

void replaceAll(std::string &str, const std::string &from, const std::string &to)
{
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos)
    {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}

string Setting::decode(const std::string &encoding, const std::string &encoded_string)
{
    if (encoding == "json")
    {
        std::cout << "json encoding" << std::endl;
        return encoded_string;
    }
    else if (encoding == "temp")
    {
        // replace #qq# with "
        // replace #q# with '
        // replace #c# with ,
        std::cout << "temp encoding" << std::endl;
        string decoded_string = encoded_string;
        replaceAll(decoded_string, "#qq#", "\"");
        replaceAll(decoded_string, "#q#", "'");
        replaceAll(decoded_string, "#c#", ",");
        return decoded_string;
    }
    else
    {
        std::cout << "Encoding is not supported" << std::endl;
        return "json";
    }
}

void Setting::load_from_json_string(const string &json_str, const string &encoding)
{
    string decoded_json_str = decode(encoding, json_str);
    find_version(decoded_json_str);
    if (version == 1)
    {
        try
        {
            json j = json::parse(decoded_json_str);

            if (j.contains("formation_name"))
            {
                formation_name = j.at("formation_name").get<std::string>();
                if (std::find(formation_options.begin(), formation_options.end(), formation_name) == formation_options.end())
                {
                    std::cout << "Formation name is not valid" << std::endl;
                    formation_name = formation_options.at(0);
                }
            }
            if (j.contains("winner_formation_name"))
            {
                winner_formation_name = j.at("winner_formation_name").get<std::string>();
                if (std::find(formation_options.begin(), formation_options.end(), winner_formation_name) == formation_options.end())
                {
                    std::cout << "Formation name is not valid" << std::endl;
                    winner_formation_name = formation_options.at(0);
                }
            }
            if (j.contains("loser_formation_name"))
            {
                loser_formation_name = j.at("loser_formation_name").get<std::string>();
                if (std::find(formation_options.begin(), formation_options.end(), loser_formation_name) == formation_options.end())
                {
                    std::cout << "Formation name is not valid" << std::endl;
                    loser_formation_name = formation_options.at(0);
                }
            }
            if (j.contains("offensive_kick_planner_use_direct_pass"))
                offensive_kick_planner_use_direct_pass = j.at("offensive_kick_planner_use_direct_pass").get<bool>();
            if (j.contains("offensive_kick_planner_use_lead_pass"))
                offensive_kick_planner_use_lead_pass = j.at("offensive_kick_planner_use_lead_pass").get<bool>();
            if (j.contains("offensive_kick_planner_use_through_pass"))
                offensive_kick_planner_use_through_pass = j.at("offensive_kick_planner_use_through_pass").get<bool>();
            if (j.contains("offensive_kick_planner_use_cross_pass"))
                offensive_kick_planner_use_cross_pass = j.at("offensive_kick_planner_use_cross_pass").get<bool>();
            if (j.contains("offensive_kick_planner_use_short_dribble"))
                offensive_kick_planner_use_short_dribble = j.at("offensive_kick_planner_use_short_dribble").get<bool>();
            if (j.contains("offensive_kick_planner_use_long_dribble"))
                offensive_kick_planner_use_long_dribble = j.at("offensive_kick_planner_use_long_dribble").get<bool>();
            if (j.contains("offensive_kick_planner_use_sample_shot"))
                offensive_kick_planner_use_sample_shot = j.at("offensive_kick_planner_use_sample_shot").get<bool>();
            if (j.contains("offensive_kick_planner_use_sample_pass"))
                offensive_kick_planner_use_sample_pass = j.at("offensive_kick_planner_use_sample_pass").get<bool>();
            if (j.contains("offensive_kick_planner_use_sample_dribble"))
                offensive_kick_planner_use_sample_dribble = j.at("offensive_kick_planner_use_sample_dribble").get<bool>();
            if (j.contains("moving_save_energy"))
            {
                moving_save_energy = j.at("moving_save_energy").get<double>();
                if (moving_save_energy < 0)
                    moving_save_energy = 0;
                if (moving_save_energy > 100)
                    moving_save_energy = 100;
            }
            if (j.contains("pressing"))
            {
                pressing = j.at("pressing").get<double>();
                if (pressing != 0)
                    pressing = 1;
                if (pressing == 0)
                    pressing = 0;
            }
            if (j.contains("offside_trap"))
            {
                offside_trap = j.at("offside_trap").get<int>();
                if (offside_trap != 0)
                    offside_trap = 1;
                if (offside_trap == 0)
                    offside_trap = 0;
            }

            if (j.contains("aggressiveness"))
            {
                aggressiveness = j.at("aggressiveness").get<double>();
                if (aggressiveness < 0)
                    aggressiveness = 0;
                if (aggressiveness > 1)
                    aggressiveness = 1;
            }
        }
        catch (exception &e)
        {
            std::cout << "Error in parsing json string: " << e.what() << std::endl;
        }
    }
    else
    {
        std::cout << "Version is not supported or it was not found in json" << std::endl;
    }
}

void Setting::read_from_file(string file_path, const string &encoding)
{
    if (file_path.find(".json") == string::npos)
    {
        if (file_path[file_path.size() - 1] == '/')
            file_path = file_path.substr(0, file_path.size() - 1);
        file_path = file_path + "/config.json";
    }
    if (!std::filesystem::exists(file_path))
    {
        std::cerr << "File does not exist - '" << file_path << "'" << std::endl;
        return;
    }
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        std::cerr << "Could not open the file - '" << file_path << "'" << std::endl;
        return;
    }
    string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    load_from_json_string(json_str, encoding);
}

void Setting::print() const
{
    std::cout << "formation_name: " << formation_name << std::endl;
    std::cout << "winner_formation_name: " << winner_formation_name << std::endl;
    std::cout << "loser_formation_name: " << loser_formation_name << std::endl;
    std::cout << "offensive_kick_planner_use_direct_pass: " << offensive_kick_planner_use_direct_pass << std::endl;
    std::cout << "offensive_kick_planner_use_lead_pass: " << offensive_kick_planner_use_lead_pass << std::endl;
    std::cout << "offensive_kick_planner_use_through_pass: " << offensive_kick_planner_use_through_pass << std::endl;
    std::cout << "offensive_kick_planner_use_cross_pass: " << offensive_kick_planner_use_cross_pass << std::endl;
    std::cout << "offensive_kick_planner_use_short_dribble: " << offensive_kick_planner_use_short_dribble << std::endl;
    std::cout << "offensive_kick_planner_use_long_dribble: " << offensive_kick_planner_use_long_dribble << std::endl;
    std::cout << "offensive_kick_planner_use_sample_shot: " << offensive_kick_planner_use_sample_shot << std::endl;
    std::cout << "offensive_kick_planner_use_sample_pass: " << offensive_kick_planner_use_sample_pass << std::endl;
    std::cout << "offensive_kick_planner_use_sample_dribble: " << offensive_kick_planner_use_sample_dribble << std::endl;
    std::cout << "moving_save_energy: " << moving_save_energy << std::endl;
    std::cout << "pressing: " << pressing << std::endl;
}

void Setting::read_from_arguments(int argc, char *argv[])
{
    for (int i = 0; i < argc; i++)
    {
        std::cout << argv[i] << std::endl;
    }
    string encoding = "json";
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-e") == 0)
        {
            if (i + 1 < argc)
            {
                encoding = argv[i + 1];
            }
        }
    }
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-j") == 0)
        {
            if (i + 1 < argc)
            {
                load_from_json_string(argv[i + 1], encoding);
            }
        }
        else if (strcmp(argv[i], "-c") == 0)
        {
            if (i + 1 < argc)
            {
                read_from_file(argv[i + 1], encoding);
            }
        }
    }
    print();
}
