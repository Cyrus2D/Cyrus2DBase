/*
    Copyright:
    Cyrus2D
    Modified by Aref Sayareh, Nader Zare, Omid Amini
*/

#ifndef CYRUS_OffensiveDataExtractor_H
#define CYRUS_OffensiveDataExtractor_H

#include <fstream>
#include <rcsc/geom.h>
#include <rcsc/player/player_agent.h>
//#include "../chain_action/action_state_pair.h"
#include "../planner/cooperative_action.h"
#include "DEState.h"

//#include "shoot_generator.h"
enum ODEDataSide {
    NONE,
    TM,
    OPP,
    BOTH,
    Kicker
};

class OffensiveDataExtractor {
private:
    struct Option {
    public:
        bool cycle;
        bool ball_pos;
        ODEDataSide unum;
        ODEDataSide pos;
        ODEDataSide relativePos;
        ODEDataSide polarPos;
        ODEDataSide isKicker;
        ODEDataSide openAnglePass;
        ODEDataSide nearestOppDist;
        ODEDataSide in_offside;

        bool use_convertor;
        Option();
    };

private:
    std::vector<double> features;
    std::ofstream fout;
    long last_update_cycle;
    std::vector<double> data;

public:

    OffensiveDataExtractor();

    ~OffensiveDataExtractor();
    Option option;
    void generate_save_data(const WorldModel & wm,
                            const CooperativeAction &action,
                            bool update_shoot=false);
    std::string get_header();

    //accessors
    static OffensiveDataExtractor &i();
    static bool active;

    void extract_output(DEState &state,
                        int category,
                        const rcsc::Vector2D &target,
                        const int &unum,
                        const char *desc,
                        double bell_speed);

    std::vector<double> get_data(DEState &state);
private:
    void init_file(DEState &state);

    void extract_ball(DEState &state);

    void extract_players(DEState &state);

    void add_null_player(int unum, ODEDataSide side);

    void extract_pos(DEPlayer *player, DEState &state, ODEDataSide side);

    void extract_pass_angle(DEPlayer *player, DEState &state, ODEDataSide side);

    void extract_base_data(DEPlayer *player, ODEDataSide side, DEState &state);

    void extract_type(DEPlayer *player, ODEDataSide side);

    uint find_unum_index(DEState &state, uint unum);

    double convertor_x(double x);

    double convertor_y(double y);

    double convertor_dist(double dist);

    double convertor_dist_x(double dist);

    double convertor_dist_y(double dist);

    double convertor_angle(double angle);

    double convertor_type(double type);

    double convertor_cycle(double cycle);

    double convertor_bv(double bv);

    double convertor_bvx(double bvx);

    double convertor_bvy(double bvy);

    double convertor_pv(double pv);

    double convertor_pvx(double pvx);

    double convertor_pvy(double pvy);

    double convertor_unum(double unum);

    double convertor_card(double card);

    double convertor_stamina(double stamina);

    double convertor_counts(double count);

    void extract_drible_angles(DEState &state);

    std::vector<DEPlayer *> sort_players(DEState &state);
};

class ODEPolar {
public:
    double r;
    double teta;

    ODEPolar(rcsc::Vector2D p);
};

class ODEOpenAngle {
public:
    int unum;
    double dist_self_to_opp;
    double dist_self_to_opp_proj;
    double dist_opp_proj;
    double open_angle;

    ODEOpenAngle(){};
    ODEOpenAngle(
        int _unum,
        double _dist_self_to_opp,
        double _dist_self_to_opp_proj,
        double _dist_opp_proj,
        double _open_angle
    ) {
        unum = _unum;
        dist_self_to_opp = _dist_self_to_opp;
        dist_self_to_opp_proj = _dist_self_to_opp_proj;
        dist_opp_proj = _dist_opp_proj;
        open_angle = _open_angle;
    };
};


#endif //CYRUS_OffensiveDataExtractor_H
