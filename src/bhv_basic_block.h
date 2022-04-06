//
// Created by nader on 2022-04-05.
//

#ifndef BHV_BASIC_BLOCK_H
#define BHV_BASIC_BLOCK_H

#include <rcsc/geom/vector_2d.h>
#include <rcsc/player/soccer_action.h>
#include <vector>
class Bhv_BasicBlock
    : public rcsc::SoccerBehavior {
        public:
    Bhv_BasicBlock()
        { }

    bool execute( rcsc::PlayerAgent * agent );
    std::vector<int> get_blockers( const rcsc::WorldModel & wm );
    std::pair<int, Vector2D> get_best_blocker( const rcsc::WorldModel & wm, std::vector<int> & tm_blockers);
    rcsc::AngleDeg dribble_direction_detector ( Vector2D dribble_pos );
    static int last_block_cycle;
    static rcsc::Vector2D last_block_pos;
};


#endif
