// -*-c++-*-

/*!
  \file strategy.cpp
  \brief team strategh Source File
*/

/*
 *Copyright:

 Cyrus2D
 Modified by Omid Amini, Nader Zare

 Gliders2d
 Modified by Mikhail Prokopenko, Peter Wang

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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "strategy.h"

#include "soccer_role.h"

#include "setting.h"

#ifndef USE_GENERIC_FACTORY
#include "role_sample.h"

#include "role_center_back.h"
#include "role_center_forward.h"
#include "role_defensive_half.h"
#include "role_goalie.h"
#include "role_offensive_half.h"
#include "role_side_back.h"
#include "role_side_forward.h"
#include "role_side_half.h"

#include "role_keepaway_keeper.h"
#include "role_keepaway_taker.h"
#endif

#include <rcsc/player/intercept_table.h>
#include <rcsc/player/world_model.h>
#include <rcsc/geom/voronoi_diagram.h>

#include <rcsc/formation/formation_parser.h>
#include <rcsc/common/logger.h>
#include <rcsc/common/server_param.h>
#include <rcsc/param/cmd_line_parser.h>
#include <rcsc/param/param_map.h>
#include <rcsc/game_mode.h>

#include <iostream>

using namespace rcsc;

const std::string Strategy::BEFORE_KICK_OFF_CONF = "defense-formation.conf";
const std::string Strategy::NORMAL_FORMATION_CONF = "defense-formation.conf";
const std::string Strategy::DEFENSE_FORMATION_CONF = "defense-formation.conf";
const std::string Strategy::OFFENSE_FORMATION_CONF = "offense-formation.conf";
const std::string Strategy::GOAL_KICK_OPP_FORMATION_CONF = "defense-formation.conf";
const std::string Strategy::GOAL_KICK_OUR_FORMATION_CONF = "offense-formation.conf";
const std::string Strategy::GOALIE_CATCH_OPP_FORMATION_CONF = "defense-formation.conf";
const std::string Strategy::GOALIE_CATCH_OUR_FORMATION_CONF = "offense-formation.conf";
const std::string Strategy::KICKIN_OUR_FORMATION_CONF = "offense-formation.conf";
const std::string Strategy::SETPLAY_OPP_FORMATION_CONF = "defense-formation.conf";
const std::string Strategy::SETPLAY_OUR_FORMATION_CONF = "offense-formation.conf";
const std::string Strategy::INDIRECT_FREEKICK_OPP_FORMATION_CONF = "defense-formation.conf";
const std::string Strategy::INDIRECT_FREEKICK_OUR_FORMATION_CONF = "offense-formation.conf";

/*-------------------------------------------------------------------*/
/*!

 */
namespace
{
    struct MyCompare
    {

        const Vector2D pos_;

        MyCompare(const Vector2D &pos)
            : pos_(pos)
        {
        }

        bool operator()(const Vector2D &lhs,
                        const Vector2D &rhs) const
        {
            return (lhs - pos_).length() < (rhs - pos_).length();
        }
    };
}

/*-------------------------------------------------------------------*/
/*!

 */
Strategy::Strategy()
    : M_goalie_unum(Unum_Unknown),
      M_current_situation(Normal_Situation),
      M_role_number(11, 0),
      M_position_types(11, Position_Center),
      M_positions(11)
{
#ifndef USE_GENERIC_FACTORY
    //
    // roles
    //

    M_role_factory[RoleSample::name()] = &RoleSample::create;

    M_role_factory[RoleGoalie::name()] = &RoleGoalie::create;
    M_role_factory[RoleCenterBack::name()] = &RoleCenterBack::create;
    M_role_factory[RoleSideBack::name()] = &RoleSideBack::create;
    M_role_factory[RoleDefensiveHalf::name()] = &RoleDefensiveHalf::create;
    M_role_factory[RoleOffensiveHalf::name()] = &RoleOffensiveHalf::create;
    M_role_factory[RoleSideHalf::name()] = &RoleSideHalf::create;
    M_role_factory[RoleSideForward::name()] = &RoleSideForward::create;
    M_role_factory[RoleCenterForward::name()] = &RoleCenterForward::create;

    // keepaway
    M_role_factory[RoleKeepawayKeeper::name()] = &RoleKeepawayKeeper::create;
    M_role_factory[RoleKeepawayTaker::name()] = &RoleKeepawayTaker::create;

#endif

    for (size_t i = 0; i < M_role_number.size(); ++i)
    {
        M_role_number[i] = i + 1;
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
Strategy &
Strategy::instance()
{
    static Strategy s_instance;
    return s_instance;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool Strategy::init(CmdLineParser &cmd_parser)
{
    ParamMap param_map("HELIOS_base options");

    // std::string fconf;
    // param_map.add()
    //    ( "fconf", "", &fconf, "another formation file." );

    //
    //
    //

    if (cmd_parser.count("help") > 0)
    {
        param_map.printHelp(std::cout);
        return false;
    }

    //
    //
    //

    cmd_parser.parse(param_map);

    return true;
}

/*-------------------------------------------------------------------*/
/*!

 */
bool Strategy::read(const std::string &formation_dir)
{
    static bool s_initialized = false;

    if (s_initialized)
    {
        std::cerr << __FILE__ << ' ' << __LINE__ << ": already initialized."
                  << std::endl;
        return false;
    }

    std::string configpath = formation_dir;
    if (!configpath.empty() && configpath[configpath.length() - 1] != '/')
    {
        configpath += '/';
    }

    // before kick off
    M_before_kick_off_formation = createFormation(configpath + BEFORE_KICK_OFF_CONF);
    if (!M_before_kick_off_formation)
    {
        std::cerr << "Failed to read before_kick_off formation" << std::endl;
        return false;
    }

    ///////////////////////////////////////////////////////////
    M_normal_formation = createFormation(configpath + NORMAL_FORMATION_CONF);
    if (!M_normal_formation)
    {
        std::cerr << "Failed to read normal formation" << std::endl;
        return false;
    }

    M_defense_formation = createFormation(configpath + DEFENSE_FORMATION_CONF);
    if (!M_defense_formation)
    {
        std::cerr << "Failed to read defense formation" << std::endl;
        return false;
    }

    M_offense_formation = createFormation(configpath + OFFENSE_FORMATION_CONF);
    if (!M_offense_formation)
    {
        std::cerr << "Failed to read offense formation" << std::endl;
        return false;
    }

    M_goal_kick_opp_formation = createFormation(configpath + GOAL_KICK_OPP_FORMATION_CONF);
    if (!M_goal_kick_opp_formation)
    {
        return false;
    }

    M_goal_kick_our_formation = createFormation(configpath + GOAL_KICK_OUR_FORMATION_CONF);
    if (!M_goal_kick_our_formation)
    {
        return false;
    }

    M_goalie_catch_opp_formation = createFormation(configpath + GOALIE_CATCH_OPP_FORMATION_CONF);
    if (!M_goalie_catch_opp_formation)
    {
        return false;
    }

    M_goalie_catch_our_formation = createFormation(configpath + GOALIE_CATCH_OUR_FORMATION_CONF);
    if (!M_goalie_catch_our_formation)
    {
        return false;
    }

    M_kickin_our_formation = createFormation(configpath + KICKIN_OUR_FORMATION_CONF);
    if (!M_kickin_our_formation)
    {
        std::cerr << "Failed to read kickin our formation" << std::endl;
        return false;
    }

    M_setplay_opp_formation = createFormation(configpath + SETPLAY_OPP_FORMATION_CONF);
    if (!M_setplay_opp_formation)
    {
        std::cerr << "Failed to read setplay opp formation" << std::endl;
        return false;
    }

    M_setplay_our_formation = createFormation(configpath + SETPLAY_OUR_FORMATION_CONF);
    if (!M_setplay_our_formation)
    {
        std::cerr << "Failed to read setplay our formation" << std::endl;
        return false;
    }

    M_indirect_freekick_opp_formation = createFormation(configpath + INDIRECT_FREEKICK_OPP_FORMATION_CONF);
    if (!M_indirect_freekick_opp_formation)
    {
        std::cerr << "Failed to read indirect freekick opp formation" << std::endl;
        return false;
    }

    M_indirect_freekick_our_formation = createFormation(configpath + INDIRECT_FREEKICK_OUR_FORMATION_CONF);
    if (!M_indirect_freekick_our_formation)
    {
        std::cerr << "Failed to read indirect freekick our formation" << std::endl;
        return false;
    }

    s_initialized = true;
    return true;
}

bool Strategy::readLoser(const std::string &formation_dir)
{
    static bool s_initialized = false;

    if (s_initialized)
    {
        std::cerr << __FILE__ << ' ' << __LINE__ << ": already initialized."
                  << std::endl;
        return false;
    }

    std::string configpath = formation_dir;
    if (!configpath.empty() && configpath[configpath.length() - 1] != '/')
    {
        configpath += '/';
    }

    // before kick off
    M_before_kick_off_formation_loser = createFormation(configpath + BEFORE_KICK_OFF_CONF);
    if (!M_before_kick_off_formation_loser)
    {
        std::cerr << "Failed to read before_kick_off formation" << std::endl;
        return false;
    }

    ///////////////////////////////////////////////////////////
    M_normal_formation_loser = createFormation(configpath + NORMAL_FORMATION_CONF);
    if (!M_normal_formation_loser)
    {
        std::cerr << "Failed to read normal formation" << std::endl;
        return false;
    }

    M_defense_formation_loser = createFormation(configpath + DEFENSE_FORMATION_CONF);
    if (!M_defense_formation_loser)
    {
        std::cerr << "Failed to read defense formation" << std::endl;
        return false;
    }

    M_offense_formation_loser = createFormation(configpath + OFFENSE_FORMATION_CONF);
    if (!M_offense_formation_loser)
    {
        std::cerr << "Failed to read offense formation" << std::endl;
        return false;
    }

    M_goal_kick_opp_formation_loser = createFormation(configpath + GOAL_KICK_OPP_FORMATION_CONF);
    if (!M_goal_kick_opp_formation_loser)
    {
        return false;
    }

    M_goal_kick_our_formation_loser = createFormation(configpath + GOAL_KICK_OUR_FORMATION_CONF);
    if (!M_goal_kick_our_formation_loser)
    {
        return false;
    }

    M_goalie_catch_opp_formation_loser = createFormation(configpath + GOALIE_CATCH_OPP_FORMATION_CONF);
    if (!M_goalie_catch_opp_formation_loser)
    {
        return false;
    }

    M_goalie_catch_our_formation_loser = createFormation(configpath + GOALIE_CATCH_OUR_FORMATION_CONF);
    if (!M_goalie_catch_our_formation_loser)
    {
        return false;
    }

    M_kickin_our_formation_loser = createFormation(configpath + KICKIN_OUR_FORMATION_CONF);
    if (!M_kickin_our_formation_loser)
    {
        std::cerr << "Failed to read kickin our formation" << std::endl;
        return false;
    }

    M_setplay_opp_formation_loser = createFormation(configpath + SETPLAY_OPP_FORMATION_CONF);
    if (!M_setplay_opp_formation_loser)
    {
        std::cerr << "Failed to read setplay opp formation" << std::endl;
        return false;
    }

    M_setplay_our_formation_loser = createFormation(configpath + SETPLAY_OUR_FORMATION_CONF);
    if (!M_setplay_our_formation_loser)
    {
        std::cerr << "Failed to read setplay our formation" << std::endl;
        return false;
    }

    M_indirect_freekick_opp_formation_loser = createFormation(configpath + INDIRECT_FREEKICK_OPP_FORMATION_CONF);
    if (!M_indirect_freekick_opp_formation_loser)
    {
        std::cerr << "Failed to read indirect freekick opp formation" << std::endl;
        return false;
    }

    M_indirect_freekick_our_formation_loser = createFormation(configpath + INDIRECT_FREEKICK_OUR_FORMATION_CONF);
    if (!M_indirect_freekick_our_formation_loser)
    {
        std::cerr << "Failed to read indirect freekick our formation" << std::endl;
        return false;
    }

    s_initialized = true;
    return true;
}

bool Strategy::readWinner(const std::string &formation_dir)
{
    static bool s_initialized = false;

    if (s_initialized)
    {
        std::cerr << __FILE__ << ' ' << __LINE__ << ": already initialized."
                  << std::endl;
        return false;
    }

    std::string configpath = formation_dir;
    if (!configpath.empty() && configpath[configpath.length() - 1] != '/')
    {
        configpath += '/';
    }

    // before kick off
    M_before_kick_off_formation_winner = createFormation(configpath + BEFORE_KICK_OFF_CONF);
    if (!M_before_kick_off_formation_winner)
    {
        std::cerr << "Failed to read before_kick_off formation" << std::endl;
        return false;
    }

    ///////////////////////////////////////////////////////////
    M_normal_formation_winner = createFormation(configpath + NORMAL_FORMATION_CONF);
    if (!M_normal_formation_winner)
    {
        std::cerr << "Failed to read normal formation" << std::endl;
        return false;
    }

    M_defense_formation_winner = createFormation(configpath + DEFENSE_FORMATION_CONF);
    if (!M_defense_formation_winner)
    {
        std::cerr << "Failed to read defense formation" << std::endl;
        return false;
    }

    M_offense_formation_winner = createFormation(configpath + OFFENSE_FORMATION_CONF);
    if (!M_offense_formation_winner)
    {
        std::cerr << "Failed to read offense formation" << std::endl;
        return false;
    }

    M_goal_kick_opp_formation_winner = createFormation(configpath + GOAL_KICK_OPP_FORMATION_CONF);
    if (!M_goal_kick_opp_formation_winner)
    {
        return false;
    }

    M_goal_kick_our_formation_winner = createFormation(configpath + GOAL_KICK_OUR_FORMATION_CONF);
    if (!M_goal_kick_our_formation_winner)
    {
        return false;
    }

    M_goalie_catch_opp_formation_winner = createFormation(configpath + GOALIE_CATCH_OPP_FORMATION_CONF);
    if (!M_goalie_catch_opp_formation_winner)
    {
        return false;
    }

    M_goalie_catch_our_formation_winner = createFormation(configpath + GOALIE_CATCH_OUR_FORMATION_CONF);
    if (!M_goalie_catch_our_formation_winner)
    {
        return false;
    }

    M_kickin_our_formation_winner = createFormation(configpath + KICKIN_OUR_FORMATION_CONF);
    if (!M_kickin_our_formation_winner)
    {
        std::cerr << "Failed to read kickin our formation" << std::endl;
        return false;
    }

    M_setplay_opp_formation_winner = createFormation(configpath + SETPLAY_OPP_FORMATION_CONF);
    if (!M_setplay_opp_formation_winner)
    {
        std::cerr << "Failed to read setplay opp formation" << std::endl;
        return false;
    }

    M_setplay_our_formation_winner = createFormation(configpath + SETPLAY_OUR_FORMATION_CONF);
    if (!M_setplay_our_formation_winner)
    {
        std::cerr << "Failed to read setplay our formation" << std::endl;
        return false;
    }

    M_indirect_freekick_opp_formation_winner = createFormation(configpath + INDIRECT_FREEKICK_OPP_FORMATION_CONF);
    if (!M_indirect_freekick_opp_formation_winner)
    {
        std::cerr << "Failed to read indirect freekick opp formation" << std::endl;
        return false;
    }

    M_indirect_freekick_our_formation_winner = createFormation(configpath + INDIRECT_FREEKICK_OUR_FORMATION_CONF);
    if (!M_indirect_freekick_our_formation_winner)
    {
        std::cerr << "Failed to read indirect freekick our formation" << std::endl;
        return false;
    }

    s_initialized = true;
    return true;
}
/*-------------------------------------------------------------------*/
/*!

 */
Formation::Ptr
Strategy::createFormation(const std::string &filepath)
{
    Formation::Ptr f = FormationParser::parse(filepath);

    if (!f)
    {
        std::cerr << "(Strategy::createFormation) Could not create a formation from " << filepath << std::endl;
        return Formation::Ptr();
    }

    //
    // check role names
    //
    for (int unum = 1; unum <= 11; ++unum)
    {
        const std::string role_name = f->roleName(unum);
        if (role_name == "Savior" || role_name == "Goalie")
        {
            if (M_goalie_unum == Unum_Unknown)
            {
                M_goalie_unum = unum;
            }

            if (M_goalie_unum != unum)
            {
                std::cerr << __FILE__ << ':' << __LINE__ << ':'
                          << " ***ERROR*** Illegal goalie's uniform number"
                          << " read unum=" << unum
                          << " expected=" << M_goalie_unum
                          << std::endl;
                f.reset();
                return f;
            }
        }

#ifdef USE_GENERIC_FACTORY
        SoccerRole::Ptr role = SoccerRole::create(role_name);
        if (!role)
        {
            std::cerr << __FILE__ << ':' << __LINE__ << ':'
                      << " ***ERROR*** Unsupported role name ["
                      << role_name << "] is appered in ["
                      << filepath << "]" << std::endl;
            f.reset();
            return f;
        }
#else
        if (M_role_factory.find(role_name) == M_role_factory.end())
        {
            std::cerr << __FILE__ << ':' << __LINE__ << ':'
                      << " ***ERROR*** Unsupported role name ["
                      << role_name << "] is appered in ["
                      << filepath << "]" << std::endl;
            f.reset();
            return f;
        }
#endif
    }

    return f;
}

/*-------------------------------------------------------------------*/
/*!

 */
void Strategy::update(const WorldModel &wm)
{
    static GameTime s_update_time(-1, 0);

    if (s_update_time == wm.time())
    {
        return;
    }
    s_update_time = wm.time();

    updateSituation(wm);
    updatePosition(wm);
}

/*-------------------------------------------------------------------*/
/*!

 */
void Strategy::exchangeRole(const int unum0,
                            const int unum1)
{
    if (unum0 < 1 || 11 < unum0 || unum1 < 1 || 11 < unum1)
    {
        std::cerr << __FILE__ << ':' << __LINE__ << ':'
                  << "(exchangeRole) Illegal uniform number. "
                  << unum0 << ' ' << unum1
                  << std::endl;
        dlog.addText(Logger::TEAM,
                     __FILE__ ":(exchangeRole) Illegal unum. %d %d",
                     unum0, unum1);
        return;
    }

    if (unum0 == unum1)
    {
        std::cerr << __FILE__ << ':' << __LINE__ << ':'
                  << "(exchangeRole) same uniform number. "
                  << unum0 << ' ' << unum1
                  << std::endl;
        dlog.addText(Logger::TEAM,
                     __FILE__ ":(exchangeRole) same unum. %d %d",
                     unum0, unum1);
        return;
    }

    int role0 = M_role_number[unum0 - 1];
    int role1 = M_role_number[unum1 - 1];

    dlog.addText(Logger::TEAM,
                 __FILE__ ":(exchangeRole) unum=%d(role=%d) <-> unum=%d(role=%d)",
                 unum0, role0,
                 unum1, role1);

    M_role_number[unum0 - 1] = role1;
    M_role_number[unum1 - 1] = role0;
}

/*-------------------------------------------------------------------*/
/*!

*/
bool Strategy::isMarkerType(const int unum) const
{
    int number = roleNumber(unum);

    if (number == 2 || number == 3 || number == 4 || number == 5)
    {
        return true;
    }

    return false;
}

/*-------------------------------------------------------------------*/
/*!

 */
SoccerRole::Ptr
Strategy::createRole(const int unum,
                     const WorldModel &world) const
{
    const int number = roleNumber(unum);

    SoccerRole::Ptr role;

    if (number < 1 || 11 < number)
    {
        std::cerr << __FILE__ << ": " << __LINE__
                  << " ***ERROR*** Invalid player number " << number
                  << std::endl;
        return role;
    }

    Formation::Ptr f = getFormation(world);
    if (!f)
    {
        std::cerr << __FILE__ << ": " << __LINE__
                  << " ***ERROR*** faled to create role. Null formation" << std::endl;
        return role;
    }

    const std::string role_name = f->roleName(number);

#ifdef USE_GENERIC_FACTORY
    role = SoccerRole::create(role_name);
#else
    RoleFactory::const_iterator factory = M_role_factory.find(role_name);
    if (factory != M_role_factory.end())
    {
        role = factory->second();
    }
#endif

    if (!role)
    {
        std::cerr << __FILE__ << ": " << __LINE__
                  << " ***ERROR*** unsupported role name ["
                  << role_name << "]"
                  << std::endl;
    }
    return role;
}

/*-------------------------------------------------------------------*/
/*!

 */
void Strategy::updateSituation(const WorldModel &wm)
{
    M_current_situation = Normal_Situation;

    if (wm.gameMode().type() != GameMode::PlayOn)
    {
        if (wm.gameMode().isPenaltyKickMode())
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": Situation PenaltyKick");
            M_current_situation = PenaltyKick_Situation;
        }
        else if (wm.gameMode().isPenaltyKickMode())
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": Situation OurSetPlay");
            M_current_situation = OurSetPlay_Situation;
        }
        else
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": Situation OppSetPlay");
            M_current_situation = OppSetPlay_Situation;
        }
        return;
    }

    int self_min = wm.interceptTable().selfStep();
    int mate_min = wm.interceptTable().teammateStep();
    int opp_min = wm.interceptTable().opponentStep();
    int our_min = std::min(self_min, mate_min);

    if (opp_min <= our_min)
    {
        dlog.addText(Logger::TEAM,
                     __FILE__ ": Situation Defense");
        M_current_situation = Defense_Situation;
        return;
    }
    else
    {
        dlog.addText(Logger::TEAM,
                     __FILE__ ": Situation Offense");
        M_current_situation = Offense_Situation;
        return;
    }

    dlog.addText(Logger::TEAM,
                 __FILE__ ": Situation Normal");
}

/*-------------------------------------------------------------------*/
/*!

 */
void Strategy::updatePosition(const WorldModel &wm)
{
    static GameTime s_update_time(0, 0);
    if (s_update_time == wm.time())
    {
        return;
    }
    s_update_time = wm.time();

    Formation::Ptr f = getFormation(wm);
    if (!f)
    {
        std::cerr << wm.teamName() << ':' << wm.self().unum() << ": "
                  << wm.time()
                  << " ***ERROR*** could not get the current formation" << std::endl;
        return;
    }


    


    int ball_step = 0;
    if (wm.gameMode().type() == GameMode::PlayOn || wm.gameMode().type() == GameMode::GoalKick_)
    {
        ball_step = std::min(1000, wm.interceptTable().teammateStep());
        ball_step = std::min(ball_step, wm.interceptTable().opponentStep());
        ball_step = std::min(ball_step, wm.interceptTable().selfStep());
    }

    Vector2D ball_pos = wm.ball().inertiaPoint(ball_step);

    dlog.addText(Logger::TEAM,
                 __FILE__ ": HOME POSITION: ball pos=(%.1f %.1f) step=%d",
                 ball_pos.x, ball_pos.y,
                 ball_step);

    M_positions.clear();
    f->getPositions(ball_pos, M_positions);

    // G2d: various states
    bool indFK = false;
    if ((wm.gameMode().type() == GameMode::BackPass_ && wm.gameMode().side() == wm.theirSide()) || (wm.gameMode().type() == GameMode::IndFreeKick_ && wm.gameMode().side() == wm.ourSide()) || (wm.gameMode().type() == GameMode::FoulCharge_ && wm.gameMode().side() == wm.theirSide()) || (wm.gameMode().type() == GameMode::FoulPush_ && wm.gameMode().side() == wm.theirSide()))
        indFK = true;

    bool dirFK = false;
    if (
        (wm.gameMode().type() == GameMode::FreeKick_ && wm.gameMode().side() == wm.ourSide()) || (wm.gameMode().type() == GameMode::FoulCharge_ && wm.gameMode().side() == wm.theirSide()) || (wm.gameMode().type() == GameMode::FoulPush_ && wm.gameMode().side() == wm.theirSide()))
        dirFK = true;

    bool cornerK = false;
    if (
        (wm.gameMode().type() == GameMode::CornerKick_ && wm.gameMode().side() == wm.ourSide()))
        cornerK = true;

    bool kickin = false;
    if (
        (wm.gameMode().type() == GameMode::KickIn_ && wm.gameMode().side() == wm.ourSide()))
        kickin = true;

    // C2D: Helios 18 Tune removed -> replace with BNN
    // bool heliosbase = false;
    // bool helios2018 = false;
    // if (wm.opponentTeamName().find("HELIOS_base") != std::string::npos)
    // 	heliosbase = true;
    // else if (wm.opponentTeamName().find("HELIOS2018") != std::string::npos)
    // 	helios2018 = true;

    if (ServerParam::i().useOffside())
    {
        double max_x = wm.offsideLineX();
        if (ServerParam::i().kickoffOffside() && (wm.gameMode().type() == GameMode::BeforeKickOff || wm.gameMode().type() == GameMode::AfterGoal_))
        {
            max_x = 0.0;
        }
        else
        {
            int mate_step = wm.interceptTable().teammateStep();
            if (mate_step < 50)
            {
                Vector2D trap_pos = wm.ball().inertiaPoint(mate_step);
                if (trap_pos.x > max_x)
                    max_x = trap_pos.x;
            }

            max_x -= 1.0;
        }
        // C2d: PlayerPtrCont::const_iterator replace with auto
        // G2d: Voronoi diagram
        bool newvel = false;

        VoronoiDiagram vd;
        // const ServerParam & SP = ServerParam::i();

        std::vector<Vector2D> vd_cont;
        std::vector<Vector2D> NOL_cont; // Near Offside Line
        std::vector<Vector2D> NOL_tmp;  // Near Offside Line tmp

        std::vector<Vector2D> OffsideSegm_cont;
        std::vector<Vector2D> OffsideSegm_tmpcont;

        Vector2D y1(wm.offsideLineX(), -34.0);
        Vector2D y2(wm.offsideLineX(), 34.0);

        if (wm.ball().pos().x > 25.0)
        {
            if (wm.ball().pos().y < 0.0)
                y2.y = 20.0;
            if (wm.ball().pos().y > 0.0)
                y1.y = -20.0;
        }

        if (wm.ball().pos().x > 36.0)
        {
            if (wm.ball().pos().y < 0.0)
                y2.y = 8.0;
            if (wm.ball().pos().y > 0.0)
                y1.y = -8.0;
        }

        if (wm.ball().pos().x > 49.0)
        {
            y1.x = y1.x - 4.0;
            y2.x = y2.x - 4.0;
        }

        for (auto o = wm.opponentsFromSelf().begin();
             o != wm.opponentsFromSelf().end();
             ++o)
        {
            if (newvel)
                vd.addPoint((*o)->pos() + (*o)->vel());
            else
                vd.addPoint((*o)->pos());
        }

        if (y1.x < 37.0)
        {
            vd.addPoint(y1);
            vd.addPoint(y2);
        }

        vd.compute();

        Line2D offsideLine(y1, y2);

        for (VoronoiDiagram::Segment2DCont::const_iterator p = vd.segments().begin(),
                                                           end = vd.segments().end();
             p != end;
             ++p)
        {
            Vector2D si = (*p).intersection(offsideLine);
            if (si.isValid() && fabs(si.y) < 34.0 && fabs(si.x) < 52.5)
            {
                OffsideSegm_tmpcont.push_back(si);
            }
        }

        std::sort(OffsideSegm_tmpcont.begin(), OffsideSegm_tmpcont.end(), MyCompare(wm.ball().pos()));

        double prevY = -1000.0;

        for (std::vector<Vector2D>::iterator p = OffsideSegm_tmpcont.begin(),
                                             end = OffsideSegm_tmpcont.end();
             p != end;
             ++p)
        {
            if (p == OffsideSegm_tmpcont.begin())
            {
                OffsideSegm_cont.push_back((*p));
                prevY = (*p).y;
                continue;
            }

            if (fabs((*p).y - prevY) > 2.0)
            {
                prevY = (*p).y;
                OffsideSegm_cont.push_back((*p));
            }
        }

        // int n_points = 0;

        for (VoronoiDiagram::Vector2DCont::const_iterator p = vd.vertices().begin(),
                                                          end = vd.vertices().end();
             p != end;
             ++p)
        {
            if ((*p).x < wm.offsideLineX() - 5.0 && (*p).x > 0.0)
            {
                vd_cont.push_back((*p));
            }
        }

        // end of Voronoi

        // G2d: assign players to Voronoi points

        Vector2D rank(y1.x, -34.0);

        Vector2D first_pt(-100.0, -100.0);
        Vector2D mid_pt(-100.0, -100.0);
        Vector2D third_pt(-100.0, -100.0);

        if (wm.ball().pos().y > 0.0)
            rank.y = 34.0;

        std::sort(OffsideSegm_cont.begin(), OffsideSegm_cont.end(), MyCompare(rank));

        // int shift = 0;

        // if (OffsideSegm_cont.size() > 4)
        // shift = 1;

        if (OffsideSegm_cont.size() > 0)
            first_pt = OffsideSegm_cont[0];

        if (OffsideSegm_cont.size() > 1)
            third_pt = OffsideSegm_cont[OffsideSegm_cont.size() - 1];

        if (OffsideSegm_cont.size() > 2)
            mid_pt = OffsideSegm_cont[2];

        int first_unum = -1;
        int sec_unum = -1;
        int third_unum = -1;

        if (wm.ball().pos().y <= 0.0)
        {
            double tmp = 100.0;
            for (int ch = 9; ch <= 11; ch++)
            {
                if (wm.ourPlayer(ch) == NULL)
                    continue;

                if (wm.ourPlayer(ch)->pos().y < tmp)
                {
                    tmp = wm.ourPlayer(ch)->pos().y;
                    first_unum = ch;
                }
            }

            tmp = 100.0;

            for (int ch = 9; ch <= 11; ch++)
            {
                if (wm.ourPlayer(ch) == NULL)
                    continue;

                if (ch == first_unum)
                    continue;

                if (wm.ourPlayer(ch)->pos().y < tmp)
                {
                    tmp = wm.ourPlayer(ch)->pos().y;
                    sec_unum = ch;
                }
            }

            for (int ch = 9; ch <= 11; ch++)
            {
                if (ch == first_unum || ch == sec_unum)
                    continue;

                if (first_unum > 0 && sec_unum > 0)
                    third_unum = ch;
            }
        }

        if (wm.ball().pos().y > 0.0)
        {
            double tmp = -100.0;
            for (int ch = 9; ch <= 11; ch++)
            {
                if (wm.ourPlayer(ch) == NULL)
                    continue;

                if (wm.ourPlayer(ch)->pos().y > tmp)
                {
                    tmp = wm.ourPlayer(ch)->pos().y;
                    first_unum = ch;
                }
            }

            tmp = -100.0;

            for (int ch = 9; ch <= 11; ch++)
            {
                if (wm.ourPlayer(ch) == NULL)
                    continue;

                if (ch == first_unum)
                    continue;

                if (wm.ourPlayer(ch)->pos().y > tmp)
                {
                    tmp = wm.ourPlayer(ch)->pos().y;
                    sec_unum = ch;
                }
            }

            for (int ch = 9; ch <= 11; ch++)
            {
                if (ch == first_unum || ch == sec_unum)
                    continue;

                if (first_unum > 0 && sec_unum > 0)
                    third_unum = ch;
            }
        }

        bool first = false;
        bool sec = false;
        bool third = false;

        double voron_depth = 42.0;
        // C2D: Helios 18 Tune removed -> replace with BNN

        // if (helios2018)
        // 	voron_depth = 36.0;
        // if (heliosbase)
        // 	voron_depth = 0.2;

        if (wm.gameMode().type() == GameMode::PlayOn && wm.ball().pos().x > voron_depth)
        {
            if (first_pt.x > -1.0 && first_unum > 0)
            {
                first = true;
                M_positions[first_unum - 1] = first_pt;
            }
            if (mid_pt.x > -1.0 && sec_unum > 0)
            {
                sec = true;
                M_positions[sec_unum - 1] = mid_pt;
            }
            if (third_pt.x > -1.0 && third_unum > 0)
            {
                third = true;
                M_positions[third_unum - 1] = third_pt;
            }
        }
        // end of assignment

        for (int unum = 1; unum <= 11; ++unum)
        {
            // G2d: skip assigned players

            if (unum == first_unum && first)
                continue;

            if (unum == sec_unum && sec)
                continue;

            if (unum == third_unum && third)
                continue;

            if (M_positions[unum - 1].x > max_x)
            {
                dlog.addText(Logger::TEAM,
                             "____ %d offside. home_pos_x %.2f -> %.2f",
                             unum,
                             M_positions[unum - 1].x, max_x);
                M_positions[unum - 1].x = max_x;
            }
        }
    }

    int self_min = wm.interceptTable().selfStep();
    int mate_min = wm.interceptTable().teammateStep();
    int opp_min = wm.interceptTable().opponentStep();

    const int our_min = std::min(self_min, mate_min);

    // G2d : wing tactic
    double wing_x = -15.0;
    double wing_y = 7.0;
    double wing_depth = 5.0;
    double wing_limit = 39.0;

    // C2D: Tune removed
    // if (mt || helios2018)
    // {
    //     wing_depth = 10.0;
    //     wing_y = 17.0;
    // }

    if (our_min < opp_min)
        if (wm.ball().pos().x > wing_x)
            if (wm.ball().pos().x < wing_limit)
                if (fabs(wm.ball().pos().y) > wing_y)
                    if (!indFK && !dirFK && !cornerK && !kickin)
                    {
                        M_positions[9 - 1].x = wm.offsideLineX() + wm.ball().vel().x;
                        M_positions[10 - 1].x = wm.offsideLineX() + wm.ball().vel().x;
                        M_positions[11 - 1].x = wm.offsideLineX() + wm.ball().vel().x;

                        if (wm.ball().pos().y > 0)
                        {
                            M_positions[9 - 1].y = 15.0;
                            M_positions[11 - 1].y = 22.5;
                            M_positions[10 - 1].y = 30.0;
                        }
                        else
                        {
                            M_positions[9 - 1].y = -30.0;
                            M_positions[11 - 1].y = -22.5;
                            M_positions[10 - 1].y = -15.0;
                        }

                        double midX = wm.offsideLineX() - wing_depth;

                        M_positions[6 - 1].x = midX;
                        M_positions[7 - 1].x = midX;
                        M_positions[8 - 1].x = midX;

                        M_positions[7 - 1].y = M_positions[9 - 1].y;
                        M_positions[6 - 1].y = M_positions[11 - 1].y;
                        M_positions[8 - 1].y = M_positions[10 - 1].y;
                    }

    M_position_types.clear();
    for (int unum = 1; unum <= 11; ++unum)
    {
        PositionType type = Position_Center;

        const RoleType role_type = f->roleType(unum);
        if (role_type.side() == RoleType::Left)
        {
            type = Position_Left;
        }
        else if (role_type.side() == RoleType::Right)
        {
            type = Position_Right;
        }

        M_position_types.push_back(type);

        dlog.addText(Logger::TEAM,
                     "__ %d home pos (%.2f %.2f) type=%d",
                     unum,
                     M_positions[unum - 1].x, M_positions[unum - 1].y,
                     type);
        dlog.addCircle(Logger::TEAM,
                       M_positions[unum - 1], 0.5,
                       "#000000");
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
PositionType
Strategy::getPositionType(const int unum) const
{
    const int number = roleNumber(unum);

    if (number < 1 || 11 < number)
    {
        std::cerr << __FILE__ << ' ' << __LINE__
                  << ": Illegal number : " << number
                  << std::endl;
        return Position_Center;
    }

    try
    {
        return M_position_types.at(number - 1);
    }
    catch (std::exception &e)
    {
        std::cerr << __FILE__ << ':' << __LINE__ << ':'
                  << " Exception caught! " << e.what()
                  << std::endl;
        return Position_Center;
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
Vector2D
Strategy::getPosition(const int unum) const
{
    const int number = roleNumber(unum);

    if (number < 1 || 11 < number)
    {
        std::cerr << __FILE__ << ' ' << __LINE__
                  << ": Illegal number : " << number
                  << std::endl;
        return Vector2D::INVALIDATED;
    }

    try
    {
        Vector2D pos = M_positions.at(number - 1);
        Setting &setting = Setting::i();
        // if aggressive mode just move all players to left
        double new_pos = pos.x + ((pos.x/10) * ((0.5 - setting.aggressiveness)/0.5));
        if (new_pos > -52.5)
            pos.x = new_pos;
        
        return pos;
    }
    catch (std::exception &e)
    {
        std::cerr << __FILE__ << ':' << __LINE__ << ':'
                  << " Exception caught! " << e.what()
                  << std::endl;
        return Vector2D::INVALIDATED;
    }
}

/*-------------------------------------------------------------------*/
/*!

 */
Formation::Ptr
Strategy::getFormation(const WorldModel &wm) const
{
    int our_score = wm.ourSide() == SideID::LEFT
                        ? wm.gameMode().scoreLeft()
                        : wm.gameMode().scoreRight();
    int opp_score = wm.ourSide() == SideID::LEFT
                        ? wm.gameMode().scoreRight()
                        : wm.gameMode().scoreLeft();

    if (our_score > opp_score)
        return getFormation_winner(wm);
    if (our_score < opp_score)
        return getFormation_loser(wm);
    return getFormation_normal(wm);
}

Formation::Ptr
Strategy::getFormation_normal(const WorldModel &wm) const
{
    //
    // play on
    //
    if (wm.gameMode().type() == GameMode::PlayOn)
    {
        switch (M_current_situation)
        {
        case Defense_Situation:
            return M_defense_formation;
        case Offense_Situation:
            return M_offense_formation;
        default:
            break;
        }
        return M_normal_formation;
    }

    //
    // kick in, corner kick
    //
    if (wm.gameMode().type() == GameMode::KickIn_ || wm.gameMode().type() == GameMode::CornerKick_)
    {
        if (wm.ourSide() == wm.gameMode().side())
        {
            // our kick-in or corner-kick
            return M_kickin_our_formation;
        }
        else
        {
            return M_setplay_opp_formation;
        }
    }

    //
    // our indirect free kick
    //
    if ((wm.gameMode().type() == GameMode::BackPass_ && wm.gameMode().side() == wm.theirSide()) || (wm.gameMode().type() == GameMode::IndFreeKick_ && wm.gameMode().side() == wm.ourSide()))
    {
        return M_indirect_freekick_our_formation;
    }

    //
    // opponent indirect free kick
    //
    if ((wm.gameMode().type() == GameMode::BackPass_ && wm.gameMode().side() == wm.ourSide()) || (wm.gameMode().type() == GameMode::IndFreeKick_ && wm.gameMode().side() == wm.theirSide()))
    {
        return M_indirect_freekick_opp_formation;
    }

    //
    // after foul
    //
    if (wm.gameMode().type() == GameMode::FoulCharge_ || wm.gameMode().type() == GameMode::FoulPush_)
    {
        if (wm.gameMode().side() == wm.ourSide())
        {
            //
            // opponent (indirect) free kick
            //
            if (wm.ball().pos().x < ServerParam::i().ourPenaltyAreaLineX() + 1.0 && wm.ball().pos().absY() < ServerParam::i().penaltyAreaHalfWidth() + 1.0)
            {
                return M_indirect_freekick_opp_formation;
            }
            else
            {
                return M_setplay_opp_formation;
            }
        }
        else
        {
            //
            // our (indirect) free kick
            //
            if (wm.ball().pos().x > ServerParam::i().theirPenaltyAreaLineX() && wm.ball().pos().absY() < ServerParam::i().penaltyAreaHalfWidth())
            {
                return M_indirect_freekick_our_formation;
            }
            else
            {
                return M_setplay_our_formation;
            }
        }
    }

    //
    // goal kick
    //
    if (wm.gameMode().type() == GameMode::GoalKick_)
    {
        if (wm.gameMode().side() == wm.ourSide())
        {
            return M_goal_kick_our_formation;
        }
        else
        {
            return M_goal_kick_opp_formation;
        }
    }

    //
    // goalie catch
    //
    if (wm.gameMode().type() == GameMode::GoalieCatch_)
    {
        if (wm.gameMode().side() == wm.ourSide())
        {
            return M_goalie_catch_our_formation;
        }
        else
        {
            return M_goalie_catch_opp_formation;
        }
    }

    //
    // before kick off
    //
    if (wm.gameMode().type() == GameMode::BeforeKickOff || wm.gameMode().type() == GameMode::AfterGoal_)
    {
        return M_before_kick_off_formation;
    }

    //
    // other set play
    //
    if (wm.gameMode().isOurSetPlay(wm.ourSide()))
    {
        return M_setplay_our_formation;
    }

    if (wm.gameMode().type() != GameMode::PlayOn)
    {
        return M_setplay_opp_formation;
    }

    //
    // unknown
    //
    switch (M_current_situation)
    {
    case Defense_Situation:
        return M_defense_formation;
    case Offense_Situation:
        return M_offense_formation;
    default:
        break;
    }

    return M_normal_formation;
}

Formation::Ptr
Strategy::getFormation_loser(const WorldModel &wm) const
{
    //
    // play on
    //
    if (wm.gameMode().type() == GameMode::PlayOn)
    {
        switch (M_current_situation)
        {
        case Defense_Situation:
            return M_defense_formation_loser;
        case Offense_Situation:
            return M_offense_formation_loser;
        default:
            break;
        }
        return M_normal_formation_loser;
    }

    //
    // kick in, corner kick
    //
    if (wm.gameMode().type() == GameMode::KickIn_ || wm.gameMode().type() == GameMode::CornerKick_)
    {
        if (wm.ourSide() == wm.gameMode().side())
        {
            // our kick-in or corner-kick
            return M_kickin_our_formation_loser;
        }
        else
        {
            return M_setplay_opp_formation_loser;
        }
    }

    //
    // our indirect free kick
    //
    if ((wm.gameMode().type() == GameMode::BackPass_ && wm.gameMode().side() == wm.theirSide()) || (wm.gameMode().type() == GameMode::IndFreeKick_ && wm.gameMode().side() == wm.ourSide()))
    {
        return M_indirect_freekick_our_formation_loser;
    }

    //
    // opponent indirect free kick
    //
    if ((wm.gameMode().type() == GameMode::BackPass_ && wm.gameMode().side() == wm.ourSide()) || (wm.gameMode().type() == GameMode::IndFreeKick_ && wm.gameMode().side() == wm.theirSide()))
    {
        return M_indirect_freekick_opp_formation_loser;
    }

    //
    // after foul
    //
    if (wm.gameMode().type() == GameMode::FoulCharge_ || wm.gameMode().type() == GameMode::FoulPush_)
    {
        if (wm.gameMode().side() == wm.ourSide())
        {
            //
            // opponent (indirect) free kick
            //
            if (wm.ball().pos().x < ServerParam::i().ourPenaltyAreaLineX() + 1.0 && wm.ball().pos().absY() < ServerParam::i().penaltyAreaHalfWidth() + 1.0)
            {
                return M_indirect_freekick_opp_formation_loser;
            }
            else
            {
                return M_setplay_opp_formation_loser;
            }
        }
        else
        {
            //
            // our (indirect) free kick
            //
            if (wm.ball().pos().x > ServerParam::i().theirPenaltyAreaLineX() && wm.ball().pos().absY() < ServerParam::i().penaltyAreaHalfWidth())
            {
                return M_indirect_freekick_our_formation_loser;
            }
            else
            {
                return M_setplay_our_formation_loser;
            }
        }
    }

    //
    // goal kick
    //
    if (wm.gameMode().type() == GameMode::GoalKick_)
    {
        if (wm.gameMode().side() == wm.ourSide())
        {
            return M_goal_kick_our_formation_loser;
        }
        else
        {
            return M_goal_kick_opp_formation_loser;
        }
    }

    //
    // goalie catch
    //
    if (wm.gameMode().type() == GameMode::GoalieCatch_)
    {
        if (wm.gameMode().side() == wm.ourSide())
        {
            return M_goalie_catch_our_formation_loser;
        }
        else
        {
            return M_goalie_catch_opp_formation_loser;
        }
    }

    //
    // before kick off
    //
    if (wm.gameMode().type() == GameMode::BeforeKickOff || wm.gameMode().type() == GameMode::AfterGoal_)
    {
        return M_before_kick_off_formation_loser;
    }

    //
    // other set play
    //
    if (wm.gameMode().isOurSetPlay(wm.ourSide()))
    {
        return M_setplay_our_formation_loser;
    }

    if (wm.gameMode().type() != GameMode::PlayOn)
    {
        return M_setplay_opp_formation_loser;
    }

    //
    // unknown
    //
    switch (M_current_situation)
    {
    case Defense_Situation:
        return M_defense_formation_loser;
    case Offense_Situation:
        return M_offense_formation_loser;
    default:
        break;
    }

    return M_normal_formation_loser;
}

Formation::Ptr
Strategy::getFormation_winner(const WorldModel &wm) const
{
    //
    // play on
    //
    if (wm.gameMode().type() == GameMode::PlayOn)
    {
        switch (M_current_situation)
        {
        case Defense_Situation:
            return M_defense_formation_winner;
        case Offense_Situation:
            return M_offense_formation_winner;
        default:
            break;
        }
        return M_normal_formation_winner;
    }

    //
    // kick in, corner kick
    //
    if (wm.gameMode().type() == GameMode::KickIn_ || wm.gameMode().type() == GameMode::CornerKick_)
    {
        if (wm.ourSide() == wm.gameMode().side())
        {
            // our kick-in or corner-kick
            return M_kickin_our_formation_winner;
        }
        else
        {
            return M_setplay_opp_formation_winner;
        }
    }

    //
    // our indirect free kick
    //
    if ((wm.gameMode().type() == GameMode::BackPass_ && wm.gameMode().side() == wm.theirSide()) || (wm.gameMode().type() == GameMode::IndFreeKick_ && wm.gameMode().side() == wm.ourSide()))
    {
        return M_indirect_freekick_our_formation_winner;
    }

    //
    // opponent indirect free kick
    //
    if ((wm.gameMode().type() == GameMode::BackPass_ && wm.gameMode().side() == wm.ourSide()) || (wm.gameMode().type() == GameMode::IndFreeKick_ && wm.gameMode().side() == wm.theirSide()))
    {
        return M_indirect_freekick_opp_formation_winner;
    }

    //
    // after foul
    //
    if (wm.gameMode().type() == GameMode::FoulCharge_ || wm.gameMode().type() == GameMode::FoulPush_)
    {
        if (wm.gameMode().side() == wm.ourSide())
        {
            //
            // opponent (indirect) free kick
            //
            if (wm.ball().pos().x < ServerParam::i().ourPenaltyAreaLineX() + 1.0 && wm.ball().pos().absY() < ServerParam::i().penaltyAreaHalfWidth() + 1.0)
            {
                return M_indirect_freekick_opp_formation_winner;
            }
            else
            {
                return M_setplay_opp_formation_winner;
            }
        }
        else
        {
            //
            // our (indirect) free kick
            //
            if (wm.ball().pos().x > ServerParam::i().theirPenaltyAreaLineX() && wm.ball().pos().absY() < ServerParam::i().penaltyAreaHalfWidth())
            {
                return M_indirect_freekick_our_formation_winner;
            }
            else
            {
                return M_setplay_our_formation_winner;
            }
        }
    }

    //
    // goal kick
    //
    if (wm.gameMode().type() == GameMode::GoalKick_)
    {
        if (wm.gameMode().side() == wm.ourSide())
        {
            return M_goal_kick_our_formation_winner;
        }
        else
        {
            return M_goal_kick_opp_formation_winner;
        }
    }

    //
    // goalie catch
    //
    if (wm.gameMode().type() == GameMode::GoalieCatch_)
    {
        if (wm.gameMode().side() == wm.ourSide())
        {
            return M_goalie_catch_our_formation_winner;
        }
        else
        {
            return M_goalie_catch_opp_formation_winner;
        }
    }

    //
    // before kick off
    //
    if (wm.gameMode().type() == GameMode::BeforeKickOff || wm.gameMode().type() == GameMode::AfterGoal_)
    {
        return M_before_kick_off_formation_winner;
    }

    //
    // other set play
    //
    if (wm.gameMode().isOurSetPlay(wm.ourSide()))
    {
        return M_setplay_our_formation_winner;
    }

    if (wm.gameMode().type() != GameMode::PlayOn)
    {
        return M_setplay_opp_formation_winner;
    }

    //
    // unknown
    //
    switch (M_current_situation)
    {
    case Defense_Situation:
        return M_defense_formation_winner;
    case Offense_Situation:
        return M_offense_formation_winner;
    default:
        break;
    }

    return M_normal_formation_winner;
}

/*-------------------------------------------------------------------*/
/*!

 */
Strategy::BallArea
Strategy::get_ball_area(const WorldModel &wm)
{
    int ball_step = 1000;
    ball_step = std::min(ball_step, wm.interceptTable().teammateStep());
    ball_step = std::min(ball_step, wm.interceptTable().opponentStep());
    ball_step = std::min(ball_step, wm.interceptTable().selfStep());

    return get_ball_area(wm.ball().inertiaPoint(ball_step));
}

/*-------------------------------------------------------------------*/
/*!

 */
Strategy::BallArea
Strategy::get_ball_area(const Vector2D &ball_pos)
{
    dlog.addLine(Logger::TEAM,
                 52.5, -17.0, -52.5, -17.0,
                 "#999999");
    dlog.addLine(Logger::TEAM,
                 52.5, 17.0, -52.5, 17.0,
                 "#999999");
    dlog.addLine(Logger::TEAM,
                 36.0, -34.0, 36.0, 34.0,
                 "#999999");
    dlog.addLine(Logger::TEAM,
                 -1.0, -34.0, -1.0, 34.0,
                 "#999999");
    dlog.addLine(Logger::TEAM,
                 -30.0, -17.0, -30.0, 17.0,
                 "#999999");
    dlog.addLine(Logger::TEAM,
                 //-36.5, -34.0, -36.5, 34.0,
                 -35.5, -34.0, -35.5, 34.0,
                 "#999999");

    if (ball_pos.x > 36.0)
    {
        if (ball_pos.absY() > 17.0)
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": get_ball_area: Cross");
            dlog.addRect(Logger::TEAM,
                         36.0, -34.0, 52.5 - 36.0, 34.0 - 17.0,
                         "#00ff00");
            dlog.addRect(Logger::TEAM,
                         36.0, 17.0, 52.5 - 36.0, 34.0 - 17.0,
                         "#00ff00");
            return BA_Cross;
        }
        else
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": get_ball_area: ShootChance");
            dlog.addRect(Logger::TEAM,
                         36.0, -17.0, 52.5 - 36.0, 34.0,
                         "#00ff00");
            return BA_ShootChance;
        }
    }
    else if (ball_pos.x > -1.0)
    {
        if (ball_pos.absY() > 17.0)
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": get_ball_area: DribbleAttack");
            dlog.addRect(Logger::TEAM,
                         -1.0, -34.0, 36.0 + 1.0, 34.0 - 17.0,
                         "#00ff00");
            dlog.addRect(Logger::TEAM,
                         -1.0, 17.0, 36.0 + 1.0, 34.0 - 17.0,
                         "#00ff00");
            return BA_DribbleAttack;
        }
        else
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": get_ball_area: OffMidField");
            dlog.addRect(Logger::TEAM,
                         -1.0, -17.0, 36.0 + 1.0, 34.0,
                         "#00ff00");
            return BA_OffMidField;
        }
    }
    else if (ball_pos.x > -30.0)
    {
        if (ball_pos.absY() > 17.0)
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": get_ball_area: DribbleBlock");
            dlog.addRect(Logger::TEAM,
                         -30.0, -34.0, -1.0 + 30.0, 34.0 - 17.0,
                         "#00ff00");
            dlog.addRect(Logger::TEAM,
                         -30.0, 17.0, -1.0 + 30.0, 34.0 - 17.0,
                         "#00ff00");
            return BA_DribbleBlock;
        }
        else
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": get_ball_area: DefMidField");
            dlog.addRect(Logger::TEAM,
                         -30.0, -17.0, -1.0 + 30.0, 34.0,
                         "#00ff00");
            return BA_DefMidField;
        }
    }
    // 2009-06-17 akiyama: -36.5 -> -35.5
    // else if ( ball_pos.x > -36.5 )
    else if (ball_pos.x > -35.5)
    {
        if (ball_pos.absY() > 17.0)
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": get_ball_area: CrossBlock");
            dlog.addRect(Logger::TEAM,
                         //-36.5, -34.0, 36.5 - 30.0, 34.0 - 17.0,
                         -35.5, -34.0, 35.5 - 30.0, 34.0 - 17.0,
                         "#00ff00");
            dlog.addRect(Logger::TEAM,
                         -35.5, 17.0, 35.5 - 30.0, 34.0 - 17.0,
                         "#00ff00");
            return BA_CrossBlock;
        }
        else
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": get_ball_area: Stopper");
            dlog.addRect(Logger::TEAM,
                         //-36.5, -17.0, 36.5 - 30.0, 34.0,
                         -35.5, -17.0, 35.5 - 30.0, 34.0,
                         "#00ff00");
            // 2009-06-17 akiyama: Stopper -> DefMidField
            // return BA_Stopper;
            return BA_DefMidField;
        }
    }
    else
    {
        if (ball_pos.absY() > 17.0)
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": get_ball_area: CrossBlock");
            dlog.addRect(Logger::TEAM,
                         -52.5, -34.0, 52.5 - 36.5, 34.0 - 17.0,
                         "#00ff00");
            dlog.addRect(Logger::TEAM,
                         -52.5, 17.0, 52.5 - 36.5, 34.0 - 17.0,
                         "#00ff00");
            return BA_CrossBlock;
        }
        else
        {
            dlog.addText(Logger::TEAM,
                         __FILE__ ": get_ball_area: Danger");
            dlog.addRect(Logger::TEAM,
                         -52.5, -17.0, 52.5 - 36.5, 34.0,
                         "#00ff00");
            return BA_Danger;
        }
    }

    dlog.addText(Logger::TEAM,
                 __FILE__ ": get_ball_area: unknown area");
    return BA_None;
}

/*-------------------------------------------------------------------*/
/*!

 */
double
Strategy::get_normal_dash_power(const WorldModel &wm)
{
    static bool s_recover_mode = false;

    // G2d: role
    int role = Strategy::i().roleNumber(wm.self().unum());

    if (wm.self().staminaModel().capacityIsEmpty())
    {
        return std::min(ServerParam::i().maxDashPower(),
                        wm.self().stamina() + wm.self().playerType().extraStamina());
    }

    const int self_min = wm.interceptTable().selfStep();
    const int mate_min = wm.interceptTable().teammateStep();
    const int opp_min = wm.interceptTable().opponentStep();

    // check recover
    if (wm.self().staminaModel().capacityIsEmpty())
    {
        s_recover_mode = false;
    }
    else if (wm.self().stamina() < ServerParam::i().staminaMax() * (static_cast<double>(Setting::i().moving_save_energy) / 100.0 - 0.1))
    {
        s_recover_mode = true;
    }
    else if (wm.self().stamina() > ServerParam::i().staminaMax() * (static_cast<double>(Setting::i().moving_save_energy) / 100.0 + 0.1))
    {
        s_recover_mode = false;
    }

    /*--------------------------------------------------------*/
    double dash_power = ServerParam::i().maxDashPower();
    const double my_inc = wm.self().playerType().staminaIncMax() * wm.self().recovery();

    if (wm.ourDefenseLineX() > wm.self().pos().x && wm.ball().pos().x < wm.ourDefenseLineX() + 20.0)
    {
        dlog.addText(Logger::TEAM,
                     __FILE__ ": (get_normal_dash_power) correct DF line. keep max power");
        // keep max power
        dash_power = ServerParam::i().maxDashPower();
    }
    else if (s_recover_mode)
    {
        dash_power = my_inc - 25.0; // preffered recover value
        if (dash_power < 0.0)
            dash_power = 0.0;

        dlog.addText(Logger::TEAM,
                     __FILE__ ": (get_normal_dash_power) recovering");
    }

    // G2d: run to offside line
    else if (wm.ball().pos().x > 0.0 && wm.self().pos().x < wm.offsideLineX() && fabs(wm.ball().pos().x - wm.self().pos().x) < 25.0)
        dash_power = ServerParam::i().maxDashPower();

    // G2d: defenders
    else if (wm.ball().pos().x < 10.0 && (role == 4 || role == 5 || role == 2 || role == 3))
        dash_power = ServerParam::i().maxDashPower();

    // G2d: midfielders
    else if (wm.ball().pos().x < -10.0 && (role == 6 || role == 7 || role == 8))
        dash_power = ServerParam::i().maxDashPower();

    // G2d: run in opp penalty area
    else if (wm.ball().pos().x > 36.0 && wm.self().pos().x > 36.0 && mate_min < opp_min - 4)
        dash_power = ServerParam::i().maxDashPower();

    // exist kickable teammate
    else if (wm.kickableTeammate() && wm.ball().distFromSelf() < 20.0)
    {
        dash_power = std::min(my_inc * 1.1,
                              ServerParam::i().maxDashPower());
        dlog.addText(Logger::TEAM,
                     __FILE__ ": (get_normal_dash_power) exist kickable teammate. dash_power=%.1f",
                     dash_power);
    }
    // in offside area
    else if (wm.self().pos().x > wm.offsideLineX())
    {
        dash_power = ServerParam::i().maxDashPower();
        dlog.addText(Logger::TEAM,
                     __FILE__ ": in offside area. dash_power=%.1f",
                     dash_power);
    }
    else if (wm.ball().pos().x > 25.0 && wm.ball().pos().x > wm.self().pos().x + 10.0 && self_min < opp_min - 6 && mate_min < opp_min - 6)
    {
        dash_power = bound(ServerParam::i().maxDashPower() * 0.1,
                           my_inc * 0.5,
                           ServerParam::i().maxDashPower());
        dlog.addText(Logger::TEAM,
                     __FILE__ ": (get_normal_dash_power) opponent ball dash_power=%.1f",
                     dash_power);
    }
    // normal
    else
    {
        dash_power = std::min(my_inc * 1.7,
                              ServerParam::i().maxDashPower());
        dlog.addText(Logger::TEAM,
                     __FILE__ ": (get_normal_dash_power) normal mode dash_power=%.1f",
                     dash_power);
    }

    return dash_power;
}
