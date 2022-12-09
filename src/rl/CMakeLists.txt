add_executable(rl_player
        ../chain_action/actgen_cross.cpp
        ../chain_action/actgen_direct_pass.cpp
        ../chain_action/actgen_self_pass.cpp
        ../chain_action/actgen_shoot.cpp
        ../chain_action/actgen_short_dribble.cpp
        ../chain_action/actgen_simple_dribble.cpp
        ../chain_action/actgen_strict_check_pass.cpp
        ../chain_action/action_chain_graph.cpp
        ../chain_action/action_chain_holder.cpp
        ../chain_action/bhv_chain_action.cpp
        ../chain_action/bhv_normal_dribble.cpp
        ../chain_action/bhv_pass_kick_find_receiver.cpp
        ../chain_action/bhv_strict_check_shoot.cpp
        ../chain_action/body_force_shoot.cpp
        ../chain_action/clear_ball.cpp
        ../chain_action/clear_generator.cpp
        ../chain_action/cooperative_action.cpp
        ../chain_action/cross_generator.cpp
        ../chain_action/dribble.cpp
        ../chain_action/field_analyzer.cpp
        ../chain_action/hold_ball.cpp
        ../chain_action/neck_turn_to_receiver.cpp
        ../chain_action/pass.cpp
        ../chain_action/predict_state.cpp
        ../chain_action/self_pass_generator.cpp
        ../chain_action/shoot.cpp
        ../chain_action/shoot_generator.cpp
        ../chain_action/short_dribble_generator.cpp
        ../chain_action/simple_pass_checker.cpp
        ../chain_action/strict_check_pass_generator.cpp
        ../chain_action/tackle_generator.cpp
        ../bhv_basic_move.cpp
        ../bhv_basic_block.cpp
        ../bhv_basic_offensive_kick.cpp
        ../bhv_basic_tackle.cpp
        ../bhv_custom_before_kick_off.cpp
        ../bhv_go_to_static_ball.cpp
        ../bhv_goalie_basic_move.cpp
        ../bhv_goalie_chase_ball.cpp
        ../bhv_goalie_free_kick.cpp
        ../bhv_prepare_set_play_kick.cpp
        ../bhv_set_play.cpp
        ../bhv_set_play_free_kick.cpp
        ../bhv_set_play_goal_kick.cpp
        ../bhv_set_play_indirect_free_kick.cpp
        ../bhv_set_play_kick_in.cpp
        ../bhv_set_play_kick_off.cpp
        ../bhv_their_goal_kick_move.cpp
        ../bhv_penalty_kick.cpp
        ../neck_default_intercept_neck.cpp
        ../neck_goalie_turn_neck.cpp
        ../neck_offensive_intercept_neck.cpp
        ../view_tactical.cpp
        ../intention_receive.cpp
        ../intention_wait_after_set_play_kick.cpp
        ../soccer_role.cpp
        ../role_center_back.cpp
        ../role_center_forward.cpp
        ../role_defensive_half.cpp
        ../role_goalie.cpp
        ../role_offensive_half.cpp
        ../role_sample.cpp
        ../role_side_back.cpp
        ../role_side_forward.cpp
        ../role_side_half.cpp
        ../role_keepaway_keeper.cpp
        ../role_keepaway_taker.cpp
        ../keepaway_communication.cpp
        ../sample_communication.cpp
        ../sample_field_evaluator.cpp
        ../sample_freeform_message_parser.cpp
        ../strategy.cpp
        ../data_extractor/DEState.cpp
        ../data_extractor/offensive_data_extractor.cpp
        ../bhv_unmark.cpp
        rl_player.cpp
        main_rl_player.cpp
        rl_feature_gen.cpp
        simple_rl_agent.cpp
        rl_by_redis.cpp
        )

find_path(HIREDIS_HEADER hiredis)
find_library(HIREDIS_LIB hiredis)
find_path(REDIS_PLUS_PLUS_HEADER sw)
find_library(REDIS_PLUS_PLUS_LIB redis++)
find_package(Threads REQUIRED)

target_include_directories(rl_player
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src/
        ${PROJECT_SOURCE_DIR}/src/chain_action
        ${PROJECT_BINARY_DIR}
        PUBLIC
        ${Boost_INCLUDE_DIRS}
        ${LIBRCSC_INCLUDE_DIR}
        ${HIREDIS_HEADER}
        ${REDIS_PLUS_PLUS_HEADER}
        )

target_link_libraries(rl_player
        PUBLIC
        ${LIBRCSC_LIB}
        Boost::system
        ZLIB::ZLIB
        ${HIREDIS_LIB}
        ${REDIS_PLUS_PLUS_LIB}
        Threads::Threads
        )

set_target_properties(rl_player
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/src
        )

# rl trainer
add_executable(rl_trainer
        main_rl_trainer.cpp
        rl_trainer.cpp
        rl_by_redis.cpp
        )

target_include_directories(rl_trainer
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src/
        ${PROJECT_BINARY_DIR}
        PUBLIC
        ${Boost_INCLUDE_DIRS}
        ${LIBRCSC_INCLUDE_DIR}
        ${HIREDIS_HEADER}
        ${REDIS_PLUS_PLUS_HEADER}
        )

target_link_libraries(rl_trainer
        PUBLIC
        ${LIBRCSC_LIB}
        Boost::system
        ZLIB::ZLIB
        ${HIREDIS_LIB}
        ${REDIS_PLUS_PLUS_LIB}
        Threads::Threads
        )

set_target_properties(rl_trainer
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/src
        )

configure_file(
        ${PROJECT_SOURCE_DIR}/src/rl/rl_train.sh.in
        ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/rl_train.sh
        @ONLY
)
file(
        COPY ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/rl_train.sh
        DESTINATION ${PROJECT_BINARY_DIR}/src
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)