
add_executable(sample_trainer
  sample_trainer.cpp
  main_trainer.cpp
  )

target_include_directories(sample_trainer
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/trainer
    ${PROJECT_BINARY_DIR}
  PUBLIC
    ${Boost_INCLUDE_DIRS}
    ${LIBRCSC_INCLUDE_DIR}
  )

target_link_libraries(sample_trainer
  PUBLIC
    ${LIBRCSC_LIB}
    Boost::system
    ZLIB::ZLIB
  )

set_target_properties(sample_trainer
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
  )
