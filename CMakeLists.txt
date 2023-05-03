cmake_minimum_required(VERSION 3.5)

project(helios-base VERSION 2023.03)

if(CMAKE_VERSION VERSION_LESS "3.7.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()

# compliler options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_FLAGS "-W -Wall")

# check librcsc
set(LIBRCSC_INSTALL_DIR "$ENV{HOME}/.local;$ENV{HOME}/local" CACHE PATH "The path where librcsc is installed.")

find_library(LIBRCSC_LIB NAMES rcsc
  PATHS ${LIBRCSC_INSTALL_DIR}
  PATH_SUFFIXES lib
  )
if(NOT LIBRCSC_LIB)
  message(FATAL_ERROR "librcsc library not found!")
endif()
get_filename_component(LIBRCSC_LIBDIR ${LIBRCSC_LIB} DIRECTORY)

find_path(LIBRCSC_INCLUDE_DIR
  NAME rcsc/types.h
  PATHS ${LIBRCSC_INSTALL_DIR}
  PATH_SUFFIXES include
  )
if(NOT LIBRCSC_INCLUDE_DIR)
  message(FATAL_ERROR "librcsc include dir not found!")
endif()

# remove variables from GUI
mark_as_advanced(
  LIBRCSC_LIB
  LIBRCSC_INCLUDE_DIR
  LIBRCSC_LIBDIR
  )

# boost
find_package(Boost 1.36.0 COMPONENTS system REQUIRED)
if(NOT Boost_FOUND)
  message(FATAL_ERROR "Boost not found!")
endif()

# zlib
find_package(ZLIB)
if(ZLIB_FOUND)
  set(HAVE_LIBZ TRUE)
endif()

# generate config.h
add_definitions(-DHAVE_CONFIG_H)
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake-config.h.in
  ${PROJECT_BINARY_DIR}/config.h
  )

# check the settings
message(STATUS "Found librcsc:")
message(STATUS "  LIBRCSC_LIBDIR=${LIBRCSC_LIBDIR}")
message(STATUS "  LIBRCSC_LIB=${LIBRCSC_LIB}")
message(STATUS "  LIBRCSC_INCLUDE_DIR=${LIBRCSC_INCLUDE_DIR}")
message(STATUS "Build settings:")
message(STATUS "  BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# sub directories
add_subdirectory(src)
