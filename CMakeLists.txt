cmake_minimum_required(VERSION 3.0)
project(paddle-mobile)
add_definitions(-DPADDLE_MOBILE_DEBUG)
add_definitions(-DENABLE_EXCEPTION)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build)

file(GLOB_RECURSE PADDLE_MOBILE_CC src/*.cc src/*.cpp src/*.c)
file(GLOB_RECURSE PADDLE_MOBILE_H src/*.h)

include_directories(src/)

option(USE_OPENMP "openmp support" ON)
if(USE_OPENMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

add_library(paddle-mobile SHARED ${PADDLE_MOBILE_CC} ${PADDLE_MOBILE_H})

add_subdirectory(test)
