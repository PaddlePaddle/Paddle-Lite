cmake_minimum_required(VERSION 3.6)
project(quali)
add_definitions(-DENABLE_EXCEPTION)

set(CMAKE_CXX_STANDARD 11)
file(GLOB_RECURSE QULIFICATON_CC src/*.cc src/*.cpp src/*.c src/*.mm)
file(GLOB_RECURSE QULIFICATON_H src/*.h)
include_directories(. src/)

#add_library(paddle-mobile SHARED ${QULIFICATON_CC} ${QULIFICATON_H} convert.cpp)

add_executable(quantify convert.cpp ${QULIFICATON_CC} ${QULIFICATON_H})