cmake_minimum_required(VERSION 3.0)
project(demo CXX C)

set(TARGET mobilenet_full_api)
# 1. lib & include
link_directories(${PROJECT_SOURCE_DIR}/lib ${PROJECT_SOURCE_DIR}/lib/pcie ${PROJECT_SOURCE_DIR}/lib/bmcompiler)
include_directories(${PROJECT_SOURCE_DIR}/include)
# 2. compile options 
add_definitions(-std=c++11 -g -O3 -DNDEBUG -pthread)
#add_definitions(-g  )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
# 3.add executable output
add_executable(${TARGET} ${TARGET}.cc)
target_link_libraries(${TARGET} -lpaddle_full_api_shared -lbmlib -lbmrt -lbmcompiler)
