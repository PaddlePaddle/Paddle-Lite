cmake_minimum_required(VERSION 2.8)

set(TARGET arm_linux_full_api)

# 1. path to Paddle-Lite lib
set(LITE_DIR "${PROJECT_SOURCE_DIR}/../../../cxx")

# 2. link Paddle-Lite directory
link_directories(${LITE_DIR}/lib)

include_directories(${LITE_DIR}/include)

# 3. compile options 
add_definitions(-std=c++11 -g -O3 -pthread)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

# 4.add executable output
add_executable(${TARGET} ${TARGET}.cc)
target_link_libraries(${TARGET} -lpaddle_full_api_shared)
#target_link_libraries(${TARGET} ${LITE_DIR}/lib/libpaddle_api_full_bundled.a)
