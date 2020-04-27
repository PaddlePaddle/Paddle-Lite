cmake_minimum_required(VERSION 2.8)

set(TARGET mobilenet_light_api)

# 1. path to Paddle-Lite lib and mklml lib
set(LITE_DIR "${PROJECT_SOURCE_DIR}/../../../cxx")
set(MKLML_DIR "${PROJECT_SOURCE_DIR}/../../../third_party/mklml/")

# 2. link mklml and Paddle-Lite directory
link_directories(${LITE_DIR}/lib ${MKLML_DIR}/lib)
include_directories(${LITE_DIR}/include/ ${MKLML_DIR}/include)

# 3. compile options 
add_definitions(-std=c++11 -g -O3 -pthread)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

# 4.add executable output
add_executable(${TARGET} ${TARGET}.cc)
target_link_libraries(${TARGET} -lpaddle_light_api_shared)
target_link_libraries(${TARGET} -lmklml_intel)
target_link_libraries(${TARGET} -ldl)
