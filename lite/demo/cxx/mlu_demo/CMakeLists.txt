cmake_minimum_required(VERSION 3.0)
project(demo CXX C)
set(MKLML_DIR "${PROJECT_SOURCE_DIR}/../../../third_party/mklml/")

set(TARGET mobilenet_full_api)
# 1. lib & include
link_directories(${PROJECT_SOURCE_DIR}/lib ${MKLML_DIR}/lib  /usr/local/neuware/lib64)
include_directories(${PROJECT_SOURCE_DIR}/include ${MKLML_DIR}/include  /usr/local/neuware/include)
# 2. compile options
add_definitions(-std=c++11 -g -O3 -DNDEBUG -pthread -DLITE_WITH_MLU -DLITE_WITH_X86)
#add_definitions(-g  )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
# 3.add executable output
add_executable(${TARGET} ${TARGET}.cc)
target_link_libraries(${TARGET} -lpaddle_full_api_shared  -liomp5 -lcnrt -lcndrv -ldl -lstdc++)
