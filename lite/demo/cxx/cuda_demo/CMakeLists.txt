project(demo CXX C)
cmake_minimum_required(VERSION 2.8)

set(TARGET demo)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

set(LITE_LIB "${PROJECT_SOURCE_DIR}/../../cxx")
set(PROTOBUF_LIB "${PROJECT_SOURCE_DIR}/../../third_party/protobuf")

include_directories("${LITE_LIB}/include")
link_directories("${LITE_LIB}/lib")
link_directories("${PROTOBUF_LIB}/lib")

add_executable(${TARGET} ${TARGET}.cc)

set(DEPS ${LITE_LIB}/lib/libpaddle_full_api_shared.so)
set(DEPS ${DEPS} protobuf-lite)
set(DEPS ${DEPS} "-lrt -lpthread -ldl")

target_link_libraries(${TARGET} ${DEPS})
