cmake_minimum_required(VERSION 2.8)
project(demo CXX C)

add_definitions(-DLITE_WITH_CUDA)

set(TARGET demo)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

set(LITE_ROOT "${PROJECT_SOURCE_DIR}/../../cxx")
set(PROTOBUF_ROOT "${PROJECT_SOURCE_DIR}/../../third_party/protobuf")

include_directories("${LITE_ROOT}/include")
link_directories("${LITE_ROOT}/lib")
link_directories("${PROTOBUF_ROOT}/lib")
# cuda lib
link_directories("/usr/local/cuda/lib64/")

add_executable(${TARGET} ${TARGET}.cc)

set(DEPS ${LITE_ROOT}/lib/libpaddle_full_api_shared.so)
set(DEPS ${DEPS} protobuf-lite)
set(DEPS ${DEPS} "-lrt -lpthread -ldl -lcudart")

target_link_libraries(${TARGET} ${DEPS})
