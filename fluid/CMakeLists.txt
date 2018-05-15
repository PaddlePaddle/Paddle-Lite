cmake_minimum_required(VERSION 3.0)
project(paddle-mobile)
add_definitions(-std=c++11)
add_definitions(-DPADDLE_MOBILE_DEBUG="true")


set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build)

file(GLOB_RECURSE PADDLE_MOBILE_CC src/*.cc src/*.cpp)
file(GLOB_RECURSE PADDLE_MOBILE_H src/*.h)

# include headers
include_directories(src/)


include_directories(/usr/local/Cellar/openblas/0.2.20_2/include/)
link_directories(/usr/local/Cellar/openblas/0.2.20_2/lib/)


# link protobuf
include_directories(third-party/protobuf/include)
link_directories(third-party/protobuf/lib)

# gen so
ADD_LIBRARY(paddle-mobile SHARED ${PADDLE_MOBILE_CC} ${PADDLE_MOBILE_H})
target_link_libraries(paddle-mobile protobuf-lite openblasp-r0.2.20)

# gen static
ADD_LIBRARY(paddle-mobile-static STATIC ${PADDLE_MOBILE_CC} ${PADDLE_MOBILE_H})
target_link_libraries(paddle-mobile-static protobuf-lite openblasp-r0.2.20)

# gen test
ADD_EXECUTABLE(paddle-mobile-test test/main.cpp test/test_helper.h)
target_link_libraries(paddle-mobile-test paddle-mobile)
