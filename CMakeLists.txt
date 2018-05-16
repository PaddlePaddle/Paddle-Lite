cmake_minimum_required(VERSION 3.0)
project(paddle-mobile)
add_definitions(-std=c++11)
add_definitions(-DPADDLE_MOBILE_DEBUG="true")


set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build)

file(GLOB_RECURSE PADDLE_MOBILE_CC src/*.cc src/*.cpp)
file(GLOB_RECURSE PADDLE_MOBILE_H src/*.h)

# include headers
include_directories(src/)

include(ExternalProject)
ExternalProject_Add(openblas_proj
        GIT_REPOSITORY "https://github.com/xianyi/OpenBLAS.git"
        GIT_TAG "v0.2.20"
        SOURCE_DIR "openblas/"
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND ""
        BUILD_COMMAND "make" "ONLY_CBLAS=1"
        INSTALL_COMMAND "make" "PREFIX=${CMAKE_BINARY_DIR}/" "install"
        )
set_target_properties(openblas_proj PROPERTIES EXCLUDE_FROM_ALL 1)
# link openblas
include_directories(${CMAKE_BINARY_DIR}/openblas)
link_directories(${CMAKE_BINARY_DIR}/lib)

# link protobuf
include_directories(third-party/protobuf/include)
link_directories(third-party/protobuf/lib)

# gen so
ADD_LIBRARY(paddle-mobile SHARED ${PADDLE_MOBILE_CC} ${PADDLE_MOBILE_H})
target_link_libraries(paddle-mobile protobuf-lite openblas)
add_dependencies(paddle-mobile openblas_proj)

# gen static
ADD_LIBRARY(paddle-mobile-static STATIC ${PADDLE_MOBILE_CC} ${PADDLE_MOBILE_H})
target_link_libraries(paddle-mobile-static protobuf-lite openblas)
add_dependencies(paddle-mobile openblas_proj)

# gen test
ADD_EXECUTABLE(paddle-mobile-test test/main.cpp test/test_helper.h)
target_link_libraries(paddle-mobile-test paddle-mobile)
