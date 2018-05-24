cmake_minimum_required(VERSION 3.0)
project(paddle-mobile)
add_definitions(-std=c++11)
add_definitions(-DPADDLE_MOBILE_DEBUG="true")


set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build)

file(GLOB_RECURSE PADDLE_MOBILE_CC src/*.cc src/*.cpp)
file(GLOB_RECURSE PADDLE_MOBILE_H src/*.h)

# include headers
include_directories(src/)

#include(ExternalProject)
#ExternalProject_Add(openblas_proj
#        GIT_REPOSITORY "https://github.com/xianyi/OpenBLAS.git"
#        GIT_TAG "v0.2.20"
#        SOURCE_DIR "openblas/"
#        BUILD_IN_SOURCE 1
#        CONFIGURE_COMMAND ""
#        BUILD_COMMAND "make" "ONLY_CBLAS=1"
#        INSTALL_COMMAND "make" "PREFIX=${CMAKE_BINARY_DIR}/" "install"
#        )
#set_target_properties(openblas_proj PROPERTIES EXCLUDE_FROM_ALL 1)
# link protobuf
include_directories(third-party/protobuf/include)
if (ANDROID)
    link_directories(third-party/protobuf/armeabi-v7a)
else()
    # link openblas
    link_directories(third-party/protobuf/lib)
endif ()

#add_dependencies(paddle-mobile openblas_proj)

# gen static
ADD_LIBRARY(paddle-mobile STATIC ${PADDLE_MOBILE_CC} ${PADDLE_MOBILE_H})
if (ANDROID)
    # openblas.a need log lib
    target_link_libraries(paddle-mobile protobuf-lite)
else()
    target_link_libraries(paddle-mobile protobuf-lite)
endif ()
#add_dependencies(paddle-mobile openblas_proj)

add_subdirectory(test)