cmake_minimum_required(VERSION 3.0)
project(paddle-mobile)
add_definitions(-DPADDLE_MOBILE_DEBUG="true")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY build)
set(ACL_ROOT /home/halsay/acl_android/ComputeLibrary)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${ACL_ROOT}/build")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -larm_compute")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -larm_compute_core")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -larm_compute_graph")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${ACL_ROOT}/build/opencl-1.2-stubs")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lOpenCL")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_ACL=1")

file(GLOB_RECURSE PADDLE_MOBILE_CC src/*.cc src/*.cpp src/*.c)
file(GLOB_RECURSE PADDLE_MOBILE_H src/*.h)

# include headers
include_directories(src/)
include_directories(${ACL_ROOT} ${ACL_ROOT}/include)
add_definitions(-DUSE_ACL=1)
add_definitions(-DUSE_OPENCL)
if(WITH_OPENGLES)
  add_definitions(-DUSE_OPENGLES)
endif(WITH_OPENGLES)

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

#add_dependencies(paddle-mobile openblas_proj)

# gen static
ADD_LIBRARY(paddle-mobile SHARED ${PADDLE_MOBILE_CC} ${PADDLE_MOBILE_H})

#add_dependencies(paddle-mobile openblas_proj)

add_subdirectory(test)
