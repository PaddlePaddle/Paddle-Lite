# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(WITH_PADDLE_MOBILE   "Use the paddle-mobile legacy build"    OFF)
if (WITH_PADDLE_MOBILE)
    add_subdirectory(mobile)
    return()
endif(WITH_PADDLE_MOBILE)

set(PADDLE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PADDLE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_CXX_STANDARD 11)

include(system)
include(cross_compiling/preproject)

project(paddle CXX C)
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
        "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "AR tools: ${CMAKE_AR}")

if(NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    find_package(CUDA QUIET)
endif()
find_package(Git REQUIRED)
find_package(Threads REQUIRED)

include(simd)

################################ Exposed Configurations #######################################
option(WITH_DSO         "Compile PaddlePaddle with dynamic linked CUDA" ON)
option(WITH_AVX         "Compile PaddlePaddle with AVX intrinsics"      ${AVX_FOUND})
option(WITH_PYTHON      "Compile PaddlePaddle with python interpreter"  ON)
option(WITH_TESTING     "Compile PaddlePaddle with unit testing"        OFF)
option(WITH_MKL         "Compile PaddlePaddle with MKL support."        ${AVX_FOUND})
option(WITH_ARM_DOTPROD "Compile PaddlePaddle with ARM dot production"  ON)
option(WITH_SYSTEM_BLAS   "Use system blas library"           OFF)
# TODO(Superjomn) Remove WITH_ANAKIN option if not needed latter.
if(ANDROID OR IOS OR ARMLINUX)
    set(WITH_GPU OFF CACHE STRING
            "Disable GPU when cross-compiling for Android and iOS" FORCE)
    set(WITH_DSO OFF CACHE STRING
            "Disable DSO when cross-compiling for Android and iOS" FORCE)
    set(WITH_AVX OFF CACHE STRING
            "Disable AVX when cross-compiling for Android and iOS" FORCE)
    set(WITH_PYTHON OFF CACHE STRING
            "Disable PYTHON when cross-compiling for Android and iOS" FORCE)
    set(WITH_RDMA OFF CACHE STRING
            "Disable RDMA when cross-compiling for Android and iOS" FORCE)
    set(WITH_MKL OFF CACHE STRING
            "Disable MKL when cross-compiling for Android and iOS" FORCE)
endif()

# for lite, both server and mobile framework.
option(LITE_WITH_JAVA "Enable Java JNI lib in lite mode" OFF)
option(LITE_WITH_CUDA "Enable CUDA in lite mode" OFF)
option(LITE_WITH_X86  "Enable X86 in lite mode"  ON)
option(LITE_WITH_ARM  "Enable ARM in lite mode"  OFF)
option(LITE_WITH_NPU  "Enable NPU in lite mode"  OFF)
option(LITE_WITH_OPENMP "Enable OpenMP in lite framework" ON)
option(LITE_WITH_OPENCL   "Enable OpenCL support in lite" OFF)
option(LITE_WITH_FPGA   "Enable FPGA support in lite" OFF)
option(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK  "Enable light-weight framework" OFF)
option(LITE_WITH_PROFILE  "Enable profile mode in lite framework"  OFF)
option(LITE_SHUTDOWN_LOG "Shutdown log system or not." OFF)
option(LITE_ON_TINY_PUBLISH "Publish tiny predictor lib." OFF)
# publish options
option(LITE_BUILD_EXTRA "Enable extra algorithm support in Lite, both kernels and operators" OFF)

set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
        "A path setting third party libraries download & build directories.")

# CMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
            "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel"
            FORCE)
endif()

# check options
if (LITE_ON_TINY_PUBLISH)
    if (NOT (WITH_LITE AND LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND NOT WITH_TESTING))#LITE_WITH_JAVA AND
        message(FATAL_ERROR "LITE_ON_TINY_PUBLISH=ON must be used with WITH_LITE=ON LITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON LITE_WITH_JAVA=ON WITH_TESTING=OFF")
        return()
    endif()
endif()

include_directories("${PADDLE_SOURCE_DIR}")

# for mobile
if (WITH_LITE AND LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    message(STATUS "Building the mobile framework")
    include(cross_compiling/postproject)
    include(cross_compiling/npu) # check and prepare NPU DDK

    # We compile the mobile deployment library when LITE_ON_TINY_PUBLISH=ON
    # So the following third party dependencies are not needed.
    if (NOT LITE_ON_TINY_PUBLISH)
        # include the necessary thirdparty dependencies
        include(external/gflags)    # download, build, install gflags
        # LITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON will disable glog
        # TODO(sangoly): refine WITH_LITE and LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
        include(external/gtest)     # download, build, install gtest
        include(ccache)             # set ccache for compilation
        include(external/protobuf)  # download, build, install protobuf
    endif()

    # for opencl
    if (LITE_WITH_OPENCL)
        include(external/opencl-headers)
        include(external/opencl-clhpp)
    endif()

    include(generic)            # simplify cmake module
    include(configure)          # add paddle env configuration

    add_subdirectory(lite)
    return()
endif()
#################################  End of mobile compile ##############################

set(WITH_MKLML ${WITH_MKL})
if (NOT DEFINED WITH_MKLDNN)
    if (WITH_MKL AND AVX2_FOUND)
        set(WITH_MKLDNN ON)
    else()
        message(STATUS "Do not have AVX2 intrinsics and disabled MKL-DNN")
        set(WITH_MKLDNN OFF)
    endif()
endif()

########################################################################################

include(external/mklml)     # download mklml package
include(external/xbyak)     # download xbyak package
include(external/libxsmm)   # download, build, install libxsmm
include(external/gflags)    # download, build, install gflags
include(external/glog)      # download, build, install glog
include(external/gtest)     # download, build, install gtest
include(external/protobuf)  # download, build, install protobuf
include(external/openblas)  # download, build, install openblas
include(external/mkldnn)    # download, build, install mkldnn
include(external/eigen)     # download eigen3
include(external/xxhash)    # download install xxhash needed for x86 jit

include(configure)          # add paddle env configuration

include(generic)            # simplify cmake module
include(ccache)             # set ccache for compilation
include(util)               # set unittest and link libs
include(version)            # set PADDLE_VERSION

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")

add_subdirectory(lite)
