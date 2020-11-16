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
include(lite_utils)

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


if(WIN32)
    option(MSVC_STATIC_CRT "use static C Runtime library by default" ON)

    set(CMAKE_SUPPRESS_REGENERATION ON)
    set(CMAKE_STATIC_LIBRARY_PREFIX lib)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")

    if(MSVC_STATIC_CRT)
      set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
      set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
      set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
      set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
    else(MSVC_STATIC_CRT)
      set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MDd")
      set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MD")
      set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MDd")
      set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MD")
    endif(MSVC_STATIC_CRT)

    add_compile_options(/wd4068 /wd4129 /wd4244 /wd4267 /wd4297 /wd4530 /wd4577 /wd4819 /wd4838)
    add_compile_options(/MP)
    message(STATUS "Using parallel compiling (/MP)")
    set(PADDLE_LINK_FLAGS "/IGNORE:4006 /IGNORE:4098 /IGNORE:4217 /IGNORE:4221")
    set(CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_STATIC_LINKER_FLAGS} ${PADDLE_LINK_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${PADDLE_LINK_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${PADDLE_LINK_FLAGS}")
endif()

if(NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    find_package(CUDA QUIET)
endif()
find_package(Git REQUIRED)
find_package(Threads REQUIRED)

include(simd)

################################ Exposed Configurations #######################################
lite_option(WITH_DSO            "Compile PaddlePaddle with dynamic linked CUDA" ON)
lite_option(WITH_AVX            "Compile PaddlePaddle with AVX intrinsics"      ON IF ${AVX_FOUND})
lite_option(WITH_TESTING        "Compile PaddlePaddle with unit testing"        OFF)
lite_option(WITH_MKL            "Compile PaddlePaddle with MKL support."        ON IF ${AVX_FOUND})
lite_option(WITH_ARM_DOTPROD    "Compile PaddlePaddle with ARM dot production"  ON)
lite_option(WITH_SYSTEM_BLAS    "Use system blas library"           OFF)

# for lite, both server and mobile framework.
lite_option(LITE_WITH_JAVA      "Enable Java JNI lib in lite mode" OFF)
lite_option(LITE_WITH_PYTHON    "Enable Python api lib in lite mode" OFF)
lite_option(LITE_WITH_CUDA      "Enable CUDA in lite mode" OFF)
lite_option(LITE_WITH_X86       "Enable X86 in lite mode"  ON)
lite_option(LITE_WITH_ARM       "Enable ARM in lite mode"  OFF)
lite_option(LITE_WITH_NPU       "Enable NPU in lite mode"  OFF)
lite_option(LITE_WITH_RKNPU     "Enable RKNPU in lite mode"  OFF)
lite_option(LITE_WITH_MLU       "Enable MLU in lite mode"  OFF)
lite_option(LITE_WITH_HUAWEI_ASCEND_NPU  "Enable HUAWEI_ASCEND_NPU in lite mode"  OFF)
lite_option(LITE_WITH_IMAGINATION_NNA    "Enable IMAGINATION_NNA in lite mode"  OFF)
lite_option(LITE_WITH_XPU       "Enable XPU in lite mode"  OFF)
lite_option(LITE_WITH_XTCL      "Enable XPU via XTCL"  OFF IF LITE_WITH_XPU)
lite_option(LITE_WITH_BM        "Enable BM in lite mode"   OFF)
lite_option(LITE_WITH_APU       "Enable APU in lite mode"  OFF)
lite_option(LITE_WITH_TRAIN     "Enable training operators and kernels in lite" OFF)
lite_option(LITE_WITH_OPENMP    "Enable OpenMP in lite framework" ON)
lite_option(LITE_WITH_OPENCL    "Enable OpenCL support in lite" OFF)
lite_option(LITE_WITH_FPGA      "Enable FPGA support in lite" OFF)
lite_option(LITE_WITH_LIGHT_WEIGHT_FRAMEWORK  "Enable light-weight framework" OFF)
lite_option(LITE_WITH_PROFILE   "Enable profile mode in lite framework"  OFF)
lite_option(LITE_WITH_PRECISION_PROFILE "Enable precision profile in profile mode ON in lite" OFF)
lite_option(LITE_WITH_LOG       "Enable log printing or not." ON)
lite_option(LITE_WITH_EXCEPTION "Enable throwing the exception when error occurs in lite" OFF)
lite_option(LITE_WITH_NVTX      "Enable nvtx or not, please enable LITE_WITH_CUDA first." OFF)
lite_option(LITE_ON_TINY_PUBLISH "Publish tiny predictor lib." OFF)
lite_option(LITE_ON_MODEL_OPTIMIZE_TOOL "Build the model optimize tool" OFF)
lite_option(LITE_WITH_BENCHMARK_TEST "Build benchmark test cases" OFF)
# publish options
lite_option(LITE_BUILD_EXTRA    "Enable extra algorithm support in Lite, both kernels and operators" OFF)
lite_option(LITE_BUILD_TAILOR   "Enable tailoring library according to model" OFF)
# cv build options
lite_option(LITE_WITH_CV        "Enable build cv image in lite" OFF)
lite_option(LITE_WITH_STATIC_CUDA "Statically link cuda libraries." OFF)
lite_option(CUDA_WITH_FP16      "Compile with cuda half support" OFF)
lite_option(LITE_WITH_ARM_CLANG "when arm lang is clang, its ON." OFF)

# TODO(Superjomn) Remove WITH_ANAKIN option if not needed latter.
if(ANDROID OR IOS OR ARMLINUX)
    set(WITH_GPU OFF CACHE STRING
            "Disable GPU when cross-compiling for Android and iOS" FORCE)
    set(WITH_DSO OFF CACHE STRING
            "Disable DSO when cross-compiling for Android and iOS" FORCE)
    set(WITH_AVX OFF CACHE STRING
            "Disable AVX when cross-compiling for Android and iOS" FORCE)
    set(WITH_RDMA OFF CACHE STRING
            "Disable RDMA when cross-compiling for Android and iOS" FORCE)
    set(WITH_MKL OFF CACHE STRING
            "Disable MKL when cross-compiling for Android and iOS" FORCE)
endif()

if(ANDROID OR IOS)
    set(LITE_WITH_PYTHON OFF CACHE STRING
            "Disable PYTHON when cross-compiling for Android and iOS" FORCE)
endif()

set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
        "A path setting third party libraries download & build directories.")

# CMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE)
    if(WIN32)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
    else()
    
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
            "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel"
            FORCE)
    endif()
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# check options
if (LITE_ON_TINY_PUBLISH)
    if (NOT (WITH_LITE AND LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND NOT WITH_TESTING))#LITE_WITH_JAVA AND
        message(FATAL_ERROR "LITE_ON_TINY_PUBLISH=ON must be used with WITH_LITE=ON LITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON LITE_WITH_JAVA=ON WITH_TESTING=OFF")
        return()
    endif()
endif()

include_directories("${PADDLE_SOURCE_DIR}")
# the generated header files.
set(LITE_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}")
include_directories("${LITE_GENERATED_INCLUDE_DIR}")

if (LITE_WITH_PYTHON)
    include(external/python)    # download, build, install python
    include(external/pybind11)    # download, build, install pybind11
endif()

if(LITE_WITH_RKNPU)
   include(device/rknpu)
endif()

if(LITE_WITH_IMAGINATION_NNA)
	include(device/imagination_nna)
endif()

include(external/flatbuffers)

# for opencl
if (LITE_WITH_OPENCL)
    include(external/opencl-headers)
    include(external/opencl-clhpp)
endif()

# for mobile
if (WITH_LITE AND LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    message(STATUS "Building the mobile framework")
    include(cross_compiling/postproject)
    include(device/npu) # check and prepare NPU DDK
    include(device/xpu) # check and prepare XPU SDK
    include(device/apu) # check and prepare APU SDK

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

if(LITE_WITH_XPU)
    include(device/xpu)
endif()

if(LITE_WITH_MLU)
    include(mlu)
endif()

if(LITE_WITH_HUAWEI_ASCEND_NPU)
    include(device/huawei_ascend_npu)
endif()

include(coveralls)

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

include(cudnn)
include(configure)          # add paddle env configuration

if(LITE_WITH_CUDA)
  include(cuda)
endif()

if(LITE_WITH_BM)
  include(bm)
endif()
include(generic)            # simplify cmake module
include(ccache)             # set ccache for compilation
include(util)               # set unittest and link libs
include(version)            # set PADDLE_VERSION
if(NOT APPLE)
  include(flags)
endif()

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")

add_subdirectory(lite)
