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
set(PADDLE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PADDLE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# c++ standard
if(LITE_WITH_METAL)
  set(CMAKE_CXX_STANDARD 14)
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

# Host system configurations
include(system)
# Paddlelite functions and macros
include(functions)
# Target os configurations
include(os/common)
# Paddlelite project
project(paddle CXX C)
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "AR tools: ${CMAKE_AR}")
message(STATUS "CMAKE GENERATOR: ${CMAKE_GENERATOR}")

if(WIN32)
  include(os/windows)
endif()
################################ Exposed Configurations #######################################
lite_option(WITH_DSO                           "Compile PaddlePaddle with dynamic linked CUDA"                        ON)
lite_option(WITH_AVX                           "Compile PaddlePaddle with AVX intrinsics"                             ON IF ${AVX_FOUND})
lite_option(WITH_TESTING                       "Compile PaddlePaddle with unit testing"                               OFF)
lite_option(WITH_MKL                           "Compile PaddlePaddle with MKL support."                               ON IF ${AVX_FOUND})
lite_option(WITH_ARM_DOTPROD                   "Compile PaddlePaddle with ARM dot production"                         ON)
lite_option(WITH_SYSTEM_BLAS                   "Use system blas library"                                              OFF)
# for lite, both server and mobile framework.
lite_option(LITE_WITH_JAVA                     "Enable Java JNI lib in lite mode"                                     OFF)
lite_option(LITE_WITH_STATIC_LIB               "Enable static cplus lib in lite mode"                                 OFF)
lite_option(LITE_WITH_PYTHON                   "Enable Python api lib in lite mode"                                   OFF)
lite_option(LITE_WITH_CUDA                     "Enable CUDA in lite mode"                                             OFF)
lite_option(LITE_WITH_X86                      "Enable X86 in lite mode"                                              ON)
lite_option(LITE_WITH_ARM                      "Enable ARM in lite mode"                                              OFF)
lite_option(LITE_WITH_SW                       "Enable SW in lite mode"                                               OFF)
lite_option(LITE_WITH_NPU                      "Enable NPU in lite mode"                                              OFF)
lite_option(LITE_WITH_MLU                      "Enable MLU in lite mode"                                              OFF)
lite_option(LITE_WITH_NNADAPTER                "Enable NNAdapter in lite mode"                                        OFF)
lite_option(LITE_WITH_XPU                      "Enable XPU in lite mode"                                              OFF)
lite_option(LITE_WITH_BM                       "Enable BM in lite mode"                                               OFF)
lite_option(LITE_WITH_TRAIN                    "Enable training operators and kernels in lite"                        OFF)
lite_option(LITE_WITH_OPENMP                   "Enable OpenMP in lite framework"                                      ON)
lite_option(LITE_WITH_OPENCL                   "Enable OpenCL support in lite"                                        OFF)
lite_option(LITE_WITH_METAL                    "Enable Metal support in lite"                                         OFF)
lite_option(LITE_WITH_FPGA                     "Enable FPGA support in lite"                                          OFF)
lite_option(LITE_WITH_INTEL_FPGA               "Enable Intel FPGA support in lite"                                    OFF)
lite_option(LITE_WITH_PROFILE                  "Enable profile mode in lite framework"                                OFF)
lite_option(LITE_WITH_PRECISION_PROFILE        "Enable precision profile in profile mode ON in lite"                  OFF)
lite_option(LITE_WITH_LOG                      "Enable log printing or not."                                          ON)
lite_option(LITE_WITH_EXCEPTION                "Enable throwing the exception when error occurs in lite"              OFF)
lite_option(LITE_WITH_NVTX                     "Enable nvtx or not, please enable LITE_WITH_CUDA first."              OFF)
lite_option(LITE_ON_TINY_PUBLISH               "Publish tiny predictor lib."                                          OFF)
lite_option(LITE_ON_MODEL_OPTIMIZE_TOOL        "Build the model optimize tool"                                        OFF)
lite_option(LITE_WITH_BENCHMARK_TEST           "Build benchmark test cases"                                           OFF)
lite_option(LITE_THREAD_POOL                   "Enable thread pool in lite"                                           OFF)
# publish options
lite_option(LITE_BUILD_EXTRA                   "Enable extra algorithm support in Lite, both kernels and operators"   OFF)
lite_option(LITE_BUILD_TAILOR                  "Enable tailoring library according to model"                          OFF)
# cv build options
lite_option(LITE_WITH_CV                       "Enable build cv image in lite"                                        OFF)
lite_option(LITE_WITH_STATIC_CUDA              "Statically link cuda libraries."                                      OFF)
lite_option(CUDA_WITH_FP16                     "Compile with cuda half support"                                       OFF)
lite_option(LITE_WITH_ARM_CLANG                "when arm lang is clang, its ON."                                      OFF)
lite_option(LITE_WITH_XCODE                    "when debug in xcode, its ON."                                         OFF)
lite_option(LITE_WITH_ARM82_FP16               "when compile with arm v8.2 fp16, it's ON."                            OFF)
lite_option(LITE_WITH_ARM8_SVE2                "Enable SVE2 instructions in ARMv8."                                   OFF)
lite_option(LITE_WITH_ARM82_INT8_SDOT          "when compile with arm v8.2 int8, it's ON."                            OFF)
lite_option(LITE_WITH_CODE_META_INFO           "include git version in the header file."                              ON)
lite_option(WITH_NODE_RAW_FS                   "(Only available when compiling by Emscripten) Whether build with NODERAWFS"               OFF)

# Thirdparty
set(THIRD_PARTY_PATH "${PADDLE_BINARY_DIR}/third_party" CACHE STRING
  "A path setting third party libraries download & build directories.")

if(NOT LITE_WITH_ARM)
  find_package(CUDA QUIET)
endif()
find_package(Git REQUIRED)
find_package(Threads REQUIRED)

# Simd configurations
include(simd)

# CMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE)
  if(WIN32)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel" FORCE)
  else()
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel" FORCE)
    endif()
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# check options
if(LITE_ON_TINY_PUBLISH)
  if(NOT (LITE_WITH_ARM AND NOT WITH_TESTING))#LITE_WITH_JAVA AND
    message(FATAL_ERROR "LITE_ON_TINY_PUBLISH=ON must be used with LITE_WITH_ARM=ON LITE_WITH_JAVA=ON WITH_TESTING=OFF")
    return()
  endif()
endif()

include_directories("${PADDLE_SOURCE_DIR}")
# the generated header files.
set(LITE_GENERATED_INCLUDE_DIR "${PADDLE_BINARY_DIR}")
include_directories("${LITE_GENERATED_INCLUDE_DIR}")

if(LITE_WITH_PYTHON)
  include(external/python)    # download, build, install python
  include(external/pybind11)    # download, build, install pybind11
endif()

# flatbuffer module for loading model
if(LITE_UPDATE_FBS_HEAD)
  include(external/flatbuffers)
else()
  include_directories("${PADDLE_SOURCE_DIR}/third-party/flatbuffers/pre-build")
  add_custom_target(fbs_headers ${TARGET}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PADDLE_SOURCE_DIR}/third-party/flatbuffers/pre-build/framework_generated.h" "${PADDLE_SOURCE_DIR}/lite/model_parser/flatbuffers/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PADDLE_SOURCE_DIR}/third-party/flatbuffers/pre-build/param_generated.h" "${PADDLE_SOURCE_DIR}/lite/model_parser/flatbuffers/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PADDLE_SOURCE_DIR}/third-party/flatbuffers/pre-build/cache_generated.h" "${PADDLE_SOURCE_DIR}/lite/backends/opencl/utils/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PADDLE_SOURCE_DIR}/third-party/flatbuffers/pre-build/tune_cache_generated.h" "${PADDLE_SOURCE_DIR}/lite/backends/opencl/utils/"
    )
endif()

if(LITE_THREAD_POOL)
  add_definitions(-DLITE_USE_THREAD_POOL)
endif()

if (LITE_THREAD_POOL OR ((ARM_TARGET_LANG STREQUAL "clang") AND (ARM_TARGET_ARCH_ABI STREQUAL "armv7")))
  set(LITE_WITH_OPENMP OFF CACHE STRING "Disable OpenMP when Thread Pool is enabled" FORCE)
endif()

set(WITH_MKLML ${WITH_MKL})
if(NOT DEFINED WITH_MKLDNN)
  if(WITH_MKL AND AVX2_FOUND)
    set(WITH_MKLDNN ON)
  else()
    message(STATUS "Do not have AVX2 intrinsics and disabled MKL-DNN")
    set(WITH_MKLDNN OFF)
  endif()
endif()

include(backends/common)
add_subdirectory(lite)
