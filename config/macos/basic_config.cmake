# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Setting Cmake Env ##
if(LITE_WITH_XTCL)
  set(CMAKE_CXX_STANDARD 14)
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

## Setting OS
set(OS_LIST "android" "armlinux" "ios" "ios64" "armmacos")
if(NOT DEFINED ARM_TARGET_OS)
  set(ARM_TARGET_OS "android")
  message(STATUS "Setting Target OS to ${ARM_TARGET_OS}")
else()
  if(NOT ARM_TARGET_OS IN_LIST OS_LIST)
    message(FATAL_ERROR "${ARM_TARGET_OS} not support!")
  else()
    message(STATUS "Target OS is ${ARM_TARGET_OS}")
  endif()
endif()

## Setting ARCH
set(ARCH_LIST "armv8" "armv7" "armv7hf" "arm64-v8a" "armeabi-v7a")
if(NOT DEFINED ARM_TARGET_ARCH_ABI)
  set(ARM_TARGET_ARCH_ABI "armv8")
  message(STATUS "Setting Target OS to ${ARM_TARGET_ARCH_ABI}")
else()
  if(NOT ARM_TARGET_ARCH_ABI IN_LIST ARCH_LIST)
    message(FATAL_ERROR "${ARM_TARGET_ARCH_ABI} not support!")
  else()
    message(STATUS "Target ARCH is ${ARM_TARGET_ARCH_ABI}")
  endif()
endif()

## Setting ToolChain
set(TOOLCHAIN_LIST "gcc" "clang")
if(NOT DEFINED ARM_TARGET_LANG)
  set(ARM_TARGET_LANG "gcc")
  message(STATUS "Using default toolchain: ${ARM_TARGET_LANG}")
else()
  if(NOT ARM_TARGET_LANG IN_LIST TOOLCHAIN_LIST)
    message(FATAL_ERROR "${ARM_TARGET_LANG} not support!")
  else()
    message(STATUS "Target ARCH is ${ARM_TARGET_LANG}")
  endif()
endif()

if(ARM_TARGET_OS STREQUAL "armmacos")
  include(cross_compiling/armmacos)
endif()
include(cross_compiling/host)

## Setting Lib Type
set(ARM_TARGET_LIB_TYPE_LIST "static" "shared")
if(NOT DEFINED ARM_TARGET_LIB_TYPE)
  set(ARM_TARGET_LIB_TYPE "static")
  message(STATUS "Setting default lib type: ${ARM_TARGET_LIB_TYPE}")
else()
  if(NOT ARM_TARGET_LIB_TYPE IN_LIST ARM_TARGET_LIB_TYPE_LIST)
    message(FATAL_ERROR "${ARM_TARGET_LIB_TYPE} not support!")
  else()
    message(STATUS "Lib type is ${ARM_TARGET_LIB_TYPE}")
  endif()
endif()

## Compile C Flags
if (LITE_ON_TINY_PUBLISH OR LITE_WITH_LTO)
  if(ARM_TARGET_LANG STREQUAL "gcc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -flto=thin")
  endif()
endif()

## Build Type
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

## TODO: Double check needed
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

## Third Party
set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
        "A path setting third party libraries download & build directories.")

## TODO: Double check needed
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--as-needed")

## Others
if (LITE_WITH_PYTHON)
    include(external/python)    # download, build, install python
    include(external/pybind11)    # download, build, install pybind11
endif()

if (LITE_WITH_OPENCL)
    include_directories("${PADDLE_SOURCE_DIR}/third-party/opencl/include")
endif()