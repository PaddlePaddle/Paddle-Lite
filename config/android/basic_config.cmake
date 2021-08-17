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
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

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

set(ANDROID_ARCH_ABI ${ARM_TARGET_ARCH_ABI} CACHE STRING "Choose Android Arch ABI")
if(ARM_TARGET_ARCH_ABI STREQUAL "armv8")
    set(ANDROID_ARCH_ABI "arm64-v8a")
endif()

if(ARM_TARGET_ARCH_ABI STREQUAL "armv7")
    set(ANDROID_ARCH_ABI "armeabi-v7a")
endif()

set(ANDROID_ARCH_ABI_LIST "arm64-v8a" "armeabi-v7a"
"armeabi-v6" "armeabi" "mips" "mips64" "x86" "x86_64")
if(NOT DEFINED ANDROID_ARCH_ABI)
  set(ANDROID_ARCH_ABI ${ANDROID_ARCH_ABI})
  message(STATUS "Setting ANDROID_ARCH_ABI to default: ${ANDROID_ARCH_ABI}")
else()
  if(NOT ANDROID_ARCH_ABI IN_LIST ANDROID_ARCH_ABI_LIST)
    message(FATAL_ERROR "ANDROID_ARCH_ABI: ${ANDROID_ARCH_ABI} not support!")
  else()
    message(STATUS "Target ANDROID_ARCH_ABI is ${ANDROID_ARCH_ABI}")
  endif()
endif()

set(ANDROID_STL_TYPE_LIST "c++_static" "gnustl_static" "c++_shared")
if(NOT DEFINED ANDROID_STL_TYPE)
  set(ANDROID_STL_TYPE "c++_static")
  message(STATUS "Setting ANDROID_STL_TYPE to default: ${ANDROID_STL_TYPE}")
else()
  if(NOT ANDROID_STL_TYPE IN_LIST ANDROID_STL_TYPE_LIST)
    message(FATAL_ERROR "ANDROID_STL_TYPE: ${ANDROID_STL_TYPE} not support!")
  else()
    message(STATUS "Target ANDROID_STL_TYPE is ${ANDROID_STL_TYPE}")
  endif()
endif()

if(ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
    message(STATUS "armeabi-v7a use softfp by default.")
    set(CMAKE_ANDROID_ARM_NEON ON)
    message(STATUS "NEON is enabled on arm-v7a with softfp.")
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

if(NOT DEFINED ANDROID_NDK)
    set(ANDROID_NDK $ENV{NDK_ROOT})
    if(NOT ANDROID_NDK)
        message(FATAL_ERROR "Must set ANDROID_NDK or env NDK_ROOT")
    endif()
endif()
message(STATUS "lsycheck ANDROID_NDK ::: ${ANDROID_NDK}")

if(NOT DEFINED ANDROID_NATIVE_API_LEVEL)
    set(ANDROID_NATIVE_API_LEVEL "21")
    if(ARM_TARGET_ARCH_ABI STREQUAL "armv7")
        if(LITE_WITH_NPU AND NOT LITE_ON_TINY_PUBLISH)
            set(ANDROID_NATIVE_API_LEVEL "24") # HIAI DDK depends on android-24
        else()
            set(ANDROID_NATIVE_API_LEVEL "16")
        endif()
    endif()
endif()

if(ARM_TARGET_LANG STREQUAL "clang")
    set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION ${ARM_TARGET_LANG})
    set(ANDROID_TOOLCHAIN clang)
    set(CMAKE_TOOLCHAIN_FILE ${ANDROID_NDK}/build/cmake/android.toolchain.cmake)
    set(ANDROID_ABI ${ANDROID_ARCH_ABI})
    set(ANDROID_STL ${ANDROID_STL_TYPE})
    if(ARM_TARGET_ARCH_ABI STREQUAL "armv8")
        set(triple aarch64-v8a-linux-android)
        if(ANDROID_STL_TYPE MATCHES "^c\\+\\+_")
            # Use CMAKE_CXX_STANDARD_LIBRARIES_INIT to ensure libunwind and libc++ is linked in the right order
            set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_CXX_STANDARD_LIBRARIES_INIT} ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${ANDROID_ARCH_ABI}/libunwind.a")
            if (ANDROID_NATIVE_API_LEVEL LESS 21)
                set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_CXX_STANDARD_LIBRARIES_INIT} ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${ANDROID_ARCH_ABI}/libandroid_support.a")
            endif()
            if(ANDROID_STL_TYPE STREQUAL "c++_shared")
                set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_CXX_STANDARD_LIBRARIES_INIT} ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${ANDROID_ARCH_ABI}/libc++_shared.so")
            elseif(ANDROID_STL_TYPE STREQUAL "c++_static")
                set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_CXX_STANDARD_LIBRARIES_INIT} ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${ANDROID_ARCH_ABI}/libc++_static.a")
                set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_CXX_STANDARD_LIBRARIES_INIT} ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${ANDROID_ARCH_ABI}/libc++abi.a")
            else()
                message(FATAL_ERROR "Invalid Android STL TYPE: ${ANDROID_STL_TYPE}.")
            endif()
        endif()
    elseif(ARM_TARGET_ARCH_ABI STREQUAL "armv7")
        set(triple arm-v7a-linux-android)
        set(ANDROID_ARM_NEON TRUE)
        set(LITE_WITH_OPENMP OFF CACHE STRING "Due to libomp's bug(For ARM64, it has been fixed by https://reviews.llvm.org/D19879, but still exists on ARM32), disable OpenMP on armv7 when cross-compiling using Clang" FORCE)
        if(ANDROID_STL_TYPE MATCHES "^c\\+\\+_")
            # Use CMAKE_CXX_STANDARD_LIBRARIES_INIT to ensure libunwind and libc++ is linked in the right order
            set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_CXX_STANDARD_LIBRARIES_INIT} ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${ANDROID_ARCH_ABI}/libunwind.a")
            if (ANDROID_NATIVE_API_LEVEL LESS 21)
                set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_CXX_STANDARD_LIBRARIES_INIT} ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${ANDROID_ARCH_ABI}/libandroid_support.a")
            endif()
            if(ANDROID_STL_TYPE STREQUAL "c++_shared")
                set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_CXX_STANDARD_LIBRARIES_INIT} ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${ANDROID_ARCH_ABI}/libc++_shared.so")
            elseif(ANDROID_STL_TYPE STREQUAL "c++_static")
                set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_CXX_STANDARD_LIBRARIES_INIT} ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${ANDROID_ARCH_ABI}/libc++_static.a")
                set(CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_CXX_STANDARD_LIBRARIES_INIT} ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${ANDROID_ARCH_ABI}/libc++abi.a")
            else()
                message(FATAL_ERROR "Invalid Android STL TYPE: ${ANDROID_STL_TYPE}.")
            endif()
        endif()
    else()
        message(FATAL_ERROR "Clang do not support this ${ARM_TARGET_ARCH_ABI}, use armv8 or armv7")
    endif()

    set(CMAKE_C_COMPILER clang)
    set(CMAKE_C_COMPILER_TARGET ${triple})
    set(CMAKE_CXX_COMPILER clang++)
    set(CMAKE_CXX_COMPILER_TARGET ${triple})
    message(STATUS "CMAKE_CXX_COMPILER_TARGET: ${CMAKE_CXX_COMPILER_TARGET}")
endif()

#include(cross_compiling/android)
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

## Python Setting
set(LITE_WITH_PYTHON OFF CACHE STRING
"Disable PYTHON when cross-compiling for Android and iOS" FORCE)

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

## Third party
set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
        "A path setting third party libraries download & build directories.")

## TODO: Double check needed
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--as-needed")

## Others
if (LITE_WITH_OPENCL)
    include_directories("${PADDLE_SOURCE_DIR}/third-party/opencl/include")
endif()

set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_SYSTEM_VERSION ${ANDROID_NATIVE_API_LEVEL})
set(CMAKE_ANDROID_ARCH_ABI ${ANDROID_ARCH_ABI})
set(CMAKE_ANDROID_NDK ${ANDROID_NDK})
set(CMAKE_ANDROID_STL_TYPE ${ANDROID_STL_TYPE})
if(ARM_TARGET_LANG STREQUAL "gcc")
    if(ARM_TARGET_ARCH_ABI STREQUAL "armv8")
        set(CMAKE_SYSTEM_PROCESSOR aarch64)
        set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
        set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
    elseif(ARM_TARGET_ARCH_ABI STREQUAL "armv7")
        set(CMAKE_SYSTEM_PROCESSOR armv7-a)
        set(CMAKE_C_COMPILER "arm-linux-gnueabi-gcc")
        set(CMAKE_CXX_COMPILER "arm-linux-gnueabi-g++")
    else()
        message(FATAL_ERROR "INVALID ARM TARGET ARCH ABI: ${ARM_TARGET_ARCH_ABI}")
    endif()
endif()

# then check input arm abi
if(ARM_TARGET_ARCH_ABI STREQUAL "armv7hf")
    message(FATAL_ERROR "ANDROID does not support hardfp on v7 use armv7 instead.")
endif()

set(ANDROID_ARCH_ABI ${ARM_TARGET_ARCH_ABI} CACHE STRING "Choose Android Arch ABI")
if(ARM_TARGET_ARCH_ABI STREQUAL "armv8")
    set(ANDROID_ARCH_ABI "arm64-v8a")
endif()

if(ARM_TARGET_ARCH_ABI STREQUAL "armv7")
    set(ANDROID_ARCH_ABI "armeabi-v7a")
endif()

## Definitions
add_definitions(-DLITE_WITH_LINUX)
add_definitions(-DLITE_WITH_ANDROID)