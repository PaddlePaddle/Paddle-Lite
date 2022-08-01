# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.10)

# Arm config
if(LITE_WITH_ARM)
  set(ARM_TARGET_OS_LIST "android" "armlinux" "ios" "ios64" "armmacos" "qnx")
  set(ARM_TARGET_ARCH_ABI_LIST "armv8" "armv7" "armv7hf" "arm64-v8a" "armeabi-v7a")
  set(ARM_TARGET_LANG_LIST "gcc" "clang")
  set(ARM_TARGET_LIB_TYPE_LIST "static" "shared")

  # OS check
  if(NOT DEFINED ARM_TARGET_OS)
    set(ARM_TARGET_OS "android")
  else()
    if(NOT ARM_TARGET_OS IN_LIST ARM_TARGET_OS_LIST)
      message(FATAL_ERROR "ARM_TARGET_OS should be one of ${ARM_TARGET_OS_LIST}")
    endif()
  endif()

  # Abi check
  if(NOT DEFINED ARM_TARGET_ARCH_ABI)
    set(ARM_TARGET_ARCH_ABI "armv8")
  else()
    if(NOT ARM_TARGET_ARCH_ABI IN_LIST ARM_TARGET_ARCH_ABI_LIST)
      message(FATAL_ERROR "ARM_TARGET_ARCH_ABI should be one of ${ARM_TARGET_ARCH_ABI_LIST}")
    endif()
  endif()

  # Toolchain check
  if(NOT DEFINED ARM_TARGET_LANG)
    set(ARM_TARGET_LANG "gcc")
  else()
    if(NOT ARM_TARGET_LANG IN_LIST ARM_TARGET_LANG_LIST)
      message(FATAL_ERROR "ARM_TARGET_LANG should be one of ${ARM_TARGET_LANG_LIST}")
    endif()
  endif()

  # Target lib check
  if(NOT DEFINED ARM_TARGET_LIB_TYPE)
    set(ARM_TARGET_LIB_TYPE "static")
  else()
    if(NOT ARM_TARGET_LIB_TYPE IN_LIST ARM_TARGET_LIB_TYPE_LIST)
      message(FATAL_ERROR "ARM_TARGET_LIB_TYPE should be one of ${ARM_TARGET_LIB_TYPE_LIST}")
    endif()
  endif()

  # Toolchain config
  if ((LITE_ON_TINY_PUBLISH OR LITE_WITH_LTO) AND NOT ARM_TARGET_OS STREQUAL "qnx")
    if(ARM_TARGET_LANG STREQUAL "gcc")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
    else()
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -flto=thin")
    endif()
  endif()
  message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

  # OS settings
  if(ARM_TARGET_OS STREQUAL "android")
    include(os/android)
  endif()
  if(ARM_TARGET_OS STREQUAL "armlinux")
    include(os/armlinux)
  endif()
  if(ARM_TARGET_OS STREQUAL "ios" OR ARM_TARGET_OS STREQUAL "ios64")
    include(os/ios)
  endif()
  if(ARM_TARGET_OS STREQUAL "armmacos")
    include(os/armmacos)
  endif()
  if(ARM_TARGET_OS STREQUAL "qnx")
    include(os/qnx)
  endif()

  # Detect origin host toolchain
  set(HOST_C_COMPILER $ENV{CC})
  set(HOST_CXX_COMPILER $ENV{CXX})
  if(IOS OR ARMMACOS)
    set(default_cc clang)
    set(default_cxx clang++)
  else()
    set(default_cc gcc)
    set(default_cxx g++)
  endif()
  if(NOT HOST_C_COMPILER)
    find_program(HOST_C_COMPILER NAMES ${default_cc} PATH
      /usr/bin
      /usr/local/bin)
  endif()
  if(NOT HOST_CXX_COMPILER)
    find_program(HOST_CXX_COMPILER NAMES ${default_cxx} PATH
      /usr/bin
      /usr/local/bin)
  endif()
  if(NOT HOST_C_COMPILER OR NOT EXISTS ${HOST_C_COMPILER})
    message(FATAL_ERROR "Cannot find host C compiler. export CC=/path/to/cc")
  endif()
  if(NOT HOST_CXX_COMPILER OR NOT EXISTS ${HOST_CXX_COMPILER})
    message(FATAL_ERROR "Cannot find host CXX compiler. export CXX=/path/to/cxx")
  endif()
  message(STATUS "Found host C compiler: " ${HOST_C_COMPILER})
  message(STATUS "Found host CXX compiler: " ${HOST_CXX_COMPILER})
  # Build type
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default use Release in android" FORCE)
  endif()
  # Third party build type
  if(NOT THIRD_PARTY_BUILD_TYPE)
    set(THIRD_PARTY_BUILD_TYPE "MinSizeRel" CACHE STRING "Default use MinSizeRel in android" FORCE)
  endif()
  message(STATUS "Lite ARM Compile ${ARM_TARGET_OS} with ${ARM_TARGET_ARCH_ABI} ${ARM_TARGET_LANG}")
endif()

if(NOT APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--as-needed")
endif()

# TODO(Superjomn) Remove WITH_ANAKIN option if not needed latter.
if(ANDROID OR IOS OR ARMLINUX OR ARMMACOS)
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

# Python
if(ANDROID OR IOS)
  set(LITE_WITH_PYTHON OFF CACHE STRING
    "Disable PYTHON when cross-compiling for Android and iOS" FORCE)
endif()

# Enable arc for metal
if(APPLE)
  if(NOT DEFINED ENABLE_ARC)
    # Unless specified, enable ARC support by default
    set(ENABLE_ARC TRUE)
    message(STATUS "Enabling ARC support by default. ENABLE_ARC not provided!")
  endif()
  set(ENABLE_ARC_INT ${ENABLE_ARC} CACHE BOOL "Whether or not to enable ARC" ${FORCE_CACHE})
  if(ENABLE_ARC_INT)
    set(FOBJC_ARC "-fobjc-arc")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES CACHE INTERNAL "")
    message(STATUS "Enabling ARC support.")
  else()
    set(FOBJC_ARC "-fno-objc-arc")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC NO CACHE INTERNAL "")
    message(STATUS "Disabling ARC support.")
  endif()
  set(CMAKE_C_FLAGS "${FOBJC_ARC} ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${FOBJC_ARC} ${CMAKE_CXX_FLAGS}")
endif()
