# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License.

cmake_minimum_required(VERSION 3.0)

SET(ADNN_LIBRARY_TYPE "default" CACHE STRING "Type of library (shared, static, or default) to build")
SET_PROPERTY(CACHE ADNN_LIBRARY_TYPE PROPERTY STRINGS default static shared)

IF(CMAKE_OSX_ARCHITECTURES)
  LIST(LENGTH CMAKE_OSX_ARCHITECTURES CMAKE_OSX_ARCHITECTURES_COUNT)
  IF(NOT CMAKE_OSX_ARCHITECTURES MATCHES "^(arm64|arm64e|arm64_32)$")
    MESSAGE(FATAL_ERROR "Unsupported CMAKE_OSX_ARCHITECTURES value \"${CMAKE_OSX_ARCHITECTURES}\"")
  ENDIF()
  SET(ADNN_TARGET_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
ELSEIF(CMAKE_GENERATOR MATCHES "^Visual Studio " AND CMAKE_GENERATOR_PLATFORM)
  IF(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
    SET(ADNN_TARGET_PROCESSOR "arm64")
  ELSEIF(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64EC")
    SET(ADNN_TARGET_PROCESSOR "arm64ec")
  ELSE()
    MESSAGE(FATAL_ERROR "Unsupported Visual Studio architecture \"${CMAKE_GENERATOR_PLATFORM}\"")
  ENDIF()
ELSEIF(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv[5-8]")
  SET(ADNN_TARGET_PROCESSOR "arm")
ELSEIF(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  SET(ADNN_TARGET_PROCESSOR "arm64")
ELSE()
  MESSAGE(FATAL_ERROR "Unsupported CMAKE_SYSTEM_PROCESSOR value \"${CMAKE_SYSTEM_PROCESSOR}\"")
ENDIF()

IF(NOT CMAKE_SYSTEM_NAME)
  MESSAGE(FATAL_ERROR "CMAKE_SYSTEM_NAME not defined")
ELSEIF(NOT CMAKE_SYSTEM_NAME MATCHES "^(Android|Darwin|iOS|Linux|Windows|CYGWIN|MSYS|QURT)$")
  MESSAGE(FATAL_ERROR "Unrecognized CMAKE_SYSTEM_NAME value \"${CMAKE_SYSTEM_NAME}\"")
ENDIF()

OPTION(ADNN_WITH_ARM "Build ADNN with Arm CPU" ON)
OPTION(ADNN_ARM_WITH_FP16 "Build ADNN with Arm FP16 kernels" ON)
OPTION(ADNN_ARM_WITH_BF16 "Build ADNN with Arm BF16 kernels" ON)
OPTION(ADNN_ARM_WITH_DOTPROD "Build ADNN with Arm DotProd (integer dot product) kernels" ON)
OPTION(ADNN_ARM_WITH_SVE "Build ADNN with Arm SVE kernels" ON)

ADD_COMPILE_DEFINITIONS("ADNN_WITH_ARM=$<BOOL:${ADNN_WITH_ARM}>")
ADD_COMPILE_DEFINITIONS("ADNN_ARM_WITH_FP16=$<BOOL:${ADNN_ARM_WITH_FP16}>")
ADD_COMPILE_DEFINITIONS("ADNN_ARM_WITH_DOTPROD=$<BOOL:${ADNN_ARM_WITH_DOTPROD}>")
ADD_COMPILE_DEFINITIONS("ADNN_ENABLE_DOTPROD=$<BOOL:${ADNN_ENABLE_DOTPROD}>")
ADD_COMPILE_DEFINITIONS("ADNN_ARM_WITH_SVE=$<BOOL:${ADNN_ARM_WITH_SVE}>")

include_directories(include)
include_directories(src)
add_subdirectory(src)
