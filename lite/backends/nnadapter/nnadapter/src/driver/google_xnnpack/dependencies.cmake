# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

CMAKE_MINIMUM_REQUIRED(VERSION 3.12 FATAL_ERROR)

if(NOT NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG)
  set(NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG $ENV{NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG})
endif()

if(NOT NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG)
  set(NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG master)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Android")
  # Get the Android NDK version, refer to https://github.com/Kitware/CMake/blob/d8f95471c7d779c8bb1606fbfff664f1dad6cde1/Modules/Platform/Android-Determine.cmake#L231 for more details.
  if(CMAKE_ANDROID_NDK AND EXISTS "${CMAKE_ANDROID_NDK}/source.properties")
    file(READ "${CMAKE_ANDROID_NDK}/source.properties" _ANDROID_NDK_SOURCE_PROPERTIES)
    set(_ANDROID_NDK_REVISION_REGEX "^Pkg\\.Desc = Android NDK\nPkg\\.Revision = ([0-9]+)\\.([0-9]+)\\.([0-9]+)(-beta([0-9]+))?")
    if(NOT _ANDROID_NDK_SOURCE_PROPERTIES MATCHES "${_ANDROID_NDK_REVISION_REGEX}")
      string(REPLACE "\n" "\n  " _ANDROID_NDK_SOURCE_PROPERTIES "${_ANDROID_NDK_SOURCE_PROPERTIES}")
      message(FATAL_ERROR "Android: Failed to parse NDK revision from:\n ${CMAKE_ANDROID_NDK}/source.properties\n with content:\n  ${_ANDROID_NDK_SOURCE_PROPERTIES}")
    endif()
  endif()
  set(_ANDROID_NDK_MAJOR "${CMAKE_MATCH_1}")
  set(_ANDROID_NDK_MINOR "${CMAKE_MATCH_2}")
  set(_ANDROID_NDK_BUILD "${CMAKE_MATCH_3}")
  set(_ANDROID_NDK_BETA "${CMAKE_MATCH_5}")
  if(_ANDROID_NDK_BETA STREQUAL "")
    set(_ANDROID_NDK_BETA "0")
  endif()
  set(CMAKE_ANDROID_NDK_VERSION "${_ANDROID_NDK_MAJOR}.${_ANDROID_NDK_MINOR}")
  unset(_ANDROID_NDK_SOURCE_PROPERTIES)
  unset(_ANDROID_NDK_REVISION_REGEX)
  unset(_ANDROID_NDK_MAJOR)
  unset(_ANDROID_NDK_MINOR)
  unset(_ANDROID_NDK_BUILD)
  unset(_ANDROID_NDK_BETA)
  if(CMAKE_ANDROID_NDK_VERSION VERSION_LESS 19)
    message(FATAL_ERROR "Upgrade to a newer Android NDK(r19c or higher), Clang in NDK r18b or before doesn't support the intrinsic functions used in some XNNPACK microkernels, refer to https://github.com/google/XNNPACK/issues/1359 for more details.")
  endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "^arm.*")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.5)
        message(FATAL_ERROR "GNU GCC version must be at least 7.5, it is strongly recommended that you install ARM's cross-compilation toolchain on Ubuntu 18.04 and above systems!")
      endif()
    else()
      message(FATAL_ERROR "Only GNU GCC is supported.")
    endif()
  endif()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    set(XNNPACK_TARGET_PROCESSOR "armv7hf")
  else()
    set(XNNPACK_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  set(CROSS_COMPILE_CMAKE_ARGS ${CROSS_COMPILE_CMAKE_ARGS} "-DCMAKE_SYSTEM_PROCESSOR=${XNNPACK_TARGET_PROCESSOR}")
endif()

include(ExternalProject)

set(GOOGLE_XNNPACK_PROJECT extern_xnnpack)
set(GOOGLE_XNNPACK_SOURCES_DIR ${THIRD_PARTY_PATH}/xnnpack)
set(GOOGLE_XNNPACK_INSTALL_DIR ${THIRD_PARTY_PATH}/install/xnnpack)
set(GOOGLE_XNNPACK_BUILD_COMMAND $(MAKE) -j)

ExternalProject_Add(
  ${GOOGLE_XNNPACK_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  GIT_REPOSITORY      "https://github.com/google/XNNPACK.git"
  GIT_TAG             ${NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG}
  GIT_CONFIG          user.name=anonymous user.email=anonymous@anonymous.com
  SOURCE_DIR          ${GOOGLE_XNNPACK_SOURCES_DIR}
  PREFIX              ${GOOGLE_XNNPACK_INSTALL_DIR}
  PATCH_COMMAND       ${GOOGLE_XNNPACK_PATCH_COMMAND}
  BUILD_COMMAND       ${GOOGLE_XNNPACK_BUILD_COMMAND}
  CMAKE_ARGS          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                      -DXNNPACK_BUILD_TESTS=OFF
                      -DXNNPACK_BUILD_BENCHMARKS=OFF
                      -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}
                      -DCMAKE_INSTALL_PREFIX=${GOOGLE_XNNPACK_INSTALL_DIR}
                      ${CROSS_COMPILE_CMAKE_ARGS}
)

set(GOOGLE_XNNPACK_LIBRARY "${GOOGLE_XNNPACK_INSTALL_DIR}/lib/libXNNPACK.a")

add_library(xnnpack_libs STATIC IMPORTED GLOBAL)
set_property(TARGET xnnpack_libs PROPERTY IMPORTED_LOCATION ${GOOGLE_XNNPACK_LIBRARY})
add_dependencies(xnnpack_libs ${GOOGLE_XNNPACK_PROJECT})

include_directories(${GOOGLE_XNNPACK_INSTALL_DIR}/include)

set(${DEVICE_NAME}_deps xnnpack_libs)
