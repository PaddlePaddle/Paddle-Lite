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
  message(STATUS "Must set NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG or env NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG when NNADAPTER_WITH_GOOGLE_XNNPACK=ON"
                 "Set NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG=${NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG} by default.")
endif()
message(STATUS "NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG: ${NNADAPTER_GOOGLE_XNNPACK_SRC_GIT_TAG}")

if(CMAKE_SYSTEM_NAME MATCHES "Android")
  if(NOT ANDROID_NDK_REVISION)
    # Get the Android NDK version, refer to https://github.com/Kitware/CMake/blob/d8f95471c7d779c8bb1606fbfff664f1dad6cde1/Modules/Platform/Android-Determine.cmake#L231 for more details.
    if(CMAKE_ANDROID_NDK AND EXISTS "${CMAKE_ANDROID_NDK}/source.properties")
      file(READ "${CMAKE_ANDROID_NDK}/source.properties" ANDROID_NDK_SOURCE_PROPERTIES)
      set(ANDROID_NDK_REVISION_REGEX "^Pkg\\.Desc = Android NDK\nPkg\\.Revision = ([0-9]+)\\.([0-9]+)\\.([0-9]+)(-beta([0-9]+))?")
      if(NOT ANDROID_NDK_SOURCE_PROPERTIES MATCHES "${ANDROID_NDK_REVISION_REGEX}")
        string(REPLACE "\n" "\n  " ANDROID_NDK_SOURCE_PROPERTIES "${ANDROID_NDK_SOURCE_PROPERTIES}")
        message(FATAL_ERROR "Android: Failed to parse NDK revision from:\n ${CMAKE_ANDROID_NDK}/source.properties\n with content:\n  ${ANDROID_NDK_SOURCE_PROPERTIES}")
      endif()
    endif()
    set(ANDROID_NDK_MAJOR "${CMAKE_MATCH_1}")
    set(ANDROID_NDK_MINOR "${CMAKE_MATCH_2}")
    set(ANDROID_NDK_BUILD "${CMAKE_MATCH_3}")
    set(ANDROID_NDK_BETA "${CMAKE_MATCH_5}")
    if(ANDROID_NDK_BETA STREQUAL "")
      set(ANDROID_NDK_BETA "0")
    endif()
    set(ANDROID_NDK_REVISION "${ANDROID_NDK_MAJOR}.${ANDROID_NDK_MINOR}.${ANDROID_NDK_BUILD}${CMAKE_MATCH_4}")
  endif()
  if(ANDROID_NDK_REVISION VERSION_LESS 19)
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

# Hack the XNNPACK and change the symbol visibility from 'internal' to 'default' to fix the compilation error 'internal symbol `xnn_x16_transpose_ukernel__8x8_reuse_dec_zip_neon' isn't defined'
set(GOOGLE_XNNPACK_PATCH_COMMAND sed -e "s/__attribute__((__visibility__(\"internal\")))/__attribute__((__visibility__(\"default\")))/g" -i src/xnnpack/common.h)

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
                      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                      -DXNNPACK_BUILD_TESTS=OFF
                      -DXNNPACK_BUILD_BENCHMARKS=OFF
                      -DXNNPACK_LIBRARY_TYPE=static
                      -DXNNPACK_ENABLE_JIT=ON
                      -DCMAKE_INSTALL_PREFIX=${GOOGLE_XNNPACK_INSTALL_DIR}
                      ${CROSS_COMPILE_CMAKE_ARGS}
)

add_library(clog_lib STATIC IMPORTED GLOBAL)
set_property(TARGET clog_lib PROPERTY IMPORTED_LOCATION ${GOOGLE_XNNPACK_INSTALL_DIR}/lib/libclog.a)
add_dependencies(clog_lib ${GOOGLE_XNNPACK_PROJECT})

add_library(cpuinfo_lib STATIC IMPORTED GLOBAL)
set_property(TARGET cpuinfo_lib PROPERTY IMPORTED_LOCATION ${GOOGLE_XNNPACK_INSTALL_DIR}/lib/libcpuinfo.a)
add_dependencies(cpuinfo_lib ${GOOGLE_XNNPACK_PROJECT})

add_library(pthreadpool_lib STATIC IMPORTED GLOBAL)
set_property(TARGET pthreadpool_lib PROPERTY IMPORTED_LOCATION ${GOOGLE_XNNPACK_INSTALL_DIR}/lib/libpthreadpool.a)
add_dependencies(pthreadpool_lib ${GOOGLE_XNNPACK_PROJECT})

add_library(xnnpack_lib STATIC IMPORTED GLOBAL)
set_property(TARGET xnnpack_lib PROPERTY IMPORTED_LOCATION ${GOOGLE_XNNPACK_INSTALL_DIR}/lib/libXNNPACK.a)
add_dependencies(xnnpack_lib ${GOOGLE_XNNPACK_PROJECT})

include_directories(${GOOGLE_XNNPACK_INSTALL_DIR}/include)

set(DEPS ${DEPS} clog_lib cpuinfo_lib pthreadpool_lib xnnpack_lib)
