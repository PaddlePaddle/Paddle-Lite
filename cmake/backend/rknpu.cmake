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

if(NOT LITE_WITH_RKNPU)
  return()
endif()

if(NOT DEFINED RKNPU_DDK_ROOT)
    set(RKNPU_DDK_ROOT $ENV{RKNPU_DDK_ROOT})
    if(NOT RKNPU_DDK_ROOT)
        message(FATAL_ERROR "Must set RKNPU_DDK_ROOT or env RKNPU_DDK_ROOT when LITE_WITH_RKNPU=ON")
    endif()
endif()

message(STATUS "RKNPU_DDK_ROOT: ${RKNPU_DDK_ROOT}")
find_path(RKNPU_DDK_INC NAMES rknpu/rknpu_pub.h
  PATHS ${RKNPU_DDK_ROOT}/include/  NO_DEFAULT_PATH)
if(NOT RKNPU_DDK_INC)
  message(FATAL_ERROR "Can not find rknpu_pub.h in ${RKNPU_DDK_ROOT}/include")
endif()

include_directories("${RKNPU_DDK_ROOT}/include")

set(RKNPU_SUB_LIB_PATH "lib64")
if(ARM_TARGET_ARCH_ABI STREQUAL "armv8")
    set(RKNPU_SUB_LIB_PATH "lib64")
endif()

if(ARM_TARGET_ARCH_ABI STREQUAL "armv7hf")
    set(RKNPU_SUB_LIB_PATH "lib")
endif()

find_library(RKNPU_DDK_FILE NAMES rknpu_ddk
  PATHS ${RKNPU_DDK_ROOT}/${RKNPU_SUB_LIB_PATH})

if(NOT RKNPU_DDK_FILE)
  message(FATAL_ERROR "Can not find RKNPU_DDK_FILE in ${RKNPU_DDK_ROOT}/${RKNPU_SUB_LIB_PATH}")
else()
  message(STATUS "Found RKNPU_DDK_FILE  Library: ${RKNPU_DDK_FILE}")
  add_library(rknpu_ddk  SHARED IMPORTED GLOBAL)
  set_property(TARGET rknpu_ddk PROPERTY IMPORTED_LOCATION ${RKNPU_DDK_FILE})
endif()

set(rknpu_runtime_libs rknpu_ddk  CACHE INTERNAL "rknpu ddk runtime libs")
