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

if(NOT LITE_WITH_NNA)
  return()
endif()

if(NOT DEFINED IMGNNA_DDK_ROOT)
  set(IMGNNA_DDK_ROOT $ENV{IMGNNA_DDK_ROOT})
  if(NOT IMGNNA_DDK_ROOT)
    message(FATAL_ERROR "Must set IMGNNA_DDK_ROOT or env IMGNNA_DDK_ROOT when LITE_WITH_IMGNNA=ON")
  endif()
endif()

message(STATUS "IMGNNA_DDK_ROOT: ${IMGNNA_DDK_ROOT}")
find_path(IMGNNA_DDK_INC NAMES imgdnn.h
  PATHS ${IMGNNA_DDK_ROOT}/include/imgdnn NO_DEFAULT_PATH)
if(NOT IMGNNA_DDK_INC)
  message(FATAL_ERROR "Can not find imgdnn.h in ${IMGNNA_DDK_ROOT}/include")
endif()

#include_directories("${IMGNNA_DDK_ROOT}/include")
include_directories(${IMGNNA_DDK_INC})

#set(IMGNNA_SUB_LIB_PATH "lib64")
#if(ARM_TARGET_ARCH_ABI STREQUAL "armv8")
#    set(IMGNNA_SUB_LIB_PATH "lib64")
#endif()
#if(ARM_TARGET_ARCH_ABI STREQUAL "armv7")
#    set(IMGNNA_SUB_LIB_PATH "lib")
#endif()
set(IMGNNA_LIB_PATH "lib")

find_library(IMGNNA_DDK_IMGDNN_FILE NAMES imgdnn
  PATHS ${IMGNNA_DDK_ROOT}/${IMGNNA_LIB_PATH})

if(NOT IMGNNA_DDK_IMGDNN_FILE)
  message(FATAL_ERROR "Can not find IMGNNA_DDK_IMGDNN_FILE in ${IMGNNA_DDK_ROOT}")
else()
  message(STATUS "Found IMGNNA_DDK IMGDNN Library: ${IMGNNA_DDK_IMGDNN_FILE}")
  add_library(nna_ddk_imgdnn SHARED IMPORTED GLOBAL)
  set_property(TARGET nna_ddk_imgdnn PROPERTY IMPORTED_LOCATION ${IMGNNA_DDK_IMGDNN_FILE})
endif()


find_library(IMGNNA_DDK_RUNTIME_FILE NAMES nnasession
  PATHS ${IMGNNA_DDK_ROOT}/${IMGNNA_LIB_PATH})

if(NOT IMGNNA_DDK_RUNTIME_FILE)
  message(FATAL_ERROR "Can not find IMGNNA_DDK_RUNTIME_FILE in ${IMGNNA_DDK_ROOT}")
else()
  message(STATUS "Found IMGNNA_DDK RUNTIME Library: ${IMGNNA_DDK_RUNTIME_FILE}")
  add_library(nna_ddk_runtime SHARED IMPORTED GLOBAL)
  set_property(TARGET nna_ddk_runtime PROPERTY IMPORTED_LOCATION ${IMGNNA_DDK_RUNTIME_FILE})
endif()

set(nna_runtime_libs nna_ddk_runtime CACHE INTERNAL "imgnna ddk runtime libs")
set(nna_builder_libs nna_ddk_imgdnn CACHE INTERNAL "imgnna ddk builder libs")
