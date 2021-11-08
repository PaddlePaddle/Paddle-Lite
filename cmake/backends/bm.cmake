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

if(NOT LITE_WITH_BM)
  return()
endif()

if(NOT DEFINED BM_SDK_ROOT)
    set(BM_SDK_ROOT $ENV{BM_SDK_ROOT})
    if(NOT BM_SDK_ROOT)
        message(FATAL_ERROR "Must set BM_SDK_ROOT or env BM_SDK_ROOT when LITE_WITH_BM=ON")
    endif()
endif()

set(BM_SDK_CPLIB_RPATH ${BM_SDK_ROOT}/lib/bmcompiler)
set(BM_SDK_LIB_RPATH ${BM_SDK_ROOT}/lib/bmnn/pcie)

message(STATUS "BM_SDK_ROOT: ${BM_SDK_ROOT}")
find_path(BM_SDK_INC NAMES bmruntime_interface.h
  PATHS ${BM_SDK_ROOT}/include/bmruntime NO_DEFAULT_PATH)
if(NOT BM_SDK_INC)
  message(FATAL_ERROR "Can not find bmruntime_interface.h in ${BM_SDK_ROOT}/include")
endif()

include_directories("${BM_SDK_ROOT}/include/bmruntime")
include_directories("${BM_SDK_ROOT}/include/bmlib")
include_directories("${BM_SDK_ROOT}/include/bmcompiler")
include_directories("${BM_SDK_ROOT}/include/bmcpu")
include_directories("${BM_SDK_ROOT}/include/bmlog")

find_library(BM_SDK_RT_LIB NAMES bmrt
  PATHS ${BM_SDK_LIB_RPATH})
if(NOT BM_SDK_RT_LIB)
  message(FATAL_ERROR "Can not find bmrt Library in ${BM_SDK_ROOT}")
else()
  message(STATUS "Found bmrt Library: ${BM_SDK_RT_LIB}")
endif()

find_library(BM_SDK_BM_LIB NAMES bmlib
  PATHS ${BM_SDK_LIB_RPATH})
if(NOT BM_SDK_BM_LIB)
  message(FATAL_ERROR "Can not find bmlib Library in ${BM_SDK_ROOT}")
else()
  message(STATUS "Found bmlib Library: ${BM_SDK_BM_LIB}")
endif()

find_library(BM_SDK_COMPILER_LIB NAMES bmcompiler
  PATHS ${BM_SDK_CPLIB_RPATH})
if(NOT BM_SDK_COMPILER_LIB)
  message(FATAL_ERROR "Can not find bmcompiler Library in ${BM_SDK_ROOT}")
else()
  message(STATUS "Found bmcompiler Library: ${BM_SDK_COMPILER_LIB}")
endif()

find_library(BM_SDK_CPU_LIB NAMES bmcpu
  PATHS ${BM_SDK_LIB_RPATH})
if(NOT BM_SDK_CPU_LIB)
  message(FATAL_ERROR "Can not find bmcpu Library in ${BM_SDK_ROOT}")
else()
  message(STATUS "Found bmcpu Library: ${BM_SDK_CPU_LIB}")
endif()

set(bm_runtime_libs bmrt bmlib bmcompiler bmcpu CACHE INTERNAL "bm runtime libs")
set(bm_builder_libs bmrt bmlib bmcompiler bmcpu CACHE INTERNAL "bm builder libs")
