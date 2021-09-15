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

if(NOT LITE_WITH_CAMBRICON_MLU)
  return()
endif()

if(NOT DEFINED CAMBRICON_MLU_SDK_ROOT)
  set(CAMBRICON_MLU_SDK_ROOT $ENV{CAMBRICON_MLU_SDK_ROOT})
  if(NOT CAMBRICON_MLU_SDK_ROOT)
    message(FATAL_ERROR "Must set CAMBRICON_MLU_SDK_ROOT or env CAMBRICON_MLU_SDK_ROOT when LITE_WITH_CAMBRICON_MLU=ON")
  endif()
endif()

message(STATUS "CAMBRICON_MLU_SDK_ROOT: ${CAMBRICON_MLU_SDK_ROOT}")
find_path(MAGICMIND_INC NAMES interface_network.h
  PATHS ${CAMBRICON_MLU_SDK_ROOT}/include NO_DEFAULT_PATH)
if(NOT MAGICMIND_INC)
  message(FATAL_ERROR "Can not find interface_network.h in ${CAMBRICON_MLU_SDK_ROOT}/include")
endif()

include_directories(${MAGICMIND_INC})

find_library(MAGICMIND_LIB_FILE NAMES magicmind_core
    PATHS ${CAMBRICON_MLU_SDK_ROOT}/lib64)

if(NOT MAGICMIND_LIB_FILE)
  message(FATAL_ERROR "Can not find MAGICMIND_LIB_FILE in ${CAMBRICON_MLU_SDK_ROOT}")
else()
  message(STATUS "Found MAGICMIND Library: ${MAGICMIND_LIB_FILE}")
  add_library(magicmind_lib SHARED IMPORTED GLOBAL)
  set_property(TARGET magicmind_lib PROPERTY IMPORTED_LOCATION ${MAGICMIND_LIB_FILE})
endif()

set(cambricon_mlu_libs magicmind_lib CACHE INTERNAL "cambricon mlu libs")
