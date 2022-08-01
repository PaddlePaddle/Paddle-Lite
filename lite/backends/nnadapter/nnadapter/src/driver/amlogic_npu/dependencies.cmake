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

if(NOT DEFINED NNADAPTER_AMLOGIC_NPU_SDK_ROOT)
  set(NNADAPTER_AMLOGIC_NPU_SDK_ROOT $ENV{NNADAPTER_AMLOGIC_NPU_SDK_ROOT})
endif()
if(NOT NNADAPTER_AMLOGIC_NPU_SDK_ROOT)
  message(FATAL_ERROR "Must set NNADAPTER_AMLOGIC_NPU_SDK_ROOT or env NNADAPTER_AMLOGIC_NPU_SDK_ROOT when NNADAPTER_WITH_AMLOGIC_NPU=ON")
endif()
message(STATUS "NNADAPTER_AMLOGIC_NPU_SDK_ROOT: ${NNADAPTER_AMLOGIC_NPU_SDK_ROOT}")

find_path(AMLOGIC_NPU_SDK_INC NAMES amlnpu/amlnpu_pub.h
  PATHS ${NNADAPTER_AMLOGIC_NPU_SDK_ROOT}/include/
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT AMLOGIC_NPU_SDK_INC)
  message(FATAL_ERROR "Missing amlnpu_pub.h in ${NNADAPTER_AMLOGIC_NPU_SDK_ROOT}/include")
endif()

include_directories("${NNADAPTER_AMLOGIC_NPU_SDK_ROOT}/include")

set(AMLOGIC_NPU_SDK_SUB_LIB_PATH "lib64")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(AMLOGIC_NPU_SDK_SUB_LIB_PATH "lib64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm")
  set(AMLOGIC_NPU_SDK_SUB_LIB_PATH "lib")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7-a")
  set(AMLOGIC_NPU_SDK_SUB_LIB_PATH "lib")
else()
  message(FATAL_ERROR "${CMAKE_SYSTEM_PROCESSOR} isn't supported by Amlogic NPU SDK.")
endif()

find_library(AMLOGIC_NPU_SDK_DDK_FILE NAMES amlnpu_ddk
  PATHS ${NNADAPTER_AMLOGIC_NPU_SDK_ROOT}/${AMLOGIC_NPU_SDK_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT AMLOGIC_NPU_SDK_DDK_FILE)
  message(FATAL_ERROR "Missing amlnpu_ddk in ${NNADAPTER_AMLOGIC_NPU_SDK_ROOT}/${AMLOGIC_NPU_SDK_SUB_LIB_PATH}")
endif()
add_library(amlnpu_ddk SHARED IMPORTED GLOBAL)
set_property(TARGET amlnpu_ddk PROPERTY IMPORTED_LOCATION ${AMLOGIC_NPU_SDK_DDK_FILE})

set(DEPS ${DEPS} amlnpu_ddk)
