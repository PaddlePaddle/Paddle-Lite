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

if(NOT DEFINED NNADAPTER_EEASYTECH_NPU_SDK_ROOT)
  set(NNADAPTER_EEASYTECH_NPU_SDK_ROOT $ENV{NNADAPTER_EEASYTECH_NPU_SDK_ROOT})
endif()
if(NOT NNADAPTER_EEASYTECH_NPU_SDK_ROOT)
  message(FATAL_ERROR "Must set NNADAPTER_EEASYTECH_NPU_SDK_ROOT or env NNADAPTER_EEASYTECH_NPU_SDK_ROOT when NNADAPTER_WITH_EEASYTECH_NPU=ON")
endif()
message(STATUS "NNADAPTER_EEASYTECH_NPU_SDK_ROOT: ${NNADAPTER_EEASYTECH_NPU_SDK_ROOT}")

find_path(EEASYTECH_NPU_SDK_INC NAMES eznpu/eznpu_pub.h
  PATHS ${NNADAPTER_EEASYTECH_NPU_SDK_ROOT}/include/
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT EEASYTECH_NPU_SDK_INC)
  message(FATAL_ERROR "Missing eznpu_pub.h in ${NNADAPTER_EEASYTECH_NPU_SDK_ROOT}/include")
endif()

include_directories("${NNADAPTER_EEASYTECH_NPU_SDK_ROOT}/include")

set(EEASYTECH_NPU_SDK_SUB_LIB_PATH "lib64")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(EEASYTECH_NPU_SDK_SUB_LIB_PATH "lib64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm")
  set(EEASYTECH_NPU_SDK_SUB_LIB_PATH "lib")
else()
  message(FATAL_ERROR "${CMAKE_SYSTEM_PROCESSOR} isn't supported by EEASYTECH NPU SDK.")
endif()

find_library(EEASYTECH_NPU_SDK_DDK_FILE NAMES eznpu_ddk
  PATHS ${NNADAPTER_EEASYTECH_NPU_SDK_ROOT}/${EEASYTECH_NPU_SDK_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT EEASYTECH_NPU_SDK_DDK_FILE)
  message(FATAL_ERROR "Missing eznpu_ddk in ${NNADAPTER_EEASYTECH_NPU_SDK_ROOT}/${EEASYTECH_NPU_SDK_SUB_LIB_PATH}")
endif()

add_library(eznpu_ddk SHARED IMPORTED GLOBAL)
set_property(TARGET eznpu_ddk PROPERTY IMPORTED_LOCATION ${EEASYTECH_NPU_SDK_DDK_FILE})

set(DEPS ${DEPS} eznpu_ddk)

