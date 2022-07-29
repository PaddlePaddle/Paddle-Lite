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

if(NOT DEFINED NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT)
  set(NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT $ENV{NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT})
endif()
if(NOT NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT)
  message(FATAL_ERROR "Must set NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT or env NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT when NNADAPTER_WITH_HUAWEI_KIRIN_NPU=ON")
endif()
message(STATUS "NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT: ${NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT}")

find_path(HUAWEI_KIRIN_NPU_SDK_INC NAMES HiAiModelManagerService.h
  PATHS ${NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT}/include/
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_KIRIN_NPU_SDK_INC)
  message(FATAL_ERROR "Missing HiAiModelManagerService.h in ${NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT}/include")
endif()

include_directories("${NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT}/include")

set(HUAWEI_KIRIN_NPU_SDK_SUB_LIB_PATH "lib64")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(HUAWEI_KIRIN_NPU_SDK_SUB_LIB_PATH "lib64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7-a")
  set(HUAWEI_KIRIN_NPU_SDK_SUB_LIB_PATH "lib")
else()
  message(FATAL_ERROR "${CMAKE_SYSTEM_PROCESSOR} isn't supported by Huawei Kirin NPU SDK.")
endif()

# libhiai.so
find_library(HUAWEI_KIRIN_NPU_SDK_HIAI_FILE NAMES hiai
  PATHS ${NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT}/${HUAWEI_KIRIN_NPU_SDK_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_KIRIN_NPU_SDK_HIAI_FILE)
  message(FATAL_ERROR "Missing libhiai.so in ${NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT}/${HUAWEI_KIRIN_NPU_SDK_SUB_LIB_PATH}")
endif()
add_library(hiai SHARED IMPORTED GLOBAL)
set_property(TARGET hiai PROPERTY IMPORTED_LOCATION ${HUAWEI_KIRIN_NPU_SDK_HIAI_FILE})

# libhiai_ir.so
find_library(HUAWEI_KIRIN_NPU_SDK_HIAI_IR_FILE NAMES hiai_ir
  PATHS ${NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT}/${HUAWEI_KIRIN_NPU_SDK_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_KIRIN_NPU_SDK_HIAI_IR_FILE)
  message(FATAL_ERROR "Missing libhiai_ir.so in ${NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT}/${HUAWEI_KIRIN_NPU_SDK_SUB_LIB_PATH}")
endif()
add_library(hiai_ir SHARED IMPORTED GLOBAL)
set_property(TARGET hiai_ir PROPERTY IMPORTED_LOCATION ${HUAWEI_KIRIN_NPU_SDK_HIAI_IR_FILE})

# libhiai_ir_build.so
find_library(HUAWEI_KIRIN_NPU_SDK_HIAI_IR_BUILD_FILE NAMES hiai_ir_build
  PATHS ${NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT}/${HUAWEI_KIRIN_NPU_SDK_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_KIRIN_NPU_SDK_HIAI_IR_BUILD_FILE)
  message(FATAL_ERROR "Missing libhiai_ir_build.so in ${NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT}/${HUAWEI_KIRIN_NPU_SDK_SUB_LIB_PATH}")
endif()
add_library(hiai_ir_build SHARED IMPORTED GLOBAL)
set_property(TARGET hiai_ir_build PROPERTY IMPORTED_LOCATION ${HUAWEI_KIRIN_NPU_SDK_HIAI_IR_BUILD_FILE})

# libhcl.so(hiai ddk 320 or later)
find_library(HUAWEI_KIRIN_NPU_SDK_HCL_FILE NAMES hcl
  PATHS ${NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT}/${HUAWEI_KIRIN_NPU_SDK_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_KIRIN_NPU_SDK_HCL_FILE)
  message(FATAL_ERROR "Missing libhcl.so in ${NNADAPTER_HUAWEI_KIRIN_NPU_SDK_ROOT}/${HUAWEI_KIRIN_NPU_SDK_SUB_LIB_PATH}")
endif()
add_library(hcl SHARED IMPORTED GLOBAL)
set_property(TARGET hcl PROPERTY IMPORTED_LOCATION ${HUAWEI_KIRIN_NPU_SDK_HCL_FILE})

set(DEPS ${DEPS} hiai_ir hiai_ir_build hiai hcl)
