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

if(NOT LITE_WITH_NPU)
  return()
endif()

if(NOT DEFINED NPU_DDK_ROOT)
  set(NPU_DDK_ROOT $ENV{NPU_DDK_ROOT})
  if(NOT NPU_DDK_ROOT)
    message(FATAL_ERROR "Must set NPU_DDK_ROOT or env NPU_DDK_ROOT when LITE_WITH_NPU=ON")
  endif()
endif()

message(STATUS "NPU_DDK_ROOT: ${NPU_DDK_ROOT}")
find_path(NPU_DDK_INC NAMES HiAiModelManagerService.h
  PATHS ${NPU_DDK_ROOT}/include/
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT NPU_DDK_INC)
  message(FATAL_ERROR "Can not find HiAiModelManagerService.h in ${NPU_DDK_ROOT}/include")
endif()

include_directories("${NPU_DDK_ROOT}/include")

set(NPU_SUB_LIB_PATH "lib64")
if(ARM_TARGET_ARCH_ABI STREQUAL "armv8")
  set(NPU_SUB_LIB_PATH "lib64")
endif()

if(ARM_TARGET_ARCH_ABI STREQUAL "armv7")
  set(NPU_SUB_LIB_PATH "lib")
endif()

find_library(NPU_DDK_HIAI_FILE NAMES hiai
  PATHS ${NPU_DDK_ROOT}/${NPU_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)

find_library(NPU_DDK_IR_FILE NAMES hiai_ir
  PATHS ${NPU_DDK_ROOT}/${NPU_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)

find_library(NPU_DDK_IR_BUILD_FILE NAMES hiai_ir_build
  PATHS ${NPU_DDK_ROOT}/${NPU_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)

# Added in HiAI DDK 320 or later version
find_library(NPU_DDK_HCL_FILE NAMES hcl
  PATHS ${NPU_DDK_ROOT}/${NPU_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)

if(NOT NPU_DDK_HIAI_FILE)
  message(FATAL_ERROR "Can not find NPU_DDK_HIAI_FILE in ${NPU_DDK_ROOT}")
else()
  message(STATUS "Found NPU_DDK HIAI Library: ${NPU_DDK_HIAI_FILE}")
  add_library(npu_ddk_hiai SHARED IMPORTED GLOBAL)
  set_property(TARGET npu_ddk_hiai PROPERTY IMPORTED_LOCATION ${NPU_DDK_HIAI_FILE})
endif()

if(NOT NPU_DDK_IR_FILE)
  message(FATAL_ERROR "Can not find NPU_DDK_IR_FILE in ${NPU_DDK_ROOT}")
else()
  message(STATUS "Found NPU_DDK IR Library: ${NPU_DDK_IR_FILE}")
  add_library(npu_ddk_ir SHARED IMPORTED GLOBAL)
  set_property(TARGET npu_ddk_ir PROPERTY IMPORTED_LOCATION ${NPU_DDK_IR_FILE})
endif()

if(NOT NPU_DDK_IR_BUILD_FILE)
  message(FATAL_ERROR "Can not find NPU_DDK_IR_BUILD_FILE in ${NPU_DDK_ROOT}")
else()
  message(STATUS "Found NPU_DDK IR_BUILD Library: ${NPU_DDK_IR_BUILD_FILE}")
  add_library(npu_ddk_ir_build SHARED IMPORTED GLOBAL)
  set_property(TARGET npu_ddk_ir_build PROPERTY IMPORTED_LOCATION ${NPU_DDK_IR_BUILD_FILE})
endif()

if(NOT NPU_DDK_HCL_FILE)
# message(FATAL_ERROR "Can not find NPU_DDK_HCL_FILE in ${NPU_DDK_ROOT}")
else()
  message(STATUS "Found NPU_DDK HCL Library: ${NPU_DDK_HCL_FILE}")
  add_library(npu_ddk_hcl SHARED IMPORTED GLOBAL)
  set_property(TARGET npu_ddk_hcl PROPERTY IMPORTED_LOCATION ${NPU_DDK_HCL_FILE})
endif()

set(npu_runtime_libs npu_ddk_hiai npu_ddk_hcl CACHE INTERNAL "npu ddk runtime libs")
set(npu_builder_libs npu_ddk_ir npu_ddk_ir_build CACHE INTERNAL "npu ddk builder libs")
