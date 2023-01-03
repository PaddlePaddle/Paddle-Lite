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

if(NOT DEFINED NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT)
  set(NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT $ENV{NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT})
endif()
if(NOT NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT)
  message(FATAL_ERROR "Must set NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT or env NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT when NNADAPTER_WITH_HUAWEI_ASCEND_NPU=ON")
endif()
message(STATUS "NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT: ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}")

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
else()
  message(FATAL_ERROR "${CMAKE_SYSTEM_PROCESSOR} isn't supported by Huawei Ascend NPU SDK.")
endif()

macro(find_huawei_ascend_npu_sdk_version huawei_ascend_npu_sdk_version_info) 
    file(READ ${huawei_ascend_npu_sdk_version_info} HUAWEI_ASCEND_NPU_SDK_VERSION_CONTENTS)
    string(REGEX MATCH "version=([0-9]+\.[0-9]+\.(RC)?[0-9]+\.(alpha)?[0-9]*)" HUAWEI_ASCEND_NPU_SDK_VERSION "${HUAWEI_ASCEND_NPU_SDK_VERSION_CONTENTS}")
    string(REGEX REPLACE "version=([0-9]+\.[0-9]+\.(RC)?[0-9]+\.(alpha)?[0-9]*)" "\\1" HUAWEI_ASCEND_NPU_SDK_VERSION "${HUAWEI_ASCEND_NPU_SDK_VERSION}")
    if(NOT HUAWEI_ASCEND_NPU_SDK_VERSION)
        message(FATAL_ERROR "Failed to extract the CANN version, please set the correct NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT")
    else()
        message(STATUS "Current Huawei Ascend NPU SDK version is ${HUAWEI_ASCEND_NPU_SDK_VERSION}")
    endif()
    string(REGEX MATCHALL "[0-9]+" CANN_VERSION_NUM_LIST "${HUAWEI_ASCEND_NPU_SDK_VERSION}")
    list(LENGTH CANN_VERSION_NUM_LIST CANN_VERSION_NUM_LIST_LENGTH)
    if(CANN_VERSION_NUM_LIST_LENGTH GREATER_EQUAL 3)
      list(GET CANN_VERSION_NUM_LIST 0 CANN_MAJOR_VERSION)
      list(GET CANN_VERSION_NUM_LIST 1 CANN_MINOR_VERSION)
      list(GET CANN_VERSION_NUM_LIST 2 CANN_PATCH_VERSION)
    endif()
    message(STATUS "NNADAPTER_HUAWEI_ASCEND_NPU_CANN_MAJOR_VERSION: ${CANN_MAJOR_VERSION}")
    message(STATUS "NNADAPTER_HUAWEI_ASCEND_NPU_CANN_MINOR_VERSION: ${CANN_MINOR_VERSION}")
    message(STATUS "NNADAPTER_HUAWEI_ASCEND_NPU_CANN_PATCH_VERSION: ${CANN_PATCH_VERSION}")
    add_definitions(-DNNADAPTER_HUAWEI_ASCEND_NPU_CANN_MAJOR_VERSION=${CANN_MAJOR_VERSION})
    add_definitions(-DNNADAPTER_HUAWEI_ASCEND_NPU_CANN_MINOR_VERSION=${CANN_MINOR_VERSION})
    add_definitions(-DNNADAPTER_HUAWEI_ASCEND_NPU_CANN_PATCH_VERSION=${CANN_PATCH_VERSION})
endmacro()

if (LITE_WITH_ARM)
  set(HUAWEI_ASCEND_NPU_SDK_INSTALL_DIR ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/arm64-linux)
else()
  set(HUAWEI_ASCEND_NPU_SDK_INSTALL_DIR ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/x86_64-linux)
endif()
find_huawei_ascend_npu_sdk_version(${HUAWEI_ASCEND_NPU_SDK_INSTALL_DIR}/ascend_toolkit_install.info)

include_directories("${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/acllib/include")
include_directories("${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/include")
include_directories("${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/opp")

# ACL libraries
# libascendcl.so 
find_library(HUAWEI_ASCEND_NPU_SDK_ACL_ASCENDCL_FILE NAMES ascendcl
  PATHS ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/acllib/lib64
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_ASCEND_NPU_SDK_ACL_ASCENDCL_FILE)
  message(FATAL_ERROR "Missing libascendcl.so in ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/acllib/lib64")
endif()
add_library(acl_ascendcl SHARED IMPORTED GLOBAL)
set_property(TARGET acl_ascendcl PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_SDK_ACL_ASCENDCL_FILE})

# ATC libraries
# libge_compiler.so
find_library(HUAWEI_ASCEND_NPU_SDK_ATC_GE_COMPILER_FILE NAMES ge_compiler
  PATHS ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_ASCEND_NPU_SDK_ATC_GE_COMPILER_FILE)
  message(FATAL_ERROR "Missing libge_compiler.so in ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64")
endif()
add_library(atc_ge_compiler SHARED IMPORTED GLOBAL)
set_property(TARGET atc_ge_compiler PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_SDK_ATC_GE_COMPILER_FILE})

# ACL libs should before ATC libs
set(DEPS ${DEPS} acl_ascendcl atc_ge_compiler)
