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
  if(NOT NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT)
    message(FATAL_ERROR "Must set NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT or env NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT when NNADAPTER_WITH_HUAWEI_ASCEND_NPU=ON")
  endif()
endif()

message(STATUS "NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT: ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}")

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
else()
  message(FATAL_ERROR "${CMAKE_SYSTEM_PROCESSOR} isn't supported by Huawei Ascend NPU SDK.")
endif()

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

# libruntime.so
find_library(HUAWEI_ASCEND_NPU_SDK_ACL_RUNTIME_FILE NAMES runtime
  PATHS ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/acllib/lib64
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_ASCEND_NPU_SDK_ACL_RUNTIME_FILE)
  message(FATAL_ERROR "Missing libruntime.so in ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/acllib/lib64")
endif()
add_library(acl_runtime SHARED IMPORTED GLOBAL)
set_property(TARGET acl_runtime PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_SDK_ACL_RUNTIME_FILE})

# ATC libraries
# libregister.so
find_library(HUAWEI_ASCEND_NPU_SDK_ATC_REGISTER_FILE NAMES register
  PATHS ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_ASCEND_NPU_SDK_ATC_REGISTER_FILE)
  message(FATAL_ERROR "Missing libregister.so in ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64")
endif()
add_library(atc_register SHARED IMPORTED GLOBAL)
set_property(TARGET atc_register PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_SDK_ATC_REGISTER_FILE})

# libascend_protobuf.so
find_library(HUAWEI_ASCEND_NPU_SDK_ATC_ASCEND_PROTOBUF_FILE NAMES ascend_protobuf
  PATHS ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_ASCEND_NPU_SDK_ATC_ASCEND_PROTOBUF_FILE)
  message(FATAL_ERROR "Missing libascend_protobuf.so in ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64")
endif()
add_library(atc_ascend_protobuf SHARED IMPORTED GLOBAL)
set_property(TARGET atc_ascend_protobuf PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_SDK_ATC_ASCEND_PROTOBUF_FILE})

# libgraph.so
find_library(HUAWEI_ASCEND_NPU_SDK_ATC_GRAPH_FILE NAMES graph
  PATHS ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_ASCEND_NPU_SDK_ATC_GRAPH_FILE)
  message(FATAL_ERROR "Missing libgraph.so in ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64")
endif()
add_library(atc_graph SHARED IMPORTED GLOBAL)
set_property(TARGET atc_graph PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_SDK_ATC_GRAPH_FILE})

# libge_compiler.so
find_library(HUAWEI_ASCEND_NPU_SDK_ATC_GE_COMPILER_FILE NAMES ge_compiler
  PATHS ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_ASCEND_NPU_SDK_ATC_GE_COMPILER_FILE)
  message(FATAL_ERROR "Missing libge_compiler.so in ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64")
endif()
add_library(atc_ge_compiler SHARED IMPORTED GLOBAL)
set_property(TARGET atc_ge_compiler PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_SDK_ATC_GE_COMPILER_FILE})

# libge_common.so
find_library(HUAWEI_ASCEND_NPU_SDK_ATC_GE_COMMON_FILE NAMES ge_common
  PATHS ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_ASCEND_NPU_SDK_ATC_GE_COMMON_FILE)
  message(FATAL_ERROR "Missing libge_common.so in ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64")
endif()
add_library(atc_ge_common SHARED IMPORTED GLOBAL)
set_property(TARGET atc_ge_common PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_SDK_ATC_GE_COMMON_FILE})

# libresource.so
find_library(HUAWEI_ASCEND_NPU_SDK_ATC_RESOURCE_FILE NAMES resource
  PATHS ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_ASCEND_NPU_SDK_ATC_RESOURCE_FILE)
  message(FATAL_ERROR "Missing libresource.so in ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64")
endif()
add_library(atc_resource SHARED IMPORTED GLOBAL)
set_property(TARGET atc_resource PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_SDK_ATC_RESOURCE_FILE})

# liberror_manager.so
find_library(HUAWEI_ASCEND_NPU_SDK_ATC_ERROR_MANAGER_FILE NAMES error_manager
  PATHS ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_ASCEND_NPU_SDK_ATC_ERROR_MANAGER_FILE)
  message(FATAL_ERROR "Missing libresource.so in ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/atc/lib64")
endif()
add_library(atc_error_manager SHARED IMPORTED GLOBAL)
set_property(TARGET atc_error_manager PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_SDK_ATC_ERROR_MANAGER_FILE})

# OPP libs
# libopsproto.so
find_library(HUAWEI_ASCEND_NPU_SDK_OPP_OPSPROTO_FILE NAMES opsproto
  PATHS ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/opp/op_proto/built-in
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT HUAWEI_ASCEND_NPU_SDK_OPP_OPSPROTO_FILE)
  message(FATAL_ERROR "Missing libopsproto.so in ${NNADAPTER_HUAWEI_ASCEND_NPU_SDK_ROOT}/opp/op_proto/built-in")
endif()
add_library(opp_opsproto SHARED IMPORTED GLOBAL)
set_property(TARGET opp_opsproto PROPERTY IMPORTED_LOCATION ${HUAWEI_ASCEND_NPU_SDK_OPP_OPSPROTO_FILE})

# ACL libs should before ATC libs
set(${DEVICE_NAME}_deps acl_ascendcl acl_runtime atc_register atc_graph atc_ge_compiler atc_ge_common atc_resource atc_error_manager opp_opsproto)
