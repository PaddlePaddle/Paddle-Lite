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

if(NOT LITE_WITH_HW_ASCEND_NPU)
  return()
endif()

if(NOT DEFINED ASCEND_HOME)
    set(ASCEND_HOME $ENV{ASCEND_HOME})
    if(NOT ASCEND_HOME)
        message(FATAL_ERROR "Must set ASCEND_HOME or env ASCEND_HOME when LITE_WITH_HW_ASCEND_NPU=ON")
    endif()
endif()

message(STATUS "LITE_WITH_HW_ASCEND_NPU: ${LITE_WITH_HW_ASCEND_NPU}")
find_path(ACL_INC NAMES acl/acl.h
  PATHS ${ASCEND_HOME}/acllib/include NO_DEFAULT_PATH)
if(NOT ACL_INC)
  message(FATAL_ERROR "Can not find acl/acl.h in ${ASCEND_HOME}/include")
endif()

include_directories("${ACL_INC}")

# find ascendcl library
find_library(ACL_LIB_FILE NAMES ascendcl PATHS ${ASCEND_HOME}/acllib/lib64)
if(NOT ACL_LIB_FILE)
  message(FATAL_ERROR "Can not find ACL Library in ${ASCEND_HOME}/acllib/lib64")
else()
  message(STATUS "Found ACL Library: ${ACL_LIB_FILE}")
  add_library(acl_lib SHARED IMPORTED GLOBAL)
  set_property(TARGET acl_lib PROPERTY IMPORTED_LOCATION ${ACL_LIB_FILE})
endif()

# find register library
find_library(REG_LIB_FILE NAMES register PATHS ${ASCEND_HOME}/acllib/lib64)
if(NOT REG_LIB_FILE)
    message(FATAL_ERROR "Can not find REG Library in ${ASCEND_HOME}/acllib/lib64")
else()
    message(STATUS "Found REG Library: ${REG_LIB_FILE}")
    add_library(register_lib SHARED IMPORTED GLOBAL)
    set_property(TARGET register_lib PROPERTY IMPORTED_LOCATION ${REG_LIB_FILE})
endif()


find_library(RT_LIB_FILE NAMES runtime PATHS ${ASCEND_HOME}/acllib/lib64)
if(NOT RT_LIB_FILE)
    message(FATAL_ERROR "Can not find RT Library in ${ASCEND_HOME}/acllib/lib64")
else()
    message(STATUS "Found RT Library: ${RT_LIB_FILE}")
    add_library(runtime_lib SHARED IMPORTED GLOBAL)
    set_property(TARGET runtime_lib PROPERTY IMPORTED_LOCATION ${RT_LIB_FILE})
endif()

set(hw_ascend_npu_runtime_libs acl_lib register_lib runtime_lib CACHE INTERNAL "ascend runtime libs")

# find atc include folder and library
find_path(ATC_INC NAMES ge/ge_ir_build.h
  PATHS ${ASCEND_HOME}/atc/include)
if (NOT ATC_INC)
  message(FATAL_ERROR "Can not find ge/ge_ir_build.h in ${ASCEND_HOME}/atc/include")
endif()
include_directories("${ATC_INC}")

find_library(GRAPH_LIB_FILE graph PATHS ${ASCEND_HOME}/atc/lib64)
if (NOT GRAPH_LIB_FILE)
  message(FATAL_ERROR "Can not find libgraph.so library in ${ASCEND_HOME}/atc/lib64")
else()
  message(STATUS "Found Graph Library: ${GRAPH_LIB_FILE}")
  add_library(graph_lib SHARED IMPORTED GLOBAL)
  set_property(TARGET graph_lib PROPERTY IMPORTED_LOCATION ${GRAPH_LIB_FILE})
endif()

# find opp include folder and library
find_path(OPP_INC NAMES all_ops.h
  PATHS ${ASCEND_HOME}/opp/op_proto/built-in/inc)
if (NOT OPP_INC)
  message(FATAL_ERROR "Can not find all_ops.h in ${ASCEND_HOME}/opp/op_proto/built-in/inc")
endif()
include_directories("${OPP_INC}")

find_library(OPP_LIB_FILE opsproto PATHS ${ASCEND_HOME}/opp/op_proto/built-in)
if (NOT OPP_LIB_FILE)
  message(FATAL_ERROR "Can not find libopsproto.so in ${ASCEND_HOME}/opp/op_proto/built-in")
else()
  message(STATUS "Found OPP Library: ${OPP_LIB_FILE}")
  add_library(opp_lib SHARED IMPORTED GLOBAL)
  set_property(TARGET opp_lib PROPERTY IMPORTED_LOCATION ${OPP_LIB_FILE})
endif()

set(hw_ascend_npu_builder_libs graph_lib opp_lib CACHE INTERNAL "ascend builder libs")

