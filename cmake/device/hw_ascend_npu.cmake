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

# find atc include folder and library
find_path(ATC_INC NAMES ge/ge_ir_build.h
  PATHS ${ASCEND_HOME}/atc/include)
if (NOT ATC_INC)
  message(FATAL_ERROR "Can not find ge/ge_ir_build.h in ${ASCEND_HOME}/atc/include")
endif()
include_directories("${ATC_INC}")

set(ATC_LIB_FILES
  ge_compiler
  graph
  _caffe_parser
  auto_tiling
  c_sec
  cce
  cce_aicore
  cce_aicpudev_online
  cce_tools
  drvdevdrv
  drvdevmm
  drvdsmi_host
  drvhdc_host
  fmk_caffe_parser
  fmk_tensorflow_parser
  ge_client
  ge_common
  ge_executor
  mmpa
  msprof
  parser_common
  register
  resource
  runtime
  slog
  te_fusion
  tiling
  tvm
  tvm_runtime
  tvm_topi
  )
set(ATC_PLUGIN_NNENGIN_LIB_FILES
  engine
  )
set(ATC_PLUGIN_OPSKERNEL_LIB_FILES
  aicpu_engine
  fe
  ge_local_engine
  rts_engine
  )


foreach (libname ${ATC_LIB_FILES})
  find_library(lib_name_path_${libname} NAMES ${libname} PATHS ${ASCEND_HOME}/atc/lib64)
  if (lib_name_path_${libname})
    add_library(atc_${libname} SHARED IMPORTED GLOBAL)
    set_property(TARGET atc_${libname} PROPERTY IMPORTED_LOCATION ${lib_name_path_${libname}})
    list(APPEND atc_libs atc_${libname})
  else()
    message(FATAL_ERROR "can not find library: ${libname}")
  endif()
endforeach()

foreach (libname ${ATC_PLUGIN_NNENGIN_LIB_FILES})
  find_library(lib_name_path_${libname} NAMES ${libname} PATHS ${ASCEND_HOME}/atc/lib64/plugin/nnengine)
  if (lib_name_path_${libname})
    add_library(atc_${libname} SHARED IMPORTED GLOBAL)
    set_property(TARGET atc_${libname} PROPERTY IMPORTED_LOCATION ${lib_name_path_${libname}})
    list(APPEND atc_libs atc_${libname})
  else()
    message(FATAL_ERROR "can not find library: ${libname}")
  endif()
endforeach()

foreach (libname ${ATC_PLUGIN_OPSKERNEL_LIB_FILES})
  find_library(lib_name_path_${libname} NAMES ${libname} PATHS ${ASCEND_HOME}/atc/lib64/plugin/opskernel)
  if (lib_name_path_${libname})
    add_library(atc_${libname} SHARED IMPORTED GLOBAL)
    set_property(TARGET atc_${libname} PROPERTY IMPORTED_LOCATION ${lib_name_path_${libname}})
    list(APPEND atc_libs atc_${libname})
  else()
    message(FATAL_ERROR "can not find library: ${libname}")
  endif()
endforeach()

# find opp include folder and library
find_path(OPP_INC NAMES all_ops.h
  PATHS ${ASCEND_HOME}/opp/op_proto/built-in/inc)
if (NOT OPP_INC)
  message(FATAL_ERROR "Can not find all_ops.h in ${ASCEND_HOME}/opp/op_proto/built-in/inc")
endif()
include_directories("${OPP_INC}")

find_library(OPP_OPSPROTO_LIB_FILE opsproto PATHS ${ASCEND_HOME}/opp/op_proto/built-in)
if (NOT OPP_OPSPROTO_LIB_FILE)
  message(FATAL_ERROR "Can not find libopsproto.so in ${ASCEND_HOME}/opp/op_proto/built-in")
else()
  message(STATUS "Found OPP proto Library: ${OPP_OPSPROTO_LIB_FILE}")
  add_library(opp_opsproto_lib SHARED IMPORTED GLOBAL)
  set_property(TARGET opp_opsproto_lib PROPERTY IMPORTED_LOCATION ${OPP_OPSPROTO_LIB_FILE})
endif()

find_library(OPP_FUSION_AICORE ops_fusion_pass_aicore PATHS ${ASCEND_HOME}/opp/fusion_pass/built_in/)
if (NOT OPP_FUSION_AICORE)
  message(FATAL_ERROR "Can not find libops_fusion_pass_aicore.so in ${ASCEND_HOME}/opp/fusion_pass/built_in/")
else()
  message(STATUS "Found fusion_pass_aicore Library: ${OPP_FUSION_AICORE}")
  add_library(opp_fusion_pass_aicore_lib SHARED IMPORTED GLOBAL)
  set_property(TARGET opp_fusion_pass_aicore_lib PROPERTY IMPORTED_LOCATION ${OPP_FUSION_AICORE})
endif()

find_library(OPP_FUSION_VECTORCORE ops_fusion_pass_vectorcore PATHS ${ASCEND_HOME}/opp/fusion_pass/built_in/vector_core)
if (NOT OPP_FUSION_VECTORCORE)
  message(FATAL_ERROR "Can not find libops_fusion_pass_vectorcore.so in ${ASCEND_HOME}/opp/fusion_pass/built_in/vector_core")
else()
  message(STATUS "Found fusion_pass_vectorcore Library: ${OPP_FUSION_VECTORCORE}")
  add_library(opp_fusion_pass_vectorcore_lib SHARED IMPORTED GLOBAL)
  set_property(TARGET opp_fusion_pass_vectorcore_lib PROPERTY IMPORTED_LOCATION ${OPP_FUSION_VECTORCORE})
endif()

add_library(hw_ascend_npu_builder_libs INTERFACE)
target_link_libraries(hw_ascend_npu_builder_libs INTERFACE
  ${atc_libs}
  opp_opsproto_lib
  opp_fusion_pass_aicore_lib
  opp_fusion_pass_vectorcore_lib)

#set(hw_ascend_npu_builder_libs
#  ${atc_libs}
#  opp_opsproto_lib
#  opp_fusion_pass_aicore_lib
#  opp_fusion_pass_vectorcore_lib
#  CACHE INTERNAL "ascend builder libs")

# find ascend cl runtime library
find_path(ACL_INC NAMES acl/acl.h
  PATHS ${ASCEND_HOME}/acllib/include NO_DEFAULT_PATH)
if(NOT ACL_INC)
  message(FATAL_ERROR "Can not find acl/acl.h in ${ASCEND_HOME}/include")
endif()

include_directories("${ACL_INC}")

set(ACL_LIB_FILES
  acl_dvpp
  ascendcl
  register
  runtime
  )

foreach (libname ${ACL_LIB_FILES})
  find_library(lib_name_path_${libname} NAMES ${libname} PATHS ${ASCEND_HOME}/acllib/lib64)
  if (lib_name_path_${libname})
    add_library(acl_${libname} SHARED IMPORTED GLOBAL)
    set_property(TARGET acl_${libname} PROPERTY IMPORTED_LOCATION ${lib_name_path_${libname}})
    list(APPEND acl_libs acl_${libname})
  else()
    message(FATAL_ERROR "can not find library: ${libname}")
  endif()
endforeach()

add_library(hw_ascend_npu_runtime_libs INTERFACE)
target_link_libraries(hw_ascend_npu_runtime_libs INTERFACE ${acl_libs})

add_library(hw_ascend_npu_libs INTERFACE)
target_link_libraries(hw_ascend_npu_libs INTERFACE
  ${atc_libs}
  opp_opsproto_lib
  opp_fusion_pass_aicore_lib
  opp_fusion_pass_vectorcore_lib
  ${acl_libs})

# set(hw_ascend_npu_runtime_libs ${acl_libs} CACHE INTERNAL "ascend runtime libs")



