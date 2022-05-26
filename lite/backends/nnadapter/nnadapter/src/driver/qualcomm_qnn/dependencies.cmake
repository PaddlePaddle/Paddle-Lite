# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

if(NOT DEFINED NNADAPTER_QUALCOMM_QNN_SDK_ROOT)
  set(NNADAPTER_INTEL_OPENVINO_SDK_ROOT $ENV{NNADAPTER_INTEL_OPENVINO_SDK_ROOT})
  if(NOT NNADAPTER_INTEL_OPENVINO_SDK_ROOT)
    message(FATAL_ERROR "Must set NNADAPTER_QUALCOMM_QNN_SDK_ROOT or env NNADAPTER_QUALCOMM_QNN_SDK_ROOT when NNADAPTER_WITH_QUALCOMM_QNN=ON")
  endif()
endif()
message(STATUS "NNADAPTER_QUALCOMM_QNN_SDK_ROOT: ${NNADAPTER_QUALCOMM_QNN_SDK_ROOT}")

include_directories(${NNADAPTER_QUALCOMM_QNN_SDK_ROOT}/include)

# only support cpu now
set(QNN_LIB_PATH ${NNADAPTER_QUALCOMM_QNN_SDK_ROOT}/target/x86_64-linux-clang/lib)
find_library(QNN_CPU_FILE NAMES QnnCpu
  PATHS ${QNN_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT QNN_CPU_FILE)
  message(FATAL_ERROR "Missing libQnnCpu.so in ${QNN_LIB_PATH}")
endif()
add_library(qnn_cpu SHARED IMPORTED GLOBAL)
set_property(TARGET qnn_cpu PROPERTY IMPORTED_LOCATION ${QNN_CPU_FILE})

set(DEPS ${DEPS} qnn_cpu)
