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

if(NOT DEFINED NNADAPTER_INTEL_OPENVINO_SDK_ROOT)
  set(NNADAPTER_INTEL_OPENVINO_SDK_ROOT $ENV{NNADAPTER_INTEL_OPENVINO_SDK_ROOT})
  if(NOT NNADAPTER_INTEL_OPENVINO_SDK_ROOT)
    message(FATAL_ERROR "Must set NNADAPTER_INTEL_OPENVINO_SDK_ROOT or env NNADAPTER_INTEL_OPENVINO_SDK_ROOT when NNADAPTER_WITH_INTEL_OPENVINO=ON")
  endif()
endif()
message(STATUS "NNADAPTER_INTEL_OPENVINO_SDK_ROOT: ${NNADAPTER_INTEL_OPENVINO_SDK_ROOT}")

set(OPENVINO_RUNTIME_PATH ${NNADAPTER_INTEL_OPENVINO_SDK_ROOT}/runtime)
execute_process(COMMAND ls WORKING_DIRECTORY ${NNADAPTER_INTEL_OPENVINO_SDK_ROOT}/runtime/lib OUTPUT_VARIABLE system_type OUTPUT_STRIP_TRAILING_WHITESPACE)
set(OPENVINO_RUNTIME_LIB_PATH ${OPENVINO_RUNTIME_PATH}/lib/${system_type})

set(OPENVINO_INC_PATH 
    ${NNADAPTER_INTEL_OPENVINO_SDK_ROOT}/runtime/include/ie
    ${OPENVINO_RUNTIME_PATH}/include
)
include_directories("${OPENVINO_INC_PATH}")

# libopenvino.so
find_library(INTEL_OPENVINO_SDK_OPENVINO_FILE NAMES openvino
  PATHS ${OPENVINO_RUNTIME_LIB_PATH}
  NO_DEFAULT_PATH)
if(NOT INTEL_OPENVINO_SDK_OPENVINO_FILE)
  message(FATAL_ERROR "Missing libopenvino.so in ${OPENVINO_RUNTIME_LIB_PATH}")
endif()
add_library(openvino SHARED IMPORTED GLOBAL)
set_property(TARGET openvino PROPERTY IMPORTED_LOCATION ${INTEL_OPENVINO_SDK_OPENVINO_FILE})

set(${DEVICE_NAME}_deps openvino)
