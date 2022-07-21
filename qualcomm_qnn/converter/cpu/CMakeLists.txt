# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
cmake_minimum_required(VERSION 3.10)

set(CPU_OP_PACKAGE_NAME qualcomm_qnn_cpu_custom_op_package)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os -Wl,-exclude-libs,ALL -Wl,--gc-sections -Wl,--strip-all")

include_directories(.)
aux_source_directory(. SRCS)

add_library(${CPU_OP_PACKAGE_NAME} SHARED ${SRCS})

# Copy libs
string(REGEX REPLACE "(.*)/converter/cpu" "\\1" QUALCOMM_QNN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
add_custom_command(TARGET ${CPU_OP_PACKAGE_NAME}
  POST_BUILD
  COMMAND cp -r "${CMAKE_CURRENT_BINARY_DIR}/*.so" "${QUALCOMM_QNN_BINARY_DIR}"
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/lib${CPU_OP_PACKAGE_NAME}.so
)
