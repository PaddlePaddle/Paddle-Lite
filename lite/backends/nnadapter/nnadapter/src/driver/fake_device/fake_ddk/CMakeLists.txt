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

# This an example, so we build fake_device_ddk 
# Normally, the ddk is provided by hardware vendors
include_directories("./include")
include_directories("./src")

if(CMAKE_SYSTEM_NAME MATCHES "Android")
  set(CMAKE_CXX_FLAGS "-llog ${CMAKE_CXX_FLAGS}")
endif()

aux_source_directory(src SRCS)

add_library(fake_ddk SHARED ${SRCS})
add_custom_command(TARGET fake_ddk
  POST_BUILD
  COMMAND cp -r "${CMAKE_CURRENT_BINARY_DIR}/libfake_ddk.so" "${CMAKE_CURRENT_BINARY_DIR}/.."
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libfake_ddk.so"
)
