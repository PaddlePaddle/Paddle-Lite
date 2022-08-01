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
cmake_minimum_required(VERSION 3.10)

if(NNADAPTER_WITH_STANDALONE)
  # Add external libnnadapter.so to build NNAdapter HAL library standalone
  add_library(nnadapter SHARED IMPORTED GLOBAL)
  set_property(TARGET nnadapter PROPERTY IMPORTED_LOCATION ${NNADAPTER_STANDALONE_RUNTIME_LIBRARY_PATH})
  file(COPY ${NNADAPTER_STANDALONE_RUNTIME_LIBRARY_PATH} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  include_directories(${NNADAPTER_STANDALONE_RUNTIME_INCLUDE_DIRECTORY})
endif()

set(DEVICE_NAME fake_device)
add_definitions(-DDEVICE_NAME=${DEVICE_NAME})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Os -fvisibility=hidden -fvisibility-inlines-hidden -fexceptions -fasynchronous-unwind-tables -funwind-tables -Wl,--allow-shlib-undefined")

include(dependencies.cmake)

include_directories(.)
aux_source_directory(converter CONVERTERS)
set(SRCS utility.cc ${CONVERTERS} engine.cc driver.cc)
set(DEPS ${DEPS} nnadapter)

add_library(${DEVICE_NAME} SHARED ${SRCS})
target_link_libraries(${DEVICE_NAME} "-Wl,--start-group" ${DEPS} "-Wl,--end-group -Wl,-exclude-libs,ALL -Wl,--gc-sections -Wl,--strip-all")
set(NNADAPTER_DEVICES ${NNADAPTER_DEVICES} ${DEVICE_NAME} CACHE INTERNAL "")
