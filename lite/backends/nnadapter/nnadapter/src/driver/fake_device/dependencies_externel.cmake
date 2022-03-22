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

if(NOT DEFINED NNADAPTER_FAKE_DEVICE_SDK_ROOT)
  set(NNADAPTER_FAKE_DEVICE_SDK_ROOT $ENV{NNADAPTER_FAKE_DEVICE_SDK_ROOT})
  if(NOT NNADAPTER_FAKE_DEVICE_SDK_ROOT)
    message(FATAL_ERROR "Must set NNADAPTER_FAKE_DEVICE_SDK_ROOT or env NNADAPTER_FAKE_DEVICE_SDK_ROOT when NNADAPTER_WITH_FAKE_DEVICE=ON")
  endif()
endif()

message(STATUS "NNADAPTER_FAKE_DEVICE_SDK_ROOT: ${NNADAPTER_FAKE_DEVICE_SDK_ROOT}")
find_path(FAKE_DEVICE_SDK_INC NAMES fakedevice_pub.h
  PATHS ${NNADAPTER_FAKE_DEVICE_SDK_ROOT}/include/
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT FAKE_DEVICE_SDK_INC)
  message(FATAL_ERROR "Missing fakedevice_pub.h in ${NNADAPTER_FAKE_DEVICE_SDK_ROOT}/include")
endif()

include_directories("${NNADAPTER_FAKE_DEVICE_SDK_ROOT}/include")

set(FAKE_DEVICE_SDK_SUB_LIB_PATH "lib64")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(FAKE_DEVICE_SDK_SUB_LIB_PATH "lib64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm")
  set(FAKE_DEVICE_SDK_SUB_LIB_PATH "lib")
else()
  message(FATAL_ERROR "${CMAKE_SYSTEM_PROCESSOR} isn't supported by fake_device DDK.")
endif()

find_library(FAKE_DEVICE_SDK_DDK_FILE NAMES fakedevice_ddk
  PATHS ${NNADAPTER_FAKE_DEVICE_SDK_ROOT}/${FAKE_DEVICE_SDK_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT FAKE_DEVICE_SDK_DDK_FILE)
  message(FATAL_ERROR "Missing fakedevice_ddk in ${NNADAPTER_FAKE_DEVICE_SDK_ROOT}/${FAKE_DEVICE_SDK_SUB_LIB_PATH}")
endif()
add_library(fakedevice_ddk SHARED IMPORTED GLOBAL)
set_property(TARGET fakedevice_ddk PROPERTY IMPORTED_LOCATION ${FAKE_DEVICE_SDK_DDK_FILE})

set(${DEVICE_NAME}_deps fakedevice_ddk)
