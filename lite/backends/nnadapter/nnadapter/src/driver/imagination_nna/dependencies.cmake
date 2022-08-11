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

if(NOT DEFINED NNADAPTER_IMAGINATION_NNA_SDK_ROOT)
  set(NNADAPTER_IMAGINATION_NNA_SDK_ROOT $ENV{NNADAPTER_IMAGINATION_NNA_SDK_ROOT})
endif()
if(NOT NNADAPTER_IMAGINATION_NNA_SDK_ROOT)
  message(FATAL_ERROR "Must set NNADAPTER_IMAGINATION_NNA_SDK_ROOT or env NNADAPTER_IMAGINATION_NNA_SDK_ROOT when NNADAPTER_WITH_IMAGINATION_NNA=ON")
endif()
message(STATUS "NNADAPTER_IMAGINATION_NNA_SDK_ROOT: ${NNADAPTER_IMAGINATION_NNA_SDK_ROOT}")

find_path(IMAGINATION_NNA_SDK_INC NAMES imgdnn.h
  PATHS ${NNADAPTER_IMAGINATION_NNA_SDK_ROOT}/include/imgdnn
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT IMAGINATION_NNA_SDK_INC)
  message(FATAL_ERROR "Missing imgdnn.h in ${NNADAPTER_IMAGINATION_NNA_SDK_ROOT}/include")
endif()

include_directories("${NNADAPTER_IMAGINATION_NNA_SDK_ROOT}/include/imgdnn")

set(IMAGINATION_NNA_SDK_SUB_LIB_PATH "lib")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(IMAGINATION_NNA_SDK_SUB_LIB_PATH "lib")
else()
  message(FATAL_ERROR "${CMAKE_SYSTEM_PROCESSOR} isn't supported by Imagination NNA SDK.")
endif()

find_library(IMAGINATION_NNA_SDK_IMGDNN_FILE NAMES imgdnn
  PATHS ${NNADAPTER_IMAGINATION_NNA_SDK_ROOT}/${IMAGINATION_NNA_SDK_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT IMAGINATION_NNA_SDK_IMGDNN_FILE)
  message(FATAL_ERROR "Missing imgdnn in ${NNADAPTER_IMAGINATION_NNA_SDK_ROOT}/${IMAGINATION_NNA_SDK_SUB_LIB_PATH}")
endif()
add_library(imgdnn SHARED IMPORTED GLOBAL)
set_property(TARGET imgdnn PROPERTY IMPORTED_LOCATION ${IMAGINATION_NNA_SDK_IMGDNN_FILE})

find_library(IMAGINATION_NNA_SDK_NNASESSION_FILE NAMES nnasession
  PATHS ${NNADAPTER_IMAGINATION_NNA_SDK_ROOT}/${IMAGINATION_NNA_SDK_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT IMAGINATION_NNA_SDK_NNASESSION_FILE)
  message(FATAL_ERROR "Missing nnasession in ${NNADAPTER_IMAGINATION_NNA_SDK_ROOT}/${IMAGINATION_NNA_SDK_SUB_LIB_PATH}")
endif()
add_library(nnasession SHARED IMPORTED GLOBAL)
set_property(TARGET nnasession PROPERTY IMPORTED_LOCATION ${IMAGINATION_NNA_SDK_NNASESSION_FILE})

set(DEPS ${DEPS} imgdnn nnasession)
