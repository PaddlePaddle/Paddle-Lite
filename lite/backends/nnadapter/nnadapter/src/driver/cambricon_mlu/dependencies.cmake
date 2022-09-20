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

if(NOT DEFINED NNADAPTER_CAMBRICON_MLU_SDK_ROOT)
  set(NNADAPTER_CAMBRICON_MLU_SDK_ROOT $ENV{NNADAPTER_CAMBRICON_MLU_SDK_ROOT})
endif()
if(NOT NNADAPTER_CAMBRICON_MLU_SDK_ROOT)
  message(FATAL_ERROR "Must set NNADAPTER_CAMBRICON_MLU_SDK_ROOT or env NNADAPTER_CAMBRICON_MLU_SDK_ROOT when NNADAPTER_WITH_CAMBRICON_MLU=ON")
endif()
message(STATUS "NNADAPTER_CAMBRICON_MLU_SDK_ROOT: ${NNADAPTER_CAMBRICON_MLU_SDK_ROOT}")

find_path(CAMBRICON_MLU_SDK_INC NAMES interface_network.h
  PATHS ${NNADAPTER_CAMBRICON_MLU_SDK_ROOT}/include/
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT CAMBRICON_MLU_SDK_INC)
  message(FATAL_ERROR "Missing interface_network.h in ${NNADAPTER_CAMBRICON_MLU_SDK_ROOT}/include")
endif()

include_directories("${NNADAPTER_CAMBRICON_MLU_SDK_ROOT}/include/")

set(CAMBRICON_MLU_SDK_SUB_LIB_PATH "lib64")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
else()
  message(FATAL_ERROR "${CMAKE_SYSTEM_PROCESSOR} isn't supported by Cambricon MLU SDK.")
endif()

find_library(CAMBRICON_MLU_SDK_MAGICMIND_FILE NAMES magicmind
  PATHS ${NNADAPTER_CAMBRICON_MLU_SDK_ROOT}/${CAMBRICON_MLU_SDK_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT CAMBRICON_MLU_SDK_MAGICMIND_FILE)
    message(FATAL_ERROR "Missing libmagicmind.so in ${NNADAPTER_CAMBRICON_MLU_SDK_ROOT}/${CAMBRICON_MLU_SDK_SUB_LIB_PATH}")
endif()
add_library(magicmind SHARED IMPORTED GLOBAL)
set_property(TARGET magicmind PROPERTY IMPORTED_LOCATION ${CAMBRICON_MLU_SDK_MAGICMIND_FILE})

find_library(CAMBRICON_MLU_SDK_MAGICMIND_RUNTIME_FILE NAMES magicmind_runtime
  PATHS ${NNADAPTER_CAMBRICON_MLU_SDK_ROOT}/${CAMBRICON_MLU_SDK_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT CAMBRICON_MLU_SDK_MAGICMIND_RUNTIME_FILE)
    message(FATAL_ERROR "Missing libmagicmind_runtime.so in ${NNADAPTER_CAMBRICON_MLU_SDK_ROOT}/${CAMBRICON_MLU_SDK_SUB_LIB_PATH}")
endif()
add_library(magicmind_runtime SHARED IMPORTED GLOBAL)
set_property(TARGET magicmind_runtime PROPERTY IMPORTED_LOCATION ${CAMBRICON_MLU_SDK_MAGICMIND_RUNTIME_FILE})

find_library(CAMBRICON_MLU_SDK_MAGICMIND_PLUGIN_FILE NAMES magicmind_plugin
  PATHS ${NNADAPTER_CAMBRICON_MLU_SDK_ROOT}/${CAMBRICON_MLU_SDK_SUB_LIB_PATH}
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT CAMBRICON_MLU_SDK_MAGICMIND_PLUGIN_FILE)
    message(FATAL_ERROR "Missing libmagicmind_plugin.so in ${NNADAPTER_CAMBRICON_MLU_SDK_ROOT}/${CAMBRICON_MLU_SDK_SUB_LIB_PATH}")
endif()
add_library(magicmind_plugin SHARED IMPORTED GLOBAL)
set_property(TARGET magicmind_plugin PROPERTY IMPORTED_LOCATION ${CAMBRICON_MLU_SDK_MAGICMIND_PLUGIN_FILE})

set(DEPS ${DEPS} magicmind magicmind_runtime magicmind_plugin)
