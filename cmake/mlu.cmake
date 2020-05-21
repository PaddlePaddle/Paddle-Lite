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

if(NOT LITE_WITH_MLU)
  return()
endif()

if(NOT DEFINED NEUWARE_HOME)
    set(NEUWARE_HOME $ENV{NEUWARE_HOME})
    if(NOT NEUWARE_HOME)
        message(FATAL_ERROR "Must set NEUWARE_HOME or env NEUWARE_HOME when LITE_WITH_MLU=ON")
    endif()
endif()

message(STATUS "LITE_WITH_MLU: ${LITE_WITH_MLU}")
find_path(CNML_INC NAMES cnml.h
  PATHS ${NEUWARE_HOME}/include NO_DEFAULT_PATH)
if(NOT CNML_INC)
  message(FATAL_ERROR "Can not find cnml.h in ${NEUWARE_HOME}/include")
endif()

find_path(CNRT_INC NAMES cnrt.h
  PATHS ${NEUWARE_HOME}/include NO_DEFAULT_PATH)
if(NOT CNRT_INC)
  message(FATAL_ERROR "Can not find cnrt.h in ${NEUWARE_HOME}/include")
endif()

include_directories("${NEUWARE_HOME}/include")

find_library(CNML_LIB_FILE NAMES cnml
  PATHS ${NEUWARE_HOME}/lib64)

if(NOT CNML_LIB_FILE)
  message(FATAL_ERROR "Can not find CNML Library in ${NEUWARE_HOME}/lib64")
else()
  message(STATUS "Found CNML Library: ${CNML_LIB_FILE}")
  add_library(cnml_lib SHARED IMPORTED GLOBAL)
  set_property(TARGET cnml_lib PROPERTY IMPORTED_LOCATION ${CNML_LIB_FILE})
endif()

find_library(CNRT_LIB_FILE NAMES cnrt
  PATHS ${NEUWARE_HOME}/lib64)

if(NOT CNRT_LIB_FILE)
  message(FATAL_ERROR "Can not find CNRT Library in ${NEUWARE_HOME}/lib64")
else()
  message(STATUS "Found CNRT Library: ${CNRT_LIB_FILE}")
  add_library(cnrt_lib SHARED IMPORTED GLOBAL)
  set_property(TARGET cnrt_lib PROPERTY IMPORTED_LOCATION ${CNRT_LIB_FILE})
endif()
