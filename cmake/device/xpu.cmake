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

if(NOT LITE_WITH_XPU)
  return()
endif()

if(NOT DEFINED XPU_SDK_ROOT)
  set(XPU_SDK_ROOT $ENV{XPU_SDK_ROOT})
  if(NOT XPU_SDK_ROOT)
    message(FATAL_ERROR "Must set XPU_SDK_ROOT or env XPU_SDK_ROOT when LITE_WITH_XPU=ON")
  endif()
endif()
message(STATUS "XPU_SDK_ROOT: ${XPU_SDK_ROOT}")

include_directories("${XPU_SDK_ROOT}/XTDK/include")

find_library(XPU_SDK_XPU_API_FILE NAMES xpuapi
  PATHS ${XPU_SDK_ROOT}/XTDK/shlib
  NO_DEFAULT_PATH)

if(NOT XPU_SDK_XPU_API_FILE)
  message(FATAL_ERROR "Can not find XPU API Library in ${XPU_SDK_ROOT}")
else()
  message(STATUS "Found XPU API Library: ${XPU_SDK_XPU_API_FILE}")
  add_library(xpu_sdk_xpu_api SHARED IMPORTED GLOBAL)
  set_property(TARGET xpu_sdk_xpu_api PROPERTY IMPORTED_LOCATION ${XPU_SDK_XPU_API_FILE})
endif()

find_library(XPU_SDK_XPU_RT_FILE NAMES xpurt
  PATHS ${XPU_SDK_ROOT}/XTDK/runtime/shlib ${XPU_SDK_ROOT}/XTDK/shlib # libxpurt.so may have been moved to XTDK/runtime/shlib
  NO_DEFAULT_PATH)

if(NOT XPU_SDK_XPU_RT_FILE)
  message(FATAL_ERROR "Can not find XPU RT Library in ${XPU_SDK_ROOT}")
else()
  message(STATUS "Found XPU RT Library: ${XPU_SDK_XPU_RT_FILE}")
  add_library(xpu_sdk_xpu_rt SHARED IMPORTED GLOBAL)
  set_property(TARGET xpu_sdk_xpu_rt PROPERTY IMPORTED_LOCATION ${XPU_SDK_XPU_RT_FILE})
endif()

set(xpu_runtime_libs xpu_sdk_xpu_api xpu_sdk_xpu_rt CACHE INTERNAL "xpu runtime libs")
set(xpu_builder_libs xpu_sdk_xpu_api xpu_sdk_xpu_rt CACHE INTERNAL "xpu builder libs")

if(LITE_WITH_XTCL)
    find_path(XPU_SDK_INC NAMES xtcl.h
      PATHS ${XPU_SDK_ROOT}/XTCL/include/xtcl NO_DEFAULT_PATH)
    if(NOT XPU_SDK_INC)
      message(FATAL_ERROR "Can not find xtcl.h in ${XPU_SDK_ROOT}/include")
    endif()
    include_directories("${XPU_SDK_ROOT}/XTCL/include")

    find_library(XPU_SDK_XTCL_FILE NAMES xtcl
      PATHS ${XPU_SDK_ROOT}/XTCL/lib
      NO_DEFAULT_PATH)

    if(NOT XPU_SDK_XTCL_FILE)
      message(FATAL_ERROR "Can not find XPU XTCL Library in ${XPU_SDK_ROOT}")
    else()
      message(STATUS "Found XPU XTCL Library: ${XPU_SDK_XTCL_FILE}")
      add_library(xpu_sdk_xtcl SHARED IMPORTED GLOBAL)
      set_property(TARGET xpu_sdk_xtcl PROPERTY IMPORTED_LOCATION ${XPU_SDK_XTCL_FILE})
    endif()

    find_library(XPU_SDK_TVM_FILE NAMES tvm
      PATHS ${XPU_SDK_ROOT}/XTCL/shlib
      NO_DEFAULT_PATH)

    if(NOT XPU_SDK_TVM_FILE)
      message(FATAL_ERROR "Can not find XPU TVM Library in ${XPU_SDK_ROOT}")
    else()
      message(STATUS "Found XPU TVM Library: ${XPU_SDK_TVM_FILE}")
      add_library(xpu_sdk_tvm SHARED IMPORTED GLOBAL)
      set_property(TARGET xpu_sdk_tvm PROPERTY IMPORTED_LOCATION ${XPU_SDK_TVM_FILE})
    endif()

    find_library(XPU_SDK_LLVM_FILE NAMES LLVM-8
      PATHS ${XPU_SDK_ROOT}/XTDK/shlib
      NO_DEFAULT_PATH)

    if(NOT XPU_SDK_LLVM_FILE)
      message(FATAL_ERROR "Can not find LLVM Library in ${XPU_SDK_ROOT}")
    else()
      message(STATUS "Found XPU LLVM Library: ${XPU_SDK_LLVM_FILE}")
      add_library(xpu_sdk_llvm SHARED IMPORTED GLOBAL)
      set_property(TARGET xpu_sdk_llvm PROPERTY IMPORTED_LOCATION ${XPU_SDK_LLVM_FILE})
    endif()

    find_library(XPU_SDK_XPU_JITC_FILE NAMES xpujitc
      PATHS ${XPU_SDK_ROOT}/XTDK/runtime/shlib ${XPU_SDK_ROOT}/XTDK/shlib # libxpujitc.so may have been moved to XTDK/runtime/shlib
      NO_DEFAULT_PATH)

    if(NOT XPU_SDK_XPU_JITC_FILE)
      message(FATAL_ERROR "Can not find XPU JITC Library in ${XPU_SDK_ROOT}")
    else()
      message(STATUS "Found XPU JITC Library: ${XPU_SDK_XPU_JITC_FILE}")
      add_library(xpu_sdk_xpu_jitc SHARED IMPORTED GLOBAL)
      set_property(TARGET xpu_sdk_xpu_jitc PROPERTY IMPORTED_LOCATION ${XPU_SDK_XPU_JITC_FILE})
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDMLC_USE_GLOG=1")

    set(xpu_runtime_libs xpu_sdk_xtcl xpu_sdk_tvm xpu_sdk_xpu_api xpu_sdk_xpu_rt xpu_sdk_llvm xpu_sdk_xpu_jitc CACHE INTERNAL "xpu runtime libs")
    set(xpu_builder_libs xpu_sdk_xtcl xpu_sdk_tvm xpu_sdk_xpu_api xpu_sdk_xpu_rt xpu_sdk_llvm xpu_sdk_xpu_jitc CACHE INTERNAL "xpu builder libs")
endif()
