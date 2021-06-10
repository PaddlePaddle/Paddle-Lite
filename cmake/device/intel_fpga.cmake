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

if(NOT LITE_WITH_INTEL_FPGA)
  return()
endif()

if(NOT DEFINED INTEL_FPGA_SDK_ROOT)
  set(INTEL_FPGA_SDK_ROOT $ENV{INTEL_FPGA_SDK_ROOT})
  if(NOT INTEL_FPGA_SDK_ROOT)
    message(FATAL_ERROR "Must set INTEL_FPGA_SDK_ROOT or env INTEL_FPGA_SDK_ROOT when LITE_WITH_INTEL_FPGA=ON")
  endif()
endif()

message(STATUS "INTEL_FPGA_SDK_ROOT: ${INTEL_FPGA_SDK_ROOT}")

find_path(INTEL_FPGA_SDK_INC NAMES intelfpga.h
  PATHS ${INTEL_FPGA_SDK_ROOT}/include NO_DEFAULT_PATH)
if (NOT INTEL_FPGA_SDK_INC)
  message(FATAL_ERROR "Can not find intelfpga.h in ${INTEL_FPGA_SDK_INC}/include")
endif()

include_directories("${INTEL_FPGA_SDK_INC}")

find_library(INTEL_FPGA_SDK_LIB NAMES vnna
  PATHS ${INTEL_FPGA_SDK_ROOT}/lib)

if(NOT INTEL_FPGA_SDK_LIB)
  message(FATAL_ERROR "Can not find INTEL_FPGA_LIB_FILE in ${INTEL_FPGA_SDK_ROOT}/lib")
else()
  message(STATUS "Found INTEL_FPGA_SDK Library: ${INTEL_FPGA_SDK_LIB}")
  add_library(intel_fpga_vnna SHARED IMPORTED GLOBAL)
  set_property(TARGET intel_fpga_vnna PROPERTY IMPORTED_LOCATION ${INTEL_FPGA_SDK_LIB})
endif()

set(intel_fpga_runtime_libs intel_fpga_vnna CACHE INTERNAL "intel fpga sdk runtime libs")
