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

if(NOT LITE_WITH_APU)
  return()
endif()

if(NOT DEFINED APU_DDK_ROOT)
    set(APU_DDK_ROOT $ENV{APU_DDK_ROOT})
    if(NOT APU_DDK_ROOT)
        message(FATAL_ERROR "Must set APU_DDK_ROOT or env APU_DDK_ROOT when LITE_WITH_APU=ON")
    endif()
endif()

message(STATUS "APU_DDK_ROOT: ${APU_DDK_ROOT}")
find_path(APU_DDK_INC NAMES NeuronAdapter.h
  PATHS ${APU_DDK_ROOT}/include NO_DEFAULT_PATH)
if(NOT APU_DDK_INC)
  message(FATAL_ERROR "Can not find NeuronAdapter.h in ${APU_DDK_ROOT}/include")
endif()
message(STATUS "APU_DDK_INC: ${APU_DDK_INC}")

include_directories("${APU_DDK_ROOT}/include")

set(APU_SUB_LIB_PATH "lib64")
if(ARM_TARGET_ARCH_ABI STREQUAL "armv8")
    set(APU_SUB_LIB_PATH "lib64")
endif()

find_library(APU_NEURON_FILE NAMES neuron
  PATHS ${APU_DDK_ROOT}/${APU_SUB_LIB_PATH})

find_library(APU_NEURON_ADAPTER_FILE NAMES neuron_adapter
  PATHS ${APU_DDK_ROOT}/${APU_SUB_LIB_PATH})

if(NOT APU_NEURON_FILE)
  message(FATAL_ERROR "Can not find APU_NEURON_FILE in ${APU_DDK_ROOT}")
else()
  message(STATUS "Found APU NEURON Library: ${APU_NEURON_FILE}")
  add_library(apu_neuron SHARED IMPORTED GLOBAL)
  set_property(TARGET apu_neuron PROPERTY IMPORTED_LOCATION ${APU_NEURON_FILE})
endif()

if(NOT APU_NEURON_ADAPTER_FILE)
  message(FATAL_ERROR "Can not find APU_NEURON_ADAPTER_FILE in ${APU_DDK_ROOT}")
else()
  message(STATUS "Found APU NEURON ADAPTER Library: ${APU_NEURON_ADAPTER_FILE}")
  add_library(apu_neuron_adapter SHARED IMPORTED GLOBAL)
  set_property(TARGET apu_neuron_adapter PROPERTY IMPORTED_LOCATION ${APU_NEURON_ADAPTER_FILE})
endif()

set(apu_runtime_libs apu_neuron apu_neuron_adapter CACHE INTERNAL "apu runtime libs")
message(STATUS "${apu_runtime_libs}")

