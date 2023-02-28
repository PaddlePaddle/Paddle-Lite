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

if(NOT DEFINED NNADAPTER_MEDIATEK_APU_SDK_ROOT)
  set(NNADAPTER_MEDIATEK_APU_SDK_ROOT $ENV{NNADAPTER_MEDIATEK_APU_SDK_ROOT})
endif()
if(NOT NNADAPTER_MEDIATEK_APU_SDK_ROOT)
  message(FATAL_ERROR "Must set NNADAPTER_MEDIATEK_APU_SDK_ROOT or env NNADAPTER_MEDIATEK_APU_SDK_ROOT when NNADAPTER_WITH_MEDIATEK_APU=ON")
endif()
message(STATUS "NNADAPTER_MEDIATEK_APU_SDK_ROOT: ${NNADAPTER_MEDIATEK_APU_SDK_ROOT}")

find_path(MEDIATEK_APU_SDK_INC NAMES NeuronAdapter.h
  PATHS ${NNADAPTER_MEDIATEK_APU_SDK_ROOT}/include/
  CMAKE_FIND_ROOT_PATH_BOTH)
if(NOT MEDIATEK_APU_SDK_INC)
  message(FATAL_ERROR "Missing NeuronAdapter.h in ${NNADAPTER_MEDIATEK_APU_SDK_ROOT}/include")
endif()

include_directories("${NNADAPTER_MEDIATEK_APU_SDK_ROOT}/include")
