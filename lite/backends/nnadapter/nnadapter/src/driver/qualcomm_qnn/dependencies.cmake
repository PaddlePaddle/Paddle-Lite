# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

if(NOT DEFINED NNADAPTER_QUALCOMM_QNN_SDK_ROOT)
  set(NNADAPTER_INTEL_OPENVINO_SDK_ROOT $ENV{NNADAPTER_INTEL_OPENVINO_SDK_ROOT})
  if(NOT NNADAPTER_INTEL_OPENVINO_SDK_ROOT)
    message(FATAL_ERROR "Must set NNADAPTER_QUALCOMM_QNN_SDK_ROOT or env NNADAPTER_QUALCOMM_QNN_SDK_ROOT when NNADAPTER_WITH_QUALCOMM_QNN=ON")
  endif()
endif()
message(STATUS "NNADAPTER_QUALCOMM_QNN_SDK_ROOT: ${NNADAPTER_QUALCOMM_QNN_SDK_ROOT}")

include_directories(${NNADAPTER_QUALCOMM_QNN_SDK_ROOT}/include)

# do not need libs during compiling.
