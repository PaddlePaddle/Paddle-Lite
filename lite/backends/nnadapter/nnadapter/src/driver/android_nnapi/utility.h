// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vector>
#include "core/types.h"
#include "driver/android_nnapi/nnapi_implementation.h"

namespace nnadapter {
namespace android_nnapi {

// The following environment variables can be used at runtime:
// Specify the list of device names, such as
// ANDROID_NNAPI_SELECTED_DEVICE_NAMES=nnapi-reference,armnn or
// ANDROID_NNAPI_SELECTED_DEVICE_NAMES=armnn,
// ANDROID_NNAPI_SELECTED_DEVICE_NAMES
// have higher priority than ANDROID_NNAPI_SELECTED_DEVICE_IDS
#define ANDROID_NNAPI_SELECTED_DEVICE_NAMES \
  "ANDROID_NNAPI_SELECTED_DEVICE_NAMES"

// Specify the list of device IDs, such as
// ANDROID_NNAPI_SELECTED_DEVICE_IDS=0,1,2,3 or
// ANDROID_NNAPI_SELECTED_DEVICE_IDS=0
#define ANDROID_NNAPI_SELECTED_DEVICE_IDS "ANDROID_NNAPI_SELECTED_DEVICE_IDS"

// Use NNAPI acclerator device only to offload the whole model to the
// accelerators, all of CPU and GPU related devices will be removed in the
// available device list
#define ANDROID_NNAPI_ONLY_USE_ACC_DEVICE "ANDROID_NNAPI_ONLY_USE_ACC_DEVICE"

// Disable NNAPI CPU device, so nnapi-reference and cpu related devices will be
// removed in the available device list
#define ANDROID_NNAPI_DISABLE_CPU_DEVICE "ANDROID_NNAPI_DISABLE_CPU_DEVICE"

// Relax computation float32 to float16
#define ANDROID_NNAPI_RELAX_FP32_TO_FP16 "ANDROID_NNAPI_RELAX_FP32_TO_FP16"

// Constants of NNAPI
#define ANDROID_NNAPI_REFERENCE_DEVICE_NAME "nnapi-reference"

// Simplify the calls of NNAPI
inline const NnApi* nnapi() { return NnApiImplementation(); }

// Get the bytes of the data type of NNAPI
int NNOperandDataTypeLength(int data_type);

// Convert NNAdapter types to NNAPI types
int ConvertToNNPrecision(NNAdapterOperandPrecisionCode precision_code);
int ConvertToNNDataLayout(NNAdapterOperandLayoutCode layout_code);
std::vector<uint32_t> ConvertToNNDimensions(int32_t* input_dimensions,
                                            uint32_t input_dimensions_count);
int32_t ConvertFuseCodeToNNFuseCode(int32_t fuse_code);

}  // namespace android_nnapi
}  // namespace nnadapter
