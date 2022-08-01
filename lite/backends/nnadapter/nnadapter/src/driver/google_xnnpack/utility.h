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

#include <xnnpack.h>
#include <vector>
#include "core/types.h"

namespace nnadapter {
namespace google_xnnpack {

// The following environment variables can be used at runtime:
// Specify the number of threads to use in XNNPACK thread pool, no thread
// pool/single-thread is used as default(default value is 0).
#define GOOGLE_XNNPACK_NUM_THREADS "GOOGLE_XNNPACK_NUM_THREADS"

// Get the bytes of the data type of XNNPACK
int XNNTensorDataTypeLength(xnn_datatype data_type);

// Convert NNAdapter types to XNNPACK types
xnn_datatype ConvertToXNNDataType(NNAdapterOperandPrecisionCode precision_code);
void ConvertToXNNDimensions(int32_t* input_dimensions,
                            uint32_t input_dimensions_count,
                            size_t* output_dimensions,
                            size_t* output_dimensions_count);
std::vector<size_t> ConvertToXNNDimensions(int32_t* input_dimensions,
                                           uint32_t input_dimensions_count);
// Convert the fused activation to the lower and upper bound for clipping output
// values
bool ConvertFuseCodeToXNNClippingRange(int32_t fuse_code,
                                       float* clipping_min,
                                       float* clipping_max);

}  // namespace google_xnnpack
}  // namespace nnadapter
