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

#include "driver/google_xnnpack/utility.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace google_xnnpack {

int XNNOperandDataTypeLength(int data_type) {
  switch (data_type) {
    default:
      NNADAPTER_LOG(FATAL) << "Failed to get the length of XNNPACK data type("
                           << data_type << ")!";
      break;
  }
  return 0;
}

int ConvertToXNNPrecision(NNAdapterOperandPrecisionCode precision_code) {
  switch (precision_code) {
    case NNADAPTER_BOOL8:
      return 0;
    case NNADAPTER_FLOAT32:
      return 0;
    case NNADAPTER_INT32:
      return 0;
    case NNADAPTER_UINT32:
      return 0;
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
      return 0;
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
      return 0;
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      return 0;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(precision_code)
          << ") to the XNNPACK datatype!";
      break;
  }
  return 0;
}

int ConvertToXNNDataLayout(NNAdapterOperandLayoutCode layout_code) {
  NNADAPTER_CHECK_EQ(layout_code, NNADAPTER_NHWC)
      << "XNNPACK only supports NHWC data layout!";
  return 0;
}

std::vector<size_t> ConvertToXNNDimensions(int32_t* input_dimensions,
                                           uint32_t input_dimensions_count) {
  std::vector<size_t> output_dimensions(input_dimensions_count);
  for (size_t i = 0; i < input_dimensions_count; i++) {
    auto dimension = input_dimensions[i];
    NNADAPTER_CHECK_GE(dimension, 0);
    output_dimensions[i] = static_cast<size_t>(dimension);
  }
  return output_dimensions;
}

int32_t ConvertFuseCodeToXNNFuseCode(int32_t fuse_code) {
  switch (fuse_code) {
    case NNADAPTER_FUSED_NONE:
      return 0;
    case NNADAPTER_FUSED_RELU:
      return 0;
    case NNADAPTER_FUSED_RELU1:
      return 0;
    case NNADAPTER_FUSED_RELU6:
      return 0;
    default:
      NNADAPTER_LOG(FATAL) << "Failed to convert the NNAdapter fuse code("
                           << fuse_code << ") to the XNNPACK fuse code!";
      break;
  }
  return 0;
}

}  // namespace google_xnnpack
}  // namespace nnadapter
