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

#include "driver/android_nnapi/utility.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace android_nnapi {

int NNOperandDataTypeLength(int data_type) {
  switch (data_type) {
    case ANEURALNETWORKS_BOOL:
    case ANEURALNETWORKS_TENSOR_BOOL8:
    case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
    case ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL:
    case ANEURALNETWORKS_TENSOR_QUANT8_SYMM:
      return 1;
    case ANEURALNETWORKS_TENSOR_FLOAT16:
    case ANEURALNETWORKS_TENSOR_QUANT16_SYMM:
      return 2;
    case ANEURALNETWORKS_INT32:
    case ANEURALNETWORKS_UINT32:
    case ANEURALNETWORKS_FLOAT32:
    case ANEURALNETWORKS_TENSOR_INT32:
    case ANEURALNETWORKS_TENSOR_FLOAT32:
      return 4;
    default:
      NNADAPTER_LOG(FATAL) << "Failed to get the length of NNAPI data type("
                           << data_type << ")!";
      break;
  }
  return 0;
}

int ConvertToNNPrecision(NNAdapterOperandPrecisionCode precision_code) {
  switch (precision_code) {
    case NNADAPTER_BOOL8:
      return ANEURALNETWORKS_BOOL;
    case NNADAPTER_FLOAT32:
      return ANEURALNETWORKS_FLOAT32;
    case NNADAPTER_INT32:
      return ANEURALNETWORKS_INT32;
    case NNADAPTER_UINT32:
      return ANEURALNETWORKS_UINT32;
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
      return ANEURALNETWORKS_TENSOR_QUANT8_SYMM;
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
      return ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL;
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      return ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(precision_code)
          << ") to the NNAPI operand precision code!";
      break;
  }
  return 0;
}

int ConvertToNNDataLayout(NNAdapterOperandLayoutCode layout_code) {
  NNADAPTER_CHECK_EQ(layout_code, NNADAPTER_NHWC)
      << "NNAPI only supports NHWC data layout!";
  return 0;
}

std::vector<uint32_t> ConvertToNNDimensions(int32_t* input_dimensions,
                                            uint32_t input_dimensions_count) {
  std::vector<uint32_t> output_dimensions(input_dimensions_count);
  for (size_t i = 0; i < input_dimensions_count; i++) {
    auto dimension = input_dimensions[i];
    NNADAPTER_CHECK_GE(dimension, 0);
    output_dimensions[i] = static_cast<uint32_t>(dimension);
  }
  return output_dimensions;
}

int32_t ConvertFuseCodeToNNFuseCode(int32_t fuse_code) {
  switch (fuse_code) {
    case NNADAPTER_FUSED_NONE:
      return ANEURALNETWORKS_FUSED_NONE;
    case NNADAPTER_FUSED_RELU:
      return ANEURALNETWORKS_FUSED_RELU;
    case NNADAPTER_FUSED_RELU1:
      return ANEURALNETWORKS_FUSED_RELU1;
    case NNADAPTER_FUSED_RELU6:
      return ANEURALNETWORKS_FUSED_RELU6;
    default:
      NNADAPTER_LOG(FATAL) << "Failed to convert the NNAdapter fuse code("
                           << fuse_code << ") to the NNAPI fuse code!";
      break;
  }
  return ANEURALNETWORKS_FUSED_NONE;
}

}  // namespace android_nnapi
}  // namespace nnadapter
