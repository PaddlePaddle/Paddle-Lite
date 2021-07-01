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

#include "driver/mediatek_apu/utility.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace mediatek_apu {

int ConvertPrecision(NNAdapterOperandPrecisionCode input_precision) {
  int output_precision = 0;
  switch (input_precision) {
    case NNADAPTER_BOOL8:
      output_precision = NEURON_BOOL;
      break;
    case NNADAPTER_FLOAT16:
      output_precision = NEURON_FLOAT16;
      break;
    case NNADAPTER_FLOAT32:
      output_precision = NEURON_FLOAT32;
      break;
    case NNADAPTER_INT32:
      output_precision = NEURON_INT32;
      break;
    case NNADAPTER_UINT32:
      output_precision = NEURON_UINT32;
      break;
    case NNADAPTER_TENSOR_BOOL8:
      output_precision = NEURON_TENSOR_BOOL8;
      break;
    case NNADAPTER_TENSOR_FLOAT16:
      output_precision = NEURON_TENSOR_FLOAT16;
      break;
    case NNADAPTER_TENSOR_FLOAT32:
      output_precision = NEURON_TENSOR_FLOAT32;
      break;
    case NNADAPTER_TENSOR_INT32:
      output_precision = NEURON_TENSOR_INT32;
      break;
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER:
      output_precision = NEURON_TENSOR_QUANT8_SYMM;
      break;
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL:
      output_precision = NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL;
      break;
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER:
      output_precision = NEURON_TENSOR_QUANT8_ASYMM;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(input_precision)
          << ") to the Neuron operand precision code!";
      break;
  }
  return output_precision;
}

int ConvertDataLayout(NNAdapterOperandLayoutCode input_layout) {
  NNADAPTER_CHECK_EQ(input_layout, NNADAPTER_NHWC)
      << "Neuron only supports NHWC data layout!";
  return 0;
}

std::vector<uint32_t> ConvertDimensions(int32_t* input_dimensions,
                                        uint32_t input_dimensions_count) {
  std::vector<uint32_t> output_dimensions(input_dimensions_count);
  for (size_t i = 0; i < input_dimensions_count; i++) {
    auto dimension = input_dimensions[i];
    NNADAPTER_CHECK_GE(dimension, 0);
    output_dimensions[i] = static_cast<uint32_t>(dimension);
  }
  return output_dimensions;
}

int32_t ConvertFuseCode(int32_t input_fuse_code) {
  int output_fuse_code = NEURON_FUSED_NONE;
  switch (input_fuse_code) {
    case NNADAPTER_FUSED_NONE:
      output_fuse_code = NEURON_FUSED_NONE;
      break;
    case NNADAPTER_FUSED_RELU:
      output_fuse_code = NEURON_FUSED_RELU;
      break;
    case NNADAPTER_FUSED_RELU1:
      output_fuse_code = NEURON_FUSED_RELU1;
      break;
    case NNADAPTER_FUSED_RELU6:
      output_fuse_code = NEURON_FUSED_RELU6;
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Failed to convert the NNAdapter fuse code("
                           << input_fuse_code << ") to the Neuron fuse code!";
      break;
  }
  return output_fuse_code;
}

int PrecisionLength(int precision) {
  switch (precision) {
    case NEURON_BOOL:
    case NEURON_TENSOR_BOOL8:
    case NEURON_TENSOR_QUANT8_ASYMM:
    case NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL:
    case NEURON_TENSOR_QUANT8_SYMM:
      return 1;
    case NEURON_FLOAT16:
    case NEURON_TENSOR_FLOAT16:
    case NEURON_TENSOR_QUANT16_ASYMM:
    case NEURON_TENSOR_QUANT16_SYMM:
      return 2;
    case NEURON_INT32:
    case NEURON_UINT32:
    case NEURON_FLOAT32:
    case NEURON_TENSOR_INT32:
    case NEURON_TENSOR_FLOAT32:
      return 4;
    default:
      NNADAPTER_LOG(FATAL) << "Failed to get the length of type(" << precision
                           << ")!";
      break;
  }
  return 0;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
