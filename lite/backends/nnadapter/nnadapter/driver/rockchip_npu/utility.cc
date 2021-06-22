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

#include "driver/rockchip_npu/utility.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace rockchip_npu {

rk::nn::PrecisionType ConvertPrecision(
    NNAdapterOperandPrecisionCode input_precision) {
  rk::nn::PrecisionType output_precision = rk::nn::PrecisionType::UNKNOWN;
  switch (input_precision) {
    case NNADAPTER_TENSOR_BOOL8:
      output_precision = rk::nn::PrecisionType::BOOL8;
      break;
    case NNADAPTER_TENSOR_INT8:
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL:
      output_precision = rk::nn::PrecisionType::INT8;
      break;
    case NNADAPTER_TENSOR_INT16:
      output_precision = rk::nn::PrecisionType::INT16;
      break;
    case NNADAPTER_TENSOR_INT32:
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL:
      output_precision = rk::nn::PrecisionType::INT32;
      break;
    case NNADAPTER_TENSOR_INT64:
      output_precision = rk::nn::PrecisionType::INT64;
      break;
    case NNADAPTER_TENSOR_UINT8:
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER:
      output_precision = rk::nn::PrecisionType::UINT8;
      break;
    case NNADAPTER_TENSOR_UINT16:
      output_precision = rk::nn::PrecisionType::UINT16;
      break;
    case NNADAPTER_TENSOR_QUANT_UINT32_ASYMM_PER_LAYER:
    case NNADAPTER_TENSOR_UINT32:
      output_precision = rk::nn::PrecisionType::UINT32;
      break;
    case NNADAPTER_TENSOR_UINT64:
      output_precision = rk::nn::PrecisionType::UINT64;
      break;
    case NNADAPTER_TENSOR_FLOAT16:
      output_precision = rk::nn::PrecisionType::FLOAT16;
      break;
    case NNADAPTER_TENSOR_FLOAT32:
      output_precision = rk::nn::PrecisionType::FLOAT32;
      break;
    case NNADAPTER_TENSOR_FLOAT64:
      output_precision = rk::nn::PrecisionType::FLOAT64;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(input_precision)
          << ") to rk::nn::PrecisionType !";
      break;
  }
  return output_precision;
}

rk::nn::DataLayoutType ConvertDataLayout(
    NNAdapterOperandLayoutCode input_layout) {
  rk::nn::DataLayoutType output_layout = rk::nn::DataLayoutType::UNKNOWN;
  switch (input_layout) {
    case NNADAPTER_NCHW:
      output_layout = rk::nn::DataLayoutType::NCHW;
      break;
    case NNADAPTER_NHWC:
      output_layout = rk::nn::DataLayoutType::NHWC;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand layout code("
          << OperandLayoutCodeToString(input_layout)
          << ") to rk::nn::DataLayoutType !";
      break;
  }
  return output_layout;
}

std::vector<int32_t> ConvertDimensions(int32_t* input_dimensions,
                                       uint32_t input_dimensions_count) {
  std::vector<int32_t> output_dimensions(input_dimensions_count);
  memcpy(&output_dimensions[0],
         input_dimensions,
         input_dimensions_count * sizeof(int32_t));
  return output_dimensions;
}

}  // namespace rockchip_npu
}  // namespace nnadapter
