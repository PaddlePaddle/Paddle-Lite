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

#include "driver/amlogic_npu/utility.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace amlogic_npu {

aml::nn::PrecisionType ConvertPrecision(
    NNAdapterOperandPrecisionCode input_precision) {
  aml::nn::PrecisionType output_precision = aml::nn::PrecisionType::UNKNOWN;
  switch (input_precision) {
    case NNADAPTER_TENSOR_BOOL8:
      output_precision = aml::nn::PrecisionType::BOOL8;
      break;
    case NNADAPTER_TENSOR_INT8:
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL:
      output_precision = aml::nn::PrecisionType::INT8;
      break;
    case NNADAPTER_TENSOR_INT16:
      output_precision = aml::nn::PrecisionType::INT16;
      break;
    case NNADAPTER_TENSOR_INT32:
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_TENSOR_QUANT_INT32_SYMM_PER_CHANNEL:
      output_precision = aml::nn::PrecisionType::INT32;
      break;
    case NNADAPTER_TENSOR_INT64:
      output_precision = aml::nn::PrecisionType::INT64;
      break;
    case NNADAPTER_TENSOR_UINT8:
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER:
      output_precision = aml::nn::PrecisionType::UINT8;
      break;
    case NNADAPTER_TENSOR_UINT16:
      output_precision = aml::nn::PrecisionType::UINT16;
      break;
    case NNADAPTER_TENSOR_QUANT_UINT32_ASYMM_PER_LAYER:
    case NNADAPTER_TENSOR_UINT32:
      output_precision = aml::nn::PrecisionType::UINT32;
      break;
    case NNADAPTER_TENSOR_UINT64:
      output_precision = aml::nn::PrecisionType::UINT64;
      break;
    case NNADAPTER_TENSOR_FLOAT16:
      output_precision = aml::nn::PrecisionType::FLOAT16;
      break;
    case NNADAPTER_TENSOR_FLOAT32:
      output_precision = aml::nn::PrecisionType::FLOAT32;
      break;
    case NNADAPTER_TENSOR_FLOAT64:
      output_precision = aml::nn::PrecisionType::FLOAT64;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(input_precision)
          << ") to aml::nn::PrecisionType !";
      break;
  }
  return output_precision;
}

aml::nn::DataLayoutType ConvertDataLayout(
    NNAdapterOperandLayoutCode input_layout) {
  aml::nn::DataLayoutType output_layout = aml::nn::DataLayoutType::UNKNOWN;
  switch (input_layout) {
    case NNADAPTER_NCHW:
      output_layout = aml::nn::DataLayoutType::NCHW;
      break;
    case NNADAPTER_NHWC:
      output_layout = aml::nn::DataLayoutType::NHWC;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand layout code("
          << OperandLayoutCodeToString(input_layout)
          << ") to aml::nn::DataLayoutType !";
      break;
  }
  return output_layout;
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

}  // namespace amlogic_npu
}  // namespace nnadapter
