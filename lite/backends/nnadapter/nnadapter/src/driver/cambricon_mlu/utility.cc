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

#include "driver/cambricon_mlu/utility.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

magicmind::DataType ConvertToMagicMindDtype(
    NNAdapterOperandPrecisionCode input_precision) {
  magicmind::DataType output_precision;
  switch (input_precision) {
    case NNADAPTER_BOOL8:
      output_precision = magicmind::DataType::BOOL;
      break;
    case NNADAPTER_INT8:
      output_precision = magicmind::DataType::INT8;
      break;
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
      output_precision = magicmind::DataType::QINT8;
      break;
    case NNADAPTER_INT16:
      output_precision = magicmind::DataType::INT16;
      break;
    case NNADAPTER_QUANT_INT16_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT16_SYMM_PER_CHANNEL:
      output_precision = magicmind::DataType::QINT16;
      break;
    case NNADAPTER_INT32:
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL:
      output_precision = magicmind::DataType::INT32;
      break;
    case NNADAPTER_INT64:
      output_precision = magicmind::DataType::INT64;
      break;
    case NNADAPTER_UINT8:
      output_precision = magicmind::DataType::UINT8;
      break;
    case NNADAPTER_UINT16:
      output_precision = magicmind::DataType::UINT16;
      break;
    case NNADAPTER_UINT32:
    case NNADAPTER_QUANT_UINT32_ASYMM_PER_LAYER:
      output_precision = magicmind::DataType::UINT32;
      break;
    case NNADAPTER_FLOAT16:
      output_precision = magicmind::DataType::FLOAT16;
      break;
    case NNADAPTER_FLOAT32:
      output_precision = magicmind::DataType::FLOAT32;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(input_precision)
          << ") to magicmind dtype !";
      break;
  }
  return output_precision;
}

magicmind::Layout ConvertToMagicMindDataLayout(
    NNAdapterOperandLayoutCode input_layout) {
  magicmind::Layout output_layout;
  switch (input_layout) {
    case NNADAPTER_NCHW:
      output_layout = magicmind::Layout::NCHW;
      break;
    case NNADAPTER_NHWC:
      output_layout = magicmind::Layout::NHWC;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand layout code("
          << OperandLayoutCodeToString(input_layout)
          << ") to magicmind layout !";
      break;
  }
  return output_layout;
}

int64_t ConvertToMagicMindAxis(NNAdapterOperandLayoutCode input_layout) {
  int64_t axis;
  switch (input_layout) {
    case NNADAPTER_NCHW:
      axis = 1;
      break;
    case NNADAPTER_NHWC:
      axis = 3;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand layout code("
          << OperandLayoutCodeToString(input_layout) << ") to magicmind axis !";
      break;
  }
  return axis;
}

magicmind::Dims ConvertToMagicMindDims(const int32_t* input_dimensions,
                                       uint32_t input_dimensions_count) {
  std::vector<int64_t> output_dimensions;
  for (uint32_t i = 0; i < input_dimensions_count; i++) {
    output_dimensions.push_back(static_cast<int64_t>(input_dimensions[i]));
  }
  return magicmind::Dims(output_dimensions);
}

bool IsDeviceMemory(magicmind::IRTTensor* pointer) {
  auto location = pointer->GetMemoryLocation();
  switch (location) {
    case magicmind::TensorLocation::kMLU:
      return true;
    case magicmind::TensorLocation::kHost:
      return false;
    default:
      NNADAPTER_LOG(WARNING) << "Unknown memory space.";
      return false;
  }
}

bool IsScalar(magicmind::Dims dim) {
  auto dim_num = dim.GetDimsNum();
  auto element_count = dim.GetElementCount();
  return dim_num == 0 && element_count == 1;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
