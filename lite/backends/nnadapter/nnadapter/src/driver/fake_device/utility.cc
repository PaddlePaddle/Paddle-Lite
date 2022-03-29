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

#include "utility.h"  // NOLINT
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace fake_device {

fake_ddk::PrecisionType ConvertToFakeDevicePrecisionType(
    NNAdapterOperandPrecisionCode input_precision) {
  fake_ddk::PrecisionType output_precision = fake_ddk::PrecisionType::FLOAT32;
  switch (input_precision) {
    case NNADAPTER_BOOL8:
      output_precision = fake_ddk::PrecisionType::BOOL8;
      break;
    case NNADAPTER_INT8:
      output_precision = fake_ddk::PrecisionType::INT8;
      break;
    case NNADAPTER_INT16:
      output_precision = fake_ddk::PrecisionType::INT16;
      break;
    case NNADAPTER_INT32:
      output_precision = fake_ddk::PrecisionType::INT32;
      break;
    case NNADAPTER_INT64:
      output_precision = fake_ddk::PrecisionType::INT64;
      break;
    case NNADAPTER_UINT8:
      output_precision = fake_ddk::PrecisionType::UINT8;
      break;
    case NNADAPTER_UINT16:
      output_precision = fake_ddk::PrecisionType::UINT16;
      break;
    case NNADAPTER_UINT32:
      output_precision = fake_ddk::PrecisionType::UINT32;
      break;
    case NNADAPTER_UINT64:
      output_precision = fake_ddk::PrecisionType::UINT64;
      break;
    case NNADAPTER_FLOAT16:
      output_precision = fake_ddk::PrecisionType::FLOAT16;
      break;
    case NNADAPTER_FLOAT32:
      output_precision = fake_ddk::PrecisionType::FLOAT32;
      break;
    case NNADAPTER_FLOAT64:
      output_precision = fake_ddk::PrecisionType::FLOAT64;
      break;
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
      output_precision = fake_ddk::PrecisionType::QUANT_INT8_SYMM_PER_LAYER;
      break;
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
      output_precision = fake_ddk::PrecisionType::QUANT_INT8_SYMM_PER_CHANNEL;
      break;
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER:
      output_precision = fake_ddk::PrecisionType::QUANT_INT32_SYMM_PER_LAYER;
      break;
    case NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL:
      output_precision = fake_ddk::PrecisionType::QUANT_INT32_SYMM_PER_CHANNEL;
      break;
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      output_precision = fake_ddk::PrecisionType::QUANT_UINT8_ASYMM_PER_LAYER;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(input_precision)
          << ") to fake_ddk::PrecisionType !";
      break;
  }
  return output_precision;
}

fake_ddk::DataLayoutType ConvertToFakeDeviceDataLayoutType(
    NNAdapterOperandLayoutCode input_layout) {
  fake_ddk::DataLayoutType output_layout = fake_ddk::DataLayoutType::NCHW;
  switch (input_layout) {
    case NNADAPTER_NCHW:
      output_layout = fake_ddk::DataLayoutType::NCHW;
      break;
    case NNADAPTER_NHWC:
      output_layout = fake_ddk::DataLayoutType::NHWC;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand layout code("
          << OperandLayoutCodeToString(input_layout)
          << ") to fake_ddk::DataLayoutType !";
      break;
  }
  return output_layout;
}

std::vector<int32_t> ConvertToFakeDeviceDimensions(
    int32_t* input_dimensions, uint32_t input_dimensions_count) {
  std::vector<int32_t> output_dimensions(input_dimensions_count);
  memcpy(&output_dimensions[0],
         input_dimensions,
         input_dimensions_count * sizeof(int32_t));
  return output_dimensions;
}

fake_ddk::FuseType ConvertFuseCodeToFakeDeviceFuseType(int32_t fuse_code) {
  switch (fuse_code) {
    case NNADAPTER_FUSED_NONE:
      return fake_ddk::FuseType::FUSE_NONE;
    case NNADAPTER_FUSED_RELU:
      return fake_ddk::FuseType::FUSE_RELU;
    case NNADAPTER_FUSED_RELU1:
      return fake_ddk::FuseType::FUSE_RELU1;
    case NNADAPTER_FUSED_RELU6:
      return fake_ddk::FuseType::FUSE_RELU6;
    default:
      NNADAPTER_LOG(FATAL) << "Failed to convert the NNAdapter fuse code("
                           << fuse_code << ") to a fake_ddk::FuseType !";
      break;
  }
  return fake_ddk::FuseType::FUSE_NONE;
}

}  // namespace fake_device
}  // namespace nnadapter
