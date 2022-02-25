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

rk::nn::PrecisionType ConvertToRknnPrecisionType(
    NNAdapterOperandPrecisionCode input_precision) {
  rk::nn::PrecisionType output_precision = rk::nn::PrecisionType::UNKNOWN;
  switch (input_precision) {
    case NNADAPTER_BOOL8:
      output_precision = rk::nn::PrecisionType::BOOL8;
      break;
    case NNADAPTER_INT8:
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
      output_precision = rk::nn::PrecisionType::INT8;
      break;
    case NNADAPTER_INT16:
      output_precision = rk::nn::PrecisionType::INT16;
      break;
    case NNADAPTER_INT32:
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL:
      output_precision = rk::nn::PrecisionType::INT32;
      break;
    case NNADAPTER_INT64:
      output_precision = rk::nn::PrecisionType::INT64;
      break;
    case NNADAPTER_UINT8:
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      output_precision = rk::nn::PrecisionType::UINT8;
      break;
    case NNADAPTER_UINT16:
      output_precision = rk::nn::PrecisionType::UINT16;
      break;
    case NNADAPTER_QUANT_UINT32_ASYMM_PER_LAYER:
    case NNADAPTER_UINT32:
      output_precision = rk::nn::PrecisionType::UINT32;
      break;
    case NNADAPTER_UINT64:
      output_precision = rk::nn::PrecisionType::UINT64;
      break;
    case NNADAPTER_FLOAT16:
      output_precision = rk::nn::PrecisionType::FLOAT16;
      break;
    case NNADAPTER_FLOAT32:
      output_precision = rk::nn::PrecisionType::FLOAT32;
      break;
    case NNADAPTER_FLOAT64:
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

rk::nn::DataLayoutType ConvertToRknnDataLayoutType(
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

std::vector<int32_t> ConvertToRknnDimensions(int32_t* input_dimensions,
                                             uint32_t input_dimensions_count) {
  std::vector<int32_t> output_dimensions(input_dimensions_count);
  memcpy(&output_dimensions[0],
         input_dimensions,
         input_dimensions_count * sizeof(int32_t));
  return output_dimensions;
}

std::shared_ptr<rk::nn::Tensor> CreateRknnTensor(
    rk::nn::Graph* graph,
    const std::string& name,
    int32_t* dimensions_data,
    uint32_t dimensions_count,
    rk::nn::PrecisionType precision,
    const float* quant_scale,
    const int32_t* zero_point,
    void* buffer,
    rk::nn::DataLayoutType layout) {
  auto attr = std::make_shared<rk::nn::TensorAttr>();
  attr->name = name;
  attr->role = buffer ? rk::nn::TensorRole::CONST : rk::nn::TensorRole::VAR;
  attr->dims = ConvertToRknnDimensions(dimensions_data, dimensions_count);
  attr->precision = precision;
  attr->layout = layout;
  if (quant_scale) {
    // Quantization types
    if (precision == rk::nn::PrecisionType::UINT8) {
      attr->qntBits = 8;
    } else if (precision == rk::nn::PrecisionType::INT32) {
      attr->qntBits = 32;
    } else {
      NNADAPTER_LOG(FATAL)
          << "Only UINT8 and INT32 is supported for quantizaion.";
    }
    if (zero_point) {
      attr->qntType = rk::nn::QuantizationType::AFFINE_ASYMMETRIC;
      attr->qntParamAffineAsymmetric.scale.resize(1);
      attr->qntParamAffineAsymmetric.scale[0] = *quant_scale;
      attr->qntParamAffineAsymmetric.zero_point.resize(1);
      attr->qntParamAffineAsymmetric.zero_point[0] = *zero_point;
    } else {
      attr->qntType = rk::nn::QuantizationType::SYMMETRIC;
      attr->qntParamSymmetric.scale.resize(1);
      attr->qntParamSymmetric.scale[0] = *quant_scale;
    }
  } else {
    // TODO(hong19860320) Supports the normal types, such as float etc.
    NNADAPTER_LOG(FATAL) << "Only quantizaion types are supported.";
  }
  auto tensor = graph->CreateTensor(attr, buffer);
  NNADAPTER_CHECK(tensor);
  return tensor;
}

std::shared_ptr<rk::nn::Tensor> CreateRknnTensor(
    rk::nn::Graph* graph,
    const std::string& name,
    const NNAdapterOperandType* type,
    void* buffer,
    std::vector<int32_t> dimensions) {
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < type->dimensions.count; i++) {
      dimensions.push_back(type->dimensions.data[i]);
    }
  }
  auto precision = ConvertToRknnPrecisionType(type->precision);
  auto layout = ConvertToRknnDataLayoutType(type->layout);
  const float* quant_scale = nullptr;
  const int32_t* zero_point = nullptr;
  switch (type->precision) {
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      quant_scale = &type->asymm_per_layer_params.scale;
      zero_point = &type->asymm_per_layer_params.zero_point;
      break;
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER:
      quant_scale = &type->symm_per_layer_params.scale;
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Can not add a rk::nn::Tensor with precision="
                           << OperandPrecisionCodeToString(type->precision)
                           << " !";
      break;
  }
  return CreateRknnTensor(graph,
                          name,
                          dimensions.data(),
                          dimensions.size(),
                          precision,
                          quant_scale,
                          zero_point,
                          buffer,
                          layout);
}

}  // namespace rockchip_npu
}  // namespace nnadapter
