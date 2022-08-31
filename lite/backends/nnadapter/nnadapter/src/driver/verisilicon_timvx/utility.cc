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

#include "driver/verisilicon_timvx/utility.h"
#include <map>
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace verisilicon_timvx {

tim::vx::DataType ConvertToTimVXDataType(
    NNAdapterOperandPrecisionCode input_precision) {
  tim::vx::DataType output_precision = tim::vx::DataType::UNKNOWN;
  switch (input_precision) {
    case NNADAPTER_BOOL8:
      output_precision = tim::vx::DataType::BOOL8;
      break;
    case NNADAPTER_INT8:
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
      output_precision = tim::vx::DataType::INT8;
      break;
    case NNADAPTER_INT16:
    case NNADAPTER_QUANT_INT16_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT16_SYMM_PER_CHANNEL:
      output_precision = tim::vx::DataType::INT16;
      break;
    case NNADAPTER_INT32:
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL:
      output_precision = tim::vx::DataType::INT32;
      break;
    case NNADAPTER_UINT8:
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      output_precision = tim::vx::DataType::UINT8;
      break;
    case NNADAPTER_UINT16:
    case NNADAPTER_QUANT_UINT16_ASYMM_PER_LAYER:
      output_precision = tim::vx::DataType::UINT16;
      break;
    case NNADAPTER_UINT32:
    case NNADAPTER_QUANT_UINT32_ASYMM_PER_LAYER:
      output_precision = tim::vx::DataType::UINT32;
      break;
    case NNADAPTER_FLOAT16:
      output_precision = tim::vx::DataType::FLOAT16;
      break;
    case NNADAPTER_FLOAT32:
      output_precision = tim::vx::DataType::FLOAT32;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(input_precision)
          << ") to tim::vx::DataType !";
      break;
  }
  return output_precision;
}

tim::vx::DataLayout ConvertToTimVXDataLayout(
    NNAdapterOperandLayoutCode input_layout) {
  tim::vx::DataLayout output_layout = tim::vx::DataLayout::ANY;
  switch (input_layout) {
    case NNADAPTER_NCHW:
      output_layout = tim::vx::DataLayout::WHCN;
      break;
    case NNADAPTER_NHWC:
      output_layout = tim::vx::DataLayout::CWHN;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand layout code("
          << OperandLayoutCodeToString(input_layout)
          << ") to tim::vx::DataLayout !";
      break;
  }
  return output_layout;
}

tim::vx::ShapeType ConvertToTimVXShapeType(int32_t* input_dimensions,
                                           uint32_t input_dimensions_count) {
  tim::vx::ShapeType output_dimensions;
  // NCHW->WHCN
  for (int i = input_dimensions_count - 1; i >= 0; i--) {
    output_dimensions.push_back(input_dimensions[i]);
  }
  return output_dimensions;
}

tim::vx::TensorAttribute ConvertToTimVXTensorAttribute(
    NNAdapterOperandLifetimeCode input_lifetime) {
  tim::vx::TensorAttribute output_lifetime;
  switch (input_lifetime) {
    case NNADAPTER_MODEL_INPUT:
      output_lifetime = tim::vx::TensorAttribute::INPUT;
      break;
    case NNADAPTER_MODEL_OUTPUT:
      output_lifetime = tim::vx::TensorAttribute::OUTPUT;
      break;
    case NNADAPTER_TEMPORARY_SHAPE:
    case NNADAPTER_TEMPORARY_VARIABLE:
      output_lifetime = tim::vx::TensorAttribute::TRANSIENT;
      break;
    case NNADAPTER_CONSTANT_COPY:
    case NNADAPTER_CONSTANT_REFERENCE:
      output_lifetime = tim::vx::TensorAttribute::CONSTANT;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand lifetime code("
          << OperandLifetimeCodeToString(input_lifetime)
          << ") to tim::vx::TensorAttribute !";
      break;
  }
  return output_lifetime;
}

std::vector<uint32_t> ConvertToTimVXPermutation(const int32_t* input_perm_data,
                                                size_t input_perm_count) {
  std::vector<uint32_t> output_perm;
  for (int i = input_perm_count - 1; i >= 0; i--) {
    output_perm.push_back(input_perm_count - 1 - input_perm_data[i]);
  }
  return output_perm;
}

int32_t ConvertToTimVXAxis(int32_t axis, size_t dimension_count) {
  return dimension_count - 1 - (axis < 0 ? dimension_count + axis : axis);
}

std::shared_ptr<tim::vx::Tensor> CreateTimVXTensor(
    tim::vx::Graph* graph,
    tim::vx::ShapeType shape,
    tim::vx::DataType data_type,
    const float* quant_scale,
    const int32_t* quant_zero_point,
    uint32_t quant_scale_count,
    int32_t quant_channel_dim,
    void* buffer,
    tim::vx::DataLayout data_layout,
    tim::vx::TensorAttribute tensor_attr) {
  tim::vx::TensorSpec tensor_spec;
  tensor_spec.SetDataType(data_type);
  tensor_spec.SetShape(shape);
  if (quant_scale) {
    NNADAPTER_CHECK_GT(quant_scale_count, 0);
    NNADAPTER_CHECK(data_type == tim::vx::DataType::INT8 ||
                    data_type == tim::vx::DataType::UINT8 ||
                    data_type == tim::vx::DataType::INT32)
        << "Only INT8, UINT8 and INT32 is supported for quantizaion.";
    // Check quantization types
    tim::vx::Quantization quantization;
    if (quant_scale_count > 1) {
      NNADAPTER_CHECK_GE(quant_channel_dim, 0);
      quantization.SetType(tim::vx::QuantType::SYMMETRIC_PER_CHANNEL);
      quantization.SetChannelDim(quant_channel_dim);
    } else {
      quantization.SetType(tim::vx::QuantType::ASYMMETRIC);
    }
    quantization.SetScales(
        std::vector<float>(quant_scale, quant_scale + quant_scale_count));
    if (quant_zero_point) {
      quantization.SetZeroPoints(std::vector<int32_t>(
          quant_zero_point, quant_zero_point + quant_scale_count));
    } else {
      quantization.SetZeroPoints(std::vector<int32_t>(quant_scale_count, 0));
    }
    tensor_spec.SetQuantization(quantization);
  } else {
    // TODO(hong19860320) Supports the normal types, such as float etc.
  }
  tensor_spec.SetAttribute(tensor_attr);
  NNADAPTER_CHECK(
      (!buffer && tensor_attr != tim::vx::TensorAttribute::CONSTANT) ||
      (buffer && tensor_attr == tim::vx::TensorAttribute::CONSTANT));
  auto tensor = buffer ? graph->CreateTensor(tensor_spec, buffer)
                       : graph->CreateTensor(tensor_spec);
  NNADAPTER_CHECK(tensor);
  return tensor;
}

std::shared_ptr<tim::vx::Tensor> CreateTimVXTensor(
    tim::vx::Graph* graph,
    const NNAdapterOperandType* type,
    void* buffer,
    std::vector<int32_t> dimensions) {
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < type->dimensions.count; i++) {
      dimensions.push_back(type->dimensions.data[i]);
    }
  }
  auto shape = ConvertToTimVXShapeType(dimensions.data(), dimensions.size());
  auto data_type = ConvertToTimVXDataType(type->precision);
  auto data_layout = ConvertToTimVXDataLayout(type->layout);
  auto tensor_attr = ConvertToTimVXTensorAttribute(type->lifetime);
  const float* quant_scale = nullptr;
  const int32_t* quant_zero_point = nullptr;
  uint32_t quant_scale_count = 0;
  int32_t quant_channel_dim = -1;
  switch (type->precision) {
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
    case NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL:
      quant_scale = type->symm_per_channel_params.scales;
      quant_scale_count = type->symm_per_channel_params.scale_count;
      quant_channel_dim =
          dimensions.size() - 1 - type->symm_per_channel_params.channel_dim;
      break;
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      quant_scale = &type->asymm_per_layer_params.scale;
      quant_zero_point = &type->asymm_per_layer_params.zero_point;
      quant_scale_count = 1;
      break;
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER:
      quant_scale = &type->symm_per_layer_params.scale;
      quant_scale_count = 1;
      break;
    case NNADAPTER_FLOAT32:
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Can't add a tim::vx::Tensor with precision="
                           << OperandPrecisionCodeToString(type->precision)
                           << " !";
      break;
  }
  return CreateTimVXTensor(graph,
                           shape,
                           data_type,
                           quant_scale,
                           quant_zero_point,
                           quant_scale_count,
                           quant_channel_dim,
                           buffer,
                           data_layout,
                           tensor_attr);
}
}  // namespace verisilicon_timvx
}  // namespace nnadapter
