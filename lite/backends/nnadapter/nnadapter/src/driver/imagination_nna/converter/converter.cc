// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/imagination_nna/converter/converter.h"
#include <unistd.h>
#include <algorithm>
#include <limits>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace imagination_nna {

#define REGISTER_CONVERTER(__op_type__, __func_name__) \
  extern int __func_name__(Converter* converter, core::Operation* operation);
#include "driver/imagination_nna/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_IMAGINATION_NNA_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

int Converter::Apply(core::Model* model) {
  // Convert the NNAdapter operations to the imgdnn operators
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
#define REGISTER_CONVERTER(__op_type__, __func_name__) \
  case NNADAPTER_##__op_type__:                        \
    __func_name__(this, operation);                    \
    break;
#include "driver/imagination_nna/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_IMAGINATION_NNA_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER
      default:
        NNADAPTER_LOG(FATAL) << "Unsupported operation("
                             << OperationTypeToString(operation->type)
                             << ") is found.";
        break;
    }
  }
  return NNADAPTER_NO_ERROR;
}

imgdnn_tensor Converter::GetMappedTensor(core::Operand* operand) {
  auto it = tensors_->find(operand);
  if (it != tensors_->end()) {
    return it->second.back();
  }
  return nullptr;
}

imgdnn_tensor Converter::UpdateTensorMap(core::Operand* operand,
                                         imgdnn_tensor tensor) {
  auto it = tensors_->find(operand);
  if (it == tensors_->end()) {
    auto result =
        tensors_->insert(std::make_pair(operand, std::vector<imgdnn_tensor>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(tensor);
  return tensor;
}

imgdnn_tensor Converter::AddTensor(int32_t* dimensions_data,
                                   uint32_t dimensions_count,
                                   imgdnn_type type,
                                   const float* quant_scales,
                                   const int32_t* zero_point,
                                   uint32_t quant_scale_count,
                                   uint32_t quant_channel_dim,
                                   void* buffer) {
  imgdnn_tensor tensor = nullptr;
  imgdnn_tensor_descriptor desc;
  desc.type = type;
  NNADAPTER_CHECK(dimensions_data);
  NNADAPTER_CHECK_GT(dimensions_count, 0);
  ConvertToImgdnnDimensions(
      dimensions_data, dimensions_count, desc.size, &desc.dimensions);
  if (quant_scales && quant_scale_count > 0) {
    // Quantization types
    if (quant_scale_count > 1) {
      // Symmetric and asymmetric per-channel quantization
      if (zero_point) {
        // Asymmetric per-channel quantization
        NNADAPTER_CHECK_EQ(type, IMGDNN_TYPE_QPA_U8);
      } else {
        // Symmetric per-channel quantization
        NNADAPTER_CHECK_EQ(type, IMGDNN_TYPE_QPA_I8);
      }
      desc.quant_param.per_axis = imgdnnCreatePerAxisQuantParam(
          quant_channel_dim, quant_scale_count, quant_scales, zero_point);
      NNADAPTER_CHECK(desc.quant_param.per_axis != nullptr);
      tensor = buffer ? imgdnn_mgr_->CreateFixedInputTensor(&desc, buffer, true)
                      : imgdnn_mgr_->CreateInputTensor(&desc);
      imgdnnDestroyPerAxisQuantParam(desc.quant_param.per_axis);
    } else {
      desc.quant_param.scale = quant_scales[0];
      if (zero_point) {
        // Asymmetric per-layer quantization
        NNADAPTER_CHECK_EQ(type, IMGDNN_TYPE_Q_U8);
        desc.quant_param.zero_point = zero_point[0];
      } else {
        // Symmetric per-layer quantization
        NNADAPTER_CHECK_EQ(type, IMGDNN_TYPE_Q_I8);
        // zeroPoint = 0
      }
      tensor = buffer ? imgdnn_mgr_->CreateFixedInputTensor(&desc, buffer, true)
                      : imgdnn_mgr_->CreateInputTensor(&desc);
    }
  } else {
    // TODO(hong19860320) Supports the normal types, such as float, int32 etc.
    NNADAPTER_CHECK_EQ(type, IMGDNN_TYPE_I32);
    NNADAPTER_CHECK(buffer);
    tensor = imgdnn_mgr_->CreateFixedInputTensor(&desc, buffer, true);
  }
  return tensor;
}

imgdnn_tensor Converter::AddTensor(const NNAdapterOperandType* type,
                                   void* buffer,
                                   std::vector<int32_t> dimensions) {
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < type->dimensions.count; i++) {
      dimensions.push_back(type->dimensions.data[i]);
    }
  }
  NNADAPTER_CHECK_EQ(type->layout, NNADAPTER_NCHW);
  const float* quant_scales = nullptr;
  const int32_t* zero_point = nullptr;
  uint32_t quant_scale_count = 0;
  uint32_t quant_channel_dim = 0;
  switch (type->precision) {
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      quant_scales = &type->asymm_per_layer_params.scale;
      zero_point = &type->asymm_per_layer_params.zero_point;
      quant_scale_count = 1;
      break;
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
      quant_scales = type->symm_per_channel_params.scales;
      quant_scale_count = type->symm_per_channel_params.scale_count;
      quant_channel_dim = type->symm_per_channel_params.channel_dim;
      break;
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL:
      // Only for bias
      NNADAPTER_CHECK(type->lifetime == NNADAPTER_CONSTANT_COPY ||
                      type->lifetime == NNADAPTER_CONSTANT_REFERENCE);
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Can not add a imgdnn_tensor with precision="
                           << OperandPrecisionCodeToString(type->precision)
                           << " !";
      break;
  }
  return AddTensor(dimensions.data(),
                   dimensions.size(),
                   ConvertToImgdnnPrecision(type->precision),
                   quant_scales,
                   zero_point,
                   quant_scale_count,
                   quant_channel_dim,
                   buffer);
}

imgdnn_tensor Converter::AddQuant8ConstantTensor(uint8_t* values,
                                                 int32_t* dimensions_data,
                                                 uint32_t dimensions_count,
                                                 float quant_scale,
                                                 int32_t zero_point) {
  return AddTensor(dimensions_data,
                   dimensions_count,
                   IMGDNN_TYPE_Q_U8,
                   &quant_scale,
                   &zero_point,
                   1,
                   0,
                   values);
}

imgdnn_tensor Converter::AddQuant8ConstantTensor(int8_t* values,
                                                 int32_t* dimensions_data,
                                                 uint32_t dimensions_count,
                                                 float* quant_scales,
                                                 uint32_t quant_scale_count,
                                                 uint32_t quant_channel_dim) {
  return AddTensor(dimensions_data,
                   dimensions_count,
                   IMGDNN_TYPE_Q_I8,
                   quant_scales,
                   nullptr,
                   quant_scale_count,
                   quant_channel_dim,
                   values);
}

imgdnn_tensor Converter::AddQuant32ConstantTensor(int32_t* values,
                                                  int32_t* dimensions_data,
                                                  uint32_t dimensions_count,
                                                  float quant_scale) {
  return AddTensor(dimensions_data,
                   dimensions_count,
                   IMGDNN_TYPE_I32,
                   &quant_scale,
                   nullptr,
                   1,
                   0,
                   values);
}

imgdnn_tensor Converter::ConvertOperand(core::Operand* operand,
                                        std::vector<int32_t> dimensions) {
  if (IsConstantOperand(operand)) {
    auto constant_tensor =
        AddTensor(&operand->type, operand->buffer, dimensions);
    UpdateTensorMap(operand, constant_tensor);
    return constant_tensor;
  } else if (IsModelInputOperand(operand)) {
    auto input_tensor = AddTensor(&operand->type, nullptr, dimensions);
    UpdateTensorMap(operand, input_tensor);
    return input_tensor;
  } else {
    NNADAPTER_LOG(FATAL) << "Only constant and model input operands can be "
                            "converted to imgdnn_tensor!"
                         << OperandToString(operand);
  }
  return nullptr;
}

}  // namespace imagination_nna
}  // namespace nnadapter
