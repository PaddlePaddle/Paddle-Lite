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

#include "driver/mediatek_apu/converter/converter.h"
#include <algorithm>
#include <utility>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace mediatek_apu {

#define REGISTER_CONVERTER(__op_type__, __func_name__) \
  extern int __func_name__(Converter* converter, core::Operation* operation);
#include "driver/mediatek_apu/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_MEDIATEK_APU_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

int Converter::Apply(core::Model* model) {
  operand_index_ = 0;
  // Convert the NNAdapter operations to the Neuron operations
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
#include "driver/mediatek_apu/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_MEDIATEK_APU_CONVERTER_ALL_H__
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

uint32_t Converter::GetMappedIndex(core::Operand* operand) {
  auto it = operand_indexes_->find(operand);
  if (it != operand_indexes_->end()) {
    return it->second.back();
  }
  return INVALID_INDEX;
}

uint32_t Converter::UpdateIndexMap(core::Operand* operand, uint32_t index) {
  auto it = operand_indexes_->find(operand);
  if (it == operand_indexes_->end()) {
    auto result = operand_indexes_->insert(
        std::make_pair(operand, std::vector<uint32_t>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(index);
  return index;
}

uint32_t Converter::AddOperand(int32_t* dimensions_data,
                               uint32_t dimensions_count,
                               int precision,
                               float* quant_scales,
                               int32_t* zero_point,
                               uint32_t quant_scale_count,
                               uint32_t quant_channel_dim,
                               void* buffer) {
  NeuronOperandType type;
  memset(&type, 0, sizeof(NeuronOperandType));
  type.type = precision;
  std::vector<uint32_t> converted_dimensions;
  if (dimensions_data && dimensions_count > 0) {
    converted_dimensions =
        ConvertToNeuronDimensions(dimensions_data, dimensions_count);
    type.dimensions = &converted_dimensions[0];
  }
  type.dimensionCount = converted_dimensions.size();
  NeuronSymmPerChannelQuantParams symm_per_channel_quant_params;
  memset(&symm_per_channel_quant_params,
         0,
         sizeof(NeuronSymmPerChannelQuantParams));
  if (quant_scales && quant_scale_count > 0) {
    // Quant type
    if (quant_scale_count > 1) {
      // Symmetric per-channel quantization
      NNADAPTER_CHECK(!zero_point &&
                          precision == NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL ||
                      precision == NEURON_TENSOR_INT32);
      symm_per_channel_quant_params.scales = quant_scales;
      symm_per_channel_quant_params.scaleCount = quant_scale_count;
      symm_per_channel_quant_params.channelDim = quant_channel_dim;
    } else {
      type.scale = quant_scales[0];
      if (zero_point) {
        // Asymmetric per-layer quantization
        NNADAPTER_CHECK(precision == NEURON_TENSOR_QUANT8_ASYMM);
        type.zeroPoint = zero_point[0];
      } else {
        // Symmetric per-layer quantization
        NNADAPTER_CHECK(precision == NEURON_TENSOR_INT32);
        // zeroPoint = 0
      }
    }
  } else {
    // Basic type, without any quantization parameters
  }
  NNADAPTER_CHECK_EQ(NeuronModel_addOperand_invoke(model_, &type),
                     NEURON_NO_ERROR);
  auto index = operand_index_++;
  if (buffer) {
    // Constant operand
    auto length = NeuronOperandDataTypeLength(precision) *
                  ProductionOfDimensions(dimensions_data, dimensions_count);
    NNADAPTER_CHECK_EQ(
        NeuronModel_setOperandValue_invoke(model_, index, buffer, length),
        NEURON_NO_ERROR);
  } else {
    // Variable/Input/Output operand
  }
  if (quant_scales && quant_scale_count > 1) {
    // Symmetric per-channel quantization
    NNADAPTER_CHECK_EQ(NeuronModel_setOperandSymmPerChannelQuantParams_invoke(
                           model_, index, &symm_per_channel_quant_params),
                       NEURON_NO_ERROR);
  }
  return index;
}

int Converter::AddOperation(NeuronOperationType type,
                            const std::vector<uint32_t>& input_indexes,
                            const std::vector<uint32_t>& output_indexes) {
  return NeuronModel_addOperation_invoke(model_,
                                         type,
                                         input_indexes.size(),
                                         input_indexes.data(),
                                         output_indexes.size(),
                                         output_indexes.data());
}

uint32_t Converter::AddBool8ConstantOperand(bool value) {
  int8_t int8_value = value ? static_cast<int8_t>(1) : static_cast<int8_t>(0);
  return AddOperand(
      nullptr, 0, NEURON_BOOL, nullptr, nullptr, 0, 0, &int8_value);
}

uint32_t Converter::AddInt32ConstantOperand(int32_t value) {
  return AddOperand(nullptr, 0, NEURON_INT32, nullptr, nullptr, 0, 0, &value);
}

uint32_t Converter::AddFloat32ConstantOperand(float value) {
  return AddOperand(nullptr, 0, NEURON_FLOAT32, nullptr, nullptr, 0, 0, &value);
}

uint32_t Converter::AddInt32ConstantOperand(int32_t* values,
                                            uint32_t num_values) {
  std::vector<int32_t> dimensions({static_cast<int32_t>(num_values)});
  return AddOperand(&dimensions[0],
                    dimensions.size(),
                    NEURON_TENSOR_INT32,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    values);
}

uint32_t Converter::AddFloat32ConstantOperand(float* values,
                                              uint32_t num_values) {
  std::vector<int32_t> dimensions({static_cast<int32_t>(num_values)});
  return AddOperand(&dimensions[0],
                    dimensions.size(),
                    NEURON_TENSOR_FLOAT32,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    values);
}

uint32_t Converter::AddInt32ConstantOperand(int32_t* values,
                                            int32_t* dimensions_data,
                                            uint32_t dimensions_count) {
  return AddOperand(dimensions_data,
                    dimensions_count,
                    NEURON_TENSOR_INT32,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    values);
}

uint32_t Converter::AddFloat32ConstantOperand(float* values,
                                              int32_t* dimensions_data,
                                              uint32_t dimensions_count) {
  return AddOperand(dimensions_data,
                    dimensions_count,
                    NEURON_TENSOR_FLOAT32,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    values);
}

uint32_t Converter::AddQuant8ConstantOperand(int8_t* values,
                                             int32_t* dimensions_data,
                                             uint32_t dimensions_count,
                                             float* quant_scales,
                                             uint32_t quant_scale_count,
                                             uint32_t quant_channel_dim) {
  return AddOperand(dimensions_data,
                    dimensions_count,
                    NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL,
                    quant_scales,
                    nullptr,
                    quant_scale_count,
                    quant_channel_dim,
                    values);
}

uint32_t Converter::AddQuant8ConstantOperand(uint8_t* values,
                                             int32_t* dimensions_data,
                                             uint32_t dimensions_count,
                                             float quant_scale,
                                             int32_t zero_point) {
  return AddOperand(dimensions_data,
                    dimensions_count,
                    NEURON_TENSOR_QUANT8_ASYMM,
                    &quant_scale,
                    &zero_point,
                    1,
                    0,
                    values);
}

uint32_t Converter::AddQuant32ConstantOperand(int32_t* values,
                                              int32_t* dimensions_data,
                                              uint32_t dimensions_count,
                                              float quant_scale) {
  return AddOperand(dimensions_data,
                    dimensions_count,
                    NEURON_TENSOR_INT32,
                    &quant_scale,
                    nullptr,
                    1,
                    0,
                    values);
}

uint32_t Converter::AddFloat32VariableOperand(int32_t* dimensions_data,
                                              uint32_t dimensions_count) {
  return AddOperand(dimensions_data, dimensions_count, NEURON_TENSOR_FLOAT32);
}

uint32_t Converter::AddQuant8VariableOperand(int32_t* dimensions_data,
                                             uint32_t dimensions_count,
                                             float quant_scale,
                                             int32_t zero_point) {
  return AddOperand(dimensions_data,
                    dimensions_count,
                    NEURON_TENSOR_QUANT8_ASYMM,
                    &quant_scale,
                    &zero_point,
                    1,
                    0);
}

uint32_t Converter::ConvertOperand(core::Operand* operand,
                                   std::vector<int32_t> dimensions) {
  auto& type = operand->type;
  auto buffer = operand->buffer;
  if (dimensions.empty()) {
    for (uint32_t i = 0; i < type.dimensions.count; i++) {
      dimensions.push_back(type.dimensions.data[i]);
    }
  }
  auto is_constant_copy = type.lifetime == NNADAPTER_CONSTANT_COPY;
  auto is_constant_reference = type.lifetime == NNADAPTER_CONSTANT_REFERENCE;
  auto is_constant = is_constant_copy || is_constant_reference;
  uint32_t index = INVALID_INDEX;
  switch (type.precision) {
    case NNADAPTER_FLOAT32: {
      if (is_constant) {
        index = AddFloat32ConstantOperand(reinterpret_cast<float*>(buffer),
                                          &dimensions[0],
                                          dimensions.size());
      } else {
        index = AddFloat32VariableOperand(&dimensions[0], dimensions.size());
      }
    } break;
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER: {
      if (is_constant) {
        index =
            AddQuant8ConstantOperand(reinterpret_cast<uint8_t*>(buffer),
                                     &dimensions[0],
                                     dimensions.size(),
                                     type.asymm_per_layer_params.scale,
                                     type.asymm_per_layer_params.zero_point);
      } else {
        index =
            AddQuant8VariableOperand(&dimensions[0],
                                     dimensions.size(),
                                     type.asymm_per_layer_params.scale,
                                     type.asymm_per_layer_params.zero_point);
      }
    } break;
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL: {
      NNADAPTER_CHECK(is_constant);
      index =
          AddQuant8ConstantOperand(reinterpret_cast<int8_t*>(buffer),
                                   &dimensions[0],
                                   dimensions.size(),
                                   type.symm_per_channel_params.scales,
                                   type.symm_per_channel_params.scale_count,
                                   type.symm_per_channel_params.channel_dim);
    } break;
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER: {
      // Only for bias
      NNADAPTER_CHECK(is_constant);
      index = AddQuant32ConstantOperand(reinterpret_cast<int32_t*>(buffer),
                                        &dimensions[0],
                                        dimensions.size(),
                                        type.symm_per_layer_params.scale);
    } break;
    case NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL: {
      // Only for bias
      NNADAPTER_CHECK(is_constant);
      index = AddInt32ConstantOperand(reinterpret_cast<int32_t*>(buffer),
                                      &dimensions[0],
                                      dimensions.size());
    } break;
    default:
      NNADAPTER_LOG(FATAL) << "Missing the processing "
                           << OperandPrecisionCodeToString(type.precision)
                           << " for the conversion of Neuron operands.";
      break;
  }
  NNADAPTER_CHECK_NE(index, INVALID_INDEX);
  UpdateIndexMap(operand, index);
  return index;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
