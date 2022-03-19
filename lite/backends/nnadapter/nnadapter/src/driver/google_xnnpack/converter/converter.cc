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

#include "driver/google_xnnpack/converter/converter.h"
#include <algorithm>
#include <utility>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace google_xnnpack {

#define REGISTER_CONVERTER(                                     \
    __op_type__, __validate_func_name__, __convert_func_name__) \
  extern int __convert_func_name__(Converter* converter,        \
                                   core::Operation* operation);
#include "driver/google_xnnpack/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_GOOGLE_XNNPACK_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

int Converter::Apply(core::Model* model) {
  // Convert the NNAdapter operations to the XNNPACK nodes
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
#define REGISTER_CONVERTER(                                     \
    __op_type__, __validate_func_name__, __convert_func_name__) \
  case NNADAPTER_##__op_type__:                                 \
    __convert_func_name__(this, operation);                     \
    break;
#include "driver/google_xnnpack/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_GOOGLE_XNNPACK_CONVERTER_ALL_H__
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

uint32_t Converter::GetMappedTensorValueId(core::Operand* operand) {
  auto it = tensor_value_ids_->find(operand);
  if (it != tensor_value_ids_->end()) {
    return it->second.back();
  }
  return XNN_INVALID_VALUE_ID;
}

uint32_t Converter::UpdateTensorValueIdMap(core::Operand* operand,
                                           uint32_t tensor_value_id) {
  auto it = tensor_value_ids_->find(operand);
  if (it == tensor_value_ids_->end()) {
    auto result = tensor_value_ids_->insert(
        std::make_pair(operand, std::vector<uint32_t>()));
    NNADAPTER_CHECK(result.second);
    it = result.first;
  }
  it->second.push_back(tensor_value_id);
  return tensor_value_id;
}

uint32_t Converter::AddTensorValue(int32_t* dimensions_data,
                                   uint32_t dimensions_count,
                                   xnn_datatype datatype,
                                   float* quant_scales,
                                   uint32_t quant_scale_count,
                                   uint32_t quant_channel_dim,
                                   void* buffer,
                                   uint32_t flags) {
  std::vector<size_t> converted_dimensions;
  if (dimensions_data && dimensions_count > 0) {
    converted_dimensions =
        ConvertToXNNDimensions(dimensions_data, dimensions_count);
  }
  uint32_t tensor_value_id = XNN_INVALID_VALUE_ID;
  if (quant_scales && quant_scale_count > 0) {
    // Quant type
    NNADAPTER_CHECK(datatype == xnn_datatype_qint8 ||
                    datatype == xnn_datatype_qint32);
    if (quant_scale_count > 1) {
      // Symmetric per-channel quantization
      xnn_define_channelwise_quantized_tensor_value(
          subgraph_,
          datatype,
          quant_scales,
          converted_dimensions.size(),
          quant_channel_dim,
          converted_dimensions.data(),
          buffer,
          static_cast<uint32_t>(XNN_INVALID_VALUE_ID),
          flags,
          &tensor_value_id);
    } else {
      // Symmetric per-layer quantization
      xnn_define_quantized_tensor_value(
          subgraph_,
          datatype,
          0,
          quant_scales[0],
          converted_dimensions.size(),
          converted_dimensions.data(),
          buffer,
          static_cast<uint32_t>(XNN_INVALID_VALUE_ID),
          flags,
          &tensor_value_id);
    }
  } else {
    // Basic type, without any quantization parameters
    xnn_define_tensor_value(subgraph_,
                            datatype,
                            converted_dimensions.size(),
                            converted_dimensions.data(),
                            buffer,
                            static_cast<uint32_t>(XNN_INVALID_VALUE_ID),
                            flags,
                            &tensor_value_id);
  }
  NNADAPTER_CHECK(tensor_value_id != XNN_INVALID_VALUE_ID);
  return tensor_value_id;
}

uint32_t Converter::AddFloat32ConstantTensorValue(float* values,
                                                  int32_t* dimensions_data,
                                                  uint32_t dimensions_count) {
  return AddTensorValue(dimensions_data,
                        dimensions_count,
                        xnn_datatype_fp32,
                        nullptr,
                        0,
                        0,
                        values);
}

uint32_t Converter::AddFloat32ConstantTensorValue(float* values,
                                                  uint32_t num_values) {
  std::vector<int32_t> dimensions({static_cast<int32_t>(num_values)});
  return AddFloat32ConstantTensorValue(
      values, dimensions.data(), dimensions.size());
}

uint32_t Converter::AddFloat32ConstantTensorValue(float value) {
  return AddFloat32ConstantTensorValue(&value, 1);
}

uint32_t Converter::AddQuant8ConstantTensorValue(int8_t* values,
                                                 int32_t* dimensions_data,
                                                 uint32_t dimensions_count,
                                                 float* quant_scales,
                                                 uint32_t quant_scale_count,
                                                 uint32_t quant_channel_dim) {
  return AddTensorValue(dimensions_data,
                        dimensions_count,
                        xnn_datatype_qint8,
                        quant_scales,
                        quant_scale_count,
                        quant_channel_dim,
                        values);
}

uint32_t Converter::AddQuant8ConstantTensorValue(int8_t* values,
                                                 int32_t* dimensions_data,
                                                 uint32_t dimensions_count,
                                                 float quant_scale) {
  return AddQuant8ConstantTensorValue(
      values, dimensions_data, dimensions_count, &quant_scale, 1, 0);
}

uint32_t Converter::AddQuant32ConstantTensorValue(int32_t* values,
                                                  int32_t* dimensions_data,
                                                  uint32_t dimensions_count,
                                                  float quant_scale) {
  return AddTensorValue(dimensions_data,
                        dimensions_count,
                        xnn_datatype_qint32,
                        &quant_scale,
                        1,
                        0,
                        values);
}

uint32_t Converter::AddFloat32VariableTensorValue(int32_t* dimensions_data,
                                                  uint32_t dimensions_count,
                                                  uint32_t flags) {
  return AddTensorValue(dimensions_data,
                        dimensions_count,
                        xnn_datatype_fp32,
                        nullptr,
                        0,
                        0,
                        nullptr,
                        flags);
}

uint32_t Converter::AddQuant8VariableTensorValue(int32_t* dimensions_data,
                                                 uint32_t dimensions_count,
                                                 float quant_scale,
                                                 uint32_t flags) {
  return AddTensorValue(dimensions_data,
                        dimensions_count,
                        xnn_datatype_qint8,
                        &quant_scale,
                        1,
                        0,
                        nullptr,
                        flags);
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
  auto is_constant = IsConstantOperandType(type);
  auto is_model_input = IsModelInputOperandType(type);
  auto is_model_output = IsModelOutputOperandType(type);
  uint32_t flags = 0;
  if (is_model_input) {
    flags |= XNN_VALUE_FLAG_EXTERNAL_INPUT;
  }
  if (is_model_output) {
    flags |= XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  }
  uint32_t tensor_value_id = XNN_INVALID_VALUE_ID;
  switch (type.precision) {
    case NNADAPTER_FLOAT32: {
      if (is_constant) {
        tensor_value_id =
            AddFloat32ConstantTensorValue(reinterpret_cast<float*>(buffer),
                                          dimensions.data(),
                                          dimensions.size());
      } else {
        tensor_value_id = AddFloat32VariableTensorValue(
            &dimensions[0], dimensions.size(), flags);
      }
    } break;
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER: {
      if (is_constant) {
        tensor_value_id =
            AddQuant8ConstantTensorValue(reinterpret_cast<int8_t*>(buffer),
                                         dimensions.data(),
                                         dimensions.size(),
                                         type.symm_per_layer_params.scale);
      } else {
        tensor_value_id =
            AddQuant8VariableTensorValue(&dimensions[0],
                                         dimensions.size(),
                                         type.symm_per_layer_params.scale,
                                         flags);
      }
    } break;
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL: {
      NNADAPTER_CHECK(is_constant);
      tensor_value_id = AddQuant8ConstantTensorValue(
          reinterpret_cast<int8_t*>(buffer),
          dimensions.data(),
          dimensions.size(),
          type.symm_per_channel_params.scales,
          type.symm_per_channel_params.scale_count,
          type.symm_per_channel_params.channel_dim);
    } break;
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER: {
      // Only for bias
      NNADAPTER_CHECK(is_constant);
      tensor_value_id =
          AddQuant32ConstantTensorValue(reinterpret_cast<int32_t*>(buffer),
                                        dimensions.data(),
                                        dimensions.size(),
                                        type.symm_per_layer_params.scale);
    } break;
    default:
      NNADAPTER_LOG(FATAL) << "Missing the processing "
                           << OperandPrecisionCodeToString(type.precision)
                           << " for the conversion of XNNPACK tensor value id.";
      break;
  }
  NNADAPTER_CHECK_NE(tensor_value_id, XNN_INVALID_VALUE_ID);
  UpdateTensorValueIdMap(operand, tensor_value_id);
  return tensor_value_id;
}

}  // namespace google_xnnpack
}  // namespace nnadapter
