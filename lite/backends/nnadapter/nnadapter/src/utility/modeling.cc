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

#include "utility/modeling.h"
#include <algorithm>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include "utility/cache.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/utility.h"

namespace nnadapter {

NNADAPTER_EXPORT void ClearModel(core::Model* model) {
  for (auto& operand : model->operands) {
    ClearOperand(&operand);
  }
}

NNADAPTER_EXPORT void ClearOperand(core::Operand* operand) {
  if (operand->type.lifetime != NNADAPTER_CONSTANT_REFERENCE &&
      operand->buffer && operand->length > 0) {
    free(operand->buffer);
    operand->buffer = nullptr;
    operand->length = 0;
  }
  if (IsPerChannelQuantType(operand->type.precision) &&
      operand->type.symm_per_channel_params.scales) {
    free(operand->type.symm_per_channel_params.scales);
    operand->type.symm_per_channel_params.scales = nullptr;
  }
  for (size_t i = 0; i < core::NNADAPTER_MAX_SIZE_OF_HINTS; i++) {
    if (operand->hints[i].handler) {
      operand->hints[i].deleter(&operand->hints[i].handler);
      operand->hints[i].handler = nullptr;
    }
  }
}

NNADAPTER_EXPORT core::Operand* AddOperand(core::Model* model) {
  model->operands.emplace_back();
  auto operand = &model->operands.back();
  memset(operand, 0, sizeof(core::Operand));
  return operand;
}

NNADAPTER_EXPORT core::Operation* AddOperation(core::Model* model) {
  model->operations.emplace_back();
  auto operation = &model->operations.back();
  memset(&operation->type, 0, sizeof(NNAdapterOperationType));
  return operation;
}

NNADAPTER_EXPORT void RemoveOperand(core::Model* model,
                                    core::Operand* operand) {
  for (auto it = model->operands.begin(); it != model->operands.end();) {
    if (&(*it) == operand) {
      ClearOperand(operand);
      it = model->operands.erase(it);
    } else {
      ++it;
    }
  }
}

NNADAPTER_EXPORT void RemoveOperation(core::Model* model,
                                      core::Operation* operation) {
  for (auto it = model->operations.begin(); it != model->operations.end();) {
    if (&(*it) == operation) {
      it = model->operations.erase(it);
    } else {
      ++it;
    }
  }
}

NNADAPTER_EXPORT void* AllocateOperand(core::Operand* operand) {
  if (operand->type.lifetime == NNADAPTER_CONSTANT_REFERENCE) {
    operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
    operand->buffer = nullptr;
    operand->length = 0;
  }
  auto length = GetOperandTypeBufferLength(operand->type);
  if (operand->length < length) {
    if (operand->buffer) {
      free(operand->buffer);
    }
    operand->buffer = malloc(length);
    NNADAPTER_CHECK(operand->buffer) << "Failed to allocate " << length
                                     << " bytes, out of memory!";
    operand->length = length;
  }
  NNADAPTER_CHECK(operand->buffer);
  NNADAPTER_CHECK_GT(operand->length, 0);
  return operand->buffer;
}

static core::Operand* AddOperand(core::Model* model,
                                 const std::vector<int32_t>& dimensions,
                                 NNAdapterOperandPrecisionCode precision,
                                 float* quant_scales = nullptr,
                                 int32_t* zero_point = nullptr,
                                 uint32_t quant_scale_count = 0,
                                 uint32_t quant_channel_dim = 0,
                                 void* buffer = nullptr,
                                 bool copy = true) {
  auto operand = AddOperand(model);
  memset(&operand->type, 0, sizeof(NNAdapterOperandType));
  operand->type.dimensions.count = dimensions.size();
  if (!dimensions.empty()) {
    memcpy(operand->type.dimensions.data,
           &dimensions[0],
           dimensions.size() * sizeof(int32_t));
  }
  operand->type.precision = precision;
  if (quant_scales && quant_scale_count > 0) {
    // Quant type
    if (quant_scale_count > 1) {
      // Symmetric per-channel quantization
      NNADAPTER_CHECK(!zero_point && IsSymmPerChannelQuantType(precision));
      float* scales =
          reinterpret_cast<float*>(malloc(quant_scale_count * sizeof(float)));
      NNADAPTER_CHECK(scales) << "Failed to allocate the Symmetric per-channel "
                                 "scale buffer for a operand.";
      memcpy(scales, quant_scales, quant_scale_count * sizeof(float));
      operand->type.symm_per_channel_params.scales = scales;
      operand->type.symm_per_channel_params.scale_count = quant_scale_count;
      operand->type.symm_per_channel_params.channel_dim = quant_channel_dim;
    } else {
      if (zero_point) {
        // Asymmetric per-layer quantization
        NNADAPTER_CHECK(IsAsymmPerLayerQuantType(precision));
        operand->type.asymm_per_layer_params.scale = quant_scales[0];
        operand->type.asymm_per_layer_params.zero_point = zero_point[0];
      } else {
        // Symmetric per-layer quantization
        NNADAPTER_CHECK(IsSymmPerLayerQuantType(precision));
        operand->type.symm_per_layer_params.scale = quant_scales[0];
      }
    }
  } else {
    // Basic type, without any quantization parameters
  }
  if (buffer) {
    // Constant operand
    operand->length = GetOperandPrecisionDataLength(precision) *
                      ProductionOfDimensions(dimensions);
    if (copy) {
      operand->buffer = malloc(operand->length);
      NNADAPTER_CHECK(operand->buffer != nullptr)
          << "Failed to allocate " << operand->length
          << " bytes for the buffer of an operand, out of memory!";
      memcpy(operand->buffer, buffer, operand->length);
      operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
    } else {
      operand->buffer = buffer;
      operand->type.lifetime = NNADAPTER_CONSTANT_REFERENCE;
    }
  } else {
    // Variable/Input/Output operand
    operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  }
  return operand;
}

NNADAPTER_EXPORT core::Operand* AddBool8ConstantOperand(core::Model* model,
                                                        bool value) {
  int8_t int8_value = value ? static_cast<int8_t>(1) : static_cast<int8_t>(0);
  return AddOperand(
      model, {}, NNADAPTER_INT32, nullptr, nullptr, 0, 0, &int8_value);
}

NNADAPTER_EXPORT core::Operand* AddInt32ConstantOperand(core::Model* model,
                                                        int32_t value) {
  return AddOperand(model, {}, NNADAPTER_INT32, nullptr, nullptr, 0, 0, &value);
}

NNADAPTER_EXPORT core::Operand* AddFloat32ConstantOperand(core::Model* model,
                                                          float value) {
  return AddOperand(
      model, {}, NNADAPTER_FLOAT32, nullptr, nullptr, 0, 0, &value);
}

NNADAPTER_EXPORT core::Operand* AddInt32ConstantOperand(
    core::Model* model, std::vector<int32_t> values) {
  std::vector<int32_t> dimensions({static_cast<int32_t>(values.size())});
  return AddOperand(
      model, dimensions, NNADAPTER_INT32, nullptr, nullptr, 0, 0, &values[0]);
}

NNADAPTER_EXPORT core::Operand* AddFloat32ConstantOperand(
    core::Model* model, std::vector<float> values) {
  std::vector<int32_t> dimensions({static_cast<int32_t>(values.size())});
  return AddOperand(
      model, dimensions, NNADAPTER_FLOAT32, nullptr, nullptr, 0, 0, &values[0]);
}

NNADAPTER_EXPORT core::Operand* AddInt32ConstantOperand(
    core::Model* model,
    int32_t* values,
    const std::vector<int32_t>& dimensions,
    bool copy) {
  return AddOperand(
      model, dimensions, NNADAPTER_INT32, nullptr, nullptr, 0, 0, values, copy);
}

NNADAPTER_EXPORT core::Operand* AddFloat32ConstantOperand(
    core::Model* model,
    float* values,
    const std::vector<int32_t>& dimensions,
    bool copy) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_FLOAT32,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    values,
                    copy);
}

NNADAPTER_EXPORT core::Operand* AddQuant8ConstantOperand(
    core::Model* model,
    int8_t* values,
    const std::vector<int32_t>& dimensions,
    float quant_scale,
    bool copy) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_QUANT_INT8_SYMM_PER_LAYER,
                    &quant_scale,
                    nullptr,
                    1,
                    0,
                    values,
                    copy);
}

NNADAPTER_EXPORT core::Operand* AddQuant8ConstantOperand(
    core::Model* model,
    int8_t* values,
    const std::vector<int32_t>& dimensions,
    float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim,
    bool copy) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL,
                    quant_scales,
                    nullptr,
                    quant_scale_count,
                    quant_channel_dim,
                    values,
                    copy);
}

NNADAPTER_EXPORT core::Operand* AddQuant8ConstantOperand(
    core::Model* model,
    uint8_t* values,
    const std::vector<int32_t>& dimensions,
    float quant_scale,
    int32_t zero_point,
    bool copy) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER,
                    &quant_scale,
                    &zero_point,
                    1,
                    0,
                    values,
                    copy);
}

NNADAPTER_EXPORT core::Operand* AddQuant32ConstantOperand(
    core::Model* model,
    int32_t* values,
    const std::vector<int32_t>& dimensions,
    float quant_scale,
    bool copy) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_QUANT_INT32_SYMM_PER_LAYER,
                    &quant_scale,
                    nullptr,
                    1,
                    0,
                    values,
                    copy);
}

NNADAPTER_EXPORT core::Operand* AddQuant32ConstantOperand(
    core::Model* model,
    int32_t* values,
    const std::vector<int32_t>& dimensions,
    float* quant_scales,
    uint32_t quant_scale_count,
    uint32_t quant_channel_dim,
    bool copy) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL,
                    quant_scales,
                    nullptr,
                    quant_scale_count,
                    quant_channel_dim,
                    values,
                    copy);
}

NNADAPTER_EXPORT core::Operand* AddQuant32ConstantOperand(
    core::Model* model,
    uint32_t* values,
    const std::vector<int32_t>& dimensions,
    float quant_scale,
    int32_t zero_point,
    bool copy) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_QUANT_UINT32_ASYMM_PER_LAYER,
                    &quant_scale,
                    &zero_point,
                    1,
                    0,
                    values,
                    copy);
}

NNADAPTER_EXPORT core::Operand* AddQuant8VariableOperand(
    core::Model* model,
    const std::vector<int32_t>& dimensions,
    float quant_scale) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_QUANT_INT8_SYMM_PER_LAYER,
                    &quant_scale,
                    nullptr,
                    1,
                    0,
                    nullptr,
                    false);
}

NNADAPTER_EXPORT core::Operand* AddQuant8VariableOperand(
    core::Model* model,
    const std::vector<int32_t>& dimensions,
    float quant_scale,
    int32_t zero_point) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER,
                    &quant_scale,
                    &zero_point,
                    1,
                    0,
                    nullptr,
                    false);
}

NNADAPTER_EXPORT core::Operand* AddFloat32VariableOperand(
    core::Model* model, const std::vector<int32_t>& dimensions) {
  return AddOperand(model,
                    dimensions,
                    NNADAPTER_FLOAT32,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    nullptr,
                    false);
}

NNADAPTER_EXPORT void ReshapeOperand(core::Operand* operand,
                                     std::vector<int32_t> dimensions) {
  ReshapeDimensions(operand->type.dimensions.data,
                    &operand->type.dimensions.count,
                    dimensions);
}

NNADAPTER_EXPORT void TransposeOperand(core::Operand* operand,
                                       std::vector<int32_t> permutation) {
  auto is_constant_copy = operand->type.lifetime == NNADAPTER_CONSTANT_COPY;
  auto is_constant_reference =
      operand->type.lifetime == NNADAPTER_CONSTANT_REFERENCE;
  auto is_constant = is_constant_copy || is_constant_reference;
  NNADAPTER_CHECK(!permutation.empty()) << "Permutation is empty!";
  NNADAPTER_CHECK_EQ(permutation.size(), operand->type.dimensions.count)
      << "The rank of permutation and operand mismatch!";
  if (is_constant) {
#define OPERAND_TRANSPOSE_DATA(bytes, dtype)                          \
  case bytes: {                                                       \
    auto src_buffer = reinterpret_cast<dtype*>(origin_buffer);        \
    auto dst_buffer = reinterpret_cast<dtype*>(transform_buffer);     \
    TransposeData<dtype>(                                             \
        src_buffer, dst_buffer, permutation, dimensions, dimensions); \
  } break;
    auto origin_buffer = operand->buffer;
    auto transform_buffer = malloc(operand->length);
    NNADAPTER_CHECK(transform_buffer) << "Out of memory!";
    auto dimensions = operand->type.dimensions.data;
    int bytes = GetOperandPrecisionDataLength(operand->type.precision);
    switch (bytes) {
      OPERAND_TRANSPOSE_DATA(1, int8_t);
      OPERAND_TRANSPOSE_DATA(2, int16_t);
      OPERAND_TRANSPOSE_DATA(4, int32_t);
      OPERAND_TRANSPOSE_DATA(8, int64_t);
      default:
        NNADAPTER_LOG(ERROR)
            << "Missing the processing of "
            << OperandPrecisionCodeToString(operand->type.precision)
            << " for the transpose of operand.";
        break;
    }
    if (is_constant_reference) {
      operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
    } else {
      // Free th origin buffer and replace it with the new one
      free(origin_buffer);
    }
    operand->buffer = transform_buffer;
#undef OPERAND_TRANSPOSE_DATA
  } else {
    // Only transpose the dimensions the non-constant operands
    TransposeDimensions(operand->type.dimensions.data, permutation);
  }
}

NNADAPTER_EXPORT void CopyOperand(core::Operand* dst_operand,
                                  core::Operand* src_operand,
                                  bool copy) {
  CopyOperandType(&dst_operand->type, src_operand->type);
  dst_operand->type.lifetime = src_operand->type.lifetime;
  if (IsTemporaryShapeOperand(dst_operand)) {
    SetTemporaryShape(dst_operand, *(GetTemporaryShape(src_operand)));
  } else if (IsConstantOperand(dst_operand)) {
    if (copy) {
      dst_operand->buffer = malloc(src_operand->length);
      NNADAPTER_CHECK(dst_operand->buffer != nullptr)
          << "Failed to allocate " << src_operand->length
          << " bytes for the buffer of an operand, out of memory!";
      memcpy(dst_operand->buffer, src_operand->buffer, src_operand->length);
      dst_operand->length = src_operand->length;
      dst_operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
    } else {
      dst_operand->buffer = src_operand->buffer;
      dst_operand->length = src_operand->length;
      dst_operand->type.lifetime = NNADAPTER_CONSTANT_REFERENCE;
    }
  }
}

NNADAPTER_EXPORT bool UpdateOperationInputOperands(
    std::vector<core::Operation*> operations,
    core::Operand* old_operand,
    core::Operand* new_operand) {
  if (operations.empty()) return false;
  // Update if any operation use the 'old_operand' as a input operand
  bool updated = false;
  for (auto& operation : operations) {
    for (auto& operand : operation->input_operands) {
      if (operand == old_operand) {
        operand = new_operand;
        updated = true;
      }
    }
  }
  return updated;
}

NNADAPTER_EXPORT bool UpdateOperationOutputOperands(
    core::Operation* operation,
    core::Operand* old_operand,
    core::Operand* new_operand) {
  if (!operation) return false;
  // Replace if the operation use the 'old_operand' as a output operand
  bool updated = false;
  for (auto& operand : operation->output_operands) {
    if (operand == old_operand) {
      operand = new_operand;
      updated = true;
    }
  }
  return updated;
}

NNADAPTER_EXPORT bool UpdateModelInputOperands(core::Model* model,
                                               core::Operand* old_operand,
                                               core::Operand* new_operand) {
  bool updated = false;
  if (IsModelInputOperand(old_operand)) {
    for (auto& operand : model->input_operands) {
      if (operand == old_operand) {
        operand = new_operand;
        updated = true;
      }
    }
    old_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
    new_operand->type.lifetime = NNADAPTER_MODEL_INPUT;
  }
  return updated;
}

NNADAPTER_EXPORT bool UpdateModelOutputOperands(core::Model* model,
                                                core::Operand* old_operand,
                                                core::Operand* new_operand) {
  bool updated = false;
  if (IsModelOutputOperand(old_operand)) {
    for (auto& operand : model->output_operands) {
      if (operand == old_operand) {
        operand = new_operand;
        updated = true;
      }
    }
    old_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
    new_operand->type.lifetime = NNADAPTER_MODEL_OUTPUT;
  }
  return updated;
}

NNADAPTER_EXPORT bool IsConstantOperand(core::Operand* operand) {
  return operand->type.lifetime == NNADAPTER_CONSTANT_COPY ||
         operand->type.lifetime == NNADAPTER_CONSTANT_REFERENCE;
}

NNADAPTER_EXPORT bool IsConstantCopyOperand(core::Operand* operand) {
  return operand->type.lifetime == NNADAPTER_CONSTANT_COPY;
}

NNADAPTER_EXPORT bool IsConstantReferenceOperand(core::Operand* operand) {
  return operand->type.lifetime == NNADAPTER_CONSTANT_REFERENCE;
}

NNADAPTER_EXPORT bool IsTemporaryVariableOperand(core::Operand* operand) {
  return operand->type.lifetime == NNADAPTER_TEMPORARY_VARIABLE;
}

NNADAPTER_EXPORT bool IsTemporaryShapeOperand(core::Operand* operand) {
  return operand->type.lifetime == NNADAPTER_TEMPORARY_SHAPE;
}

NNADAPTER_EXPORT bool IsModelInputOperand(core::Operand* operand) {
  return operand->type.lifetime == NNADAPTER_MODEL_INPUT;
}

NNADAPTER_EXPORT bool IsModelOutputOperand(core::Operand* operand) {
  return operand->type.lifetime == NNADAPTER_MODEL_OUTPUT;
}

NNADAPTER_EXPORT bool IsOperandWithDynamicShape(core::Operand* operand) {
  for (size_t i = 0; i < operand->type.dimensions.count; i++) {
    if (operand->type.dimensions.data[i] == NNADAPTER_UNKNOWN) {
      return true;
    }
  }
  return false;
}

NNADAPTER_EXPORT bool IsOperationWithAllInputConstantOperands(
    core::Operation* operation) {
  auto& input_operands = operation->input_operands;
  for (auto input_operand : input_operands) {
    if (!IsConstantOperand(input_operand)) {
      return false;
    }
  }
  return true;
}

std::vector<core::Operation*> GetOperandConsumers(core::Model* model,
                                                  core::Operand* operand) {
  std::vector<core::Operation*> consumers;
  if (operand) {
    for (auto& operation : model->operations) {
      auto& input_operands = operation.input_operands;
      if (std::find(input_operands.begin(), input_operands.end(), operand) ==
          input_operands.end())
        continue;
      consumers.push_back(&operation);
    }
  }
  return consumers;
}

NNADAPTER_EXPORT core::Operation* GetOperandProducer(core::Model* model,
                                                     core::Operand* operand) {
  core::Operation* producer = nullptr;
  if (operand) {
    for (auto& operation : model->operations) {
      auto& output_operands = operation.output_operands;
      if (std::find(output_operands.begin(), output_operands.end(), operand) ==
          output_operands.end())
        continue;
      // a operand has only one producer
      NNADAPTER_CHECK(producer == nullptr);
      producer = &operation;
    }
  }
  return producer;
}

NNADAPTER_EXPORT int GetModelInputOperandIndex(core::Model* model,
                                               core::Operand* operand) {
  if (IsModelInputOperand(operand)) {
    for (size_t i = 0; i < model->input_operands.size(); i++) {
      if (model->input_operands[i] == operand) {
        return i;
      }
    }
  }
  return -1;
}

NNADAPTER_EXPORT int GetModelOutputOperandIndex(core::Model* model,
                                                core::Operand* operand) {
  if (IsModelOutputOperand(operand)) {
    for (size_t i = 0; i < model->output_operands.size(); i++) {
      if (model->output_operands[i] == operand) {
        return i;
      }
    }
  }
  return -1;
}

core::Operand* AddTransposeOperation(core::Model* model,
                                     core::Operand* reference_operand,
                                     std::vector<int32_t> permutation,
                                     bool after = true) {
  auto target_operand = AddOperand(model);
  CopyOperandType(&target_operand->type, reference_operand->type);
  if (!IsTemporaryShapeOperand(reference_operand)) {
    target_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  }
  auto target_permutation =
      after ? permutation : InversePermutation(permutation);
  TransposeDimensions(target_operand->type.dimensions.data, target_permutation);
  for (uint32_t i = 0; i < target_operand->type.dimensions.dynamic_count; i++) {
    TransposeDimensions(target_operand->type.dimensions.dynamic_data[i],
                        target_permutation);
  }
  auto perm_operand = AddInt32ConstantOperand(model, permutation);
  auto transpose_operation = AddOperation(model);
  transpose_operation->type = NNADAPTER_TRANSPOSE;
  transpose_operation->input_operands = {
      after ? reference_operand : target_operand, perm_operand};
  transpose_operation->output_operands = {after ? target_operand
                                                : reference_operand};
  return target_operand;
}

NNADAPTER_EXPORT core::Operand* AppendTransposeOperation(
    core::Model* model,
    core::Operand* input_operand,
    std::vector<int32_t> permutation) {
  return AddTransposeOperation(model, input_operand, permutation, true);
}

NNADAPTER_EXPORT core::Operand* InsertTransposeOperation(
    core::Model* model,
    core::Operand* output_operand,
    std::vector<int32_t> permutation) {
  return AddTransposeOperation(model, output_operand, permutation, false);
}

NNADAPTER_EXPORT core::Operand* AppendReshapeOperation(
    core::Model* model,
    core::Operand* input_operand,
    std::vector<int32_t> shape) {
  auto output_operand = AddOperand(model);
  CopyOperandType(&output_operand->type, input_operand->type);
  if (!IsTemporaryShapeOperand(input_operand)) {
    output_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  }
  ReshapeDimensions(output_operand->type.dimensions.data,
                    &output_operand->type.dimensions.count,
                    shape);
  for (uint32_t i = 0; i < output_operand->type.dimensions.dynamic_count; i++) {
    ReshapeDimensions(output_operand->type.dimensions.dynamic_data[i],
                      &output_operand->type.dimensions.count,
                      shape);
  }
  auto shape_operand = AddInt32ConstantOperand(model, shape);
  auto reshape_operation = AddOperation(model);
  reshape_operation->type = NNADAPTER_RESHAPE;
  reshape_operation->input_operands = {input_operand, shape_operand};
  reshape_operation->output_operands = {output_operand};
  return output_operand;
}

NNADAPTER_EXPORT core::Operand* InsertReshapeOperation(
    core::Model* model,
    core::Operand* output_operand,
    const NNAdapterOperandDimensionType& input_dimensions,
    std::vector<int32_t> shape) {
  auto input_operand = AddOperand(model);
  CopyOperandType(&input_operand->type, output_operand->type);
  input_operand->type.dimensions = input_dimensions;
  if (!IsTemporaryShapeOperand(output_operand)) {
    input_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  }
  if (shape.empty()) {
    NNADAPTER_CHECK(!IsOperandWithDynamicShape(output_operand));
    shape = std::vector<int32_t>(output_operand->type.dimensions.data,
                                 output_operand->type.dimensions.data +
                                     output_operand->type.dimensions.count);
  }
  auto shape_operand = AddInt32ConstantOperand(model, shape);
  auto reshape_operation = AddOperation(model);
  reshape_operation->type = NNADAPTER_RESHAPE;
  reshape_operation->input_operands = {input_operand, shape_operand};
  reshape_operation->output_operands = {output_operand};
  return input_operand;
}

core::Operand* AddDummyOperation(core::Model* model,
                                 core::Operand* reference_operand,
                                 bool after = true) {
  auto target_operand = AddOperand(model);
  CopyOperandType(&target_operand->type, reference_operand->type);
  if (!IsTemporaryShapeOperand(reference_operand)) {
    target_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  }
  // Add a zero addend operand
  auto zero_operand = AddOperand(model);
  CopyOperandType(&zero_operand->type, reference_operand->type);
  zero_operand->type.dimensions.count = 1;
  zero_operand->type.dimensions.data[0] = 1;
  zero_operand->length =
      GetOperandPrecisionDataLength(zero_operand->type.precision);
  zero_operand->buffer = malloc(zero_operand->length);
  NNADAPTER_CHECK(zero_operand->buffer != nullptr)
      << "Failed to allocate " << zero_operand->length
      << " bytes for the buffer of an operand, out of memory!";
  memset(zero_operand->buffer, 0, zero_operand->length);
  zero_operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
  auto fuse_code_operand = AddInt32ConstantOperand(model, 0);
  // Insert a new ADD operation
  auto dummy_add_operation = AddOperation(model);
  dummy_add_operation->type = NNADAPTER_ADD;
  dummy_add_operation->input_operands = {
      after ? reference_operand : target_operand,
      zero_operand,
      fuse_code_operand};
  dummy_add_operation->output_operands = {after ? target_operand
                                                : reference_operand};
  return target_operand;
}

NNADAPTER_EXPORT core::Operand* AppendDummyOperation(
    core::Model* model, core::Operand* input_operand) {
  return AddDummyOperation(model, input_operand, true);
}

NNADAPTER_EXPORT core::Operand* InsertDummyOperation(
    core::Model* model, core::Operand* output_operand) {
  return AddDummyOperation(model, output_operand, false);
}

core::Operand* AddUnaryOperation(core::Model* model,
                                 core::Operand* reference_operand,
                                 NNAdapterOperationType operation_type,
                                 bool after = true) {
  auto target_operand = AddOperand(model);
  CopyOperandType(&target_operand->type, reference_operand->type);
  if (!IsTemporaryShapeOperand(reference_operand)) {
    target_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  }
  auto unary_operation = AddOperation(model);
  unary_operation->type = operation_type;
  unary_operation->input_operands = {after ? reference_operand
                                           : target_operand};
  unary_operation->output_operands = {after ? target_operand
                                            : reference_operand};
  return target_operand;
}

NNADAPTER_EXPORT core::Operand* AppendUnaryOperation(
    core::Model* model,
    core::Operand* input_operand,
    NNAdapterOperationType operation_type) {
  return AddUnaryOperation(model, input_operand, operation_type, true);
}

NNADAPTER_EXPORT core::Operand* InsertUnaryOperation(
    core::Model* model,
    core::Operand* output_operand,
    NNAdapterOperationType operation_type) {
  return AddUnaryOperation(model, output_operand, operation_type, false);
}

core::Operand* AddRequantOperation(core::Model* model,
                                   core::Operand* reference_operand,
                                   void* target_quant_params,
                                   bool after = true) {
  auto target_operand = AddOperand(model);
  CopyOperandType(&target_operand->type, reference_operand->type);
  if (!IsTemporaryShapeOperand(reference_operand)) {
    target_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  }
  // Copy the target_quant_params to target_operand
  if (IsSymmPerLayerQuantType(target_operand->type.precision)) {
    target_operand->type.symm_per_layer_params =
        *reinterpret_cast<NNAdapterSymmPerLayerQuantParams*>(
            target_quant_params);
  } else if (IsSymmPerChannelQuantType(target_operand->type.precision) &&
             target_operand->type.symm_per_channel_params.scales) {
    free(target_operand->type.symm_per_channel_params.scales);
    auto target_symm_per_channel_params =
        reinterpret_cast<NNAdapterSymmPerChannelQuantParams*>(
            target_quant_params);
    auto scale_size =
        target_symm_per_channel_params->scale_count * sizeof(float);
    auto scales = reinterpret_cast<float*>(malloc(scale_size));
    NNADAPTER_CHECK(scales) << "Failed to allocate the scale buffer for a symm "
                               "per-channel quant type!";
    memcpy(scales, target_symm_per_channel_params->scales, scale_size);
    target_operand->type.symm_per_channel_params.scales = scales;
  } else if (IsAsymmPerLayerQuantType(target_operand->type.precision)) {
    target_operand->type.asymm_per_layer_params =
        *reinterpret_cast<NNAdapterAsymmPerLayerQuantParams*>(
            target_quant_params);
  } else {
    NNADAPTER_LOG(FATAL)
        << "Unknown precision type("
        << OperandPrecisionCodeToString(target_operand->type.precision)
        << ") to identity the type of quantization parameters!";
  }
  // Add a zero addend operand
  auto zero_operand = AddOperand(model);
  CopyOperandType(&zero_operand->type, reference_operand->type);
  zero_operand->type.asymm_per_layer_params.zero_point = 0;
  zero_operand->type.dimensions.count = 1;
  zero_operand->type.dimensions.data[0] = 1;
  zero_operand->length =
      GetOperandPrecisionDataLength(zero_operand->type.precision);
  zero_operand->buffer = malloc(zero_operand->length);
  NNADAPTER_CHECK(zero_operand->buffer != nullptr)
      << "Failed to allocate " << zero_operand->length
      << " bytes for the buffer of an operand, out of memory!";
  memset(zero_operand->buffer, 0, zero_operand->length);
  zero_operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
  auto fuse_code_operand = AddInt32ConstantOperand(model, 0);
  // Insert a dummy ADD operation before/after reference_operand
  auto dummy_add_operation = AddOperation(model);
  dummy_add_operation->type = NNADAPTER_ADD;
  dummy_add_operation->input_operands = {
      after ? reference_operand : target_operand,
      zero_operand,
      fuse_code_operand};
  dummy_add_operation->output_operands = {after ? target_operand
                                                : reference_operand};
  return target_operand;
}

NNADAPTER_EXPORT core::Operand* AppendRequantOperation(
    core::Model* model,
    core::Operand* input_operand,
    void* output_quant_params) {
  return AddRequantOperation(model, input_operand, output_quant_params, true);
}

NNADAPTER_EXPORT core::Operand* InsertRequantOperation(
    core::Model* model,
    core::Operand* output_operand,
    void* input_quant_params) {
  return AddRequantOperation(model, output_operand, input_quant_params, false);
}

#define SORT_OPERATIONS_IN_TOPOLOGICAL_ORDER(T)                               \
  NNADAPTER_EXPORT std::vector<T core::Operation*>                            \
  SortOperationsInTopologicalOrder(T core::Model* model) {                    \
    NNADAPTER_VLOG(5) << "model total operands: " << model->operands.size();  \
    NNADAPTER_VLOG(5) << "model input operands: "                             \
                      << model->input_operands.size();                        \
    NNADAPTER_VLOG(5) << "model output operands: "                            \
                      << model->output_operands.size();                       \
    NNADAPTER_VLOG(5) << "model total operations: "                           \
                      << model->operations.size();                            \
    /* Operations in topological order */                                     \
    std::vector<T core::Operation*> operations;                               \
    std::vector<T core::Operation*> queue;                                    \
    /* Use to find all of adjacent operations according to a given operand.*/ \
    std::multimap<T core::Operand*, T core::Operation*> map;                  \
    /* The counters of variable inputs for all of operations. */              \
    std::map<T core::Operation*, uint32_t> counts;                            \
    for (auto& operation : model->operations) {                               \
      uint32_t count = 0;                                                     \
      for (auto operand : operation.input_operands) {                         \
        NNAdapterOperandLifetimeCode lifetime{NNADAPTER_CONSTANT_COPY};       \
        if (operand != nullptr) {                                             \
          lifetime = operand->type.lifetime;                                  \
        }                                                                     \
        if (lifetime == NNADAPTER_TEMPORARY_VARIABLE ||                       \
            lifetime == NNADAPTER_TEMPORARY_SHAPE ||                          \
            lifetime == NNADAPTER_MODEL_OUTPUT) {                             \
          count++;                                                            \
          map.insert(std::pair<T core::Operand*, T core::Operation*>(         \
              operand, &operation));                                          \
        }                                                                     \
      }                                                                       \
      if (count == 0) {                                                       \
        /* The operation which only depends the model inputs and constants */ \
        queue.push_back(&operation);                                          \
      }                                                                       \
      counts[&operation] = count;                                             \
    }                                                                         \
    while (queue.size() > 0) {                                                \
      auto operation = queue.back();                                          \
      queue.pop_back();                                                       \
      operations.push_back(operation);                                        \
      for (auto operand : operation->output_operands) {                       \
        auto range = map.equal_range(operand);                                \
        for (auto i = range.first; i != range.second; i++) {                  \
          uint32_t& count = counts[i->second];                                \
          if (--count == 0) {                                                 \
            queue.push_back(i->second);                                       \
          }                                                                   \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    return operations;                                                        \
  }

SORT_OPERATIONS_IN_TOPOLOGICAL_ORDER()
SORT_OPERATIONS_IN_TOPOLOGICAL_ORDER(const)

#undef SORT_OPERATIONS_IN_TOPOLOGICAL_ORDER

static const char* NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_OPERANDS_KEY =
    "operands";
static const char* NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_OPERATIONS_KEY =
    "operations";
static const char* NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_INPUT_OPERANDS_KEY =
    "input_operands";
static const char* NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_OUTPUT_OPERANDS_KEY =
    "output_operands";

inline void SerializeData(std::vector<uint8_t>* buffer,
                          size_t* offset,
                          void* data,
                          size_t length) {
  memcpy(buffer->data() + *offset, data, length);
  *offset += length;
}

template <typename T>
void SerializeData(std::vector<uint8_t>* buffer, size_t* offset, T value) {
  SerializeData(buffer, offset, &value, sizeof(T));
}

inline void DeserializeData(std::vector<uint8_t>* buffer,
                            size_t* offset,
                            void* data,
                            size_t length) {
  memcpy(data, buffer->data() + *offset, length);
  *offset += length;
}

template <typename T>
void DeserializeData(std::vector<uint8_t>* buffer, size_t* offset, T* value) {
  DeserializeData(buffer, offset, value, sizeof(T));
}

NNADAPTER_EXPORT bool SerializeModel(core::Model* model,
                                     std::vector<uint8_t>* buffer) {
  auto helper = std::make_shared<Cache>();
  // Serialize the model operands
  std::vector<uint8_t> value;
  size_t offset = sizeof(size_t);  // number of operands
  for (auto& operand : model->operands) {
    offset += sizeof(NNAdapterOperandType);
    if (IsPerChannelQuantType(operand.type.precision)) {
      offset +=
          operand.type.symm_per_channel_params.scale_count * sizeof(float);
    }
    if (IsConstantOperand(&operand)) {
      offset += sizeof(size_t) + operand.length;
    }
  }
  value.resize(offset);
  // Flatten the model operands
  offset = 0;
  SerializeData(&value, &offset, model->operands.size());
  int64_t operand_index = 0;
  std::unordered_map<core::Operand*, int64_t> operand_to_index;
  operand_to_index[0] = -1;  // Map to -1 if operand is nullptr
  for (auto& operand : model->operands) {
    // The type of the operand
    SerializeData(&value, &offset, &operand.type, sizeof(NNAdapterOperandType));
    // The quantization parameters of the operand
    if (IsPerChannelQuantType(operand.type.precision)) {
      NNADAPTER_CHECK(operand.type.symm_per_channel_params.scales);
      SerializeData(
          &value,
          &offset,
          operand.type.symm_per_channel_params.scales,
          operand.type.symm_per_channel_params.scale_count * sizeof(float));
    }
    // The constant values of the operand
    if (IsConstantOperandType(operand.type)) {
      NNADAPTER_CHECK(operand.buffer);
      SerializeData(&value, &offset, static_cast<size_t>(operand.length));
      SerializeData(&value, &offset, operand.buffer, operand.length);
    }
    operand_to_index[&operand] = operand_index++;
  }
  NNADAPTER_CHECK(
      helper->Set(NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_OPERANDS_KEY, value));
  // Serialize the model operations
  offset = sizeof(size_t);  // number of operations
  for (auto& operation : model->operations) {
    offset += sizeof(NNAdapterOperationType) + sizeof(size_t) +
              operation.input_operands.size() * sizeof(int64_t) +
              sizeof(size_t) +
              operation.output_operands.size() * sizeof(int64_t);
  }
  value.resize(offset);
  // Flatten the model operations
  offset = 0;
  SerializeData(&value, &offset, model->operations.size());
  for (auto& operation : model->operations) {
    // The type of the operation
    SerializeData(
        &value, &offset, &operation.type, sizeof(NNAdapterOperationType));
    // The indexes of the input operands of the operation
    SerializeData(&value, &offset, operation.input_operands.size());
    for (auto operand : operation.input_operands) {
      NNADAPTER_CHECK(operand_to_index.count(operand))
          << "Operand @" << OperandIdToString(operand) << " not found!";
      SerializeData(&value, &offset, operand_to_index[operand]);
    }
    // The indexes of the output operands of the operation
    SerializeData(&value, &offset, operation.output_operands.size());
    for (auto operand : operation.output_operands) {
      NNADAPTER_CHECK(operand_to_index.count(operand))
          << "Operand @" << OperandIdToString(operand) << " not found!";
      SerializeData(&value, &offset, operand_to_index[operand]);
    }
  }
  NNADAPTER_CHECK(
      helper->Set(NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_OPERATIONS_KEY, value));
  // Serialize the model input operands
  size_t num_input_operands = model->input_operands.size();
  value.resize(sizeof(size_t) + num_input_operands * sizeof(int64_t));
  // Flatten the model input operands
  offset = 0;
  SerializeData(&value, &offset, num_input_operands);
  for (auto operand : model->input_operands) {
    NNADAPTER_CHECK(operand_to_index.count(operand))
        << "Operand @" << OperandIdToString(operand) << " not found!";
    SerializeData(&value, &offset, operand_to_index[operand]);
  }
  NNADAPTER_CHECK(helper->Set(
      NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_INPUT_OPERANDS_KEY, value));
  // Serialize the model output operands
  size_t num_output_operands = model->output_operands.size();
  value.resize(sizeof(size_t) + num_output_operands * sizeof(int64_t));
  // Flatten the model output operands
  offset = 0;
  SerializeData(&value, &offset, num_output_operands);
  for (auto operand : model->output_operands) {
    NNADAPTER_CHECK(operand_to_index.count(operand))
        << "Operand @" << OperandIdToString(operand) << " not found!";
    SerializeData(&value, &offset, operand_to_index[operand]);
  }
  NNADAPTER_CHECK(helper->Set(
      NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_OUTPUT_OPERANDS_KEY, value));
  // Serialize all of model into buffer
  auto size = helper->GetSerializedSize();
  buffer->resize(size);
  return helper->Serialize(buffer->data(), size);
}

NNADAPTER_EXPORT bool DeserializeModel(void* buffer,
                                       uint64_t size,
                                       core::Model** model) {
  NNADAPTER_CHECK(model);
  *model = new core::Model();
  NNADAPTER_CHECK(*model)
      << "Failed to allocate the core::Model for restoring from buffer!";
  // Deserialize all of model from buffer
  auto helper = std::make_shared<Cache>();
  if (!helper->Deserialize(buffer, size)) {
    return false;
  }
  std::vector<uint8_t> value;
  // Deserialize the model operands
  NNADAPTER_CHECK(
      helper->Get(NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_OPERANDS_KEY, &value));
  size_t offset = 0;
  size_t num_operands;
  DeserializeData(&value, &offset, &num_operands);
  std::vector<core::Operand*> index_to_operand(num_operands);
  for (size_t i = 0; i < num_operands; i++) {
    auto operand = AddOperand(*model);
    // The type of the operand
    DeserializeData(
        &value, &offset, &operand->type, sizeof(NNAdapterOperandType));
    // The quantization parameters of the operand
    if (IsPerChannelQuantType(operand->type.precision)) {
      size_t length =
          operand->type.symm_per_channel_params.scale_count * sizeof(float);
      auto buffer = reinterpret_cast<float*>(malloc(length));
      NNADAPTER_CHECK(buffer) << "Failed to allocate the scale buffer for a "
                                 "symm per-channel quant type!";
      DeserializeData(&value, &offset, buffer, length);
      operand->type.symm_per_channel_params.scales = buffer;
    }
    // The constant values of the operand
    if (IsConstantOperandType(operand->type)) {
      size_t length;
      DeserializeData(&value, &offset, &length);
      auto buffer = reinterpret_cast<void*>(malloc(length));
      NNADAPTER_CHECK(buffer)
          << "Failed to allocate the buffer for a constant operand!";
      DeserializeData(&value, &offset, buffer, length);
      operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
      operand->buffer = buffer;
      operand->length = length;
    }
    index_to_operand[i] = operand;
  }
  // Deserialize the model operations
  NNADAPTER_CHECK(
      helper->Get(NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_OPERATIONS_KEY, &value));
  offset = 0;
  size_t num_operations;
  DeserializeData(&value, &offset, &num_operations);
  for (size_t i = 0; i < num_operations; i++) {
    auto operation = AddOperation(*model);
    // The type of the operation
    DeserializeData(
        &value, &offset, &operation->type, sizeof(NNAdapterOperationType));
    // The indexes of the input operands of the operation
    size_t num_input_operands;
    DeserializeData(&value, &offset, &num_input_operands);
    operation->input_operands.resize(num_input_operands);
    for (size_t j = 0; j < num_input_operands; j++) {
      int64_t operand_index;
      DeserializeData(&value, &offset, &operand_index);
      NNADAPTER_CHECK(operand_index == -1 ||
                      (operand_index >= 0 && operand_index < num_operands));
      operation->input_operands[j] =
          operand_index == -1 ? nullptr : index_to_operand[operand_index];
    }
    // The indexes of the output operands of the operation
    size_t num_output_operands;
    DeserializeData(&value, &offset, &num_output_operands);
    operation->output_operands.resize(num_output_operands);
    for (size_t j = 0; j < num_output_operands; j++) {
      int64_t operand_index;
      DeserializeData(&value, &offset, &operand_index);
      NNADAPTER_CHECK(operand_index == -1 ||
                      (operand_index >= 0 && operand_index < num_operands));
      operation->output_operands[j] =
          operand_index == -1 ? nullptr : index_to_operand[operand_index];
    }
  }
  // Deserialize the model input operands
  NNADAPTER_CHECK(helper->Get(
      NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_INPUT_OPERANDS_KEY, &value));
  offset = 0;
  size_t num_input_operands;
  DeserializeData(&value, &offset, &num_input_operands);
  (*model)->input_operands.resize(num_input_operands);
  for (size_t i = 0; i < num_input_operands; i++) {
    int64_t operand_index;
    DeserializeData(&value, &offset, &operand_index);
    NNADAPTER_CHECK(operand_index == -1 ||
                    (operand_index >= 0 && operand_index < num_operands));
    (*model)->input_operands[i] =
        operand_index == -1 ? nullptr : index_to_operand[operand_index];
  }
  // Deserialize the model output operands
  NNADAPTER_CHECK(helper->Get(
      NNADAPTER_RUNTIME_CACHE_CACHE_MODEL_OUTPUT_OPERANDS_KEY, &value));
  offset = 0;
  size_t num_output_operands;
  DeserializeData(&value, &offset, &num_output_operands);
  (*model)->output_operands.resize(num_output_operands);
  for (size_t i = 0; i < num_output_operands; i++) {
    int64_t operand_index;
    DeserializeData(&value, &offset, &operand_index);
    NNADAPTER_CHECK(operand_index == -1 ||
                    (operand_index >= 0 && operand_index < num_operands));
    (*model)->output_operands[i] =
        operand_index == -1 ? nullptr : index_to_operand[operand_index];
  }
  return true;
}

}  // namespace nnadapter
