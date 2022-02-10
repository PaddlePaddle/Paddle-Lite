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
#include <utility>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/utility.h"

namespace nnadapter {

NNADAPTER_EXPORT hal::Operand* AddOperand(hal::Model* model) {
  model->operands.emplace_back();
  auto operand = &model->operands.back();
  memset(operand, 0, sizeof(hal::Operand));
  return operand;
}

NNADAPTER_EXPORT hal::Operation* AddOperation(hal::Model* model) {
  model->operations.emplace_back();
  auto operation = &model->operations.back();
  memset(&operation->type, 0, sizeof(NNAdapterOperationType));
  return operation;
}

NNADAPTER_EXPORT void RemoveOperand(hal::Model* model, hal::Operand* operand) {
  for (auto it = model->operands.begin(); it != model->operands.end();) {
    if (&(*it) == operand) {
      if ((operand->type.lifetime == NNADAPTER_CONSTANT_COPY ||
           operand->type.lifetime == NNADAPTER_TEMPORARY_SHAPE) &&
          operand->buffer) {
        free(operand->buffer);
      }
      if ((operand->type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL ||
           operand->type.precision == NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL) &&
          operand->type.symm_per_channel_params.scales) {
        free(operand->type.symm_per_channel_params.scales);
      }
      for (size_t i = 0; i < hal::NNADAPTER_MAX_SIZE_OF_HINTS; i++) {
        if (operand->hints[i].handler) {
          operand->hints[i].deleter(&operand->hints[i].handler);
        }
      }
      it = model->operands.erase(it);
    } else {
      ++it;
    }
  }
}

NNADAPTER_EXPORT void RemoveOperation(hal::Model* model,
                                      hal::Operation* operation) {
  for (auto it = model->operations.begin(); it != model->operations.end();) {
    if (&(*it) == operation) {
      it = model->operations.erase(it);
    } else {
      ++it;
    }
  }
}

static hal::Operand* AddOperand(hal::Model* model,
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
      operand->type.symm_per_channel_params.scales = quant_scales;
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

NNADAPTER_EXPORT hal::Operand* AddBool8ConstantOperand(hal::Model* model,
                                                       bool value) {
  int8_t int8_value = value ? static_cast<int8_t>(1) : static_cast<int8_t>(0);
  return AddOperand(
      model, {}, NNADAPTER_INT32, nullptr, nullptr, 0, 0, &int8_value);
}

NNADAPTER_EXPORT hal::Operand* AddInt32ConstantOperand(hal::Model* model,
                                                       int32_t value) {
  return AddOperand(model, {}, NNADAPTER_INT32, nullptr, nullptr, 0, 0, &value);
}

NNADAPTER_EXPORT hal::Operand* AddFloat32ConstantOperand(hal::Model* model,
                                                         float value) {
  return AddOperand(
      model, {}, NNADAPTER_FLOAT32, nullptr, nullptr, 0, 0, &value);
}

NNADAPTER_EXPORT hal::Operand* AddInt32ConstantOperand(
    hal::Model* model, std::vector<int32_t> values) {
  std::vector<int32_t> dimensions({static_cast<int32_t>(values.size())});
  return AddOperand(
      model, dimensions, NNADAPTER_INT32, nullptr, nullptr, 0, 0, &values[0]);
}

NNADAPTER_EXPORT hal::Operand* AddFloat32ConstantOperand(
    hal::Model* model, std::vector<float> values) {
  std::vector<int32_t> dimensions({static_cast<int32_t>(values.size())});
  return AddOperand(
      model, dimensions, NNADAPTER_FLOAT32, nullptr, nullptr, 0, 0, &values[0]);
}

NNADAPTER_EXPORT hal::Operand* AddInt32ConstantOperand(
    hal::Model* model,
    int32_t* values,
    const std::vector<int32_t>& dimensions,
    bool copy) {
  return AddOperand(
      model, dimensions, NNADAPTER_INT32, nullptr, nullptr, 0, 0, values, copy);
}

NNADAPTER_EXPORT hal::Operand* AddFloat32ConstantOperand(
    hal::Model* model,
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

NNADAPTER_EXPORT hal::Operand* AddQuant8ConstantOperand(
    hal::Model* model,
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

NNADAPTER_EXPORT hal::Operand* AddQuant8ConstantOperand(
    hal::Model* model,
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

NNADAPTER_EXPORT hal::Operand* AddQuant8ConstantOperand(
    hal::Model* model,
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

NNADAPTER_EXPORT hal::Operand* AddQuant32ConstantOperand(
    hal::Model* model,
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

NNADAPTER_EXPORT hal::Operand* AddQuant32ConstantOperand(
    hal::Model* model,
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

NNADAPTER_EXPORT hal::Operand* AddQuant32ConstantOperand(
    hal::Model* model,
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

NNADAPTER_EXPORT hal::Operand* AddQuant8VariableOperand(
    hal::Model* model,
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

NNADAPTER_EXPORT hal::Operand* AddQuant8VariableOperand(
    hal::Model* model,
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

NNADAPTER_EXPORT hal::Operand* AddFloat32VariableOperand(
    hal::Model* model, const std::vector<int32_t>& dimensions) {
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

NNADAPTER_EXPORT void ReshapeOperand(hal::Operand* operand,
                                     std::vector<int32_t> dimensions) {
  ReshapeDimensions(operand->type.dimensions.data,
                    &operand->type.dimensions.count,
                    dimensions);
}

NNADAPTER_EXPORT void TransposeOperand(hal::Operand* operand,
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

NNADAPTER_EXPORT bool UpdateOperationInputOperands(
    std::vector<hal::Operation*> operations,
    hal::Operand* old_operand,
    hal::Operand* new_operand) {
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

NNADAPTER_EXPORT bool UpdateOperationOutputOperands(hal::Operation* operation,
                                                    hal::Operand* old_operand,
                                                    hal::Operand* new_operand) {
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

NNADAPTER_EXPORT bool UpdateModelInputOperands(hal::Model* model,
                                               hal::Operand* old_operand,
                                               hal::Operand* new_operand) {
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

NNADAPTER_EXPORT bool UpdateModelOutputOperands(hal::Model* model,
                                                hal::Operand* old_operand,
                                                hal::Operand* new_operand) {
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

NNADAPTER_EXPORT bool IsConstantOperand(hal::Operand* operand) {
  return operand->type.lifetime == NNADAPTER_CONSTANT_COPY ||
         operand->type.lifetime == NNADAPTER_CONSTANT_REFERENCE;
}

NNADAPTER_EXPORT bool IsTemporaryShapeOperand(hal::Operand* operand) {
  return operand->type.lifetime == NNADAPTER_TEMPORARY_SHAPE;
}

NNADAPTER_EXPORT bool IsModelInputOperand(hal::Operand* operand) {
  return operand->type.lifetime == NNADAPTER_MODEL_INPUT;
}

NNADAPTER_EXPORT bool IsModelOutputOperand(hal::Operand* operand) {
  return operand->type.lifetime == NNADAPTER_MODEL_OUTPUT;
}

NNADAPTER_EXPORT bool IsOperandWithDynamicShape(hal::Operand* operand) {
  for (size_t i = 0; i < operand->type.dimensions.count; i++) {
    if (operand->type.dimensions.data[i] == NNADAPTER_UNKNOWN) {
      return true;
    }
  }
  return false;
}

NNADAPTER_EXPORT bool IsOperationWithAllInputConstantOperands(
    hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  for (auto input_operand : input_operands) {
    if (!IsConstantOperand(input_operand)) {
      return false;
    }
  }
  return true;
}

std::vector<hal::Operation*> GetOperandConsumers(hal::Model* model,
                                                 hal::Operand* operand) {
  std::vector<hal::Operation*> consumers;
  for (auto& operation : model->operations) {
    auto& input_operands = operation.input_operands;
    if (std::find(input_operands.begin(), input_operands.end(), operand) ==
        input_operands.end())
      continue;
    consumers.push_back(&operation);
  }
  return consumers;
}

NNADAPTER_EXPORT hal::Operation* GetOperandProducer(hal::Model* model,
                                                    hal::Operand* operand) {
  hal::Operation* producer = nullptr;
  for (auto& operation : model->operations) {
    auto& output_operands = operation.output_operands;
    if (std::find(output_operands.begin(), output_operands.end(), operand) ==
        output_operands.end())
      continue;
    // a operand has only one producer
    NNADAPTER_CHECK(producer == nullptr);
    producer = &operation;
  }
  return producer;
}

NNADAPTER_EXPORT int GetModelInputOperandIndex(hal::Model* model,
                                               hal::Operand* operand) {
  if (IsModelInputOperand(operand)) {
    for (size_t i = 0; i < model->input_operands.size(); i++) {
      if (model->input_operands[i] == operand) {
        return i;
      }
    }
  }
  return -1;
}

NNADAPTER_EXPORT int GetModelOutputOperandIndex(hal::Model* model,
                                                hal::Operand* operand) {
  if (IsModelOutputOperand(operand)) {
    for (size_t i = 0; i < model->output_operands.size(); i++) {
      if (model->output_operands[i] == operand) {
        return i;
      }
    }
  }
  return -1;
}

hal::Operand* AddTransposeOperation(hal::Model* model,
                                    hal::Operand* reference_operand,
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

NNADAPTER_EXPORT hal::Operand* AppendTransposeOperation(
    hal::Model* model,
    hal::Operand* input_operand,
    std::vector<int32_t> permutation) {
  return AddTransposeOperation(model, input_operand, permutation, true);
}

NNADAPTER_EXPORT hal::Operand* InsertTransposeOperation(
    hal::Model* model,
    hal::Operand* output_operand,
    std::vector<int32_t> permutation) {
  return AddTransposeOperation(model, output_operand, permutation, false);
}

NNADAPTER_EXPORT hal::Operand* AppendReshapeOperation(
    hal::Model* model,
    hal::Operand* input_operand,
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

NNADAPTER_EXPORT hal::Operand* InsertReshapeOperation(
    hal::Model* model,
    hal::Operand* output_operand,
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
  return output_operand;
}

hal::Operand* AddDummyOperation(hal::Model* model,
                                hal::Operand* reference_operand,
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

NNADAPTER_EXPORT hal::Operand* AppendDummyOperation(
    hal::Model* model, hal::Operand* input_operand) {
  return AddDummyOperation(model, input_operand, true);
}

NNADAPTER_EXPORT hal::Operand* InsertDummyOperation(
    hal::Model* model, hal::Operand* output_operand) {
  return AddDummyOperation(model, output_operand, false);
}

hal::Operand* AddUnaryOperation(hal::Model* model,
                                hal::Operand* reference_operand,
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

NNADAPTER_EXPORT hal::Operand* AppendUnaryOperation(
    hal::Model* model,
    hal::Operand* input_operand,
    NNAdapterOperationType operation_type) {
  return AddUnaryOperation(model, input_operand, operation_type, true);
}

NNADAPTER_EXPORT hal::Operand* InsertUnaryOperation(
    hal::Model* model,
    hal::Operand* output_operand,
    NNAdapterOperationType operation_type) {
  return AddUnaryOperation(model, output_operand, operation_type, false);
}

NNADAPTER_EXPORT hal::Operand* AddRequantOperation(
    hal::Model* model,
    hal::Operation* operation,
    hal::Operand* target_operand,
    hal::Operand* reference_operand) {
  // Insert a new operand before output_operand
  auto immediate_operand = AddOperand(model);
  // Make the quantization parameters and precision of the immediate operand the
  // same as output's
  CopyOperandType(&immediate_operand->type, reference_operand->type);
  if (!IsTemporaryShapeOperand(reference_operand)) {
    immediate_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  }
  // Update to the dimensions of input operand, and update the input operand of
  // the operation with immediate_operand
  immediate_operand->type.dimensions = target_operand->type.dimensions;
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
  // Insert a dummy ADD operation before or after target_operand
  auto dummy_add_operation = AddOperation(model);
  dummy_add_operation->type = NNADAPTER_ADD;
  // After target_operand
  for (auto& operand : operation->input_operands) {
    if (operand == target_operand) {
      dummy_add_operation->input_operands = {
          target_operand, zero_operand, fuse_code_operand};
      dummy_add_operation->output_operands = {immediate_operand};
      operand = immediate_operand;
      break;
    }
  }
  // Before target_operand
  for (auto& operand : operation->output_operands) {
    if (operand == target_operand) {
      dummy_add_operation->input_operands = {
          immediate_operand, zero_operand, fuse_code_operand};
      dummy_add_operation->output_operands = {target_operand};
      operand = immediate_operand;
      break;
    }
  }
  return immediate_operand;
}

NNADAPTER_EXPORT std::vector<hal::Operation*> SortOperationsInTopologicalOrder(
    hal::Model* model) {
  NNADAPTER_VLOG(5) << "model total operands: " << model->operands.size();
  NNADAPTER_VLOG(5) << "model input operands: " << model->input_operands.size();
  NNADAPTER_VLOG(5) << "model output operands: "
                    << model->output_operands.size();
  NNADAPTER_VLOG(5) << "model total operations: " << model->operations.size();
  std::vector<hal::Operation*> operations;  // Operations in topological order
  std::vector<hal::Operation*> queue;
  // Use to find all of adjacent operations according to a given operand.
  std::multimap<hal::Operand*, hal::Operation*> map;
  // The counters of variable inputs for all of operations.
  std::map<hal::Operation*, uint32_t> counts;
  for (auto& operation : model->operations) {
    uint32_t count = 0;
    for (auto operand : operation.input_operands) {
      NNAdapterOperandLifetimeCode lifetime{NNADAPTER_CONSTANT_COPY};
      if (operand != nullptr) {
        lifetime = operand->type.lifetime;
      }
      if (lifetime == NNADAPTER_TEMPORARY_VARIABLE ||
          lifetime == NNADAPTER_TEMPORARY_SHAPE ||
          lifetime == NNADAPTER_MODEL_OUTPUT) {
        count++;
        map.insert(
            std::pair<hal::Operand*, hal::Operation*>(operand, &operation));
      }
    }
    if (count == 0) {
      // The operation which only depends the model inputs and constants
      queue.push_back(&operation);
    }
    counts[&operation] = count;
  }
  while (queue.size() > 0) {
    auto operation = queue.back();
    queue.pop_back();
    operations.push_back(operation);
    for (auto operand : operation->output_operands) {
      auto range = map.equal_range(operand);
      for (auto i = range.first; i != range.second; i++) {
        uint32_t& count = counts[i->second];
        if (--count == 0) {
          queue.push_back(i->second);
        }
      }
    }
  }
  return operations;
}

}  // namespace nnadapter
