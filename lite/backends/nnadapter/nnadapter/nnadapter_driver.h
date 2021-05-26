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

#pragma once

#include <cstring>
#include <list>
#include <string>
#include <vector>
#include "nnadapter_logging.h"  // NOLINT
#include "nnadapter_micros.h"   // NOLINT
#include "nnadapter_types.h"    // NOLINT

namespace nnadapter {
namespace driver {

typedef struct Operand {
  NNAdapterOperandType type;
  void* buffer;
  uint32_t length;
} Operand;

typedef struct Argument {
  int index;
  uint32_t dimension_count;
  int32_t dimensions[NNADAPTER_MAX_SIZE_OF_DIMENSIONS];
  void* buffer;
  uint32_t length;
} Argument;

typedef struct Operation {
  NNAdapterOperationType type;
  std::vector<Operand*> input_operands;
  std::vector<Operand*> output_operands;
} Operation;

typedef struct Cache {
  std::string cache_key;
  void* cache_buffer;
  uint32_t cache_length;
  std::string cache_dir;
  std::vector<NNAdapterOperandType> input_types;
  std::vector<NNAdapterOperandType> output_types;
} Cache;

typedef struct Model {
  std::list<Operand> operands;
  std::list<Operation> operations;
  std::vector<Operand*> input_operands;
  std::vector<Operand*> output_operands;
} Model;

typedef struct Device {
  const char* name;
  const char* vendor;
  NNAdapterDeviceType type;
  int32_t version;
  int (*create_context)(void** context);
  void (*destroy_context)(void* context);
  int (*create_program)(void* context,
                        Model* model,
                        Cache* cache,
                        void** program);
  void (*destroy_program)(void* program);
  int (*execute_program)(void* program,
                         uint32_t input_count,
                         Argument* input_arguments,
                         uint32_t output_count,
                         Argument* output_arguments);
} Device;

// Utilities
int32_t ProductionOfDimensions(int32_t* input_dimensions,
                               uint32_t input_dimensions_count);
void TransposeDimensions(int32_t* input_dimensions,
                         const std::vector<int32_t>& permutation,
                         int32_t* output_dimensions_ptr = nullptr);
void TransposeOperand(Operand* operand, std::vector<int32_t> permutation);
// A naive implementation of transpose operation
template <typename T>
void TransposeData(const T* input,
                   T* output,
                   const std::vector<int32_t>& permutation,
                   const int32_t* input_dimensions,
                   int32_t* output_dimensions_ptr = nullptr) {
  auto permutation_count = permutation.size();
  NNADAPTER_CHECK_GE(permutation_count, 2);
  std::vector<int32_t> output_dimensions(permutation_count);
  for (size_t i = 0; i < permutation_count; i++) {
    output_dimensions[i] = input_dimensions[i];
  }
  for (size_t i = 0; i < permutation_count; i++) {
    output_dimensions[i] = input_dimensions[permutation[i]];
  }
  std::vector<int64_t> input_strides(permutation_count, 1);
  std::vector<int64_t> output_strides(permutation_count, 1);
  for (int i = permutation_count - 2; i >= 0; i--) {
    input_strides[i] = input_strides[i + 1] * input_dimensions[i + 1];
    output_strides[i] = output_strides[i + 1] * output_dimensions[i + 1];
  }
  auto element_count = input_strides[0] * input_dimensions[0];
  for (int64_t i = 0; i < element_count; i++) {
    // Calculate the indexes for input
    int64_t input_offset = i;
    std::vector<int64_t> input_index(permutation_count, 0);
    for (size_t j = 0; j < permutation_count; j++) {
      input_index[j] = input_offset / input_strides[j];
      input_offset %= input_strides[j];
    }
    // Calculate the transposed indexes for output
    std::vector<int64_t> output_index(permutation_count, 0);
    for (size_t j = 0; j < permutation_count; j++) {
      output_index[j] = input_index[permutation[j]];
    }
    // Calculate the element offset for output
    int64_t output_offset = 0;
    for (size_t j = 0; j < permutation_count; j++) {
      output_offset += output_strides[j] * output_index[j];
    }
    output[output_offset] = input[i];
  }
  if (output_dimensions_ptr) {
    for (size_t i = 0; i < permutation_count; i++) {
      output_dimensions_ptr[i] = output_dimensions[i];
    }
  }
}
std::string OperandTypeToString(NNAdapterOperandType* type);
std::string OperandValueToString(Operand* operand);
// Visualize model to dot file for debugging
std::string Visualize(Model* model);
// Sort the operations of the model in toplogical order
std::vector<Operation*> SortOperationsInTopologicalOrder(Model* model);
// Helper functions for mutating the model
Operand* AddOperand(Model* model);
template <typename T>
Operand* AddScalarConstantOperand(
    Model* model, T value, NNAdapterOperandPrecisionCode precision_code) {
  auto operand = AddOperand(model);
  memset(&operand->type, 0, sizeof(NNAdapterOperandType));
  operand->type.precision = precision_code;
  operand->type.dimension_count = 0;
  operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
  operand->length = sizeof(T);
  operand->buffer = malloc(operand->length);
  NNADAPTER_CHECK(operand->buffer != nullptr)
      << "Failed to allocate " << operand->length
      << " bytes for the buffer of an operand, out of memory!";
  memcpy(operand->buffer, &value, operand->length);
  return operand;
}
template <typename T>
Operand* AddVectorConstantOperand(
    Model* model,
    const T* values,
    uint32_t num_values,
    NNAdapterOperandPrecisionCode precision_code) {
  auto operand = AddOperand(model);
  memset(&operand->type, 0, sizeof(NNAdapterOperandType));
  operand->type.precision = precision_code;
  operand->type.dimension_count = 1;
  operand->type.dimensions[0] = num_values;
  operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
  operand->length = sizeof(T) * num_values;
  operand->buffer = malloc(operand->length);
  NNADAPTER_CHECK(operand->buffer != nullptr)
      << "Failed to allocate " << operand->length
      << " bytes for the buffer of an operand, out of memory!";
  memcpy(operand->buffer, values, operand->length);
  return operand;
}
Operand* AddScalarInt32ConstantOperand(Model* model, int32_t value);
Operand* AddScalarFloat32ConstantOperand(Model* model, float value);
Operand* AddVectorInt32ConstantOperand(Model* model,
                                       const int32_t* values,
                                       uint32_t num_values);
Operand* AddVectorFloat32ConstantOperand(Model* model,
                                         const float* values,
                                         uint32_t num_values);
Operand* AddVectorInt32ConstantOperand(Model* model,
                                       const std::vector<int32_t>& values);
Operand* AddVectorFloat32ConstantOperand(Model* model,
                                         const std::vector<float>& values);
Operation* AddOperation(Model* model);

}  // namespace driver
}  // namespace nnadapter
