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

#include <algorithm>
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

// Utilities for model debugging
std::string OperandToString(Operand* operand);
std::string OperandTypeToString(NNAdapterOperandType* type);
std::string OperandValueToString(Operand* operand);
// Visualize model to dot file
std::string Visualize(Model* model);
// Sort the operations of the model in toplogical order
std::vector<Operation*> SortOperationsInTopologicalOrder(Model* model);

// Utilities for model building
Operand* AddOperand(Model* model);
Operation* AddOperation(Model* model);
Operand* AddBool8ConstantOperand(Model* model, bool value);
Operand* AddInt32ConstantOperand(Model* model, int32_t value);
Operand* AddFloat32ConstantOperand(Model* model, float value);
Operand* AddInt32ConstantOperand(Model* model, std::vector<int32_t> values);
Operand* AddFloat32ConstantOperand(Model* model, std::vector<float> values);
Operand* AddInt32ConstantOperand(Model* model,
                                 int32_t* values,
                                 const std::vector<int32_t>& dimensions,
                                 bool copy = true);
Operand* AddFloat32ConstantOperand(Model* model,
                                   float* values,
                                   const std::vector<int32_t>& dimensions,
                                   bool copy = true);
// Quant8 constant operand with symmetric per-layer quantizion
Operand* AddQuant8ConstantOperand(Model* model,
                                  int8_t* values,
                                  const std::vector<int32_t>& dimensions,
                                  float quant_scale,
                                  bool copy = true);
// Quant8 constant operand with symmetric per-channel quantizion
Operand* AddQuant8ConstantOperand(Model* model,
                                  int8_t* values,
                                  const std::vector<int32_t>& dimensions,
                                  float* quant_scales,
                                  uint32_t quant_scale_count,
                                  uint32_t quant_channel_dim = 0,
                                  bool copy = true);
// Quant8 constant operand with asymmetric per-layer quantizion
Operand* AddQuant8ConstantOperand(Model* model,
                                  uint8_t* values,
                                  const std::vector<int32_t>& dimensions,
                                  float quant_scale,
                                  int32_t zero_point,
                                  bool copy = true);
// Quant32 constant operand with symmetric per-layer quantizion
Operand* AddQuant32ConstantOperand(Model* model,
                                   int32_t* values,
                                   const std::vector<int32_t>& dimensions,
                                   float quant_scale,
                                   bool copy = true);
// Quant32 constant operand with symmetric per-channel quantizion
Operand* AddQuant32ConstantOperand(Model* model,
                                   int32_t* values,
                                   const std::vector<int32_t>& dimensions,
                                   float* quant_scales,
                                   uint32_t quant_scale_count,
                                   uint32_t quant_channel_dim = 0,
                                   bool copy = true);
// Quant32 constant operand with asymmetric per-layer quantizion
Operand* AddQuant32ConstantOperand(Model* model,
                                   uint32_t* values,
                                   const std::vector<int32_t>& dimensions,
                                   float quant_scale,
                                   int32_t zero_point,
                                   bool copy = true);
// Quant8 variable operand with symmetric per-layer quantizion
Operand* AddQuant8VariableOperand(Model* model,
                                  const std::vector<int32_t>& dimensions,
                                  float quant_scale);
// Quant8 variable operand with asymmetric per-layer quantizion
Operand* AddQuant8VariableOperand(Model* model,
                                  const std::vector<int32_t>& dimensions,
                                  float quant_scale,
                                  int32_t zero_point);
Operand* AddFloat32VariableOperand(Model* model,
                                   const std::vector<int32_t>& dimensions);
void TransposeOperand(Operand* operand, std::vector<int32_t> permutation);
void ReshapeOperand(Operand* operand, std::vector<int32_t> dimensions);
bool ReplaceOperand(Model* model,
                    const Operand* pattern,
                    Operand* replace,
                    bool remove = true);
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

template <typename T>
void QuantizeData(const float* input_data,
                  size_t input_data_count,
                  float* input_scale,
                  size_t input_scale_count,
                  T* output_data) {
  bool per_layer = input_scale_count == 1;
  NNADAPTER_CHECK(per_layer || input_data_count == input_scale_count)
      << "Only input_scale_count == 1 and input_scale_count == "
         "input_data_count is supported.";
  int quant_bits = sizeof(T) * 8;
  auto dtype_max = static_cast<int>((1 << (quant_bits - 1)) - 1);
  auto dtype_min = static_cast<int>(0 - dtype_max);
  for (size_t i = 0; i < input_data_count; i++) {
    int scale_index = per_layer ? 0 : i;
    output_data[i] = std::min(
        std::max(static_cast<T>(input_data[i] / input_scale[scale_index]),
                 dtype_min),
        dtype_max);
  }
}

template <typename T>
void DequantizeData(const T* input_data,
                    size_t input_data_count,
                    float* input_scale,
                    size_t input_scale_count,
                    float* output_data) {
  bool per_layer = input_scale_count == 1;
  NNADAPTER_CHECK(per_layer || input_data_count == input_scale_count)
      << "Only input_scale_count == 1 and input_scale_count == "
         "input_data_count is supported.";
  int quant_bits = sizeof(T) * 8;
  auto dtype_max = static_cast<int>((1 << (quant_bits - 1)) - 1);
  auto dtype_min = static_cast<int>(0 - dtype_max);
  for (size_t i = 0; i < input_data_count; i++) {
    int scale_index = per_layer ? 0 : i;
    output_data[i] = std::min(std::max(input_data[i], dtype_min), dtype_max) *
                     input_scale[scale_index];
  }
}

}  // namespace driver
}  // namespace nnadapter
