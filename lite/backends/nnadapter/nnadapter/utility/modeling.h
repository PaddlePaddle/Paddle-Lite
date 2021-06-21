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

#include <vector>
#include "core/hal/types.h"

namespace nnadapter {

// Append a operand into a model
hal::Operand* AddOperand(hal::Model* model);
// Append a operation into a model
hal::Operation* AddOperation(hal::Model* model);
// Add a bool8 scalar constant operand
hal::Operand* AddBool8ConstantOperand(hal::Model* model, bool value);
// Add a int32 scalar constant operand
hal::Operand* AddInt32ConstantOperand(hal::Model* model, int32_t value);
// Add a float32 scalar constant operand
hal::Operand* AddFloat32ConstantOperand(hal::Model* model, float value);
// Add a int32 vector constant operand
hal::Operand* AddInt32ConstantOperand(hal::Model* model,
                                      std::vector<int32_t> values);
// Add a float32 vector constant operand
hal::Operand* AddFloat32ConstantOperand(hal::Model* model,
                                        std::vector<float> values);
// Add a int32 constant operand
hal::Operand* AddInt32ConstantOperand(hal::Model* model,
                                      int32_t* values,
                                      const std::vector<int32_t>& dimensions,
                                      bool copy = true);
// Add a float32 constant operand
hal::Operand* AddFloat32ConstantOperand(hal::Model* model,
                                        float* values,
                                        const std::vector<int32_t>& dimensions,
                                        bool copy = true);
// Add a quant8 constant operand with symmetric per-layer quantizion
hal::Operand* AddQuant8ConstantOperand(hal::Model* model,
                                       int8_t* values,
                                       const std::vector<int32_t>& dimensions,
                                       float quant_scale,
                                       bool copy = true);
// Add a quant8 constant operand with symmetric per-channel quantizion
hal::Operand* AddQuant8ConstantOperand(hal::Model* model,
                                       int8_t* values,
                                       const std::vector<int32_t>& dimensions,
                                       float* quant_scales,
                                       uint32_t quant_scale_count,
                                       uint32_t quant_channel_dim = 0,
                                       bool copy = true);
// Add a quant8 constant operand with asymmetric per-layer quantizion
hal::Operand* AddQuant8ConstantOperand(hal::Model* model,
                                       uint8_t* values,
                                       const std::vector<int32_t>& dimensions,
                                       float quant_scale,
                                       int32_t zero_point,
                                       bool copy = true);
// Add a quant32 constant operand with symmetric per-layer quantizion
hal::Operand* AddQuant32ConstantOperand(hal::Model* model,
                                        int32_t* values,
                                        const std::vector<int32_t>& dimensions,
                                        float quant_scale,
                                        bool copy = true);
// Add a quant32 constant operand with symmetric per-channel quantizion
hal::Operand* AddQuant32ConstantOperand(hal::Model* model,
                                        int32_t* values,
                                        const std::vector<int32_t>& dimensions,
                                        float* quant_scales,
                                        uint32_t quant_scale_count,
                                        uint32_t quant_channel_dim = 0,
                                        bool copy = true);
// Add a quant32 constant operand with asymmetric per-layer quantizion
hal::Operand* AddQuant32ConstantOperand(hal::Model* model,
                                        uint32_t* values,
                                        const std::vector<int32_t>& dimensions,
                                        float quant_scale,
                                        int32_t zero_point,
                                        bool copy = true);
// Add a quant8 variable operand with symmetric per-layer quantizion
hal::Operand* AddQuant8VariableOperand(hal::Model* model,
                                       const std::vector<int32_t>& dimensions,
                                       float quant_scale);
// Add a quant8 variable operand with asymmetric per-layer quantizion
hal::Operand* AddQuant8VariableOperand(hal::Model* model,
                                       const std::vector<int32_t>& dimensions,
                                       float quant_scale,
                                       int32_t zero_point);
// Add a float32 variable operand
hal::Operand* AddFloat32VariableOperand(hal::Model* model,
                                        const std::vector<int32_t>& dimensions);

// Transpose the data of a constant operand(with NNADAPTER_CONSTANT_COPY or
// NNADAPTER_CONSTANT_REFERENCE lifetime) or only dimensions of non-constant
// operand(with NNADAPTER_TEMPORARY_VARIABLE, NNADAPTER_MODEL_INPUT or
// NNADAPTER_MODEL_OUTPUT lifetime)
void TransposeOperand(hal::Operand* operand, std::vector<int32_t> permutation);
// Reshapes the dimensions of a operand, similar to numpy.reshape
void ReshapeOperand(hal::Operand* operand, std::vector<int32_t> dimensions);
// Replace a operand with a new one, similar to numpy.transpose
bool ReplaceOperand(hal::Model* model,
                    hal::Operand* pattern,
                    hal::Operand* replace,
                    bool remove = true);
// Add a transpose operation which set input_operand as its input operand,
// create a output operand with the permutated dimensions, and update all of
// operations
hal::Operand* AddTransposeOperation(hal::Model* model,
                                    hal::Operand* input_operand,
                                    std::vector<int32_t> permutation);

// Sort the operations of the specified model in topological order
std::vector<hal::Operation*> SortOperationsInTopologicalOrder(
    hal::Model* model);

}  // namespace nnadapter
