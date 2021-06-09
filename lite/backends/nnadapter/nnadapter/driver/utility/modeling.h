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
#include "driver/driver.h"

namespace nnadapter {
namespace driver {

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

// Sort the operations of the specified model in topological order
std::vector<Operation*> SortOperationsInTopologicalOrder(Model* model);

}  // namespace driver
}  // namespace nnadapter
