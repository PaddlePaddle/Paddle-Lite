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
// Remove a operand into a model
void RemoveOperand(hal::Model* model, hal::Operand* operand);
// Remove a operation into a model
void RemoveOperation(hal::Model* model, hal::Operation* operation);
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
// Update the input/output operands of the operations which equal to the
// old_operand with the new_operand
// For example:
//   op0 -> var0 -> op1
//            |---> op2
// case0: UpdateOperationInputOperands, operations={op1}, old_operand=var0,
// new_operand=new_var
//   op0 -> var0   new_var -> op1
//            |---> op2
// case1: UpdateOperationOutputOperands, operation=op0, old_operand=var0,
// new_operand=new_var
//   op0 -> new_var   var0 -> op1
//                      |---> op2
bool UpdateOperationInputOperands(std::vector<hal::Operation*> operations,
                                  hal::Operand* old_operand,
                                  hal::Operand* new_operand);
bool UpdateOperationOutputOperands(hal::Operation* operation,
                                   hal::Operand* old_operand,
                                   hal::Operand* new_operand);
bool UpdateModelInputOperands(hal::Model* model,
                              hal::Operand* old_operand,
                              hal::Operand* new_operand);
bool UpdateModelOutputOperands(hal::Model* model,
                               hal::Operand* old_operand,
                               hal::Operand* new_operand);
// Check if it is a constant operand
bool IsConstantOperand(hal::Operand* operand);
// Check if it is a temporary shape operand
bool IsTemporaryShapeOperand(hal::Operand* operand);
bool IsModelInputOperand(hal::Operand* operand);
bool IsModelOutputOperand(hal::Operand* operand);
bool IsOperandWithDynamicShape(hal::Operand* operand);
bool IsOperationWithAllInputConstantOperands(hal::Operation* operation);
// Find the operations that consumes the operand
std::vector<hal::Operation*> GetOperandConsumers(hal::Model* model,
                                                 hal::Operand* operand);
// Find the operation that produced the operand
hal::Operation* GetOperandProducer(hal::Model* model, hal::Operand* operand);

// Get the index of model input and output operands
int GetModelInputOperandIndex(hal::Model* model, hal::Operand* operand);
int GetModelOutputOperandIndex(hal::Model* model, hal::Operand* operand);

// Append or insert a transpose operation
hal::Operand* AppendTransposeOperation(hal::Model* model,
                                       hal::Operand* input_operand,
                                       std::vector<int32_t> permutation);
hal::Operand* InsertTransposeOperation(hal::Model* model,
                                       hal::Operand* output_operand,
                                       std::vector<int32_t> permutation);
// Append or insert a reshape operation
hal::Operand* AppendReshapeOperation(hal::Model* model,
                                     hal::Operand* input_operand,
                                     std::vector<int32_t> shape);
hal::Operand* InsertReshapeOperation(
    hal::Model* model,
    hal::Operand* output_operand,
    const NNAdapterOperandDimensionType& input_dimensions,
    std::vector<int32_t> shape = {});
// Append or insert a dummy add operation, set the addend to a zero operand
hal::Operand* AppendDummyOperation(hal::Model* model,
                                   hal::Operand* input_operand);
hal::Operand* InsertDummyOperation(hal::Model* model,
                                   hal::Operand* output_operand);
// Append or insert a unary activiation or other operation which has only one
// input and output operand
hal::Operand* AppendUnaryOperation(hal::Model* model,
                                   hal::Operand* input_operand,
                                   NNAdapterOperationType operation_type);
hal::Operand* InsertUnaryOperation(hal::Model* model,
                                   hal::Operand* output_operand,
                                   NNAdapterOperationType operation_type);
// Add a dummy ADD to simulate the REQUANT operation
// i.e.
// target_operand(target_quant_params)->CONCAT->reference_operand(reference_quant_params),
// After applying this,
// target_operand(target_quant_params)->ADD->immediate_operand(reference_quant_params)->CONCAT->reference_operand(reference_quant_params)
// i.e.
// reference_operand(reference_quant_params)->SPLIT->target_operand(target_quant_params),
// After applying this,
// reference_operand(reference_quant_params)->SPLIT->immediate_operand(reference_quant_params)->ADD->target_operand(target_quant_params)
hal::Operand* AddRequantOperation(hal::Model* model,
                                  hal::Operation* operation,
                                  hal::Operand* target_operand,
                                  hal::Operand* reference_operand);

// Sort the operations of the specified model in topological order
std::vector<hal::Operation*> SortOperationsInTopologicalOrder(
    hal::Model* model);

}  // namespace nnadapter
