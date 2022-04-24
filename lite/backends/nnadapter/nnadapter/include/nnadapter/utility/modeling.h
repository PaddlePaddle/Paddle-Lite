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
#include "core/types.h"

namespace nnadapter {

// Clear the operands and operations of a model
void ClearModel(core::Model* model);
// Clear the type and the buffer of a operand
void ClearOperand(core::Operand* operand);
// Append a operand into a model
core::Operand* AddOperand(core::Model* model);
// Append a operation into a model
core::Operation* AddOperation(core::Model* model);
// Remove a operand into a model
void RemoveOperand(core::Model* model, core::Operand* operand);
// Remove a operation into a model
void RemoveOperation(core::Model* model, core::Operation* operation);
// Allocate memory based on the data type and dimensions for a operand, and do
// not reallocate if memory is sufficient
void* AllocateOperand(core::Operand* operand);
// Add a bool8 scalar constant operand
core::Operand* AddBool8ConstantOperand(core::Model* model, bool value);
// Add a int32 scalar constant operand
core::Operand* AddInt32ConstantOperand(core::Model* model, int32_t value);
// Add a float32 scalar constant operand
core::Operand* AddFloat32ConstantOperand(core::Model* model, float value);
// Add a int32 vector constant operand
core::Operand* AddInt32ConstantOperand(core::Model* model,
                                       std::vector<int32_t> values);
// Add a float32 vector constant operand
core::Operand* AddFloat32ConstantOperand(core::Model* model,
                                         std::vector<float> values);
// Add a int32 constant operand
core::Operand* AddInt32ConstantOperand(core::Model* model,
                                       int32_t* values,
                                       const std::vector<int32_t>& dimensions,
                                       bool copy = true);
// Add a float32 constant operand
core::Operand* AddFloat32ConstantOperand(core::Model* model,
                                         float* values,
                                         const std::vector<int32_t>& dimensions,
                                         bool copy = true);
// Add a quant8 constant operand with symmetric per-layer quantizion
core::Operand* AddQuant8ConstantOperand(core::Model* model,
                                        int8_t* values,
                                        const std::vector<int32_t>& dimensions,
                                        float quant_scale,
                                        bool copy = true);
// Add a quant8 constant operand with symmetric per-channel quantizion
core::Operand* AddQuant8ConstantOperand(core::Model* model,
                                        int8_t* values,
                                        const std::vector<int32_t>& dimensions,
                                        float* quant_scales,
                                        uint32_t quant_scale_count,
                                        uint32_t quant_channel_dim = 0,
                                        bool copy = true);
// Add a quant8 constant operand with asymmetric per-layer quantizion
core::Operand* AddQuant8ConstantOperand(core::Model* model,
                                        uint8_t* values,
                                        const std::vector<int32_t>& dimensions,
                                        float quant_scale,
                                        int32_t zero_point,
                                        bool copy = true);
// Add a quant32 constant operand with symmetric per-layer quantizion
core::Operand* AddQuant32ConstantOperand(core::Model* model,
                                         int32_t* values,
                                         const std::vector<int32_t>& dimensions,
                                         float quant_scale,
                                         bool copy = true);
// Add a quant32 constant operand with symmetric per-channel quantizion
core::Operand* AddQuant32ConstantOperand(core::Model* model,
                                         int32_t* values,
                                         const std::vector<int32_t>& dimensions,
                                         float* quant_scales,
                                         uint32_t quant_scale_count,
                                         uint32_t quant_channel_dim = 0,
                                         bool copy = true);
// Add a quant32 constant operand with asymmetric per-layer quantizion
core::Operand* AddQuant32ConstantOperand(core::Model* model,
                                         uint32_t* values,
                                         const std::vector<int32_t>& dimensions,
                                         float quant_scale,
                                         int32_t zero_point,
                                         bool copy = true);
// Add a quant8 variable operand with symmetric per-layer quantizion
core::Operand* AddQuant8VariableOperand(core::Model* model,
                                        const std::vector<int32_t>& dimensions,
                                        float quant_scale);
// Add a quant8 variable operand with asymmetric per-layer quantizion
core::Operand* AddQuant8VariableOperand(core::Model* model,
                                        const std::vector<int32_t>& dimensions,
                                        float quant_scale,
                                        int32_t zero_point);
// Add a float32 variable operand
core::Operand* AddFloat32VariableOperand(
    core::Model* model, const std::vector<int32_t>& dimensions);

// Transpose the data of a constant operand(with NNADAPTER_CONSTANT_COPY or
// NNADAPTER_CONSTANT_REFERENCE lifetime) or only dimensions of non-constant
// operand(with NNADAPTER_TEMPORARY_VARIABLE, NNADAPTER_MODEL_INPUT or
// NNADAPTER_MODEL_OUTPUT lifetime)
void TransposeOperand(core::Operand* operand, std::vector<int32_t> permutation);
// Reshapes the dimensions of a operand, similar to numpy.reshape
void ReshapeOperand(core::Operand* operand, std::vector<int32_t> dimensions);
// Copy the operand and its buffer
void CopyOperand(core::Operand* dst_operand,
                 core::Operand* src_operand,
                 bool copy = true);
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
bool UpdateOperationInputOperands(std::vector<core::Operation*> operations,
                                  core::Operand* old_operand,
                                  core::Operand* new_operand);
bool UpdateOperationOutputOperands(core::Operation* operation,
                                   core::Operand* old_operand,
                                   core::Operand* new_operand);
bool UpdateModelInputOperands(core::Model* model,
                              core::Operand* old_operand,
                              core::Operand* new_operand);
bool UpdateModelOutputOperands(core::Model* model,
                               core::Operand* old_operand,
                               core::Operand* new_operand);
// Check if it is a constant operand
bool IsConstantOperand(core::Operand* operand);
bool IsConstantCopyOperand(core::Operand* operand);
bool IsConstantReferenceOperand(core::Operand* operand);
// Check if it is a temporary variable operand
bool IsTemporaryVariableOperand(core::Operand* operand);
// Check if it is a temporary shape operand
bool IsTemporaryShapeOperand(core::Operand* operand);
bool IsModelInputOperand(core::Operand* operand);
bool IsModelOutputOperand(core::Operand* operand);
bool IsOperandWithDynamicShape(core::Operand* operand);
bool IsOperationWithAllInputConstantOperands(core::Operation* operation);
// Find the operations that consumes the operand
std::vector<core::Operation*> GetOperandConsumers(core::Model* model,
                                                  core::Operand* operand);
// Find the operation that produced the operand
core::Operation* GetOperandProducer(core::Model* model, core::Operand* operand);

// Get the index of model input and output operands
int GetModelInputOperandIndex(core::Model* model, core::Operand* operand);
int GetModelOutputOperandIndex(core::Model* model, core::Operand* operand);

// Append or insert a transpose operation
core::Operand* AppendTransposeOperation(core::Model* model,
                                        core::Operand* input_operand,
                                        std::vector<int32_t> permutation);
core::Operand* InsertTransposeOperation(core::Model* model,
                                        core::Operand* output_operand,
                                        std::vector<int32_t> permutation);
// Append or insert a reshape operation
core::Operand* AppendReshapeOperation(core::Model* model,
                                      core::Operand* input_operand,
                                      std::vector<int32_t> shape);
core::Operand* InsertReshapeOperation(
    core::Model* model,
    core::Operand* output_operand,
    const NNAdapterOperandDimensionType& input_dimensions,
    std::vector<int32_t> shape = {});
// Append or insert a dummy add operation, set the addend to a zero operand
core::Operand* AppendDummyOperation(core::Model* model,
                                    core::Operand* input_operand);
core::Operand* InsertDummyOperation(core::Model* model,
                                    core::Operand* output_operand);
// Append or insert a unary activiation or other operation which has only one
// input and output operand
core::Operand* AppendUnaryOperation(core::Model* model,
                                    core::Operand* input_operand,
                                    NNAdapterOperationType operation_type);
core::Operand* InsertUnaryOperation(core::Model* model,
                                    core::Operand* output_operand,
                                    NNAdapterOperationType operation_type);
// Append or insert a dummy ADD to simulate the REQUANT operation
// input_operand(input_quant_params)->ADD->output_operand(output_quant_params)
core::Operand* AppendRequantOperation(core::Model* model,
                                      core::Operand* input_operand,
                                      void* output_quant_params);
core::Operand* InsertRequantOperation(core::Model* model,
                                      core::Operand* output_operand,
                                      void* input_quant_params);

// Sort the operations of the specified model in topological order
std::vector<const core::Operation*> SortOperationsInTopologicalOrder(
    const core::Model* model);
std::vector<core::Operation*> SortOperationsInTopologicalOrder(
    core::Model* model);

// Serialize/deserialize core::Model into/from the binary buffer
bool SerializeModel(core::Model* model, std::vector<uint8_t>* buffer);
bool DeserializeModel(void* buffer, uint64_t size, core::Model** model);

}  // namespace nnadapter
