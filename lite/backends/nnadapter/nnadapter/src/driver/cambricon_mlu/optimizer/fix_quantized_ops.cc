// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "driver/cambricon_mlu/optimizer/fix_quantized_ops.h"
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace cambricon_mlu {

static bool NeedInsertQuant(core::Model* model, core::Operand* operand) {
  auto pre_operation = GetOperandProducer(model, operand);
  if (pre_operation == nullptr || pre_operation->type == NNADAPTER_QUANTIZE) {
    return false;
  }
  if (pre_operation->type == NNADAPTER_RESHAPE) {
    auto prepre_operation =
        GetOperandProducer(model, pre_operation->input_operands[0]);
    if (prepre_operation != nullptr &&
        prepre_operation->type == NNADAPTER_QUANTIZE) {
      return false;
    }
  }
  return true;
}

static bool NeedDeleteDequant(core::Model* model, core::Operand* operand) {
  auto next_operations = GetOperandConsumers(model, operand);
  return !next_operations.empty() &&
         next_operations[0]->type == NNADAPTER_DEQUANTIZE;
}

// Add a quant operation after input_operand, before reference_operations
static core::Operand* AddQuantOperation(
    core::Model* model,
    core::Operand* input_operand,
    const std::vector<core::Operation*>& reference_operations = {}) {
  // Insert a new operand after input_operand
  auto output_operand = AddOperand(model);
  CopyOperandType(&output_operand->type, input_operand->type);
  if (!IsTemporaryShapeOperand(input_operand)) {
    output_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  }
  UpdateOperationInputOperands(
      reference_operations, input_operand, output_operand);
  output_operand->type.precision = NNADAPTER_QUANT_INT8_SYMM_PER_LAYER;
  // Insert a new quant operation between input_operand and output_operand
  auto quant_operation = AddOperation(model);
  quant_operation->type = NNADAPTER_QUANTIZE;
  core::Operand* axis_operand = AddInt32ConstantOperand(model, 1);
  float scale = input_operand->type.symm_per_layer_params.scale;
  core::Operand* scale_operand = AddFloat32ConstantOperand(model, scale);
  core::Operand* zero_point_operand = AddInt32ConstantOperand(model, 0);
  quant_operation->input_operands = {
      input_operand, axis_operand, scale_operand, zero_point_operand};
  quant_operation->output_operands = {output_operand};
  return output_operand;
}

// Del a dequant operation after output_operand
static core::Operand* DelDequantOperation(core::Model* model,
                                          core::Operand* output_operand) {
  auto next_operations = GetOperandConsumers(model, output_operand);
  auto dequant_operation = next_operations[0];
  auto dequant_output_operand = dequant_operation->output_operands[0];
  auto dequant_post_operations =
      GetOperandConsumers(model, dequant_output_operand);
  for (auto operation : dequant_post_operations) {
    for (int i = 0; i < operation->input_operands.size(); i++) {
      if (dequant_output_operand == operation->input_operands[i]) {
        operation->input_operands[i] = output_operand;
      }
    }
  }
  RemoveOperand(model, dequant_output_operand);
  RemoveOperation(model, dequant_operation);
  return output_operand;
}

// Return quant operations after/before reference_operand
// For example:
// If graph is:
//
//     conv0_int8 -> var -> conv3_int8
//     conv1_int8 --|   |--> conv4_int8
//  conv2_float32 --|   |--> conv5_float32
//
// return {conv0_int8, conv1_int8}, if after is true.
// return {conv3_int8, conv4_int8}, if after is false.
static std::vector<core::Operation*> GetQuantOpsAroundOperand(
    const std::vector<core::Operation*>& operations,
    core::Operand* reference_operand,
    bool after,
    const std::vector<NNAdapterOperationCode>& valid_quant_ops_type) {
  std::vector<core::Operation*> target_operations;
  for (auto operation : operations) {
    if (std::find(valid_quant_ops_type.begin(),
                  valid_quant_ops_type.end(),
                  operation->type) == valid_quant_ops_type.end() ||
        operation->output_operands[0]->type.precision !=
            NNADAPTER_QUANT_INT8_SYMM_PER_LAYER) {
      continue;
    }
    std::vector<nnadapter::core::Operand*> operands;
    if (after) {
      operands = operation->input_operands;
    } else {
      operands = operation->output_operands;
    }
    if (std::find(operands.begin(), operands.end(), reference_operand) !=
        operands.end()) {
      target_operations.push_back(operation);
    }
  }
  return target_operations;
}

static void InsertQuant(core::Model* model) {
  std::vector<NNAdapterOperationCode> valid_quant_ops_type{
      NNADAPTER_CONV_2D, NNADAPTER_FULLY_CONNECTED};
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    auto input_operands = operation->input_operands;
    if (std::find(valid_quant_ops_type.begin(),
                  valid_quant_ops_type.end(),
                  operation->type) == valid_quant_ops_type.end() ||
        input_operands[0]->type.precision == NNADAPTER_FLOAT32) {
      continue;
    }
    if (NeedInsertQuant(model, input_operands[0])) {
      // Brother int8 ops share the same input.
      auto reference_operations = GetQuantOpsAroundOperand(
          operations, input_operands[0], true, valid_quant_ops_type);
      AddQuantOperation(model, input_operands[0], reference_operations);
    }
  }
}

static void DeleteDequant(core::Model* model) {
  std::vector<NNAdapterOperationCode> valid_quant_ops_type{
      NNADAPTER_CONV_2D, NNADAPTER_FULLY_CONNECTED};
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    auto input_operands = operation->input_operands;
    if (std::find(valid_quant_ops_type.begin(),
                  valid_quant_ops_type.end(),
                  operation->type) == valid_quant_ops_type.end() ||
        input_operands[0]->type.precision == NNADAPTER_FLOAT32) {
      continue;
    }
    auto output_operand = operation->output_operands[0];
    if (NeedDeleteDequant(model, output_operand)) {
      DelDequantOperation(model, output_operand);
    }
  }
}

// MLU int8 conv or fc only support float32 out.
static void ChangeQuantizedOpOutPrecision(core::Model* model) {
  std::vector<NNAdapterOperationCode> valid_quant_ops_type{
      NNADAPTER_CONV_2D, NNADAPTER_FULLY_CONNECTED};
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    if (std::find(valid_quant_ops_type.begin(),
                  valid_quant_ops_type.end(),
                  operation->type) != valid_quant_ops_type.end()) {
      auto output_operand = operation->output_operands[0];
      output_operand->type.precision = NNADAPTER_FLOAT32;
    }
  }
}

void FixQuantizedOps(core::Model* model) {
  InsertQuant(model);
  DeleteDequant(model);
  ChangeQuantizedOpOutPrecision(model);
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
