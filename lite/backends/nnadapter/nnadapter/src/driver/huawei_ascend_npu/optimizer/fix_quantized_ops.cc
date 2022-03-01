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

#include "driver/huawei_ascend_npu/optimizer/fix_quantized_ops.h"
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

static bool NeedPreQuant(core::Model* model, core::Operand* operand) {
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

static bool NeedNextDequant(core::Model* model, core::Operand* operand) {
  auto next_operations = GetOperandConsumers(model, operand);
  return !next_operations.empty() &&
         next_operations[0]->type != NNADAPTER_DEQUANTIZE;
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

// Add a quant operation after input_operand, before reference_operations
static core::Operand* AddQuantOperation(
    core::Model* model,
    core::Operand* input_operand,
    const std::vector<core::Operation*>& reference_operations = {}) {
  NNADAPTER_CHECK_EQ(input_operand->type.precision, NNADAPTER_FLOAT32);
  NNADAPTER_CHECK_GE(input_operand->type.symm_per_layer_params.scale, 0.f);
  // Insert a new operand after input_operand
  auto output_operand = AddOperand(model);
  CopyOperandType(&output_operand->type, input_operand->type);
  if (!IsTemporaryShapeOperand(input_operand)) {
    output_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  }
  UpdateOperationInputOperands(
      reference_operations, input_operand, output_operand);
  UpdateModelOutputOperands(model, input_operand, output_operand);
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

// Add a dequant operation before output_operand, after reference_operations
static core::Operand* AddDequantOperation(
    core::Model* model,
    core::Operand* output_operand,
    const std::vector<core::Operation*>& reference_operations = {}) {
  NNADAPTER_CHECK_EQ(output_operand->type.precision,
                     NNADAPTER_QUANT_INT8_SYMM_PER_LAYER);
  // Insert a new operand before output_operand
  auto input_operand = AddOperand(model);
  CopyOperandType(&input_operand->type, output_operand->type);
  if (!IsTemporaryShapeOperand(output_operand)) {
    input_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  }
  input_operand->type.precision = NNADAPTER_QUANT_INT32_SYMM_PER_LAYER;
  output_operand->type.precision = NNADAPTER_FLOAT32;
  // Insert a new dequant operation between input_operand and output_operand
  auto dequant_operation = AddOperation(model);
  dequant_operation->type = NNADAPTER_DEQUANTIZE;
  dequant_operation->input_operands = {input_operand};
  dequant_operation->output_operands = {output_operand};
  NNADAPTER_CHECK_LE(reference_operations.size(), 1);
  UpdateOperationOutputOperands(
      reference_operations.empty() ? nullptr : reference_operations[0],
      output_operand,
      input_operand);
  return input_operand;
}

float GetDequantScale(core::Model* model, core::Operation* dequant) {
  NNADAPTER_CHECK(dequant);
  NNADAPTER_CHECK_EQ(dequant->type, NNADAPTER_DEQUANTIZE);
  auto pre_operation = GetOperandProducer(model, dequant->input_operands[0]);
  NNADAPTER_CHECK(pre_operation);

  auto pre_op_type = pre_operation->type;
  switch (pre_op_type) {
    case NNADAPTER_CONV_2D:
    case NNADAPTER_FULLY_CONNECTED: {
      auto input_operand = pre_operation->input_operands[0];
      auto weight_operand = pre_operation->input_operands[1];
      NNADAPTER_CHECK_EQ(input_operand->type.precision,
                         NNADAPTER_QUANT_INT8_SYMM_PER_LAYER);
      NNADAPTER_CHECK_EQ(weight_operand->type.precision,
                         NNADAPTER_QUANT_INT8_SYMM_PER_LAYER);
      return input_operand->type.symm_per_layer_params.scale *
             weight_operand->type.symm_per_layer_params.scale;
    } break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported quanted operation: "
                           << OperationTypeToString(pre_op_type);
      break;
  }
  return 0.f;
}

/**
 * before:
 *   op(not_quant) -> in -> conv(quant)
 * after:
 *   op(not_quant) -> in -> quant -> in1 -> conv(quant)
 *
 * before:
 *   conv(quant) -> out -> op(not_dequant)
 * after:
 *   conv(quant) -> out1 -> dequant -> out -> op(not_dequant)
 */
static void InsertExtraQuantDequant(core::Model* model) {
  std::vector<NNAdapterOperationCode> valid_quant_ops_type{
      NNADAPTER_CONV_2D, NNADAPTER_FULLY_CONNECTED};
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    auto output_operand = operation->output_operands[0];
    if (std::find(valid_quant_ops_type.begin(),
                  valid_quant_ops_type.end(),
                  operation->type) == valid_quant_ops_type.end() ||
        output_operand->type.precision == NNADAPTER_FLOAT32) {
      continue;
    }

    auto input_operands = operation->input_operands;
    if (NeedPreQuant(model, input_operands[0])) {
      // Brother int8 ops share the same input.
      auto reference_operations = GetQuantOpsAroundOperand(
          operations, input_operands[0], true, valid_quant_ops_type);
      AddQuantOperation(model, input_operands[0], reference_operations);
    }

    if (NeedNextDequant(model, output_operand)) {
      // Brother int8 ops share the same output.
      auto reference_operations = GetQuantOpsAroundOperand(
          operations, output_operand, false, valid_quant_ops_type);
      AddDequantOperation(model, output_operand, reference_operations);
    }
  }
}

// Ascend int8 conv or fc only support int32 out. If last op is a quanted op,
// its out should be int8.
// For example:
// before:
//    conv -> model_out_var
// after:
//    conv -> var0 -> dequant -> var1 -> quant -> model_out_var
static void FixLastQuantizedOp(core::Model* model) {
  std::vector<NNAdapterOperationCode> valid_quant_ops_type{
      NNADAPTER_CONV_2D, NNADAPTER_FULLY_CONNECTED};
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    if (std::find(valid_quant_ops_type.begin(),
                  valid_quant_ops_type.end(),
                  operation->type) == valid_quant_ops_type.end()) {
      continue;
    }
    auto output_operand = operation->output_operands[0];
    auto next_operations = GetOperandConsumers(model, output_operand);
    if (output_operand->type.precision == NNADAPTER_FLOAT32 ||
        !next_operations.empty()) {
      continue;
    }

    AddDequantOperation(model, output_operand);
    AddQuantOperation(model, output_operand);
  }
}

// dequant's input's precision should be NNADAPTER_QUANT_INT32_SYMM_PER_LAYER.
// dequant's input's scale should be calculated.
static void FixDequantOps(core::Model* model) {
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    if (operation->type == NNADAPTER_DEQUANTIZE) {
      operation->input_operands[0]->type.precision =
          NNADAPTER_QUANT_INT32_SYMM_PER_LAYER;
      operation->input_operands[0]->type.symm_per_layer_params.scale =
          GetDequantScale(model, operation);
    }
  }
}

void FixQuantizedOps(core::Model* model) {
  InsertExtraQuantDequant(model);
  FixLastQuantizedOp(model);
  FixDequantOps(model);
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
