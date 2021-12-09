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

#include "driver/huawei_ascend_npu/optimizer/fix_quant_ops.h"
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

static bool NeedPreQuant(hal::Model* model, hal::Operand* operand) {
  auto pre_operation = GetOperandProducer(model, operand);
  return pre_operation != nullptr && pre_operation->type != NNADAPTER_QUANTIZE;
}

static bool NeedNextDequant(hal::Model* model, hal::Operand* operand) {
  auto next_operation = GetOperandConsumers(model, operand)[0];
  return next_operation != nullptr &&
         next_operation->type != NNADAPTER_DEQUANTIZE;
}

// Add a quant operation after input_operand
static hal::Operand* AddQuantOperation(hal::Model* model,
                                       hal::Operand* input_operand) {
  NNADAPTER_CHECK_EQ(input_operand->type.precision, NNADAPTER_FLOAT32);
  NNADAPTER_CHECK_GE(input_operand->type.symm_per_layer_params.scale, 0.f);
  // Insert a new operand after input_operand
  auto output_operand = AddOperand(model);
  memcpy(&output_operand->type,
         &input_operand->type,
         sizeof(NNAdapterOperandType));
  InsertOperand(model, input_operand, output_operand, true);
  output_operand->type.precision = NNADAPTER_QUANT_INT8_SYMM_PER_LAYER;
  // Insert a new quant operation between input_operand and output_operand
  auto quant_operation = AddOperation(model);
  quant_operation->type = NNADAPTER_QUANTIZE;
  hal::Operand* axis_operand = AddInt32ConstantOperand(model, 1);
  float scale = input_operand->type.symm_per_layer_params.scale;
  hal::Operand* scale_operand = AddFloat32ConstantOperand(model, scale);
  hal::Operand* zero_point_operand = AddInt32ConstantOperand(model, 0);
  quant_operation->input_operands = {
      input_operand, axis_operand, scale_operand, zero_point_operand};
  quant_operation->output_operands = {output_operand};
  return output_operand;
}

// Add a dequant operation before output_operand
static hal::Operand* AddDequantOperation(hal::Model* model,
                                         hal::Operand* output_operand) {
  NNADAPTER_CHECK_EQ(output_operand->type.precision,
                     NNADAPTER_QUANT_INT8_SYMM_PER_LAYER);
  // Insert a new operand before output_operand
  auto input_operand = AddOperand(model);
  memcpy(&input_operand->type,
         &output_operand->type,
         sizeof(NNAdapterOperandType));
  InsertOperand(model, output_operand, input_operand, false);
  input_operand->type.precision = NNADAPTER_QUANT_INT32_SYMM_PER_LAYER;
  output_operand->type.precision = NNADAPTER_FLOAT32;
  // Insert a new dequant operation between input_operand and output_operand
  auto dequant_operation = AddOperation(model);
  dequant_operation->type = NNADAPTER_DEQUANTIZE;
  dequant_operation->input_operands = {input_operand};
  dequant_operation->output_operands = {output_operand};
  return input_operand;
}

float GetDequantScale(hal::Model* model, hal::Operation* dequant) {
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
 *   conv(quant) -> out1 -> dequant -> (out2 -> act ->) out -> op(not_dequant)
 */
static void FixQuantConv(hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    if (operation->type != NNADAPTER_CONV_2D) {
      continue;
    }
    auto output_operand = operation->output_operands[0];
    if (output_operand->type.precision != NNADAPTER_QUANT_INT8_SYMM_PER_LAYER) {
      continue;
    }

    auto input_operands = operation->input_operands;
    if (NeedPreQuant(model, input_operands[0])) {
      AddQuantOperation(model, input_operands[0]);
    }

    if (NeedNextDequant(model, output_operand)) {
      AddDequantOperation(model, output_operand);
    }

    // Unpack activations after dequant
    auto next_operation = GetOperandConsumers(model, output_operand)[0];
    if (next_operation->type == NNADAPTER_DEQUANTIZE) {
      output_operand = next_operation->output_operands[0];
    }
    NNADAPTER_CHECK_EQ(GetOperandProducer(model, output_operand)->type,
                       NNADAPTER_DEQUANTIZE);
    auto fuse_code = reinterpret_cast<int32_t*>(input_operands[8]->buffer);
    switch (*fuse_code) {
      case NNADAPTER_FUSED_RELU:
        AddUnaryOperation(model, output_operand, NNADAPTER_RELU);
        break;
      case NNADAPTER_FUSED_RELU6:
        AddUnaryOperation(model, output_operand, NNADAPTER_RELU6);
        break;
      default:
        NNADAPTER_CHECK_EQ(*fuse_code, NNADAPTER_FUSED_NONE)
            << "Unsupported fuse code: "
            << FuseCodeToString(static_cast<NNAdapterFuseCode>(*fuse_code));
    }
    *fuse_code = NNADAPTER_FUSED_NONE;
  }
}

static void SwapQuantReshape(hal::Operation* quantize_operation,
                             hal::Operation* reshape_operation) {
  auto quant_in = quantize_operation->input_operands[0];
  auto reshape_in = reshape_operation->input_operands[0];
  auto reshape_out = reshape_operation->output_operands[0];
  CopyOperandTypeExceptQuantParams(&reshape_in->type, reshape_out->type);
  reshape_in->type.precision = quant_in->type.precision;

  quantize_operation->input_operands[0] = reshape_in;
  quantize_operation->output_operands[0] = reshape_out;
  reshape_operation->input_operands[0] = quant_in;
  reshape_operation->output_operands[0] = reshape_in;
}

/**
 * step 1:
 * before:
 *   quant -> reshape -> fc(quant)
 * after:
 *   reshape -> quant -> fc(quant)
 *
 * step 2:
 * before:
 *   op(not_quant) -> in -> fc(quant)
 * after:
 *   op(not_quant) -> in -> quant -> in1 -> fc(quant)
 *
 * before:
 *   fc(quant) -> out -> op(not_dequant)
 * after:
 *   fc(quant) -> out1 -> dequant -> out -> op(not_dequant)
 *
 * step3:
 * before:
 *   fc(quant) -> dequant
 * after:
 *   fc(quant) -> dequant -> reshape
 */
static void FixQuantFullyConnected(hal::Model* model) {
  std::vector<hal::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    if (operation->type != NNADAPTER_FULLY_CONNECTED) {
      continue;
    }
    auto output_operand = operation->output_operands[0];
    if (output_operand->type.precision != NNADAPTER_QUANT_INT8_SYMM_PER_LAYER) {
      continue;
    }

    // step1, swap quantize and reshape
    auto input_operands = operation->input_operands;
    auto pre_operation = GetOperandProducer(model, input_operands[0]);
    if (pre_operation != nullptr && pre_operation->type == NNADAPTER_RESHAPE) {
      auto reshape_operation = pre_operation;
      pre_operation =
          GetOperandProducer(model, reshape_operation->input_operands[0]);
      if (pre_operation != nullptr &&
          pre_operation->type == NNADAPTER_QUANTIZE) {
        SwapQuantReshape(pre_operation, reshape_operation);
      }
    }

    // step2, add quantize and dequantize
    if (NeedPreQuant(model, input_operands[0])) {
      AddQuantOperation(model, input_operands[0]);
    }
    if (NeedPreQuant(model, input_operands[1])) {
      AddQuantOperation(model, input_operands[1]);
    }

    if (NeedNextDequant(model, output_operand)) {
      AddDequantOperation(model, output_operand);
    }

    // step3, add reshape after dequantize
    output_operand = operation->output_operands[0];
    auto next_operations = GetOperandConsumers(model, output_operand);
    if (next_operations.size() == 1 &&
        next_operations[0]->type == NNADAPTER_DEQUANTIZE) {
      auto dims_data = output_operand->type.dimensions.data;
      AddReshapeOperation(model,
                          next_operations[0]->output_operands[0],
                          {dims_data[0], dims_data[1]});
    }
  }
}

// dequant's input's precision should be NNADAPTER_QUANT_INT32_SYMM_PER_LAYER.
// dequant's input's scale should be calculated.
static void FixDequant(hal::Model* model) {
  std::vector<hal::Operation*> operations =
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

void FixQuantOps(hal::Model* model) {
  FixQuantConv(model);
  FixQuantFullyConnected(model);
  FixDequant(model);
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
