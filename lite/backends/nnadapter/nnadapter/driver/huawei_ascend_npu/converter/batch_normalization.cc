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

#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertBatchNormalization(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 6);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input_operand: " << OperandToString(input_operand);

  auto scale_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "scale_operand: " << OperandToString(scale_operand);

  auto bias_operand = input_operands[2];
  NNADAPTER_VLOG(5) << "bias_operand: " << OperandToString(bias_operand);

  auto mean_operand = input_operands[3];
  NNADAPTER_VLOG(5) << "mean_operand: " << OperandToString(mean_operand);

  auto variance_operand = input_operands[4];
  NNADAPTER_VLOG(5) << "variance_operand: "
                    << OperandToString(variance_operand);

  auto epsilon_operand = input_operands[5];
  NNADAPTER_VLOG(5) << "epsilon_operand: " << OperandToString(epsilon_operand);

  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output_operand: " << OperandToString(output_operand);

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }

  auto scale_operator = GetMappedOperator(scale_operand);
  if (!scale_operator) {
    scale_operator = ConvertOperand(scale_operand);
  }

  auto offset_operator = GetMappedOperator(bias_operand);
  if (!offset_operator) {
    offset_operator = ConvertOperand(bias_operand);
  }

  auto mean_operator = GetMappedOperator(mean_operand);
  if (!mean_operator) {
    mean_operator = ConvertOperand(mean_operand);
  }

  auto variance_operator = GetMappedOperator(variance_operand);
  if (!variance_operator) {
    variance_operator = ConvertOperand(variance_operand);
  }

  auto epsilon = *reinterpret_cast<float*>(epsilon_operand->buffer);
  auto batch_norm_name = GetOperatorName(output_operand);
  auto batch_norm_op = std::make_shared<ge::op::BatchNorm>(batch_norm_name);
  batch_norm_op->set_attr_epsilon(epsilon);
  batch_norm_op->set_attr_data_format("NCHW");
  batch_norm_op->set_attr_is_training(false);
  SET_INPUT(batch_norm_op, x, input_operator);
  SET_INPUT(batch_norm_op, scale, scale_operator);
  SET_INPUT(batch_norm_op, offset, offset_operator);
  SET_INPUT(batch_norm_op, mean, mean_operator);
  SET_INPUT(batch_norm_op, variance, variance_operator);
  MAP_OUTPUT(batch_norm_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
