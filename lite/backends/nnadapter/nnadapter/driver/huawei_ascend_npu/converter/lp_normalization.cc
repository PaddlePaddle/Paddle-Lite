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

#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertLpNormalization(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 5);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  // Axis
  auto axis_operand = input_operands[1];
  auto axis_count = axis_operand->length / sizeof(int32_t);
  auto axis_data = reinterpret_cast<int32_t*>(axis_operand->buffer);
  ge::Operator::OpListInt axis;
  for (uint32_t i = 0; i < axis_count; i++) {
    NNADAPTER_VLOG(5) << "axis[" << i << "]=" << axis_data[i];
    axis.push_back(axis_data[i]);
  }
  // P
  auto p = *reinterpret_cast<int32_t*>(input_operands[2]->buffer);
  NNADAPTER_VLOG(5) << "p: " << p;
  // Epsilon
  auto epsilon = *reinterpret_cast<float*>(input_operands[3]->buffer);
  NNADAPTER_VLOG(5) << "epsilon: " << epsilon;
  // Keepdim
  auto keepdim = *reinterpret_cast<bool*>(input_operands[4]->buffer);
  NNADAPTER_VLOG(5) << "keepdim: " << keepdim;

  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }

  // Convert to GE operators
  if (p == 2 && keepdim) {
    auto l2_norm_name = GetOperatorName(output_operand);
    auto l2_norm_op = std::make_shared<ge::op::L2Normalize>(l2_norm_name);
    l2_norm_op->set_attr_axis(axis);
    l2_norm_op->set_attr_eps(epsilon);
    SET_INPUT(l2_norm_op, x, input_operator);
    MAP_OUTPUT(l2_norm_op, y, output_operand);
    return NNADAPTER_NO_ERROR;
  } else if (p == INT_MAX || p == INT_MIN || p == 0) {
    auto p_norm_name = GetOperatorName(output_operand);
    auto p_norm_op = std::make_shared<ge::op::LpNorm>(p_norm_name);
    p_norm_op->set_attr_p(p);
    p_norm_op->set_attr_axes(axis);
    p_norm_op->set_attr_epsilon(epsilon);
    p_norm_op->set_attr_keepdim(keepdim);
    SET_INPUT(p_norm_op, x, input_operator);
    MAP_OUTPUT(p_norm_op, y, output_operand);
    return NNADAPTER_NO_ERROR;
  } else {
    auto power_name_1 = GetOperatorName(output_operand) + "/power";
    auto power_op_1 = std::make_shared<ge::op::Power>(power_name_1);
    power_op_1->set_attr_power(static_cast<float>(p));
    power_op_1->set_attr_scale(1.0f);
    power_op_1->set_attr_shift(0.0f);
    SET_INPUT(power_op_1, x, input_operator);
    auto power_operator_1 = MAP_OUTPUT(power_op_1, y, output_operand);

    auto reduce_name = GetOperatorName(output_operand) + "/reduce";
    auto reduce_op = std::make_shared<ge::op::ReduceSumD>(reduce_name);
    reduce_op->set_attr_axes(axis);
    reduce_op->set_attr_keep_dims(keepdim);
    SET_INPUT(reduce_op, x, power_operator_1);
    auto reduce_operator = MAP_OUTPUT(reduce_op, y, output_operand);

    auto power_name_2 = GetOperatorName(output_operand);
    auto power_op_2 = std::make_shared<ge::op::Power>(power_name_2);
    power_op_2->set_attr_power(1.0f / p);
    power_op_2->set_attr_scale(1.0f);
    power_op_2->set_attr_shift(0.0f);
    SET_INPUT(power_op_2, x, reduce_operator);
    MAP_OUTPUT(power_op_2, y, output_operand);
    return NNADAPTER_NO_ERROR;
  }

  return NNADAPTER_INVALID_PARAMETER;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
