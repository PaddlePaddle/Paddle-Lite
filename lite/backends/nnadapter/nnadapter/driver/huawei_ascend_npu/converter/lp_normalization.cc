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
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);

  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);

  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Axis
  auto axis = *reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  NNADAPTER_VLOG(5) << "axis: " << axis;

  // P
  auto p = *reinterpret_cast<int32_t*>(input_operands[2]->buffer);
  NNADAPTER_VLOG(5) << "p: " << p;
  NNADAPTER_CHECK_EQ(p, 2) << "Only supports P=2 yet!";

  // Epsilon
  auto epsilon = *reinterpret_cast<float*>(input_operands[3]->buffer);
  NNADAPTER_VLOG(5) << "epsilon: " << epsilon;

  // Convert to GE operators
  if (p == 2) {
    auto input_operator = GetMappedOperator(input_operand);
    if (!input_operator) {
      input_operator = ConvertOperand(input_operand);
    }
    auto l2_norm_name = GetOperatorName(output_operand);
    auto l2_norm_op = std::make_shared<ge::op::L2Normalize>(l2_norm_name);
    l2_norm_op->set_attr_axis(ge::Operator::OpListInt({axis}));
    l2_norm_op->set_attr_eps(epsilon);
    SET_INPUT(l2_norm_op, x, input_operator);
    MAP_OUTPUT(l2_norm_op, y, output_operand);
    return NNADAPTER_NO_ERROR;
  }

  return NNADAPTER_INVALID_PARAMETER;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
