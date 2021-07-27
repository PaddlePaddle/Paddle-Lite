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

int Program::ConvertPow(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand_0 = input_operands[0];
  NNADAPTER_VLOG(5) << "input0: " << OperandToString(input_operand_0);

  auto input_operand_1 = input_operands[1];
  NNADAPTER_VLOG(5) << "input1: " << OperandToString(input_operand_1);

  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to GE operators
  auto input_operator_0 = GetMappedOperator(input_operand_0);
  if (!input_operator_0) {
    input_operator_0 = ConvertOperand(input_operand_0);
  }
  auto input_operator_1 = GetMappedOperator(input_operand_1);
  if (!input_operator_1) {
    input_operator_1 = ConvertOperand(input_operand_1);
  }

  auto pow_name = GetOperatorName(output_operand);
  auto pow_op = std::make_shared<ge::op::Pow>(pow_name);
  SET_INPUT(pow_op, x1, input_operator_0);
  SET_INPUT(pow_op, x2, input_operator_1);
  MAP_OUTPUT(pow_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
