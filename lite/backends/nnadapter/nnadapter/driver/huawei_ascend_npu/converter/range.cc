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

int Program::ConvertRange(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Start
  auto start_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "start_operand: " << OperandToString(start_operand);
  // Limit
  auto limit_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "limit_operand: " << OperandToString(limit_operand);
  // Delta
  auto delta_operand = input_operands[2];
  NNADAPTER_VLOG(5) << "delta_operand: " << OperandToString(delta_operand);
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output_operand: " << OperandToString(output_operand);

  // Convert to GE operators
  auto start_operator = GetMappedOperator(start_operand);
  if (!start_operator) {
    start_operator = ConvertOperand(start_operand);
  }
  auto limit_operator = GetMappedOperator(limit_operand);
  if (!limit_operator) {
    limit_operator = ConvertOperand(limit_operand);
  }
  auto delta_operator = GetMappedOperator(delta_operand);
  if (!delta_operator) {
    delta_operator = ConvertOperand(delta_operand);
  }

  auto range_name = GetOperatorName(output_operand);
  auto range_op = std::make_shared<ge::op::Range>(range_name);
  SET_INPUT(range_op, start, start_operator);
  SET_INPUT(range_op, limit, limit_operator);
  SET_INPUT(range_op, delta, delta_operator);
  MAP_OUTPUT(range_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
