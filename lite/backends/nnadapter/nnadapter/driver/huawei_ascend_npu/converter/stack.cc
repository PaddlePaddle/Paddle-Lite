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

#include "core/operation/stack.h"
#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertStack(hal::Operation* operation) {
  STACK_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto N = input_count - 1;
  auto stack_name = GetOperatorName(output_operand);
  auto stack_op = std::make_shared<ge::op::Pack>(stack_name);
  stack_op->set_attr_axis(axis);
  stack_op->set_attr_N(N);
  stack_op->create_dynamic_input_x(N);
  for (int i = 0; i < N; i++) {
    auto input_operand = input_operands[i];
    auto input_operator = GetMappedOperator(input_operand);
    if (!input_operator) {
      input_operator = ConvertOperand(input_operand);
    }
    SET_DYNAMIC_INPUT(stack_op, x, i, input_operator);
  }
  MAP_OUTPUT(stack_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
