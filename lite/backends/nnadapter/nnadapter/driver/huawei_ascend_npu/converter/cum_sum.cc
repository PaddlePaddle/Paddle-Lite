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

#include "core/operation/cum_sum.h"
#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertCumSum(hal::Operation* operation) {
  CUM_SUM_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (input_operator == nullptr) {
    input_operator = ConvertOperand(input_operand);
  }
  auto axis_operand = input_operands[1];
  auto axis_operator = GetMappedOperator(axis_operand);
  if (axis_operator == nullptr) {
    axis_operator = ConvertOperand(axis_operand);
  }
  auto cum_sum_name = GetOperatorName(output_operand);
  auto cum_sum_op = std::make_shared<ge::op::Cumsum>(cum_sum_name);
  cum_sum_op->set_attr_exclusive(exclusive);
  cum_sum_op->set_attr_reverse(reverse);
  SET_INPUT(cum_sum_op, x, input_operator);
  SET_INPUT(cum_sum_op, axis, axis_operator);
  MAP_OUTPUT(cum_sum_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
