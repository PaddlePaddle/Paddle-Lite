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

#include "operation/sum.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertSum(Converter* converter, core::Operation* operation) {
  SUM_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto sum_op = converter->AddOperator<ge::op::AddN>(output_operand);
  sum_op->set_attr_N(input_count);
  sum_op->create_dynamic_input_x(input_count);
  for (int i = 0; i < input_count; i++) {
    auto input_operand = input_operands[i];
    auto input_operator = converter->GetMappedOperator(input_operand);
    if (!input_operator) {
      input_operator = converter->ConvertOperand(input_operand);
    }
    SET_DYNAMIC_INPUT(sum_op, x, i, input_operator);
  }
  MAP_OUTPUT(sum_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
