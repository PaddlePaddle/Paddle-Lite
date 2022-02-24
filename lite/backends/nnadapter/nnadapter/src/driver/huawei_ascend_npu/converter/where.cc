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

#include "operation/where.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertWhere(Converter* converter, core::Operation* operation) {
  WHERE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto condition_operator = converter->GetMappedOperator(condition_operand);
  if (!condition_operator) {
    condition_operator = converter->ConvertOperand(condition_operand);
  }
  auto input0_operator = converter->GetMappedOperator(input0_operand);
  if (!input0_operator) {
    input0_operator = converter->ConvertOperand(input0_operand);
  }
  auto input1_operator = converter->GetMappedOperator(input1_operand);
  if (!input1_operator) {
    input1_operator = converter->ConvertOperand(input1_operand);
  }
  auto where_op = converter->AddOperator<ge::op::Select>(output_operand);
  SET_INPUT(where_op, condition, condition_operator);
  SET_INPUT(where_op, x1, input0_operator);
  SET_INPUT(where_op, x2, input1_operator);
  MAP_OUTPUT(where_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
