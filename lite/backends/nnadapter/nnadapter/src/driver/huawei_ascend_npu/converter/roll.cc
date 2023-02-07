// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/roll.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertRoll(Converter* converter, core::Operation* operation) {
  ROLL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto shifts_operator = converter->GetMappedOperator(shifts_operand);
  if (!shifts_operator) {
    shifts_operator = converter->ConvertOperand(shifts_operand);
  }
  auto axes_operator = converter->GetMappedOperator(axes_operand);
  if (!axes_operator) {
    axes_operator = converter->ConvertOperand(axes_operand);
  }
  auto roll_op = converter->AddOperator<ge::op::RollV2>(output_operand);
  SET_INPUT(roll_op, input, input_operator);
  SET_INPUT(roll_op, shift, shifts_operator);
  SET_INPUT(roll_op, axes, axes_operator);
  MAP_OUTPUT(roll_op, output, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
