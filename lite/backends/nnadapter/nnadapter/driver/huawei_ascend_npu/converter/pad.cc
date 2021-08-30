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

int Program::ConvertPad(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Pads
  auto pads_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "pads: " << OperandToString(pads_operand);
  // Mode
  auto mode_operand = input_operands[2];
  auto mode_code = *reinterpret_cast<int32_t*>(mode_operand->buffer);
  std::string mode = ConvertPadMode(mode_code);
  NNADAPTER_CHECK_EQ(mode, "constant")
      << "Ascend npu only support mode=constant right now, "
         "but received mode is "
      << mode;
  NNADAPTER_VLOG(5) << "mode: " << OperandToString(mode_operand);
  // Value
  auto value_operand = input_operands[3];
  auto value = *reinterpret_cast<float*>(value_operand->buffer);
  NNADAPTER_VLOG(5) << "value: " << OperandToString(value_operand);
  NNADAPTER_CHECK_LT(std::abs(value), 1e-6)
      << "Ascend npu only support constant_values=0 right now, "
         "but received constant_value is "
      << value;
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }
  auto pads_operator = GetMappedOperator(pads_operand);
  if (!pads_operator) {
    pads_operator = ConvertOperand(pads_operand);
  }
  auto value_operator = ConvertOperand(value_operand);
  auto pad_name = GetOperatorName(output_operand);
  auto pad_op = std::make_shared<ge::op::PadV3>(pad_name);
  pad_op->set_attr_mode(mode);
  SET_INPUT(pad_op, x, input_operator);
  SET_INPUT(pad_op, paddings, pads_operator);
  SET_INPUT(pad_op, constant_values, value_operator);
  MAP_OUTPUT(pad_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
