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

#include "core/operation/pad.h"
#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertPad(hal::Operation* operation) {
  PAD_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  std::string pad_mode = ConvertPadModeCodeToGEPadMode(mode);
  auto value = *reinterpret_cast<float*>(value_operand->buffer);
  NNADAPTER_CHECK_EQ(pad_mode, "constant")
      << "HuaewiAscendNPU only support mode=constant right now, "
         "but received mode is "
      << pad_mode;
  NNADAPTER_CHECK_LT(std::abs(value), 1e-6)
      << "HuaewiAscendNPU only support constant_values=0 right now, "
         "but received constant_value is "
      << value;
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
  pad_op->set_attr_mode(pad_mode);
  SET_INPUT(pad_op, x, input_operator);
  SET_INPUT(pad_op, paddings, pads_operator);
  SET_INPUT(pad_op, constant_values, value_operator);
  MAP_OUTPUT(pad_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
