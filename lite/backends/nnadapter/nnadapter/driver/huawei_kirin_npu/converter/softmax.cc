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

#include "driver/huawei_kirin_npu/converter.h"
#include "utility/debug.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int Program::ConvertSoftmax(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Axis
  auto axis = *reinterpret_cast<int32_t*>(input_operands[1]->buffer);
  if (axis < 0) {
    axis += input_operand->type.dimension_count;
  }
  NNADAPTER_VLOG(5) << "axis=" << axis;
  NNADAPTER_CHECK(!(axis == 2 && input_operand->type.dimension_count > 3 &&
                    input_operand->type.dimensions[3] != 1))
      << "Unsupported case: axis = " << axis
      << " dims = " << DimensionsToString(input_operand->type.dimensions,
                                          input_operand->type.dimension_count);
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to HiAI operators
  auto input_operator = ConvertOperand(input_operand);
  auto softmax_operator = AddOperator<ge::op::Softmax>(output_operand);
  softmax_operator->set_input_x(*input_operator);
  softmax_operator->set_attr_axis(axis);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
