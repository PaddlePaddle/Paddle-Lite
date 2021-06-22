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

int Program::ConvertElementwise(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input0
  auto input0_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input0: " << OperandToString(input0_operand);
  // Input1
  auto input1_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "input1: " << OperandToString(input1_operand);
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[2]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to HiAI operators
  auto input0_operator = ConvertOperand(input0_operand);
  auto input1_operator = ConvertOperand(input1_operand);
  std::shared_ptr<ge::Operator> eltwise_operator = nullptr;
  if (operation->type == NNADAPTER_ADD) {
    auto add_operator = AddOperator<ge::op::Add>(output_operand);
    add_operator->set_input_x1(*input0_operator);
    add_operator->set_input_x2(*input1_operator);
    eltwise_operator = add_operator;
  } else if (operation->type == NNADAPTER_SUB) {
    auto sub_operator = AddOperator<ge::op::Sub>(output_operand);
    sub_operator->set_input_x1(*input0_operator);
    sub_operator->set_input_x2(*input1_operator);
    eltwise_operator = sub_operator;
  } else if (operation->type == NNADAPTER_MUL) {
    auto mul_operator = AddOperator<ge::op::Mul>(output_operand);
    mul_operator->set_input_x(*input0_operator);
    mul_operator->set_input_y(*input1_operator);
    eltwise_operator = mul_operator;
  } else if (operation->type == NNADAPTER_DIV) {
    auto div_operator = AddOperator<ge::op::RealDiv>(output_operand);
    div_operator->set_input_x1(*input0_operator);
    div_operator->set_input_x2(*input1_operator);
    eltwise_operator = div_operator;
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported element-wise operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  if (fuse_code != NNADAPTER_FUSED_NONE) {
    auto act_operator = AddOperator<ge::op::Activation>(output_operand);
    act_operator->set_input_x(*eltwise_operator);
    act_operator->set_attr_mode(ConvertFuseCode(fuse_code));
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
