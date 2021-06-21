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
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int Program::ConvertFullyConnected(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Weight
  auto weight_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "weight: " << OperandToString(weight_operand);
  NNADAPTER_CHECK_EQ(weight_operand->type.dimension_count, 2);
  auto num_units = weight_operand->type.dimensions[0];
  auto input_size = weight_operand->type.dimensions[1];
  auto batch_size =
      ProductionOfDimensions(input_operand->type.dimensions,
                             input_operand->type.dimension_count) /
      input_size;
  // Bias
  auto bias_operand = input_operands[2];
  NNADAPTER_VLOG(5) << "bias: " << OperandToString(bias_operand);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimension_count, 1);
  NNADAPTER_CHECK_EQ(num_units, bias_operand->type.dimensions[0]);
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[3]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to HiAI operators
  // Add input operator and reshape it to (batch_size, input_size, 1, 1)
  auto input_operator = ConvertOperand(input_operand);
  auto reshaped_input_operator = AddOperator<ge::op::Reshape>();
  reshaped_input_operator->set_input_tensor(*input_operator);
  reshaped_input_operator->set_attr_shape({batch_size, input_size, 1, 1});
  reshaped_input_operator->set_attr_axis(0);
  // Add weight operator and reshape to to (num_units, input_size, 1, 1)
  auto weight_operator =
      ConvertOperand(weight_operand, {num_units, input_size, 1, 1});
  // Add bias operator and reshape to to (1, num_units, 1, 1)
  auto bias_operator = ConvertOperand(bias_operand, {1, num_units, 1, 1});
  // Add fc operator and reshape back to the origin dimension of output operand
  auto fc_operator = AddOperator<ge::op::FullConnection>(output_operand);
  fc_operator->set_input_x(*reshaped_input_operator);
  fc_operator->set_input_w(*weight_operator);
  fc_operator->set_input_b(*bias_operator);
  auto reshaped_fc_operator = AddOperator<ge::op::Reshape>(output_operand);
  reshaped_fc_operator->set_input_tensor(*fc_operator);
  auto output_dimensions = ConvertDimensions(
      output_operand->type.dimensions, output_operand->type.dimension_count);
  reshaped_fc_operator->set_attr_shape(ge::AttrValue::LIST_INT(
      output_dimensions.begin(), output_dimensions.end()));
  reshaped_fc_operator->set_attr_axis(0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
