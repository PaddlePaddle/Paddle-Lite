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
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int Program::ConvertSplit(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_GE(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Axis
  auto axis_operand = input_operands[1];
  auto axis = *reinterpret_cast<int32_t*>(axis_operand->buffer);
  if (axis < 0) {
    axis += input_operand->type.dimension_count;
  }
  NNADAPTER_VLOG(5) << "axis=" << axis;
  // Split
  auto split_operand = input_operands[2];
  auto split_count = split_operand->length / sizeof(int32_t);
  auto split_data = reinterpret_cast<int32_t*>(split_operand->buffer);
  NNADAPTER_CHECK_EQ(split_count, output_count);
  int split_sum = 0;
  for (uint32_t i = 0; i < split_count; i++) {
    NNADAPTER_VLOG(5) << "split[" << i << "]=" << split_data[i];
    split_sum += split_data[i];
  }
  NNADAPTER_CHECK_EQ(split_sum, input_operand->type.dimensions[axis]);
  // Output
  for (size_t i = 0; i < output_count; i++) {
    NNADAPTER_VLOG(5) << "output" << i << ": "
                      << OperandToString(output_operands[i]);
  }

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }
  auto split_operator = AddInt32ConstantOperator(
      std::vector<int32_t>(split_data, split_data + split_count));
  auto axis_operator = AddInt32ConstantOperator(std::vector<int32_t>({axis}));
  auto split_name = GetOperatorName(output_operands[0]);
  auto split_op = std::make_shared<hiai::op::SplitV>(split_name);
  split_op->set_attr_num_split(split_count);
  SET_INPUT(split_op, x, input_operator);
  SET_INPUT(split_op, size_splits, split_operator);
  SET_INPUT(split_op, split_dim, axis_operator);
  for (int i = 0; i < split_count; i++) {
    // Start from 1 for dynamic output in HiAI
    MAP_DYNAMIC_OUTPUT(split_op, y, i + 1, output_operands[i]);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
