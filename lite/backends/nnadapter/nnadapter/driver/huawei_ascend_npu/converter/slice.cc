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

int Program::ConvertSlice(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 4);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input_operand: " << OperandToString(input_operand);
  auto input_dimension_count = input_operand->type.dimension_count;
  auto input_dimensions = input_operand->type.dimensions;

  auto axes_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "axes_operand: " << OperandToString(axes_operand);
  auto axes = reinterpret_cast<int32_t*>(axes_operand->buffer);
  NNADAPTER_CHECK_EQ(axes_operand->type.dimension_count, 1);
  auto axes_size = axes_operand->type.dimensions[0];
  NNADAPTER_CHECK_LE(axes_size, input_dimension_count);

  auto starts_operand = input_operands[2];
  NNADAPTER_VLOG(5) << "starts_operand: " << OperandToString(starts_operand);
  auto starts = reinterpret_cast<int32_t*>(starts_operand->buffer);
  NNADAPTER_CHECK_EQ(starts_operand->type.dimension_count, 1);
  auto starts_size = starts_operand->type.dimensions[0];

  auto ends_operand = input_operands[3];
  NNADAPTER_VLOG(5) << "ends_operand: " << OperandToString(ends_operand);
  auto ends = reinterpret_cast<int32_t*>(ends_operand->buffer);
  NNADAPTER_CHECK_EQ(ends_operand->type.dimension_count, 1);
  auto ends_size = ends_operand->type.dimensions[0];

  NNADAPTER_CHECK_EQ(axes_size, starts_size);
  NNADAPTER_CHECK_EQ(starts_size, ends_size);

  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output_operand: " << OperandToString(output_operand);

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }

  auto axes_operator = GetMappedOperator(axes_operand);
  if (!axes_operator) {
    axes_operator = ConvertOperand(axes_operand);
  }

  std::vector<int> offsets_vec(axes_size, 0);
  std::vector<int> size_vec(axes_size, 0);
  // Get begin/offset based on axes and starts
  for (int i = 0; i < axes_size; i++) {
    auto axis = axes[i];
    NNADAPTER_CHECK_LE(axis, input_dimension_count);
    NNADAPTER_CHECK_LE(starts[i], input_dimensions[axis]);
    offsets_vec[axis] = starts[i];
    size_vec[axis] = ends[i] - starts[i];
  }

  auto offsets_operator = AddInt32ConstantOperator(offsets_vec);
  auto size_operator = AddInt32ConstantOperator(size_vec);

  auto slice_name = GetOperatorName(output_operand);
  auto slice_op = std::make_shared<ge::op::Slice>(slice_name);
  SET_INPUT(slice_op, x, input_operator);
  SET_INPUT(slice_op, offsets, offsets_operator);
  SET_INPUT(slice_op, size, size_operator);
  MAP_OUTPUT(slice_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
