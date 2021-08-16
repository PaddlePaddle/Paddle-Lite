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
  NNADAPTER_CHECK_EQ(input_count, 5);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input_operand: " << OperandToString(input_operand);
  auto input_dimension_count = input_operand->type.dimension_count;
  auto input_dimensions = input_operand->type.dimensions;
  // Axes
  auto axes_operand = input_operands[1];
  auto axes_count =
      axes_operand->length / static_cast<uint32_t>(sizeof(int32_t));
  NNADAPTER_CHECK_LE(axes_count, input_dimension_count);
  auto axes_data = reinterpret_cast<int32_t*>(axes_operand->buffer);
  std::vector<int32_t> origin_axes(axes_data, axes_data + axes_count);
  std::vector<int32_t> axes;
  for (uint32_t i = 0; i < input_dimension_count; i++) {
    axes.push_back(static_cast<int32_t>(i));
    NNADAPTER_VLOG(5) << "axes[" << i << "] = " << i;
  }
  // Starts
  auto starts_operand = input_operands[2];
  auto starts_count =
      starts_operand->length / static_cast<uint32_t>(sizeof(int32_t));
  auto starts_data = reinterpret_cast<int32_t*>(starts_operand->buffer);
  std::vector<int32_t> origin_starts(starts_data, starts_data + starts_count);
  std::vector<int32_t> starts;
  for (uint32_t i = 0; i < input_dimension_count; i++) {
    auto iter = std::find(
        origin_axes.begin(), origin_axes.end(), static_cast<int32_t>(i));
    if (iter == origin_axes.end()) {
      starts.push_back(0);
    } else {
      int32_t index = std::distance(origin_axes.begin(), iter);
      starts.push_back(origin_starts[index]);
    }
    NNADAPTER_VLOG(5) << "starts[" << i << "] = " << starts[i];
  }
  NNADAPTER_CHECK_EQ(axes.size(), starts.size());
  // Ends
  auto ends_operand = input_operands[3];
  auto ends_count =
      ends_operand->length / static_cast<uint32_t>(sizeof(int32_t));
  auto ends_data = reinterpret_cast<int32_t*>(ends_operand->buffer);
  std::vector<int32_t> origin_ends(ends_data, ends_data + ends_count);
  std::vector<int32_t> ends;
  for (uint32_t i = 0; i < input_dimension_count; i++) {
    auto iter = std::find(
        origin_axes.begin(), origin_axes.end(), static_cast<int32_t>(i));
    if (iter == origin_axes.end()) {
      ends.push_back(input_dimensions[i]);
    } else {
      int32_t index = std::distance(origin_axes.begin(), iter);
      int end = origin_ends[index];
      if (end < 0) {
        end += input_dimensions[i];
      }
      end = std::min(end, input_dimensions[i]);
      ends.push_back(end);
    }
    NNADAPTER_VLOG(5) << "ends[" << i << "] = " << ends[i];
  }
  NNADAPTER_CHECK_EQ(axes.size(), ends.size());
  // Steps
  auto steps_operand = input_operands[4];
  auto steps_count =
      steps_operand->length / static_cast<uint32_t>(sizeof(int32_t));
  auto steps_data = reinterpret_cast<int32_t*>(steps_operand->buffer);
  for (uint32_t i = 0; i < steps_count; i++) {
    if (steps_data[i] != 1) {
      NNADAPTER_LOG(WARNING) << "Only support step == 1 now.";
      return NNADAPTER_INVALID_PARAMETER;
    }
  }
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output_operand: " << OperandToString(output_operand);

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }
  std::vector<int> size(axes.size(), 0);
  // Get begin/offset based on axes and starts
  for (size_t i = 0; i < axes.size(); i++) {
    size[i] = ends[i] - starts[i];
  }
  auto offsets_operator = AddInt32ConstantOperator(starts);
  auto size_operator = AddInt32ConstantOperator(size);
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
