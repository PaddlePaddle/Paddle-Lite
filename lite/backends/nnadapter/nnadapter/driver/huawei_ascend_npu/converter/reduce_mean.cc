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

int Program::ConvertReduceMean(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 3);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input_operand: " << OperandToString(input_operand);
  // Axes
  auto axes_operand = input_operands[1];
  int axes_size = axes_operand->length / sizeof(int32_t);
  auto axes_data = reinterpret_cast<int32_t*>(axes_operand->buffer);
  for (int i = 0; i < axes_size; i++) {
    NNADAPTER_VLOG(5) << "axes[" << i << "]: " << axes_data[i];
  }
  // Keep_dim
  auto keep_dim_operand = input_operands[2];
  auto keep_dims = *reinterpret_cast<int8_t*>(keep_dim_operand->buffer);
  NNADAPTER_VLOG(5) << "keep_dims: " << keep_dims;
  bool keep_dim = keep_dims ? true : false;
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output_operand: " << OperandToString(output_operand);

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }
  auto axes_operator = ConvertOperand(axes_operand);
  auto reduce_mean_name = GetOperatorName(output_operand);
  // If keep_dim == false && reduce_all, need add reshape op, trans scalar to 1D
  // tensor with shape[1].
  if (!keep_dim && (axes_size == input_operand->type.dimension_count)) {
    auto reduce_mean_op = std::make_shared<ge::op::ReduceMean>(
        reduce_mean_name + "/reshape_before");
    reduce_mean_op->set_attr_keep_dims(keep_dim);
    SET_INPUT(reduce_mean_op, x, input_operator);
    SET_INPUT(reduce_mean_op, axes, axes_operator);
    std::shared_ptr<Operator> reduce_mean_operator =
        MAP_OUTPUT(reduce_mean_op, y, output_operand);
    // Add reshape op
    auto reshape_op = std::make_shared<ge::op::Reshape>(reduce_mean_name);
    auto shape_operator = AddInt32ConstantOperator({1});
    SET_INPUT(reshape_op, x, reduce_mean_operator);
    SET_INPUT(reshape_op, shape, shape_operator);
    MAP_OUTPUT(reshape_op, y, output_operand);
  } else {
    auto reduce_mean_op =
        std::make_shared<ge::op::ReduceMean>(reduce_mean_name);
    reduce_mean_op->set_attr_keep_dims(keep_dim);
    SET_INPUT(reduce_mean_op, x, input_operator);
    SET_INPUT(reduce_mean_op, axes, axes_operator);
    MAP_OUTPUT(reduce_mean_op, y, output_operand);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
