// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/group_normalization.h"

#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertGroupNormalization(Converter* converter,
                              core::Operation* operation) {
  GROUP_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS
  if (IsDynamicShapeOperandType(input_operand->type)) {
    NNADAPTER_LOG(FATAL)
        << "GroupNormalization does not support dynamic shape.";
  }

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (input_operator == nullptr) {
    input_operator = converter->ConvertOperand(input_operand);
  }

  /**
   * Use small operators to calculate, and the formula is as follows:
   * input = reshape(input, shape=[batch_size, groups, -1])
   * mean = reduce_mean(input, axis=2, keep_dims=True)
   * var = reduce_sum(square(input - mean), axis=2, keep_dims=True) / (channel *
   * height * width / groups)
   * std = sqrt(var + epsilon)
   * output = (input - mean) / std
   * output = reshape(output, shape=[batch_size, channel, height, width])
   * output = output * scale + bias
   *
   */
  // Reshape Input to [batch_size, groups, -1]
  auto shape = std::vector<int32_t>(
      {input_operand->type.dimensions.data[0], groups, -1});
  auto input_shape_operator = converter->AddInt32ConstantOperator(shape);
  auto reshape_input_op =
      converter->AddOperator<ge::op::Reshape>(output_operand, "reshape_input");
  SET_INPUT(reshape_input_op, x, input_operator);
  SET_INPUT(reshape_input_op, shape, input_shape_operator);
  auto reshape_input_operator = MAP_OUTPUT(reshape_input_op, y, output_operand);
  // Mean
  auto reduce_mean_op =
      converter->AddOperator<ge::op::ReduceMean>(output_operand, "reduce_mean");
  auto reduce_mean_axes_operator =
      converter->AddInt32ConstantOperator(std::vector<int32_t>({2}));
  reduce_mean_op->set_attr_keep_dims(true);
  SET_INPUT(reduce_mean_op, x, reshape_input_operator);
  SET_INPUT(reduce_mean_op, axes, reduce_mean_axes_operator);
  auto reduce_mean_operator = MAP_OUTPUT(reduce_mean_op, y, output_operand);
  // Input - Mean
  auto sub_op =
      converter->AddOperator<ge::op::Sub>(output_operand, "input_sub_mean");
  SET_INPUT(sub_op, x1, reshape_input_operator);
  SET_INPUT(sub_op, x2, reduce_mean_operator);
  auto sub_operator = MAP_OUTPUT(sub_op, y, output_operand);
  // Square
  auto square_op =
      converter->AddOperator<ge::op::Square>(output_operand, "square");
  SET_INPUT(square_op, x, sub_operator);
  auto square_operator = MAP_OUTPUT(square_op, y, output_operand);
  // ReduceSum
  auto reduce_sum_op =
      converter->AddOperator<ge::op::ReduceSum>(output_operand, "reduce_sum");
  auto reduce_sum_axes_operator =
      converter->AddInt32ConstantOperator(std::vector<int32_t>({2}));
  reduce_sum_op->set_attr_keep_dims(true);
  SET_INPUT(reduce_sum_op, x, square_operator);
  SET_INPUT(reduce_sum_op, axes, reduce_sum_axes_operator);
  auto reduce_sum_operator = MAP_OUTPUT(reduce_sum_op, y, output_operand);
  // Variance
  auto div_op = converter->AddOperator<ge::op::Xdivy>(output_operand, "div");
  float block_num = input_operand->type.dimensions.data[1] *
                    input_operand->type.dimensions.data[2] *
                    input_operand->type.dimensions.data[3] / groups;
  auto block_num_operator =
      converter->AddFloat32ConstantOperator(std::vector<float>({block_num}));
  SET_INPUT(div_op, x1, reduce_sum_operator);
  SET_INPUT(div_op, x2, block_num_operator);
  auto variance_operator = MAP_OUTPUT(div_op, y, output_operand);
  // Add
  auto add_op = converter->AddOperator<ge::op::Add>(output_operand, "add");
  auto epsilon_operator =
      converter->AddFloat32ConstantOperator(std::vector<float>({epsilon}));
  SET_INPUT(add_op, x1, variance_operator);
  SET_INPUT(add_op, x2, epsilon_operator);
  auto add_operator = MAP_OUTPUT(add_op, y, output_operand);
  // Sqrt
  auto sqrt_op = converter->AddOperator<ge::op::Sqrt>(output_operand, "sqrt");
  SET_INPUT(sqrt_op, x, add_operator);
  auto std_operator = MAP_OUTPUT(sqrt_op, y, output_operand);
  // Input Normalization
  auto input_normalization_div_op = converter->AddOperator<ge::op::Xdivy>(
      output_operand, "input_normalization");
  SET_INPUT(input_normalization_div_op, x1, sub_operator);
  SET_INPUT(input_normalization_div_op, x2, std_operator);
  auto input_normalization_div_operator =
      MAP_OUTPUT(input_normalization_div_op, y, output_operand);
  // Reshape output
  auto output_shape =
      std::vector<int32_t>(input_operand->type.dimensions.data,
                           input_operand->type.dimensions.data +
                               input_operand->type.dimensions.count);
  auto output_shape_operator =
      converter->AddInt32ConstantOperator(output_shape);
  auto reshape_output_op =
      converter->AddOperator<ge::op::Reshape>(output_operand, "reshape_output");
  SET_INPUT(reshape_output_op, x, input_normalization_div_operator);
  SET_INPUT(reshape_output_op, shape, output_shape_operator);
  auto reshape_output_operator =
      MAP_OUTPUT(reshape_output_op, y, output_operand);
  // Scale and bias
  std::shared_ptr<Operator> mul_scale_operator = nullptr;
  if (scale_operand) {
    auto scale_operator = converter->ConvertOperand(
        scale_operand,
        std::vector<int32_t>(
            {1, scale_operand->type.dimensions.data[0], 1, 1}));
    auto mul_scale_op =
        converter->AddOperator<ge::op::Mul>(output_operand, "mul_scale");
    SET_INPUT(mul_scale_op, x1, reshape_output_operator);
    SET_INPUT(mul_scale_op, x2, scale_operator);
    mul_scale_operator = MAP_OUTPUT(mul_scale_op, y, output_operand);
  } else {
    mul_scale_operator = reshape_output_operator;
  }
  if (bias_operand) {
    auto bias_operator = converter->ConvertOperand(
        bias_operand,
        std::vector<int32_t>({1, bias_operand->type.dimensions.data[0], 1, 1}));
    auto add_bias_op =
        converter->AddOperator<ge::op::Add>(output_operand, "add_bias");
    SET_INPUT(add_bias_op, x1, mul_scale_operator);
    SET_INPUT(add_bias_op, x2, bias_operator);
    MAP_OUTPUT(add_bias_op, y, output_operand);
  }

  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
