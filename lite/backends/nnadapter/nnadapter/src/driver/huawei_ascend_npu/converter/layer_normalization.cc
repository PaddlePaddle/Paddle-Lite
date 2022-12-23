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

#include "operation/layer_normalization.h"

#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertLayerNormalization(Converter* converter,
                              core::Operation* operation) {
  LAYER_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (input_operator == nullptr) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  if (IsDynamicShapeOperandType(input_operand->type)) {
    auto scale_operator = converter->GetMappedOperator(scale_operand);
    if (scale_operator == nullptr) {
      scale_operator = converter->ConvertOperand(scale_operand);
    }
    auto bias_operator = converter->GetMappedOperator(bias_operand);
    if (bias_operator == nullptr) {
      bias_operator = converter->ConvertOperand(bias_operand);
    }

    // Layer normalization
    auto layer_norm_op =
        converter->AddOperator<ge::op::LayerNorm>(output_operand);
    layer_norm_op->set_attr_epsilon(epsilon);
    layer_norm_op->set_attr_begin_norm_axis(begin_norm_axis);
    layer_norm_op->set_attr_begin_params_axis(begin_norm_axis);
    SET_INPUT(layer_norm_op, x, input_operator);
    SET_INPUT(layer_norm_op, beta, bias_operator);
    SET_INPUT(layer_norm_op, gamma, scale_operator);
    MAP_OUTPUT(layer_norm_op, y, output_operand);
  } else {
    /**
     * Use small operators to calculate, and the formula is as follows:
     * mean = np.mean(x, reduce_axis, keepdims=True)
     * variance = np.mean(np.power((x - mean), 2), reduce_axis, keepdims=True)
     * output = scale *((x - mean) / np.sqrt(variance + epsilon)) + bias
     *
     */
    auto batch_size = ProductionOfDimensions(
        input_operand->type.dimensions.data, begin_norm_axis);
    auto inner_size = ProductionOfDimensions(
        input_operand->type.dimensions.data + begin_norm_axis,
        input_operand->type.dimensions.count - begin_norm_axis);
    // Reshape
    auto shape = std::vector<int32_t>(
        {static_cast<int32_t>(batch_size), static_cast<int32_t>(inner_size)});
    auto shape_operator = converter->AddInt32ConstantOperator(shape);
    auto reshape_op = converter->AddOperator<ge::op::Reshape>(output_operand,
                                                              "input_reshape");
    SET_INPUT(reshape_op, x, input_operator);
    SET_INPUT(reshape_op, shape, shape_operator);
    auto input_reshape_operator = MAP_OUTPUT(reshape_op, y, output_operand);
    // Mean: np.mean(x, reduce_axis, keepdims=True)
    auto input_reduce_mean_op = converter->AddOperator<ge::op::ReduceMean>(
        output_operand, "input_reduce_mean");
    auto input_reduce_mean_axes_operator =
        converter->AddInt32ConstantOperator(std::vector<int32_t>({1}));
    input_reduce_mean_op->set_attr_keep_dims(true);
    SET_INPUT(input_reduce_mean_op, x, input_reshape_operator);
    SET_INPUT(input_reduce_mean_op, axes, input_reduce_mean_axes_operator);
    auto input_mean_operator =
        MAP_OUTPUT(input_reduce_mean_op, y, output_operand);
    // Sub: x - mean
    auto sub_op =
        converter->AddOperator<ge::op::Sub>(output_operand, "input_sub_mean");
    SET_INPUT(sub_op, x1, input_reshape_operator);
    SET_INPUT(sub_op, x2, input_mean_operator);
    auto sub_operator = MAP_OUTPUT(sub_op, y, output_operand);
    // Square: np.power((x - mean),2
    auto square_op =
        converter->AddOperator<ge::op::Square>(output_operand, "square");
    SET_INPUT(square_op, x, sub_operator);
    auto square_operator = MAP_OUTPUT(square_op, y, output_operand);
    // ReduceMean for variance:  np.mean(np.power((x - mean),2),reduce_axis,
    // keepdims=True)
    auto variance_reduce_mean_op = converter->AddOperator<ge::op::ReduceMean>(
        output_operand, "variance_reduce_mean");
    auto variance_reduce_mean_axes_operator =
        converter->AddInt32ConstantOperator(std::vector<int32_t>({1}));
    variance_reduce_mean_op->set_attr_keep_dims(true);
    SET_INPUT(variance_reduce_mean_op, x, square_operator);
    SET_INPUT(
        variance_reduce_mean_op, axes, variance_reduce_mean_axes_operator);
    auto variance_operator =
        MAP_OUTPUT(variance_reduce_mean_op, y, output_operand);
    // Add: variance + epsilon
    auto add_op = converter->AddOperator<ge::op::Add>(output_operand, "add");
    auto epsilon_operator =
        converter->AddFloat32ConstantOperator(std::vector<float>({epsilon}));
    SET_INPUT(add_op, x1, variance_operator);
    SET_INPUT(add_op, x2, epsilon_operator);
    auto add_operator = MAP_OUTPUT(add_op, y, output_operand);
    // Sqrt: np.sqrt(variance + epsilon)
    auto sqrt_op = converter->AddOperator<ge::op::Sqrt>(output_operand, "sqrt");
    SET_INPUT(sqrt_op, x, add_operator);
    auto sqrt_operator = MAP_OUTPUT(sqrt_op, y, output_operand);
    // Div: (x - mean) / np.sqrt(variance + epsilon)
    auto div_op = converter->AddOperator<ge::op::Xdivy>(output_operand, "div");
    SET_INPUT(div_op, x1, sub_operator);
    SET_INPUT(div_op, x2, sqrt_operator);
    auto div_operator = MAP_OUTPUT(div_op, y, output_operand);
    // Reshape output to origin shape
    auto output_shape =
        std::vector<int32_t>(input_operand->type.dimensions.data,
                             input_operand->type.dimensions.data +
                                 input_operand->type.dimensions.count);
    auto output_shape_operator =
        converter->AddInt32ConstantOperator(output_shape);
    auto reshape_output_op = converter->AddOperator<ge::op::Reshape>(
        output_operand, "reshape_output");
    SET_INPUT(reshape_output_op, x, div_operator);
    SET_INPUT(reshape_output_op, shape, output_shape_operator);
    auto reshape_output_operator =
        MAP_OUTPUT(reshape_output_op, y, output_operand);
    // Scale
    auto scale_shape =
        std::vector<int32_t>(scale_operand->type.dimensions.data,
                             scale_operand->type.dimensions.data +
                                 scale_operand->type.dimensions.count);
    scale_shape.insert(scale_shape.begin(), begin_norm_axis, 1);
    auto scale_operator = converter->ConvertOperand(scale_operand, scale_shape);
    auto mul_scale_op =
        converter->AddOperator<ge::op::Mul>(output_operand, "mul_scale");
    SET_INPUT(mul_scale_op, x1, reshape_output_operator);
    SET_INPUT(mul_scale_op, x2, scale_operator);
    auto mul_scale_operator = MAP_OUTPUT(mul_scale_op, y, output_operand);
    // Bias
    auto bias_shape =
        std::vector<int32_t>(bias_operand->type.dimensions.data,
                             bias_operand->type.dimensions.data +
                                 bias_operand->type.dimensions.count);
    bias_shape.insert(bias_shape.begin(), begin_norm_axis, 1);
    auto bias_operator = converter->ConvertOperand(bias_operand, bias_shape);
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
