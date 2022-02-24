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
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {
/**
 * hiai::op::LayerNorm only support begin_norm_axis = 1, so reshape the tensor
 * if necessary
 * Conditions:
 * 1. (n, c, h, w), axis=1 -> no need
 * 2. (n, c, h, w), axis=2 -> (n * c, h, w, 1)
 * 3. (n, c, h, w), axis=3 -> (n * c * h, w, 1)
 * 4. (n, h, w), axis=1 -> (n, h, w, 1)
 * 5. (n, h, w), axis=2 -> (n * h, w, 1, 1)
 * 6. (h, w), axis=1 -> (h, w, 1, 1)
 *
 * The output will be reshaped to original shape.
 */
int ConvertLayerNormalization(Converter* converter,
                              core::Operation* operation) {
  LAYER_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto scale_operator = converter->ConvertOperand(scale_operand);
  auto bias_operator = converter->ConvertOperand(bias_operand);
  bool need_reshape =
      (input_operand->type.dimensions.count != 4 || begin_norm_axis != 1)
          ? true
          : false;
  // Reshape input tensor if necessary
  if (need_reshape) {
    std::vector<int32_t> reshape_size = {};
    int32_t assis = 1;
    for (int32_t i = 0; i < begin_norm_axis; i++) {
      assis *= input_operand->type.dimensions.data[i];
    }
    reshape_size.push_back(assis);
    for (int32_t i = begin_norm_axis; i < input_operand->type.dimensions.count;
         i++) {
      reshape_size.push_back(input_operand->type.dimensions.data[i]);
    }
    int32_t pad_size = 4 - reshape_size.size();
    for (int32_t i = 0; i < pad_size; i++) {
      reshape_size.push_back(1);
    }
    // Reshape input
    auto shape_operator = converter->AddInt32ConstantOperator(reshape_size);
    auto input_reshape_op =
        converter->AddOperator<hiai::op::Reshape>(output_operand);
    SET_INPUT(input_reshape_op, x, input_operator);
    SET_INPUT(input_reshape_op, shape, shape_operator);
    input_operator = MAP_OUTPUT(input_reshape_op, y, output_operand);
    // Reshape scale and bias
    std::vector<int32_t> scale_bias_shape = {
        1, reshape_size[1], reshape_size[2], reshape_size[3]};
    auto scale_shape_operator =
        converter->AddInt32ConstantOperator(scale_bias_shape);
    auto scale_reshape_op =
        converter->AddOperator<hiai::op::Reshape>(output_operand);
    SET_INPUT(scale_reshape_op, x, scale_operator);
    SET_INPUT(scale_reshape_op, shape, scale_shape_operator);
    scale_operator = MAP_OUTPUT(scale_reshape_op, y, scale_operand);
    auto bias_shape_operator =
        converter->AddInt32ConstantOperator(scale_bias_shape);
    auto bias_reshape_op =
        converter->AddOperator<hiai::op::Reshape>(output_operand);
    SET_INPUT(bias_reshape_op, x, bias_operator);
    SET_INPUT(bias_reshape_op, shape, bias_shape_operator);
    bias_operator = MAP_OUTPUT(bias_reshape_op, y, bias_operand);
  }
  // Layer normalization
  auto layer_norm_op =
      converter->AddOperator<hiai::op::LayerNorm>(output_operand);
  layer_norm_op->set_attr_epsilon(epsilon);
  SET_INPUT(layer_norm_op, x, input_operator);
  SET_INPUT(layer_norm_op, gamma, scale_operator);
  SET_INPUT(layer_norm_op, beta, bias_operator);
  auto output_operator = MAP_OUTPUT(layer_norm_op, y, output_operand);
  // Reshape the output if necessary
  if (need_reshape) {
    auto original_shape = input_operand->type.dimensions.data;
    auto original_shape_count = input_operand->type.dimensions.count;
    std::vector<int32_t> output_shape(original_shape,
                                      original_shape + original_shape_count);
    auto output_shape_operator =
        converter->AddInt32ConstantOperator(output_shape);
    auto output_reshape_op =
        converter->AddOperator<hiai::op::Reshape>(output_operand);
    SET_INPUT(output_reshape_op, x, output_operator);
    SET_INPUT(output_reshape_op, shape, output_shape_operator);
    MAP_OUTPUT(output_reshape_op, y, output_operand);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
