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

void Split(Converter* converter,
           std::shared_ptr<Operator> input_operator,
           core::Operand* output_operand,
           const int32_t split_num,
           const std::string name,
           std::vector<std::shared_ptr<Operator>>* split_outs) {
  auto axis_operator =
      converter->AddInt32ConstantOperator(std::vector<int32_t>({1}));
  auto split_op = converter->AddOperator<ge::op::Split>(output_operand, name);
  split_op->set_attr_num_split(split_num);
  split_op->create_dynamic_output_y(split_num);
  SET_INPUT(split_op, x, input_operator);
  SET_INPUT(split_op, split_dim, axis_operator);
  for (uint32_t i = 0; i < split_num; i++) {
    split_outs->push_back(MAP_DYNAMIC_OUTPUT(split_op, y, i, output_operand));
  }
}

int ConvertGroupNormalization(Converter* converter,
                              core::Operation* operation) {
  GROUP_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (input_operator == nullptr) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto scale_operator = converter->ConvertOperand(
      scale_operand,
      std::vector<int32_t>({1, scale_operand->type.dimensions.data[0], 1, 1}));
  auto bias_operator = converter->ConvertOperand(
      bias_operand,
      std::vector<int32_t>({1, bias_operand->type.dimensions.data[0], 1, 1}));
  // Split depends on groups
  int32_t channel_axis = 1;
  auto input_channel = input_operand->type.dimensions.data[channel_axis];
  NNADAPTER_CHECK_GT(input_channel, 0);
  NNADAPTER_CHECK_EQ(input_channel % groups, 0);
  int32_t split_num = groups;
  std::vector<std::shared_ptr<Operator>> split_input_outs;
  std::vector<std::shared_ptr<Operator>> split_scale_outs;
  std::vector<std::shared_ptr<Operator>> split_bias_outs;
  if (groups > 1) {
    Split(converter,
          input_operator,
          output_operand,
          split_num,
          "split_input",
          &split_input_outs);
    Split(converter,
          scale_operator,
          output_operand,
          split_num,
          "split_scale",
          &split_scale_outs);
    Split(converter,
          bias_operator,
          output_operand,
          split_num,
          "split_bias",
          &split_bias_outs);
  } else {
    split_input_outs.push_back(input_operator);
    split_scale_outs.push_back(scale_operator);
    split_bias_outs.push_back(bias_operator);
  }
  // Use layer normalization
  std::vector<float> gammas(input_operand->type.dimensions.data[1] *
                                input_operand->type.dimensions.data[2] *
                                input_operand->type.dimensions.data[3] / groups,
                            1);
  std::vector<float> betas(input_operand->type.dimensions.data[1] *
                               input_operand->type.dimensions.data[2] *
                               input_operand->type.dimensions.data[3] / groups,
                           0);
  std::vector<int32_t> input_dimensions(
      {input_operand->type.dimensions.data[1] / groups,
       input_operand->type.dimensions.data[2],
       input_operand->type.dimensions.data[3]});
  auto dummy_gamma_operator =
      converter->AddFloat32ConstantOperator(gammas, input_dimensions);
  auto dummy_beta_operator =
      converter->AddFloat32ConstantOperator(betas, input_dimensions);
  std::vector<std::shared_ptr<Operator>> layer_norm_outs;
  for (uint32_t i = 0; i < split_num; i++) {
    auto layer_norm_op = converter->AddOperator<ge::op::LayerNorm>(
        output_operand, "layer_norm_" + std::to_string(i));
    layer_norm_op->set_attr_epsilon(epsilon);
    layer_norm_op->set_attr_begin_norm_axis(channel_axis);
    layer_norm_op->set_attr_begin_params_axis(channel_axis);
    SET_INPUT(layer_norm_op, x, split_input_outs[i]);
    SET_INPUT(layer_norm_op, beta, dummy_beta_operator);
    SET_INPUT(layer_norm_op, gamma, dummy_gamma_operator);
    auto layer_norm_operator = MAP_OUTPUT(layer_norm_op, y, output_operand);
    // Use eltwise_mul op to process scale
    auto mul_op = converter->AddOperator<ge::op::Mul>(output_operand);
    SET_INPUT(mul_op, x1, layer_norm_operator);
    SET_INPUT(mul_op, x2, split_scale_outs[i]);
    auto mul_operator = MAP_OUTPUT(mul_op, y, output_operand);
    // Use eltwise_add op to process bias
    auto add_op = converter->AddOperator<ge::op::Add>(output_operand);
    SET_INPUT(add_op, x1, mul_operator);
    SET_INPUT(add_op, x2, split_bias_outs[i]);
    layer_norm_outs.push_back(MAP_OUTPUT(add_op, y, output_operand));
  }
  if (layer_norm_outs.size() > 1) {
    // Concat
    auto concat_op = converter->AddOperator<ge::op::ConcatD>(output_operand);
    concat_op->set_attr_concat_dim(channel_axis);
    concat_op->set_attr_N(split_num);
    concat_op->create_dynamic_input_x(split_num);
    for (uint32_t i = 0; i < split_num; i++) {
      SET_DYNAMIC_INPUT(concat_op, x, i, layer_norm_outs[i]);
    }
    MAP_OUTPUT(concat_op, y, output_operand);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
