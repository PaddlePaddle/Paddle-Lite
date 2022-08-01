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

#include "operation/instance_normalization.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertInstanceNormalization(Converter* converter,
                                 core::Operation* operation) {
  INSTANCE_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

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

  // Betas and gammas
  std::vector<float> betas(input_operand->type.dimensions.data[2] *
                               input_operand->type.dimensions.data[3],
                           0);
  std::vector<float> gammas(input_operand->type.dimensions.data[2] *
                                input_operand->type.dimensions.data[3],
                            1);
  std::vector<int32_t> input_dimensions(
      input_operand->type.dimensions.data + 2,
      input_operand->type.dimensions.data +
          input_operand->type.dimensions.count);
  auto dummy_beta_operator =
      converter->AddFloat32ConstantOperator(betas, input_dimensions);
  auto dummy_gamma_operator =
      converter->AddFloat32ConstantOperator(gammas, input_dimensions);
  // Use layer norm instead of instance norm
  auto layer_norm_op =
      converter->AddOperator<ge::op::LayerNorm>(output_operand);
  layer_norm_op->set_attr_epsilon(epsilon);
  layer_norm_op->set_attr_begin_norm_axis(2);
  layer_norm_op->set_attr_begin_params_axis(2);
  SET_INPUT(layer_norm_op, x, input_operator);
  SET_INPUT(layer_norm_op, beta, dummy_beta_operator);
  SET_INPUT(layer_norm_op, gamma, dummy_gamma_operator);
  MAP_OUTPUT(layer_norm_op, y, output_operand);
  auto layer_norm_operator = MAP_OUTPUT(layer_norm_op, y, output_operand);
  // Use eltwise_mul op to process scale
  auto mul_op = converter->AddOperator<ge::op::Mul>(output_operand);
  SET_INPUT(mul_op, x1, layer_norm_operator);
  SET_INPUT(mul_op, x2, scale_operator);
  auto mul_operator = MAP_OUTPUT(mul_op, y, output_operand);
  // Use eltwise_add op to process bias
  auto add_op = converter->AddOperator<ge::op::Add>(output_operand);
  SET_INPUT(add_op, x1, mul_operator);
  SET_INPUT(add_op, x2, bias_operator);
  MAP_OUTPUT(add_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
