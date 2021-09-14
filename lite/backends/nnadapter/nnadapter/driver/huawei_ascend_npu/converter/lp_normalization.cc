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

#include "core/operation/lp_normalization.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertLpNormalization(Converter* converter, hal::Operation* operation) {
  LP_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  ge::Operator::OpListInt axis;
  for (uint32_t i = 0; i < axis_count; i++) {
    axis.push_back(axis_data[i]);
  }
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  if (p == 2 && keepdim) {
    auto l2_norm_op =
        converter->AddOperator<ge::op::L2Normalize>(output_operand);
    l2_norm_op->set_attr_axis(axis);
    l2_norm_op->set_attr_eps(epsilon);
    SET_INPUT(l2_norm_op, x, input_operator);
    MAP_OUTPUT(l2_norm_op, y, output_operand);
  } else if (p == INT_MAX || p == INT_MIN || p == 0) {
    auto p_norm_op = converter->AddOperator<ge::op::LpNorm>(output_operand);
    p_norm_op->set_attr_p(p);
    p_norm_op->set_attr_axes(axis);
    p_norm_op->set_attr_epsilon(epsilon);
    p_norm_op->set_attr_keepdim(keepdim);
    SET_INPUT(p_norm_op, x, input_operator);
    MAP_OUTPUT(p_norm_op, y, output_operand);
  } else {
    // pow(input, p)
    auto power_op =
        converter->AddOperator<ge::op::Power>(output_operand, "power");
    power_op->set_attr_power(static_cast<float>(p));
    power_op->set_attr_scale(1.0f);
    power_op->set_attr_shift(0.0f);
    SET_INPUT(power_op, x, input_operator);
    auto power_operator = MAP_OUTPUT(power_op, y, output_operand);
    // reduce_sum(pow(input, p))
    auto reduce_sum_op = converter->AddOperator<ge::op::ReduceSumD>(
        output_operand, "reduce_sum");
    reduce_sum_op->set_attr_axes(axis);
    reduce_sum_op->set_attr_keep_dims(keepdim);
    SET_INPUT(reduce_sum_op, x, power_operator);
    auto reduce_sum_operator = MAP_OUTPUT(reduce_sum_op, y, output_operand);
    // pow(reduce_sum(pow(input, p)), 1/p)
    auto root_op =
        converter->AddOperator<ge::op::Power>(output_operand, "root");
    root_op->set_attr_power(1.0f / p);
    root_op->set_attr_scale(1.0f);
    root_op->set_attr_shift(0.0f);
    SET_INPUT(root_op, x, reduce_sum_operator);
    MAP_OUTPUT(root_op, y, output_operand);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
