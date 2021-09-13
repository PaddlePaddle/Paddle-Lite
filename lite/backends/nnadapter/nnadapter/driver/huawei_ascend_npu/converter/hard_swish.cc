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

#include "core/operation/hard_swish.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertHardSwish(Converter* converter, hal::Operation* operation) {
  HARD_SWISH_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Alpha
  auto alpha = 1.0 / threshold;
  NNADAPTER_VLOG(5) << "alpha: " << alpha;
  // Beta
  auto beta = offset / threshold;
  NNADAPTER_VLOG(5) << "beta: " << beta;
  // Clip Min
  auto clip_min = 0.0;
  // Clip Max
  auto clip_max = 1.0;
  std::vector<int32_t> input_dimensions(
      input_operand->type.dimensions,
      input_operand->type.dimensions + input_operand->type.dimension_count);
  auto count = ProductionOfDimensions(input_operand->type.dimensions,
                                      input_operand->type.dimension_count);
  NNADAPTER_VLOG(5) << "count: " << count;

  // Convert to GE operators
  // y = mul(x, clip(0, scale(alpha*x + beta)), 1)
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  // Prepare data
  std::vector<float> alphas(count, alpha);
  std::vector<float> betas(count, beta);
  std::vector<float> clipmins(count, clip_min);
  std::vector<float> clipmaxs(count, clip_max);
  // Scale op
  auto scale_op =
      converter->AddOperator<ge::op::Scale>(output_operand, "scale");
  auto alpha_operator =
      converter->AddFloat32ConstantOperator(alphas, input_dimensions);
  auto beta_operator =
      converter->AddFloat32ConstantOperator(betas, input_dimensions);
  scale_op->set_attr_axis(0);
  scale_op->set_attr_num_axes(-1);
  scale_op->set_attr_scale_from_blob(true);
  SET_INPUT(scale_op, x, input_operator);
  SET_INPUT(scale_op, scale, alpha_operator);
  SET_INPUT(scale_op, bias, beta_operator);
  auto scale_operator = MAP_OUTPUT(scale_op, y, output_operand);
  // Clip op
  auto clip_op =
      converter->AddOperator<ge::op::ClipByValue>(output_operand, "clip");
  auto clip_min_operator =
      converter->AddFloat32ConstantOperator(clipmins, input_dimensions);
  auto clip_max_operator =
      converter->AddFloat32ConstantOperator(clipmaxs, input_dimensions);
  SET_INPUT(clip_op, x, scale_operator);
  SET_INPUT(clip_op, clip_value_min, clip_min_operator);
  SET_INPUT(clip_op, clip_value_max, clip_max_operator);
  auto clip_operator = MAP_OUTPUT(clip_op, y, output_operand);
  // Mul op
  auto mul_op = converter->AddOperator<ge::op::Mul>(output_operand);
  SET_INPUT(mul_op, x1, input_operator);
  SET_INPUT(mul_op, x2, clip_operator);
  MAP_OUTPUT(mul_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
