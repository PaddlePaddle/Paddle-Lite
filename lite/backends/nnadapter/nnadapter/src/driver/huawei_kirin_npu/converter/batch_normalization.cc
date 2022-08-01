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

#include "operation/batch_normalization.h"
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertBatchNormalization(Converter* converter,
                              core::Operation* operation) {
  BATCH_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto scale_operator = converter->ConvertOperand(scale_operand);
  auto offset_operator = converter->ConvertOperand(bias_operand);
  auto mean_operator = converter->ConvertOperand(mean_operand);
  auto variance_operator = converter->ConvertOperand(variance_operand);
  auto batch_norm_op =
      converter->AddOperator<hiai::op::BNInference>(output_operand);
  batch_norm_op->set_attr_epsilon(epsilon);
  SET_INPUT(batch_norm_op, x, input_operator);
  SET_INPUT(batch_norm_op, scale, scale_operator);
  SET_INPUT(batch_norm_op, offset, offset_operator);
  SET_INPUT(batch_norm_op, mean, mean_operator);
  SET_INPUT(batch_norm_op, variance, variance_operator);
  MAP_OUTPUT(batch_norm_op, y, output_operand);
  NNADAPTER_VLOG(5)
      << "Use Default param: momentum(0.9f) mode(1) use_global_stats(true)";
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
