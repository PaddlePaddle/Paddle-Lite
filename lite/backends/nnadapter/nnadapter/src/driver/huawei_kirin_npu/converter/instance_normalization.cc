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
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertInstanceNormalization(Converter* converter,
                                 core::Operation* operation) {
  INSTANCE_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto scale_operator = converter->ConvertOperand(scale_operand);
  auto bias_operator = converter->ConvertOperand(bias_operand);
  auto instance_norm_op =
      converter->AddOperator<hiai::op::InstanceNorm>(output_operand);
  instance_norm_op->set_attr_epsilon(epsilon);
  SET_INPUT(instance_norm_op, x, input_operator);
  SET_INPUT(instance_norm_op, gamma, scale_operator);
  SET_INPUT(instance_norm_op, beta, bias_operator);
  MAP_OUTPUT(instance_norm_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
