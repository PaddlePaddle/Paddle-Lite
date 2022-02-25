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

#include "driver/huawei_ascend_npu/converter/converter.h"
#include "operation/hard_sigmoid_swish.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertHardSigmoid(Converter* converter, core::Operation* operation) {
  HARD_SIGMOID_SWISH_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto hard_sigmoid_op =
      converter->AddOperator<ge::op::HardSigmoid>(output_operand);
  hard_sigmoid_op->set_attr_alpha(alpha);
  hard_sigmoid_op->set_attr_beta(beta);
  SET_INPUT(hard_sigmoid_op, input_x, input_operator);
  MAP_OUTPUT(hard_sigmoid_op, output_y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
