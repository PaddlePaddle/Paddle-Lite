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
#include "operation/unary_activations.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertSwish(Converter* converter, core::Operation* operation) {
  UNARY_ACTIVATIONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  // output = input * sigmoid(input)
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto sigmoid_op = converter->AddOperator<ge::op::Sigmoid>(output_operand);
  SET_INPUT(sigmoid_op, x, input_operator);
  auto sigmoid_operator = MAP_OUTPUT(sigmoid_op, y, output_operand);
  auto eltwise_op = converter->AddOperator<ge::op::Mul>(output_operand);
  SET_INPUT(eltwise_op, x1, input_operator);
  SET_INPUT(eltwise_op, x2, sigmoid_operator);
  MAP_OUTPUT(eltwise_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
