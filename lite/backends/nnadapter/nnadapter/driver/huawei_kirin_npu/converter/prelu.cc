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

#include "core/operation/prelu.h"
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertPRelu(Converter* converter, hal::Operation* operation) {
  PRELU_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto slope_operator = converter->GetMappedOperator(slope_operand);
  if (!slope_operator) {
    slope_operator = converter->ConvertOperand(slope_operand);
  }
  auto prelu_op = converter->AddOperator<hiai::op::PRelu>(output_operand);
  SET_INPUT(prelu_op, x, input_operator);
  SET_INPUT(prelu_op, weight, slope_operator);
  MAP_OUTPUT(prelu_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
