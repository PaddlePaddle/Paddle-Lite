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

#include "operation/softplus.h"
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertSoftplus(Converter* converter, core::Operation* operation) {
  SOFTPLUS_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(fabs(beta - 1.0f) <= 1e-5f && fabs(threshold - 20.0) <= 1e-5f)
      << "Only supports beta = 1.0f and threshold = 20.0f!";

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto softplus_op =
      converter->AddOperator<hiai::op::Activation>(output_operand);
  softplus_op->set_attr_mode(9);
  SET_INPUT(softplus_op, x, input_operator);
  MAP_OUTPUT(softplus_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
