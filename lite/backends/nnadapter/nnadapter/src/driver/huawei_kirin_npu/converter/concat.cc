// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/concat.h"
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertConcat(Converter* converter, core::Operation* operation) {
  CONCAT_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto N = input_count - 1;
  auto concat_op = converter->AddOperator<hiai::op::ConcatD>(output_operand);
  concat_op->set_attr_concat_dim(axis);
  concat_op->set_attr_N(N);
  concat_op->create_dynamic_input_x(N);
  for (int i = 0; i < N; i++) {
    auto input_operand = input_operands[i];
    auto input_operator = converter->GetMappedOperator(input_operand);
    if (!input_operator) {
      input_operator = converter->ConvertOperand(input_operand);
    }
    // Start from 1 for dynamic input in HiAI
    SET_DYNAMIC_INPUT(concat_op, x, i + 1, input_operator);
  }
  MAP_OUTPUT(concat_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
