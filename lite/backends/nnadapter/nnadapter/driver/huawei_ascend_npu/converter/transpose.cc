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

#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertTranspose(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  // Perm
  auto perm_operand = input_operands[1];
  auto perm_count = perm_operand->length / sizeof(int32_t);
  auto perm_data = reinterpret_cast<int32_t*>(perm_operand->buffer);
  for (uint32_t i = 0; i < perm_count; i++) {
    NNADAPTER_VLOG(5) << "perm[" << i << "]=" << perm_data[i];
  }
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }
  auto transpose_name = GetOperatorName(output_operand);
  auto transpose_op = std::make_shared<ge::op::Transpose>(transpose_name);
  auto perm_operator = AddInt32ConstantOperator(
      std::vector<int32_t>(perm_data, perm_data + perm_count));
  SET_INPUT(transpose_op, x, input_operator);
  SET_INPUT(transpose_op, perm, perm_operator);
  MAP_OUTPUT(transpose_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
