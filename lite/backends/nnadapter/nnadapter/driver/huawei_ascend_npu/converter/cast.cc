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

#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertCast(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 2);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);

  // Dtype
  NNAdapterOperandPrecisionCode dtype =
      *reinterpret_cast<NNAdapterOperandPrecisionCode*>(
          input_operands[1]->buffer);
  NNADAPTER_VLOG(5) << "dtype: " << dtype;
  ge::DataType otype = ConvertPrecision(dtype);

  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }
  auto cast_name = GetOperatorName(output_operand);
  auto cast_op = std::make_shared<ge::op::Cast>(cast_name);
  cast_op->set_attr_dst_type(otype);
  SET_INPUT(cast_op, x, input_operator);
  MAP_OUTPUT(cast_op, y, output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
