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

#include "operation/split.h"
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertSplit(Converter* converter, core::Operation* operation) {
  SPLIT_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(IsConstantOperand(axis_operand));
  NNADAPTER_CHECK(IsConstantOperand(split_operand));

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto split_operator = converter->AddInt32ConstantOperator(split);
  auto axis_operator =
      converter->AddInt32ConstantOperator(std::vector<int32_t>({axis}));
  auto split_op = converter->AddOperator<hiai::op::SplitV>(output_operands[0]);
  int split_count = split.size();
  split_op->set_attr_num_split(split_count);
  SET_INPUT(split_op, x, input_operator);
  SET_INPUT(split_op, size_splits, split_operator);
  SET_INPUT(split_op, split_dim, axis_operator);
  split_op->create_dynamic_output_y(split_count);
  for (int i = 0; i < split_count; i++) {
    // Start from 1 for dynamic output in HiAI
    MAP_DYNAMIC_OUTPUT(split_op, y, i + 1, output_operands[i]);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
