// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <vector>
#include "utility/modeling.h"

namespace nnadapter {
namespace operation {

#define UNSTACK_OPERATION_EXTRACT_INPUTS_OUTPUTS                    \
  auto& input_operands = operation->input_operands;                 \
  auto& output_operands = operation->output_operands;               \
  auto input_count = input_operands.size();                         \
  auto output_count = output_operands.size();                       \
  NNADAPTER_CHECK_EQ(input_count, 3);                               \
  NNADAPTER_CHECK_GE(output_count, 1);                              \
  /* Input */                                                       \
  auto input_operand = input_operands[0];                           \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand); \
  /* Axis */                                                        \
  auto axis_operand = input_operands[1];                            \
  NNADAPTER_CHECK(IsConstantOperand(axis_operand));                 \
  auto axis = *reinterpret_cast<int32_t*>(axis_operand->buffer);    \
  if (axis < 0) {                                                   \
    axis += input_operand->type.dimensions.count;                   \
  }                                                                 \
  NNADAPTER_VLOG(5) << "axis: " << axis;                            \
  NNADAPTER_CHECK_GE(axis, 0);                                      \
  NNADAPTER_CHECK_LT(axis, input_operand->type.dimensions.count);   \
  /* Num */                                                         \
  auto num_operand = input_operands[2];                             \
  NNADAPTER_CHECK(IsConstantOperand(num_operand));                  \
  auto num = *reinterpret_cast<int32_t*>(num_operand->buffer);      \
  NNADAPTER_VLOG(5) << "num: " << num;                              \
  NNADAPTER_CHECK_EQ(num, output_count);                            \
  /* Output */                                                      \
  for (size_t i = 0; i < output_count; i++) {                       \
    NNADAPTER_VLOG(5) << "output" << i << ": "                      \
                      << OperandToString(output_operands[i]);       \
  }

}  // namespace operation
}  // namespace nnadapter
