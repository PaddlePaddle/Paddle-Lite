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

#pragma once

#include <vector>
#include "utility/modeling.h"

namespace nnadapter {
namespace operation {

#define SPLIT_OPERATION_EXTRACT_INPUTS_OUTPUTS                           \
  auto& input_operands = operation->input_operands;                      \
  auto& output_operands = operation->output_operands;                    \
  auto input_count = input_operands.size();                              \
  auto output_count = output_operands.size();                            \
  NNADAPTER_CHECK_EQ(input_count, 3);                                    \
  NNADAPTER_CHECK_GE(output_count, 1);                                   \
  /* Input */                                                            \
  auto input_operand = input_operands[0];                                \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);      \
  /* Axis */                                                             \
  auto axis_operand = input_operands[1];                                 \
  int axis = -1;                                                         \
  if (IsConstantOperand(axis_operand)) {                                 \
    axis = *reinterpret_cast<int32_t*>(axis_operand->buffer);            \
    if (axis < 0) {                                                      \
      axis += input_operand->type.dimensions.count;                      \
    }                                                                    \
    NNADAPTER_VLOG(5) << "axis: " << axis;                               \
  } else {                                                               \
    NNADAPTER_VLOG(5) << "axis: " << OperandToString(axis_operand);      \
  }                                                                      \
  NNADAPTER_CHECK_LT(axis, input_operand->type.dimensions.count);        \
  /* Split */                                                            \
  auto split_operand = input_operands[2];                                \
  std::vector<int> split;                                                \
  if (IsConstantOperand(split_operand)) {                                \
    auto split_count = split_operand->length / sizeof(int32_t);          \
    auto split_data = reinterpret_cast<int32_t*>(split_operand->buffer); \
    split = std::vector<int>(split_data, split_data + split_count);      \
    NNADAPTER_CHECK_EQ(split_count, output_count);                       \
    for (uint32_t i = 0; i < split_count; i++) {                         \
      NNADAPTER_VLOG(5) << "split[" << i << "]: " << split_data[i];      \
    }                                                                    \
  } else {                                                               \
    NNADAPTER_VLOG(5) << "split: " << OperandToString(split_operand);    \
  }                                                                      \
  /* Output */                                                           \
  for (size_t i = 0; i < output_count; i++) {                            \
    NNADAPTER_VLOG(5) << "output" << i << ": "                           \
                      << OperandToString(output_operands[i]);            \
  }

}  // namespace operation
}  // namespace nnadapter
