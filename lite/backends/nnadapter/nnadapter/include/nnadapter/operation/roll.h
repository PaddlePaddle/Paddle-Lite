// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

namespace nnadapter {
namespace operation {

#define ROLL_OPERATION_EXTRACT_INPUTS_OUTPUTS                               \
  auto& input_operands = operation->input_operands;                         \
  auto& output_operands = operation->output_operands;                       \
  auto input_count = input_operands.size();                                 \
  auto output_count = output_operands.size();                               \
  NNADAPTER_CHECK_EQ(input_count, 3);                                       \
  NNADAPTER_CHECK_EQ(output_count, 1);                                      \
  /* Input */                                                               \
  auto input_operand = input_operands[0];                                   \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);         \
  /* Shifts */                                                              \
  auto shifts_operand = input_operands[1];                                  \
  std::vector<int32_t> shifts;                                              \
  if (IsConstantOperand(shifts_operand)) {                                  \
    auto shifts_count = shifts_operand->length / sizeof(int32_t);           \
    auto shifts_data = reinterpret_cast<int32_t*>(shifts_operand->buffer);  \
    shifts = std::vector<int32_t>(shifts_data, shifts_data + shifts_count); \
    for (size_t i = 0; i < shifts.size(); i++) {                            \
      NNADAPTER_VLOG(5) << "shifts[" << i << "]: " << shifts[i];            \
    }                                                                       \
  } else {                                                                  \
    NNADAPTER_VLOG(5) << "shifts: " << OperandToString(shifts_operand);     \
  }                                                                         \
  /* Axes */                                                                \
  auto axes_operand = input_operands[2];                                    \
  std::vector<int32_t> axes;                                                \
  if (IsConstantOperand(axes_operand)) {                                    \
    auto axes_count = axes_operand->length / sizeof(int32_t);               \
    auto axes_data = reinterpret_cast<int32_t*>(axes_operand->buffer);      \
    axes = std::vector<int32_t>(axes_data, axes_data + axes_count);         \
    for (size_t i = 0; i < axes.size(); i++) {                              \
      NNADAPTER_VLOG(5) << "axes[" << i << "]: " << axes[i];                \
    }                                                                       \
  } else {                                                                  \
    NNADAPTER_VLOG(5) << "axes: " << OperandToString(axes_operand);         \
  }                                                                         \
  /* Output */                                                              \
  auto output_operand = output_operands[0];                                 \
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

}  // namespace operation
}  // namespace nnadapter
