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

#define SLICE_OPERATION_EXTRACT_INPUTS_OUTPUTS                              \
  auto& input_operands = operation->input_operands;                         \
  auto& output_operands = operation->output_operands;                       \
  auto input_count = input_operands.size();                                 \
  auto output_count = output_operands.size();                               \
  NNADAPTER_CHECK_EQ(input_count, 5);                                       \
  NNADAPTER_CHECK_EQ(output_count, 1);                                      \
  /* Input */                                                               \
  auto input_operand = input_operands[0];                                   \
  NNADAPTER_VLOG(5) << "input_operand: " << OperandToString(input_operand); \
  /* Axes */                                                                \
  auto axes_operand = input_operands[1];                                    \
  auto axes_count =                                                         \
      axes_operand->length / static_cast<uint32_t>(sizeof(int32_t));        \
  auto axes = reinterpret_cast<int32_t*>(axes_operand->buffer);             \
  for (uint32_t i = 0; i < axes_count; i++) {                               \
    NNADAPTER_VLOG(5) << "axes[" << i << "] = " << axes[i];                 \
  }                                                                         \
  /* Starts */                                                              \
  auto starts_operand = input_operands[2];                                  \
  std::vector<int32_t> starts;                                              \
  if (IsConstantOperand(starts_operand)) {                                  \
    auto starts_count = starts_operand->length / sizeof(int32_t);           \
    auto starts_data = reinterpret_cast<int32_t*>(starts_operand->buffer);  \
    starts = std::vector<int32_t>(starts_data, starts_data + starts_count); \
    for (size_t i = 0; i < starts.size(); i++) {                            \
      NNADAPTER_VLOG(5) << "starts[" << i << "]: " << starts[i];            \
    }                                                                       \
  } else {                                                                  \
    NNADAPTER_VLOG(5) << "starts: " << OperandToString(starts_operand);     \
  }                                                                         \
  /* Ends */                                                                \
  auto ends_operand = input_operands[3];                                    \
  std::vector<int32_t> ends;                                                \
  if (IsConstantOperand(ends_operand)) {                                    \
    auto ends_count = ends_operand->length / sizeof(int32_t);               \
    auto ends_data = reinterpret_cast<int32_t*>(ends_operand->buffer);      \
    ends = std::vector<int32_t>(ends_data, ends_data + ends_count);         \
    for (size_t i = 0; i < ends.size(); i++) {                              \
      NNADAPTER_VLOG(5) << "ends[" << i << "]: " << ends[i];                \
    }                                                                       \
  } else {                                                                  \
    NNADAPTER_VLOG(5) << "ends: " << OperandToString(ends_operand);         \
  }                                                                         \
  /* Steps */                                                               \
  auto steps_operand = input_operands[4];                                   \
  std::vector<int32_t> steps;                                               \
  if (IsConstantOperand(steps_operand)) {                                   \
    auto steps_count = steps_operand->length / sizeof(int32_t);             \
    auto steps_data = reinterpret_cast<int32_t*>(steps_operand->buffer);    \
    steps = std::vector<int32_t>(steps_data, steps_data + steps_count);     \
    for (size_t i = 0; i < steps.size(); i++) {                             \
      NNADAPTER_VLOG(5) << "steps[" << i << "]: " << steps[i];              \
    }                                                                       \
  } else {                                                                  \
    NNADAPTER_VLOG(5) << "steps: " << OperandToString(steps_operand);       \
  }                                                                         \
  /* Output */                                                              \
  auto output_operand = output_operands[0];                                 \
  NNADAPTER_VLOG(5) << "output_operand: " << OperandToString(output_operand);

}  // namespace operation
}  // namespace nnadapter
