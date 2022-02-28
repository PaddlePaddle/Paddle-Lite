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

namespace nnadapter {
namespace operation {

#define LP_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS              \
  auto& input_operands = operation->input_operands;                    \
  auto& output_operands = operation->output_operands;                  \
  auto input_count = input_operands.size();                            \
  auto output_count = output_operands.size();                          \
  NNADAPTER_CHECK_EQ(input_count, 4);                                  \
  NNADAPTER_CHECK_EQ(output_count, 1);                                 \
  /* Input */                                                          \
  auto input_operand = input_operands[0];                              \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);    \
  /* Output */                                                         \
  auto output_operand = output_operands[0];                            \
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);  \
  /* Axis */                                                           \
  auto axis_operand = input_operands[1];                               \
  auto axis_count = axis_operand->length / sizeof(int32_t);            \
  auto axis_data = reinterpret_cast<int32_t*>(axis_operand->buffer);   \
  for (uint32_t i = 0; i < axis_count; i++) {                          \
    NNADAPTER_VLOG(5) << "axis[" << i << "]=" << axis_data[i];         \
  }                                                                    \
  /* P */                                                              \
  auto p = *reinterpret_cast<int32_t*>(input_operands[2]->buffer);     \
  NNADAPTER_VLOG(5) << "p: " << p;                                     \
  NNADAPTER_CHECK(p == 1 || p == 2)                                    \
      << "lp normalization only support p = 1 or p = 2.";              \
  /* Epsilon */                                                        \
  auto epsilon = *reinterpret_cast<float*>(input_operands[3]->buffer); \
  NNADAPTER_VLOG(5) << "epsilon: " << epsilon;

}  // namespace operation
}  // namespace nnadapter
