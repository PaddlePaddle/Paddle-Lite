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

#define REDUCE_OPERATION_EXTRACT_INPUTS_OUTPUTS                             \
  auto& input_operands = operation->input_operands;                         \
  auto& output_operands = operation->output_operands;                       \
  auto input_count = input_operands.size();                                 \
  auto output_count = output_operands.size();                               \
  NNADAPTER_CHECK_EQ(input_count, 3);                                       \
  NNADAPTER_CHECK_EQ(output_count, 1);                                      \
  /* Input */                                                               \
  auto input_operand = input_operands[0];                                   \
  auto input_dimension_count = input_operand->type.dimensions.count;        \
  NNADAPTER_VLOG(5) << "input_operand: " << OperandToString(input_operand); \
  /* Axes */                                                                \
  auto axes_operand = input_operands[1];                                    \
  int axes_size = axes_operand->length / sizeof(int32_t);                   \
  auto axes_data = reinterpret_cast<int32_t*>(axes_operand->buffer);        \
  for (int i = 0; i < axes_size; i++) {                                     \
    axes_data[i] = axes_data[i] < 0 ? axes_data[i] + input_dimension_count  \
                                    : axes_data[i];                         \
    NNADAPTER_VLOG(5) << "axes[" << i << "]: " << axes_data[i];             \
  }                                                                         \
  /* Keep_dim */                                                            \
  auto keep_dim_operand = input_operands[2];                                \
  auto keep_dims = *reinterpret_cast<int8_t*>(keep_dim_operand->buffer);    \
  NNADAPTER_VLOG(5) << "keep_dims: " << keep_dims;                          \
  bool keep_dim = keep_dims ? true : false;                                 \
  /* Output */                                                              \
  auto output_operand = output_operands[0];                                 \
  NNADAPTER_VLOG(5) << "output_operand: " << OperandToString(output_operand);

}  // namespace operation
}  // namespace nnadapter
