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

#define FULLY_CONNECTED_OPERATION_EXTRACT_INPUTS_OUTPUTS                   \
  auto& input_operands = operation->input_operands;                        \
  auto& output_operands = operation->output_operands;                      \
  auto input_count = input_operands.size();                                \
  auto output_count = output_operands.size();                              \
  NNADAPTER_CHECK_EQ(input_count, 4);                                      \
  NNADAPTER_CHECK_EQ(output_count, 1);                                     \
  /* Input */                                                              \
  auto input_operand = input_operands[0];                                  \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);        \
  /* Weight */                                                             \
  auto weight_operand = input_operands[1];                                 \
  NNADAPTER_VLOG(5) << "weight: " << OperandToString(weight_operand);      \
  NNADAPTER_CHECK_EQ(weight_operand->type.dimensions.count, 2);            \
  auto num_units = weight_operand->type.dimensions.data[0];                \
  NNADAPTER_VLOG(5) << "num_units: " << num_units;                         \
  auto input_size = weight_operand->type.dimensions.data[1];               \
  NNADAPTER_VLOG(5) << "input_size: " << input_size;                       \
  /* Bias */                                                               \
  auto bias_operand = input_operands[2];                                   \
  NNADAPTER_VLOG(5) << "bias: " << OperandToString(bias_operand);          \
  NNADAPTER_CHECK_EQ(bias_operand->type.dimensions.count, 1);              \
  NNADAPTER_CHECK_EQ(num_units, bias_operand->type.dimensions.data[0]);    \
  /* Fuse code */                                                          \
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[3]->buffer); \
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;                          \
  /* Output */                                                             \
  auto output_operand = output_operands[0];                                \
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

}  // namespace operation
}  // namespace nnadapter
