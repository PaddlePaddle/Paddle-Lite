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

#define QUANTIZE_OPERATION_EXTRACT_INPUTS_OUTPUTS                        \
  auto& input_operands = operation->input_operands;                      \
  auto& output_operands = operation->output_operands;                    \
  auto input_count = input_operands.size();                              \
  auto output_count = output_operands.size();                            \
  NNADAPTER_CHECK_EQ(input_count, 4);                                    \
  NNADAPTER_CHECK_EQ(output_count, 1);                                   \
  /* Input */                                                            \
  auto input_operand = input_operands[0];                                \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);      \
  /* Axis */                                                             \
  auto axis = *reinterpret_cast<int32_t*>(input_operands[1]->buffer);    \
  if (axis < 0) {                                                        \
    axis += static_cast<int32_t>(input_operand->type.dimensions.count);  \
  }                                                                      \
  NNADAPTER_VLOG(5) << "axis: " << axis;                                 \
  /* Scale */                                                            \
  auto scale_operand = input_operands[2];                                \
  uint32_t scale_count = scale_operand->length / sizeof(float);          \
  NNADAPTER_CHECK_GT(scale_count, 0U);                                   \
  float* scale_data = reinterpret_cast<float*>(scale_operand->buffer);   \
  bool is_per_layer_quant = scale_count == 1;                            \
  NNADAPTER_VLOG(5) << "scale_count: " << scale_count                    \
                    << ", scale_data[0]: " << scale_data[0]              \
                    << ", is_per_layer_quant: " << is_per_layer_quant;   \
  /* Zero_point */                                                       \
  auto zero_point_operand = input_operands[3];                           \
  uint32_t zero_point_count = scale_operand->length / sizeof(int32_t);   \
  int32_t* zero_point_data =                                             \
      reinterpret_cast<int32_t*>(zero_point_operand->buffer);            \
  bool is_symm_quant = zero_point_count == 1 && zero_point_data[0] == 0; \
  NNADAPTER_VLOG(5) << "zero_point_count: " << zero_point_count          \
                    << ", zero_point_data[0]: " << zero_point_data[0]    \
                    << ", is_symm_quant: " << is_symm_quant;             \
  /* Output */                                                           \
  auto output_operand = output_operands[0];                              \
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

}  // namespace operation
}  // namespace nnadapter
