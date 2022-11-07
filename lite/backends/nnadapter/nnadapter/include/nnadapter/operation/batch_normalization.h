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

#define BATCH_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS                \
  auto& input_operands = operation->input_operands;                         \
  auto& output_operands = operation->output_operands;                       \
  auto input_count = input_operands.size();                                 \
  auto output_count = output_operands.size();                               \
  NNADAPTER_CHECK_EQ(input_count, 6);                                       \
  NNADAPTER_CHECK_EQ(output_count, 1);                                      \
  /* Input */                                                               \
  auto input_operand = input_operands[0];                                   \
  NNADAPTER_VLOG(5) << "input_operand: " << OperandToString(input_operand); \
  /* Scale */                                                               \
  auto scale_operand = input_operands[1];                                   \
  NNADAPTER_VLOG(5) << "scale_operand: " << OperandToString(scale_operand); \
  int32_t scale_count = scale_operand->length / sizeof(float);              \
  auto scale_data = reinterpret_cast<float*>(scale_operand->buffer);        \
  for (uint32_t i = 0; i < scale_count && i < 8; i++) {                     \
    NNADAPTER_VLOG(5) << "scale[" << i << "]=" << scale_data[i];            \
  }                                                                         \
  /* Bias */                                                                \
  auto bias_operand = input_operands[2];                                    \
  NNADAPTER_VLOG(5) << "bias_operand: " << OperandToString(bias_operand);   \
  int32_t bias_count = bias_operand->length / sizeof(float);                \
  NNADAPTER_CHECK_EQ(bias_count, scale_count);                              \
  auto bias_data = reinterpret_cast<float*>(bias_operand->buffer);          \
  for (uint32_t i = 0; i < bias_count && i < 8; i++) {                      \
    NNADAPTER_VLOG(5) << "bias[" << i << "]=" << bias_data[i];              \
  }                                                                         \
  /* Mean */                                                                \
  auto mean_operand = input_operands[3];                                    \
  NNADAPTER_VLOG(5) << "mean_operand: " << OperandToString(mean_operand);   \
  int32_t mean_count = mean_operand->length / sizeof(float);                \
  NNADAPTER_CHECK_EQ(mean_count, scale_count);                              \
  auto mean_data = reinterpret_cast<float*>(mean_operand->buffer);          \
  for (uint32_t i = 0; i < mean_count && i < 8; i++) {                      \
    NNADAPTER_VLOG(5) << "mean[" << i << "]=" << mean_data[i];              \
  }                                                                         \
  /* Variance */                                                            \
  auto variance_operand = input_operands[4];                                \
  NNADAPTER_VLOG(5) << "variance_operand: "                                 \
                    << OperandToString(variance_operand);                   \
  int32_t variance_count = variance_operand->length / sizeof(float);        \
  NNADAPTER_CHECK_EQ(variance_count, scale_count);                          \
  auto variance_data = reinterpret_cast<float*>(variance_operand->buffer);  \
  for (uint32_t i = 0; i < variance_count && i < 8; i++) {                  \
    NNADAPTER_VLOG(5) << "variance[" << i << "]=" << variance_data[i];      \
  }                                                                         \
  /* Epsilon */                                                             \
  auto epsilon = *reinterpret_cast<float*>(input_operands[5]->buffer);      \
  NNADAPTER_VLOG(5) << "epsilon :" << epsilon;                              \
  /* Output */                                                              \
  auto output_operand = output_operands[0];                                 \
  NNADAPTER_VLOG(5) << "output_operand: " << OperandToString(output_operand);

}  // namespace operation
}  // namespace nnadapter
