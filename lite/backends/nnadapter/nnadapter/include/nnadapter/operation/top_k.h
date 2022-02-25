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

#include "utility/modeling.h"

namespace nnadapter {
namespace operation {

#define TOP_K_OPERATION_EXTRACT_INPUTS_OUTPUTS                           \
  auto& input_operands = operation->input_operands;                      \
  auto& output_operands = operation->output_operands;                    \
  auto input_count = input_operands.size();                              \
  auto output_count = output_operands.size();                            \
  NNADAPTER_CHECK_EQ(input_count, 6);                                    \
  NNADAPTER_CHECK_EQ(output_count, 2);                                   \
  /* Input */                                                            \
  auto input_operand = input_operands[0];                                \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);      \
  /* K */                                                                \
  auto k_operand = input_operands[1];                                    \
  NNADAPTER_VLOG(5) << "k: " << OperandToString(k_operand);              \
  int64_t k = NNADAPTER_UNKNOWN;                                         \
  if (IsConstantOperand(k_operand)) {                                    \
    auto k_precision = k_operand->type.precision;                        \
    auto k_buffer = k_operand->buffer;                                   \
    if (k_precision == NNADAPTER_INT32) {                                \
      k = *reinterpret_cast<int32_t*>(k_buffer);                         \
    } else if (k_precision == NNADAPTER_INT64) {                         \
      k = *reinterpret_cast<int64_t*>(k_buffer);                         \
    } else {                                                             \
      NNADAPTER_LOG(FATAL) << "Unsupported the precision type:"          \
                           << static_cast<int>(k_precision);             \
    }                                                                    \
  }                                                                      \
  NNADAPTER_VLOG(5) << "k: " << k;                                       \
  /* Axis */                                                             \
  auto axis = *reinterpret_cast<int32_t*>(input_operands[2]->buffer);    \
  NNADAPTER_VLOG(5) << "axis: " << axis;                                 \
  if (axis < 0) {                                                        \
    axis += input_operand->type.dimensions.count;                        \
  }                                                                      \
  /* Largest */                                                          \
  bool largest = *reinterpret_cast<int8_t*>(input_operands[3]->buffer);  \
  NNADAPTER_VLOG(5) << "largest: " << largest;                           \
  /* Sorted */                                                           \
  bool sorted = *reinterpret_cast<int8_t*>(input_operands[4]->buffer);   \
  NNADAPTER_VLOG(5) << "sorted: " << sorted;                             \
  /* ReturnIndicesDtype */                                               \
  auto return_indices_dtype =                                            \
      *reinterpret_cast<int32_t*>(input_operands[5]->buffer);            \
  NNADAPTER_VLOG(5) << "return_indices_dtype: " << return_indices_dtype; \
  /* Output */                                                           \
  auto output_operand = output_operands[0];                              \
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);    \
  /* Indices */                                                          \
  auto indices_operand = output_operands[1];                             \
  NNADAPTER_VLOG(5) << "indices: " << OperandToString(indices_operand);

}  // namespace operation
}  // namespace nnadapter
