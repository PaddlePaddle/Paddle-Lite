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

#define RANGE_OPERATION_EXTRACT_INPUTS_OUTPUTS                              \
  auto& input_operands = operation->input_operands;                         \
  auto& output_operands = operation->output_operands;                       \
  auto input_count = input_operands.size();                                 \
  auto output_count = output_operands.size();                               \
  NNADAPTER_CHECK_EQ(input_count, 3);                                       \
  NNADAPTER_CHECK_EQ(output_count, 1);                                      \
  /* Start */                                                               \
  auto start_operand = input_operands[0];                                   \
  NNADAPTER_VLOG(5) << "start_operand: " << OperandToString(start_operand); \
  /* Limit */                                                               \
  auto limit_operand = input_operands[1];                                   \
  NNADAPTER_VLOG(5) << "limit_operand: " << OperandToString(limit_operand); \
  /* Delta */                                                               \
  auto delta_operand = input_operands[2];                                   \
  NNADAPTER_VLOG(5) << "delta_operand: " << OperandToString(delta_operand); \
  /* Output */                                                              \
  auto output_operand = output_operands[0];                                 \
  NNADAPTER_VLOG(5) << "output_operand: " << OperandToString(output_operand);

}  // namespace operation
}  // namespace nnadapter
