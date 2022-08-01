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

#define RESIZE_NEAREST_OPERATION_EXTRACT_INPUTS_OUTPUTS                       \
  auto& input_operands = operation->input_operands;                           \
  auto& output_operands = operation->output_operands;                         \
  auto input_count = input_operands.size();                                   \
  auto output_count = output_operands.size();                                 \
  NNADAPTER_CHECK_EQ(input_count, 4);                                         \
  NNADAPTER_CHECK_EQ(output_count, 1);                                        \
  /* Input */                                                                 \
  auto input_operand = input_operands[0];                                     \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);           \
  /* Shape */                                                                 \
  auto shape_operand = input_operands[1];                                     \
  if (shape_operand == nullptr) {                                             \
    NNADAPTER_VLOG(5) << "Shape is null, please use scales.";                 \
  } else {                                                                    \
    NNADAPTER_VLOG(5) << "shape: " << OperandToString(shape_operand);         \
  }                                                                           \
  /* Scales */                                                                \
  auto scales_operand = input_operands[2];                                    \
  if (scales_operand == nullptr) {                                            \
    NNADAPTER_VLOG(5) << "Scales is null, please use shape.";                 \
  } else {                                                                    \
    NNADAPTER_VLOG(5) << "scales: " << OperandToString(scales_operand);       \
  }                                                                           \
  NNADAPTER_CHECK(shape_operand != nullptr || scales_operand != nullptr)      \
      << "shape_operand and scales_operand should not both be null.";         \
  /* Align_corners */                                                         \
  bool align_corners = reinterpret_cast<bool*>(input_operands[3]->buffer)[0]; \
  NNADAPTER_VLOG(5) << "align_corners: " << align_corners;                    \
  /* Output */                                                                \
  auto* output_operand = output_operands[0];                                  \
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

}  // namespace operation
}  // namespace nnadapter
