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

#define PRIOR_BOX_OPERATION_EXTRACT_INPUTS_OUTPUTS                           \
  auto& input_operands = operation->input_operands;                          \
  auto& output_operands = operation->output_operands;                        \
  auto input_count = input_operands.size();                                  \
  auto output_count = output_operands.size();                                \
  NNADAPTER_CHECK_EQ(input_count, 12);                                       \
  NNADAPTER_CHECK_EQ(output_count, 2);                                       \
  /* Input */                                                                \
  auto input_operand = input_operands[0];                                    \
  NNADAPTER_VLOG(5) << "Input: " << OperandToString(input_operand);          \
  /* Image */                                                                \
  auto image_operand = input_operands[1];                                    \
  NNADAPTER_VLOG(5) << "Image: " << OperandToString(image_operand);          \
  /* min_sizes */                                                            \
  auto min_sizes_operand = input_operands[2];                                \
  NNADAPTER_VLOG(5) << "min_sizes: " << OperandToString(min_sizes_operand);  \
  /* max_sizes */                                                            \
  auto max_sizes_operand = input_operands[3];                                \
  NNADAPTER_VLOG(5) << "max_sizes: " << OperandToString(max_sizes_operand);  \
  /* max_sizes */                                                            \
  auto aspect_ratios_operand = input_operands[4];                            \
  NNADAPTER_VLOG(5) << "aspect_ratios: "                                     \
                    << OperandToString(aspect_ratios_operand);               \
  /* variances */                                                            \
  auto variances_operand = input_operands[5];                                \
  NNADAPTER_VLOG(5) << "variances: " << OperandToString(variances_operand);  \
  /* flip */                                                                 \
  auto flip_operand = input_operands[6];                                     \
  NNADAPTER_VLOG(5) << "flip: " << OperandToString(flip_operand);            \
  /* clip */                                                                 \
  auto clip_operand = input_operands[7];                                     \
  NNADAPTER_VLOG(5) << "clip: " << OperandToString(clip_operand);            \
  /* step_w */                                                               \
  auto step_w_operand = input_operands[8];                                   \
  NNADAPTER_VLOG(5) << "step_w: " << OperandToString(step_w_operand);        \
  /* step_h */                                                               \
  auto step_h_operand = input_operands[9];                                   \
  NNADAPTER_VLOG(5) << "step_h: " << OperandToString(step_h_operand);        \
  /* offset */                                                               \
  auto offset_operand = input_operands[10];                                  \
  NNADAPTER_VLOG(5) << "offset: " << OperandToString(offset_operand);        \
  /* min_max_aspect_ratios_order */                                          \
  auto min_max_aspect_ratios_order_operand = input_operands[11];             \
  NNADAPTER_VLOG(5) << "min_max_aspect_ratios_order: "                       \
                    << OperandToString(min_max_aspect_ratios_order_operand); \
  /* Boxes */                                                                \
  auto boxes_operand = output_operands[0];                                   \
  NNADAPTER_VLOG(5) << "Boxes: " << OperandToString(boxes_operand);          \
  /* Variances */                                                            \
  auto Variances_operand = output_operands[1];                               \
  NNADAPTER_VLOG(5) << "Variances: " << OperandToString(Variances_operand);  \
  auto input_type = input_operands[0]->type;                                 \
  auto image_type = input_operands[1]->type;                                 \
  NNADAPTER_CHECK_EQ(input_type.dimensions.count, 4);                        \
  NNADAPTER_CHECK_EQ(image_type.dimensions.count, 4);

}  // namespace operation
}  // namespace nnadapter
