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

namespace nnadapter {
namespace operation {

#define YOLO_BOX_HEAD_OPERATION_EXTRACT_INPUTS_OUTPUTS                        \
  auto& input_operands = operation->input_operands;                           \
  auto& output_operands = operation->output_operands;                         \
  auto input_count = input_operands.size();                                   \
  auto output_count = output_operands.size();                                 \
  NNADAPTER_CHECK_EQ(input_count, 7);                                         \
  NNADAPTER_CHECK_EQ(output_count, 1);                                        \
  /* Input */                                                                 \
  auto input_operand = input_operands[0];                                     \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);           \
  /* anchors */                                                               \
  auto anchors_operand = input_operands[1];                                   \
  NNADAPTER_VLOG(5) << "anchors: " << OperandToString(anchors_operand);       \
  auto anchors_count = anchors_operand->length / sizeof(int32_t);             \
  auto anchors_data = reinterpret_cast<int32_t*>(anchors_operand->buffer);    \
  auto anchors =                                                              \
      std::vector<int32_t>(anchors_data, anchors_data + anchors_count);       \
  for (size_t i = 0; i < anchors.size(); i++) {                               \
    NNADAPTER_VLOG(5) << "anchors[" << i << "]: " << anchors[i];              \
  }                                                                           \
  /* various attrs */                                                         \
  auto class_num = *reinterpret_cast<int*>(input_operands[2]->buffer);        \
  auto conf_thresh = *reinterpret_cast<float*>(input_operands[3]->buffer);    \
  auto downsample_ratio = *reinterpret_cast<int*>(input_operands[4]->buffer); \
  auto clip_bbox = *reinterpret_cast<bool*>(input_operands[5]->buffer);       \
  auto scale_x_y = *reinterpret_cast<float*>(input_operands[6]->buffer);      \
  /* Output */                                                                \
  auto output_operand = output_operands[0];                                   \
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
}  // namespace operation
}  // namespace nnadapter
