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

#define YOLO_BOX_PARSER_OPERATION_EXTRACT_INPUTS_OUTPUTS                      \
  auto& input_operands = operation->input_operands;                           \
  auto& output_operands = operation->output_operands;                         \
  auto input_count = input_operands.size();                                   \
  auto output_count = output_operands.size();                                 \
  NNADAPTER_CHECK_EQ(input_count, 15);                                        \
  NNADAPTER_CHECK_EQ(output_count, 1);                                        \
  /* Input */                                                                 \
  auto x0_operand = input_operands[0];                                     \
  NNADAPTER_VLOG(5) << "x0: " << OperandToString(x0_operand);           \
  auto x1_operand = input_operands[1];                                     \
  NNADAPTER_VLOG(5) << "x1: " << OperandToString(x1_operand);           \
  auto x2_operand = input_operands[2];                                     \
  NNADAPTER_VLOG(5) << "x2: " << OperandToString(x2_operand);           \
  auto image_shape_operand = input_operands[3];                                   \
  NNADAPTER_VLOG(5) << "image_shape: " << OperandToString(image_shape_operand);         \
  auto image_scale_operand = input_operands[4];                                   \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(image_scale_operand);         \
  /* anchors */                                                               \
  auto anchors_operand0 = input_operands[5];                                   \
  NNADAPTER_VLOG(5) << "anchors0: " << OperandToString(anchors_operand0);       \
  auto anchors_count0 = anchors_operand0->length / sizeof(int32_t);             \
  auto anchors_data0 = reinterpret_cast<int32_t*>(anchors_operand0->buffer);    \
  auto anchors0 =                                                              \
      std::vector<int32_t>(anchors_data0, anchors_data0 + anchors_count0);       \
  auto anchors_operand1 = input_operands[6];                                   \
  NNADAPTER_VLOG(5) << "anchors: " << OperandToString(anchors_operand1);       \
  auto anchors_count1 = anchors_operand1->length / sizeof(int32_t);             \
  auto anchors_data1 = reinterpret_cast<int32_t*>(anchors_operand1->buffer);    \
  auto anchors1 =                                                              \
      std::vector<int32_t>(anchors_data1, anchors_data1 + anchors_count1);       \
  auto anchors_operand2 = input_operands[7];                                   \
  NNADAPTER_VLOG(5) << "anchors: " << OperandToString(anchors_operand2);       \
  auto anchors_count2 = anchors_operand2->length / sizeof(int32_t);             \
  auto anchors_data2 = reinterpret_cast<int32_t*>(anchors_operand2->buffer);    \
  auto anchors2 =                                                              \
      std::vector<int32_t>(anchors_data2, anchors_data2 + anchors_count2);       \
  /* various attrs */                                                         \
  auto class_num = *reinterpret_cast<int*>(input_operands[8]->buffer);        \
  auto conf_thresh = *reinterpret_cast<float*>(input_operands[9]->buffer);    \
  auto downsample_ratio0 = *reinterpret_cast<int*>(input_operands[10]->buffer); \
  auto downsample_ratio1 = *reinterpret_cast<int*>(input_operands[11]->buffer); \
  auto downsample_ratio2 = *reinterpret_cast<int*>(input_operands[12]->buffer); \
  auto clip_bbox = *reinterpret_cast<bool*>(input_operands[13]->buffer);       \
  auto scale_x_y = *reinterpret_cast<float*>(input_operands[14]->buffer);      \
  /* Output */                                                                \
  auto boxes_scores_operand = output_operands[0];                             \
  NNADAPTER_VLOG(5) << "output: " << OperandToString(boxes_scores_operand);
}  // namespace operation
}  // namespace nnadapter
