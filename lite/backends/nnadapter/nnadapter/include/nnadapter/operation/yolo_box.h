// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#define YOLO_BOX_OPERATION_EXTRACT_INPUTS_OUTPUTS                             \
  auto& input_operands = operation->input_operands;                           \
  auto& output_operands = operation->output_operands;                         \
  auto input_count = input_operands.size();                                   \
  auto output_count = output_operands.size();                                 \
  NNADAPTER_CHECK_EQ(input_count, 8);                                         \
  NNADAPTER_CHECK_EQ(output_count, 2);                                        \
  /* x */                                                                     \
  auto x_operand = input_operands[0];                                         \
  NNADAPTER_VLOG(5) << "x: " << OperandToString(x_operand);                   \
  /* image_size */                                                            \
  auto image_size_operand = input_operands[1];                                \
  NNADAPTER_VLOG(5) << "image_size: " << OperandToString(image_size_operand); \
  /* anchors */                                                               \
  auto anchors_operand = input_operands[2];                                   \
  NNADAPTER_VLOG(5) << "anchors: " << OperandToString(anchors_operand);       \
  auto anchors_count = anchors_operand->length / sizeof(int32_t);             \
  auto anchors_data = reinterpret_cast<int32_t*>(anchors_operand->buffer);    \
  anchors = std::vector<int32_t>(anchors_data, anchors_data + anchors_count); \
  /* class_num */                                                             \
  auto class_num_operand = input_operands[3];                                 \
  NNADAPTER_VLOG(5) << "class_num: " << OperandToString(class_num_operand);   \
  /* conf_thresh */                                                           \
  auto conf_thresh_operand = input_operands[4];                               \
  NNADAPTER_VLOG(5) << "conf_thresh: "                                        \
                    << OperandToString(conf_thresh_operand);                  \
  /* downsample_ratio */                                                      \
  auto downsample_ratio_operand = input_operands[5];                          \
  NNADAPTER_VLOG(5) << "downsample_ratio: "                                   \
                    << OperandToString(adownsample_ratio_operand);            \
  /* clip_bbox */                                                             \
  auto clip_bbox_operand = input_operands[6];                                 \
  NNADAPTER_VLOG(5) << "clip_bbox: " << OperandToString(clip_bbox_operand);   \
  /* scale_x_y */                                                             \
  auto scale_x_y_operand = input_operands[7];                                 \
  NNADAPTER_VLOG(5) << "scale_x_y: " << OperandToString(scale_x_y_operand);   \
  /* Output box*/                                                             \
  auto output_box_operand = output_operands[0];                               \
  NNADAPTER_VLOG(5) << "output_box: " << OperandToString(output_box_operand); \
  /* Output score*/                                                           \
  auto output_score_operand = output_operands[1];                             \
  NNADAPTER_VLOG(5) << "output_score: "                                       \
                    << OperandToString(output_score_operand);

}  // namespace operation
}  // namespace nnadapter
