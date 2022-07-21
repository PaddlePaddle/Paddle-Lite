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

#define CUSTOM_YOLO_BOX_3D_NMS_FUSER_OPERATION_EXTRACT_INPUTS_OUTPUTS          \
  auto& input_operands = operation->input_operands;                            \
  auto& output_operands = operation->output_operands;                          \
  auto input_count = input_operands.size();                                    \
  auto output_count = output_operands.size();                                  \
  NNADAPTER_CHECK_EQ(input_count, 20);                                         \
  NNADAPTER_CHECK_EQ(output_count, 6);                                         \
  /* Inputs */                                                                 \
  auto input0_operand = input_operands[0];                                     \
  NNADAPTER_VLOG(5) << "input0: " << OperandToString(input0_operand);          \
  auto input1_operand = input_operands[1];                                     \
  NNADAPTER_VLOG(5) << "input1: " << OperandToString(input1_operand);          \
  auto input2_operand = input_operands[2];                                     \
  NNADAPTER_VLOG(5) << "input2: " << OperandToString(input2_operand);          \
  auto imgsize_operand = input_operands[3];                                    \
  NNADAPTER_VLOG(5) << "imgsize: " << OperandToString(imgsize_operand);        \
  /* anchors */                                                                \
  auto anchors0_operand = input_operands[4];                                   \
  NNADAPTER_VLOG(5) << "anchors0: " << OperandToString(anchors0_operand);      \
  auto anchors0_count = anchors0_operand->length / sizeof(int32_t);            \
  auto anchors0_data = reinterpret_cast<int32_t*>(anchors0_operand->buffer);   \
  auto anchors0 =                                                              \
      std::vector<int32_t>(anchors0_data, anchors0_data + anchors0_count);     \
  for (size_t i = 0; i < anchors0.size(); i++) {                               \
    NNADAPTER_VLOG(5) << "anchors0[" << i << "]: " << anchors0[i];             \
  }                                                                            \
  auto anchors1_operand = input_operands[5];                                   \
  NNADAPTER_VLOG(5) << "anchors1: " << OperandToString(anchors1_operand);      \
  auto anchors1_count = anchors1_operand->length / sizeof(int32_t);            \
  auto anchors1_data = reinterpret_cast<int32_t*>(anchors1_operand->buffer);   \
  auto anchors1 =                                                              \
      std::vector<int32_t>(anchors1_data, anchors1_data + anchors1_count);     \
  for (size_t i = 0; i < anchors1.size(); i++) {                               \
    NNADAPTER_VLOG(5) << "anchors1[" << i << "]: " << anchors1[i];             \
  }                                                                            \
  auto anchors2_operand = input_operands[6];                                   \
  NNADAPTER_VLOG(5) << "anchors2: " << OperandToString(anchors2_operand);      \
  auto anchors2_count = anchors2_operand->length / sizeof(int32_t);            \
  auto anchors2_data = reinterpret_cast<int32_t*>(anchors2_operand->buffer);   \
  auto anchors2 =                                                              \
      std::vector<int32_t>(anchors2_data, anchors2_data + anchors2_count);     \
  for (size_t i = 0; i < anchors2.size(); i++) {                               \
    NNADAPTER_VLOG(5) << "anchors2[" << i << "]: " << anchors2[i];             \
  }                                                                            \
  /* various attrs */                                                          \
  auto class_num = *reinterpret_cast<int*>(input_operands[7]->buffer);         \
  NNADAPTER_VLOG(5) << "class_num: " << class_num;                             \
  auto conf_thresh = *reinterpret_cast<float*>(input_operands[8]->buffer);     \
  NNADAPTER_VLOG(5) << "conf_thresh: " << conf_thresh;                         \
  auto downsample_ratio0 = *reinterpret_cast<int*>(input_operands[9]->buffer); \
  NNADAPTER_VLOG(5) << "downsample_ratio0: " << downsample_ratio0;             \
  auto downsample_ratio1 =                                                     \
      *reinterpret_cast<int*>(input_operands[10]->buffer);                     \
  NNADAPTER_VLOG(5) << "downsample_ratio1: " << downsample_ratio1;             \
  auto downsample_ratio2 =                                                     \
      *reinterpret_cast<int*>(input_operands[11]->buffer);                     \
  NNADAPTER_VLOG(5) << "downsample_ratio2: " << downsample_ratio2;             \
  auto scale_x_y = *reinterpret_cast<float*>(input_operands[12]->buffer);      \
  NNADAPTER_VLOG(5) << "scale_x_y: " << scale_x_y;                             \
  /* background_label */                                                       \
  auto background_label =                                                      \
      *reinterpret_cast<int32_t*>(input_operands[13]->buffer);                 \
  NNADAPTER_VLOG(5) << "background_label: " << background_label;               \
  /* score_threshold */                                                        \
  auto score_threshold =                                                       \
      *reinterpret_cast<float*>(input_operands[14]->buffer);                   \
  NNADAPTER_VLOG(5) << "score_threshold: " << score_threshold;                 \
  /* nms_top_k */                                                              \
  auto nms_top_k = *reinterpret_cast<int32_t*>(input_operands[15]->buffer);    \
  NNADAPTER_VLOG(5) << "nms_top_k: " << nms_top_k;                             \
  /* nms_threshold */                                                          \
  auto nms_threshold = *reinterpret_cast<float*>(input_operands[16]->buffer);  \
  NNADAPTER_VLOG(5) << "nms_threshold: " << nms_threshold;                     \
  /* nms_eta */                                                                \
  auto nms_eta = *reinterpret_cast<float*>(input_operands[17]->buffer);        \
  NNADAPTER_VLOG(5) << "nms_eta: " << nms_eta;                                 \
  /* keep_top_k */                                                             \
  auto keep_top_k = *reinterpret_cast<int32_t*>(input_operands[18]->buffer);   \
  NNADAPTER_VLOG(5) << "keep_top_k: " << keep_top_k;                           \
  /* normalized */                                                             \
  auto normalized = *reinterpret_cast<bool*>(input_operands[19]->buffer);      \
  NNADAPTER_VLOG(5) << "normalized: " << normalized;                           \
  /* Output box_res*/                                                          \
  auto output_box_operand = output_operands[0];                                \
  NNADAPTER_VLOG(5) << "output_box: " << OperandToString(output_box_operand);  \
  /* Output nms_rois_num*/                                                     \
  auto output_nms_rois_num_operand = output_operands[1];                       \
  NNADAPTER_VLOG(5) << "output_nms_rois_num: "                                 \
                    << OperandToString(output_nms_rois_num_operand);           \
  /* Output index*/                                                            \
  auto output_index_operand = output_operands[2];                              \
  NNADAPTER_VLOG(5) << "output_index: "                                        \
                    << OperandToString(output_index_operand);                  \
  /* Output location*/                                                         \
  auto location_operand = output_operands[3];                                  \
  NNADAPTER_VLOG(5) << "location: " << OperandToString(location_operand);      \
  /* Output dim*/                                                              \
  auto dim_operand = output_operands[4];                                       \
  NNADAPTER_VLOG(5) << "dim: " << OperandToString(dim_operand);                \
  /* Output alpha*/                                                            \
  auto alpha_operand = output_operands[5];                                     \
  NNADAPTER_VLOG(5) << "alpha: " << OperandToString(alpha_operand);

}  // namespace operation
}  // namespace nnadapter
