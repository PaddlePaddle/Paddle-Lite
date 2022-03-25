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

namespace nnadapter {
namespace operation {

#define MULTICLASS_NMS_OPERATION_EXTRACT_INPUTS_OUTPUTS                       \
  auto& input_operands = operation->input_operands;                           \
  auto& output_operands = operation->output_operands;                         \
  auto input_count = input_operands.size();                                   \
  auto output_count = output_operands.size();                                 \
  NNADAPTER_CHECK_EQ(input_count, 9);                                         \
  NNADAPTER_CHECK_EQ(output_count, 2);                                        \
  /* box */                                                                   \
  auto box_operand = input_operands[0];                                       \
  NNADAPTER_VLOG(5) << "box: " << OperandToString(box_operand);               \
  /* scores */                                                                \
  auto scores_operand = input_operands[1];                                    \
  NNADAPTER_VLOG(5) << "scores: " << OperandToString(scores_operand);         \
  /* background_label */                                                      \
  auto background_label_operand = input_operands[2];                          \
  NNADAPTER_VLOG(5) << "background_label: "                                   \
                    << OperandToString(background_label_operand);             \
  /* score_threshold */                                                       \
  auto score_threshold_operand = input_operands[3];                           \
  NNADAPTER_VLOG(5) << "score_threshold: "                                    \
                    << OperandToString(score_threshold_operand);              \
  /* nms_top_k */                                                             \
  auto nms_top_k_operand = input_operands[4];                                 \
  NNADAPTER_VLOG(5) << "nms_top_k: " << OperandToString(nms_top_k_operand);   \
  /* nms_threshold */                                                         \
  auto nms_threshold_operand = input_operands[5];                             \
  NNADAPTER_VLOG(5) << "nms_threshold: "                                      \
                    << OperandToString(nms_threshold_operand);                \
  /* nms_eta */                                                               \
  auto nms_eta_operand = input_operands[6];                                   \
  NNADAPTER_VLOG(5) << "nms_eta: " << OperandToString(nms_eta_operand);       \
  /* keep_top_k */                                                            \
  auto keep_top_k_operand = input_operands[7];                                \
  NNADAPTER_VLOG(5) << "keep_top_k: " << OperandToString(keep_top_k_operand); \
  /* normalized */                                                            \
  auto normalized_operand = input_operands[8];                                \
  NNADAPTER_VLOG(5) << "normalized: " << OperandToString(normalized_operand); \
  /* Output box_res*/                                                         \
  auto output_box_operand = output_operands[0];                               \
  NNADAPTER_VLOG(5) << "output_box: " << OperandToString(output_box_operand); \
  /* Output nms_rois_num*/                                                    \
  auto output_nms_rois_num_operand = output_operands[1];                      \
  NNADAPTER_VLOG(5) << "output_nms_rois_num: "                                \
                    << OperandToString(output_nms_rois_num_operand);

}  // namespace operation
}  // namespace nnadapter
