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

#define NON_MAX_SUPPRESSION_OPERATION_EXTRACT_INPUTS_OUTPUTS                   \
  auto& input_operands = operation->input_operands;                            \
  auto& output_operands = operation->output_operands;                          \
  auto input_count = input_operands.size();                                    \
  auto output_count = output_operands.size();                                  \
  NNADAPTER_CHECK_EQ(input_count, 11);                                         \
  NNADAPTER_CHECK_GE(output_count, 2);                                         \
  /* box */                                                                    \
  auto bboxes_operand = input_operands[0];                                     \
  NNADAPTER_VLOG(5) << "box: " << OperandToString(bboxes_operand);             \
  /* scores */                                                                 \
  auto scores_operand = input_operands[1];                                     \
  NNADAPTER_VLOG(5) << "scores: " << OperandToString(scores_operand);          \
  /* rois_num */                                                               \
  auto rois_num_operand = input_operands[2];                                   \
  NNADAPTER_VLOG(5) << "rois_num: " << OperandToString(rois_num_operand);      \
  /* background_label */                                                       \
  auto background_label =                                                      \
      *reinterpret_cast<int32_t*>(input_operands[3]->buffer);                  \
  NNADAPTER_VLOG(5) << "background_label: " << background_label;               \
  /* score_threshold */                                                        \
  auto score_threshold = *reinterpret_cast<float*>(input_operands[4]->buffer); \
  NNADAPTER_VLOG(5) << "score_threshold: " << score_threshold;                 \
  /* nms_top_k */                                                              \
  auto nms_top_k = *reinterpret_cast<int32_t*>(input_operands[5]->buffer);     \
  NNADAPTER_VLOG(5) << "nms_top_k: " << nms_top_k;                             \
  /* nms_threshold */                                                          \
  auto nms_threshold = *reinterpret_cast<float*>(input_operands[6]->buffer);   \
  NNADAPTER_VLOG(5) << "nms_threshold: " << nms_threshold;                     \
  /* nms_eta */                                                                \
  auto nms_eta = *reinterpret_cast<float*>(input_operands[7]->buffer);         \
  NNADAPTER_VLOG(5) << "nms_eta: " << nms_eta;                                 \
  /* keep_top_k */                                                             \
  auto keep_top_k = *reinterpret_cast<int32_t*>(input_operands[8]->buffer);    \
  NNADAPTER_VLOG(5) << "keep_top_k: " << keep_top_k;                           \
  /* normalized */                                                             \
  auto normalized = *reinterpret_cast<bool*>(input_operands[9]->buffer);       \
  NNADAPTER_VLOG(5) << "normalized: " << normalized;                           \
  /* return_index */                                                           \
  auto return_index = *reinterpret_cast<bool*>(input_operands[10]->buffer);    \
  NNADAPTER_VLOG(5) << "return_index: " << return_index;                       \
  /* Output box_res*/                                                          \
  auto output_box_operand = output_operands[0];                                \
  NNADAPTER_VLOG(5) << "output_box: " << OperandToString(output_box_operand);  \
  /* Output nms_rois_num*/                                                     \
  auto output_nms_rois_num_operand = output_operands[1];                       \
  NNADAPTER_VLOG(5) << "output_nms_rois_num: "                                 \
                    << OperandToString(output_nms_rois_num_operand);           \
  nnadapter::core::Operand* output_index_operand = nullptr;                    \
  if (return_index) {                                                          \
    /* Output index*/                                                          \
    output_index_operand = output_operands[2];                                 \
    NNADAPTER_VLOG(5) << "output_index: "                                      \
                      << OperandToString(output_index_operand);                \
  }
}  // namespace operation
}  // namespace nnadapter
