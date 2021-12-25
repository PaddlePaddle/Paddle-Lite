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

#define ROI_ALIGN_OPERATION_EXTRACT_INPUTS_OUTPUTS                             \
  auto& input_operands = operation->input_operands;                            \
  auto& output_operands = operation->output_operands;                          \
  auto input_count = input_operands.size();                                    \
  auto output_count = output_operands.size();                                  \
  NNADAPTER_CHECK_EQ(input_count, 8);                                          \
  NNADAPTER_CHECK_EQ(output_count, 1);                                         \
  /* Input */                                                                  \
  auto input_operand = input_operands[0];                                      \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);            \
  /* ROIs */                                                                   \
  auto rois_operand = input_operands[1];                                       \
  NNADAPTER_VLOG(5) << "rois: " << OperandToString(rois_operand);              \
  /* Batch indices */                                                          \
  auto batch_indices_operand = input_operands[2];                              \
  NNADAPTER_VLOG(5) << "batch indices: "                                       \
                    << OperandToString(batch_indices_operand);                 \
  /* Output height */                                                          \
  auto output_height = *reinterpret_cast<int32_t*>(input_operands[3]->buffer); \
  NNADAPTER_VLOG(5) << "output height = " << output_height;                    \
  /* Output width */                                                           \
  auto output_width = *reinterpret_cast<int32_t*>(input_operands[4]->buffer);  \
  NNADAPTER_VLOG(5) << "output width = " << output_width;                      \
  /* Sampling ratio */                                                         \
  auto sampling_ratio =                                                        \
      *reinterpret_cast<int32_t*>(input_operands[5]->buffer);                  \
  NNADAPTER_VLOG(5) << "sampling ratio = " << sampling_ratio;                  \
  /* Spatial scale */                                                          \
  auto spatial_scale = *reinterpret_cast<float*>(input_operands[6]->buffer);   \
  NNADAPTER_VLOG(5) << "spatial scale = " << spatial_scale;                    \
  /* Aligned */                                                                \
  auto aligned = *reinterpret_cast<bool*>(input_operands[7]->buffer);          \
  NNADAPTER_VLOG(5) << "aligned = " << aligned;                                \
  /* Output */                                                                 \
  auto output_operand = output_operands[0];                                    \
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

}  // namespace operation
}  // namespace nnadapter
