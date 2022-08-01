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

#define DEFORMABLE_CONV_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS                    \
  auto& input_operands = operation->input_operands;                            \
  auto& output_operands = operation->output_operands;                          \
  auto input_count = input_operands.size();                                    \
  auto output_count = output_operands.size();                                  \
  NNADAPTER_CHECK_EQ(input_count, 11);                                         \
  NNADAPTER_CHECK_EQ(output_count, 1);                                         \
  /* Input */                                                                  \
  auto input_operand = input_operands[0];                                      \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);            \
  /* Offset */                                                                 \
  auto offset_operand = input_operands[1];                                     \
  NNADAPTER_VLOG(5) << "offset: " << OperandToString(offset_operand);          \
  /* Mask */                                                                   \
  auto mask_operand = input_operands[2];                                       \
  NNADAPTER_VLOG(5) << "mask: " << OperandToString(mask_operand);              \
  /* Filter */                                                                 \
  auto filter_operand = input_operands[3];                                     \
  NNADAPTER_VLOG(5) << "filter: " << OperandToString(filter_operand);          \
  auto output_channel_size = filter_operand->type.dimensions.data[0];          \
  auto filter_channel_size = filter_operand->type.dimensions.data[1];          \
  auto filter_height = filter_operand->type.dimensions.data[2];                \
  auto filter_width = filter_operand->type.dimensions.data[3];                 \
  NNADAPTER_VLOG(5) << "filter dims = [" << output_channel_size << ","         \
                    << filter_channel_size << "," << filter_height << ","      \
                    << filter_width << "]";                                    \
  /* Bias */                                                                   \
  auto bias_operand = input_operands[4];                                       \
  NNADAPTER_VLOG(5) << "bias: " << OperandToString(bias_operand);              \
  /* Paddings */                                                               \
  auto pads_buffer = reinterpret_cast<int32_t*>(input_operands[5]->buffer);    \
  NNADAPTER_CHECK_EQ(input_operands[5]->length / sizeof(int32_t), 4);          \
  std::vector<int32_t> pads(pads_buffer, pads_buffer + 4);                     \
  for (size_t i = 0; i < pads.size(); i++) {                                   \
    NNADAPTER_VLOG(5) << "pads[" << i << "]: " << pads[i];                     \
  }                                                                            \
  /* Strides */                                                                \
  auto strides_buffer = reinterpret_cast<int32_t*>(input_operands[6]->buffer); \
  NNADAPTER_CHECK_EQ(input_operands[6]->length / sizeof(int32_t), 2);          \
  std::vector<int32_t> strides(strides_buffer, strides_buffer + 2);            \
  for (size_t i = 0; i < strides.size(); i++) {                                \
    NNADAPTER_VLOG(5) << "strides[" << i << "]: " << strides[i];               \
  }                                                                            \
  /* Group */                                                                  \
  auto group = *reinterpret_cast<int32_t*>(input_operands[7]->buffer);         \
  NNADAPTER_VLOG(5) << "group: " << group;                                     \
  /* Deformable groups */                                                      \
  auto deformable_group =                                                      \
      *reinterpret_cast<int32_t*>(input_operands[8]->buffer);                  \
  NNADAPTER_VLOG(5) << "deformable_group: " << deformable_group;               \
  /* Dilations */                                                              \
  auto dilations_buffer =                                                      \
      reinterpret_cast<int32_t*>(input_operands[9]->buffer);                   \
  NNADAPTER_CHECK_EQ(input_operands[9]->length / sizeof(int32_t), 2);          \
  std::vector<int32_t> dilations(dilations_buffer, dilations_buffer + 2);      \
  for (size_t i = 0; i < dilations.size(); i++) {                              \
    NNADAPTER_VLOG(5) << "dilations[" << i << "]: " << dilations[i];           \
  }                                                                            \
  /* Fuse code */                                                              \
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[10]->buffer);    \
  NNADAPTER_VLOG(5) << "fuse_code: " << fuse_code;                             \
  /* Output */                                                                 \
  auto output_operand = output_operands[0];                                    \
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

}  // namespace operation
}  // namespace nnadapter
