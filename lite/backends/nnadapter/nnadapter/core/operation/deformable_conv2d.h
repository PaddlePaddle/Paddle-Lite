// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#define DEFORMABLE_CONV_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS                    \
  auto& input_operands = operation->input_operands;                            \
  auto& output_operands = operation->output_operands;                          \
  auto input_count = input_operands.size();                                    \
  auto output_count = output_operands.size();                                  \
  NNADAPTER_CHECK_EQ(input_count, 16);                                         \
  NNADAPTER_CHECK_EQ(output_count, 1);                                         \
  /* Input */                                                                  \
  auto input_operand = input_operands[0];                                      \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);            \
  /* Offset */                                                                 \
  auto offset_operand = input_operands[1];                                     \
  NNADAPTER_VLOG(5) << "offset: " << OperandToString(offset_operand);          \
  auto offset_channel = offset_operand->type.dimensions.data[1];               \
  NNADAPTER_VLOG(5) << "offset_channel: " << offset_channel;                   \
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
  auto padding_width_left =                                                    \
      *reinterpret_cast<int32_t*>(input_operands[5]->buffer);                  \
  auto padding_width_right =                                                   \
      *reinterpret_cast<int32_t*>(input_operands[6]->buffer);                  \
  auto padding_height_top =                                                    \
      *reinterpret_cast<int32_t*>(input_operands[7]->buffer);                  \
  auto padding_height_bottom =                                                 \
      *reinterpret_cast<int32_t*>(input_operands[8]->buffer);                  \
  NNADAPTER_VLOG(5) << "paddings=[" << padding_width_left << ","               \
                    << padding_width_right << "," << padding_height_top << "," \
                    << padding_height_bottom << "]";                           \
  /* Strides */                                                                \
  auto stride_width = *reinterpret_cast<int32_t*>(input_operands[9]->buffer);  \
  auto stride_height =                                                         \
      *reinterpret_cast<int32_t*>(input_operands[10]->buffer);                 \
  NNADAPTER_VLOG(5) << "strides=[" << stride_width << "," << stride_height     \
                    << "]";                                                    \
  /* Group */                                                                  \
  auto group = *reinterpret_cast<int32_t*>(input_operands[11]->buffer);        \
  NNADAPTER_VLOG(5) << "group=" << group;                                      \
  /* Deformable groups */                                                      \
  auto deformable_groups =                                                     \
      *reinterpret_cast<int32_t*>(input_operands[12]->buffer);                 \
  NNADAPTER_VLOG(5) << "deformable_groups=" << deformable_groups;              \
  /* Fuse code */                                                              \
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[13]->buffer);    \
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;                              \
  /* Dilations */                                                              \
  auto dilation_width =                                                        \
      *reinterpret_cast<int32_t*>(input_operands[14]->buffer);                 \
  auto dilation_height =                                                       \
      *reinterpret_cast<int32_t*>(input_operands[15]->buffer);                 \
  NNADAPTER_VLOG(5) << "dilations=[" << dilation_width << ","                  \
                    << dilation_height << "]";                                 \
  /* Output */                                                                 \
  auto output_operand = output_operands[0];                                    \
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);

}  // namespace operation
}  // namespace nnadapter
