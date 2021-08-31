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

#define CONV_2D_TRANSPOSE_OPERATION_EXTRACT_INPUTS_OUTPUTS                     \
  auto& input_operands = operation->input_operands;                            \
  auto& output_operands = operation->output_operands;                          \
  auto input_count = input_operands.size();                                    \
  auto output_count = output_operands.size();                                  \
  NNADAPTER_CHECK_EQ(input_count, 11);                                         \
  NNADAPTER_CHECK_EQ(output_count, 1);                                         \
  /* Input */                                                                  \
  auto input_operand = input_operands[0];                                      \
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);            \
  auto input_channel_size = input_operand->type.dimensions[1];                 \
  /* Filter */                                                                 \
  auto filter_operand = input_operands[1];                                     \
  NNADAPTER_VLOG(5) << "filter: " << OperandToString(filter_operand);          \
  auto output_channel_size = filter_operand->type.dimensions[0];               \
  auto filter_channel_size = filter_operand->type.dimensions[1];               \
  auto filter_height = filter_operand->type.dimensions[2];                     \
  auto filter_width = filter_operand->type.dimensions[3];                      \
  NNADAPTER_VLOG(5) << "filter_dims: [" << output_channel_size << ", "         \
                    << filter_channel_size << ", " << filter_height << ", "    \
                    << filter_width << "]";                                    \
  /* Bias */                                                                   \
  auto bias_operand = input_operands[2];                                       \
  NNADAPTER_VLOG(5) << "bias: " << OperandToString(bias_operand);              \
  /* Auto pad: not support auto_pad. */                                        \
  /* Pads: Pads are transed according to auto_pad, so pads are used. */        \
  uint32_t pads_size =                                                         \
      input_operands[4]->length / static_cast<uint32_t>(sizeof(int32_t));      \
  NNADAPTER_CHECK_EQ(pads_size, 4U);                                           \
  auto pads_buffer = reinterpret_cast<int32_t*>(input_operands[4]->buffer);    \
  auto pad_height_top = pads_buffer[0];                                        \
  auto pad_height_bottom = pads_buffer[1];                                     \
  auto pad_width_left = pads_buffer[2];                                        \
  auto pad_width_right = pads_buffer[3];                                       \
  NNADAPTER_VLOG(5) << "paddings = [" << pad_height_top << ", "                \
                    << pad_height_bottom << ", " << pad_width_left << ", "     \
                    << pad_width_right << "]";                                 \
  /* Strides */                                                                \
  uint32_t strides_size =                                                      \
      input_operands[5]->length / static_cast<uint32_t>(sizeof(int32_t));      \
  NNADAPTER_CHECK_EQ(strides_size, 2U);                                        \
  auto strides_buffer = reinterpret_cast<int32_t*>(input_operands[5]->buffer); \
  auto stride_height = strides_buffer[0];                                      \
  auto stride_width = strides_buffer[1];                                       \
  NNADAPTER_VLOG(5) << "strides = [" << stride_height << ", " << stride_width  \
                    << "]";                                                    \
  /* Group */                                                                  \
  auto group = *reinterpret_cast<int32_t*>(input_operands[6]->buffer);         \
  NNADAPTER_VLOG(5) << "group = " << group;                                    \
  /* Dilations */                                                              \
  uint32_t dilations_size =                                                    \
      input_operands[7]->length / static_cast<uint32_t>(sizeof(int32_t));      \
  NNADAPTER_CHECK_EQ(dilations_size, 2U);                                      \
  auto dilations_buffer =                                                      \
      reinterpret_cast<int32_t*>(input_operands[7]->buffer);                   \
  auto dilation_height = dilations_buffer[0];                                  \
  auto dilation_width = dilations_buffer[1];                                   \
  NNADAPTER_VLOG(5) << "dilations = [" << dilation_height << ", "              \
                    << dilation_width << "]";                                  \
  /* Output_padding */                                                         \
  int output_padding_height = 0;                                               \
  int output_padding_width = 0;                                                \
  if (input_operands[8] != nullptr) {                                          \
    uint32_t output_padding_size =                                             \
        input_operands[8]->length / static_cast<uint32_t>(sizeof(int32_t));    \
    NNADAPTER_CHECK_EQ(output_padding_size, 2U);                               \
    auto output_padding_buffer =                                               \
        reinterpret_cast<int32_t*>(input_operands[8]->buffer);                 \
    auto output_padding_height = output_padding_buffer[0];                     \
    auto output_padding_width = output_padding_buffer[1];                      \
  }                                                                            \
  NNADAPTER_VLOG(5) << "output_padding = [" << output_padding_height << ", "   \
                    << output_padding_width << "]";                            \
  if (output_padding_height != 0 || output_padding_width != 0) {               \
    NNADAPTER_LOG(WARNING)                                                     \
        << "Only support output_padding_height/output_padding_width == 0.";    \
    return NNADAPTER_INVALID_PARAMETER;                                        \
  }                                                                            \
  /* Output_shape */                                                           \
  int output_shape_height = -1;                                                \
  int output_shape_width = -1;                                                 \
  if (input_operands[9] != nullptr) {                                          \
    uint32_t output_shape_size =                                               \
        input_operands[9]->length / static_cast<uint32_t>(sizeof(int32_t));    \
    NNADAPTER_CHECK_EQ(output_shape_size, 2U);                                 \
    auto output_shape_buffer =                                                 \
        reinterpret_cast<int32_t*>(input_operands[9]->buffer);                 \
    auto output_shape_height = output_shape_buffer[0];                         \
    auto output_shape_width = output_shape_buffer[1];                          \
  }                                                                            \
  NNADAPTER_VLOG(5) << "output_shape = [" << output_shape_height << ", "       \
                    << output_shape_width << "]";                              \
  /* Fuse code */                                                              \
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[10]->buffer);    \
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;                              \
  /* Output */                                                                 \
  auto output_operand = output_operands[0];                                    \
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);          \
  /* Check depthwise mode */                                                   \
  bool is_depthwise_mode = (group != 1 && input_channel_size == group);        \
  NNADAPTER_VLOG(5) << "depthwise mode(" << is_depthwise_mode << ").";

}  // namespace operation
}  // namespace nnadapter
