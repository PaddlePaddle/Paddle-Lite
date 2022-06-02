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

#include "core/types.h"

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
  auto input_layout = input_operand->type.layout;                              \
  auto input_channel_size = input_operand->type.dimensions                     \
                                .data[input_layout == NNADAPTER_NCHW ? 1 : 3]; \
  /* Filter */                                                                 \
  auto filter_operand = input_operands[1];                                     \
  NNADAPTER_VLOG(5) << "filter: " << OperandToString(filter_operand);          \
  /* (1) For NCHW, [C_in, C_out, filter_height, filter_width] */               \
  /* (2) For NHWC, [C_out, filter_height, filter_width, C_in] */               \
  auto filter_layout = filter_operand->type.layout;                            \
  auto output_channel_size =                                                   \
      filter_operand->type.dimensions                                          \
          .data[filter_layout == NNADAPTER_NCHW ? 1 : 0];                      \
  auto filter_channel_size =                                                   \
      filter_operand->type.dimensions                                          \
          .data[filter_layout == NNADAPTER_NCHW ? 0 : 3];                      \
  auto filter_height = filter_operand->type.dimensions                         \
                           .data[filter_layout == NNADAPTER_NCHW ? 2 : 1];     \
  auto filter_width = filter_operand->type.dimensions                          \
                          .data[filter_layout == NNADAPTER_NCHW ? 3 : 2];      \
  NNADAPTER_VLOG(5) << "input_channel_size: " << input_channel_size;           \
  NNADAPTER_VLOG(5) << "output_channel_size: " << output_channel_size;         \
  NNADAPTER_VLOG(5) << "filter_channel_size: " << filter_channel_size;         \
  NNADAPTER_VLOG(5) << "filter_height: " << filter_height;                     \
  NNADAPTER_VLOG(5) << "filter_width: " << filter_width;                       \
  /* Bias */                                                                   \
  auto bias_operand = input_operands[2];                                       \
  NNADAPTER_VLOG(5) << "bias: " << OperandToString(bias_operand);              \
  /* Auto pad */                                                               \
  auto auto_pad = static_cast<NNAdapterAutoPadCode>(                           \
      *reinterpret_cast<int32_t*>(input_operands[3]->buffer));                 \
  NNADAPTER_VLOG(5) << "auto_pad: " << AutoPadCodeToString(auto_pad);          \
  /* Pads: Pads are transed according to auto_pad, so pads are used. */        \
  auto pads_size = input_operands[4]->length / sizeof(int32_t);                \
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
  auto strides_size = input_operands[5]->length / sizeof(int32_t);             \
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
  auto dilations_size = input_operands[7]->length / sizeof(int32_t);           \
  NNADAPTER_CHECK_EQ(dilations_size, 2U);                                      \
  auto dilations_buffer =                                                      \
      reinterpret_cast<int32_t*>(input_operands[7]->buffer);                   \
  auto dilation_height = dilations_buffer[0];                                  \
  auto dilation_width = dilations_buffer[1];                                   \
  NNADAPTER_VLOG(5) << "dilations = [" << dilation_height << ", "              \
                    << dilation_width << "]";                                  \
  /* Output_padding */                                                         \
  auto output_padding_size = input_operands[8]->length / sizeof(int32_t);      \
  NNADAPTER_CHECK_EQ(output_padding_size, 2U);                                 \
  auto output_padding_buffer =                                                 \
      reinterpret_cast<int32_t*>(input_operands[8]->buffer);                   \
  int output_padding_height = output_padding_buffer[0];                        \
  int output_padding_width = output_padding_buffer[1];                         \
  NNADAPTER_VLOG(5) << "output_padding = [" << output_padding_height << ", "   \
                    << output_padding_width << "]";                            \
  /* Output_shape */                                                           \
  int output_shape_height = -1;                                                \
  int output_shape_width = -1;                                                 \
  if (input_operands[9] != nullptr) {                                          \
    auto output_shape_size = input_operands[9]->length / sizeof(int32_t);      \
    NNADAPTER_CHECK_EQ(output_shape_size, 2U);                                 \
    auto output_shape_buffer =                                                 \
        reinterpret_cast<int32_t*>(input_operands[9]->buffer);                 \
    output_shape_height = output_shape_buffer[0];                              \
    output_shape_width = output_shape_buffer[1];                               \
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

// Calculate the height or width of the output operand of Conv2DTranspose
// according to the pads, dilation, stride and etc.
int32_t CalcConv2DTransposeOutputSize(int32_t input_size,
                                      int32_t filter_height_or_width,
                                      NNAdapterAutoPadCode auto_pad,
                                      int32_t pad_top_or_left,
                                      int32_t pad_bottom_or_right,
                                      int32_t stride_height_or_width,
                                      int32_t dilation_height_or_width);

}  // namespace operation
}  // namespace nnadapter
