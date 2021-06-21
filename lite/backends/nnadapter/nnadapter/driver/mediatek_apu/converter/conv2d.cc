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

#include "driver/mediatek_apu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace mediatek_apu {

int Program::ConvertConv2D(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 13);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input"
                    << " : " << OperandToString(input_operand);
  auto input_channel_size = input_operand->type.dimensions[3];
  // Filter
  auto filter_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "filter: " << OperandToString(filter_operand);
  NNADAPTER_CHECK(filter_operand && filter_operand->buffer);
  auto filter_height = filter_operand->type.dimensions[1];
  auto filter_width = filter_operand->type.dimensions[2];
  // Bias
  auto bias_operand = input_operands[2];
  NNADAPTER_VLOG(5) << "bias: " << OperandToString(bias_operand);
  NNADAPTER_CHECK(bias_operand && bias_operand->buffer);
  // Paddings
  auto padding_width_left =
      *reinterpret_cast<int32_t*>(input_operands[3]->buffer);
  auto padding_width_right =
      *reinterpret_cast<int32_t*>(input_operands[4]->buffer);
  auto padding_height_top =
      *reinterpret_cast<int32_t*>(input_operands[5]->buffer);
  auto padding_height_bottom =
      *reinterpret_cast<int32_t*>(input_operands[6]->buffer);
  NNADAPTER_VLOG(5) << "paddings=[" << padding_width_left << ","
                    << padding_width_right << "," << padding_height_top << ","
                    << padding_height_bottom << "]";
  // Strides
  auto stride_width = *reinterpret_cast<int32_t*>(input_operands[7]->buffer);
  auto stride_height = *reinterpret_cast<int32_t*>(input_operands[8]->buffer);
  NNADAPTER_VLOG(5) << "strides=[" << stride_width << "," << stride_height
                    << "]";
  // Group
  auto group = *reinterpret_cast<int32_t*>(input_operands[9]->buffer);
  NNADAPTER_VLOG(5) << "group=" << group;
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[10]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code=" << fuse_code;
  // Dilations
  auto dilation_width = *reinterpret_cast<int32_t*>(input_operands[11]->buffer);
  auto dilation_height =
      *reinterpret_cast<int32_t*>(input_operands[12]->buffer);
  NNADAPTER_CHECK_EQ(dilation_width, 1);
  NNADAPTER_CHECK_EQ(dilation_height, 1);
  NNADAPTER_VLOG(5) << "dilations=[" << dilation_width << "," << dilation_height
                    << "]";
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  // Check depthwise mode
  bool is_depthwise_mode = (group != 1 && input_channel_size == group);
  NNADAPTER_VLOG(5) << "depthwise mode(" << is_depthwise_mode << ").";

  // Convert to Neuron operands and operations
  auto input_index = ConvertOperand(input_operand);
  bool is_per_channel = filter_operand->type.precision ==
                        NNADAPTER_TENSOR_QUANT_INT8_SYMM_PER_CHANNEL;
  NNADAPTER_CHECK(filter_operand->type.precision ==
                      NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER ||
                  is_per_channel);
  NNADAPTER_VLOG(5) << "is_per_channel:" << is_per_channel;
  auto filter_index = ConvertOperand(filter_operand);
  NNADAPTER_VLOG(5) << "filter_index:" << filter_index;
  NNADAPTER_VLOG(5) << "bias_buffer:" << std::hex << bias_operand->buffer;
  auto bias_index = ConvertOperand(bias_operand);
  NNADAPTER_VLOG(5) << "bias_index:" << bias_index;
  auto padding_width_left_index = AddInt32ConstantOperand(padding_width_left);
  auto padding_width_right_index = AddInt32ConstantOperand(padding_width_right);
  auto padding_height_top_index = AddInt32ConstantOperand(padding_height_top);
  auto padding_height_bottom_index =
      AddInt32ConstantOperand(padding_height_bottom);
  auto stride_width_index = AddInt32ConstantOperand(stride_width);
  auto stride_height_index = AddInt32ConstantOperand(stride_height);
  auto fuse_code_index = AddInt32ConstantOperand(ConvertFuseCode(fuse_code));
  auto output_index = ConvertOperand(output_operand);
  std::vector<uint32_t> input_indexes = {input_index,
                                         filter_index,
                                         bias_index,
                                         padding_width_left_index,
                                         padding_width_right_index,
                                         padding_height_top_index,
                                         padding_height_bottom_index,
                                         stride_width_index,
                                         stride_height_index};
  std::vector<uint32_t> output_indexes = {output_index};
  if (is_depthwise_mode) {
    int32_t multiplier =
        filter_operand->type.dimensions[3] / input_channel_size;
    auto multiplier_index = AddInt32ConstantOperand(multiplier);
    input_indexes.push_back(multiplier_index);
    input_indexes.push_back(fuse_code_index);
    NNADAPTER_CHECK_EQ(
        AddOperation(NEURON_DEPTHWISE_CONV_2D, &input_indexes, &output_indexes),
        NEURON_NO_ERROR);
  } else {
    input_indexes.push_back(fuse_code_index);
    NNADAPTER_CHECK_EQ(
        AddOperation(NEURON_CONV_2D, &input_indexes, &output_indexes),
        NEURON_NO_ERROR);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
