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

#include "core/operation/conv2d.h"
#include "driver/mediatek_apu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace mediatek_apu {

int Program::ConvertConv2D(hal::Operation* operation) {
  CONV2D_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Dynamic shapes are still not supported
  NNADAPTER_CHECK_EQ(input_operand->type.dynamic_dimension_count, 0);
  operation::UpdateConv2DPadAndDilation(input_operand->type.dimensions[1],
                                        filter_height,
                                        auto_pad,
                                        &pad_height_top,
                                        &pad_height_bottom,
                                        stride_height,
                                        &dilation_height);
  operation::UpdateConv2DPadAndDilation(input_operand->type.dimensions[2],
                                        filter_width,
                                        auto_pad,
                                        &pad_width_left,
                                        &pad_width_right,
                                        stride_width,
                                        &dilation_width);
  // NHWC
  input_channel_size = input_operand->type.dimensions[3];
  is_depthwise_mode = group != 1 && input_channel_size == group;
  NNADAPTER_VLOG(5) << "Update depthwise mode(" << is_depthwise_mode << ").";
  NNADAPTER_CHECK_EQ(dilation_height, 1)
      << "MediaTek APU only supports dilations = [1,1]";
  NNADAPTER_CHECK_EQ(dilation_width, 1)
      << "MediaTek APU only supports dilations = [1,1]";

  // Convert to Neuron operands and operations
  auto input_index = GetMappedIndex(input_operand);
  if (input_index == INVALID_INDEX) {
    input_index = ConvertOperand(input_operand);
  }
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
  auto padding_width_left_index = AddInt32ConstantOperand(pad_width_left);
  auto padding_width_right_index = AddInt32ConstantOperand(pad_width_right);
  auto padding_height_top_index = AddInt32ConstantOperand(pad_height_top);
  auto padding_height_bottom_index = AddInt32ConstantOperand(pad_height_bottom);
  auto stride_width_index = AddInt32ConstantOperand(stride_width);
  auto stride_height_index = AddInt32ConstantOperand(stride_height);
  auto fuse_code_index =
      AddInt32ConstantOperand(ConvertFuseCodeToNeuronFuseCode(fuse_code));
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
    int32_t multiplier = filter_operand->type.dimensions[3] / group;
    NNADAPTER_CHECK_EQ(multiplier, 1)
        << "MediaTek APU only supports multiplier=1, but recieved multiplier="
        << multiplier << " which C_out=" << filter_operand->type.dimensions[3]
        << " and group=" << group;
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
