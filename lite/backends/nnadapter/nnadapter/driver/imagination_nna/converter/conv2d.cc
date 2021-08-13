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

#include "driver/imagination_nna/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace imagination_nna {

int Program::ConvertConv2D(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 9);
  NNADAPTER_CHECK_EQ(output_count, 1);
  // Input
  auto input_operand = input_operands[0];
  NNADAPTER_VLOG(5) << "input: " << OperandToString(input_operand);
  auto input_channel_size = input_operand->type.dimensions[1];
  // Filter
  auto filter_operand = input_operands[1];
  NNADAPTER_VLOG(5) << "filter: " << OperandToString(filter_operand);
  auto output_channel_size = filter_operand->type.dimensions[0];
  auto filter_channel_size = filter_operand->type.dimensions[1];
  auto filter_height = filter_operand->type.dimensions[2];
  auto filter_width = filter_operand->type.dimensions[3];
  // Bias
  auto bias_operand = input_operands[2];
  NNADAPTER_VLOG(5) << "bias: " << OperandToString(bias_operand);
  // Auto pad: not support auto_pad.
  // Pads: Pads are transed according to auto_pad, so pads are used.
  uint32_t pads_size =
      input_operands[4]->length / static_cast<uint32_t>(sizeof(int32_t));
  NNADAPTER_CHECK_EQ(pads_size, 4U);
  auto pads_buffer = reinterpret_cast<int32_t*>(input_operands[4]->buffer);
  auto pad_height_top = pads_buffer[0];
  auto pad_height_bottom = pads_buffer[1];
  auto pad_width_left = pads_buffer[2];
  auto pad_width_right = pads_buffer[3];
  NNADAPTER_VLOG(5) << "paddings = [" << pad_height_top << ", "
                    << pad_height_bottom << ", " << pad_width_left << ", "
                    << pad_width_right << "]";
  // Strides
  uint32_t strides_size =
      input_operands[5]->length / static_cast<uint32_t>(sizeof(int32_t));
  NNADAPTER_CHECK_EQ(strides_size, 2U);
  auto strides_buffer = reinterpret_cast<int32_t*>(input_operands[5]->buffer);
  auto stride_height = strides_buffer[0];
  auto stride_width = strides_buffer[1];
  NNADAPTER_VLOG(5) << "strides = [" << stride_height << ", " << stride_width
                    << "]";
  // Group
  auto group = *reinterpret_cast<int32_t*>(input_operands[6]->buffer);
  NNADAPTER_VLOG(5) << "group = " << group;
  // Dilations
  uint32_t dilations_size =
      input_operands[7]->length / static_cast<uint32_t>(sizeof(int32_t));
  NNADAPTER_CHECK_EQ(dilations_size, 2U);
  auto dilations_buffer = reinterpret_cast<int32_t*>(input_operands[7]->buffer);
  auto dilation_height = dilations_buffer[0];
  auto dilation_width = dilations_buffer[1];
  NNADAPTER_VLOG(5) << "dilations = [" << dilation_height << ", "
                    << dilation_width << "]";
  // Fuse code
  auto fuse_code = *reinterpret_cast<int32_t*>(input_operands[8]->buffer);
  NNADAPTER_VLOG(5) << "fuse_code = " << fuse_code;
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  // Check depthwise mode
  bool is_depthwise_mode = (group != 1 && input_channel_size == group);
  NNADAPTER_VLOG(5) << "depthwise mode(" << is_depthwise_mode << ").";

  // Convert to imgdnn tensors and operators
  auto input_tensor = GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = ConvertOperand(input_operand);
  }
  auto filter_tensor = ConvertOperand(filter_operand);
  // Expand bias tensor from (c) to (1, c)
  auto bias_tensor =
      ConvertOperand(bias_operand, {1, bias_operand->type.dimensions[0]});
  unsigned int ksizes[2] = {static_cast<unsigned int>(filter_height),
                            static_cast<unsigned int>(filter_width)};
  unsigned int strides[2] = {static_cast<unsigned int>(stride_height),
                             static_cast<unsigned int>(stride_width)};
  // Top and left
  unsigned int pad_to_begin[2] = {static_cast<unsigned int>(pad_height_top),
                                  static_cast<unsigned int>(pad_width_left)};
  // Bottom and right
  unsigned int pad_to_end[2] = {static_cast<unsigned int>(pad_height_bottom),
                                static_cast<unsigned int>(pad_width_right)};
  unsigned int dilations[2] = {static_cast<unsigned int>(dilation_height),
                               static_cast<unsigned int>(dilation_width)};
  NNADAPTER_CHECK(
      IsUInt8AsymmPerLayerQuantization(output_operand->type.precision));
  imgdnn_quant_param output_quant_param;
  output_quant_param.scale = output_operand->type.asymm_per_layer_params.scale;
  output_quant_param.zero_point =
      output_operand->type.asymm_per_layer_params.zero_point;
  auto output_tensor = imgdnn_mgr_.CreateConvolutionLayer(input_tensor,
                                                          filter_tensor,
                                                          bias_tensor,
                                                          output_quant_param,
                                                          strides,
                                                          pad_to_begin,
                                                          pad_to_end,
                                                          dilations,
                                                          is_depthwise_mode);
  // fuse RELU ?
  if (fuse_code == NNADAPTER_FUSED_NONE) {
  } else if (fuse_code == NNADAPTER_FUSED_RELU) {
    output_tensor = imgdnn_mgr_.CreateReLULayer(
        output_tensor, true, 0.0, false, 0.0, false);
  } else if (fuse_code == NNADAPTER_FUSED_RELU1) {
    output_tensor =
        imgdnn_mgr_.CreateReLULayer(output_tensor, true, 0.0, true, 1.0, false);
  } else if (fuse_code == NNADAPTER_FUSED_RELU6) {
    output_tensor =
        imgdnn_mgr_.CreateReLULayer(output_tensor, true, 0.0, true, 6.0, false);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                         << ") is found.";
  }
  UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace imagination_nna
}  // namespace nnadapter
