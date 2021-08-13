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

#include "driver/amlogic_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace amlogic_npu {

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

  // Convert to amlnpu tensors and operators
  auto input_tensor = GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = ConvertOperand(input_operand);
  }
  std::shared_ptr<aml::nn::Tensor> filter_tensor = nullptr;
  if (is_depthwise_mode) {
    filter_tensor = ConvertOperand(filter_operand,
                                   {filter_channel_size,
                                    output_channel_size,
                                    filter_height,
                                    filter_width});
  } else {
    filter_tensor = ConvertOperand(filter_operand);
  }
  auto bias_tensor = ConvertOperand(bias_operand);
  auto output_tensor = ConvertOperand(output_operand);
  aml::nn::Conv2DAttr attr;
  attr.ksize[0] = filter_height;
  attr.ksize[1] = filter_width;
  attr.stride[0] = stride_height;
  attr.stride[1] = stride_width;
  attr.pad[0] = pad_height_top;
  attr.pad[1] = pad_height_bottom;
  attr.pad[2] = pad_width_left;
  attr.pad[3] = pad_width_right;
  attr.group = group;
  attr.multiplier = is_depthwise_mode ? output_channel_size / group : 0;
  attr.weights = output_channel_size;
  attr.dilation[0] = dilation_height;
  attr.dilation[1] = dilation_width;
  attr.pad_type = aml::nn::PadType::AUTO;
  // fuse RELU ?
  if (fuse_code != NNADAPTER_FUSED_NONE) {
    NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                         << ") is found.";
  }
  std::vector<std::shared_ptr<aml::nn::Tensor>> input_tensors = {
      input_tensor, filter_tensor, bias_tensor};
  std::vector<std::shared_ptr<aml::nn::Tensor>> output_tensors = {
      output_tensor};
  graph_->AddOperator(
      aml::nn::OperatorType::CONV2D, input_tensors, output_tensors, &attr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace amlogic_npu
}  // namespace nnadapter
