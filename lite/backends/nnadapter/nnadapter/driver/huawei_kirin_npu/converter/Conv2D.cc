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

#include "driver/huawei_kirin_npu/converter.h"
#include "utility/debug.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int Program::ConvertConv2D(hal::Operation* operation) {
  auto& input_operands = operation->input_operands;
  auto& output_operands = operation->output_operands;
  auto input_count = input_operands.size();
  auto output_count = output_operands.size();
  NNADAPTER_CHECK_EQ(input_count, 13);
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
  NNADAPTER_VLOG(5) << "dilations=[" << dilation_width << "," << dilation_height
                    << "]";
  // Output
  auto output_operand = output_operands[0];
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  // Check depthwise mode
  bool is_depthwise_mode = (group != 1 && input_channel_size == group);
  NNADAPTER_VLOG(5) << "depthwise mode(" << is_depthwise_mode << ").";

  // Convert to HiAI operators
  // Check depthwise mode, and decide whether use ConvolutionDepthwise
  bool use_depthwise_conv =
      false;  // Whether use ge::op::ConvolutionDepthwise ?
  if (is_depthwise_mode &&
      !((group == 1 || group >= 5) && dilation_width == 1 &&
        dilation_height == 1)) {
    use_depthwise_conv = true;
    NNADAPTER_LOG(WARNING) << "For depthwise mode, dilation = 1 and group >= 5 "
                              "(or group = 1) is only supported in "
                              "Convolution, so force to use "
                              "ConvolutionDepthwise, but may lead poor "
                              "performance.";
  }
  auto input_operator = ConvertOperand(input_operand);
  auto filter_operator = ConvertOperand(filter_operand);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimension_count, 1);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimensions[0], output_channel_size);
  // Limitations in HiAI, force to pad the dimension of bias to 4-D
  std::vector<int64_t> bias_dimensions = {1, output_channel_size, 1, 1};
  auto bias_operator = ConvertOperand(bias_operand, bias_dimensions);
  std::shared_ptr<ge::Operator> conv_operator = nullptr;
  if (use_depthwise_conv && is_depthwise_mode) {
    auto depthwise_conv_operator =
        AddOperator<ge::op::ConvolutionDepthwise>(output_operand);
    depthwise_conv_operator->set_input_x(*input_operator);
    depthwise_conv_operator->set_input_filter(*filter_operator);
    depthwise_conv_operator->set_attr_mode(1);
    depthwise_conv_operator->set_attr_algo(0);
    depthwise_conv_operator->set_attr_format(0);    // NCHW
    depthwise_conv_operator->set_attr_pad_mode(5);  // VALID
    depthwise_conv_operator->set_attr_group(group);
    depthwise_conv_operator->set_attr_pad(
        ge::AttrValue::LIST_INT({padding_height_bottom,
                                 padding_height_top,
                                 padding_width_right,
                                 padding_width_left}));
    depthwise_conv_operator->set_attr_dilation(
        ge::AttrValue::LIST_INT({dilation_height, dilation_width}));
    depthwise_conv_operator->set_attr_stride(
        ge::AttrValue::LIST_INT({stride_height, stride_width}));
    depthwise_conv_operator->set_attr_kernel(
        ge::AttrValue::LIST_INT({filter_height, filter_width}));
    // ConvolutionDepthwise doesn't support bias, so append Add operator to
    // support bias
    auto add_operator = AddOperator<ge::op::Add>(output_operand);
    add_operator->set_input_x1(*depthwise_conv_operator);
    add_operator->set_input_x2(*bias_operator);
    conv_operator = add_operator;
  } else {
    auto normal_conv_operator =
        AddOperator<ge::op::Convolution>(output_operand);
    normal_conv_operator->set_input_x(*input_operator);
    normal_conv_operator->set_input_w(*filter_operator);
    normal_conv_operator->set_attr_mode(1);
    normal_conv_operator->set_attr_pad_mode(0);  // NOTSET
    normal_conv_operator->set_attr_group(group);
    normal_conv_operator->set_attr_pad(
        ge::AttrValue::LIST_INT({padding_height_bottom,
                                 padding_height_top,
                                 padding_width_right,
                                 padding_width_left}));
    normal_conv_operator->set_attr_dilation(
        ge::AttrValue::LIST_INT({dilation_height, dilation_width}));
    normal_conv_operator->set_attr_stride(
        ge::AttrValue::LIST_INT({stride_height, stride_width}));
    normal_conv_operator->set_attr_kernel(
        ge::AttrValue::LIST_INT({filter_height, filter_width}));
    // Convolution only support bias with dimension {1, oc, 1, 1},
    normal_conv_operator->set_input_b(*bias_operator);
    conv_operator = normal_conv_operator;
  }
  if (fuse_code != NNADAPTER_FUSED_NONE) {
    auto act_operator = AddOperator<ge::op::Activation>(output_operand);
    act_operator->set_input_x(*conv_operator);
    act_operator->set_attr_mode(ConvertFuseCode(fuse_code));
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
