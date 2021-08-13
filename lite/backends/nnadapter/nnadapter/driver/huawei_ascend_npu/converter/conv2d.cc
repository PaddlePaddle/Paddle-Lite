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

#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

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

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }
  // Check depthwise mode, and decide whether use ConvolutionDepthwise
  std::shared_ptr<Operator> filter_operator = nullptr;
  bool use_depthwise_conv = false;  // Whether use ge::op::DepthwiseConv2D ?
  if (is_depthwise_mode && dilation_width == 1 && dilation_height == 1) {
    use_depthwise_conv = true;
    // [C_out, 1, filter_height, filter_width] -> [1, C_out, filter_height,
    // filter_width]
    NNADAPTER_CHECK_EQ(filter_channel_size, 1);
    filter_operator = ConvertOperand(
        filter_operand,
        std::vector<int32_t>(
            {1, output_channel_size, filter_height, filter_width}));
  } else {
    filter_operator = ConvertOperand(filter_operand);
  }
  NNADAPTER_CHECK_EQ(bias_operand->type.dimension_count, 1);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimensions[0], output_channel_size);
  auto bias_operator = ConvertOperand(bias_operand);
  auto conv_name = GetOperatorName(output_operand);
  std::shared_ptr<Operator> conv_operator = nullptr;
  if (use_depthwise_conv && is_depthwise_mode) {
    auto depthwise_conv_op =
        std::make_shared<ge::op::DepthwiseConv2D>(conv_name);
    depthwise_conv_op->set_attr_pads(ge::Operator::OpListInt(
        {pad_height_top, pad_height_bottom, pad_width_left, pad_width_right}));
    depthwise_conv_op->set_attr_dilations(
        ge::Operator::OpListInt({1, 1, dilation_height, dilation_width}));
    depthwise_conv_op->set_attr_strides(
        ge::Operator::OpListInt({1, 1, stride_height, stride_width}));
    depthwise_conv_op->set_attr_data_format("NCHW");
    SET_INPUT(depthwise_conv_op, x, input_operator);
    SET_INPUT(depthwise_conv_op, filter, filter_operator);
    SET_INPUT(depthwise_conv_op, bias, bias_operator);
    conv_operator = MAP_OUTPUT(depthwise_conv_op, y, output_operand);
  } else {
    auto normal_conv_op = std::make_shared<ge::op::Conv2D>(conv_name);
    normal_conv_op->set_attr_pads(ge::Operator::OpListInt(
        {pad_height_top, pad_height_bottom, pad_width_left, pad_width_right}));
    normal_conv_op->set_attr_dilations(
        ge::Operator::OpListInt({1, 1, dilation_height, dilation_width}));
    normal_conv_op->set_attr_strides(
        ge::Operator::OpListInt({1, 1, stride_height, stride_width}));
    normal_conv_op->set_attr_groups(group);
    normal_conv_op->set_attr_data_format("NCHW");
    SET_INPUT(normal_conv_op, x, input_operator);
    SET_INPUT(normal_conv_op, filter, filter_operator);
    SET_INPUT(normal_conv_op, bias, bias_operator);
    conv_operator = MAP_OUTPUT(normal_conv_op, y, output_operand);
  }
  // fuse activations ?
  auto act_name = GetOperatorName(output_operand);
  switch (fuse_code) {
#define CONVERT_UNARY_ACTIVATION(type, class_name)                \
  case NNADAPTER_FUSED_##type: {                                  \
    auto act_op = std::make_shared<ge::op::class_name>(act_name); \
    SET_INPUT(act_op, x, conv_operator);                          \
    MAP_OUTPUT(act_op, y, output_operand);                        \
  } break;
    CONVERT_UNARY_ACTIVATION(RELU, Relu);
    CONVERT_UNARY_ACTIVATION(RELU6, Relu6);
#undef CONVERT_UNARY_ACTIVATION
    case NNADAPTER_FUSED_NONE:
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                           << ") is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
