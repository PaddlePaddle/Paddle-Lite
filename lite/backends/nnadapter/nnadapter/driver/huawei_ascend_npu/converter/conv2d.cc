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
#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertConv2D(hal::Operation* operation) {
  CONV2D_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Dynamic shapes are still not supported
  NNADAPTER_CHECK_EQ(input_operand->type.dynamic_dimension_count, 0);
  operation::UpdateConv2DPadAndDilation(input_operand->type.dimensions[2],
                                        filter_height,
                                        auto_pad,
                                        &pad_height_top,
                                        &pad_height_bottom,
                                        stride_height,
                                        &dilation_height);
  operation::UpdateConv2DPadAndDilation(input_operand->type.dimensions[3],
                                        filter_width,
                                        auto_pad,
                                        &pad_width_left,
                                        &pad_width_right,
                                        stride_width,
                                        &dilation_width);

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
