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
#include "driver/huawei_kirin_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int Program::ConvertConv2D(hal::Operation* operation) {
  CONV2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = ConvertOperand(input_operand);
  }
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
  auto filter_operator = ConvertOperand(filter_operand);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimension_count, 1);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimensions[0], output_channel_size);
  auto bias_operator = ConvertOperand(bias_operand);
  auto conv_name = GetOperatorName(output_operand);
  std::shared_ptr<Operator> conv_operator = nullptr;
  if (use_depthwise_conv && is_depthwise_mode) {
    auto depthwise_conv_op =
        std::make_shared<hiai::op::ConvolutionDepthwise>(conv_name);
    depthwise_conv_op->set_attr_pad_mode("SPECIFIC");
    depthwise_conv_op->set_attr_pads(ge::AttrValue::LIST_INT(
        {pad_height_top, pad_height_bottom, pad_width_left, pad_width_right}));
    depthwise_conv_op->set_attr_dilations(
        ge::AttrValue::LIST_INT({dilation_height, dilation_width}));
    depthwise_conv_op->set_attr_strides(
        ge::AttrValue::LIST_INT({stride_height, stride_width}));
    SET_INPUT(depthwise_conv_op, x, input_operator);
    SET_INPUT(depthwise_conv_op, filter, filter_operator);
    SET_INPUT(depthwise_conv_op, bias, bias_operator);
    conv_operator = MAP_OUTPUT(depthwise_conv_op, y, output_operand);
  } else {
    auto normal_conv_op = std::make_shared<hiai::op::Convolution>(conv_name);
    normal_conv_op->set_attr_pad_mode("SPECIFIC");
    normal_conv_op->set_attr_pads(ge::AttrValue::LIST_INT(
        {pad_height_top, pad_height_bottom, pad_width_left, pad_width_right}));
    normal_conv_op->set_attr_dilations(
        ge::AttrValue::LIST_INT({dilation_height, dilation_width}));
    normal_conv_op->set_attr_strides(
        ge::AttrValue::LIST_INT({stride_height, stride_width}));
    normal_conv_op->set_attr_groups(group);
    SET_INPUT(normal_conv_op, x, input_operator);
    SET_INPUT(normal_conv_op, filter, filter_operator);
    SET_INPUT(normal_conv_op, bias, bias_operator);
    conv_operator = MAP_OUTPUT(normal_conv_op, y, output_operand);
  }
  if (fuse_code != NNADAPTER_FUSED_NONE) {
    auto act_name = GetOperatorName(output_operand);
    auto act_op = std::make_shared<hiai::op::Activation>(act_name);
    act_op->set_attr_mode(ConvertFuseCode(fuse_code));
    SET_INPUT(act_op, x, conv_operator);
    MAP_OUTPUT(act_op, y, output_operand);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
