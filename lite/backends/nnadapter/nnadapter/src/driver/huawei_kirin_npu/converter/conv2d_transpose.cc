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

#include "operation/conv2d_transpose.h"
#include "driver/huawei_kirin_npu/converter/converter.h"
#include "operation/conv2d.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

int ConvertConv2DTranspose(Converter* converter, core::Operation* operation) {
  CONV_2D_TRANSPOSE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  if (auto_pad != NNADAPTER_AUTO_PAD_NONE) {
    operation::UpdateConv2DPadAndDilation(
        input_operand->type.dimensions.data[2],
        filter_height,
        auto_pad,
        &pad_height_top,
        &pad_height_bottom,
        stride_height,
        &dilation_height);
    operation::UpdateConv2DPadAndDilation(
        input_operand->type.dimensions.data[3],
        filter_width,
        auto_pad,
        &pad_width_left,
        &pad_width_right,
        stride_width,
        &dilation_width);
  }
  auto pad_mode = "";
  switch (auto_pad) {
    case 0:
      pad_mode = "SPECIFIC";
      break;
    case 1:
      pad_mode = "SAME";
      break;
    case 2:
      pad_mode = "VALID";
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Invalid pad mode!";
      break;
  }
  std::vector<int32_t> output_shape = {};
  for (int32_t i = 0; i < output_operand->type.dimensions.count; i++) {
    output_shape.push_back(output_operand->type.dimensions.data[i]);
  }
  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (input_operator == nullptr) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto filter_operator = converter->GetMappedOperator(filter_operand);
  if (filter_operator == nullptr) {
    filter_operator = converter->ConvertOperand(filter_operand);
  }
  auto bias_operator = converter->GetMappedOperator(bias_operand);
  if (bias_operator == nullptr) {
    bias_operator = converter->ConvertOperand(bias_operand);
  }
  auto output_shape_operator =
      converter->AddInt32ConstantOperator(output_shape);
  auto conv_transpose_op =
      converter->AddOperator<hiai::op::ConvTranspose>(output_operand);
  conv_transpose_op->set_attr_strides(
      ge::AttrValue::LIST_INT({stride_height, stride_width}));
  conv_transpose_op->set_attr_pads(ge::AttrValue::LIST_INT(
      {pad_height_top, pad_height_bottom, pad_width_left, pad_width_right}));
  conv_transpose_op->set_attr_pad_mode(pad_mode);
  conv_transpose_op->set_attr_dilations(
      ge::AttrValue::LIST_INT({dilation_height, dilation_width}));
  conv_transpose_op->set_attr_groups(group);
  SET_INPUT(conv_transpose_op, output_shape, output_shape_operator);
  SET_INPUT(conv_transpose_op, x, input_operator);
  SET_INPUT(conv_transpose_op, filter, filter_operator);
  SET_INPUT(conv_transpose_op, bias, bias_operator);
  auto conv_transpose_operator =
      MAP_OUTPUT(conv_transpose_op, y, output_operand);
  // fuse activations
  switch (fuse_code) {
#define CONVERT_UNARY_ACTIVATION(type, act_mode)                      \
  case NNADAPTER_FUSED_##type: {                                      \
    auto act_op =                                                     \
        converter->AddOperator<hiai::op::Activation>(output_operand); \
    act_op->set_attr_mode(act_mode);                                  \
    SET_INPUT(act_op, x, conv_transpose_operator);                    \
    MAP_OUTPUT(act_op, y, output_operand);                            \
  } break;
    CONVERT_UNARY_ACTIVATION(RELU, 1);
    CONVERT_UNARY_ACTIVATION(RELU1, 7);
    CONVERT_UNARY_ACTIVATION(RELU6, 14);
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

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
