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
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "operation/conv2d.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int ConvertConv2DTranspose(Converter* converter, core::Operation* operation) {
  CONV_2D_TRANSPOSE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  auto output_dimensions = output_operand->type.dimensions.data;
  NNADAPTER_CHECK_NE(output_dimensions[0], NNADAPTER_UNKNOWN)
      << "output_dimensions[0] is unknown, dynamic shape is still not "
         "supported.";
  NNADAPTER_CHECK_NE(output_dimensions[1], NNADAPTER_UNKNOWN)
      << "output_dimensions[1] is unknown, dynamic shape is still not "
         "supported.";
  NNADAPTER_CHECK_NE(output_dimensions[2], NNADAPTER_UNKNOWN)
      << "output_dimensions[2] is unknown, dynamic shape is still not "
         "supported.";
  NNADAPTER_CHECK_NE(output_dimensions[3], NNADAPTER_UNKNOWN)
      << "output_dimensions[3] is unknown, dynamic shape is still not "
         "supported.";
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
  // Group only supports 1
  NNADAPTER_CHECK_EQ(group, 1);

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto filter_operator = converter->GetMappedOperator(filter_operand);
  if (!filter_operator) {
    filter_operator = converter->ConvertOperand(filter_operand);
  }
  auto bias_operator = converter->GetMappedOperator(bias_operand);
  if (!bias_operator) {
    bias_operator = converter->ConvertOperand(bias_operand);
  }
  auto conv2d_transpose_op =
      converter->AddOperator<ge::op::Conv2DTransposeD>(output_operand);
  SET_INPUT(conv2d_transpose_op, x, input_operator);
  SET_INPUT(conv2d_transpose_op, filter, filter_operator);
  SET_INPUT(conv2d_transpose_op, bias, bias_operator);
  auto conv2d_transpose_operator =
      MAP_OUTPUT(conv2d_transpose_op, y, output_operand);
  conv2d_transpose_op->set_attr_input_size(
      ge::Operator::OpListInt({output_dimensions[0],
                               output_dimensions[1],
                               output_dimensions[2],
                               output_dimensions[3]}));
  conv2d_transpose_op->set_attr_strides(
      ge::Operator::OpListInt({1, 1, stride_height, stride_width}));
  conv2d_transpose_op->set_attr_pads(ge::Operator::OpListInt(
      {pad_height_top, pad_height_bottom, pad_width_left, pad_width_right}));
  conv2d_transpose_op->set_attr_dilations(
      ge::Operator::OpListInt({1, 1, dilation_height, dilation_width}));
  conv2d_transpose_op->set_attr_groups(group);
  conv2d_transpose_op->set_attr_data_format("NCHW");
  conv2d_transpose_op->set_attr_output_padding(ge::Operator::OpListInt(
      {0, 0, output_padding_height, output_padding_width}));

  // fuse activations
  switch (fuse_code) {
#define CONVERT_UNARY_ACTIVATION(type, class_name)                            \
  case NNADAPTER_FUSED_##type: {                                              \
    auto act_op = converter->AddOperator<ge::op::class_name>(output_operand); \
    SET_INPUT(act_op, x, conv2d_transpose_operator);                          \
    MAP_OUTPUT(act_op, y, output_operand);                                    \
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
