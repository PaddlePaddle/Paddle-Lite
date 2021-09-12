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

#include "core/operation/conv2d_transpose.h"
#include "driver/huawei_ascend_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

int Program::ConvertConv2DTranspose(hal::Operation* operation) {
  CONV_2D_TRANSPOSE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  auto out_dims = output_operand->type.dimensions;
  NNADAPTER_CHECK_NE(out_dims[2], NNADAPTER_UNKNOWN)
      << "AscendNPU must set out shape.";
  NNADAPTER_CHECK_NE(out_dims[3], NNADAPTER_UNKNOWN)
      << "AscendNPU must set out shape.";
  // Group of AscendNPU may be different from paddle.
  NNADAPTER_CHECK_EQ(group, 1);

  // Convert to GE operators
  auto input_operator = GetMappedOperator(input_operand);
  if (input_operator == nullptr) {
    input_operator = ConvertOperand(input_operand);
  }
  auto filter_operator = GetMappedOperator(filter_operand);
  if (filter_operator == nullptr) {
    filter_operator = ConvertOperand(filter_operand);
  }
  auto bias_operator = GetMappedOperator(bias_operand);
  if (bias_operator == nullptr) {
    bias_operator = ConvertOperand(bias_operand);
  }
  auto conv2d_transpose_name = GetOperatorName(output_operand);
  auto conv2d_transpose_op =
      std::make_shared<ge::op::Conv2DTransposeD>(conv2d_transpose_name);
  SET_INPUT(conv2d_transpose_op, x, input_operator);
  SET_INPUT(conv2d_transpose_op, filter, filter_operator);
  SET_INPUT(conv2d_transpose_op, bias, bias_operator);
  MAP_OUTPUT(conv2d_transpose_op, y, output_operand);
  conv2d_transpose_op->set_attr_input_size(
      ge::Operator::OpListInt({input_operand->type.dimensions[0],
                               output_channel_size,
                               out_dims[2],
                               out_dims[3]}));
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
  return NNADAPTER_NO_ERROR;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
