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

#include "operation/deformable_conv2d.h"
#include "driver/huawei_ascend_npu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

/**
 * fuse deformable_offset and conv2d as deformable_conv2d
 *
 * [input] [offsets]  [mask]
 *    \             \ /
 *     \             |
 *      \          concat
 *       \         /
 *        \       /
 * deformable_offset(input) [filter]  [bias]
 *                       \     |     /
 *                           \   /
 *                           conv2d
 *                             |
 *                      deformable_conv2d
 */
int ConvertDeformableConv2d(Converter* converter, core::Operation* operation) {
  DEFORMABLE_CONV_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to GE operators
  auto input_operator = converter->GetMappedOperator(input_operand);
  if (!input_operator) {
    input_operator = converter->ConvertOperand(input_operand);
  }
  auto offset_operator = converter->GetMappedOperator(offset_operand);
  if (!offset_operator) {
    offset_operator = converter->ConvertOperand(offset_operand);
  }
  auto mask_operator = converter->GetMappedOperator(mask_operand);
  if (!mask_operator) {
    mask_operator = converter->ConvertOperand(mask_operand);
  }

  /**
   * Convert the offsets data arrangement in deformable_offsets
   *
   * NNAdapter:     yyyyy          ->      Ascend:   xxxxx
   *                xxxxx                            xxxxx
   *                yyyyy                            xxxxx
   *                xxxxx                            yyyyy
   *                yyyyy                            yyyyy
   *                xxxxx                            yyyyy
   *
   */
  // Slice offset_x in the channel dimension
  auto x_begin_operator =
      converter->AddInt32ConstantOperator(std::vector<int32_t>({1}));
  auto offset_channel = offset_operand->type.dimensions.data[1];
  auto x_end_operator = converter->AddInt32ConstantOperator(
      std::vector<int32_t>({offset_channel}));
  auto x_strides_operator = converter->AddInt32ConstantOperator(2);
  auto x_axes_operator = converter->AddInt32ConstantOperator(1);
  auto slice_x_op =
      converter->AddOperator<ge::op::StridedSliceV2>(output_operand, "split_x");
  SET_INPUT(slice_x_op, x, offset_operator);
  SET_INPUT(slice_x_op, begin, x_begin_operator);
  SET_INPUT(slice_x_op, end, x_end_operator);
  SET_INPUT(slice_x_op, strides, x_strides_operator);
  SET_INPUT(slice_x_op, axes, x_axes_operator);
  auto slice_x_operator = MAP_OUTPUT(slice_x_op, y, output_operand);

  // Slice offset_y in the channel dimension
  auto y_begin_operator =
      converter->AddInt32ConstantOperator(std::vector<int32_t>({0}));
  auto y_end_operator = converter->AddInt32ConstantOperator(
      std::vector<int32_t>({offset_channel - 1}));
  auto y_strides_operator = converter->AddInt32ConstantOperator(2);
  auto y_axes_operator = converter->AddInt32ConstantOperator(1);
  auto slice_y_op =
      converter->AddOperator<ge::op::StridedSliceV2>(output_operand, "split_y");
  SET_INPUT(slice_y_op, x, offset_operator);
  SET_INPUT(slice_y_op, begin, y_begin_operator);
  SET_INPUT(slice_y_op, end, y_end_operator);
  SET_INPUT(slice_y_op, strides, y_strides_operator);
  SET_INPUT(slice_y_op, axes, y_axes_operator);
  auto slice_y_operator = MAP_OUTPUT(slice_y_op, y, output_operand);

  // Concat the offset and mask in the channel dimension
  auto concat_op =
      converter->AddOperator<ge::op::ConcatD>(output_operand, "concat");
  concat_op->set_attr_concat_dim(1);
  concat_op->set_attr_N(3);
  concat_op->create_dynamic_input_x(3);
  SET_DYNAMIC_INPUT(concat_op, x, 0, slice_x_operator);
  SET_DYNAMIC_INPUT(concat_op, x, 1, slice_y_operator);
  SET_DYNAMIC_INPUT(concat_op, x, 2, mask_operator);
  auto concat_operator = MAP_OUTPUT(concat_op, y, output_operand);

  /**
   * Create deformable_offsets operator
   *
   * deformable_offset args:
   *   tensor:
   *      x:          use input_operand
   *      offsets:    use concat_operator output
   *   attrs:
   *      [ksize]:                {filter_height, filter_width}
   *      [strides]:              {1, 1, stride_height, stride_width}
   *      [paddings]:             {padding_height_top,
   *                               padding_height_bottom,
   *                               padding_width_left,
   *                               padding_width_right}
   *      [dilations]:            {1, 1, dilation_height, dilation_width}
   *      [data_format]:          NCHW
   *      [deformable_groups]:    deformable_groups
   *      [modulated]:            true
   */
  auto deformable_offsets_op =
      converter->AddOperator<ge::op::DeformableOffsets>(output_operand);
  deformable_offsets_op->set_attr_strides(
      ge::Operator::OpListInt({1, 1, strides[0], strides[1]}));
  deformable_offsets_op->set_attr_pads(
      ge::Operator::OpListInt(pads.begin(), pads.end()));
  deformable_offsets_op->set_attr_ksize(
      ge::Operator::OpListInt({filter_height, filter_width}));
  deformable_offsets_op->set_attr_dilations(
      ge::Operator::OpListInt({1, 1, dilations[0], dilations[1]}));
  deformable_offsets_op->set_attr_deformable_groups(deformable_group);
  deformable_offsets_op->set_attr_data_format("NCHW");
  deformable_offsets_op->set_attr_modulated(true);
  SET_INPUT(deformable_offsets_op, x, input_operator);
  SET_INPUT(deformable_offsets_op, offsets, concat_operator);
  auto deformable_offsets_operator =
      MAP_OUTPUT(deformable_offsets_op, y, output_operand);

  /** Create deformable_offsets operator
   *
   * conv2d args:
   *   tensor:
   *      x:              use deformable_offsets operator output
   *      filter:         use filter operator
   *      bias:           use bias operator
   *   attrs:
   *      [strides]:      {1, 1, filter_height, filter_width}
   *      [pads]:         defualt:{0, 0, 0, 0}
   *      [dilations]:    defualt:{1, 1, 1, 1}
   *      [data_format]:  NCHW
   */
  auto filter_operator = converter->ConvertOperand(filter_operand);
  auto bias_operator = converter->ConvertOperand(bias_operand);
  auto conv_op = converter->AddOperator<ge::op::Conv2D>(output_operand);
  conv_op->set_attr_pads(ge::Operator::OpListInt({0, 0, 0, 0}));
  conv_op->set_attr_dilations(ge::Operator::OpListInt({1, 1, 1, 1}));
  conv_op->set_attr_strides(
      ge::Operator::OpListInt({1, 1, filter_height, filter_width}));
  conv_op->set_attr_data_format("NCHW");
  SET_INPUT(conv_op, x, deformable_offsets_operator);
  SET_INPUT(conv_op, filter, filter_operator);
  SET_INPUT(conv_op, bias, bias_operator);
  auto conv_operator = MAP_OUTPUT(conv_op, y, output_operand);

  // Fuse activations
  switch (fuse_code) {
#define CONVERT_UNARY_ACTIVATION(type, class_name)                            \
  case NNADAPTER_FUSED_##type: {                                              \
    auto act_op = converter->AddOperator<ge::op::class_name>(output_operand); \
    SET_INPUT(act_op, x, conv_operator);                                      \
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
