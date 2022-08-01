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
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "operation/conv2d_transpose.h"
#include "driver/intel_openvino/converter/converter.h"
#include "operation/conv2d.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertConv2DTranspose(Converter* converter, core::Operation* operation) {
  CONV_2D_TRANSPOSE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  NNADAPTER_CHECK_EQ((output_channel_size % group), 0);
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
  auto IsOutputSet = [=]() -> bool {
    return output_shape_height != -1 && output_shape_width != -1;
  };

  // Convert operand to OpenVINO tensor
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto filter_tensor = converter->ConvertOperand(filter_operand);
  auto ov_strides = ov::Strides(
      {static_cast<size_t>(stride_height), static_cast<size_t>(stride_width)});
  auto ov_diliations = ov::Strides({static_cast<size_t>(dilation_height),
                                    static_cast<size_t>(dilation_width)});
  auto ov_pads_begin =
      ov::CoordinateDiff({static_cast<std::ptrdiff_t>(pad_height_top),
                          static_cast<std::ptrdiff_t>(pad_width_left)});
  auto ov_pads_end =
      ov::CoordinateDiff({static_cast<std::ptrdiff_t>(pad_height_bottom),
                          static_cast<std::ptrdiff_t>(pad_width_right)});
  auto output_padding =
      ov::CoordinateDiff({static_cast<std::ptrdiff_t>(output_padding_height),
                          static_cast<std::ptrdiff_t>(output_padding_width)});
  auto output_shape = ov::Shape({static_cast<size_t>(output_shape_height),
                                 static_cast<size_t>(output_shape_width)});
  auto output_shape_tensor = converter->AddConstantTensor<int32_t>(
      {output_shape_height, output_shape_width});
  std::shared_ptr<Tensor> output_tensor;

  // For group > 1, reshape filter.
  // Filters' layout is [I,O,W,H].
  // The final grouped filters' layout is [groups, grouped_i, O, W, H].
  std::shared_ptr<Operator> reshape_op;
  if (group > 1) {
    if (IsOperandWithDynamicShape(input_operand)) {
      reshape_op = converter->GetGroupConvFilterShape(filter_tensor, group);
    } else {
      auto filter_reshape_tensor =
          converter->AddConstantTensor<int64_t>({group,
                                                 filter_channel_size / group,
                                                 output_channel_size,
                                                 filter_height,
                                                 filter_width});
      reshape_op = std::make_shared<default_opset::Reshape>(
          *filter_tensor, *filter_reshape_tensor, false);
    }
    auto conv2d_transpose_op =
        IsOutputSet()
            ? std::make_shared<default_opset::GroupConvolutionBackpropData>(
                  *input_tensor,
                  reshape_op->output(0),
                  *output_shape_tensor,
                  ov_strides,
                  ov_pads_begin,
                  ov_pads_end,
                  ov_diliations,
                  ov::op::PadType::EXPLICIT,
                  output_padding)
            : std::make_shared<default_opset::GroupConvolutionBackpropData>(
                  *input_tensor,
                  reshape_op->output(0),
                  ov_strides,
                  ov_pads_begin,
                  ov_pads_end,
                  ov_diliations,
                  ov::op::PadType::EXPLICIT,
                  output_padding);
    output_tensor = MAP_OUTPUT(output_operand, conv2d_transpose_op, 0);
  } else {
    auto conv2d_transpose_op =
        IsOutputSet()
            ? std::make_shared<default_opset::ConvolutionBackpropData>(
                  *input_tensor,
                  *filter_tensor,
                  *output_shape_tensor,
                  ov_strides,
                  ov_pads_begin,
                  ov_pads_end,
                  ov_diliations,
                  ov::op::PadType::EXPLICIT,
                  output_padding)
            : std::make_shared<default_opset::ConvolutionBackpropData>(
                  *input_tensor,
                  *filter_tensor,
                  ov_strides,
                  ov_pads_begin,
                  ov_pads_end,
                  ov_diliations,
                  ov::op::PadType::EXPLICIT,
                  output_padding);
    output_tensor = MAP_OUTPUT(output_operand, conv2d_transpose_op, 0);
  }
  // Bias
  auto bias_tensor = converter->ConvertOperand(bias_operand);
  auto unsqueeze_op = converter->AddUnsqueezeOperator(
      bias_tensor, std::vector<int64_t>({0, 2, 3}));
  auto add_op = std::make_shared<default_opset::Add>(*output_tensor,
                                                     unsqueeze_op->output(0));
  output_tensor = MAP_OUTPUT(output_operand, add_op, 0);
  // Fuse activation
  std::shared_ptr<Operator> act_op;
  switch (fuse_code) {
#define CONVERT_UNARY_ACTIVATION(type, class_name)                        \
  case NNADAPTER_FUSED_##type: {                                          \
    act_op = std::make_shared<default_opset::class_name>(*output_tensor); \
    MAP_OUTPUT(output_operand, act_op, 0);                                \
  } break;
    CONVERT_UNARY_ACTIVATION(RELU, Relu);
#undef CONVERT_UNARY_ACTIVATION
    case NNADAPTER_FUSED_RELU6:
      act_op =
          std::make_shared<default_opset::Clamp>(*output_tensor, 0.0f, 6.0f);
      MAP_OUTPUT(output_operand, act_op, 0);
      break;
    case NNADAPTER_FUSED_NONE:
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                           << ") is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace intel_openvino
}  // namespace nnadapter
