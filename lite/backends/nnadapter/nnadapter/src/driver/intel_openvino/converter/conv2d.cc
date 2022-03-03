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

#include "operation/conv2d.h"
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertConv2D(Converter* converter, core::Operation* operation) {
  CONV_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS
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

  // Convert operand to OpenVINO tensor
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto filter_tensor = converter->ConvertOperand(filter_operand);
  auto ov_auto_pad = ConvertToOVPadType(auto_pad);
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
  auto conv2d_op = std::make_shared<default_opset::Convolution>(*input_tensor,
                                                                *filter_tensor,
                                                                ov_strides,
                                                                ov_pads_begin,
                                                                ov_pads_end,
                                                                ov_diliations,
                                                                ov_auto_pad);
  auto output_tensor = MAP_OUTPUT(output_operand, conv2d_op, 0);
  // Bias
  auto bias_tensor = converter->ConvertOperand(bias_operand);
  auto unsqueeze_op = converter->AddUnsqueezeOperator(
      bias_tensor, std::vector<int64_t>({0, 2, 3}));
  auto add_op = std::make_shared<default_opset::Add>(*output_tensor,
                                                     unsqueeze_op->output(0));
  output_tensor = MAP_OUTPUT(output_operand, add_op, 0);
  // Fuse activation
  switch (fuse_code) {
#define CONVERT_UNARY_ACTIVATION(type, class_name)                             \
  case NNADAPTER_FUSED_##type: {                                               \
    auto act_op = std::make_shared<default_opset::class_name>(*output_tensor); \
    MAP_OUTPUT(output_operand, act_op, 0);                                     \
  } break;
    CONVERT_UNARY_ACTIVATION(RELU, Relu);
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

}  // namespace intel_openvino
}  // namespace nnadapter
