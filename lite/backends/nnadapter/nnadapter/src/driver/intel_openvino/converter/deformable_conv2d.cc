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
#include "driver/intel_openvino/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace intel_openvino {

int ConvertDeformableConv2d(Converter* converter, core::Operation* operation) {
  DEFORMABLE_CONV_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  NNADAPTER_CHECK_EQ(dilations[0], dilations[1])
      << "Only supports dilations[0] == dilations[1] !";
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto offset_tensor = converter->GetMappedTensor(offset_operand);
  if (!offset_tensor) {
    offset_tensor = converter->ConvertOperand(offset_operand);
  }
  auto mask_tensor = converter->GetMappedTensor(mask_operand);
  if (!mask_tensor) {
    mask_tensor = converter->ConvertOperand(mask_operand);
  }
  auto filter_tensor = converter->ConvertOperand(filter_operand);
  auto bias_tensor = converter->ConvertOperand(bias_operand);
  auto ov_strides = ov::Strides(
      {static_cast<size_t>(strides[0]), static_cast<size_t>(strides[1])});
  auto ov_diliations = ov::Strides(
      {static_cast<size_t>(dilations[0]), static_cast<size_t>(dilations[1])});
  auto ov_pads_begin =
      ov::CoordinateDiff({static_cast<std::ptrdiff_t>(pads[0]),
                          static_cast<std::ptrdiff_t>(pads[2])});
  auto ov_pads_end = ov::CoordinateDiff({static_cast<std::ptrdiff_t>(pads[1]),
                                         static_cast<std::ptrdiff_t>(pads[3])});
  auto deformable_conv2d_op =
      std::make_shared<default_opset::DeformableConvolution>(
          *input_tensor,
          *offset_tensor,
          *filter_tensor,
          *mask_tensor,
          ov_strides,
          ov_pads_begin,
          ov_pads_end,
          ov_diliations,
          ov::op::PadType::EXPLICIT,
          group,
          deformable_group,
          true);
  auto unsqueeze_op = converter->AddUnsqueezeOperator(
      bias_tensor, std::vector<int64_t>({0, 2, 3}));
  auto add_op = std::make_shared<default_opset::Add>(
      deformable_conv2d_op->output(0), unsqueeze_op->output(0));
  auto output_tensor = MAP_OUTPUT(output_operand, add_op, 0);
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
