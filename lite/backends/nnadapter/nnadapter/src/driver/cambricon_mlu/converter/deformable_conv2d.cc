// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertDeformableConv2d(Converter* converter, core::Operation* operation) {
  DEFORMABLE_CONV_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto filter_tensor = converter->GetMappedTensor(filter_operand);
  if (!filter_tensor) {
    filter_tensor = converter->ConvertOperand(filter_operand);
  }
  auto offset_tensor = converter->GetMappedTensor(offset_operand);
  if (!offset_tensor) {
    offset_tensor = converter->ConvertOperand(offset_operand);
  }
  auto mask_tensor = converter->GetMappedTensor(mask_operand);
  if (!mask_tensor) {
    mask_tensor = converter->ConvertOperand(mask_operand);
  }
  auto bias_tensor = converter->GetMappedTensor(bias_operand);
  if (!bias_tensor) {
    bias_tensor = converter->ConvertOperand(bias_operand);
  }
  auto deformconv_node = converter->network()->AddIDeformConv2DNode(
      input_tensor, filter_tensor, offset_tensor, mask_tensor, bias_tensor);
  NNADAPTER_CHECK(deformconv_node) << "Failed to add deformconv node.";
  magicmind::Layout layout =
      ConvertToMagicMindDataLayout(input_operand->type.layout);
  deformconv_node->SetLayout(layout, layout, layout, layout, layout);
  deformconv_node->SetStride(static_cast<int64_t>(strides_buffer[0]),
                             static_cast<int64_t>(strides_buffer[1]));
  deformconv_node->SetPad(static_cast<int64_t>(pads[0]),
                          static_cast<int64_t>(pads[1]),
                          static_cast<int64_t>(pads[2]),
                          static_cast<int64_t>(pads[3]));
  deformconv_node->SetDilation(static_cast<int64_t>(dilations[0]),
                               static_cast<int64_t>(dilations[1]));

  auto output_tensor = deformconv_node->GetOutput(0);
  // fuse activations ?
  switch (fuse_code) {
#define CONVERT_ACTIVATION(type, mm_type)                                 \
  case NNADAPTER_FUSED_##type: {                                          \
    auto activation_node =                                                \
        converter->network()->AddIActivationNode(output_tensor, mm_type); \
    auto fuse_out_tensor = activation_node->GetOutput(0);                 \
    converter->UpdateTensorMap(output_operand, fuse_out_tensor);          \
    break;                                                                \
  }
    CONVERT_ACTIVATION(RELU, magicmind::IActivation::RELU);
    CONVERT_ACTIVATION(RELU6, magicmind::IActivation::RELU6);
#undef CONVERT_ACTIVATION
    case NNADAPTER_FUSED_NONE:
      converter->UpdateTensorMap(output_operand, output_tensor);
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                           << ") is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
