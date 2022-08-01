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
#include "driver/cambricon_mlu/converter.h"
#include "operation/conv2d.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

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

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (input_tensor == nullptr) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto filter_tensor = converter->GetMappedTensor(filter_operand);
  if (filter_tensor == nullptr) {
    filter_tensor = converter->ConvertOperand(filter_operand);
  }
  auto bias_tensor = converter->GetMappedTensor(bias_operand);
  if (bias_tensor == nullptr) {
    bias_tensor = converter->ConvertOperand(bias_operand);
  }

  magicmind::ITensor* output_shape_tensor = nullptr;
  std::vector<float> output_shape_vec = {};
  if (input_operands[9] != nullptr) {
    output_shape_vec.push_back(output_operand->type.dimensions.data[0]);
    output_shape_vec.push_back(output_operand->type.dimensions.data[1]);
    output_shape_vec.push_back(static_cast<float>(output_shape_height));
    output_shape_vec.push_back(static_cast<float>(output_shape_width));
    output_shape_tensor =
        converter->AddFloat32ConstantTensor(output_shape_vec.data(), {2});
  }
  auto conv2d_transpose_node = converter->network()->AddIDeconvNode(
      input_tensor, filter_tensor, bias_tensor, output_shape_tensor);
  NNADAPTER_CHECK(conv2d_transpose_node)
      << "Failed to add convolution transpose node.";

  magicmind::Layout layout =
      ConvertToMagicMindDataLayout(input_operand->type.layout);
  conv2d_transpose_node->SetLayout(layout, layout, layout);
  conv2d_transpose_node->SetStride(static_cast<int64_t>(stride_height),
                                   static_cast<int64_t>(stride_width));
  conv2d_transpose_node->SetPad(static_cast<int64_t>(pad_height_top),
                                static_cast<int64_t>(pad_height_bottom),
                                static_cast<int64_t>(pad_width_left),
                                static_cast<int64_t>(pad_width_right));
  conv2d_transpose_node->SetDilation(static_cast<int64_t>(dilation_height),
                                     static_cast<int64_t>(dilation_width));
  conv2d_transpose_node->SetOutpad(static_cast<int64_t>(output_padding_height));
  conv2d_transpose_node->SetGroup(static_cast<int64_t>(group));
  auto output_tensor = conv2d_transpose_node->GetOutput(0);
  // fuse activations
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
