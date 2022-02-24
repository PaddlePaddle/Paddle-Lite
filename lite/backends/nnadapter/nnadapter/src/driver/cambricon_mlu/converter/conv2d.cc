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
#include "driver/cambricon_mlu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace cambricon_mlu {

int ConvertConv2D(Converter* converter, core::Operation* operation) {
  CONV_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to magicmind tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  // Check depthwise mode, and decide whether use ConvolutionDepthwise
  bool use_depthwise_conv = false;
  auto filter_tensor = converter->ConvertOperand(filter_operand);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimensions.count, 1);
  NNADAPTER_CHECK_EQ(bias_operand->type.dimensions.data[0],
                     output_channel_size);
  magicmind::ITensor* bias_tensor = nullptr;
  auto bias_tmp_tensor = converter->ConvertOperand(bias_operand);
  if (input_operand->type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_LAYER) {
    auto cast_node = converter->network()->AddICastNode(
        bias_tmp_tensor, magicmind::DataType::FLOAT32);
    auto cast_out_tensor = cast_node->GetOutput(0);

    float bias_scale = bias_operand->type.symm_per_layer_params.scale;
    auto scale_tensor = converter->AddFloat32ConstantTensor(&bias_scale, {1});
    auto dequantize_node = converter->network()->AddIElementwiseNode(
        cast_out_tensor, scale_tensor, magicmind::IElementwise::MUL);
    auto dequant_out_tensor = dequantize_node->GetOutput(0);
    bias_tensor = dequant_out_tensor;
  } else {
    bias_tensor = bias_tmp_tensor;
  }

  if (use_depthwise_conv && is_depthwise_mode) {
  } else {
    auto conv_node = converter->network()->AddIConvNode(
        input_tensor, filter_tensor, bias_tensor);
    NNADAPTER_CHECK(conv_node) << "Failed to add convolution node.";
    auto pre_h = pads_buffer[0];
    auto post_h = pads_buffer[1];
    auto pre_w = pads_buffer[2];
    auto post_w = pads_buffer[3];
    conv_node->SetPad(static_cast<int64_t>(pre_h),
                      static_cast<int64_t>(post_h),
                      static_cast<int64_t>(pre_w),
                      static_cast<int64_t>(post_w));
    conv_node->SetStride(static_cast<int64_t>(stride_height),
                         static_cast<int64_t>(stride_width));
    conv_node->SetDilation(static_cast<int64_t>(dilation_height),
                           static_cast<int64_t>(dilation_width));
    conv_node->SetGroup(static_cast<int64_t>(group));
    magicmind::Layout input_layout =
        ConvertToMagicMindDataLayout(input_operand->type.layout);
    conv_node->SetLayout(input_layout, input_layout, input_layout);
    if (input_operand->type.precision == NNADAPTER_QUANT_INT8_SYMM_PER_LAYER) {
      float input_scale = input_operand->type.symm_per_layer_params.scale;
      auto input_tensor_range = magicmind::UniformQuantParamToRangeWithQuantAlg(
          {input_scale, 0}, 8, "symmetric");
      auto input = conv_node->GetInput(0);
      input->SetDynamicRange(input_tensor_range, true);

      float filter_scale = filter_operand->type.symm_per_layer_params.scale;
      auto filter_tensor_range =
          magicmind::UniformQuantParamToRangeWithQuantAlg(
              {filter_scale, 0}, 8, "symmetric");
      auto filter = conv_node->GetInput(1);
      filter->SetDynamicRange(filter_tensor_range, true);
    }

    auto output_tensor = conv_node->GetOutput(0);
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
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace cambricon_mlu
}  // namespace nnadapter
