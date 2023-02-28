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

#include "operation/conv2d_transpose.h"
#include "driver/nvidia_tensorrt/converter/converter.h"
#include "operation/conv2d.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int ConvertConv2DTranspose(Converter* converter, core::Operation* operation) {
  CONV_2D_TRANSPOSE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // TensorRT doesn't support output_channel_size % groups != 0 case
  NNADAPTER_CHECK_EQ((output_channel_size % group), 0);
  NNADAPTER_CHECK_EQ(fuse_code, NNADAPTER_FUSED_NONE);
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

  // Convert to trt tensors and node
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto weight = converter->OperandToWeights(filter_operand);
  auto bias = converter->OperandToWeights(bias_operand);
  auto conv_layer = converter->network()->addDeconvolutionNd(
      *input_tensor,
      output_channel_size,
      nvinfer1::Dims2(filter_height, filter_width),
      weight,
      bias);
  NNADAPTER_CHECK(conv_layer);
  conv_layer->setStrideNd(nvinfer1::Dims2(stride_height, stride_width));
  conv_layer->setPrePadding(nvinfer1::Dims2(pad_height_top, pad_width_left));
  conv_layer->setPostPadding(
      nvinfer1::Dims2(pad_height_bottom, pad_width_right));
  conv_layer->setDilationNd(nvinfer1::Dims2(dilation_height, dilation_width));
  conv_layer->setNbGroups(group);
  auto output_tensor = conv_layer->getOutput(0);
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
