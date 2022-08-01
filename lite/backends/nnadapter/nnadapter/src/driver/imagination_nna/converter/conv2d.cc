// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "driver/imagination_nna/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace imagination_nna {

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

  // Convert to imgdnn tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  auto filter_tensor = converter->ConvertOperand(filter_operand);
  // Expand bias tensor from (c) to (1, c)
  auto bias_tensor = converter->ConvertOperand(
      bias_operand, {1, bias_operand->type.dimensions.data[0]});
  unsigned int ksizes[2] = {static_cast<unsigned int>(filter_height),
                            static_cast<unsigned int>(filter_width)};
  unsigned int strides[2] = {static_cast<unsigned int>(stride_height),
                             static_cast<unsigned int>(stride_width)};
  // Top and left
  unsigned int pad_to_begin[2] = {static_cast<unsigned int>(pad_height_top),
                                  static_cast<unsigned int>(pad_width_left)};
  // Bottom and right
  unsigned int pad_to_end[2] = {static_cast<unsigned int>(pad_height_bottom),
                                static_cast<unsigned int>(pad_width_right)};
  unsigned int dilations[2] = {static_cast<unsigned int>(dilation_height),
                               static_cast<unsigned int>(dilation_width)};
  NNADAPTER_CHECK(
      IsUInt8AsymmPerLayerQuantType(output_operand->type.precision));
  imgdnn_quant_param output_quant_param;
  output_quant_param.scale = output_operand->type.asymm_per_layer_params.scale;
  output_quant_param.zero_point =
      output_operand->type.asymm_per_layer_params.zero_point;
  auto output_tensor = ADD_OPERATOR(CreateConvolutionLayer,
                                    input_tensor,
                                    filter_tensor,
                                    bias_tensor,
                                    output_quant_param,
                                    strides,
                                    pad_to_begin,
                                    pad_to_end,
                                    dilations,
                                    is_depthwise_mode);
  // fuse RELU ?
  if (fuse_code == NNADAPTER_FUSED_NONE) {
  } else if (fuse_code == NNADAPTER_FUSED_RELU) {
    output_tensor = ADD_OPERATOR(
        CreateReLULayer, output_tensor, true, 0.0, false, 0.0, false);
  } else if (fuse_code == NNADAPTER_FUSED_RELU1) {
    output_tensor = ADD_OPERATOR(
        CreateReLULayer, output_tensor, true, 0.0, true, 1.0, false);
  } else if (fuse_code == NNADAPTER_FUSED_RELU6) {
    output_tensor = ADD_OPERATOR(
        CreateReLULayer, output_tensor, true, 0.0, true, 6.0, false);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                         << ") is found.";
  }
  converter->UpdateTensorMap(output_operand, output_tensor);
  return NNADAPTER_NO_ERROR;
}

}  // namespace imagination_nna
}  // namespace nnadapter
