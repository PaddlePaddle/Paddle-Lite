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

#include "core/operation/conv2d_transpose.h"
#include "driver/amlogic_npu/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace amlogic_npu {

int Program::ConvertConv2DTranspose(hal::Operation* operation) {
  CONV_2D_TRANSPOSE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to amlnpu tensors and operators
  auto input_tensor = GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = ConvertOperand(input_operand);
  }
  std::shared_ptr<aml::nn::Tensor> filter_tensor = nullptr;
  if (is_depthwise_mode) {
    filter_tensor = ConvertOperand(filter_operand,
                                   {filter_channel_size,
                                    output_channel_size,
                                    filter_height,
                                    filter_width});
  } else {
    filter_tensor = ConvertOperand(filter_operand);
  }
  auto bias_tensor = ConvertOperand(bias_operand);
  auto output_tensor = ConvertOperand(output_operand);
  aml::nn::Conv2DAttr attr;
  attr.ksize[0] = filter_height;
  attr.ksize[1] = filter_width;
  attr.stride[0] = stride_width;
  attr.stride[1] = stride_height;
  attr.pad[0] = pad_width_left;
  attr.pad[1] = pad_width_right;
  attr.pad[2] = pad_height_top;
  attr.pad[3] = pad_height_bottom;
  attr.group = group;
  attr.multiplier = 0;
  attr.weights = filter_channel_size;
  attr.dilation[0] = dilation_width;
  attr.dilation[1] = dilation_height;
  attr.pad_type = aml::nn::PadType::AUTO;
  // fuse RELU ?
  if (fuse_code != NNADAPTER_FUSED_NONE) {
    NNADAPTER_LOG(FATAL) << "Unsupported fuse_code(" << fuse_code
                         << ") is found.";
  }
  std::vector<std::shared_ptr<aml::nn::Tensor>> input_tensors = {
      input_tensor, filter_tensor, bias_tensor};
  std::vector<std::shared_ptr<aml::nn::Tensor>> output_tensors = {
      output_tensor};
  graph_->AddOperator(aml::nn::OperatorType::DECONVOLUTION,
                      input_tensors,
                      output_tensors,
                      &attr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace amlogic_npu
}  // namespace nnadapter
