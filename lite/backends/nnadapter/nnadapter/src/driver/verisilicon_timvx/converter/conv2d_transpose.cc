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

#include "operation/conv2d_transpose.h"
#include "driver/verisilicon_timvx/converter/converter.h"
#include "operation/conv2d.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace verisilicon_timvx {

int ConvertConv2DTranspose(Converter* converter, core::Operation* operation) {
  CONV_2D_TRANSPOSE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_EQ(output_shape_height, -1)
      << "Only supports output_shape_height = -1 and output_shape_width = -1.";
  NNADAPTER_CHECK_EQ(output_shape_width, -1)
      << "Only supports output_shape_height = -1 and output_shape_width = -1.";
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

  // Convert to tim-vx tensors and operators
  auto input_tensor = converter->GetMappedTensor(input_operand);
  if (!input_tensor) {
    input_tensor = converter->ConvertOperand(input_operand);
  }
  // [C_in, C_out, filter_height, filter_width]->[C_out, C_in, filter_height,
  // filter_width]
  std::vector<int32_t> filter_permutation = {1, 0, 2, 3};
  TransposeOperand(filter_operand, filter_permutation);
  int32_t multiplier = 0;
  std::vector<int32_t> filter_dimensions(
      filter_operand->type.dimensions.data,
      filter_operand->type.dimensions.data +
          filter_operand->type.dimensions.count);
  if (is_depthwise_mode) {
    multiplier = output_channel_size / group;
    NNADAPTER_CHECK_GT(filter_operand->type.dimensions.count, 2);
    // Oc,1,H,W -> 1,Oc,H,W
    filter_dimensions[0] = filter_dimensions[1];
    filter_dimensions[1] = output_channel_size;
  }
  auto filter_tensor =
      converter->ConvertOperand(filter_operand, filter_dimensions);
  auto bias_tensor = converter->ConvertOperand(bias_operand);
  if (!bias_tensor) {
    bias_tensor = converter->ConvertOperand(bias_operand);
  }
  auto output_tensor = converter->ConvertOperand(output_operand);
  auto conv2dtranspose_op =
      converter->graph()->CreateOperation<tim::vx::ops::DeConv2d>(
          filter_dimensions[0],
          tim::vx::PadType::AUTO,
          std::array<uint32_t, 2>({static_cast<uint32_t>(filter_width),
                                   static_cast<uint32_t>(filter_height)}),
          std::array<uint32_t, 2>({static_cast<uint32_t>(stride_width),
                                   static_cast<uint32_t>(stride_height)}),
          std::array<uint32_t, 2>(
              {static_cast<uint32_t>(output_padding_width),
               static_cast<uint32_t>(output_padding_height)}),
          std::array<uint32_t, 4>({static_cast<uint32_t>(pad_width_left),
                                   static_cast<uint32_t>(pad_width_right),
                                   static_cast<uint32_t>(pad_height_top),
                                   static_cast<uint32_t>(pad_height_bottom)}),
          1);
  conv2dtranspose_op->BindInputs({input_tensor, filter_tensor, bias_tensor});
  conv2dtranspose_op->BindOutputs({output_tensor});
  NNADAPTER_CHECK_EQ(fuse_code, NNADAPTER_FUSED_NONE)
      << "Missing the processing of fuse_code(" << fuse_code
      << ") in unpack_op_fusion.cc";
  return NNADAPTER_NO_ERROR;
}

}  // namespace verisilicon_timvx
}  // namespace nnadapter
