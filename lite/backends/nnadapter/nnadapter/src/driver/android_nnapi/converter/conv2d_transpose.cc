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
#include "driver/android_nnapi/converter/converter.h"
#include "driver/android_nnapi/converter/validator.h"
#include "operation/conv2d.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace android_nnapi {

bool ValidateConv2DTranspose(Validator* validator,
                             const core::Operation* operation) {
  return true;
}

int ConvertConv2DTranspose(Converter* converter, core::Operation* operation) {
  CONV_2D_TRANSPOSE_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_EQ(output_padding_height, 0)
      << "Only supports output_padding_height = 0 and output_padding_width = "
         "0.";
  NNADAPTER_CHECK_EQ(output_padding_width, 0)
      << "Only supports output_padding_height = 0 and output_padding_width = "
         "0.";
  NNADAPTER_CHECK_EQ(output_shape_height, -1)
      << "Only supports output_shape_height = -1 and output_shape_width = -1.";
  NNADAPTER_CHECK_EQ(output_shape_width, -1)
      << "Only supports output_shape_height = -1 and output_shape_width = -1.";
  if (auto_pad != NNADAPTER_AUTO_PAD_NONE) {
    // NHWC
    operation::UpdateConv2DPadAndDilation(
        input_operand->type.dimensions.data[1],
        filter_height,
        auto_pad,
        &pad_height_top,
        &pad_height_bottom,
        stride_height,
        &dilation_height);
    operation::UpdateConv2DPadAndDilation(
        input_operand->type.dimensions.data[2],
        filter_width,
        auto_pad,
        &pad_width_left,
        &pad_width_right,
        stride_width,
        &dilation_width);
  }
  NNADAPTER_CHECK_EQ(dilation_height, 1) << "Only supports dilations = [1,1]";
  NNADAPTER_CHECK_EQ(dilation_width, 1) << "Only supports dilations = [1,1]";

  // Convert to NNAPI operands and operations
  auto input_index = converter->GetMappedIndex(input_operand);
  if (input_index == INVALID_INDEX) {
    input_index = converter->ConvertOperand(input_operand);
  }
  auto filter_index = converter->ConvertOperand(filter_operand);
  NNADAPTER_VLOG(5) << "filter_index:" << filter_index;
  auto bias_index = converter->ConvertOperand(bias_operand);
  NNADAPTER_VLOG(5) << "bias_index:" << bias_index;
  auto padding_width_left_index =
      converter->AddInt32ConstantOperand(pad_width_left);
  auto padding_width_right_index =
      converter->AddInt32ConstantOperand(pad_width_right);
  auto padding_height_top_index =
      converter->AddInt32ConstantOperand(pad_height_top);
  auto padding_height_bottom_index =
      converter->AddInt32ConstantOperand(pad_height_bottom);
  auto stride_width_index = converter->AddInt32ConstantOperand(stride_width);
  auto stride_height_index = converter->AddInt32ConstantOperand(stride_height);
  auto fuse_code_index = converter->AddInt32ConstantOperand(
      ConvertFuseCodeToNNFuseCode(fuse_code));
  auto is_nchw_index = converter->AddBool8ConstantOperand(false);
  auto output_index = converter->ConvertOperand(output_operand);
  NNADAPTER_CHECK_EQ(converter->AddOperation(ANEURALNETWORKS_TRANSPOSE_CONV_2D,
                                             {input_index,
                                              filter_index,
                                              bias_index,
                                              padding_width_left_index,
                                              padding_width_right_index,
                                              padding_height_top_index,
                                              padding_height_bottom_index,
                                              stride_width_index,
                                              stride_height_index,
                                              fuse_code_index,
                                              is_nchw_index},
                                             {output_index}),
                     ANEURALNETWORKS_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace android_nnapi
}  // namespace nnadapter
