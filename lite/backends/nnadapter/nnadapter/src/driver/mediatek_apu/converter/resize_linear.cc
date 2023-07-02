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

#include "operation/resize_linear.h"
#include "driver/mediatek_apu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace mediatek_apu {

int ConvertResizeBilinear(Converter* converter, core::Operation* operation) {
  RESIZE_LINEAR_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to Neuron operands and operations
  auto input_index = converter->GetMappedIndex(input_operand);
  if (input_index == INVALID_INDEX) {
    input_index = converter->ConvertOperand(input_operand);
  }
  auto output_index = converter->ConvertOperand(output_operand);
  // NHWC
  auto output_w = converter->AddInt32ConstantOperand(
      output_operand->type.dimensions.data[2]);
  auto output_h = converter->AddInt32ConstantOperand(
      output_operand->type.dimensions.data[1]);
  NNADAPTER_CHECK_NE(output_operand->type.dimensions.data[2], NNADAPTER_UNKNOWN)
      << "The output width should not be dynamic!";
  NNADAPTER_CHECK_NE(output_operand->type.dimensions.data[1], NNADAPTER_UNKNOWN)
      << "The output height should not be dynamic!";
  auto is_nchw_index = converter->AddBool8ConstantOperand(false);
  auto align_corners_index = converter->AddBool8ConstantOperand(align_corners);
  auto half_pixel_centers_index = converter->AddBool8ConstantOperand(false);
  if (align_corners) {
    NNADAPTER_CHECK_EQ(converter->AddOperation(NEURON_RESIZE_BILINEAR,
                                               {input_index,
                                                output_w,
                                                output_h,
                                                is_nchw_index,
                                                align_corners_index,
                                                half_pixel_centers_index},
                                               {output_index}),
                       NEURON_NO_ERROR);
  } else {
    NNADAPTER_CHECK_EQ(converter->AddOperation(
                           NEURON_RESIZE_BILINEAR,
                           {input_index, output_w, output_h, is_nchw_index},
                           {output_index}),
                       NEURON_NO_ERROR);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
