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
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT void UpdateConv2DPadAndDilation(
    int32_t input_size,
    int32_t filter_height_or_width,
    NNAdapterAutoPadCode auto_pad,
    int32_t* pad_top_or_left,
    int32_t* pad_bottom_or_right,
    int32_t stride_height_or_width,
    int32_t* dilation_height_or_width) {
  if (auto_pad == NNADAPTER_AUTO_PAD_SAME) {
    NNADAPTER_CHECK_NE(input_size, NNADAPTER_UNKNOWN);
    auto output_size =
        (input_size + stride_height_or_width - 1) / stride_height_or_width;
    auto pad_size = (output_size - 1) * stride_height_or_width +
                    filter_height_or_width - input_size;
    pad_size = pad_size < 0 ? 0 : pad_size;
    *pad_top_or_left = pad_size / 2;
    *pad_bottom_or_right = pad_size - *pad_top_or_left;
    *dilation_height_or_width = 1;
  } else if (auto_pad == NNADAPTER_AUTO_PAD_VALID) {
    *pad_top_or_left = 0;
    *pad_bottom_or_right = 0;
  }
}

NNADAPTER_EXPORT int32_t
CalcConv2DOutputSize(int32_t input_size,
                     int32_t filter_height_or_width,
                     NNAdapterAutoPadCode auto_pad,
                     int32_t pad_top_or_left,
                     int32_t pad_bottom_or_right,
                     int32_t stride_height_or_width,
                     int32_t dilation_height_or_width) {
  if (input_size == NNADAPTER_UNKNOWN) {
    return NNADAPTER_UNKNOWN;
  }
  UpdateConv2DPadAndDilation(input_size,
                             filter_height_or_width,
                             auto_pad,
                             &pad_top_or_left,
                             &pad_bottom_or_right,
                             stride_height_or_width,
                             &dilation_height_or_width);
  auto dkernel = dilation_height_or_width * (filter_height_or_width - 1) + 1;
  return (input_size + (pad_top_or_left + pad_bottom_or_right) - dkernel) /
             stride_height_or_width +
         1;
}

int PrepareConv2D(core::Operation* operation) {
  CONV_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type, input_operand->type);
  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions) {
    output_dimensions[0] = input_dimensions[0];
    output_dimensions[1] = output_channel_size;
    output_dimensions[2] = CalcConv2DOutputSize(input_dimensions[2],
                                                filter_height,
                                                auto_pad,
                                                pad_height_top,
                                                pad_height_bottom,
                                                stride_height,
                                                dilation_height);
    output_dimensions[3] = CalcConv2DOutputSize(input_dimensions[3],
                                                filter_width,
                                                auto_pad,
                                                pad_width_left,
                                                pad_width_right,
                                                stride_width,
                                                dilation_width);
  };
  infer_output_shape(input_operand->type.dimensions.data,
                     output_operand->type.dimensions.data);
  for (uint32_t i = 0; i < input_operand->type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_operand->type.dimensions.dynamic_data[i],
                       output_operand->type.dimensions.dynamic_data[i]);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
