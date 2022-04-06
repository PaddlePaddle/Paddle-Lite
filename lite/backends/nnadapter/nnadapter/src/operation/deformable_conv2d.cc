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

#include "operation/deformable_conv2d.h"
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT int32_t DeformableConvOutputSize(int32_t input_size,
                                                  int32_t filter_size,
                                                  int32_t dilation,
                                                  int32_t pad_left,
                                                  int32_t pad_right,
                                                  int32_t stride) {
  if (input_size == NNADAPTER_UNKNOWN) {
    return NNADAPTER_UNKNOWN;
  }
  const int32_t dkernel = dilation * (filter_size - 1) + 1;
  return (input_size + (pad_left + pad_right) - dkernel) / stride + 1;
}

NNADAPTER_EXPORT bool ValidateDeformableConv2D(
    const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareDeformableConv2D(core::Operation* operation) {
  DEFORMABLE_CONV_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type, input_operand->type);
  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions) {
    output_dimensions[0] = input_dimensions[0];
    output_dimensions[1] = output_channel_size;
    output_dimensions[2] = DeformableConvOutputSize(input_dimensions[2],
                                                    filter_height,
                                                    dilations[0],
                                                    pads[0],
                                                    pads[1],
                                                    strides[0]);
    output_dimensions[3] = DeformableConvOutputSize(input_dimensions[3],
                                                    filter_width,
                                                    dilations[1],
                                                    pads[2],
                                                    pads[3],
                                                    strides[1]);
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

NNADAPTER_EXPORT int ExecuteDeformableConv2D(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
