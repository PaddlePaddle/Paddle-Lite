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

#include "core/operation/conv2d.h"
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareConv2D(hal::Operation* operation) {
  CONV2D_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type, input_operand->type);
  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions) {
    output_dimensions[0] = input_dimensions[0];
    output_dimensions[1] = output_channel_size;
    int dkernel = dilation_height * (filter_height - 1) + 1;
    output_dimensions[2] =
        input_dimensions[2] == NNADAPTER_UNKNOWN
            ? NNADAPTER_UNKNOWN
            : ((input_dimensions[2] + (pad_height_top + pad_height_bottom) -
                dkernel) /
                   stride_height +
               1);
    dkernel = dilation_width * (filter_width - 1) + 1;
    output_dimensions[3] =
        input_dimensions[3] == NNADAPTER_UNKNOWN
            ? NNADAPTER_UNKNOWN
            : ((input_dimensions[3] + (pad_width_left + pad_width_right) -
                dkernel) /
                   stride_width +
               1);
  };
  infer_output_shape(input_operand->type.dimensions,
                     output_operand->type.dimensions);
  for (uint32_t i = 0; i < input_operand->type.dynamic_dimension_count; i++) {
    infer_output_shape(input_operand->type.dynamic_dimensions[i],
                       output_operand->type.dynamic_dimensions[i]);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
