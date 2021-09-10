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

#include "core/operation/flatten.h"
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareFlatten(hal::Operation* operation) {
  FLATTEN_OPERATION_EXTRACT_INPUTS_OUTPUTS
  CopyOperandTypeExceptQuantParams(&output_operand->type, input_operand->type);
  // Infer the shape and type of output operands
  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions,
                                const uint32_t input_dimension_count,
                                int32_t start,
                                int32_t stop) {
    stop = stop < 0 ? stop + input_dimension_count : stop;
    uint32_t input_dimension_index = 0;
    uint32_t output_dimension_index = 0;
    // Init output dim with value 1.
    for (int i = 0; i < input_dimension_count; i++) {
      output_dimensions[i] = 1;
    }
    // Calc dim
    while (input_dimension_index < input_dimension_count) {
      if (input_dimension_index >= start && input_dimension_index <= stop) {
        output_dimensions[output_dimension_index] *=
            input_dimensions[input_dimension_index];
        if (input_dimension_index == stop &&
            input_dimension_index < input_dimension_count) {
          output_dimension_index++;
        }
      } else {
        output_dimensions[output_dimension_index] =
            input_dimensions[input_dimension_index];
        output_dimension_index++;
      }
      input_dimension_index++;
    }
    return output_dimension_index;
  };
  output_operand->type.dimension_count =
      infer_output_shape(input_operand->type.dimensions,
                         output_operand->type.dimensions,
                         input_operand->type.dimension_count,
                         start,
                         stop);

  for (uint32_t i = 0; i < input_operand->type.dynamic_dimension_count; i++) {
    infer_output_shape(input_operand->type.dynamic_dimensions[i],
                       output_operand->type.dynamic_dimensions[i],
                       input_operand->type.dimension_count,
                       start,
                       stop);
  }

  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
