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

#include "operation/flatten.h"
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateFlatten(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareFlatten(core::Operation* operation) {
  FLATTEN_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  CopyOperandTypeWithQuantParams(&output_operand->type, input_operand->type);
  end_axis =
      end_axis < 0 ? input_operand->type.dimensions.count + end_axis : end_axis;
  output_operand->type.dimensions.count =
      input_operand->type.dimensions.count - end_axis + start_axis;
  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions,
                                const uint32_t input_dimension_count,
                                int32_t start_axis,
                                int32_t end_axis) {
    uint32_t output_dimension_index = 0;
    for (int i = 0; i < start_axis; i++) {
      output_dimensions[output_dimension_index++] = input_dimensions[i];
    }
    int32_t outer = 1;
    for (int i = start_axis; i <= end_axis; i++) {
      if (input_dimensions[i] == NNADAPTER_UNKNOWN ||
          outer == NNADAPTER_UNKNOWN) {
        outer = NNADAPTER_UNKNOWN;
      } else {
        outer *= input_dimensions[i];
      }
    }
    output_dimensions[output_dimension_index++] = outer;
    for (int i = end_axis + 1; i < input_dimension_count; i++) {
      output_dimensions[output_dimension_index++] = input_dimensions[i];
    }
  };
  infer_output_shape(input_operand->type.dimensions.data,
                     output_operand->type.dimensions.data,
                     input_operand->type.dimensions.count,
                     start_axis,
                     end_axis);

  for (uint32_t i = 0; i < input_operand->type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_operand->type.dimensions.dynamic_data[i],
                       output_operand->type.dimensions.dynamic_data[i],
                       input_operand->type.dimensions.count,
                       start_axis,
                       end_axis);
  }

  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteFlatten(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
