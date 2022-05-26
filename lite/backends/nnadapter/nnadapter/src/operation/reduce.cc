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

#include "operation/reduce.h"
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateReduce(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareReduce(core::Operation* operation) {
  REDUCE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operand->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeExceptQuantParams(&output_type, input_type);
  // special case
  if (input_type.dimensions.count == 1) {
    keep_dim = true;
  }
  if (keep_dim) {
    output_type.dimensions.count = input_type.dimensions.count;
  } else {
    output_type.dimensions.count =
        (input_type.dimensions.count == axes_size)
            ? 1
            : input_type.dimensions.count - axes_size;
  }
  auto infer_output_shape = [&](const int32_t* input_dimensions,
                                const int32_t input_dimension_count,
                                int32_t* output_dimensions) {
    for (int i = 0; i < input_dimension_count; i++) {
      output_dimensions[i] = input_dimensions[i];
    }
    const int kDelFlag = -2;
    for (int i = 0; i < axes_size; i++) {
      auto axis = axes_data[i] >= 0 ? axes_data[i]
                                    : axes_data[i] + input_dimension_count;
      if (keep_dim) {
        output_dimensions[axis] = 1;
      } else {
        output_dimensions[axis] = kDelFlag;
      }
    }
    int output_dimension_index = 0;
    for (int i = 0; i < input_dimension_count;) {
      if (output_dimensions[i] == kDelFlag) {
        i++;
      } else {
        output_dimensions[output_dimension_index++] = output_dimensions[i++];
      }
    }
    if (output_dimension_index == 0 &&
        output_dimensions[output_dimension_index] == kDelFlag) {
      output_dimensions[0] = 1;
    }
  };
  infer_output_shape(input_type.dimensions.data,
                     input_type.dimensions.count,
                     output_type.dimensions.data);
  for (uint32_t i = 0; i < input_type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_type.dimensions.dynamic_data[i],
                       input_type.dimensions.count,
                       output_type.dimensions.dynamic_data[i]);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteReduce(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
