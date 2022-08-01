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

#include "operation/arg_min_max.h"
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateArgMinMax(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareArgMinMax(core::Operation* operation) {
  ARG_MIN_MAX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operand->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeExceptQuantParams(&output_type, input_type);
  output_type.precision = dtype;
  if (!keepdim && input_type.dimensions.count > 1) {
    output_type.dimensions.count = input_type.dimensions.count - 1;
  }
  const uint32_t input_dimensions_count = input_type.dimensions.count;
  auto infer_output_shape = [&](int32_t* input_dimensions_data,
                                int32_t* output_dimensions_data) {
    int j = 0;
    for (uint32_t i = 0; i < input_dimensions_count; i++) {
      if (i == axis) {
        if (keepdim) {
          output_dimensions_data[j++] = 1;
        }
      } else {
        output_dimensions_data[j++] = input_dimensions_data[i];
      }
    }
  };
  infer_output_shape(input_type.dimensions.data, output_type.dimensions.data);
  for (uint32_t i = 0; i < input_type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_type.dimensions.dynamic_data[i],
                       output_type.dimensions.dynamic_data[i]);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteArgMinMax(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
