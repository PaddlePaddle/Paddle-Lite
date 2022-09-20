// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/unstack.h"
#include <vector>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateUnstack(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareUnstack(core::Operation* operation) {
  UNSTACK_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operand->type;
  auto infer_output_shape = [&](const int32_t* input_dimensions_data,
                                uint32_t input_dimensions_count,
                                int32_t* output_dimensions_data,
                                uint32_t* output_dimensions_count) {
    for (uint32_t i = 0; i < axis; i++) {
      output_dimensions_data[i] = input_dimensions_data[i];
    }
    for (uint32_t i = axis + 1; i < input_dimensions_count; i++) {
      output_dimensions_data[i - 1] = input_dimensions_data[i];
    }
    *output_dimensions_count = input_dimensions_count - 1;
  };
  for (size_t i = 0; i < output_count; i++) {
    CopyOperandTypeExceptQuantParams(&output_operands[i]->type, input_type);
    infer_output_shape(input_type.dimensions.data,
                       input_type.dimensions.count,
                       output_operands[i]->type.dimensions.data,
                       &output_operands[i]->type.dimensions.count);
    for (uint32_t j = 0; j < input_type.dimensions.dynamic_count; j++) {
      infer_output_shape(input_type.dimensions.dynamic_data[j],
                         input_type.dimensions.count,
                         output_operands[i]->type.dimensions.dynamic_data[j],
                         &output_operands[i]->type.dimensions.count);
    }
    NNADAPTER_VLOG(5) << "output" << i << ": "
                      << OperandToString(output_operands[i]);
  }
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteUnstack(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
