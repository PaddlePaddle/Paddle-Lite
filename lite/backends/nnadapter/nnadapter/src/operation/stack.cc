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

#include "operation/stack.h"
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateStack(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareStack(core::Operation* operation) {
  STACK_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operands[0]->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeWithQuantParams(&output_type, input_type);
  auto infer_output_shape = [&](const int32_t* input_dimensions_data,
                                uint32_t input_dimensions_count,
                                int32_t* output_dimensions_data,
                                uint32_t* output_dimensions_count) {
    for (uint32_t i = 0; i < axis; i++) {
      output_dimensions_data[i] = input_dimensions_data[i];
    }
    output_dimensions_data[axis] = input_count - 1;
    for (uint32_t i = axis; i < input_dimensions_count; i++) {
      output_dimensions_data[i + 1] = input_dimensions_data[i];
    }
    *output_dimensions_count = input_dimensions_count + 1;
  };
  infer_output_shape(input_type.dimensions.data,
                     input_type.dimensions.count,
                     output_type.dimensions.data,
                     &output_type.dimensions.count);
  for (uint32_t i = 0; i < input_type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_type.dimensions.dynamic_data[i],
                       input_type.dimensions.count,
                       output_type.dimensions.dynamic_data[i],
                       &output_type.dimensions.count);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteStack(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
