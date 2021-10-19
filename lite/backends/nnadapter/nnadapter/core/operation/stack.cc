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

#include "core/operation/stack.h"
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareStack(hal::Operation* operation) {
  STACK_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operands[0]->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeWithQuantParams(&output_type, input_type);

  auto input_dimensions_count = input_type.dimensions.count;
  if (axis < 0) axis += (input_dimensions_count + 1);
  NNADAPTER_CHECK_GE(axis, 0);
  NNADAPTER_CHECK_LE(axis, input_dimensions_count);
  output_type.dimensions.count = input_dimensions_count + 1;

  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions,
                                const uint32_t input_dimensions_count) {
    for (uint32_t j = input_dimensions_count; j > axis; j--) {
      output_dimensions[j] = output_dimensions[j - 1];
    }
    output_dimensions[axis] = input_count - 1;
  };

  infer_output_shape(input_type.dimensions.data,
                     output_type.dimensions.data,
                     input_dimensions_count);
  for (uint32_t i = 0; i < output_type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_type.dimensions.dynamic_data[i],
                       output_type.dimensions.dynamic_data[i],
                       input_dimensions_count);
  }

  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
