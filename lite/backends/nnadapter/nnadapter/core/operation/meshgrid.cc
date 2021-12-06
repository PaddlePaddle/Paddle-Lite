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

#include "core/operation/meshgrid.h"
#include <vector>
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareMeshgrid(hal::Operation* operation) {
  MESHGRID_OPERATION_EXTRACT_INPUTS_OUTPUTS
  for (auto output_operand : output_operands) {
    CopyOperandTypeExceptQuantParams(&output_operand->type,
                                     input_operands[0]->type);
    output_operand->type.dimensions.count = input_count;
  }

  // Infer the shape and type of output operands
  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions) {
    std::vector<int> output_shape(input_count, 0);
    for (int i = 0; i < input_count; i++) {
      output_shape[i] = input_dimensions[0];
    }

    for (auto output_operand : output_operands) {
      for (int j = 0; j < input_count; j++) {
        output_dimensions[j] = output_shape[j];
      }
    }
  };

  for (size_t i = 0; i < input_count; i++) {
    infer_output_shape(input_operands[i]->type.dimensions.data,
                       output_operands[i]->type.dimensions.data);
  }
  for (uint32_t i = 0; i < output_operands[0]->type.dimensions.dynamic_count;
       i++) {
    for (size_t j = 0; j < input_count; j++) {
      infer_output_shape(input_operands[j]->type.dimensions.dynamic_data[i],
                         output_operands[j]->type.dimensions.dynamic_data[i]);
    }
  }

  for (size_t i = 0; i < output_count; i++) {
    NNADAPTER_VLOG(5) << "output" << i << ": "
                      << OperandToString(output_operands[i]);
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
