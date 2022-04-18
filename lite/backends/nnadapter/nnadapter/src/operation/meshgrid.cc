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

#include "operation/meshgrid.h"
#include <vector>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateMeshgrid(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareMeshgrid(core::Operation* operation) {
  MESHGRID_OPERATION_EXTRACT_INPUTS_OUTPUTS
  for (auto output_operand : output_operands) {
    output_operand->type.dimensions.count = input_count;
    CopyOperandTypeExceptQuantParams(&output_operand->type,
                                     input_operands[0]->type);
  }

  // Infer the shape and type of output operands
  std::vector<int> output_shape(input_count, 0);
  for (int i = 0; i < input_count; i++) {
    NNADAPTER_CHECK_EQ(input_operands[i]->type.dimensions.count, 1)
        << "Input" << i << "operand dimensions should be 1D.";
    output_shape[i] = input_operands[i]->type.dimensions.data[0];
  }
  for (auto output_operand : output_operands) {
    for (int j = 0; j < input_count; j++) {
      output_operand->type.dimensions.data[j] = output_shape[j];
    }
  }
  // Dynamic shape
  if (input_operands[0]->type.dimensions.dynamic_count != 0) {
    for (uint32_t i = 0; i < input_operands[0]->type.dimensions.dynamic_count;
         i++) {
      std::vector<int> output_shape(input_count, 0);
      for (size_t j = 0; j < input_count; j++) {
        output_shape[i] = input_operands[i]->type.dimensions.dynamic_data[i][0];
      }
      for (auto output_operand : output_operands) {
        for (int j = 0; j < input_count; j++) {
          output_operand->type.dimensions.dynamic_data[i][j] = output_shape[j];
        }
      }
    }
  }

  for (size_t i = 0; i < output_count; i++) {
    NNADAPTER_VLOG(5) << "output" << i << ": "
                      << OperandToString(output_operands[i]);
  }
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteMeshgrid(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
