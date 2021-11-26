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

  // Infer the shape and type of output operands
  std::vector<int> output_shape(input_count, 0);
  for (int i = 0; i < input_count; i++) {
    NNADAPTER_CHECK(input_operands[i]->type.dimensions.count == 1)
        << "Input" << i << "operand dimensions should be 1D.";
    output_shape[i] = input_operands[i]->type.dimensions.data[0];
  }

  for (auto output_operand : output_operands) {
    CopyOperandTypeExceptQuantParams(&output_operand->type,
                                     input_operands[0]->type);
    output_operand->type.dimensions.count = input_count;
    for (int j = 0; j < input_count; j++) {
      output_operand->type.dimensions.data[j] = output_shape[j];
    }
  }

  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
