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

#include "core/operation/unsqueeze.h"
#include <vector>
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareUnsqueeze(hal::Operation* operation) {
  UNSQUEEZE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto in_type = input_operand->type;
  auto& out_type = output_operand->type;
  CopyOperandTypeExceptQuantParams(&out_type, in_type);
  out_type.dimension_count += axes_count;
  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions) {
    uint32_t cur_size = in_type.dimension_count;
    for (uint32_t i = 0; i < axes_count; i++) {
      int32_t axis = axes_ptr[i] < 0 ? axes_ptr[i] + cur_size + 1 : axes_ptr[i];
      NNADAPTER_CHECK_GE(axis, 0);
      NNADAPTER_CHECK_LE(axis, cur_size);
      for (uint32_t j = cur_size; j > axis; j--) {
        output_dimensions[j] = output_dimensions[j - 1];
      }
      output_dimensions[axis] = 1;
      cur_size++;
    }
  };
  infer_output_shape(in_type.dimensions, out_type.dimensions);
  for (uint32_t i = 0; i < in_type.dynamic_dimension_count; i++) {
    infer_output_shape(in_type.dynamic_dimensions[i],
                       out_type.dynamic_dimensions[i]);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
