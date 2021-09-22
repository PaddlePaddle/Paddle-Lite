// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "core/operation/squeeze.h"
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareSqueeze(hal::Operation* operation) {
  SQUEEZE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operand->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeExceptQuantParams(&output_type, input_type);
  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions) {
    size_t num_squeeze_dims = axes.size();
    int cnt_squeezed_dims = 0;
    bool should_squeeze[9] = {false};
    if (num_squeeze_dims == 0) {
      for (size_t idx = 0; idx < input_type.dimensions.count; ++idx) {
        if (input_dimensions[idx] == 1) {
          should_squeeze[idx] = true;
          ++cnt_squeezed_dims;
        }
      }
    } else {
      for (size_t idx = 0; idx < num_squeeze_dims; ++idx) {
        int current =
            axes[idx] < 0 ? axes[idx] + input_type.dimensions.count : axes[idx];
        // Check current index, the upper limit has been checked.
        NNADAPTER_CHECK_GE(current, 0)
            << "Invalid axis, the negative axis is out of range.";
        NNADAPTER_CHECK_EQ(input_dimensions[current], 1)
            << "Invalid axis index, the axis that "
               "will be squeezed should be equal "
               "to 1.";
        if (!(should_squeeze[current])) {
          ++cnt_squeezed_dims;
        }
        should_squeeze[current] = true;
      }
    }

    output_type.dimensions.count -= cnt_squeezed_dims;
    for (size_t in_idx = 0, out_idx = 0; in_idx < input_type.dimensions.count;
         ++in_idx) {
      if (!should_squeeze[in_idx]) {
        output_dimensions[out_idx++] = input_dimensions[in_idx];
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

}  // namespace operation
}  // namespace nnadapter
