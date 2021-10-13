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

#include "core/operation/expand.h"
#include <vector>
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareExpand(hal::Operation* operation) {
  EXPAND_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operand->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeWithQuantParams(&output_type, input_type);

  output_type.dimensions.count = shape_count;

  auto infer_output_shape = [&](int32_t* input_dimensions_data,
                                uint32_t input_dimensions_count,
                                int32_t* output_dimensions_data) {
    std::vector<int> input_dims_vec;
    for (uint32_t i = 0; i < input_dimensions_count; i++) {
      input_dims_vec.push_back(input_dimensions_data[i]);
    }
    auto diff = shape_count - input_dimensions_count;
    input_dims_vec.insert(input_dims_vec.begin(), diff, 1);
    std::vector<int> final_expand_shape(input_dimensions_count);
    for (uint32_t i = 0; i < input_dims_vec.size(); ++i) {
      NNADAPTER_CHECK_NE(shape_data[i], 0)
          << "The expanded size cannot be zero.";
      if (i < diff) {
        // shape_data = [3,4,-1,-1], X = [10,2] --> // final_expand_shape =
        // [3,4,10,2]
        NNADAPTER_CHECK_GT(shape_data[i], 0)
            << "The expanded size " << shape_data[i]
            << "for non-existing dimensions must be positive for expand_v2 op.";
        final_expand_shape[i] = shape_data[i];
      } else if (shape_data[i] > 0) {
        // shape_data = [3,4,10,4], X = [10,1] --> final_expand_shape =
        // [3,4,10,4]
        if (input_dims_vec[i] != 1) {
          NNADAPTER_CHECK_EQ(input_dims_vec[i], shape_data[i])
              << "The value " << input_dims_vec[i]
              << " of the non-singleton dimension does not match the "
                 "corresponding value "
              << shape_data[i] << " in shape for expand_v2 op.";
          final_expand_shape[i] = shape_data[i];
        } else {
          final_expand_shape[i] = shape_data[i];
        }
      } else {
        // shape_data = [3,4,-1,-1], X = [10,2] --> final_expand_shape =
        // [3,4,10,2]
        NNADAPTER_CHECK_EQ(shape_data[i], -1)
            << "When the value in shape is negative for expand_v2 op, "
               "only -1 is supported, but the value received is "
            << shape_data[i];
        final_expand_shape[i] = input_dims_vec[i];
      }
    }

    for (uint32_t i = 0; i < shape_count; ++i) {
      shape_data[i] = final_expand_shape[i];
      output_dimensions_data[i] = final_expand_shape[i];
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

}  // namespace operation
}  // namespace nnadapter
