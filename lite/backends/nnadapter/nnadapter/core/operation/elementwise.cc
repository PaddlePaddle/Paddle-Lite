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

#include "core/operation/elementwise.h"
#include <algorithm>
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareElementwise(hal::Operation* operation) {
  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto x_type = input0_operand->type;
  auto y_type = input1_operand->type;
  auto& output_type = output_operand->type;
  int32_t x_size = x_type.dimensions.count;
  int32_t y_size = y_type.dimensions.count;
  int32_t max_size = std::max(x_size, y_size);
  auto infer_output_shape = [&](int32_t* x_dimensions,
                                int32_t* y_dimensions,
                                int32_t* output_dimensions) {
    int32_t xi = x_size - 1;
    int32_t yi = y_size - 1;
    for (int32_t i = max_size - 1; i >= 0; i--) {
      if (xi < 0) {
        NNADAPTER_CHECK_GE(yi, 0);
        output_dimensions[i] = y_dimensions[yi];
      } else if (yi < 0) {
        NNADAPTER_CHECK_GE(xi, 0);
        output_dimensions[i] = x_dimensions[xi];
      } else {
        int32_t x_data = x_dimensions[xi];
        int32_t y_data = y_dimensions[yi];
        if (x_data == y_data) {
          output_dimensions[i] = x_data;
        } else if (x_data == 1) {
          output_dimensions[i] = y_data;
        } else if (y_data == 1) {
          output_dimensions[i] = x_data;
        } else {
          NNADAPTER_LOG(ERROR) << "Cannot broadcast x: " << x_data
                               << ", y: " << y_data;
        }
      }
      xi--;
      yi--;
    }
  };
  output_type.dimensions.count = max_size;
  output_type.dimensions.dynamic_count = x_type.dimensions.dynamic_count;
  infer_output_shape(x_type.dimensions.data,
                     y_type.dimensions.data,
                     output_type.dimensions.data);
  for (uint32_t i = 0; i < x_type.dimensions.dynamic_count; i++) {
    infer_output_shape(x_type.dimensions.dynamic_data[i],
                       y_type.dimensions.dynamic_data[i],
                       output_type.dimensions.dynamic_data[i]);
  }
  output_type.precision = x_type.precision;
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
