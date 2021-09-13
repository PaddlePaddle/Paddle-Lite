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

#include "core/operation/mat_mul.h"
#include <vector>
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareMatMul(hal::Operation* operation) {
  MAT_MUL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto infer_output_shape = [&](
      std::vector<int32_t> x_shape,
      std::vector<int32_t> y_shape,
      const bool transpose_x,
      const bool transpose_y) -> std::vector<int32_t> {
    NNADAPTER_CHECK_GE(x_shape.size(), 1UL);
    NNADAPTER_CHECK_GE(y_shape.size(), 1UL);
    bool squeeze_first = false;
    bool squeeze_last = false;
    if (x_shape.size() == 1) {
      x_shape.insert(x_shape.begin(), 1);
      squeeze_first = true;
    }
    if (y_shape.size() == 1) {
      y_shape.push_back(1);
      squeeze_last = true;
    }
    auto x_size = x_shape.size();
    auto y_size = y_shape.size();
    if (x_size < y_size) {
      x_shape.insert(x_shape.begin(), y_size - x_size, 1);
    } else if (y_size < x_size) {
      y_shape.insert(y_shape.begin(), x_size - y_size, 1);
    }
    NNADAPTER_CHECK_EQ(x_shape.size(), y_shape.size());
    NNADAPTER_CHECK_GE(x_shape.size(), 2UL);

    std::vector<int32_t> out_shape;
    for (size_t i = 0; i < x_shape.size() - 2; i++) {
      if (x_shape[i] == y_shape[i]) {
        out_shape.push_back(x_shape[i]);
      } else if (x_shape[i] == 1) {
        out_shape.push_back(y_shape[i]);
      } else if (y_shape[i] == 1) {
        out_shape.push_back(x_shape[i]);
      } else {
        NNADAPTER_LOG(ERROR) << "Not match x_shape[i] = " << x_shape[i]
                             << " and y_shape[i] = " << y_shape[i];
      }
    }
    if (transpose_x) {
      out_shape.push_back(x_shape.back());
    } else {
      out_shape.push_back(x_shape[x_shape.size() - 2]);
    }
    if (transpose_y) {
      out_shape.push_back(y_shape[y_shape.size() - 2]);
    } else {
      out_shape.push_back(y_shape.back());
    }
    if (squeeze_first) {
      NNADAPTER_CHECK_EQ(*out_shape.begin(), 1);
      out_shape.erase(out_shape.begin());
    }
    if (squeeze_last) {
      NNADAPTER_CHECK_EQ(*out_shape.end(), 1);
      out_shape.erase(out_shape.end());
    }
    if (out_shape.empty()) {
      out_shape.push_back(1);
    }
    return out_shape;
  };

  const auto x_type = x_operand->type;
  auto x_size = x_type.dimension_count;
  auto x_shape =
      std::vector<int32_t>(x_type.dimensions, x_type.dimensions + x_size);
  const auto y_type = y_operand->type;
  auto y_size = y_type.dimension_count;
  auto y_shape =
      std::vector<int32_t>(y_type.dimensions, y_type.dimensions + y_size);
  auto out_shape =
      infer_output_shape(x_shape, y_shape, transpose_x, transpose_y);
  auto& out_type = output_operand->type;
  out_type.dimension_count = static_cast<uint32_t>(out_shape.size());
  memcpy(out_type.dimensions,
         out_shape.data(),
         out_shape.size() * sizeof(int32_t));

  out_type.dynamic_dimension_count = x_type.dynamic_dimension_count;
  for (uint32_t i = 0; i < out_type.dynamic_dimension_count; i++) {
    x_shape = std::vector<int32_t>(x_type.dynamic_dimensions[i],
                                   x_type.dynamic_dimensions[i] + x_size);
    y_shape = std::vector<int32_t>(y_type.dynamic_dimensions[i],
                                   y_type.dynamic_dimensions[i] + y_size);
    out_shape = infer_output_shape(x_shape, y_shape, transpose_x, transpose_y);
    NNADAPTER_CHECK_EQ(out_type.dimension_count,
                       static_cast<uint32_t>(out_shape.size()));
    memcpy(out_type.dynamic_dimensions[i],
           out_shape.data(),
           out_shape.size() * sizeof(int32_t));
  }

  out_type.precision = x_type.precision;
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
