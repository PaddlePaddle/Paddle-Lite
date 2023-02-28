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

#include "operation/mat_mul.h"
#include <vector>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateMatMul(const core::Operation* operation) {
  return true;
}

NNADAPTER_EXPORT int PrepareMatMul(core::Operation* operation) {
  MAT_MUL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  const auto& x_type = x_operand->type;
  const auto& y_type = y_operand->type;
  auto x_size = x_type.dimensions.count;
  auto y_size = y_type.dimensions.count;
  NNADAPTER_CHECK_GE(x_size, 1U);
  NNADAPTER_CHECK_GE(y_size, 1U);

  auto infer_output_shape = [&](const int32_t* x_dims,
                                const int32_t* y_dims,
                                const bool transpose_x,
                                const bool transpose_y,
                                uint32_t* out_count,
                                int32_t* out_dims) {
    std::vector<int32_t> x_shape(x_dims, x_dims + x_size);
    std::vector<int32_t> y_shape(y_dims, y_dims + y_size);
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
    if (x_shape.size() < y_shape.size()) {
      x_shape.insert(x_shape.begin(), y_shape.size() - x_shape.size(), 1);
    } else if (y_shape.size() < x_shape.size()) {
      y_shape.insert(y_shape.begin(), x_shape.size() - y_shape.size(), 1);
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
    if (transpose_x && x_size != 1) {
      out_shape.push_back(x_shape.back());
    } else {
      out_shape.push_back(x_shape[x_shape.size() - 2]);
    }
    if (transpose_y && y_size != 1) {
      out_shape.push_back(y_shape[y_shape.size() - 2]);
    } else {
      out_shape.push_back(y_shape.back());
    }
    if (squeeze_first) {
      NNADAPTER_CHECK_EQ(*out_shape.begin(), 1);
      out_shape.erase(out_shape.begin());
    }
    if (squeeze_last) {
      NNADAPTER_CHECK_EQ(*(out_shape.end() - 1), 1);
      out_shape.erase(out_shape.end() - 1);
    }
    if (out_shape.empty()) {
      out_shape.push_back(1);
    }
    out_count[0] = out_shape.size();
    memcpy(out_dims, out_shape.data(), out_shape.size() * sizeof(int32_t));
  };

  auto& out_type = output_operand->type;
  infer_output_shape(x_type.dimensions.data,
                     y_type.dimensions.data,
                     transpose_x,
                     transpose_y,
                     &(out_type.dimensions.count),
                     out_type.dimensions.data);
  out_type.dimensions.dynamic_count = x_type.dimensions.dynamic_count;
  for (uint32_t i = 0; i < out_type.dimensions.dynamic_count; i++) {
    uint32_t out_count;
    infer_output_shape(x_type.dimensions.dynamic_data[i],
                       y_type.dimensions.dynamic_data[i],
                       transpose_x,
                       transpose_y,
                       &out_count,
                       out_type.dimensions.dynamic_data[i]);
    NNADAPTER_CHECK_EQ(out_count, out_type.dimensions.count);
  }

  out_type.precision = x_type.precision;
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteMatMul(core::Operation* operation) {
  NNADAPTER_LOG(FATAL) << OperationTypeToString(operation->type)
                       << " is not implemented!";
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
