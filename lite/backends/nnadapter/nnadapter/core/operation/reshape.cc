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

#include "core/operation/reshape.h"
#include <vector>
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareReshape(hal::Operation* operation) {
  RESHAPE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  NNADAPTER_CHECK(IsConstantOperand(shape_operand))
      << "Only support constant shape now.";
  auto& in_type = input_operand->type;
  bool input_has_unk_dim = false;
  for (uint32_t i = 0; i < in_type.dimension_count; i++) {
    if (in_type.dimensions[i] == NNADAPTER_UNKNOWN) {
      NNADAPTER_CHECK(!input_has_unk_dim)
          << "Only support input has one dimension = -1.";
      input_has_unk_dim = true;
    }
  }
  bool shape_has_unk_dim = false;
  for (uint32_t i = 0; i < shape_count; i++) {
    if (shape_data[i] == 0) {
      NNADAPTER_CHECK_LT(i, in_type.dimension_count);
      shape_data[i] = in_type.dimensions[i];
    }
    if (shape_data[i] == -1) {
      NNADAPTER_CHECK(!shape_has_unk_dim)
          << "Only support shape has one dimension = -1.";
      shape_has_unk_dim = true;
    }
  }
  if (shape_has_unk_dim && !input_has_unk_dim) {
    int64_t size = 1;
    for (uint32_t i = 0; i < in_type.dimension_count; i++) {
      size *= static_cast<int64_t>(in_type.dimensions[i]);
    }
    uint32_t unk_idx = 0;
    for (uint32_t i = 0; i < shape_count; i++) {
      if (shape_data[i] != -1) {
        size /= static_cast<int64_t>(shape_data[i]);
      } else {
        unk_idx = i;
      }
    }
    shape_data[unk_idx] = static_cast<int32_t>(size);
    shape_has_unk_dim = false;
  }

  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions) {
    memcpy(output_dimensions, shape_data, shape_count * sizeof(int32_t));
    if (shape_has_unk_dim) {
      int64_t size = 1;
      for (uint32_t i = 0; i < in_type.dimension_count; i++) {
        size *= static_cast<int64_t>(input_dimensions[i]);
      }
      uint32_t unk_idx = 0;
      for (uint32_t i = 0; i < shape_count; i++) {
        if (shape_data[i] != -1) {
          size /= static_cast<int64_t>(shape_data[i]);
        } else {
          unk_idx = i;
        }
      }
      output_dimensions[unk_idx] = static_cast<int32_t>(size);
    }
  };
  auto& out_type = output_operand->type;
  out_type.dimension_count = shape_count;
  out_type.dynamic_dimension_count = in_type.dynamic_dimension_count;
  infer_output_shape(in_type.dimensions, out_type.dimensions);
  for (uint32_t i = 0; i < in_type.dynamic_dimension_count; i++) {
    infer_output_shape(in_type.dynamic_dimensions[i],
                       out_type.dynamic_dimensions[i]);
  }
  out_type.precision = in_type.precision;
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
