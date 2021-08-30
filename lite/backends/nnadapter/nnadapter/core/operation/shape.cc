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

#include "core/operation/shape.h"
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareShape(hal::Operation* operation) {
  SHAPE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto& in_type = input_operand->type;
  auto& out_type = output_operand->type;
  out_type.dimension_count = 1;
  int32_t shape_size = in_type.dimension_count;
  out_type.dimensions[0] = shape_size;
  out_type.dynamic_dimension_count = in_type.dynamic_dimension_count;
  for (uint32_t i = 0; i < in_type.dynamic_dimension_count; i++) {
    out_type.dynamic_dimensions[i][0] = shape_size;
  }
  out_type.precision = NNADAPTER_TENSOR_INT32;

  if (out_type.lifetime == NNADAPTER_TEMPORARY_SHAPE) {
    uint32_t size = static_cast<uint32_t>(sizeof(hal::TemporaryShape));
    void* shape_ptr = malloc(size);
    memset(shape_ptr, 0, size);
    output_operand->length = size;
    output_operand->buffer = shape_ptr;
    hal::TemporaryShape* tmp_shape =
        reinterpret_cast<hal::TemporaryShape*>(shape_ptr);
    for (uint32_t i = 0; i < in_type.dimension_count; i++) {
      tmp_shape->shape.push_back(in_type.dimensions[i]);
    }
    for (uint32_t i = 0; i < in_type.dynamic_dimension_count; i++) {
      tmp_shape->dynamic_shape.push_back({});
      for (uint32_t j = 0; j < in_type.dimension_count; j++) {
        tmp_shape->dynamic_shape.back().push_back(
            in_type.dynamic_dimensions[i][j]);
      }
    }
  }

  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
