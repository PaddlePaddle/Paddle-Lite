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

#include "core/operation/fill.h"
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareFill(hal::Operation* operation) {
  FILL_OPERATION_EXTRACT_INPUTS_OUTPUTS
  // Infer the shape and type of output operands

  auto& shape_type = shape_operand->type;
  auto& out_type = output_operand->type;
  if (shape_type.lifetime == NNADAPTER_CONSTANT_COPY) {
    uint32_t length = shape_operand->length;
    auto shape_precision = shape_type.precision;
    switch (shape_precision) {
      case NNADAPTER_TENSOR_INT32: {
        int32_t* shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
        out_type.dimension_count = length;
        memcpy(out_type.dimensions, shape_data, length * sizeof(int32_t));
        break;
      }
      case NNADAPTER_TENSOR_INT64: {
        int64_t* shape_data = reinterpret_cast<int64_t*>(shape_operand->buffer);
        out_type.dimension_count = length;
        for (uint32_t i = 0; i < length; i++) {
          out_type.dimensions[i] = static_cast<int32_t>(shape_data[i]);
        }
        break;
      }
      default:
        NNADAPTER_LOG(ERROR) << "Unsupported shape precision: "
                             << static_cast<int32_t>(shape_precision);
        break;
    }
  } else if (shape_type.lifetime == NNADAPTER_TEMPORARY_SHAPE) {
    auto tmp_shape =
        reinterpret_cast<hal::TemporaryShape*>(shape_operand->buffer);
    auto shape = tmp_shape->shape;
    out_type.dimension_count = static_cast<uint32_t>(shape.size());
    memcpy(out_type.dimensions,
           shape.data(),
           sizeof(int32_t) * out_type.dimension_count);
    auto dynamic_shape = tmp_shape->dynamic_shape;
    out_type.dynamic_dimension_count =
        static_cast<uint32_t>(dynamic_shape.size());
    for (size_t i = 0; i < dynamic_shape.size(); i++) {
      memcpy(out_type.dynamic_dimensions[i],
             dynamic_shape[i].data(),
             sizeof(int32_t) * dynamic_shape[i].size());
    }
  } else {
    NNADAPTER_LOG(ERROR) << "Unsupported shape lifetime: "
                         << static_cast<int32_t>(shape_type.lifetime);
  }

  out_type.precision = value_operand->type.precision;

  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
