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
  if (dtype == static_cast<int32_t>(NNADAPTER_TENSOR_INT32)) {
    out_type.precision = NNADAPTER_TENSOR_INT32;
  } else if (dtype == static_cast<int32_t>(NNADAPTER_TENSOR_INT64)) {
    out_type.precision = NNADAPTER_TENSOR_INT64;
  }

  out_type.lifetime = NNADAPTER_TEMPORARY_SHAPE;
  uint32_t size = static_cast<uint32_t>(sizeof(NNAdapterOperandDimensionType));
  void* shape_ptr = malloc(size);
  memset(shape_ptr, 0, size);
  output_operand->length = size;
  output_operand->buffer = shape_ptr;
  NNAdapterOperandDimensionType* tmp_shape =
      reinterpret_cast<NNAdapterOperandDimensionType*>(shape_ptr);
  tmp_shape->count = in_type.dimension_count;
  memcpy(tmp_shape->data,
         in_type.dimensions,
         NNADAPTER_MAX_SIZE_OF_DIMENSIONS * sizeof(int32_t));
  tmp_shape->dynamic_count = in_type.dynamic_dimension_count;
  memcpy(tmp_shape->dynamic_data,
         in_type.dynamic_dimensions,
         NNADAPTER_MAX_SIZE_OF_DYNAMIC_DIMENSIONS *
             NNADAPTER_MAX_SIZE_OF_DIMENSIONS * sizeof(int32_t));

  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
