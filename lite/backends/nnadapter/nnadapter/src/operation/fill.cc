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

#include "operation/fill.h"
#include "core/types.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateFill(const core::Operation* operation) {
  return true;
}

NNADAPTER_EXPORT int ExecuteFill(core::Operation* operation) {
  FILL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Allocate and calculate the output operands
  auto out_buffer = reinterpret_cast<uint8_t*>(AllocateOperand(output_operand));
  auto out_length = output_operand->length;
  auto value_buffer = reinterpret_cast<uint8_t*>(value_operand->buffer);
  auto value_length = value_operand->length;
  for (int i = 0; i < out_length; i += value_length) {
    memcpy(out_buffer, value_buffer, value_length);
    out_buffer += value_length;
  }
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int PrepareFill(core::Operation* operation) {
  FILL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto& shape_type = shape_operand->type;
  auto& output_type = output_operand->type;
  if (IsConstantOperand(shape_operand)) {
    uint32_t length = shape_operand->length;
    auto shape_precision = shape_type.precision;
    switch (shape_precision) {
      case NNADAPTER_INT32: {
        int32_t* shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
        uint32_t size = length / sizeof(int32_t);
        output_type.dimensions.count = size;
        memcpy(output_type.dimensions.data, shape_data, size * sizeof(int32_t));
        break;
      }
      case NNADAPTER_INT64: {
        int64_t* shape_data = reinterpret_cast<int64_t*>(shape_operand->buffer);
        uint32_t size = length / sizeof(int64_t);
        output_type.dimensions.count = size;
        for (uint32_t i = 0; i < size; i++) {
          output_type.dimensions.data[i] = static_cast<int32_t>(shape_data[i]);
        }
        break;
      }
      default:
        NNADAPTER_LOG(ERROR) << "Unsupported shape precision: "
                             << OperandPrecisionCodeToString(shape_precision);
        break;
    }
  } else if (IsTemporaryShapeOperand(shape_operand)) {
    auto& temporary_shape = *(GetTemporaryShape(shape_operand));
    memcpy(&output_type.dimensions,
           &temporary_shape,
           sizeof(NNAdapterOperandDimensionType));
  } else {
    NNADAPTER_LOG(ERROR) << "Unsupported shape lifetime: "
                         << static_cast<int32_t>(shape_type.lifetime);
    return NNADAPTER_INVALID_PARAMETER;
  }
  output_type.precision = value_operand->type.precision;
  output_type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;

  if (IsConstantOperand(shape_operand) && IsConstantOperand(value_operand)) {
    ExecuteFill(operation);
    output_type.lifetime = NNADAPTER_CONSTANT_COPY;
  }

  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
