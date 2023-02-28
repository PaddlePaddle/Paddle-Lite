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

#include "operation/shape.h"
#include "core/types.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateShape(const core::Operation* operation) {
  return true;
}

NNADAPTER_EXPORT int ExecuteShape(core::Operation* operation) {
  SHAPE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Allocate and calculate the output operands
  auto output_buffer = AllocateOperand(output_operand);
  int size = input_operand->type.dimensions.count;
  auto in_dims = input_operand->type.dimensions.data;
  switch (dtype) {
    case NNADAPTER_INT32: {
      auto out_buf = reinterpret_cast<int32_t*>(output_buffer);
      for (int i = 0; i < size; i++) {
        out_buf[i] = in_dims[i];
      }
    } break;
    case NNADAPTER_INT64: {
      auto out_buf = reinterpret_cast<int64_t*>(output_buffer);
      for (int i = 0; i < size; i++) {
        out_buf[i] = in_dims[i];
      }
    } break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported precision code("
                           << OperandPrecisionCodeToString(dtype) << ") for "
                           << OperationTypeToString(operation->type)
                           << " is found!";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int PrepareShape(core::Operation* operation) {
  SHAPE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto& input_type = input_operand->type;
  auto& output_type = output_operand->type;
  output_type.dimensions.count = 1;
  int32_t shape_size = input_type.dimensions.count;
  output_type.dimensions.data[0] = shape_size;
  output_type.precision = static_cast<NNAdapterOperandPrecisionCode>(dtype);
  output_type.lifetime = NNADAPTER_TEMPORARY_SHAPE;
  if (IsDynamicShapeOperandType(input_operand->type)) {
    SetTemporaryShape(output_operand, input_type.dimensions);
  } else {
    ExecuteShape(operation);
    output_operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
