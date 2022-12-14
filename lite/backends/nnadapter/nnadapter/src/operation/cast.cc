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

#include "operation/cast.h"
#include "core/types.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

template <class InType, class OutType>
OutType TransOp(InType in) {
  return static_cast<OutType>(in);
}

NNADAPTER_EXPORT bool ValidateCast(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int ExecuteCast(core::Operation* operation) {
  CAST_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Allocate and calculate the output operands
  auto output_buffer = AllocateOperand(output_operand);
  auto size = ProductionOfDimensions(input_operand->type.dimensions.data,
                                     input_operand->type.dimensions.count);
  auto input_precision = input_operand->type.precision;
  if (input_precision == NNADAPTER_INT32 && dtype == NNADAPTER_INT32) {
    auto input_data = reinterpret_cast<int32_t*>(input_operand->buffer);
    auto output_data = reinterpret_cast<int32_t*>(output_buffer);
    memcpy(output_data, input_data, sizeof(int32_t) * size);
  } else if (input_precision == NNADAPTER_INT64 && dtype == NNADAPTER_INT64) {
    auto input_data = reinterpret_cast<int64_t*>(input_operand->buffer);
    auto output_data = reinterpret_cast<int64_t*>(output_buffer);
    memcpy(output_data, input_data, sizeof(int64_t) * size);
  } else if (input_precision == NNADAPTER_FLOAT32 &&
             dtype == NNADAPTER_FLOAT32) {
    auto input_data = reinterpret_cast<float*>(input_operand->buffer);
    auto output_data = reinterpret_cast<float*>(output_buffer);
    memcpy(output_data, input_data, sizeof(float) * size);
  } else if (input_precision == NNADAPTER_INT32 && dtype == NNADAPTER_INT64) {
    auto input_data = reinterpret_cast<int32_t*>(input_operand->buffer);
    auto output_data = reinterpret_cast<int64_t*>(output_buffer);
    std::transform(
        input_data, input_data + size, output_data, TransOp<int32_t, int64_t>);
  } else if (input_precision == NNADAPTER_INT64 && dtype == NNADAPTER_INT32) {
    auto input_data = reinterpret_cast<int64_t*>(input_operand->buffer);
    auto output_data = reinterpret_cast<int32_t*>(output_buffer);
    std::transform(
        input_data, input_data + size, output_data, TransOp<int64_t, int32_t>);
  } else if (input_precision == NNADAPTER_FLOAT32 && dtype == NNADAPTER_INT64) {
    auto input_data = reinterpret_cast<float*>(input_operand->buffer);
    auto output_data = reinterpret_cast<int64_t*>(output_buffer);
    std::transform(
        input_data, input_data + size, output_data, TransOp<float, int64_t>);
  } else if (input_precision == NNADAPTER_INT64 && dtype == NNADAPTER_FLOAT32) {
    auto input_data = reinterpret_cast<int64_t*>(input_operand->buffer);
    auto output_data = reinterpret_cast<float*>(output_buffer);
    std::transform(
        input_data, input_data + size, output_data, TransOp<int64_t, float>);
  } else if (input_precision == NNADAPTER_INT32 && dtype == NNADAPTER_FLOAT32) {
    auto input_data = reinterpret_cast<int32_t*>(input_operand->buffer);
    auto output_data = reinterpret_cast<float*>(output_buffer);
    std::transform(
        input_data, input_data + size, output_data, TransOp<int32_t, float>);
  } else if (input_precision == NNADAPTER_FLOAT32 && dtype == NNADAPTER_INT32) {
    auto input_data = reinterpret_cast<float*>(input_operand->buffer);
    auto output_data = reinterpret_cast<int32_t*>(output_buffer);
    std::transform(
        input_data, input_data + size, output_data, TransOp<float, int32_t>);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported input precision code("
                         << OperandPrecisionCodeToString(input_precision) << ")"
                         << "and output precision code("
                         << OperandPrecisionCodeToString(dtype) << ") for "
                         << OperationTypeToString(operation->type)
                         << " is found!";
  }
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int PrepareCast(core::Operation* operation) {
  CAST_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type, input_operand->type);
  output_operand->type.precision = dtype;
  if (IsTemporaryShapeOperand(input_operand)) {
    SetTemporaryShape(output_operand, input_operand->type.dimensions);
  } else if (IsConstantOperand(input_operand)) {
    ExecuteCast(operation);
    output_operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
  }

  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
