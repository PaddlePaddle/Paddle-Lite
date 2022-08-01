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

#include "operation/reshape.h"
#include <vector>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateReshape(const core::Operation* operation) {
  return true;
}

NNADAPTER_EXPORT int PrepareReshape(core::Operation* operation) {
  RESHAPE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operand->type;
  auto& shape_type = shape_operand->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeWithQuantParams(&output_type, input_type);

  uint32_t shape_count;
  int32_t* shape_data = nullptr;
  if (IsTemporaryShapeOperand(shape_operand)) {
    auto& temporary_shape = *(GetTemporaryShape(shape_operand));
    shape_count = temporary_shape.count;
    shape_data = temporary_shape.data;
  } else if (IsConstantOperand(shape_operand)) {
    shape_count = shape_operand->length / sizeof(int32_t);
    shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported shape lifetime: "
                         << OperandLifetimeCodeToString(shape_type.lifetime);
    return NNADAPTER_INVALID_PARAMETER;
  }
  for (uint32_t i = 0; i < shape_count; i++) {
    NNADAPTER_VLOG(5) << "shape[" << i << "] = " << shape_data[i];
  }
  output_type.dimensions.count = shape_count;

  auto infer_output_shape = [&](int32_t* input_dimensions_data,
                                uint32_t input_dimensions_count,
                                int32_t* output_dimensions_data) {
    for (uint32_t i = 0; i < shape_count; i++) {
      output_dimensions_data[i] =
          shape_data[i] == 0 ? input_dimensions_data[i] : shape_data[i];
    }
    int64_t size = 1;
    for (uint32_t i = 0; i < input_dimensions_count; i++) {
      if (input_dimensions_data[i] == NNADAPTER_UNKNOWN) {
        size = -1;
        break;
      } else {
        size *= static_cast<int64_t>(input_dimensions_data[i]);
      }
    }
    if (size != -1) {
      int32_t unk_idx = -1;
      for (uint32_t i = 0; i < shape_count; i++) {
        if (output_dimensions_data[i] == -1) {
          NNADAPTER_CHECK_EQ(unk_idx, -1) << "Should only has one unk idx.";
          unk_idx = i;
        } else {
          size /= static_cast<int64_t>(output_dimensions_data[i]);
        }
      }
      if (unk_idx != -1) {
        output_dimensions_data[unk_idx] = size;
      }
    }
  };
  infer_output_shape(input_type.dimensions.data,
                     input_type.dimensions.count,
                     output_type.dimensions.data);
  for (uint32_t i = 0; i < input_type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_type.dimensions.dynamic_data[i],
                       input_type.dimensions.count,
                       output_type.dimensions.dynamic_data[i]);
  }
  output_type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteReshape(core::Operation* operation) {
  RESHAPE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Allocate and calculate the output operands
  auto& input_type = input_operand->type;
  auto input_shape = std::vector<int32_t>(
      input_type.dimensions.data,
      input_type.dimensions.data + input_type.dimensions.count);
  const auto input_buffer = input_operand->buffer;
  NNADAPTER_CHECK(input_buffer);
  auto& output_type = output_operand->type;
  auto output_shape = std::vector<int32_t>(
      output_type.dimensions.data,
      output_type.dimensions.data + output_type.dimensions.count);
  auto output_buffer = AllocateOperand(output_operand);
  NNADAPTER_CHECK_EQ(input_type.precision, output_type.precision);
  NNADAPTER_CHECK_EQ(ProductionOfDimensions(input_shape),
                     ProductionOfDimensions(output_shape));
  memcpy(output_buffer, input_buffer, GetOperandTypeBufferLength(input_type));
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
