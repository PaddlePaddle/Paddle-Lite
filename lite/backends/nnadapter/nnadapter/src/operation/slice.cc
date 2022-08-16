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

#include "operation/slice.h"
#include <algorithm>
#include <vector>
#include "core/types.h"
#include "operation/math/slice.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateSlice(const core::Operation* operation) {
  return true;
}

NNADAPTER_EXPORT int ExecuteSlice(core::Operation* operation) {
  SLICE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Allocate and calculate the output operands
  int status = -1;
  auto output_buffer = AllocateOperand(output_operand);
  auto input_dims_count = input_operand->type.dimensions.count;
  auto input_dims_data = input_operand->type.dimensions.data;
  std::vector<int32_t> input_shape(input_dims_data,
                                   input_dims_data + input_dims_count);
  auto input_precision = input_operand->type.precision;
  switch (input_precision) {
    case NNADAPTER_FLOAT32:
      status = math::slice(reinterpret_cast<float*>(input_operand->buffer),
                           input_shape,
                           axes_count,
                           axes,
                           starts,
                           ends,
                           steps,
                           reinterpret_cast<float*>(output_buffer));
      break;
    case NNADAPTER_INT32:
      status = math::slice(reinterpret_cast<int32_t*>(input_operand->buffer),
                           input_shape,
                           axes_count,
                           axes,
                           starts,
                           ends,
                           steps,
                           reinterpret_cast<int32_t*>(output_buffer));
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported precision code("
                           << OperandPrecisionCodeToString(input_precision)
                           << ") for " << OperationTypeToString(operation->type)
                           << " is found!";
      break;
  }
  NNADAPTER_CHECK_EQ(status, 0);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int PrepareSlice(core::Operation* operation) {
  SLICE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type, input_operand->type);
  auto infer_output_shape = [&](int32_t* output_dimensions) {
    for (size_t i = 0; i < axes_count; ++i) {
      int dim = output_dimensions[axes[i]];
      if (dim > 0) {
        int start = starts[i] < 0 ? (starts[i] + dim) : starts[i];
        int end = ends[i] < 0 ? (ends[i] + dim) : ends[i];
        int step = std::abs(steps[i]);
        start = std::max(start, 0);
        end = std::max(end, 0);
        end = std::min(end, dim);
        output_dimensions[axes[i]] = (std::abs(end - start) + step - 1) / step;
      }
    }
  };

  infer_output_shape(output_operand->type.dimensions.data);
  for (uint32_t i = 0; i < input_operand->type.dimensions.dynamic_count; i++) {
    infer_output_shape(output_operand->type.dimensions.dynamic_data[i]);
  }

  if (IsTemporaryShapeOperand(input_operand)) {
    output_operand->type.lifetime = NNADAPTER_TEMPORARY_SHAPE;
    auto& temporary_shape = *(GetTemporaryShape(input_operand));
    NNADAPTER_CHECK(temporary_shape.data);
    NNADAPTER_CHECK(temporary_shape.data[0]);
    NNAdapterOperandDimensionType dimension_type;
    dimension_type.count = output_operand->type.dimensions.data[0];
    dimension_type.dynamic_count = input_operand->type.dimensions.dynamic_count;
    math::slice<int32_t>(
        temporary_shape.data,
        std::vector<int32_t>({static_cast<int32_t>(temporary_shape.count)}),
        axes_count,
        axes,
        starts,
        ends,
        steps,
        dimension_type.data);
    for (uint32_t i = 0; i < dimension_type.dynamic_count; i++) {
      math::slice<int32_t>(
          temporary_shape.dynamic_data[i],
          std::vector<int32_t>({static_cast<int32_t>(temporary_shape.count)}),
          axes_count,
          axes,
          starts,
          ends,
          steps,
          dimension_type.dynamic_data[i]);
    }
    SetTemporaryShape(output_operand, dimension_type);
  } else if (IsConstantOperand(input_operand)) {
    ExecuteSlice(operation);
    output_operand->type.lifetime = NNADAPTER_CONSTANT_COPY;
  }

  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
