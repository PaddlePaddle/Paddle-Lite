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

#include "operation/gather.h"
#include "core/types.h"
#include "operation/math/gather.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateGather(const core::Operation* operation) {
  return true;
}

NNADAPTER_EXPORT int PrepareGather(core::Operation* operation) {
  GATHER_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type, input_operand->type);
  output_operand->type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  auto& in_dims = input_operand->type.dimensions;
  auto& ids_dims = indices_operand->type.dimensions;
  auto& out_dims = output_operand->type.dimensions;
  int32_t in_count = in_dims.count;
  int32_t ids_count = ids_dims.count;
  out_dims.count = in_count + ids_count - 1;

  auto infer_output_shape = [&](const int32_t* in_dims_data,
                                const int32_t* ids_dims_data,
                                int32_t* out_dims_data) {
    memcpy(out_dims_data, in_dims_data, sizeof(int32_t) * axis);
    memcpy(out_dims_data + axis, ids_dims_data, sizeof(int32_t) * ids_count);
    memcpy(out_dims_data + axis + ids_count,
           in_dims_data + axis + 1,
           sizeof(int32_t) * (in_count - axis));
  };

  infer_output_shape(in_dims.data, ids_dims.data, out_dims.data);
  for (uint32_t i = 0; i < in_dims.dynamic_count; i++) {
    infer_output_shape(in_dims.dynamic_data[i],
                       ids_dims.dynamic_data[i],
                       out_dims.dynamic_data[i]);
  }

  if (IsTemporaryShapeOperand(input_operand) &&
      IsConstantOperand(indices_operand)) {
    auto indices = reinterpret_cast<int32_t*>(indices_operand->buffer);
    auto indices_count =
        indices_operand->length / static_cast<uint32_t>(sizeof(int32_t));
    output_operand->type.lifetime = NNADAPTER_TEMPORARY_SHAPE;
    auto& temporary_shape = *(GetTemporaryShape(input_operand));
    NNADAPTER_CHECK(temporary_shape.data);
    NNADAPTER_CHECK(temporary_shape.data[0]);
    NNAdapterOperandDimensionType dimension_type;
    dimension_type.count = output_operand->type.dimensions.data[0];
    dimension_type.dynamic_count = input_operand->type.dimensions.dynamic_count;
    math::gather<int32_t>(
        temporary_shape.data,
        std::vector<int32_t>({static_cast<int32_t>(temporary_shape.count)}),
        indices,
        std::vector<int32_t>({static_cast<int32_t>(indices_count)}),
        axis,
        dimension_type.data);
    for (uint32_t i = 0; i < dimension_type.dynamic_count; i++) {
      math::gather<int32_t>(
          temporary_shape.dynamic_data[i],
          std::vector<int32_t>({static_cast<int32_t>(temporary_shape.count)}),
          indices,
          std::vector<int32_t>({static_cast<int32_t>(indices_count)}),
          axis,
          dimension_type.dynamic_data[i]);
    }
    SetTemporaryShape(output_operand, dimension_type);
  }

  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteGather(core::Operation* operation) {
  GATHER_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Allocate and calculate the output operands
  int status = -1;
  auto output_buffer = AllocateOperand(output_operand);
  auto in_dims_data = input_operand->type.dimensions.data;
  auto in_dims_count = input_operand->type.dimensions.count;
  std::vector<int32_t> in_dims(in_dims_data, in_dims_data + in_dims_count);
  auto idx_dims_data = indices_operand->type.dimensions.data;
  auto idx_dims_count = indices_operand->type.dimensions.count;
  std::vector<int32_t> idx_dims(idx_dims_data, idx_dims_data + idx_dims_count);
  auto input_precision = input_operand->type.precision;
  auto precision_size = GetOperandPrecisionDataLength(input_precision);
  switch (precision_size) {
    case 4:
      if (indices_operand->type.precision == NNADAPTER_INT32) {
        status =
            math::gather(reinterpret_cast<float*>(input_operand->buffer),
                         in_dims,
                         reinterpret_cast<int32_t*>(indices_operand->buffer),
                         idx_dims,
                         axis,
                         reinterpret_cast<float*>(output_buffer));
      } else {
        status =
            math::gather(reinterpret_cast<float*>(input_operand->buffer),
                         in_dims,
                         reinterpret_cast<int64_t*>(indices_operand->buffer),
                         idx_dims,
                         axis,
                         reinterpret_cast<float*>(output_buffer));
      }
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported input precision code("
                           << OperandPrecisionCodeToString(input_precision)
                           << ") for " << OperationTypeToString(operation->type)
                           << " is found!";
      break;
  }

  NNADAPTER_CHECK_EQ(status, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
