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

#include "operation/expand.h"
#include <vector>
#include "core/types.h"
#include "operation/math/expand.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT void UpdateExpandInferOutputShape(
    int32_t* input_dimensions_data,
    uint32_t input_dimensions_count,
    int32_t* output_dimensions_data,
    uint32_t shape_count,
    int32_t* shape_data) {
  std::vector<int> input_shape(input_dimensions_data,
                               input_dimensions_data + input_dimensions_count);
  auto diff = shape_count - input_dimensions_count;
  input_shape.insert(input_shape.begin(), diff, 1);
  for (uint32_t i = 0; i < shape_count; ++i) {
    NNADAPTER_CHECK_NE(shape_data[i], 0) << "The expanded size cannot be zero.";
    if (i < diff) {
      // input = [10, 2], shape_data = [3,4,-1,-1]
      // --> output_dimensions_data = [3,4,10,2]
      NNADAPTER_CHECK_GT(shape_data[i], 0)
          << "The expanded size " << shape_data[i]
          << "for non-existing dimensions must be positive for expand_v2 op.";
      output_dimensions_data[i] = shape_data[i];
    } else if (shape_data[i] > 0) {
      // input = [10,1], shape_data = [3,4,10,4]
      // --> output_dimensions_data = [3,4,10,4]
      if (input_shape[i] != 1) {
        NNADAPTER_CHECK_EQ(input_shape[i], shape_data[i])
            << "The value " << input_shape[i]
            << " of the non-singleton dimension does not match the "
               "corresponding value "
            << shape_data[i] << " in shape for expand_v2 op.";
        output_dimensions_data[i] = shape_data[i];
      } else {
        output_dimensions_data[i] = shape_data[i];
      }
    } else {
      // input = [10, 2], shape_data = [3,4,-1,-1]
      // --> output_dimensions_data = [3,4,10,2]
      NNADAPTER_CHECK_EQ(shape_data[i], -1)
          << "When the value in shape is negative for expand_v2 op, "
             "only -1 is supported, but the value received is "
          << shape_data[i];
      output_dimensions_data[i] = input_shape[i];
    }
  }
}

NNADAPTER_EXPORT bool ValidateExpand(const core::Operation* operation) {
  return true;
}

NNADAPTER_EXPORT int PrepareExpand(core::Operation* operation) {
  EXPAND_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operand->type;
  auto& shape_type = shape_operand->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeWithQuantParams(&output_type, input_type);

  uint32_t shape_count;
  int32_t* shape_data;
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
  output_type.lifetime = NNADAPTER_TEMPORARY_VARIABLE;
  UpdateExpandInferOutputShape(input_type.dimensions.data,
                               input_type.dimensions.count,
                               output_type.dimensions.data,
                               shape_count,
                               shape_data);
  for (uint32_t i = 0; i < input_type.dimensions.dynamic_count; i++) {
    UpdateExpandInferOutputShape(input_type.dimensions.dynamic_data[i],
                                 input_type.dimensions.count,
                                 output_type.dimensions.dynamic_data[i],
                                 shape_count,
                                 shape_data);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteExpand(core::Operation* operation) {
  EXPAND_OPERATION_EXTRACT_INPUTS_OUTPUTS

  auto in_dims_data = input_operand->type.dimensions.data;
  auto in_dims_count = input_operand->type.dimensions.count;
  std::vector<int32_t> in_dims(in_dims_data, in_dims_data + in_dims_count);
  auto out_dims_data = input_operand->type.dimensions.data;
  auto out_dims_count = input_operand->type.dimensions.count;
  std::vector<int32_t> out_dims(out_dims_data, out_dims_data + out_dims_count);
  auto in_dtype_length =
      GetOperandPrecisionDataLength(input_operand->type.precision);
  auto output_buffer = AllocateOperand(output_operand);
  int status = -1;
  switch (in_dtype_length) {
    case 1:
      status = math::expand(reinterpret_cast<int8_t*>(input_operand->buffer),
                            in_dims,
                            static_cast<int8_t*>(output_buffer),
                            out_dims);
      break;
    case 2:
      status = math::expand(reinterpret_cast<int16_t*>(input_operand->buffer),
                            in_dims,
                            static_cast<int16_t*>(output_buffer),
                            out_dims);
      break;
    case 4:
      status = math::expand(reinterpret_cast<int32_t*>(input_operand->buffer),
                            in_dims,
                            static_cast<int32_t*>(output_buffer),
                            out_dims);
      break;
    case 8:
      status = math::expand(reinterpret_cast<int64_t*>(input_operand->buffer),
                            in_dims,
                            static_cast<int64_t*>(output_buffer),
                            out_dims);
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Not support data type length: "
                           << in_dtype_length;
      break;
  }
  NNADAPTER_CHECK_EQ(status, 0);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
