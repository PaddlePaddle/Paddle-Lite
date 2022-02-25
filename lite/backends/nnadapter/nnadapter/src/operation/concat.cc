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

#include "operation/concat.h"
#include <vector>
#include "core/types.h"
#include "operation/math/concat.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareConcat(core::Operation* operation) {
  CONCAT_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type,
                                   input_operands[0]->type);

  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions,
                                const uint32_t input_dimension_count) {
    NNADAPTER_CHECK_EQ(input_dimension_count,
                       output_operand->type.dimensions.count);
    for (size_t i = 0; i < input_dimension_count; i++) {
      if (output_dimensions[i] == NNADAPTER_UNKNOWN ||
          input_dimensions[i] == NNADAPTER_UNKNOWN) {
        output_dimensions[i] = NNADAPTER_UNKNOWN;
        continue;
      }
      if (i == axis) {
        output_dimensions[i] += input_dimensions[i];
      } else {
        NNADAPTER_CHECK_EQ(output_dimensions[i], input_dimensions[i]);
      }
    }
  };

  // Derive the dimensions of concat's output operand
  for (size_t i = 1; i < input_count - 1; i++) {
    infer_output_shape(input_operands[i]->type.dimensions.data,
                       output_operand->type.dimensions.data,
                       input_operands[i]->type.dimensions.count);
  }
  for (size_t i = 0; i < output_operand->type.dimensions.dynamic_count; i++) {
    for (size_t j = 1; j < input_count - 1; j++) {
      infer_output_shape(input_operands[j]->type.dimensions.dynamic_data[i],
                         output_operand->type.dimensions.dynamic_data[i],
                         input_operands[j]->type.dimensions.count);
    }
  }

  // Check if inputs are temporary shape or constant operands
  bool has_temporary_shape = false;
  for (size_t i = 0; i < input_count - 1; i++) {
    if (IsTemporaryShapeOperand(input_operands[i])) {
      has_temporary_shape = true;
    }
  }
  for (size_t i = 0; i < input_count - 1; i++) {
    if (has_temporary_shape && !IsTemporaryShapeOperand(input_operands[i]) &&
        !IsConstantOperand(input_operands[i])) {
      NNADAPTER_LOG(FATAL) << "Temporary shape operand can only be used with "
                              "constant operand, current operand lifetime is "
                           << OperandLifetimeCodeToString(
                                  input_operands[i]->type.lifetime);
    }
  }
  // Derive the shape value of concat's output operand
  if (has_temporary_shape) {
    auto peek_input_data = [&](
        std::vector<int32_t*>& input_data_ptrs,
        std::vector<std::vector<int32_t>>& input_shapes) {
      for (size_t i = 0; i < input_count - 1; i++) {
        if (IsTemporaryShapeOperand(input_operands[i])) {
          auto& temporary_shape = *(GetTemporaryShape(input_operands[i]));
          NNADAPTER_CHECK(temporary_shape.data);
          NNADAPTER_CHECK(temporary_shape.data[0]);
          input_data_ptrs.push_back(temporary_shape.data);
        } else {  // Constant Operand
          auto input_data =
              reinterpret_cast<int32_t*>(input_operands[i]->buffer);
          input_data_ptrs.push_back(input_data);
        }
        std::vector<int32_t> input_dims;
        for (size_t j = 0; j < input_operands[i]->type.dimensions.count; j++) {
          input_dims.push_back(input_operands[i]->type.dimensions.data[j]);
        }
        input_shapes.push_back(input_dims);
      }
    };
    // Static shape
    std::vector<int32_t*> input_data_ptrs;
    std::vector<std::vector<int32_t>> input_shapes;
    peek_input_data(input_data_ptrs, input_shapes);
    // Dynamic shape
    std::vector<std::vector<int32_t*>> dynamic_input_data_ptrs;
    std::vector<std::vector<std::vector<int32_t>>> dynamic_input_shapes;
    for (size_t i = 0; i < output_operand->type.dimensions.dynamic_count; i++) {
      peek_input_data(dynamic_input_data_ptrs[i], dynamic_input_shapes[i]);
    }

    NNAdapterOperandDimensionType dimension_type;
    dimension_type.count = output_operand->type.dimensions.data[0];
    dimension_type.dynamic_count =
        output_operand->type.dimensions.dynamic_count;
    math::concat<int32_t>(
        input_data_ptrs, input_shapes, axis, dimension_type.data);
    for (size_t i = 0; i < output_operand->type.dimensions.dynamic_count; i++) {
      math::concat<int32_t>(dynamic_input_data_ptrs[i],
                            dynamic_input_shapes[i],
                            axis,
                            dimension_type.dynamic_data[i]);
    }
    output_operand->type.lifetime = NNADAPTER_TEMPORARY_SHAPE;
    SetTemporaryShape(output_operand, dimension_type);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
