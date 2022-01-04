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

#include "core/operation/concat.h"
#include <vector>
#include "core/hal/types.h"
#include "core/math/concat_compute.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareConcat(hal::Operation* operation) {
  CONCAT_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type,
                                   input_operands[0]->type);

  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions,
                                const uint32_t input_dimension_count) {
    NNADAPTER_CHECK_EQ(input_dimension_count,
                       output_operand->type.dimensions.count);
    for (uint32_t i = 0; i < input_dimension_count; i++) {
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

  // For operand shape info
  for (size_t i = 1; i < input_count - 1; i++) {
    infer_output_shape(input_operands[i]->type.dimensions.data,
                       output_operand->type.dimensions.data,
                       input_operands[i]->type.dimensions.count);
  }
  for (uint32_t i = 0; i < output_operand->type.dimensions.dynamic_count; i++) {
    for (size_t j = 1; j < input_count - 1; j++) {
      infer_output_shape(input_operands[j]->type.dimensions.dynamic_data[i],
                         output_operand->type.dimensions.dynamic_data[i],
                         input_operands[j]->type.dimensions.count);
    }
  }

  // For operand dims value info
  bool temporary_shape_flag = false;
  for (size_t i = 0; i < input_count - 1; i++) {
    if (input_operands[i]->type.lifetime == NNADAPTER_TEMPORARY_SHAPE) {
      temporary_shape_flag = true;
    }
  }
  for (size_t i = 0; i < input_count - 1; i++) {
    if (temporary_shape_flag &&
        input_operands[i]->type.lifetime != NNADAPTER_TEMPORARY_SHAPE &&
        !IsConstantOperand(input_operands[i])) {
      NNADAPTER_LOG(FATAL) << "Tempory shape operand can only be used with "
                              "constant operand, current operand lifetime is "
                           << OperandLifetimeCodeToString(
                                  input_operands[i]->type.lifetime);
    }
  }
  if (temporary_shape_flag) {
    // Static shape
    std::vector<int32_t*> concat_inputs;
    std::vector<std::vector<int32_t>> concat_inputs_dims;
    for (size_t i = 0; i < input_count - 1; i++) {
      if (input_operands[i]->type.lifetime == NNADAPTER_TEMPORARY_SHAPE) {
        auto& tempory_shape_info = *(GetTemporyShapeInfo(input_operands[i]));
        NNADAPTER_CHECK(tempory_shape_info.data);
        NNADAPTER_CHECK(tempory_shape_info.data[0]);
        concat_inputs.push_back(tempory_shape_info.data);
      } else {  // Constant Operand
        auto input_data = reinterpret_cast<int32_t*>(input_operands[i]->buffer);
        concat_inputs.push_back(input_data);
      }
      std::vector<int32_t> input_dims;
      for (uint32_t j = 0; j < input_operands[i]->type.dimensions.count; j++) {
        input_dims.push_back(input_operands[i]->type.dimensions.data[j]);
      }
      concat_inputs_dims.push_back(input_dims);
    }
    // Dynamic shape
    std::vector<std::vector<int32_t*>> dynamic_concat_inputs;
    std::vector<std::vector<std::vector<int32_t>>> dynamic_concat_inputs_dims;
    for (uint32_t i = 0; i < output_operand->type.dimensions.dynamic_count;
         i++) {
      for (size_t j = 0; j < input_count - 1; j++) {
        if (input_operands[j]->type.lifetime == NNADAPTER_TEMPORARY_SHAPE) {
          auto& tempory_shape_info = *(GetTemporyShapeInfo(input_operands[i]));
          NNADAPTER_CHECK(tempory_shape_info.data);
          NNADAPTER_CHECK(tempory_shape_info.data[0]);
          dynamic_concat_inputs[i].push_back(tempory_shape_info.data);
        } else {  // Constant Operand
          auto input_data =
              reinterpret_cast<int32_t*>(input_operands[j]->buffer);
          dynamic_concat_inputs[i].push_back(input_data);
        }
        std::vector<int32_t> input_dims;
        for (uint32_t k = 0; k < input_operands[j]->type.dimensions.count;
             k++) {
          input_dims.push_back(input_operands[j]->type.dimensions.data[k]);
        }
        dynamic_concat_inputs_dims[i].push_back(input_dims);
      }
    }

    NNAdapterOperandDimensionType dimension_type;
    dimension_type.count = output_operand->type.dimensions.data[0];
    dimension_type.dynamic_count =
        output_operand->type.dimensions.dynamic_count;
    ConcatCompute<int32_t>(
        concat_inputs, concat_inputs_dims, axis, dimension_type.data);
    for (uint32_t i = 0; i < output_operand->type.dimensions.dynamic_count;
         i++) {
      ConcatCompute<int32_t>(dynamic_concat_inputs[i],
                             dynamic_concat_inputs_dims[i],
                             axis,
                             dimension_type.dynamic_data[i]);
    }
    output_operand->type.lifetime = NNADAPTER_TEMPORARY_SHAPE;
    SetTemporyShapeInfo(output_operand, dimension_type);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
