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

#include "operation/elementwise.h"
#include <algorithm>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT void CalcEltwiseBinaryOperationsOutputSize(
    const NNAdapterOperandType& input0_type,
    const NNAdapterOperandType& input1_type,
    NNAdapterOperandType* output_type) {
  // Infer the shape and type of output operands
  int32_t input0_size = input0_type.dimensions.count;
  int32_t input1_size = input1_type.dimensions.count;
  int32_t max_size = std::max(input0_size, input1_size);
  auto infer_output_shape = [&](const int32_t* input0_dimensions_data,
                                const int32_t* input1_dimensions_data,
                                int32_t* output_dimensions_data) {
    int32_t input0_i = input0_size - 1;
    int32_t input1_i = input1_size - 1;
    for (int32_t i = max_size - 1; i >= 0; i--) {
      if (input0_i < 0) {
        NNADAPTER_CHECK_GE(input1_i, 0);
        output_dimensions_data[i] = input1_dimensions_data[input1_i];
      } else if (input1_i < 0) {
        NNADAPTER_CHECK_GE(input0_i, 0);
        output_dimensions_data[i] = input0_dimensions_data[input0_i];
      } else {
        int32_t input0_data = input0_dimensions_data[input0_i];
        int32_t input1_data = input1_dimensions_data[input1_i];
        if (input0_data == input1_data) {
          output_dimensions_data[i] = input0_data;
        } else if (input0_data == 1) {
          output_dimensions_data[i] = input1_data;
        } else if (input1_data == 1) {
          output_dimensions_data[i] = input0_data;
        } else {
          NNADAPTER_LOG(ERROR) << "Cannot broadcast input0: " << input0_data
                               << ", input1: " << input1_data;
        }
      }
      input0_i--;
      input1_i--;
    }
  };
  output_type->dimensions.count = max_size;
  output_type->dimensions.dynamic_count = input0_type.dimensions.dynamic_count;
  infer_output_shape(input0_type.dimensions.data,
                     input1_type.dimensions.data,
                     output_type->dimensions.data);
  for (uint32_t i = 0; i < input0_type.dimensions.dynamic_count; i++) {
    infer_output_shape(input0_type.dimensions.dynamic_data[i],
                       input1_type.dimensions.dynamic_data[i],
                       output_type->dimensions.dynamic_data[i]);
  }
}

int PrepareElementwise(core::Operation* operation) {
  ELEMENTWISE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  if (IsConstantOperand(input0_operand) && !IsConstantOperand(input1_operand)) {
    input0_operand->type.dimensions.dynamic_count =
        input1_operand->type.dimensions.dynamic_count;
    for (size_t i = 0; i < input0_operand->type.dimensions.dynamic_count; i++) {
      for (size_t j = 0; j < input1_operand->type.dimensions.count; j++) {
        input0_operand->type.dimensions.dynamic_data[i][j] = 1;
      }
    }
  } else if (IsConstantOperand(input1_operand) &&
             !IsConstantOperand(input0_operand)) {
    input1_operand->type.dimensions.dynamic_count =
        input0_operand->type.dimensions.dynamic_count;
    for (size_t i = 0; i < input1_operand->type.dimensions.dynamic_count; i++) {
      for (size_t j = 0; j < input0_operand->type.dimensions.count; j++) {
        input1_operand->type.dimensions.dynamic_data[i][j] = 1;
      }
    }
  }

  CalcEltwiseBinaryOperationsOutputSize(
      input0_operand->type, input1_operand->type, &output_operand->type);
  output_operand->type.precision = input0_operand->type.precision;
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
