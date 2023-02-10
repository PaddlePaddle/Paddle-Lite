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

#include "operation/stack.h"
#include "core/types.h"
#include "operation/math/stack.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateStack(const core::Operation* operation) {
  return true;
}

NNADAPTER_EXPORT int PrepareStack(core::Operation* operation) {
  STACK_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operands[0]->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeWithQuantParams(&output_type, input_type);
  auto infer_output_shape = [&](const int32_t* input_dimensions_data,
                                uint32_t input_dimensions_count,
                                int32_t* output_dimensions_data,
                                uint32_t* output_dimensions_count) {
    for (uint32_t i = 0; i < axis; i++) {
      output_dimensions_data[i] = input_dimensions_data[i];
    }
    output_dimensions_data[axis] = input_count - 1;
    for (uint32_t i = axis; i < input_dimensions_count; i++) {
      output_dimensions_data[i + 1] = input_dimensions_data[i];
    }
    *output_dimensions_count = input_dimensions_count + 1;
  };
  infer_output_shape(input_type.dimensions.data,
                     input_type.dimensions.count,
                     output_type.dimensions.data,
                     &output_type.dimensions.count);
  for (uint32_t i = 0; i < input_type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_type.dimensions.dynamic_data[i],
                       input_type.dimensions.count,
                       output_type.dimensions.dynamic_data[i],
                       &output_type.dimensions.count);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteStack(core::Operation* operation) {
  STACK_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Allocate and calculate the output operands
  int status = -1;
  auto output_buffer = AllocateOperand(output_operand);
  auto input_precision = input_operands[0]->type.precision;
  std::vector<std::vector<int32_t>> input_shapes;
  for (int i = 0; i < input_count - 1; i++) {
    input_shapes.push_back(
        std::vector<int32_t>(input_operands[i]->type.dimensions.data,
                             input_operands[i]->type.dimensions.data +
                                 input_operands[i]->type.dimensions.count));
  }
  switch (input_precision) {
    case NNADAPTER_INT32: {
      std::vector<int32_t*> input_datas;
      for (int i = 0; i < input_count - 1; i++) {
        input_datas.push_back(
            reinterpret_cast<int32_t*>(input_operands[i]->buffer));
      }
      status = math::stack(input_datas,
                           input_shapes,
                           axis,
                           reinterpret_cast<int32_t*>(output_buffer));
    } break;
    case NNADAPTER_FLOAT32: {
      std::vector<float*> input_datas;
      for (int i = 0; i < input_count - 1; i++) {
        input_datas.push_back(
            reinterpret_cast<float*>(input_operands[i]->buffer));
      }
      status = math::stack(input_datas,
                           input_shapes,
                           axis,
                           reinterpret_cast<float*>(output_buffer));
    } break;
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER: {
      std::vector<int8_t*> input_datas;
      for (int i = 0; i < input_count - 1; i++) {
        input_datas.push_back(
            reinterpret_cast<int8_t*>(input_operands[i]->buffer));
      }
      status = math::stack(input_datas,
                           input_shapes,
                           axis,
                           reinterpret_cast<int8_t*>(output_buffer));
    } break;
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

}  // namespace operation
}  // namespace nnadapter
