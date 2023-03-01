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

#include "operation/meshgrid.h"
#include <vector>
#include "core/types.h"
#include "operation/math/meshgrid.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateMeshgrid(const core::Operation* operation) {
  return true;
}

NNADAPTER_EXPORT int PrepareMeshgrid(core::Operation* operation) {
  MESHGRID_OPERATION_EXTRACT_INPUTS_OUTPUTS
  for (auto output_operand : output_operands) {
    CopyOperandTypeExceptQuantParams(&output_operand->type,
                                     input_operands[0]->type);
    output_operand->type.dimensions.count = input_count;
  }

  // Infer the shape and type of output operands
  std::vector<int> output_shape(input_count, 0);
  for (int i = 0; i < input_count; i++) {
    NNADAPTER_CHECK_EQ(input_operands[i]->type.dimensions.count, 1)
        << "Input" << i << "operand dimensions should be 1D.";
    output_shape[i] = input_operands[i]->type.dimensions.data[0];
  }
  for (auto output_operand : output_operands) {
    for (int j = 0; j < input_count; j++) {
      output_operand->type.dimensions.data[j] = output_shape[j];
    }
  }
  // Dynamic shape
  if (input_operands[0]->type.dimensions.dynamic_count != 0) {
    for (uint32_t i = 0; i < input_operands[0]->type.dimensions.dynamic_count;
         i++) {
      std::vector<int> output_shape(input_count, 0);
      for (size_t j = 0; j < input_count; j++) {
        output_shape[i] = input_operands[i]->type.dimensions.dynamic_data[i][0];
      }
      for (auto output_operand : output_operands) {
        for (int j = 0; j < input_count; j++) {
          output_operand->type.dimensions.dynamic_data[i][j] = output_shape[j];
        }
      }
    }
  }

  for (size_t i = 0; i < output_count; i++) {
    NNADAPTER_VLOG(5) << "output" << i << ": "
                      << OperandToString(output_operands[i]);
  }
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteMeshgrid(core::Operation* operation) {
  MESHGRID_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Allocate and calculate the output operands
  int status = -1;
  auto input_precision = input_operands[0]->type.precision;
  std::vector<std::vector<int32_t>> input_shapes;
  for (int i = 0; i < input_count; i++) {
    auto in_dims = input_operands[i]->type.dimensions.data;
    auto in_dims_count = input_operands[i]->type.dimensions.count;
    input_shapes.push_back(
        std::vector<int32_t>(in_dims, in_dims + in_dims_count));
  }
  switch (input_precision) {
    case NNADAPTER_FLOAT32: {
      std::vector<float*> input_datas;
      for (int i = 0; i < input_count; i++) {
        input_datas.push_back(
            reinterpret_cast<float*>(input_operands[i]->buffer));
      }
      std::vector<float*> output_datas;
      for (int i = 0; i < output_count; i++) {
        output_datas.push_back(
            reinterpret_cast<float*>(AllocateOperand(output_operands[i])));
      }
      status = math::meshgrid(input_datas, input_shapes, output_datas);
    } break;
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER: {
      std::vector<int8_t*> input_datas;
      for (int i = 0; i < input_count; i++) {
        input_datas.push_back(
            reinterpret_cast<int8_t*>(input_operands[i]->buffer));
      }
      std::vector<int8_t*> output_datas;
      for (int i = 0; i < output_count; i++) {
        output_datas.push_back(
            reinterpret_cast<int8_t*>(AllocateOperand(output_operands[i])));
      }
      status = math::meshgrid(input_datas, input_shapes, output_datas);
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
