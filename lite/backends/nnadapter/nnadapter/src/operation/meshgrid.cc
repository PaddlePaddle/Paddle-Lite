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
  return false;
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
  // SOFTMAX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // // Allocate and calculate the output operands
  // int status = -1;
  // auto& input_type = input_operand->type;
  // auto input_shape = std::vector<int32_t>(
  //     input_type.dimensions.data,
  //     input_type.dimensions.data + input_type.dimensions.count);
  // const auto input_buffer = input_operand->buffer;
  // NNADAPTER_CHECK(input_buffer);
  // auto& output_type = output_operand->type;
  // auto output_buffer = AllocateOperand(output_operand);
  // NNADAPTER_CHECK_EQ(input_type.precision, output_type.precision);
  // if (input_type.precision == NNADAPTER_FLOAT32) {
  //   const auto input_data = reinterpret_cast<const float*>(input_buffer);
  //   auto output_data = reinterpret_cast<float*>(output_buffer);
  //   status = math::softmax<float>(input_data, input_shape, axis,
  //   output_data);
  // } else {
  //   NNADAPTER_LOG(FATAL) << "Unsupported precision code("
  //                        <<
  //                        OperandPrecisionCodeToString(input_type.precision)
  //                        << ") for " <<
  //                        OperationTypeToString(operation->type)
  //                        << " is found!";
  // }
  // NNADAPTER_CHECK_EQ(status, 0);
  // return NNADAPTER_NO_ERROR;
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
