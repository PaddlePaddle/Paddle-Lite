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

#include "operation/cum_sum.h"
#include "core/types.h"
#include "operation/math/cum_sum.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateCumSum(const core::Operation* operation) {
  return true;
}

NNADAPTER_EXPORT int PrepareCumSum(core::Operation* operation) {
  CUM_SUM_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type, input_operand->type);
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteCumSum(core::Operation* operation) {
  CUM_SUM_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Allocate and calculate the output operands
  int status = -1;
  auto output_buffer = AllocateOperand(output_operand);
  auto input_precision = input_operand->type.precision;
  auto in_dims_data = input_operand->type.dimensions.data;
  auto in_dims_count = input_operand->type.dimensions.count;
  std::vector<int32_t> in_dims(in_dims_data, in_dims_data + in_dims_count);
  switch (input_precision) {
    case NNADAPTER_FLOAT32:
      status = math::cum_sum(reinterpret_cast<float*>(input_operand->buffer),
                             in_dims,
                             reinterpret_cast<float*>(output_buffer),
                             axis,
                             reverse,
                             exclusive);
      break;
    case NNADAPTER_INT32:
      status = math::cum_sum(reinterpret_cast<int32_t*>(input_operand->buffer),
                             in_dims,
                             reinterpret_cast<int32_t*>(output_buffer),
                             axis,
                             reverse,
                             exclusive);
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

}  // namespace operation
}  // namespace nnadapter
