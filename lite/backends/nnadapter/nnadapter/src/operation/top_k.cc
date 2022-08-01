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

#include "operation/top_k.h"
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateTopK(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareTopK(core::Operation* operation) {
  TOP_K_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operand->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeExceptQuantParams(&output_type, input_type);
  NNADAPTER_CHECK_GE(axis, 0);
  NNADAPTER_CHECK_LT(axis, input_type.dimensions.count);
  output_type.dimensions.data[axis] = k;
  for (uint32_t i = 0; i < input_type.dimensions.dynamic_count; i++) {
    output_type.dimensions.dynamic_data[i][axis] = static_cast<int32_t>(k);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  auto& indices_type = indices_operand->type;
  CopyOperandTypeExceptQuantParams(&indices_type, output_type);
  indices_type.precision =
      static_cast<NNAdapterOperandPrecisionCode>(return_indices_dtype);
  NNADAPTER_VLOG(5) << "indices: " << OperandToString(indices_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteTopK(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
