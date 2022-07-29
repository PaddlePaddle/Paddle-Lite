// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "operation/unstack.h"
#include <vector>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateUnstack(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareUnstack(core::Operation* operation) {
  UNSTACK_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(IsConstantOperand(axis_operand));
  NNADAPTER_CHECK(IsConstantOperand(num_operand));

  // Infer the shape and type of output operands
  std::vector<int32_t> output_dimensions{};
  auto& input_dimensions = input_operand->type.dimensions;
  for (int i = 0; i < input_dimensions.count; i++) {
    if (i == axis) continue;
    output_dimensions.push_back(input_dimensions.data[i]);
  }

  std::vector<std::vector<int32_t>> output_dynamic_dimensions{};
  for (uint32_t i = 0; i < input_dimensions.dynamic_count; i++) {
    std::vector<int32_t> dynamic_dimensions{};
    for (uint32_t j = 0; j < input_dimensions.count; j++) {
      if (j == axis) continue;
      dynamic_dimensions.push_back(input_dimensions.dynamic_data[i][j]);
    }
    output_dynamic_dimensions.emplace_back(std::move(dynamic_dimensions));
  }

  for (size_t i = 0; i < output_count; i++) {
    CopyOperandTypeExceptQuantParams(&output_operands[i]->type,
                                     input_operand->type);
    auto& out_dimensions = output_operands[i]->type.dimensions;
    out_dimensions.count = output_dimensions.size();
    for (int i = 0; i < output_dimensions.size(); i++) {
      out_dimensions.data[i] = output_dimensions[i];
    }
    for (uint32_t i = 0; i < out_dimensions.dynamic_count; i++) {
      for (uint32_t j = 0; j < output_dynamic_dimensions[i].size(); j++) {
        out_dimensions.dynamic_data[i][j] = output_dynamic_dimensions[i][j];
      }
    }
    NNADAPTER_VLOG(5) << "output" << i << ": "
                      << OperandToString(output_operands[i]);
  }
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteUnstack(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
