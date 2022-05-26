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

#include "operation/split.h"
#include <vector>
#include "core/types.h"
#include "operation/math/split.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateSplit(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareSplit(core::Operation* operation) {
  SPLIT_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(IsConstantOperand(axis_operand));
  NNADAPTER_CHECK(IsConstantOperand(split_operand));

  // Infer the shape and type of output operands
  for (size_t i = 0; i < output_count; i++) {
    CopyOperandTypeExceptQuantParams(&output_operands[i]->type,
                                     input_operand->type);

    auto& out_dimensions = output_operands[i]->type.dimensions;
    out_dimensions.data[axis] = split[i];
    for (uint32_t j = 0; j < out_dimensions.dynamic_count; i++) {
      out_dimensions.dynamic_data[j][axis] = split[i];
    }

    NNADAPTER_VLOG(5) << "output" << i << ": "
                      << OperandToString(output_operands[i]);
  }

  if (IsTemporaryShapeOperand(input_operand)) {
    std::vector<int32_t*> output_data_ptrs;
    std::vector<std::vector<int32_t*>> dynamic_output_data_ptrs;
    std::vector<NNAdapterOperandDimensionType> dimension_types;

    for (size_t i = 0; i < output_count; i++) {
      output_operands[i]->type.lifetime = NNADAPTER_TEMPORARY_SHAPE;
      dimension_types[i].count = output_operands[i]->type.dimensions.data[0];
      dimension_types[i].dynamic_count =
          input_operand->type.dimensions.dynamic_count;
      output_data_ptrs.push_back(dimension_types[i].data);
      for (size_t i = 0; i < output_operands[i]->type.dimensions.dynamic_count;
           i++) {
        dynamic_output_data_ptrs[i].push_back(
            dimension_types[i].dynamic_data[i]);
      }
    }

    auto& temporary_shape = *(GetTemporaryShape(input_operand));
    NNADAPTER_CHECK(temporary_shape.data);
    NNADAPTER_CHECK(temporary_shape.data[0]);
    math::split<int32_t>(
        temporary_shape.data,
        std::vector<int32_t>({static_cast<int32_t>(temporary_shape.count)}),
        split.data(),
        split.size(),
        axis,
        output_data_ptrs);
    for (uint32_t i = 0; i < input_operand->type.dimensions.dynamic_count;
         i++) {
      math::split<int32_t>(
          temporary_shape.dynamic_data[i],
          std::vector<int32_t>({static_cast<int32_t>(temporary_shape.count)}),
          split.data(),
          split.size(),
          axis,
          dynamic_output_data_ptrs[i]);
    }
    for (size_t i = 0; i < output_count; i++) {
      SetTemporaryShape(output_operands[i], dimension_types[i]);
    }
  }
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteSplit(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
