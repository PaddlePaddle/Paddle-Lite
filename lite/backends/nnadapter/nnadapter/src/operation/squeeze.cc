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

#include "operation/squeeze.h"
#include <set>
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateSqueeze(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PrepareSqueeze(core::Operation* operation) {
  SQUEEZE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  auto input_type = input_operand->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeExceptQuantParams(&output_type, input_type);
  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions,
                                uint32_t input_dimensions_count) {
    size_t axes_count = axes.size();
    std::set<int32_t> squeezed_dims;
    if (axes_count == 0) {
      for (size_t idx = 0; idx < input_dimensions_count; ++idx) {
        if (input_dimensions[idx] == 1) {
          squeezed_dims.insert(idx);
        }
      }
    } else {
      for (size_t idx = 0; idx < axes_count; ++idx) {
        int axis =
            axes[idx] < 0 ? axes[idx] + input_dimensions_count : axes[idx];
        NNADAPTER_CHECK_GE(axis, 0)
            << "Invalid axis, the negative axis is out of range.";
        NNADAPTER_CHECK_EQ(input_dimensions[axis], 1)
            << "Invalid axis index, the axis that "
               "will be squeezed should be equal "
               "to 1.";
        NNADAPTER_CHECK_GE(input_dimensions_count, axis)
            << "Invalid axis index, axis needs to be smaller than the "
               "dimension of input";
        squeezed_dims.insert(axis);
      }
    }

    for (size_t in_idx = 0, out_idx = 0; in_idx < input_dimensions_count;
         ++in_idx) {
      if (!squeezed_dims.count(in_idx)) {
        output_dimensions[out_idx++] = input_dimensions[in_idx];
      }
    }
    if (static_cast<uint32_t>(squeezed_dims.size()) == input_dimensions_count) {
      output_dimensions[0] = 1;
    }
    return squeezed_dims.size();
  };

  auto squeezed_dimensions_count =
      infer_output_shape(input_type.dimensions.data,
                         output_type.dimensions.data,
                         input_type.dimensions.count);
  output_type.dimensions.count =
      input_type.dimensions.count - squeezed_dimensions_count;
  output_type.dimensions.count =
      output_type.dimensions.count > 0 ? output_type.dimensions.count : 1;
  for (uint32_t i = 0; i < input_type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_type.dimensions.dynamic_data[i],
                       output_type.dimensions.dynamic_data[i],
                       input_type.dimensions.count);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteSqueeze(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
