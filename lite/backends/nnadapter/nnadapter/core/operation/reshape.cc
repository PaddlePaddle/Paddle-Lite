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

#include "core/operation/reshape.h"
#include <vector>
#include "core/hal/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

int PrepareReshape(hal::Operation* operation) {
  RESHAPE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  NNADAPTER_CHECK(IsConstantOperand(shape_operand))
      << "Only support constant shape now.";
  auto input_type = input_operand->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeWithQuantParams(&output_type, input_type);
  output_type.dimensions.count = shape_count;
  for (uint32_t i = 0; i < shape_count; i++) {
    if (shape_data[i] == 0 &&
        input_type.dimensions.data[i] != NNADAPTER_UNKNOWN) {
      shape_data[i] = input_type.dimensions.data[i];
    }
  }

  auto infer_output_shape = [&](int32_t* input_dimensions_data,
                                uint32_t input_dimensions_count,
                                int32_t* output_dimensions_data) {
    for (uint32_t i = 0; i < shape_count; i++) {
      output_dimensions_data[i] =
          shape_data[i] == 0 ? input_dimensions_data[i] : shape_data[i];
    }
    int64_t size = 1;
    for (uint32_t i = 0; i < input_dimensions_count; i++) {
      if (input_dimensions_data[i] == NNADAPTER_UNKNOWN) {
        size = -1;
        break;
      } else {
        size *= static_cast<int64_t>(input_dimensions_data[i]);
      }
    }
    if (size != -1) {
      int32_t unk_idx = -1;
      for (uint32_t i = 0; i < shape_count; i++) {
        if (output_dimensions_data[i] == -1) {
          NNADAPTER_CHECK_EQ(unk_idx, -1) << "Should only has one unk idx.";
          unk_idx = i;
        } else {
          size /= static_cast<int64_t>(output_dimensions_data[i]);
        }
      }
      if (unk_idx != -1) {
        output_dimensions_data[unk_idx] = size;
      }
    }
  };
  infer_output_shape(input_type.dimensions.data,
                     input_type.dimensions.count,
                     output_type.dimensions.data);
  for (uint32_t i = 0; i < input_type.dimensions.dynamic_count; i++) {
    infer_output_shape(input_type.dimensions.dynamic_data[i],
                       input_type.dimensions.count,
                       output_type.dimensions.dynamic_data[i]);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
