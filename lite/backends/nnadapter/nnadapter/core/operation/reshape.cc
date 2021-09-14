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
  auto in_type = input_operand->type;
  auto& out_type = output_operand->type;
  CopyOperandTypeWithQuantParams(&out_type, in_type);
  out_type.dimension_count = shape_count;
  for (uint32_t i = 0; i < shape_count; i++) {
    if (shape_data[i] == 0 && in_type.dimensions[i] != NNADAPTER_UNKNOWN) {
      shape_data[i] = in_type.dimensions[i];
    }
  }

  auto infer_output_shape = [&](int32_t* input_dimensions,
                                int32_t* output_dimensions) {
    for (uint32_t i = 0; i < shape_count; i++) {
      output_dimensions[i] =
          shape_data[i] == 0 ? input_dimensions[i] : shape_data[i];
    }
    int64_t size = 1;
    for (uint32_t i = 0; i < in_type.dimension_count; i++) {
      if (in_type.dimensions[i] == NNADAPTER_UNKNOWN) {
        size = -1;
        break;
      } else {
        size *= static_cast<int64_t>(in_type.dimensions[i]);
      }
    }
    if (size != -1) {
      int32_t unk_idx = -1;
      for (uint32_t i = 0; i < shape_count; i++) {
        if (output_dimensions[i] == -1) {
          NNADAPTER_CHECK_EQ(unk_idx, -1) << "Should only has one unk idx.";
          unk_idx = i;
        } else {
          size /= static_cast<int64_t>(output_dimensions[i]);
        }
      }
      if (unk_idx != -1) {
        output_dimensions[unk_idx] = size;
      }
    }
  };
  infer_output_shape(in_type.dimensions, out_type.dimensions);
  for (uint32_t i = 0; i < in_type.dynamic_dimension_count; i++) {
    infer_output_shape(in_type.dynamic_dimensions[i],
                       out_type.dynamic_dimensions[i]);
  }
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
