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

#include "operation/pad.h"
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidatePad(const core::Operation* operation) {
  return false;
}

NNADAPTER_EXPORT int PreparePad(core::Operation* operation) {
  PAD_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(IsConstantOperand(pads_operand))
      << "Only support constant pads now.";
  uint32_t pads_count = pads_operand->length / sizeof(int32_t);
  NNADAPTER_CHECK_EQ(pads_count, 2 * input_operand->type.dimensions.count);
  auto pads_data = reinterpret_cast<int32_t*>(pads_operand->buffer);

  // Infer the shape and type of output operands
  auto input_type = input_operand->type;
  auto& output_type = output_operand->type;
  CopyOperandTypeExceptQuantParams(&output_type, input_type);
  auto infer_output_shape = [&](const int32_t* input_dimensions,
                                const int32_t input_dimension_count,
                                int32_t* output_dimensions) {
    for (int i = 0; i < input_dimension_count; i++) {
      if (input_dimensions[i] == NNADAPTER_UNKNOWN) {
        output_dimensions[i] = NNADAPTER_UNKNOWN;
      } else {
        output_dimensions[i] =
            input_dimensions[i] + pads_data[i * 2] + pads_data[2 * i + 1];
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

NNADAPTER_EXPORT int ExecutePad(core::Operation* operation) {
  return NNADAPTER_FEATURE_NOT_SUPPORTED;
}

}  // namespace operation
}  // namespace nnadapter
