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

#include "operation/fill_like.h"
#include "core/types.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace operation {

NNADAPTER_EXPORT bool ValidateFillLike(const core::Operation* operation) {
  return true;
}

NNADAPTER_EXPORT int PrepareFillLike(core::Operation* operation) {
  FILL_LIKE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Infer the shape and type of output operands
  CopyOperandTypeExceptQuantParams(&output_operand->type, input_operand->type);
  output_operand->type.precision = value_operand->type.precision;
  NNADAPTER_VLOG(5) << "output: " << OperandToString(output_operand);
  return NNADAPTER_NO_ERROR;
}

NNADAPTER_EXPORT int ExecuteFillLike(core::Operation* operation) {
  FILL_LIKE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Allocate and calculate the output operands
  auto output_buffer = AllocateOperand(output_operand);
  auto out_uint8 = reinterpret_cast<uint8_t*>(output_buffer);
  auto out_length = output_operand->length;
  auto value_uint8 = reinterpret_cast<uint8_t*>(value_operand->buffer);
  auto value_length = value_operand->length;
  for (int i = 0; i < out_length; i += value_length) {
    memcpy(out_uint8, value_uint8, value_length);
    out_uint8 += value_length;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace operation
}  // namespace nnadapter
