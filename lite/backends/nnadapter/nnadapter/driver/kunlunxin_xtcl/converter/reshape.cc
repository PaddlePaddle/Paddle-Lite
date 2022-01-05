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
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertReshape(Converter* converter, hal::Operation* operation) {
  RESHAPE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  uint32_t shape_count = 0;
  int32_t* shape_data = nullptr;
  if (IsTemporaryShapeOperand(shape_operand)) {
    shape_count = output_operand->type.dimensions.count;
    shape_data = output_operand->type.dimensions.data;
  } else if (IsConstantOperand(shape_operand)) {
    shape_count = shape_operand->length / sizeof(int32_t);
    shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
    for (uint32_t i = 0; i < shape_count; i++) {
      if (shape_data[i] == 0 &&
          input_operand->type.dimensions.data[i] != NNADAPTER_UNKNOWN) {
        shape_data[i] = input_operand->type.dimensions.data[i];
      }
    }
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported shape lifetime: "
                         << OperandLifetimeCodeToString(
                                shape_operand->type.lifetime);
    return NNADAPTER_INVALID_PARAMETER;
  }
  auto reshape_expr = converter->builder()->CreateReshape(
      input_expr, ConvertToXTCLArray<xtcl::Integer>(shape_data, shape_count));
  converter->UpdateExprMap(output_operand, reshape_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
