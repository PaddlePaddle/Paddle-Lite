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

#include "operation/expand.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/hints.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertExpand(Converter* converter, core::Operation* operation) {
  EXPAND_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  // Input expr
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  // shape
  std::vector<int> expand_shape;
  if (IsTemporaryShapeOperand(shape_operand)) {
    if (IsOperandWithDynamicShape(shape_operand)) {
      NNADAPTER_LOG(FATAL) << "Unsupported dynamic shape";
      return NNADAPTER_INVALID_PARAMETER;
    } else {
      auto& temporary_shape = *(GetTemporaryShape(shape_operand));
      auto shape_count = temporary_shape.count;
      auto shape_data = temporary_shape.data;
      expand_shape.resize(shape_count);
      operation::UpdateExpandInferOutputShape(
          input_operand->type.dimensions.data,
          input_operand->type.dimensions.count,
          expand_shape.data(),
          shape_count,
          shape_data);
    }
  } else if (IsConstantOperand(shape_operand)) {
    auto shape_count = shape_operand->length / sizeof(int32_t);
    auto shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
    expand_shape.resize(shape_count);
    operation::UpdateExpandInferOutputShape(
        input_operand->type.dimensions.data,
        input_operand->type.dimensions.count,
        expand_shape.data(),
        shape_count,
        shape_data);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported shape lifetime: "
                         << OperandLifetimeCodeToString(
                                shape_operand->type.lifetime);
    return NNADAPTER_INVALID_PARAMETER;
  }

  auto expand_expr = converter->builder()->CreateBroadCastTo(
      input_expr, ConvertToXTCLArray<xtcl::Integer>(expand_shape));
  converter->UpdateExprMap(output_operand, expand_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
