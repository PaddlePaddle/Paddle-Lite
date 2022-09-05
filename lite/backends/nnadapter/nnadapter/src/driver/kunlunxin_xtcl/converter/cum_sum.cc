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

#include "operation/cum_sum.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertCumSum(Converter* converter, core::Operation* operation) {
  CUM_SUM_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  // Reverse: whether to perform the cumsum in the reversed direction
  xtcl::xExpr cum_sum_expr;
  if (reverse) {
    auto reverse_expr = converter->builder()->CreateReverse(input_expr, axis);
    cum_sum_expr = converter->builder()->CreateCumsum(
        reverse_expr,
        axis,
        ConvertToXTCLDataType(input_operand->type.precision),
        exclusive);
    cum_sum_expr = converter->builder()->CreateReverse(cum_sum_expr, axis);
  } else {
    cum_sum_expr = converter->builder()->CreateCumsum(
        input_expr,
        axis,
        ConvertToXTCLDataType(input_operand->type.precision),
        exclusive);
  }
  converter->UpdateExprMap(output_operand, cum_sum_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
