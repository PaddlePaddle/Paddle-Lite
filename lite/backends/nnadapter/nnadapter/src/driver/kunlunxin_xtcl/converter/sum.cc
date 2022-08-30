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

#include "operation/sum.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertSum(Converter* converter, core::Operation* operation) {
  SUM_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_operand = input_operands[0];
  auto sum_expr = converter->GetMappedExpr(input_operand);
  if (!sum_expr.defined()) {
    sum_expr = converter->ConvertOperand(input_operand);
  }
  for (int i = 1; i < input_count; i++) {
    auto input_operand = input_operands[i];
    auto input_expr = converter->GetMappedExpr(input_operand);
    if (!input_expr.defined()) {
      input_expr = converter->ConvertOperand(input_operand);
    }
    sum_expr =
        converter->builder()->CreateBinaryOp("add", input_expr, sum_expr);
  }
  converter->UpdateExprMap(output_operand, sum_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
