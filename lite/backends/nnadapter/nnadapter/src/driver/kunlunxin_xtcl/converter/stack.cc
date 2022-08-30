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

#include "operation/stack.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertStack(Converter* converter, core::Operation* operation) {
  STACK_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto N = input_count - 1;
  xtcl::Array<xtcl::xExpr> input_exprs;
  for (int i = 0; i < N; i++) {
    auto input_operand = input_operands[i];
    auto input_expr = converter->GetMappedExpr(input_operand);
    if (!input_expr.defined()) {
      input_expr = converter->ConvertOperand(input_operand);
    }
    input_exprs.push_back(input_expr);
  }
  auto stack_expr = converter->builder()->CreateStack(
      xtcl::network::Tuple(input_exprs), axis);
  converter->UpdateExprMap(output_operand, stack_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
