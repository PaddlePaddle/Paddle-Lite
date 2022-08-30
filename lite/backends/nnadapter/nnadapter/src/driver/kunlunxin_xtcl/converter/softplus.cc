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

#include "operation/softplus.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertSoftplus(Converter* converter, core::Operation* operation) {
  SOFTPLUS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  auto beta_expr = converter->builder()->CreateConstant({1}, beta);
  auto constant_expr = converter->builder()->CreateConstant({1}, 1.0f);
  // `output` = log(1 + exp^(`beta` * `input`)) / `beta`
  // When: `beta` * `input` > threshold:  `output` = `input`
  xtcl::xExpr softplus_expr =
      converter->builder()->CreateBinaryOp("multiply", beta_expr, input_expr);
  softplus_expr = converter->builder()->CreateUnaryOp("exp", softplus_expr);
  softplus_expr =
      converter->builder()->CreateBinaryOp("add", constant_expr, softplus_expr);
  softplus_expr = converter->builder()->CreateUnaryOp("log", softplus_expr);
  softplus_expr =
      converter->builder()->CreateBinaryOp("divide", softplus_expr, beta_expr);
  converter->UpdateExprMap(output_operand, softplus_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
