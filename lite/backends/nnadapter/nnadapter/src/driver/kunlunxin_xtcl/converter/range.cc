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

#include "operation/range.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertRange(Converter* converter, core::Operation* operation) {
  RANGE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto start_expr = converter->GetMappedExpr(start_operand);
  if (!start_expr.defined()) {
    start_expr = converter->ConvertOperand(start_operand);
  }
  auto limit_expr = converter->GetMappedExpr(limit_operand);
  if (!limit_expr.defined()) {
    limit_expr = converter->ConvertOperand(limit_operand);
  }
  auto delta_expr = converter->GetMappedExpr(delta_operand);
  if (!delta_expr.defined()) {
    delta_expr = converter->ConvertOperand(delta_operand);
  }
  auto range_expr = converter->builder()->CreateArange(
      start_expr,
      limit_expr,
      delta_expr,
      ConvertToXTCLDataType(start_operand->type.precision));
  converter->UpdateExprMap(output_operand, range_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
