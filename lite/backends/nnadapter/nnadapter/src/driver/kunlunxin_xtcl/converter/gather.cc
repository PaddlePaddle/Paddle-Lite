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

#include "operation/gather.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertGather(Converter* converter, core::Operation* operation) {
  GATHER_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  auto indices_expr = converter->GetMappedExpr(indices_operand);
  if (!indices_expr.defined()) {
    indices_expr = converter->ConvertOperand(indices_operand);
  }
  if (indices_operand->type.dimensions.count != 1) {
    indices_expr = converter->builder()->CreateReshape(indices_expr, {-1});
  }
  // Reshape the gather expr with the inferred shape as the output expr
  auto gather_expr =
      converter->builder()->CreateTake(input_expr, indices_expr, axis);
  converter->UpdateExprMap(output_operand, gather_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
