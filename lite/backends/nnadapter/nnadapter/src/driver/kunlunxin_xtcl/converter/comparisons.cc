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

#include "operation/comparisons.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertComparisons(Converter* converter, core::Operation* operation) {
  COMPARISONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input0_expr = converter->GetMappedExpr(input0_operand);
  if (!input0_expr.defined()) {
    input0_expr = converter->ConvertOperand(input0_operand);
  }
  auto input1_expr = converter->GetMappedExpr(input1_operand);
  if (!input1_expr.defined()) {
    input1_expr = converter->ConvertOperand(input1_operand);
  }
  xtcl::xExpr comparisons_expr;
  switch (operation->type) {
#define CONVERT_COMPARISON(type, xtcl_type)                     \
  case NNADAPTER_##type: {                                      \
    comparisons_expr = converter->builder()->CreateBinaryOp(    \
        #xtcl_type, input0_expr, input1_expr);                  \
    converter->UpdateExprMap(output_operand, comparisons_expr); \
  } break;
    CONVERT_COMPARISON(EQUAL, equal);
    CONVERT_COMPARISON(NOT_EQUAL, not_equal);
    CONVERT_COMPARISON(GREATER, greater);
    CONVERT_COMPARISON(GREATER_EQUAL, greater_equal);
    CONVERT_COMPARISON(LESS, less);
    CONVERT_COMPARISON(LESS_EQUAL, less_equal);
#undef CONVERT_COMPARISON
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported comparison operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
