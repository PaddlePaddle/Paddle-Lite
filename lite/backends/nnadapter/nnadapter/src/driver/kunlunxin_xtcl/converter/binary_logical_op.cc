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

#include "operation/binary_logical_op.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertBinaryLogicalOp(Converter* converter, core::Operation* operation) {
  BINARY_LOGICAL_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input0_expr = converter->GetMappedExpr(input0_operand);
  if (!input0_expr.defined()) {
    input0_expr = converter->ConvertOperand(input0_operand);
  }
  auto input1_expr = converter->GetMappedExpr(input1_operand);
  if (!input1_expr.defined()) {
    input1_expr = converter->ConvertOperand(input1_operand);
  }
  xtcl::xExpr binary_logical_expr;
  switch (operation->type) {
#define CONVERT_BINARY_LOGICAL_OP(type, xtcl_type)                 \
  case NNADAPTER_##type: {                                         \
    binary_logical_expr = converter->builder()->CreateBinaryOp(    \
        #xtcl_type, input0_expr, input1_expr);                     \
    converter->UpdateExprMap(output_operand, binary_logical_expr); \
  } break;
    CONVERT_BINARY_LOGICAL_OP(AND, logical_and);
    CONVERT_BINARY_LOGICAL_OP(OR, logical_or);
#undef CONVERT_BINARY_LOGICAL_OP
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported unary logical operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
