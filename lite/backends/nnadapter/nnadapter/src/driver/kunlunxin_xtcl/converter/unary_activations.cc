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

#include "operation/unary_activations.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertUnaryActivations(Converter* converter, core::Operation* operation) {
  UNARY_ACTIVATIONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  switch (operation->type) {
#define CONVERT_UNARY_ACTIVATION(type, func)                              \
  case NNADAPTER_##type:                                                  \
    converter->UpdateExprMap(output_operand, converter->builder()->func); \
    break;
    CONVERT_UNARY_ACTIVATION(RELU, CreateRelu(input_expr));
    CONVERT_UNARY_ACTIVATION(RELU6, CreateRelu6(input_expr));
    CONVERT_UNARY_ACTIVATION(SIGMOID, CreateUnaryOp("sigmoid", input_expr));
    CONVERT_UNARY_ACTIVATION(TANH, CreateUnaryOp("tanh", input_expr));
    CONVERT_UNARY_ACTIVATION(LOG, CreateUnaryOp("log", input_expr));
    CONVERT_UNARY_ACTIVATION(ABS, CreateUnaryOp("abs", input_expr));
    CONVERT_UNARY_ACTIVATION(EXP, CreateUnaryOp("exp", input_expr));
    CONVERT_UNARY_ACTIVATION(FLOOR, CreateUnaryOp("floor", input_expr));
    CONVERT_UNARY_ACTIVATION(
        SQUARE, CreateBinaryOp("multiply", input_expr, input_expr));
#undef CONVERT_UNARY_ACTIVATION
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported activation operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
