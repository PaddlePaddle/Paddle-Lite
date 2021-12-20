// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "core/operation/unary_activations.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertUnaryActivations(Converter* converter, hal::Operation* operation) {
  UNARY_ACTIVATIONS_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  xtcl::xExpr act_expr = nullptr;
  switch (operation->type) {
    case NNADAPTER_RELU:
      act_expr = converter->builder()->CreateRelu(input_expr);
      break;
    case NNADAPTER_TANH:
      act_expr = converter->builder()->CreateUnaryOp("tanh", input_expr);
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported activation unary operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  UpdateExprMap(output_operand, act_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
