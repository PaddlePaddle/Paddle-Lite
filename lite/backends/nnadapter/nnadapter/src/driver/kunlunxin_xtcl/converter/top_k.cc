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

#include "operation/top_k.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertTopK(Converter* converter, core::Operation* operation) {
  TOP_K_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_NE(k, NNADAPTER_UNKNOWN) << "";

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  bool is_ascend = !largest;
  auto dtype = ConvertToXTCLDataType(
      static_cast<NNAdapterOperandPrecisionCode>(return_indices_dtype));
  auto top_k_expr = converter->builder()->CreateTopK(
      input_expr, k, axis, "both", is_ascend, dtype);
  converter->UpdateExprMap(output_operand,
                           converter->builder()->GetField(top_k_expr, 0));
  converter->UpdateExprMap(indices_operand,
                           converter->builder()->GetField(top_k_expr, 1));
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
