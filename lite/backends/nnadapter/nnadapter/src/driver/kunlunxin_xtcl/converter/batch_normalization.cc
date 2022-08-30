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

#include "operation/batch_normalization.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertBatchNormalization(Converter* converter,
                              core::Operation* operation) {
  BATCH_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  auto scale_expr = converter->GetMappedExpr(scale_operand);
  if (!scale_expr.defined()) {
    scale_expr = converter->ConvertOperand(scale_operand);
  }
  auto bias_expr = converter->GetMappedExpr(bias_operand);
  if (!bias_expr.defined()) {
    bias_expr = converter->ConvertOperand(bias_operand);
  }
  auto mean_expr = converter->GetMappedExpr(mean_operand);
  if (!mean_expr.defined()) {
    mean_expr = converter->ConvertOperand(mean_operand);
  }
  auto variance_expr = converter->GetMappedExpr(variance_operand);
  if (!variance_expr.defined()) {
    variance_expr = converter->ConvertOperand(variance_operand);
  }
  auto batch_norm_expr = converter->builder()->CreateBatchNorm(
      input_expr, scale_expr, bias_expr, mean_expr, variance_expr, 1, epsilon);
  converter->UpdateExprMap(output_operand,
                           converter->builder()->GetField(batch_norm_expr, 0));
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
