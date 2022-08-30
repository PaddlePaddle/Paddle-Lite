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

#include "operation/instance_normalization.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertInstanceNormalization(Converter* converter,
                                 core::Operation* operation) {
  INSTANCE_NORMALIZATION_OPERATION_EXTRACT_INPUTS_OUTPUTS

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
  auto instance_norm_expr = converter->builder()->CreateInstanceNorm(
      input_expr, scale_expr, bias_expr, 1, epsilon, true, true);
  converter->UpdateExprMap(output_operand, instance_norm_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
