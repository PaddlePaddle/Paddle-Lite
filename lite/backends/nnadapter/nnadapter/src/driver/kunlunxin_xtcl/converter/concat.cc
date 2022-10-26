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

#include "operation/concat.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertConcat(Converter* converter, core::Operation* operation) {
  CONCAT_OPERATION_EXTRACT_INPUTS_OUTPUTS
  auto dims_count_0 = input_operands[0]->type.dimensions.count;
  for (int i = 1; i < input_count - 1; i++) {
    auto dims_count = input_operands[i]->type.dimensions.count;
    NNADAPTER_CHECK_EQ(dims_count_0, dims_count)
        << "Expect all inputs have the same dimensions count, but receive: "
        << dims_count_0 << ", " << dims_count;
  }

  // Convert to XTCL exprs
  auto N = input_count - 1;
  xtcl::Array<xtcl::xExpr> input_exprs;
  for (int i = 0; i < N; i++) {
    auto input_operand = input_operands[i];
    auto input_expr = converter->GetMappedExpr(input_operand);
    if (!input_expr.defined()) {
      input_expr = converter->ConvertOperand(input_operand);
    }
    input_exprs.push_back(input_expr);
  }
  auto concat_expr = converter->builder()->CreateConcatenate(
      xtcl::network::Tuple(input_exprs), axis);
  converter->UpdateExprMap(output_operand, concat_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
