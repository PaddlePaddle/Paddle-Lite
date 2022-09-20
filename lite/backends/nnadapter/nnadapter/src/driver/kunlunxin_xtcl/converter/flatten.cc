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

#include "operation/flatten.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertFlatten(Converter* converter, core::Operation* operation) {
  FLATTEN_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  auto input_rank = input_operand->type.dimensions.count;
  if (start_axis < 0) {
    start_axis += input_rank;
  }
  if (end_axis < 0) {
    end_axis += input_rank;
  }
  NNADAPTER_CHECK_EQ(end_axis, input_rank - 1)
      << "XTCL only support end_axis = -1 or rank - 1";
  auto flatten_expr =
      converter->builder()->CreateBatchFlatten(input_expr, start_axis);
  converter->UpdateExprMap(output_operand, flatten_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
