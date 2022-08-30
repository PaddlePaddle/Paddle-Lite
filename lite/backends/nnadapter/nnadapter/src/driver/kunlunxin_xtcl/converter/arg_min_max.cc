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

#include "operation/arg_min_max.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertArgMinMax(Converter* converter, core::Operation* operation) {
  ARG_MIN_MAX_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK_EQ(input_operand->type.dimensions.count, 1)
      << "Expect input dimensions count: 1"
      << ", but receive: " << input_operand->type.dimensions.count;

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  xtcl::xExpr arg_min_max_expr;
  if (operation->type == NNADAPTER_ARG_MAX) {
    arg_min_max_expr = converter->builder()->CreateReduceArgMax(
        input_expr,
        ConvertToXTCLArray<xtcl::Integer>(std::vector<int>({axis})),
        keepdim,
        false);
  } else if (operation->type == NNADAPTER_ARG_MIN) {
    arg_min_max_expr = converter->builder()->CreateReduceArgMin(
        input_expr,
        ConvertToXTCLArray<xtcl::Integer>(std::vector<int>({axis})),
        keepdim,
        false);
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported arg operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  if (dtype == NNADAPTER_INT64) {
    arg_min_max_expr = converter->builder()->CreateCast(
        arg_min_max_expr, ConvertToXTCLDataType(dtype));
  }
  converter->UpdateExprMap(output_operand, arg_min_max_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
