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

#include "operation/reduce.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertReduce(Converter* converter, core::Operation* operation) {
  REDUCE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  auto axes_xtcl_array =
      ConvertToXTCLArray<xtcl::Integer>(axes_data, axes_size);
  xtcl::xExpr reduce_expr;
  switch (operation->type) {
#define CONVERT_REDUCE(type, func)                         \
  case NNADAPTER_##type: {                                 \
    reduce_expr = converter->builder()->func;              \
    converter->UpdateExprMap(output_operand, reduce_expr); \
  } break;
    CONVERT_REDUCE(REDUCE_MEAN,
                   CreateReduceMean(input_expr, axes_xtcl_array, keep_dim));
    CONVERT_REDUCE(REDUCE_SUM,
                   CreateReduceSum(input_expr, axes_xtcl_array, keep_dim));
#undef CONVERT_REDUCE
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported reduce operation type "
                           << OperationTypeToString(operation->type)
                           << " is found.";
      break;
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
