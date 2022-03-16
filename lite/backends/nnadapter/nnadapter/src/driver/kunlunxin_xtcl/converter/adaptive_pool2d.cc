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

#include "operation/adaptive_pool2d.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/utility.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertAdaptivePool2D(Converter* converter, core::Operation* operation) {
  ADAPTIVE_POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  xtcl::xExpr adaptive_pool2d_expr;
  auto output_size = ConvertToXTCLArray<xtcl::xIndexExpr>(
      std::vector<int>({output_height, output_width}));
  if (operation->type == NNADAPTER_ADAPTIVE_MAX_POOL_2D) {
    adaptive_pool2d_expr = converter->builder()->CreateAdaptiveMaxPool2D(
        input_expr, output_size, "NCHW");
  } else if (operation->type == NNADAPTER_ADAPTIVE_AVERAGE_POOL_2D) {
    adaptive_pool2d_expr = converter->builder()->CreateAdaptiveAvgPool2D(
        input_expr, output_size, "NCHW");
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported adaptive pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  converter->UpdateExprMap(output_operand, adaptive_pool2d_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
