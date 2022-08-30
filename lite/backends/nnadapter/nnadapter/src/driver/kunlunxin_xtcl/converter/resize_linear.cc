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

#include "operation/resize_linear.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertResizeLinear(Converter* converter, core::Operation* operation) {
  RESIZE_LINEAR_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  // shape_operand may be invalid, get output height and width from
  // output_operand
  NNADAPTER_CHECK_EQ(output_operand->type.dimensions.count, 4)
      << "Expect output_operand dimensions count: 4"
      << ", but receive: " << output_operand->type.dimensions.count;
  std::vector<int> output_h_w(output_operand->type.dimensions.data + 2,
                              output_operand->type.dimensions.data + 4);
  for (const auto& item : output_h_w) {
    NNADAPTER_VLOG(5) << "output_h_w : " << item;
  }
  auto size = ConvertToXTCLArray<xtcl::xIndexExpr>(output_h_w);
  auto resize_nearest_expr = converter->builder()->CreateInterpolate(
      input_expr, size, "NCHW", "bilinear", align_corners);
  converter->UpdateExprMap(output_operand, resize_nearest_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
