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

#include "operation/resize_nearest.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertResizeNearest(Converter* converter, core::Operation* operation) {
  RESIZE_NEAREST_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Input expr
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }

  auto shape_count = shape_operand->length / sizeof(int32_t);
  NNADAPTER_CHECK_EQ(shape_count, 2);
  auto shape_data = reinterpret_cast<int32_t*>(shape_operand->buffer);
  std::vector<int> output_shape_size(shape_data, shape_data + shape_count);
  auto size = ConvertToXTCLArray<xtcl::xIndexExpr>(output_shape_size);
  auto resize_nearest_expr = converter->builder()->CreateInterpolate(
      input_expr, size, "NCHW", "nearest_neighbor", align_corners);
  converter->UpdateExprMap(output_operand, resize_nearest_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
