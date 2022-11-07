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

#include "operation/slice.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertSlice(Converter* converter, core::Operation* operation) {
  SLICE_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  auto input_dim = input_operand->type.dimensions;
  auto input_dim_count = input_dim.count;
  auto input_dim_data = input_dim.data;
  xtcl::Array<xtcl::Integer> begin, end, strides;
  std::vector<int32_t> axes_vec(axes, axes + axes_count);
  for (size_t i = 0; i < input_dim_count; ++i) {
    auto it = std::find(axes_vec.cbegin(), axes_vec.cend(), i);
    if (it == axes_vec.cend()) {
      // Don't slice this axis
      begin.push_back(0);
      end.push_back(input_dim_data[i]);
      strides.push_back(1);
    } else {
      int dis = it - axes_vec.cbegin();
      int s = starts[dis];
      int e = ends[dis];
      begin.push_back(s);
      end.push_back(e);
      strides.push_back(1);
    }
  }
  // slice_mode: "end" or "size"
  auto slice_expr = converter->builder()->CreateStridedSlice(
      input_expr, begin, end, strides, "end");
  converter->UpdateExprMap(output_operand, slice_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
