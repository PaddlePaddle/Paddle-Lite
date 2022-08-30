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

#include "operation/split.h"
#include <numeric>
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertSplit(Converter* converter, core::Operation* operation) {
  SPLIT_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_CHECK(IsConstantOperand(axis_operand));
  NNADAPTER_CHECK(IsConstantOperand(split_operand));
  int split_section_sum = std::accumulate(split.begin(), split.end(), 0);
  NNADAPTER_CHECK_EQ(split_section_sum,
                     input_operand->type.dimensions.data[axis])
      << "The sum of sections must match the input.shape[axis]";

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  // Convert split_sections to split_indices
  int32_t index_sum = 0;
  std::vector<int32_t> split_indices;
  for (size_t index = 0; index < split.size() - 1; ++index) {
    index_sum += split[index];
    split_indices.push_back(index_sum);
  }
  auto split_expr = converter->builder()->CreateSplit(
      input_expr, ConvertToXTCLArray<xtcl::Integer>(split_indices), axis);
  for (size_t i = 0; i < split.size(); i++) {
    converter->UpdateExprMap(output_operands[i],
                             converter->builder()->GetField(split_expr, i));
  }
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
