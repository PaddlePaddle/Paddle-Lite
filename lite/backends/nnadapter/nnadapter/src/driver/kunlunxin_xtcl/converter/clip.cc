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

#include "operation/clip.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertClip(Converter* converter, core::Operation* operation) {
  CLIP_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  // Min
  uint32_t min_size =
      min_operand->length / static_cast<uint32_t>(sizeof(float));
  // Must set min
  NNADAPTER_CHECK_EQ(min_size, 1U) << "Expect min_size: 1, but receive: "
                                   << min_size;
  auto min_buffer = *reinterpret_cast<float*>(min_operand->buffer);
  // Max
  uint32_t max_size =
      max_operand->length / static_cast<uint32_t>(sizeof(float));
  // Must set min
  NNADAPTER_CHECK_EQ(max_size, 1U) << "Expect max_size: 1, but receive: "
                                   << max_size;
  auto max_buffer = *reinterpret_cast<float*>(max_operand->buffer);
  auto clip_expr =
      converter->builder()->CreateClip(input_expr, min_buffer, max_buffer);
  converter->UpdateExprMap(output_operand, clip_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
