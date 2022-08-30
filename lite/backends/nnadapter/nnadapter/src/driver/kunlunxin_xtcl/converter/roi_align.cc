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

#include "operation/roi_align.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertRoiAlign(Converter* converter, core::Operation* operation) {
  ROI_ALIGN_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  auto rois_expr = converter->GetMappedExpr(rois_operand);
  if (!rois_expr.defined()) {
    rois_expr = converter->ConvertOperand(rois_operand);
  }
  auto pooled_size = ConvertToXTCLArray<xtcl::xIndexExpr>(
      std::vector<int>({output_height, output_width}));
  auto roi_align_expr = converter->builder()->CreateROIAlign(input_expr,
                                                             rois_expr,
                                                             pooled_size,
                                                             spatial_scale,
                                                             sampling_ratio,
                                                             "NCHW",
                                                             "avg");
  converter->UpdateExprMap(output_operand, roi_align_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
