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

#include "operation/pool2d.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertPool2D(Converter* converter, core::Operation* operation) {
  POOL_2D_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to XTCL exprs
  auto input_expr = converter->GetMappedExpr(input_operand);
  if (!input_expr.defined()) {
    input_expr = converter->ConvertOperand(input_operand);
  }
  auto ksize = ConvertToXTCLArray<xtcl::xIndexExpr>(
      std::vector<int>({kernel_height, kernel_width}));
  auto strides = ConvertToXTCLArray<xtcl::xIndexExpr>(
      std::vector<int>({stride_height, stride_width}));
  auto paddings = ConvertToXTCLArray<xtcl::xIndexExpr>(std::vector<int>(
      {pad_height_top, pad_width_left, pad_height_bottom, pad_width_right}));
  auto output_height = output_operand->type.dimensions.data[2];
  auto output_width = output_operand->type.dimensions.data[3];
  auto output_size = ConvertToXTCLArray<xtcl::xIndexExpr>(
      std::vector<int>({output_height, output_width}));
  xtcl::xExpr pool2d_expr;
  if (operation->type == NNADAPTER_AVERAGE_POOL_2D) {
    if (global_pooling) {
      pool2d_expr =
          converter->builder()->CreateGlobalAvgPool2D(input_expr, "NCHW");
    } else {
      pool2d_expr = converter->builder()->CreateAvgPool2D(
          input_expr, ksize, strides, paddings, "NCHW", "", ceil_mode, flag);
    }
  } else if (operation->type == NNADAPTER_MAX_POOL_2D) {
    if (global_pooling) {
      pool2d_expr =
          converter->builder()->CreateGlobalMaxPool2D(input_expr, "NCHW");
    } else {
      pool2d_expr = converter->builder()->CreateMaxPool2D(
          input_expr, ksize, strides, paddings, "NCHW", "", ceil_mode);
    }
  } else if (operation->type == NNADAPTER_ADAPTIVE_AVERAGE_POOL_2D) {
    pool2d_expr = converter->builder()->CreateAdaptiveAvgPool2D(
        input_expr, output_size, "NCHW");
  } else if (operation->type == NNADAPTER_ADAPTIVE_MAX_POOL_2D) {
    pool2d_expr = converter->builder()->CreateAdaptiveMaxPool2D(
        input_expr, output_size, "NCHW");
  } else {
    NNADAPTER_LOG(FATAL) << "Unsupported pooling operation type "
                         << OperationTypeToString(operation->type)
                         << " is found.";
  }
  converter->UpdateExprMap(output_operand, pool2d_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
