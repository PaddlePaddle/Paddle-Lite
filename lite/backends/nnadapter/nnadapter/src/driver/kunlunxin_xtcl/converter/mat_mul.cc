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

#include "operation/mat_mul.h"
#include "driver/kunlunxin_xtcl/converter/converter.h"
#include "utility/debug.h"
#include "utility/utility.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

int ConvertMatMul(Converter* converter, core::Operation* operation) {
  MAT_MUL_OPERATION_EXTRACT_INPUTS_OUTPUTS
  NNADAPTER_VLOG(5) << "x_operand dimensions count = "
                    << x_operand->type.dimensions.count;
  NNADAPTER_VLOG(5) << "y_operand dimensions count = "
                    << y_operand->type.dimensions.count;

  // Convert to XTCL exprs
  auto x_expr = converter->GetMappedExpr(x_operand);
  if (!x_expr.defined()) {
    x_expr = converter->ConvertOperand(x_operand);
  }
  auto y_expr = converter->GetMappedExpr(y_operand);
  if (!y_expr.defined()) {
    y_expr = converter->ConvertOperand(y_operand);
  }
  xtcl::xExpr matmul_expr;
  auto x_dims = x_operand->type.dimensions;
  auto y_dims = y_operand->type.dimensions;
  auto x_dims_size = x_dims.count;
  auto y_dims_size = y_dims.count;
  if (x_dims_size > 2 && y_dims_size >= 2) {
    // x: [B, ..., M, K], y: [B, ..., K, N], out: [B, ..., M, N]
    // x: [B, M, K], y: [K, N], out: [B, M, N]
    // Reshape to 3 dimension and transposed x expr
    auto x_m = static_cast<int>(x_dims.data[x_dims_size - 2]);
    auto x_k = static_cast<int>(x_dims.data[x_dims_size - 1]);
    auto x_batch_size = static_cast<int>(
        ProductionOfDimensions(x_dims.data, x_dims_size) / (x_m * x_k));
    std::vector<int> x_shape{x_batch_size, x_m, x_k};
    if (x_dims_size != 3) {
      x_expr = converter->builder()->CreateReshape(x_expr, {-1, x_m, x_k});
    }
    if (transpose_x) {
      x_expr = converter->builder()->CreateTranspose(x_expr, {0, 2, 1});
      x_shape[1] = x_k;
      x_shape[2] = x_m;
    }
    // Reshape to 3 dimension and transposed y expr
    auto y_k = static_cast<int>(y_dims.data[y_dims_size - 2]);
    auto y_n = static_cast<int>(y_dims.data[y_dims_size - 1]);
    auto y_batch_size = static_cast<int>(
        ProductionOfDimensions(y_dims.data, y_dims_size) / (y_k * y_n));
    std::vector<int> y_shape{y_batch_size, y_k, y_n};
    if (y_dims_size == 2) {
      y_expr = converter->builder()->CreateExpandDims(y_expr, 0);
    }
    if (y_dims_size > 3) {
      y_expr = converter->builder()->CreateReshape(y_expr, {-1, y_k, y_n});
    }
    if (!transpose_y) {
      y_expr = converter->builder()->CreateTranspose(y_expr, {0, 2, 1});
      y_shape[1] = y_n;
      y_shape[2] = y_k;
    }
    NNADAPTER_CHECK(x_shape[0] == y_shape[0] || x_shape[0] == 1 ||
                    y_shape[0] == 1)
        << "Batch dimensions don't match, but recieved x_shape[0]: "
        << x_shape[0] << ", y_shape[0]: " << y_shape[0];
    NNADAPTER_CHECK_EQ(x_shape[2], y_shape[2])
        << "Expect x_shape[2] == y_shape[2], but recieved x_shape[2]: "
        << x_shape[2] << ", y_shape[2]: " << y_shape[2];
    matmul_expr = converter->builder()->CreateBatchMatmul(x_expr, y_expr);
    auto output_dims = output_operand->type.dimensions;
    if (output_dims.count != 3) {
      auto newshape = ConvertToXTCLArray<xtcl::Integer>(output_dims.data,
                                                        output_dims.count);
      matmul_expr = converter->builder()->CreateReshape(matmul_expr, newshape);
    }
  } else if (x_dims_size == 2 && y_dims_size == 2) {
    // x: [M, K], y: [K, N], out: [M, N]
    if (transpose_x) {
      x_expr = converter->builder()->CreateTranspose(x_expr, {1, 0});
    }
    if (!transpose_y) {
      y_expr = converter->builder()->CreateTranspose(y_expr, {1, 0});
    }
    // Add batch dimension
    x_expr = converter->builder()->CreateExpandDims(x_expr, 0);
    y_expr = converter->builder()->CreateExpandDims(y_expr, 0);
    matmul_expr = converter->builder()->CreateBatchMatmul(x_expr, y_expr);
    // Remove batch dimension
    matmul_expr = converter->builder()->CreateSqueeze(matmul_expr, {0});
  } else {
    // x: [K], y: [K], out: [1]
    // x: [M], y: [N], x_transpose: true, y_transpose: true, out: [M, N]
    NNADAPTER_LOG(FATAL) << "Unsupported dims"
                         << ", x_dims_size: " << x_dims_size
                         << ", y_dims_size: " << y_dims_size;
    return NNADAPTER_FEATURE_NOT_SUPPORTED;
  }
  converter->UpdateExprMap(output_operand, matmul_expr);
  return NNADAPTER_NO_ERROR;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
