// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "lite/backends/cuda/cuda_utils.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/matmul_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T, PrecisionType PType>
void MatMulCompute<T, PType>::Run() {
  auto& context = this->ctx_->template As<CUDAContext>();
  auto& param = this->template Param<param_t>();

  const auto* x_data = param.X->template data<T>();
  const auto* y_data = param.Y->template data<T>();
  auto* out_data = param.Out->template mutable_data<T>(TARGET(kCUDA));
  bool transpose_x = param.transpose_X;
  bool transpose_y = param.transpose_Y;
  float alpha = param.alpha;

  auto x_dims = param.X->dims();
  auto y_dims = param.Y->dims();

  int m = 0;
  int k = 0;
  int n = 0;
  int batch = 0;
  int64_t stride_x = 0;
  int64_t stride_y = 0;

  if (x_dims.size() >= 2 && y_dims.size() >= 2 &&
      (x_dims.size() != 2 || y_dims.size() != 2)) {
    // x: [B, ..., M, K], y: [B, ..., K, N], out: [B, ..., M, N]
    // x: [B, M, K], y: [K, N], out: [B, M, N]
    // or
    // x: [M, K], y: [B, ..., K, N], out: [B, ..., M, N]
    // x: [M, K], y: [B, K, N], out: [B, M, N]
    strided_gemm_impl_->init(transpose_x, transpose_y, &context);
    m = transpose_x ? x_dims[x_dims.size() - 1] : x_dims[x_dims.size() - 2];
    k = transpose_x ? x_dims[x_dims.size() - 2] : x_dims[x_dims.size() - 1];
    n = transpose_y ? y_dims[y_dims.size() - 2] : y_dims[y_dims.size() - 1];
    int batch_x = x_dims.size() == 2 ? 0 : x_dims.count(0, x_dims.size() - 2);
    int batch_y = y_dims.size() == 2 ? 0 : y_dims.count(0, y_dims.size() - 2);
    CHECK(batch_x == batch_y || batch_x == 0 || batch_y == 0)
        << "batch_size x should be equal to batch_size y, or "
           "one of batch_size x and batch_size y should be 0. "
           "But got batch_size x = "
        << batch_x << ", batch_size y = " << batch_y;
    batch = batch_x == 0 ? batch_y : batch_x;
    stride_x = x_dims.size() == 2 ? 0 : m * k;
    stride_y = y_dims.size() == 2 ? 0 : k * n;
    strided_gemm_impl_->run(alpha,
                            0.f,
                            m,
                            n,
                            k,
                            x_data,
                            y_data,
                            out_data,
                            batch,
                            stride_x,
                            stride_y);
  } else if (x_dims.size() == 2 && y_dims.size() == 2) {
    // x: [M, K], y: [K, N], out: [M, N]
    m = transpose_x ? x_dims[1] : x_dims[0];
    k = transpose_x ? x_dims[0] : x_dims[1];
    n = transpose_y ? y_dims[0] : y_dims[1];
    gemm_impl_->init(transpose_x, transpose_y, m, n, k, &context);
    gemm_impl_->run(alpha, 0.0f, x_data, y_data, out_data, &context);
  } else if (x_dims.size() > 2 && y_dims.size() == 1) {
    // x: [B, M, K], y: [K], out: [B, M]
    strided_gemm_impl_->init(transpose_x, transpose_y, &context);
    m = transpose_x ? x_dims[x_dims.size() - 1] : x_dims[x_dims.size() - 2];
    k = transpose_x ? x_dims[x_dims.size() - 2] : x_dims[x_dims.size() - 1];
    n = 1;
    batch = x_dims.count(0, x_dims.size() - 2);
    stride_x = m * k;
    stride_y = 0;
    strided_gemm_impl_->run(alpha,
                            0.f,
                            m,
                            n,
                            k,
                            x_data,
                            y_data,
                            out_data,
                            batch,
                            stride_x,
                            stride_y);
  } else if (x_dims.size() == 1 && y_dims.size() == 1) {
    if (!transpose_x && !transpose_y) {
      // x: [K], y: [K], out: [1]
      m = 1;
      k = x_dims[0];
      n = 1;
      CHECK_EQ(x_dims[0], y_dims[0])
          << "x_dims[0] should be equal to y_dims[0]";
      gemm_impl_->init(false, false, m, n, k, &context);
      gemm_impl_->run(alpha, 0.0f, x_data, y_data, out_data, &context);
    } else if (transpose_x && transpose_y) {
      // x: [M], y: [N], x_transpose: true, y_transpose: true, out: [M, N]
      m = x_dims[0];
      k = 1;
      n = y_dims[0];
      gemm_impl_->init(false, false, m, n, k, &context);
      gemm_impl_->run(alpha, 0.0f, x_data, y_data, out_data, &context);
    } else {
      LOG(FATAL) << "not supported x_dims(" << x_dims << ") and y_dims("
                 << y_dims << "), transpose_x(" << transpose_x
                 << "), transpose_y(" << transpose_y << ")";
    }
  } else {
    LOG(FATAL) << "not supported x_dims(" << x_dims << ") and y_dims(" << y_dims
               << ")";
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using MatMulFp32 =
    paddle::lite::kernels::cuda::MatMulCompute<float, PRECISION(kFloat)>;

using MatMulFp16 =
    paddle::lite::kernels::cuda::MatMulCompute<half, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(matmul, kCUDA, kFloat, kNCHW, MatMulFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(matmul, kCUDA, kFP16, kNCHW, MatMulFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();
