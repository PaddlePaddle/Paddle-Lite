// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/matmul_compute.h"
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

namespace math = paddle::lite::xpu::math;

template <typename TGEMM,
          typename TW,
          typename DX,
          typename DY,
          PrecisionType PType>
void MatMulCompute<TGEMM, TW, DX, DY, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* x = param.X;
  auto* y = param.Y;
  auto* out = param.Out;

  // max
  int max_ptr_size = ctx.GetRawContext()->max_ptr_size();
  XPUScratchPadGuard input_max_guard_ =
      TargetWrapperXPU::MallocScratchPad(max_ptr_size * sizeof(float));
  if (param.enable_int8) {  // for paddle slim int8 quant
    std::vector<float> cpu_input_max(max_ptr_size, 127 * param.input_scale);
    lite::TargetWrapperXPU::MemcpySync(input_max_guard_->addr_,
                                       cpu_input_max.data(),
                                       sizeof(float) * max_ptr_size,
                                       IoDirection::HtoD);
  }

  const float* x_maxptr = nullptr;
  const float* w_maxptr = nullptr;
  if (param.enable_int8 && x == y) {
    x_maxptr = reinterpret_cast<float*>(input_max_guard_->addr_);
    w_maxptr = reinterpret_cast<float*>(input_max_guard_->addr_);
  }

  auto& x_dims = x->dims();
  auto& y_dims = y->dims();
  auto mat_dim_a = math::CreateMatrixDescriptor(
      math::RowMatrixFromVector(x_dims), 0, param.transpose_X);
  auto mat_dim_b = math::CreateMatrixDescriptor(
      math::ColumnMatrixFromVector(y_dims), 0, param.transpose_Y);

  if (x_dims.size() >= 3 && y_dims.size() <= 2) {
    if (!param.transpose_X) {
      mat_dim_a.height_ *= mat_dim_a.batch_size_;
      mat_dim_a.batch_size_ = 0;
    } else {
      mat_dim_b.batch_size_ = mat_dim_a.batch_size_;
      mat_dim_b.height_ = mat_dim_b.height_ / mat_dim_b.batch_size_;
    }
  } else if (x_dims.size() <= 2 && y_dims.size() >= 3) {
    if (!param.transpose_Y) {
      mat_dim_b.height_ *= mat_dim_b.batch_size_;
      mat_dim_b.batch_size_ = 0;
    } else {
      mat_dim_a.batch_size_ = mat_dim_b.batch_size_;
      mat_dim_a.height_ = mat_dim_a.height_ / mat_dim_a.batch_size_;
    }
  }

  CHECK_EQ(mat_dim_a.width_, mat_dim_b.height_);
  CHECK_EQ(mat_dim_a.batch_size_, mat_dim_b.batch_size_);

  int lda = (mat_dim_a.trans_ ? mat_dim_a.height_ : mat_dim_a.width_);
  int ldb = (mat_dim_b.trans_ ? mat_dim_b.height_ : mat_dim_b.width_);
  int ldc = mat_dim_b.width_;

  int r = 0;
  if (mat_dim_a.batch_size_ == 0 || mat_dim_a.batch_size_ == 1) {
    r = xdnn::fc_fusion<DX, TW, DY, TGEMM>(
        ctx.GetRawContext(),                           // ctx
        x->template data<DX>(),                        // x
        y->template data<TW>(),                        // w
        out->template mutable_data<DY>(TARGET(kXPU)),  // y
        mat_dim_a.height_,                             // m
        mat_dim_b.width_,                              // n
        mat_dim_a.width_,                              // k
        mat_dim_a.trans_,                              // x_trans
        mat_dim_b.trans_,                              // w_trans
        x_maxptr,                                      // x_maxptr
        w_maxptr,                                      // w_maxptr
        nullptr,                                       // y_maxptr
        lda,                                           // ldx
        ldb,                                           // ldw
        ldc,                                           // ldy
        param.alpha,                                   // alpha
        0.0f,                                          // beta
        nullptr,                                       // bias
        xdnn::Activation_t::LINEAR);                   // act
  } else {
    // batch matmul
    r = xdnn::fc_batched<DX, TW, DY, TGEMM>(
        ctx.GetRawContext(),                          /* context */
        mat_dim_a.batch_size_,                        /* batch_size */
        mat_dim_a.trans_,                             /* TransA */
        mat_dim_b.trans_,                             /* TransB */
        mat_dim_a.height_,                            /* M */
        mat_dim_b.width_,                             /* N */
        mat_dim_a.width_,                             /* K */
        param.alpha,                                  /* alpha */
        x->template data<DX>(),                       /* A */
        mat_dim_a.stride_,                            /* stride_a */
        y->template data<TW>(),                       /* B */
        mat_dim_b.stride_,                            /* stride_b */
        0.0f,                                         /* beta */
        out->template mutable_data<DY>(TARGET(kXPU)), /* C */
        mat_dim_a.height_ * mat_dim_b.width_,         /* stride_c */
        x_maxptr,                                     /* x_maxptr */
        w_maxptr);                                    /* w_maxptr */
  }
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

// tgemm w x y
using XPUMATMUL_FP32 =
    xpu::MatMulCompute<int16_t, float, float, float, PRECISION(kFloat)>;

using XPUMATMUL_Int8_FP32_FP32 =
    xpu::MatMulCompute<int8_t, float, float, float, PRECISION(kInt8)>;

REGISTER_LITE_KERNEL(matmul, kXPU, kFloat, kNCHW, XPUMATMUL_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(matmul, kXPU, kInt8, kNCHW, XPUMATMUL_Int8_FP32_FP32, int8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(matmul_v2, kXPU, kFloat, kNCHW, XPUMATMUL_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
