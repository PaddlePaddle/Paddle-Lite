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
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

namespace math = paddle::lite::xpu::math;

void MatMulCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* x = param.X;
  auto* y = param.Y;
  auto* out = param.Out;

  if (param.enable_int8) {
    LOG(FATAL) << "xpu don't support matmul int8 outside encoder";
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
    r = xdnn::fc_fusion<float, float, float, int16_t>(
        ctx.GetRawContext(),                     // ctx
        x->data<float>(),                        // x
        y->data<float>(),                        // w
        out->mutable_data<float>(TARGET(kXPU)),  // y
        mat_dim_a.height_,                       // m
        mat_dim_b.width_,                        // n
        mat_dim_a.width_,                        // k
        mat_dim_a.trans_,                        // x_trans
        mat_dim_b.trans_,                        // w_trans
        nullptr,                                 // x_maxptr
        nullptr,                                 // w_maxptr
        nullptr,                                 // y_maxptr
        lda,                                     // ldx
        ldb,                                     // ldw
        ldc,                                     // ldy
        param.alpha,                             // alpha
        0.0f,                                    // beta
        nullptr,                                 // bias
        xdnn::Activation_t::LINEAR);             // act
  } else {
    // batch matmul
    r = xdnn::fc_batched<float, float, float, int16_t>(
        ctx.GetRawContext(),                    /* context */
        mat_dim_a.batch_size_,                  /* batch_size */
        mat_dim_a.trans_,                       /* TransA */
        mat_dim_b.trans_,                       /* TransB */
        mat_dim_a.height_,                      /* M */
        mat_dim_b.width_,                       /* N */
        mat_dim_a.width_,                       /* K */
        param.alpha,                            /* alpha */
        x->data<float>(),                       /* A */
        mat_dim_a.stride_,                      /* stride_a */
        y->data<float>(),                       /* B */
        mat_dim_b.stride_,                      /* stride_b */
        0.0f,                                   /* beta */
        out->mutable_data<float>(TARGET(kXPU)), /* C */
        mat_dim_a.height_ * mat_dim_b.width_,   /* stride_c */
        nullptr,                                /* x_maxptr */
        nullptr /* w_maxptr */);
  }
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    matmul, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::MatMulCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
REGISTER_LITE_KERNEL(matmul_v2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::MatMulCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
