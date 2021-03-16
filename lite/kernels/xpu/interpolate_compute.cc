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

#include "lite/kernels/xpu/interpolate_compute.h"
#include <iostream>
#include <memory>
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void BilinearInterpCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  lite::Tensor* X = param.X;
  int n = X->dims()[0];
  int c = X->dims()[1];
  int in_h = X->dims()[2];
  int in_w = X->dims()[3];

  lite::Tensor* Out = param.Out;
  int out_h = Out->dims()[2];
  int out_w = Out->dims()[3];
  int align_mode = param.align_mode;
  bool align_corners = param.align_corners;

  int trans_mode = -1;
  if (align_corners == true) {
    trans_mode = 0;
  } else if ((align_corners == false) && (align_mode == 0)) {
    trans_mode = 1;
  } else {
    trans_mode = 2;
  }
  int r = xdnn::interpolate2d<float>(ctx.GetRawContext(),
                                     X->data<float>(),
                                     Out->mutable_data<float>(TARGET(kXPU)),
                                     n,
                                     c,
                                     in_h,
                                     in_w,
                                     out_h,
                                     out_w,
                                     false,
                                     trans_mode,
                                     true);
  CHECK_EQ(r, 0);
}

void NearestInterpCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  lite::Tensor* X = param.X;
  int n = X->dims()[0];
  int c = X->dims()[1];
  int in_h = X->dims()[2];
  int in_w = X->dims()[3];

  lite::Tensor* Out = param.Out;
  int out_h = Out->dims()[2];
  int out_w = Out->dims()[3];
  bool align_corners = param.align_corners;
  int trans_mode = (align_corners == true) ? 0 : 1;

  int r = xdnn::interpolate2d<float>(ctx.GetRawContext(),
                                     X->data<float>(),
                                     Out->mutable_data<float>(TARGET(kXPU)),
                                     n,
                                     c,
                                     in_h,
                                     in_w,
                                     out_h,
                                     out_w,
                                     true,
                                     trans_mode,
                                     true);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(bilinear_interp,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::BilinearInterpCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(bilinear_interp_v2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::BilinearInterpCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(nearest_interp,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::NearestInterpCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();

REGISTER_LITE_KERNEL(nearest_interp_v2,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::NearestInterpCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
