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
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void BilinearInterpCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  auto x_dims = param.X->dims();
  CHECK_EQ(x_dims.size(), 4);
  int n = x_dims[0];
  int c = x_dims[1];
  int in_h = x_dims[2];
  int in_w = x_dims[3];

  int out_w = param.out_w;
  int out_h = param.out_h;
  float scale = param.scale;
  if (scale > 0) {
    out_h = static_cast<int>(in_h * scale);
    out_w = static_cast<int>(in_w * scale);
  }
  if (param.OutSize != nullptr) {
    out_h = param.OutSize->data<int>()[0];
    out_w = param.OutSize->data<int>()[1];
  }
  bool align_corners = param.align_corners;
  CHECK_EQ(align_corners, 1) << "XPU only support align corners = 1";

  int r = xdnn::bilinear_interp(ctx.GetRawContext(), /* context */
                                param.X->data<float>(),
                                param.Out->mutable_data<float>(TARGET(kXPU)),
                                n,
                                c,
                                in_h,
                                in_w,
                                out_h,
                                out_w,
                                align_corners,
                                1);
  CHECK_EQ(r, 0);
}

void NearestInterpCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  auto x_dims = param.X->dims();
  CHECK_EQ(x_dims.size(), 4);
  int n = x_dims[0];
  int c = x_dims[1];
  int in_h = x_dims[2];
  int in_w = x_dims[3];

  int out_w = param.out_w;
  int out_h = param.out_h;
  float scale = param.scale;
  if (scale > 0) {
    out_h = static_cast<int>(in_h * scale);
    out_w = static_cast<int>(in_w * scale);
  }

  if (param.OutSize != nullptr) {
    out_h = param.OutSize->data<int>()[0];
    out_w = param.OutSize->data<int>()[1];
  }
  bool align_corners = param.align_corners;

  int r = xdnn::interpolate(ctx.GetRawContext(), /* context */
                            param.X->data<float>(),
                            param.Out->mutable_data<float>(TARGET(kXPU)),
                            n,
                            c,
                            in_h,
                            in_w,
                            out_h,
                            out_w,
                            align_corners);
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
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kXPU))})
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
               {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
