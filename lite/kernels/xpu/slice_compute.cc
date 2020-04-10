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

#include "lite/kernels/xpu/slice_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SliceCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto x_dims = param.X->dims();
  x_shape_.reserve(x_dims.size());
  x_dim_begin_.reserve(x_dims.size());
  x_dim_end_.reserve(x_dims.size());
}

void SliceCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  auto x_dims = param.X->dims();
  for (size_t i = 0; i < x_dims.size(); ++i) {
    x_shape_[i] = x_dims[i];
    x_dim_begin_[i] = 0;
    x_dim_end_[i] = x_dims[i];
  }
  for (size_t i = 0; i < param.axes.size(); ++i) {
    int axis = param.axes[i];
    x_dim_begin_[axis] = param.starts[i];
    x_dim_end_[axis] = param.ends[i];
  }

  int ndim = param.X->dims().size();
  int r = xdnn::slice_forward(
      ctx.GetRawContext(),    /* context */
      &x_shape_[0],           /* shape */
      &x_dim_begin_[0],       /* starts */
      &x_dim_end_[0],         /* ends */
      ndim,                   /* n */
      param.X->data<float>(), /* in */
      param.Out->mutable_data<float>(TARGET(kXPU)) /* out */);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    slice, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::SliceCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
