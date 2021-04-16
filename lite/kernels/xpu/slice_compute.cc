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

template <class T>
void SliceCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto x_dims = param.X->dims();
  auto x_shape = x_dims.Vectorize();
  std::vector<int> x_shape_(x_shape.begin(), x_shape.end());
  std::vector<int> x_dim_begin_(x_dims.size(), 0);
  std::vector<int> x_dim_end_(x_shape_);

  for (size_t i = 0; i < param.axes.size(); ++i) {
    int axis = param.axes[i];
    x_dim_begin_[axis] = param.starts[i] < 0
                             ? param.starts[i] + static_cast<int>(x_dims[axis])
                             : param.starts[i];
    int end = param.ends[i] < 0 ? param.ends[i] + static_cast<int>(x_dims[axis])
                                : param.ends[i];
    x_dim_end_[axis] = (std::min)(end, static_cast<int>(x_dims[axis]));
  }

  int r =
      xdnn::slice(ctx.GetRawContext(),         /* context */
                  param.X->template data<T>(), /* in */
                  param.Out->template mutable_data<T>(TARGET(kXPU)), /* out */
                  x_shape_,
                  x_dim_begin_,
                  x_dim_end_);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using SliceFloat32 = paddle::lite::kernels::xpu::SliceCompute<float>;
REGISTER_LITE_KERNEL(slice, kXPU, kFloat, kAny, SliceFloat32, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

using SliceInt32 = paddle::lite::kernels::xpu::SliceCompute<int32_t>;
REGISTER_LITE_KERNEL(slice, kXPU, kFloat, kAny, SliceInt32, int32)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();
