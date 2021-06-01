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

#include "lite/kernels/xpu/fill_zeros_like_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <class T>
void FillZerosLikeCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  T zero(0);
  int ret = xdnn::constant<T>(ctx.GetRawContext(),
                              param.Out->template mutable_data<T>(TARGET(kXPU)),
                              param.Out->numel(),
                              zero);

  CHECK_EQ(ret, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fill_zeros_like,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::FillZerosLikeCompute<float>,
                     float32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(fill_zeros_like,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::FillZerosLikeCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(fill_zeros_like,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::FillZerosLikeCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .Finalize();
