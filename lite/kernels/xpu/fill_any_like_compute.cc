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

#include "lite/kernels/xpu/fill_any_like_compute.h"
#include <iostream>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void FillAnyLikeCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  int write_size = param.in->numel();

  int r = xdnn::constant<float>(ctx.GetRawContext(), /* context */
                                param.out->mutable_data<float>(TARGET(kXPU)),
                                write_size,
                                param.value);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fill_any_like,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::FillAnyLikeCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .Finalize();
