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

#include "lite/kernels/xpu/dropout_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void DropoutCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  float scale = 1.0f;
  if (param.dropout_implementation == "upscale_in_train") {
    scale = 1.0f;
  } else {
    scale = 1.0f - param.dropout_prob;
  }
  int r = xdnn::scale<float>(
      ctx.GetRawContext(),                             /* context */
      param.x->data<float>(),                          /* src */
      param.output->mutable_data<float>(TARGET(kXPU)), /* dst */
      param.x->numel(),
      false,
      scale,
      0.0f);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(dropout,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::DropoutCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Seed", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Mask", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
