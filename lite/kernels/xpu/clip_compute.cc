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

#include "lite/kernels/xpu/clip_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void ClipCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto min_tensor = param.min_tensor;
  auto max_tensor = param.max_tensor;
  float min = param.min;
  float max = param.max;
  if (min_tensor != nullptr) {
    min = min_tensor->data<float>()[0];
  }
  if (max_tensor != nullptr) {
    max = max_tensor->data<float>()[0];
  }
  int r = xdnn::clip_v2<float>(ctx.GetRawContext(),
                               param.x->data<float>(),
                               param.out->mutable_data<float>(TARGET(kXPU)),
                               param.x->numel(),
                               min,
                               max);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    clip, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::ClipCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Min", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Max", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
