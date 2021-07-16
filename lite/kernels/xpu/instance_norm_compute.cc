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

#include "lite/kernels/xpu/instance_norm_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void InstanceNormCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  auto x_dims = param.x->dims();
  CHECK_EQ(x_dims.size(), 4);
  int n = x_dims[0];
  int c = x_dims[1];
  int h = x_dims[2];
  int w = x_dims[3];

  int ret = xdnn::instance_norm<float>(
      ctx.GetRawContext(),
      param.x->data<float>(),
      param.out->mutable_data<float>(TARGET(kXPU)),
      n,
      c,
      h,
      w,
      param.epsilon,
      param.scale->data<float>(),
      param.bias->data<float>(),
      param.saved_mean->mutable_data<float>(TARGET(kXPU)),
      param.saved_variance->mutable_data<float>(TARGET(kXPU)),
      true);

  CHECK_EQ(ret, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(instance_norm,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::InstanceNormCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
