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

#include "lite/kernels/xpu/norm_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void NormCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  auto x_dims = param.X->dims();
  int axis = param.axis;
  float epsilon = param.epsilon;
  if (axis < 0) {
    axis += x_dims.size();
  }
  CHECK_GE(axis, 0) << " axis < 0: " << axis;
  CHECK_LT(axis, x_dims.size()) << " axis >= rank: " << axis;

  int m = 1;
  for (int i = 0; i < axis; i++) {
    m = m * x_dims[i];
  }
  int t = x_dims[axis];
  int n = 1;
  for (int i = axis + 1; i < x_dims.size(); i++) {
    n = n * x_dims[i];
  }

  int r = xdnn::l2_normalize(ctx.GetRawContext(),
                             param.X->data<float>(),
                             param.Out->mutable_data<float>(TARGET(kXPU)),
                             nullptr,
                             epsilon,
                             m,
                             t,
                             n,
                             true);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    norm, kXPU, kFloat, kNCHW, paddle::lite::kernels::xpu::NormCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Norm", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
