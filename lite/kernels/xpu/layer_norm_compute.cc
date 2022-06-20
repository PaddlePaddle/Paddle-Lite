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

#include "lite/kernels/xpu/layer_norm_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void LayerNormCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto x_dims = param.X->dims();
  auto axis = param.begin_norm_axis;
  auto matrix_dim = x_dims.Flatten2D(axis);
  float epsilon = param.epsilon;

  int r = xdnn::layer_norm(ctx.GetRawContext(),    /* context */
                           param.X->data<float>(), /* in */
                           param.Y->mutable_data<float>(TARGET(kXPU)), /* out */
                           matrix_dim[0],                              /* m */
                           matrix_dim[1],                              /* n */
                           epsilon,                    /* epsilon */
                           param.Scale->data<float>(), /* scale */
                           param.Bias->data<float>(),  /* bias */
                           nullptr,
                           nullptr);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(layer_norm,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::LayerNormCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Mean", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Variance", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
