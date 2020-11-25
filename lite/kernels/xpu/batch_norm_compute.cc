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

#include "lite/kernels/xpu/batch_norm_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void BatchNormCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  float epsilon = param.epsilon;
  auto& x_dims = param.x->dims();
  CHECK_LE(x_dims.size(), 4);
  std::vector<int> x_shape(4, 1);
  for (int i = 0; i < x_dims.size(); i++) {
    x_shape[i] = x_dims[i];
  }

  int r =
      xdnn::batch_norm_infer_forward(ctx.GetRawContext(),
                                     epsilon,
                                     x_shape[0],
                                     x_shape[1],
                                     x_shape[2],
                                     x_shape[3],
                                     param.x->data<float>(),
                                     param.y->mutable_data<float>(TARGET(kXPU)),
                                     param.scale->data<float>(),
                                     param.bias->data<float>(),
                                     param.mean->data<float>(),
                                     param.variance->data<float>());
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(batch_norm,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::BatchNormCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
