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

template <class T, PrecisionType PType>
void BatchNormCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  float epsilon = param.epsilon;
  auto& x_dims = param.x->dims();
  CHECK_LE(x_dims.size(), 5);
  std::vector<int> x_shape(5, 1);
  for (int i = 0; i < x_dims.size(); i++) {
    x_shape[i] = x_dims[i];
  }
  if (x_dims.size() == 5) {
    x_shape[3] *= x_shape[4];
  }

  int r =
      xdnn::batch_norm_infer<T>(ctx.GetRawContext(),
                                param.x->template data<T>(),
                                param.y->template mutable_data<T>(TARGET(kXPU)),
                                x_shape[0],
                                x_shape[1],
                                x_shape[2],
                                x_shape[3],
                                epsilon,
                                param.scale->template data<float>(),
                                param.bias->template data<float>(),
                                param.mean->template data<float>(),
                                param.variance->template data<float>(),
                                true);

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace xpu = paddle::lite::kernels::xpu;

using BatchNorm_FP32 = xpu::BatchNormCompute<float, PRECISION(kFloat)>;
using BatchNorm_FP16 = xpu::BatchNormCompute<float16, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(batch_norm, kXPU, kFloat, kNCHW, BatchNorm_FP32, def)
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

REGISTER_LITE_KERNEL(
    batch_norm, kXPU, kFP16, kNCHW, BatchNorm_FP16, DISABLE_XPU1_fp16)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFP16))})
    .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
