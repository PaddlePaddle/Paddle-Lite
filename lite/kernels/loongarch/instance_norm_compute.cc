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

#include "lite/kernels/loongarch/instance_norm_compute.h"
#include "lite/backends/loongarch/xxl.h"
#include <cmath>
#include "lite/backends/loongarch/math/include/instance_norm.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace loongarch {

void InstanceNormCompute::PrepareForRun() {}

void InstanceNormCompute::Run() {
  auto& param = this->Param<param_t>();
  const float* in = param.x->data<float>();
  const float* scale =
      param.scale == nullptr ? nullptr : param.scale->data<float>();
  const float* bias =
      param.bias == nullptr ? nullptr : param.bias->data<float>();
  float* out = param.out->mutable_data<float>();
  float* saved_mean = param.saved_mean->mutable_data<float>();
  float* saved_variance = param.saved_variance->mutable_data<float>();
  float epsilon = param.epsilon;

  int n = param.x->dims()[0];
  int c = param.x->dims()[1];
  int height = param.x->dims()[2];
  int width = param.x->dims()[3];
  if (param.x->dims().size() == 5) {
    width = param.x->dims()[3] * param.x->dims()[4];
  }

  lite::loongarch::math::instance_norm(in,
                                 out,
                                 n,
                                 c,
                                 height,
                                 width,
                                 epsilon,
                                 scale,
                                 bias,
                                 saved_mean,
                                 saved_variance);
}

}  // namespace loongarch
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(instance_norm,
                     kLoongArch,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::loongarch::InstanceNormCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kLoongArch))})
    .Finalize();
