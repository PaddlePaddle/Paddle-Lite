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

#include "lite/kernels/arm/squeeze_compute.h"
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void SqueezeCompute::Run() {
  auto& param = Param<operators::SqueezeParam>();
  auto x = param.X;
  auto output = param.Out;
  auto x_dims = x->dims();
  auto* x_data = x->data<float>();
  auto* out_data = output->mutable_data<float>();
  memcpy(out_data, x_data, x_dims.production() * sizeof(float));
}

void Squeeze2Compute::Run() {
  auto& param = Param<operators::SqueezeParam>();
  auto x = param.X;
  auto output = param.Out;
  auto xshape = param.XShape;
  auto x_dims = x->dims();
  auto* x_data = x->data<float>();
  auto* out_data = output->mutable_data<float>();
  auto* xshape_data = xshape->mutable_data<float>();
  memcpy(out_data, x_data, x_dims.production() * sizeof(float));
  memcpy(xshape_data, x_data, x_dims.production() * sizeof(float));
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(squeeze,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::SqueezeCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(squeeze2,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::Squeeze2Compute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
