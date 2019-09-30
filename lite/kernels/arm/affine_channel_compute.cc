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

#include "lite/kernels/arm/affine_channel_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void AffineChannelCompute::Run() {
  auto& param = Param<operators::AffineChannelParam>();
  const lite::Tensor* x = param.X;
  const lite::Tensor* scale = param.Scale;
  const lite::Tensor* bias = param.Bias;
  const std::string data_layout = param.data_layout;
  lite::Tensor* out = param.Out;

  auto x_dims = x->dims();
  int num = x_dims[0];
  int channel = 0;
  int h = 0;
  int w = 0;
  if (data_layout == "NCHW") {
    channel = x_dims[1];
    h = x_dims[2];
    w = x_dims[3];
  } else if (data_layout == "NHWC") {
    channel = x_dims[3];
    h = x_dims[1];
    w = x_dims[2];
  }
  lite::arm::math::affine_channel_func(x->data<float>(),
                                       scale->data<float>(),
                                       bias->data<float>(),
                                       data_layout,
                                       num,
                                       channel,
                                       h,
                                       w,
                                       out->mutable_data<float>());
  return;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(affine_channel,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::AffineChannelCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
