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

#include "lite/kernels/arm/shuffle_channel_compute.h"
#include "lite/backends/arm/math/funcs.h"

#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <>
void ShuffleChannelCompute<float, PRECISION(kFloat)>::Run() {
  auto& param = Param<operators::ShuffleChannelParam>();
  const float* x_data = param.X->data<float>();
  float* output_data = param.Out->mutable_data<float>();
  DDim x_dims = param.X->dims();
  int group = param.group;
  int num = param.X->dims()[0];
  int channel = param.X->dims()[1];
  int height = param.X->dims()[2];
  int width = param.X->dims()[3];
  lite::arm::math::shuffle_channel(
      x_data, output_data, group, num, channel, height, width);
}

#ifdef ENABLE_ARM_FP16
template <>
void ShuffleChannelCompute<float16_t, PRECISION(kFP16)>::Run() {
  auto& param = Param<operators::ShuffleChannelParam>();
  const float16_t* x_data = param.X->data<float16_t>();
  auto* output_data = param.Out->mutable_data<float16_t>();
  DDim x_dims = param.X->dims();
  int group = param.group;
  int num = param.X->dims()[0];
  int channel = param.X->dims()[1];
  int height = param.X->dims()[2];
  int width = param.X->dims()[3];
  lite::arm::math::fp16::shuffle_channel(
      x_data, output_data, group, num, channel, height, width);
}
#endif

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using float_shuffle =
    paddle::lite::kernels::arm::ShuffleChannelCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(shuffle_channel, kARM, kFloat, kNCHW, float_shuffle, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

#ifdef ENABLE_ARM_FP16
using fp16_shuffle =
    paddle::lite::kernels::arm::ShuffleChannelCompute<float16_t,
                                                      PRECISION(kFP16)>;
REGISTER_LITE_KERNEL(shuffle_channel, kARM, kFP16, kNCHW, fp16_shuffle, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
#endif
