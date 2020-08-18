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

#include "lite/kernels/arm/pixel_shuffle_compute.h"
#include "lite/backends/arm/math/pixel_shuffle.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void PixelShuffleCompute::Run() {
  auto& param = Param<operators::PixelShuffleParam>();

  const float* x_data = param.x->data<float>();
  float* output_data = param.output->mutable_data<float>();
  const int upscale_factor = param.upscale_factor;

  const int batch_size = param.x->dims()[0];
  const int height = param.x->dims()[2];
  const int width = param.x->dims()[3];
  const int out_channels = param.output->dims()[1];
  const int out_height = param.output->dims()[2];
  const int out_width = param.output->dims()[3];

  if (upscale_factor == 2) {
    lite::arm::math::pixel_shuffle_scale2_fp32(x_data,
                                               output_data,
                                               batch_size,
                                               height,
                                               width,
                                               out_channels,
                                               out_height,
                                               out_width);
  } else if (upscale_factor == 3) {
    lite::arm::math::pixel_shuffle_scale3_fp32(x_data,
                                               output_data,
                                               batch_size,
                                               height,
                                               width,
                                               out_channels,
                                               out_height,
                                               out_width);
  } else if (upscale_factor == 4) {
    lite::arm::math::pixel_shuffle_scale4_fp32(x_data,
                                               output_data,
                                               batch_size,
                                               height,
                                               width,
                                               out_channels,
                                               out_height,
                                               out_width);
  } else {
    lite::arm::math::pixel_shuffle_native_fp32(x_data,
                                               output_data,
                                               batch_size,
                                               height,
                                               width,
                                               out_channels,
                                               out_height,
                                               out_width,
                                               upscale_factor);
  }

#ifdef LITE_WITH_PROFILE
  kernel_func_name_ = "pixel_shuffle_func";
#endif
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(pixel_shuffle,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::PixelShuffleCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
