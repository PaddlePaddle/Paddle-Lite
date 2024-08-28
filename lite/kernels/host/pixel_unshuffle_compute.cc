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

#include "lite/kernels/host/pixel_unshuffle_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void PixelUnShuffleCompute::Run() {
  auto& param = Param<operators::PixelUnShuffleParam>();

  const float* x_data = param.x->data<float>();
  float* output_data = param.output->mutable_data<float>();
  int downscale_factor = param.downscale_factor;

  int batch_size = param.x->dims()[0];
  int in_channels = param.x->dims()[1];
  int height = param.x->dims()[2];
  int width = param.x->dims()[3];
  int out_channels = param.output->dims()[1];
  int out_height = param.output->dims()[2];
  int out_width = param.output->dims()[3];

  for (int b = 0; b < batch_size; ++b) {
    for (int c = 0; c < in_channels; ++c) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          int out_c = c * downscale_factor * downscale_factor +
                      (y % downscale_factor) * downscale_factor +
                      (x % downscale_factor);
          int out_y = y / downscale_factor;
          int out_x = x / downscale_factor;
          int in_index = ((b * in_channels + c) * height + y) * width + x;
          int out_index =
              ((b * out_channels + out_c) * out_height + out_y) * out_width +
              out_x;
          output_data[out_index] = x_data[in_index];
        }
      }
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(pixel_unshuffle,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::PixelUnShuffleCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
