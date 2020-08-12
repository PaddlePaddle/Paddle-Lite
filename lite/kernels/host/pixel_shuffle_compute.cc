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

#include "lite/kernels/host/pixel_shuffle_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void PixelShuffleCompute::Run() {
  auto& param = Param<operators::PixelShuffleParam>();

  const float* x_data = param.x->data<float>();
  float* output_data = param.output->mutable_data<float>();
  int upscale_factor = param.upscale_factor;

  int batch_size = param.x->dims()[0];
  int height = param.x->dims()[2];
  int width = param.x->dims()[3];
  int out_channels = param.output->dims()[1];
  int out_height = param.output->dims()[2];
  int out_width = param.output->dims()[3];

  for (int nc = 0; nc < batch_size * out_channels; nc++) {
    const float* inptr = x_data + nc * out_height * out_width;
    float* outptr_nc = output_data + nc * out_height * out_width;

    for (int sh = 0; sh < upscale_factor; sh++) {
      for (int sw = 0; sw < upscale_factor; sw++) {
        float* outptr = outptr_nc + sh * out_width + sw;
        for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
            outptr[0] = inptr[0];
            inptr++;
            outptr += upscale_factor;
          }
          outptr += (upscale_factor - 1) * out_width;
        }
      }
    }
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(pixel_shuffle,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::PixelShuffleCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
