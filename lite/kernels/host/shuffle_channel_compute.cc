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

#include "lite/kernels/host/shuffle_channel_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {
void shuffle_kernel(
    float* output, const float* input, int group_row, int group_col, int len) {
  for (int i = 0; i < group_row; ++i) {
    for (int j = 0; j < group_col; ++j) {
      const float* p_i = input + (i * group_col + j) * len;
      float* p_o = output + (j * group_row + i) * len;
      memcpy(p_o, p_i, len * sizeof(float));
    }
  }
}

void shuffle_channel(const float* inputs,
                     float* outputs,
                     int group,
                     int num,
                     int channel,
                     int height,
                     int width) {
  int fea_size = channel * height * width;
  int spatial_size = height * width;
  int group_row = group;
  int group_col = channel / group;
  for (int i = 0; i < num; ++i) {
    shuffle_kernel(outputs + i * fea_size,
                   inputs + i * fea_size,
                   group_row,
                   group_col,
                   spatial_size);
  }
}
void ShuffleChannelCompute::Run() {
  auto& param = Param<operators::ShuffleChannelParam>();
  const float* x_data = param.X->data<float>();
  float* output_data = param.Out->mutable_data<float>();
  DDim x_dims = param.X->dims();
  int group = param.group;
  int num = param.X->dims()[0];
  int channel = param.X->dims()[1];
  int height = param.X->dims()[2];
  int width = param.X->dims()[3];
  shuffle_channel(x_data, output_data, group, num, channel, height, width);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(shuffle_channel,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::ShuffleChannelCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
