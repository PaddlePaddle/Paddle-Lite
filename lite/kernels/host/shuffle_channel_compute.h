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
#pragma once
#include <random>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void shuffle_kernel(
    T* output, const T* input, int group_row, int group_col, int len) {
  for (int i = 0; i < group_row; ++i) {
    for (int j = 0; j < group_col; ++j) {
      const T* p_i = input + (i * group_col + j) * len;
      T* p_o = output + (j * group_row + i) * len;
      memcpy(p_o, p_i, len * sizeof(T));
    }
  }
}

template <typename T>
void shuffle_channel(const T* inputs,
                     T* outputs,
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
    shuffle_kernel<T>(outputs + i * fea_size,
                      inputs + i * fea_size,
                      group_row,
                      group_col,
                      spatial_size);
  }
}

template <typename T, TargetType TType, PrecisionType PType>
class ShuffleChannelCompute : public KernelLite<TType, PType> {
 public:
  using param_t = operators::ShuffleChannelParam;

  void Run() {
    auto& param = this->template Param<operators::ShuffleChannelParam>();
    const T* x_data = param.X->template data<T>();
    T* output_data = param.Out->template mutable_data<T>();
    DDim x_dims = param.X->dims();
    int group = param.group;
    int num = param.X->dims()[0];
    int channel = param.X->dims()[1];
    int height = param.X->dims()[2];
    int width = param.X->dims()[3];
    shuffle_channel<T>(x_data, output_data, group, num, channel, height, width);
  }

  virtual ~ShuffleChannelCompute() = default;
};

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#ifdef ENABLE_ARM_FP16
using float16_t = __fp16;
using shufflechannelfp16 =
    paddle::lite::kernels::host::ShuffleChannelCompute<float16_t,
                                                       TARGET(kARM),
                                                       PRECISION(kFP16)>;
#endif
