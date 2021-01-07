/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/x86/math/conv_bias.h"
#include <algorithm>

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void bias_add_broadcast(const float* dinx,
                        const float* diny,
                        float* dout,
                        int batch,
                        int channels,
                        int num) {
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const float* din_ptr = dinx + offset;
      const float diny_data = diny[j];
      float* dout_ptr = dout + offset;
      for (int k = 0; k < num; ++k) {
        *dout_ptr = *din_ptr + diny_data;
        dout_ptr++;
        din_ptr++;
      }
    }
  }
}

void bias_add_relu_broadcast(const float* dinx,
                             const float* diny,
                             float* dout,
                             int batch,
                             int channels,
                             int num) {
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const float* din_ptr = dinx + offset;
      const float diny_data = diny[j];
      float* dout_ptr = dout + offset;
      for (int k = 0; k < num; ++k) {
        *dout_ptr = (std::max)(0.f, *din_ptr + diny_data);
        dout_ptr++;
        din_ptr++;
      }
    }
  }
}

void bias_add_relu6_broadcast(const float* dinx,
                              const float* diny,
                              float* dout,
                              int batch,
                              int channels,
                              int num) {
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const float* din_ptr = dinx + offset;
      const float diny_data = diny[j];
      float* dout_ptr = dout + offset;
      for (int k = 0; k < num; ++k) {
        *dout_ptr = (std::min)(6.f, (std::max)(0.f, *din_ptr + diny_data));
        dout_ptr++;
        din_ptr++;
      }
    }
  }
}

}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
