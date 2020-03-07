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

#include "lite/backends/arm/math/affine_channel.h"
#include <algorithm>
#include <limits>
#include <memory>
#include "lite/backends/arm/math/axpy.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/saturate.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void affine_channel_func(const float* x,
                         const float* scale,
                         const float* bias,
                         const std::string data_layout,
                         int num,
                         int channel,
                         int height,
                         int width,
                         float* out) {
  if (data_layout == "NCHW") {
    int hw_size = height * width;
    for (int n = 0; n < num; n++) {
      for (int c = 0; c < channel; c++) {
        const float* x_ptr = x + n * channel * hw_size + c * hw_size;
        const float* scale_ptr = scale + c;
        const float* bias_ptr = bias + c;
        float* out_ptr = out + n * channel * hw_size + c * hw_size;
        for (int i = 0; i < hw_size; i++) {
          *out_ptr = (*x_ptr) * (*scale_ptr) + (*bias_ptr);
          x_ptr++;
          out_ptr++;
        }
      }
    }
  } else if (data_layout == "NHWC") {
    int nhw = num * height * width;
    for (int i = 0; i < nhw; i++) {
      const float* x_ptr = x + i * channel;
      float* out_ptr = out + i * channel;
      for (int c = 0; c < channel; c++) {
        *out_ptr = (*x_ptr) * scale[c] + bias[c];
        x_ptr++;
        out_ptr++;
      }
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
