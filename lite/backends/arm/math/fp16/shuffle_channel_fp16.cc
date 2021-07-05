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

#include "lite/backends/arm/math/fp16/shuffle_channel_fp16.h"
#include <cstring>

namespace paddle {
namespace lite_metal {
namespace arm {
namespace math {
namespace fp16 {

void shuffle_kernel(float16_t* output,
                    const float16_t* input,
                    int group_row,
                    int group_col,
                    int len) {
  for (int i = 0; i < group_row; ++i) {
    for (int j = 0; j < group_col; ++j) {
      const float16_t* p_i = input + (i * group_col + j) * len;
      float16_t* p_o = output + (j * group_row + i) * len;
      memcpy(p_o, p_i, len * sizeof(float16_t));
    }
  }
}

void shuffle_channel(const float16_t* inputs,
                     float16_t* outputs,
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

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
