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

#include "lite/backends/arm/math/shuffle_channel.h"
#include <typeinfo>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <typename Dtype>
void shuffle_kernel(
    Dtype* output, const Dtype* input, int group_row, int group_col, int len) {
  for (int i = 0; i < group_row; ++i) {
    for (int j = 0; j < group_col; ++j) {
      const Dtype* p_i = input + (i * group_col + j) * len;
      Dtype* p_o = output + (j * group_row + i) * len;
      memcpy(p_o, p_i, len * sizeof(Dtype));
    }
  }
}

template <>
void shuffle_channel<float>(const float* inputs,
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

template <>
void shuffle_channel<char>(const char* inputs,
                           char* outputs,
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

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
