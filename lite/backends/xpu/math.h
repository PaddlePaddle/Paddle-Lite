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

namespace paddle {
namespace lite {
namespace xpu {
namespace math {

static inline float FindMaxAbs(const float* data, int len) {
  float max_f = 0.0f;
  for (int i = 0; i < len; ++i) {
    float max = std::abs(data[i]);
    if (max > max_f) {
      max_f = max;
    }
  }
  return max_f;
}

template<typename T>
static inline void Transpose(const T* in, T* out, int h, int w) {
  for (int h1 = 0; h1 < w; ++h1) {
    for (int w1 = 0; w1 < h; ++w1) {
      out[h1 * h + w1] = in[w1 * w + h1];
    }
  }
}

}  // namespace math
}  // namespace xpu
}  // namespace lite
}  // namespace paddle
