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

#include "lite/backends/arm/math/sequence_pool_grad.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void seq_pool_sum_grad<float>(const float* din,
                              const float* dout_grad,
                              float* din_grad,
                              const std::vector<uint64_t> lod,
                              int64_t width) {
  for (int i = 0; i < static_cast<int>(lod.size()) - 1; i++) {
    int64_t height = static_cast<int64_t>(lod[i + 1] - lod[i]);
    const float* dout_grad_ptr = dout_grad + i * width;
    float* din_grad_ptr = din_grad + lod[i] * width;
    if (height > 0) {
      if (width == 1) {
        for (int h = 0; h < height; ++h) {
          din_grad_ptr[h] = dout_grad_ptr[h];
        }
      } else {
        for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
            din_grad_ptr[w] = dout_grad_ptr[w];
          }
          din_grad_ptr += width;
        }
      }
    }
  }
}

template <>
void seq_pool_average_grad<float>(const float* din,
                                  const float* dout_grad,
                                  float* din_grad,
                                  const std::vector<uint64_t> lod,
                                  int64_t width) {
  for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
    int64_t height = static_cast<int64_t>(lod[i + 1] - lod[i]);
    const float* dout_grad_ptr = dout_grad + i * width;
    float* din_grad_ptr = din_grad + lod[i] * width;
    float alpha = 1.0 / height;
    if (height > 0) {
      if (width == 1) {
        float sum = 0.f;
        for (int h = 0; h < height; ++h) {
          din_grad_ptr[h] = alpha * dout_grad_ptr[h];
        }
      } else {
        for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
            din_grad_ptr[w] = alpha * dout_grad_ptr[w];
          }
          din_grad_ptr += width;
        }
      }
    }
  }
}

template <>
void seq_pool_sqrt_grad<float>(const float* din,
                               const float* dout_grad,
                               float* din_grad,
                               const std::vector<uint64_t> lod,
                               int64_t width) {
  for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
    int64_t height = static_cast<int64_t>(lod[i + 1] - lod[i]);
    const float* dout_grad_ptr = dout_grad + i * width;
    float* din_grad_ptr = din_grad + lod[i] * width;
    float alpha = 1.0 / sqrtf(height);
    if (height > 0) {
      if (width == 1) {
        float sum = 0.f;
        for (int h = 0; h < height; ++h) {
          din_grad_ptr[h] = alpha * dout_grad_ptr[h];
        }
      } else {
        for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; w++) {
            din_grad_ptr[w] = alpha * dout_grad_ptr[w];
          }
          din_grad_ptr += width;
        }
      }
    }
  }
}

template <>
void seq_pool_max_grad<float>(const float* din,
                              const float* dout_grad,
                              const int64_t* index_grad,
                              float* din_grad,
                              const std::vector<uint64_t> lod,
                              int64_t width) {
  for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
    int64_t height = lod[i + 1] - lod[i];
    const float* dout_grad_ptr = dout_grad + i * width;
    const int64_t* index_grad_ptr = index_grad + i * width;
    float* din_grad_ptr = din_grad + lod[i] * width;
    if (height > 0) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          if (h == index_grad_ptr[w]) {
            din_grad_ptr[w] = dout_grad_ptr[w];
          } else {
            din_grad_ptr[w] = 0.f;
          }
        }
        din_grad_ptr += width;
      }
    }
  }
}

template <>
void seq_pool_first_grad<float>(const float* din,
                                const float* dout_grad,
                                float* din_grad,
                                const std::vector<uint64_t> lod,
                                int64_t width) {
  for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
    int64_t height = lod[i + 1] - lod[i];
    const float* dout_grad_ptr = dout_grad + i * width;
    float* din_grad_ptr = din_grad + lod[i] * width;
    if (height > 0) {
      for (int w = 0; w < width; w++) {
        din_grad_ptr[w] = dout_grad_ptr[w];
      }
      din_grad_ptr += width;
      for (int h = 1; h < height; h++) {
        for (int w = 0; w < width; w++) {
          din_grad_ptr[w] = 0.f;
        }
        din_grad_ptr += width;
      }
    }
  }
}

template <>
void seq_pool_last_grad<float>(const float* din,
                               const float* dout_grad,
                               float* din_grad,
                               const std::vector<uint64_t> lod,
                               int64_t width) {
  for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
    int64_t height = lod[i + 1] - lod[i];
    const float* dout_grad_ptr = dout_grad + i * width;
    float* din_grad_ptr = din_grad + lod[i] * width;
    if (height > 0) {
      for (int h = 0; h < height - 1; h++) {
        for (int w = 0; w < width; w++) {
          din_grad_ptr[w] = 0.f;
        }
        din_grad_ptr += width;
      }
      // last
      for (int w = 0; w < width; w++) {
        din_grad_ptr[w] = dout_grad_ptr[w];
      }
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
