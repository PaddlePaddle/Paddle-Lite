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

#include "lite/backends/arm/math/sequence_pool.h"
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
void seq_pool_sum<float>(const float* din,
                         float* dout,
                         const std::vector<uint64_t> lod,
                         int64_t width) {
  for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
    const float* din_ptr = din + lod[i] * width;
    float* dout_ptr = dout + i * width;
    int64_t height = static_cast<int64_t>(lod[i + 1] - lod[i]);
    if (height > 0) {
      if (width == 1) {
        float sum = 0.f;
        for (int h = 0; h < height; ++h) {
          sum += din_ptr[h];
        }
        *dout_ptr = sum;
      } else {
        memcpy(dout_ptr, din_ptr, width * sizeof(float));
        din_ptr += width;
        height = height - 1;
        for (int h = 0; h < height; h++) {
          for (int w = 0; w < width; ++w) {
            dout_ptr[w] += din_ptr[w];
          }
          din_ptr += width;
        }
      }
    }
  }
}

template <>
void seq_pool_average<float>(const float* din,
                             float* dout,
                             const std::vector<uint64_t> lod,
                             int64_t width) {
  for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
    const float* din_ptr = din + lod[i] * width;
    float* dout_ptr = dout + i * width;
    int64_t height = static_cast<int64_t>(lod[i + 1] - lod[i]);
    if (height > 0) {
      if (width == 1) {
        float sum = 0.f;
        for (int h = 0; h < height; ++h) {
          sum += din_ptr[h];
        }
        *dout_ptr = sum / height;
      } else {
        memcpy(dout_ptr, din_ptr, width * sizeof(float));
        din_ptr += width;
        int remain_h = height - 1;
        for (int h = 0; h < remain_h; h++) {
          for (int w = 0; w < width; ++w) {
            dout_ptr[w] += din_ptr[w];
          }
          din_ptr += width;
        }
        for (int w = 0; w < width; ++w) {
          dout_ptr[w] /= height;
        }
      }
    }
  }
}

template <>
void seq_pool_sqrt<float>(const float* din,
                          float* dout,
                          const std::vector<uint64_t> lod,
                          int64_t width) {
  for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
    const float* din_ptr = din + lod[i] * width;
    float* dout_ptr = dout + i * width;
    int64_t height = static_cast<int64_t>(lod[i + 1] - lod[i]);
    if (height > 0) {
      float sqrt_len = sqrtf(height);
      if (width == 1) {
        float sum = 0.f;
        for (int h = 0; h < height; ++h) {
          sum += din_ptr[h];
        }
        *dout_ptr = sum / sqrt_len;
      } else {
        memcpy(dout_ptr, din_ptr, width * sizeof(float));
        din_ptr += width;
        int remain_h = height - 1;
        for (int h = 0; h < remain_h; h++) {
          for (int w = 0; w < width; ++w) {
            dout_ptr[w] += din_ptr[w];
          }
          din_ptr += width;
        }
        for (int w = 0; w < width; ++w) {
          dout_ptr[w] /= sqrt_len;
        }
      }
    }
  }
}

template <>
void seq_pool_max<float>(const float* din,
                         float* dout,
                         int64_t* index,
                         const std::vector<uint64_t> lod,
                         int64_t width) {
  for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
    const float* din_ptr = din + lod[i] * width;
    float* dout_ptr = dout + i * width;
    int64_t* index_ptr = index + i * width;
    int64_t height = static_cast<int64_t>(lod[i + 1] - lod[i]);
    if (height > 0) {
      if (width == 1) {
        float max = -std::numeric_limits<float>::max();
        int64_t max_index = -1;
        for (int h = 0; h < height; ++h) {
          max = std::max(max, din_ptr[h]);
          max_index = max >= din_ptr[h] ? h : max_index;
        }
        *dout_ptr = max;
        *index_ptr = max_index;
      } else {
        memcpy(dout_ptr, din_ptr, width * sizeof(float));
        memset(index_ptr, 0, width * sizeof(int64_t));
        din_ptr += width;
        int remain_h = height - 1;
        for (int h = 0; h < remain_h; h++) {
          for (int w = 0; w < width; w++) {
            dout_ptr[w] = std::max(dout_ptr[w], din_ptr[w]);
            index_ptr[w] = dout_ptr[w] > din_ptr[w] ? index_ptr[w] : h;
          }
          din_ptr += width;
        }
      }
    }
  }
}

template <>
void seq_pool_min<float>(const float* din,
                         float* dout,
                         int64_t* index,
                         const std::vector<uint64_t> lod,
                         int64_t width) {
  for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
    const float* din_ptr = din + lod[i] * width;
    float* dout_ptr = dout + i * width;
    int64_t* index_ptr = index + i * width;
    int64_t height = static_cast<int64_t>(lod[i + 1] - lod[i]);
    if (height > 0) {
      if (width == 1) {
        float min = std::numeric_limits<float>::max();
        int64_t min_index = -1;
        for (int h = 0; h < height; ++h) {
          min = std::min(min, din_ptr[h]);
          min_index = min >= din_ptr[h] ? h : min_index;
        }
        *dout_ptr = min;
        *index_ptr = min_index;
      } else {
        memcpy(dout_ptr, din_ptr, width * sizeof(float));
        memset(index_ptr, 0, width * sizeof(int64_t));
        din_ptr += width;
        int remain_h = height - 1;
        for (int h = 0; h < remain_h; h++) {
          for (int w = 0; w < width; w++) {
            dout_ptr[w] = std::min(dout_ptr[w], din_ptr[w]);
            index_ptr[w] = dout_ptr[w] < din_ptr[w] ? index_ptr[w] : h;
          }
          din_ptr += width;
        }
      }
    }
  }
}

template <>
void seq_pool_first<float>(const float* din,
                           float* dout,
                           const std::vector<uint64_t> lod,
                           int64_t width) {
  for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
    int64_t height = lod[i + 1] - lod[i];
    const float* din_ptr = din + width * lod[i];
    float* dout_ptr = dout + i * width;
    if (height > 0) {
      memcpy(dout_ptr, din_ptr, width * sizeof(float));
    }
  }
}

template <>
void seq_pool_last<float>(const float* din,
                          float* dout,
                          const std::vector<uint64_t> lod,
                          int64_t width) {
  for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
    int64_t height = lod[i + 1] - lod[i];
    int64_t seq_len = static_cast<int64_t>(lod[i + 1] - lod[0]);
    const float* din_ptr = din + width * seq_len;
    float* dout_ptr = dout + i * width;
    if (height > 0) {
      memcpy(dout_ptr, din_ptr - width, width * sizeof(float));
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
