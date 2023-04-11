/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/arm/math/reduce.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void mean_grad<float>(const float *out_grad, float *in_grad, int size) {
  float grad = out_grad[0] / size;
  float32x4_t grad_v = vdupq_n_f32(grad);
  int loop = size >> 2;
  int remain = size & 3;

  LITE_PARALLEL_BEGIN(i, tid, loop) {
    vst1q_f32(in_grad, grad_v);
    in_grad += 4;
  }
  LITE_PARALLEL_END()
  for (int i = 0; i < remain; ++i) {
    in_grad[i] = grad;
  }
}

#define ReduceSumKernel_FP32(avg)                               \
  int size_cur_post = cur * post;                               \
  for (int i = 0; i < pre; i++) {                               \
    for (int k = 0; k < post; k++) {                            \
      int j = 0;                                                \
      int out_idx = i * post + k;                               \
      float scalar_sum = 0;                                     \
      float32x4_t vec_sum = vdupq_n_f32(scalar_sum);            \
      for (; j + 3 < cur; j += 4) {                             \
        int in_idx = i * size_cur_post + j * post + k;          \
        float32x4_t vec_cur = {input[in_idx],                   \
                               input[in_idx + post],            \
                               input[in_idx + 2 * post],        \
                               input[in_idx + 3 * post]};       \
        vec_sum = vaddq_f32(vec_sum, vec_cur);                  \
      }                                                         \
      for (; j < cur; j++) {                                    \
        int in_idx = i * size_cur_post + j * post + k;          \
        scalar_sum += input[in_idx];                            \
      }                                                         \
      for (int id = 0; id < 4; id++) scalar_sum += vec_sum[id]; \
      output[out_idx] = scalar_sum / avg;                       \
    }                                                           \
  }

#define ReduceSumKernel_INT32(avg)                              \
  int size_cur_post = cur * post;                               \
  for (int i = 0; i < pre; i++) {                               \
    for (int k = 0; k < post; k++) {                            \
      int j = 0;                                                \
      int out_idx = i * post + k;                               \
      int scalar_sum = 0;                                       \
      int32x4_t vec_sum = vdupq_n_s32(scalar_sum);              \
      for (; j + 3 < cur; j += 4) {                             \
        int in_idx = i * size_cur_post + j * post + k;          \
        int32x4_t vec_cur = {input[in_idx],                     \
                             input[in_idx + post],              \
                             input[in_idx + 2 * post],          \
                             input[in_idx + 3 * post]};         \
        vec_sum = vaddq_s32(vec_sum, vec_cur);                  \
      }                                                         \
      for (; j < cur; j++) {                                    \
        int in_idx = i * size_cur_post + j * post + k;          \
        scalar_sum += input[in_idx];                            \
      }                                                         \
      for (int id = 0; id < 4; id++) scalar_sum += vec_sum[id]; \
      output[out_idx] = scalar_sum / avg;                       \
    }                                                           \
  }

#define ReduceSumKernel_INT64(avg)                                 \
  int size_cur_post = cur * post;                                  \
  for (int i = 0; i < pre; i++) {                                  \
    for (int k = 0; k < post; k++) {                               \
      int j = 0;                                                   \
      int out_idx = i * post + k;                                  \
      int64_t scalar_sum = 0;                                      \
      int64x2_t vec_sum = vdupq_n_s64(scalar_sum);                 \
      for (; j + 1 < cur; j += 2) {                                \
        int in_idx = i * size_cur_post + j * post + k;             \
        int64x2_t vec_cur = {input[in_idx], input[in_idx + post]}; \
        vec_sum = vaddq_s64(vec_sum, vec_cur);                     \
      }                                                            \
      for (; j < cur; j++) {                                       \
        int in_idx = i * size_cur_post + j * post + k;             \
        scalar_sum += input[in_idx];                               \
      }                                                            \
      for (int id = 0; id < 2; id++) scalar_sum += vec_sum[id];    \
      output[out_idx] = scalar_sum / avg;                          \
    }                                                              \
  }

// support sum/mean/max/min/prod
template <typename T>
void reduce_common_sum(const T *input, T *output, int pre, int cur, int post);
template <>
void reduce_common_sum<float>(
    const float *input, float *output, int pre, int cur, int post) {
  ReduceSumKernel_FP32(1)
}
template <>
void reduce_common_sum<int>(
    const int *input, int *output, int pre, int cur, int post) {
  ReduceSumKernel_INT32(1)
}
template <>
void reduce_common_sum<int64_t>(
    const int64_t *input, int64_t *output, int pre, int cur, int post) {
  ReduceSumKernel_INT64(1)
}

template <typename T>
void reduce_common_mean(const T *input, T *output, int pre, int cur, int post);
template <>
void reduce_common_mean<float>(
    const float *input, float *output, int pre, int cur, int post) {
  ReduceSumKernel_FP32(cur)
}
template <>
void reduce_common_mean<int>(
    const int *input, int *output, int pre, int cur, int post) {
  ReduceSumKernel_INT32(cur)
}
template <>
void reduce_common_mean<int64_t>(
    const int64_t *input, int64_t *output, int pre, int cur, int post) {
  ReduceSumKernel_INT64(cur)
}

template <typename T>
void reduce_common_max(const T *input, T *output, int pre, int cur, int post);
template <>
void reduce_common_max<float>(
    const float *input, float *output, int pre, int cur, int post) {
  int size_cur_post = cur * post;
  for (int i = 0; i < pre; i++) {
    for (int k = 0; k < post; k++) {
      int j = 0;
      int out_idx = i * post + k;
      float scalar_sum = std::numeric_limits<float>::min();
      float32x4_t vec_sum = vdupq_n_f32(scalar_sum);
      // reduce axis
      for (; j + 3 < cur; j += 4) {
        int in_idx = i * size_cur_post + j * post + k;
        float32x4_t vec_cur = {input[in_idx],
                               input[in_idx + post],
                               input[in_idx + 2 * post],
                               input[in_idx + 3 * post]};
        vec_sum = vmaxq_f32(vec_sum, vec_cur);
      }
      for (; j < cur; j++) {
        int in_idx = i * size_cur_post + j * post + k;
        scalar_sum = std::max(input[in_idx], scalar_sum);
      }
      for (int id = 0; id < 4; id++)
        scalar_sum = std::max(static_cast<float>(vec_sum[id]), scalar_sum);
      output[out_idx] = scalar_sum;
    }
  }
}
template <>
void reduce_common_max<int>(
    const int *input, int *output, int pre, int cur, int post) {
  int size_cur_post = cur * post;
  for (int i = 0; i < pre; i++) {
    for (int k = 0; k < post; k++) {
      int j = 0;
      int out_idx = i * post + k;
      int scalar_sum = std::numeric_limits<int>::min();
      int32x4_t vec_sum = vdupq_n_s32(scalar_sum);
      // reduce axis
      for (; j + 3 < cur; j += 4) {
        int in_idx = i * size_cur_post + j * post + k;
        int32x4_t vec_cur = {input[in_idx],
                             input[in_idx + post],
                             input[in_idx + 2 * post],
                             input[in_idx + 3 * post]};
        vec_sum = vmaxq_s32(vec_sum, vec_cur);
      }
      for (; j < cur; j++) {
        int in_idx = i * size_cur_post + j * post + k;
        scalar_sum = std::max(input[in_idx], scalar_sum);
      }
      for (int id = 0; id < 4; id++)
        scalar_sum = std::max(static_cast<int>(vec_sum[id]), scalar_sum);
      output[out_idx] = scalar_sum;
    }
  }
}
template <>
void reduce_common_max<int64_t>(
    const int64_t *input, int64_t *output, int pre, int cur, int post) {
  int size_cur_post = cur * post;
  for (int i = 0; i < pre; i++) {
    for (int k = 0; k < post; k++) {
      int out_idx = i * post + k;
      int64_t scalar_sum = std::numeric_limits<int64_t>::min();
      // reduce axis
      for (int j = 0; j < cur; j++) {
        int in_idx = i * size_cur_post + j * post + k;
        scalar_sum = std::max(input[in_idx], scalar_sum);
      }
      output[out_idx] = scalar_sum;
    }
  }
}

template <typename T>
void reduce_common_min(const T *input, T *output, int pre, int cur, int post);
template <>
void reduce_common_min<float>(
    const float *input, float *output, int pre, int cur, int post) {
  int size_cur_post = cur * post;
  for (int i = 0; i < pre; i++) {
    for (int k = 0; k < post; k++) {
      int j = 0;
      int out_idx = i * post + k;
      float scalar_sum = std::numeric_limits<float>::max();
      float32x4_t vec_sum = vdupq_n_f32(scalar_sum);
      // reduce axis
      for (; j + 3 < cur; j += 4) {
        int in_idx = i * size_cur_post + j * post + k;
        float32x4_t vec_cur = {input[in_idx],
                               input[in_idx + post],
                               input[in_idx + 2 * post],
                               input[in_idx + 3 * post]};
        vec_sum = vminq_f32(vec_sum, vec_cur);
      }
      for (; j < cur; j++) {
        int in_idx = i * size_cur_post + j * post + k;
        scalar_sum = std::min(input[in_idx], scalar_sum);
      }
      for (int id = 0; id < 4; id++)
        scalar_sum = std::min(static_cast<float>(vec_sum[id]), scalar_sum);
      output[out_idx] = scalar_sum;
    }
  }
}
template <>
void reduce_common_min<int>(
    const int *input, int *output, int pre, int cur, int post) {
  int size_cur_post = cur * post;
  for (int i = 0; i < pre; i++) {
    for (int k = 0; k < post; k++) {
      int j = 0;
      int out_idx = i * post + k;
      int scalar_sum = std::numeric_limits<int>::max();
      int32x4_t vec_sum = vdupq_n_s32(scalar_sum);
      // reduce axis
      for (; j + 3 < cur; j += 4) {
        int in_idx = i * size_cur_post + j * post + k;
        int32x4_t vec_cur = {input[in_idx],
                             input[in_idx + post],
                             input[in_idx + 2 * post],
                             input[in_idx + 3 * post]};
        vec_sum = vminq_s32(vec_sum, vec_cur);
      }
      for (; j < cur; j++) {
        int in_idx = i * size_cur_post + j * post + k;
        scalar_sum = std::min(input[in_idx], scalar_sum);
      }
      for (int id = 0; id < 4; id++)
        scalar_sum = std::min(static_cast<int>(vec_sum[id]), scalar_sum);
      output[out_idx] = scalar_sum;
    }
  }
}
template <>
void reduce_common_min<int64_t>(
    const int64_t *input, int64_t *output, int pre, int cur, int post) {
  int size_cur_post = cur * post;
  for (int i = 0; i < pre; i++) {
    for (int k = 0; k < post; k++) {
      int j = 0;
      int out_idx = i * post + k;
      int64_t scalar_sum = std::numeric_limits<int64_t>::max();
      // reduce axis
      for (; j < cur; j++) {
        int in_idx = i * size_cur_post + j * post + k;
        scalar_sum = std::min(input[in_idx], scalar_sum);
      }
      output[out_idx] = scalar_sum;
    }
  }
}

template <typename T>
void reduce_common_prod(const T *input, T *output, int pre, int cur, int post);
template <>
void reduce_common_prod<float>(
    const float *input, float *output, int pre, int cur, int post) {
  int size_cur_post = cur * post;
  for (int i = 0; i < pre; i++) {
    for (int k = 0; k < post; k++) {
      int j = 0;
      int out_idx = i * post + k;
      float scalar_sum = 1.f;
      float32x4_t vec_sum = vdupq_n_f32(scalar_sum);
      // reduce axis
      for (; j + 3 < cur; j += 4) {
        int in_idx = i * size_cur_post + j * post + k;
        float32x4_t vec_cur = {input[in_idx],
                               input[in_idx + post],
                               input[in_idx + 2 * post],
                               input[in_idx + 3 * post]};
        vec_sum = vmulq_f32(vec_sum, vec_cur);
      }
      for (; j < cur; j++) {
        int in_idx = i * size_cur_post + j * post + k;
        scalar_sum *= input[in_idx];
      }
      for (int id = 0; id < 4; id++) scalar_sum *= vec_sum[id];
      output[out_idx] = scalar_sum;
    }
  }
}
template <>
void reduce_common_prod<int>(
    const int *input, int *output, int pre, int cur, int post) {
  int size_cur_post = cur * post;
  for (int i = 0; i < pre; i++) {
    for (int k = 0; k < post; k++) {
      int j = 0;
      int out_idx = i * post + k;
      int scalar_sum = 1;
      int32x4_t vec_sum = vdupq_n_s32(scalar_sum);
      // reduce axis
      for (; j + 3 < cur; j += 4) {
        int in_idx = i * size_cur_post + j * post + k;
        int32x4_t vec_cur = {input[in_idx],
                             input[in_idx + post],
                             input[in_idx + 2 * post],
                             input[in_idx + 3 * post]};
        vec_sum = vmulq_s32(vec_sum, vec_cur);
      }
      for (; j < cur; j++) {
        int in_idx = i * size_cur_post + j * post + k;
        scalar_sum *= input[in_idx];
      }
      for (int id = 0; id < 4; id++) scalar_sum *= vec_sum[id];
      output[out_idx] = scalar_sum;
    }
  }
}
template <>
void reduce_common_prod<int64_t>(
    const int64_t *input, int64_t *output, int pre, int cur, int post) {
  int size_cur_post = cur * post;
  for (int i = 0; i < pre; i++) {
    for (int k = 0; k < post; k++) {
      int j = 0;
      int out_idx = i * post + k;
      int scalar_sum = 1;
      // reduce axis
      for (; j < cur; j++) {
        int in_idx = i * size_cur_post + j * post + k;
        scalar_sum *= input[in_idx];
      }
      output[out_idx] = scalar_sum;
    }
  }
}

#define ReduceSumContKernel_FP32(avg)                         \
  int size_cur_post = cur;                                    \
  for (int i = 0; i < pre; i++) {                             \
    int j = 0;                                                \
    int out_idx = i;                                          \
    float scalar_sum = 0;                                     \
    float32x4_t vec_sum = vdupq_n_f32(scalar_sum);            \
    for (; j + 3 < cur; j += 4) {                             \
      int in_idx = i * size_cur_post + j;                     \
      float32x4_t vec_cur = vld1q_f32(input + in_idx);        \
      vec_sum = vaddq_f32(vec_sum, vec_cur);                  \
    }                                                         \
    for (; j < cur; j++) {                                    \
      int in_idx = i * size_cur_post + j;                     \
      scalar_sum += input[in_idx];                            \
    }                                                         \
    for (int id = 0; id < 4; id++) scalar_sum += vec_sum[id]; \
    output[out_idx] = scalar_sum / avg;                       \
  }

#define ReduceSumContKernel_INT32(avg)                        \
  int size_cur_post = cur;                                    \
  for (int i = 0; i < pre; i++) {                             \
    int j = 0;                                                \
    int out_idx = i;                                          \
    int scalar_sum = 0;                                       \
    int32x4_t vec_sum = vdupq_n_s32(scalar_sum);              \
    for (; j + 3 < cur; j += 4) {                             \
      int in_idx = i * size_cur_post + j;                     \
      int32x4_t vec_cur = vld1q_s32(input + in_idx);          \
      vec_sum = vaddq_s32(vec_sum, vec_cur);                  \
    }                                                         \
    for (; j < cur; j++) {                                    \
      int in_idx = i * size_cur_post + j;                     \
      scalar_sum += input[in_idx];                            \
    }                                                         \
    for (int id = 0; id < 4; id++) scalar_sum += vec_sum[id]; \
    output[out_idx] = scalar_sum / avg;                       \
  }

#define ReduceSumContKernel_INT64(avg)                        \
  int size_cur_post = cur;                                    \
  for (int i = 0; i < pre; i++) {                             \
    int j = 0;                                                \
    int out_idx = i;                                          \
    int scalar_sum = 0;                                       \
    int64x2_t vec_sum = vdupq_n_s64(scalar_sum);              \
    for (; j + 1 < cur; j += 2) {                             \
      int in_idx = i * size_cur_post + j;                     \
      int64x2_t vec_cur = vld1q_s64(input + in_idx);          \
      vec_sum = vaddq_s64(vec_sum, vec_cur);                  \
    }                                                         \
    for (; j < cur; j++) {                                    \
      int in_idx = i * size_cur_post + j;                     \
      scalar_sum += input[in_idx];                            \
    }                                                         \
    for (int id = 0; id < 2; id++) scalar_sum += vec_sum[id]; \
    output[out_idx] = scalar_sum / avg;                       \
  }

// contiguous memory
template <typename T>
void reduce_cont_sum(const T *input, T *output, int pre, int cur);
template <>
void reduce_cont_sum<float>(const float *input,
                            float *output,
                            int pre,
                            int cur) {
  ReduceSumContKernel_FP32(1)
}
template <>
void reduce_cont_sum<int>(const int *input, int *output, int pre, int cur) {
  ReduceSumContKernel_INT32(1)
}
template <>
void reduce_cont_sum<int64_t>(const int64_t *input,
                              int64_t *output,
                              int pre,
                              int cur) {
  ReduceSumContKernel_INT64(1)
}

template <typename T>
void reduce_cont_mean(const T *input, T *output, int pre, int cur);
template <>
void reduce_cont_mean<float>(const float *input,
                             float *output,
                             int pre,
                             int cur) {
  ReduceSumContKernel_FP32(cur)
}
template <>
void reduce_cont_mean<int>(const int *input, int *output, int pre, int cur) {
  ReduceSumContKernel_INT32(cur)
}
template <>
void reduce_cont_mean<int64_t>(const int64_t *input,
                               int64_t *output,
                               int pre,
                               int cur) {
  ReduceSumContKernel_INT64(cur)
}

template <typename T>
void reduce_cont_max(const T *input, T *output, int pre, int cur);
template <>
void reduce_cont_max<float>(const float *input,
                            float *output,
                            int pre,
                            int cur) {
  int size_cur_post = cur;
  for (int i = 0; i < pre; i++) {
    int j = 0;
    int out_idx = i;
    float scalar_sum = std::numeric_limits<float>::min();
    float32x4_t vec_sum = vdupq_n_f32(scalar_sum);
    // reduce axis
    for (; j + 3 < cur; j += 4) {
      int in_idx = i * size_cur_post + j;
      float32x4_t vec_cur = vld1q_f32(input + in_idx);
      vec_sum = vmaxq_f32(vec_sum, vec_cur);
    }
    for (; j < cur; j++) {
      int in_idx = i * size_cur_post + j;
      scalar_sum = std::max(input[in_idx], scalar_sum);
    }
    for (int id = 0; id < 4; id++)
      scalar_sum = std::max(static_cast<float>(vec_sum[id]), scalar_sum);
    output[out_idx] = scalar_sum;
  }
}
template <>
void reduce_cont_max<int>(const int *input, int *output, int pre, int cur) {
  int size_cur_post = cur;
  for (int i = 0; i < pre; i++) {
    int j = 0;
    int out_idx = i;
    int scalar_sum = std::numeric_limits<int>::min();
    int32x4_t vec_sum = vdupq_n_s32(scalar_sum);
    // reduce axis
    for (; j + 3 < cur; j += 4) {
      int in_idx = i * size_cur_post + j;
      int32x4_t vec_cur = vld1q_s32(input + in_idx);
      vec_sum = vmaxq_s32(vec_sum, vec_cur);
    }
    for (; j < cur; j++) {
      int in_idx = i * size_cur_post + j;
      scalar_sum = std::max(input[in_idx], scalar_sum);
    }
    for (int id = 0; id < 4; id++)
      scalar_sum = std::max(static_cast<int>(vec_sum[id]), scalar_sum);
    output[out_idx] = scalar_sum;
  }
}
template <>
void reduce_cont_max<int64_t>(const int64_t *input,
                              int64_t *output,
                              int pre,
                              int cur) {
  int size_cur_post = cur;
  for (int i = 0; i < pre; i++) {
    int64_t scalar_sum = std::numeric_limits<int64_t>::min();
    // reduce axis
    for (int j = 0; j < cur; j++) {
      int in_idx = i * size_cur_post + j;
      scalar_sum = std::max(input[in_idx], scalar_sum);
    }
    output[i] = scalar_sum;
  }
}

template <typename T>
void reduce_cont_min(const T *input, T *output, int pre, int cur);
template <>
void reduce_cont_min<float>(const float *input,
                            float *output,
                            int pre,
                            int cur) {
  int size_cur_post = cur;
  for (int i = 0; i < pre; i++) {
    int j = 0;
    int out_idx = i;
    float scalar_sum = std::numeric_limits<float>::max();
    float32x4_t vec_sum = vdupq_n_f32(scalar_sum);
    // reduce axis
    for (; j + 3 < cur; j += 4) {
      int in_idx = i * size_cur_post + j;
      float32x4_t vec_cur = vld1q_f32(input + in_idx);
      vec_sum = vminq_f32(vec_sum, vec_cur);
    }
    for (; j < cur; j++) {
      int in_idx = i * size_cur_post + j;
      scalar_sum = std::min(input[in_idx], scalar_sum);
    }
    for (int id = 0; id < 4; id++)
      scalar_sum = std::min(static_cast<float>(vec_sum[id]), scalar_sum);
    output[out_idx] = scalar_sum;
  }
}
template <>
void reduce_cont_min<int>(const int *input, int *output, int pre, int cur) {
  int size_cur_post = cur;
  for (int i = 0; i < pre; i++) {
    int j = 0;
    int out_idx = i;
    int scalar_sum = std::numeric_limits<int>::max();
    int32x4_t vec_sum = vdupq_n_s32(scalar_sum);
    // reduce axis
    for (; j + 3 < cur; j += 4) {
      int in_idx = i * size_cur_post + j;
      int32x4_t vec_cur = vld1q_s32(input + in_idx);
      vec_sum = vminq_s32(vec_sum, vec_cur);
    }
    for (; j < cur; j++) {
      int in_idx = i * size_cur_post + j;
      scalar_sum = std::min(input[in_idx], scalar_sum);
    }
    for (int id = 0; id < 4; id++)
      scalar_sum = std::min(static_cast<int>(vec_sum[id]), scalar_sum);
    output[out_idx] = scalar_sum;
  }
}
template <>
void reduce_cont_min<int64_t>(const int64_t *input,
                              int64_t *output,
                              int pre,
                              int cur) {
  int size_cur_post = cur;
  for (int i = 0; i < pre; i++) {
    int64_t scalar_sum = std::numeric_limits<int64_t>::max();
    // reduce axis
    for (int j = 0; j < cur; j++) {
      int in_idx = i * size_cur_post + j;
      scalar_sum = std::min(input[in_idx], scalar_sum);
    }
    output[i] = scalar_sum;
  }
}

template <typename T>
void reduce_cont_prod(const T *input, T *output, int pre, int cur);
template <>
void reduce_cont_prod<float>(const float *input,
                             float *output,
                             int pre,
                             int cur) {
  int size_cur_post = cur;
  for (int i = 0; i < pre; i++) {
    int j = 0;
    float scalar_sum = 1.f;
    float32x4_t vec_sum = vdupq_n_f32(scalar_sum);
    // reduce axis
    for (; j + 3 < cur; j += 4) {
      int in_idx = i * size_cur_post + j;
      float32x4_t vec_cur = vld1q_f32(input + in_idx);
      vec_sum = vmulq_f32(vec_sum, vec_cur);
    }
    for (; j < cur; j++) {
      int in_idx = i * size_cur_post + j;
      scalar_sum *= input[in_idx];
    }
    for (int id = 0; id < 4; id++) scalar_sum *= vec_sum[id];
    output[i] = scalar_sum;
  }
}
template <>
void reduce_cont_prod<int>(const int *input, int *output, int pre, int cur) {
  int size_cur_post = cur;
  for (int i = 0; i < pre; i++) {
    int j = 0;
    int scalar_sum = 1;
    int32x4_t vec_sum = vdupq_n_s32(scalar_sum);
    // reduce axis
    for (; j + 3 < cur; j += 4) {
      int in_idx = i * size_cur_post + j;
      int32x4_t vec_cur = vld1q_s32(input + in_idx);
      vec_sum = vmulq_s32(vec_sum, vec_cur);
    }
    for (; j < cur; j++) {
      int in_idx = i * size_cur_post + j;
      scalar_sum *= input[in_idx];
    }
    for (int id = 0; id < 4; id++) scalar_sum *= vec_sum[id];
    output[i] = scalar_sum;
  }
}
template <>
void reduce_cont_prod<int64_t>(const int64_t *input,
                               int64_t *output,
                               int pre,
                               int cur) {
  int size_cur_post = cur;
  for (int i = 0; i < pre; i++) {
    int64_t scalar_sum = 1;
    // reduce axis
    for (int j = 0; j < cur; j++) {
      int in_idx = i * size_cur_post + j;
      scalar_sum *= input[in_idx];
    }
    output[i] = scalar_sum;
  }
}

int production(const std::vector<int64_t> &x_dims, int start, int end) {
  int res = 1;
  for (int i = start; i <= end; i++) res *= static_cast<int>(x_dims[i]);
  return res;
}

#define REDUCE_PROCESS(dim_num, input_dims, input_ptr, output_ptr)     \
  if (dim_num == x_dims_size - 1) {                                    \
    func_cont(input_ptr,                                               \
              output_ptr,                                              \
              production(input_dims, 0, x_dims_size - 2),              \
              input_dims[x_dims_size - 1]);                            \
    input_dims[dim_num] = 1;                                           \
  } else {                                                             \
    func_common(input_ptr,                                             \
                output_ptr,                                            \
                production(input_dims, 0, dim_num - 1),                \
                input_dims[dim_num],                                   \
                production(input_dims, dim_num + 1, x_dims_size - 1)); \
    input_dims[dim_num] = 1;                                           \
  }

template <typename T, typename CommonFunctor, typename ContFunctor>
void Reduce_n_dims(const T *X,
                   const std::vector<int64_t> &x_dims,
                   T *Out,
                   const std::vector<int64_t> &out_dims,
                   const std::vector<int> &dim,
                   CommonFunctor func_common,
                   ContFunctor func_cont) {
  Tensor tmp_buf_0, tmp_buf_1;
  tmp_buf_0.Resize(x_dims);
  tmp_buf_1.Resize(x_dims);
  T *tmp_buf_ptr_0 = tmp_buf_0.template mutable_data<T>();
  T *tmp_buf_ptr_1 = tmp_buf_1.template mutable_data<T>();
  auto input_dims = x_dims;
  auto x_dims_size = x_dims.size();
  auto dim_size = dim.size();
  if (dim_size == 1) {
    REDUCE_PROCESS(dim[0], input_dims, X, Out);
  } else if (dim_size == 2) {
    REDUCE_PROCESS(dim[0], input_dims, X, tmp_buf_ptr_0);
    REDUCE_PROCESS(dim[1], input_dims, tmp_buf_ptr_0, Out);
  } else if (dim_size == 3) {
    REDUCE_PROCESS(dim[0], input_dims, X, tmp_buf_ptr_0);
    REDUCE_PROCESS(dim[1], input_dims, tmp_buf_ptr_0, tmp_buf_ptr_1);
    REDUCE_PROCESS(dim[2], input_dims, tmp_buf_ptr_1, Out);
  } else if (dim_size == 4) {
    REDUCE_PROCESS(dim[0], input_dims, X, tmp_buf_ptr_0);
    REDUCE_PROCESS(dim[1], input_dims, tmp_buf_ptr_0, tmp_buf_ptr_1);
    REDUCE_PROCESS(dim[2], input_dims, tmp_buf_ptr_1, tmp_buf_ptr_0);
    REDUCE_PROCESS(dim[3], input_dims, tmp_buf_ptr_0, Out);
  } else if (dim_size == 5) {
    REDUCE_PROCESS(dim[0], input_dims, X, tmp_buf_ptr_0);
    REDUCE_PROCESS(dim[1], input_dims, tmp_buf_ptr_0, tmp_buf_ptr_1);
    REDUCE_PROCESS(dim[2], input_dims, tmp_buf_ptr_1, tmp_buf_ptr_0);
    REDUCE_PROCESS(dim[3], input_dims, tmp_buf_ptr_0, tmp_buf_ptr_1);
    REDUCE_PROCESS(dim[4], input_dims, tmp_buf_ptr_1, Out);
  } else {
    LOG(FATAL) << "ReduceOP unsupport dim(should be in [1, 5]) = " << dim_size;
  }
}

template <typename T>
void ReduceKernel(const T *X,
                  const std::vector<int64_t> &x_dims,
                  T *Out,
                  const std::vector<int64_t> &out_dims,
                  const std::vector<int> &dim,
                  bool reduce_all,
                  ReduceProcessType op_name) {
  int x_rank = x_dims.size();
  int dim_size = dim.size();
  if (reduce_all) {
    int dim_prod = production(x_dims, 0, x_rank - 1);
    switch (op_name) {
      case ReduceProcessType::mean:
        reduce_cont_mean(X, Out, 1, dim_prod);
        break;
      case ReduceProcessType::sum:
        reduce_cont_sum(X, Out, 1, dim_prod);
        break;
      case ReduceProcessType::min:
        reduce_cont_min(X, Out, 1, dim_prod);
        break;
      case ReduceProcessType::max:
        reduce_cont_max(X, Out, 1, dim_prod);
        break;
      case ReduceProcessType::prod:
        reduce_cont_prod(X, Out, 1, dim_prod);
        break;
      default:
        LOG(FATAL) << "Reduce unsupported process type.";
    }
  } else {
    switch (op_name) {
      case ReduceProcessType::mean:
        Reduce_n_dims<T,
                      std::function<void(const T *, T *, int, int, int)>,
                      std::function<void(const T *, T *, int, int)>>(
            X,
            x_dims,
            Out,
            out_dims,
            dim,
            reduce_common_mean<T>,
            reduce_cont_mean<T>);
        break;
      case ReduceProcessType::sum:
        Reduce_n_dims<T,
                      std::function<void(const T *, T *, int, int, int)>,
                      std::function<void(const T *, T *, int, int)>>(
            X,
            x_dims,
            Out,
            out_dims,
            dim,
            reduce_common_sum<T>,
            reduce_cont_sum<T>);
        break;
      case ReduceProcessType::min:
        Reduce_n_dims<T,
                      std::function<void(const T *, T *, int, int, int)>,
                      std::function<void(const T *, T *, int, int)>>(
            X,
            x_dims,
            Out,
            out_dims,
            dim,
            reduce_common_min<T>,
            reduce_cont_min<T>);
        break;
      case ReduceProcessType::max:
        Reduce_n_dims<T,
                      std::function<void(const T *, T *, int, int, int)>,
                      std::function<void(const T *, T *, int, int)>>(
            X,
            x_dims,
            Out,
            out_dims,
            dim,
            reduce_common_max<T>,
            reduce_cont_max<T>);
        break;
      case ReduceProcessType::prod:
        Reduce_n_dims<T,
                      std::function<void(const T *, T *, int, int, int)>,
                      std::function<void(const T *, T *, int, int)>>(
            X,
            x_dims,
            Out,
            out_dims,
            dim,
            reduce_common_prod<T>,
            reduce_cont_prod<T>);
        break;
      default:
        LOG(FATAL) << "Reduce unsupported process type.";
    }
  }
}

template <>
void ReduceImpl(const float *X,
                const std::vector<int64_t> &x_dims,
                float *Out,
                const std::vector<int64_t> &out_dims,
                const std::vector<int> &dim,
                bool reduce_all,
                ReduceProcessType op_name) {
  ReduceKernel<float>(X, x_dims, Out, out_dims, dim, reduce_all, op_name);
}
template <>
void ReduceImpl(const int *X,
                const std::vector<int64_t> &x_dims,
                int *Out,
                const std::vector<int64_t> &out_dims,
                const std::vector<int> &dim,
                bool reduce_all,
                ReduceProcessType op_name) {
  ReduceKernel<int>(X, x_dims, Out, out_dims, dim, reduce_all, op_name);
}
template <>
void ReduceImpl(const int64_t *X,
                const std::vector<int64_t> &x_dims,
                int64_t *Out,
                const std::vector<int64_t> &out_dims,
                const std::vector<int> &dim,
                bool reduce_all,
                ReduceProcessType op_name) {
  ReduceKernel<int64_t>(X, x_dims, Out, out_dims, dim, reduce_all, op_name);
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
