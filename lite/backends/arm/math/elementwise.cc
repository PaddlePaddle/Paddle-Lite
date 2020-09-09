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
#include "lite/backends/arm/math/elementwise.h"
#include <math.h>
#include <algorithm>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void elementwise_add<float>(const float* dinx,
                            const float* diny,
                            float* dout,
                            int num) {
  int cnt = num >> 4;
  int remain = num % 16;
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const float* dinx_ptr = dinx + (i << 4);
    const float* diny_ptr = diny + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t dinx0 = vld1q_f32(dinx_ptr);
    float32x4_t dinx1 = vld1q_f32(dinx_ptr + 4);
    float32x4_t dinx2 = vld1q_f32(dinx_ptr + 8);
    float32x4_t dinx3 = vld1q_f32(dinx_ptr + 12);

    float32x4_t diny0 = vld1q_f32(diny_ptr);
    float32x4_t diny1 = vld1q_f32(diny_ptr + 4);
    float32x4_t diny2 = vld1q_f32(diny_ptr + 8);
    float32x4_t diny3 = vld1q_f32(diny_ptr + 12);

    dinx0 = vaddq_f32(dinx0, diny0);
    dinx1 = vaddq_f32(dinx1, diny1);
    dinx2 = vaddq_f32(dinx2, diny2);
    dinx3 = vaddq_f32(dinx3, diny3);

    vst1q_f32(dout_ptr, dinx0);
    vst1q_f32(dout_ptr + 4, dinx1);
    vst1q_f32(dout_ptr + 8, dinx2);
    vst1q_f32(dout_ptr + 12, dinx3);
  }
  if (remain > 0) {
    const float* dinx_ptr = dinx + (cnt << 4);
    const float* diny_ptr = diny + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      *dout_ptr = *dinx_ptr + *diny_ptr;
      dout_ptr++;
      dinx_ptr++;
      diny_ptr++;
    }
  }
}

template <>
void elementwise_add_relu<float>(const float* dinx,
                                 const float* diny,
                                 float* dout,
                                 int num) {
  int cnt = num >> 4;
  int remain = num % 16;
  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const float* dinx_ptr = dinx + (i << 4);
    const float* diny_ptr = diny + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t dinx0 = vld1q_f32(dinx_ptr);
    float32x4_t dinx1 = vld1q_f32(dinx_ptr + 4);
    float32x4_t dinx2 = vld1q_f32(dinx_ptr + 8);
    float32x4_t dinx3 = vld1q_f32(dinx_ptr + 12);

    float32x4_t diny0 = vld1q_f32(diny_ptr);
    float32x4_t diny1 = vld1q_f32(diny_ptr + 4);
    float32x4_t diny2 = vld1q_f32(diny_ptr + 8);
    float32x4_t diny3 = vld1q_f32(diny_ptr + 12);

    dinx0 = vaddq_f32(dinx0, diny0);
    dinx1 = vaddq_f32(dinx1, diny1);
    dinx2 = vaddq_f32(dinx2, diny2);
    dinx3 = vaddq_f32(dinx3, diny3);

    // relu
    dinx0 = vmaxq_f32(dinx0, vzero);
    dinx1 = vmaxq_f32(dinx1, vzero);
    dinx2 = vmaxq_f32(dinx2, vzero);
    dinx3 = vmaxq_f32(dinx3, vzero);

    vst1q_f32(dout_ptr, dinx0);
    vst1q_f32(dout_ptr + 4, dinx1);
    vst1q_f32(dout_ptr + 8, dinx2);
    vst1q_f32(dout_ptr + 12, dinx3);
  }
  if (remain > 0) {
    const float* dinx_ptr = dinx + (cnt << 4);
    const float* diny_ptr = diny + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      float tmp = *dinx_ptr + *diny_ptr;
      *dout_ptr = tmp > 0.f ? tmp : 0.f;
      dout_ptr++;
      dinx_ptr++;
      diny_ptr++;
    }
  }
}
template <>
void elementwise_add_tanh<float>(const float* dinx,
                                 const float* diny,
                                 float* dout,
                                 int num) {
  int cnt = num >> 4;
  int remain = num % 16;
  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const float* dinx_ptr = dinx + (i << 4);
    const float* diny_ptr = diny + (i << 4);
    float* dout_ptr = dout + (i << 4);

    // Elementwise_add
    float32x4_t dinx0 = vld1q_f32(dinx_ptr);
    float32x4_t diny0 = vld1q_f32(diny_ptr);
    float32x4_t dinx1 = vld1q_f32(dinx_ptr + 4);
    float32x4_t diny1 = vld1q_f32(diny_ptr + 4);
    float32x4_t dinx2 = vld1q_f32(dinx_ptr + 8);
    float32x4_t diny2 = vld1q_f32(diny_ptr + 8);
    float32x4_t dinx3 = vld1q_f32(dinx_ptr + 12);
    float32x4_t diny3 = vld1q_f32(diny_ptr + 12);

    dinx0 = vaddq_f32(dinx0, diny0);
    dinx1 = vaddq_f32(dinx1, diny1);
    dinx2 = vaddq_f32(dinx2, diny2);
    dinx3 = vaddq_f32(dinx3, diny3);

    for (int j = 0; j < 4; j++) {
      dinx0[j] = (expf(dinx0[j]) - expf(-dinx0[j])) /
                 (expf(dinx0[j]) + expf(-dinx0[j]));
      dinx1[j] = (expf(dinx1[j]) - expf(-dinx1[j])) /
                 (expf(dinx1[j]) + expf(-dinx1[j]));
      dinx2[j] = (expf(dinx2[j]) - expf(-dinx2[j])) /
                 (expf(dinx2[j]) + expf(-dinx2[j]));
      dinx3[j] = (expf(dinx3[j]) - expf(-dinx3[j])) /
                 (expf(dinx3[j]) + expf(-dinx3[j]));
    }
    vst1q_f32(dout_ptr, dinx0);
    vst1q_f32(dout_ptr + 4, dinx1);
    vst1q_f32(dout_ptr + 8, dinx2);
    vst1q_f32(dout_ptr + 12, dinx3);
  }
  if (remain > 0) {
    const float* dinx_ptr = dinx + (cnt << 4);
    const float* diny_ptr = diny + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      float tmp = *dinx_ptr + *diny_ptr;
      *dout_ptr = (expf(tmp) - expf(-tmp)) / (expf(tmp) + expf(-tmp));
      dout_ptr++;
      dinx_ptr++;
      diny_ptr++;
    }
  }
}

template <>
void elementwise_add_broadcast<float>(const float* dinx,
                                      const float* diny,
                                      float* dout,
                                      int batch,
                                      int channels,
                                      int num) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const float* din_ptr = dinx + offset;
      const float diny_data = diny[j];
      float* dout_ptr = dout + offset;

      int cnt = num >> 4;
      int remain = num % 16;
      float32x4_t rb = vdupq_n_f32(diny_data);
      for (int k = 0; k < cnt; ++k) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        float32x4_t din2 = vld1q_f32(din_ptr + 8);
        float32x4_t din3 = vld1q_f32(din_ptr + 12);

        din0 = vaddq_f32(din0, rb);
        din1 = vaddq_f32(din1, rb);
        din2 = vaddq_f32(din2, rb);
        din3 = vaddq_f32(din3, rb);

        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        vst1q_f32(dout_ptr + 8, din2);
        vst1q_f32(dout_ptr + 12, din3);
        din_ptr += 16;
        dout_ptr += 16;
      }
      if (remain >= 8) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        din0 = vaddq_f32(din0, rb);
        din1 = vaddq_f32(din1, rb);
        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        din_ptr += 8;
        dout_ptr += 8;
        remain -= 8;
      }
      if (remain >= 4) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        din0 = vaddq_f32(din0, rb);
        vst1q_f32(dout_ptr, din0);
        din_ptr += 4;
        dout_ptr += 4;
        remain -= 4;
      }
      if (remain > 0) {
        for (int p = 0; p < remain; p++) {
          *dout_ptr = *din_ptr + diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  }
}

template <>
void elementwise_add_relu_broadcast<float>(const float* dinx,
                                           const float* diny,
                                           float* dout,
                                           int batch,
                                           int channels,
                                           int num) {
  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const float* din_ptr = dinx + offset;
      const float diny_data = diny[j];
      float* dout_ptr = dout + offset;

      int cnt = num >> 4;
      int remain = num % 16;
      float32x4_t rb = vdupq_n_f32(diny_data);
      for (int k = 0; k < cnt; ++k) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        float32x4_t din2 = vld1q_f32(din_ptr + 8);
        float32x4_t din3 = vld1q_f32(din_ptr + 12);

        din0 = vaddq_f32(din0, rb);
        din1 = vaddq_f32(din1, rb);
        din2 = vaddq_f32(din2, rb);
        din3 = vaddq_f32(din3, rb);

        // relu
        din0 = vmaxq_f32(din0, vzero);
        din1 = vmaxq_f32(din1, vzero);
        din2 = vmaxq_f32(din2, vzero);
        din3 = vmaxq_f32(din3, vzero);

        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        vst1q_f32(dout_ptr + 8, din2);
        vst1q_f32(dout_ptr + 12, din3);
        din_ptr += 16;
        dout_ptr += 16;
      }
      if (remain >= 8) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        din0 = vaddq_f32(din0, rb);
        din1 = vaddq_f32(din1, rb);
        // relu
        din0 = vmaxq_f32(din0, vzero);
        din1 = vmaxq_f32(din1, vzero);
        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        din_ptr += 8;
        dout_ptr += 8;
        remain -= 8;
      }
      if (remain >= 4) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        din0 = vaddq_f32(din0, rb);
        // relu
        din0 = vmaxq_f32(din0, vzero);
        vst1q_f32(dout_ptr, din0);
        din_ptr += 4;
        dout_ptr += 4;
        remain -= 4;
      }
      if (remain > 0) {
        for (int p = 0; p < remain; p++) {
          float tmp = *din_ptr + diny_data;
          *dout_ptr = tmp > 0.f ? tmp : 0.f;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  }
}

template <>
void elementwise_add_grad<float>(const float* dout_grad,
                                 float* x_grad,
                                 int num) {
  int cnt = num >> 4;
  int remain = num & 0x0f;
#pragma omp parallel for
  for (int i = 0; i < cnt; ++i) {
    const float* out_data = dout_grad + 16 * i;
    float* x_data = x_grad + 16 * i;
    float32x4_t din0 = vld1q_f32(out_data);
    float32x4_t din1 = vld1q_f32(out_data + 4);
    float32x4_t din2 = vld1q_f32(out_data + 8);
    float32x4_t din3 = vld1q_f32(out_data + 12);
    vst1q_f32(x_data, din0);
    vst1q_f32(x_data + 4, din1);
    vst1q_f32(x_data + 8, din2);
    vst1q_f32(x_data + 12, din3);
  }
  if (remain > 0) {
    const float* out_data = dout_grad + 16 * cnt;
    float* x_data = x_grad + 16 * cnt;
    for (int i = 0; i < remain; ++i) {
      x_data[i] = out_data[i];
    }
  }
}
// we assume that y_data numel less than x_data, otherwise, call this function
// by change x_grad and y_grad position
template <>
void elementwise_add_grad_broadcast<float>(const float* dout_grad,
                                           float* x_grad,
                                           float* y_grad,
                                           int pre,
                                           int n,
                                           int post) {
  if (x_grad != nullptr) {
    elementwise_add_grad(dout_grad, x_grad, pre * n * post);
  }
  if (y_grad != nullptr) {
    memset(y_grad, 0, n * sizeof(float));
#pragma omp parallel for
    for (int i = 0; i < pre; ++i) {
      for (int j = 0; j < n; ++j) {
        float sum = 0;
        int cnt = post >> 2;
        int remain = post & 0x03;
        const float* out_data = dout_grad + (i * n + j) * post;
        float32x4_t sum_v = vdupq_n_f32(0);
        for (int ci = 0; ci < cnt; ++ci) {
          float32x4_t din = vld1q_f32(out_data + 4 * ci);
          sum_v = vaddq_f32(sum_v, din);
        }
        out_data += 4 * cnt;
        for (int ci = 0; ci < remain; ++ci) {
          sum += out_data[ci];
        }
        float32x2_t high = vget_high_f32(sum_v);
        float32x2_t low = vget_low_f32(sum_v);
        sum += vget_lane_f32(high, 0) + vget_lane_f32(high, 1) +
               vget_lane_f32(low, 0) + vget_lane_f32(low, 1);
        y_grad[j] += sum;
      }
    }
  }
}
template <>
void elementwise_sub<float>(const float* dinx,
                            const float* diny,
                            float* dout,
                            int num) {
  int cnt = num >> 4;
  int remain = num % 16;
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const float* dinx_ptr = dinx + (i << 4);
    const float* diny_ptr = diny + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t dinx0 = vld1q_f32(dinx_ptr);
    float32x4_t dinx1 = vld1q_f32(dinx_ptr + 4);
    float32x4_t dinx2 = vld1q_f32(dinx_ptr + 8);
    float32x4_t dinx3 = vld1q_f32(dinx_ptr + 12);

    float32x4_t diny0 = vld1q_f32(diny_ptr);
    float32x4_t diny1 = vld1q_f32(diny_ptr + 4);
    float32x4_t diny2 = vld1q_f32(diny_ptr + 8);
    float32x4_t diny3 = vld1q_f32(diny_ptr + 12);

    dinx0 = vsubq_f32(dinx0, diny0);
    dinx1 = vsubq_f32(dinx1, diny1);
    dinx2 = vsubq_f32(dinx2, diny2);
    dinx3 = vsubq_f32(dinx3, diny3);

    vst1q_f32(dout_ptr, dinx0);
    vst1q_f32(dout_ptr + 4, dinx1);
    vst1q_f32(dout_ptr + 8, dinx2);
    vst1q_f32(dout_ptr + 12, dinx3);
  }
  if (remain > 0) {
    const float* dinx_ptr = dinx + (cnt << 4);
    const float* diny_ptr = diny + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      *dout_ptr = *dinx_ptr - *diny_ptr;
      dout_ptr++;
      dinx_ptr++;
      diny_ptr++;
    }
  }
}

template <>
void elementwise_sub_relu<float>(const float* dinx,
                                 const float* diny,
                                 float* dout,
                                 int num) {
  int cnt = num >> 4;
  int remain = num % 16;
  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const float* dinx_ptr = dinx + (i << 4);
    const float* diny_ptr = diny + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t dinx0 = vld1q_f32(dinx_ptr);
    float32x4_t dinx1 = vld1q_f32(dinx_ptr + 4);
    float32x4_t dinx2 = vld1q_f32(dinx_ptr + 8);
    float32x4_t dinx3 = vld1q_f32(dinx_ptr + 12);

    float32x4_t diny0 = vld1q_f32(diny_ptr);
    float32x4_t diny1 = vld1q_f32(diny_ptr + 4);
    float32x4_t diny2 = vld1q_f32(diny_ptr + 8);
    float32x4_t diny3 = vld1q_f32(diny_ptr + 12);

    dinx0 = vsubq_f32(dinx0, diny0);
    dinx1 = vsubq_f32(dinx1, diny1);
    dinx2 = vsubq_f32(dinx2, diny2);
    dinx3 = vsubq_f32(dinx3, diny3);

    // relu
    dinx0 = vmaxq_f32(dinx0, vzero);
    dinx1 = vmaxq_f32(dinx1, vzero);
    dinx2 = vmaxq_f32(dinx2, vzero);
    dinx3 = vmaxq_f32(dinx3, vzero);

    vst1q_f32(dout_ptr, dinx0);
    vst1q_f32(dout_ptr + 4, dinx1);
    vst1q_f32(dout_ptr + 8, dinx2);
    vst1q_f32(dout_ptr + 12, dinx3);
  }
  if (remain > 0) {
    const float* dinx_ptr = dinx + (cnt << 4);
    const float* diny_ptr = diny + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      float tmp = *dinx_ptr - *diny_ptr;
      *dout_ptr = tmp > 0.f ? tmp : 0.f;
      dout_ptr++;
      dinx_ptr++;
      diny_ptr++;
    }
  }
}

template <>
void elementwise_sub_broadcast<float>(const float* dinx,
                                      const float* diny,
                                      float* dout,
                                      int batch,
                                      int channels,
                                      int num) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const float* din_ptr = dinx + offset;
      const float diny_data = diny[j];
      float* dout_ptr = dout + offset;

      int cnt = num >> 4;
      int remain = num % 16;
      float32x4_t rb = vdupq_n_f32(diny_data);
      for (int k = 0; k < cnt; ++k) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        float32x4_t din2 = vld1q_f32(din_ptr + 8);
        float32x4_t din3 = vld1q_f32(din_ptr + 12);

        din0 = vsubq_f32(din0, rb);
        din1 = vsubq_f32(din1, rb);
        din2 = vsubq_f32(din2, rb);
        din3 = vsubq_f32(din3, rb);

        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        vst1q_f32(dout_ptr + 8, din2);
        vst1q_f32(dout_ptr + 12, din3);
        din_ptr += 16;
        dout_ptr += 16;
      }
      if (remain >= 8) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        din0 = vsubq_f32(din0, rb);
        din1 = vsubq_f32(din1, rb);
        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        din_ptr += 8;
        dout_ptr += 8;
        remain -= 8;
      }
      if (remain >= 4) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        din0 = vsubq_f32(din0, rb);
        vst1q_f32(dout_ptr, din0);
        din_ptr += 4;
        dout_ptr += 4;
        remain -= 4;
      }
      if (remain > 0) {
        for (int p = 0; p < remain; p++) {
          *dout_ptr = *din_ptr - diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  }
}

template <>
void elementwise_sub_relu_broadcast<float>(const float* dinx,
                                           const float* diny,
                                           float* dout,
                                           int batch,
                                           int channels,
                                           int num) {
  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const float* din_ptr = dinx + offset;
      const float diny_data = diny[j];
      float* dout_ptr = dout + offset;

      int cnt = num >> 4;
      int remain = num % 16;
      float32x4_t rb = vdupq_n_f32(diny_data);
      for (int k = 0; k < cnt; ++k) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        float32x4_t din2 = vld1q_f32(din_ptr + 8);
        float32x4_t din3 = vld1q_f32(din_ptr + 12);

        din0 = vsubq_f32(din0, rb);
        din1 = vsubq_f32(din1, rb);
        din2 = vsubq_f32(din2, rb);
        din3 = vsubq_f32(din3, rb);

        // relu
        din0 = vmaxq_f32(din0, vzero);
        din1 = vmaxq_f32(din1, vzero);
        din2 = vmaxq_f32(din2, vzero);
        din3 = vmaxq_f32(din3, vzero);

        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        vst1q_f32(dout_ptr + 8, din2);
        vst1q_f32(dout_ptr + 12, din3);
        din_ptr += 16;
        dout_ptr += 16;
      }
      if (remain >= 8) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        din0 = vsubq_f32(din0, rb);
        din1 = vsubq_f32(din1, rb);
        // relu
        din0 = vmaxq_f32(din0, vzero);
        din1 = vmaxq_f32(din1, vzero);
        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        din_ptr += 8;
        dout_ptr += 8;
        remain -= 8;
      }
      if (remain >= 4) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        din0 = vsubq_f32(din0, rb);
        // relu
        din0 = vmaxq_f32(din0, vzero);
        vst1q_f32(dout_ptr, din0);
        din_ptr += 4;
        dout_ptr += 4;
        remain -= 4;
      }
      if (remain > 0) {
        for (int p = 0; p < remain; p++) {
          float tmp = *din_ptr - diny_data;
          *dout_ptr = tmp > 0.f ? tmp : 0.f;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  }
}
// we assume the formula is x-y
template <>
void elementwise_sub_grad<float>(const float* dout_grad,
                                 float* x_grad,
                                 float* y_grad,
                                 int num) {
  if (x_grad != nullptr) {
    elementwise_add_grad(dout_grad, x_grad, num);
  }
  if (y_grad != nullptr) {
    int cnt = num >> 4;
    int remain = num & 0x0f;
    float32x4_t minus = vdupq_n_f32(-1);
#pragma omp parallel for
    for (int i = 0; i < cnt; ++i) {
      const float* out_data = dout_grad + 16 * i;
      float* y_data = y_grad + 16 * i;
      float32x4_t din0 = vld1q_f32(out_data);
      float32x4_t din1 = vld1q_f32(out_data + 4);
      float32x4_t din2 = vld1q_f32(out_data + 8);
      float32x4_t din3 = vld1q_f32(out_data + 12);
      din0 = vmulq_f32(din0, minus);
      din1 = vmulq_f32(din1, minus);
      din2 = vmulq_f32(din2, minus);
      din3 = vmulq_f32(din3, minus);
      vst1q_f32(y_data, din0);
      vst1q_f32(y_data + 4, din1);
      vst1q_f32(y_data + 8, din2);
      vst1q_f32(y_data + 12, din3);
    }
    if (remain > 0) {
      const float* out_data = dout_grad + 16 * cnt;
      float* y_data = y_grad + 16 * cnt;
      for (int i = 0; i < remain; ++i) {
        y_data[i] = -out_data[i];
      }
    }
  }
}
// we assume that y_data numel less than x_data, otherwise, call this function
// by change x_grad and y_grad position
template <>
void elementwise_sub_grad_broadcast<float>(const float* dout_grad,
                                           float* x_grad,
                                           float* y_grad,
                                           int pre,
                                           int n,
                                           int post) {
  if (x_grad != nullptr) {
    elementwise_add_grad(dout_grad, x_grad, pre * n * post);
  }
  if (y_grad != nullptr) {
    memset(y_grad, 0, n * sizeof(float));
#pragma omp parallel for
    for (int i = 0; i < pre; ++i) {
      for (int j = 0; j < n; ++j) {
        float sum = 0;
        int cnt = post << 2;
        int remain = post & 0x03;
        const float* out_data = dout_grad + (i * n + j) * post;
        float32x4_t sum_v = vdupq_n_f32(0);
        for (int ci = 0; ci < cnt; ++ci) {
          float32x4_t din = vld1q_f32(out_data + 4 * ci);
          sum_v = vaddq_f32(sum_v, din);
        }
        out_data += 4 * cnt;
        for (int ci = 0; ci < remain; ++ci) {
          sum -= out_data[ci];
        }
        float32x2_t high = vget_high_f32(sum_v);
        float32x2_t low = vget_low_f32(sum_v);
        sum -= vget_lane_f32(high, 0) + vget_lane_f32(high, 1) +
               vget_lane_f32(low, 0) + vget_lane_f32(low, 1);
        y_grad[j] += sum;
      }
    }
  }
}

template <>
void elementwise_mul<float>(const float* dinx,
                            const float* diny,
                            float* dout,
                            int num) {
  int cnt = num >> 4;
  int remain = num % 16;
#pragma omp parallel for
  for (int i = 0; i < cnt; ++i) {
    const float* dinx_ptr = dinx + (i << 4);
    const float* diny_ptr = diny + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t dinx0 = vld1q_f32(dinx_ptr);
    float32x4_t dinx1 = vld1q_f32(dinx_ptr + 4);
    float32x4_t dinx2 = vld1q_f32(dinx_ptr + 8);
    float32x4_t dinx3 = vld1q_f32(dinx_ptr + 12);

    float32x4_t diny0 = vld1q_f32(diny_ptr);
    float32x4_t diny1 = vld1q_f32(diny_ptr + 4);
    float32x4_t diny2 = vld1q_f32(diny_ptr + 8);
    float32x4_t diny3 = vld1q_f32(diny_ptr + 12);

    dinx0 = vmulq_f32(dinx0, diny0);
    dinx1 = vmulq_f32(dinx1, diny1);
    dinx2 = vmulq_f32(dinx2, diny2);
    dinx3 = vmulq_f32(dinx3, diny3);

    vst1q_f32(dout_ptr, dinx0);
    vst1q_f32(dout_ptr + 4, dinx1);
    vst1q_f32(dout_ptr + 8, dinx2);
    vst1q_f32(dout_ptr + 12, dinx3);
  }
  if (remain > 0) {
    const float* dinx_ptr = dinx + (cnt << 4);
    const float* diny_ptr = diny + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      *dout_ptr = *dinx_ptr * *diny_ptr;
      dout_ptr++;
      dinx_ptr++;
      diny_ptr++;
    }
  }
}

template <>
void elementwise_mul<int>(const int* dinx,
                          const int* diny,
                          int* dout,
                          int num) {
  int cnt = num >> 4;
  int remain = num % 16;
#pragma omp parallel for
  for (int i = 0; i < cnt; ++i) {
    const int* dinx_ptr = dinx + (i << 4);
    const int* diny_ptr = diny + (i << 4);
    int* dout_ptr = dout + (i << 4);

    int32x4_t dinx0 = vld1q_s32(dinx_ptr);
    int32x4_t dinx1 = vld1q_s32(dinx_ptr + 4);
    int32x4_t dinx2 = vld1q_s32(dinx_ptr + 8);
    int32x4_t dinx3 = vld1q_s32(dinx_ptr + 12);

    int32x4_t diny0 = vld1q_s32(diny_ptr);
    int32x4_t diny1 = vld1q_s32(diny_ptr + 4);
    int32x4_t diny2 = vld1q_s32(diny_ptr + 8);
    int32x4_t diny3 = vld1q_s32(diny_ptr + 12);

    dinx0 = vmulq_s32(dinx0, diny0);
    dinx1 = vmulq_s32(dinx1, diny1);
    dinx2 = vmulq_s32(dinx2, diny2);
    dinx3 = vmulq_s32(dinx3, diny3);

    vst1q_s32(dout_ptr, dinx0);
    vst1q_s32(dout_ptr + 4, dinx1);
    vst1q_s32(dout_ptr + 8, dinx2);
    vst1q_s32(dout_ptr + 12, dinx3);
  }
  if (remain > 0) {
    const int* dinx_ptr = dinx + (cnt << 4);
    const int* diny_ptr = diny + (cnt << 4);
    int* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      *dout_ptr = *dinx_ptr * *diny_ptr;
      dout_ptr++;
      dinx_ptr++;
      diny_ptr++;
    }
  }
}

template <>
void elementwise_mul<int64_t>(const int64_t* dinx,
                              const int64_t* diny,
                              int64_t* dout,
                              int num) {
  for (int i = 0; i < num; i++) {
    dout[i] = dinx[i] * diny[i];
  }
}

template <>
void elementwise_mul_relu<float>(const float* dinx,
                                 const float* diny,
                                 float* dout,
                                 int num) {
  int cnt = num >> 4;
  int remain = num % 16;
  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for
  for (int i = 0; i < cnt; ++i) {
    const float* dinx_ptr = dinx + (i << 4);
    const float* diny_ptr = diny + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t dinx0 = vld1q_f32(dinx_ptr);
    float32x4_t dinx1 = vld1q_f32(dinx_ptr + 4);
    float32x4_t dinx2 = vld1q_f32(dinx_ptr + 8);
    float32x4_t dinx3 = vld1q_f32(dinx_ptr + 12);

    float32x4_t diny0 = vld1q_f32(diny_ptr);
    float32x4_t diny1 = vld1q_f32(diny_ptr + 4);
    float32x4_t diny2 = vld1q_f32(diny_ptr + 8);
    float32x4_t diny3 = vld1q_f32(diny_ptr + 12);

    dinx0 = vmulq_f32(dinx0, diny0);
    dinx1 = vmulq_f32(dinx1, diny1);
    dinx2 = vmulq_f32(dinx2, diny2);
    dinx3 = vmulq_f32(dinx3, diny3);

    // relu
    dinx0 = vmaxq_f32(dinx0, vzero);
    dinx1 = vmaxq_f32(dinx1, vzero);
    dinx2 = vmaxq_f32(dinx2, vzero);
    dinx3 = vmaxq_f32(dinx3, vzero);

    vst1q_f32(dout_ptr, dinx0);
    vst1q_f32(dout_ptr + 4, dinx1);
    vst1q_f32(dout_ptr + 8, dinx2);
    vst1q_f32(dout_ptr + 12, dinx3);
  }
  if (remain > 0) {
    const float* dinx_ptr = dinx + (cnt << 4);
    const float* diny_ptr = diny + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      float tmp = *dinx_ptr * *diny_ptr;
      *dout_ptr = tmp > 0.f ? tmp : 0.f;
      dout_ptr++;
      dinx_ptr++;
      diny_ptr++;
    }
  }
}

template <>
void elementwise_mul_relu<int64_t>(const int64_t* dinx,
                                   const int64_t* diny,
                                   int64_t* dout,
                                   int num) {
  for (int i = 0; i < num; i++) {
    int64_t tmp = dinx[i] * diny[i];
    dout[i] = tmp > 0 ? tmp : 0;
  }
}

template <>
void elementwise_mul_broadcast<float>(const float* dinx,
                                      const float* diny,
                                      float* dout,
                                      int batch,
                                      int channels,
                                      int num) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const float* din_ptr = dinx + offset;
      const float diny_data = diny[j];
      float* dout_ptr = dout + offset;

      int cnt = num >> 4;
      int remain = num % 16;
      float32x4_t rb = vdupq_n_f32(diny_data);
      for (int k = 0; k < cnt; ++k) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        float32x4_t din2 = vld1q_f32(din_ptr + 8);
        float32x4_t din3 = vld1q_f32(din_ptr + 12);

        din0 = vmulq_f32(din0, rb);
        din1 = vmulq_f32(din1, rb);
        din2 = vmulq_f32(din2, rb);
        din3 = vmulq_f32(din3, rb);

        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        vst1q_f32(dout_ptr + 8, din2);
        vst1q_f32(dout_ptr + 12, din3);

        din_ptr += 16;
        dout_ptr += 16;
      }
      if (remain >= 8) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        din0 = vmulq_f32(din0, rb);
        din1 = vmulq_f32(din1, rb);
        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        din_ptr += 8;
        dout_ptr += 8;
        remain -= 8;
      }
      if (remain >= 4) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        din0 = vmulq_f32(din0, rb);
        vst1q_f32(dout_ptr, din0);
        din_ptr += 4;
        dout_ptr += 4;
        remain -= 4;
      }
      if (remain > 0) {
        for (int p = 0; p < remain; ++p) {
          *dout_ptr = *din_ptr * diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  }
}

template <>
void elementwise_mul_broadcast<int>(const int* dinx,
                                    const int* diny,
                                    int* dout,
                                    int batch,
                                    int channels,
                                    int num) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const int* din_ptr = dinx + offset;
      const int diny_data = diny[j];
      int* dout_ptr = dout + offset;

      int cnt = num >> 4;
      int remain = num % 16;
      int32x4_t rb = vdupq_n_s32(diny_data);
      for (int k = 0; k < cnt; ++k) {
        int32x4_t din0 = vld1q_s32(din_ptr);
        int32x4_t din1 = vld1q_s32(din_ptr + 4);
        int32x4_t din2 = vld1q_s32(din_ptr + 8);
        int32x4_t din3 = vld1q_s32(din_ptr + 12);

        din0 = vmulq_s32(din0, rb);
        din1 = vmulq_s32(din1, rb);
        din2 = vmulq_s32(din2, rb);
        din3 = vmulq_s32(din3, rb);

        vst1q_s32(dout_ptr, din0);
        vst1q_s32(dout_ptr + 4, din1);
        vst1q_s32(dout_ptr + 8, din2);
        vst1q_s32(dout_ptr + 12, din3);

        din_ptr += 16;
        dout_ptr += 16;
      }
      if (remain >= 8) {
        int32x4_t din0 = vld1q_s32(din_ptr);
        int32x4_t din1 = vld1q_s32(din_ptr + 4);
        din0 = vmulq_s32(din0, rb);
        din1 = vmulq_s32(din1, rb);
        vst1q_s32(dout_ptr, din0);
        vst1q_s32(dout_ptr + 4, din1);
        din_ptr += 8;
        dout_ptr += 8;
        remain -= 8;
      }
      if (remain >= 4) {
        int32x4_t din0 = vld1q_s32(din_ptr);
        din0 = vmulq_s32(din0, rb);
        vst1q_s32(dout_ptr, din0);
        din_ptr += 4;
        dout_ptr += 4;
        remain -= 4;
      }
      if (remain > 0) {
        for (int p = 0; p < remain; ++p) {
          *dout_ptr = *din_ptr * diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  }
}

template <>
void elementwise_mul_broadcast<int64_t>(const int64_t* dinx,
                                        const int64_t* diny,
                                        int64_t* dout,
                                        int batch,
                                        int channels,
                                        int num) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const int64_t* dinx_ptr = dinx + offset;
      const int64_t diny_data = diny[j];
      int64_t* dout_ptr = dout + offset;
      for (int k = 0; k < num; ++k) {
        *dout_ptr = *dinx_ptr * diny_data;
        dout_ptr++;
        dinx_ptr++;
      }
    }
  }
}

template <>
void elementwise_mul_relu_broadcast<float>(const float* dinx,
                                           const float* diny,
                                           float* dout,
                                           int batch,
                                           int channels,
                                           int num) {
  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const float* din_ptr = dinx + offset;
      const float diny_data = diny[j];
      float* dout_ptr = dout + offset;

      int cnt = num >> 4;
      int remain = num % 16;
      float32x4_t rb = vdupq_n_f32(diny_data);
      for (int k = 0; k < cnt; ++k) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        float32x4_t din2 = vld1q_f32(din_ptr + 8);
        float32x4_t din3 = vld1q_f32(din_ptr + 12);

        din0 = vmulq_f32(din0, rb);
        din1 = vmulq_f32(din1, rb);
        din2 = vmulq_f32(din2, rb);
        din3 = vmulq_f32(din3, rb);

        // relu
        din0 = vmaxq_f32(din0, vzero);
        din1 = vmaxq_f32(din1, vzero);
        din2 = vmaxq_f32(din2, vzero);
        din3 = vmaxq_f32(din3, vzero);

        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        vst1q_f32(dout_ptr + 8, din2);
        vst1q_f32(dout_ptr + 12, din3);
        din_ptr += 16;
        dout_ptr += 16;
      }
      if (remain >= 8) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        din0 = vmulq_f32(din0, rb);
        din1 = vmulq_f32(din1, rb);
        // relu
        din0 = vmaxq_f32(din0, vzero);
        din1 = vmaxq_f32(din1, vzero);
        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        din_ptr += 8;
        dout_ptr += 8;
        remain -= 8;
      }
      if (remain >= 4) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        din0 = vmulq_f32(din0, rb);
        // relu
        din0 = vmaxq_f32(din0, vzero);
        vst1q_f32(dout_ptr, din0);
        din_ptr += 4;
        dout_ptr += 4;
        remain -= 4;
      }
      if (remain > 0) {
        for (int p = 0; p < remain; ++p) {
          float tmp = *din_ptr * diny_data;
          *dout_ptr = tmp > 0.f ? tmp : 0.f;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  }
}

template <>
void elementwise_mul_relu_broadcast<int64_t>(const int64_t* dinx,
                                             const int64_t* diny,
                                             int64_t* dout,
                                             int batch,
                                             int channels,
                                             int num) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const int64_t* dinx_ptr = dinx + offset;
      const int64_t diny_data = diny[j];
      int64_t* dout_ptr = dout + offset;
      for (int k = 0; k < num; ++k) {
        int64_t tmp = *dinx_ptr * diny_data;
        *dout_ptr = tmp > 0 ? tmp : 0;
        dout_ptr++;
        dinx_ptr++;
      }
    }
  }
}

template <>
void elementwise_max<float>(const float* dinx,
                            const float* diny,
                            float* dout,
                            int num) {
  int cnt = num >> 4;
  int remain = num % 16;
#pragma omp parallel for
  for (int i = 0; i < cnt; ++i) {
    const float* dinx_ptr = dinx + (i << 4);
    const float* diny_ptr = diny + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t dinx0 = vld1q_f32(dinx_ptr);
    float32x4_t dinx1 = vld1q_f32(dinx_ptr + 4);
    float32x4_t dinx2 = vld1q_f32(dinx_ptr + 8);
    float32x4_t dinx3 = vld1q_f32(dinx_ptr + 12);

    float32x4_t diny0 = vld1q_f32(diny_ptr);
    float32x4_t diny1 = vld1q_f32(diny_ptr + 4);
    float32x4_t diny2 = vld1q_f32(diny_ptr + 8);
    float32x4_t diny3 = vld1q_f32(diny_ptr + 12);

    dinx0 = vmaxq_f32(dinx0, diny0);
    dinx1 = vmaxq_f32(dinx1, diny1);
    dinx2 = vmaxq_f32(dinx2, diny2);
    dinx3 = vmaxq_f32(dinx3, diny3);

    vst1q_f32(dout_ptr, dinx0);
    vst1q_f32(dout_ptr + 4, dinx1);
    vst1q_f32(dout_ptr + 8, dinx2);
    vst1q_f32(dout_ptr + 12, dinx3);
  }
  if (remain > 0) {
    const float* dinx_ptr = dinx + (cnt << 4);
    const float* diny_ptr = diny + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; ++i) {
      *(dout_ptr++) = std::max(*(dinx_ptr++), *(diny_ptr++));
    }
  }
}

template <>
void elementwise_max_relu<float>(const float* dinx,
                                 const float* diny,
                                 float* dout,
                                 int num) {
  int cnt = num >> 4;
  int remain = num % 16;
  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for
  for (int i = 0; i < cnt; ++i) {
    const float* dinx_ptr = dinx + (i << 4);
    const float* diny_ptr = diny + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t dinx0 = vld1q_f32(dinx_ptr);
    float32x4_t dinx1 = vld1q_f32(dinx_ptr + 4);
    float32x4_t dinx2 = vld1q_f32(dinx_ptr + 8);
    float32x4_t dinx3 = vld1q_f32(dinx_ptr + 12);

    float32x4_t diny0 = vld1q_f32(diny_ptr);
    float32x4_t diny1 = vld1q_f32(diny_ptr + 4);
    float32x4_t diny2 = vld1q_f32(diny_ptr + 8);
    float32x4_t diny3 = vld1q_f32(diny_ptr + 12);

    dinx0 = vmaxq_f32(dinx0, diny0);
    dinx1 = vmaxq_f32(dinx1, diny1);
    dinx2 = vmaxq_f32(dinx2, diny2);
    dinx3 = vmaxq_f32(dinx3, diny3);

    // relu
    dinx0 = vmaxq_f32(dinx0, vzero);
    dinx1 = vmaxq_f32(dinx1, vzero);
    dinx2 = vmaxq_f32(dinx2, vzero);
    dinx3 = vmaxq_f32(dinx3, vzero);

    vst1q_f32(dout_ptr, dinx0);
    vst1q_f32(dout_ptr + 4, dinx1);
    vst1q_f32(dout_ptr + 8, dinx2);
    vst1q_f32(dout_ptr + 12, dinx3);
  }
  if (remain > 0) {
    const float* dinx_ptr = dinx + (cnt << 4);
    const float* diny_ptr = diny + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; ++i) {
      float tmp = std::max(*(dinx_ptr++), *(diny_ptr++));
      *(dout_ptr++) = tmp > 0.f ? tmp : 0.f;
    }
  }
}

template <>
void elementwise_max_broadcast<float>(const float* dinx,
                                      const float* diny,
                                      float* dout,
                                      int batch,
                                      int channels,
                                      int num) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const float* din_ptr = dinx + offset;
      const float diny_data = diny[j];
      float* dout_ptr = dout + offset;

      int cnt = num >> 4;
      int remain = num % 16;
      float32x4_t rb = vdupq_n_f32(diny_data);
      for (int k = 0; k < cnt; ++k) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        float32x4_t din2 = vld1q_f32(din_ptr + 8);
        float32x4_t din3 = vld1q_f32(din_ptr + 12);

        din0 = vmaxq_f32(din0, rb);
        din1 = vmaxq_f32(din1, rb);
        din2 = vmaxq_f32(din2, rb);
        din3 = vmaxq_f32(din3, rb);

        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        vst1q_f32(dout_ptr + 8, din2);
        vst1q_f32(dout_ptr + 12, din3);

        din_ptr += 16;
        dout_ptr += 16;
      }
      if (remain >= 8) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        din0 = vmaxq_f32(din0, rb);
        din1 = vmaxq_f32(din1, rb);
        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        din_ptr += 8;
        dout_ptr += 8;
        remain -= 8;
      }
      if (remain >= 4) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        din0 = vmaxq_f32(din0, rb);
        vst1q_f32(dout_ptr, din0);
        din_ptr += 4;
        dout_ptr += 4;
        remain -= 4;
      }
      if (remain > 0) {
        for (int p = 0; p < remain; ++p) {
          *dout_ptr = std::max(*din_ptr, diny_data);
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  }
}

template <>
void elementwise_max_relu_broadcast<float>(const float* dinx,
                                           const float* diny,
                                           float* dout,
                                           int batch,
                                           int channels,
                                           int num) {
  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const float* din_ptr = dinx + offset;
      const float diny_data = diny[j];
      float* dout_ptr = dout + offset;

      int cnt = num >> 4;
      int remain = num % 16;
      float32x4_t rb = vdupq_n_f32(diny_data);
      for (int k = 0; k < cnt; ++k) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        float32x4_t din2 = vld1q_f32(din_ptr + 8);
        float32x4_t din3 = vld1q_f32(din_ptr + 12);

        din0 = vmaxq_f32(din0, rb);
        din1 = vmaxq_f32(din1, rb);
        din2 = vmaxq_f32(din2, rb);
        din3 = vmaxq_f32(din3, rb);

        // relu
        din0 = vmaxq_f32(din0, vzero);
        din1 = vmaxq_f32(din1, vzero);
        din2 = vmaxq_f32(din2, vzero);
        din3 = vmaxq_f32(din3, vzero);

        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        vst1q_f32(dout_ptr + 8, din2);
        vst1q_f32(dout_ptr + 12, din3);
        din_ptr += 16;
        dout_ptr += 16;
      }
      if (remain >= 8) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        din0 = vmaxq_f32(din0, rb);
        din1 = vmaxq_f32(din1, rb);
        // relu
        din0 = vmaxq_f32(din0, vzero);
        din1 = vmaxq_f32(din1, vzero);
        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        din_ptr += 8;
        dout_ptr += 8;
        remain -= 8;
      }
      if (remain >= 4) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        din0 = vmaxq_f32(din0, rb);
        // relu
        din0 = vmaxq_f32(din0, vzero);
        vst1q_f32(dout_ptr, din0);
        din_ptr += 4;
        dout_ptr += 4;
        remain -= 4;
      }
      if (remain > 0) {
        for (int p = 0; p < remain; ++p) {
          float tmp = std::max(*din_ptr, diny_data);
          *dout_ptr = tmp > 0.f ? tmp : 0.f;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  }
}

template <>
void elementwise_div<int64_t>(const int64_t* dinx,
                              const int64_t* diny,
                              int64_t* dout,
                              int num) {
  for (int i = 0; i < num; i++) {
    *dout = *dinx / *diny;
    dout++;
    dinx++;
    diny++;
  }
}

template <>
void elementwise_div<float>(const float* dinx,
                            const float* diny,
                            float* dout,
                            int num) {
  int cnt = num >> 4;
  int remain = num % 16;
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const float* dinx_ptr = dinx + (i << 4);
    const float* diny_ptr = diny + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t dinx0 = vld1q_f32(dinx_ptr);
    float32x4_t dinx1 = vld1q_f32(dinx_ptr + 4);
    float32x4_t dinx2 = vld1q_f32(dinx_ptr + 8);
    float32x4_t dinx3 = vld1q_f32(dinx_ptr + 12);

    float32x4_t diny0 = vld1q_f32(diny_ptr);
    float32x4_t diny1 = vld1q_f32(diny_ptr + 4);
    float32x4_t diny2 = vld1q_f32(diny_ptr + 8);
    float32x4_t diny3 = vld1q_f32(diny_ptr + 12);

#ifdef __aarch64__
    dinx0 = vdivq_f32(dinx0, diny0);
    dinx1 = vdivq_f32(dinx1, diny1);
    dinx2 = vdivq_f32(dinx2, diny2);
    dinx3 = vdivq_f32(dinx3, diny3);
#else
    dinx0 = div_ps(dinx0, diny0);
    dinx1 = div_ps(dinx1, diny1);
    dinx2 = div_ps(dinx2, diny2);
    dinx3 = div_ps(dinx3, diny3);
#endif
    vst1q_f32(dout_ptr, dinx0);
    vst1q_f32(dout_ptr + 4, dinx1);
    vst1q_f32(dout_ptr + 8, dinx2);
    vst1q_f32(dout_ptr + 12, dinx3);
  }
  if (remain > 0) {
    const float* dinx_ptr = dinx + (cnt << 4);
    const float* diny_ptr = diny + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      *dout_ptr = *dinx_ptr / *diny_ptr;
      dout_ptr++;
      dinx_ptr++;
      diny_ptr++;
    }
  }
}

template <>
void elementwise_div_broadcast<int64_t>(const int64_t* dinx,
                                        const int64_t* diny,
                                        int64_t* dout,
                                        int batch,
                                        int channels,
                                        int num) {
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const int64_t* din_ptr = dinx + offset;
      const int64_t diny_data = diny[j];
      int64_t* dout_ptr = dout + offset;
      for (int p = 0; p < num; p++) {
        *dout_ptr = *din_ptr / diny_data;
        dout_ptr++;
        din_ptr++;
      }
    }
  }
}

template <>
void elementwise_div_broadcast<float>(const float* dinx,
                                      const float* diny,
                                      float* dout,
                                      int batch,
                                      int channels,
                                      int num) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const float* din_ptr = dinx + offset;
      const float diny_data = diny[j];
      float* dout_ptr = dout + offset;

      int cnt = num >> 4;
      int remain = num % 16;
      float32x4_t rb = vdupq_n_f32(diny_data);
      for (int k = 0; k < cnt; ++k) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        float32x4_t din2 = vld1q_f32(din_ptr + 8);
        float32x4_t din3 = vld1q_f32(din_ptr + 12);

#ifdef __aarch64__
        din0 = vdivq_f32(din0, rb);
        din1 = vdivq_f32(din1, rb);
        din2 = vdivq_f32(din2, rb);
        din3 = vdivq_f32(din3, rb);
#else
        din0 = div_ps(din0, rb);
        din1 = div_ps(din1, rb);
        din2 = div_ps(din2, rb);
        din3 = div_ps(din3, rb);
#endif

        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        vst1q_f32(dout_ptr + 8, din2);
        vst1q_f32(dout_ptr + 12, din3);
        din_ptr += 16;
        dout_ptr += 16;
      }
      if (remain >= 8) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
#ifdef __aarch64__
        din0 = vdivq_f32(din0, rb);
        din1 = vdivq_f32(din1, rb);
#else
        din0 = div_ps(din0, rb);
        din1 = div_ps(din1, rb);
#endif
        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        din_ptr += 8;
        dout_ptr += 8;
        remain -= 8;
      }
      if (remain >= 4) {
        float32x4_t din0 = vld1q_f32(din_ptr);
#ifdef __aarch64__
        din0 = vdivq_f32(din0, rb);
#else
        din0 = div_ps(din0, rb);
#endif
        vst1q_f32(dout_ptr, din0);
        din_ptr += 4;
        dout_ptr += 4;
        remain -= 4;
      }
      if (remain > 0) {
        for (int p = 0; p < remain; p++) {
          *dout_ptr = *din_ptr / diny_data;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  }
}

template <>
void elementwise_div_relu<float>(const float* dinx,
                                 const float* diny,
                                 float* dout,
                                 int num) {
  int cnt = num >> 4;
  int remain = num % 16;
  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for
  for (int i = 0; i < cnt; ++i) {
    const float* dinx_ptr = dinx + (i << 4);
    const float* diny_ptr = diny + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t dinx0 = vld1q_f32(dinx_ptr);
    float32x4_t dinx1 = vld1q_f32(dinx_ptr + 4);
    float32x4_t dinx2 = vld1q_f32(dinx_ptr + 8);
    float32x4_t dinx3 = vld1q_f32(dinx_ptr + 12);

    float32x4_t diny0 = vld1q_f32(diny_ptr);
    float32x4_t diny1 = vld1q_f32(diny_ptr + 4);
    float32x4_t diny2 = vld1q_f32(diny_ptr + 8);
    float32x4_t diny3 = vld1q_f32(diny_ptr + 12);

#ifdef __aarch64__
    dinx0 = vdivq_f32(dinx0, diny0);
    dinx1 = vdivq_f32(dinx1, diny1);
    dinx2 = vdivq_f32(dinx2, diny2);
    dinx3 = vdivq_f32(dinx3, diny3);
#else
    dinx0 = div_ps(dinx0, diny0);
    dinx1 = div_ps(dinx1, diny1);
    dinx2 = div_ps(dinx2, diny2);
    dinx3 = div_ps(dinx3, diny3);
#endif
    // relu
    dinx0 = vmaxq_f32(dinx0, vzero);
    dinx1 = vmaxq_f32(dinx1, vzero);
    dinx2 = vmaxq_f32(dinx2, vzero);
    dinx3 = vmaxq_f32(dinx3, vzero);

    vst1q_f32(dout_ptr, dinx0);
    vst1q_f32(dout_ptr + 4, dinx1);
    vst1q_f32(dout_ptr + 8, dinx2);
    vst1q_f32(dout_ptr + 12, dinx3);
  }
  if (remain > 0) {
    const float* dinx_ptr = dinx + (cnt << 4);
    const float* diny_ptr = diny + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; ++i) {
      float tmp = *dinx_ptr / *diny_ptr;
      *(dout_ptr++) = tmp > 0.f ? tmp : 0.f;
      dinx_ptr++;
      diny_ptr++;
    }
  }
}

template <>
void elementwise_div_relu_broadcast<float>(const float* dinx,
                                           const float* diny,
                                           float* dout,
                                           int batch,
                                           int channels,
                                           int num) {
  float32x4_t vzero = vdupq_n_f32(0.f);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const float* din_ptr = dinx + offset;
      const float diny_data = diny[j];
      float* dout_ptr = dout + offset;

      int cnt = num >> 4;
      int remain = num % 16;
      float32x4_t rb = vdupq_n_f32(diny_data);
      for (int k = 0; k < cnt; ++k) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
        float32x4_t din2 = vld1q_f32(din_ptr + 8);
        float32x4_t din3 = vld1q_f32(din_ptr + 12);

#ifdef __aarch64__
        din0 = vdivq_f32(din0, rb);
        din1 = vdivq_f32(din1, rb);
        din2 = vdivq_f32(din2, rb);
        din3 = vdivq_f32(din3, rb);
#else
        din0 = div_ps(din0, rb);
        din1 = div_ps(din1, rb);
        din2 = div_ps(din2, rb);
        din3 = div_ps(din3, rb);
#endif
        // relu
        din0 = vmaxq_f32(din0, vzero);
        din1 = vmaxq_f32(din1, vzero);
        din2 = vmaxq_f32(din2, vzero);
        din3 = vmaxq_f32(din3, vzero);

        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        vst1q_f32(dout_ptr + 8, din2);
        vst1q_f32(dout_ptr + 12, din3);
        din_ptr += 16;
        dout_ptr += 16;
      }
      if (remain >= 8) {
        float32x4_t din0 = vld1q_f32(din_ptr);
        float32x4_t din1 = vld1q_f32(din_ptr + 4);
#ifdef __aarch64__
        din0 = vdivq_f32(din0, rb);
        din1 = vdivq_f32(din1, rb);
#else
        din0 = div_ps(din0, rb);
        din1 = div_ps(din1, rb);
#endif
        // relu
        din0 = vmaxq_f32(din0, vzero);
        din1 = vmaxq_f32(din1, vzero);
        vst1q_f32(dout_ptr, din0);
        vst1q_f32(dout_ptr + 4, din1);
        din_ptr += 8;
        dout_ptr += 8;
        remain -= 8;
      }
      if (remain >= 4) {
        float32x4_t din0 = vld1q_f32(din_ptr);
#ifdef __aarch64__
        din0 = vdivq_f32(din0, rb);
#else
        din0 = div_ps(din0, rb);
#endif
        // relu
        din0 = vmaxq_f32(din0, vzero);
        vst1q_f32(dout_ptr, din0);
        din_ptr += 4;
        dout_ptr += 4;
        remain -= 4;
      }
      if (remain > 0) {
        for (int p = 0; p < remain; p++) {
          float tmp = *din_ptr / diny_data;
          *dout_ptr = tmp > 0.f ? tmp : 0.f;
          dout_ptr++;
          din_ptr++;
        }
      }
    }
  }
}

template <typename T>
void elementwise_mod_broadcast(
    const T* dinx, const T* diny, T* dout, int batch, int channels, int num) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const T* din_ptr = dinx + offset;
      const T diny_data = diny[j];
      T* dout_ptr = dout + offset;

      int cnt = num >> 2;
      int remain = num % 4;
      for (int k = 0; k < cnt; ++k) {
        register T dinx0 = din_ptr[0];
        register T dinx1 = din_ptr[1];
        register T dinx2 = din_ptr[2];
        register T dinx3 = din_ptr[3];
        dout_ptr[0] = dinx0 % diny_data;
        dout_ptr[1] = dinx1 % diny_data;
        dout_ptr[2] = dinx2 % diny_data;
        dout_ptr[3] = dinx3 % diny_data;
        din_ptr += 4;
        dout_ptr += 4;
      }
      if (remain > 0) {
        for (int p = 0; p < remain; p++) {
          *dout_ptr++ = *din_ptr++ % diny_data;
        }
      }
    }
  }
}

template <typename T>
void elementwise_mod(const T* dinx, const T* diny, T* dout, int num) {
  int cnt = num >> 2;
  int remain = num % 4;
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const T* dinx_ptr = dinx + (i << 2);
    const T* diny_ptr = diny + (i << 2);
    T* dout_ptr = dout + (i << 2);

    register T dinx0 = dinx_ptr[0];
    register T dinx1 = dinx_ptr[1];
    register T dinx2 = dinx_ptr[2];
    register T dinx3 = dinx_ptr[3];

    register T diny0 = diny_ptr[0];
    register T diny1 = diny_ptr[1];
    register T diny2 = diny_ptr[2];
    register T diny3 = diny_ptr[3];

    dout_ptr[0] = dinx0 % diny0;
    dout_ptr[1] = dinx1 % diny1;
    dout_ptr[2] = dinx2 % diny2;
    dout_ptr[3] = dinx3 % diny3;
  }
  if (remain > 0) {
    const T* dinx_ptr = dinx + (cnt << 2);
    const T* diny_ptr = diny + (cnt << 2);
    T* dout_ptr = dout + (cnt << 2);
    for (int i = 0; i < remain; i++) {
      *dout_ptr++ = *dinx_ptr++ % *diny_ptr++;
    }
  }
}

template void elementwise_mod<int64_t>(const int64_t* dinx,
                                       const int64_t* diny,
                                       int64_t* dout,
                                       int num);

template void elementwise_mod_broadcast<int64_t>(const int64_t* dinx,
                                                 const int64_t* diny,
                                                 int64_t* dout,
                                                 int batch,
                                                 int channels,
                                                 int num);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
