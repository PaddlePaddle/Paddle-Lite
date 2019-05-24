/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <algorithm>
#include "framework/tensor.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

/* preprocessing weights
 * input weights: [chout, chin/ group, kh, kw] --> outputs weights: [chout / n,
 * chin/ group, kh, kw, n]
 */
template <typename dtype>
void slidingwindow_transform_weight(const framework::Tensor& weight,
                                    framework::Tensor* output) {
  int chout = weight.dims()[0];
  int chin = weight.dims()[1];
  int kernel_size = weight.dims()[2] * weight.dims()[3];
  const int n = 4;
  int cround = (chout + n - 1) / n * n;
  const dtype* din = weight.data<dtype>();
  dtype* dout = output->mutable_data<dtype>({cround, chin, 3, 3});
  int c_loop = chout / n;
  int chout_round = (chout + n - 1) / n;
  int win_stride = chin * kernel_size;
  int wout_stride = n * win_stride;
  int co = 0;
  for (; co < c_loop; ++co) {
    dtype* dout_c = dout + co * wout_stride;
    const dtype* din_array[n];
    din_array[0] = din + co * wout_stride;
    for (int i = 1; i < n; i++) {
      din_array[i] = din_array[i - 1] + win_stride;
    }
    for (int ci = 0; ci < chin; ++ci) {
      for (int k = 0; k < kernel_size; ++k) {
        for (int i = 0; i < n; i++) {
          *(dout_c++) = *(din_array[i]++);
        }
      }
    }
  }
  // pad final chout
  if (chout_round > c_loop) {
    dtype* dout_c = dout + c_loop * wout_stride;
    const dtype* din_array[n];
    din_array[0] = din + c_loop * wout_stride;
    for (int i = 1; i < n; i++) {
      din_array[i] = din_array[i - 1] + win_stride;
    }
    // deal remain
    int cremain = chout_round * n - chout;
    for (int i = 1; i <= cremain; i++) {
      din_array[n - i] = din_array[0];
    }
    for (int ci = 0; ci < chin; ++ci) {
      for (int k = 0; k < kernel_size; ++k) {
        for (int i = 0; i < n; i++) {
          *(dout_c++) = *(din_array[i]++);
        }
      }
    }
  }
}

/* preprocessing inputs
 * input din: [1, chin, he-hs, we - ws] --> outputs dout: [n, chin, 1, we - ws]
 * n = he - hs
 */
template <typename dtype>
void slidingwindow_prepack_input(const dtype* din, dtype* dout, int cs, int ce,
                                 int hs, int he, int ws, int we, int channel,
                                 int width, int height, dtype* zero_ptr) {
  int n = he - hs;
  int w0 = ws < 0 ? 0 : ws;
  int w1 = we > width ? width : we;

  int size_w = we - ws;
  int size_wc_len = size_w * channel;
  int size_c = width * height;

  int valid_w = w1 - w0;
  size_t valid_w_byte = valid_w * sizeof(dtype);

  dtype* out_array[n];
  out_array[0] = dout;
  for (int i = 1; i < n; i++) {
    out_array[i] = out_array[i - 1] + size_wc_len;
  }

  for (int c = 0; c < channel; ++c) {
    int j = 0;
    // valid height
    for (int i = hs; i < he; i++) {
      // get address
      const dtype* in_array;
      if (i < 0 || i >= height) {
        in_array = zero_ptr;
      } else {
        in_array = din + i * width;
      }

      for (int w = ws; w < w0; ++w) {
        *(out_array[j]++) = 0.f;
      }
      memcpy(out_array[j], in_array, valid_w_byte);
      out_array[j] += valid_w;
      for (int w = w1; w < we; ++w) {
        *(out_array[j]++) = 0.f;
      }
      j++;
    }
    din += size_c;
  }
}

inline void slidingwindow_fill_bias(float* dout, const float* bias, int size) {
  float32x4_t vb = vld1q_f32(bias);
  int cnt = size / 4;
  for (int i = 0; i < cnt; ++i) {
    vst1q_f32(dout, vb);
    dout += 4;
  }
}

void slidingwindow_fill_bias(float* dout, const float* bias, int ch_num,
                             int ch_size);

void slidingwindow_writeout_c1_fp32(const float* din, float* dout, int cs,
                                    int ce, int hs, int he, int ws, int we,
                                    int channel, int height, int width,
                                    bool flag_relu, float* trash_ptr);

void slidingwindow_writeout_c4_fp32(const float* din, float* dout, int cs,
                                    int ce, int hs, int he, int ws, int we,
                                    int channel, int height, int width,
                                    bool flag_relu, float* trash_ptr);
}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
