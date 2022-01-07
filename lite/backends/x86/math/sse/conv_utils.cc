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

#include "lite/backends/x86/math/sse/conv_utils.h"
#include <algorithm>

namespace paddle {
namespace lite {
namespace x86 {
namespace math {

void packC4_common(const float* din,
                   float* dout,
                   const std::vector<int>& pad,
                   int h_in,
                   int w_in,
                   int channel) {
  int top = pad[0];
  int bottom = pad[1];
  int left = pad[2];
  int right = pad[3];
  int w_out = (w_in + left + right);
  int h_out = (h_in + top + bottom);
  int block_channel = 4;
  const float* din_init = din;
  float* dout_init = dout;

  for (int c = 0; c < channel; c += block_channel) {
    din = din_init + c * h_in * w_in;
    dout = dout_init + c * w_out * h_out;

    memset(dout, 0, top * w_out * block_channel * sizeof(float));
    auto dout_block = dout + top * w_out * block_channel;

    for (int i = 0; i < h_in; i++) {
      float* douth = dout_block + i * w_out * block_channel;
      const float* dinh = din + i * w_in;
      memset(douth, 0, left * block_channel * sizeof(float));
      douth += left * block_channel;
      int kernel_size = h_in * w_in;
      auto dinr0 = dinh;
      auto dinr1 = dinr0 + kernel_size;
      auto dinr2 = dinr1 + kernel_size;
      auto dinr3 = dinr2 + kernel_size;

      int j = 0;
      if (c + 3 < channel) {
        for (; j + 3 < w_in; j += 4) {
          __m128 _row0 = _mm_loadu_ps(dinr0);
          __m128 _row1 = _mm_loadu_ps(dinr1);
          __m128 _row2 = _mm_loadu_ps(dinr2);
          __m128 _row3 = _mm_loadu_ps(dinr3);
          transpose4_ps(_row0, _row1, _row2, _row3);
          _mm_storeu_ps(douth, _row0);
          _mm_storeu_ps(douth + 4, _row1);
          _mm_storeu_ps(douth + 8, _row2);
          _mm_storeu_ps(douth + 12, _row3);
          dinr0 += 4;
          dinr1 += 4;
          dinr2 += 4;
          dinr3 += 4;
          douth += 16;
        }

        for (; j < w_in; j++) {
          douth[0] = *dinr0++;
          douth[1] = *dinr1++;
          douth[2] = *dinr2++;
          douth[3] = *dinr3++;
          douth += 4;
        }
      } else {
        __m128 _row0 = _mm_setzero_ps();
        __m128 _row1 = _mm_setzero_ps();
        __m128 _row2 = _mm_setzero_ps();
        __m128 _row3 = _mm_setzero_ps();
        for (; j + 3 < w_in; j += 4) {
          _row0 = _mm_loadu_ps(dinr0);
          if (channel - c > 1) _row1 = _mm_loadu_ps(dinr1);
          if (channel - c > 2) _row2 = _mm_loadu_ps(dinr2);
          if (channel - c > 3) _row3 = _mm_loadu_ps(dinr3);
          transpose4_ps(_row0, _row1, _row2, _row3);
          _mm_storeu_ps(douth, _row0);
          _mm_storeu_ps(douth + 4, _row1);
          _mm_storeu_ps(douth + 8, _row2);
          _mm_storeu_ps(douth + 12, _row3);
          dinr0 += 4;
          dinr1 += 4;
          dinr2 += 4;
          dinr3 += 4;
          douth += 16;
        }

        for (; j < w_in; j++) {
          douth[0] = *dinr0++;
          douth[1] = channel - c > 1 ? *dinr1++ : 0;
          douth[2] = channel - c > 2 ? *dinr2++ : 0;
          douth[3] = channel - c > 3 ? *dinr3++ : 0;
          douth += 4;
        }
      }
      memset(douth, 0, right * block_channel * sizeof(float));
    }
    memset(dout + (h_in + top) * w_out * block_channel,
           0,
           bottom * w_out * block_channel * sizeof(float));
  }
}

void unpackC4_common(const float* din,
                     float* dout,
                     int size_out_channel,
                     int channel) {
  int block_channel = 4;
  float* dout_init = dout;

  for (int c = 0; c < channel; c += block_channel) {
    dout = dout_init + c * size_out_channel;
    auto doutr0 = dout;
    auto doutr1 = doutr0 + size_out_channel;
    auto doutr2 = doutr1 + size_out_channel;
    auto doutr3 = doutr2 + size_out_channel;
    int j = 0;
    if (c + 3 < channel) {
      for (; j + 3 < size_out_channel; j += 4) {
        __m128 _row0 = _mm_loadu_ps(din);
        __m128 _row1 = _mm_loadu_ps(din + 4);
        __m128 _row2 = _mm_loadu_ps(din + 8);
        __m128 _row3 = _mm_loadu_ps(din + 12);
        transpose4_ps(_row0, _row1, _row2, _row3);
        _mm_storeu_ps(doutr0, _row0);
        _mm_storeu_ps(doutr1, _row1);
        _mm_storeu_ps(doutr2, _row2);
        _mm_storeu_ps(doutr3, _row3);
        doutr0 += 4;
        doutr1 += 4;
        doutr2 += 4;
        doutr3 += 4;
        din += 16;
      }

      for (; j < size_out_channel; j++) {
        *doutr0++ = *din++;
        *doutr1++ = *din++;
        *doutr2++ = *din++;
        *doutr3++ = *din++;
      }
    } else {
      for (; j + 3 < size_out_channel; j += 4) {
        __m128 _row0 = _mm_loadu_ps(din);
        __m128 _row1 = _mm_loadu_ps(din + 4);
        __m128 _row2 = _mm_loadu_ps(din + 8);
        __m128 _row3 = _mm_loadu_ps(din + 12);
        transpose4_ps(_row0, _row1, _row2, _row3);
        _mm_storeu_ps(doutr0, _row0);
        if (channel - c > 1) _mm_storeu_ps(doutr1, _row1);
        if (channel - c > 2) _mm_storeu_ps(doutr2, _row2);
        if (channel - c > 3) _mm_storeu_ps(doutr3, _row3);
        doutr0 += 4;
        doutr1 += 4;
        doutr2 += 4;
        doutr3 += 4;
        din += 16;
      }

      for (; j < size_out_channel; j++) {
        *doutr0++ = *din;
        if (channel - c > 1) *doutr1++ = *(din + 1);
        if (channel - c > 2) *doutr2++ = *(din + 2);
        if (channel - c > 3) *doutr3++ = *(din + 3);
        din += 4;
      }
    }
  }
}
}  // namespace math
}  // namespace x86
}  // namespace lite
}  // namespace paddle
