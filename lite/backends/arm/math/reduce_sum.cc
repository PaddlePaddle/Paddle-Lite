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

#include "lite/backends/arm/math/reduce_sum.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void reduce_sum_n<float>(const float* src,
                         float* dst,
                         int num_in,
                         int channel_in,
                         int height_in,
                         int width_in) {
  int chw_size = channel_in * height_in * width_in;
  if (num_in == 1) {
    memcpy(dst, src, sizeof(float) * chw_size);
  } else {
    int cnt_n = num_in >> 2;
    int remain_n = num_in & 3;
    int cnt_chw = chw_size >> 3;
    int cnt_rem = chw_size & 7;
    int stride = chw_size << 2;
    int stride_c = 0;
    for (int c = 0; c < cnt_chw; c++) {
      float32x4_t vsum0 = vdupq_n_f32(0.f);
      float32x4_t vsum1 = vdupq_n_f32(0.f);
      const float* din_ptr0 = src + stride_c;
      const float* din_ptr1 = din_ptr0 + chw_size;
      const float* din_ptr2 = din_ptr1 + chw_size;
      const float* din_ptr3 = din_ptr2 + chw_size;
      for (int n = 0; n < cnt_n; n++) {
        float32x4_t va0 = vld1q_f32(din_ptr0);
        float32x4_t vb0 = vld1q_f32(din_ptr1);
        float32x4_t va1 = vld1q_f32(din_ptr0 + 4);
        float32x4_t vb1 = vld1q_f32(din_ptr1 + 4);
        float32x4_t vc0 = vld1q_f32(din_ptr2);
        float32x4_t vd0 = vld1q_f32(din_ptr3);
        float32x4_t vs00 = vaddq_f32(va0, vb0);
        float32x4_t vc1 = vld1q_f32(din_ptr2 + 4);
        float32x4_t vs10 = vaddq_f32(va1, vb1);
        float32x4_t vd1 = vld1q_f32(din_ptr3 + 4);
        float32x4_t vs01 = vaddq_f32(vc0, vd0);
        vsum0 = vaddq_f32(vsum0, vs00);
        float32x4_t vs11 = vaddq_f32(vc1, vd1);
        vsum1 = vaddq_f32(vsum1, vs10);
        din_ptr0 += stride;
        din_ptr1 += stride;
        vsum0 = vaddq_f32(vsum0, vs01);
        din_ptr2 += stride;
        din_ptr3 += stride;
        vsum1 = vaddq_f32(vsum1, vs11);
      }
      for (int n = 0; n < remain_n; n++) {
        float32x4_t va0 = vld1q_f32(din_ptr0);
        float32x4_t va1 = vld1q_f32(din_ptr0 + 4);
        vsum0 = vaddq_f32(vsum0, va0);
        din_ptr0 += chw_size;
        vsum1 = vaddq_f32(vsum1, va1);
      }
      vst1q_f32(dst, vsum0);
      dst += 4;
      stride_c += 8;
      vst1q_f32(dst, vsum1);
      dst += 4;
    }
    if (cnt_rem > 3) {
      float32x4_t vsum0 = vdupq_n_f32(0.f);
      const float* din_ptr0 = src + stride_c;
      const float* din_ptr1 = din_ptr0 + chw_size;
      const float* din_ptr2 = din_ptr1 + chw_size;
      const float* din_ptr3 = din_ptr2 + chw_size;
      for (int n = 0; n < cnt_n; n++) {
        float32x4_t va0 = vld1q_f32(din_ptr0);
        float32x4_t vb0 = vld1q_f32(din_ptr1);
        float32x4_t vc0 = vld1q_f32(din_ptr2);
        float32x4_t vd0 = vld1q_f32(din_ptr3);
        float32x4_t vs00 = vaddq_f32(va0, vb0);
        float32x4_t vs01 = vaddq_f32(vc0, vd0);
        vsum0 = vaddq_f32(vsum0, vs00);
        din_ptr0 += stride;
        din_ptr1 += stride;
        vsum0 = vaddq_f32(vsum0, vs01);
        din_ptr2 += stride;
        din_ptr3 += stride;
      }
      for (int n = 0; n < remain_n; n++) {
        float32x4_t va0 = vld1q_f32(din_ptr0);
        din_ptr0 += chw_size;
        vsum0 = vaddq_f32(vsum0, va0);
      }
      stride_c += 4;
      vst1q_f32(dst, vsum0);
      dst += 4;
      cnt_rem -= 4;
    }
    for (int c = 0; c < cnt_rem; c++) {
      const float* din_ptr0 = src + stride_c;
      const float* din_ptr1 = din_ptr0 + chw_size;
      const float* din_ptr2 = din_ptr1 + chw_size;
      const float* din_ptr3 = din_ptr2 + chw_size;
      float sum = 0.0;
      for (int n = 0; n < cnt_n; n++) {
        float tmp0 = din_ptr0[0] + din_ptr1[0];
        float tmp1 = din_ptr2[0] + din_ptr3[0];
        din_ptr0 += stride;
        din_ptr1 += stride;
        sum += tmp0;
        din_ptr2 += stride;
        din_ptr3 += stride;
        sum += tmp1;
      }
      for (int n = 0; n < remain_n; n++) {
        sum += din_ptr0[0];
        din_ptr0 += chw_size;
      }
      stride_c++;
      dst[0] = sum;
      dst++;
    }
  }
}

template <>
void reduce_sum_w<float>(const float* src,
                         float* dst,
                         int num_in,
                         int channel_in,
                         int height_in,
                         int width_in) {
  int nch_size = num_in * channel_in * height_in;
  int cnt_w = width_in >> 3;
  int cnt_n = nch_size >> 2;
  int rem_w = width_in & 7;
  int rem_n = nch_size & 3;
  int stride = 0;
  int stride_n = width_in << 2;
  for (int n = 0; n < cnt_n; n++) {
    const float* din_ptr0 = src + stride;
    const float* din_ptr1 = din_ptr0 + width_in;
    const float* din_ptr2 = din_ptr1 + width_in;
    const float* din_ptr3 = din_ptr2 + width_in;
    float32x4_t vsum = vdupq_n_f32(0.f);
    int tmp = rem_w;
    for (int w = 0; w < cnt_w; w++) {
      float32x4_t va0 = vld1q_f32(din_ptr0);
      float32x4_t va1 = vld1q_f32(din_ptr0 + 4);
      float32x4_t vb0 = vld1q_f32(din_ptr1);
      float32x4_t vb1 = vld1q_f32(din_ptr1 + 4);
      float32x4_t vc0 = vld1q_f32(din_ptr2);
      float32x4_t vc1 = vld1q_f32(din_ptr2 + 4);
      float32x4_t vs0 = vaddq_f32(va0, va1);
      float32x4_t vd0 = vld1q_f32(din_ptr3);
      float32x4_t vs1 = vaddq_f32(vb0, vb1);
      float32x4_t vd1 = vld1q_f32(din_ptr3 + 4);
      float32x4_t vs2 = vaddq_f32(vc0, vc1);
      din_ptr0 += 8;
      float32x4_t vs3 = vaddq_f32(vd0, vd1);
      din_ptr1 += 8;
      float32x4_t vs00 = vpaddq_f32(vs0, vs1);
      din_ptr2 += 8;
      float32x4_t vs01 = vpaddq_f32(vs2, vs3);
      din_ptr3 += 8;
      float32x4_t vs = vpaddq_f32(vs00, vs01);
      vsum = vaddq_f32(vs, vsum);
    }
    if (tmp > 3) {
      float32x4_t va0 = vld1q_f32(din_ptr0);
      float32x4_t vb0 = vld1q_f32(din_ptr1);
      float32x4_t vc0 = vld1q_f32(din_ptr2);
      float32x4_t vd0 = vld1q_f32(din_ptr3);
      din_ptr0 += 4;
      din_ptr1 += 4;
      float32x4_t vs00 = vpaddq_f32(va0, vb0);
      float32x4_t vs01 = vpaddq_f32(vc0, vd0);
      din_ptr2 += 4;
      din_ptr3 += 4;
      float32x4_t vs = vpaddq_f32(vs00, vs01);
      vsum = vaddq_f32(vs, vsum);
      tmp -= 4;
    }
    for (int w = 0; w < tmp; w++) {
      vsum[0] += *din_ptr0++;
      vsum[1] += *din_ptr1++;
      vsum[2] += *din_ptr2++;
      vsum[3] += *din_ptr3++;
    }
    stride += stride_n;
    vst1q_f32(dst, vsum);
    dst += 4;
  }
  if (rem_n > 1) {
    const float* din_ptr0 = src + stride;
    const float* din_ptr1 = din_ptr0 + width_in;
    float32x4_t vsum = vdupq_n_f32(0.f);
    for (int w = 0; w < cnt_w; w++) {
      float32x4_t va0 = vld1q_f32(din_ptr0);
      din_ptr0 += 4;
      float32x4_t vb0 = vld1q_f32(din_ptr1);
      din_ptr1 += 4;
      float32x4_t va1 = vld1q_f32(din_ptr0);
      float32x4_t vb1 = vld1q_f32(din_ptr1);
      float32x4_t vs0 = vpaddq_f32(va0, vb0);
      din_ptr0 += 4;
      float32x4_t vs1 = vpaddq_f32(va1, vb1);
      din_ptr1 += 4;
      float32x4_t vs00 = vpaddq_f32(vs0, vs1);
      vsum = vaddq_f32(vs00, vsum);
    }
    int tmp = rem_w;
    if (tmp > 3) {
      float32x4_t va0 = vld1q_f32(din_ptr0);
      float32x4_t vb0 = vld1q_f32(din_ptr1);
      din_ptr0 += 4;
      din_ptr1 += 4;
      float32x4_t vs00 = vpaddq_f32(va0, vb0);
      tmp -= 4;
      vsum[0] += vs00[0];
      vsum[2] += vs00[1];
      vsum[1] += vs00[2];
      vsum[3] += vs00[3];
    }
    vsum[0] += vsum[2];
    vsum[1] += vsum[3];
    for (int w = 0; w < tmp; w++) {
      vsum[0] += *din_ptr0++;
      vsum[1] += *din_ptr1++;
    }
    stride += width_in;
    *dst++ = vsum[0];
    stride += width_in;
    *dst++ = vsum[1];
    rem_n -= 2;
  }
  for (int n = 0; n < rem_n; n++) {
    const float* din_ptr0 = src + stride;
    float32x4_t vsum = vdupq_n_f32(0.f);
    for (int w = 0; w < cnt_w; w++) {
      float32x4_t va0 = vld1q_f32(din_ptr0);
      float32x4_t va1 = vld1q_f32(din_ptr0 + 4);
      float32x4_t vs0 = vaddq_f32(va0, va1);
      din_ptr0 += 8;
      vsum = vaddq_f32(vs0, vsum);
    }
    if (rem_w > 3) {
      float32x4_t va0 = vld1q_f32(din_ptr0);
      din_ptr0 += 4;
      vsum = vaddq_f32(vsum, va0);
      rem_w -= 4;
    }
    vsum[1] += vsum[2];
    for (int w = 0; w < rem_w; w++) {
      vsum[0] += *din_ptr0++;
    }
    vsum[1] += vsum[3];
    vsum[0] += vsum[1];
    *dst++ = vsum[0];
  }
}

template <>
void reduce_sum_all<float>(const float* src, float* dst, int all_size) {
  int cnt_n = all_size >> 4;
  int rem_n = all_size & 15;
  int cnt_rem = rem_n >> 2;
  int rem_rem = rem_n & 3;
  float32x4_t vsum = vdupq_n_f32(0.f);
  for (int n = 0; n < cnt_n; n++) {
    float32x4_t va0 = vld1q_f32(src);
    float32x4_t va1 = vld1q_f32(src + 4);
    float32x4_t va2 = vld1q_f32(src + 8);
    float32x4_t va3 = vld1q_f32(src + 12);
    src += 16;
    float32x4_t vs0 = vaddq_f32(va0, va1);
    float32x4_t vs1 = vaddq_f32(va2, va3);
    float32x4_t vs = vpaddq_f32(vs0, vs1);
    vsum = vaddq_f32(vsum, vs);
  }
  for (int n = 0; n < cnt_rem; n++) {
    float32x4_t va0 = vld1q_f32(src);
    src += 4;
    vsum = vaddq_f32(vsum, va0);
  }
  vsum[1] += vsum[2];
  for (int n = 0; n < rem_rem; n++) {
    vsum[0] += *src++;
  }
  vsum[1] += vsum[3];
  vsum[0] += vsum[1];
  dst[0] = vsum[0];
}

template <>
void reduce_sum_n<int32_t>(const int32_t* src,
                           int32_t* dst,
                           int num_in,
                           int channel_in,
                           int height_in,
                           int width_in) {
  int chw_size = channel_in * height_in * width_in;
  if (num_in == 1) {
    memcpy(dst, src, sizeof(int32_t) * chw_size);
  } else {
    int cnt_n = num_in >> 2;
    int remain_n = num_in & 3;
    int cnt_chw = chw_size >> 3;
    int cnt_rem = chw_size & 7;
    int stride = chw_size << 2;
    int stride_c = 0;
    for (int c = 0; c < cnt_chw; c++) {
      int32x4_t vsum0 = vdupq_n_s32(0.f);
      int32x4_t vsum1 = vdupq_n_s32(0.f);
      const int32_t* din_ptr0 = src + stride_c;
      const int32_t* din_ptr1 = din_ptr0 + chw_size;
      const int32_t* din_ptr2 = din_ptr1 + chw_size;
      const int32_t* din_ptr3 = din_ptr2 + chw_size;
      for (int n = 0; n < cnt_n; n++) {
        int32x4_t va0 = vld1q_s32(din_ptr0);
        int32x4_t vb0 = vld1q_s32(din_ptr1);
        int32x4_t va1 = vld1q_s32(din_ptr0 + 4);
        int32x4_t vb1 = vld1q_s32(din_ptr1 + 4);
        int32x4_t vc0 = vld1q_s32(din_ptr2);
        int32x4_t vd0 = vld1q_s32(din_ptr3);
        int32x4_t vs00 = vaddq_s32(va0, vb0);
        int32x4_t vc1 = vld1q_s32(din_ptr2 + 4);
        int32x4_t vs10 = vaddq_s32(va1, vb1);
        int32x4_t vd1 = vld1q_s32(din_ptr3 + 4);
        int32x4_t vs01 = vaddq_s32(vc0, vd0);
        vsum0 = vaddq_s32(vsum0, vs00);
        int32x4_t vs11 = vaddq_s32(vc1, vd1);
        vsum1 = vaddq_s32(vsum1, vs10);
        din_ptr0 += stride;
        din_ptr1 += stride;
        vsum0 = vaddq_s32(vsum0, vs01);
        din_ptr2 += stride;
        din_ptr3 += stride;
        vsum1 = vaddq_s32(vsum1, vs11);
      }
      for (int n = 0; n < remain_n; n++) {
        int32x4_t va0 = vld1q_s32(din_ptr0);
        int32x4_t va1 = vld1q_s32(din_ptr0 + 4);
        vsum0 = vaddq_s32(vsum0, va0);
        din_ptr0 += chw_size;
        vsum1 = vaddq_s32(vsum1, va1);
      }
      vst1q_s32(dst, vsum0);
      dst += 4;
      stride_c += 8;
      vst1q_s32(dst, vsum1);
      dst += 4;
    }
    if (cnt_rem > 3) {
      int32x4_t vsum0 = vdupq_n_s32(0.f);
      const int32_t* din_ptr0 = src + stride_c;
      const int32_t* din_ptr1 = din_ptr0 + chw_size;
      const int32_t* din_ptr2 = din_ptr1 + chw_size;
      const int32_t* din_ptr3 = din_ptr2 + chw_size;
      for (int n = 0; n < cnt_n; n++) {
        int32x4_t va0 = vld1q_s32(din_ptr0);
        int32x4_t vb0 = vld1q_s32(din_ptr1);
        int32x4_t vc0 = vld1q_s32(din_ptr2);
        int32x4_t vd0 = vld1q_s32(din_ptr3);
        int32x4_t vs00 = vaddq_s32(va0, vb0);
        int32x4_t vs01 = vaddq_s32(vc0, vd0);
        vsum0 = vaddq_s32(vsum0, vs00);
        din_ptr0 += stride;
        din_ptr1 += stride;
        vsum0 = vaddq_s32(vsum0, vs01);
        din_ptr2 += stride;
        din_ptr3 += stride;
      }
      for (int n = 0; n < remain_n; n++) {
        int32x4_t va0 = vld1q_s32(din_ptr0);
        din_ptr0 += chw_size;
        vsum0 = vaddq_s32(vsum0, va0);
      }
      stride_c += 4;
      vst1q_s32(dst, vsum0);
      dst += 4;
      cnt_rem -= 4;
    }
    for (int c = 0; c < cnt_rem; c++) {
      const int32_t* din_ptr0 = src + stride_c;
      const int32_t* din_ptr1 = din_ptr0 + chw_size;
      const int32_t* din_ptr2 = din_ptr1 + chw_size;
      const int32_t* din_ptr3 = din_ptr2 + chw_size;
      int32_t sum = 0.0;
      for (int n = 0; n < cnt_n; n++) {
        int32_t tmp0 = din_ptr0[0] + din_ptr1[0];
        int32_t tmp1 = din_ptr2[0] + din_ptr3[0];
        din_ptr0 += stride;
        din_ptr1 += stride;
        sum += tmp0;
        din_ptr2 += stride;
        din_ptr3 += stride;
        sum += tmp1;
      }
      for (int n = 0; n < remain_n; n++) {
        sum += din_ptr0[0];
        din_ptr0 += chw_size;
      }
      stride_c++;
      dst[0] = sum;
      dst++;
    }
  }
}

template <>
void reduce_sum_w<int32_t>(const int32_t* src,
                           int32_t* dst,
                           int num_in,
                           int channel_in,
                           int height_in,
                           int width_in) {
  int nch_size = num_in * channel_in * height_in;
  int cnt_w = width_in >> 3;
  int cnt_n = nch_size >> 2;
  int rem_w = width_in & 7;
  int rem_n = nch_size & 3;
  int stride = 0;
  int stride_n = width_in << 2;
  for (int n = 0; n < cnt_n; n++) {
    const int32_t* din_ptr0 = src + stride;
    const int32_t* din_ptr1 = din_ptr0 + width_in;
    const int32_t* din_ptr2 = din_ptr1 + width_in;
    const int32_t* din_ptr3 = din_ptr2 + width_in;
    int32x4_t vsum = vdupq_n_s32(0.f);
    int tmp = rem_w;
    for (int w = 0; w < cnt_w; w++) {
      int32x4_t va0 = vld1q_s32(din_ptr0);
      int32x4_t va1 = vld1q_s32(din_ptr0 + 4);
      int32x4_t vb0 = vld1q_s32(din_ptr1);
      int32x4_t vb1 = vld1q_s32(din_ptr1 + 4);
      int32x4_t vc0 = vld1q_s32(din_ptr2);
      int32x4_t vc1 = vld1q_s32(din_ptr2 + 4);
      int32x4_t vs0 = vaddq_s32(va0, va1);
      int32x4_t vd0 = vld1q_s32(din_ptr3);
      int32x4_t vs1 = vaddq_s32(vb0, vb1);
      int32x4_t vd1 = vld1q_s32(din_ptr3 + 4);
      int32x4_t vs2 = vaddq_s32(vc0, vc1);
      din_ptr0 += 8;
      int32x4_t vs3 = vaddq_s32(vd0, vd1);
      din_ptr1 += 8;
      din_ptr2 += 8;
#ifdef __aarch64__
      int32x4_t vs00 = vpaddq_s32(vs0, vs1);
      int32x4_t vs01 = vpaddq_s32(vs2, vs3);
      din_ptr3 += 8;
      int32x4_t vs = vpaddq_s32(vs00, vs01);
      vsum = vaddq_s32(vs, vsum);
#else
      int32x4_t vs;
      vs[0] = vs0[0] + vs0[1] + vs0[2] + vs0[3];
      vs[1] = vs1[0] + vs1[1] + vs1[2] + vs1[3];
      vs[2] = vs2[0] + vs2[1] + vs2[2] + vs2[3];
      vs[3] = vs3[0] + vs3[1] + vs3[2] + vs3[3];
      din_ptr3 += 8;
      vsum = vaddq_s32(vs, vsum);
#endif
    }
    if (tmp > 3) {
      int32x4_t va0 = vld1q_s32(din_ptr0);
      int32x4_t vb0 = vld1q_s32(din_ptr1);
      int32x4_t vc0 = vld1q_s32(din_ptr2);
      int32x4_t vd0 = vld1q_s32(din_ptr3);
      din_ptr0 += 4;
      din_ptr1 += 4;
#ifdef __aarch64__
      int32x4_t vs00 = vpaddq_s32(va0, vb0);
      int32x4_t vs01 = vpaddq_s32(vc0, vd0);
      din_ptr2 += 4;
      din_ptr3 += 4;
      int32x4_t vs = vpaddq_s32(vs00, vs01);
      vsum = vaddq_s32(vs, vsum);
#else
      int32x4_t vs;
      vs[0] = va0[0] + va0[1] + va0[2] + va0[3];
      vs[1] = vb0[0] + vb0[1] + vb0[2] + vb0[3];
      vs[2] = vc0[0] + vc0[1] + vc0[2] + vc0[3];
      vs[3] = vd0[0] + vd0[1] + vd0[2] + vd0[3];
      din_ptr2 += 4;
      din_ptr3 += 4;
      vsum = vaddq_s32(vs, vsum);
#endif
      tmp -= 4;
    }
    for (int w = 0; w < tmp; w++) {
      vsum[0] += *din_ptr0++;
      vsum[1] += *din_ptr1++;
      vsum[2] += *din_ptr2++;
      vsum[3] += *din_ptr3++;
    }
    stride += stride_n;
    vst1q_s32(dst, vsum);
    dst += 4;
  }
  if (rem_n > 1) {
    const int32_t* din_ptr0 = src + stride;
    const int32_t* din_ptr1 = din_ptr0 + width_in;
    int32x4_t vsum = vdupq_n_s32(0.f);
    for (int w = 0; w < cnt_w; w++) {
      int32x4_t va0 = vld1q_s32(din_ptr0);
      din_ptr0 += 4;
      int32x4_t vb0 = vld1q_s32(din_ptr1);
      din_ptr1 += 4;
      int32x4_t va1 = vld1q_s32(din_ptr0);
      int32x4_t vb1 = vld1q_s32(din_ptr1);
#ifdef __aarch64__
      int32x4_t vs0 = vpaddq_s32(va0, vb0);
      din_ptr0 += 4;
      int32x4_t vs1 = vpaddq_s32(va1, vb1);
      din_ptr1 += 4;
      int32x4_t vs00 = vpaddq_s32(vs0, vs1);
      vsum = vaddq_s32(vs00, vsum);
#else
      int32x4_t vs;
      vs[0] = va0[0] + va0[1] + va0[2] + va0[3];
      vs[1] = vb0[0] + vb0[1] + vb0[2] + vb0[3];
      vs[2] = va1[0] + va1[1] + va1[2] + va1[3];
      vs[3] = vb1[0] + vb1[1] + vb1[2] + vb1[3];
      din_ptr0 += 4;
      din_ptr1 += 4;
      vsum = vaddq_s32(vs, vsum);
#endif
    }
    int tmp = rem_w;
    if (tmp > 3) {
      int32x4_t va0 = vld1q_s32(din_ptr0);
      int32x4_t vb0 = vld1q_s32(din_ptr1);
      din_ptr0 += 4;
      din_ptr1 += 4;
      int32x4_t vs00;
#ifdef __aarch64__
      vs00 = vpaddq_s32(va0, vb0);
#else
      vs00[0] = va0[0] + va0[1];
      vs00[1] = va0[2] + va0[3];
      vs00[2] = vb0[0] + vb0[1];
      vs00[3] = vb0[2] + vb0[3];
#endif
      tmp -= 4;
      vsum[0] += vs00[0];
      vsum[2] += vs00[1];
      vsum[1] += vs00[2];
      vsum[3] += vs00[3];
    }
    vsum[0] += vsum[2];
    vsum[1] += vsum[3];
    for (int w = 0; w < tmp; w++) {
      vsum[0] += *din_ptr0++;
      vsum[1] += *din_ptr1++;
    }
    stride += width_in;
    *dst++ = vsum[0];
    stride += width_in;
    *dst++ = vsum[1];
    rem_n -= 2;
  }
  for (int n = 0; n < rem_n; n++) {
    const int32_t* din_ptr0 = src + stride;
    int32x4_t vsum = vdupq_n_s32(0.f);
    for (int w = 0; w < cnt_w; w++) {
      int32x4_t va0 = vld1q_s32(din_ptr0);
      int32x4_t va1 = vld1q_s32(din_ptr0 + 4);
      int32x4_t vs0 = vaddq_s32(va0, va1);
      din_ptr0 += 8;
      vsum = vaddq_s32(vs0, vsum);
    }
    if (rem_w > 3) {
      int32x4_t va0 = vld1q_s32(din_ptr0);
      din_ptr0 += 4;
      vsum = vaddq_s32(vsum, va0);
      rem_w -= 4;
    }
    vsum[1] += vsum[2];
    for (int w = 0; w < rem_w; w++) {
      vsum[0] += *din_ptr0++;
    }
    vsum[1] += vsum[3];
    vsum[0] += vsum[1];
    *dst++ = vsum[0];
  }
}

template <>
void reduce_sum_all<int32_t>(const int32_t* src, int32_t* dst, int all_size) {
  int cnt_n = all_size >> 4;
  int rem_n = all_size & 15;
  int cnt_rem = rem_n >> 2;
  int rem_rem = rem_n & 3;
  int32x4_t vsum = vdupq_n_s32(0.f);
  for (int n = 0; n < cnt_n; n++) {
    int32x4_t va0 = vld1q_s32(src);
    int32x4_t va1 = vld1q_s32(src + 4);
    int32x4_t va2 = vld1q_s32(src + 8);
    int32x4_t va3 = vld1q_s32(src + 12);
    src += 16;
    int32x4_t vs0 = vaddq_s32(va0, va1);
    int32x4_t vs1 = vaddq_s32(va2, va3);
#ifdef __aarch64__
    int32x4_t vs = vpaddq_s32(vs0, vs1);
    vsum = vaddq_s32(vsum, vs);
#else
    vsum[0] = vsum[0] + vs0[0] + vs0[1];
    vsum[1] = vsum[1] + vs0[2] + vs0[3];
    vsum[2] = vsum[2] + vs1[0] + vs1[1];
    vsum[3] = vsum[3] + vs1[2] + vs1[3];
#endif
  }
  for (int n = 0; n < cnt_rem; n++) {
    int32x4_t va0 = vld1q_s32(src);
    src += 4;
    vsum = vaddq_s32(vsum, va0);
  }
  vsum[1] += vsum[2];
  for (int n = 0; n < rem_rem; n++) {
    vsum[0] += *src++;
  }
  vsum[1] += vsum[3];
  vsum[0] += vsum[1];
  dst[0] = vsum[0];
}

template <typename T>
void reduce_sum_c(const T* src,
                  T* dst,
                  int num_in,
                  int channel_in,
                  int height_in,
                  int width_in) {
  int hw_size = height_in * width_in;
  int chw_size = hw_size * channel_in;
  for (int n = 0; n < num_in; ++n) {
    reduce_sum_n(src, dst, channel_in, 1, height_in, width_in);
    src += chw_size;
    dst += hw_size;
  }
}

template <typename T>
void reduce_sum_h(const T* src,
                  T* dst,
                  int num_in,
                  int channel_in,
                  int height_in,
                  int width_in) {
  int nc_size = num_in * channel_in;
  int hw_size = height_in * width_in;
  for (int n = 0; n < nc_size; ++n) {
    reduce_sum_n(src, dst, height_in, 1, 1, width_in);
    src += hw_size;
    dst += width_in;
  }
}

template <typename T>
void reduce_sum_nc(const T* src,
                   T* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in) {
  // reduce nc.
  int num = num_in * channel_in;
  int size = height_in * width_in;
  reduce_sum_n(src, dst, num, size, 1, 1);
}

template <typename T>
void reduce_sum_ch(const T* src,
                   T* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in) {
  int ch_size = channel_in * height_in;
  int chw_size = ch_size * width_in;
  for (int n = 0; n < num_in; n++) {
    reduce_sum_n(src, dst, ch_size, 1, 1, width_in);
    src += chw_size;
    dst += width_in;
  }
}

template <typename T>
void reduce_sum_hw(const T* src,
                   T* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in) {
  int hw_size = height_in * width_in;
  int nc_size = num_in * channel_in;
  reduce_sum_w(src, dst, nc_size, 1, 1, hw_size);
}
template void reduce_sum_c<float>(const float* src,
                                  float* dst,
                                  int num_in,
                                  int channel_in,
                                  int height_in,
                                  int width_in);
template void reduce_sum_c<int32_t>(const int32_t* src,
                                    int32_t* dst,
                                    int num_in,
                                    int channel_in,
                                    int height_in,
                                    int width_in);
template void reduce_sum_h<float>(const float* src,
                                  float* dst,
                                  int num_in,
                                  int channel_in,
                                  int height_in,
                                  int width_in);
template void reduce_sum_h<int32_t>(const int32_t* src,
                                    int32_t* dst,
                                    int num_in,
                                    int channel_in,
                                    int height_in,
                                    int width_in);
template void reduce_sum_nc<float>(const float* src,
                                   float* dst,
                                   int num_in,
                                   int channel_in,
                                   int height_in,
                                   int width_in);
template void reduce_sum_nc<int32_t>(const int32_t* src,
                                     int32_t* dst,
                                     int num_in,
                                     int channel_in,
                                     int height_in,
                                     int width_in);
template void reduce_sum_ch<float>(const float* src,
                                   float* dst,
                                   int num_in,
                                   int channel_in,
                                   int height_in,
                                   int width_in);
template void reduce_sum_ch<int32_t>(const int32_t* src,
                                     int32_t* dst,
                                     int num_in,
                                     int channel_in,
                                     int height_in,
                                     int width_in);
template void reduce_sum_hw<float>(const float* src,
                                   float* dst,
                                   int num_in,
                                   int channel_in,
                                   int height_in,
                                   int width_in);
template void reduce_sum_hw<int32_t>(const int32_t* src,
                                     int32_t* dst,
                                     int num_in,
                                     int channel_in,
                                     int height_in,
                                     int width_in);
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
