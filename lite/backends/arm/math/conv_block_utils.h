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

#pragma once
#include <arm_neon.h>
#include <cmath>
#include "lite/backends/arm/math/gemm_s8.h"
#include "lite/backends/arm/math/saturate.h"
#include "lite/backends/arm/math/sgemm.h"
#include "lite/backends/arm/math/type_trans.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

#define LITEMAX(a, b) ((a) > (b) ? (a) : (b))
#define LITEMIN(a, b) ((a) < (b) ? (a) : (b))
#define ROUNDUP(a, b) ((((a) + (b)-1) / (b)) * (b))

template <PrecisionType Ptype>
inline void trans_gemm_weights(const Tensor& tin,
                               Tensor& tout,  // NOLINT
                               int group,
                               ARMContext* ctx);

template <>
inline void trans_gemm_weights<PRECISION(kFloat)>(const Tensor& tin,
                                                  Tensor& tout,  // NOLINT
                                                  int group,
                                                  ARMContext* ctx) {
  CHECK_EQ(tin.dims().size(), 4) << "conv weights dims size must = 4";
  int m = tin.dims()[0] / group;
  int k = tin.dims().count(1, 4);
  int hblock = lite::arm::math::get_hblock(ctx);
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int group_size_round_up = ((m_roundup * k + 15) / 16) * 16;
  float* w_trans_ptr = nullptr;
  tout.Resize({group_size_round_up * group});
  w_trans_ptr = tout.mutable_data<float>();
  const auto* w_data = tin.data<float>();
  for (int g = 0; g < group; ++g) {
    const float* weights_group = w_data + g * m * k;
    float* weights_trans_ptr = w_trans_ptr + g * group_size_round_up;
    lite::arm::math::prepackA(
        weights_trans_ptr, weights_group, 1.f, k, 0, m, 0, k, false, ctx);
  }
}

template <>
inline void trans_gemm_weights<PRECISION(kInt8)>(const Tensor& tin,
                                                 Tensor& tout,  // NOLINT
                                                 int group,
                                                 ARMContext* ctx) {
  CHECK_EQ(tin.dims().size(), 4) << "conv weights dims size must = 4";
  int m = tin.dims()[0] / group;
  int k = tin.dims().count(1, 4);
  prepackA_int8(&tout, tin, m, k, group, false, ctx);
}

inline void fill_packed_biasc4(float* dout, const float* bias, int size) {
  float32x4_t vb = vld1q_f32(bias);
  int cnt = size / 4;
  for (int i = 0; i < cnt; ++i) {
    vst1q_f32(dout, vb);
    dout += 4;
  }
}

/*preprocessing weights
* input weights: [chout, chin/ group, kh, kw] --> outputs weights: [chout / n,
* chin/ group, kh, kw, n]
*/
template <typename dtype>
static bool conv_trans_weights_numc(const dtype* din,
                                    dtype* dout,
                                    int chout,
                                    int chin,
                                    int n,
                                    int kernel_size) {
  if (n <= 0) {
    LOG(ERROR) << "ch_n and hei_n are more than zero";
    return false;
  }
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
  return true;
}
// for example: m = 4, n = 4
// din = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9 , 10 ,11], [12, 13, 14, 15]]
// dout = [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
/*
  m = 8 n = 8: 0 1 2 3 4 5 6 7           0 8 16 24 32 40 48 56
               16 17 18 19 20 21 22 23   2 10 18 26 34 42 50 58
               24 25 26 27 28 29 30 31   3 11 19 27 35 43 51 59
               32 33 34 35 36 37 38 39   4 12 20 28 36 44 52 60           ...
    }
  }
*/
template <typename Dtype>
void local_transpose(const Dtype* din, Dtype* dout, int m, int n) {
  // n % 4 == 0 && m % 4 == 0
  // n * m ==> n * m data trans
  int offset_m = m << 2;
  const Dtype* din_ptr = din;
  Dtype* dout_ptr = dout;
  for (int i = 0; i < n; i += 4) {
    Dtype* out_ptr0 = dout_ptr;
    Dtype* out_ptr1 = dout_ptr + m;
    Dtype* out_ptr2 = out_ptr1 + m;
    Dtype* out_ptr3 = out_ptr2 + m;
    const Dtype* in_ptr0 = din_ptr;
    const Dtype* in_ptr1 = din_ptr + m;
    const Dtype* in_ptr2 = in_ptr1 + m;
    const Dtype* in_ptr3 = in_ptr2 + m;
    for (int j = 0; j < m; j += 4) {
      float32x4_t vin0 = vld1q_f32(in_ptr0);
      float32x4_t vin1 = vld1q_f32(in_ptr1);
      float32x4_t vin2 = vld1q_f32(in_ptr2);
      float32x4_t vin3 = vld1q_f32(in_ptr3);
      // a00 b00 a02 b02 a01 b01 a03 b03
      float32x4x2_t tmp0 = vtrnq_f32(vin0, vin1);
      // c00 d00 c02 d02 c01 d01 c03 d03
      float32x4x2_t tmp2 = vtrnq_f32(vin2, vin3);
      in_ptr0 = in_ptr3 + m;
      in_ptr1 = in_ptr3 + 2 * m;
      float tmp_val1 = tmp0.val[0][2];
      float tmp_val2 = tmp0.val[0][3];
      tmp0.val[0][2] = tmp2.val[0][0];
      tmp0.val[0][3] = tmp2.val[0][1];
      float tmp_val3 = tmp0.val[1][2];
      float tmp_val4 = tmp0.val[1][3];
      tmp2.val[0][0] = tmp_val1;
      tmp2.val[0][1] = tmp_val2;
      tmp0.val[1][2] = tmp2.val[1][0];
      tmp0.val[1][3] = tmp2.val[1][1];
      tmp2.val[1][0] = tmp_val3;
      tmp2.val[1][1] = tmp_val4;
      in_ptr2 = in_ptr1 + m;
      in_ptr3 = in_ptr1 + 2 * m;
      vst1q_f32(out_ptr0, tmp0.val[0]);
      vst1q_f32(out_ptr1, tmp0.val[1]);
      out_ptr0 += 4;
      out_ptr1 += 4;
      vst1q_f32(out_ptr2, tmp2.val[0]);
      vst1q_f32(out_ptr3, tmp2.val[1]);
      out_ptr2 += 4;
      out_ptr3 += 4;
    }
    dout_ptr += offset_m;
    din_ptr += 4;
  }
}
template <typename Dtype>
void transpose(const Dtype* din, Dtype* dout, int m, int n) {
  // nxm == mxn
  // 4x4
  int cnt_n = n >> 2;
  int remain_n = n & 3;
  int cnt_m = m >> 2;
  int remain_m = m & 3;
  int nn_num = n << 2;  // n * 4
  int mm_num = m << 2;  // m * 4
  for (int x = 0; x < cnt_n; x++) {
    const Dtype* din_ptr0 = din + x * mm_num;
    const Dtype* din_ptr1 = din_ptr0 + m;
    const Dtype* din_ptr2 = din_ptr1 + m;
    const Dtype* din_ptr3 = din_ptr2 + m;
    Dtype* dout_ptr0 = dout + x * 4;
    for (int y = 0; y < cnt_m; y++) {
      float32x4_t din0 = vld1q_f32(din_ptr0);  // a00 a01 a02 a03
      float32x4_t din1 = vld1q_f32(din_ptr1);
      float32x4_t din2 = vld1q_f32(din_ptr2);
      float32x4_t din3 = vld1q_f32(din_ptr3);
      Dtype* dout_ptr1 = dout_ptr0 + n;
      Dtype* dout_ptr2 = dout_ptr1 + n;
      Dtype* dout_ptr3 = dout_ptr2 + n;
      // a00 b00 a02 b02 a01 b01 a03 b03
      float32x4x2_t tmp0 = vtrnq_f32(din0, din1);
      // c00 d00 c02 d02 c01 d01 c03 d03
      float32x4x2_t tmp2 = vtrnq_f32(din2, din3);
      din_ptr0 += 4;
      din_ptr1 += 4;
      // a00 b00 c00 d00 a02 b02 c02 d02
      // a01 b01 c01 d01 a03 b03 c03 d03
      float tmp_val1 = tmp0.val[0][2];
      float tmp_val2 = tmp0.val[0][3];
      tmp0.val[0][2] = tmp2.val[0][0];
      tmp0.val[0][3] = tmp2.val[0][1];
      float tmp_val3 = tmp0.val[1][2];
      float tmp_val4 = tmp0.val[1][3];
      tmp2.val[0][0] = tmp_val1;
      tmp2.val[0][1] = tmp_val2;
      tmp0.val[1][2] = tmp2.val[1][0];
      tmp0.val[1][3] = tmp2.val[1][1];
      tmp2.val[1][0] = tmp_val3;
      tmp2.val[1][1] = tmp_val4;
      din_ptr2 += 4;
      din_ptr3 += 4;
      vst1q_f32(dout_ptr0, tmp0.val[0]);
      vst1q_f32(dout_ptr1, tmp0.val[1]);
      dout_ptr0 += nn_num;
      vst1q_f32(dout_ptr2, tmp2.val[0]);
      vst1q_f32(dout_ptr3, tmp2.val[1]);
    }
    for (int y = 0; y < remain_m; y++) {
      *dout_ptr0++ = *din_ptr0++;
      *dout_ptr0++ = *din_ptr1++;
      *dout_ptr0++ = *din_ptr2++;
      *dout_ptr0++ = *din_ptr3++;
    }
  }
  const Dtype* din_ptr0 = din + cnt_n * mm_num;
  dout = dout + cnt_n * 4;
  for (int x = 0; x < remain_n; x++) {
    Dtype* dout_ptr0 = dout + x * 4;
    for (int y = 0; y < cnt_m; y++) {
      float32x4_t din0 = vld1q_f32(din_ptr0);
      Dtype* dout_ptr1 = dout_ptr0 + n;
      Dtype* dout_ptr2 = dout_ptr1 + n;
      Dtype* dout_ptr3 = dout_ptr2 + n;
      din_ptr0 += 4;
      *dout_ptr0 = din0[0];
      *dout_ptr1 = din0[1];
      dout_ptr0 += nn_num;
      *dout_ptr2 = din0[2];
      *dout_ptr3 = din0[3];
    }
    for (int y = 0; y < remain_m; y++) {
      *dout_ptr0++ = *din_ptr0++;
    }
  }
}
/*preprocessing inputs
* input din: [1, chin, he-hs, we - ws] --> outputs dout: [n, chin, 1, we - ws]
* n = he - hs
*/
template <typename dtype>
static bool prepack_input_nxw(const dtype* din,
                              dtype* dout,
                              int cs,
                              int ce,
                              int hs,
                              int he,
                              int ws,
                              int we,
                              int channel,
                              int width,
                              int height,
                              dtype* zero_ptr) {
  int n = he - hs;
  if (n <= 0) {
    LOG(ERROR) << "hei_n is more than zero";
    return false;
  }
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
  return true;
}

inline void transpose_4x4(float32x4_t v0,
                          float32x4_t v1,
                          float32x4_t v2,
                          float32x4_t v3,
                          float* dout) {
#ifdef __aarch64__
  asm volatile(
      "trn1   v0.4s, %[v0].4s, %[v1].4s\n" /* trans q0, q1, a0b0a2b2*/
      "trn2   v1.4s, %[v0].4s, %[v1].4s\n" /* trans q0, q1, a1b1a3b3*/
      "trn1   v2.4s, %[v2].4s, %[v3].4s\n" /* trans q2, q3, c0d0c2d2*/
      "trn2   v3.4s, %[v2].4s, %[v3].4s\n" /* trans q2, q3, c1d1c3d3*/
      "trn1   v4.2d, v0.2d, v2.2d\n"       /* trans q0, q2, a0b0c0d0*/
      "trn2   v6.2d, v0.2d, v2.2d\n"       /* trans q0, q2, a2b2c2d2*/
      "trn1   v5.2d, v1.2d, v3.2d\n"       /* trans q1, q3, a1b1c1d1*/
      "trn2   v7.2d, v1.2d, v3.2d\n"       /* trans q1, q3, a3b3c3d3*/
      "stp  q4, q5, [%[dout]], #32\n"
      "stp  q6, q7, [%[dout]]\n"
      : [dout] "+r"(dout)
      : [v0] "w"(v0), [v1] "w"(v1), [v2] "w"(v2), [v3] "w"(v3)
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else
  asm volatile(
      "vtrn.32 %q[v0], %q[v1]\n" /* trans q0, q1, a0b0a2b2, a1b1a3b3*/
      "vtrn.32 %q[v2], %q[v3]\n" /* trans q2, q3, c0d0c2d2, c1d1c3d3*/
      "vswp   %f[v0], %e[v2]\n"  /* trans q0, q2, a0b0c0d0, a2b2c2d2*/
      "vswp   %f[v1], %e[v3]\n"  /* trans q1, q3, a1b1c1d1, a3b3c3d3*/
      "vst1.32  {%e[v0], %f[v0]}, [%[dout]]!\n"
      "vst1.32  {%e[v1], %f[v1]}, [%[dout]]!\n"
      "vst1.32  {%e[v2], %f[v2]}, [%[dout]]!\n"
      "vst1.32  {%e[v3], %f[v3]}, [%[dout]]\n"
      : [dout] "+r"(dout)
      : [v0] "w"(v0), [v1] "w"(v1), [v2] "w"(v2), [v3] "w"(v3)
      :);
#endif
}

inline void prepack_input_nxwc4_dw(const float* din,
                                   float* dout,
                                   int cs,
                                   int hs,
                                   int he,
                                   int ws,
                                   int we,
                                   int channel,
                                   int width,
                                   int height,
                                   float* zero_ptr) {
  int n = he - hs;
  if (n <= 0) {
    LOG(FATAL) << "prepack_dw_input, valid height must > zero";
  }
  float32x4_t vzero = vdupq_n_f32(0.f);
  auto out_data = dout;

  int size_w = we - ws;
  int w0 = ws < 0 ? 0 : ws;
  int w1 = we > width ? width : we;
  int valid_w = w1 - w0;

  int mask[4] = {0, 1, 2, 3};

  int pad_l = ws < 0 ? -ws : 0;
  int pad_r = we > width ? we - width : 0;
  int cnt_l = pad_l / 4;
  int left_remain = pad_l - cnt_l * 4;

  bool flag_ext_l = left_remain > 0;
  int left_sl = 4 - left_remain;
  int left_valid_sl = left_sl > width ? width : left_sl;
  uint32x4_t vmask_padl;
  bool flag_mask_l = false;
  if (flag_ext_l) {
    if (valid_w < 3) {
      flag_mask_l = true;
      vmask_padl = vcltq_s32(vld1q_s32(mask), vdupq_n_s32(valid_w));
    }
    valid_w -= left_sl;
    valid_w = valid_w > 0 ? valid_w : 0;
  }
  int cnt_valid = valid_w / 4;
  int valid_sl = valid_w - cnt_valid * 4;
  bool flag_mask_valid = valid_sl > 0;
  uint32x4_t vmask_valid;
  if (flag_mask_valid) {
    vmask_valid = vcltq_s32(vld1q_s32(mask), vdupq_n_s32(valid_sl));
    pad_r -= 4 - valid_sl;
    pad_r = pad_r > 0 ? pad_r : 0;
  }
  int size_c = width * height;
  for (int h = hs; h < he; ++h) {
    dout = out_data + (h - hs) * 4 * size_w;
    auto ptr_c0 = din + cs * size_c + h * width;
    auto ptr_c1 = ptr_c0 + size_c;
    auto ptr_c2 = ptr_c1 + size_c;
    auto ptr_c3 = ptr_c2 + size_c;
    if (h < 0 || h >= height) {
      memset(dout, 0, sizeof(float) * size_w * 4);
      dout += size_w * 4;
      continue;
    } else if (cs + 4 > channel) {
      switch (cs + 4 - channel) {
        case 3:
          ptr_c1 = zero_ptr;
        case 2:
          ptr_c2 = zero_ptr;
        case 1:
          ptr_c3 = zero_ptr;
        default:
          break;
      }
    }
    /// left padding
    if (cnt_l > 0) {
      memset(dout, 0, sizeof(float) * 16 * cnt_l);
      dout += 16 * cnt_l;
    }
    /// left mask
    if (flag_ext_l) {
      float32x4_t vc0 = vld1q_f32(ptr_c0);
      float32x4_t vc1 = vld1q_f32(ptr_c1);
      float32x4_t vc2 = vld1q_f32(ptr_c2);
      float32x4_t vc3 = vld1q_f32(ptr_c3);
      if (flag_mask_l) {
        vc0 = vbslq_f32(vmask_padl, vc0, vzero);
        vc1 = vbslq_f32(vmask_padl, vc1, vzero);
        vc2 = vbslq_f32(vmask_padl, vc2, vzero);
        vc3 = vbslq_f32(vmask_padl, vc3, vzero);
      }
      switch (left_sl) {
        case 1:
          vc0 = vextq_f32(vzero, vc0, 1);
          vc1 = vextq_f32(vzero, vc1, 1);
          vc2 = vextq_f32(vzero, vc2, 1);
          vc3 = vextq_f32(vzero, vc3, 1);
          break;
        case 2:
          vc0 = vextq_f32(vzero, vc0, 2);
          vc1 = vextq_f32(vzero, vc1, 2);
          vc2 = vextq_f32(vzero, vc2, 2);
          vc3 = vextq_f32(vzero, vc3, 2);
          break;
        case 3:
          vc0 = vextq_f32(vzero, vc0, 3);
          vc1 = vextq_f32(vzero, vc1, 3);
          vc2 = vextq_f32(vzero, vc2, 3);
          vc3 = vextq_f32(vzero, vc3, 3);
          break;
        default:
          break;
      }
      transpose_4x4(vc0, vc1, vc2, vc3, dout);
      dout += 16;
      ptr_c0 += left_valid_sl;
      ptr_c1 += left_valid_sl;
      ptr_c2 += left_valid_sl;
      ptr_c3 += left_valid_sl;
    }
    /// valid
    for (int i = 0; i < cnt_valid; ++i) {
      float32x4_t vc0 = vld1q_f32(ptr_c0);
      float32x4_t vc1 = vld1q_f32(ptr_c1);
      float32x4_t vc2 = vld1q_f32(ptr_c2);
      float32x4_t vc3 = vld1q_f32(ptr_c3);
      transpose_4x4(vc0, vc1, vc2, vc3, dout);
      dout += 16;
      ptr_c0 += 4;
      ptr_c1 += 4;
      ptr_c2 += 4;
      ptr_c3 += 4;
    }
    if (flag_mask_valid) {
      float32x4_t vc0 = vld1q_f32(ptr_c0);
      float32x4_t vc1 = vld1q_f32(ptr_c1);
      float32x4_t vc2 = vld1q_f32(ptr_c2);
      float32x4_t vc3 = vld1q_f32(ptr_c3);
      vc0 = vbslq_f32(vmask_valid, vc0, vzero);
      vc1 = vbslq_f32(vmask_valid, vc1, vzero);
      vc2 = vbslq_f32(vmask_valid, vc2, vzero);
      vc3 = vbslq_f32(vmask_valid, vc3, vzero);
      transpose_4x4(vc0, vc1, vc2, vc3, dout);
      dout += 16;
    }
    /// right padding
    if (pad_r > 0) {
      memset(dout, 0, sizeof(float) * 4 * pad_r);
      dout += 4 * pad_r;
    }
  }
}

inline void prepack_input_nxwc8_int8_dw(const int8_t* din,
                                        int8_t* dout,
                                        int cs,
                                        int hs,
                                        int he,
                                        int ws,
                                        int we,
                                        int channel,
                                        int width,
                                        int height) {
  int n = he - hs;
  if (n <= 0) {
    LOG(FATAL) << "prepack_dw_input_int8, valid height must > zero";
  }
  int size_w = we - ws;
  int w0 = ws < 0 ? 0 : ws;
  int w1 = we > width ? width : we;
  int valid_w = w1 - w0;
  int pad_l = ws < 0 ? -ws : 0;
  int pad_r = we > width ? we - width : 0;
  int size_c = width * height;

  int valid_cnt = valid_w >> 3;
  int remain = valid_w & 7;

  int8_t zero_ptr[size_w * 2];  // NOLINT
  memset(zero_ptr, 0, size_w * 2);

  for (int h = hs; h < he; ++h) {
    const int8_t* ptr_c0 = din + h * width + cs * size_c;
    const int8_t* ptr_c1 = ptr_c0 + size_c;
    const int8_t* ptr_c2 = ptr_c1 + size_c;
    const int8_t* ptr_c3 = ptr_c2 + size_c;
    const int8_t* ptr_c4 = ptr_c3 + size_c;
    const int8_t* ptr_c5 = ptr_c4 + size_c;
    const int8_t* ptr_c6 = ptr_c5 + size_c;
    const int8_t* ptr_c7 = ptr_c6 + size_c;
    if (h < 0 || h >= height) {
      memset(dout, 0, 8 * size_w * sizeof(int8_t));
      dout += size_w * 8;
      continue;
    } else if (cs + 8 > channel) {
      switch (cs + 8 - channel) {
        case 7:
          ptr_c1 = zero_ptr;
        case 6:
          ptr_c2 = zero_ptr;
        case 5:
          ptr_c3 = zero_ptr;
        case 4:
          ptr_c4 = zero_ptr;
        case 3:
          ptr_c5 = zero_ptr;
        case 2:
          ptr_c6 = zero_ptr;
        case 1:
          ptr_c7 = zero_ptr;
        default:
          break;
      }
    }
    if (pad_l) {
      memset(dout, 0, pad_l * 8 * sizeof(int8_t));
      dout += pad_l * 8;
    }
    if (valid_cnt) {
      int cnt = valid_cnt;
#ifdef __aarch64__
      asm volatile(
          /* main loop */
          "1:\n"
          "ldr d0,    [%[r0]], #8\n"
          "ldr d1,    [%[r1]], #8\n"
          "ldr d2,    [%[r2]], #8\n"
          "ldr d3,    [%[r3]], #8\n"
          "ldr d4,    [%[r4]], #8\n"
          "ldr d5,    [%[r5]], #8\n"
          "ldr d6,    [%[r6]], #8\n"
          "ldr d7,    [%[r7]], #8\n"
          "trn1 v8.8b,  v0.8b, v1.8b\n"
          "trn2 v9.8b,  v0.8b, v1.8b\n"
          "trn1 v10.8b, v2.8b, v3.8b\n"
          "trn2 v11.8b, v2.8b, v3.8b\n"
          "trn1 v12.8b, v4.8b, v5.8b\n"
          "trn2 v13.8b, v4.8b, v5.8b\n"
          "trn1 v14.8b, v6.8b, v7.8b\n"
          "trn2 v15.8b, v6.8b, v7.8b\n"
          "trn1 v0.4h,  v8.4h, v10.4h\n"
          "trn2 v1.4h,  v8.4h, v10.4h\n"
          "trn1 v2.4h,  v9.4h, v11.4h\n"
          "trn2 v3.4h,  v9.4h, v11.4h\n"
          "trn1 v4.4h,  v12.4h, v14.4h\n"
          "trn2 v5.4h,  v12.4h, v14.4h\n"
          "trn1 v6.4h,  v13.4h, v15.4h\n"
          "trn2 v7.4h,  v13.4h, v15.4h\n"
          "trn1 v8.2s,  v0.2s, v4.2s\n"
          "trn1 v9.2s,  v2.2s, v6.2s\n"
          "trn1 v10.2s, v1.2s, v5.2s\n"
          "trn1 v11.2s, v3.2s, v7.2s\n"
          "stp d8, d9, [%[ptr_out]], #16\n"
          "trn2 v12.2s, v0.2s, v4.2s\n"
          "trn2 v13.2s, v2.2s, v6.2s\n"
          "stp d10, d11, [%[ptr_out]], #16\n"
          "trn2 v14.2s, v1.2s, v5.2s\n"
          "trn2 v15.2s, v3.2s, v7.2s\n"
          "subs %w[cnt], %w[cnt], #1\n"
          "stp d12, d13, [%[ptr_out]], #16\n"
          "stp d14, d15, [%[ptr_out]], #16\n"
          "bne    1b\n"
          : [cnt] "+r"(cnt),
            [r0] "+r"(ptr_c0),
            [r1] "+r"(ptr_c1),
            [r2] "+r"(ptr_c2),
            [r3] "+r"(ptr_c3),
            [r4] "+r"(ptr_c4),
            [r5] "+r"(ptr_c5),
            [r6] "+r"(ptr_c6),
            [r7] "+r"(ptr_c7),
            [ptr_out] "+r"(dout)
          :
          : "cc",
            "memory",
            "v0",
            "v1",
            "v2",
            "v3",
            "v4",
            "v5",
            "v6",
            "v7",
            "v8",
            "v9",
            "v10",
            "v11",
            "v12",
            "v13",
            "v14",
            "v15");
#else
      asm volatile(
          /* main loop */
          "1:\n"
          "vld1.32 {d0},  [%[r0]]!\n"
          "vld1.32 {d1},  [%[r1]]!\n"
          "vld1.32 {d2},  [%[r2]]!\n"
          "vld1.32 {d3},  [%[r3]]!\n"
          "vld1.32 {d4},  [%[r4]]!\n"
          "vld1.32 {d5},  [%[r5]]!\n"
          "vld1.32 {d6},  [%[r6]]!\n"
          "vld1.32 {d7},  [%[r7]]!\n"
          "vtrn.8   d0, d1\n"
          "vtrn.8   d2, d3\n"
          "vtrn.8   d4, d5\n"
          "vtrn.8   d6, d7\n"
          "vtrn.16  d0, d2\n"
          "vtrn.16  d1, d3\n"
          "vtrn.16  d4, d6\n"
          "vtrn.16  d5, d7\n"
          "vtrn.32  d0, d4\n"
          "vtrn.32  d2, d6\n"
          "vtrn.32  d1, d5\n"
          "vtrn.32  d3, d7\n"
          "subs %[cnt], #1\n"
          "vst1.32 {d0-d3}, [%[ptr_out]]!\n"
          "vst1.32 {d4-d7}, [%[ptr_out]]!\n"
          "bne    1b\n"
          : [cnt] "+r"(cnt),
            [r0] "+r"(ptr_c0),
            [r1] "+r"(ptr_c1),
            [r2] "+r"(ptr_c2),
            [r3] "+r"(ptr_c3),
            [r4] "+r"(ptr_c4),
            [r5] "+r"(ptr_c5),
            [r6] "+r"(ptr_c6),
            [r7] "+r"(ptr_c7),
            [ptr_out] "+r"(dout)
          :
          : "cc", "memory", "q0", "q1", "q2", "q3");
#endif  // __aarch64__
    }
    for (int i = 0; i < remain; ++i) {
      dout[0] = *(ptr_c0++);
      dout[1] = *(ptr_c1++);
      dout[2] = *(ptr_c2++);
      dout[3] = *(ptr_c3++);
      dout[4] = *(ptr_c4++);
      dout[5] = *(ptr_c5++);
      dout[6] = *(ptr_c6++);
      dout[7] = *(ptr_c7++);
      dout += 8;
    }
    if (pad_r) {
      memset(dout, 0, pad_r * 8 * sizeof(int8_t));
      dout += pad_r * 8;
    }
  }
}
// clang-format off
#ifdef __aarch64__
#define NCHWC1_TRANS_FP32_COMPUTE                                      \
  "ldr q0, [%[ptr_din]], #16      \n" /* load data, c0r0, c1r0, c0r1*/ \
  "ldr q1, [%[ptr_din]], #16      \n" /* load data, c0r0, c1r0, c0r1*/ \
  "ldr q2, [%[ptr_din]], #16      \n" /* load data, c0r0, c1r0, c0r1*/ \
  "ldr q3, [%[ptr_din]], #16      \n" /* load data, c0r0, c1r0, c0r1*/ \
  "movi v20.4s, #0                \n" /* for relu */                   \
  "1:                             \n" /* main loop*/

#define NCHWC1_TRANS_FP32_RELU                 \
  "fmax   v0.4s, v0.4s, v20.4s    \n" /*relu*/ \
  "fmax   v1.4s, v1.4s, v20.4s    \n" /*relu*/ \
  "fmax   v2.4s, v2.4s, v20.4s    \n" /*relu*/ \
  "fmax   v3.4s, v3.4s, v20.4s    \n" /*relu*/

#define NCHWC1_TRANS_FP32_RELU6                    \
  "fmin   v0.4s, v0.4s, %[six].4s  \n" /* relu6 */ \
  "fmin   v1.4s, v1.4s, %[six].4s  \n" /* relu6 */ \
  "fmin   v2.4s, v2.4s, %[six].4s  \n" /* relu6 */ \
  "fmin   v3.4s, v3.4s, %[six].4s  \n" /* relu6 */

#define NCHWC1_TRANS_FP32_LEAKY_RELU                   \
  "fcmge v4.4s, v0.4s, v20.4s \n"      /* vcgeq_f32 */ \
  "fcmge v5.4s, v1.4s, v20.4s \n"      /* vcgeq_f32 */ \
  "fcmge v6.4s, v2.4s, v20.4s \n"      /* vcgeq_f32 */ \
  "fcmge v7.4s, v3.4s, v20.4s \n"      /* vcgeq_f32 */ \
  "fmul v8.4s, v0.4s, %[scale].4s  \n" /* mul */       \
  "fmul v9.4s, v1.4s, %[scale].4s  \n" /* mul */       \
  "fmul v10.4s, v2.4s, %[scale].4s \n" /* mul */       \
  "fmul v11.4s, v3.4s, %[scale].4s \n" /* mul */       \
  "bif  v0.16b, v8.16b, v4.16b  \n"    /* choose*/     \
  "bif  v1.16b, v9.16b, v5.16b  \n"    /* choose*/     \
  "bif  v2.16b, v10.16b, v6.16b \n"    /* choose*/     \
  "bif  v3.16b, v11.16b, v7.16b \n"    /* choose*/

#define NCHWC1_TRANS_FP32_STORE                                        \
  "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/               \
                                                                       \
  "str    q0, [%[doutc0r0]], #16  \n" /* store c0r0*/                  \
  "str    q1, [%[doutc0r0]], #16  \n" /* store c0r0*/                  \
  "ldr q0, [%[ptr_din]], #16      \n" /* load data, c0r0, c1r0, c0r1*/ \
  "ldr q1, [%[ptr_din]], #16      \n" /* load data, c0r0, c1r0, c0r1*/ \
  "str    q2, [%[doutc0r0]], #16  \n" /* store c0r0*/                  \
  "str    q3, [%[doutc0r0]], #16  \n" /* store c2r0*/                  \
  "ldr q2, [%[ptr_din]], #16      \n" /* load data, c0r0, c1r0, c0r1*/ \
  "ldr q3, [%[ptr_din]], #16      \n" /* load data, c0r0, c1r0, c0r1*/ \
                                                                       \
  "bne    1b                      \n" /* jump to main loop*/
#else
#define NCHWC1_TRANS_FP32_COMPUTE                                       \
  "vld1.32 {d0-d3}, [%[ptr_din]]!                 @ load data, c0r0 \n" \
  "vld1.32 {d4-d7}, [%[ptr_din]]!                 @ load data, c0r0 \n" \
  "vmov.u32 q15, #0                       @ dump zero\n"                \
  "1:                                     @ main loop\n"

#define NCHWC1_TRANS_FP32_RELU                      \
  "vmax.f32   q0, q0, q15                 @ relu\n" \
  "vmax.f32   q1, q1, q15                 @ relu\n" \
  "vmax.f32   q2, q2, q15                 @ relu\n" \
  "vmax.f32   q3, q3, q15                 @ relu\n"

#define NCHWC1_TRANS_FP32_RELU6                  \
  "vmin.f32   q0, q0, %q[six]        @ relu6 \n" \
  "vmin.f32   q1, q1, %q[six]        @ relu6 \n" \
  "vmin.f32   q2, q2, %q[six]        @ relu6 \n" \
  "vmin.f32   q3, q3, %q[six]        @ relu6 \n"

#define NCHWC1_TRANS_FP32_LEAKY_RELU          \
  "vcge.f32   q5, q0, q15        @ q0 > 0 \n" \
  "vcge.f32   q6, q1, q15        @ q0 > 0 \n" \
  "vcge.f32   q7, q2, q15        @ q0 > 0 \n" \
  "vcge.f32   q8, q3, q15        @ q0 > 0 \n" \
  "vmul.f32 q9, q0, %q[scale] \n"             \
  "vmul.f32 q10, q1, %q[scale] \n"            \
  "vmul.f32 q11, q2, %q[scale] \n"            \
  "vmul.f32 q12, q3, %q[scale] \n"            \
  "vbif q0, q9, q5 @ choose \n"               \
  "vbif q1, q10, q6 @ choose \n"              \
  "vbif q2, q11, q7 @ choose \n"              \
  "vbif q3, q12, q8 @ choose \n"

#define NCHWC1_TRANS_FP32_STORE                                 \
  "vst1.32  {d0-d1}, [%[doutc0r0]]!       @ store result  \n"   \
  "vst1.32  {d2-d3}, [%[doutc0r0]]!       @ store result, \n"   \
  "subs   %[cnt], %[cnt], #1              @ loop count - 1\n"   \
                                                                \
  "vld1.32 {d0-d3}, [%[ptr_din]]!         @ load data      \n"  \
  "vst1.32  {d4-d5}, [%[doutc0r0]]!       @ store result   \n"  \
  "vst1.32  {d6-d7}, [%[doutc0r0]]!       @ store result,  \n"  \
                                                                \
  "vld1.32 {d4-d7}, [%[ptr_din]]!         @ load data     \n"   \
                                                                \
  "bne    1b                              @ jump to main loop\n"
#endif
// clang-format on
inline void act_switch_c1_fp32(const float* din_ptr,
                               float* doutc0_ptr,
                               int cnt_loop,
                               const operators::ActivationParam* act_param) {
  if (act_param != nullptr && act_param->has_active) {
    float32x4_t six = vdupq_n_f32(act_param->Relu_clipped_coef);
    float32x4_t scale = vdupq_n_f32(act_param->Leaky_relu_alpha);
    switch (act_param->active_type) {
      case lite_api::ActivationType::kRelu:
#ifdef __aarch64__
        asm volatile(NCHWC1_TRANS_FP32_COMPUTE NCHWC1_TRANS_FP32_RELU
                         NCHWC1_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [cnt] "+r"(cnt_loop),
                       [ptr_din] "+r"(din_ptr)
                     :
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v4",
                       "v5",
                       "v6",
                       "v7",
                       "v8",
                       "v9",
                       "v10",
                       "v20");
#else
        asm volatile(NCHWC1_TRANS_FP32_COMPUTE NCHWC1_TRANS_FP32_RELU
                         NCHWC1_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [ptr_din] "+r"(din_ptr),
                       [cnt] "+r"(cnt_loop)
                     :
                     : "cc", "memory", "q0", "q1", "q2", "q3", "q15");
#endif
        break;
      case lite_api::ActivationType::kRelu6:
/* 0 <= din <= 6 */
#ifdef __aarch64__
        asm volatile(NCHWC1_TRANS_FP32_COMPUTE NCHWC1_TRANS_FP32_RELU
                         NCHWC1_TRANS_FP32_RELU6 NCHWC1_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [cnt] "+r"(cnt_loop),
                       [ptr_din] "+r"(din_ptr)
                     : [six] "w"(six)
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v4",
                       "v5",
                       "v6",
                       "v7",
                       "v8",
                       "v9",
                       "v10",
                       "v20");
#else
        asm volatile(NCHWC1_TRANS_FP32_COMPUTE NCHWC1_TRANS_FP32_RELU
                         NCHWC1_TRANS_FP32_RELU6 NCHWC1_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [ptr_din] "+r"(din_ptr),
                       [cnt] "+r"(cnt_loop)
                     : [six] "w"(six)
                     : "cc", "memory", "q0", "q1", "q2", "q3", "q15");
#endif
        break;
      case lite_api::ActivationType::kLeakyRelu:
/*din = din >= 0 ? din : din * scale*/
#ifdef __aarch64__
        asm volatile(NCHWC1_TRANS_FP32_COMPUTE NCHWC1_TRANS_FP32_LEAKY_RELU
                         NCHWC1_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [cnt] "+r"(cnt_loop),
                       [ptr_din] "+r"(din_ptr)
                     : [scale] "w"(scale)
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v4",
                       "v5",
                       "v6",
                       "v7",
                       "v8",
                       "v9",
                       "v10",
                       "v11",
                       "v20");
#else
        asm volatile(NCHWC1_TRANS_FP32_COMPUTE NCHWC1_TRANS_FP32_LEAKY_RELU
                         NCHWC1_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [ptr_din] "+r"(din_ptr),
                       [cnt] "+r"(cnt_loop)
                     : [scale] "w"(scale)
                     : "cc",
                       "memory",
                       "q0",
                       "q1",
                       "q2",
                       "q3",
                       "q5",
                       "q6",
                       "q7",
                       "q8",
                       "q9",
                       "q10",
                       "q11",
                       "q12",
                       "q15");
#endif
        break;
      default:
        LOG(FATAL) << "this act_type: "
                   << static_cast<int>(act_param->active_type)
                   << " fuse not support";
    }
  } else {
#ifdef __aarch64__
    asm volatile(NCHWC1_TRANS_FP32_COMPUTE NCHWC1_TRANS_FP32_STORE
                 : [doutc0r0] "+r"(doutc0_ptr),
                   [cnt] "+r"(cnt_loop),
                   [ptr_din] "+r"(din_ptr)
                 :
                 : "cc", "memory", "v0", "v1", "v2", "v3", "v20");
#else
    asm volatile(NCHWC1_TRANS_FP32_COMPUTE NCHWC1_TRANS_FP32_STORE
                 : [doutc0r0] "+r"(doutc0_ptr),
                   [ptr_din] "+r"(din_ptr),
                   [cnt] "+r"(cnt_loop)
                 :
                 : "cc", "memory", "q0", "q1", "q2", "q3", "q15");
#endif
  }
}
/*wirte result in outputs
* input din: [n, c, h, w], output dout: [n, c, h, w]
*/
inline bool write_to_output_c1_fp32(const float* din,
                                    float* dout,
                                    int cs,
                                    int ce,
                                    int hs,
                                    int he,
                                    int ws,
                                    int we,
                                    int channel,
                                    int height,
                                    int width,
                                    bool flag_relu,
                                    float* trash_ptr,
                                    operators::ActivationParam* act_param) {
  if (cs > channel) {
    return true;
  }

  const int c1 = 1;
  const int w4 = 16;

  int size_c_out = width * height;

  float* doutc0r0 = dout + cs * size_c_out + hs * width + ws;

  const float* ptr_din = din;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int w_round = we - ws;
  int cnt = (width - ws) / w4;
  int remain = (width - ws) % w4;
  for (int i = 0; i < size_h; i++) {
    int size_w = i * width;
    float* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
    const float* din_hei_ptr = ptr_din + i * w_round * c1;
    if (cnt > 0) {
      int cnt_loop = cnt;
      act_switch_c1_fp32(din_hei_ptr, doutc0_ptr, cnt_loop, act_param);
    }
    if (remain > 0) {
      int offset = i * w_round * c1 + c1 * w4 * cnt;
      din_hei_ptr = ptr_din + offset;
      doutc0_ptr += w4 * cnt;
      int j = w4 * cnt;
      if (act_param != nullptr && act_param->has_active) {
        float six = act_param->Relu_clipped_coef;
        float scale = act_param->Leaky_relu_alpha;
        switch (act_param->active_type) {
          case lite_api::ActivationType::kRelu:
            for (; j < width; ++j) {
              *(doutc0_ptr++) = LITEMAX(din_hei_ptr[0], 0.f);
              din_hei_ptr++;
            }
            break;
          case lite_api::ActivationType::kRelu6:
            /* 0 <= din <= 6 */
            for (; j < width; ++j) {
              float tmp = LITEMAX(din_hei_ptr[0], 0.f);
              *(doutc0_ptr++) = LITEMIN(tmp, six);
              din_hei_ptr++;
            }
            break;
          case lite_api::ActivationType::kLeakyRelu:
            /*din = din >= 0 ? din : din * scale*/
            for (; j < width; ++j) {
              if (din_hei_ptr[0] >= 0) {
                *(doutc0_ptr++) = din_hei_ptr[0];
              } else {
                *(doutc0_ptr++) = din_hei_ptr[0] * scale;
              }
              din_hei_ptr++;
            }
            break;
          default:
            LOG(FATAL) << "this act_type: "
                       << static_cast<int>(act_param->active_type)
                       << " fuse not support";
        }
      } else {
        for (; j < width; ++j) {
          *(doutc0_ptr++) = *(din_hei_ptr++);
        }
      }
    }
  }
  return true;
}
// clang-format off
#ifdef __aarch64__
#define NCHWC2_TRANS_FP32_COMPUTE                                      \
  "ldp q0, q1, [%[ptr_din]], #32  \n" /* load data, c0r0, c1r0, c0r1*/ \
  "movi v20.4s, #0                \n" /* for relu */                   \
  "1:                             \n" /* main loop*/                   \
  "trn1   v2.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/                \
  "trn2   v3.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/                \
  "ldp q0, q1, [%[ptr_din]], #32  \n" /* load data, c0r0, c1r0, c0r1*/ \
  "trn1   v4.2d, v2.2d, v3.2d     \n" /* trans q8, q10*/               \
  "trn2   v5.2d, v2.2d, v3.2d     \n" /* trans q8, q10*/

#define NCHWC2_TRANS_FP32_RELU                 \
  "fmax   v2.4s, v4.4s, v20.4s    \n" /*relu*/ \
  "fmax   v3.4s, v5.4s, v20.4s    \n" /*relu*/

#define NCHWC2_TRANS_FP32_RELU6                    \
  "fmin   v2.4s, v2.4s, %[six].4s  \n" /* relu6 */ \
  "fmin   v3.4s, v3.4s, %[six].4s  \n" /* relu6 */

#define NCHWC2_TRANS_FP32_LEAKY_RELU                   \
  "fcmge v6.4s, v2.4s, v20.4s \n"      /* vcgeq_f32 */ \
  "fcmge v7.4s, v3.4s, v20.4s \n"      /* vcgeq_f32 */ \
  "fmul v4.4s, v2.4s, %[scale].4s \n" /* mul */        \
  "fmul v5.4s, v3.4s, %[scale].4s \n" /* mul */        \
  "bif  v2.16b, v4.16b, v6.16b \n"    /* choose*/      \
  "bif  v3.16b, v5.16b, v7.16b \n"    /* choose*/

#define NCHWC2_TRANS_FP32_STORE                          \
  "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/ \
                                                         \
  "str    q2, [%[doutc0r0]], #16  \n" /* store c0r0*/    \
  "str    q3, [%[doutc1r0]], #16  \n" /* store c2r0*/    \
                                                         \
  "bne    1b                      \n" /* jump to main loop*/
#else
#define NCHWC2_TRANS_FP32_COMPUTE                                      \
  "vld1.32 {d0-d3}, [%[ptr_din]]!         @ load data, c0r0, c1r0 \n"  \
  "vmov.u32 q15, #0                       @ dump zero\n"               \
  "1:                                     @ main loop\n"               \
  "vtrn.32 d0, d1                         @ trans data:c0r0, c0r1, "   \
  "c1r0, c1r1 \n"                                                      \
  "vtrn.32 d2, d3                         @ trans data:c0r2, c0r3, "   \
  "c1r2, c1r3 \n"                                                      \
                                                                       \
  "vswp  d1, d2                           @ swap data\n"

#define NCHWC2_TRANS_FP32_RELU                      \
  "vmax.f32   q0, q0, q15                 @ relu\n" \
  "vmax.f32   q1, q1, q15                 @ relu\n"

#define NCHWC2_TRANS_FP32_RELU6                  \
  "vmin.f32   q0, q0, %q[six]        @ relu6 \n" \
  "vmin.f32   q1, q1, %q[six]        @ relu6 \n"

#define NCHWC2_TRANS_FP32_LEAKY_RELU          \
  "vcge.f32   q5, q0, q15        @ q0 > 0 \n" \
  "vcge.f32   q6, q1, q15        @ q0 > 0 \n" \
  "vmul.f32 q9, q0, %q[scale] \n"             \
  "vmul.f32 q10, q1, %q[scale] \n"            \
  "vbif q0, q9, q5 @ choose \n"               \
  "vbif q1, q10, q6 @ choose \n"

#define NCHWC2_TRANS_FP32_STORE                                 \
  "vst1.32  {d0-d1}, [%[doutc0r0]]!       @ store result, add pointer\n"   \
  "vst1.32  {d2-d3}, [%[doutc1r0]]!       @ store result, add pointer\n"   \
                                                                \
  "subs   %[cnt], %[cnt], #1              @ loop count - 1\n"   \
                                                                \
  "vld1.32 {d0-d3}, [%[ptr_din]]!         @ load data \n"       \
                                                                \
  "bne    1b                              @ jump to main loop\n"
#endif
// clang-format on
inline void act_switch_c2_fp32(const float* din_ptr,
                               float* doutc0_ptr,
                               float* doutc1_ptr,
                               int cnt_loop,
                               const operators::ActivationParam* act_param) {
  if (act_param != nullptr && act_param->has_active) {
    float32x4_t six = vdupq_n_f32(act_param->Relu_clipped_coef);
    float32x4_t scale = vdupq_n_f32(act_param->Leaky_relu_alpha);
    switch (act_param->active_type) {
      case lite_api::ActivationType::kRelu:
#ifdef __aarch64__
        asm volatile(NCHWC2_TRANS_FP32_COMPUTE NCHWC2_TRANS_FP32_RELU
                         NCHWC2_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [cnt] "+r"(cnt_loop),
                       [ptr_din] "+r"(din_ptr)
                     :
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v4",
                       "v5",
                       "v6",
                       "v7",
                       "v8",
                       "v9",
                       "v10",
                       "v20");
#else
        asm volatile(NCHWC2_TRANS_FP32_COMPUTE NCHWC2_TRANS_FP32_RELU
                         NCHWC2_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [ptr_din] "+r"(din_ptr),
                       [cnt] "+r"(cnt_loop)
                     :
                     : "cc", "memory", "q0", "q1", "q2", "q3", "q15");
#endif
        break;
      case lite_api::ActivationType::kRelu6:
/* 0 <= din <= 6 */
#ifdef __aarch64__
        asm volatile(NCHWC2_TRANS_FP32_COMPUTE NCHWC2_TRANS_FP32_RELU
                         NCHWC2_TRANS_FP32_RELU6 NCHWC2_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [cnt] "+r"(cnt_loop),
                       [ptr_din] "+r"(din_ptr)
                     : [six] "w"(six)
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v4",
                       "v5",
                       "v6",
                       "v7",
                       "v8",
                       "v9",
                       "v10",
                       "v20");
#else
        asm volatile(NCHWC2_TRANS_FP32_COMPUTE NCHWC2_TRANS_FP32_RELU
                         NCHWC2_TRANS_FP32_RELU6 NCHWC2_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [ptr_din] "+r"(din_ptr),
                       [cnt] "+r"(cnt_loop)
                     : [six] "w"(six)
                     : "cc", "memory", "q0", "q1", "q2", "q3", "q15");
#endif
        break;
      case lite_api::ActivationType::kLeakyRelu:
/*din = din >= 0 ? din : din * scale*/
#ifdef __aarch64__
        asm volatile(NCHWC2_TRANS_FP32_COMPUTE NCHWC2_TRANS_FP32_LEAKY_RELU
                         NCHWC2_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [cnt] "+r"(cnt_loop),
                       [ptr_din] "+r"(din_ptr)
                     : [scale] "w"(scale)
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v4",
                       "v5",
                       "v6",
                       "v7",
                       "v8",
                       "v9",
                       "v10",
                       "v20");
#else
        asm volatile(NCHWC2_TRANS_FP32_COMPUTE NCHWC2_TRANS_FP32_LEAKY_RELU
                         NCHWC2_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [ptr_din] "+r"(din_ptr),
                       [cnt] "+r"(cnt_loop)
                     : [scale] "w"(scale)
                     : "cc",
                       "memory",
                       "q0",
                       "q1",
                       "q2",
                       "q3",
                       "q5",
                       "q6",
                       "q7",
                       "q8",
                       "q9",
                       "q10",
                       "q11",
                       "q12",
                       "q15");
#endif
        break;
      default:
        LOG(FATAL) << "this act_type: "
                   << static_cast<int>(act_param->active_type)
                   << " fuse not support";
    }
  } else {
#ifdef __aarch64__
    asm volatile(NCHWC2_TRANS_FP32_COMPUTE NCHWC2_TRANS_FP32_STORE
                 : [doutc0r0] "+r"(doutc0_ptr),
                   [doutc1r0] "+r"(doutc1_ptr),
                   [cnt] "+r"(cnt_loop),
                   [ptr_din] "+r"(din_ptr)
                 :
                 : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v20");
#else
    asm volatile(NCHWC2_TRANS_FP32_COMPUTE NCHWC2_TRANS_FP32_STORE
                 : [doutc0r0] "+r"(doutc0_ptr),
                   [doutc1r0] "+r"(doutc1_ptr),
                   [ptr_din] "+r"(din_ptr),
                   [cnt] "+r"(cnt_loop)
                 :
                 : "cc", "memory", "q0", "q1", "q2", "q3", "q15");
#endif
  }
}
/*wirte result in outputs
* input din: [n, c / 4, h, w * 4], output dout: [n, c, h, w]
*/
inline bool write_to_output_c2_fp32(const float* din,
                                    float* dout,
                                    int cs,
                                    int ce,
                                    int hs,
                                    int he,
                                    int ws,
                                    int we,
                                    int channel,
                                    int height,
                                    int width,
                                    bool flag_relu,
                                    float* trash_ptr,
                                    operators::ActivationParam* act_param) {
  if (cs > channel) {
    return true;
  }
  const int c2 = 2;
  const int w4 = 4;

  //    float trash_ptr[width];

  int size_c_out = width * height;

  float* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
  float* doutc1r0 = doutc0r0 + size_c_out;

  const float* ptr_din = din;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int w_round = we - ws;
  int cnt = (width - ws) / w4;

  for (int i = 0; i < size_h; i++) {
    int size_w = i * width;
    float* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
    float* doutc1_ptr = doutc1r0 + size_w;
    if (ce > channel) {
      switch (ce - channel) {
        case 1:
          doutc1_ptr = trash_ptr;
        default:
          break;
      }
    }
    const float* din_hei_ptr = ptr_din + i * w_round * c2;
    if (cnt > 0) {
      int cnt_loop = cnt;
      act_switch_c2_fp32(
          din_hei_ptr, doutc0_ptr, doutc1_ptr, cnt_loop, act_param);
    }
    if (we > width) {
      int offset = i * w_round * c2 + c2 * w4 * cnt;
      din_hei_ptr = ptr_din + offset;
      doutc0_ptr += w4 * cnt;
      doutc1_ptr += w4 * cnt;
      int j = we - w4;
      if (act_param != nullptr && act_param->has_active) {
        float six = act_param->Relu_clipped_coef;
        float scale = act_param->Leaky_relu_alpha;
        switch (act_param->active_type) {
          case lite_api::ActivationType::kRelu:
            for (; j < width; ++j) {
              *(doutc0_ptr++) = LITEMAX(din_hei_ptr[0], 0.f);
              *(doutc1_ptr++) = LITEMAX(din_hei_ptr[1], 0.f);
              din_hei_ptr += 2;
            }
            break;
          case lite_api::ActivationType::kRelu6:
            /* 0 <= din <= 6 */
            for (; j < width; ++j) {
              float tmp1 = LITEMAX(din_hei_ptr[0], 0.f);
              float tmp2 = LITEMAX(din_hei_ptr[1], 0.f);
              *(doutc0_ptr++) = LITEMIN(tmp1, six);
              *(doutc1_ptr++) = LITEMIN(tmp2, six);
              din_hei_ptr += 2;
            }
            break;
          case lite_api::ActivationType::kLeakyRelu:
            /*din = din >= 0 ? din : din * scale*/
            for (; j < width; ++j) {
              if (din_hei_ptr[0] >= 0) {
                *(doutc0_ptr++) = din_hei_ptr[0];
              } else {
                *(doutc0_ptr++) = din_hei_ptr[0] * scale;
              }
              if (din_hei_ptr[1] >= 0) {
                *(doutc1_ptr++) = din_hei_ptr[1];
              } else {
                *(doutc1_ptr++) = din_hei_ptr[1] * scale;
              }
              din_hei_ptr += 2;
            }
            break;
          default:
            LOG(FATAL) << "this act_type: "
                       << static_cast<int>(act_param->active_type)
                       << " fuse not support";
        }
      } else {
        for (; j < width; ++j) {
          *(doutc0_ptr++) = *(din_hei_ptr++);
          *(doutc1_ptr++) = *(din_hei_ptr++);
        }
      }
    }
  }
  return true;
}
// clang-format off
#ifdef __aarch64__
#define NCHWC4_TRANS_FP32_COMPUTE                                   \
  "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */ \
  "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */ \
  "movi v20.4s, #0                \n" /* for relu */                \
  "1:                             \n" /* main loop*/                \
  "trn1   v8.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/             \
  "trn2   v9.4s, v0.4s, v1.4s     \n" /* trans q0, q1*/             \
  "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */ \
  "trn1   v10.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/             \
  "trn2   v11.4s, v2.4s, v3.4s    \n" /* trans q2, q3*/             \
  "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */ \
  "trn1   v16.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/            \
  "trn2   v17.2d, v8.2d, v10.2d   \n" /* trans q8, q10*/            \
  "trn1   v18.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/            \
  "trn2   v19.2d, v9.2d, v11.2d   \n" /* trans q9, q11*/

#define NCHWC4_TRANS_FP32_RELU                 \
  "fmax   v16.4s, v16.4s, v20.4s  \n" /*relu*/ \
  "fmax   v17.4s, v17.4s, v20.4s  \n" /*relu*/ \
  "fmax   v18.4s, v18.4s, v20.4s  \n" /*relu*/ \
  "fmax   v19.4s, v19.4s, v20.4s  \n" /*relu*/

#define NCHWC4_TRANS_FP32_RELU6                      \
  "fmin   v16.4s, v16.4s, %[six].4s  \n" /* relu6 */ \
  "fmin   v17.4s, v17.4s, %[six].4s  \n" /* relu6 */ \
  "fmin   v18.4s, v18.4s, %[six].4s  \n" /* relu6 */ \
  "fmin   v19.4s, v19.4s, %[six].4s  \n" /* relu6 */

#define NCHWC4_TRANS_FP32_LEAKY_RELU                    \
  "fcmge v8.4s, v16.4s, v20.4s  \n"     /* vcgeq_f32 */ \
  "fcmge v9.4s, v17.4s, v20.4s  \n"     /* vcgeq_f32 */ \
  "fcmge v10.4s, v18.4s, v20.4s \n"     /* vcgeq_f32 */ \
  "fcmge v11.4s, v19.4s, v20.4s \n"     /* vcgeq_f32 */ \
  "fmul v4.4s, v16.4s, %[scale].4s \n"  /* mul */       \
  "fmul v5.4s, v17.4s, %[scale].4s \n"  /* mul */       \
  "fmul v6.4s, v18.4s, %[scale].4s \n"  /* mul */       \
  "fmul v7.4s, v19.4s, %[scale].4s \n"  /* mul */       \
  "bif  v16.16b, v4.16b, v8.16b  \n"    /* choose*/     \
  "bif  v17.16b, v5.16b, v9.16b  \n"    /* choose*/     \
  "bif  v18.16b, v6.16b, v10.16b \n"    /* choose*/     \
  "bif  v19.16b, v7.16b, v11.16b \n"    /* choose*/

#define NCHWC4_TRANS_FP32_STORE                          \
  "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0*/    \
  "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0*/    \
  "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0*/    \
  "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0*/    \
                                                         \
  "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/ \
  "bne    1b                      \n" /* jump to main loop*/
#else
#define NCHWC4_TRANS_FP32_COMPUTE                                     \
  "vld1.32 {d0-d3}, [%[ptr_din]]!                 @load data \n"      \
  "vld1.32 {d4-d7}, [%[ptr_din]]!         @load data \n"              \
  "vmov.u32 q15, #0                       @ dump zero\n"              \
  "1:                                     @ main loop\n"              \
  "vtrn.32 q0, q1                         @ trans data:c00c01c20c21 " \
  "\n"                                                                \
  "vtrn.32 q2, q3                         @ trans data:c02c03c22c23 " \
  "\n"                                                                \
                                                                      \
  "vswp   d1, d4                          @ swap data\n"              \
  "vswp   d3, d6                          @ swap data\n"

#define NCHWC4_TRANS_FP32_RELU             \
  "vmax.f32   q0, q0, q15        @ relu\n" \
  "vmax.f32   q1, q1, q15        @ relu\n" \
  "vmax.f32   q2, q2, q15        @ relu\n" \
  "vmax.f32   q3, q3, q15        @ relu\n"

#define NCHWC4_TRANS_FP32_RELU6                  \
  "vmin.f32   q0, q0, %q[six]        @ relu6 \n" \
  "vmin.f32   q1, q1, %q[six]        @ relu6 \n" \
  "vmin.f32   q2, q2, %q[six]        @ relu6 \n" \
  "vmin.f32   q3, q3, %q[six]        @ relu6 \n"

#define NCHWC4_TRANS_FP32_LEAKY_RELU          \
  "vcge.f32   q5, q0, q15        @ q0 > 0 \n" \
  "vcge.f32   q6, q1, q15        @ q0 > 0 \n" \
  "vcge.f32   q7, q2, q15        @ q0 > 0 \n" \
  "vcge.f32   q8, q3, q15        @ q0 > 0 \n" \
  "vmul.f32 q9, q0, %q[scale] \n"             \
  "vmul.f32 q10, q1, %q[scale] \n"            \
  "vmul.f32 q11, q2, %q[scale] \n"            \
  "vmul.f32 q12, q3, %q[scale] \n"            \
  "vbif q0, q9, q5 @ choose \n"               \
  "vbif q1, q10, q6 @ choose \n"              \
  "vbif q2, q11, q7 @ choose \n"              \
  "vbif q3, q12, q8 @ choose \n"

#define NCHWC4_TRANS_FP32_STORE                                        \
  "vst1.32  {d0-d1}, [%[doutc0r0]]!     @ store result, add pointer\n" \
  "vst1.32  {d2-d3}, [%[doutc1r0]]!     @ store result, add pointer\n" \
  "vst1.32  {d4-d5}, [%[doutc2r0]]!     @ store result, add pointer\n" \
  "vst1.32  {d6-d7}, [%[doutc3r0]]!     @ store result, add pointer\n" \
                                                                       \
  "subs   %[cnt], %[cnt], #1    @ loop count - 1\n"                    \
                                                                       \
  "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"                \
  "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"                \
                                                                       \
  "bne    1b                            @ jump to main loop\n"
#endif
// clang-format on
inline void act_switch_c4_fp32(const float* din_ptr,
                               float* doutc0_ptr,
                               float* doutc1_ptr,
                               float* doutc2_ptr,
                               float* doutc3_ptr,
                               int cnt_loop,
                               const operators::ActivationParam* act_param) {
  if (act_param != nullptr && act_param->has_active) {
    float32x4_t six = vdupq_n_f32(act_param->Relu_clipped_coef);
    float32x4_t scale = vdupq_n_f32(act_param->Leaky_relu_alpha);
    switch (act_param->active_type) {
      case lite_api::ActivationType::kRelu:
#ifdef __aarch64__
        asm volatile(NCHWC4_TRANS_FP32_COMPUTE NCHWC4_TRANS_FP32_RELU
                         NCHWC4_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [doutc2r0] "+r"(doutc2_ptr),
                       [doutc3r0] "+r"(doutc3_ptr),
                       [cnt] "+r"(cnt_loop),
                       [ptr_din] "+r"(din_ptr)
                     :
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v4",
                       "v5",
                       "v6",
                       "v7",
                       "v8",
                       "v9",
                       "v10",
                       "v11",
                       "v12",
                       "v13",
                       "v14",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20");
#else
        asm volatile(NCHWC4_TRANS_FP32_COMPUTE NCHWC4_TRANS_FP32_RELU
                         NCHWC4_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [doutc2r0] "+r"(doutc2_ptr),
                       [doutc3r0] "+r"(doutc3_ptr),
                       [ptr_din] "+r"(din_ptr),
                       [cnt] "+r"(cnt_loop)
                     :
                     : "cc", "memory", "q0", "q1", "q2", "q3", "q15");
#endif
        break;
      case lite_api::ActivationType::kRelu6:
/* 0 <= din <= 6 */
#ifdef __aarch64__
        asm volatile(NCHWC4_TRANS_FP32_COMPUTE NCHWC4_TRANS_FP32_RELU
                         NCHWC4_TRANS_FP32_RELU6 NCHWC4_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [doutc2r0] "+r"(doutc2_ptr),
                       [doutc3r0] "+r"(doutc3_ptr),
                       [cnt] "+r"(cnt_loop),
                       [ptr_din] "+r"(din_ptr)
                     : [six] "w"(six)
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v4",
                       "v5",
                       "v6",
                       "v7",
                       "v8",
                       "v9",
                       "v10",
                       "v11",
                       "v12",
                       "v13",
                       "v14",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20");
#else
        asm volatile(NCHWC4_TRANS_FP32_COMPUTE NCHWC4_TRANS_FP32_RELU
                         NCHWC4_TRANS_FP32_RELU6 NCHWC4_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [doutc2r0] "+r"(doutc2_ptr),
                       [doutc3r0] "+r"(doutc3_ptr),
                       [ptr_din] "+r"(din_ptr),
                       [cnt] "+r"(cnt_loop)
                     : [six] "w"(six)
                     : "cc", "memory", "q0", "q1", "q2", "q3", "q15");
#endif
        break;
      case lite_api::ActivationType::kLeakyRelu:
/*din = din >= 0 ? din : din * scale*/
#ifdef __aarch64__
        asm volatile(NCHWC4_TRANS_FP32_COMPUTE NCHWC4_TRANS_FP32_LEAKY_RELU
                         NCHWC4_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [doutc2r0] "+r"(doutc2_ptr),
                       [doutc3r0] "+r"(doutc3_ptr),
                       [cnt] "+r"(cnt_loop),
                       [ptr_din] "+r"(din_ptr)
                     : [scale] "w"(scale)
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v4",
                       "v5",
                       "v6",
                       "v7",
                       "v8",
                       "v9",
                       "v10",
                       "v11",
                       "v12",
                       "v13",
                       "v14",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20");
#else
        asm volatile(NCHWC4_TRANS_FP32_COMPUTE NCHWC4_TRANS_FP32_LEAKY_RELU
                         NCHWC4_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [doutc2r0] "+r"(doutc2_ptr),
                       [doutc3r0] "+r"(doutc3_ptr),
                       [ptr_din] "+r"(din_ptr),
                       [cnt] "+r"(cnt_loop)
                     : [scale] "w"(scale)
                     : "cc",
                       "memory",
                       "q0",
                       "q1",
                       "q2",
                       "q3",
                       "q5",
                       "q6",
                       "q7",
                       "q8",
                       "q9",
                       "q10",
                       "q11",
                       "q12",
                       "q15");
#endif
        break;
      default:
        LOG(FATAL) << "this act_type: "
                   << static_cast<int>(act_param->active_type)
                   << " fuse not support";
    }
  } else {
#ifdef __aarch64__
    asm volatile(NCHWC4_TRANS_FP32_COMPUTE NCHWC4_TRANS_FP32_STORE
                 : [doutc0r0] "+r"(doutc0_ptr),
                   [doutc1r0] "+r"(doutc1_ptr),
                   [doutc2r0] "+r"(doutc2_ptr),
                   [doutc3r0] "+r"(doutc3_ptr),
                   [cnt] "+r"(cnt_loop),
                   [ptr_din] "+r"(din_ptr)
                 :
                 : "cc",
                   "memory",
                   "v0",
                   "v1",
                   "v2",
                   "v3",
                   "v8",
                   "v9",
                   "v10",
                   "v11",
                   "v16",
                   "v17",
                   "v18",
                   "v19");
#else
    asm volatile(NCHWC4_TRANS_FP32_COMPUTE NCHWC4_TRANS_FP32_STORE
                 : [doutc0r0] "+r"(doutc0_ptr),
                   [doutc1r0] "+r"(doutc1_ptr),
                   [doutc2r0] "+r"(doutc2_ptr),
                   [doutc3r0] "+r"(doutc3_ptr),
                   [ptr_din] "+r"(din_ptr),
                   [cnt] "+r"(cnt_loop)
                 :
                 : "cc", "memory", "q0", "q1", "q2", "q3", "q15");
#endif
  }
}
/*wirte result in outputs
* input din: [n, c / 4, h, w * 4], output dout: [n, c, h, w]
*/
inline bool write_to_output_c4_fp32(const float* din,
                                    float* dout,
                                    int cs,
                                    int ce,
                                    int hs,
                                    int he,
                                    int ws,
                                    int we,
                                    int channel,
                                    int height,
                                    int width,
                                    bool flag_relu,
                                    float* trash_ptr,
                                    operators::ActivationParam* act_param) {
  const int c4 = 4;
  const int w4 = 4;
  const int w_round = we - ws;
  const int ch_n = ce - cs;

  if (ch_n != 4) {
    LOG(ERROR) << "write_to_output_c4_fp32 ch_n must be equal 4 and hei_n is "
                  "more than zero";
    return false;
  }
  int size_c_out = width * height;

  float* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
  float* doutc1r0 = doutc0r0 + size_c_out;
  float* doutc2r0 = doutc1r0 + size_c_out;
  float* doutc3r0 = doutc2r0 + size_c_out;

  const float* ptr_din = din;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int valid_we = we > width ? width : we;
  int cnt = (valid_we - ws) / w4;
  int remain = valid_we - ws - cnt * w4;

  for (int i = 0; i < size_h; i++) {
    int size_w = i * width;
    float* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
    float* doutc1_ptr = doutc1r0 + size_w;
    float* doutc2_ptr = doutc2r0 + size_w;
    float* doutc3_ptr = doutc3r0 + size_w;
    if (ce > channel) {
      switch (ce - channel) {
        case 3:
          doutc1_ptr = trash_ptr;
        case 2:
          doutc2_ptr = trash_ptr;
        case 1:
          doutc3_ptr = trash_ptr;
        default:
          break;
      }
    }
    const float* din_hei_ptr = ptr_din + i * w_round * ch_n;
    if (cnt > 0) {
      int cnt_loop = cnt;
      act_switch_c4_fp32(din_hei_ptr,
                         doutc0_ptr,
                         doutc1_ptr,
                         doutc2_ptr,
                         doutc3_ptr,
                         cnt_loop,
                         act_param);
    }
    if (remain > 0) {
      int offset = i * w_round * c4 + c4 * w4 * cnt;
      din_hei_ptr = ptr_din + offset;
      doutc0_ptr += w4 * cnt;
      doutc1_ptr += w4 * cnt;
      doutc2_ptr += w4 * cnt;
      doutc3_ptr += w4 * cnt;
      int j = 0;
      if (act_param != nullptr && act_param->has_active) {
        float six = act_param->Relu_clipped_coef;
        float scale = act_param->Leaky_relu_alpha;
        switch (act_param->active_type) {
          case lite_api::ActivationType::kRelu:
            for (; j < remain; ++j) {
              *(doutc0_ptr++) = LITEMAX(din_hei_ptr[0], 0.f);
              *(doutc1_ptr++) = LITEMAX(din_hei_ptr[1], 0.f);
              *(doutc2_ptr++) = LITEMAX(din_hei_ptr[2], 0.f);
              *(doutc3_ptr++) = LITEMAX(din_hei_ptr[3], 0.f);
              din_hei_ptr += 4;
            }
            break;
          case lite_api::ActivationType::kRelu6:
            /* 0 <= din <= 6 */
            for (; j < remain; ++j) {
              float tmp1 = LITEMAX(din_hei_ptr[0], 0.f);
              float tmp2 = LITEMAX(din_hei_ptr[1], 0.f);
              float tmp3 = LITEMAX(din_hei_ptr[2], 0.f);
              float tmp4 = LITEMAX(din_hei_ptr[3], 0.f);
              *(doutc0_ptr++) = LITEMIN(tmp1, six);
              *(doutc1_ptr++) = LITEMIN(tmp2, six);
              *(doutc2_ptr++) = LITEMIN(tmp3, six);
              *(doutc3_ptr++) = LITEMIN(tmp4, six);
              din_hei_ptr += 4;
            }
            break;
          case lite_api::ActivationType::kLeakyRelu:
            /*din = din >= 0 ? din : din * scale*/
            for (; j < remain; ++j) {
              if (din_hei_ptr[0] >= 0) {
                *(doutc0_ptr++) = din_hei_ptr[0];
              } else {
                *(doutc0_ptr++) = din_hei_ptr[0] * scale;
              }
              if (din_hei_ptr[1] >= 0) {
                *(doutc1_ptr++) = din_hei_ptr[1];
              } else {
                *(doutc1_ptr++) = din_hei_ptr[1] * scale;
              }
              if (din_hei_ptr[2] >= 0) {
                *(doutc2_ptr++) = din_hei_ptr[2];
              } else {
                *(doutc2_ptr++) = din_hei_ptr[2] * scale;
              }
              if (din_hei_ptr[3] >= 0) {
                *(doutc3_ptr++) = din_hei_ptr[3];
              } else {
                *(doutc3_ptr++) = din_hei_ptr[3] * scale;
              }
              din_hei_ptr += 4;
            }
            break;
          default:
            LOG(FATAL) << "this act_type: "
                       << static_cast<int>(act_param->active_type)
                       << " fuse not support";
        }
      } else {
        for (; j < remain; ++j) {
          *(doutc0_ptr++) = din_hei_ptr[0];
          *(doutc1_ptr++) = din_hei_ptr[1];
          *(doutc2_ptr++) = din_hei_ptr[2];
          *(doutc3_ptr++) = din_hei_ptr[3];
          din_hei_ptr += 4;
        }
      }
    }
  }
  return true;
}
// clang-format off
#ifdef __aarch64__
#define NCHWC8_TRANS_FP32_COMPUTE                                    \
  "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */  \
  "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */  \
  "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */  \
  "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */  \
  "movi v20.4s, #0                \n" /* for relu */                 \
  "1:                             \n" /* main loop*/                 \
  "trn1   v8.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/              \
  "trn2   v9.4s, v0.4s, v2.4s     \n" /* trans q0, q1*/              \
  "trn1   v10.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/              \
  "trn2   v11.4s, v1.4s, v3.4s    \n" /* trans q2, q3*/              \
  "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */  \
                                                                     \
  "trn1   v12.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/              \
  "trn2   v13.4s, v4.4s, v6.4s    \n" /* trans q0, q1*/              \
  "trn1   v14.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/              \
  "trn2   v15.4s, v5.4s, v7.4s    \n" /* trans q2, q3*/              \
  "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */  \
                                                                     \
  "trn1   v16.2d, v8.2d, v12.2d   \n" /* trans q8, q10 00 01 02 03*/ \
  "trn2   v17.2d, v8.2d, v12.2d   \n" /* trans q8, q10 20 21 22 23*/ \
  "trn1   v18.2d, v9.2d, v13.2d   \n" /* trans q9, q11 10 11 12 13*/ \
  "trn2   v19.2d, v9.2d, v13.2d   \n" /* trans q9, q11 30 31 32 33*/ \
  "ldp q4, q5, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */  \
                                                                     \
  "trn1   v8.2d, v10.2d, v14.2d   \n" /* trans q8, q10 40 41 42 43*/ \
  "trn2   v9.2d, v10.2d, v14.2d   \n" /* trans q8, q10 60 61 62 63*/ \
  "trn1   v12.2d, v11.2d, v15.2d  \n" /* trans q9, q11 50 51 52 53*/ \
  "trn2   v13.2d, v11.2d, v15.2d  \n" /* trans q9, q11 70 71 72 73*/ \
  "ldp q6, q7, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */

#define NCHWC8_TRANS_FP32_RELU                 \
  "fmax   v16.4s, v16.4s, v20.4s  \n" /*relu*/ \
  "fmax   v17.4s, v17.4s, v20.4s  \n" /*relu*/ \
  "fmax   v18.4s, v18.4s, v20.4s  \n" /*relu*/ \
  "fmax   v19.4s, v19.4s, v20.4s  \n" /*relu*/ \
                                               \
  "fmax   v8.4s,  v8.4s,  v20.4s  \n" /*relu*/ \
  "fmax   v9.4s,  v9.4s,  v20.4s  \n" /*relu*/ \
  "fmax   v12.4s, v12.4s, v20.4s  \n" /*relu*/ \
  "fmax   v13.4s, v13.4s, v20.4s  \n" /*relu*/

#define NCHWC8_TRANS_FP32_RELU6                    \
  "fmin   v16.4s, v16.4s, %[six].4s  \n" /*relu6*/ \
  "fmin   v17.4s, v17.4s, %[six].4s  \n" /*relu6*/ \
  "fmin   v18.4s, v18.4s, %[six].4s  \n" /*relu6*/ \
  "fmin   v19.4s, v19.4s, %[six].4s  \n" /*relu6*/ \
                                                   \
  "fmin   v8.4s,  v8.4s,  %[six].4s  \n" /*relu6*/ \
  "fmin   v9.4s,  v9.4s,  %[six].4s  \n" /*relu6*/ \
  "fmin   v12.4s, v12.4s, %[six].4s  \n" /*relu6*/ \
  "fmin   v13.4s, v13.4s, %[six].4s  \n" /*relu6*/

#define NCHWC8_TRANS_FP32_LEAKY_RELU                \
  "fcmge v10.4s, v16.4s, v20.4s \n" /* vcgeq_u32 */ \
  "fcmge v11.4s, v17.4s, v20.4s \n" /* vcgeq_u32 */ \
  "fcmge v14.4s, v18.4s, v20.4s \n" /* vcgeq_u32 */ \
  "fcmge v15.4s, v19.4s, v20.4s \n" /* vcgeq_u32 */ \
                                                    \
  "fcmge v21.4s, v8.4s, v20.4s \n"  /* vcgeq_u32 */ \
  "fcmge v22.4s, v9.4s, v20.4s \n"  /* vcgeq_u32 */ \
  "fcmge v23.4s, v12.4s, v20.4s \n" /* vcgeq_u32 */ \
  "fcmge v24.4s, v13.4s, v20.4s \n" /* vcgeq_u32 */ \
                                                    \
  "fmul v25.4s, v16.4s, %[scale].4s \n" /* mul */   \
  "fmul v26.4s, v17.4s, %[scale].4s \n" /* mul */   \
  "fmul v27.4s, v18.4s, %[scale].4s \n" /* mul */   \
  "fmul v28.4s, v19.4s, %[scale].4s \n" /* mul */   \
                                                    \
  "fmul v29.4s, v8.4s, %[scale].4s \n"  /* mul */   \
  "fmul v30.4s, v9.4s, %[scale].4s \n"  /* mul */   \
  "fmul v31.4s, v12.4s, %[scale].4s \n" /* mul */   \
                                                    \
  "bif  v16.16b, v25.16b, v10.16b \n"   /* choose*/ \
  "bif  v17.16b, v26.16b, v11.16b \n"   /* choose*/ \
  "bif  v18.16b, v27.16b, v14.16b \n"   /* choose*/ \
  "bif  v19.16b, v28.16b, v15.16b \n"   /* choose*/ \
  "fmul v25.4s, v13.4s, %[scale].4s \n" /* mul */   \
                                                    \
  "bif  v8.16b, v29.16b, v21.16b \n"  /* choose*/   \
  "bif  v9.16b, v30.16b, v22.16b \n"  /* choose*/   \
  "bif  v12.16b, v31.16b, v23.16b \n" /* choose*/   \
  "bif  v13.16b, v25.16b, v24.16b \n" /* choose*/

#define NCHWC8_TRANS_FP32_STORE                          \
  "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0*/    \
  "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0*/    \
  "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0*/    \
  "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0*/    \
                                                         \
  "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1*/ \
  "str    q8,  [%[doutc4r0]], #16 \n" /* store c0r0*/    \
  "str    q9,  [%[doutc6r0]], #16 \n" /* store c2r0*/    \
  "str    q12, [%[doutc5r0]], #16 \n" /* store c1r0*/    \
  "str    q13, [%[doutc7r0]], #16 \n" /* store c3r0*/    \
                                                         \
  "bne    1b                      \n" /* jump to main loop*/

#else
#define NCHWC8_TRANS_FP32_COMPUTE                           \
  "vld1.32 {d0-d3}, [%[ptr_din]]!        @load data \n"     \
  "vld1.32 {d4-d7}, [%[ptr_din]]!        @load data \n"     \
  "vld1.32 {d8-d11}, [%[ptr_din]]!       @load data \n"     \
  "vld1.32 {d12-d15}, [%[ptr_din]]!      @load data \n"     \
  "vmov.u32 q15, #0                      @ dump zero\n"     \
  "1:                                    @ main loop\n"     \
  "vtrn.32   q0, q2                      @ trans q0, q2 \n" \
  "vtrn.32   q4, q6                      @ trans q4, q6 \n" \
  "vswp.32   d1, d8                      @ swap  d1, d8 \n" \
  "vswp.32   d5, d12                     @ swap  d5, d12\n" \
                                                            \
  "vtrn.32   q1, q3                      @ trans q1, q3 \n" \
  "vtrn.32   q5, q7                      @ trans q5, q7 \n" \
  "vswp.32   d3, d10                     @ swap  d3, d10\n" \
  "vswp.32   d7, d14                     @ swap  d7, d14\n"

#define NCHWC8_TRANS_FP32_RELU                     \
  "vmax.f32  q0, q0, q15                 @ relu\n" \
  "vmax.f32  q1, q1, q15                 @ relu\n" \
  "vmax.f32  q2, q2, q15                 @ relu\n" \
  "vmax.f32  q3, q3, q15                 @ relu\n" \
                                                   \
  "vmax.f32  q4, q4, q15                 @ relu\n" \
  "vmax.f32  q5, q5, q15                 @ relu\n" \
  "vmax.f32  q6, q6, q15                 @ relu\n" \
  "vmax.f32  q7, q7, q15                 @ relu\n"

#define NCHWC8_TRANS_FP32_RELU6                         \
  "vmin.f32  q0, q0, %q[six]                 @ relu6\n" \
  "vmin.f32  q1, q1, %q[six]                 @ relu6\n" \
  "vmin.f32  q2, q2, %q[six]                 @ relu6\n" \
  "vmin.f32  q3, q3, %q[six]                 @ relu6\n" \
                                                        \
  "vmin.f32  q4, q4, %q[six]                 @ relu6\n" \
  "vmin.f32  q5, q5, %q[six]                 @ relu6\n" \
  "vmin.f32  q6, q6, %q[six]                 @ relu6\n" \
  "vmin.f32  q7, q7, %q[six]                 @ relu6\n"

#define NCHWC8_TRANS_FP32_LEAKY_RELU           \
  "vcge.f32   q9, q0,  q15        @ q0 > 0 \n" \
  "vcge.f32   q10, q1, q15        @ q0 > 0 \n" \
  "vcge.f32   q11, q2, q15        @ q0 > 0 \n" \
  "vcge.f32   q12, q3, q15        @ q0 > 0 \n" \
  "vmul.f32 q13, q0, %q[scale] \n"             \
  "vmul.f32 q14, q1, %q[scale] \n"             \
  "vmul.f32 q15, q2, %q[scale] \n"             \
                                               \
  "vbif q0, q13, q9 @ choose \n"               \
  "vmul.f32 q9, q3, %q[scale] \n"              \
                                               \
  "vbif q1, q14, q10 @ choose \n"              \
  "vbif q2, q15, q11 @ choose \n"              \
  "vbif q3, q9, q12 @ choose \n"               \
                                               \
  "vcge.f32   q9, q4, q15        @ q0 > 0 \n"  \
  "vcge.f32   q10, q5, q15        @ q0 > 0 \n" \
  "vcge.f32   q11, q6, q15        @ q0 > 0 \n" \
  "vcge.f32   q12, q7, q15        @ q0 > 0 \n" \
  "vmul.f32 q13, q4, %q[scale] \n"             \
  "vmul.f32 q14, q5, %q[scale] \n"             \
  "vmul.f32 q15, q6, %q[scale] \n"             \
                                               \
  "vbif q4, q13, q9 @ choose \n"               \
  "vmul.f32 q9, q7, %q[scale] \n"              \
                                               \
  "vbif q5, q14, q10 @ choose \n"              \
  "vbif q6, q15, q11 @ choose \n"              \
  "vbif q7, q9, q12 @ choose \n"

#define NCHWC8_TRANS_FP32_STORE                                \
  "subs   %[cnt], %[cnt], #1             @ loop count - 1\n"   \
  "vst1.32   {d0-d1}, [%[doutc0r0]]!     @ store result, add " \
  "pointer\n"                                                  \
  "vst1.32   {d2-d3}, [%[doutc4r0]]!     @ store result, add " \
  "pointer\n"                                                  \
  "vst1.32   {d4-d5}, [%[doutc1r0]]!     @ store result, add " \
  "pointer\n"                                                  \
  "vst1.32   {d6-d7}, [%[doutc5r0]]!     @ store result, add " \
  "pointer\n"                                                  \
                                                               \
  "vld1.32   {d0-d3}, [%[ptr_din]]!      @load data \n"        \
  "vld1.32   {d4-d7}, [%[ptr_din]]!      @load data \n"        \
                                                               \
  "vst1.32   {d8-d9},   [%[doutc2r0]]!   @ store result, add " \
  "pointer\n"                                                  \
  "vst1.32   {d10-d11}, [%[doutc6r0]]!   @ store result, add " \
  "pointer\n"                                                  \
  "vst1.32   {d12-d13}, [%[doutc3r0]]!   @ store result, add " \
  "pointer\n"                                                  \
  "vst1.32   {d14-d15}, [%[doutc7r0]]!   @ store result, add " \
  "pointer\n"                                                  \
                                                               \
  "vld1.32 {d8-d11}, [%[ptr_din]]!       @load data \n"        \
  "vld1.32 {d12-d15}, [%[ptr_din]]!      @load data \n"        \
                                                               \
  "bne    1b                             @ jump to main loop\n"

#endif
// clang-format on
inline void act_switch_c8_fp32(const float* din_ptr,
                               float* doutc0_ptr,
                               float* doutc1_ptr,
                               float* doutc2_ptr,
                               float* doutc3_ptr,
                               float* doutc4_ptr,
                               float* doutc5_ptr,
                               float* doutc6_ptr,
                               float* doutc7_ptr,
                               int cnt_loop,
                               const operators::ActivationParam* act_param) {
  if (act_param != nullptr && act_param->has_active) {
    float32x4_t six = vdupq_n_f32(act_param->Relu_clipped_coef);
    float32x4_t scale = vdupq_n_f32(act_param->Leaky_relu_alpha);
    switch (act_param->active_type) {
      case lite_api::ActivationType::kRelu:
#ifdef __aarch64__
        asm volatile(NCHWC8_TRANS_FP32_COMPUTE NCHWC8_TRANS_FP32_RELU
                         NCHWC8_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [doutc2r0] "+r"(doutc2_ptr),
                       [doutc3r0] "+r"(doutc3_ptr),
                       [doutc4r0] "+r"(doutc4_ptr),
                       [doutc5r0] "+r"(doutc5_ptr),
                       [doutc6r0] "+r"(doutc6_ptr),
                       [doutc7r0] "+r"(doutc7_ptr),
                       [cnt] "+r"(cnt_loop),
                       [ptr_din] "+r"(din_ptr)
                     :
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v4",
                       "v5",
                       "v6",
                       "v7",
                       "v8",
                       "v9",
                       "v10",
                       "v11",
                       "v12",
                       "v13",
                       "v14",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20");
#else
        asm volatile(NCHWC8_TRANS_FP32_COMPUTE NCHWC8_TRANS_FP32_RELU
                         NCHWC8_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [doutc2r0] "+r"(doutc2_ptr),
                       [doutc3r0] "+r"(doutc3_ptr),
                       [doutc4r0] "+r"(doutc4_ptr),
                       [doutc5r0] "+r"(doutc5_ptr),
                       [doutc6r0] "+r"(doutc6_ptr),
                       [doutc7r0] "+r"(doutc7_ptr),
                       [ptr_din] "+r"(din_ptr),
                       [cnt] "+r"(cnt_loop)
                     :
                     : "cc",
                       "memory",
                       "q0",
                       "q1",
                       "q2",
                       "q3",
                       "q4",
                       "q5",
                       "q6",
                       "q7",
                       "q15");
#endif
        break;
      case lite_api::ActivationType::kRelu6:
/* 0 <= din <= 6 */
#ifdef __aarch64__
        asm volatile(NCHWC8_TRANS_FP32_COMPUTE NCHWC8_TRANS_FP32_RELU6
                         NCHWC8_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [doutc2r0] "+r"(doutc2_ptr),
                       [doutc3r0] "+r"(doutc3_ptr),
                       [doutc4r0] "+r"(doutc4_ptr),
                       [doutc5r0] "+r"(doutc5_ptr),
                       [doutc6r0] "+r"(doutc6_ptr),
                       [doutc7r0] "+r"(doutc7_ptr),
                       [cnt] "+r"(cnt_loop),
                       [ptr_din] "+r"(din_ptr)
                     : [six] "w"(six)
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v4",
                       "v5",
                       "v6",
                       "v7",
                       "v8",
                       "v9",
                       "v10",
                       "v11",
                       "v12",
                       "v13",
                       "v14",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20");
#else
        asm volatile(NCHWC4_TRANS_FP32_COMPUTE NCHWC4_TRANS_FP32_RELU
                         NCHWC4_TRANS_FP32_RELU6 NCHWC4_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [doutc2r0] "+r"(doutc2_ptr),
                       [doutc3r0] "+r"(doutc3_ptr),
                       [ptr_din] "+r"(din_ptr),
                       [cnt] "+r"(cnt_loop)
                     : [six] "w"(six)
                     : "cc",
                       "memory",
                       "q0",
                       "q1",
                       "q2",
                       "q3",
                       "q4",
                       "q5",
                       "q6",
                       "q7",
                       "q15");
#endif
        break;
      case lite_api::ActivationType::kLeakyRelu:
/*din = din >= 0 ? din : din * scale*/
#ifdef __aarch64__
        asm volatile(NCHWC8_TRANS_FP32_COMPUTE NCHWC8_TRANS_FP32_LEAKY_RELU
                         NCHWC8_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [doutc2r0] "+r"(doutc2_ptr),
                       [doutc3r0] "+r"(doutc3_ptr),
                       [doutc4r0] "+r"(doutc4_ptr),
                       [doutc5r0] "+r"(doutc5_ptr),
                       [doutc6r0] "+r"(doutc6_ptr),
                       [doutc7r0] "+r"(doutc7_ptr),
                       [cnt] "+r"(cnt_loop),
                       [ptr_din] "+r"(din_ptr)
                     : [scale] "w"(scale)
                     : "cc",
                       "memory",
                       "v0",
                       "v1",
                       "v2",
                       "v3",
                       "v4",
                       "v5",
                       "v6",
                       "v7",
                       "v8",
                       "v9",
                       "v10",
                       "v11",
                       "v12",
                       "v13",
                       "v14",
                       "v16",
                       "v17",
                       "v18",
                       "v19",
                       "v20",
                       "v21",
                       "v22",
                       "v23",
                       "v24",
                       "v25",
                       "v26",
                       "v27",
                       "v28",
                       "v29",
                       "v30",
                       "v31");
#else
        asm volatile(NCHWC8_TRANS_FP32_COMPUTE NCHWC8_TRANS_FP32_LEAKY_RELU
                         NCHWC8_TRANS_FP32_STORE
                     : [doutc0r0] "+r"(doutc0_ptr),
                       [doutc1r0] "+r"(doutc1_ptr),
                       [doutc2r0] "+r"(doutc2_ptr),
                       [doutc3r0] "+r"(doutc3_ptr),
                       [doutc4r0] "+r"(doutc4_ptr),
                       [doutc5r0] "+r"(doutc5_ptr),
                       [doutc6r0] "+r"(doutc6_ptr),
                       [doutc7r0] "+r"(doutc7_ptr),
                       [ptr_din] "+r"(din_ptr),
                       [cnt] "+r"(cnt_loop)
                     : [scale] "w"(scale)
                     : "cc",
                       "memory",
                       "q0",
                       "q1",
                       "q2",
                       "q3",
                       "q4",
                       "q5",
                       "q6",
                       "q7",
                       "q9",
                       "q10",
                       "q11",
                       "q12",
                       "q13",
                       "q14",
                       "q15");
#endif
        break;
      default:
        LOG(FATAL) << "this act_type: "
                   << static_cast<int>(act_param->active_type)
                   << " fuse not support";
    }
  } else {
#ifdef __aarch64__
    asm volatile(NCHWC8_TRANS_FP32_COMPUTE NCHWC8_TRANS_FP32_STORE
                 : [doutc0r0] "+r"(doutc0_ptr),
                   [doutc1r0] "+r"(doutc1_ptr),
                   [doutc2r0] "+r"(doutc2_ptr),
                   [doutc3r0] "+r"(doutc3_ptr),
                   [doutc4r0] "+r"(doutc4_ptr),
                   [doutc5r0] "+r"(doutc5_ptr),
                   [doutc6r0] "+r"(doutc6_ptr),
                   [doutc7r0] "+r"(doutc7_ptr),
                   [cnt] "+r"(cnt_loop),
                   [ptr_din] "+r"(din_ptr)
                 :
                 : "cc",
                   "memory",
                   "v0",
                   "v1",
                   "v2",
                   "v3",
                   "v4",
                   "v5",
                   "v6",
                   "v7",
                   "v8",
                   "v9",
                   "v10",
                   "v11",
                   "v12",
                   "v13",
                   "v14",
                   "v15",
                   "v16",
                   "v17",
                   "v18",
                   "v19",
                   "v20");
#else
    asm volatile(NCHWC8_TRANS_FP32_COMPUTE NCHWC8_TRANS_FP32_STORE
                 : [doutc0r0] "+r"(doutc0_ptr),
                   [doutc1r0] "+r"(doutc1_ptr),
                   [doutc2r0] "+r"(doutc2_ptr),
                   [doutc3r0] "+r"(doutc3_ptr),
                   [doutc4r0] "+r"(doutc4_ptr),
                   [doutc5r0] "+r"(doutc5_ptr),
                   [doutc6r0] "+r"(doutc6_ptr),
                   [doutc7r0] "+r"(doutc7_ptr),
                   [ptr_din] "+r"(din_ptr),
                   [cnt] "+r"(cnt_loop)
                 :
                 : "cc",
                   "memory",
                   "q0",
                   "q1",
                   "q2",
                   "q3",
                   "q4",
                   "q5",
                   "q6",
                   "q7",
                   "q15");
#endif
  }
}

#ifdef __aarch64__
#define LOAD_DATA                                               \
  "1:                               \n"                         \
  "ld1 {v0.4s}, [%[din_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/ \
  "ld1 {v1.4s}, [%[din_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/ \
  "ld1 {v2.4s}, [%[din_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/ \
  "ld1 {v3.4s}, [%[din_ptr]], #16   \n" /*vld1q_f32(din_ptr0)*/
#define DO_RELU                                           \
  "fmax v0.4s, v0.4s, %[vzero].4s   \n" /* vmaxq_f32() */ \
  "fmax v1.4s, v1.4s, %[vzero].4s   \n" /* vmaxq_f32() */ \
  "fmax v2.4s, v2.4s, %[vzero].4s   \n" /* vmaxq_f32() */ \
  "fmax v3.4s, v3.4s, %[vzero].4s   \n" /* vmaxq_f32() */
#define DO_RELU6                                         \
  "fmin v0.4s, v0.4s, %[vsix].4s   \n" /* vmaxq_f32() */ \
  "fmin v1.4s, v1.4s, %[vsix].4s   \n" /* vmaxq_f32() */ \
  "fmin v2.4s, v2.4s, %[vsix].4s   \n" /* vmaxq_f32() */ \
  "fmin v3.4s, v3.4s, %[vsix].4s   \n" /* vmaxq_f32() */
#define DO_LEAKY_RELU                                    \
  "fcmge v4.4s, v0.4s,  %[vzero].4s  \n" /* vcgeq_f32 */ \
  "fmul v5.4s, v0.4s, %[vscale].4s   \n" /* vmulq_f32 */ \
  "fcmge v6.4s, v1.4s,  %[vzero].4s  \n" /* vcgeq_f32 */ \
  "fmul v7.4s, v1.4s, %[vscale].4s   \n" /* vmulq_f32 */ \
  "fcmge v8.4s, v2.4s,  %[vzero].4s  \n" /* vcgeq_f32 */ \
  "fmul v9.4s, v2.4s, %[vscale].4s   \n" /* vmulq_f32 */ \
  "fcmge v10.4s, v3.4s,  %[vzero].4s \n" /* vcgeq_f32 */ \
  "fmul v11.4s, v3.4s, %[vscale].4s  \n" /* vmulq_f32 */ \
  "bif v0.16b, v5.16b, v4.16b        \n" /* choose*/     \
  "bif v1.16b, v7.16b, v6.16b        \n" /* choose*/     \
  "bif v2.16b, v9.16b, v8.16b        \n" /* choose*/     \
  "bif v3.16b, v11.16b, v10.16b      \n" /* choose*/
#define DO_STORE                                         \
  "subs %w[cnt], %w[cnt], #1                    \n"      \
  "st1 {v0.4s}, [%[dout_ptr]], #16 \n" /* vst1q_f32() */ \
  "st1 {v1.4s}, [%[dout_ptr]], #16 \n" /* vst1q_f32() */ \
  "st1 {v2.4s}, [%[dout_ptr]], #16 \n" /* vst1q_f32() */ \
  "st1 {v3.4s}, [%[dout_ptr]], #16 \n" /* vst1q_f32() */ \
  "bne  1b                                    \n"
#else
#define LOAD_DATA                                            \
  "1:                               \n"                      \
  "vld1.32 {d6-d7}, [%[din_ptr]]!   @ vld1q_f32(din_ptr) \n" \
  "vld1.32 {d8-d9}, [%[din_ptr]]!   @ vld1q_f32(din_ptr) \n" \
  "vld1.32 {d10-d11}, [%[din_ptr]]! @ vld1q_f32(din_ptr) \n" \
  "vld1.32 {d12-d13}, [%[din_ptr]]! @ vld1q_f32(din_ptr) \n"
#define DO_RELU                                 \
  "vmax.f32 q3, q3, %q[vzero] @ vmaxq_f32() \n" \
  "vmax.f32 q4, q4, %q[vzero] @ vmaxq_f32() \n" \
  "vmax.f32 q5, q5, %q[vzero] @ vmaxq_f32() \n" \
  "vmax.f32 q6, q6, %q[vzero] @ vmaxq_f32() \n"
#define DO_RELU6                               \
  "vmin.f32 q3, q3, %q[vsix] @ vminq_f32() \n" \
  "vmin.f32 q4, q4, %q[vsix] @ vmaxq_f32() \n" \
  "vmin.f32 q5, q5, %q[vsix] @ vmaxq_f32() \n" \
  "vmin.f32 q6, q6, %q[vsix] @ vmaxq_f32() \n"
#define DO_LEAKY_RELU                            \
  "vcge.f32 q7, q3, %q[vzero]   @ vcgeq_u32 \n"  \
  "vmul.f32 q8, q3, %q[vscale]  @ vmulq_f32 \n"  \
  "vcge.f32 q9, q4, %q[vzero]   @ vcgeq_u32 \n"  \
  "vmul.f32 q10, q4, %q[vscale]  @ vmulq_f32 \n" \
  "vcge.f32 q11, q5, %q[vzero]   @ vcgeq_u32 \n" \
  "vmul.f32 q12, q5, %q[vscale]  @ vmulq_f32 \n" \
  "vcge.f32 q13, q6, %q[vzero]   @ vcgeq_u32 \n" \
  "vmul.f32 q14, q6, %q[vscale]  @ vmulq_f32 \n" \
  "vbif q3, q8, q7               @ choose \n"    \
  "vbif q4, q10, q9              @ choose \n"    \
  "vbif q5, q12, q11             @ choose \n"    \
  "vbif q6, q14, q13             @ choose \n"
#define DO_STORE                                            \
  "subs %[cnt], #1                                \n"       \
  "vst1.32 {d6-d7}, [%[dout_ptr]]!       @ vst1q_f32()  \n" \
  "vst1.32 {d8-d9}, [%[dout_ptr]]!       @ vst1q_f32()  \n" \
  "vst1.32 {d10-d11}, [%[dout_ptr]]!     @ vst1q_f32()  \n" \
  "vst1.32 {d12-d13}, [%[dout_ptr]]!     @ vst1q_f32()  \n" \
  "bne  1b                                    \n"
#endif
/*
* Data do activation process
* Now support relu relu6 leakyrelu act
*/
inline void act_switch_process(float* src,
                               float* dst,
                               int size,
                               const operators::ActivationParam* act_param) {
  int cnt = size >> 4;
  int remain = size % 16;
  float32x4_t vzero = vdupq_n_f32(0.f);
  if (act_param != nullptr) {
    float32x4_t vsix = vdupq_n_f32(act_param->Relu_clipped_coef);
    float32x4_t vscale = vdupq_n_f32(act_param->Leaky_relu_alpha);
    if (cnt > 0) {
      switch (act_param->active_type) {
        case lite_api::ActivationType::kRelu:
#ifdef __aarch64__
          asm volatile(
              LOAD_DATA DO_RELU DO_STORE
              : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
              : [vzero] "w"(vzero)
              : "memory", "cc", "v0", "v1", "v2", "v3");
#else
          asm volatile(
              LOAD_DATA DO_RELU DO_STORE
              : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
              : [vzero] "w"(vzero)
              : "memory", "cc", "q3", "q4", "q5", "q6");
#endif
          break;
        case lite_api::ActivationType::kRelu6:
#ifdef __aarch64__
          asm volatile(
              LOAD_DATA DO_RELU DO_RELU6 DO_STORE
              : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
              : [vzero] "w"(vzero), [vsix] "w"(vsix)
              : "memory", "cc", "v0", "v1", "v2", "v3");
#else
          asm volatile(
              LOAD_DATA DO_RELU DO_RELU6 DO_STORE
              : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
              : [vzero] "w"(vzero), [vsix] "w"(vsix)
              : "memory", "cc", "q3", "q4", "q5", "q6");
#endif
          break;
        case lite_api::ActivationType::kLeakyRelu:
#ifdef __aarch64__
          asm volatile(
              LOAD_DATA DO_LEAKY_RELU DO_STORE
              : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
              : [vzero] "w"(vzero), [vscale] "w"(vscale)
              : "memory",
                "cc",
                "v0",
                "v1",
                "v2",
                "v3",
                "v4",
                "v5",
                "v6",
                "v7",
                "v8",
                "v9",
                "v10",
                "v11");
#else
          asm volatile(
              LOAD_DATA DO_LEAKY_RELU DO_STORE
              : [din_ptr] "+r"(src), [dout_ptr] "+r"(dst), [cnt] "+r"(cnt)
              : [vzero] "w"(vzero), [vscale] "w"(vscale)
              : "memory",
                "cc",
                "q3",
                "q4",
                "q5",
                "q6",
                "q7",
                "q8",
                "q9",
                "q10",
                "q11",
                "q12",
                "q13",
                "q14");
#endif
          break;
        default:
          LOG(FATAL) << "this act_type: "
                     << static_cast<int>(act_param->active_type)
                     << " fuse not support";
      }
    }
    // remain
    switch (act_param->active_type) {
      case lite_api::ActivationType::kRelu:
        for (int i = 0; i < remain; i++) {
          *dst = *src >= 0.f ? *src : 0.f;
          src++;
          dst++;
        }
        break;
      case lite_api::ActivationType::kRelu6:
        for (int i = 0; i < remain; i++) {
          float tmp = *src >= 0.f ? *src : 0.f;
          *dst = tmp <= act_param->Relu_clipped_coef
                     ? tmp
                     : act_param->Relu_clipped_coef;
          src++;
          dst++;
        }
        break;
      case lite_api::ActivationType::kLeakyRelu:
        for (int i = 0; i < remain; i++) {
          if (*src >= 0.f) {
            *dst = *src;
          } else {
            *dst = *src * act_param->Leaky_relu_alpha;
          }
          src++;
          dst++;
        }
        break;
      default:
        LOG(FATAL) << "this act_type: "
                   << static_cast<int>(act_param->active_type)
                   << " fuse not support";
    }
  }
}

/*wirte result in outputs
* input din: [n, c / 8, h, w * 8], output dout: [n, c, h, w]
*/
inline bool write_to_output_c8_fp32(const float* din,
                                    float* dout,
                                    int ch_n,
                                    int hei_n,
                                    int cs,
                                    int ce,
                                    int hs,
                                    int he,
                                    int ws,
                                    int we,
                                    int channel,
                                    int height,
                                    int width,
                                    bool flag_relu,
                                    float* trash_ptr,
                                    operators::ActivationParam* act_param) {
  if (ch_n != 8 || hei_n <= 0) {
    LOG(ERROR) << "ch_n must be equal 8 and hei_n is more than zero";
    return false;
  }
  int size_c_out = width * height;

  float* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
  float* doutc1r0 = doutc0r0 + size_c_out;
  float* doutc2r0 = doutc1r0 + size_c_out;
  float* doutc3r0 = doutc2r0 + size_c_out;
  float* doutc4r0 = doutc3r0 + size_c_out;
  float* doutc5r0 = doutc4r0 + size_c_out;
  float* doutc6r0 = doutc5r0 + size_c_out;
  float* doutc7r0 = doutc6r0 + size_c_out;

  const float* ptr_din = din;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int valid_w = we - ws;
  int w4 = 4;
  int cnt = valid_w / 4;

  if (we > width) {
    cnt--;
  }
  for (int i = 0; i < size_h; i++) {
    int size_w = i * width;
    float* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
    float* doutc1_ptr = doutc1r0 + size_w;
    float* doutc2_ptr = doutc2r0 + size_w;
    float* doutc3_ptr = doutc3r0 + size_w;
    float* doutc4_ptr = doutc4r0 + size_w;
    float* doutc5_ptr = doutc5r0 + size_w;
    float* doutc6_ptr = doutc6r0 + size_w;
    float* doutc7_ptr = doutc7r0 + size_w;
    if (ce > channel) {
      switch (ce - channel) {
        case 7:
          doutc1_ptr = trash_ptr;
        case 6:
          doutc2_ptr = trash_ptr;
        case 5:
          doutc3_ptr = trash_ptr;
        case 4:
          doutc4_ptr = trash_ptr;
        case 3:
          doutc5_ptr = trash_ptr;
        case 2:
          doutc6_ptr = trash_ptr;
        case 1:
          doutc7_ptr = trash_ptr;
        default:
          break;
      }
    }
    ptr_din = din + i * valid_w * ch_n;
    const float* din_hei_ptr = ptr_din;
    if (cnt > 0) {
      int cnt_loop = cnt;
      act_switch_c8_fp32(din_hei_ptr,
                         doutc0_ptr,
                         doutc1_ptr,
                         doutc2_ptr,
                         doutc3_ptr,
                         doutc4_ptr,
                         doutc5_ptr,
                         doutc6_ptr,
                         doutc7_ptr,
                         cnt_loop,
                         act_param);
    }
    if (we > width) {
      int offset = 32 * (valid_w / 4 - 1);
      din_hei_ptr = ptr_din + offset;
      doutc0_ptr += w4 * cnt;
      doutc1_ptr += w4 * cnt;
      doutc2_ptr += w4 * cnt;
      doutc3_ptr += w4 * cnt;
      doutc4_ptr += w4 * cnt;
      doutc5_ptr += w4 * cnt;
      doutc6_ptr += w4 * cnt;
      doutc7_ptr += w4 * cnt;
      int i = we - 4;
      if (act_param != nullptr && act_param->has_active) {
        float six = act_param->Relu_clipped_coef;
        float scale = act_param->Leaky_relu_alpha;
        switch (act_param->active_type) {
          case lite_api::ActivationType::kRelu:
            for (; i < width; ++i) {
              *(doutc0_ptr++) = LITEMAX(din_hei_ptr[0], 0.f);
              *(doutc1_ptr++) = LITEMAX(din_hei_ptr[1], 0.f);
              *(doutc2_ptr++) = LITEMAX(din_hei_ptr[2], 0.f);
              *(doutc3_ptr++) = LITEMAX(din_hei_ptr[3], 0.f);
              *(doutc4_ptr++) = LITEMAX(din_hei_ptr[4], 0.f);
              *(doutc5_ptr++) = LITEMAX(din_hei_ptr[5], 0.f);
              *(doutc6_ptr++) = LITEMAX(din_hei_ptr[6], 0.f);
              *(doutc7_ptr++) = LITEMAX(din_hei_ptr[7], 0.f);
              din_hei_ptr += 8;
            }
            break;
          case lite_api::ActivationType::kRelu6:
            /* 0 <= din <= 6 */
            for (; i < width; ++i) {
              float tmp1 = LITEMAX(din_hei_ptr[0], 0.f);
              float tmp2 = LITEMAX(din_hei_ptr[1], 0.f);
              float tmp3 = LITEMAX(din_hei_ptr[2], 0.f);
              float tmp4 = LITEMAX(din_hei_ptr[3], 0.f);
              float tmp5 = LITEMAX(din_hei_ptr[4], 0.f);
              float tmp6 = LITEMAX(din_hei_ptr[5], 0.f);
              float tmp7 = LITEMAX(din_hei_ptr[6], 0.f);
              float tmp8 = LITEMAX(din_hei_ptr[7], 0.f);
              *(doutc0_ptr++) = LITEMIN(tmp1, six);
              *(doutc1_ptr++) = LITEMIN(tmp2, six);
              *(doutc2_ptr++) = LITEMIN(tmp3, six);
              *(doutc3_ptr++) = LITEMIN(tmp4, six);
              *(doutc4_ptr++) = LITEMIN(tmp5, six);
              *(doutc5_ptr++) = LITEMIN(tmp6, six);
              *(doutc6_ptr++) = LITEMIN(tmp7, six);
              *(doutc7_ptr++) = LITEMIN(tmp8, six);
              din_hei_ptr += 8;
            }
            break;
          case lite_api::ActivationType::kLeakyRelu:
            /*din = din >= 0 ? din : din * scale*/
            for (; i < width; ++i) {
              if (din_hei_ptr[0] >= 0) {
                *(doutc0_ptr++) = din_hei_ptr[0];
              } else {
                *(doutc0_ptr++) = din_hei_ptr[0] * scale;
              }
              if (din_hei_ptr[1] >= 0) {
                *(doutc1_ptr++) = din_hei_ptr[1];
              } else {
                *(doutc1_ptr++) = din_hei_ptr[1] * scale;
              }
              if (din_hei_ptr[2] >= 0) {
                *(doutc2_ptr++) = din_hei_ptr[2];
              } else {
                *(doutc2_ptr++) = din_hei_ptr[2] * scale;
              }
              if (din_hei_ptr[3] >= 0) {
                *(doutc3_ptr++) = din_hei_ptr[3];
              } else {
                *(doutc3_ptr++) = din_hei_ptr[3] * scale;
              }
              if (din_hei_ptr[4] >= 0) {
                *(doutc4_ptr++) = din_hei_ptr[4];
              } else {
                *(doutc4_ptr++) = din_hei_ptr[4] * scale;
              }
              if (din_hei_ptr[4] >= 0) {
                *(doutc5_ptr++) = din_hei_ptr[5];
              } else {
                *(doutc5_ptr++) = din_hei_ptr[5] * scale;
              }
              if (din_hei_ptr[6] >= 0) {
                *(doutc6_ptr++) = din_hei_ptr[6];
              } else {
                *(doutc6_ptr++) = din_hei_ptr[6] * scale;
              }
              if (din_hei_ptr[7] >= 0) {
                *(doutc7_ptr++) = din_hei_ptr[7];
              } else {
                *(doutc7_ptr++) = din_hei_ptr[7] * scale;
              }
              din_hei_ptr += 8;
            }
            break;
          default:
            LOG(FATAL) << "this act_type: "
                       << static_cast<int>(act_param->active_type)
                       << " fuse not support";
        }
      } else {
        for (; i < width; ++i) {
          *(doutc0_ptr++) = din_hei_ptr[0];
          *(doutc1_ptr++) = din_hei_ptr[1];
          *(doutc2_ptr++) = din_hei_ptr[2];
          *(doutc3_ptr++) = din_hei_ptr[3];
          *(doutc4_ptr++) = din_hei_ptr[4];
          *(doutc5_ptr++) = din_hei_ptr[5];
          *(doutc6_ptr++) = din_hei_ptr[6];
          *(doutc7_ptr++) = din_hei_ptr[7];
          din_hei_ptr += 8;
        }
      }
    }
  }
  return true;
}

template <typename Dtype>
inline void int32_nchwc4_kernel(Dtype*& dout0,        // NOLINT
                                Dtype*& dout1,        // NOLINT
                                Dtype*& dout2,        // NOLINT
                                Dtype*& dout3,        // NOLINT
                                const int32_t*& din,  // NOLINT
                                int cnt,
                                float32x4_t scale,
                                float32x4_t bias,
                                int flag_act,
                                float* alpha);

#ifdef __aarch64__
#define NCHWC4_TRANS_INT32                                      \
  "ldp q0, q1, [%[ptr_din]], #32\n"                             \
  "ldp q2, q3, [%[ptr_din]], #32\n"                             \
  "1:\n"                                                        \
  "trn1   v8.4s, v0.4s, v1.4s\n"                                \
  "trn2   v9.4s, v0.4s, v1.4s\n"                                \
  "ldp q0, q1, [%[ptr_din]], #32\n"                             \
  "trn1   v10.4s, v2.4s, v3.4s\n"                               \
  "trn2   v11.4s, v2.4s, v3.4s\n"                               \
  "ldp q2, q3, [%[ptr_din]], #32\n"                             \
  "trn1   v16.2d, v8.2d, v10.2d\n"                              \
  "trn2   v17.2d, v8.2d, v10.2d\n"                              \
  "trn1   v18.2d, v9.2d, v11.2d\n"                              \
  "trn2   v19.2d, v9.2d, v11.2d\n" /* int32 --> fp32 */         \
  "scvtf   v4.4s, v16.4s\n"                                     \
  "scvtf   v5.4s, v17.4s\n"                                     \
  "scvtf   v6.4s, v18.4s\n"                                     \
  "scvtf   v7.4s, v19.4s\n" /* add bias */                      \
  "dup    v16.4s, %[bias].s[0]\n"                               \
  "dup    v17.4s, %[bias].s[2]\n"                               \
  "dup    v18.4s, %[bias].s[1]\n"                               \
  "dup    v19.4s, %[bias].s[3]\n" /* mul scale */               \
  "fmla    v16.4s, v4.4s, %[scale].s[0]\n"                      \
  "fmla    v17.4s, v5.4s, %[scale].s[2]\n"                      \
  "fmla    v18.4s, v6.4s, %[scale].s[1]\n"                      \
  "fmla    v19.4s, v7.4s, %[scale].s[3]\n"                      \
  "cmp    %w[flag_act],   #1\n"                                 \
  "bne    12f                     \n"                           \
  "movi   v20.4s,  #0             \n" /* for relu*/             \
  "fmax   v16.4s, v16.4s, v20.4s  \n"                           \
  "fmax   v17.4s, v17.4s, v20.4s  \n"                           \
  "fmax   v18.4s, v18.4s, v20.4s  \n"                           \
  "fmax   v19.4s, v19.4s, v20.4s  \n"                           \
  "b      2f                      \n"   /* relu end */          \
  "12:                            \n"   /* no relu */           \
  "cmp    %w[flag_act],  #0       \n"   /* check no act */      \
  "beq    2f                      \n"   /* no act end */        \
  "cmp    %w[flag_act],  #2       \n"   /* check relu6 */       \
  "bne    13f                     \n"   /* jump no relu6*/      \
  "movi   v8.4s, #0               \n"   /* for relu6 */         \
  "ld1    {v9.4s}, [%[alpha]]     \n"   /* relu6 alpha */       \
  "fmax   v16.4s, v16.4s, v8.4s  \n"    /* relu6 */             \
  "fmax   v17.4s, v17.4s, v8.4s  \n"    /* relu6 */             \
  "fmax   v18.4s, v18.4s, v8.4s  \n"    /* relu6 */             \
  "fmax   v19.4s, v19.4s, v8.4s  \n"    /* relu6 */             \
  "fmin   v16.4s, v16.4s, v9.4s  \n"    /* relu6 */             \
  "fmin   v17.4s, v17.4s, v9.4s  \n"    /* relu6 */             \
  "fmin   v18.4s, v18.4s, v9.4s  \n"    /* relu6 */             \
  "fmin   v19.4s, v19.4s, v9.4s  \n"    /* relu6 */             \
  "b      2f                     \n"    /* relu6 end */         \
  "13:                              \n" /* leakey relu */       \
  "movi   v12.4s,   #0              \n" /* for leakey relu */   \
  "ld1    {v13.4s}, [%[alpha]]      \n" /* leakey relu alpha */ \
  "fcmge  v4.4s,   v16.4s,  v12.4s  \n" /* vcgeq_f32 */         \
  "fmul   v5.4s,   v16.4s,  v13.4s  \n" /* vmulq_f32 */         \
  "fcmge  v6.4s,   v17.4s,  v12.4s  \n" /* vcgeq_f32 */         \
  "fmul   v7.4s,   v17.4s,  v13.4s  \n" /* vmulq_f32 */         \
  "fcmge  v8.4s,   v18.4s,  v12.4s  \n" /* vcgeq_f32 */         \
  "fmul   v9.4s,   v18.4s,  v13.4s  \n" /* vmulq_f32 */         \
  "fcmge  v10.4s,  v19.4s,  v12.4s  \n" /* vcgeq_f32 */         \
  "fmul   v11.4s,  v19.4s,  v13.4s  \n" /* vmulq_f32 */         \
  "bif    v16.16b, v5.16b,  v4.16b  \n" /* choose*/             \
  "bif    v17.16b, v7.16b,  v6.16b  \n" /* choose*/             \
  "bif    v18.16b, v9.16b,  v8.16b  \n" /* choose*/             \
  "bif    v19.16b, v11.16b, v10.16b \n" /* choose*/             \
  "2:                               \n" /* act end */

#else
#define NCHWC4_TRANS_INT32                        \
  "vld1.32    {d4-d7}, [%[ptr_din]]!\n"           \
  "vld1.32    {d8-d11}, [%[ptr_din]]!\n"          \
  "1:\n" /* transpose */                          \
  "vtrn.32    q2, q3\n"                           \
  "vtrn.32    q4, q5\n"                           \
  "vswp.32    d5, d8\n"                           \
  "vswp.32    d7, d10\n" /* int32-> fp32 */       \
  "vcvt.f32.s32   q6, q2\n"                       \
  "vcvt.f32.s32   q7, q3\n"                       \
  "vcvt.f32.s32   q8, q4\n"                       \
  "vcvt.f32.s32   q9, q5\n" /* add bias */        \
  "vdup.32    q10, %e[bias][0]\n"                 \
  "vdup.32    q11, %e[bias][1]\n"                 \
  "vdup.32    q12, %f[bias][0]\n"                 \
  "vdup.32    q13, %f[bias][1]\n" /* mul scale */ \
  "vmla.f32  q10, q6, %e[scale][0]\n"             \
  "vmla.f32  q11, q7, %e[scale][1]\n"             \
  "vmla.f32  q12, q8, %f[scale][0]\n"             \
  "vmla.f32  q13, q9, %f[scale][1]\n"             \
  "vmov.u32   q15, #0              \n"            \
  "cmp    %[flag_act],   #1        \n"            \
  "bne    12f                      \n"            \
  "vmax.f32   q10, q10, q15        \n"            \
  "vmax.f32   q11, q11, q15        \n"            \
  "vmax.f32   q12, q12, q15        \n"            \
  "vmax.f32   q13, q13, q15        \n"            \
  "b      2f                       \n"            \
  "12:                             \n"            \
  "cmp    %[flag_act],  #0         \n"            \
  "beq    2f                       \n"            \
  "cmp    %[flag_act],  #2         \n"            \
  "bne    13f                      \n"            \
  "vld1.f32  {d14-d15}, [%[alpha]] \n"            \
  "vmax.f32   q10, q10, q15        \n"            \
  "vmax.f32   q11, q11, q15        \n"            \
  "vmax.f32   q12, q12, q15        \n"            \
  "vmax.f32   q13, q13, q15        \n"            \
  "vmin.f32   q10, q10, q7         \n"            \
  "vmin.f32   q11, q11, q7         \n"            \
  "vmin.f32   q12, q12, q7         \n"            \
  "vmin.f32   q13, q13, q7         \n"            \
  "b      2f                       \n"            \
  "13:                             \n"            \
  "vld1.f32  {d6-d7}, [%[alpha]]   \n"            \
  "vcge.f32  q6,  q10, q15         \n"            \
  "vmul.f32  q7,  q10, q3          \n"            \
  "vcge.f32  q8,  q11, q15         \n"            \
  "vmul.f32  q9,  q11, q3          \n"            \
  "vbif      q10, q7,  q6          \n"            \
  "vbif      q11, q9,  q8          \n"            \
  "vcge.f32  q6,  q12, q15         \n"            \
  "vmul.f32  q7,  q12, q3          \n"            \
  "vcge.f32  q8,  q13, q15         \n"            \
  "vmul.f32  q9,  q13, q3          \n"            \
  "vbif      q12, q7,  q6          \n"            \
  "vbif      q13, q9,  q8          \n"            \
  "2:\n"

#endif

template <>
inline void int32_nchwc4_kernel(float*& dout0,        // NOLINT
                                float*& dout1,        // NOLINT
                                float*& dout2,        // NOLINT
                                float*& dout3,        // NOLINT
                                const int32_t*& din,  // NOLINT
                                int cnt,
                                float32x4_t scale,
                                float32x4_t bias,
                                int flag_act,
                                float* alpha) {
#ifdef __aarch64__
  asm volatile(NCHWC4_TRANS_INT32
               "subs   %w[cnt], %w[cnt], #1\n"
               /* store result */
               "str    q16, [%[doutc0r0]], #16\n"
               "str    q17, [%[doutc2r0]], #16\n"
               "str    q18, [%[doutc1r0]], #16\n"
               "str    q19, [%[doutc3r0]], #16\n"
               "bne    1b\n"
               : [doutc0r0] "+r"(dout0),
                 [doutc1r0] "+r"(dout1),
                 [doutc2r0] "+r"(dout2),
                 [doutc3r0] "+r"(dout3),
                 [ptr_din] "+r"(din),
                 [cnt] "+r"(cnt)
               : [scale] "w"(scale),
                 [bias] "w"(bias),
                 [flag_act] "r"(flag_act),
                 [alpha] "r"(alpha)
               : "cc",
                 "memory",
                 "v0",
                 "v1",
                 "v2",
                 "v3",
                 "v4",
                 "v5",
                 "v6",
                 "v7",
                 "v8",
                 "v9",
                 "v10",
                 "v11",
                 "v12",
                 "v13",
                 "v14",
                 "v15",
                 "v16",
                 "v17",
                 "v18",
                 "v19",
                 "v20",
                 "v31");
#else
  asm volatile(NCHWC4_TRANS_INT32
               "subs   %[cnt], %[cnt], #1\n"
               /* store result */
               "vld1.32 {d4-d7}, [%[ptr_din]]!\n"
               "vst1.32  {d20-d21}, [%[doutc0r0]]!\n"
               "vst1.32  {d22-d23}, [%[doutc1r0]]!\n"
               "vld1.32 {d8-d11}, [%[ptr_din]]!\n"
               "vst1.32  {d24-d25}, [%[doutc2r0]]!\n"
               "vst1.32  {d26-d27}, [%[doutc3r0]]!\n"
               "bne    1b\n"
               : [doutc0r0] "+r"(dout0),
                 [doutc1r0] "+r"(dout1),
                 [doutc2r0] "+r"(dout2),
                 [doutc3r0] "+r"(dout3),
                 [ptr_din] "+r"(din),
                 [cnt] "+r"(cnt)
               : [scale] "w"(scale),
                 [bias] "w"(bias),
                 [flag_act] "r"(flag_act),
                 [alpha] "r"(alpha)
               : "cc",
                 "memory",
                 "q2",
                 "q3",
                 "q4",
                 "q5",
                 "q6",
                 "q7",
                 "q8",
                 "q9",
                 "q10",
                 "q11",
                 "q12",
                 "q13",
                 "q14",
                 "q15");
#endif
}

template <>
inline void int32_nchwc4_kernel(int8_t*& dout0,       // NOLINT
                                int8_t*& dout1,       // NOLINT
                                int8_t*& dout2,       // NOLINT
                                int8_t*& dout3,       // NOLINT
                                const int32_t*& din,  // NOLINT
                                int cnt,
                                float32x4_t scale,
                                float32x4_t bias,
                                int flag_act,
                                float* alpha) {
#ifdef __aarch64__
  float32x4_t vmax = vdupq_n_f32(-127.f);
  asm volatile(NCHWC4_TRANS_INT32
               "subs   %w[cnt], %w[cnt], #1\n"
               /* data >= -127 */
               "fcmge v4.4s, v16.4s, %[vmax].4s             \n"
               "fcmge v5.4s, v18.4s, %[vmax].4s             \n"
               "fcmge v6.4s, v17.4s, %[vmax].4s            \n"
               "fcmge v7.4s, v19.4s, %[vmax].4s            \n"
               "bif v16.16b, %[vmax].16b, v4.16b            \n"
               "bif v18.16b, %[vmax].16b, v5.16b            \n"
               "bif v17.16b, %[vmax].16b, v6.16b            \n"
               "bif v19.16b, %[vmax].16b, v7.16b            \n"
               /* fp32-int32 */
               "fcvtas  v4.4s, v16.4s\n"
               "fcvtas  v5.4s, v18.4s\n"
               "fcvtas  v6.4s, v17.4s\n"
               "fcvtas  v7.4s, v19.4s\n"
               /* int32-int16 */
               "sqxtn   v8.4h, v4.4s\n"
               "sqxtn   v9.4h, v5.4s\n"
               "sqxtn   v10.4h, v6.4s\n"
               "sqxtn   v11.4h, v7.4s\n"
               /* int16-int8 */
               "sqxtn  v16.8b, v8.8h\n"
               "sqxtn  v17.8b, v9.8h\n"
               "sqxtn  v18.8b, v10.8h\n"
               "sqxtn  v19.8b, v11.8h\n"
               /* store result */
               "str     s16, [%[doutc0r0]], #4\n"
               "str     s17, [%[doutc1r0]], #4\n"
               "str     s18, [%[doutc2r0]], #4\n"
               "str     s19, [%[doutc3r0]], #4\n"
               "bne    1b\n"
               : [doutc0r0] "+r"(dout0),
                 [doutc1r0] "+r"(dout1),
                 [doutc2r0] "+r"(dout2),
                 [doutc3r0] "+r"(dout3),
                 [ptr_din] "+r"(din),
                 [cnt] "+r"(cnt)
               : [scale] "w"(scale),
                 [vmax] "w"(vmax),
                 [bias] "w"(bias),
                 [flag_act] "r"(flag_act),
                 [alpha] "r"(alpha)
               : "cc",
                 "memory",
                 "v0",
                 "v1",
                 "v2",
                 "v3",
                 "v4",
                 "v5",
                 "v6",
                 "v7",
                 "v8",
                 "v9",
                 "v10",
                 "v11",
                 "v12",
                 "v13",
                 "v14",
                 "v15",
                 "v16",
                 "v17",
                 "v18",
                 "v19",
                 "v20",
                 "v31");
#else
  float vmax[4] = {-127.f, -127.f, -127.f, -127.f};
  asm volatile(NCHWC4_TRANS_INT32
               /* set 0.5 offset */
               "vmov.f32 q2, #0.5\n"
               "vmov.f32 q14, #-0.5\n"
               "vand.i32   q3, q2, q2    @ set offset, 0.5\n"
               "vand.i32   q4, q2, q2    @ set offset, 0.5\n"
               "vand.i32   q5, q2, q2    @ set offset, 0.5\n"
               "vcgt.f32   q6, q10, q15  @ get mask > 0, in0\n"
               "vcgt.f32   q7, q11, q15  @ get mask > 0, in1\n"
               "vcgt.f32   q8, q12, q15  @ get mask > 0, in2\n"
               "vcgt.f32   q9, q13, q15  @ get mask > 0, in3\n"
               /* set 0.5 offset */
               "vbif.f32   q2, q14, q6   @ get right offset\n"
               "vbif.f32   q3, q14, q7   @ get right offset\n"
               "vbif.f32   q4, q14, q8   @ get right offset\n"
               "vbif.f32   q5, q14, q9   @ get right offset\n"
               "vld1.32 {d28-d29}, [%[vmax]] \n"
               /* add offset */
               "vadd.f32   q10, q2, q10\n"
               "vadd.f32   q11, q3, q11\n"
               "vadd.f32   q12, q4, q12\n"
               "vadd.f32   q13, q5, q13\n"
               /* data >= -127 */
               "vcge.f32 q6, q10, q14     @ q10 >= vmax \n"
               "vcge.f32 q7, q11, q14     @ q11 >= vmax \n"
               "vcge.f32 q8, q12, q14     @ q12 >= vmax \n"
               "vcge.f32 q9, q13, q14     @ q13 >= vmax \n"
               "vbif q10, q14, q6         @ choose \n"
               "vbif q11, q14, q7         @ choose \n"
               "vbif q12, q14, q8         @ choose \n"
               "vbif q13, q14, q9         @ choose \n"
               /* fp32 to int32 */
               "vcvt.s32.f32  q6, q10    @ cvt to int32\n"
               "vcvt.s32.f32  q7, q11    @ cvt to int32\n"
               "vcvt.s32.f32  q8, q12    @ cvt to int32\n"
               "vcvt.s32.f32  q9, q13    @ cvt to int32\n"
               /* int32 to int16 */
               "vqmovn.s32 d20, q6       @ cnt to int16\n"
               "vqmovn.s32 d22, q7       @ cnt to int16\n"
               "vqmovn.s32 d24, q8       @ cnt to int16\n"
               "vqmovn.s32 d26, q9       @ cnt to int16\n"
               /* int16 to int8 */
               "vqmovn.s16 d12, q10       @ cnt to int8\n"
               "vqmovn.s16 d13, q11       @ cnt to int8\n"
               "vqmovn.s16 d14, q12      @ cnt to int8\n"
               "vqmovn.s16 d15, q13      @ cnt to int8\n"
               "subs   %[cnt], %[cnt], #1\n"
               /* store data*/
               "vld1.32 {d4-d7}, [%[ptr_din]]!\n"
               "vst1.32 {d12[0]},    [%[doutc0r0]]!\n"
               "vst1.32 {d13[0]},    [%[doutc1r0]]!\n"
               "vld1.32 {d8-d11}, [%[ptr_din]]!\n"
               "vst1.32 {d14[0]},    [%[doutc2r0]]!\n"
               "vst1.32 {d15[0]},    [%[doutc3r0]]!\n"
               "bne    1b                @ jump to main loop\n"
               : [doutc0r0] "+r"(dout0),
                 [doutc1r0] "+r"(dout1),
                 [doutc2r0] "+r"(dout2),
                 [doutc3r0] "+r"(dout3),
                 [ptr_din] "+r"(din),
                 [cnt] "+r"(cnt)
               : [scale] "w"(scale),
                 [bias] "w"(bias),
                 [vmax] "r"(vmax),
                 [flag_act] "r"(flag_act),
                 [alpha] "r"(alpha)
               : "cc",
                 "memory",
                 "q2",
                 "q3",
                 "q4",
                 "q5",
                 "q6",
                 "q7",
                 "q8",
                 "q9",
                 "q10",
                 "q11",
                 "q12",
                 "q13",
                 "q14",
                 "q15");
#endif
}

template <typename Dtype>
inline Dtype cvt_kernel(
    int din, float scale, float bias, int flag_act, float alpha);

template <>
inline float cvt_kernel(
    int din, float scale, float bias, int flag_act, float alpha) {
  if (flag_act == 1) {
    return LITEMAX(din * scale + bias, 0);
  } else if (flag_act == 0) {
    return din * scale + bias;
  } else if (flag_act == 2) {
    float max = LITEMAX(din * scale + bias, 0);
    return LITEMIN(max, alpha);
  } else {
    float result = din * scale + bias;
    return result > 0 ? result : alpha * result;
  }
}

template <>
inline int8_t cvt_kernel(
    int din, float scale, float bias, int flag_act, float alpha) {
  if (flag_act == 1) {
    auto tmp = saturate_cast<int8_t>(round(LITEMAX(din * scale + bias, 0)));
    return tmp < -127 ? -127 : tmp;
  } else if (flag_act == 0) {
    auto tmp = saturate_cast<int8_t>(round(din * scale + bias));
    return tmp < -127 ? -127 : tmp;
  } else if (flag_act == 2) {
    float max = LITEMAX(din * scale + bias, 0);
    float relu6_result = LITEMIN(max, alpha);
    auto tmp = saturate_cast<int8_t>(round(relu6_result));
    return tmp < -127 ? -127 : tmp;
  } else {
    float result = din * scale + bias;
    float leaky_result = result > 0 ? result : alpha * result;
    auto tmp = saturate_cast<int8_t>(round(leaky_result));
    return tmp < -127 ? -127 : tmp;
  }
}

template <typename Dtype>
inline void write_int32_nchwc4_to_nchw(const int* din,
                                       Dtype* dout,
                                       int cs,
                                       int ce,
                                       int hs,
                                       int he,
                                       int ws,
                                       int we,
                                       int channel,
                                       int height,
                                       int width,
                                       int flag_act,
                                       float* alpha,
                                       float* bias,
                                       bool flag_bias,
                                       Dtype* trash_ptr,
                                       const float* scale) {
  int size_c_out = width * height;

  Dtype* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
  Dtype* doutc1r0 = doutc0r0 + size_c_out;
  Dtype* doutc2r0 = doutc1r0 + size_c_out;
  Dtype* doutc3r0 = doutc2r0 + size_c_out;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int valid_w = we - ws;
  int cnt = valid_w / 4;

  float32x4_t w_scale = vld1q_f32(scale);
  float32x4_t w_bias = flag_bias ? vld1q_f32(bias) : vdupq_n_f32(0.f);

  if (we > width) {
    cnt--;
  }
  for (int i = 0; i < size_h; i++) {
    int size_w = i * width;
    Dtype* doutc0_ptr = doutc0r0 + size_w;
    Dtype* doutc1_ptr = doutc1r0 + size_w;
    Dtype* doutc2_ptr = doutc2r0 + size_w;
    Dtype* doutc3_ptr = doutc3r0 + size_w;
    if (ce > channel) {
      switch (ce - channel) {
        case 3:
          doutc1_ptr = trash_ptr;
        case 2:
          doutc2_ptr = trash_ptr;
        case 1:
          doutc3_ptr = trash_ptr;
        default:
          break;
      }
    }
    int index = i * valid_w * 4;
    const int* din_hei_ptr = din + index;
    if (cnt > 0) {
      int32_nchwc4_kernel<Dtype>(doutc0_ptr,
                                 doutc1_ptr,
                                 doutc2_ptr,
                                 doutc3_ptr,
                                 din_hei_ptr,
                                 cnt,
                                 w_scale,
                                 w_bias,
                                 flag_act,
                                 alpha);
    }
    if (we > width) {
      int offset = 16 * (valid_w / 4 - 1);
      din_hei_ptr = din + index + offset;
      int j = we - 4;
      for (; j < width; ++j) {
        *(doutc0_ptr++) = cvt_kernel<Dtype>(
            din_hei_ptr[0], scale[0], bias[0], flag_act, alpha[0]);
        *(doutc1_ptr++) = cvt_kernel<Dtype>(
            din_hei_ptr[1], scale[1], bias[1], flag_act, alpha[0]);
        *(doutc2_ptr++) = cvt_kernel<Dtype>(
            din_hei_ptr[2], scale[2], bias[2], flag_act, alpha[0]);
        *(doutc3_ptr++) = cvt_kernel<Dtype>(
            din_hei_ptr[3], scale[3], bias[3], flag_act, alpha[0]);
        din_hei_ptr += 4;
      }
    }
  }
}

template <typename Dtype>
inline void int32_nchwc8_kernel(Dtype*& dout0,        // NOLINT
                                Dtype*& dout1,        // NOLINT
                                Dtype*& dout2,        // NOLINT
                                Dtype*& dout3,        // NOLINT
                                Dtype*& dout4,        // NOLINT
                                Dtype*& dout5,        // NOLINT
                                Dtype*& dout6,        // NOLINT
                                Dtype*& dout7,        // NOLINT
                                const int32_t*& din,  // NOLINT
                                int cnt,
                                float32x4_t scale0,
                                float32x4_t scale1,
                                float32x4_t bias0,
                                float32x4_t bias1,
                                int flag_act,
                                float* alpha);

// clang-format off
#ifdef __aarch64__
#define INT32_NCHWC8_TO_NCHW_FP32                                  \
  "ldp q0, q1, [%[ptr_din]], #32\n" /* load r00, r01 to q0, q1 */  \
  "ldp q2, q3, [%[ptr_din]], #32\n" /* load r02, r03 to q2, q3 */  \
  "ldp q4, q5, [%[ptr_din]], #32\n" /* load r00, r01 to q0, q1 */  \
  "ldp q6, q7, [%[ptr_din]], #32\n" /* load r02, r03 to q2, q3 */  \
  "1:\n"                                                           \
  "trn1   v8.4s, v0.4s, v2.4s\n"    /* trans q0, q1*/              \
  "trn2   v9.4s, v0.4s, v2.4s\n"    /* trans q0, q1*/              \
  "trn1   v10.4s, v1.4s, v3.4s\n"   /* trans q2, q3*/              \
  "trn2   v11.4s, v1.4s, v3.4s\n"   /* trans q2, q3*/              \
  "ldp q0, q1, [%[ptr_din]], #32\n" /* load r00, r01 to q0, q1 */  \
  "trn1   v12.4s, v4.4s, v6.4s\n"   /* trans q0, q1*/              \
  "trn2   v13.4s, v4.4s, v6.4s\n"   /* trans q0, q1*/              \
  "trn1   v14.4s, v5.4s, v7.4s\n"   /* trans q2, q3*/              \
  "trn2   v15.4s, v5.4s, v7.4s\n"   /* trans q2, q3*/              \
  "ldp q2, q3, [%[ptr_din]], #32\n" /* load r02, r03 to q2, q3 */  \
  "trn1   v16.2d, v8.2d, v12.2d\n"  /* trans q8, q10 00 01 02 03*/ \
  "trn2   v17.2d, v8.2d, v12.2d\n"  /* trans q8, q10 20 21 22 23*/ \
  "trn1   v18.2d, v9.2d, v13.2d\n"  /* trans q9, q11 10 11 12 13*/ \
  "trn2   v19.2d, v9.2d, v13.2d\n"  /* trans q9, q11 30 31 32 33*/ \
  "ldp q4, q5, [%[ptr_din]], #32\n" /* load r00, r01 to q0, q1 */  \
  "trn1   v8.2d, v10.2d, v14.2d\n"  /* trans q8, q10 40 41 42 43*/ \
  "trn2   v9.2d, v10.2d, v14.2d\n"  /* trans q8, q10 60 61 62 63*/ \
  "trn1   v12.2d, v11.2d, v15.2d\n" /* trans q9, q11 50 51 52 53*/ \
  "trn2   v13.2d, v11.2d, v15.2d\n" /* trans q9, q11 70 71 72 73*/ \
  "ldp q6, q7, [%[ptr_din]], #32\n" /* load r02, r03 to q2, q3 */  \
  /* int32->fp32 */                                                \
  "scvtf   v10.4s, v16.4s\n"                                       \
  "scvtf   v11.4s, v17.4s\n"                                       \
  "scvtf   v14.4s, v18.4s\n"                                       \
  "scvtf   v15.4s, v19.4s\n"                                       \
  /* add bias */                                                   \
  "dup    v16.4s, %[bias0].s[0]\n"                                 \
  "dup    v17.4s, %[bias0].s[2]\n"                                 \
  "dup    v18.4s, %[bias0].s[1]\n"                                 \
  "dup    v19.4s, %[bias0].s[3]\n"                                 \
  /* mul scale */                                                  \
  "fmla    v16.4s, v10.4s, %[scale0].s[0]\n"                       \
  "fmla    v17.4s, v11.4s, %[scale0].s[2]\n"                       \
  "fmla    v18.4s, v14.4s, %[scale0].s[1]\n"                       \
  "fmla    v19.4s, v15.4s, %[scale0].s[3]\n"                       \
  "scvtf   v10.4s, v8.4s\n"                                        \
  "scvtf   v11.4s, v9.4s\n"                                        \
  "scvtf   v14.4s, v12.4s\n"                                       \
  "scvtf   v15.4s, v13.4s\n"                                       \
  /* add bias */                                                   \
  "dup    v8.4s, %[bias1].s[0]\n"                                  \
  "dup    v9.4s, %[bias1].s[2]\n"                                  \
  "dup    v12.4s, %[bias1].s[1]\n"                                 \
  "dup    v13.4s, %[bias1].s[3]\n"                                 \
  /* mul scale */                                                  \
  "fmla    v8.4s, v10.4s, %[scale1].s[0]\n"                        \
  "fmla    v9.4s, v11.4s, %[scale1].s[2]\n"                        \
  "fmla    v12.4s, v14.4s, %[scale1].s[1]\n"                       \
  "fmla    v13.4s, v15.4s, %[scale1].s[3]\n"                       \
  /* activation */                                                 \
  "cmp    %w[flag_act],   #1\n"                                    \
  "bne    12f                     \n"                              \
  "movi   v31.4s,  #0             \n" /* for relu*/                \
  "fmax   v16.4s, v16.4s, v31.4s  \n" /*relu*/                     \
  "fmax   v17.4s, v17.4s, v31.4s  \n" /*relu*/                     \
  "fmax   v18.4s, v18.4s, v31.4s  \n" /*relu*/                     \
  "fmax   v19.4s, v19.4s, v31.4s  \n" /*relu*/                     \
  "fmax   v8.4s,  v8.4s,  v31.4s  \n" /*relu*/                     \
  "fmax   v9.4s,  v9.4s,  v31.4s  \n" /*relu*/                     \
  "fmax   v12.4s, v12.4s, v31.4s  \n" /*relu*/                     \
  "fmax   v13.4s, v13.4s, v31.4s  \n" /*relu*/                     \
  "b      2f                      \n" /* relu end */               \
  "12:                            \n" /* no relu */                \
  "cmp    %w[flag_act],  #0       \n" /* check no act */           \
  "beq    2f                      \n" /* no act end */             \
  "cmp    %w[flag_act],  #2       \n" /* check relu6 */            \
  "bne    13f                     \n" /* jump no relu6*/           \
  "movi   v20.4s, #0              \n" /* for relu6 */              \
  "ld1    {v21.4s}, [%[alpha]]    \n" /* relu6 alpha */            \
  "fmax   v16.4s, v16.4s, v20.4s  \n" /* relu6 */                  \
  "fmax   v17.4s, v17.4s, v20.4s  \n" /* relu6 */                  \
  "fmax   v18.4s, v18.4s, v20.4s  \n" /* relu6 */                  \
  "fmax   v19.4s, v19.4s, v20.4s  \n" /* relu6 */                  \
  "fmax   v8.4s,  v8.4s,  v20.4s  \n" /* relu6 */                  \
  "fmax   v9.4s,  v9.4s,  v20.4s  \n" /* relu6 */                  \
  "fmax   v12.4s, v12.4s, v20.4s  \n" /* relu6 */                  \
  "fmax   v13.4s, v13.4s, v20.4s  \n" /* relu6 */                  \
  "fmin   v16.4s, v16.4s, v21.4s  \n" /* relu6 */                  \
  "fmin   v17.4s, v17.4s, v21.4s  \n" /* relu6 */                  \
  "fmin   v18.4s, v18.4s, v21.4s  \n" /* relu6 */                  \
  "fmin   v19.4s, v19.4s, v21.4s  \n" /* relu6 */                  \
  "fmin   v8.4s,  v8.4s,  v21.4s  \n" /* relu6 */                  \
  "fmin   v9.4s,  v9.4s,  v21.4s  \n" /* relu6 */                  \
  "fmin   v12.4s, v12.4s, v21.4s  \n" /* relu6 */                  \
  "fmin   v13.4s, v13.4s, v21.4s  \n" /* relu6 */                  \
  "b      2f                      \n" /* relu6 end */              \
  "13:                               \n" /* leakey relu */         \
  "movi   v20.4s,   #0               \n" /* for leakey relu */     \
  "ld1    {v21.4s}, [%[alpha]]       \n" /* leakey relu alpha */   \
  "fcmge  v10.4s,   v16.4s,  v20.4s  \n" /* vcgeq_f32 */           \
  "fmul   v11.4s,   v16.4s,  v21.4s  \n" /* vmulq_f32 */           \
  "fcmge  v14.4s,   v17.4s,  v20.4s  \n" /* vcgeq_f32 */           \
  "fmul   v15.4s,   v17.4s,  v21.4s  \n" /* vmulq_f32 */           \
  "fcmge  v22.4s,   v18.4s,  v20.4s  \n" /* vcgeq_f32 */           \
  "fmul   v23.4s,   v18.4s,  v21.4s  \n" /* vmulq_f32 */           \
  "fcmge  v24.4s,   v19.4s,  v20.4s  \n" /* vcgeq_f32 */           \
  "fmul   v25.4s,   v19.4s,  v21.4s  \n" /* vmulq_f32 */           \
  "bif    v16.16b, v11.16b,  v10.16b \n" /* choose*/               \
  "bif    v17.16b, v15.16b,  v14.16b \n" /* choose*/               \
  "bif    v18.16b, v23.16b,  v22.16b \n" /* choose*/               \
  "bif    v19.16b, v25.16b,  v24.16b \n" /* choose*/               \
  "fcmge  v10.4s,   v8.4s,   v20.4s  \n" /* vcgeq_f32 */           \
  "fmul   v11.4s,   v8.4s,   v21.4s  \n" /* vmulq_f32 */           \
  "fcmge  v14.4s,   v9.4s,   v20.4s  \n" /* vcgeq_f32 */           \
  "fmul   v15.4s,   v9.4s,   v21.4s  \n" /* vmulq_f32 */           \
  "fcmge  v22.4s,   v12.4s,  v20.4s  \n" /* vcgeq_f32 */           \
  "fmul   v23.4s,   v12.4s,  v21.4s  \n" /* vmulq_f32 */           \
  "fcmge  v24.4s,   v13.4s,  v20.4s  \n" /* vcgeq_f32 */           \
  "fmul   v25.4s,   v13.4s,  v21.4s  \n" /* vmulq_f32 */           \
  "bif    v8.16b,  v11.16b,  v10.16b \n" /* choose*/               \
  "bif    v9.16b,  v15.16b,  v14.16b \n" /* choose*/               \
  "bif    v12.16b, v23.16b,  v22.16b \n" /* choose*/               \
  "bif    v13.16b, v25.16b,  v24.16b \n" /* choose*/               \
  "2:                                \n" /* act end */

#else
#define INT32_NCHWC8_TO_NCHW_FP32                                 \
  "1:                                 @ main loop\n"              \
  "vld1.32 {d0-d3},   [%[ptr_din]]!   @load data \n"              \
  "vld1.32 {d4-d7},   [%[ptr_din]]!   @load data \n"              \
  "vld1.32 {d8-d11},  [%[ptr_din]]!   @load data \n"              \
  "vld1.32 {d12-d15}, [%[ptr_din]]!   @load data \n"              \
  /* int32-> fp32 */                                              \
  "vcvt.f32.s32   q8, q0\n"                                       \
  "vcvt.f32.s32   q9, q1\n"                                       \
  "vcvt.f32.s32   q10, q2\n"                                      \
  "vcvt.f32.s32   q11, q3\n"                                      \
  "vand.32   q0, %q[bias0], %q[bias0]\n"                          \
  "vand.32   q1, %q[bias1], %q[bias1]\n"                          \
  "vand.32   q2, %q[bias0], %q[bias0]\n"                          \
  "vand.32   q3, %q[bias1], %q[bias1]\n"                          \
  /* mul scale */                                                 \
  "vmla.f32  q0, q8, %q[scale0]\n"                                \
  "vmla.f32  q1, q9, %q[scale1]\n"                                \
  "vmla.f32  q2, q10, %q[scale0]\n"                               \
  "vmla.f32  q3, q11, %q[scale1]\n"                               \
  /* int32-> fp32 */                                              \
  "vcvt.f32.s32   q8, q4\n"                                       \
  "vcvt.f32.s32   q9, q5\n"                                       \
  "vcvt.f32.s32   q10, q6\n"                                      \
  "vcvt.f32.s32   q11, q7\n"                                      \
  "vand.32   q4, %q[bias0], %q[bias0]\n"                          \
  "vand.32   q5, %q[bias1], %q[bias1]\n"                          \
  "vand.32   q6, %q[bias0], %q[bias0]\n"                          \
  "vand.32   q7, %q[bias1], %q[bias1]\n"                          \
  /* mul scale */                                                 \
  "vmla.f32  q4, q8, %q[scale0]\n"                                \
  "vmla.f32  q5, q9, %q[scale1]\n"                                \
  "vmla.f32  q6, q10, %q[scale0]\n"                               \
  "vmla.f32  q7, q11, %q[scale1]\n"                               \
  /* transpose */                                                 \
  "vtrn.32    q0, q2\n"                                           \
  "vtrn.32    q1, q3\n"                                           \
  "vtrn.32    q4, q6\n"                                           \
  "vtrn.32    q5, q7\n"                                           \
  "vswp    d1, d8\n"  /* q0: a0-a3, q4: c0-c3 */                  \
  "vswp    d5, d12\n" /* q2: b0-b3, q6: d0-d3 */                  \
  "vswp    d3, d10\n" /* q1: e0-e3, q5: g0-g3 */                  \
  "vswp    d7, d14\n" /* q3: f0-f3, q7: h0-h3 */                  \
  /* activation */                                                \
  "vmov.u32   q8, #0                \n"                           \
  "cmp    %[flag_act],   #1         \n"                           \
  "bne    12f                       \n"                           \
  "vmax.f32 q0, q0, q8              \n" /*relu*/                  \
  "vmax.f32 q2, q2, q8              \n" /*relu*/                  \
  "vmax.f32 q4, q4, q8              \n" /*relu*/                  \
  "vmax.f32 q6, q6, q8              \n" /*relu*/                  \
  "vmax.f32 q1, q1, q8              \n" /*relu*/                  \
  "vmax.f32 q3, q3, q8              \n" /*relu*/                  \
  "vmax.f32 q5, q5, q8              \n" /*relu*/                  \
  "vmax.f32 q7, q7, q8              \n" /*relu*/                  \
  "b      2f                        \n"                           \
  "12:                              \n"                           \
  "cmp    %[flag_act],  #0          \n"                           \
  "beq    2f                        \n"                           \
  "cmp    %[flag_act],  #2          \n"                           \
  "bne    13f                       \n"                           \
  "vld1.f32  {d18-d19}, [%[alpha]]  \n"                           \
  "vmax.f32   q0, q0, q8            \n"                           \
  "vmax.f32   q2, q2, q8            \n"                           \
  "vmax.f32   q4, q4, q8            \n"                           \
  "vmax.f32   q6, q6, q8            \n"                           \
  "vmax.f32   q1, q1, q8            \n"                           \
  "vmax.f32   q3, q3, q8            \n"                           \
  "vmax.f32   q5, q5, q8            \n"                           \
  "vmax.f32   q7, q7, q8            \n"                           \
  "vmin.f32   q0, q0, q9            \n"                           \
  "vmin.f32   q2, q2, q9            \n"                           \
  "vmin.f32   q4, q4, q9            \n"                           \
  "vmin.f32   q6, q6, q9            \n"                           \
  "vmin.f32   q1, q1, q9            \n"                           \
  "vmin.f32   q3, q3, q9            \n"                           \
  "vmin.f32   q5, q5, q9            \n"                           \
  "vmin.f32   q7, q7, q9            \n"                           \
  "b      2f                        \n"                           \
  "13:                              \n"                           \
  "vld1.f32  {d18-d19}, [%[alpha]]  \n"                           \
  "vcge.f32  q10,  q0,  q8          \n"                           \
  "vmul.f32  q11,  q0,  q9          \n"                           \
  "vbif      q0,   q11, q10         \n"                           \
  "vcge.f32  q10,  q2,  q8          \n"                           \
  "vmul.f32  q11,  q2,  q9          \n"                           \
  "vbif      q2,   q11, q10         \n"                           \
  "vcge.f32  q10,  q4,  q8          \n"                           \
  "vmul.f32  q11,  q4,  q9          \n"                           \
  "vbif      q4,   q11, q10         \n"                           \
  "vcge.f32  q10,  q6,  q8          \n"                           \
  "vmul.f32  q11,  q6,  q9          \n"                           \
  "vbif      q6,   q11, q10         \n"                           \
  "vcge.f32  q10,  q1,  q8          \n"                           \
  "vmul.f32  q11,  q1,  q9          \n"                           \
  "vbif      q1,   q11, q10         \n"                           \
  "vcge.f32  q10,  q3,  q8          \n"                           \
  "vmul.f32  q11,  q3,  q9          \n"                           \
  "vbif      q3,   q11, q10         \n"                           \
  "vcge.f32  q10,  q5,  q8          \n"                           \
  "vmul.f32  q11,  q5,  q9          \n"                           \
  "vbif      q5,   q11, q10         \n"                           \
  "vcge.f32  q10,  q7,  q8          \n"                           \
  "vmul.f32  q11,  q7,  q9          \n"                           \
  "vbif      q7,   q11, q10         \n"                           \
  "2:\n"

#endif
// clang-format on

template <>
inline void int32_nchwc8_kernel(float*& dout0,        // NOLINT
                                float*& dout1,        // NOLINT
                                float*& dout2,        // NOLINT
                                float*& dout3,        // NOLINT
                                float*& dout4,        // NOLINT
                                float*& dout5,        // NOLINT
                                float*& dout6,        // NOLINT
                                float*& dout7,        // NOLINT
                                const int32_t*& din,  // NOLINT
                                int cnt,
                                float32x4_t scale0,
                                float32x4_t scale1,
                                float32x4_t bias0,
                                float32x4_t bias1,
                                int flag_act,
                                float* alpha) {
// clang-format off
#ifdef __aarch64__
  asm volatile(INT32_NCHWC8_TO_NCHW_FP32
               "subs   %w[cnt], %w[cnt],  #1\n"   /* loop count -1*/
               "str    q16, [%[doutc0r0]], #16\n" /* store c0r0*/
               "str    q17, [%[doutc2r0]], #16\n" /* store c2r0*/
               "str    q18, [%[doutc1r0]], #16\n" /* store c1r0*/
               "str    q19, [%[doutc3r0]], #16\n" /* store c3r0*/
               "str    q8, [%[doutc4r0]], #16\n"  /* store c4r0*/
               "str    q9, [%[doutc6r0]], #16\n"  /* store c6r0*/
               "str    q12, [%[doutc5r0]], #16\n" /* store c5r0*/
               "str    q13, [%[doutc7r0]], #16\n" /* store c7r0*/
               "bne    1b\n"                      /* jump to main loop*/
               : [doutc0r0] "+r"(dout0),
                 [doutc1r0] "+r"(dout1),
                 [doutc2r0] "+r"(dout2),
                 [doutc3r0] "+r"(dout3),
                 [doutc4r0] "+r"(dout4),
                 [doutc5r0] "+r"(dout5),
                 [doutc6r0] "+r"(dout6),
                 [doutc7r0] "+r"(dout7),
                 [ptr_din] "+r"(din),
                 [cnt] "+r"(cnt)
               : [scale0] "w"(scale0),
                 [scale1] "w"(scale1),
                 [bias0] "w"(bias0),
                 [bias1] "w"(bias1),
                 [flag_act] "r"(flag_act), 
                 [alpha] "r"(alpha)
               : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                 "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                 "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                 "v25", "v31"
               );
#else
  asm volatile(INT32_NCHWC8_TO_NCHW_FP32
               "subs    %[cnt],  #1\n"               /* loop count -1*/
               "vst1.32 {d0-d1}, [%[doutc0r0]]!\n"   /* store c0r0*/
               "vst1.32 {d4-d5}, [%[doutc1r0]]!\n"   /* store c0r0*/
               "vst1.32 {d8-d9}, [%[doutc2r0]]!\n"   /* store c0r0*/
               "vst1.32 {d12-d13}, [%[doutc3r0]]!\n" /* store c0r0*/
               "vst1.32 {d2-d3}, [%[doutc4r0]]!\n"   /* store c0r0*/
               "vst1.32 {d6-d7}, [%[doutc5r0]]!\n"   /* store c0r0*/
               "vst1.32 {d10-d11}, [%[doutc6r0]]!\n" /* store c0r0*/
               "vst1.32 {d14-d15}, [%[doutc7r0]]!\n" /* store c0r0*/
               "bne    1b\n"                         /* jump to main loop*/
               : [doutc0r0] "+r"(dout0),
                 [doutc1r0] "+r"(dout1),
                 [doutc2r0] "+r"(dout2),
                 [doutc3r0] "+r"(dout3),
                 [doutc4r0] "+r"(dout4),
                 [doutc5r0] "+r"(dout5),
                 [doutc6r0] "+r"(dout6),
                 [doutc7r0] "+r"(dout7),
                 [ptr_din] "+r"(din),
                 [cnt] "+r"(cnt)
               : [scale0] "w"(scale0),
                 [scale1] "w"(scale1),
                 [bias0] "w"(bias0),
                 [bias1] "w"(bias1),
                 [flag_act] "r"(flag_act), 
                 [alpha] "r"(alpha)
               : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                 "q7", "q8", "q9", "q10", "q11"
               );
#endif
  // clang-format on
}

template <>
inline void int32_nchwc8_kernel(int8_t*& dout0,       // NOLINT
                                int8_t*& dout1,       // NOLINT
                                int8_t*& dout2,       // NOLINT
                                int8_t*& dout3,       // NOLINT
                                int8_t*& dout4,       // NOLINT
                                int8_t*& dout5,       // NOLINT
                                int8_t*& dout6,       // NOLINT
                                int8_t*& dout7,       // NOLINT
                                const int32_t*& din,  // NOLINT
                                int cnt,
                                float32x4_t scale0,
                                float32x4_t scale1,
                                float32x4_t bias0,
                                float32x4_t bias1,
                                int flag_act,
                                float* alpha) {
// clang-format off
#ifdef __aarch64__
  float32x4_t vmax = vdupq_n_f32(-127.f);
  asm volatile(INT32_NCHWC8_TO_NCHW_FP32 /* fp32-int32 */
               /* data >= -127 */
               "fcmge v10.4s, v16.4s, %[vmax].4s             \n"
               "fcmge v11.4s, v17.4s, %[vmax].4s             \n"
               "fcmge v14.4s, v18.4s, %[vmax].4s            \n"
               "fcmge v15.4s, v19.4s, %[vmax].4s            \n"
               "fcmge v20.4s, v8.4s, %[vmax].4s             \n"
               "fcmge v21.4s, v9.4s, %[vmax].4s             \n"
               "fcmge v22.4s, v12.4s, %[vmax].4s            \n"
               "fcmge v23.4s, v13.4s, %[vmax].4s            \n"
               /* choose data */
               "bif v16.16b, %[vmax].16b, v10.16b            \n"
               "bif v17.16b, %[vmax].16b, v11.16b            \n"
               "bif v18.16b, %[vmax].16b, v14.16b            \n"
               "bif v19.16b, %[vmax].16b, v15.16b            \n"
               "bif v8.16b, %[vmax].16b, v20.16b            \n"
               "bif v9.16b, %[vmax].16b, v21.16b            \n"
               "bif v12.16b, %[vmax].16b, v22.16b            \n"
               "bif v13.16b, %[vmax].16b, v23.16b            \n"
               /* fp32 - int32 */
               "fcvtas  v10.4s, v16.4s\n"
               "fcvtas  v11.4s, v17.4s\n"
               "fcvtas  v14.4s, v18.4s\n"
               "fcvtas  v15.4s, v19.4s\n"
               "fcvtas  v20.4s, v8.4s\n"
               "fcvtas  v21.4s, v9.4s\n"
               "fcvtas  v22.4s, v12.4s\n"
               "fcvtas  v23.4s, v13.4s\n"
               /* int32-int16 */
               "sqxtn   v16.4h, v10.4s\n"
               "sqxtn   v17.4h, v11.4s\n"
               "sqxtn   v18.4h, v14.4s\n"
               "sqxtn   v19.4h, v15.4s\n"
               "sqxtn   v8.4h, v20.4s\n"
               "sqxtn   v9.4h, v21.4s\n"
               "sqxtn   v12.4h, v22.4s\n"
               "sqxtn   v13.4h, v23.4s\n"
               /* int16-int8 */
               "sqxtn  v10.8b, v16.8h\n"
               "sqxtn  v11.8b, v17.8h\n"
               "sqxtn  v14.8b, v18.8h\n"
               "sqxtn  v15.8b, v19.8h\n"
               "sqxtn  v20.8b, v8.8h\n"
               "sqxtn  v21.8b, v9.8h\n"
               "sqxtn  v22.8b, v12.8h\n"
               "sqxtn  v23.8b, v13.8h\n"
               "str    s10, [%[doutc0r0]], #4 \n"  /* store c0r0*/
               "str    s11, [%[doutc2r0]], #4 \n"  /* store c2r0*/
               "str    s14, [%[doutc1r0]], #4 \n"  /* store c1r0*/
               "str    s15, [%[doutc3r0]], #4 \n"  /* store c3r0*/
               "subs   %w[cnt], %w[cnt],  #1   \n" /* loop count -1*/
               "str    s20, [%[doutc4r0]], #4  \n" /* store c0r0*/
               "str    s21, [%[doutc6r0]], #4  \n" /* store c2r0*/
               "str    s22, [%[doutc5r0]], #4 \n"  /* store c1r0*/
               "str    s23, [%[doutc7r0]], #4 \n"  /* store c3r0*/
               "bne    1b                      \n" /* jump to main loop*/
               : [doutc0r0] "+r"(dout0),
                 [doutc1r0] "+r"(dout1),
                 [doutc2r0] "+r"(dout2),
                 [doutc3r0] "+r"(dout3),
                 [doutc4r0] "+r"(dout4),
                 [doutc5r0] "+r"(dout5),
                 [doutc6r0] "+r"(dout6),
                 [doutc7r0] "+r"(dout7),
                 [ptr_din] "+r"(din),
                 [cnt] "+r"(cnt)
               : [scale0] "w"(scale0),
                 [scale1] "w"(scale1),
                 [bias0] "w"(bias0),
                 [bias1] "w"(bias1),
                 [vmax] "w"(vmax),
                 [flag_act] "r"(flag_act), 
                 [alpha] "r"(alpha)
               : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                 "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                 "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                 "v25", "v31"
               );
#else
  float vmax[4] = {-127.f, -127.f, -127.f, -127.f};
  asm volatile(INT32_NCHWC8_TO_NCHW_FP32 /* set +-0.5 offset */
               "vmov.f32 q10, #-0.5\n"
               "vmov.f32 q9, #0.5\n"
               "vcgt.f32   q11, q0, q8   @ get mask > 0, in0\n"
               "vbif.f32   q9, q10, q11   @ get right offset\n"
               "vadd.f32   q0, q0, q9\n"
               "vmov.f32 q9, #0.5\n"
               "vcgt.f32   q11, q2, q8   @ get mask > 0, in0\n"
               "vbif.f32   q9, q10, q11   @ get right offset\n"
               "vadd.f32   q2, q2, q9\n"
               "vmov.f32 q9, #0.5\n"
               "vcgt.f32   q11, q4, q8   @ get mask > 0, in0\n"
               "vbif.f32   q9, q10, q11   @ get right offset\n"
               "vadd.f32   q4, q4, q9\n"
               "vmov.f32 q9, #0.5\n"
               "vcgt.f32   q11, q6, q8   @ get mask > 0, in0\n"
               "vbif.f32   q9, q10, q11   @ get right offset\n"
               "vadd.f32   q6, q6, q9\n"
               "vmov.f32 q9, #0.5\n"
               "vcgt.f32   q11, q1, q8   @ get mask > 0, in0\n"
               "vbif.f32   q9, q10, q11   @ get right offset\n"
               "vadd.f32   q1, q1, q9\n"
               "vmov.f32 q9, #0.5\n"
               "vcgt.f32   q11, q3, q8   @ get mask > 0, in0\n"
               "vbif.f32   q9, q10, q11   @ get right offset\n"
               "vadd.f32   q3, q3, q9\n"
               "vmov.f32 q9, #0.5\n"
               "vcgt.f32   q11, q5, q8   @ get mask > 0, in0\n"
               "vbif.f32   q9, q10, q11   @ get right offset\n"
               "vadd.f32   q5, q5, q9\n"
               "vmov.f32 q9, #0.5\n"
               "vcgt.f32   q11, q7, q8   @ get mask > 0, in0\n"
               "vbif.f32   q9, q10, q11   @ get right offset\n"
               "vld1.32 {d22-d23}, [%[vmax]] \n"
               "vadd.f32   q7, q7, q9\n"
               /* data >= -127 */
               "vcge.f32 q8, q0, q11     @ q10 >= vmax \n"
               "vcge.f32 q9, q2, q11     @ q10 >= vmax \n"
               "vcge.f32 q10, q4, q11     @ q10 >= vmax \n"
               /* choose data */
               "vbif q0, q11, q8    @ choose \n"
               "vcge.f32 q8, q6, q11     @ q10 >= vmax \n"
               "vbif q2, q11, q9    @ choose \n"
               "vbif q4, q11, q10    @ choose \n"
               "vbif q6, q11, q8    @ choose \n"
               /* fp32 to int32 */
               "vcvt.s32.f32  q8, q0    @ cvt to int32\n"
               "vcvt.s32.f32  q9, q2    @ cvt to int32\n"
               "vcvt.s32.f32  q10, q4   @ cvt to int32\n"
               "vcvt.s32.f32  q11, q6   @ cvt to int32\n"
               /* int32 to int16 */
               "vqmovn.s32 d0, q8       @ cnt to int16\n"
               "vqmovn.s32 d4, q9       @ cnt to int16\n"
               "vqmovn.s32 d8, q10      @ cnt to int16\n"
               "vqmovn.s32 d12, q11      @ cnt to int16\n"
               /* data >= -127 */
               "vld1.32 {d22-d23}, [%[vmax]] \n"
               "vcge.f32 q8, q1, q11     @ q10 >= vmax \n"
               "vcge.f32 q9, q3, q11     @ q10 >= vmax \n"
               "vcge.f32 q10, q5, q11     @ q10 >= vmax \n"
               /* choose data */
               "vbif q1, q11, q8    @ choose \n"
               "vcge.f32 q8, q7, q11     @ q10 >= vmax \n"
               "vbif q3, q11, q9    @ choose \n"
               "vbif q5, q11, q10    @ choose \n"
               "vbif q7, q11, q8    @ choose \n"
               /* fp32 to int32 */
               "vcvt.s32.f32  q8, q1    @ cvt to int32\n"
               "vcvt.s32.f32  q9, q3    @ cvt to int32\n"
               "vcvt.s32.f32  q10, q5   @ cvt to int32\n"
               "vcvt.s32.f32  q11, q7   @ cvt to int32\n"
               /* int32 to int16 */
               "vqmovn.s32 d2, q8       @ cnt to int16\n"
               "vqmovn.s32 d6, q9       @ cnt to int16\n"
               "vqmovn.s32 d10, q10     @ cnt to int16\n"
               "vqmovn.s32 d14, q11     @ cnt to int16\n"
               /* int16 to int8 */
               "vqmovn.s16 d16, q0      @ cnt to int8\n"
               "vqmovn.s16 d17, q2      @ cnt to int8\n"
               "vqmovn.s16 d18, q4      @ cnt to int8\n"
               "vqmovn.s16 d19, q6      @ cnt to int8\n"
               "vst1.32    {d16[0]}, [%[doutc0r0]]!\n"
               "vqmovn.s16 d20, q1      @ cnt to int8\n"
               "vst1.32    {d17[0]}, [%[doutc1r0]]!\n"
               "vqmovn.s16 d21, q3      @ cnt to int8\n"
               "vst1.32    {d18[0]}, [%[doutc2r0]]!\n"
               "vqmovn.s16 d22, q5      @ cnt to int8\n"
               "vst1.32    {d19[0]}, [%[doutc3r0]]!\n"
               "vqmovn.s16 d23, q7      @ cnt to int8\n"
               "subs   %[cnt], #1\n"
               "vst1.32    {d20[0]}, [%[doutc4r0]]!\n"
               "vst1.32    {d21[0]}, [%[doutc5r0]]!\n"
               "vst1.32    {d22[0]}, [%[doutc6r0]]!\n"
               "vst1.32    {d23[0]}, [%[doutc7r0]]!\n"
               "bne    1b\n" /* jump to main loop*/
               : [doutc0r0] "+r"(dout0),
                 [doutc1r0] "+r"(dout1),
                 [doutc2r0] "+r"(dout2),
                 [doutc3r0] "+r"(dout3),
                 [doutc4r0] "+r"(dout4),
                 [doutc5r0] "+r"(dout5),
                 [doutc6r0] "+r"(dout6),
                 [doutc7r0] "+r"(dout7),
                 [ptr_din] "+r"(din),
                 [cnt] "+r"(cnt)
               : [scale0] "w"(scale0),
                 [scale1] "w"(scale1),
                 [bias0] "w"(bias0),
                 [bias1] "w"(bias1),
                 [vmax] "r"(vmax),
                 [flag_act] "r"(flag_act), 
                 [alpha] "r"(alpha)
               : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                 "q7", "q8", "q9", "q10", "q11"
               );
#endif
  // clang-format on
}

/*wirte result in outputs
* input din: [n, c / 8, h, w * 8], output dout: [n, c, h, w]
*/
template <typename Dtype>
inline void write_int32_nchwc8_to_nchw(const int* din,
                                       Dtype* dout,
                                       int cs,
                                       int ce,
                                       int hs,
                                       int he,
                                       int ws,
                                       int we,
                                       int channel,
                                       int height,
                                       int width,
                                       int flag_act,
                                       float* alpha,
                                       const float* bias,
                                       bool flag_bias,
                                       Dtype* trash_ptr,
                                       const float* scale) {
  int size_c_out = width * height;

  Dtype* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
  Dtype* doutc1r0 = doutc0r0 + size_c_out;
  Dtype* doutc2r0 = doutc1r0 + size_c_out;
  Dtype* doutc3r0 = doutc2r0 + size_c_out;
  Dtype* doutc4r0 = doutc3r0 + size_c_out;
  Dtype* doutc5r0 = doutc4r0 + size_c_out;
  Dtype* doutc6r0 = doutc5r0 + size_c_out;
  Dtype* doutc7r0 = doutc6r0 + size_c_out;

  const int* ptr_din = din;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int w_stride = we - ws;
  int valid_w = (we > width ? width : we) - ws;
  int cnt = valid_w / 4;
  int remain = valid_w & 3;

  float32x4_t w_scale0 = vld1q_f32(scale);
  float32x4_t w_scale1 = vld1q_f32(scale + 4);
  float32x4_t w_bias0 = flag_bias ? vld1q_f32(bias) : vdupq_n_f32(0.f);
  float32x4_t w_bias1 = flag_bias ? vld1q_f32(bias + 4) : vdupq_n_f32(0.f);

  for (int i = 0; i < size_h; i++) {
    int size_w = i * width;
    Dtype* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
    Dtype* doutc1_ptr = doutc1r0 + size_w;
    Dtype* doutc2_ptr = doutc2r0 + size_w;
    Dtype* doutc3_ptr = doutc3r0 + size_w;
    Dtype* doutc4_ptr = doutc4r0 + size_w;
    Dtype* doutc5_ptr = doutc5r0 + size_w;
    Dtype* doutc6_ptr = doutc6r0 + size_w;
    Dtype* doutc7_ptr = doutc7r0 + size_w;
    if (ce > channel) {
      switch (ce - channel) {
        case 7:
          doutc1_ptr = trash_ptr;
        case 6:
          doutc2_ptr = trash_ptr;
        case 5:
          doutc3_ptr = trash_ptr;
        case 4:
          doutc4_ptr = trash_ptr;
        case 3:
          doutc5_ptr = trash_ptr;
        case 2:
          doutc6_ptr = trash_ptr;
        case 1:
          doutc7_ptr = trash_ptr;
        default:
          break;
      }
    }
    ptr_din = din + i * w_stride * 8;
    const int* din_hei_ptr = ptr_din;
    if (cnt > 0) {
      int32_nchwc8_kernel(doutc0_ptr,
                          doutc1_ptr,
                          doutc2_ptr,
                          doutc3_ptr,
                          doutc4_ptr,
                          doutc5_ptr,
                          doutc6_ptr,
                          doutc7_ptr,
                          din_hei_ptr,
                          cnt,
                          w_scale0,
                          w_scale1,
                          w_bias0,
                          w_bias1,
                          flag_act,
                          alpha);
    }
    if (remain > 0) {
      int offset = 32 * cnt;
      din_hei_ptr = ptr_din + offset;
      for (int j = 0; j < remain; ++j) {
        if (flag_bias) {
          *(doutc0_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[0], scale[0], bias[0], flag_act, alpha[0]);
          *(doutc1_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[1], scale[1], bias[1], flag_act, alpha[0]);
          *(doutc2_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[2], scale[2], bias[2], flag_act, alpha[0]);
          *(doutc3_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[3], scale[3], bias[3], flag_act, alpha[0]);
          *(doutc4_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[4], scale[4], bias[4], flag_act, alpha[0]);
          *(doutc5_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[5], scale[5], bias[5], flag_act, alpha[0]);
          *(doutc6_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[6], scale[6], bias[6], flag_act, alpha[0]);
          *(doutc7_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[7], scale[7], bias[7], flag_act, alpha[0]);
        } else {
          *(doutc0_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[0], scale[0], 0.f, flag_act, alpha[0]);
          *(doutc1_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[1], scale[1], 0.f, flag_act, alpha[0]);
          *(doutc2_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[2], scale[2], 0.f, flag_act, alpha[0]);
          *(doutc3_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[3], scale[3], 0.f, flag_act, alpha[0]);
          *(doutc4_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[4], scale[4], 0.f, flag_act, alpha[0]);
          *(doutc5_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[5], scale[5], 0.f, flag_act, alpha[0]);
          *(doutc6_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[6], scale[6], 0.f, flag_act, alpha[0]);
          *(doutc7_ptr++) = cvt_kernel<Dtype>(
              din_hei_ptr[7], scale[7], 0.f, flag_act, alpha[0]);
        }
        din_hei_ptr += 8;
      }
    }
  }
}

/*
* din [n, hei_n, ch_n, w]
* dout [n, ch_n, hei_n, w]
*/
template <typename dtype>
static bool write_to_output_numc(const dtype* din,
                                 dtype* dout,
                                 int ch_n,
                                 int hei_n,
                                 int cs,
                                 int ce,
                                 int hs,
                                 int he,
                                 int ws,
                                 int we,
                                 int channel,
                                 int height,
                                 int width,
                                 bool flag_relu,
                                 dtype* trash_ptr) {
  if (ch_n <= 0 || hei_n <= 0) {
    LOG(ERROR) << "ch_n and hei_n are more than zero";
    return false;
  }
  int size_c_out = width * height;

  dtype* out_array[ch_n];
  out_array[0] = dout + cs * size_c_out + hs * width + ws;

  for (int i = 1; i < ch_n; i++) {
    out_array[i] = out_array[i - 1] + size_c_out;
  }

  const dtype* ptr_din = din;

  int cremain = ce - channel;
  for (int i = 1; i <= cremain; i++) {
    out_array[ch_n - i] = trash_ptr;
  }

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int size_w = we - ws;

  int size_c_in = ch_n * size_w;

  size_t valid_w_byte = width * sizeof(dtype);

  if (flag_relu) {
    for (int h = 0; h < size_h; h++) {
      const dtype* din_ptr = din + h * size_c_in;
      for (int i = 0; i < ch_n; i++) {
        dtype* dout_ptr = out_array[i] + h * width;
        for (int k = 0; k < width; k++) {
          *(dout_ptr++) = LITEMAX(din_ptr[k], 0);
        }
        din_ptr += size_w;
      }
    }
  } else {
    for (int h = 0; h < size_h; h++) {
      const dtype* din_ptr = din + h * size_c_in;
      for (int i = 0; i < ch_n; i++) {
        dtype* dout_ptr = out_array[i] + h * width;
        memcpy(dout_ptr, din_ptr, valid_w_byte);
        din_ptr += size_w;
      }
    }
  }
  return true;
}

/// ch_n == ce - cs ??
/// hei_n == he - hs ??
/// channel height width ? -> output
template <typename ditype, typename dotype>
static bool write2_to_output_numc(const ditype* din,
                                  dotype* dout,
                                  int ch_n,
                                  int hei_n,
                                  int cs,
                                  int ce,
                                  int hs,
                                  int he,
                                  int ws,
                                  int we,
                                  int channel,
                                  int height,
                                  int width,
                                  bool flag_relu,
                                  dotype* trash_ptr,
                                  float const* scales) {
  // static_assert(std::is_same<dotype, float>::value, "just support float");

  if (ch_n <= 0 || hei_n <= 0) {
    LOG(ERROR) << "ch_n and hei_n are more than zero";
    return false;
  }

  int size_c_out = width * height;

  dotype* out_array[ch_n];
  out_array[0] = dout + cs * size_c_out + hs * width + ws;

  for (int i = 1; i < ch_n; i++) {
    out_array[i] = out_array[i - 1] + size_c_out;
  }

  const ditype* ptr_din = din;

  int cremain = ce - channel;
  for (int i = 1; i <= cremain; i++) {
    out_array[ch_n - i] = trash_ptr;
  }

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int size_w = we - ws;

  int size_c_in = ch_n * size_w;

  size_t valid_w_byte = width * sizeof(ditype);

  if (flag_relu) {
    for (int h = 0; h < size_h; h++) {
      ditype const* din_ptr = din + h * size_c_in;
      for (int i = 0; i < ch_n; i++) {
        float const ws = scales[(i + cs) % ch_n];
        dotype* dout_ptr = out_array[i] + h * width;
        for (int k = 0; k < width; k++) {
          *(dout_ptr++) = LITEMAX(din_ptr[k] * ws, 0);
        }
        din_ptr += size_w;
      }
    }
  } else {
    for (int h = 0; h < size_h; h++) {
      ditype const* din_ptr = din + h * size_c_in;
      for (int i = 0; i < ch_n; i++) {
        dotype* dout_ptr = out_array[i] + h * width;

        float const* ws = &scales[(i + cs) % ch_n];
        int32_to_dtype(din_ptr, dout_ptr, ws, 1, 1, width);

        din_ptr += size_w;
      }
    }
  }
  return true;
}
/**
* innput din: nchwc(num)
*/
inline bool fill_packed_bias_nxmw_fp32(
    const float* bias, float* dout, int ch_n, int hei_n, int wround) {
  if (ch_n <= 0 || hei_n <= 0) {
    LOG(ERROR) << "ch_n and hei_n are more than zero";
    return false;
  }
  int cnt_ch = ch_n / 4;
  int size = wround * ch_n;
  for (int h = 0; h < hei_n; h++) {
    float* dout_ptr = dout + h * size;
    for (int i = 0; i < wround; i++) {
      const float* bias_ptr = bias;
      int j = 0;
      for (; j < cnt_ch; j++) {
        float32x4_t vb = vld1q_f32(bias_ptr);
        bias_ptr += 4;

        vst1q_f32(dout_ptr, vb);
        dout_ptr += 4;
      }
      j = j * 4;
      for (; j < ch_n; j++) {
        *dout_ptr = *bias_ptr;
        dout_ptr++;
        bias_ptr++;
      }
    }
  }
}

inline bool fill_packed_bias_nxmw_int8(
    const int* bias, int* dout, int ch_n, int hei_n, int wround) {
  if (ch_n <= 0 || hei_n <= 0) {
    LOG(ERROR) << "ch_n and hei_n are more than zero";
    return false;
  }
  int cnt_ch = ch_n / 4;
  int size = wround * ch_n;
  for (int h = 0; h < hei_n; h++) {
    int* dout_ptr = dout + h * size;
    for (int i = 0; i < wround; i++) {
      const int* bias_ptr = bias;
      int j = 0;
      for (; j < cnt_ch; j++) {
        int32x4_t vb = vld1q_s32(bias_ptr);
        bias_ptr += 4;

        vst1q_s32(dout_ptr, vb);
        dout_ptr += 4;
      }
      j = j * 4;
      for (; j < ch_n; j++) {
        *dout_ptr = *bias_ptr;
        dout_ptr++;
        bias_ptr++;
      }
    }
  }
  return true;
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
