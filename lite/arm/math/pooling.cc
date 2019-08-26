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

#include "lite/arm/math/pooling.h"
#include <algorithm>
#include <limits>
#include "lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void pooling_basic(const float* din,
                   float* dout,
                   int num,
                   int chout,
                   int hout,
                   int wout,
                   int chin,
                   int hin,
                   int win,
                   const std::vector<int>& ksize,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   bool global_pooling,
                   bool exclusive,
                   bool adaptive,
                   bool ceil_mode,
                   bool use_quantizer,
                   const std::string& pooling_type) {
  // no need to pad input tensor, border is zero pad inside this function
  memset(dout, 0, num * chout * hout * wout * sizeof(float));
  int kernel_h = ksize[0];
  int kernel_w = ksize[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[1];
  int size_channel_in = win * hin;
  int size_channel_out = wout * hout;
  if (global_pooling) {
    if (pooling_type == "max") {  // Pooling_max
      for (int n = 0; n < num; ++n) {
        float* dout_batch = dout + n * chout * size_channel_out;
        const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
        for (int c = 0; c < chout; ++c) {
          const float* din_ch = din_batch + c * size_channel_in;  // in address
          float tmp1 = din_ch[0];
          for (int i = 0; i < size_channel_in; ++i) {
            float tmp2 = din_ch[i];
            tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
          }
          dout_batch[c] = tmp1;
        }
      }
    } else if (pooling_type == "avg") {
      // Pooling_average_include_padding
      // Pooling_average_exclude_padding
      for (int n = 0; n < num; ++n) {
        float* dout_batch = dout + n * chout * size_channel_out;
        const float* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
        for (int c = 0; c < chout; ++c) {
          const float* din_ch = din_batch + c * size_channel_in;  // in address
          float sum = 0.f;
          for (int i = 0; i < size_channel_in; ++i) {
            sum += din_ch[i];
          }
          dout_batch[c] = sum / size_channel_in;
        }
      }
    } else {
      LOG(FATAL) << "unsupported pooling type: " << pooling_type;
    }
  } else {
    for (int ind_n = 0; ind_n < num; ++ind_n) {
      for (int ind_c = 0; ind_c < chin; ++ind_c) {
        for (int ind_h = 0; ind_h < hout; ++ind_h) {
          int sh = ind_h * stride_h;
          int eh = sh + kernel_h;
          sh = (sh - pad_h) < 0 ? 0 : sh - pad_h;
          eh = (eh - pad_h) > hin ? hin : eh - pad_h;
          for (int ind_w = 0; ind_w < wout; ++ind_w) {
            int sw = ind_w * stride_w;
            int ew = sw + kernel_w;
            sw = (sw - pad_w) < 0 ? 0 : sw - pad_w;
            ew = (ew - pad_w) > win ? win : ew - pad_w;
            float result = static_cast<float>(0);
            int dst_ind = (ind_n * chout + ind_c) * size_channel_out +
                          ind_h * wout + ind_w;
            for (int kh = sh; kh < eh; ++kh) {
              for (int kw = sw; kw < ew; ++kw) {
                int src_ind =
                    (ind_n * chin + ind_c) * size_channel_in + kh * win + kw;
                if (kh == sh && kw == sw) {
                  result = din[src_ind];
                } else {
                  if (pooling_type == "max") {
                    result = result >= din[src_ind] ? result : din[src_ind];
                  } else if (pooling_type == "avg") {
                    result += din[src_ind];
                  }
                }
              }
            }
            if (pooling_type == "avg") {
              if (exclusive) {
                int div = (ew - sw) * (eh - sh);
                div = div > 0 ? div : 1;
                result /= div;
              } else {
                int bh = kernel_h;
                int bw = kernel_w;
                if (ew == win) {
                  bw = sw + kernel_w >= win + pad_w ? win + pad_w
                                                    : sw + kernel_w;
                  bw -= sw;
                }
                if (eh == hin) {
                  bh = sh + kernel_h >= hin + pad_h ? hin + pad_h
                                                    : sh + kernel_h;
                  bh -= sh;
                }
                result /= bh * bw;
              }
            }
            dout[dst_ind] = result;
          }
        }
      }
    }
  }
}

void pooling_global_max(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win) {
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);

  int cnt = size_channel_in / 16;

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; ++c) {
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      int i = 0;
      float32x4_t vmax = vdupq_n_f32(std::numeric_limits<float>::lowest());
#ifdef __aarch64__
      for (; i < cnt; i++) {
        float32x4_t vdin1 = vld1q_f32(data_in_channel);
        float32x4_t vdin2 = vld1q_f32(data_in_channel + 4);
        float32x4_t vdin3 = vld1q_f32(data_in_channel + 8);
        float32x4_t vdin4 = vld1q_f32(data_in_channel + 12);
        float32x4_t vtmp1 = vmaxq_f32(vdin1, vdin2);
        float32x4_t vtmp2 = vmaxq_f32(vdin3, vdin4);
        float32x4_t vtmp3 = vmaxq_f32(vtmp1, vtmp2);
        vmax = vmaxq_f32(vmax, vtmp3);
        data_in_channel += 16;
      }
#else
      if (cnt > 0) {
        int num = cnt;
        asm volatile(
            "1:                                               @ main loop\n"
            "vld1.f32   {d0-d3}, [%[data_in_channel]]!        @ load q0, q1, "
            "data_in_channel\n"
            "vld1.f32   {d4-d7}, [%[data_in_channel]]!        @ load q2, q3, "
            "data_in_channel\n"
            "vmax.f32   q4, q0, q1                            @ max vmax, "
            "vmax, data_in_channel\n"
            "vmax.f32   q5, q2, q3                            @ max vmax, "
            "vmax, data_in_channel\n"
            "vmax.f32   q0, q4, q5                            @ max vmax, "
            "vmax, data_in_channel\n"
            "vmax.f32   %q[vmax], %q[vmax], q0                @ max vmax, "
            "vmax, data_in_channel\n"
            "subs       %[num], #1                            @ subs num, 1\n"
            "bne        1b                                    @ bne num\n"
            : [data_in_channel] "+r"(data_in_channel),
              [num] "+r"(num),
              [vmax] "+w"(vmax)
            :
            : "cc", "memory", "q0", "q1");
      }
#endif  //  __aarch64__
      float32x2_t vmax_tmp = vmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
      float tmp1 = vget_lane_f32(vmax_tmp, 0);
      float tmp2 = vget_lane_f32(vmax_tmp, 1);
      float max_tmp = tmp1 > tmp2 ? tmp1 : tmp2;
      for (i = cnt * 16; i < size_channel_in; ++i) {
        max_tmp = max_tmp > data_in_channel[0] ? max_tmp : data_in_channel[0];
        data_in_channel++;
      }
      data_out_batch[c] = max_tmp;
    }
  }
}

void pooling_global_avg(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win) {
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);

  int cnt = size_channel_in / 16;

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      const float* data_in_channel =
          data_in_batch + c * size_channel_in;  // in address
      int i = 0;
      float32x4_t vsum = vdupq_n_f32(0.0f);
#ifdef __aarch64__
      for (; i < cnt; i++) {  //
        vsum = vaddq_f32(vld1q_f32(data_in_channel), vsum);
        data_in_channel += 4;
      }
#else
      int num = cnt;
      if (num > 0) {
        asm volatile(
            "1:                                        @main loop\n"
            "vld1.f32   {d0-d1}, [%[data_in_channel]]!        @load q1, "
            "data_in_channel\n"
            "vadd.f32   %q[vsum], %q[vsum], q0                @add vmax, vmax, "
            "data_in_channel\n"
            "subs        %[num], #1                           @subs num, 1\n"
            "bne        1b                              @bne num\n"
            : [data_in_channel] "+r"(data_in_channel),
              [num] "+r"(num),
              [vsum] "+w"(vsum)
            :
            : "cc", "memory", "q0");
      }
#endif  //  __aarch64__
      float32x2_t vsum_tmp = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
      float sum = vget_lane_f32(vsum_tmp, 0) + vget_lane_f32(vsum_tmp, 1);
      for (i = cnt * 4; i < size_channel_in; i++) {
        sum += data_in_channel[0];
        data_in_channel++;
      }
      data_out_batch[c] = sum / size_channel_in;
    }
  }
}

void pooling2x2s2_max(const float* din,
                      float* dout,
                      int num,
                      int chout,
                      int hout,
                      int wout,
                      int chin,
                      int hin,
                      int win) {
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);

  const int K = 2;
  const int P = 0;
  const int S = 2;

  int w_unroll_size = wout / 4;
  int w_unroll_remian = wout - w_unroll_size * 4;

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout * size_channel_out;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + win;
      for (int h = 0; h < hout; h++) {
        float* dr_out = data_out_channel;
        auto dr0 = r0;
        auto dr1 = r1;
        if (h * S + K - P > hin) {
          dr1 = r0;
        }
#ifdef __aarch64__
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              "1: \n"                                /* load bias to q2, q3*/
              "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/
              "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/
              "fmax  v4.4s, v0.4s, v1.4s\n" /*  max 0, 2, 4, 6 and 1, 3, 5, 7,
                                               r0 */
              "fmax  v5.4s, v2.4s, v3.4s\n" /*  max 0, 2, 4, 6 and 1, 3, 5, 7,
                                               r1 */
              "fmax  v0.4s, v4.4s, v5.4s\n" /* max reduce */
              "st1  {v0.4s}, [%[dr_out]], #16\n"    /* store 4 out, dr_out */
              "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/
              "bne       1b\n"                      /* bne s3_max_loop_mid */
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              :
              : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5");
        }
#else
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              "1:                                             @ main loop\n"
              "vld2.f32  {d0-d3}, [%[dr0]]!                   @ load d0-d3, "
              "dr0, 0-7\n"
              "vld2.f32  {d4-d7}, [%[dr1]]!                   @ load d4-d7, "
              "dr1, 0-7\n"
              "vmax.f32  q4, q0, q1                           @ max 0, 2, 4, 6 "
              "and 1, 3, 5, 7, r0\n"
              "vmax.f32  q5, q2, q3                           @ max 0, 2, 4, 6 "
              "and 1, 3, 5, 7, r1\n"
              "vmax.f32  q0, q4, q5                           @ max reduce\n"
              "vst1.f32  {d0-d1}, [%[dr_out]]!                @ store 4 out, "
              "dr_out\n"
              "subs      %[cnt_num], #1                       @ subs cnt_num, "
              "#1\n"
              "bne       1b                                   @ bne "
              "s3_max_loop_mid\n"
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              :
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5");
        }
#endif
        // deal with right pad
        int rem = win - (w_unroll_size * 4) * S;
        int wstart = 0;
        for (int j = 0; j < w_unroll_remian; ++j) {
          int wend = std::min(wstart + K, rem);
          float tmp = dr0[wstart];
          for (int i = wstart; i < wend; i++) {
            tmp = std::max(tmp, dr0[i]);
            tmp = std::max(tmp, dr1[i]);
          }
          *(dr_out++) = tmp;
          wstart += S;
        }
        r0 = r1 + win;
        r1 = r0 + win;
        data_out_channel += wout;
      }
    }
  }
}

void pooling2x2s2_avg(const float* din,
                      float* dout,
                      int num,
                      int chout,
                      int hout,
                      int wout,
                      int chin,
                      int hin,
                      int win,
                      bool exclusive) {
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);

  const int K = 2;
  const int P = 0;
  const int S = 2;

  int w_unroll_size = wout / 4;
  int w_unroll_remian = wout - w_unroll_size * 4;
  float32x4_t vcoef = vdupq_n_f32(0.25f);  // divided by 4

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout * size_channel_out;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + win;
      for (int h = 0; h < hout; h++) {
        float* dr_out = data_out_channel;
        auto dr0 = r0;
        auto dr1 = r1;
        if (h * S + K - P > hin) {
          dr1 = r0;
        }
#ifdef __aarch64__
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              "1: \n"                                /* load bias to q2, q3*/
              "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/
              "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/
              "fadd v4.4s, v0.4s, v1.4s\n" /*  add 0, 2, 4, 6 and 1, 3, 5, 7, r0
                                              */
              "fadd v5.4s, v2.4s, v3.4s\n" /*  add 0, 2, 4, 6 and 1, 3, 5, 7, r1
                                              */
              "fadd v1.4s, v4.4s, v5.4s\n" /* add reduce */
              "fmul v0.4s, v1.4s, %[vcoef].4s\n"    /* mul coef */
              "st1  {v0.4s}, [%[dr_out]], #16\n"    /* store 4 out, dr_out */
              "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/
              "bne       1b\n"                      /* bne s3_max_loop_mid */
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vcoef] "w"(vcoef)
              : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5");
        }
#else
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              "1:                                             @ main loop\n"
              "vld2.f32  {d0-d3}, [%[dr0]]!                   @ load d0-d3, "
              "dr0, 0-7\n"
              "vld2.f32  {d4-d7}, [%[dr1]]!                   @ load d4-d7, "
              "dr1, 0-7\n"
              "vadd.f32  q4, q0, q1                           @ add 0, 2, 4, 6 "
              "and 1, 3, 5, 7, r0\n"
              "vadd.f32  q5, q2, q3                           @ add 0, 2, 4, 6 "
              "and 1, 3, 5, 7, r1\n"
              "vadd.f32  q1, q4, q5                           @ add reduce\n"
              "vmul.f32  q0, q1, %q[vcoef]                    @ mul coef\n"
              "vst1.f32  {d0-d1}, [%[dr_out]]!                @ store 4 out, "
              "dr_out\n"
              "subs      %[cnt_num], #1                       @ subs cnt_num, "
              "#1\n"
              "bne       1b                                   @ bne "
              "s3_max_loop_mid\n"
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vcoef] "w"(vcoef)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5");
        }
#endif
        // deal with right pad
        int rem = win - (w_unroll_size * 4) * S;
        int wstart = 0;
        for (int j = 0; j < w_unroll_remian; ++j) {
          int wend = std::min(wstart + K, rem);
          float coef = 0.5f / (wend - wstart);
          float tmp = 0.f;
          for (int i = wstart; i < wend; i++) {
            tmp += dr0[i] + dr1[i];
          }
          *(dr_out++) = tmp * coef;
          wstart += S;
        }

        r0 = r1 + win;
        r1 = r0 + win;
        data_out_channel += wout;
      }
    }
  }
}

void pooling3x3s1p1_max(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win) {
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);

  const int K = 3;
  const int P = 1;
  const int S = 1;
  const int WUNROLL = 4;

  int w_unroll_size = wout / WUNROLL;
  int w_unroll_remian = wout - w_unroll_size * WUNROLL;
  if (w_unroll_remian == 0) {
    w_unroll_size -= 1;
    w_unroll_remian = wout - w_unroll_size * WUNROLL;
  }

  float32x4_t vmin = vdupq_n_f32(std::numeric_limits<float>::lowest());

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout * size_channel_out;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + win;
      const float* r2 = r1 + win;
      for (int h = 0; h < hout; h++) {
        float* dr_out = data_out_channel;
        auto dr0 = r0;
        auto dr1 = r1;
        auto dr2 = r2;
        if (h == 0) {
          dr0 = r0;
          dr1 = r0;
          dr2 = r1;
        } else {
          r0 = r1;
          r1 = r2;
          r2 = r1 + win;
        }
        if (h * S + K - P > hin) {
          switch (h * S + K - P - hin) {
            case 2:
              dr1 = dr0;
            case 1:
              dr2 = dr0;
            default:
              break;
          }
        }
#ifdef __aarch64__
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              /* preocess left */
              "ldr  q0, [%[dr0]], #16\n"           /* load q0, dr0, 0-3*/
              "ldr  d3, [%[dr0]]\n"                /* load q3, dr0, 4-5*/
              "ldr  q1, [%[dr1]], #16\n"           /* load q1, dr1, 0-3*/
              "ldr  d4, [%[dr1]]\n"                /* load q4, dr1, 4-5*/
              "ldr  q2, [%[dr2]], #16\n"           /* load q2, dr2, 0-3*/
              "ldr  d5, [%[dr2]]\n"                /* load q5, dr2, 4-5*/
              "ext   v6.16b, v0.16b, v3.16b, #4\n" /* ext 1, 2, 3, 4, r0 */
              "ext   v7.16b, v1.16b, v4.16b, #4\n" /* ext 1, 2, 3, 4, r1 */
              "ext   v8.16b, v2.16b, v5.16b, #4\n" /* ext 1, 2, 3, 4, r2 */
              "ext   v9.16b, %[vmin].16b, v0.16b, #12\n"  /* ext -1, 0, 1, 2, r0
                                                             */
              "ext   v10.16b, %[vmin].16b, v1.16b, #12\n" /* ext -1, 0, 1, 2, r1
                                                             */
              "ext   v11.16b, %[vmin].16b, v2.16b, #12\n" /* ext -1, 0, 1, 2, r2
                                                             */
              "fmax v3.4s, v0.4s, v1.4s\n"
              "fmax v4.4s, v2.4s, v6.4s\n"
              "fmax v5.4s, v7.4s, v8.4s\n"
              "fmax v6.4s, v9.4s, v10.4s\n"
              "fmax v0.4s, v11.4s, v3.4s\n"
              "fmax v1.4s, v4.4s, v5.4s\n"
              "fmax v2.4s, v6.4s, v0.4s\n"
              "fmax v3.4s, v2.4s, v1.4s\n"
              "subs %[dr0], %[dr0], #4\n"
              "subs %[dr1], %[dr1], #4\n"
              "subs %[dr2], %[dr2], #4\n"
              "st1  {v3.4s}, [%[dr_out]], #16\n"    /* store 4 out, dr_out */
              "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/
              "ble       2f\n"                      /* jump to end */
              "1: \n"                               /* */
              "ldr  q0, [%[dr0]], #16\n"            /* load q0, dr0, 0-3*/
              "ldr  d3, [%[dr0]]\n"                 /* load q3, dr0, 4-5*/
              "ldr  q1, [%[dr1]], #16\n"            /* load q1, dr1, 0-3*/
              "ldr  d4, [%[dr1]]\n"                 /* load q4, dr1, 4-5*/
              "ldr  q2, [%[dr2]], #16\n"            /* load q2, dr2, 0-3*/
              "ldr  d5, [%[dr2]]\n"                 /* load q5, dr2, 4-5*/
              "ext   v6.16b, v0.16b, v3.16b, #4\n"  /* ext 1, 2, 3, 4, r0 */
              "ext   v7.16b, v1.16b, v4.16b, #4\n"  /* ext 1, 2, 3, 4, r1 */
              "ext   v8.16b, v2.16b, v5.16b, #4\n"  /* ext 1, 2, 3, 4, r2 */
              "ext   v9.16b, v0.16b, v3.16b, #8\n"  /* ext 2, 3, 4, 5, r0 */
              "ext   v10.16b, v1.16b, v4.16b, #8\n" /* ext 2, 3, 4, 5, r1 */
              "ext   v11.16b, v2.16b, v5.16b, #8\n" /* ext 2, 3, 4, 5, r2 */
              "fmax v3.4s, v0.4s, v1.4s\n"
              "fmax v4.4s, v2.4s, v6.4s\n"
              "fmax v5.4s, v7.4s, v8.4s\n"
              "fmax v6.4s, v9.4s, v10.4s\n"
              "fmax v0.4s, v11.4s, v3.4s\n"
              "fmax v1.4s, v4.4s, v5.4s\n"
              "fmax v2.4s, v6.4s, v0.4s\n"
              "fmax v3.4s, v2.4s, v1.4s\n"
              "st1  {v3.4s}, [%[dr_out]], #16\n"    /* store 4 out, dr_out */
              "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/
              "bne       1b\n"                      /* bne s3_max_loop_mid */
              "2: \n"                               /* end */
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vmin] "w"(vmin)
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
                "v11");
        }
#else
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              /* preocess left */
              "vld1.32  {d0-d2}, [%[dr0]]!\n"    /* load q0, dr0, 0-5*/
              "vld1.32  {d4-d6}, [%[dr1]]!\n"    /* load q2, dr0, 0-5*/
              "vld1.32  {d8-d10}, [%[dr2]]!\n"   /* load q4, dr0, 0-5*/
              "vext.32  q6, q0, q1, #1\n"        /* ext 1, 2, 3, 4, r0 */
              "vext.32  q7, q2, q3, #1\n"        /* ext 1, 2, 3, 4, r1 */
              "vext.32  q8, q4, q5, #1\n"        /* ext 1, 2, 3, 4, r2 */
              "vext.32  q9, %q[vmin], q0, #3\n"  /* ext -1, 0, 1, 2, r0 */
              "vext.32  q10, %q[vmin], q2, #3\n" /* ext -1, 0, 1, 2, r1 */
              "vext.32  q11, %q[vmin], q4, #3\n" /* ext -1, 0, 1, 2, r2 */
              "vmax.f32 q1, q0, q2\n"
              "vmax.f32 q3, q4, q6\n"
              "vmax.f32 q5, q7, q8\n"
              "vmax.f32 q6, q9, q10\n"
              "vmax.f32 q0, q11, q1\n"
              "vmax.f32 q7, q3, q5\n"
              "vmax.f32 q8, q6, q0\n"
              "vmax.f32 q8, q8, q7\n"
              "subs %[dr0], %[dr0], #12\n"
              "subs %[dr1], %[dr1], #12\n"
              "subs %[dr2], %[dr2], #12\n"
              "vst1.32  {d16-d17}, [%[dr_out]]!\n" /* store 4 out, dr_out */
              "subs %[cnt_num], %[cnt_num], #1\n"  /* subs cnt_num, #1*/
              "ble       2f\n"                     /* jump to end */
              "1: \n"                              /* */
              "vld1.32  {d0-d1}, [%[dr0]]!\n"      /* load q0, dr0, 0-3*/
              "vld1.32  {d2}, [%[dr0]]\n"          /* load q1, dr0, 4-5*/
              "vld1.32  {d4-d5}, [%[dr1]]!\n"      /* load q2, dr1, 0-3*/
              "vld1.32  {d6}, [%[dr1]]\n"          /* load q3, dr1, 4-5*/
              "vld1.32  {d8-d9}, [%[dr2]]!\n"      /* load q4, dr2, 0-3*/
              "vld1.32  {d10}, [%[dr2]]\n"         /* load q5, dr2, 4-5*/
              "vext.32  q6, q0, q1, #1\n"          /* ext 1, 2, 3, 4, r0 */
              "vext.32  q7, q2, q3, #1\n"          /* ext 1, 2, 3, 4, r1 */
              "vext.32  q8, q4, q5, #1\n"          /* ext 1, 2, 3, 4, r2 */
              "vext.32  q9, q0, q1, #2\n"          /* ext 2, 3, 4, 5, r0 */
              "vext.32  q10, q2, q3, #2\n"         /* ext 2, 3, 4, 5, r1 */
              "vext.32  q11, q4, q5, #2\n"         /* ext 2, 3, 4, 5, r2 */
              "vmax.f32 q1, q0, q2\n"
              "vmax.f32 q3, q4, q6\n"
              "vmax.f32 q5, q7, q8\n"
              "vmax.f32 q6, q9, q10\n"
              "vmax.f32 q0, q11, q1\n"
              "vmax.f32 q7, q3, q5\n"
              "vmax.f32 q8, q6, q0\n"
              "vmax.f32 q8, q8, q7\n"
              "vst1.32  {d16-d17}, [%[dr_out]]!\n" /* store 4 out, dr_out */
              "subs %[cnt_num], %[cnt_num], #1\n"  /* subs cnt_num, #1*/
              "bne       1b\n"                     /* bne s3_max_loop_mid */
              "2: \n"                              /* end */
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vmin] "w"(vmin)
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
                "q8",
                "q9",
                "q10",
                "q11");
        }
#endif
        // deal with right pad
        int wstart = w_unroll_size * 4 * S - P;
        for (int j = 0; j < w_unroll_remian; ++j) {
          int wend = std::min(wstart + K, win);
          int st = wstart > 0 ? wstart : 0;
          float tmp = dr0[0];
          for (int i = 0; i < wend - st; i++) {
            tmp = std::max(tmp, dr0[i]);
            tmp = std::max(tmp, dr1[i]);
            tmp = std::max(tmp, dr2[i]);
          }
          *(dr_out++) = tmp;
          dr0 += S - (st - wstart);
          dr1 += S - (st - wstart);
          dr2 += S - (st - wstart);
          wstart += S;
        }
        data_out_channel += wout;
      }
    }
  }
}

void pooling3x3s1p1_avg(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win,
                        bool exclusive) {
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);

  const int K = 3;
  const int P = 1;
  const int S = 1;
  const int WUNROLL = 4;

  int w_unroll_size = wout / WUNROLL;
  int w_unroll_remian = wout - w_unroll_size * WUNROLL;
  if (w_unroll_remian == 0) {
    w_unroll_size -= 1;
    w_unroll_remian = wout - w_unroll_size * WUNROLL;
  }

  auto zero_ptr =
      static_cast<float*>(TargetMalloc(TARGET(kARM), win * sizeof(float)));
  memset(zero_ptr, 0, win * sizeof(float));

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout * size_channel_out;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + win;
      const float* r2 = r1 + win;
      for (int h = 0; h < hout; h++) {
        float coef_h = 1.f / 3;
        float* dr_out = data_out_channel;
        auto dr0 = r0;
        auto dr1 = r1;
        auto dr2 = r2;
        if (h == 0) {
          if (exclusive) {
            coef_h = 0.5f;
          }
          dr0 = zero_ptr;
          dr1 = r0;
          dr2 = r1;
        } else {
          r0 = r1;
          r1 = r2;
          r2 = r1 + win;
        }
        if (h * S + K - P > hin) {
          switch (h * S + K - P - hin) {
            case 2:
              dr1 = zero_ptr;
              dr2 = zero_ptr;
              if (exclusive) {
                coef_h = 1.f;
              } else {
                coef_h = 0.5f;
              }
              break;
            case 1:
              dr2 = zero_ptr;
              if (exclusive) {
                if (fabsf(coef_h - 0.5f) < 1e-6f) {
                  coef_h = 1.f;
                } else {
                  coef_h = 0.5f;
                }
              } else {
                coef_h = 1.f / 3;
              }
            default:
              break;
          }
        }
        float32x4_t vcoef = vdupq_n_f32(coef_h / 3);
        float coef_left_most = exclusive ? coef_h / 2 : coef_h / 3;
        float coef_left[4] = {
            coef_left_most, coef_h / 3, coef_h / 3, coef_h / 3};
        float32x4_t vcoef_left = vld1q_f32(coef_left);
#ifdef __aarch64__
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              /* preocess left */
              "movi v31.4s, #0\n"
              "ldr  q0, [%[dr0]], #16\n"              /* load q0, dr0, 0-3*/
              "ldr  d3, [%[dr0]]\n"                   /* load q3, dr0, 4-5*/
              "ldr  q1, [%[dr1]], #16\n"              /* load q1, dr1, 0-3*/
              "ldr  d4, [%[dr1]]\n"                   /* load q4, dr1, 4-5*/
              "ldr  q2, [%[dr2]], #16\n"              /* load q2, dr2, 0-3*/
              "ldr  d5, [%[dr2]]\n"                   /* load q5, dr2, 4-5*/
              "ext   v6.16b, v0.16b, v3.16b, #4\n"    /* ext 1, 2, 3, 4, r0 */
              "ext   v7.16b, v1.16b, v4.16b, #4\n"    /* ext 1, 2, 3, 4, r1 */
              "ext   v8.16b, v2.16b, v5.16b, #4\n"    /* ext 1, 2, 3, 4, r2 */
              "ext   v9.16b, v31.16b, v0.16b, #12\n"  /* ext -1, 0, 1, 2, r0 */
              "ext   v10.16b, v31.16b, v1.16b, #12\n" /* ext -1, 0, 1, 2, r1 */
              "ext   v11.16b, v31.16b, v2.16b, #12\n" /* ext -1, 0, 1, 2, r2 */
              "fadd v3.4s, v0.4s, v1.4s\n"
              "fadd v4.4s, v2.4s, v6.4s\n"
              "fadd v5.4s, v7.4s, v8.4s\n"
              "fadd v6.4s, v9.4s, v10.4s\n"
              "fadd v0.4s, v11.4s, v3.4s\n"
              "fadd v1.4s, v4.4s, v5.4s\n"
              "fadd v2.4s, v6.4s, v0.4s\n"
              "fadd v3.4s, v2.4s, v1.4s\n"
              "fmul v4.4s, v3.4s, %[vcoef_left].4s\n"
              "subs %[dr0], %[dr0], #4\n"
              "subs %[dr1], %[dr1], #4\n"
              "subs %[dr2], %[dr2], #4\n"
              "st1  {v4.4s}, [%[dr_out]], #16\n"    /* store 4 out, dr_out */
              "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/
              "ble       2f\n"                      /* jump to end */
              "1: \n"                               /* */
              "ldr  q0, [%[dr0]], #16\n"            /* load q0, dr0, 0-3*/
              "ldr  d3, [%[dr0]]\n"                 /* load q3, dr0, 4-5*/
              "ldr  q1, [%[dr1]], #16\n"            /* load q1, dr1, 0-3*/
              "ldr  d4, [%[dr1]]\n"                 /* load q4, dr1, 4-5*/
              "ldr  q2, [%[dr2]], #16\n"            /* load q2, dr2, 0-3*/
              "ldr  d5, [%[dr2]]\n"                 /* load q5, dr2, 4-5*/
              "ext   v6.16b, v0.16b, v3.16b, #4\n"  /* ext 1, 2, 3, 4, r0 */
              "ext   v7.16b, v1.16b, v4.16b, #4\n"  /* ext 1, 2, 3, 4, r1 */
              "ext   v8.16b, v2.16b, v5.16b, #4\n"  /* ext 1, 2, 3, 4, r2 */
              "ext   v9.16b, v0.16b, v3.16b, #8\n"  /* ext 2, 3, 4, 5, r0 */
              "ext   v10.16b, v1.16b, v4.16b, #8\n" /* ext 2, 3, 4, 5, r1 */
              "ext   v11.16b, v2.16b, v5.16b, #8\n" /* ext 2, 3, 4, 5, r2 */
              "fadd v3.4s, v0.4s, v1.4s\n"
              "fadd v4.4s, v2.4s, v6.4s\n"
              "fadd v5.4s, v7.4s, v8.4s\n"
              "fadd v6.4s, v9.4s, v10.4s\n"
              "fadd v0.4s, v11.4s, v3.4s\n"
              "fadd v1.4s, v4.4s, v5.4s\n"
              "fadd v2.4s, v6.4s, v0.4s\n"
              "fadd v3.4s, v2.4s, v1.4s\n"
              "fmul v4.4s, v3.4s, %[vcoef].4s\n"
              "st1  {v4.4s}, [%[dr_out]], #16\n"    /* store 4 out, dr_out */
              "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/
              "bne       1b\n"                      /* bne s3_max_loop_mid */
              "2: \n"                               /* end */
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vcoef] "w"(vcoef), [vcoef_left] "w"(vcoef_left)
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
                "v31");
        }
#else
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              /* preocess left */
              "vmov.i32   q15, #0\n"
              "vld1.32  {d0-d2}, [%[dr0]]!\n"  /* load q0, dr0, 0-5*/
              "vld1.32  {d4-d6}, [%[dr1]]!\n"  /* load q2, dr0, 0-5*/
              "vld1.32  {d8-d10}, [%[dr2]]!\n" /* load q4, dr0, 0-5*/
              "vext.32  q6, q0, q1, #1\n"      /* ext 1, 2, 3, 4, r0 */
              "vext.32  q7, q2, q3, #1\n"      /* ext 1, 2, 3, 4, r1 */
              "vext.32  q8, q4, q5, #1\n"      /* ext 1, 2, 3, 4, r2 */
              "vext.32  q9, q15, q0, #3\n"     /* ext -1, 0, 1, 2, r0 */
              "vext.32  q10, q15, q2, #3\n"    /* ext -1, 0, 1, 2, r1 */
              "vext.32  q11, q15, q4, #3\n"    /* ext -1, 0, 1, 2, r2 */
              "vadd.f32 q1, q0, q2\n"
              "vadd.f32 q3, q4, q6\n"
              "vadd.f32 q5, q7, q8\n"
              "vadd.f32 q6, q9, q10\n"
              "vadd.f32 q0, q11, q1\n"
              "vadd.f32 q7, q3, q5\n"
              "vadd.f32 q8, q6, q0\n"
              "vadd.f32 q8, q8, q7\n"
              "vmul.f32 q9, q8, %q[vcoef_left]\n"
              "subs %[dr0], %[dr0], #12\n"
              "subs %[dr1], %[dr1], #12\n"
              "subs %[dr2], %[dr2], #12\n"
              "vst1.32  {d18-d19}, [%[dr_out]]!\n" /* store 4 out, dr_out */
              "subs %[cnt_num], %[cnt_num], #1\n"  /* subs cnt_num, #1*/
              "ble       2f\n"                     /* jump to end */
              "1: \n"                              /* */
              "vld1.32  {d0-d1}, [%[dr0]]!\n"      /* load q0, dr0, 0-3*/
              "vld1.32  {d2}, [%[dr0]]\n"          /* load q1, dr0, 4-5*/
              "vld1.32  {d4-d5}, [%[dr1]]!\n"      /* load q2, dr1, 0-3*/
              "vld1.32  {d6}, [%[dr1]]\n"          /* load q3, dr1, 4-5*/
              "vld1.32  {d8-d9}, [%[dr2]]!\n"      /* load q4, dr2, 0-3*/
              "vld1.32  {d10}, [%[dr2]]\n"         /* load q5, dr2, 4-5*/
              "vext.32  q6, q0, q1, #1\n"          /* ext 1, 2, 3, 4, r0 */
              "vext.32  q7, q2, q3, #1\n"          /* ext 1, 2, 3, 4, r1 */
              "vext.32  q8, q4, q5, #1\n"          /* ext 1, 2, 3, 4, r2 */
              "vext.32  q9, q0, q1, #2\n"          /* ext 2, 3, 4, 5, r0 */
              "vext.32  q10, q2, q3, #2\n"         /* ext 2, 3, 4, 5, r1 */
              "vext.32  q11, q4, q5, #2\n"         /* ext 2, 3, 4, 5, r2 */
              "vadd.f32 q1, q0, q2\n"
              "vadd.f32 q3, q4, q6\n"
              "vadd.f32 q5, q7, q8\n"
              "vadd.f32 q6, q9, q10\n"
              "vadd.f32 q0, q11, q1\n"
              "vadd.f32 q7, q3, q5\n"
              "vadd.f32 q8, q6, q0\n"
              "vadd.f32 q8, q8, q7\n"
              "vmul.f32 q9, q8, %q[vcoef]\n"
              "vst1.32  {d18-d19}, [%[dr_out]]!\n" /* store 4 out, dr_out */
              "subs %[cnt_num], %[cnt_num], #1\n"  /* subs cnt_num, #1*/
              "bne       1b\n"                     /* bne s3_max_loop_mid */
              "2: \n"                              /* end */
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vcoef] "w"(vcoef), [vcoef_left] "w"(vcoef_left)
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
                "q8",
                "q9",
                "q10",
                "q11",
                "q15");
        }
#endif
        // deal with right pad
        int wstart = w_unroll_size * 4 * S - P;
        for (int j = 0; j < w_unroll_remian; ++j) {
          int wend = wstart + K;  // std::min(wstart + K, win);
          float coef = coef_h / 3.f;
          if (wstart + K > win) {
            wend = win;
            if (!exclusive && wstart + K - win == 2) {
              coef = coef_h / 2;
            }
          }
          int st = wstart > 0 ? wstart : 0;
          if (exclusive) {
            coef = coef_h / (wend - st);
          }
          float tmp = 0.f;
          for (int i = 0; i < wend - st; i++) {
            tmp += dr0[i] + dr1[i] + dr2[i];
          }
          *(dr_out++) = tmp * coef;
          dr0 += S - (st - wstart);
          dr1 += S - (st - wstart);
          dr2 += S - (st - wstart);
          wstart += S;
        }
        data_out_channel += wout;
      }
    }
  }
  TargetFree(TARGET(kARM), zero_ptr);
}

void pooling3x3s2p1_max(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win) {
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);

  const int K = 3;
  const int P = 1;
  const int S = 2;

  int w_unroll_size = wout / 4;
  int w_unroll_remian = wout - w_unroll_size * 4;
  if (w_unroll_remian == 0) {
    w_unroll_size -= 1;
    w_unroll_remian = wout - w_unroll_size * 4;
  }
  float32x4_t vmin = vdupq_n_f32(std::numeric_limits<float>::lowest());

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout * size_channel_out;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + win;
      const float* r2 = r1 + win;
      for (int h = 0; h < hout; h++) {
        float* dr_out = data_out_channel;
        auto dr0 = r0;
        auto dr1 = r1;
        auto dr2 = r2;
        if (h == 0) {
          dr0 = r0;
          dr1 = r0;
          dr2 = r1;
          r0 = r1;
          r1 = r2;
          r2 = r1 + win;
        } else {
          r0 = r2;
          r1 = r0 + win;
          r2 = r1 + win;
        }
        if (h * S + K - P > hin) {
          switch (h * S + K - P - hin) {
            case 2:
              dr1 = dr0;
            case 1:
              dr2 = dr0;
            default:
              break;
          }
        }
#ifdef __aarch64__
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              /* preocess left */
              "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/
              "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/
              "ld2  {v4.4s, v5.4s}, [%[dr2]], #32\n" /* load q4-q5, dr2, 0-7*/
              "fmax v6.4s, v0.4s, v1.4s\n"
              "fmax v7.4s, v2.4s, v3.4s\n"
              "fmax v8.4s, v4.4s, v5.4s\n"
              "ext   v0.16b, %[vmin].16b, v1.16b, #12\n" /* ext 0, 1, 3, 5, r0
                                                            */
              "ext   v2.16b, %[vmin].16b, v3.16b, #12\n" /* ext 0, 1, 3, 5, r1
                                                            */
              "ext   v4.16b, %[vmin].16b, v5.16b, #12\n" /* ext 0, 1, 3, 5, r2
                                                            */
              "fmax v1.4s, v6.4s, v0.4s\n"
              "fmax v3.4s, v7.4s, v2.4s\n"
              "fmax v5.4s, v8.4s, v4.4s\n"
              "fmax v0.4s, v1.4s, v3.4s\n" /* reduce */
              "fmax v0.4s, v0.4s, v5.4s\n" /* reduce */
              "subs %[dr0], %[dr0], #4\n"
              "subs %[dr1], %[dr1], #4\n"
              "subs %[dr2], %[dr2], #4\n"
              "st1  {v0.4s}, [%[dr_out]], #16\n"     /* store 4 out, dr_out */
              "subs %w[cnt_num], %w[cnt_num], #1\n"  /* subs cnt_num, #1*/
              "ble       2f\n"                       /* jump to end */
              "1: \n"                                /* load bias to q2, q3*/
              "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/
              "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/
              "ld2  {v4.4s, v5.4s}, [%[dr2]], #32\n" /* load q4-q5, dr2, 0-7*/
              "ld1  {v6.2s}, [%[dr0]]\n"             /* load d6, dr0, 8,9 */
              "ld1  {v7.2s}, [%[dr1]]\n"             /* load d7, dr1, 8,9 */
              "ld1  {v8.2s}, [%[dr2]]\n"             /* load d8, dr2, 8,9 */
              "fmax   v9.4s, v0.4s, v1.4s\n"  /*  add 0, 2, 4, 6 and 1, 3, 5, 7,
                                                 r0 */
              "fmax   v10.4s, v2.4s, v3.4s\n" /*  add 0, 2, 4, 6 and 1, 3, 5, 7,
                                                 r1 */
              "fmax   v11.4s, v4.4s, v5.4s\n" /*  add 0, 2, 4, 6 and 1, 3, 5, 7,
                                                 r2 */
              "ext    v1.16b, v0.16b, v6.16b, #4\n" /* ext 2, 4, 6, 8, r0 */
              "ext    v3.16b, v2.16b, v7.16b, #4\n" /* ext 2, 4, 6, 8, r1 */
              "ext    v5.16b, v4.16b, v8.16b, #4\n" /* ext 2, 4, 6, 8, r2 */
              "fmax  v0.4s, v9.4s, v1.4s\n"         /* max */
              "fmax  v2.4s, v10.4s, v3.4s\n"        /* max */
              "fmax  v4.4s, v11.4s, v5.4s\n"        /* max */
              "fmax  v1.4s, v0.4s, v2.4s\n"         /* max reduce */
              "fmax  v0.4s, v1.4s, v4.4s\n"         /* max reduce */
              "st1  {v0.4s}, [%[dr_out]], #16\n"    /* store 4 out, dr_out */
              "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/
              "bne       1b\n"                      /* bne s3_max_loop_mid */
              "2: \n"                               /* end */
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vmin] "w"(vmin)
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
                "v11");
        }
#else
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              /* preocess left */
              "vld2.f32  {d0-d3}, [%[dr0]]!\n"  /* load q0-q1, dr0, 0-7*/
              "vld2.f32  {d4-d7}, [%[dr1]]!\n"  /* load q2-q3, dr1, 0-7*/
              "vld2.f32  {d8-d11}, [%[dr2]]!\n" /* load q4-q5, dr2, 0-7*/
              "vmax.f32 q6, q0, q1\n"
              "vmax.f32 q7, q2, q3\n"
              "vmax.f32 q8, q4, q5\n"
              "vext.32   q0, %q[vmin], q1, #3\n" /* ext 0, 1, 3, 5, r0 */
              "vext.32   q2, %q[vmin], q3, #3\n" /* ext 0, 1, 3, 5, r1 */
              "vext.32   q4, %q[vmin], q5, #3\n" /* ext 0, 1, 3, 5, r2 */
              "vmax.f32 q1, q6, q0\n"
              "vmax.f32 q3, q7, q2\n"
              "vmax.f32 q5, q8, q4\n"
              "vmax.f32 q0, q1, q3\n" /* reduce */
              "vmax.f32 q0, q0, q5\n" /* reduce */
              "subs %[dr0], %[dr0], #4\n"
              "subs %[dr1], %[dr1], #4\n"
              "subs %[dr2], %[dr2], #4\n"
              "vst1.32  {d0-d1}, [%[dr_out]]!\n" /* store 4 out, dr_out */
              "subs %[cnt_num], #1\n"            /* subs cnt_num, #1*/
              "ble       2f\n"                   /* jump to end */
              "1: \n"                            /* load bias to q2, q3*/
              "vld2.f32  {d0-d3}, [%[dr0]]!\n"   /* load q0-q1, dr0, 0-7*/
              "vld2.f32  {d4-d7}, [%[dr1]]!\n"   /* load q2-q3, dr1, 0-7*/
              "vld2.f32  {d8-d11}, [%[dr2]]!\n"  /* load q4-q5, dr2, 0-7*/
              "vld1.f32  {d12-d13}, [%[dr0]]\n"  /* load d6, dr0, 8,9 */
              "vld1.f32  {d14-d15}, [%[dr1]]\n"  /* load d7, dr1, 8,9 */
              "vld1.f32  {d16-d17}, [%[dr2]]\n"  /* load d8, dr2, 8,9 */
              "vmax.f32   q9, q0, q1\n" /*  add 0, 2, 4, 6 and 1, 3, 5, 7, r0 */
              "vmax.f32   q10, q2, q3\n" /*  add 0, 2, 4, 6 and 1, 3, 5, 7, r1
                                            */
              "vmax.f32   q11, q4, q5\n" /*  add 0, 2, 4, 6 and 1, 3, 5, 7, r2
                                            */
              "vext.32    q1, q0, q6, #1\n"       /* ext 2, 4, 6, 8, r0 */
              "vext.32    q3, q2, q7, #1\n"       /* ext 2, 4, 6, 8, r1 */
              "vext.32    q5, q4, q8, #1\n"       /* ext 2, 4, 6, 8, r2 */
              "vmax.f32  q0, q9, q1\n"            /* add */
              "vmax.f32  q2, q10, q3\n"           /* add */
              "vmax.f32  q4, q11, q5\n"           /* add */
              "vmax.f32  q1, q0, q2\n"            /* max reduce */
              "vmax.f32  q0, q1, q4\n"            /* max reduce */
              "vst1.32  {d0-d1}, [%[dr_out]]!\n"  /* store 4 out, dr_out */
              "subs %[cnt_num], %[cnt_num], #1\n" /* subs cnt_num, #1*/
              "bne       1b\n"                    /* bne s3_max_loop_mid */
              "2: \n"                             /* end */
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vmin] "w"(vmin)
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
                "q8",
                "q9",
                "q10",
                "q11");
        }
#endif
        // deal with right pad
        int wstart = w_unroll_size * 4 * S - P;
        for (int j = 0; j < w_unroll_remian; ++j) {
          int wend = std::min(wstart + K, win);
          int st = wstart > 0 ? wstart : 0;
          float tmp = dr0[0];
          for (int i = 0; i < wend - st; i++) {
            tmp = std::max(tmp, dr0[i]);
            tmp = std::max(tmp, dr1[i]);
            tmp = std::max(tmp, dr2[i]);
          }
          *(dr_out++) = tmp;
          dr0 += S - (st - wstart);
          dr1 += S - (st - wstart);
          dr2 += S - (st - wstart);
          wstart += S;
        }
        data_out_channel += wout;
      }
    }
  }
}

void pooling3x3s2p1_avg(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win,
                        bool exclusive) {
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);

  const int K = 3;
  const int P = 1;
  const int S = 2;

  int w_unroll_size = wout / 4;
  int w_unroll_remian = wout - w_unroll_size * 4;
  if (w_unroll_remian == 0) {
    w_unroll_size -= 1;
    w_unroll_remian = wout - w_unroll_size * 4;
  }

  auto zero_ptr =
      static_cast<float*>(TargetMalloc(TARGET(kARM), win * sizeof(float)));
  memset(zero_ptr, 0, win * sizeof(float));

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout * size_channel_out;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + win;
      const float* r2 = r1 + win;
      for (int h = 0; h < hout; h++) {
        float coef_h = 1.f / 3;
        float* dr_out = data_out_channel;
        auto dr0 = r0;
        auto dr1 = r1;
        auto dr2 = r2;
        if (h == 0) {
          if (exclusive) {
            coef_h = 0.5f;
          }
          dr0 = zero_ptr;
          dr1 = r0;
          dr2 = r1;
          r0 = r1;
          r1 = r2;
          r2 = r1 + win;
        } else {
          r0 = r2;
          r1 = r0 + win;
          r2 = r1 + win;
        }
        if (h * S + K - P > hin) {
          switch (h * S + K - P - hin) {
            case 2:
              dr1 = zero_ptr;
              dr2 = zero_ptr;
              if (exclusive) {
                coef_h = 1.f;
              } else {
                coef_h = 0.5f;
              }
              break;
            case 1:
              dr2 = zero_ptr;
              if (exclusive) {
                if (fabsf(coef_h - 0.5f) < 1e-6f) {
                  coef_h = 1.f;
                } else {
                  coef_h = 0.5f;
                }
              } else {
                coef_h = 1.f / 3;
              }
            default:
              break;
          }
        }
        float32x4_t vcoef = vdupq_n_f32(coef_h / 3);
        float coef_left_most = exclusive ? coef_h / 2 : coef_h / 3;
        float coef_left[4] = {
            coef_left_most, coef_h / 3, coef_h / 3, coef_h / 3};
        float32x4_t vcoef_left = vld1q_f32(coef_left);
#ifdef __aarch64__
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              "movi v31.4s, #0\n"
              /* preocess left */
              "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/
              "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/
              "ld2  {v4.4s, v5.4s}, [%[dr2]], #32\n" /* load q4-q5, dr2, 0-7*/
              "fadd v6.4s, v0.4s, v1.4s\n"
              "fadd v7.4s, v2.4s, v3.4s\n"
              "fadd v8.4s, v4.4s, v5.4s\n"
              "ext   v0.16b, v31.16b, v1.16b, #12\n" /* ext 0, 1, 3, 5, r0 */
              "ext   v2.16b, v31.16b, v3.16b, #12\n" /* ext 0, 1, 3, 5, r1 */
              "ext   v4.16b, v31.16b, v5.16b, #12\n" /* ext 0, 1, 3, 5, r2 */
              "fadd v1.4s, v6.4s, v0.4s\n"
              "fadd v3.4s, v7.4s, v2.4s\n"
              "fadd v5.4s, v8.4s, v4.4s\n"
              "fadd v0.4s, v1.4s, v3.4s\n" /* reduce */
              "fadd v0.4s, v0.4s, v5.4s\n" /* reduce */
              "subs %[dr0], %[dr0], #4\n"
              "subs %[dr1], %[dr1], #4\n"
              "subs %[dr2], %[dr2], #4\n"
              "fmul v6.4s, v0.4s, %[vcoef_left].4s\n"
              "st1  {v6.4s}, [%[dr_out]], #16\n"     /* store 4 out, dr_out */
              "subs %w[cnt_num], %w[cnt_num], #1\n"  /* subs cnt_num, #1*/
              "ble       2f\n"                       /* jump to end */
              "1: \n"                                /* load bias to q2, q3*/
              "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/
              "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/
              "ld2  {v4.4s, v5.4s}, [%[dr2]], #32\n" /* load q4-q5, dr2, 0-7*/
              "ld1  {v6.2s}, [%[dr0]]\n"             /* load d6, dr0, 8,9 */
              "ld1  {v7.2s}, [%[dr1]]\n"             /* load d7, dr1, 8,9 */
              "ld1  {v8.2s}, [%[dr2]]\n"             /* load d8, dr2, 8,9 */
              "fadd   v9.4s, v0.4s, v1.4s\n"  /*  add 0, 2, 4, 6 and 1, 3, 5, 7,
                                                 r0 */
              "fadd   v10.4s, v2.4s, v3.4s\n" /*  add 0, 2, 4, 6 and 1, 3, 5, 7,
                                                 r1 */
              "fadd   v11.4s, v4.4s, v5.4s\n" /*  add 0, 2, 4, 6 and 1, 3, 5, 7,
                                                 r2 */
              "ext    v1.16b, v0.16b, v6.16b, #4\n" /* ext 2, 4, 6, 8, r0 */
              "ext    v3.16b, v2.16b, v7.16b, #4\n" /* ext 2, 4, 6, 8, r1 */
              "ext    v5.16b, v4.16b, v8.16b, #4\n" /* ext 2, 4, 6, 8, r2 */
              "fadd  v0.4s, v9.4s, v1.4s\n"         /* max */
              "fadd  v2.4s, v10.4s, v3.4s\n"        /* max */
              "fadd  v4.4s, v11.4s, v5.4s\n"        /* max */
              "fadd  v1.4s, v0.4s, v2.4s\n"         /* max reduce */
              "fadd  v0.4s, v1.4s, v4.4s\n"         /* max reduce */
              "fmul v6.4s, v0.4s, %[vcoef].4s\n"
              "st1  {v6.4s}, [%[dr_out]], #16\n"    /* store 4 out, dr_out */
              "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/
              "bne       1b\n"                      /* bne s3_max_loop_mid */
              "2: \n"                               /* end */
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vcoef] "w"(vcoef), [vcoef_left] "w"(vcoef_left)
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
                "v31");
        }
#else
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              "vmov.i32 q15, #0\n"
              /* preocess left */
              "vld2.f32  {d0-d3}, [%[dr0]]!\n"  /* load q0-q1, dr0, 0-7*/
              "vld2.f32  {d4-d7}, [%[dr1]]!\n"  /* load q2-q3, dr1, 0-7*/
              "vld2.f32  {d8-d11}, [%[dr2]]!\n" /* load q4-q5, dr2, 0-7*/
              "vadd.f32 q6, q0, q1\n"
              "vadd.f32 q7, q2, q3\n"
              "vadd.f32 q8, q4, q5\n"
              "vext.32   q0, q15, q1, #3\n" /* ext 0, 1, 3, 5, r0 */
              "vext.32   q2, q15, q3, #3\n" /* ext 0, 1, 3, 5, r1 */
              "vext.32   q4, q15, q5, #3\n" /* ext 0, 1, 3, 5, r2 */
              "vadd.f32 q1, q6, q0\n"
              "vadd.f32 q3, q7, q2\n"
              "vadd.f32 q5, q8, q4\n"
              "vadd.f32 q0, q1, q3\n" /* reduce */
              "vadd.f32 q0, q0, q5\n" /* reduce */
              "subs %[dr0], %[dr0], #4\n"
              "subs %[dr1], %[dr1], #4\n"
              "subs %[dr2], %[dr2], #4\n"
              "vmul.f32 q6, q0, %q[vcoef_left]\n"
              "vst1.32  {d12-d13}, [%[dr_out]]!\n" /* store 4 out, dr_out */
              "subs %[cnt_num], #1\n"              /* subs cnt_num, #1*/
              "ble       2f\n"                     /* jump to end */
              "1: \n"                              /* load bias to q2, q3*/
              "vld2.f32  {d0-d3}, [%[dr0]]!\n"     /* load q0-q1, dr0, 0-7*/
              "vld2.f32  {d4-d7}, [%[dr1]]!\n"     /* load q2-q3, dr1, 0-7*/
              "vld2.f32  {d8-d11}, [%[dr2]]!\n"    /* load q4-q5, dr2, 0-7*/
              "vld1.f32  {d12-d13}, [%[dr0]]\n"    /* load d6, dr0, 8,9 */
              "vld1.f32  {d14-d15}, [%[dr1]]\n"    /* load d7, dr1, 8,9 */
              "vld1.f32  {d16-d17}, [%[dr2]]\n"    /* load d8, dr2, 8,9 */
              "vadd.f32   q9, q0, q1\n" /*  add 0, 2, 4, 6 and 1, 3, 5, 7, r0 */
              "vadd.f32   q10, q2, q3\n" /*  add 0, 2, 4, 6 and 1, 3, 5, 7, r1
                                            */
              "vadd.f32   q11, q4, q5\n" /*  add 0, 2, 4, 6 and 1, 3, 5, 7, r2
                                            */
              "vext.32    q1, q0, q6, #1\n" /* ext 2, 4, 6, 8, r0 */
              "vext.32    q3, q2, q7, #1\n" /* ext 2, 4, 6, 8, r1 */
              "vext.32    q5, q4, q8, #1\n" /* ext 2, 4, 6, 8, r2 */
              "vadd.f32  q0, q9, q1\n"      /* add */
              "vadd.f32  q2, q10, q3\n"     /* add */
              "vadd.f32  q4, q11, q5\n"     /* add */
              "vadd.f32  q1, q0, q2\n"      /* max reduce */
              "vadd.f32  q0, q1, q4\n"      /* max reduce */
              "vmul.f32 q6, q0, %q[vcoef]\n"
              "vst1.32  {d12-d13}, [%[dr_out]]!\n" /* store 4 out, dr_out */
              "subs %[cnt_num], %[cnt_num], #1\n"  /* subs cnt_num, #1*/
              "bne       1b\n"                     /* bne s3_max_loop_mid */
              "2: \n"                              /* end */
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vcoef] "w"(vcoef), [vcoef_left] "w"(vcoef_left)
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
                "q8",
                "q9",
                "q10",
                "q11",
                "q15");
        }
#endif
        // deal with right pad
        int wstart = w_unroll_size * 4 * S - P;
        for (int j = 0; j < w_unroll_remian; ++j) {
          int wend = wstart + K;  // std::min(wstart + K, win);
          float coef = coef_h / 3.f;
          if (wstart + K > win) {
            wend = win;
            if (!exclusive && wstart + K - win == 2) {
              coef = coef_h / 2;
            }
          }
          int st = wstart > 0 ? wstart : 0;
          if (exclusive) {
            coef = coef_h / (wend - st);
          }
          float tmp = 0.f;
          for (int i = 0; i < wend - st; i++) {
            tmp += dr0[i] + dr1[i] + dr2[i];
          }
          *(dr_out++) = tmp * coef;
          dr0 += S - (st - wstart);
          dr1 += S - (st - wstart);
          dr2 += S - (st - wstart);
          wstart += S;
        }
        data_out_channel += wout;
      }
    }
  }
  TargetFree(TARGET(kARM), zero_ptr);
}

void pooling3x3s2p0_max(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win) {
  const int K = 3;
  const int P = 0;
  const int S = 2;

  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);

  int w_unroll_size = wout / 4;
  int w_unroll_remian = wout - w_unroll_size * 4;
  if (w_unroll_remian == 0 && w_unroll_size * 4 * S + K > win) {
    w_unroll_size -= 1;
    w_unroll_remian = wout - w_unroll_size * 4;
  }

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout * size_channel_out;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + win;
      const float* r2 = r1 + win;
      for (int h = 0; h < hout; h++) {
        float* dr_out = data_out_channel;
        auto dr0 = r0;
        auto dr1 = r1;
        auto dr2 = r2;
        if (h * S + K - P > hin) {
          switch (h * S + K - P - hin) {
            case 2:
              dr1 = r0;
            case 1:
              dr2 = r0;
            default:
              break;
          }
        }
#ifdef __aarch64__
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              "1: \n"                                /* load bias to q2, q3*/
              "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/
              "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/
              "ld2  {v4.4s, v5.4s}, [%[dr2]], #32\n" /* load q4-q5, dr2, 0-7*/
              "ld1  {v6.2s}, [%[dr0]]\n"             /* load d6, dr0, 8,9 */
              "ld1  {v7.2s}, [%[dr1]]\n"             /* load d7, dr1, 8,9 */
              "ld1  {v8.2s}, [%[dr2]]\n"             /* load d8, dr2, 8,9 */
              "fmax   v9.4s, v0.4s, v1.4s\n"  /*  max 0, 2, 4, 6 and 1, 3, 5, 7,
                                                 r0 */
              "fmax   v10.4s, v2.4s, v3.4s\n" /*  max 0, 2, 4, 6 and 1, 3, 5, 7,
                                                 r1 */
              "fmax   v11.4s, v4.4s, v5.4s\n" /*  max 0, 2, 4, 6 and 1, 3, 5, 7,
                                                 r2 */
              "ext    v0.16b, v0.16b, v6.16b, #4\n" /* ext 2, 4, 6, 8, r0 */
              "ext    v2.16b, v2.16b, v7.16b, #4\n" /* ext 2, 4, 6, 8, r1 */
              "ext    v4.16b, v4.16b, v8.16b, #4\n" /* ext 2, 4, 6, 8, r2 */
              "fmax  v1.4s, v9.4s, v0.4s\n"         /* max */
              "fmax  v3.4s, v10.4s, v2.4s\n"        /* max */
              "fmax  v5.4s, v11.4s, v4.4s\n"        /* max */
              "fmax  v0.4s, v1.4s, v3.4s\n"         /* max reduce */
              "fmax  v0.4s, v0.4s, v5.4s\n"         /* max reduce */
              "st1  {v0.4s}, [%[dr_out]], #16\n"    /* store 4 out, dr_out */
              "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/
              "bne       1b\n"                      /* bne s3_max_loop_mid */
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
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
                "v11");
        }
#else
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              "1:                                             @ main loop\n"
              "vld2.f32  {d0-d3}, [%[dr0]]!                   @ load d0-d3, "
              "dr0, 0-7\n"
              "vld2.f32  {d4-d7}, [%[dr1]]!                   @ load d4-d7, "
              "dr1, 0-7\n"
              "vld2.f32  {d8-d11}, [%[dr2]]!                  @ load d7-d11, "
              "dr2, 0-7\n"
              "vld1.f32  {d12}, [%[dr0]]                      @ load d12, dr0, "
              "8,9\n"
              "vld1.f32  {d14}, [%[dr1]]                      @ load d14, dr1, "
              "8,9\n"
              "vld1.f32  {d16}, [%[dr2]]                      @ load d16, dr2, "
              "8,9\n"
              "vmax.f32  q9, q0, q1                           @ max 0, 2, 4, 6 "
              "and 1, 3, 5, 7, r0\n"
              "vmax.f32  q10, q2, q3                          @ max 0, 2, 4, 6 "
              "and 1, 3, 5, 7, r1\n"
              "vmax.f32  q11, q4, q5                          @ max 0, 2, 4, 6 "
              "and 1, 3, 5, 7, r2\n"
              "vext.f32  q0, q0, q6, #1                       @ vext 2, 4, 6, "
              "8, r0\n"
              "vext.f32  q2, q2, q7, #1                       @ vext 2, 4, 6, "
              "8, r1\n"
              "vext.f32  q4, q4, q8, #1                       @ vext 2, 4, 6, "
              "8, r2\n"
              "vmax.f32  q1, q9, q0                           @ max\n"
              "vmax.f32  q3, q10, q2                          @ max\n"
              "vmax.f32  q5, q11, q4                          @ max\n"
              "vmax.f32 q0, q1, q3                            @ max reduce\n"
              "vmax.f32 q0, q0, q5                            @ max reduce\n"
              "vst1.f32  {d0-d1}, [%[dr_out]]!                @ store 4 out, "
              "dr_out\n"
              "subs      %[cnt_num], #1                       @ subs cnt_num, "
              "#1\n"
              "bne       1b                                   @ bne "
              "s3_max_loop_mid\n"
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
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
                "q8",
                "q9",
                "q10",
                "q11");
        }
#endif
        // deal with right pad
        int rem = win - (w_unroll_size * 4) * S;
        int wstart = 0;
        for (int j = 0; j < w_unroll_remian; ++j) {
          int wend = std::min(wstart + K, rem);
          float tmp = dr0[wstart];  // std::numeric_limits<float>::min();
          for (int i = wstart; i < wend; i++) {
            tmp = std::max(tmp, dr0[i]);
            tmp = std::max(tmp, dr1[i]);
            tmp = std::max(tmp, dr2[i]);
          }
          *(dr_out++) = tmp;
          wstart += S;
        }

        r0 = r2;
        r1 = r0 + win;
        r2 = r1 + win;
        data_out_channel += wout;
      }
    }
  }
}

void pooling3x3s2p0_avg(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win,
                        bool exclusive) {
  const int K = 3;
  const int P = 0;
  const int S = 2;

  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);

  int w_unroll_size = wout / 4;
  int w_unroll_remian = wout - w_unroll_size * 4;
  if (w_unroll_remian == 0 && w_unroll_size * 4 * S + K > win) {
    w_unroll_size -= 1;
    w_unroll_remian = wout - w_unroll_size * 4;
  }

  auto zero_ptr =
      static_cast<float*>(TargetMalloc(TARGET(kARM), win * sizeof(float)));
  memset(zero_ptr, 0, win * sizeof(float));

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout * size_channel_out;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      const float* r0 = data_in_channel;
      const float* r1 = r0 + win;
      const float* r2 = r1 + win;
      for (int h = 0; h < hout; h++) {
        float coef_h = 1.f / 3;
        float* dr_out = data_out_channel;
        auto dr0 = r0;
        auto dr1 = r1;
        auto dr2 = r2;
        if (h * S + K - P > hin) {
          switch (h * S + K - P - hin) {
            case 2:
              dr1 = zero_ptr;
              dr2 = zero_ptr;
              coef_h = 1.f;
              break;
            case 1:
              dr2 = zero_ptr;
              coef_h = 0.5f;
              break;
            default:
              break;
          }
        }
        float32x4_t vcoef = vdupq_n_f32(coef_h / 3);
#ifdef __aarch64__
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              "1: \n"                                /* load bias to q2, q3*/
              "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/
              "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/
              "ld2  {v4.4s, v5.4s}, [%[dr2]], #32\n" /* load q4-q5, dr2, 0-7*/
              "ld1  {v6.2s}, [%[dr0]]\n"             /* load d6, dr0, 8,9 */
              "ld1  {v7.2s}, [%[dr1]]\n"             /* load d7, dr1, 8,9 */
              "ld1  {v8.2s}, [%[dr2]]\n"             /* load d8, dr2, 8,9 */
              "fadd   v9.4s, v0.4s, v1.4s\n"  /*  max 0, 2, 4, 6 and 1, 3, 5, 7,
                                                 r0 */
              "fadd   v10.4s, v2.4s, v3.4s\n" /*  max 0, 2, 4, 6 and 1, 3, 5, 7,
                                                 r1 */
              "fadd   v11.4s, v4.4s, v5.4s\n" /*  max 0, 2, 4, 6 and 1, 3, 5, 7,
                                                 r2 */
              "ext    v0.16b, v0.16b, v6.16b, #4\n" /* ext 2, 4, 6, 8, r0 */
              "ext    v2.16b, v2.16b, v7.16b, #4\n" /* ext 2, 4, 6, 8, r1 */
              "ext    v4.16b, v4.16b, v8.16b, #4\n" /* ext 2, 4, 6, 8, r2 */
              "fadd  v1.4s, v9.4s, v0.4s\n"         /* max */
              "fadd  v3.4s, v10.4s, v2.4s\n"        /* max */
              "fadd  v5.4s, v11.4s, v4.4s\n"        /* max */
              "fadd  v0.4s, v1.4s, v3.4s\n"         /* max reduce */
              "fadd  v0.4s, v0.4s, v5.4s\n"         /* max reduce */
              "fmul   v6.4s, v0.4s, %[vcoef].4s\n"
              "st1  {v6.4s}, [%[dr_out]], #16\n"    /* store 4 out, dr_out */
              "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/
              "bne       1b\n"                      /* bne s3_max_loop_mid */
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vcoef] "w"(vcoef)
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
                "v11");
        }
#else
        if (w_unroll_size > 0) {
          int cnt_num = w_unroll_size;
          asm volatile(
              "1:                                             @ main loop\n"
              "vld2.f32  {d0-d3}, [%[dr0]]!                   @ load d0-d3, "
              "dr0, 0-7\n"
              "vld2.f32  {d4-d7}, [%[dr1]]!                   @ load d4-d7, "
              "dr1, 0-7\n"
              "vld2.f32  {d8-d11}, [%[dr2]]!                  @ load d7-d11, "
              "dr2, 0-7\n"
              "vld1.f32  {d12}, [%[dr0]]                      @ load d12, dr0, "
              "8,9\n"
              "vld1.f32  {d14}, [%[dr1]]                      @ load d14, dr1, "
              "8,9\n"
              "vld1.f32  {d16}, [%[dr2]]                      @ load d16, dr2, "
              "8,9\n"
              "vadd.f32  q9, q0, q1                           @ max 0, 2, 4, 6 "
              "and 1, 3, 5, 7, r0\n"
              "vadd.f32  q10, q2, q3                          @ max 0, 2, 4, 6 "
              "and 1, 3, 5, 7, r1\n"
              "vadd.f32  q11, q4, q5                          @ max 0, 2, 4, 6 "
              "and 1, 3, 5, 7, r2\n"
              "vext.f32  q0, q0, q6, #1                       @ vext 2, 4, 6, "
              "8, r0\n"
              "vext.f32  q2, q2, q7, #1                       @ vext 2, 4, 6, "
              "8, r1\n"
              "vext.f32  q4, q4, q8, #1                       @ vext 2, 4, 6, "
              "8, r2\n"
              "vadd.f32  q1, q9, q0                           @ max\n"
              "vadd.f32  q3, q10, q2                          @ max\n"
              "vadd.f32  q5, q11, q4                          @ max\n"
              "vadd.f32 q0, q1, q3                            @ max reduce\n"
              "vadd.f32 q0, q0, q5                            @ max reduce\n"
              "vmul.f32   q6, q0, %q[vcoef]\n"
              "vst1.f32  {d12-d13}, [%[dr_out]]!              @ store 4 out, "
              "dr_out\n"
              "subs      %[cnt_num], #1                       @ subs cnt_num, "
              "#1\n"
              "bne       1b                                   @ bne "
              "s3_max_loop_mid\n"
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vcoef] "w"(vcoef)
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
                "q8",
                "q9",
                "q10",
                "q11");
        }
#endif

        // deal with right pad
        int rem = win - (w_unroll_size * 4) * S;
        int wstart = 0;
        for (int j = 0; j < w_unroll_remian; ++j) {
          int wend = std::min(wstart + K, rem);
          float coef = coef_h / (wend - wstart);
          float tmp = 0.f;
          for (int i = wstart; i < wend; i++) {
            tmp += dr0[i];
            tmp += dr1[i];
            tmp += dr2[i];
          }
          tmp *= coef;
          *(dr_out++) = tmp;
          wstart += S;
        }

        r0 = r2;
        r1 = r0 + win;
        r2 = r1 + win;
        data_out_channel += wout;
      }
    }
  }
  TargetFree(TARGET(kARM), zero_ptr);
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
