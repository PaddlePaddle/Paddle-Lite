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

#include "lite/backends/arm/math/pooling.h"
#include <algorithm>
#include <limits>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

int AdaptStartIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      floor(static_cast<double>(ph * input_size) / output_size));
}

int AdaptEndIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      ceil(static_cast<double>((ph + 1) * input_size) / output_size));
}

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
  int pad_w = paddings[2];
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
#pragma omp parallel for
      for (int ind_c = 0; ind_c < chin; ++ind_c) {
        for (int ind_h = 0; ind_h < hout; ++ind_h) {
          int sh, eh;
          if (adaptive) {
            sh = AdaptStartIndex(ind_h, hin, hout);
            eh = AdaptEndIndex(ind_h, hin, hout);
          } else {
            sh = ind_h * stride_h;
            eh = sh + kernel_h;
            sh = (sh - pad_h) < 0 ? 0 : sh - pad_h;
            eh = (eh - pad_h) > hin ? hin : eh - pad_h;
          }
          for (int ind_w = 0; ind_w < wout; ++ind_w) {
            int sw, ew;
            if (adaptive) {
              sw = AdaptStartIndex(ind_w, win, wout);
              ew = AdaptEndIndex(ind_w, win, wout);
            } else {
              sw = ind_w * stride_w;
              ew = sw + kernel_w;
              sw = (sw - pad_w) < 0 ? 0 : sw - pad_w;
              ew = (ew - pad_w) > win ? win : ew - pad_w;
            }
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
                  bw = (sw + kernel_w) >= (win + paddings[3])
                           ? (win + paddings[3])
                           : (sw + kernel_w);
                  bw -= sw;
                  if ((sw - pad_w) < 0 &&
                      (sw + kernel_w) > (win + paddings[3])) {
                    bw += pad_w;
                  }
                }
                if (eh == hin) {
                  bh = (sh + kernel_h) >= (hin + paddings[1])
                           ? (hin + paddings[1])
                           : (sh + kernel_h);
                  bh -= sh;
                  if ((sh - pad_h) < 0 &&
                      (sh + kernel_h) > (hin + paddings[1])) {
                    bh += pad_h;
                  }
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

#ifdef __aarch64__
#define GLOBAL_INIT                                    \
  "ld1 {v0.4s-v1.4s}, [%[data_in_channel]], #32    \n" \
  "ld1 {v2.4s-v3.4s}, [%[data_in_channel]], #32    \n"
#define GLOBAL_MAX                                     \
  "1:                                              \n" \
  "fmax v4.4s, v0.4s, v2.4s \n"                        \
  "fmax v5.4s, v1.4s, v3.4s \n"                        \
  "ld1 {v0.4s-v1.4s}, [%[data_in_channel]], #32    \n" \
  "ld1 {v2.4s-v3.4s}, [%[data_in_channel]], #32    \n" \
  "fmax v6.4s, v4.4s, v5.4s \n"                        \
  "subs %w[cnt], %w[cnt], #1 \n"                       \
  "fmax %[vmax].4s, %[vmax].4s, v6.4s \n"              \
  "bne 1b \n"
#define GLOBAL_AVG                                  \
  "1: \n"                                           \
  "fadd %[vsum].4s, %[vsum].4s, v0.4s \n"           \
  "fadd v4.4s, v1.4s, v2.4s \n"                     \
  "ld1 {v0.4s-v1.4s}, [%[data_in_channel]], #32 \n" \
  "fadd %[vsum].4s, %[vsum].4s, v3.4s \n"           \
  "subs %w[cnt], %w[cnt], #1 \n"                    \
  "fadd %[vsum].4s, %[vsum].4s, v4.4s \n"           \
  "ld1 {v2.4s-v3.4s}, [%[data_in_channel]], #32 \n" \
  "bne 1b \n"

#define P2x2S2_INIT                                                \
  "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/ \
  "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/

#define P2x2S2P1_MAX                                                  \
  "ext v6.16b, %[vzero].16b, v1.16b, #12\n" /* 1357-0135 */           \
  "ext v8.16b, %[vzero].16b, v3.16b, #12\n" /* 1357-0135 */           \
  "sub %[dr0], %[dr0], #4\n"                /* sub */                 \
  "sub %[dr1], %[dr1], #4\n"                /* sub */                 \
  "fmax  v4.4s, v0.4s, v6.4s\n"             /*  max */                \
  "fmax  v5.4s, v2.4s, v8.4s\n"             /*  max */                \
  "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n"    /* load q0-q1, dr0, 0-7*/ \
  "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n"    /* load q2-q3, dr1, 0-7*/ \
  "fmax  v6.4s, v4.4s, v5.4s\n"             /* max reduce */          \
  "subs %w[cnt_num], %w[cnt_num], #1\n"     /* subs cnt_num, #1*/     \
  "st1  {v6.4s}, [%[dr_out]], #16\n"        /* store 4 out, dr_out */ \
  "ble       2f\n"                          /* bne s3_max_loop_mid */

#define P2x2S2P0_MAX                                               \
  "1: \n"                                                          \
  "fmax  v4.4s, v0.4s, v1.4s\n"          /*  max */                \
  "fmax  v5.4s, v2.4s, v3.4s\n"          /*  max */                \
  "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/ \
  "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/ \
  "fmax  v6.4s, v4.4s, v5.4s\n"          /* max reduce */          \
  "subs %w[cnt_num], %w[cnt_num], #1\n"  /* subs cnt_num, #1*/     \
  "st1  {v6.4s}, [%[dr_out]], #16\n"     /* store 4 out, dr_out */ \
  "bne       1b\n"                       /* bne s3_max_loop_mid */

#define P2x2S2P1_AVG                                                          \
  "ext v6.16b, %[vzero].16b, v1.16b, #12\n" /* 1357-0135 */                   \
  "ext v8.16b, %[vzero].16b, v3.16b, #12\n" /* 1357-0135 */                   \
  "sub %[dr0], %[dr0], #4\n"                /* sub */                         \
  "sub %[dr1], %[dr1], #4\n"                /* sub */                         \
  "fadd v4.4s, v0.4s, v6.4s\n"            /* add 0, 2, 4, 6 and 1, 3, 5, 7 */ \
  "fadd v5.4s, v2.4s, v8.4s\n"            /* add 0, 2, 4, 6 and 1, 3, 5, 7 */ \
  "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n"  /* load q0-q1, dr0, 0-7*/           \
  "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n"  /* load q2-q3, dr1, 0-7*/           \
  "fadd v6.4s, v4.4s, v5.4s\n"            /* add reduce */                    \
  "subs %w[cnt_num], %w[cnt_num], #1\n"   /* subs cnt_num, #1*/               \
  "fmul v4.4s, v6.4s, %[vcoef_left].4s\n" /* mul coef */                      \
  "st1  {v4.4s}, [%[dr_out]], #16\n"      /* store 4 out, dr_out */           \
  "ble       2f\n"                        /* bne s3_max_loop_mid */

#define P2x2S2P0_AVG                                                         \
  "1: \n"                                /* load bias to q2, q3*/            \
  "fadd v4.4s, v0.4s, v1.4s\n"           /* add 0, 2, 4, 6 and 1, 3, 5, 7 */ \
  "fadd v5.4s, v2.4s, v3.4s\n"           /* add 0, 2, 4, 6 and 1, 3, 5, 7 */ \
  "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/           \
  "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/           \
  "fadd v6.4s, v4.4s, v5.4s\n"           /* add reduce */                    \
  "subs %w[cnt_num], %w[cnt_num], #1\n"  /* subs cnt_num, #1*/               \
  "fmul v4.4s, v6.4s, %[vcoef].4s\n"     /* mul coef */                      \
  "st1  {v4.4s}, [%[dr_out]], #16\n"     /* store 4 out, dr_out */           \
  "bne       1b\n"                       /* bne s3_max_loop_mid */

#define P3x3S1_INIT                                 \
  "ldr  q0, [%[dr0]], #16\n" /* load q0, dr0, 0-3*/ \
  "ldr  q1, [%[dr1]], #16\n" /* load q1, dr1, 0-3*/ \
  "ldr  q2, [%[dr2]], #16\n" /* load q2, dr2, 0-3*/ \
  "ldr  d3, [%[dr0]]\n"      /* load q3, dr0, 4-5*/ \
  "ldr  d4, [%[dr1]]\n"      /* load q4, dr1, 4-5*/ \
  "ldr  d5, [%[dr2]]\n"      /* load q5, dr2, 4-5*/

#define P3x3S1P1_MAX                                                   \
  "ext   v6.16b, v0.16b, v3.16b, #4\n"        /* ext 1, 2, 3, 4, r0 */ \
  "ext   v7.16b, v1.16b, v4.16b, #4\n"        /* ext 1, 2, 3, 4, r1 */ \
  "ext   v8.16b, v2.16b, v5.16b, #4\n"        /* ext 1, 2, 3, 4, r2 */ \
  "ext   v9.16b, %[vmin].16b, v0.16b, #12\n"  /* ext -1, 0, 1, 2 */    \
  "ext   v10.16b, %[vmin].16b, v1.16b, #12\n" /* ext -1, 0, 1, 2 */    \
  "ext   v11.16b, %[vmin].16b, v2.16b, #12\n" /* ext -1, 0, 1, 2 */    \
  "fmax v3.4s, v0.4s, v1.4s\n"                                         \
  "fmax v4.4s, v2.4s, v6.4s\n"                                         \
  "fmax v5.4s, v7.4s, v8.4s\n"                                         \
                                                                       \
  "fmax v6.4s, v9.4s, v10.4s\n"                                        \
  "fmax v7.4s, v11.4s, v3.4s\n"                                        \
  "fmax v8.4s, v4.4s, v5.4s\n"                                         \
  "subs %[dr0], %[dr0], #4\n"                                          \
  "subs %[dr1], %[dr1], #4\n"                                          \
  "subs %[dr2], %[dr2], #4\n"                                          \
                                                                       \
  "fmax v9.4s, v6.4s, v7.4s\n"                                         \
  "ldr  q0, [%[dr0]], #16\n" /* load q0, dr0, 0-3*/                    \
  "ldr  q1, [%[dr1]], #16\n" /* load q0, dr0, 0-3*/                    \
  "ldr  q2, [%[dr2]], #16\n" /* load q0, dr0, 0-3*/                    \
  "fmax v7.4s, v8.4s, v9.4s\n"                                         \
  "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/          \
  "ldr  d3, [%[dr0]] \n"                /* load q0, dr0, 0-3*/         \
  "ldr  d4, [%[dr1]]\n"                 /* load q4, dr1, 4-5*/         \
  "ldr  d5, [%[dr2]]\n"                 /* load q4, dr1, 4-5*/         \
  "st1  {v7.4s}, [%[dr_out]], #16\n"    /* store 4 out, dr_out */      \
  "ble       2f\n"                      /* jump to end */

#define P3x3S1P0_MAX                                              \
  "1: \n"                               /* */                     \
  "ext   v6.16b, v0.16b, v3.16b, #4\n"  /* ext 1, 2, 3, 4, r0 */  \
  "ext   v7.16b, v1.16b, v4.16b, #4\n"  /* ext 1, 2, 3, 4, r1 */  \
  "ext   v8.16b, v2.16b, v5.16b, #4\n"  /* ext 1, 2, 3, 4, r2 */  \
  "ext   v9.16b, v0.16b, v3.16b, #8\n"  /* ext 2, 3, 4, 5, r0 */  \
  "ext   v10.16b, v1.16b, v4.16b, #8\n" /* ext 2, 3, 4, 5, r1 */  \
  "ext   v11.16b, v2.16b, v5.16b, #8\n" /* ext 2, 3, 4, 5, r2 */  \
  "fmax v3.4s, v0.4s, v1.4s\n"                                    \
  "fmax v4.4s, v2.4s, v6.4s\n"                                    \
  "fmax v5.4s, v7.4s, v8.4s\n"                                    \
  "fmax v6.4s, v9.4s, v10.4s\n"                                   \
                                                                  \
  "fmax v7.4s, v11.4s, v3.4s\n"                                   \
  "fmax v8.4s, v4.4s, v5.4s\n"                                    \
  "fmax v9.4s, v6.4s, v7.4s\n"                                    \
  "ldr  q0, [%[dr0]], #16\n" /* load q0, dr0, 0-3*/               \
  "ldr  q1, [%[dr1]], #16\n" /* load q0, dr0, 0-3*/               \
  "ldr  q2, [%[dr2]], #16\n" /* load q0, dr0, 0-3*/               \
                                                                  \
  "fmax v7.4s, v8.4s, v9.4s\n"                                    \
  "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/     \
  "ldr  d3, [%[dr0]] \n"                /* load q0, dr0, 0-3*/    \
  "ldr  d4, [%[dr1]]\n"                 /* load q4, dr1, 4-5*/    \
  "ldr  d5, [%[dr2]]\n"                 /* load q4, dr1, 4-5*/    \
  "st1  {v7.4s}, [%[dr_out]], #16\n"    /* store 4 out, dr_out */ \
  "bne       1b\n"                      /* bne s3_max_loop_mid */

#define P3x3S1P1_AVG                                                \
  "ext   v6.16b, v0.16b, v3.16b, #4\n"    /* ext 1, 2, 3, 4, r0 */  \
  "ext   v7.16b, v1.16b, v4.16b, #4\n"    /* ext 1, 2, 3, 4, r1 */  \
  "ext   v8.16b, v2.16b, v5.16b, #4\n"    /* ext 1, 2, 3, 4, r2 */  \
  "ext   v9.16b, v31.16b, v0.16b, #12\n"  /* ext -1, 0, 1, 2, r0 */ \
  "ext   v10.16b, v31.16b, v1.16b, #12\n" /* ext -1, 0, 1, 2, r1 */ \
  "ext   v11.16b, v31.16b, v2.16b, #12\n" /* ext -1, 0, 1, 2, r2 */ \
                                                                    \
  "fadd v3.4s, v0.4s, v1.4s\n"                                      \
  "fadd v4.4s, v2.4s, v6.4s\n"                                      \
  "fadd v5.4s, v7.4s, v8.4s\n"                                      \
  "fadd v6.4s, v9.4s, v10.4s\n"                                     \
  "fadd v7.4s, v11.4s, v3.4s\n"                                     \
                                                                    \
  "subs %[dr0], %[dr0], #4\n"                                       \
  "subs %[dr1], %[dr1], #4\n"                                       \
  "subs %[dr2], %[dr2], #4\n"                                       \
                                                                    \
  "fadd v8.4s, v4.4s, v5.4s\n"                                      \
  "fadd v9.4s, v6.4s, v7.4s\n"                                      \
  "ldr  q0, [%[dr0]], #16\n" /* load q0, dr0, 0-3*/                 \
  "ldr  q1, [%[dr1]], #16\n" /* load q1, dr1, 0-3*/                 \
  "ldr  q2, [%[dr2]], #16\n" /* load q2, dr2, 0-3*/                 \
                                                                    \
  "fadd v10.4s, v8.4s, v9.4s\n"                                     \
  "ldr  d3, [%[dr0]]\n" /* load q3, dr0, 4-5*/                      \
  "ldr  d4, [%[dr1]]\n" /* load q4, dr1, 4-5*/                      \
                                                                    \
  "fmul v11.4s, v10.4s, %[vcoef_left].4s\n"                         \
  "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/       \
  "ldr  d5, [%[dr2]]\n"                 /* load q5, dr2, 4-5*/      \
                                                                    \
  "st1  {v11.4s}, [%[dr_out]], #16\n" /* store 4 out, dr_out */     \
  "ble       2f\n"                    /* jump to end */
#define P3x3S1P0_AVG                                              \
  "1: \n"                               /* */                     \
  "ext   v6.16b, v0.16b, v3.16b, #4\n"  /* ext 1, 2, 3, 4, r0 */  \
  "ext   v7.16b, v1.16b, v4.16b, #4\n"  /* ext 1, 2, 3, 4, r1 */  \
  "ext   v8.16b, v2.16b, v5.16b, #4\n"  /* ext 1, 2, 3, 4, r2 */  \
  "ext   v9.16b, v0.16b, v3.16b, #8\n"  /* ext 2, 3, 4, 5, r0 */  \
  "ext   v10.16b, v1.16b, v4.16b, #8\n" /* ext 2, 3, 4, 5, r1 */  \
  "ext   v11.16b, v2.16b, v5.16b, #8\n" /* ext 2, 3, 4, 5, r2 */  \
                                                                  \
  "fadd v3.4s, v0.4s, v1.4s\n"                                    \
  "fadd v4.4s, v2.4s, v6.4s\n"                                    \
  "fadd v5.4s, v7.4s, v8.4s\n"                                    \
  "fadd v6.4s, v9.4s, v10.4s\n"                                   \
  "fadd v7.4s, v11.4s, v3.4s\n"                                   \
                                                                  \
  "fadd v8.4s, v4.4s, v5.4s\n"                                    \
  "fadd v9.4s, v6.4s, v7.4s\n"                                    \
                                                                  \
  "ldr  q0, [%[dr0]], #16\n" /* load q0, dr0, 0-3*/               \
  "ldr  q1, [%[dr1]], #16\n" /* load q1, dr1, 0-3*/               \
  "ldr  q2, [%[dr2]], #16\n" /* load q2, dr2, 0-3*/               \
  "fadd v10.4s, v8.4s, v9.4s\n"                                   \
                                                                  \
  "ldr  d3, [%[dr0]]\n" /* load q3, dr0, 4-5*/                    \
  "ldr  d4, [%[dr1]]\n" /* load q4, dr1, 4-5*/                    \
  "fmul v11.4s, v10.4s, %[vcoef].4s\n"                            \
                                                                  \
  "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/     \
  "ldr  d5, [%[dr2]]\n"                 /* load q3, dr0, 4-5*/    \
  "st1  {v11.4s}, [%[dr_out]], #16\n"   /* store 4 out, dr_out */ \
  "bne       1b\n"                      /* bne s3_max_loop_mid */

#define P3x3S2_INIT                                                \
  "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/ \
  "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/ \
  "ld2  {v4.4s, v5.4s}, [%[dr2]], #32\n" /* load q4-q5, dr2, 0-7*/

#define P3x3S2P0_INIT                                              \
  "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/ \
  "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/ \
  "ld2  {v4.4s, v5.4s}, [%[dr2]], #32\n" /* load q4-q5, dr2, 0-7*/ \
  "ld1  {v6.2s}, [%[dr0]]\n"             /* load d6, dr0, 8,9 */   \
  "ld1  {v7.2s}, [%[dr1]]\n"             /* load d7, dr1, 8,9 */   \
  "ld1  {v8.2s}, [%[dr2]]\n"             /* load d8, dr2, 8,9 */

#define P3x3S2P1_MAX                                               \
  "fmax v6.4s, v0.4s, v1.4s\n"                                     \
  "fmax v7.4s, v2.4s, v3.4s\n"                                     \
  "fmax v8.4s, v4.4s, v5.4s\n"                                     \
  "ext   v0.16b, %[vmin].16b, v1.16b, #12\n" /* ext 0, 1, 3, 5 */  \
  "ext   v2.16b, %[vmin].16b, v3.16b, #12\n" /* ext 0, 1, 3, 5 */  \
  "ext   v4.16b, %[vmin].16b, v5.16b, #12\n" /* ext 0, 1, 3, 5 */  \
  "fmax v1.4s, v6.4s, v0.4s\n"                                     \
  "fmax v3.4s, v7.4s, v2.4s\n"                                     \
  "fmax v11.4s, v8.4s, v4.4s\n"                                    \
                                                                   \
  "subs %[dr0], %[dr0], #4\n"                                      \
  "subs %[dr1], %[dr1], #4\n"                                      \
  "subs %[dr2], %[dr2], #4\n"                                      \
                                                                   \
  "fmax v9.4s, v1.4s, v3.4s\n"           /* reduce */              \
  "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/ \
  "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/ \
  "ld2  {v4.4s, v5.4s}, [%[dr2]], #32\n" /* load q4-q5, dr2, 0-7*/ \
                                                                   \
  "fmax v10.4s, v9.4s, v11.4s\n"        /* reduce */               \
  "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/      \
  "ld1  {v6.2s}, [%[dr0]]\n"            /* load d6, dr0, 8,9 */    \
  "ld1  {v7.2s}, [%[dr1]]\n"            /* load d7, dr1, 8,9 */    \
  "ld1  {v8.2s}, [%[dr2]]\n"            /* load d8, dr2, 8,9 */    \
                                                                   \
  "st1  {v10.4s}, [%[dr_out]], #16\n" /* store 4 out, dr_out */    \
  "ble       2f\n"                    /* jump to end */

#define P3x3S2P0_MAX                                                         \
  "1: \n"                               /* load bias to q2, q3*/             \
  "fmax   v9.4s, v0.4s, v1.4s\n"        /*  add 0, 2, 4, 6 and 1, 3, 5, 7 */ \
  "fmax   v10.4s, v2.4s, v3.4s\n"       /*  add 0, 2, 4, 6 and 1, 3, 5, 7 */ \
  "fmax   v11.4s, v4.4s, v5.4s\n"       /*  add 0, 2, 4, 6 and 1, 3, 5, 7 */ \
  "ext    v1.16b, v0.16b, v6.16b, #4\n" /* ext 2, 4, 6, 8, r0 */             \
  "ext    v3.16b, v2.16b, v7.16b, #4\n" /* ext 2, 4, 6, 8, r1 */             \
  "ext    v5.16b, v4.16b, v8.16b, #4\n" /* ext 2, 4, 6, 8, r2 */             \
                                                                             \
  "fmax  v6.4s, v9.4s, v1.4s\n"  /* max */                                   \
  "fmax  v7.4s, v10.4s, v3.4s\n" /* max */                                   \
  "fmax  v8.4s, v11.4s, v5.4s\n" /* max */                                   \
                                                                             \
  "fmax  v9.4s, v6.4s, v7.4s\n"          /* max reduce */                    \
  "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/           \
  "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/           \
  "ld2  {v4.4s, v5.4s}, [%[dr2]], #32\n" /* load q4-q5, dr2, 0-7*/           \
                                                                             \
  "fmax  v10.4s, v8.4s, v9.4s\n"        /* max reduce */                     \
  "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/                \
  "ld1  {v6.2s}, [%[dr0]]\n"            /* load d6, dr0, 8,9 */              \
  "ld1  {v7.2s}, [%[dr1]]\n"            /* load d7, dr1, 8,9 */              \
  "ld1  {v8.2s}, [%[dr2]]\n"            /* load d8, dr2, 8,9 */              \
                                                                             \
  "st1  {v10.4s}, [%[dr_out]], #16\n" /* store 4 out, dr_out */              \
  "bne       1b\n"                    /* bne s3_max_loop_mid */

#define P3x3S2P1_AVG                                               \
  "fadd v6.4s, v0.4s, v1.4s\n"                                     \
  "fadd v7.4s, v2.4s, v3.4s\n"                                     \
  "fadd v8.4s, v4.4s, v5.4s\n"                                     \
  "ext   v0.16b, v31.16b, v1.16b, #12\n" /* ext 0, 1, 3, 5, r0 */  \
  "ext   v2.16b, v31.16b, v3.16b, #12\n" /* ext 0, 1, 3, 5, r1 */  \
  "ext   v4.16b, v31.16b, v5.16b, #12\n" /* ext 0, 1, 3, 5, r2 */  \
                                                                   \
  "fadd v1.4s, v6.4s, v0.4s\n"                                     \
  "fadd v3.4s, v7.4s, v2.4s\n"                                     \
  "fadd v5.4s, v8.4s, v4.4s\n"                                     \
                                                                   \
  "fadd v9.4s, v1.4s, v3.4s\n" /* reduce */                        \
  "subs %[dr0], %[dr0], #4\n"                                      \
  "subs %[dr1], %[dr1], #4\n"                                      \
  "subs %[dr2], %[dr2], #4\n"                                      \
                                                                   \
  "fadd v10.4s, v9.4s, v5.4s\n"          /* reduce */              \
  "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/ \
  "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/ \
  "ld2  {v4.4s, v5.4s}, [%[dr2]], #32\n" /* load q4-q5, dr2, 0-7*/ \
                                                                   \
  "fmul v11.4s, v10.4s, %[vcoef_left].4s\n"                        \
  "subs %w[cnt_num], %w[cnt_num], #1\n" /* subs cnt_num, #1*/      \
  "ld1  {v6.2s}, [%[dr0]]\n"            /* load d6, dr0, 8,9 */    \
  "ld1  {v7.2s}, [%[dr1]]\n"            /* load d7, dr1, 8,9 */    \
  "ld1  {v8.2s}, [%[dr2]]\n"            /* load d8, dr2, 8,9 */    \
                                                                   \
  "st1  {v11.4s}, [%[dr_out]], #16\n" /* store 4 out, dr_out */    \
  "ble       2f\n"                    /* jump to end */

#define P3x3S2P0_AVG                                                         \
  "1: \n"                               /* load bias to q2, q3*/             \
  "fadd   v9.4s, v0.4s, v1.4s\n"        /*  add 0, 2, 4, 6 and 1, 3, 5, 7 */ \
  "fadd   v10.4s, v2.4s, v3.4s\n"       /*  add 0, 2, 4, 6 and 1, 3, 5, 7 */ \
  "fadd   v11.4s, v4.4s, v5.4s\n"       /*  add 0, 2, 4, 6 and 1, 3, 5, 7 */ \
  "ext    v1.16b, v0.16b, v6.16b, #4\n" /* ext 2, 4, 6, 8, r0 */             \
  "ext    v3.16b, v2.16b, v7.16b, #4\n" /* ext 2, 4, 6, 8, r1 */             \
  "ext    v5.16b, v4.16b, v8.16b, #4\n" /* ext 2, 4, 6, 8, r2 */             \
                                                                             \
  "fadd  v9.4s, v9.4s, v1.4s\n"   /* max */                                  \
  "fadd  v10.4s, v10.4s, v3.4s\n" /* max */                                  \
  "fadd  v11.4s, v11.4s, v5.4s\n" /* max */                                  \
                                                                             \
  "fadd  v9.4s, v9.4s, v10.4s\n"         /* max reduce */                    \
  "ld2  {v0.4s, v1.4s}, [%[dr0]], #32\n" /* load q0-q1, dr0, 0-7*/           \
  "ld2  {v2.4s, v3.4s}, [%[dr1]], #32\n" /* load q2-q3, dr1, 0-7*/           \
                                                                             \
  "fadd  v9.4s, v9.4s, v11.4s\n"         /* max reduce */                    \
  "ld2  {v4.4s, v5.4s}, [%[dr2]], #32\n" /* load q4-q5, dr2, 0-7*/           \
  "subs %w[cnt_num], %w[cnt_num], #1\n"  /* subs cnt_num, #1*/               \
                                                                             \
  "fmul v10.4s, v9.4s, %[vcoef].4s\n"                                        \
  "ld1  {v6.2s}, [%[dr0]]\n" /* load d6, dr0, 8,9 */                         \
  "ld1  {v7.2s}, [%[dr1]]\n" /* load d7, dr1, 8,9 */                         \
  "ld1  {v8.2s}, [%[dr2]]\n" /* load d8, dr2, 8,9 */                         \
                                                                             \
  "st1  {v10.4s}, [%[dr_out]], #16\n" /* store 4 out, dr_out */              \
  "bne       1b\n"                    /* bne s3_max_loop_mid */

#else
#define GLOBAL_INIT                                                 \
  "vld1.f32   {d0-d3}, [%[data_in_channel]]!        @ load data \n" \
  "vld1.f32   {d4-d7}, [%[data_in_channel]]!        @ load data \n"
#define GLOBAL_MAX                                                   \
  "1:                                               @ main loop\n"   \
  "vmax.f32   q4, q0, q1                            @ max \n"        \
  "vmax.f32   q5, q2, q3                            @ max vmax \n"   \
  "vld1.f32   {d0-d3}, [%[data_in_channel]]!        @ load data \n"  \
  "vld1.f32   {d4-d7}, [%[data_in_channel]]!        @ load data \n"  \
  "vmax.f32   q6, q4, q5                            @ max vmax \n"   \
  "subs       %[cnt], #1                            @ subs num, 1\n" \
  "vmax.f32   %q[vmax], %q[vmax], q6                @ max vmax \n"   \
  "bne        1b                                    @ bne num\n"
#define GLOBAL_AVG                                                  \
  "1:                                        @main loop\n"          \
  "vadd.f32   %q[vsum], %q[vsum], q0                @add vmax \n"   \
  "vadd.f32   q4, q2, q1                @add vmax \n"               \
  "vld1.f32   {d0-d3}, [%[data_in_channel]]!        @load q1 \n"    \
  "vadd.f32   %q[vsum], %q[vsum], q3                @add vmax \n"   \
  "subs        %[cnt], #1                           @subs num, 1\n" \
  "vadd.f32   %q[vsum], %q[vsum], q4                @add vmax \n"   \
  "vld1.f32   {d4-d7}, [%[data_in_channel]]!        @load q1 \n"    \
  "bne        1b                              @bne num\n"

#define P2x2S2_INIT                                          \
  "vld2.f32  {d0-d3}, [%[dr0]]!                   @ load \n" \
  "vld2.f32  {d4-d7}, [%[dr1]]!                   @ load \n"

#define P2x2S2P1_MAX                                                 \
  "vext.32 q4, %q[vzero], q1, #3                  @ 1357-0135\n"     \
  "vext.32 q5, %q[vzero], q3, #3                  @ 1357-0135\n"     \
  "sub %[dr0], #4                                 @sub \n"           \
  "sub %[dr1], #4                                 @sub \n"           \
  "vmax.f32  q8, q0, q4                           @ max \n"          \
  "vmax.f32  q9, q2, q5                           @ max \n"          \
  "vld2.f32  {d0-d3}, [%[dr0]]!                   @ load \n"         \
  "vld2.f32  {d4-d7}, [%[dr1]]!                   @ load \n"         \
  "vmax.f32  q5, q9, q8                           @ max reduce\n"    \
  "subs   %[cnt_num], #1                          @ subs cnt_num \n" \
  "vst1.f32  {d10-d11}, [%[dr_out]]!              @ store 4 out \n"  \
  "ble       2f                                   @ bne \n"

#define P2x2S2P0_MAX                                                  \
  "1:                                             @ main loop\n"      \
  "vmax.f32  q4, q0, q1                           @ max \n"           \
  "vmax.f32  q5, q2, q3                           @ max \n"           \
  "vld2.f32  {d0-d3}, [%[dr0]]!                   @ load \n"          \
  "vld2.f32  {d4-d7}, [%[dr1]]!                   @ load \n"          \
  "vmax.f32  q8, q4, q5                           @ max reduce\n"     \
  "subs      %[cnt_num], #1                       @ subs cnt_num \n"  \
  "vst1.f32  {d16-d17}, [%[dr_out]]!                @ store 4 out \n" \
  "bne       1b                                   @ bne \n"

#define P2x2S2P1_AVG                                                 \
  "vext.32 q4, %q[vzero], q1, #3                  @ 1357-0135\n"     \
  "vext.32 q5, %q[vzero], q3, #3                  @ 1357-0135\n"     \
  "sub %[dr0], #4                                 @sub \n"           \
  "sub %[dr1], #4                                 @sub \n"           \
  "vadd.f32  q9, q0, q4                           @ max \n"          \
  "vadd.f32  q8, q2, q5                           @ max \n"          \
  "vld2.f32  {d0-d3}, [%[dr0]]!                   @ load \n"         \
  "vld2.f32  {d4-d7}, [%[dr1]]!                   @ load \n"         \
  "vadd.f32  q5, q9, q8                           @ max reduce\n"    \
  "subs      %[cnt_num], #1                       @ subs cnt_num \n" \
  "vmul.f32  q4, q5, %q[vcoef_left]               @ mul coef \n"     \
  "vst1.f32  {d8-d9}, [%[dr_out]]!                @ store 4 out \n"  \
  "ble       2f                                   @ bne\n"

#define P2x2S2P0_AVG                                                   \
  "1:                                             @ main loop\n"       \
  "vadd.f32  q4, q0, q1                           @ add 0, 2, 4, 6 \n" \
  "vadd.f32  q5, q2, q3                           @ add 0, 2, 4, 6 \n" \
  "vld2.f32  {d0-d3}, [%[dr0]]!                   @ load d0-d3 \n"     \
  "vld2.f32  {d4-d7}, [%[dr1]]!                  @ load d4-d7 \n"      \
  "vadd.f32  q8, q4, q5                           @ add reduce \n"     \
  "subs      %[cnt_num], #1                       @ subs \n"           \
  "vmul.f32  q4, q8, %q[vcoef]                    @ mul coef \n"       \
  "vst1.f32  {d8-d9}, [%[dr_out]]!                @ store 4 out \n"    \
  "bne       1b                                   @ bne \n"

#define P3x3S1_INIT                                       \
  "vld1.32  {d0-d2}, [%[dr0]]!\n"  /* load q0, dr0, 0-5*/ \
  "vld1.32  {d4-d6}, [%[dr1]]!\n"  /* load q2, dr0, 0-5*/ \
  "vld1.32  {d8-d10}, [%[dr2]]!\n" /* load q4, dr0, 0-5*/
#define P3x3S1P0_INIT                                    \
  "vld1.32  {d0-d1}, [%[dr0]]!\n" /* load q0, dr0, 0-5*/ \
  "vld1.32  {d4-d5}, [%[dr1]]!\n" /* load q2, dr0, 0-5*/ \
  "vld1.32  {d8-d9}, [%[dr2]]!\n" /* load q4, dr0, 0-5*/ \
  "vld1.32  {d2}, [%[dr0]]\n"     /* load q1, dr0, 4-5*/ \
  "vld1.32  {d6}, [%[dr1]]\n"     /* load q1, dr0, 4-5*/ \
  "vld1.32  {d10}, [%[dr2]]\n"    /* load q1, dr0, 4-5*/
#define P3x3S1P1_MAX                                             \
  "vext.32  q6, q0, q1, #1\n"        /* ext 1, 2, 3, 4, r0 */    \
  "vext.32  q7, q2, q3, #1\n"        /* ext 1, 2, 3, 4, r1 */    \
  "vext.32  q8, q4, q5, #1\n"        /* ext 1, 2, 3, 4, r2 */    \
  "vext.32  q9, %q[vmin], q0, #3\n"  /* ext -1, 0, 1, 2, r0 */   \
  "vext.32  q10, %q[vmin], q2, #3\n" /* ext -1, 0, 1, 2, r1 */   \
  "vext.32  q11, %q[vmin], q4, #3\n" /* ext -1, 0, 1, 2, r2 */   \
                                                                 \
  "vmax.f32 q1, q0, q2\n"                                        \
  "vmax.f32 q3, q4, q6\n"                                        \
  "vmax.f32 q5, q7, q8\n"                                        \
  "vmax.f32 q6, q9, q10\n"                                       \
  "vmax.f32 q7, q11, q1\n"                                       \
                                                                 \
  "subs %[dr0], %[dr0], #12\n"                                   \
  "subs %[dr1], %[dr1], #12\n"                                   \
  "subs %[dr2], %[dr2], #12\n"                                   \
                                                                 \
  "vmax.f32 q8, q3, q5\n"                                        \
  "vmax.f32 q9, q6, q7\n"                                        \
  "vld1.32  {d0-d1}, [%[dr0]]!\n" /* load q0, dr0, 0-3*/         \
  "vld1.32  {d4-d5}, [%[dr1]]!\n" /* load q0, dr0, 0-3*/         \
  "vld1.32  {d8-d9}, [%[dr2]]!\n" /* load q0, dr0, 0-3*/         \
  "vmax.f32 q6, q8, q9\n"                                        \
                                                                 \
  "subs %[cnt_num], %[cnt_num], #1\n" /* subs cnt_num, #1*/      \
  "vld1.32  {d2}, [%[dr0]]\n"         /* load q1, dr0, 4-5*/     \
  "vld1.32  {d6}, [%[dr1]]\n"         /* load q1, dr0, 4-5*/     \
  "vld1.32  {d10}, [%[dr2]]\n"        /* load q1, dr0, 4-5*/     \
                                                                 \
  "vst1.32  {d12-d13}, [%[dr_out]]!\n" /* store 4 out, dr_out */ \
  "ble       2f\n"                     /* jump to end */

#define P3x3S1P0_MAX                                             \
  "1: \n"                      /* */                             \
  "vext.32  q6, q0, q1, #1\n"  /* ext 1, 2, 3, 4, r0 */          \
  "vext.32  q7, q2, q3, #1\n"  /* ext 1, 2, 3, 4, r1 */          \
  "vext.32  q8, q4, q5, #1\n"  /* ext 1, 2, 3, 4, r2 */          \
  "vext.32  q9, q0, q1, #2\n"  /* ext 2, 3, 4, 5, r0 */          \
  "vext.32  q10, q2, q3, #2\n" /* ext 2, 3, 4, 5, r1 */          \
  "vext.32  q11, q4, q5, #2\n" /* ext 2, 3, 4, 5, r2 */          \
                                                                 \
  "vmax.f32 q1, q0, q2\n"                                        \
  "vmax.f32 q3, q4, q6\n"                                        \
  "vmax.f32 q5, q7, q8\n"                                        \
  "vmax.f32 q6, q9, q10\n"                                       \
  "vmax.f32 q7, q11, q1\n"                                       \
                                                                 \
  "vmax.f32 q8, q3, q5\n"                                        \
  "vmax.f32 q9, q6, q7\n"                                        \
  "vld1.32  {d0-d1}, [%[dr0]]!\n" /* load q0, dr0, 0-3*/         \
  "vld1.32  {d4-d5}, [%[dr1]]!\n" /* load q0, dr0, 0-3*/         \
  "vld1.32  {d8-d9}, [%[dr2]]!\n" /* load q0, dr0, 0-3*/         \
                                                                 \
  "vmax.f32 q6, q8, q9\n"                                        \
  "subs %[cnt_num], %[cnt_num], #1\n" /* subs cnt_num, #1*/      \
  "vld1.32  {d2}, [%[dr0]]\n"         /* load q1, dr0, 4-5*/     \
  "vld1.32  {d6}, [%[dr1]]\n"         /* load q1, dr0, 4-5*/     \
  "vld1.32  {d10}, [%[dr2]]\n"        /* load q1, dr0, 4-5*/     \
                                                                 \
  "vst1.32  {d12-d13}, [%[dr_out]]!\n" /* store 4 out, dr_out */ \
  "bne       1b\n"                     /* bne s3_max_loop_mid */

#define P3x3S1P1_AVG                                             \
  "vext.32  q6, q0, q1, #1\n"   /* ext 1, 2, 3, 4, r0 */         \
  "vext.32  q7, q2, q3, #1\n"   /* ext 1, 2, 3, 4, r1 */         \
  "vext.32  q8, q4, q5, #1\n"   /* ext 1, 2, 3, 4, r2 */         \
  "vext.32  q9, q15, q0, #3\n"  /* ext -1, 0, 1, 2, r0 */        \
  "vext.32  q10, q15, q2, #3\n" /* ext -1, 0, 1, 2, r1 */        \
  "vext.32  q11, q15, q4, #3\n" /* ext -1, 0, 1, 2, r2 */        \
                                                                 \
  "vadd.f32 q1, q0, q2\n"                                        \
  "vadd.f32 q3, q4, q6\n"                                        \
  "vadd.f32 q5, q7, q8\n"                                        \
  "vadd.f32 q6, q9, q10\n"                                       \
  "vadd.f32 q7, q11, q1\n"                                       \
                                                                 \
  "vadd.f32 q8, q3, q5\n"                                        \
  "vadd.f32 q9, q6, q7\n"                                        \
                                                                 \
  "subs %[dr0], %[dr0], #12\n"                                   \
  "subs %[dr1], %[dr1], #12\n"                                   \
  "subs %[dr2], %[dr2], #12\n"                                   \
  "vadd.f32 q10, q8, q9\n"                                       \
                                                                 \
  "vld1.32  {d0-d1}, [%[dr0]]!\n" /* load q0, dr0, 0-3*/         \
  "vld1.32  {d4-d5}, [%[dr1]]!\n" /* load q2, dr1, 0-3*/         \
  "vld1.32  {d8-d9}, [%[dr2]]!\n" /* load q4, dr2, 0-3*/         \
  "vmul.f32 q11, q10, %q[vcoef_left]\n"                          \
                                                                 \
  "subs %[cnt_num], %[cnt_num], #1\n" /* subs cnt_num, #1*/      \
  "vld1.32  {d2}, [%[dr0]]\n"         /* load q1, dr0, 4-5*/     \
  "vld1.32  {d6}, [%[dr1]]\n"         /* load q3, dr1, 4-5*/     \
  "vld1.32  {d10}, [%[dr2]]\n"        /* load q5, dr2, 4-5*/     \
                                                                 \
  "vst1.32  {d22-d23}, [%[dr_out]]!\n" /* store 4 out, dr_out */ \
  "ble       2f\n"                     /* jump to end */         \
  "1: \n"                              /* */

#define P3x3S1P0_AVG                                             \
  "1: \n"                      /* */                             \
  "vext.32  q6, q0, q1, #1\n"  /* ext 1, 2, 3, 4, r0 */          \
  "vext.32  q7, q2, q3, #1\n"  /* ext 1, 2, 3, 4, r1 */          \
  "vext.32  q8, q4, q5, #1\n"  /* ext 1, 2, 3, 4, r2 */          \
  "vext.32  q9, q0, q1, #2\n"  /* ext 2, 3, 4, 5, r0 */          \
  "vext.32  q10, q2, q3, #2\n" /* ext 2, 3, 4, 5, r1 */          \
  "vext.32  q11, q4, q5, #2\n" /* ext 2, 3, 4, 5, r2 */          \
                                                                 \
  "vadd.f32 q1, q0, q2\n"                                        \
  "vadd.f32 q3, q4, q6\n"                                        \
  "vadd.f32 q5, q7, q8\n"                                        \
  "vadd.f32 q6, q9, q10\n"                                       \
  "vadd.f32 q7, q11, q1\n"                                       \
                                                                 \
  "vadd.f32 q8, q3, q5\n"                                        \
  "vadd.f32 q9, q6, q7\n"                                        \
  "vld1.32  {d0-d1}, [%[dr0]]!\n" /* load q0, dr0, 0-3*/         \
  "vld1.32  {d4-d5}, [%[dr1]]!\n" /* load q2, dr1, 0-3*/         \
                                                                 \
  "vadd.f32 q10, q8, q9\n"                                       \
  "vld1.32  {d8-d9}, [%[dr2]]!\n" /* load q4, dr2, 0-3*/         \
  "vld1.32  {d2}, [%[dr0]]\n"     /* load q1, dr0, 4-5*/         \
                                                                 \
  "vmul.f32 q11, q10, %q[vcoef]\n"                               \
  "subs %[cnt_num], %[cnt_num], #1\n" /* subs cnt_num, #1*/      \
  "vld1.32  {d6}, [%[dr1]]\n"         /* load q3, dr1, 4-5*/     \
  "vld1.32  {d10}, [%[dr2]]\n"        /* load q5, dr2, 4-5*/     \
                                                                 \
  "vst1.32  {d22-d23}, [%[dr_out]]!\n" /* store 4 out, dr_out */ \
  "bne       1b\n"                     /* bne s3_max_loop_mid */

#define P3x3S2_INIT                                           \
  "vld2.f32  {d0-d3}, [%[dr0]]!\n"  /* load q0-q1, dr0, 0-7*/ \
  "vld2.f32  {d4-d7}, [%[dr1]]!\n"  /* load q2-q3, dr1, 0-7*/ \
  "vld2.f32  {d8-d11}, [%[dr2]]!\n" /* load q4-q5, dr2, 0-7*/
#define P3x3S2P0_INIT                                         \
  "vld2.f32  {d0-d3}, [%[dr0]]!\n"  /* load q0-q1, dr0, 0-7*/ \
  "vld2.f32  {d4-d7}, [%[dr1]]!\n"  /* load q2-q3, dr1, 0-7*/ \
  "vld2.f32  {d8-d11}, [%[dr2]]!\n" /* load q4-q5, dr2, 0-7*/ \
  "vld1.f32  {d12-d13}, [%[dr0]]\n" /* load d6, dr0, 8,9 */   \
  "vld1.f32  {d14-d15}, [%[dr1]]\n" /* load d7, dr1, 8,9 */   \
  "vld1.f32  {d16-d17}, [%[dr2]]\n" /* load d8, dr2, 8,9 */
#define P3x3S2P1_MAX                                             \
  "vmax.f32 q6, q0, q1\n"                                        \
  "vmax.f32 q7, q2, q3\n"                                        \
  "vmax.f32 q8, q4, q5\n"                                        \
  "vext.32   q0, %q[vmin], q1, #3\n" /* ext 0, 1, 3, 5, r0 */    \
  "vext.32   q2, %q[vmin], q3, #3\n" /* ext 0, 1, 3, 5, r1 */    \
  "vext.32   q4, %q[vmin], q5, #3\n" /* ext 0, 1, 3, 5, r2 */    \
                                                                 \
  "vmax.f32 q9, q6, q0\n"                                        \
  "vmax.f32 q10, q7, q2\n"                                       \
  "vmax.f32 q11, q8, q4\n"                                       \
                                                                 \
  "subs %[dr0], %[dr0], #4\n"                                    \
  "subs %[dr1], %[dr1], #4\n"                                    \
  "subs %[dr2], %[dr2], #4\n"                                    \
                                                                 \
  "vmax.f32 q6, q9, q10\n"          /* reduce */                 \
  "vld2.f32  {d0-d3}, [%[dr0]]!\n"  /* load q0-q1, dr0, 0-7*/    \
  "vld2.f32  {d4-d7}, [%[dr1]]!\n"  /* load q2-q3, dr1, 0-7*/    \
  "vld2.f32  {d8-d11}, [%[dr2]]!\n" /* load q4-q5, dr2, 0-7*/    \
                                                                 \
  "vmax.f32 q10, q6, q11\n"         /* reduce */                 \
  "subs %[cnt_num], #1\n"           /* subs cnt_num, #1*/        \
  "vld1.f32  {d12-d13}, [%[dr0]]\n" /* load d6, dr0, 8,9 */      \
  "vld1.f32  {d14-d15}, [%[dr1]]\n" /* load d7, dr1, 8,9 */      \
  "vld1.f32  {d16-d17}, [%[dr2]]\n" /* load d8, dr2, 8,9 */      \
                                                                 \
  "vst1.32  {d20-d21}, [%[dr_out]]!\n" /* store 4 out, dr_out */ \
  "ble       2f\n"                     /* jump to end */
#define P3x3S2P0_MAX                                                 \
  "1: \n"                       /* load bias to q2, q3*/             \
  "vmax.f32   q9, q0, q1\n"     /*  add 0, 2, 4, 6 and 1, 3, 5, 7 */ \
  "vmax.f32   q10, q2, q3\n"    /*  add 0, 2, 4, 6 and 1, 3, 5, 7 */ \
  "vmax.f32   q11, q4, q5\n"    /*  add 0, 2, 4, 6 and 1, 3, 5, 7 */ \
  "vext.32    q1, q0, q6, #1\n" /* ext 2, 4, 6, 8, r0 */             \
  "vext.32    q3, q2, q7, #1\n" /* ext 2, 4, 6, 8, r1 */             \
  "vext.32    q5, q4, q8, #1\n" /* ext 2, 4, 6, 8, r2 */             \
                                                                     \
  "vmax.f32  q6, q9, q1\n"  /* add */                                \
  "vmax.f32  q7, q10, q3\n" /* add */                                \
  "vmax.f32  q8, q11, q5\n" /* add */                                \
                                                                     \
  "vmax.f32  q9, q6, q7\n"          /* max reduce */                 \
  "vld2.f32  {d0-d3}, [%[dr0]]!\n"  /* load q0-q1, dr0, 0-7*/        \
  "vld2.f32  {d4-d7}, [%[dr1]]!\n"  /* load q2-q3, dr1, 0-7*/        \
  "vld2.f32  {d8-d11}, [%[dr2]]!\n" /* load q4-q5, dr2, 0-7*/        \
                                                                     \
  "vmax.f32  q10, q9, q8\n"           /* max reduce */               \
  "subs %[cnt_num], %[cnt_num], #1\n" /* subs cnt_num, #1*/          \
  "vld1.f32  {d12-d13}, [%[dr0]]\n"   /* load d6, dr0, 8,9 */        \
  "vld1.f32  {d14-d15}, [%[dr1]]\n"   /* load d7, dr1, 8,9 */        \
  "vld1.f32  {d16-d17}, [%[dr2]]\n"   /* load d8, dr2, 8,9 */        \
                                                                     \
  "vst1.32  {d20-d21}, [%[dr_out]]!\n" /* store 4 out, dr_out */     \
  "bne       1b\n"                     /* bne s3_max_loop_mid */

#define P3x3S2P1_AVG                                             \
  "vadd.f32 q6, q0, q1\n"                                        \
  "vadd.f32 q7, q2, q3\n"                                        \
  "vadd.f32 q8, q4, q5\n"                                        \
  "vext.32   q0, q15, q1, #3\n" /* ext 0, 1, 3, 5, r0 */         \
  "vext.32   q2, q15, q3, #3\n" /* ext 0, 1, 3, 5, r1 */         \
  "vext.32   q4, q15, q5, #3\n" /* ext 0, 1, 3, 5, r2 */         \
  "vadd.f32 q6, q6, q0\n"                                        \
  "vadd.f32 q7, q7, q2\n"                                        \
  "vadd.f32 q8, q8, q4\n"                                        \
                                                                 \
  "vadd.f32 q9, q6, q7\n" /* reduce */                           \
  "subs %[dr0], %[dr0], #4\n"                                    \
  "subs %[dr1], %[dr1], #4\n"                                    \
  "subs %[dr2], %[dr2], #4\n"                                    \
                                                                 \
  "vadd.f32 q10, q9, q8\n"          /* reduce */                 \
  "vld2.f32  {d0-d3}, [%[dr0]]!\n"  /* load q0-q1, dr0, 0-7*/    \
  "vld2.f32  {d4-d7}, [%[dr1]]!\n"  /* load q2-q3, dr1, 0-7*/    \
  "vld2.f32  {d8-d11}, [%[dr2]]!\n" /* load q4-q5, dr2, 0-7*/    \
                                                                 \
  "vmul.f32 q11, q10, %q[vcoef_left]\n"                          \
  "subs %[cnt_num], #1\n"           /* subs cnt_num, #1*/        \
  "vld1.f32  {d12-d13}, [%[dr0]]\n" /* load d6, dr0, 8,9 */      \
  "vld1.f32  {d14-d15}, [%[dr1]]\n" /* load d7, dr1, 8,9 */      \
  "vld1.f32  {d16-d17}, [%[dr2]]\n" /* load d8, dr2, 8,9 */      \
                                                                 \
  "vst1.32  {d22-d23}, [%[dr_out]]!\n" /* store 4 out, dr_out */ \
  "ble       2f\n"                     /* jump to end */

#define P3x3S2P0_AVG                                                   \
  "1: \n"                                                              \
  "vadd.f32   q9, q0, q1\n"     /*  add 0, 2, 4, 6 and 1, 3, 5, 7,  */ \
  "vadd.f32   q10, q2, q3\n"    /*  add 0, 2, 4, 6 and 1, 3, 5, 7, */  \
  "vadd.f32   q11, q4, q5\n"    /*  add 0, 2, 4, 6 and 1, 3, 5, 7, */  \
  "vext.32    q1, q0, q6, #1\n" /* ext 2, 4, 6, 8, r0 */               \
  "vext.32    q3, q2, q7, #1\n" /* ext 2, 4, 6, 8, r1 */               \
  "vext.32    q5, q4, q8, #1\n" /* ext 2, 4, 6, 8, r2 */               \
                                                                       \
  "vadd.f32  q9, q9, q1\n"   /* add */                                 \
  "vadd.f32  q10, q10, q3\n" /* add */                                 \
  "vadd.f32  q11, q11, q5\n" /* add */                                 \
                                                                       \
  "vadd.f32  q9, q9, q10 \n"       /* max reduce */                    \
  "vld2.f32  {d0-d3}, [%[dr0]]!\n" /* load q0-q1, dr0, 0-7*/           \
  "vld2.f32  {d4-d7}, [%[dr1]]!\n" /* load q2-q3, dr1, 0-7*/           \
                                                                       \
  "vadd.f32  q10, q9, q11 \n"         /* max reduce */                 \
  "vld2.f32  {d8-d11}, [%[dr2]]!\n"   /* load q4-q5, dr2, 0-7*/        \
  "subs %[cnt_num], %[cnt_num], #1\n" /* subs cnt_num, #1*/            \
                                                                       \
  "vmul.f32 q11, q10, %q[vcoef]\n"                                     \
  "vld1.f32  {d12-d13}, [%[dr0]]\n" /* load d6, dr0, 8,9 */            \
  "vld1.f32  {d14-d15}, [%[dr1]]\n" /* load d7, dr1, 8,9 */            \
  "vld1.f32  {d16-d17}, [%[dr2]]\n" /* load d8, dr2, 8,9 */            \
                                                                       \
  "vst1.32  {d22-d23}, [%[dr_out]]!\n" /* store 4 out, dr_out */       \
  "bne       1b\n"                     /* bne s3_max_loop_mid */

#endif

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
      int size_cnt = cnt;
      if (cnt > 0) {
#ifdef __aarch64__
        asm volatile(
            GLOBAL_INIT GLOBAL_MAX
            : [data_in_channel] "+r"(data_in_channel),
              [cnt] "+r"(size_cnt),
              [vmax] "+w"(vmax)
            :
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6");
#else
        asm volatile(
            GLOBAL_INIT GLOBAL_MAX
            : [data_in_channel] "+r"(data_in_channel),
              [cnt] "+r"(size_cnt),
              [vmax] "+w"(vmax)
            :
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
#endif  //  __aarch64__
        data_in_channel -= 16;
      }
      float32x2_t vmax_tmp = vmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
      float max_tmp = vmax_tmp[0] > vmax_tmp[1] ? vmax_tmp[0] : vmax_tmp[1];
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
      int size_cnt = cnt;
      if (cnt > 0) {
#ifdef __aarch64__
        asm volatile(GLOBAL_INIT GLOBAL_AVG
                     : [data_in_channel] "+r"(data_in_channel),
                       [cnt] "+r"(size_cnt),
                       [vsum] "+w"(vsum)
                     :
                     : "cc", "memory", "v0", "v1", "v2", "v3", "v4");
#else
        asm volatile(GLOBAL_INIT GLOBAL_AVG
                     : [data_in_channel] "+r"(data_in_channel),
                       [cnt] "+r"(size_cnt),
                       [vsum] "+w"(vsum)
                     :
                     : "cc", "memory", "q0", "q1", "q2", "q3", "q4");
#endif  //  __aarch64__
        data_in_channel -= 16;
      }
      float32x2_t vsum_tmp = vadd_f32(vget_low_f32(vsum), vget_high_f32(vsum));
      float sum = vsum_tmp[0] + vsum_tmp[1];
      for (i = cnt * 16; i < size_channel_in; i++) {
        sum += data_in_channel[0];
        data_in_channel++;
      }
      data_out_batch[c] = sum / size_channel_in;
    }
  }
}

void pooling1x1s2p0_max(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win,
                        int pad_bottom,
                        int pad_right) {
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);

  int w_unroll_size = wout / 4;
  int w_unroll_remian = wout - w_unroll_size * 4;
  int win_ext = w_unroll_size * 8;
  auto zero_ptr =
      static_cast<float*>(TargetMalloc(TARGET(kARM), win * sizeof(float)));
  memset(zero_ptr, 0, win * sizeof(float));
  auto write_ptr =
      static_cast<float*>(TargetMalloc(TARGET(kARM), wout * sizeof(float)));

  for (int n = 0; n < num; ++n) {
    float* data_out_batch = data_out + n * chout * size_channel_out;
    const float* data_in_batch = data_in + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      float* data_out_channel = data_out_batch + c * size_channel_out;
      const float* data_in_channel = data_in_batch + c * size_channel_in;
      for (int h = 0; h < hout; h += 4) {
        const float* din0_ptr = data_in_channel + h * 2 * win;
        const float* din1_ptr = din0_ptr + 2 * win;
        const float* din2_ptr = din1_ptr + 2 * win;
        const float* din3_ptr = din2_ptr + 2 * win;

        float* doutr0 = data_out_channel + h * wout;
        float* doutr1 = doutr0 + wout;
        float* doutr2 = doutr1 + wout;
        float* doutr3 = doutr2 + wout;
        if (h + 4 > hout) {
          switch (h + 4 - hout) {
            case 3:
              doutr1 = write_ptr;
            case 2:
              doutr2 = write_ptr;
            case 1:
              doutr3 = write_ptr;
            default:
              break;
          }
        }
        if (h * 2 + 7 > hin) {
          switch (h * 2 + 7 - hin) {
            case 7:
              din0_ptr = zero_ptr;
            case 6:
            case 5:
              din1_ptr = zero_ptr;
            case 4:
            case 3:
              din2_ptr = zero_ptr;
            case 2:
            case 1:
              din3_ptr = zero_ptr;
            default:
              break;
          }
        }
        for (int i = 0; i < w_unroll_size; i++) {
          float32x4x2_t din0 = vld2q_f32(din0_ptr);
          float32x4x2_t din1 = vld2q_f32(din1_ptr);
          float32x4x2_t din2 = vld2q_f32(din2_ptr);
          float32x4x2_t din3 = vld2q_f32(din3_ptr);
          din0_ptr += 8;
          din1_ptr += 8;
          din2_ptr += 8;
          din3_ptr += 8;

          vst1q_f32(doutr0, din0.val[0]);
          vst1q_f32(doutr1, din1.val[0]);
          vst1q_f32(doutr2, din2.val[0]);
          vst1q_f32(doutr3, din3.val[0]);

          doutr0 += 4;
          doutr1 += 4;
          doutr2 += 4;
          doutr3 += 4;
        }
        int j = win_ext;
        for (int i = 0; i < w_unroll_remian; i++) {
          if (j >= win) {
            *doutr0++ = 0.f;
            *doutr1++ = 0.f;
            *doutr2++ = 0.f;
            *doutr3++ = 0.f;
          } else {
            *doutr0++ = *din0_ptr;
            *doutr1++ = *din1_ptr;
            *doutr2++ = *din2_ptr;
            *doutr3++ = *din3_ptr;
            din0_ptr += 2;
            din1_ptr += 2;
            din2_ptr += 2;
            din3_ptr += 2;
          }
          j += 2;
        }
      }
    }
  }
  TargetFree(TARGET(kARM), zero_ptr);
  TargetFree(TARGET(kARM), write_ptr);
}

void pooling2x2s2p0_max(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win,
                        int pad_bottom,
                        int pad_right) {
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
        int cnt_num = w_unroll_size;
        if (w_unroll_size > 0) {
#ifdef __aarch64__
          asm volatile(
              P2x2S2_INIT P2x2S2P0_MAX
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              :
              : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6");
#else
          asm volatile(
              P2x2S2_INIT P2x2S2P0_MAX
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              :
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q8");
#endif
          dr0 -= 8;
          dr1 -= 8;
        }
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

void pooling2x2s2p0_avg(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win,
                        bool exclusive,
                        int pad_bottom,
                        int pad_right) {
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
      vcoef = vdupq_n_f32(0.25f);
      for (int h = 0; h < hout; h++) {
        float* dr_out = data_out_channel;
        auto dr0 = r0;
        auto dr1 = r1;
        if (h * S + K - P > hin) {
          dr1 = zero_ptr;
          vcoef = vdupq_n_f32(0.5f);
        }
        int cnt_num = w_unroll_size;
        if (w_unroll_size > 0) {
#ifdef __aarch64__
          asm volatile(
              P2x2S2_INIT P2x2S2P0_AVG
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vcoef] "w"(vcoef)
              : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6");
#else
          asm volatile(
              P2x2S2_INIT P2x2S2P0_AVG
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vcoef] "w"(vcoef)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q8");
#endif
          dr0 -= 8;
          dr1 -= 8;
        }
        // deal with right pad
        int rem = win - (w_unroll_size * 4) * S;
        int wstart = 0;
        for (int j = 0; j < w_unroll_remian; ++j) {
          int wend = std::min(wstart + K, rem);
          float coef = 0.25f;
          float tmp = 0.f;
          if (wend - wstart == 1 && pad_right == 0) {
            coef *= 2;
          }
          if (h * S + K - P > hin && pad_bottom == 0) {
            coef *= 2;
          }
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
  TargetFree(TARGET(kARM), zero_ptr);
}

void pooling2x2s2p1_max(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win,
                        int pad_bottom,
                        int pad_right) {
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);

  const int K = 2;
  const int P = 1;
  const int S = 2;

  int w_unroll_size = wout / 4;
  int w_unroll_remian = wout - w_unroll_size * 4;
  float32x4_t vzero = vdupq_n_f32(std::numeric_limits<float>::lowest());
  if (w_unroll_remian == 0) {
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
      for (int h = 0; h < hout; h++) {
        float* dr_out = data_out_channel;
        auto dr0 = r0;
        auto dr1 = r1;
        if (h == 0) {
          dr0 = r0;
          dr1 = r0;
          r0 = r1;
          r1 = r0 + win;
        } else {
          r0 = r1 + win;
          r1 = r0 + win;
        }
        if (h * S + K - P > hin) {
          dr1 = dr0;
          if (h * S + K - P > hin + 1) {
            memset(dr_out, 0, wout * sizeof(float));
            continue;
          }
        }
        int cnt_num = w_unroll_size;
        if (w_unroll_size > 0) {
#ifdef __aarch64__
          asm volatile(
              P2x2S2_INIT P2x2S2P1_MAX P2x2S2P0_MAX "2: \n" /* end */
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vzero] "w"(vzero)
              : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v8");
#else
          asm volatile(
              P2x2S2_INIT P2x2S2P1_MAX P2x2S2P0_MAX "2: \n" /* end */
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vzero] "w"(vzero)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q8", "q9");
#endif
          dr0 -= 8;
          dr1 -= 8;
        }
        // deal with right pad
        int wstart = w_unroll_size * 4 * S - P;
        for (int j = 0; j < w_unroll_remian; ++j) {
          int wend = std::min(wstart + K, win);
          int st = wstart > 0 ? wstart : 0;
          float tmp = wend == st ? 0.f : dr0[0];
          for (int i = 0; i < wend - st; i++) {
            tmp = std::max(tmp, dr0[i]);
            tmp = std::max(tmp, dr1[i]);
          }
          *(dr_out++) = tmp;
          dr0 += S - (st - wstart);
          dr1 += S - (st - wstart);
          wstart += S;
        }
        data_out_channel += wout;
      }
    }
  }
}

void pooling2x2s2p1_avg(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win,
                        bool exclusive,
                        int pad_bottom,
                        int pad_right) {
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);

  const int K = 2;
  const int P = 1;
  const int S = 2;

  int w_unroll_size = wout / 4;
  int w_unroll_remian = wout - w_unroll_size * 4;
  auto zero_ptr =
      static_cast<float*>(TargetMalloc(TARGET(kARM), win * sizeof(float)));
  float32x4_t vzero = vdupq_n_f32(0.f);
  memset(zero_ptr, 0, win * sizeof(float));

  if (w_unroll_remian == 0) {
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
      for (int h = 0; h < hout; h++) {
        float* dr_out = data_out_channel;
        auto dr0 = r0;
        auto dr1 = r1;
        float coef_h = 0.5f;
        if (h == 0) {
          dr0 = zero_ptr;
          dr1 = r0;
          r0 = r1;
          r1 = r0 + win;
          if (exclusive) {
            coef_h = 1.f;
          }
        } else {
          r0 = r1 + win;
          r1 = r0 + win;
        }
        if (h * S + K - P > hin) {
          dr1 = zero_ptr;
          if (exclusive) {
            coef_h = 1.f;
          }
          if (h * S + K - P > hin + 1) {
            memset(dr_out, 0, wout * sizeof(float));
            continue;
          }
        }
        float coef_left_most = exclusive ? coef_h : coef_h / 2;
        float32x4_t vcoef = vdupq_n_f32(coef_h / 2);
        float coef_left[4] = {
            coef_left_most, coef_h / 2, coef_h / 2, coef_h / 2};
        float32x4_t vcoef_left = vld1q_f32(coef_left);
        int cnt_num = w_unroll_size;
        if (w_unroll_size > 0) {
#ifdef __aarch64__
          asm volatile(
              P2x2S2_INIT P2x2S2P1_AVG P2x2S2P0_AVG "2: \n"
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vcoef] "w"(vcoef),
                [vzero] "w"(vzero),
                [vcoef_left] "w"(vcoef_left)
              : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v8");
#else
          asm volatile(
              P2x2S2_INIT P2x2S2P1_AVG P2x2S2P0_AVG "2: \n"
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr_out] "+r"(dr_out),
                [cnt_num] "+r"(cnt_num)
              : [vcoef] "w"(vcoef),
                [vzero] "w"(vzero),
                [vcoef_left] "w"(vcoef_left)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q8", "q9");
#endif
          dr0 -= 8;
          dr1 -= 8;
        }
        // deal with right pad
        int wstart = w_unroll_size * 4 * S - P;
        for (int j = 0; j < w_unroll_remian; ++j) {
          int wend = std::min(wstart + K, win);
          int st = wstart > 0 ? wstart : 0;
          float tmp = 0.f;
          float coef = coef_h / 2;
          if (exclusive && wend - st == 1) {
            coef = coef_h;
          }
          for (int i = 0; i < wend - st; i++) {
            tmp += dr0[i] + dr1[i];
          }
          *(dr_out++) = tmp * coef;
          dr0 += S - (st - wstart);
          dr1 += S - (st - wstart);
          wstart += S;
        }
        data_out_channel += wout;
      }
    }
  }
  TargetFree(TARGET(kARM), zero_ptr);
}

void pooling3x3s1p1_max(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win,
                        int pad_bottom,
                        int pad_right) {
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
        int cnt_num = w_unroll_size;
        if (w_unroll_size > 0) {
#ifdef __aarch64__
          asm volatile(
              /* preocess left */
              P3x3S1_INIT P3x3S1P1_MAX P3x3S1P0_MAX "2: \n" /* end */
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
                "v11",
                "v31");
#else
          asm volatile(
              /* preocess left */
              P3x3S1_INIT P3x3S1P1_MAX P3x3S1P0_MAX "2: \n" /* end */
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
                "q11",
                "q15");
#endif
          dr0 -= 4;
          dr1 -= 4;
          dr2 -= 4;
        }
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
                        bool exclusive,
                        int pad_bottom,
                        int pad_right) {
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
                if (pad_bottom > 1) {
                  coef_h = 1.f / 3;
                } else if (pad_bottom == 1) {
                  coef_h = 0.5f;
                } else {
                  coef_h = 1.f;
                }
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
                if (pad_bottom >= 1) {
                  coef_h = 1.f / 3;
                } else {
                  coef_h = 0.5f;
                }
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
        int cnt_num = w_unroll_size;
        if (w_unroll_size > 0) {
#ifdef __aarch64__
          asm volatile("movi v31.4s, #0\n"
                       /* preocess left */
                       P3x3S1_INIT P3x3S1P1_AVG P3x3S1P0_AVG "2: \n" /* end */
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
#else
          asm volatile("vmov.i32 q15, #0\n"
                       /* preocess left */
                       P3x3S1_INIT P3x3S1P1_AVG P3x3S1P0_AVG "2: \n" /* end */
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
#endif
          dr0 -= 4;
          dr1 -= 4;
          dr2 -= 4;
        }
        // deal with right pad
        int wstart = w_unroll_size * 4 * S - P;
        for (int j = 0; j < w_unroll_remian; ++j) {
          int wend = wstart + K;  // std::min(wstart + K, win);
          float coef = coef_h / 3.f;
          int st = wstart > 0 ? wstart : 0;
          if (wstart + K > win) {
            wend = win;
            if (!exclusive) {
              if (wstart + K - pad_right - win == 1) {
                coef = coef_h / 2;
              } else if (wstart + K - pad_right - win == 2) {
                coef = coef_h;
              }
            }
          }
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

void pooling3x3s1p0_max(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win,
                        int pad_bottom,
                        int pad_right) {
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);

  const int K = 3;
  const int P = 0;
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
        int cnt_num = w_unroll_size;
        if (w_unroll_size > 0) {
#ifdef __aarch64__
          asm volatile(
              /* preocess left */
              P3x3S1_INIT P3x3S1P0_MAX "2: \n" /* end */
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
                "v11",
                "v31");
#else
          asm volatile(
              /* preocess left */
              P3x3S1P0_INIT P3x3S1P0_MAX
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
                "q11",
                "q15");
#endif
          dr0 -= 4;
          dr1 -= 4;
          dr2 -= 4;
        }
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
        r0 = r1;
        r1 = r2;
        r2 = r1 + win;
        data_out_channel += wout;
      }
    }
  }
}

void pooling3x3s1p0_avg(const float* din,
                        float* dout,
                        int num,
                        int chout,
                        int hout,
                        int wout,
                        int chin,
                        int hin,
                        int win,
                        bool exclusive,
                        int pad_bottom,
                        int pad_right) {
  int size_channel_out = wout * hout;
  int size_channel_in = win * hin;
  auto data_out = static_cast<float*>(dout);
  auto data_in = static_cast<const float*>(din);

  const int K = 3;
  const int P = 0;
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
        if (h * S + K - P > hin) {
          switch (h * S + K - P - hin) {
            case 2:
              dr1 = zero_ptr;
              dr2 = zero_ptr;
              if (exclusive) {
                coef_h = 1.f;
              } else {
                if (pad_bottom > 1) {
                  coef_h = 1.f / 3;
                } else if (pad_bottom == 1) {
                  coef_h = 0.5f;
                } else {
                  coef_h = 1.f;
                }
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
                if (pad_bottom >= 1) {
                  coef_h = 1.f / 3;
                } else {
                  coef_h = 0.5f;
                }
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
        int cnt_num = w_unroll_size;
        if (w_unroll_size > 0) {
#ifdef __aarch64__
          asm volatile("movi v31.4s, #0\n" P3x3S1_INIT P3x3S1P0_AVG
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
#else
          asm volatile("vmov.i32 q15, #0\n" P3x3S1P0_INIT P3x3S1P0_AVG
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
#endif
          dr0 -= 4;
          dr1 -= 4;
          dr2 -= 4;
        }
        // deal with right pad
        int wstart = w_unroll_size * 4 * S - P;
        for (int j = 0; j < w_unroll_remian; ++j) {
          int wend = wstart + K;  // std::min(wstart + K, win);
          float coef = coef_h / 3.f;
          int st = wstart > 0 ? wstart : 0;
          if (wstart + K > win) {
            wend = win;
            if (!exclusive) {
              if (wstart + K - pad_right - win == 1) {
                coef = coef_h / 2;
              } else if (wstart + K - pad_right - win == 2) {
                coef = coef_h;
              }
            }
          }
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
        r0 = r1;
        r1 = r2;
        r2 = r1 + win;
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
                        int win,
                        int pad_bottom,
                        int pad_right) {
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
  int w_needed = wout * 2 + 1;
  int pad_right_ = w_needed - win - pad_bottom;
  int w_2 = pad_right_ > 0 ? w_unroll_remian : w_unroll_remian + 1;
  w_2 = w_unroll_size <= 0 ? w_2 - 1 : w_2;

  float minval = std::numeric_limits<float>::lowest();
  float32x4_t vmin = vdupq_n_f32(minval);

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

        auto pr0 = dr0;
        auto pr1 = dr1;
        auto pr2 = dr2;

        int cnt_num = w_unroll_size;
        if (w_unroll_size > 0) {
#ifdef __aarch64__
          asm volatile(
              /* preocess left */
              P3x3S2_INIT P3x3S2P1_MAX P3x3S2P0_MAX "2: \n" /* end */
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
                "v11",
                "v31");
#else
          asm volatile(
              /* preocess left */
              P3x3S2_INIT P3x3S2P1_MAX P3x3S2P0_MAX "2: \n" /* end */
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
                "q11",
                "q15");
#endif

          dr0 -= 8;
          dr1 -= 8;
          dr2 -= 8;
        } else {
          float tmp = minval;
          for (int i = 0; i < 2; i++) {
            tmp = std::max(tmp, std::max(dr0[i], dr1[i]));
            tmp = std::max(tmp, dr2[i]);
          }

          dr_out[0] = tmp;
          dr0++;
          dr1++;
          dr2++;
          dr_out++;
        }

        for (int w = 0; w < w_2 - 1; w += 1) {
          float32x4_t vr0 = vld1q_f32(dr0);
          float32x4_t vr1 = vld1q_f32(dr1);
          float32x4_t vr2 = vld1q_f32(dr2);
          vr0 = vsetq_lane_f32(minval, vr0, 3);
          vr1 = vsetq_lane_f32(minval, vr1, 3);
          vr2 = vsetq_lane_f32(minval, vr2, 3);
          float32x4_t vmax1 = vmaxq_f32(vr0, vr1);
          vmax1 = vmaxq_f32(vmax1, vr2);
          float32x2_t vmax2 =
              vpmax_f32(vget_low_f32(vmax1), vget_high_f32(vmax1));
          float32x2_t vmax = vpmax_f32(vmax2, vmax2);
          dr_out[0] = vget_lane_f32(vmax, 0);
          dr_out++;

          dr0 += 2;
          dr1 += 2;
          dr2 += 2;
        }

        if (pad_right_) {
          float tmp = minval;
          for (int i = 1; i < 3; i++) {
            tmp = std::max(tmp, std::max(pr0[win - i], pr1[win - i]));
            tmp = std::max(tmp, pr2[win - i]);
          }
          dr_out[0] = tmp;
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
                        bool exclusive,
                        int pad_bottom,
                        int pad_right) {
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
                if (pad_bottom > 1) {
                  coef_h = 1.f / 3;
                } else if (pad_bottom == 1) {
                  coef_h = 0.5f;
                } else {
                  coef_h = 1.f;
                }
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
                if (pad_bottom == 0) {
                  coef_h = 1.f / 2;
                } else {
                  coef_h = 1.f / 3;
                }
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
        int cnt_num = w_unroll_size;
        if (w_unroll_size > 0) {
#ifdef __aarch64__
          asm volatile("movi v31.4s, #0\n"
                       /* preocess left */
                       P3x3S2_INIT P3x3S2P1_AVG P3x3S2P0_AVG "2: \n" /* end */
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
#else
          asm volatile("vmov.i32 q15, #0\n"
                       /* preocess left */
                       P3x3S2_INIT P3x3S2P1_AVG P3x3S2P0_AVG "2: \n" /* end */
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
#endif
          dr0 -= 8;
          dr1 -= 8;
          dr2 -= 8;
        }
        // deal with right pad
        int wstart = w_unroll_size * 4 * S - P;
        for (int j = 0; j < w_unroll_remian; ++j) {
          int wend = wstart + K;  // std::min(wstart + K, win);
          float coef = coef_h / 3.f;
          if (wstart + K > win) {
            wend = win;
            if (!exclusive) {
              if (wstart + K - pad_right - win == 1) {
                coef = coef_h / 2;
              } else if (wstart + K - pad_right - win == 2) {
                coef = coef_h;
              }
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
                        int win,
                        int pad_bottom,
                        int pad_right) {
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

  int remain = w_unroll_remian - 1;
  int right = wout * 2 + 1 - win;  // if need right pad

  int w_2 = right > 0 ? w_unroll_remian : w_unroll_remian + 1;
  w_2 = w_unroll_size <= 0 ? w_2 - 1 : w_2;
  float minval = std::numeric_limits<float>::lowest();

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
        int cnt_num = w_unroll_size;
        int cnt_remain = remain;
        if (w_unroll_size > 0) {
#ifdef __aarch64__
          asm volatile(P3x3S2P0_INIT P3x3S2P0_MAX
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
          dr0 -= 8;
          dr1 -= 8;
          dr2 -= 8;

          for (int w = 0; w < w_2 - 1; w += 1) {
            float32x4_t vr0 = vld1q_f32(dr0);
            float32x4_t vr1 = vld1q_f32(dr1);
            float32x4_t vr2 = vld1q_f32(dr2);
            vr0 = vsetq_lane_f32(minval, vr0, 3);
            vr1 = vsetq_lane_f32(minval, vr1, 3);
            vr2 = vsetq_lane_f32(minval, vr2, 3);
            float32x4_t vmax1 = vmaxq_f32(vr0, vr1);
            vmax1 = vmaxq_f32(vmax1, vr2);
            float32x2_t vmax2 =
                vpmax_f32(vget_low_f32(vmax1), vget_high_f32(vmax1));
            float32x2_t vmax = vpmax_f32(vmax2, vmax2);
            dr_out[0] = vget_lane_f32(vmax, 0);
            dr_out++;
            dr0 += 2;
            dr1 += 2;
            dr2 += 2;
          }
#else
          asm volatile(
              P3x3S2P0_INIT P3x3S2P0_MAX
              "cmp       %[remain], #0                         @cmp cnt_num\n"
              "sub       %[dr0], #32                           @sub - 8\n"
              "sub       %[dr1], #32                           @sub - 8\n"
              "sub       %[dr2], #32                           @sub - 8\n"
              "ble       4f                                    @ble exit1\n"
              "2:                                              @mid loop\n"
              "vld1.f32  {d0-d1}, [%[dr0]]!                    @load \n"
              "vld1.f32  {d2-d3}, [%[dr1]]!                    @load \n"
              "vld1.f32  {d4-d5}, [%[dr2]]!                    @load \n"
              "vmov.f32  s3,s2                                 @mov \n"
              "vmov.f32  s7,s6                                 @mov \n"
              "vmov.f32  s11,s10                               @mov \n"
              "vmax.f32  q0, q0, q1                            @max n"
              "sub       %[dr0], #8                            @add w \n"
              "sub       %[dr1], #8                            @add w \n"
              "sub       %[dr2], #8                            @add w \n"
              "vmax.f32  q0, q0, q2                            @max \n"
              "vpmax.f32 d0, d0, d1                            @pmax \n"
              "vpmax.f32 d0, d0, d0                            @pmax \n"
              "subs      %[remain], #1                         @subs \n"
              "vst1.f32  d0[0], [%[dr_out]]!                   @vst \n"
              "bne       2b                                    @bne \n"
              "4:                                              @exit\n"
              : [dr0] "+r"(dr0),
                [dr1] "+r"(dr1),
                [dr2] "+r"(dr2),
                [dr_out] "+r"(dr_out),
                [remain] "+r"(cnt_remain),
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
          if (right) {
            int wstart = (w_unroll_size * 4 + remain) * S;
            int wend = std::min(wstart + K, win);
            float tmp = dr0[wstart];  // std::numeric_limits<float>::min();
            for (int i = wstart; i < wend; i++) {
              tmp = std::max(tmp, std::max(dr0[i], dr1[i]));
              tmp = std::max(tmp, dr2[i]);
            }
            *(dr_out++) = tmp;
          }
#endif
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
                        bool exclusive,
                        int pad_bottom,
                        int pad_right) {
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
  //  do overflow process
  w_unroll_size -= 1;
  w_unroll_remian += 4;
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
              if (exclusive) {
                coef_h = 1.f;
              } else {
                if (pad_bottom >= 2) {
                  coef_h = 1.f / 3;
                } else if (pad_bottom == 1) {
                  coef_h = 0.5f;
                } else {
                  coef_h = 1.0f;
                }
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
                if (pad_bottom >= 1) {
                  coef_h = 1.0f / 3;
                } else {
                  coef_h = 0.5f;
                }
              }
              break;
            default:
              break;
          }
        }
        float32x4_t vcoef = vdupq_n_f32(coef_h / 3);
        int cnt_num = w_unroll_size;
        if (w_unroll_size > 0) {
#ifdef __aarch64__
          asm volatile(P3x3S2P0_INIT P3x3S2P0_AVG
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
#else
          asm volatile(P3x3S2P0_INIT P3x3S2P0_AVG
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
#endif
          dr0 -= 8;
          dr1 -= 8;
          dr2 -= 8;
        }
        // deal with right pad
        int wstart = w_unroll_size * 4 * S - P;
        for (int j = 0; j < w_unroll_remian; ++j) {
          int wend = wstart + K;  // std::min(wstart + K, win);
          float coef = coef_h / 3.f;
          if (wstart + K > win) {
            wend = win;
            if (!exclusive) {
              if (wstart + K - pad_right - win == 1) {
                coef = coef_h / 2;
              } else if (wstart + K - pad_right - win == 2) {
                coef = coef_h;
              }
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
