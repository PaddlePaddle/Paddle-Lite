// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/arm/math/fp16/pooling_fp16.h"
#include <algorithm>
#include <limits>
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

#ifdef __aarch64__
#define CHANGEED_REG_0_11 \
: "cc",                   \
  "memory",               \
  "v0",                   \
  "v1",                   \
  "v2",                   \
  "v3",                   \
  "v4",                   \
  "v5",                   \
  "v6",                   \
  "v7",                   \
  "v8",                   \
  "v9",                   \
  "v10",                  \
  "v11"
#else
#define CHANGEED_REG_0_11 \
: "cc",                   \
  "memory",               \
  "q0",                   \
  "q1",                   \
  "q2",                   \
  "q3",                   \
  "q4",                   \
  "q5",                   \
  "q6",                   \
  "q7",                   \
  "q8",                   \
  "q9",                   \
  "q10",                  \
  "q11"
#endif

int AdaptStartIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      floor(static_cast<double>(ph * input_size) / output_size));
}

int AdaptEndIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      ceil(static_cast<double>((ph + 1) * input_size) / output_size));
}

// hin mod hout is 0,  win mod wout is 0, win/wout mod 4 is 0
void pooling_avg_fp16_adaptive_exclusive_p0(POOLING_PARAM) {
  int size_channel_in = win * hin;
  int size_channel_out = wout * hout;
  int kernel_h = hin / hout;
  int kernel_w = win / wout;
  float kernel_size = 1.f / (kernel_h * kernel_w);

  for (int n = 0; n < num; ++n) {
    LITE_PARALLEL_BEGIN(c, tid, chin) {
      for (int h = 0; h < hout; ++h) {
        for (int w = 0; w < wout; ++w) {
          const float16_t *input = din + (n * chin + c) * size_channel_in +
                                   h * kernel_h * win + w * kernel_w;
          float16_t *output =
              dout + (n * chout + c) * size_channel_out + h * wout + w;
          int kh = 0, kw = 0;
          float16x8_t sum = vdupq_n_f16(0);
          float16x4_t sum1 = vdup_n_f16(0);
          for (kh = 0; kh + 1 < kernel_h; kh += 2) {
            const float16_t *line0 = input + kh * win;
            const float16_t *line1 = line0 + win;
            for (kw = 0; kw + 7 < kernel_w; kw += 8) {
              sum = vaddq_f16(vld1q_f16(line0 + kw), sum);
              sum = vaddq_f16(vld1q_f16(line1 + kw), sum);
            }
            for (; kw + 3 < kernel_w; kw += 4) {
              sum1 = vadd_f16(vld1_f16(line0 + kw), sum1);
              sum1 = vadd_f16(vld1_f16(line1 + kw), sum1);
            }
          }
          for (; kh < kernel_h; kh++) {
            const float16_t *line0 = input + kh * win;
            const float16_t *line1 = line0 + win;
            for (kw = 0; kw + 7 < kernel_w; kw += 8) {
              sum = vaddq_f16(vld1q_f16(line0 + kw), sum);
              sum = vaddq_f16(vld1q_f16(line1 + kw), sum);
            }
            for (; kw + 3 < kernel_w; kw += 4) {
              sum1 = vadd_f16(vld1_f16(line0 + kw), sum1);
              sum1 = vadd_f16(vld1_f16(line1 + kw), sum1);
            }
          }
          float16x4_t vsum = vadd_f16(vget_low_f16(sum), vget_high_f16(sum));
          float16x4_t vsum_half = vadd_f16(vsum, sum1);
          vsum_half = vpadd_f16(vsum_half, vsum_half);
          vsum_half = vpadd_f16(vsum_half, vsum_half);
          output[0] = vsum_half[0] * kernel_size;
        }
      }
    }
    LITE_PARALLEL_END()
  }
}

void pooling_basic_fp16(POOLING_PARAM,
                        const std::vector<int> &ksize,
                        const std::vector<int> &strides,
                        const std::vector<int> &paddings,
                        bool global_pooling,
                        bool exclusive,
                        bool adaptive,
                        bool ceil_mode,
                        bool use_quantizer,
                        const std::string &pooling_type) {
  int kernel_h = ksize[0];
  int kernel_w = ksize[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[2];
  int size_channel_in = win * hin;
  int size_channel_out = wout * hout;

  if (exclusive && adaptive && pad_h == 0 && pad_w == 0 && hin % hout == 0 &&
      win % wout == 0 && pooling_type == "avg" && !global_pooling) {
    int scale = win / wout;
    if (scale % 4 == 0) {
      pooling_avg_fp16_adaptive_exclusive_p0(
          din, dout, num, chout, hout, wout, chin, hin, win);
      return;
    }
  }

  // no need to pad input tensor, border is zero pad inside this function
  memset(dout, 0, num * chout * hout * wout * sizeof(float16_t));
  if (global_pooling) {
    if (pooling_type == "max") {  // Pooling_max
      for (int n = 0; n < num; ++n) {
        float16_t *dout_batch = dout + n * chout * size_channel_out;
        const float16_t *din_batch = din + n * chin * size_channel_in;

        LITE_PARALLEL_BEGIN(c, tid, chout) {
          const float16_t *din_ch =
              din_batch + c * size_channel_in;  // in address
          float16_t tmp1 = din_ch[0];
          for (int i = 0; i < size_channel_in; ++i) {
            float16_t tmp2 = din_ch[i];
            tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
          }
          dout_batch[c] = tmp1;
        }
        LITE_PARALLEL_END()
      }
    } else if (pooling_type == "avg") {
      // Pooling_average_include_padding
      for (int n = 0; n < num; ++n) {
        float16_t *dout_batch = dout + n * chout * size_channel_out;
        const float16_t *din_batch = din + n * chin * size_channel_in;

        LITE_PARALLEL_BEGIN(c, tid, chout) {
          const float16_t *din_ch =
              din_batch + c * size_channel_in;  // in address
          float16_t sum = 0.f;
          for (int i = 0; i < size_channel_in; ++i) {
            sum += din_ch[i];
          }
          dout_batch[c] = sum / size_channel_in;
        }
        LITE_PARALLEL_END()
      }
    } else {
      LOG(FATAL) << "unsupported pooling type: " << pooling_type;
    }
  } else {
    for (int ind_n = 0; ind_n < num; ++ind_n) {
      LITE_PARALLEL_BEGIN(ind_c, tid, chin) {
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
            float16_t result = static_cast<float16_t>(0);
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
      LITE_PARALLEL_END()
    }
  }
}

#ifdef __aarch64__
#define GLOBAL_INIT                         \
  "cmp %w[cnt], #1\n"                       \
  "ldp q0, q1, [%[data_in_channel]], #32\n" \
  "ldp q2, q3, [%[data_in_channel]], #32\n" \
  "prfm  pldl1keep, [%[data_in_channel]]\n" \
  "blt 4f\n"

#define GLOBAL_MAX                          \
  "1:\n"                                    \
  "fmax v4.8h, v0.8h, v2.8h\n"              \
  "fmax v5.8h, v1.8h, v3.8h\n"              \
  "ldp q0, q1, [%[data_in_channel]], #32\n" \
  "fmax %[vmax].8h, %[vmax].8h, v4.8h\n"    \
  "ldp q2, q3, [%[data_in_channel]], #32\n" \
  "subs %w[cnt], %w[cnt], #1 \n"            \
  "fmax %[vmax].8h, %[vmax].8h, v5.8h\n"    \
  "bne 1b\n"

#define GLOBAL_AVG                          \
  "1: \n"                                   \
  "fadd v4.8h, v0.8h, v2.8h\n"              \
  "fadd v5.8h, v1.8h, v3.8h\n"              \
  "ldp q0, q1, [%[data_in_channel]], #32\n" \
  "fadd %[vsum].8h, %[vsum].8h, v4.8h\n"    \
  "ldp q2, q3, [%[data_in_channel]], #32\n" \
  "subs %w[cnt], %w[cnt], #1 \n"            \
  "fadd %[vsum].8h, %[vsum].8h, v5.8h\n"    \
  "bne 1b \n"

#define GLOBAL_MAX_REMAIN                             \
  "4: \n"                                             \
  "cmp %w[remain], #1\n"                              \
  "sub %[data_in_channel], %[data_in_channel], #48\n" \
  "blt 3f\n"                                          \
  "2: \n"                                             \
  "subs %w[remain], %w[remain], #1 \n"                \
  "fmax %[vmax].8h, %[vmax].8h, v0.8h\n"              \
  "ld1 {v0.8h}, [%[data_in_channel]], #16\n"          \
  "bne 2b \n"                                         \
  "3: \n"

#define GLOBAL_AVG_REMAIN                             \
  "4: \n"                                             \
  "cmp %w[remain], #1\n"                              \
  "sub %[data_in_channel], %[data_in_channel], #48\n" \
  "blt 3f\n"                                          \
  "2: \n"                                             \
  "subs %w[remain], %w[remain], #1 \n"                \
  "fadd %[vsum].8h, %[vsum].8h, v0.8h\n"              \
  "ld1 {v0.8h}, [%[data_in_channel]], #16\n"          \
  "bne 2b \n"                                         \
  "3: \n"
#define P3x3S2P1_INIT                  \
  "cmp %w[cnt_num], #1\n"              \
  "ld2 {v0.8h-v1.8h}, [%[dr0]], #32\n" \
  "ld2 {v2.8h-v3.8h}, [%[dr1]], #32\n" \
  "ld2 {v4.8h-v5.8h}, [%[dr2]], #32\n" \
  "blt 0f\n"

#define P3x3S2P0_INIT                  \
  "cmp %w[cnt_num], #1\n"              \
  "ld2 {v0.8h-v1.8h}, [%[dr0]], #32\n" \
  "ld2 {v2.8h-v3.8h}, [%[dr1]], #32\n" \
  "ld2 {v4.8h-v5.8h}, [%[dr2]], #32\n" \
  "ld1 {v6.4h}, [%[dr0]]\n"            \
  "ld1 {v7.4h}, [%[dr1]]\n"            \
  "ld1 {v8.4h}, [%[dr2]]\n"            \
  "blt 0f\n"

#define P3x3S2P1_MAX                       \
  "fmax v9.8h, v0.8h, v1.8h\n"             \
  "fmax v10.8h, v2.8h, v3.8h\n"            \
  "fmax v11.8h, v4.8h, v5.8h\n"            \
  "ext v0.16b, %[vmin].16b, v1.16b, #14\n" \
  "ext v2.16b, %[vmin].16b, v3.16b, #14\n" \
  "ext v4.16b, %[vmin].16b, v5.16b, #14\n" \
  "fmax v6.8h, v9.8h, v0.8h\n"             \
  "fmax v7.8h, v10.8h, v2.8h\n"            \
  "fmax v8.8h, v11.8h, v4.8h\n"            \
  "sub %[dr0], %[dr0], #2\n"               \
  "sub %[dr1], %[dr1], #2\n"               \
  "sub %[dr2], %[dr2], #2\n"               \
  "fmax v9.8h, v6.8h, v7.8h\n"             \
  "ld2 {v0.8h-v1.8h}, [%[dr0]], #32\n"     \
  "ld2 {v2.8h-v3.8h}, [%[dr1]], #32\n"     \
  "fmax v10.8h, v8.8h, v9.8h\n"            \
  "ld2 {v4.8h-v5.8h}, [%[dr2]], #32\n"     \
  "subs %w[cnt_num], %w[cnt_num], #1\n"    \
  "ld1 {v6.4h}, [%[dr0]]\n"                \
  "ld1 {v7.4h}, [%[dr1]]\n"                \
  "ld1 {v8.4h}, [%[dr2]]\n"                \
  "st1  {v10.8h}, [%[dr_out]], #16\n"      \
  "ble 0f\n"

#define P3x3S2P0_MAX                    \
  "2: \n"                               \
  "fmax v9.8h, v0.8h, v1.8h\n"          \
  "fmax v10.8h, v2.8h, v3.8h\n"         \
  "fmax v11.8h, v4.8h, v5.8h\n"         \
  "ext v1.16b, v0.16b, v6.16b, #2\n"    \
  "ext v3.16b, v2.16b, v7.16b, #2\n"    \
  "ext v5.16b, v4.16b, v8.16b, #2\n"    \
  "fmax v6.8h, v9.8h, v1.8h\n"          \
  "fmax v7.8h, v10.8h, v3.8h\n"         \
  "fmax v8.8h, v11.8h, v5.8h\n"         \
  "ld2 {v0.8h-v1.8h}, [%[dr0]], #32\n"  \
  "fmax v9.8h, v6.8h, v7.8h\n"          \
  "ld2 {v2.8h-v3.8h}, [%[dr1]], #32\n"  \
  "ld2 {v4.8h-v5.8h}, [%[dr2]], #32\n"  \
  "ld1 {v6.4h}, [%[dr0]]\n"             \
  "fmax v10.8h, v8.8h, v9.8h\n"         \
  "ld1 {v7.4h}, [%[dr1]]\n"             \
  "subs %w[cnt_num], %w[cnt_num], #1\n" \
  "ld1 {v8.4h}, [%[dr2]]\n"             \
  "st1  {v10.8h}, [%[dr_out]], #16\n"   \
  "bne 2b\n"

#define P3x3S2_REMIN     \
  "0: \n"                \
  "cmp %w[remain], #1\n" \
  "blt 1f\n"

#define P3x3S2P1_MAX_REMAIN                \
  "cmp %w[win_less], #0\n"                 \
  "beq 4f\n"                               \
  "fmax v9.4h, v0.4h, v1.4h\n"             \
  "fmax v10.4h, v2.4h, v3.4h\n"            \
  "fmax v11.4h, v4.4h, v5.4h\n"            \
  "ext v0.16b, %[vmin].16b, v1.16b, #14\n" \
  "ext v2.16b, %[vmin].16b, v3.16b, #14\n" \
  "ext v4.16b, %[vmin].16b, v5.16b, #14\n" \
  "fmax v6.4h, v9.4h, v0.4h\n"             \
  "fmax v7.4h, v10.4h, v2.4h\n"            \
  "fmax v8.4h, v11.4h, v4.4h\n"            \
  "sub %[dr0], %[dr0], #18\n"              \
  "fmax v9.4h, v6.4h, v7.4h\n"             \
  "sub %[dr1], %[dr1], #18\n"              \
  "sub %[dr2], %[dr2], #18\n"              \
  "fmax v10.4h, v8.4h, v9.4h\n"            \
  "st1  {v10.4h}, [%[dr_out]], #8\n"       \
  "b 3f\n"

#define P3x3S2P0_MAX_REMAIN          \
  "4: \n"                            \
  "fmax v9.4h, v0.4h, v1.4h\n"       \
  "fmax v10.4h, v2.4h, v3.4h\n"      \
  "fmax v11.4h, v4.4h, v5.4h\n"      \
  "ext v1.16b, v0.16b, v6.16b, #2\n" \
  "ext v3.16b, v2.16b, v7.16b, #2\n" \
  "ext v5.16b, v4.16b, v8.16b, #2\n" \
  "sub %[dr0], %[dr0], #16\n"        \
  "fmax v6.4h, v9.4h, v1.4h\n"       \
  "fmax v7.4h, v10.4h, v3.4h\n"      \
  "fmax v8.4h, v11.4h, v5.4h\n"      \
  "sub %[dr1], %[dr1], #16\n"        \
  "fmax v9.4h, v6.4h, v7.4h\n"       \
  "sub %[dr2], %[dr2], #16\n"        \
  "fmax v10.4h, v8.4h, v9.4h\n"      \
  "st1  {v10.4h}, [%[dr_out]], #8\n" \
  "b 3f\n"                           \
  "1: \n"                            \
  "sub %[dr0], %[dr0], #32\n"        \
  "sub %[dr1], %[dr1], #32\n"        \
  "sub %[dr2], %[dr2], #32\n"        \
  "3: \n"

#define P3x3S2P1_AVG                        \
  "fadd v9.8h, v0.8h, v1.8h\n"              \
  "fadd v10.8h, v2.8h, v3.8h\n"             \
  "fadd v11.8h, v4.8h, v5.8h\n"             \
  "ext v0.16b, %[vmin].16b, v1.16b, #14\n"  \
  "ext v2.16b, %[vmin].16b, v3.16b, #14\n"  \
  "ext v4.16b, %[vmin].16b, v5.16b, #14\n"  \
  "fadd v6.8h, v9.8h, v0.8h\n"              \
  "fadd v7.8h, v10.8h, v2.8h\n"             \
  "fadd v8.8h, v11.8h, v4.8h\n"             \
  "sub %[dr0], %[dr0], #2\n"                \
  "sub %[dr1], %[dr1], #2\n"                \
  "sub %[dr2], %[dr2], #2\n"                \
  "fadd v9.8h, v6.8h, v7.8h\n"              \
  "ld2 {v0.8h-v1.8h}, [%[dr0]], #32\n"      \
  "ld2 {v2.8h-v3.8h}, [%[dr1]], #32\n"      \
  "ld2 {v4.8h-v5.8h}, [%[dr2]], #32\n"      \
  "fadd v10.8h, v8.8h, v9.8h\n"             \
  "subs %w[cnt_num], %w[cnt_num], #1\n"     \
  "ld1 {v6.4h}, [%[dr0]]\n"                 \
  "ld1 {v7.4h}, [%[dr1]]\n"                 \
  "fmul v10.8h, v10.8h, %[vcoef_left].8h\n" \
  "ld1 {v8.4h}, [%[dr2]]\n"                 \
  "st1  {v10.8h}, [%[dr_out]], #16\n"       \
  "ble 0f\n"

#define P3x3S2P0_AVG                    \
  "2: \n"                               \
  "fadd v9.8h, v0.8h, v1.8h\n"          \
  "fadd v10.8h, v2.8h, v3.8h\n"         \
  "fadd v11.8h, v4.8h, v5.8h\n"         \
  "ext v1.16b, v0.16b, v6.16b, #2\n"    \
  "ext v3.16b, v2.16b, v7.16b, #2\n"    \
  "ext v5.16b, v4.16b, v8.16b, #2\n"    \
  "fadd v6.8h, v9.8h, v1.8h\n"          \
  "fadd v7.8h, v10.8h, v3.8h\n"         \
  "fadd v8.8h, v11.8h, v5.8h\n"         \
  "ld2 {v0.8h-v1.8h}, [%[dr0]], #32\n"  \
  "fadd v9.8h, v6.8h, v7.8h\n"          \
  "ld2 {v2.8h-v3.8h}, [%[dr1]], #32\n"  \
  "ld2 {v4.8h-v5.8h}, [%[dr2]], #32\n"  \
  "ld1 {v6.4h}, [%[dr0]]\n"             \
  "fadd v10.8h, v8.8h, v9.8h\n"         \
  "ld1 {v7.4h}, [%[dr1]]\n"             \
  "ld1 {v8.4h}, [%[dr2]]\n"             \
  "fmul v10.8h, v10.8h, %[vcoef].8h\n"  \
  "subs %w[cnt_num], %w[cnt_num], #1\n" \
  "st1  {v10.8h}, [%[dr_out]], #16\n"   \
  "bne 2b\n"

#define P3x3S2P1_AVG_REMAIN                 \
  "cmp %w[win_less], #0\n"                  \
  "beq 4f\n"                                \
  "fadd v9.4h, v0.4h, v1.4h\n"              \
  "fadd v10.4h, v2.4h, v3.4h\n"             \
  "fadd v11.4h, v4.4h, v5.4h\n"             \
  "ext v0.16b, %[vmin].16b, v1.16b, #14\n"  \
  "ext v2.16b, %[vmin].16b, v3.16b, #14\n"  \
  "ext v4.16b, %[vmin].16b, v5.16b, #14\n"  \
  "fadd v6.4h, v9.4h, v0.4h\n"              \
  "fadd v7.4h, v10.4h, v2.4h\n"             \
  "fadd v8.4h, v11.4h, v4.4h\n"             \
  "sub %[dr0], %[dr0], #18\n"               \
  "fadd v9.4h, v6.4h, v7.4h\n"              \
  "sub %[dr1], %[dr1], #18\n"               \
  "sub %[dr2], %[dr2], #18\n"               \
  "fadd v10.4h, v8.4h, v9.4h\n"             \
  "fmul v10.4h, v10.4h, %[vcoef_left].4h\n" \
  "st1  {v10.4h}, [%[dr_out]], #8\n"        \
  "b 3f\n"

#define P3x3S2P0_AVG_REMAIN            \
  "4: \n"                              \
  "fadd v9.4h, v0.4h, v1.4h\n"         \
  "fadd v10.4h, v2.4h, v3.4h\n"        \
  "fadd v11.4h, v4.4h, v5.4h\n"        \
  "ext v1.16b, v0.16b, v6.16b, #2\n"   \
  "ext v3.16b, v2.16b, v7.16b, #2\n"   \
  "ext v5.16b, v4.16b, v8.16b, #2\n"   \
  "sub %[dr0], %[dr0], #16\n"          \
  "fadd v6.4h, v9.4h, v1.4h\n"         \
  "fadd v7.4h, v10.4h, v3.4h\n"        \
  "fadd v8.4h, v11.4h, v5.4h\n"        \
  "sub %[dr1], %[dr1], #16\n"          \
  "fadd v9.4h, v6.4h, v7.4h\n"         \
  "sub %[dr2], %[dr2], #16\n"          \
  "fadd v10.4h, v8.4h, v9.4h\n"        \
  "fmul v10.4h, v10.4h, %[vcoef].4h\n" \
  "st1  {v10.4h}, [%[dr_out]], #8\n"   \
  "b 3f\n"                             \
  "1: \n"                              \
  "sub %[dr0], %[dr0], #32\n"          \
  "sub %[dr1], %[dr1], #32\n"          \
  "sub %[dr2], %[dr2], #32\n"          \
  "3: \n"
#else
#define GLOBAL_INIT                          \
  "cmp %[cnt], #1\n"                         \
  "vld1.16 {d0-d3}, [%[data_in_channel]]!\n" \
  "vld1.16 {d4-d7}, [%[data_in_channel]]!\n" \
  "blt 4f\n"

#define GLOBAL_MAX                           \
  "1:\n"                                     \
  "vmax.f16 q4, q0, q2\n"                    \
  "vmax.f16 q5, q1, q3\n"                    \
  "vld1.16 {d0-d3}, [%[data_in_channel]]!\n" \
  "subs %[cnt], %[cnt], #1\n"                \
  "vmax.f16 %q[vmax], %q[vmax], q4\n"        \
  "vld1.16 {d4-d7}, [%[data_in_channel]]!\n" \
  "vmax.f16 %q[vmax], %q[vmax], q5\n"        \
  "bne 1b\n"

#define GLOBAL_MAX_REMAIN                             \
  "4: \n"                                             \
  "cmp %[remain], #1\n"                               \
  "sub %[data_in_channel], %[data_in_channel], #48\n" \
  "blt 3f\n"                                          \
  "2: \n"                                             \
  "subs %[remain], %[remain], #1 \n"                  \
  "vmax.f16 %q[vmax], %q[vmax], q0\n"                 \
  "vld1.16 {d0, d1}, [%[data_in_channel]]!\n"         \
  "bne 2b \n"                                         \
  "3: \n"

#define GLOBAL_AVG                           \
  "1: \n"                                    \
  "vadd.f16 q4, q0, q2\n"                    \
  "vadd.f16 q5, q1, q3\n"                    \
  "vld1.16 {d0-d3}, [%[data_in_channel]]!\n" \
  "vadd.f16 %q[vsum], %q[vsum], q4\n"        \
  "vld1.16 {d4-d7}, [%[data_in_channel]]!\n" \
  "vadd.f16 %q[vsum], %q[vsum], q5\n"        \
  "subs %[cnt], %[cnt], #1\n"                \
  "bne 1b\n"

#define GLOBAL_AVG_REMAIN                             \
  "4: \n"                                             \
  "cmp %[remain], #1\n"                               \
  "sub %[data_in_channel], %[data_in_channel], #48\n" \
  "blt 3f\n"                                          \
  "2:\n"                                              \
  "subs %[remain], %[remain], #1\n"                   \
  "vadd.f16 %q[vsum], %q[vsum], q0\n"                 \
  "vld1.16 {d0, d1}, [%[data_in_channel]]!\n"         \
  "bne 2b \n"                                         \
  "3: \n"

#define P3x3S2P0_INIT              \
  "cmp %[cnt_num], #1\n"           \
  "vld2.16 {d0-d3}, [%[dr0]]!\n"   \
  "vld2.16 {d4-d7}, [%[dr1]]!\n"   \
  "vld2.16 {d8-d11}, [%[dr2]]!\n"  \
  "vld1.16 {d12-d15}, [%[dr0]]\n"  \
  "vld1.16 {d16, d17}, [%[dr2]]\n" \
  "blt 0f\n"

#define P3x3S2P0_MAX                   \
  "2: \n"                              \
  "vmax.f16 q9, q0, q1\n"              \
  "vmax.f16 q10, q2, q3\n"             \
  "vmax.f16 q11, q4, q5\n"             \
  "vext.8 q1, q0, q6, #2\n"            \
  "vext.8 q3, q2, q7, #2\n"            \
  "vext.8 q5, q4, q8, #2\n"            \
  "vmax.f16 q6, q9, q1\n"              \
  "vmax.f16 q7, q10, q3\n"             \
  "vmax.f16 q8, q11, q5\n"             \
  "vld2.16 {d0-d3}, [%[dr0]]!\n"       \
  "vmax.f16 q9, q6, q7\n"              \
  "vld2.16 {d4-d7}, [%[dr1]]!\n"       \
  "vld2.16 {d8-d11}, [%[dr2]]!\n"      \
  "vld1.16 {d12, d13}, [%[dr0]]\n"     \
  "vmax.f16 q10, q8, q9\n"             \
  "vld1.16 {d14, d15}, [%[dr1]]\n"     \
  "subs %[cnt_num], %[cnt_num], #1\n"  \
  "vld1.16 {d16, d17}, [%[dr2]]\n"     \
  "vst1.16 {d20, d21}, [%[dr_out]]!\n" \
  "bne 2b\n"

#define P3x3S2P0_MAX_REMAIN       \
  "4: \n"                         \
  "vmax.f16 q9, q0, q1\n"         \
  "vmax.f16 q10, q2, q3\n"        \
  "vmax.f16 q11, q4, q5\n"        \
  "vext.8 q1, q0, q6, #2\n"       \
  "vext.8 q3, q2, q7, #2\n"       \
  "vext.8 q5, q4, q8, #2\n"       \
  "sub %[dr0], %[dr0], #16\n"     \
  "vmax.f16 q6, q9, q1\n"         \
  "vmax.f16 q7, q10, q3\n"        \
  "vmax.f16 q8, q11, q5\n"        \
  "sub %[dr1], %[dr1], #16\n"     \
  "vmax.f16 q9, q6, q7\n"         \
  "sub %[dr2], %[dr2], #16\n"     \
  "vmax.f16 q10, q8, q9\n"        \
  "vst1.16 {d20}, [%[dr_out]]!\n" \
  "b 3f\n"                        \
  "1: \n"                         \
  "sub %[dr0], %[dr0], #32\n"     \
  "sub %[dr1], %[dr1], #32\n"     \
  "sub %[dr2], %[dr2], #32\n"     \
  "3: \n"

#define P3x3S2_REMIN    \
  "0: \n"               \
  "cmp %[remain], #1\n" \
  "blt 1f\n"

#define P3x3S2P0_AVG                    \
  "2: \n"                               \
  "vadd.f16 q9, q0, q1\n"               \
  "vadd.f16 q10, q2, q3\n"              \
  "vadd.f16 q11, q4, q5\n"              \
  "vext.8 q1, q0, q6, #2\n"             \
  "vext.8 q3, q2, q7, #2\n"             \
  "vext.8 q5, q4, q8, #2\n"             \
  "vadd.f16 q6, q9, q1\n"               \
  "vadd.f16 q7, q10, q3\n"              \
  "vadd.f16 q8, q11, q5\n"              \
  "vld2.16 {d0-d3}, [%[dr0]]!\n"        \
  "vadd.f16 q9, q6, q7\n"               \
  "vld2.16 {d4-d7}, [%[dr1]]!\n"        \
  "vld2.16 {d8-d11}, [%[dr2]]!\n"       \
  "vld1.16 {d12, d13}, [%[dr0]]\n"      \
  "vadd.f16 q10, q8, q9\n"              \
  "vld1.16 {d14-d17}, [%[dr1]]\n"       \
  "vmul.f16 q10, q10, %q[vcoef]\n"      \
  "subs %[cnt_num], %[cnt_num], #1\n"   \
  "vst1.16  {d20, d21}, [%[dr_out]]!\n" \
  "bne 2b\n"

#define P3x3S2P0_AVG_REMAIN        \
  "4: \n"                          \
  "vadd.f16 q9, q0, q1\n"          \
  "vadd.f16 q10, q2, q3\n"         \
  "vadd.f16 q11, q4, q5\n"         \
  "vext.8 q1, q0, q6, #2\n"        \
  "vext.8 q3, q2, q7, #2\n"        \
  "vext.8 q5, q4, q8, #2\n"        \
  "sub %[dr0], %[dr0], #16\n"      \
  "vadd.f16 q6, q9, q1\n"          \
  "vadd.f16 q7, q10, q3\n"         \
  "vadd.f16 q8, q11, q5\n"         \
  "sub %[dr1], %[dr1], #16\n"      \
  "vadd.f16 q9, q6, q7\n"          \
  "sub %[dr2], %[dr2], #16\n"      \
  "vadd.f16 q10, q8, q9\n"         \
  "vmul.f16 q10, q10, %q[vcoef]\n" \
  "vst1.16  {d20}, [%[dr_out]]!\n" \
  "b 3f\n"                         \
  "1: \n"                          \
  "sub %[dr0], %[dr0], #32\n"      \
  "sub %[dr1], %[dr1], #32\n"      \
  "sub %[dr2], %[dr2], #32\n"      \
  "3: \n"

#define P3x3S2P1_INIT             \
  "cmp %[cnt_num], #1\n"          \
  "vld2.16 {d0-d3}, [%[dr0]]!\n"  \
  "vld2.16 {d4-d7}, [%[dr1]]!\n"  \
  "vld2.16 {d8-d11}, [%[dr2]]!\n" \
  "blt 0f\n"

#define P3x3S2P1_MAX                    \
  "vmax.f16 q9 , q0, q1\n"              \
  "vmax.f16 q10, q2, q3\n"              \
  "vmax.f16 q11, q4, q5\n"              \
  "vext.8 q0, %q[vmin], q1, #14\n"      \
  "vext.8 q2, %q[vmin], q3, #14\n"      \
  "vext.8 q4, %q[vmin], q5, #14\n"      \
  "vmax.f16 q6, q9,  q0\n"              \
  "vmax.f16 q7, q10, q2\n"              \
  "vmax.f16 q8, q11, q4\n"              \
  "sub %[dr0], %[dr0], #2\n"            \
  "sub %[dr1], %[dr1], #2\n"            \
  "sub %[dr2], %[dr2], #2\n"            \
  "vmax.f16 q9, q6, q7\n"               \
  "vld2.16 {d0-d3}, [%[dr0]]!\n"        \
  "vld2.16 {d4-d7}, [%[dr1]]!\n"        \
  "vmax.f16 q10, q8, q9\n"              \
  "vld2.16 {d8-d11}, [%[dr2]]!\n"       \
  "subs %[cnt_num], %[cnt_num], #1\n"   \
  "vld1.16 {d12-d15}, [%[dr0]]\n"       \
  "vld1.16 {d16, d17}, [%[dr2]]\n"      \
  "vst1.16  {d20, d21}, [%[dr_out]]!\n" \
  "ble 0f\n"

#define P3x3S2P1_AVG                    \
  "vadd.f16 q9 , q0, q1\n"              \
  "vadd.f16 q10, q2, q3\n"              \
  "vadd.f16 q11, q4, q5\n"              \
  "vext.8 q0, %q[vmin], q1, #14\n"      \
  "vext.8 q2, %q[vmin], q3, #14\n"      \
  "vext.8 q4, %q[vmin], q5, #14\n"      \
  "vadd.f16 q6, q9 , q0\n"              \
  "vadd.f16 q7, q10, q2\n"              \
  "vadd.f16 q8, q11, q4\n"              \
  "sub %[dr0], %[dr0], #2\n"            \
  "sub %[dr1], %[dr1], #2\n"            \
  "sub %[dr2], %[dr2], #2\n"            \
  "vadd.f16 q9, q6, q7\n"               \
  "vld2.16 {d0-d3}, [%[dr0]]!\n"        \
  "vld2.16 {d4-d7}, [%[dr1]]!\n"        \
  "vld2.16 {d8-d11}, [%[dr2]]!\n"       \
  "vadd.f16 q10, q8, q9\n"              \
  "subs %[cnt_num], %[cnt_num], #1\n"   \
  "vld1.16 {d12-d15}, [%[dr0]]\n"       \
  "vmul.f16 q10, q10, %q[vcoef_left]\n" \
  "vld1.16 {d16, d17}, [%[dr2]]\n"      \
  "vst1.16  {d20, d21}, [%[dr_out]]!\n" \
  "ble 0f\n"

#define P3x3S2P1_AVG_REMAIN             \
  "cmp %[win_less], #0\n"               \
  "beq 4f\n"                            \
  "vadd.f16 q9 , q0, q1\n"              \
  "vadd.f16 q10, q2, q3\n"              \
  "vadd.f16 q11, q4, q5\n"              \
  "vext.8 q0, %q[vmin], q1, #14\n"      \
  "vext.8 q2, %q[vmin], q3, #14\n"      \
  "vext.8 q4, %q[vmin], q5, #14\n"      \
  "vadd.f16 q6, q9 , q0\n"              \
  "vadd.f16 q7, q10, q2\n"              \
  "vadd.f16 q8, q11, q4\n"              \
  "sub %[dr0], %[dr0], #18\n"           \
  "vadd.f16 q9, q6, q7\n"               \
  "sub %[dr1], %[dr1], #18\n"           \
  "sub %[dr2], %[dr2], #18\n"           \
  "vadd.f16 q10, q8, q9\n"              \
  "vmul.f16 q10, q10, %q[vcoef_left]\n" \
  "vst1.16  {d20}, [%[dr_out]]!\n"      \
  "b 3f\n"

#define P3x3S2P1_MAX_REMAIN        \
  "cmp %[win_less], #0\n"          \
  "beq 4f\n"                       \
  "vmax.f16 q9 , q0, q1\n"         \
  "vmax.f16 q10, q2, q3\n"         \
  "vmax.f16 q11, q4, q5\n"         \
  "vext.8 q0, %q[vmin], q1, #14\n" \
  "vext.8 q2, %q[vmin], q3, #14\n" \
  "vext.8 q4, %q[vmin], q5, #14\n" \
  "vmax.f16 q6, q9 , q0\n"         \
  "vmax.f16 q7, q10, q2\n"         \
  "vmax.f16 q8, q11, q4\n"         \
  "sub %[dr0], %[dr0], #18\n"      \
  "vmax.f16 q9, q6, q7\n"          \
  "sub %[dr1], %[dr1], #18\n"      \
  "sub %[dr2], %[dr2], #18\n"      \
  "vmax.f16 q10, q8, q9\n"         \
  "vst1.16  {d20}, [%[dr_out]]!\n" \
  "b 3f\n"
#endif

#define POOL_CNT_COMPUTE                                    \
  int size_channel_out = wout * hout;                       \
  int size_channel_in = win * hin;                          \
  int w_unroll_size = wout >> 3;                            \
  int w_unroll_remian = wout & 7;                           \
  /* wout = (win + P + pad_right - K) / S + 1*/             \
  int right_remain = (wout * S - S + K - P) - win;          \
  if (w_unroll_remian == 0 && right_remain > 0) {           \
    w_unroll_size -= 1;                                     \
    w_unroll_remian = 8;                                    \
  }                                                         \
  int cnt = w_unroll_remian >> 2;                           \
  int cnt_remain = w_unroll_remian & 3;                     \
  if (cnt_remain == 0 && right_remain > 0) {                \
    cnt -= 1;                                               \
    cnt_remain = 4;                                         \
  }                                                         \
  int remain_num = win - (w_unroll_size * 8 * S - P);       \
  /* cnt = 1, right > 8; cnt = 2, right > 16*/              \
  int right = (remain_num - (cnt * 4 * S - P)) > 0 ? 1 : 0; \
  int wend = std::min((wout - 1) * S - P + K, win) - ((wout - 1) * S - P);

#define MAX_ONE_COMPUTE(                                                   \
    dr0, dr1, dr2, dr_out, cnt_remain, minval, right_remain, wend, stride) \
  for (int w = 0; w < cnt_remain; w += 1) {                                \
    float16x4_t vr0 = vld1_f16(dr0);                                       \
    float16x4_t vr1 = vld1_f16(dr1);                                       \
    float16x4_t vr2 = vld1_f16(dr2);                                       \
    vr0 = vset_lane_f16(minval, vr0, 3);                                   \
    vr1 = vset_lane_f16(minval, vr1, 3);                                   \
    vr2 = vset_lane_f16(minval, vr2, 3);                                   \
    float16x4_t vmax1 = vmax_f16(vr0, vr1);                                \
    vmax1 = vmax_f16(vmax1, vr2);                                          \
    float16x4_t vmax2 = vpmax_f16(vmax1, vmax1);                           \
    float16x4_t vmax = vpmax_f16(vmax2, vmax2);                            \
    dr_out[0] = vget_lane_f16(vmax, 0);                                    \
    dr0 += stride;                                                         \
    dr1 += stride;                                                         \
    dr2 += stride;                                                         \
    dr_out++;                                                              \
  }                                                                        \
  if (right_remain > 0) {                                                  \
    float16_t tmp = dr0[0];                                                \
    for (int i = 0; i < wend; i++) {                                       \
      tmp = std::max(tmp, std::max(dr0[i], dr1[i]));                       \
      tmp = std::max(tmp, dr2[i]);                                         \
    }                                                                      \
    *(dr_out++) = tmp;                                                     \
  }

#define AVG_ONE_COMPUTE(                                                   \
    dr0, dr1, dr2, dr_out, cnt_remain, minval, right_remain, wend, stride) \
  for (int w = 0; w < cnt_remain; w += 1) {                                \
    float16x4_t vr0 = vld1_f16(dr0);                                       \
    float16x4_t vr1 = vld1_f16(dr1);                                       \
    float16x4_t vr2 = vld1_f16(dr2);                                       \
    vr0 = vset_lane_f16(0.f, vr0, 3);                                      \
    vr1 = vset_lane_f16(0.f, vr1, 3);                                      \
    vr2 = vset_lane_f16(0.f, vr2, 3);                                      \
    float16x4_t vsum1 = vadd_f16(vr0, vr1);                                \
    vsum1 = vadd_f16(vsum1, vr2);                                          \
    float16x4_t vsum2 = vpadd_f16(vsum1, vsum1);                           \
    float16x4_t vsum = vpadd_f16(vsum2, vsum2);                            \
    dr_out[0] = vget_lane_f16(vsum, 0) * vcoef[0];                         \
    dr0 += stride;                                                         \
    dr1 += stride;                                                         \
    dr2 += stride;                                                         \
    dr_out++;                                                              \
  }                                                                        \
  if (right_remain > 0) {                                                  \
    float16_t sum = 0.f;                                                   \
    for (int i = 0; i < wend; i++) {                                       \
      sum += dr0[i] + dr1[i] + dr2[i];                                     \
    }                                                                      \
    sum *= vcoef[0] * 3.f;                                                 \
    if (exclusive) {                                                       \
      sum /= wend;                                                         \
    } else {                                                               \
      sum /= (wend + pad_right);                                           \
    }                                                                      \
    *(dr_out++) = sum;                                                     \
  }

#define P3x3S2_MAX_PTR_CHOOSE(dr0, dr1, dr2, S, K, P, h, hin) \
  if (h * S + K - P > hin) {                                  \
    switch (h * S + K - P - hin) {                            \
      case 2:                                                 \
        dr1 = dr0;                                            \
      case 1:                                                 \
        dr2 = dr0;                                            \
      default:                                                \
        break;                                                \
    }                                                         \
  }

#define P3x3S1_MAX_PTR_CHOOSE(dr0, dr1, dr2, S, K, P, h, hin) \
  if (h * S + K - P > hin) {                                  \
    if (h * S + K - P - hin == 1) {                           \
      dr2 = dr0;                                              \
    }                                                         \
    if (h * S + K - P - hin == 2) {                           \
      dr1 = dr0;                                              \
    }                                                         \
  }

#define P3x3s2_AVG_PTR_CHOOSE(                                          \
    dr1, dr2, zero_ptr, S, K, P, h, hin, coef_h, pad_bottom, exclusive) \
  if (h * S + K - P > hin) {                                            \
    switch (h * S + K - P - hin) {                                      \
      case 2:                                                           \
        dr1 = zero_ptr;                                                 \
        dr2 = zero_ptr;                                                 \
        if (exclusive) {                                                \
          coef_h = 1.f;                                                 \
        } else {                                                        \
          if (pad_bottom > 1) {                                         \
            coef_h = 1.f / 3;                                           \
          } else if (pad_bottom == 1) {                                 \
            coef_h = 0.5f;                                              \
          } else {                                                      \
            coef_h = 1.f;                                               \
          }                                                             \
        }                                                               \
        break;                                                          \
      case 1:                                                           \
        dr2 = zero_ptr;                                                 \
        if (exclusive) {                                                \
          if (fabsf(coef_h - 0.5f) < 1e-6f) {                           \
            coef_h = 1.f;                                               \
          } else {                                                      \
            coef_h = 0.5f;                                              \
          }                                                             \
        } else {                                                        \
          if (pad_bottom == 0) {                                        \
            coef_h = 1.f / 2;                                           \
          } else {                                                      \
            coef_h = 1.f / 3;                                           \
          }                                                             \
        }                                                               \
      default:                                                          \
        break;                                                          \
    }                                                                   \
  }

#define P3x3S1P0_INIT_INTRIN                    \
  float16x8_t dr0_first8 = vld1q_f16(dr0);      \
  float16x8_t dr0_second8 = vld1q_f16(dr0 + 8); \
  float16x8_t dr1_first8 = vld1q_f16(dr1);      \
  float16x8_t dr1_second8 = vld1q_f16(dr1 + 8); \
  float16x8_t dr2_first8 = vld1q_f16(dr2);      \
  float16x8_t dr2_second8 = vld1q_f16(dr2 + 8);

#define P3x3S1P1_WINLESS_INTRIN                                               \
  float16x4_t vmin_4 = vget_low_f16(vmin);                                    \
  float16x4_t dr0_first4 = vld1_f16(dr0);                                     \
  float16x4_t dr0_second4 = vld1_f16(dr0 + 4);                                \
                                                                              \
  float16x4_t dr1_first4 = vld1_f16(dr1);                                     \
  float16x4_t dr1_second4 = vld1_f16(dr1 + 4);                                \
                                                                              \
  float16x4_t dr2_first4 = vld1_f16(dr2);                                     \
  float16x4_t dr2_second4 = vld1_f16(dr2 + 4);                                \
                                                                              \
  const float16x4_t dr0_first4_offset = vext_f16(dr0_first4, dr0_second4, 1); \
  const float16x4_t dr1_first4_offset = vext_f16(dr1_first4, dr1_second4, 1); \
  const float16x4_t dr2_first4_offset = vext_f16(dr2_first4, dr2_second4, 1); \
                                                                              \
  const float16x4_t dr0_pad_4 = vext_f16(vmin_4, dr0_first4, 3);              \
  const float16x4_t dr1_pad_4 = vext_f16(vmin_4, dr1_first4, 3);              \
  const float16x4_t dr2_pad_4 = vext_f16(vmin_4, dr2_first4, 3);              \
                                                                              \
  float16x4_t dr0_max4 = vmax_f16(dr0_first4, dr0_first4_offset);             \
  dr0_max4 = vmax_f16(dr0_max4, dr0_pad_4);                                   \
                                                                              \
  float16x4_t dr1_max4 = vmax_f16(dr1_first4, dr1_first4_offset);             \
  dr1_max4 = vmax_f16(dr1_max4, dr1_pad_4);                                   \
                                                                              \
  float16x4_t dr2_max4 = vmax_f16(dr2_first4, dr2_first4_offset);             \
  dr2_max4 = vmax_f16(dr2_max4, dr2_pad_4);                                   \
                                                                              \
  float16x4_t pad_left_max4 = vmax_f16(dr0_max4, dr1_max4);                   \
  pad_left_max4 = vmax_f16(pad_left_max4, dr2_max4);                          \
  vst1_f16(dr_out, pad_left_max4);                                            \
                                                                              \
  dr0 += 3;                                                                   \
  dr1 += 3;                                                                   \
  dr2 += 3;                                                                   \
  dr_out += 4;                                                                \
  cnt_remain_4--;

#define P3x3S1P1_AVG_WINLESS_INTRIN                                           \
  float16x4_t dr0_first4 = vld1_f16(dr0);                                     \
  float16x4_t dr0_second4 = vld1_f16(dr0 + 4);                                \
                                                                              \
  float16x4_t dr1_first4 = vld1_f16(dr1);                                     \
  float16x4_t dr1_second4 = vld1_f16(dr1 + 4);                                \
                                                                              \
  float16x4_t dr2_first4 = vld1_f16(dr2);                                     \
  float16x4_t dr2_second4 = vld1_f16(dr2 + 4);                                \
                                                                              \
  const float16x4_t dr0_first4_offset = vext_f16(dr0_first4, dr0_second4, 1); \
  const float16x4_t dr1_first4_offset = vext_f16(dr1_first4, dr1_second4, 1); \
  const float16x4_t dr2_first4_offset = vext_f16(dr2_first4, dr2_second4, 1); \
                                                                              \
  const float16x4_t dr0_pad_4 = vext_f16(vmin_4, dr0_first4, 3);              \
  const float16x4_t dr1_pad_4 = vext_f16(vmin_4, dr1_first4, 3);              \
  const float16x4_t dr2_pad_4 = vext_f16(vmin_4, dr2_first4, 3);              \
                                                                              \
  float16x4_t dr0_sum4 = vadd_f16(dr0_first4, dr0_first4_offset);             \
  dr0_sum4 = vadd_f16(dr0_sum4, dr0_pad_4);                                   \
                                                                              \
  float16x4_t dr1_sum4 = vadd_f16(dr1_first4, dr1_first4_offset);             \
  dr1_sum4 = vadd_f16(dr1_sum4, dr1_pad_4);                                   \
                                                                              \
  float16x4_t dr2_sum4 = vadd_f16(dr2_first4, dr2_first4_offset);             \
  dr2_sum4 = vadd_f16(dr2_sum4, dr2_pad_4);                                   \
                                                                              \
  float16x4_t pad_left_sum4 = vadd_f16(dr0_sum4, dr1_sum4);                   \
  pad_left_sum4 = vadd_f16(pad_left_sum4, dr2_sum4);                          \
  float16x4_t pad_left_avg4 = vmul_f16(pad_left_sum4, vcoef_left4);           \
  vst1_f16(dr_out, pad_left_avg4);                                            \
                                                                              \
  dr0 += 3;                                                                   \
  dr1 += 3;                                                                   \
  dr2 += 3;                                                                   \
  dr_out += 4;                                                                \
  cnt_remain_4--;

#define P3x3S1P1_INIT_INTRIN                                                   \
  float16x8_t dr0_first8 = vld1q_f16(dr0);                                     \
  float16x8_t dr0_second8 = vld1q_f16(dr0 + 8);                                \
                                                                               \
  float16x8_t dr1_first8 = vld1q_f16(dr1);                                     \
  float16x8_t dr1_second8 = vld1q_f16(dr1 + 8);                                \
                                                                               \
  float16x8_t dr2_first8 = vld1q_f16(dr2);                                     \
  float16x8_t dr2_second8 = vld1q_f16(dr2 + 8);                                \
                                                                               \
  const float16x8_t dr0_first8_offset = vextq_f16(dr0_first8, dr0_second8, 1); \
  const float16x8_t dr1_first8_offset = vextq_f16(dr1_first8, dr1_second8, 1); \
  const float16x8_t dr2_first8_offset = vextq_f16(dr2_first8, dr2_second8, 1); \
                                                                               \
  const float16x8_t dr0_pad_8 = vextq_f16(vmin, dr0_first8, 7);                \
  const float16x8_t dr1_pad_8 = vextq_f16(vmin, dr1_first8, 7);                \
  const float16x8_t dr2_pad_8 = vextq_f16(vmin, dr2_first8, 7);                \
                                                                               \
  float16x8_t dr0_max8 = vmaxq_f16(dr0_first8, dr0_first8_offset);             \
  dr0_max8 = vmaxq_f16(dr0_max8, dr0_pad_8);                                   \
                                                                               \
  float16x8_t dr1_max8 = vmaxq_f16(dr1_first8, dr1_first8_offset);             \
  dr1_max8 = vmaxq_f16(dr1_max8, dr1_pad_8);                                   \
                                                                               \
  float16x8_t dr2_max8 = vmaxq_f16(dr2_first8, dr2_first8_offset);             \
  dr2_max8 = vmaxq_f16(dr2_max8, dr2_pad_8);                                   \
                                                                               \
  float16x8_t pad_left_max8 = vmaxq_f16(dr0_max8, dr1_max8);                   \
  pad_left_max8 = vmaxq_f16(pad_left_max8, dr2_max8);                          \
  vst1q_f16(dr_out, pad_left_max8);                                            \
  dr0_first8 = vextq_f16(dr0_first8, dr0_second8, 7);                          \
  dr1_first8 = vextq_f16(dr1_first8, dr1_second8, 7);                          \
  dr2_first8 = vextq_f16(dr2_first8, dr2_second8, 7);                          \
                                                                               \
  dr0 += 7;                                                                    \
  dr1 += 7;                                                                    \
  dr2 += 7;                                                                    \
  dr0_second8 = vld1q_f16(dr0 + 8);                                            \
  dr1_second8 = vld1q_f16(dr1 + 8);                                            \
  dr2_second8 = vld1q_f16(dr2 + 8);                                            \
                                                                               \
  dr_out += 8;                                                                 \
  cnt_num--;

#define P3x3S1P1_AVG_INIT_INTRIN                                               \
  float16x8_t dr0_first8 = vld1q_f16(dr0);                                     \
  float16x8_t dr0_second8 = vld1q_f16(dr0 + 8);                                \
                                                                               \
  float16x8_t dr1_first8 = vld1q_f16(dr1);                                     \
  float16x8_t dr1_second8 = vld1q_f16(dr1 + 8);                                \
                                                                               \
  float16x8_t dr2_first8 = vld1q_f16(dr2);                                     \
  float16x8_t dr2_second8 = vld1q_f16(dr2 + 8);                                \
                                                                               \
  const float16x8_t dr0_first8_offset = vextq_f16(dr0_first8, dr0_second8, 1); \
  const float16x8_t dr1_first8_offset = vextq_f16(dr1_first8, dr1_second8, 1); \
  const float16x8_t dr2_first8_offset = vextq_f16(dr2_first8, dr2_second8, 1); \
                                                                               \
  const float16x8_t dr0_pad_8 = vextq_f16(vmin, dr0_first8, 7);                \
  const float16x8_t dr1_pad_8 = vextq_f16(vmin, dr1_first8, 7);                \
  const float16x8_t dr2_pad_8 = vextq_f16(vmin, dr2_first8, 7);                \
                                                                               \
  float16x8_t dr0_sum8 = vaddq_f16(dr0_first8, dr0_first8_offset);             \
  dr0_sum8 = vaddq_f16(dr0_sum8, dr0_pad_8);                                   \
                                                                               \
  float16x8_t dr1_sum8 = vaddq_f16(dr1_first8, dr1_first8_offset);             \
  dr1_sum8 = vaddq_f16(dr1_sum8, dr1_pad_8);                                   \
                                                                               \
  float16x8_t dr2_sum8 = vaddq_f16(dr2_first8, dr2_first8_offset);             \
  dr2_sum8 = vaddq_f16(dr2_sum8, dr2_pad_8);                                   \
                                                                               \
  float16x8_t pad_left_sum8 = vaddq_f16(dr0_sum8, dr1_sum8);                   \
  pad_left_sum8 = vaddq_f16(pad_left_sum8, dr2_sum8);                          \
  float16x8_t pad_left_avg8 = vmulq_f16(pad_left_sum8, vcoef_left);            \
  vst1q_f16(dr_out, pad_left_avg8);                                            \
  dr0_first8 = vextq_f16(dr0_first8, dr0_second8, 7);                          \
  dr1_first8 = vextq_f16(dr1_first8, dr1_second8, 7);                          \
  dr2_first8 = vextq_f16(dr2_first8, dr2_second8, 7);                          \
                                                                               \
  dr0 += 7;                                                                    \
  dr1 += 7;                                                                    \
  dr2 += 7;                                                                    \
  dr0_second8 = vld1q_f16(dr0 + 8);                                            \
  dr1_second8 = vld1q_f16(dr1 + 8);                                            \
  dr2_second8 = vld1q_f16(dr2 + 8);                                            \
                                                                               \
  dr_out += 8;                                                                 \
  cnt_num--;

#define P3x3S1P0_MAX_8TIMES_INTRIN                                           \
  while (cnt_num > 0) {                                                      \
    const float16x8_t dr0_first8_offset1 =                                   \
        vextq_f16(dr0_first8, dr0_second8, 1);                               \
    const float16x8_t dr0_first8_offset2 =                                   \
        vextq_f16(dr0_first8, dr0_second8, 2);                               \
    float16x8_t dr0_max_first8 = vmaxq_f16(dr0_first8, dr0_first8_offset1);  \
    const float16x8_t dr1_first8_offset1 =                                   \
        vextq_f16(dr1_first8, dr1_second8, 1);                               \
    const float16x8_t dr1_first8_offset2 =                                   \
        vextq_f16(dr1_first8, dr1_second8, 2);                               \
    dr0_max_first8 = vmaxq_f16(dr0_max_first8, dr0_first8_offset2);          \
    float16x8_t dr1_max_first8 = vmaxq_f16(dr1_first8, dr1_first8_offset1);  \
    const float16x8_t dr2_first8_offset1 =                                   \
        vextq_f16(dr2_first8, dr2_second8, 1);                               \
    const float16x8_t dr2_first8_offset2 =                                   \
        vextq_f16(dr2_first8, dr2_second8, 2);                               \
    dr1_max_first8 = vmaxq_f16(dr1_max_first8, dr1_first8_offset2);          \
    float16x8_t dr2_max_first8 = vmaxq_f16(dr2_first8, dr2_first8_offset1);  \
                                                                             \
    float16x8_t col_0_3_row_0_8 = vmaxq_f16(dr0_max_first8, dr1_max_first8); \
    dr2_max_first8 = vmaxq_f16(dr2_max_first8, dr2_first8_offset2);          \
    col_0_3_row_0_8 = vmaxq_f16(col_0_3_row_0_8, dr2_max_first8);            \
    vst1q_f16(dr_out, col_0_3_row_0_8);                                      \
    dr_out += 8;                                                             \
                                                                             \
    dr0_first8 = dr0_second8;                                                \
    dr1_first8 = dr1_second8;                                                \
    dr2_first8 = dr2_second8;                                                \
                                                                             \
    dr0 += 8;                                                                \
    dr1 += 8;                                                                \
    dr2 += 8;                                                                \
    dr0_second8 = vld1q_f16(dr0 + 8);                                        \
    dr1_second8 = vld1q_f16(dr1 + 8);                                        \
    dr2_second8 = vld1q_f16(dr2 + 8);                                        \
    cnt_num--;                                                               \
  }

#define P3x3S1P0_MAX_4TIMES_INTRIN                                         \
  if (cnt_remain_4 > 0) {                                                  \
    const float16x4_t dr0_first4 = vget_low_f16(dr0_first8);               \
    const float16x4_t dr1_first4 = vget_low_f16(dr1_first8);               \
    const float16x4_t dr2_first4 = vget_low_f16(dr2_first8);               \
    const float16x4_t dr0_second4 = vget_high_f16(dr0_first8);             \
    const float16x4_t dr1_second4 = vget_high_f16(dr1_first8);             \
    const float16x4_t dr2_second4 = vget_high_f16(dr2_first8);             \
                                                                           \
    const float16x4_t dr0_first4_offset1 =                                 \
        vext_f16(dr0_first4, dr0_second4, 1);                              \
    const float16x4_t dr0_first4_offset2 =                                 \
        vext_f16(dr0_first4, dr0_second4, 2);                              \
    float16x4_t dr0_max_first4 = vmax_f16(dr0_first4, dr0_first4_offset1); \
    const float16x4_t dr1_first4_offset1 =                                 \
        vext_f16(dr1_first4, dr1_second4, 1);                              \
    const float16x4_t dr1_first4_offset2 =                                 \
        vext_f16(dr1_first4, dr1_second4, 2);                              \
    dr0_max_first4 = vmax_f16(dr0_max_first4, dr0_first4_offset2);         \
                                                                           \
    float16x4_t dr1_max_first4 = vmax_f16(dr1_first4, dr1_first4_offset1); \
    const float16x4_t dr2_first4_offset1 =                                 \
        vext_f16(dr2_first4, dr2_second4, 1);                              \
    const float16x4_t dr2_first4_offset2 =                                 \
        vext_f16(dr2_first4, dr2_second4, 2);                              \
    dr1_max_first4 = vmax_f16(dr1_max_first4, dr1_first4_offset2);         \
                                                                           \
    float16x4_t dr2_max_first4 = vmax_f16(dr2_first4, dr2_first4_offset1); \
    float16x4_t pool_row_result_0_4 =                                      \
        vmax_f16(dr0_max_first4, dr1_max_first4);                          \
    dr2_max_first4 = vmax_f16(dr2_max_first4, dr2_first4_offset2);         \
    pool_row_result_0_4 = vmax_f16(pool_row_result_0_4, dr2_max_first4);   \
    vst1_f16(dr_out, pool_row_result_0_4);                                 \
    dr0 += 4;                                                              \
    dr1 += 4;                                                              \
    dr2 += 4;                                                              \
    dr_out += 4;                                                           \
  }

#define P3x3S1P0_AVG_8TIMES_INTRIN                                           \
  while (cnt_num > 0) {                                                      \
    const float16x8_t dr0_first8_offset1 =                                   \
        vextq_f16(dr0_first8, dr0_second8, 1);                               \
    const float16x8_t dr0_first8_offset2 =                                   \
        vextq_f16(dr0_first8, dr0_second8, 2);                               \
    float16x8_t dr0_sum_first8 = vaddq_f16(dr0_first8, dr0_first8_offset1);  \
    const float16x8_t dr1_first8_offset1 =                                   \
        vextq_f16(dr1_first8, dr1_second8, 1);                               \
    const float16x8_t dr1_first8_offset2 =                                   \
        vextq_f16(dr1_first8, dr1_second8, 2);                               \
    dr0_sum_first8 = vaddq_f16(dr0_sum_first8, dr0_first8_offset2);          \
    float16x8_t dr1_sum_first8 = vaddq_f16(dr1_first8, dr1_first8_offset1);  \
    const float16x8_t dr2_first8_offset1 =                                   \
        vextq_f16(dr2_first8, dr2_second8, 1);                               \
    const float16x8_t dr2_first8_offset2 =                                   \
        vextq_f16(dr2_first8, dr2_second8, 2);                               \
    dr1_sum_first8 = vaddq_f16(dr1_sum_first8, dr1_first8_offset2);          \
    float16x8_t dr2_sum_first8 = vaddq_f16(dr2_first8, dr2_first8_offset1);  \
                                                                             \
    float16x8_t col_0_3_row_0_8 = vaddq_f16(dr0_sum_first8, dr1_sum_first8); \
    dr2_sum_first8 = vaddq_f16(dr2_sum_first8, dr2_first8_offset2);          \
    col_0_3_row_0_8 = vaddq_f16(col_0_3_row_0_8, dr2_sum_first8);            \
    col_0_3_row_0_8 = vmulq_f16(col_0_3_row_0_8, vcoef); /* avg = sum / 9 */ \
    vst1q_f16(dr_out, col_0_3_row_0_8);                                      \
    dr_out += 8;                                                             \
                                                                             \
    dr0_first8 = dr0_second8;                                                \
    dr1_first8 = dr1_second8;                                                \
    dr2_first8 = dr2_second8;                                                \
                                                                             \
    dr0 += 8;                                                                \
    dr1 += 8;                                                                \
    dr2 += 8;                                                                \
    dr0_second8 = vld1q_f16(dr0 + 8);                                        \
    dr1_second8 = vld1q_f16(dr1 + 8);                                        \
    dr2_second8 = vld1q_f16(dr2 + 8);                                        \
    cnt_num--;                                                               \
  }

#define P3x3S1P0_AVG_4TIMES_INTRIN                                         \
  if (cnt_remain_4 > 0) {                                                  \
    const float16x4_t dr0_first4 = vget_low_f16(dr0_first8);               \
    const float16x4_t dr1_first4 = vget_low_f16(dr1_first8);               \
    const float16x4_t dr2_first4 = vget_low_f16(dr2_first8);               \
    const float16x4_t dr0_second4 = vget_high_f16(dr0_first8);             \
    const float16x4_t dr1_second4 = vget_high_f16(dr1_first8);             \
    const float16x4_t dr2_second4 = vget_high_f16(dr2_first8);             \
                                                                           \
    const float16x4_t dr0_first4_offset1 =                                 \
        vext_f16(dr0_first4, dr0_second4, 1);                              \
    const float16x4_t dr0_first4_offset2 =                                 \
        vext_f16(dr0_first4, dr0_second4, 2);                              \
    float16x4_t dr0_sum_first4 = vadd_f16(dr0_first4, dr0_first4_offset1); \
    const float16x4_t dr1_first4_offset1 =                                 \
        vext_f16(dr1_first4, dr1_second4, 1);                              \
    const float16x4_t dr1_first4_offset2 =                                 \
        vext_f16(dr1_first4, dr1_second4, 2);                              \
    dr0_sum_first4 = vadd_f16(dr0_sum_first4, dr0_first4_offset2);         \
                                                                           \
    float16x4_t dr1_sum_first4 = vadd_f16(dr1_first4, dr1_first4_offset1); \
    const float16x4_t dr2_first4_offset1 =                                 \
        vext_f16(dr2_first4, dr2_second4, 1);                              \
    const float16x4_t dr2_first4_offset2 =                                 \
        vext_f16(dr2_first4, dr2_second4, 2);                              \
    dr1_sum_first4 = vadd_f16(dr1_sum_first4, dr1_first4_offset2);         \
                                                                           \
    float16x4_t dr2_sum_first4 = vadd_f16(dr2_first4, dr2_first4_offset1); \
    float16x4_t pool_row_result_0_4 =                                      \
        vadd_f16(dr0_sum_first4, dr1_sum_first4);                          \
    dr2_sum_first4 = vadd_f16(dr2_sum_first4, dr2_first4_offset2);         \
    pool_row_result_0_4 = vadd_f16(pool_row_result_0_4, dr2_sum_first4);   \
    pool_row_result_0_4 = vmul_f16(pool_row_result_0_4, vcoef_4);          \
    vst1_f16(dr_out, pool_row_result_0_4);                                 \
    dr0 += 4;                                                              \
    dr1 += 4;                                                              \
    dr2 += 4;                                                              \
    dr_out += 4;                                                           \
  }

void pooling_global_max_fp16(POOLING_PARAM) {
  int size_channel_in = win * hin;

  int cnt = size_channel_in >> 5;
  int remain = size_channel_in & 31;
  int cnt_8 = remain >> 3;
  int remain_8 = remain & 7;
  for (int n = 0; n < num; ++n) {
    float16_t *data_out_batch = dout + n * chout;
    const float16_t *data_in_batch = din + n * chin * size_channel_in;

    LITE_PARALLEL_BEGIN(c, tid, chout) {
      const float16_t *data_in_channel = data_in_batch + c * size_channel_in;
      float16x8_t vmax = vdupq_n_f16(data_in_channel[0]);
      int size_cnt = cnt;
      int size_remain = cnt_8;
      asm volatile(GLOBAL_INIT GLOBAL_MAX GLOBAL_MAX_REMAIN
                   : [data_in_channel] "+r"(data_in_channel),
                     [cnt] "+r"(size_cnt),
                     [remain] "+r"(size_remain),
                     [vmax] "+w"(vmax)
                   :
#ifdef __aarch64__
                   : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6");
#else
                   : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
#endif
      data_in_channel -= 8;
      float16x4_t vtmp = vmax_f16(vget_low_f16(vmax), vget_high_f16(vmax));
      float16x4_t vtmp1 = vpmax_f16(vtmp, vtmp);
      float16x4_t vtmp2 = vpmax_f16(vtmp1, vtmp1);
      for (int i = 0; i < remain_8; ++i) {
        vtmp2[0] =
            vtmp2[0] > data_in_channel[0] ? vtmp2[0] : data_in_channel[0];
        data_in_channel++;
      }
      data_out_batch[c] = vtmp2[0];
    }
    LITE_PARALLEL_END()
  }
}

void pooling_global_avg_fp16(POOLING_PARAM) {
  int size_channel_in = win * hin;

  int cnt = size_channel_in >> 5;
  int remain = size_channel_in & 31;
  int cnt_8 = remain >> 3;
  int remain_8 = remain & 7;
  for (int n = 0; n < num; ++n) {
    float16_t *data_out_batch = dout + n * chout;
    const float16_t *data_in_batch = din + n * chin * size_channel_in;

    LITE_PARALLEL_BEGIN(c, tid, chout) {
      const float16_t *data_in_channel =
          data_in_batch + c * size_channel_in;  // in address
      float16x8_t vsum = vdupq_n_f16(0.0f);
      int size_cnt = cnt;
      int size_remain = cnt_8;
      asm volatile(GLOBAL_INIT GLOBAL_AVG GLOBAL_AVG_REMAIN
                   : [data_in_channel] "+r"(data_in_channel),
                     [cnt] "+r"(size_cnt),
                     [remain] "+r"(size_remain),
                     [vsum] "+w"(vsum)
                   :
#ifdef __aarch64__
                   : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6");
#else
                   : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
#endif
      data_in_channel -= 8;
      float16x4_t vsum_tmp = vadd_f16(vget_low_f16(vsum), vget_high_f16(vsum));
      float16x4_t vtmp1 = vpadd_f16(vsum_tmp, vsum_tmp);
      float16x4_t vtmp2 = vpadd_f16(vtmp1, vtmp1);
      for (int i = 0; i < remain_8; i++) {
        vtmp2[0] += data_in_channel[0];
        data_in_channel++;
      }
      data_out_batch[c] = vtmp2[0] / size_channel_in;
    }
    LITE_PARALLEL_END()
  }
}

void pooling3x3s2p0_max_fp16(POOLING_PARAM, int pad_bottom, int pad_right) {
  const int K = 3;
  const int P = 0;
  const int S = 2;
  POOL_CNT_COMPUTE
  float minval_fp32 = std::numeric_limits<float>::lowest();
  if (right == 0) {
    cnt--;
    cnt_remain = (cnt_remain == 0) ? 4 : cnt_remain;
  }
  if (right_remain > 0) {
    cnt_remain--;
  }
  float16_t minval = minval_fp32;
  float16x8_t vmin = vdupq_n_f16(minval);
  for (int n = 0; n < num; ++n) {
    float16_t *data_out_batch = dout + n * chout * size_channel_out;
    const float16_t *data_in_batch = din + n * chin * size_channel_in;

    LITE_PARALLEL_BEGIN(c, tid, chout) {
      float16_t *data_out_channel = data_out_batch + c * size_channel_out;
      const float16_t *data_in_channel = data_in_batch + c * size_channel_in;
      const float16_t *r0 = data_in_channel;
      const float16_t *r1 = r0 + win;
      const float16_t *r2 = r1 + win;
      for (int h = 0; h < hout; h++) {
        float16_t *dr_out = data_out_channel;
        auto dr0 = r0;
        auto dr1 = r1;
        auto dr2 = r2;
        P3x3S2_MAX_PTR_CHOOSE(dr0, dr1, dr2, S, K, P, h, hin) int cnt_num =
            w_unroll_size;
        int cnt_remain_4 = cnt;
        asm volatile(P3x3S2P0_INIT P3x3S2P0_MAX P3x3S2_REMIN P3x3S2P0_MAX_REMAIN
                     : [dr0] "+r"(dr0),
                       [dr1] "+r"(dr1),
                       [dr2] "+r"(dr2),
                       [dr_out] "+r"(dr_out),
                       [cnt_num] "+r"(cnt_num)
                     : [remain] "r"(cnt_remain_4),
                       [vmin] "w"(vmin)CHANGEED_REG_0_11);
        MAX_ONE_COMPUTE(
            dr0, dr1, dr2, dr_out, cnt_remain, minval, right_remain, wend, S)
        r0 = r2;
        r1 = r0 + win;
        r2 = r1 + win;
        data_out_channel += wout;
      }
    }
    LITE_PARALLEL_END()
  }
}

void pooling3x3s2p0_avg_fp16(POOLING_PARAM,
                             bool exclusive,
                             int pad_bottom,
                             int pad_right) {
  const int K = 3;
  const int P = 0;
  const int S = 2;
  POOL_CNT_COMPUTE
  if (right == 0) {
    cnt--;
    cnt_remain = (cnt_remain == 0) ? 4 : cnt_remain;
  }
  if (right_remain > 0) {
    cnt_remain--;
  }
  auto zero_ptr = static_cast<float16_t *>(
      TargetMalloc(TARGET(kARM), win * sizeof(float16_t)));
  memset(zero_ptr, 0, win * sizeof(float16_t));

  float16x8_t vzero = vdupq_n_f16(0.f);
  for (int n = 0; n < num; ++n) {
    float16_t *data_out_batch = dout + n * chout * size_channel_out;
    const float16_t *data_in_batch = din + n * chin * size_channel_in;

    LITE_PARALLEL_BEGIN(c, tid, chout) {
      float16_t *data_out_channel = data_out_batch + c * size_channel_out;
      const float16_t *data_in_channel = data_in_batch + c * size_channel_in;
      const float16_t *r0 = data_in_channel;
      const float16_t *r1 = r0 + win;
      const float16_t *r2 = r1 + win;
      for (int h = 0; h < hout; h++) {
        float16_t coef_h = 1.f / 3;
        float16_t *dr_out = data_out_channel;
        auto dr0 = r0;
        auto dr1 = r1;
        auto dr2 = r2;
        P3x3s2_AVG_PTR_CHOOSE(
            dr1, dr2, zero_ptr, S, K, P, h, hin, coef_h, pad_bottom, exclusive);
        float16x8_t vcoef = vdupq_n_f16(coef_h / 3);
        int cnt_num = w_unroll_size;
        int cnt_remain_4 = cnt;

        asm volatile(P3x3S2P0_INIT P3x3S2P0_AVG P3x3S2_REMIN P3x3S2P0_AVG_REMAIN
                     : [dr0] "+r"(dr0),
                       [dr1] "+r"(dr1),
                       [dr2] "+r"(dr2),
                       [dr_out] "+r"(dr_out),
                       [cnt_num] "+r"(cnt_num)
                     : [remain] "r"(cnt_remain_4),
                       [vcoef] "w"(vcoef),
                       [vmin] "w"(vzero)CHANGEED_REG_0_11);
        AVG_ONE_COMPUTE(
            dr0, dr1, dr2, dr_out, cnt_remain, minval, right_remain, wend, S)
        r0 = r2;
        r1 = r0 + win;
        r2 = r1 + win;
        data_out_channel += wout;
      }
    }
    LITE_PARALLEL_END()
  }
  TargetFree(TARGET(kARM), zero_ptr);
}

void pooling3x3s2p1_max_fp16(POOLING_PARAM, int pad_bottom, int pad_right) {
  const int K = 3;
  const int P = 1;
  const int S = 2;
  POOL_CNT_COMPUTE
  float minval_fp32 = std::numeric_limits<float>::lowest();
  right = win > 7 ? 1 : 0;
  if (right == 0) {
    cnt = 0;
    cnt_remain = (cnt_remain == 0) ? 4 : cnt_remain;
  }
  if (right_remain > 0) {
    cnt_remain--;
  }
  float16_t minval = minval_fp32;
  int win_less = (w_unroll_size == 0) ? 1 : 0;

  float16x8_t vmin = vdupq_n_f16(minval);
  for (int n = 0; n < num; ++n) {
    float16_t *data_out_batch = dout + n * chout * size_channel_out;
    const float16_t *data_in_batch = din + n * chin * size_channel_in;

    LITE_PARALLEL_BEGIN(c, tid, chout) {
      float16_t *data_out_channel = data_out_batch + c * size_channel_out;
      const float16_t *data_in_channel = data_in_batch + c * size_channel_in;
      const float16_t *r0 = data_in_channel;
      const float16_t *r1 = r0 + win;
      const float16_t *r2 = r1 + win;
      for (int h = 0; h < hout; h++) {
        float16_t *dr_out = data_out_channel;
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
        P3x3S2_MAX_PTR_CHOOSE(dr0, dr1, dr2, S, K, P, h, hin) int cnt_num =
            w_unroll_size;
        int cnt_remain_4 = cnt;
        asm volatile(P3x3S2P1_INIT P3x3S2P1_MAX P3x3S2P0_MAX P3x3S2_REMIN
                         P3x3S2P1_MAX_REMAIN P3x3S2P0_MAX_REMAIN
                     : [dr0] "+r"(dr0),
                       [dr1] "+r"(dr1),
                       [dr2] "+r"(dr2),
                       [dr_out] "+r"(dr_out),
                       [cnt_num] "+r"(cnt_num)
                     : [remain] "r"(cnt_remain_4),
                       [win_less] "r"(win_less),
                       [vmin] "w"(vmin)CHANGEED_REG_0_11);
        int win_remain = cnt_remain;
        if (win_less && (cnt == 0) && win_remain > 0) {
          float16_t tmp = dr0[0];
          for (int i = 0; i < 2; i++) {
            tmp = std::max(tmp, std::max(dr0[i], dr1[i]));
            tmp = std::max(tmp, dr2[i]);
          }
          dr_out[0] = tmp;
          dr0++;
          dr1++;
          dr2++;
          dr_out++;
          win_remain--;
        }
        MAX_ONE_COMPUTE(
            dr0, dr1, dr2, dr_out, win_remain, minval, right_remain, wend, S)
        data_out_channel += wout;
      }
    }
    LITE_PARALLEL_END()
  }
}

void pooling3x3s1p0_max_fp16(POOLING_PARAM, int pad_bottom, int pad_right) {
  const int K = 3;
  const int P = 0;
  const int S = 1;
  POOL_CNT_COMPUTE
  float minval_fp32 = std::numeric_limits<float>::lowest();
  if (right == 0) {
    cnt--;
    cnt_remain = cnt_remain + 4;
  }
  if (right_remain > 0) {
    cnt_remain--;
  }
  float16_t minval = minval_fp32;
  for (int n = 0; n < num; ++n) {
    float16_t *data_out_batch = dout + n * chout * size_channel_out;
    const float16_t *data_in_batch = din + n * chin * size_channel_in;

    LITE_PARALLEL_BEGIN(c, tid, chout) {
      float16_t *data_out_channel = data_out_batch + c * size_channel_out;
      const float16_t *data_in_channel = data_in_batch + c * size_channel_in;
      const float16_t *r0 = data_in_channel;
      const float16_t *r1 = r0 + win;
      const float16_t *r2 = r1 + win;
      for (int h = 0; h < hout; h++) {
        float16_t *dr_out = data_out_channel;
        auto dr0 = r0;
        auto dr1 = r1;
        auto dr2 = r2;
        P3x3S2_MAX_PTR_CHOOSE(dr0, dr1, dr2, S, K, P, h, hin) int cnt_num =
            w_unroll_size;
        int cnt_remain_4 = cnt;
        P3x3S1P0_INIT_INTRIN P3x3S1P0_MAX_8TIMES_INTRIN
            P3x3S1P0_MAX_4TIMES_INTRIN;
        MAX_ONE_COMPUTE(
            dr0, dr1, dr2, dr_out, cnt_remain, minval, right_remain, wend, S)
        r0 = r1;
        r1 = r2;
        r2 = r1 + win;
        data_out_channel += wout;
      }
    }
    LITE_PARALLEL_END()
  }
}

void pooling3x3s1p1_max_fp16(POOLING_PARAM, int pad_bottom, int pad_right) {
  const int K = 3;
  const int P = 1;
  const int S = 1;
  POOL_CNT_COMPUTE
  right = win > 7 ? 1 : 0;
  float minval_fp32 = std::numeric_limits<float>::lowest();
  if (right == 0) {
    cnt--;
    cnt_remain = cnt < 0 ? cnt_remain : cnt_remain + 4;
  }
  if (right_remain > 0) {
    cnt_remain--;
  }
  float16_t minval = minval_fp32;
  float16x8_t vmin = vdupq_n_f16(minval);
  int win_less = (w_unroll_size == 0) ? 1 : 0;
  for (int n = 0; n < num; ++n) {
    float16_t *data_out_batch = dout + n * chout * size_channel_out;
    const float16_t *data_in_batch = din + n * chin * size_channel_in;

    LITE_PARALLEL_BEGIN(c, tid, chout) {
      float16_t *data_out_channel = data_out_batch + c * size_channel_out;
      const float16_t *data_in_channel = data_in_batch + c * size_channel_in;
      const float16_t *r0 = data_in_channel;
      const float16_t *r1 = r0 + win;
      const float16_t *r2 = r1 + win;
      for (int h = 0; h < hout; h++) {
        float16_t *dr_out = data_out_channel;
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
        P3x3S1_MAX_PTR_CHOOSE(dr0, dr1, dr2, S, K, P, h, hin) int cnt_num =
            w_unroll_size;
        int cnt_remain_4 = cnt;
        if (!win_less) {
          P3x3S1P1_INIT_INTRIN P3x3S1P0_MAX_8TIMES_INTRIN
              P3x3S1P0_MAX_4TIMES_INTRIN
        } else if (cnt_remain_4 > 0) {
          P3x3S1P1_WINLESS_INTRIN
        }
        int win_remain = cnt_remain;
        if (win_less && (cnt <= 0) && win_remain > 0) {
          float16_t tmp = dr0[0];
          for (int i = 0; i < 2; i++) {
            tmp = std::max(tmp, std::max(dr0[i], dr1[i]));
            tmp = std::max(tmp, dr2[i]);
          }
          dr_out[0] = tmp;
          dr_out++;
          win_remain--;
        }
        MAX_ONE_COMPUTE(
            dr0, dr1, dr2, dr_out, win_remain, minval, right_remain, wend, S)

        data_out_channel += wout;
      }
    }
    LITE_PARALLEL_END()
  }
}

void pooling3x3s2p1_avg_fp16(POOLING_PARAM,
                             bool exclusive,
                             int pad_bottom,
                             int pad_right) {
  const int K = 3;
  const int P = 1;
  const int S = 2;
  POOL_CNT_COMPUTE
  right = win > 7 ? 1 : 0;
  if (right == 0) {
    cnt = 0;
    cnt_remain = (cnt_remain == 0) ? 4 : cnt_remain;
  }
  if (right_remain > 0) {
    cnt_remain--;
  }
  int win_less = (w_unroll_size == 0) ? 1 : 0;

  float16x8_t vzero = vdupq_n_f16(0.f);
  auto zero_ptr = static_cast<float16_t *>(
      TargetMalloc(TARGET(kARM), win * sizeof(float16_t)));
  memset(zero_ptr, 0, win * sizeof(float16_t));
  for (int n = 0; n < num; ++n) {
    float16_t *data_out_batch = dout + n * chout * size_channel_out;
    const float16_t *data_in_batch = din + n * chin * size_channel_in;

    LITE_PARALLEL_BEGIN(c, tid, chout) {
      float16_t *data_out_channel = data_out_batch + c * size_channel_out;
      const float16_t *data_in_channel = data_in_batch + c * size_channel_in;
      const float16_t *r0 = data_in_channel;
      const float16_t *r1 = r0 + win;
      const float16_t *r2 = r1 + win;
      for (int h = 0; h < hout; h++) {
        float16_t coef_h = 1.f / 3;
        float16_t *dr_out = data_out_channel;
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
        P3x3s2_AVG_PTR_CHOOSE(
            dr1, dr2, zero_ptr, S, K, P, h, hin, coef_h, pad_bottom, exclusive)
            float16x8_t vcoef = vdupq_n_f16(coef_h / 3);
        float16_t coef_left_most = exclusive ? coef_h / 2 : coef_h / 3;
        float16_t coef_left_norm = coef_h / 3;
        float16_t coef_left[8] = {coef_left_most,
                                  coef_left_norm,
                                  coef_left_norm,
                                  coef_left_norm,
                                  coef_left_norm,
                                  coef_left_norm,
                                  coef_left_norm,
                                  coef_left_norm};
        float16x8_t vcoef_left = vld1q_f16(coef_left);
        int cnt_num = w_unroll_size;
        int cnt_remain_4 = cnt;
        asm volatile(P3x3S2P1_INIT P3x3S2P1_AVG P3x3S2P0_AVG P3x3S2_REMIN
                         P3x3S2P1_AVG_REMAIN P3x3S2P0_AVG_REMAIN
                     : [dr0] "+r"(dr0),
                       [dr1] "+r"(dr1),
                       [dr2] "+r"(dr2),
                       [dr_out] "+r"(dr_out),
                       [cnt_num] "+r"(cnt_num)
                     : [remain] "r"(cnt_remain_4),
                       [win_less] "r"(win_less),
                       [vmin] "w"(vzero),
                       [vcoef_left] "w"(vcoef_left),
                       [vcoef] "w"(vcoef)CHANGEED_REG_0_11);
        int win_remain = cnt_remain;
        if (win_less && (cnt == 0) && win_remain > 0) {
          float16_t sum = 0.f;
          for (int i = 0; i < 2; i++) {
            sum += dr0[i] + dr1[i] + dr2[i];
          }
          dr_out[0] = sum * coef_left_most;
          dr0++;
          dr1++;
          dr2++;
          dr_out++;
          win_remain--;
        }
        AVG_ONE_COMPUTE(
            dr0, dr1, dr2, dr_out, win_remain, minval, right_remain, wend, S)
        data_out_channel += wout;
      }
    }
    LITE_PARALLEL_END()
  }
  TargetFree(TARGET(kARM), zero_ptr);
}

void pooling3x3s1p0_avg_fp16(POOLING_PARAM,
                             bool exclusive,
                             int pad_bottom,
                             int pad_right) {
  const int K = 3;
  const int P = 0;
  const int S = 1;
  POOL_CNT_COMPUTE
  if (right == 0) {
    cnt--;
    cnt_remain = (cnt_remain == 0) ? 4 : cnt_remain;
  }
  if (right_remain > 0) {
    cnt_remain--;
  }
  auto zero_ptr = static_cast<float16_t *>(
      TargetMalloc(TARGET(kARM), win * sizeof(float16_t)));
  memset(zero_ptr, 0, win * sizeof(float16_t));

  for (int n = 0; n < num; ++n) {
    float16_t *data_out_batch = dout + n * chout * size_channel_out;
    const float16_t *data_in_batch = din + n * chin * size_channel_in;

    LITE_PARALLEL_BEGIN(c, tid, chout) {
      float16_t *data_out_channel = data_out_batch + c * size_channel_out;
      const float16_t *data_in_channel = data_in_batch + c * size_channel_in;
      const float16_t *r0 = data_in_channel;
      const float16_t *r1 = r0 + win;
      const float16_t *r2 = r1 + win;
      for (int h = 0; h < hout; h++) {
        float16_t coef_h = 1.f / 3;
        float16_t *dr_out = data_out_channel;
        auto dr0 = r0;
        auto dr1 = r1;
        auto dr2 = r2;
        P3x3s2_AVG_PTR_CHOOSE(
            dr1, dr2, zero_ptr, S, K, P, h, hin, coef_h, pad_bottom, exclusive)
            float16x8_t vcoef = vdupq_n_f16(coef_h / 3);
        float16x4_t vcoef_4 = vget_low_f16(vcoef);
        int cnt_num = w_unroll_size;
        P3x3S1P0_INIT_INTRIN P3x3S1P0_AVG_8TIMES_INTRIN AVG_ONE_COMPUTE(
            dr0,
            dr1,
            dr2,
            dr_out,
            cnt_remain + cnt * 4,
            minval,
            right_remain,
            wend,
            S) r0 = r1;
        r1 = r2;
        r2 = r1 + win;
        data_out_channel += wout;
      }
    }
    LITE_PARALLEL_END()
  }
  TargetFree(TARGET(kARM), zero_ptr);
}

void pooling3x3s1p1_avg_fp16(POOLING_PARAM,
                             bool exclusive,
                             int pad_bottom,
                             int pad_right) {
  const int K = 3;
  const int P = 1;
  const int S = 1;
  POOL_CNT_COMPUTE
  right = win > 7 ? 1 : 0;
  if (right == 0) {
    cnt--;
    cnt_remain = cnt < 0 ? cnt_remain : cnt_remain + 4;
  }
  if (right_remain > 0) {
    cnt_remain--;
  }
  int win_less = (w_unroll_size == 0) ? 1 : 0;

  float16x8_t vzero = vdupq_n_f16(0.f);
  auto zero_ptr = static_cast<float16_t *>(
      TargetMalloc(TARGET(kARM), win * sizeof(float16_t)));
  memset(zero_ptr, 0, win * sizeof(float16_t));
  for (int n = 0; n < num; ++n) {
    float16_t *data_out_batch = dout + n * chout * size_channel_out;
    const float16_t *data_in_batch = din + n * chin * size_channel_in;

    LITE_PARALLEL_BEGIN(c, tid, chout) {
      float16_t *data_out_channel = data_out_batch + c * size_channel_out;
      const float16_t *data_in_channel = data_in_batch + c * size_channel_in;
      const float16_t *r0 = data_in_channel;
      const float16_t *r1 = r0 + win;
      const float16_t *r2 = r1 + win;
      for (int h = 0; h < hout; h++) {
        float16_t coef_h = 1.f / 3;
        float16_t *dr_out = data_out_channel;
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
        P3x3s2_AVG_PTR_CHOOSE(
            dr1, dr2, zero_ptr, S, K, P, h, hin, coef_h, pad_bottom, exclusive);
        float16x8_t vcoef = vdupq_n_f16(coef_h / 3);
        float16x4_t vcoef_4 = vget_low_f16(vcoef);
        float16_t coef_left_most = exclusive ? coef_h / 2 : coef_h / 3;
        float16_t coef_left_norm = coef_h / 3;
        float16_t coef_left[8] = {coef_left_most,
                                  coef_left_norm,
                                  coef_left_norm,
                                  coef_left_norm,
                                  coef_left_norm,
                                  coef_left_norm,
                                  coef_left_norm,
                                  coef_left_norm};
        float16x8_t vcoef_left = vld1q_f16(coef_left);
        float16x8_t vmin = vzero;
        float16x4_t vmin_4 = vget_low_f16(vzero);
        float16x4_t vcoef_left4 = vget_low_f16(vcoef_left);
        int cnt_num = w_unroll_size;
        int cnt_remain_4 = cnt;
        if (!win_less) {
          P3x3S1P1_AVG_INIT_INTRIN P3x3S1P0_AVG_8TIMES_INTRIN
              P3x3S1P0_AVG_4TIMES_INTRIN
        } else if (cnt_remain_4 > 0) {
          P3x3S1P1_AVG_WINLESS_INTRIN
        }
        int win_remain = cnt_remain;
        if (win_less && (cnt <= 0) && win_remain > 0) {
          float16_t sum = 0.f;
          for (int i = 0; i < 2; i++) {
            sum += dr0[i] + dr1[i] + dr2[i];
          }
          dr_out[0] = sum * coef_left_most;
          dr_out++;
          win_remain--;
        }
        AVG_ONE_COMPUTE(
            dr0, dr1, dr2, dr_out, win_remain, minval, right_remain, wend, S)
        data_out_channel += wout;
      }
    }
    LITE_PARALLEL_END()
  }
  TargetFree(TARGET(kARM), zero_ptr);
}

#undef CHANGEED_REG_0_11
#undef GLOBAL_INIT
#undef GLOBAL_MAX
#undef GLOBAL_AVG
#undef GLOBAL_MAX_REMAIN
#undef GLOBAL_AVG_REMAIN
#undef P3x3S2P1_INIT
#undef P3x3S2P0_INIT
#undef P3x3S2P1_MAX
#undef P3x3S2P0_MAX
#undef P3x3S2_REMIN
#undef P3x3S2P1_MAX_REMAIN
#undef P3x3S2P0_MAX_REMAIN
#undef P3x3S2P1_AVG
#undef P3x3S2P0_AVG
#undef P3x3S2P1_AVG_REMAIN
#undef P3x3S2P0_AVG_REMAIN
#undef POOL_CNT_COMPUTE
#undef MAX_ONE_COMPUTE
#undef AVG_ONE_COMPUTE
#undef P3x3S1_MAX_PTR_CHOOSE
#undef P3x3S2_MAX_PTR_CHOOSE
#undef P3x3s2_AVG_PTR_CHOOSE
#undef P3x3S1P0_INIT_INTRIN
#undef P3x3S1P1_WINLESS_INTRIN
#undef P3x3S1P1_AVG_WINLESS_INTRIN
#undef P3x3S1P1_INIT_INTRIN
#undef P3x3S1P1_AVG_INIT_INTRIN
#undef P3x3S1P0_MAX_8TIMES_INTRIN
#undef P3x3S1P0_MAX_4TIMES_INTRIN
#undef P3x3S1P0_AVG_8TIMES_INTRIN
#undef P3x3S1P0_AVG_4TIMES_INTRIN
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
