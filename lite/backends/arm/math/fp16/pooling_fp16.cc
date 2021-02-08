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

#include "lite/backends/arm/math/fp16/pooling_fp16.h"
#include <algorithm>
#include <limits>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

int AdaptStartIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      floor(static_cast<double>(ph * input_size) / output_size));
}

int AdaptEndIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      ceil(static_cast<double>((ph + 1) * input_size) / output_size));
}

void pooling_basic_fp16(const float16_t* din,
                        float16_t* dout,
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
  memset(dout, 0, num * chout * hout * wout * sizeof(float16_t));
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
        float16_t* dout_batch = dout + n * chout * size_channel_out;
        const float16_t* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
        for (int c = 0; c < chout; ++c) {
          const float16_t* din_ch =
              din_batch + c * size_channel_in;  // in address
          float16_t tmp1 = din_ch[0];
          for (int i = 0; i < size_channel_in; ++i) {
            float16_t tmp2 = din_ch[i];
            tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
          }
          dout_batch[c] = tmp1;
        }
      }
    } else if (pooling_type == "avg") {
      // Pooling_average_include_padding
      for (int n = 0; n < num; ++n) {
        float16_t* dout_batch = dout + n * chout * size_channel_out;
        const float16_t* din_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
        for (int c = 0; c < chout; ++c) {
          const float16_t* din_ch =
              din_batch + c * size_channel_in;  // in address
          float16_t sum = 0.f;
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

#endif

void pooling_global_max_fp16(const float16_t* din,
                             float16_t* dout,
                             int num,
                             int chout,
                             int hout,
                             int wout,
                             int chin,
                             int hin,
                             int win) {
  int size_channel_in = win * hin;

  int cnt = size_channel_in >> 5;
  int remain = size_channel_in & 31;
  int cnt_8 = remain >> 3;
  int remain_8 = remain & 7;

  for (int n = 0; n < num; ++n) {
    float16_t* data_out_batch = dout + n * chout;
    const float16_t* data_in_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; ++c) {
      const float16_t* data_in_channel = data_in_batch + c * size_channel_in;
      float16x8_t vmax = vdupq_n_f16(data_in_channel[0]);
      int size_cnt = cnt;
      int size_remain = cnt_8;
#ifdef __aarch64__
      asm volatile(GLOBAL_INIT GLOBAL_MAX GLOBAL_MAX_REMAIN
                   : [data_in_channel] "+r"(data_in_channel),
                     [cnt] "+r"(size_cnt),
                     [remain] "+r"(size_remain),
                     [vmax] "+w"(vmax)
                   :
                   : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6");
#else
#endif  //  __aarch64__
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
  }
}

void pooling_global_avg_fp16(const float16_t* din,
                             float16_t* dout,
                             int num,
                             int chout,
                             int hout,
                             int wout,
                             int chin,
                             int hin,
                             int win) {
  int size_channel_in = win * hin;

  int cnt = size_channel_in >> 5;
  int remain = size_channel_in & 31;
  int cnt_8 = remain >> 3;
  int remain_8 = remain & 7;

  for (int n = 0; n < num; ++n) {
    float16_t* data_out_batch = dout + n * chout;
    const float16_t* data_in_batch = din + n * chin * size_channel_in;
#pragma omp parallel for
    for (int c = 0; c < chout; c++) {
      const float16_t* data_in_channel =
          data_in_batch + c * size_channel_in;  // in address
      float16x8_t vsum = vdupq_n_f16(0.0f);
      int size_cnt = cnt;
      int size_remain = cnt_8;
#ifdef __aarch64__
      asm volatile(GLOBAL_INIT GLOBAL_AVG GLOBAL_AVG_REMAIN
                   : [data_in_channel] "+r"(data_in_channel),
                     [cnt] "+r"(size_cnt),
                     [remain] "+r"(size_remain),
                     [vsum] "+w"(vsum)
                   :
                   : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6");
#else
#endif  //  __aarch64__
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
  }
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
