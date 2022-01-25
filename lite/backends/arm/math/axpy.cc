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

#include "lite/backends/arm/math/axpy.h"
#include <algorithm>
#include <limits>
#include <memory>
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/saturate.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void axpy_kernel_fp32(const float* scale,
                      const float* din,
                      const float* bias,
                      float* dout,
                      int num,
                      int channel,
                      int size,
                      int in_channel) {
  int cnt = size >> 3;
  int remain = size % 8;
  for (int n = 0; n < num; n++) {
    const float* din_ptr = din + n * in_channel;
    const float* scale_ptr = scale + n * channel;
    const float* bias_ptr = bias + n * in_channel;
    float* dout_ptr = dout + n * in_channel;
    LITE_PARALLEL_BEGIN(c, tid, channel) {
      const float* din_ch_ptr = din_ptr + c * size;
      const float* bias_ch_ptr = bias_ptr + c * size;
      float* dout_ch_ptr = dout_ptr + c * size;
      float32x4_t scale_val = vdupq_n_f32(scale_ptr[c]);
      int col_cnt = cnt;
      if (cnt > 0) {
#ifdef __aarch64__
        asm volatile(
            "ld1 {v0.4s}, [%[din_ptr]], #16     \n"
            "ld1 {v1.4s}, [%[bias_ptr]], #16    \n"
            "1:                                     \n"
            "ld1 {v2.4s}, [%[din_ptr]], #16     \n"
            "ld1 {v3.4s}, [%[bias_ptr]], #16    \n"
            "fmul v4.4s ,  v0.4s,  %[scale].4s  \n"
            "fmul v5.4s ,  v2.4s,  %[scale].4s  \n"
            "fadd v4.4s, v4.4s, v1.4s           \n"
            "fadd v5.4s, v5.4s, v3.4s           \n"
            "ld1 {v0.4s}, [%[din_ptr]], #16     \n"
            "ld1 {v1.4s}, [%[bias_ptr]], #16    \n"
            "subs %[cnt], %[cnt], #1 \n"
            "st1 {v4.4s}, [%[dout_ptr]], #16     \n"
            "st1 {v5.4s}, [%[dout_ptr]], #16     \n"
            "bne        1b                          \n"
            : [din_ptr] "+r"(din_ch_ptr),
              [bias_ptr] "+r"(bias_ch_ptr),
              [dout_ptr] "+r"(dout_ch_ptr),
              [cnt] "+r"(col_cnt)
            : [scale] "w"(scale_val)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5");
#else
        asm volatile(
            "vld1.32 {d2-d3}, [%[din_ptr]]!        \n"
            "vld1.32 {d4-d5}, [%[bias_ptr]]!       \n"
            "1:                                     \n"
            "vld1.32 {d6-d7}, [%[din_ptr]]!        \n"
            "vld1.32 {d8-d9}, [%[bias_ptr]]!       \n"
            "vmul.f32 q5, q1, %q[scale]            \n"
            "vmul.f32 q6, q3, %q[scale]            \n"
            "vadd.f32 q5, q5, q2                   \n"
            "vadd.f32 q6, q6, q4                   \n"
            "vld1.f32 {d2-d3}, [%[din_ptr]]!        \n"
            "vld1.f32 {d4-d5}, [%[bias_ptr]]!       \n"
            "subs    %[cnt], #1                 \n"
            "vst1.32 {d10-d11}, [%[dout_ptr]]!    \n"
            "vst1.32 {d12-d13}, [%[dout_ptr]]!    \n"
            "bne        1b                          \n"
            : [din_ptr] "+r"(din_ch_ptr),
              [bias_ptr] "+r"(bias_ch_ptr),
              [dout_ptr] "+r"(dout_ch_ptr),
              [cnt] "+r"(col_cnt)
            : [scale] "w"(scale_val)
            : "cc", "memory", "q1", "q2", "q3", "q4", "q5", "q6");
#endif
      }
      din_ch_ptr = din_ptr + c * size + cnt * 8;
      bias_ch_ptr = bias_ptr + c * size + cnt * 8;
      for (int i = 0; i < remain; i++) {
        *dout_ch_ptr = (*din_ch_ptr) * scale_ptr[c] + (*bias_ch_ptr);
        dout_ch_ptr++;
        din_ch_ptr++;
        bias_ch_ptr++;
      }
    }
    LITE_PARALLEL_END()
  }
}

void axpy_kernel_int8(const int8_t* scale,
                      const int8_t* din,
                      const int8_t* bias,
                      int8_t* dout,
                      int num,
                      int channel,
                      int size,
                      int in_channel) {
  int cnt = size >> 4;
  int remain = size % 16;
  for (int n = 0; n < num; n++) {
    const int8_t* din_ptr = din + n * in_channel;
    const int8_t* scale_ptr = scale + n * channel;
    const int8_t* bias_ptr = bias + n * in_channel;
    int8_t* dout_ptr = dout + n * in_channel;

    LITE_PARALLEL_BEGIN(c, tid, channel) {
      const int8_t* din_ch_ptr = din_ptr + c * size;
      const int8_t* bias_ch_ptr = bias_ptr + c * size;
      int8_t* dout_ch_ptr = dout_ptr + c * size;
      int8x8_t scale_val = vdup_n_s8(scale_ptr[c]);
      int col_cnt = cnt;
      if (col_cnt > 0) {
#ifdef __aarch64__
        asm volatile(
            "ld1 {v0.8b}, [%[din_ptr]], #8     \n"
            "ld1 {v1.8b}, [%[bias_ptr]], #8    \n"
            "1:                                     \n"
            "ld1 {v2.8b}, [%[din_ptr]], #8     \n"
            "ld1 {v3.8b}, [%[bias_ptr]], #8    \n"
            "smull  v4.8h,  v0.8b,  %[scale].8b \n"
            "smull  v5.8h,  v2.8b,  %[scale].8b \n"
            "saddw v4.8h, v4.8h, v1.8b           \n"
            "saddw v5.8h, v5.8h, v3.8b           \n"
            "ld1 {v0.8b}, [%[din_ptr]], #8     \n"
            "ld1 {v1.8b}, [%[bias_ptr]], #8    \n"
            "subs %[cnt], %[cnt], #1 \n"
            // int16->int8
            "sqxtn  v6.8b, v4.8h               \n"
            "sqxtn  v7.8b, v5.8h               \n"
            "st1   {v6.8b}, [%[dout_ptr]], #8 \n" /* store c0r0*/
            "st1   {v7.8b}, [%[dout_ptr]], #8 \n" /* store c2r0*/
            "bne        1b                          \n"
            : [din_ptr] "+r"(din_ch_ptr),
              [bias_ptr] "+r"(bias_ch_ptr),
              [dout_ptr] "+r"(dout_ch_ptr),
              [cnt] "+r"(col_cnt)
            : [scale] "w"(scale_val)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5");
#else
        asm volatile(
            "vdup.s8 d0, %[scale]          \n"
            "vld1.8 {d2}, [%[din_ptr]]!        \n"
            "vld1.8 {d4}, [%[bias_ptr]]!       \n"
            "1:                                     \n"
            "vld1.8 {d3}, [%[din_ptr]]!        \n"
            "vld1.8 {d5}, [%[bias_ptr]]!       \n"
            "vmull.s8 q4, d2, d0            \n"
            "vmull.s8 q5, d3, d0            \n"
            "vaddw.s16 q4, q4, d4                   \n"
            "vaddw.s16 q5, q5, d5                   \n"
            "vld1.8 {d2}, [%[din_ptr]]!        \n"
            "vld1.8 {d4}, [%[bias_ptr]]!       \n"
            "subs       %[cnt], #1                 \n"
            // int16->int8
            "vqmovn.s16 d12, q4                     @ cnt to int8\n"
            "vqmovn.s16 d13, q5                     @ cnt to int8\n"
            "vst1.32 {d12-d13}, [%[dout_ptr]]!    \n"
            "bne        1b                          \n"
            : [din_ptr] "+r"(din_ch_ptr),
              [bias_ptr] "+r"(bias_ch_ptr),
              [dout_ptr] "+r"(dout_ch_ptr),
              [cnt] "+r"(col_cnt)
            : [scale] "r"(scale_val)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
#endif
      }
      din_ch_ptr = din_ptr + c * size + cnt * 16;
      bias_ch_ptr = bias_ptr + c * size + cnt * 16;
      for (int i = 0; i < remain; i++) {
        *dout_ch_ptr = saturate_cast<int8_t>(
            roundf((*din_ch_ptr) * scale_ptr[c] + (*bias_ch_ptr)));
        dout_ch_ptr++;
        din_ch_ptr++;
        bias_ch_ptr++;
      }
    }
    LITE_PARALLEL_END()
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
