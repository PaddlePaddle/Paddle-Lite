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

#include "lite/backends/arm/math/layout.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
#ifdef __aarch64__
#define TRANS_C4                                                \
  "ld1 {v0.4s}, [%[din0_ptr]]   \n"                             \
  "ld1 {v1.4s}, [%[din1_ptr]]   \n"                             \
  "ld1 {v2.4s}, [%[din2_ptr]]   \n"                             \
  "ld1 {v3.4s}, [%[din3_ptr]]   \n"                             \
                                                                \
  "1: \n"                                                       \
  "trn1 v4.4s, v0.4s, v1.4s \n" /*00 10 02 12 */                \
  "trn1 v5.4s, v2.4s, v3.4s \n" /*20 30 22 32 */                \
  "trn2 v6.4s, v0.4s, v1.4s \n" /*01 11 03 13 */                \
  "trn2 v7.4s, v2.4s, v3.4s \n" /*21 31 23 33 */                \
                                                                \
  "add %[din0_ptr], %[din0_ptr], %[stride] \n" /* din+=c*size*/ \
  "add %[din1_ptr], %[din1_ptr], %[stride] \n" /* din+=c*size*/ \
  "add %[din2_ptr], %[din2_ptr], %[stride] \n" /* din+=c*size*/ \
  "add %[din3_ptr], %[din3_ptr], %[stride] \n" /* din+=c*size*/ \
                                                                \
  "trn1 v8.2d, v4.2d, v5.2d \n"  /*00 10 20 30 */               \
  "trn1 v9.2d, v6.2d, v7.2d \n"  /*01 11 21 31 */               \
  "trn2 v10.2d, v4.2d, v5.2d \n" /*02 12 22 32 */               \
  "trn2 v11.2d, v6.2d, v7.2d \n" /*03 13 23 33 */               \
                                                                \
  "ld1 {v0.4s}, [%[din0_ptr]]   \n"                             \
  "ld1 {v1.4s}, [%[din1_ptr]]   \n"                             \
  "ld1 {v2.4s}, [%[din2_ptr]]   \n"                             \
  "ld1 {v3.4s}, [%[din3_ptr]]   \n"                             \
                                                                \
  "subs %w[cnt], %w[cnt], #1 \n"                                \
  "str q8, [%[out0_ptr]], #16 \n"                               \
  "str q9, [%[out1_ptr]], #16 \n"                               \
  "str q10, [%[out2_ptr]], #16 \n"                              \
  "str q11, [%[out3_ptr]], #16 \n"                              \
  "bne 1b \n"

#define TRANS_C8                                                  \
  "1: \n"                                                         \
  "ld1 {v0.8b}, [%[din0_ptr]]   \n"                               \
  "ld1 {v1.8b}, [%[din1_ptr]]   \n"                               \
  "ld1 {v2.8b}, [%[din2_ptr]]   \n"                               \
  "ld1 {v3.8b}, [%[din3_ptr]]   \n"                               \
                                                                  \
  "add %[din0_ptr], %[din0_ptr], %[stride_w] \n" /* din+=c*size*/ \
  "add %[din1_ptr], %[din1_ptr], %[stride_w] \n" /* din+=c*size*/ \
  "add %[din2_ptr], %[din2_ptr], %[stride_w] \n" /* din+=c*size*/ \
  "add %[din3_ptr], %[din3_ptr], %[stride_w] \n" /* din+=c*size*/ \
                                                                  \
  "trn1 v8.8b, v0.8b, v1.8b \n"  /*00 10 02 12 04 14 06 16 */     \
  "trn1 v9.8b, v2.8b, v3.8b \n"  /*20 30 22 32 */                 \
  "trn2 v12.8b, v0.8b, v1.8b \n" /*01 11 03 13 05 15 07 17 */     \
  "trn2 v13.8b, v2.8b, v3.8b \n" /*21 31 23 33 */                 \
                                                                  \
  "ld1 {v4.8b}, [%[din0_ptr]]   \n"                               \
  "ld1 {v5.8b}, [%[din1_ptr]]   \n"                               \
  "ld1 {v6.8b}, [%[din2_ptr]]   \n"                               \
  "ld1 {v7.8b}, [%[din3_ptr]]   \n"                               \
                                                                  \
  "trn1 v10.8b, v4.8b, v5.8b \n" /*40 50 42 52 */                 \
  "trn1 v11.8b, v6.8b, v7.8b \n" /*60 70 62 72 */                 \
  "trn2 v14.8b, v4.8b, v5.8b \n" /*41 51 43 53 */                 \
  "trn2 v15.8b, v6.8b, v7.8b \n" /*61 71 63 73 */                 \
                                                                  \
  "trn1 v0.4h, v8.4h, v9.4h \n"   /*00 10 20 30 04 14 24 34*/     \
  "trn1 v2.4h, v12.4h, v13.4h \n" /*01 11 21 31 05 15 25 35*/     \
  "trn1 v1.4h, v10.4h, v11.4h \n" /*40 50 60 70 44 54 64 74*/     \
  "trn1 v3.4h, v14.4h, v15.4h \n" /*41 51 61 71 45 55 65 75*/     \
                                                                  \
  "trn2 v4.4h, v8.4h, v9.4h \n"   /*02 10 20 30 06 14 24 34*/     \
  "trn2 v6.4h, v12.4h, v13.4h \n" /*03 11 21 31 07 15 25 35*/     \
  "trn2 v5.4h, v10.4h, v11.4h \n" /*42 50 60 70 46 54 64 74*/     \
  "trn2 v7.4h, v14.4h, v15.4h \n" /*43 51 61 71 47 55 65 75*/     \
                                                                  \
  "trn1 v8.2s, v0.2s, v1.2s \n"  /*00 10 20 30 40 50 60 70*/      \
  "trn1 v9.2s, v2.2s, v3.2s \n"  /*01 11 21 31 41 51 61 71*/      \
  "trn1 v10.2s, v4.2s, v5.2s \n" /*02 12 22 32 42 50 60 70*/      \
  "trn1 v11.2s, v6.2s, v7.2s \n" /*03 13 23 33 41 51 61 71*/      \
                                                                  \
  "trn2 v12.2s, v0.2s, v1.2s \n" /*04 14 24 34 44 54 64 74*/      \
  "trn2 v13.2s, v2.2s, v3.2s \n" /*05 15 25 35  45 55 65 75*/     \
  "trn2 v14.2s, v4.2s, v5.2s \n" /*06 16 22 32 42 50 60 70*/      \
  "trn2 v15.2s, v6.2s, v7.2s \n" /*07 17 23 33 41 51 61 71*/      \
                                                                  \
  "add %[din0_ptr], %[din0_ptr], %[stride_w] \n" /* din+=c*size*/ \
  "add %[din1_ptr], %[din1_ptr], %[stride_w] \n" /* din+=c*size*/ \
  "add %[din2_ptr], %[din2_ptr], %[stride_w] \n" /* din+=c*size*/ \
  "add %[din3_ptr], %[din3_ptr], %[stride_w] \n" /* din+=c*size*/ \
                                                                  \
  "subs %w[cnt], %w[cnt], #1 \n"                                  \
  "st1 {v8.8b}, [%[out0_ptr]], #8 \n"                             \
  "st1 {v9.8b}, [%[out1_ptr]], #8 \n"                             \
  "st1 {v10.8b}, [%[out2_ptr]], #8 \n"                            \
  "st1 {v11.8b}, [%[out3_ptr]], #8 \n"                            \
                                                                  \
  "st1 {v11.8b}, [%[out4_ptr]], #8 \n"                            \
  "st1 {v12.8b}, [%[out5_ptr]], #8 \n"                            \
  "st1 {v13.8b}, [%[out6_ptr]], #8 \n"                            \
  "st1 {v14.8b}, [%[out7_ptr]], #8 \n"                            \
  "bne 1b \n"

#else
#define TRANS_C4                                                \
  "1: \n"                                                       \
  "vld1.32 {d0-d1}, [%[din0_ptr]] \n"                           \
  "vld1.32 {d2-d3}, [%[din1_ptr]] \n"                           \
  "vld1.32 {d4-d5}, [%[din2_ptr]] \n"                           \
  "vld1.32 {d6-d7}, [%[din3_ptr]] \n"                           \
                                                                \
  "vtrn.32 q0, q1 \n" /*00 10 02 12 01 11 03 13*/               \
  "vtrn.32 q2, q3 \n" /*20 30 22 32 21 31 23 33 */              \
                                                                \
  "add %[din0_ptr], %[din0_ptr], %[stride] \n" /* din+=c*size*/ \
  "add %[din1_ptr], %[din1_ptr], %[stride] \n" /* din+=c*size*/ \
  "add %[din2_ptr], %[din2_ptr], %[stride] \n" /* din+=c*size*/ \
  "add %[din3_ptr], %[din3_ptr], %[stride] \n" /* din+=c*size*/ \
  "vswp d1, d4 \n"                                              \
  "vswp d3, d6 \n"                                              \
                                                                \
  "subs %[cnt], %[cnt], #1 \n"                                  \
  "vst1.32  {d0-d1}, [%[out0_ptr]]! \n"                         \
  "vst1.32  {d2-d3}, [%[out1_ptr]]! \n"                         \
  "vst1.32  {d4-d5}, [%[out2_ptr]]! \n"                         \
  "vst1.32  {d6-d7}, [%[out3_ptr]]! \n"                         \
  "bne 1b \n"

#define TRANS_C8                                                  \
  "1: \n"                                                         \
  "vld1.8 d0, [%[din0_ptr]] \n"                                   \
  "vld1.8 d1, [%[din1_ptr]] \n"                                   \
  "vld1.8 d2, [%[din2_ptr]] \n"                                   \
  "vld1.8 d3, [%[din3_ptr]] \n"                                   \
                                                                  \
  "add %[din0_ptr], %[din0_ptr], %[stride_w] \n" /* din+=c*size*/ \
  "add %[din1_ptr], %[din1_ptr], %[stride_w] \n" /* din+=c*size*/ \
  "add %[din2_ptr], %[din2_ptr], %[stride_w] \n" /* din+=c*size*/ \
  "add %[din3_ptr], %[din3_ptr], %[stride_w] \n" /* din+=c*size*/ \
                                                                  \
  "vtrn.8 d0, d1 \n" /*00 10 02 12 04 14 06 16*/                  \
  "vtrn.8 d2, d3 \n" /*20 30 22 32 24 34 26 36 */                 \
                                                                  \
  "vld1.8 d4, [%[din0_ptr]] \n"                                   \
  "vld1.8 d5, [%[din1_ptr]] \n"                                   \
  "vld1.8 d6, [%[din2_ptr]] \n"                                   \
  "vld1.8 d7, [%[din3_ptr]] \n"                                   \
                                                                  \
  "vtrn.16 d0, d2 \n" /*00 10 20 30 04 14 24 34*/                 \
  "vtrn.16 d1, d3 \n" /* 01 11 21 31 05 15 25 35 */               \
  "vtrn.8 d4, d5 \n"  /*40 50 02 12 04 14 06 16*/                 \
  "vtrn.8 d6, d7 \n"  /*60 70 22 32 24 34 26 36 */                \
                                                                  \
  "add %[din0_ptr], %[din0_ptr], %[stride_w] \n" /* din+=c*size*/ \
  "add %[din1_ptr], %[din1_ptr], %[stride_w] \n" /* din+=c*size*/ \
  "add %[din2_ptr], %[din2_ptr], %[stride_w] \n" /* din+=c*size*/ \
  "add %[din3_ptr], %[din3_ptr], %[stride_w] \n" /* din+=c*size*/ \
                                                                  \
  "vtrn.16 d4, d6 \n" /*40 50 60 70 04 14 24 34*/                 \
  "vtrn.16 d5, d7 \n" /* 41 51 61 71 05 15 25 35 */               \
                                                                  \
  "vtrn.32 d0, d4 \n" /*00 10 20 30 40 50 60 70*/                 \
  "vtrn.32 d1, d5 \n" /* 01 11 21 31 41 51 61 71 */               \
  "vtrn.32 d2, d6 \n" /*02 12 22 32 42 52 62 72*/                 \
  "vtrn.32 d3, d7 \n" /* 03 11 21 33 43 53 63 73 */               \
                                                                  \
  "subs %[cnt], %[cnt], #1 \n"                                    \
  "vst1.8  {d0}, [%[out0_ptr]]! \n"                               \
  "vst1.8  {d1}, [%[out1_ptr]]! \n"                               \
  "vst1.8  {d2}, [%[out2_ptr]]! \n"                               \
  "vst1.8  {d3}, [%[out3_ptr]]! \n"                               \
  "vst1.8  {d4}, [%[out4_ptr]]! \n"                               \
  "vst1.8  {d5}, [%[out5_ptr]]! \n"                               \
  "vst1.8  {d6}, [%[out6_ptr]]! \n"                               \
  "vst1.8  {d7}, [%[out7_ptr]]! \n"                               \
  "bne 1b \n"

#endif
template <>
void NCHW2NHWC<float>(int N, int C, int size, const float* X, float* Y) {
  int cnt = C >> 2;
  int remain = C % 4;
  int sum = C * size;
  int stride = size << 4;  // 4 * size
  int stride_w = stride >> 2;
  for (int n = 0; n < N; n++) {
    const float* din = X + n * sum;
    float* dout = Y + n * sum;
    int s = 0;

    LITE_PARALLEL_COMMON_BEGIN(s, tid, size - 3, 0, 4) {
      const float* din0_ptr = din + s;
      const float* din1_ptr = din0_ptr + size;
      const float* din2_ptr = din1_ptr + size;
      const float* din3_ptr = din2_ptr + size;
      float* out0_ptr = dout + s * C;
      float* out1_ptr = out0_ptr + C;
      float* out2_ptr = out1_ptr + C;
      float* out3_ptr = out2_ptr + C;
      int cnt_num = cnt;
      if (cnt_num > 0) {
#ifdef __aarch64__
        asm volatile(TRANS_C4
                     : [din0_ptr] "+r"(din0_ptr),
                       [din1_ptr] "+r"(din1_ptr),
                       [din2_ptr] "+r"(din2_ptr),
                       [din3_ptr] "+r"(din3_ptr),
                       [out0_ptr] "+r"(out0_ptr),
                       [out1_ptr] "+r"(out1_ptr),
                       [out2_ptr] "+r"(out2_ptr),
                       [out3_ptr] "+r"(out3_ptr),
                       [cnt] "+r"(cnt_num),
                       [stride] "+r"(stride)
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
                       "v12");
#else
        asm volatile(TRANS_C4
                     : [din0_ptr] "+r"(din0_ptr),
                       [din1_ptr] "+r"(din1_ptr),
                       [din2_ptr] "+r"(din2_ptr),
                       [din3_ptr] "+r"(din3_ptr),
                       [out0_ptr] "+r"(out0_ptr),
                       [out1_ptr] "+r"(out1_ptr),
                       [out2_ptr] "+r"(out2_ptr),
                       [out3_ptr] "+r"(out3_ptr),
                       [cnt] "+r"(cnt_num),
                       [stride] "+r"(stride)
                     :
                     : "cc", "memory", "q0", "q1", "q2", "q3");
#endif
      }
      for (int i = 0; i < remain; i++) {
        const float* ptr = din0_ptr;
        *out0_ptr++ = *ptr++;
        *out1_ptr++ = *ptr++;
        *out2_ptr++ = *ptr++;
        *out3_ptr++ = *ptr++;
        din0_ptr += size;
      }
    }
    LITE_PARALLEL_COMMON_END()
    // remain size
    for (; s < size; s++) {
      const float* din0_ptr = din + s;
      const float* din1_ptr = din0_ptr + size;
      const float* din2_ptr = din1_ptr + size;
      const float* din3_ptr = din2_ptr + size;
      float* out0_ptr = dout + s * C;
      for (int i = 0; i < cnt; i++) {
        *out0_ptr++ = *din0_ptr;
        *out0_ptr++ = *din1_ptr;
        *out0_ptr++ = *din2_ptr;
        *out0_ptr++ = *din3_ptr;
        din0_ptr += stride_w;
        din1_ptr += stride_w;
        din2_ptr += stride_w;
        din3_ptr += stride_w;
      }
      for (int i = 0; i < remain; i++) {
        *out0_ptr++ = *din0_ptr;
        din0_ptr += size;
      }
    }
  }
}
template <>
void NCHW2NHWC<int8_t>(int N, int C, int size, const int8_t* X, int8_t* Y) {
  int cnt = C >> 3;
  int remain = C % 8;
  int sum = C * size;
  int stride = size << 3;    // 8 * size
  int stride_w = size << 4;  // 4 * size * 4
  for (int n = 0; n < N; n++) {
    const int8_t* din = X + n * sum;
    int8_t* dout = Y + n * sum;
    int s = 0;
    LITE_PARALLEL_COMMON_BEGIN(s, tid, size - 7, 0, 8) {
      const int8_t* din0_ptr = din + s;
      const int8_t* din1_ptr = din0_ptr + size;
      const int8_t* din2_ptr = din1_ptr + size;
      const int8_t* din3_ptr = din2_ptr + size;
      int8_t* out0_ptr = dout + s * C;
      int8_t* out1_ptr = out0_ptr + C;
      int8_t* out2_ptr = out1_ptr + C;
      int8_t* out3_ptr = out2_ptr + C;
      int8_t* out4_ptr = out3_ptr + C;
      int8_t* out5_ptr = out4_ptr + C;
      int8_t* out6_ptr = out5_ptr + C;
      int8_t* out7_ptr = out6_ptr + C;
      int cnt_num = cnt;
      if (cnt_num > 0) {
#ifdef __aarch64__
        asm volatile(TRANS_C8
                     : [din0_ptr] "+r"(din0_ptr),
                       [din1_ptr] "+r"(din1_ptr),
                       [din2_ptr] "+r"(din2_ptr),
                       [din3_ptr] "+r"(din3_ptr),
                       [out0_ptr] "+r"(out0_ptr),
                       [out1_ptr] "+r"(out1_ptr),
                       [out2_ptr] "+r"(out2_ptr),
                       [out3_ptr] "+r"(out3_ptr),
                       [out4_ptr] "+r"(out4_ptr),
                       [out5_ptr] "+r"(out5_ptr),
                       [out6_ptr] "+r"(out6_ptr),
                       [out7_ptr] "+r"(out7_ptr),
                       [cnt] "+r"(cnt_num),
                       [stride_w] "+r"(stride_w)
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
#if 0  // TOOD(ysh329): caused assembly code error with register for armv7
       // **clang** compile
        asm volatile(TRANS_C8
                     : [din0_ptr] "+r"(din0_ptr),
                       [din1_ptr] "+r"(din1_ptr),
                       [din2_ptr] "+r"(din2_ptr),
                       [din3_ptr] "+r"(din3_ptr),
                       [out0_ptr] "+r"(out0_ptr),
                       [out1_ptr] "+r"(out1_ptr),
                       [out2_ptr] "+r"(out2_ptr),
                       [out3_ptr] "+r"(out3_ptr),
                       [out4_ptr] "+r"(out4_ptr),
                       [out5_ptr] "+r"(out5_ptr),
                       [out6_ptr] "+r"(out6_ptr),
                       [out7_ptr] "+r"(out7_ptr),
                       [cnt] "+r"(cnt_num),
                       [stride_w] "+r"(stride_w)
                     :
                     : "cc", "memory", "q0", "q1", "q2", "q3");
#endif
#endif
      }
      // const int8_t* din_ptr = din + 8 * cnt * size + s; // remain channel
      for (int i = 0; i < remain; i++) {
        const int8_t* ptr = din0_ptr;
        *out0_ptr = *ptr++;
        *out1_ptr = *ptr++;
        *out2_ptr = *ptr++;
        *out3_ptr = *ptr++;
        din0_ptr += size;
        *out4_ptr = *ptr++;
        *out5_ptr = *ptr++;
        *out6_ptr = *ptr++;
        *out7_ptr = *ptr++;
      }
    }
    LITE_PARALLEL_COMMON_END()
    // remain size
    for (; s < size; s++) {
      const int8_t* din0_ptr = din + s;
      const int8_t* din1_ptr = din0_ptr + size;
      const int8_t* din2_ptr = din1_ptr + size;
      const int8_t* din3_ptr = din2_ptr + size;
      const int8_t* din4_ptr = din3_ptr + size;
      const int8_t* din5_ptr = din4_ptr + size;
      const int8_t* din6_ptr = din5_ptr + size;
      const int8_t* din7_ptr = din6_ptr + size;
      int8_t* out0_ptr = dout + s * C;
      for (int i = 0; i < cnt; i++) {
        *out0_ptr++ = *din0_ptr;
        *out0_ptr++ = *din1_ptr;
        *out0_ptr++ = *din2_ptr;
        *out0_ptr++ = *din3_ptr;
        *out0_ptr++ = *din4_ptr;
        *out0_ptr++ = *din5_ptr;
        *out0_ptr++ = *din6_ptr;
        *out0_ptr++ = *din7_ptr;
        din0_ptr += stride;
        din1_ptr += stride;
        din2_ptr += stride;
        din3_ptr += stride;
        din4_ptr += stride;
        din5_ptr += stride;
        din6_ptr += stride;
        din7_ptr += stride;
      }
      for (int i = 0; i < remain; i++) {
        *out0_ptr++ = *din0_ptr;
        din0_ptr += size;
      }
    }
  }
}
template <>
void NHWC2NCHW<float>(int N, int C, int size, const float* X, float* Y) {
  int cnt = size >> 2;
  int remain = size % 4;
  int sum = C * size;
  int stride = C << 4;  // 4 * size
  int stride_w = C << 2;
  for (int n = 0; n < N; n++) {
    const float* din = X + n * sum;
    float* dout = Y + n * sum;
    int s = 0;
    LITE_PARALLEL_COMMON_BEGIN(s, tid, C - 3, 0, 4) {
      const float* din0_ptr = din + s;
      const float* din1_ptr = din0_ptr + C;
      const float* din2_ptr = din1_ptr + C;
      const float* din3_ptr = din2_ptr + C;
      float* out0_ptr = dout + s * size;
      float* out1_ptr = out0_ptr + size;
      float* out2_ptr = out1_ptr + size;
      float* out3_ptr = out2_ptr + size;
      int cnt_num = cnt;
      if (cnt_num > 0) {
#ifdef __aarch64__
        asm volatile(TRANS_C4
                     : [din0_ptr] "+r"(din0_ptr),
                       [din1_ptr] "+r"(din1_ptr),
                       [din2_ptr] "+r"(din2_ptr),
                       [din3_ptr] "+r"(din3_ptr),
                       [out0_ptr] "+r"(out0_ptr),
                       [out1_ptr] "+r"(out1_ptr),
                       [out2_ptr] "+r"(out2_ptr),
                       [out3_ptr] "+r"(out3_ptr),
                       [cnt] "+r"(cnt_num),
                       [stride] "+r"(stride)
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
#else
#if 0  // TOOD(ysh329): caused assembly code error with register for armv7
       // **clang** compile
        asm volatile(TRANS_C4
                     : [din0_ptr] "+r"(din0_ptr),
                       [din1_ptr] "+r"(din1_ptr),
                       [din2_ptr] "+r"(din2_ptr),
                       [din3_ptr] "+r"(din3_ptr),
                       [out0_ptr] "+r"(out0_ptr),
                       [out1_ptr] "+r"(out1_ptr),
                       [out2_ptr] "+r"(out2_ptr),
                       [out3_ptr] "+r"(out3_ptr),
                       [cnt] "+r"(cnt_num),
                       [stride] "+r"(stride)
                     :
                     : "cc", "memory", "q0", "q1", "q2", "q3");
#endif
#endif
      }
      for (int i = 0; i < remain; i++) {
        const float* ptr = din0_ptr;
        *out0_ptr++ = *ptr++;
        *out1_ptr++ = *ptr++;
        *out2_ptr++ = *ptr++;
        *out3_ptr++ = *ptr++;
        din0_ptr += C;
      }
    }
    LITE_PARALLEL_COMMON_END()
    // remain size
    for (; s < C; s++) {
      const float* din0_ptr = din + s;
      const float* din1_ptr = din0_ptr + C;
      const float* din2_ptr = din1_ptr + C;
      const float* din3_ptr = din2_ptr + C;
      float* out0_ptr = dout + s * size;
      for (int i = 0; i < cnt; i++) {
        *out0_ptr++ = *din0_ptr;
        *out0_ptr++ = *din1_ptr;
        *out0_ptr++ = *din2_ptr;
        *out0_ptr++ = *din3_ptr;
        din0_ptr += stride_w;
        din1_ptr += stride_w;
        din2_ptr += stride_w;
        din3_ptr += stride_w;
      }
      for (int i = 0; i < remain; i++) {
        *out0_ptr++ = *din0_ptr;
        din0_ptr += C;
      }
    }
  }
}
template <>
void NHWC2NCHW<int8_t>(int N, int C, int size, const int8_t* X, int8_t* Y) {
  int cnt = size >> 3;
  int remain = size % 8;
  int sum = C * size;
  int stride = C << 3;    // 8 * size
  int stride_w = C << 4;  // 4 * size
  for (int n = 0; n < N; n++) {
    const int8_t* din = X + n * sum;
    int8_t* dout = Y + n * sum;
    int s = 0;
    LITE_PARALLEL_COMMON_BEGIN(s, tid, C - 7, 0, 8) {
      const int8_t* din0_ptr = din + s;
      const int8_t* din1_ptr = din0_ptr + C;
      const int8_t* din2_ptr = din1_ptr + C;
      const int8_t* din3_ptr = din2_ptr + C;
      const int8_t* din4_ptr = din3_ptr + C;
      const int8_t* din5_ptr = din4_ptr + C;
      const int8_t* din6_ptr = din5_ptr + C;
      const int8_t* din7_ptr = din6_ptr + C;
      int8_t* out0_ptr = dout + s * size;
      int8_t* out1_ptr = out0_ptr + size;
      int8_t* out2_ptr = out1_ptr + size;
      int8_t* out3_ptr = out2_ptr + size;
      int8_t* out4_ptr = out3_ptr + size;
      int8_t* out5_ptr = out4_ptr + size;
      int8_t* out6_ptr = out5_ptr + size;
      int8_t* out7_ptr = out6_ptr + size;
      int cnt_num = cnt;
      if (cnt_num > 0) {
#ifdef __aarch64__
        asm volatile(TRANS_C8
                     : [din0_ptr] "+r"(din0_ptr),
                       [din1_ptr] "+r"(din1_ptr),
                       [din2_ptr] "+r"(din2_ptr),
                       [din3_ptr] "+r"(din3_ptr),
                       [out0_ptr] "+r"(out0_ptr),
                       [out1_ptr] "+r"(out1_ptr),
                       [out2_ptr] "+r"(out2_ptr),
                       [out3_ptr] "+r"(out3_ptr),
                       [out4_ptr] "+r"(out4_ptr),
                       [out5_ptr] "+r"(out5_ptr),
                       [out6_ptr] "+r"(out6_ptr),
                       [out7_ptr] "+r"(out7_ptr),
                       [cnt] "+r"(cnt_num),
                       [stride_w] "+r"(stride_w)
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
#if 0  // TOOD(ysh329): caused assembly code error with register for armv7
       // **clang** compile
        asm volatile(TRANS_C8
                     : [din0_ptr] "+r"(din0_ptr),
                       [din1_ptr] "+r"(din1_ptr),
                       [din2_ptr] "+r"(din2_ptr),
                       [din3_ptr] "+r"(din3_ptr),
                       [out0_ptr] "+r"(out0_ptr),
                       [out1_ptr] "+r"(out1_ptr),
                       [out2_ptr] "+r"(out2_ptr),
                       [out3_ptr] "+r"(out3_ptr),
                       [out4_ptr] "+r"(out4_ptr),
                       [out5_ptr] "+r"(out5_ptr),
                       [out6_ptr] "+r"(out6_ptr),
                       [out7_ptr] "+r"(out7_ptr),
                       [cnt] "+r"(cnt_num),
                       [stride_w] "+r"(stride_w)
                     :
                     : "cc", "memory", "q0", "q1", "q2", "q3");
#endif
#endif
      }
      for (int i = 0; i < remain; i++) {
        const int8_t* ptr = din0_ptr;
        *out0_ptr++ = *ptr++;
        *out1_ptr++ = *ptr++;
        *out2_ptr++ = *ptr++;
        *out3_ptr++ = *ptr++;
        *out4_ptr++ = *ptr++;
        *out5_ptr++ = *ptr++;
        *out6_ptr++ = *ptr++;
        *out7_ptr++ = *ptr++;
        din0_ptr += C;
      }
    }
    LITE_PARALLEL_COMMON_END()
    // remain size
    for (; s < C; s++) {
      const int8_t* din0_ptr = din + s;
      const int8_t* din1_ptr = din0_ptr + C;
      const int8_t* din2_ptr = din1_ptr + C;
      const int8_t* din3_ptr = din2_ptr + C;
      const int8_t* din4_ptr = din3_ptr + C;
      const int8_t* din5_ptr = din4_ptr + C;
      const int8_t* din6_ptr = din5_ptr + C;
      const int8_t* din7_ptr = din6_ptr + C;
      int8_t* out0_ptr = dout + s * size;
      for (int i = 0; i < cnt; i++) {
        *out0_ptr++ = *din0_ptr;
        *out0_ptr++ = *din1_ptr;
        *out0_ptr++ = *din2_ptr;
        *out0_ptr++ = *din3_ptr;
        *out0_ptr++ = *din4_ptr;
        *out0_ptr++ = *din5_ptr;
        *out0_ptr++ = *din6_ptr;
        *out0_ptr++ = *din7_ptr;
        din0_ptr += stride;
        din1_ptr += stride;
        din2_ptr += stride;
        din3_ptr += stride;
        din4_ptr += stride;
        din5_ptr += stride;
        din6_ptr += stride;
        din7_ptr += stride;
      }
      for (int i = 0; i < remain; i++) {
        *out0_ptr++ = *din0_ptr;
        din0_ptr += C;
      }
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
