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
#include "lite/backends/arm/math/fp16/elementwise_fp16.h"
#include <algorithm>
#include "lite/backends/arm/math/fp16/funcs_fp16.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
#define LOOP_CNT(num)        \
  int cnt = num >> 5;        \
  int remain = num & 31;     \
  int rem_cnt = remain >> 3; \
  int rem_rem = remain & 7;

#ifdef __aarch64__
#define INIT_1                    \
  "ldr q0, [%[dinx_ptr]], #16 \n" \
  "ldr q4, [%[diny_ptr]], #16 \n" \
  "1: \n"

#define ADD_COMPUTE_1                   \
  "fadd v8.8h, v0.8h, v4.8h  \n"        \
  "subs %w[cnt_num], %w[cnt_num], #1\n" \
  "ldr q0, [%[dinx_ptr]], #16 \n"       \
  "ldr q4, [%[diny_ptr]], #16 \n"

#define INIT_1_BROADCAST          \
  "ldr q0, [%[dinx_ptr]], #16 \n" \
  "1: \n"

#define ADD_COMPUTE_1_BROADCAST         \
  "fadd v8.8h, v0.8h, %[val_y].8h\n"    \
  "subs %w[cnt_num], %w[cnt_num], #1\n" \
  "ldr q0, [%[dinx_ptr]], #16 \n"

#define RELU_1 "fmax v8.8h, v8.8h, %[vzero].8h\n"
#define STORE_1                    \
  "str q8, [%[dout_ptr]], #16  \n" \
  "bne 1b\n"

#define INIT                     \
  "ldr q0, [%[dinx_ptr]]     \n" \
  "ldr q4, [%[diny_ptr]]     \n" \
  "ldr q1, [%[dinx_ptr], #16]\n" \
  "ldr q5, [%[diny_ptr], #16]\n" \
  "ldr q2, [%[dinx_ptr], #32]\n" \
  "ldr q6, [%[diny_ptr], #32]\n" \
  "ldr q3, [%[dinx_ptr], #48]\n" \
  "ldr q7, [%[diny_ptr], #48]\n"

#define ADD_COMPUTE             \
  "fadd v8.8h, v0.8h, v4.8h\n"  \
  "fadd v9.8h, v1.8h, v5.8h\n"  \
  "fadd v10.8h, v2.8h, v6.8h\n" \
  "fadd v11.8h, v3.8h, v7.8h\n"

#define INIT_BROADCAST           \
  "ldr q0, [%[dinx_ptr]]     \n" \
  "ldr q1, [%[dinx_ptr], #16]\n" \
  "ldr q2, [%[dinx_ptr], #32]\n" \
  "ldr q3, [%[dinx_ptr], #48]\n"

#define ADD_COMPUTE_BROADCAST         \
  "fadd v8.8h, v0.8h, %[val_y].8h\n"  \
  "fadd v9.8h, v1.8h, %[val_y].8h\n"  \
  "fadd v10.8h, v2.8h, %[val_y].8h\n" \
  "fadd v11.8h, v3.8h, %[val_y].8h\n"

#define RELU                           \
  "fmax v8.8h, v8.8h, %[vzero].8h\n"   \
  "fmax v9.8h, v9.8h, %[vzero].8h\n"   \
  "fmax v10.8h, v10.8h, %[vzero].8h\n" \
  "fmax v11.8h, v11.8h, %[vzero].8h\n"

#define STORE                     \
  "str q8, [%[dout_ptr]]      \n" \
  "str q9, [%[dout_ptr], #16] \n" \
  "str q10, [%[dout_ptr], #32]\n" \
  "str q11, [%[dout_ptr], #48]\n"
#endif
template <>
void elementwise_add<float16_t>(const float16_t* dinx,
                                const float16_t* diny,
                                float16_t* dout,
                                int num) {
  LOOP_CNT(num)
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    int stride = i << 5;
    const float16_t* dinx_ptr = dinx + stride;
    const float16_t* diny_ptr = diny + stride;
    float16_t* dout_ptr = dout + stride;
    asm volatile(INIT ADD_COMPUTE STORE
                 :
                 : [dinx_ptr] "r"(dinx_ptr),
                   [diny_ptr] "r"(diny_ptr),
                   [dout_ptr] "r"(dout_ptr)
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
  int stride = cnt << 5;
  if (rem_cnt > 0) {
    const float16_t* dinx_ptr = dinx + stride;
    const float16_t* diny_ptr = diny + stride;
    float16_t* dout_ptr = dout + stride;
    int cnt_num = rem_cnt;
    asm volatile(INIT_1 ADD_COMPUTE_1 STORE_1
                 : [cnt_num] "+r"(cnt_num),
                   [dinx_ptr] "+r"(dinx_ptr),
                   [diny_ptr] "+r"(diny_ptr),
                   [dout_ptr] "+r"(dout_ptr)
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
  if (rem_rem > 0) {
    stride += (rem_cnt << 3);
    const float16_t* dinx_ptr = dinx + stride;
    const float16_t* diny_ptr = diny + stride;
    float16_t* dout_ptr = dout + stride;
    for (int i = 0; i < rem_rem; i++) {
      *dout_ptr = *dinx_ptr + *diny_ptr;
      dout_ptr++;
      dinx_ptr++;
      diny_ptr++;
    }
  }
}

template <>
void elementwise_add_relu<float16_t>(const float16_t* dinx,
                                     const float16_t* diny,
                                     float16_t* dout,
                                     int num) {
  LOOP_CNT(num)
  float16x8_t vzero = vdupq_n_f16(0.f);
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    int stride = i << 5;
    const float16_t* dinx_ptr = dinx + stride;
    const float16_t* diny_ptr = diny + stride;
    float16_t* dout_ptr = dout + stride;
    asm volatile(INIT ADD_COMPUTE RELU STORE
                 :
                 : [dinx_ptr] "r"(dinx_ptr),
                   [diny_ptr] "r"(diny_ptr),
                   [dout_ptr] "r"(dout_ptr),
                   [vzero] "w"(vzero)
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
  int stride = cnt << 5;
  if (rem_cnt > 0) {
    const float16_t* dinx_ptr = dinx + stride;
    const float16_t* diny_ptr = diny + stride;
    float16_t* dout_ptr = dout + stride;
    int cnt_num = rem_cnt;
    asm volatile(INIT_1 ADD_COMPUTE_1 RELU_1 STORE_1
                 : [cnt_num] "+r"(cnt_num),
                   [dinx_ptr] "+r"(dinx_ptr),
                   [diny_ptr] "+r"(diny_ptr),
                   [dout_ptr] "+r"(dout_ptr)
                 : [vzero] "w"(vzero)
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
  if (rem_rem > 0) {
    stride += (rem_cnt << 3);
    const float16_t* dinx_ptr = dinx + stride;
    const float16_t* diny_ptr = diny + stride;
    float16_t* dout_ptr = dout + stride;
    for (int i = 0; i < rem_rem; i++) {
      float16_t tmp_val = *dinx_ptr + *diny_ptr;
      dinx_ptr++;
      diny_ptr++;
      *dout_ptr++ = tmp_val > 0.f ? tmp_val : 0.f;
    }
  }
}

template <>
void elementwise_add_broadcast<float16_t>(const float16_t* dinx,
                                          const float16_t* diny,
                                          float16_t* dout,
                                          int batch,
                                          int channels,
                                          int num) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const auto* dinx_ptr = dinx + offset;
      const auto* diny_ptr = diny + j;
      auto* dout_ptr = dout + offset;

      LOOP_CNT(num)
      for (int k = 0; k < cnt; k++) {
        int stride = k << 5;
        const float16_t* dinx_ptr_1 = dinx_ptr + stride;
        float16_t* dout_ptr_1 = dout_ptr + stride;
        float16x8_t val_y = vdupq_n_f16(diny_ptr[stride]);
        asm volatile(INIT_BROADCAST ADD_COMPUTE_BROADCAST STORE
                     :
                     : [dinx_ptr] "r"(dinx_ptr_1),
                       [dout_ptr] "r"(dout_ptr_1),
                       [val_y] "w"(val_y)
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
      int stride = cnt << 5;
      if (rem_cnt > 0) {
        const float16_t* dinx_ptr_1 = dinx_ptr + stride;
        float16_t* dout_ptr_1 = dout_ptr + stride;
        float16x8_t val_y = vdupq_n_f16(diny_ptr[stride]);
        int cnt_num = rem_cnt;
        asm volatile(INIT_1_BROADCAST ADD_COMPUTE_1_BROADCAST STORE_1
                     : [cnt_num] "+r"(cnt_num),
                       [dinx_ptr] "+r"(dinx_ptr_1),
                       [dout_ptr] "+r"(dout_ptr_1)
                     : [val_y] "w"(val_y)
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
      if (rem_rem > 0) {
        stride += (rem_cnt << 3);
        const float16_t* dinx_ptr_1 = dinx_ptr + stride;
        float16_t* dout_ptr_1 = dout_ptr + stride;
        float16_t val = diny_ptr[stride];
        for (int i = 0; i < rem_rem; i++) {
          *dout_ptr_1 = *dinx_ptr_1 + val;
          dinx_ptr_1++;
          dout_ptr_1++;
        }
      }
    }
  }
}

template <>
void elementwise_add_relu_broadcast<float16_t>(const float16_t* dinx,
                                               const float16_t* diny,
                                               float16_t* dout,
                                               int batch,
                                               int channels,
                                               int num) {
  float16x8_t vzero = vdupq_n_f16(0.f);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int offset = (i * channels + j) * num;
      const auto* dinx_ptr = dinx + offset;
      const auto* diny_ptr = diny + j;
      auto* dout_ptr = dout + offset;

      LOOP_CNT(num)
      for (int k = 0; k < cnt; k++) {
        int stride = k << 5;
        const float16_t* dinx_ptr_1 = dinx_ptr + stride;
        float16_t* dout_ptr_1 = dout_ptr + stride;
        float16x8_t val_y = vdupq_n_f16(diny_ptr[stride]);
        asm volatile(INIT_BROADCAST ADD_COMPUTE_BROADCAST RELU STORE
                     :
                     : [dinx_ptr] "r"(dinx_ptr_1),
                       [dout_ptr] "r"(dout_ptr_1),
                       [val_y] "w"(val_y),
                       [vzero] "w"(vzero)
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
      int stride = cnt << 5;
      if (rem_cnt > 0) {
        const float16_t* dinx_ptr_1 = dinx_ptr + stride;
        float16_t* dout_ptr_1 = dout_ptr + stride;
        float16x8_t val_y = vdupq_n_f16(diny_ptr[stride]);
        int cnt_num = rem_cnt;
        asm volatile(INIT_1_BROADCAST ADD_COMPUTE_1_BROADCAST RELU_1 STORE_1
                     : [cnt_num] "+r"(cnt_num),
                       [dinx_ptr] "+r"(dinx_ptr_1),
                       [dout_ptr] "+r"(dout_ptr_1)
                     : [vzero] "w"(vzero), [val_y] "w"(val_y)
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
      if (rem_rem > 0) {
        stride += (rem_cnt << 3);
        const float16_t* dinx_ptr_1 = dinx_ptr + stride;
        float16_t* dout_ptr_1 = dout_ptr + stride;
        float16_t val = diny_ptr[stride];
        for (int i = 0; i < rem_rem; i++) {
          float16_t tmp_val = *dinx_ptr_1 + val;
          dinx_ptr_1++;
          *dout_ptr_1++ = tmp_val > 0.f ? tmp_val : 0.f;
        }
      }
    }
  }
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
