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
  int cnt_rem = remain >> 3; \
  int cnt_rem = remain & 7;

#ifdef __aarch64__
#define INIT_1                   \
  "ldr q0, [%[dinx_ptr]], #8 \n" \
  "ldr q4, [%[diny_ptr]], #8 \n" \
  "1: \n"

#define ADD_COMPUTE_1            \
  "fadd v8.8h, v0.8h, v4.8h  \n" \
  "subs %w[cnt], %w[cnt], #1 \n" \
  "ldr q0, [%[dinx_ptr]], #8 \n" \
  "ldr q4, [%[diny_ptr]], #8 \n"

#define RELU_1 "fmax v8.8h, v8.8h, %[vzero].8h\n"

#define STORE_1                   \
  "str q8, [%[dout_ptr]], #8  \n" \
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

#define ADD_COMPUTE              \
  "fadd v8.8h, v0.8h, v4.8h  \n" \
  "fadd v9.8h, v1.8h, v5.8h  \n" \
  "fadd v10.8h, v2.8h, v6.8h \n" \
  "fadd v11.8h, v3.8h, v7.8h \n"

#define RELU                           \
  "fmax v8.8h, v8.8h, %[vzero].8h\n"   \
  "fmax v9.8h, v9.8h, %[v[zero].8h\n"  \
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
  //   int cnt = num >> 5;
  //   int remain = num & 31;
  //   int cnt_rem = remain >> 3;
  //   int cnt_rem = remain & 7;
  LOOP_CNT(num)
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    int stride = i << 5;
    const float16_t* dinx_ptr = dinx + stride;
    const float16_t* diny_ptr = diny + stride;
    float16_t* dout_ptr = dout + stride;
    asm volatile(INIT COMPUTE STORE
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

    // float16x8_t dinx0 = vld1q_f16(dinx_ptr);
    // float16x8_t diny0 = vld1q_f16(diny_ptr);
    // float16x8_t dinx1 = vld1q_f16(dinx_ptr + 8);
    // float16x8_t diny1 = vld1q_f16(diny_ptr + 8);
    // float16x8_t dinx2 = vld1q_f16(dinx_ptr + 16);
    // float16x8_t diny2 = vld1q_f16(diny_ptr + 16);
    // float16x8_t dinx3 = vld1q_f16(dinx_ptr + 24);
    // float16x8_t diny3 = vld1q_f16(diny_ptr + 24);

    // dinx0 = vaddq_f16(dinx0, diny0);
    // dinx1 = vaddq_f16(dinx1, diny1);
    // dinx2 = vaddq_f16(dinx2, diny2);
    // dinx3 = vaddq_f16(dinx3, diny3);

    // vst1q_f16(dout_ptr, dinx0);
    // vst1q_f16(dout_ptr + 8, dinx1);
    // vst1q_f16(dout_ptr + 16, dinx2);
    // vst1q_f16(dout_ptr + 24, dinx3);
  }
  int stride = cnt << 5;
  if (cnt_rem > 0) {
    const float16_t* dinx_ptr = dinx + stride;
    const float16_t* diny_ptr = diny + stride;
    float16_t* dout_ptr = dout + stride;
    int cnt_num = cnt_rem;
    asm volatile(INIT_1 COMPUTE_1 STORE_1
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
    // for (int i = 0; i < cnt_rem; i++) {
    //   float16x8_t dinx0 = vld1q_f16(dinx_ptr);
    //   float16x8_t diny0 = vld1q_f16(diny_ptr);
    //   dinx_ptr += 8;
    //   diny_ptr += 8;
    //   dinx0 = vaddq_f16(dinx0, diny0);
    //   vst1q_f16(dout_ptr, dinx0);
    //   dout_ptr += 8;
    // }
  }
  if (cnt_rem > 0) {
    stride += (cnt_rem << 3);
    const float* dinx_ptr = dinx + stride;
    const float* diny_ptr = diny + stride;
    float* dout_ptr = dout + stride;
    for (int i = 0; i < cnt_rem; i++) {
      *dout_ptr = *dinx_ptr + *diny_ptr;
      dout_ptr++;
      dinx_ptr++;
      diny_ptr++;
    }
  }
}

template <>
void elementwise_add_relu<float>(const float* dinx,
                                 const float* diny,
                                 float* dout,
                                 int num) {
  LOOP_CNT(num)
  float16x8_t vzero = vdupq_n_f16(0.f);
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    int stride = i << 5;
    const float16_t* dinx_ptr = dinx + stride;
    const float16_t* diny_ptr = diny + stride;
    float16_t* dout_ptr = dout + stride;
    asm volatile(INIT COMPUTE RELU STORE
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
  if (cnt_rem > 0) {
    const float16_t* dinx_ptr = dinx + stride;
    const float16_t* diny_ptr = diny + stride;
    float16_t* dout_ptr = dout + stride;
    int cnt_num = cnt_rem;
    asm volatile(INIT_1 COMPUTE_1 RELU_1 STORE_1
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
  if (cnt_rem > 0) {
    stride += (cnt_rem << 3);
    const float* dinx_ptr = dinx + stride;
    const float* diny_ptr = diny + stride;
    float* dout_ptr = dout + stride;
    for (int i = 0; i < cnt_rem; i++) {
      float16_t tmp_val = *dinx_ptr + *diny_ptr;
      dinx_ptr++;
      diny_ptr++;
      *dout_ptr++ = tmp_val > 0.f ? tmp_val : 0.f;
    }
  }
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
