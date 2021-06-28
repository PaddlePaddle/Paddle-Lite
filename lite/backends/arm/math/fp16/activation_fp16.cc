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

#include "lite/backends/arm/math/fp16/activation_fp16.h"
#include <algorithm>
#include "lite/backends/arm/math/fp16/funcs_fp16.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

template <>
void act_relu<float16_t>(const float16_t* din,
                         float16_t* dout,
                         int size,
                         int threads) {
  int nums_per_thread = size / threads;
  int remain = size - threads * nums_per_thread;
  int neon_loop_cnt = nums_per_thread >> 5;
  int neon_loop_rem = nums_per_thread & 31;
  int neon_loop_rem_cnt = neon_loop_rem >> 3;
  int neon_loop_rem_rem = neon_loop_rem & 7;
  int stride = neon_loop_rem_cnt << 3;
  float16x8_t vzero = vdupq_n_f16(0.f);
#pragma omp parallel for
  for (int i = 0; i < threads; ++i) {
    const float16_t* ptr_in_thread = din + i * nums_per_thread;
    float16_t* ptr_out_thread = dout + i * nums_per_thread;
    int cnt = neon_loop_cnt;
    if (neon_loop_cnt > 0 || neon_loop_rem_cnt > 0) {
#ifdef __aarch64__
      asm volatile(
          "cmp %w[cnt], #1          \n"
          "ldr q0, [%[din_ptr]], #16\n"
          "ldr q1, [%[din_ptr]], #16\n"
          "ldr q2, [%[din_ptr]], #16\n"
          "ldr q3, [%[din_ptr]], #16\n"
          "blt 0f\n"
          "1: \n"
          "fmax v4.8h, v0.8h, %[vzero].8h\n"
          "fmax v5.8h, v1.8h, %[vzero].8h\n"
          "ldr q0, [%[din_ptr]], #16\n"
          "fmax v6.8h, v2.8h, %[vzero].8h\n"
          "ldr q1, [%[din_ptr]], #16\n"
          "fmax v7.8h, v3.8h, %[vzero].8h\n"
          "subs %w[cnt], %w[cnt], #1\n"
          "ldr q2, [%[din_ptr]], #16\n"
          "ldr q3, [%[din_ptr]], #16\n"
          "stp q4, q5, [%[dout_ptr]], #32\n"
          "stp q6, q7, [%[dout_ptr]], #32\n"
          "bne 1b\n"
          "0: \n"
          "cmp %w[rem_cnt], #0\n"
          "beq 2f\n"
          "cmp %w[rem_cnt], #1\n"
          "beq 3f\n"
          "cmp %w[rem_cnt], #2\n"
          "beq 4f\n"
          "cmp %w[rem_cnt], #3\n"
          "beq 5f\n"
          "3: \n"
          "fmax v4.8h, v0.8h, %[vzero].8h\n"
          "str q4, [%[dout_ptr]], #16\n"
          "b 2f\n"
          "4: \n"
          "fmax v4.8h, v0.8h, %[vzero].8h\n"
          "fmax v5.8h, v1.8h, %[vzero].8h\n"
          "stp q4, q5, [%[dout_ptr]], #32\n"
          "b 2f\n"
          "5: \n"
          "fmax v4.8h, v0.8h, %[vzero].8h\n"
          "fmax v5.8h, v1.8h, %[vzero].8h\n"
          "fmax v6.8h, v2.8h, %[vzero].8h\n"
          "stp q4, q5, [%[dout_ptr]], #32\n"
          "str q6, [%[dout_ptr]], #16\n"
          "2: \n"
          : [din_ptr] "+r"(ptr_in_thread),
            [dout_ptr] "+r"(ptr_out_thread),
            [cnt] "+r"(cnt)
          : [rem_cnt] "r"(neon_loop_rem_cnt), [vzero] "w"(vzero)
          : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#else
#endif
    }
    ptr_in_thread -= stride;
    for (int j = 0; j < neon_loop_rem_rem; ++j) {
      ptr_out_thread[0] = ptr_in_thread[0] > 0.f ? ptr_in_thread[0] : 0.f;
      ptr_in_thread++;
      ptr_out_thread++;
    }
  }
  float16_t* out_ptr_remain = dout + threads * nums_per_thread;
  const float16_t* in_ptr_remain = din + threads * nums_per_thread;
  for (int j = 0; j < remain; ++j) {
    out_ptr_remain[0] = in_ptr_remain[0] > 0.f ? in_ptr_remain[0] : 0.f;
    in_ptr_remain++;
    out_ptr_remain++;
  }
}

template <>
void act_hard_sigmoid<float16_t>(const float16_t* din,
                                 float16_t* dout,
                                 const int size,
                                 const float slope,
                                 const float offset,
                                 int threads) {
  int cnt = size >> 5;
  int remain = size & 31;

  int cnt_4 = remain >> 3;
  int remain_4 = remain & 7;

  float16x8_t vzero_8 = vdupq_n_f16(float16_t(0));
  float16x8_t vone_8 = vdupq_n_f16(float16_t(1));
  float16x8_t vslope_8 = vdupq_n_f16(float16_t(slope));
  float16x8_t voffset_8 = vdupq_n_f16(float16_t(offset));
#ifdef __aarch64__
  asm volatile(
      "cmp %w[cnt], #1          \n"
      "ldr q0, [%[din_ptr]], #16\n"
      "ldr q1, [%[din_ptr]], #16\n"
      "ldr q2, [%[din_ptr]], #16\n"
      "ldr q3, [%[din_ptr]], #16\n"
      "blt 0f\n"
      "1: \n"
      "fmul v4.8h, v0.8h, %[vslope_8].8h\n"
      "fmul v5.8h, v1.8h, %[vslope_8].8h\n"
      "fmul v6.8h, v2.8h, %[vslope_8].8h\n"
      "fmul v7.8h, v3.8h, %[vslope_8].8h\n"
      "fadd v0.8h, v4.8h, %[voffset_8].8h\n"
      "fadd v1.8h, v5.8h, %[voffset_8].8h\n"
      "fadd v2.8h, v6.8h, %[voffset_8].8h\n"
      "fadd v3.8h, v7.8h, %[voffset_8].8h\n"
      "fcmgt v4.8h, v0.8h, %[vzero_8].8h\n"
      "fcmgt v5.8h, v1.8h, %[vzero_8].8h\n"
      "fcmgt v6.8h, v2.8h, %[vzero_8].8h\n"
      "fcmgt v7.8h, v3.8h, %[vzero_8].8h\n"

      "bsl v4.16b, v0.16b, %[vzero_8].16b\n"
      "bsl v5.16b, v1.16b, %[vzero_8].16b\n"
      "bsl v6.16b, v2.16b, %[vzero_8].16b\n"
      "bsl v7.16b, v3.16b, %[vzero_8].16b\n"

      "fcmlt v8.8h, v0.8h, %[vone_8].8h\n"
      "fcmlt v9.8h, v1.8h, %[vone_8].8h\n"
      "fcmlt v10.8h, v2.8h, %[vone_8].8h\n"
      "fcmlt v11.8h, v3.8h, %[vone_8].8h\n"

      "ldr q0, [%[din_ptr]], #16\n"
      "bsl v8.16b, v4.16b, %[vzero_8].16b\n"
      "ldr q1, [%[din_ptr]], #16\n"
      "ldr q2, [%[din_ptr]], #16\n"
      "bsl v9.16b, v5.16b, %[vzero_8].16b\n"
      "bsl v10.16b, v6.16b, %[vzero_8].16b\n"
      "ldr q3, [%[din_ptr]], #16\n"
      "bsl v11.16b, v7.16b, %[vzero_8].16b\n"

      "subs %w[cnt], %w[cnt], #1\n"
      "str q8, [%[dout_ptr]], #16\n"
      "str q9, [%[dout_ptr]], #16\n"
      "str q10, [%[dout_ptr]], #16\n"
      "str q11, [%[dout_ptr]], #16\n"
      "bne 1b\n"
      "0: \n"
      "sub %[din_ptr], %[din_ptr], #64\n"
      "cmp %w[cnt_4], #1\n"
      "ld1 {v0.4h}, [%[din_ptr]], #8\n"
      "ld1 {v1.4h}, [%[din_ptr]], #8\n"
      "blt 3f\n"
      "2: \n"
      "fmul v4.4h, v0.4h, %[vslope_8].4h\n"
      "fmul v5.4h, v1.4h, %[vslope_8].4h\n"
      "ld1 {v0.4h}, [%[din_ptr]], #8\n"

      "fadd v2.4h, v4.4h, %[voffset_8].4h\n"
      "fadd v3.4h, v5.4h, %[voffset_8].4h\n"
      "ld1 {v1.4h}, [%[din_ptr]], #8\n"

      "fcmgt v4.4h, v2.4h, %[vzero_8].4h\n"
      "fcmgt v5.4h, v3.4h, %[vzero_8].4h\n"

      "bsl v4.8b, v2.8b, %[vzero_8].8b\n"
      "bsl v5.8b, v3.8b, %[vzero_8].8b\n"

      "fcmlt v8.4h, v4.4h, %[vone_8].4h\n"
      "fcmlt v9.4h, v5.4h, %[vone_8].4h\n"

      "bsl v8.8b, v4.8b, %[vone_8].8b\n"
      "bsl v9.8b, v5.8b, %[vone_8].8b\n"

      "subs %w[cnt_4], %w[cnt_4], #1\n"
      "st1 {v8.4h}, [%[dout_ptr]], #8\n"
      "st1 {v9.4h}, [%[dout_ptr]], #8\n"
      "bne 2b\n"
      "3: \n"
      "sub %[din_ptr], %[din_ptr], #16\n "
      : [din_ptr] "+r"(din),
        [dout_ptr] "+r"(dout),
        [cnt] "+r"(cnt),
        [cnt_4] "+r"(cnt_4)
      : [rem_cnt] "r"(remain_4),
        [vzero_8] "w"(vzero_8),
        [vone_8] "w"(vone_8),
        [vslope_8] "w"(vslope_8),
        [voffset_8] "w"(voffset_8)
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
#endif
  for (int64_t i = 0; i < remain_4; i++) {
    dout[0] = din[0] * slope + offset;
    dout[0] = dout[0] < 1.0f ? dout[0] : 1.0f;
    dout[0] = dout[0] > 0.0f ? dout[0] : 0.0f;
    ++din;
    ++dout;
  }
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
