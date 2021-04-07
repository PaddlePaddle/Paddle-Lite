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

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
