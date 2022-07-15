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

#include "lite/backends/arm/math/scale.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
inline void scale_compute_fp32(const float* din,
                               float* dout,
                               int cnt,
                               int remain_cnt,
                               int remain_rem,
                               float32x4_t vscale,
                               float32x4_t vbias) {
#ifdef __aarch64__
  asm volatile(
      "cmp %w[cnt], #1                          \n"
      "blt 0f                                   \n"
      "1:                                       \n"
      "ld1  {v4.4s}, [%[din]], #16              \n"
      "and  v8.16b, %[vbias].16b, %[vbias].16b  \n"
      "ld1  {v5.4s}, [%[din]], #16              \n"
      "and  v9.16b, %[vbias].16b, %[vbias].16b  \n"
      "ld1  {v6.4s}, [%[din]], #16              \n"
      "and  v10.16b, %[vbias].16b, %[vbias].16b \n"
      "ld1  {v7.4s}, [%[din]], #16              \n"
      "and  v11.16b, %[vbias].16b, %[vbias].16b \n"

      "fmla v8.4s, v4.4s, %[vscale].4s          \n"
      "fmla v9.4s, v5.4s, %[vscale].4s          \n"
      "fmla v10.4s, v6.4s, %[vscale].4s         \n"
      "fmla v11.4s, v7.4s, %[vscale].4s         \n"

      "stp  q8, q9, [%[dout]], #32              \n"
      "subs %w[cnt], %w[cnt],  #1               \n"
      "stp  q10, q11, [%[dout]], #32            \n"
      "bne    1b                                \n"
      "0:                                       \n"
      "cmp %w[remain_cnt], #1                   \n"
      "blt 2f                                   \n"
      "3:                                       \n"
      "ld1  {v4.4s}, [%[din]], #16              \n"
      "and  v8.16b, %[vbias].16b, %[vbias].16b  \n"
      "fmla v8.4s, v4.4s, %[vscale].4s          \n"
      "subs %w[remain_cnt], %w[remain_cnt],  #1 \n"
      "str q8, [%[dout]], #16                   \n"
      "bne    3b                                \n"
      "2:                                       \n"
      : [dout] "+r"(dout),
        [din] "+r"(din),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_cnt)
      : [vscale] "w"(vscale), [vbias] "w"(vbias)
      : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else
  asm volatile(
      "cmp %[cnt], #1                           \n"
      "blt 0f                                   \n"
      "1:                                     @ loop header \n"
      "vld1.32  {d8-d11}, [%[din]]!           @ load din 0 \n"
      "vand.32 q8, %q[vbias], %q[vbias]       @ out bias \n"
      "vand.32 q9, %q[vbias], %q[vbias]       @ out bias \n"
      "vld1.32  {d12-d15}, [%[din]]!          @ load din 0 \n"
      "vand.32 q10, %q[vbias], %q[vbias]      @ out bias \n"
      "vand.32 q11, %q[vbias], %q[vbias]      @ out bias \n"

      "vmla.f32 q8, q4, %q[vscale]            @ mla \n"
      "vmla.f32 q9, q5, %q[vscale]            @ mla \n"
      "vmla.f32 q10, q6, %q[vscale]           @ mla \n"
      "vmla.f32 q11, q7, %q[vscale]           @ mla \n"

      "vst1.32  {d16-d19}, [%[dout]]!         @ store result, add pointer\n"
      "subs %[cnt], #1                        @ loop count minus 1\n"
      "vst1.32  {d20-d23}, [%[dout]]!         @ store result, add pointer\n"
      "bne    1b                              @ jump to main loop start \n"
      "0:                                       \n"
      "cmp %[remain_cnt], #1                    \n"
      "blt 2f                                   \n"
      "3:                                       \n"
      "vld1.32  {d8-d9}, [%[din]]!            @ load din 0 \n"
      "vand.32 q8, %q[vbias], %q[vbias]       @ out bias \n"
      "vmla.f32 q8, q4, %q[vscale]            @ mla \n"
      "subs %[remain_cnt],  #1                  \n"
      "vst1.32  {d16-d17}, [%[dout]]!           \n"
      "bne    3b                                \n"
      "2:                                       \n"
      : [dout] "+r"(dout),
        [din] "+r"(din),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_cnt)
      : [vscale] "w"(vscale), [vbias] "w"(vbias)
      : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
  for (int i = 0; i < remain_rem; i++) {
    *dout = *din * vscale[0] + vbias[0];
    dout++;
    din++;
  }
}
template <>
void scale<float>(
    const float* din, float* dout, int num, float scale, float bias) {
  int cnt = num >> 4;
  int remain = num & 15;
  int remain_cnt = remain >> 2;
  int remain_rem = remain & 3;
  float32x4_t vscale = vdupq_n_f32(scale);
  float32x4_t vbias = vdupq_n_f32(bias);
  scale_compute_fp32(din, dout, cnt, remain_cnt, remain_rem, vscale, vbias);
}

template <>
void scale_relu<float>(
    const float* din, float* dout, int num, float scale, float bias) {
  int cnt = num >> 4;
  int remain = num & 15;
  int remain_cnt = remain >> 2;
  int remain_rem = remain & 3;
  float32x4_t vscale = vdupq_n_f32(scale);
  float32x4_t vbias = vdupq_n_f32(bias);
  float32x4_t vzero = vdupq_n_f32(0.f);
#ifdef __aarch64__
  asm volatile(
      "cmp %w[cnt], #1                          \n"
      "blt 0f                                   \n"
      "1:                                      \n"
      "ld1  {v4.4s}, [%[din]], #16             \n"
      "and  v8.16b, %[vbias].16b, %[vbias].16b \n"
      "ld1  {v5.4s}, [%[din]], #16             \n"
      "and  v9.16b, %[vbias].16b, %[vbias].16b \n"
      "ld1  {v6.4s}, [%[din]], #16             \n"
      "and  v10.16b, %[vbias].16b, %[vbias].16b\n"
      "ld1  {v7.4s}, [%[din]], #16             \n"
      "and  v11.16b, %[vbias].16b, %[vbias].16b\n"

      "fmla v8.4s, v4.4s, %[vscale].4s       \n"
      "fmla v9.4s, v5.4s, %[vscale].4s       \n"
      "fmla v10.4s, v6.4s, %[vscale].4s      \n"
      "fmla v11.4s, v7.4s, %[vscale].4s      \n"

      "fmax v8.4s, v8.4s, %[vzero].4s        \n"
      "fmax v9.4s, v9.4s, %[vzero].4s        \n"
      "fmax v10.4s, v10.4s, %[vzero].4s      \n"
      "fmax v11.4s, v11.4s, %[vzero].4s      \n"

      "stp  q8, q9, [%[dout]], #32           \n"
      "subs %w[cnt], %w[cnt], #1             \n"
      "stp  q10, q11, [%[dout]], #32         \n"
      "bne    1b                             \n"
      "0:                                       \n"
      "cmp %w[remain_cnt], #1                   \n"
      "blt 2f                                   \n"
      "3:                                       \n"
      "ld1  {v4.4s}, [%[din]], #16              \n"
      "and  v8.16b, %[vbias].16b, %[vbias].16b  \n"
      "fmla v8.4s, v4.4s, %[vscale].4s          \n"
      "subs %w[remain_cnt], %w[remain_cnt],  #1 \n"
      "fmax v8.4s, v8.4s, %[vzero].4s        \n"
      "str q8, [%[dout]], #16                   \n"
      "bne    3b                                \n"
      "2:                                       \n"
      : [dout] "+r"(dout),
        [din] "+r"(din),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_cnt)
      : [vscale] "w"(vscale), [vbias] "w"(vbias), [vzero] "w"(vzero)
      : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else
  asm volatile(
      "cmp %[cnt], #1                           \n"
      "blt 0f                                   \n"
      "1:                                     @ loop header \n"
      "vld1.32  {d8-d11}, [%[din]]!           @ load din 0 \n"
      "vand.32 q8, %q[vbias], %q[vbias]       @ out bias \n"
      "vand.32 q9, %q[vbias], %q[vbias]       @ out bias \n"
      "vld1.32  {d12-d15}, [%[din]]!          @ load din 0 \n"

      "vand.32 q10, %q[vbias], %q[vbias]      @ out bias \n"
      "vand.32 q11, %q[vbias], %q[vbias]      @ out bias \n"

      "vmla.f32 q8, q4, %q[vscale]            @ mla \n"
      "vmla.f32 q9, q5, %q[vscale]            @ mla \n"
      "vmla.f32 q10, q6, %q[vscale]           @ mla \n"
      "vmla.f32 q11, q7, %q[vscale]           @ mla \n"

      "vmax.f32 q8, q8, %q[vzero]             @ relu \n"
      "vmax.f32 q9, q9, %q[vzero]             @ relu \n"
      "vmax.f32 q10, q10, %q[vzero]           @ relu \n"
      "vmax.f32 q11, q11, %q[vzero]           @ relu \n"

      "vst1.32  {d16-d19}, [%[dout]]!         @ store result, add pointer\n"
      "subs %[cnt], #1                        @ loop count minus 1\n"
      "vst1.32  {d20-d23}, [%[dout]]!         @ store result, add pointer\n"

      "bne    1b                              @ jump to main loop start \n"
      "0:                                       \n"
      "cmp %[remain_cnt], #1                   \n"
      "blt 2f                                   \n"
      "3:                                       \n"
      "vld1.32  {d8-d9}, [%[din]]!            @ load din 0 \n"
      "vand.32 q8, %q[vbias], %q[vbias]       @ out bias \n"
      "vmla.f32 q8, q4, %q[vscale]            @ mla \n"
      "subs %[remain_cnt],  #1                \n"
      "vmax.f32 q8, q8, %q[vzero]             @ relu \n"
      "vst1.32  {d16-d17}, [%[dout]]!           \n"
      "bne    3b                                \n"
      "2:                                       \n"
      : [dout] "+r"(dout),
        [din] "+r"(din),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_cnt)
      : [vscale] "w"(vscale), [vbias] "w"(vbias), [vzero] "w"(vzero)
      : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
  for (int i = 0; i < remain_rem; i++) {
    *dout = *din * scale + bias;
    *dout = *dout > 0.f ? *dout : 0.f;
    dout++;
    din++;
  }
}

template <>
void scale_relu6<float>(const float* din,
                        float* dout,
                        int num,
                        float scale,
                        float bias,
                        float alpha) {
  int cnt = num >> 4;
  int remain = num & 15;
  float32x4_t vscale = vdupq_n_f32(scale);
  float32x4_t vbias = vdupq_n_f32(bias);
  float32x4_t vzero = vdupq_n_f32(0.f);
  float32x4_t valpha = vdupq_n_f32(alpha);
  if (cnt > 0) {
#ifdef __aarch64__
    asm volatile(
        "1:                                       \n"
        "ld1  {v4.4s}, [%[din]], #16              \n"
        "and  v8.16b, %[vbias].16b, %[vbias].16b  \n"
        "ld1  {v5.4s}, [%[din]], #16              \n"
        "and  v9.16b, %[vbias].16b, %[vbias].16b  \n"
        "ld1  {v6.4s}, [%[din]], #16              \n"
        "and  v10.16b, %[vbias].16b, %[vbias].16b \n"
        "ld1  {v7.4s}, [%[din]], #16              \n"
        "and  v11.16b, %[vbias].16b, %[vbias].16b \n"

        "fmla v8.4s, v4.4s, %[vscale].4s       \n"
        "fmla v9.4s, v5.4s, %[vscale].4s       \n"
        "fmla v10.4s, v6.4s, %[vscale].4s      \n"
        "fmla v11.4s, v7.4s, %[vscale].4s      \n"

        "fmax v8.4s, v8.4s, %[vzero].4s        \n"
        "fmax v9.4s, v9.4s, %[vzero].4s        \n"
        "fmax v10.4s, v10.4s, %[vzero].4s      \n"
        "fmax v11.4s, v11.4s, %[vzero].4s      \n"

        "fmin v8.4s, v8.4s, %[valpha].4s       \n"
        "fmin v9.4s, v9.4s, %[valpha].4s       \n"
        "fmin v10.4s, v10.4s, %[valpha].4s     \n"
        "fmin v11.4s, v11.4s, %[valpha].4s     \n"

        "stp  q8, q9, [%[dout]], #32           \n"
        "subs %w[cnt], %w[cnt], #1             \n"
        "stp  q10, q11, [%[dout]], #32         \n"
        "bne    1b                             \n"
        "0:   \n"
        : [dout] "+r"(dout), [din] "+r"(din), [cnt] "+r"(cnt)
        : [vscale] "w"(vscale),
          [vbias] "w"(vbias),
          [vzero] "w"(vzero),
          [valpha] "w"(valpha)
        : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else
    asm volatile(
        "1:                                     @ loop header \n"
        "vld1.32  {d8-d11}, [%[din]]!           @ load din 0 \n"
        "vand.32 q8, %q[vbias], %q[vbias]       @ out bias \n"
        "vand.32 q9, %q[vbias], %q[vbias]       @ out bias \n"
        "vld1.32  {d12-d15}, [%[din]]!          @ load din 0 \n"

        "vand.32 q10, %q[vbias], %q[vbias]      @ out bias \n"
        "vand.32 q11, %q[vbias], %q[vbias]      @ out bias \n"

        "vmla.f32 q8, q4, %q[vscale]            @ mla \n"
        "vmla.f32 q9, q5, %q[vscale]            @ mla \n"
        "vmla.f32 q10, q6, %q[vscale]           @ mla \n"
        "vmla.f32 q11, q7, %q[vscale]           @ mla \n"

        "vmax.f32 q8, q8, %q[vzero]             @ relu \n"
        "vmax.f32 q9, q9, %q[vzero]             @ relu \n"
        "vmax.f32 q10, q10, %q[vzero]           @ relu \n"
        "vmax.f32 q11, q11, %q[vzero]           @ relu \n"

        "vmin.f32 q8, q8, %q[valpha]             @ relu \n"
        "vmin.f32 q9, q9, %q[valpha]             @ relu \n"
        "vmin.f32 q10, q10, %q[valpha]           @ relu \n"
        "vmin.f32 q11, q11, %q[valpha]           @ relu \n"

        "vst1.32  {d16-d19}, [%[dout]]!         @ store result, add pointer\n"
        "subs %[cnt], #1                        @ loop count minus 1\n"
        "vst1.32  {d20-d23}, [%[dout]]!         @ store result, add pointer\n"

        "bne    1b                              @ jump to main loop start "
        "2: \n"
        : [dout] "+r"(dout), [din] "+r"(din), [cnt] "+r"(cnt)
        : [vscale] "w"(vscale),
          [vbias] "w"(vbias),
          [vzero] "w"(vzero),
          [valpha] "w"(valpha)
        : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
  }
  if (remain > 0) {
    for (int i = 0; i < remain; i++) {
      *dout = *din * scale + bias;
      *dout = *dout > 0.f ? (*dout < alpha ? *dout : alpha) : 0.f;
      dout++;
      din++;
    }
  }
}

template <>
void scale_leaky_relu<float>(const float* din,
                             float* dout,
                             int num,
                             float scale,
                             float bias,
                             float alpha) {
  int cnt = num >> 4;
  int remain = num & 15;
  float32x4_t vscale = vdupq_n_f32(scale);
  float32x4_t vbias = vdupq_n_f32(bias);
  float32x4_t vzero = vdupq_n_f32(0.f);
  float32x4_t valpha = vdupq_n_f32(alpha);
  if (cnt > 0) {
#ifdef __aarch64__
    asm volatile(
        "1:                                       \n"
        "ld1  {v4.4s}, [%[din]], #16              \n"
        "and  v8.16b, %[vbias].16b, %[vbias].16b  \n"
        "ld1  {v5.4s}, [%[din]], #16              \n"
        "and  v9.16b, %[vbias].16b, %[vbias].16b  \n"
        "ld1  {v6.4s}, [%[din]], #16              \n"
        "and  v10.16b, %[vbias].16b, %[vbias].16b \n"
        "ld1  {v7.4s}, [%[din]], #16              \n"
        "and  v11.16b, %[vbias].16b, %[vbias].16b \n"

        "fmla v8.4s, v4.4s, %[vscale].4s       \n"
        "fmla v9.4s, v5.4s, %[vscale].4s       \n"
        "fmla v10.4s, v6.4s, %[vscale].4s      \n"
        "fmla v11.4s, v7.4s, %[vscale].4s      \n"

        "fcmge v12.4s, v8.4s, %[vzero].4s       \n"
        "fmul v16.4s, v8.4s, %[valpha].4s       \n"

        "fcmge v13.4s, v9.4s, %[vzero].4s       \n"
        "fmul v17.4s, v9.4s, %[valpha].4s        \n"

        "fcmge v14.4s, v10.4s, %[vzero].4s      \n"
        "fmul v18.4s, v10.4s, %[valpha].4s      \n"

        "fcmge v15.4s, v11.4s, %[vzero].4s      \n"
        "fmul v19.4s, v11.4s, %[valpha].4s      \n"

        "bif  v8.16b, v16.16b, v12.16b \n"  /* choose*/
        "bif  v9.16b, v17.16b, v13.16b \n"  /* choose*/
        "bif  v10.16b, v18.16b, v14.16b \n" /* choose*/
        "bif  v11.16b, v19.16b, v15.16b \n" /* choose*/

        "stp  q8, q9, [%[dout]], #32           \n"
        "subs %w[cnt], %w[cnt], #1             \n"
        "stp  q10, q11, [%[dout]], #32         \n"
        "bne    1b                             \n"
        "0:   \n"
        : [dout] "+r"(dout), [din] "+r"(din), [cnt] "+r"(cnt)
        : [vscale] "w"(vscale),
          [vbias] "w"(vbias),
          [vzero] "w"(vzero),
          [valpha] "w"(valpha)
        : "cc",
          "memory",
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
          "v15",
          "v16",
          "v17",
          "v18",
          "v19");
#else
    asm volatile(
        "1:                                     @ loop header \n"
        "vld1.32  {d8-d11}, [%[din]]!           @ load din 0 \n"
        "vand.32 q8, %q[vbias], %q[vbias]       @ out bias \n"
        "vand.32 q9, %q[vbias], %q[vbias]       @ out bias \n"
        "vld1.32  {d12-d15}, [%[din]]!          @ load din 0 \n"

        "vand.32 q10, %q[vbias], %q[vbias]      @ out bias \n"
        "vand.32 q11, %q[vbias], %q[vbias]      @ out bias \n"

        "vmla.f32 q8, q4, %q[vscale]            @ mla \n"
        "vmla.f32 q9, q5, %q[vscale]            @ mla \n"
        "vmla.f32 q10, q6, %q[vscale]           @ mla \n"
        "vmla.f32 q11, q7, %q[vscale]           @ mla \n"

        "vcge.f32 q12, q8, %q[vzero]             @ relu \n"
        "vmul.f32 q14, q8, %q[valpha]            @ mul \n"
        "vcge.f32 q13, q9, %q[vzero]             @ relu \n"
        "vmul.f32 q15, q9, %q[valpha]            @ mul \n"
        "vbif q8, q14, q12                       @ choose \n"
        "vbif q9, q15, q13                      @ choose \n"

        "vcge.f32 q12, q10, %q[vzero]             @ relu \n"
        "vmul.f32 q14, q10, %q[valpha]            @ mul \n"
        "vcge.f32 q13, q11, %q[vzero]             @ relu \n"
        "vmul.f32 q15, q11, %q[valpha]            @ mul \n"

        "vst1.32  {d16-d19}, [%[dout]]!         @ store result, add pointer\n"

        "vbif q10, q14, q12                       @ choose \n"
        "vbif q11, q15, q13                      @ choose \n"
        "subs %[cnt], #1                        @ loop count minus 1\n"
        "vst1.32  {d20-d23}, [%[dout]]!         @ store result, add pointer\n"

        "bne    1b                              @ jump to main loop start "
        "2: \n"
        : [dout] "+r"(dout), [din] "+r"(din), [cnt] "+r"(cnt)
        : [vscale] "w"(vscale),
          [vbias] "w"(vbias),
          [vzero] "w"(vzero),
          [valpha] "w"(valpha)
        : "cc",
          "memory",
          "q4",
          "q5",
          "q6",
          "q7",
          "q8",
          "q9",
          "q10",
          "q11",
          "q12",
          "q13",
          "q14",
          "q15");
#endif
  }
  if (remain > 0) {
    for (int i = 0; i < remain; i++) {
      *dout = *din * scale + bias;
      *dout = *dout > 0.f ? *dout : (*dout * alpha);
      dout++;
      din++;
    }
  }
}

template <>
void scale<int>(const int* din, int* dout, int num, int scale, int bias) {
  int cnt = num >> 4;
  int remain = num % 16;
  int32x4_t vscale = vdupq_n_s32(scale);
  int32x4_t vbias = vdupq_n_s32(bias);
  LITE_PARALLEL_BEGIN(i, tid, cnt) {
    const int* din_ptr = din + (i << 4);
    int* dout_ptr = dout + (i << 4);

    int32x4_t din0 = vld1q_s32(din_ptr);
    int32x4_t din1 = vld1q_s32(din_ptr + 4);
    int32x4_t din2 = vld1q_s32(din_ptr + 8);
    int32x4_t din3 = vld1q_s32(din_ptr + 12);

    int32x4_t vsum1 = vmlaq_s32(vbias, din0, vscale);
    int32x4_t vsum2 = vmlaq_s32(vbias, din1, vscale);
    int32x4_t vsum3 = vmlaq_s32(vbias, din2, vscale);
    int32x4_t vsum4 = vmlaq_s32(vbias, din3, vscale);

    vst1q_s32(dout_ptr, vsum1);
    vst1q_s32(dout_ptr + 4, vsum2);
    vst1q_s32(dout_ptr + 8, vsum3);
    vst1q_s32(dout_ptr + 12, vsum4);
  }
  LITE_PARALLEL_END()
  if (remain > 0) {
    const int* din_ptr = din + (cnt << 4);
    int* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      *dout_ptr = *din_ptr * scale + bias;
      dout_ptr++;
      din_ptr++;
    }
  }
}

template <>
void scale_relu<int>(const int* din, int* dout, int num, int scale, int bias) {
  int cnt = num >> 4;
  int remain = num % 16;
  int32x4_t vscale = vdupq_n_s32(scale);
  int32x4_t vbias = vdupq_n_s32(bias);
  int32x4_t vzero = vdupq_n_s32(0);
  LITE_PARALLEL_BEGIN(i, tid, cnt) {
    const int* din_ptr = din + (i << 4);
    int* dout_ptr = dout + (i << 4);

    int32x4_t din0 = vld1q_s32(din_ptr);
    int32x4_t din1 = vld1q_s32(din_ptr + 4);
    int32x4_t din2 = vld1q_s32(din_ptr + 8);
    int32x4_t din3 = vld1q_s32(din_ptr + 12);

    int32x4_t vsum1 = vmlaq_s32(vbias, din0, vscale);
    int32x4_t vsum2 = vmlaq_s32(vbias, din1, vscale);
    int32x4_t vsum3 = vmlaq_s32(vbias, din2, vscale);
    int32x4_t vsum4 = vmlaq_s32(vbias, din3, vscale);

    vsum1 = vmaxq_s32(vsum1, vzero);
    vsum2 = vmaxq_s32(vsum2, vzero);
    vsum3 = vmaxq_s32(vsum3, vzero);
    vsum4 = vmaxq_s32(vsum4, vzero);

    vst1q_s32(dout_ptr, vsum1);
    vst1q_s32(dout_ptr + 4, vsum2);
    vst1q_s32(dout_ptr + 8, vsum3);
    vst1q_s32(dout_ptr + 12, vsum4);
  }
  LITE_PARALLEL_END()
  if (remain > 0) {
    const int* din_ptr = din + (cnt << 4);
    int* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      *dout_ptr = *din_ptr * scale + bias;
      *dout_ptr = *dout_ptr > 0 ? *dout_ptr : 0;
      dout_ptr++;
      din_ptr++;
    }
  }
}

template <>
void scale_relu6<int>(
    const int* din, int* dout, int num, int scale, int bias, int alpha) {
  int cnt = num >> 4;
  int remain = num % 16;
  int32x4_t vscale = vdupq_n_s32(scale);
  int32x4_t vbias = vdupq_n_s32(bias);
  int32x4_t vzero = vdupq_n_s32(0);
  int32x4_t valpha = vdupq_n_s32(alpha);
  LITE_PARALLEL_BEGIN(i, tid, cnt) {
    const int* din_ptr = din + (i << 4);
    int* dout_ptr = dout + (i << 4);

    int32x4_t din0 = vld1q_s32(din_ptr);
    int32x4_t din1 = vld1q_s32(din_ptr + 4);
    int32x4_t din2 = vld1q_s32(din_ptr + 8);
    int32x4_t din3 = vld1q_s32(din_ptr + 12);

    int32x4_t vsum1 = vmlaq_s32(vbias, din0, vscale);
    int32x4_t vsum2 = vmlaq_s32(vbias, din1, vscale);
    int32x4_t vsum3 = vmlaq_s32(vbias, din2, vscale);
    int32x4_t vsum4 = vmlaq_s32(vbias, din3, vscale);

    vsum1 = vmaxq_s32(vsum1, vzero);
    vsum2 = vmaxq_s32(vsum2, vzero);
    vsum3 = vmaxq_s32(vsum3, vzero);
    vsum4 = vmaxq_s32(vsum4, vzero);

    vsum1 = vminq_s32(vsum1, valpha);
    vsum2 = vminq_s32(vsum2, valpha);
    vsum3 = vminq_s32(vsum3, valpha);
    vsum4 = vminq_s32(vsum4, valpha);

    vst1q_s32(dout_ptr, vsum1);
    vst1q_s32(dout_ptr + 4, vsum2);
    vst1q_s32(dout_ptr + 8, vsum3);
    vst1q_s32(dout_ptr + 12, vsum4);
  }
  LITE_PARALLEL_END()
  if (remain > 0) {
    const int* din_ptr = din + (cnt << 4);
    int* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      *dout_ptr = *din_ptr * scale + bias;
      *dout_ptr = *dout_ptr > 0 ? (*dout_ptr > alpha ? alpha : *dout_ptr) : 0;
      dout_ptr++;
      din_ptr++;
    }
  }
}

template <>
void scale_leaky_relu<int>(
    const int* din, int* dout, int num, int scale, int bias, int alpha) {
  int cnt = num >> 4;
  int remain = num % 16;
  int32x4_t vscale = vdupq_n_s32(scale);
  int32x4_t vbias = vdupq_n_s32(bias);
  int32x4_t vzero = vdupq_n_s32(0);
  int32x4_t valpha = vdupq_n_s32(alpha);
  LITE_PARALLEL_BEGIN(i, tid, cnt) {
    const int* din_ptr = din + (i << 4);
    int* dout_ptr = dout + (i << 4);

    int32x4_t din0 = vld1q_s32(din_ptr);
    int32x4_t din1 = vld1q_s32(din_ptr + 4);
    int32x4_t din2 = vld1q_s32(din_ptr + 8);
    int32x4_t din3 = vld1q_s32(din_ptr + 12);

    int32x4_t vsum1 = vmlaq_s32(vbias, din0, vscale);
    int32x4_t vsum2 = vmlaq_s32(vbias, din1, vscale);
    int32x4_t vsum3 = vmlaq_s32(vbias, din2, vscale);
    int32x4_t vsum4 = vmlaq_s32(vbias, din3, vscale);

    uint32x4_t v1 = vcgeq_s32(vsum1, vzero);
    uint32x4_t v2 = vcgeq_s32(vsum2, vzero);
    uint32x4_t v3 = vcgeq_s32(vsum3, vzero);
    uint32x4_t v4 = vcgeq_s32(vsum4, vzero);

    int32x4_t v11 = vmulq_s32(vsum1, valpha);
    int32x4_t v21 = vmulq_s32(vsum1, valpha);
    int32x4_t v31 = vmulq_s32(vsum1, valpha);
    int32x4_t v41 = vmulq_s32(vsum1, valpha);

    vsum1 = vbslq_s32(v1, vsum1, v11);
    vsum2 = vbslq_s32(v2, vsum2, v21);
    vsum3 = vbslq_s32(v3, vsum3, v31);
    vsum4 = vbslq_s32(v4, vsum4, v41);

    vst1q_s32(dout_ptr, vsum1);
    vst1q_s32(dout_ptr + 4, vsum2);
    vst1q_s32(dout_ptr + 8, vsum3);
    vst1q_s32(dout_ptr + 12, vsum4);
  }
  LITE_PARALLEL_END()

  if (remain > 0) {
    const int* din_ptr = din + (cnt << 4);
    int* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      *dout_ptr = *din_ptr * scale + bias;
      *dout_ptr = *dout_ptr > 0 ? *dout_ptr : (*dout_ptr) * alpha;
      dout_ptr++;
      din_ptr++;
    }
  }
}

template <>
void scale<int64_t>(
    const int64_t* din, int64_t* dout, int num, int64_t scale, int64_t bias) {
  const int64_t* din_ptr = din;
  int64_t* dout_ptr = dout;
  for (int i = 0; i < num; i++) {
    *dout_ptr = *din_ptr * scale + bias;
    dout_ptr++;
    din_ptr++;
  }
}

template <>
void scale_relu<int64_t>(
    const int64_t* din, int64_t* dout, int num, int64_t scale, int64_t bias) {
  const int64_t* din_ptr = din;
  int64_t* dout_ptr = dout;
  for (int i = 0; i < num; i++) {
    *dout_ptr = *din_ptr * scale + bias;
    *dout_ptr = *dout_ptr > 0 ? *dout_ptr : 0;
    dout_ptr++;
    din_ptr++;
  }
}

template <>
void scale_relu6<int64_t>(const int64_t* din,
                          int64_t* dout,
                          int num,
                          int64_t scale,
                          int64_t bias,
                          int64_t alpha) {
  const int64_t* din_ptr = din;
  int64_t* dout_ptr = dout;
  for (int i = 0; i < num; i++) {
    *dout_ptr = *din_ptr * scale + bias;
    *dout_ptr = *dout_ptr > 0 ? (*dout_ptr > alpha ? alpha : *dout_ptr) : 0;
    dout_ptr++;
    din_ptr++;
  }
}

template <>
void scale_leaky_relu<int64_t>(const int64_t* din,
                               int64_t* dout,
                               int num,
                               int64_t scale,
                               int64_t bias,
                               int64_t alpha) {
  const int64_t* din_ptr = din;
  int64_t* dout_ptr = dout;
  for (int i = 0; i < num; i++) {
    *dout_ptr = *din_ptr * scale + bias;
    *dout_ptr = *dout_ptr > 0 ? *dout_ptr : (*dout_ptr) * alpha;
    dout_ptr++;
    din_ptr++;
  }
}

template <>
void scale<float>(const float* din,
                  float* dout,
                  int outer_dim,
                  int scale_dim,
                  int inner_dim,
                  const float* scale_data,
                  const float* bias_data) {
  int cnt = inner_dim >> 4;
  int remain = inner_dim & 15;
  int remain_cnt = remain >> 2;
  int remain_rem = remain & 3;
  int size = inner_dim * scale_dim;
  for (int n = 0; n < outer_dim; n++) {
    const float* din_ptr_n = din + n * size;
    float* dout_ptr_n = dout + n * size;
    LITE_PARALLEL_BEGIN(i, tid, scale_dim) {
      const float* din_ptr = din_ptr_n + i * inner_dim;
      float* dout_ptr = dout_ptr_n + i * inner_dim;
      float32x4_t vscale = vdupq_n_f32(scale_data[i]);
      float32x4_t vbias = vdupq_n_f32(bias_data[i]);
      scale_compute_fp32(
          din_ptr, dout_ptr, cnt, remain_cnt, remain_rem, vscale, vbias);
    }
    LITE_PARALLEL_END()
  }
}

template <>
void scale<float>(const float* din,
                  float* dout,
                  int outer_dim,
                  int scale_dim,
                  const float* scale_data,
                  const float* bias_data) {
  int cnt = scale_dim >> 4;
  int remain = scale_dim % 16;
  for (int n = 0; n < outer_dim; n++) {
    const float* din_ptr_n = din + n * scale_dim;
    float* dout_ptr_n = dout + n * scale_dim;
    LITE_PARALLEL_BEGIN(i, tid, cnt) {
      int idx = i << 4;
      const float* din_ptr = din_ptr_n + idx;
      const float* scale_ptr = scale_data + idx;
      const float* bias_ptr = bias_data + idx;
      float* dout_ptr = dout_ptr_n + idx;

      float32x4_t din0 = vld1q_f32(din_ptr);
      float32x4_t vscale0 = vld1q_f32(scale_ptr);
      float32x4_t vbias0 = vld1q_f32(bias_ptr);

      float32x4_t din1 = vld1q_f32(din_ptr + 4);
      float32x4_t vscale1 = vld1q_f32(scale_ptr + 4);
      float32x4_t vbias1 = vld1q_f32(bias_ptr + 4);

      float32x4_t din2 = vld1q_f32(din_ptr + 8);
      float32x4_t vscale2 = vld1q_f32(scale_ptr + 8);
      float32x4_t vbias2 = vld1q_f32(bias_ptr + 8);

      float32x4_t vsum1 = vmlaq_f32(vbias0, din0, vscale0);
      float32x4_t vsum2 = vmlaq_f32(vbias1, din1, vscale1);

      float32x4_t din3 = vld1q_f32(din_ptr + 12);
      float32x4_t vscale3 = vld1q_f32(scale_ptr + 12);
      float32x4_t vbias3 = vld1q_f32(bias_ptr + 12);

      vst1q_f32(dout_ptr, vsum1);
      vst1q_f32(dout_ptr + 4, vsum2);

      float32x4_t vsum3 = vmlaq_f32(vbias2, din2, vscale2);
      float32x4_t vsum4 = vmlaq_f32(vbias3, din3, vscale3);

      vst1q_f32(dout_ptr + 8, vsum3);
      vst1q_f32(dout_ptr + 12, vsum4);
    }
    LITE_PARALLEL_END()
    int idx = cnt << 4;
    const float* din_ptr = din_ptr_n + idx;
    float* dout_ptr = dout_ptr_n + idx;
    const float* scale_ptr = scale_data + idx;
    const float* bias_ptr = bias_data + idx;
    for (int j = 0; j < remain; j++) {
      *dout_ptr = *din_ptr * (*scale_ptr) + (*bias_ptr);
      dout_ptr++;
      din_ptr++;
      scale_ptr++;
      bias_ptr++;
    }
  }
}

#ifdef ENABLE_ARM_FP16
typedef __fp16 flaot16_t;

inline void scale_compute_fp16(const flaot16_t* din,
                               float16_t* dout,
                               int cnt,
                               int remain_cnt,
                               int remain_rem,
                               float16x8_t vscale,
                               float16x8_t vbias) {
#ifdef __aarch64__
  asm volatile(
      "cmp %w[cnt], #1                          \n"
      "blt 0f                                   \n"
      "1:                                       \n"
      "ld1  {v4.8h}, [%[din]], #16              \n"
      "and  v8.16b, %[vbias].16b, %[vbias].16b  \n"
      "ld1  {v5.8h}, [%[din]], #16              \n"
      "and  v9.16b, %[vbias].16b, %[vbias].16b  \n"

      "fmla v8.8h, v4.8h, %[vscale].8h          \n"
      "fmla v9.8h, v5.8h, %[vscale].8h          \n"

      "subs %w[cnt], %w[cnt],  #1               \n"
      "stp  q8, q9, [%[dout]], #32              \n"
      "bne    1b                                \n"
      "0:                                       \n"
      "cmp %w[remain_cnt], #1                   \n"
      "blt 2f                                   \n"
      "3:                                       \n"
      "ld1  {v4.4h}, [%[din]], #8               \n"
      "and  v8.16b, %[vbias].16b, %[vbias].16b  \n"
      "fmla v8.4h, v4.4h, %[vscale].4h          \n"
      "subs %w[remain_cnt], %w[remain_cnt],  #1 \n"
      "st1 {v8.4h}, [%[dout]], #8               \n"
      "bne    3b                                \n"
      "2:                                       \n"
      : [dout] "+r"(dout),
        [din] "+r"(din),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_cnt)
      : [vscale] "w"(vscale), [vbias] "w"(vbias)
      : "cc", "memory", "v4", "v5", "v8", "v9");
#else
  asm volatile(
      "cmp  %[cnt], #1                          \n"
      "blt 0f                                   \n"
      "1:                                       \n"
      "vld1.16  {d8-d9}, [%[din]]!              \n"
      "vmov     q8, %q[vbias] \n"
      "vld1.16  {d10-d11}, [%[din]]!              \n"
      "vmov     q9, %q[vbias] \n"

      "vmla.f16 q8, q4, %q[vscale]          \n"
      "vmla.f16 q9, q5, %q[vscale]         \n"

      "subs %[cnt], %[cnt],  #1               \n"
      "vst1.16 {d16-d19}, [%[dout]]!             \n"
      "bne    1b                                \n"
      "0:                                       \n"
      "cmp  %[remain_cnt], #1                   \n"
      "blt 2f                                   \n"
      "3:                                       \n"
      "vld1.16  {d8}, [%[din]]!               \n"
      "vmov     d16, %e[vbias] \n"
      "vmla.f16 d16, d8, %e[vscale]          \n"
      "subs %[remain_cnt], %[remain_cnt],  #1 \n"
      "vst1.16 {d16}, [%[dout]]!              \n"
      "bne    3b                                \n"
      "2:                                       \n"
      : [dout] "+r"(dout),
        [din] "+r"(din),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_cnt)
      : [vscale] "w"(vscale), [vbias] "w"(vbias)
      : "cc", "memory", "q4", "q5", "q8", "q9");

#endif
  for (int j = 0; j < remain_rem; j++) {
    *dout = *din * vscale[0] + vbias[0];
    dout++;
    din++;
  }
}

template <>
void scale<float16_t>(const flaot16_t* din,
                      flaot16_t* dout,
                      int outer_dim,
                      int scale_dim,
                      int inner_dim,
                      const flaot16_t* scale_data,
                      const flaot16_t* bias_data) {
  int cnt = inner_dim >> 4;
  int remain = inner_dim % 16;
  int remain_num = remain >> 2;
  int remain_rem = remain & 3;
  int size = inner_dim * scale_dim;
  for (int n = 0; n < outer_dim; n++) {
    const flaot16_t* din_ptr_n = din + n * size;
    flaot16_t* dout_ptr_n = dout + n * size;
    LITE_PARALLEL_BEGIN(i, tid, scale_dim) {
      const flaot16_t* din_ptr = din_ptr_n + i * inner_dim;
      flaot16_t* dout_ptr = dout_ptr_n + i * inner_dim;
      float16x8_t vscale = vdupq_n_f16(scale_data[i]);
      float16x8_t vbias = vdupq_n_f16(bias_data[i]);
      scale_compute_fp16(
          din_ptr, dout_ptr, cnt, remain_num, remain_rem, vscale, vbias);
    }
    LITE_PARALLEL_END()
  }
}

template <>
void scale<float16_t>(const float16_t* din,
                      float16_t* dout,
                      int outer_dim,
                      int scale_dim,
                      const float16_t* scale_data,
                      const float16_t* bias_data) {
  int cnt = scale_dim >> 4;
  int remain = scale_dim % 16;
  for (int n = 0; n < outer_dim; n++) {
    const float16_t* din_ptr_n = din + n * scale_dim;
    float16_t* dout_ptr_n = dout + n * scale_dim;
    LITE_PARALLEL_BEGIN(i, tid, cnt) {
      int idx = i << 4;
      const float16_t* din_ptr = din_ptr_n + idx;
      const float16_t* scale_ptr = scale_data + idx;
      const float16_t* bias_ptr = bias_data + idx;
      float16_t* dout_ptr = dout_ptr_n + idx;

      float16x8_t din0 = vld1q_f16(din_ptr);
      float16x8_t vscale0 = vld1q_f16(scale_ptr);
      float16x8_t vbias0 = vld1q_f16(bias_ptr);

      float16x8_t din1 = vld1q_f16(din_ptr + 8);
      float16x8_t vscale1 = vld1q_f16(scale_ptr + 8);
      float16x8_t vbias1 = vld1q_f16(bias_ptr + 8);

      float16x8_t vsum1 = vfmaq_f16(vbias0, din0, vscale0);
      float16x8_t vsum2 = vfmaq_f16(vbias1, din1, vscale1);

      vst1q_f16(dout_ptr, vsum1);
      vst1q_f16(dout_ptr + 8, vsum2);
    }
    LITE_PARALLEL_END()
    int idx = cnt << 4;
    const float16_t* din_ptr = din_ptr_n + idx;
    float16_t* dout_ptr = dout_ptr_n + idx;
    const float16_t* scale_ptr = scale_data + idx;
    const float16_t* bias_ptr = bias_data + idx;
    for (int j = 0; j < remain; j++) {
      *dout_ptr = *din_ptr * (*scale_ptr) + (*bias_ptr);
      dout_ptr++;
      din_ptr++;
      scale_ptr++;
      bias_ptr++;
    }
  }
}

template <>
void scale<float16_t>(const float16_t* din,
                      float16_t* dout,
                      int num,
                      float16_t scale,
                      float16_t bias) {
  int cnt = num >> 4;
  int remain = num % 16;
  int remain_num = remain >> 2;
  int remain_rem = remain & 3;
  float16x8_t vscale = vdupq_n_f16(scale);
  float16x8_t vbias = vdupq_n_f16(bias);
  scale_compute_fp16(din, dout, cnt, remain_num, remain_rem, vscale, vbias);
}

template <>
void scale_relu<float16_t>(const float16_t* din,
                           float16_t* dout,
                           int num,
                           float16_t scale,
                           float16_t bias) {
  int cnt = num >> 4;
  int remain = num % 16;
  int remain_num = remain >> 2;
  int remain_rem = remain & 3;
  float16x8_t vscale = vdupq_n_f16(scale);
  float16x8_t vbias = vdupq_n_f16(bias);
  float16x8_t vzero = vdupq_n_f16(0.f);
#ifdef __aarch64__
  asm volatile(
      "cmp %w[cnt], #1                         \n"
      "blt 0f                                  \n"
      "1:                                      \n"
      "ld1  {v4.8h}, [%[din]], #16             \n"
      "and  v8.16b, %[vbias].16b, %[vbias].16b \n"
      "ld1  {v5.8h}, [%[din]], #16             \n"
      "and  v9.16b, %[vbias].16b, %[vbias].16b \n"

      "fmla v8.8h, v4.8h, %[vscale].8h         \n"
      "fmla v9.8h, v5.8h, %[vscale].8h         \n"

      "fmax v8.8h, v8.8h, %[vzero].8h          \n"
      "fmax v9.8h, v9.8h, %[vzero].8h          \n"
      "subs %w[cnt], %w[cnt], #1               \n"
      "stp  q8, q9, [%[dout]], #32             \n"
      "bne    1b                               \n"
      "0:                                      \n"
      "cmp %w[remain_cnt], #1                  \n"
      "blt 2f                                  \n"
      "3:                                      \n"
      "ld1  {v4.4h}, [%[din]], #8              \n"
      "and  v8.16b, %[vbias].16b, %[vbias].16b \n"
      "fmla v8.4h, v4.4h, %[vscale].4h         \n"
      "subs %w[remain_cnt], %w[remain_cnt],  #1\n"
      "fmax v8.4h, v8.4h, %[vzero].4h          \n"
      "st1 {v8.4h}, [%[dout]], #8              \n"
      "bne    3b                               \n"
      "2:                                      \n"
      : [dout] "+r"(dout),
        [din] "+r"(din),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_num)
      : [vscale] "w"(vscale), [vbias] "w"(vbias), [vzero] "w"(vzero)
      : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else
  asm volatile(
      "cmp  %[cnt], #1                          \n"
      "blt 0f                                   \n"
      "1:                                       \n"
      "vld1.16  {d8-d9}, [%[din]]!              \n"
      "vmov     q8, %q[vbias] \n"
      "vld1.16  {d10-d11}, [%[din]]!              \n"
      "vmov     q9, %q[vbias] \n"

      "vmla.f16 q8, q4, %q[vscale]          \n"
      "vmla.f16 q9, q5, %q[vscale]         \n"
      "vmax.f16 q8, q8, %q[vzero]         \n"
      "vmax.f16 q9, q9, %q[vzero]          \n"

      "subs %[cnt], %[cnt],  #1               \n"
      "vst1.16 {d16-d19}, [%[dout]]!             \n"
      "bne    1b                                \n"
      "0:                                       \n"
      "cmp  %[remain_cnt], #1                   \n"
      "blt 2f                                   \n"
      "3:                                       \n"
      "vld1.16  {d8}, [%[din]]!               \n"
      "vmov     d16, %e[vbias] \n"
      "vmla.f16 d16, d8, %e[vscale]          \n"
      "vmax.f16 d16, d16, %e[vzero]         \n"
      "subs %[remain_cnt], %[remain_cnt],  #1 \n"
      "vst1.16 {d16}, [%[dout]]!              \n"
      "bne    3b                                \n"
      "2:                                       \n"
      : [dout] "+r"(dout),
        [din] "+r"(din),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_num)
      : [vscale] "w"(vscale), [vbias] "w"(vbias), [vzero] "w"(vzero)
      : "cc", "memory", "q4", "q5", "q8", "q9");

#endif
  for (int i = 0; i < remain_rem; i++) {
    *dout = *din * scale + bias;
    *dout = *dout > 0.f ? *dout : 0.f;
    dout++;
    din++;
  }
}

template <>
void scale_relu6<float16_t>(const float16_t* din,
                            float16_t* dout,
                            int num,
                            float16_t scale,
                            float16_t bias,
                            float16_t alpha) {
  int cnt = num >> 4;
  int remain = num % 16;
  int remain_num = remain >> 2;
  int remain_rem = remain & 3;
  float16x8_t vscale = vdupq_n_f16(scale);
  float16x8_t vbias = vdupq_n_f16(bias);
  float16x8_t vzero = vdupq_n_f16(0.f);
  float16x8_t valpha = vdupq_n_f16(alpha);
#ifdef __aarch64__
  asm volatile(
      "cmp %w[cnt], #1                         \n"
      "blt 0f                                  \n"
      "1:                                      \n"
      "ld1  {v4.8h}, [%[din]], #16             \n"
      "and  v8.16b, %[vbias].16b, %[vbias].16b \n"
      "ld1  {v5.8h}, [%[din]], #16             \n"
      "and  v9.16b, %[vbias].16b, %[vbias].16b \n"

      "fmla v8.8h, v4.8h, %[vscale].8h         \n"
      "fmla v9.8h, v5.8h, %[vscale].8h         \n"

      "fmax v8.8h, v8.8h, %[vzero].8h          \n"
      "fmax v9.8h, v9.8h, %[vzero].8h          \n"

      "fmin v8.8h, v8.8h, %[valpha].8h         \n"
      "fmin v9.8h, v9.8h, %[valpha].8h         \n"

      "subs %w[cnt], %w[cnt], #1               \n"
      "stp  q8, q9, [%[dout]], #32             \n"
      "bne    1b                               \n"
      "0:                                      \n"
      "cmp %w[remain_cnt], #1                  \n"
      "blt 2f                                  \n"
      "3:                                      \n"
      "ld1  {v4.4h}, [%[din]], #8              \n"
      "and  v8.16b, %[vbias].16b, %[vbias].16b \n"
      "fmla v8.4h, v4.4h, %[vscale].4h         \n"
      "subs %w[remain_cnt], %w[remain_cnt],  #1\n"
      "fmax v8.4h, v8.4h, %[vzero].4h          \n"
      "fmin v8.4h, v8.4h, %[valpha].4h         \n"
      "st1 {v8.4h}, [%[dout]], #8              \n"
      "bne    3b                               \n"
      "2:                                      \n"
      : [dout] "+r"(dout),
        [din] "+r"(din),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_num)
      : [vscale] "w"(vscale),
        [vbias] "w"(vbias),
        [vzero] "w"(vzero),
        [valpha] "w"(valpha)
      : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else
  asm volatile(
      "cmp  %[cnt], #1                          \n"
      "blt 0f                                   \n"
      "1:                                       \n"
      "vld1.16  {d8-d9}, [%[din]]!              \n"
      "vmov     q8, %q[vbias] \n"
      "vld1.16  {d10-d11}, [%[din]]!              \n"
      "vmov     q9, %q[vbias] \n"

      "vmla.f16 q8, q4, %q[vscale]          \n"
      "vmla.f16 q9, q5, %q[vscale]         \n"
      "vmax.f16 q8, q8, %q[vzero]         \n"
      "vmax.f16 q9, q9, %q[vzero]          \n"
      "vmin.f16 q8, q8, %q[valpha]         \n"
      "vmin.f16 q9, q9, %q[valpha]          \n"

      "subs %[cnt], %[cnt],  #1               \n"
      "vst1.16 {d16-d19}, [%[dout]]!             \n"
      "bne    1b                                \n"
      "0:                                       \n"
      "cmp  %[remain_cnt], #1                   \n"
      "blt 2f                                   \n"
      "3:                                       \n"
      "vld1.16  {d8}, [%[din]]!               \n"
      "vmov     d16, %e[vbias] \n"
      "vmla.f16 d16, d8, %e[vscale]          \n"
      "vmax.f16 d16, d16, %e[vzero]         \n"
      "vmin.f16 d16, d16, %e[valpha]         \n"
      "subs %[remain_cnt], %[remain_cnt],  #1 \n"
      "vst1.16 {d16}, [%[dout]]!              \n"
      "bne    3b                                \n"
      "2:                                       \n"
      : [dout] "+r"(dout),
        [din] "+r"(din),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_num)
      : [vscale] "w"(vscale),
        [vbias] "w"(vbias),
        [vzero] "w"(vzero),
        [valpha] "w"(valpha)
      : "cc", "memory", "q4", "q5", "q8", "q9");

#endif
  for (int i = 0; i < remain_rem; i++) {
    *dout = *din * scale + bias;
    *dout = *dout > 0.f ? (*dout < alpha ? *dout : alpha) : 0.f;
    dout++;
    din++;
  }
}

template <>
void scale_leaky_relu<float16_t>(const float16_t* din,
                                 float16_t* dout,
                                 int num,
                                 float16_t scale,
                                 float16_t bias,
                                 float16_t alpha) {
  int cnt = num >> 4;
  int remain = num % 16;
  int remain_num = remain >> 2;
  int remain_rem = remain & 3;
  float16x8_t vscale = vdupq_n_f16(scale);
  float16x8_t vbias = vdupq_n_f16(bias);
  float16x8_t vzero = vdupq_n_f16(0.f);
  float16x8_t valpha = vdupq_n_f16(alpha);
#ifdef __aarch64__
  asm volatile(
      "cmp %w[cnt], #1                         \n"
      "blt 0f                                  \n"
      "1:                                      \n"
      "ld1  {v4.8h}, [%[din]], #16             \n"
      "and  v8.16b, %[vbias].16b, %[vbias].16b \n"
      "ld1  {v5.8h}, [%[din]], #16             \n"
      "and  v9.16b, %[vbias].16b, %[vbias].16b \n"

      "fmla v8.8h, v4.8h, %[vscale].8h         \n"
      "fmla v9.8h, v5.8h, %[vscale].8h         \n"

      "fcmge v10.8h, v8.8h, %[vzero].8h        \n"
      "fmul v11.8h, v8.8h, %[valpha].8h        \n"
      "fcmge v12.8h, v9.8h, %[vzero].8h        \n"
      "fmul v13.8h, v9.8h, %[valpha].8h        \n"

      "subs %w[cnt], %w[cnt], #1               \n"
      "bif  v8.16b, v11.16b, v10.16b           \n"
      "bif  v9.16b, v13.16b, v12.16b           \n"
      "stp  q8, q9, [%[dout]], #32             \n"
      "bne    1b                               \n"
      "0:                                      \n"
      "cmp %w[remain_cnt], #1                  \n"
      "blt 2f                                  \n"
      "3:                                      \n"
      "ld1  {v4.4h}, [%[din]], #8              \n"
      "and  v8.16b, %[vbias].16b, %[vbias].16b \n"
      "fmla v8.4h, v4.4h, %[vscale].4h         \n"
      "subs %w[remain_cnt], %w[remain_cnt],  #1\n"
      "fcmge v10.4h, v8.4h, %[vzero].4h        \n"
      "fmul v11.4h, v8.4h, %[valpha].4h        \n"
      "bif  v8.16b, v11.16b, v10.16b           \n"
      "st1 {v8.4h}, [%[dout]], #8              \n"
      "bne    3b                               \n"
      "2:                                      \n"
      : [dout] "+r"(dout),
        [din] "+r"(din),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_num)
      : [vscale] "w"(vscale),
        [vbias] "w"(vbias),
        [vzero] "w"(vzero),
        [valpha] "w"(valpha)
      : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
#else
  asm volatile(
      "cmp  %[cnt], #1                          \n"
      "blt 0f                                   \n"
      "1:                                       \n"
      "vld1.16  {d8-d9}, [%[din]]!              \n"
      "vmov     q8, %q[vbias] \n"
      "vld1.16  {d10-d11}, [%[din]]!              \n"
      "vmov     q9, %q[vbias] \n"

      "vmla.f16 q8, q4, %q[vscale]          \n"
      "vmla.f16 q9, q5, %q[vscale]         \n"
      "vcge.f16  q10,  q8, %q[vzero] \n"
      "vmul.f16  q11,  q8, %q[valpha]\n"
      "vcge.f16  q12,  q9, %q[vzero] \n"
      "vmul.f16  q13,  q9, %q[valpha]\n"

      "subs %[cnt], %[cnt],  #1       \n"
      "vbif      q8,  q11,  q10        \n"
      "vbif      q9,  q13, q12       \n"
      "vst1.16 {d16-d19}, [%[dout]]!             \n"
      "bne    1b                                \n"
      "0:                                       \n"
      "cmp  %[remain_cnt], #1                   \n"
      "blt 2f                                   \n"
      "3:                                       \n"
      "vld1.16  {d8}, [%[din]]!               \n"
      "vmov     d16, %e[vbias] \n"
      "vmla.f16 d16, d8, %e[vscale]          \n"
      "vcge.f16  d20,  d16, %e[vzero] \n"
      "vmul.f16  d22,  d16, %e[valpha]\n"
      "subs %[remain_cnt], %[remain_cnt],  #1 \n"
      "vbif      d16,  d22,  d20        \n"
      "vst1.16 {d16}, [%[dout]]!              \n"
      "bne    3b                                \n"
      "2:                                       \n"
      : [dout] "+r"(dout),
        [din] "+r"(din),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_num)
      : [vscale] "w"(vscale),
        [vbias] "w"(vbias),
        [vzero] "w"(vzero),
        [valpha] "w"(valpha)
      : "cc", "memory", "q4", "q5", "q8", "q9", "q10", "q11");

#endif
  for (int i = 0; i < remain_rem; i++) {
    *dout = *din * scale + bias;
    *dout = *dout > 0.f ? *dout : (*dout * alpha);
    dout++;
    din++;
  }
}
#endif

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
