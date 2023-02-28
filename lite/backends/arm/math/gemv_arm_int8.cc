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

#include "lite/backends/arm/math/gemv_arm_int8.h"
#include <arm_neon.h>
#include <algorithm>
#include "lite/backends/arm/math/saturate.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
template <typename dtype>
inline void write_gemv_out(const int* in,
                           dtype* out,
                           const float* scale,
                           const float* bias,
                           int size,
                           bool flag_act,
                           lite_api::ActivationType act,
                           float six,
                           float alpha,
                           float offset,
                           float threshold);

template <>
inline void write_gemv_out(const int* in,
                           float* out,
                           const float* scale,
                           const float* bias_ptr,
                           int size,
                           bool flag_act,
                           lite_api::ActivationType act,
                           float six,
                           float alpha,
                           float offset,
                           float threshold) {
  int cnt = size >> 3;
  int remain = size & 7;
  float32x4_t vzero = vdupq_n_f32(0.f);
  int cnt_4 = remain >> 2;
  int cnt_remain = remain & 3;
  if (flag_act) {
    if (act == lite_api::ActivationType::kRelu) {
#ifdef __aarch64__
      asm volatile(
          "cmp %w[cnt], #1\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "blt 2f\n"
          "1: \n"
          "ldr q5, [%[in_ptr]], #0x10\n"
          "ldr q3, [%[bias_ptr]], #0x10\n"
          "ldr q4, [%[scale_ptr]], #0x10\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "scvtf v7.4s, v5.4s\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "fmla v3.4s, v7.4s, v4.4s\n"
          "fmax v0.4s, v0.4s, %[vzero].4s\n"
          "subs %w[cnt], %w[cnt], #1\n"
          "fmax v3.4s, v3.4s, %[vzero].4s\n"
          "str q0, [%[out_ptr]], #0x10\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "str q3, [%[out_ptr]], #0x10\n"
          "bne 1b\n"
          "2: \n"
          "cmp %w[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "fmax v0.4s, v0.4s, %[vzero].4s\n"
          "str q0, [%[out_ptr]], #0x10\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4), [cnt_remain] "r"(cnt_remain), [vzero] "w"(vzero)
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
            "v8");
#else
      asm volatile(
          "cmp %[cnt], #1\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "blt 2f\n"
          "1: \n"
          "vld1.32 {d14-d15}, [%[in_ptr]]!\n"
          "vld1.32 {d16-d17}, [%[bias_ptr]]!\n"
          "vld1.32 {d18-d19}, [%[scale_ptr]]!\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vcvt.f32.s32 q11, q7\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vmla.f32 q8, q11, q9\n"
          "vmax.f32 q5, q5, %q[vzero]\n"
          "subs %[cnt], #1\n"
          "vmax.f32 q8, q8, %q[vzero]\n"
          "vst1.32 {d10-d11}, [%[out_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vst1.32 {d16-d17}, [%[out_ptr]]!\n"
          "bne 1b\n"
          "2: \n"
          "cmp %[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vmax.f32 q5, q5, %q[vzero]\n"
          "vst1.32 {d10-d11}, [%[out_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4), [cnt_remain] "r"(cnt_remain), [vzero] "w"(vzero)
          : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
      in -= 4;
      scale -= 4;
      bias_ptr -= 4;
      for (int i = 0; i < cnt_remain; i++) {
        out[0] = *(bias_ptr++) + *(in++) * *(scale)++;
        out[0] = out[0] > 0.f ? out[0] : 0.f;
        out++;
      }
    } else if (act == lite_api::ActivationType::kRelu6) {
      float32x4_t vsix = vdupq_n_f32(six);
#ifdef __aarch64__
      asm volatile(
          "cmp %w[cnt], #1\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "blt 2f\n"
          "1: \n"
          "ldr q5, [%[in_ptr]], #0x10\n"
          "ldr q3, [%[bias_ptr]], #0x10\n"
          "ldr q4, [%[scale_ptr]], #0x10\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "scvtf v7.4s, v5.4s\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "fmla v3.4s, v7.4s, v4.4s\n"
          "fmax v0.4s, v0.4s, %[vzero].4s\n"
          "fmax v3.4s, v3.4s, %[vzero].4s\n"
          "subs %w[cnt], %w[cnt], #1\n"
          "fmin v0.4s, v0.4s, %[vsix].4s\n"
          "fmin v3.4s, v3.4s, %[vsix].4s\n"
          "str q0, [%[out_ptr]], #0x10\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "str q3, [%[out_ptr]], #0x10\n"
          "bne 1b\n"
          "2: \n"
          "cmp %w[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "fmax v0.4s, v0.4s, %[vzero].4s\n"
          "fmin v0.4s, v0.4s, %[vsix].4s\n"
          "str q0, [%[out_ptr]], #0x10\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4),
            [cnt_remain] "r"(cnt_remain),
            [vzero] "w"(vzero),
            [vsix] "w"(vsix)
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
            "v8");
#else
      asm volatile(
          "cmp %[cnt], #1\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "blt 2f\n"
          "1: \n"
          "vld1.32 {d14-d15}, [%[in_ptr]]!\n"
          "vld1.32 {d16-d17}, [%[bias_ptr]]!\n"
          "vld1.32 {d18-d19}, [%[scale_ptr]]!\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vcvt.f32.s32 q11, q7\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vmla.f32 q8, q11, q9\n"
          "vmax.f32 q5, q5, %q[vzero]\n"
          "vmax.f32 q8, q8, %q[vzero]\n"
          "subs %[cnt], #1\n"
          "vmin.f32 q5, q5, %q[vsix]\n"
          "vmin.f32 q8, q8, %q[vsix]\n"
          "vst1.32 {d10-d11}, [%[out_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vst1.32 {d16-d17}, [%[out_ptr]]!\n"
          "bne 1b\n"
          "2: \n"
          "cmp %[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vmax.f32 q5, q5, %q[vzero]\n"
          "vmin.f32 q5, q5, %q[vsix]\n"
          "vst1.32 {d10-d11}, [%[out_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4),
            [cnt_remain] "r"(cnt_remain),
            [vzero] "w"(vzero),
            [vsix] "w"(vsix)
          : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
      in -= 4;
      scale -= 4;
      bias_ptr -= 4;
      for (int i = 0; i < cnt_remain; i++) {
        out[0] = *(bias_ptr++) + *(in++) * *(scale)++;
        out[0] = out[0] > 0.f ? (out[0] < six ? out[0] : six) : 0.f;
        out++;
      }
    } else if (act == lite_api::ActivationType::kLeakyRelu) {
      float32x4_t valpha = vdupq_n_f32(alpha);
#ifdef __aarch64__
      asm volatile(
          "cmp %w[cnt], #1\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "blt 2f\n"
          "1: \n"
          "ldr q5, [%[in_ptr]], #0x10\n"
          "ldr q3, [%[bias_ptr]], #0x10\n"
          "ldr q4, [%[scale_ptr]], #0x10\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "scvtf v7.4s, v5.4s\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "fmla v3.4s, v7.4s, v4.4s\n"
          "fcmge v4.4s, v0.4s,  %[vzero].4s\n"
          "fmul v5.4s, v0.4s,  %[valpha].4s\n"
          "subs %w[cnt], %w[cnt], #1\n"
          "fcmge v6.4s, v3.4s,  %[vzero].4s\n"
          "fmul v7.4s, v3.4s,  %[valpha].4s\n"
          "bif v0.16b, v5.16b, v4.16b\n"
          "bif v3.16b, v7.16b, v6.16b\n"
          "str q0, [%[out_ptr]], #0x10\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "str q3, [%[out_ptr]], #0x10\n"
          "bne 1b\n"
          "2: \n"
          "cmp %w[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "fcmge v4.4s, v0.4s,  %[vzero].4s\n"
          "fmul v5.4s, v0.4s,  %[valpha].4s\n"
          "bif v0.16b, v5.16b, v4.16b\n"
          "str q0, [%[out_ptr]], #0x10\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4),
            [cnt_remain] "r"(cnt_remain),
            [vzero] "w"(vzero),
            [valpha] "w"(valpha)
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
            "v8");
#else
      asm volatile(
          "cmp %[cnt], #1\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "blt 2f\n"
          "1: \n"
          "vld1.32 {d14-d15}, [%[in_ptr]]!\n"
          "vld1.32 {d16-d17}, [%[bias_ptr]]!\n"
          "vld1.32 {d18-d19}, [%[scale_ptr]]!\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vcvt.f32.s32 q11, q7\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vmla.f32 q8, q11, q9\n"
          "vcge.f32 q7, q5, %q[vzero]\n"
          "vmul.f32 q9, q5, %q[valpha]\n"
          "subs %[cnt], #1\n"
          "vcge.f32 q10, q8, %q[vzero]\n"
          "vmul.f32 q11, q8, %q[valpha]\n"
          "vbif q5, q9, q7\n"
          "vbif q8, q11, q10\n"
          "vst1.32 {d10-d11}, [%[out_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vst1.32 {d16-d17}, [%[out_ptr]]!\n"
          "bne 1b\n"
          "2: \n"
          "cmp %[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vcge.f32 q7, q5, %q[vzero]\n"
          "vmul.f32 q9, q5, %q[valpha]\n"
          "vbif q5, q9, q7\n"
          "vst1.32 {d10-d11}, [%[out_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4),
            [cnt_remain] "r"(cnt_remain),
            [vzero] "w"(vzero),
            [valpha] "w"(valpha)
          : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
      in -= 4;
      scale -= 4;
      bias_ptr -= 4;
      for (int i = 0; i < cnt_remain; i++) {
        out[0] = *(bias_ptr++) + *(in++) * *(scale)++;
        out[0] = out[0] > 0.f ? out[0] : out[0] * alpha;
        out++;
      }
    } else if (act == lite_api::ActivationType::kHardSwish) {
      float32x4_t valpha = vdupq_n_f32(alpha);
      float32x4_t voffset = vdupq_n_f32(offset);
      float32x4_t vthreshold = vdupq_n_f32(threshold);
#ifdef __aarch64__
      asm volatile(
          "cmp %w[cnt], #1\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "blt 2f\n"
          "1: \n"
          "ldr q5, [%[in_ptr]], #0x10\n"
          "ldr q3, [%[bias_ptr]], #0x10\n"
          "ldr q4, [%[scale_ptr]], #0x10\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "scvtf v7.4s, v5.4s\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "fmla v3.4s, v7.4s, v4.4s\n"
          "fadd  v4.4s, v0.4s,  %[voffset].4s\n"
          "fmul  v5.4s, v0.4s,  %[valpha].4s\n"
          "fmax  v4.4s, v4.4s,  %[vzero].4s\n"
          "subs %w[cnt], %w[cnt], #1\n"
          "fadd  v6.4s, v3.4s,  %[voffset].4s\n"
          "fmin  v4.4s, v4.4s,  %[vthreshold].4s\n"
          "fmul  v7.4s, v3.4s,  %[valpha].4s\n"
          "fmax  v6.4s, v6.4s,  %[vzero].4s\n"
          "fmul  v0.4s, v5.4s,  v4.4s\n"
          "fmin  v6.4s, v6.4s,  %[vthreshold].4s\n"
          "str q0, [%[out_ptr]], #0x10\n"
          "fmul  v3.4s, v7.4s,  v6.4s\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "str q3, [%[out_ptr]], #0x10\n"
          "bne 1b\n"
          "2: \n"
          "cmp %w[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "fadd  v4.4s, v0.4s,  %[voffset].4s\n"
          "fmul  v5.4s, v0.4s,  %[valpha].4s\n"
          "fmax  v4.4s, v4.4s,  %[vzero].4s\n"
          "fmin  v4.4s, v4.4s,  %[vthreshold].4s\n"
          "fmul  v0.4s, v5.4s,  v4.4s\n"
          "str q0, [%[out_ptr]], #0x10\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4),
            [cnt_remain] "r"(cnt_remain),
            [vzero] "w"(vzero),
            [voffset] "w"(voffset),
            [vthreshold] "w"(vthreshold),
            [valpha] "w"(valpha)
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
            "v8");
#else
      asm volatile(
          "cmp %[cnt], #1\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "blt 2f\n"
          "1: \n"
          "vld1.32 {d14-d15}, [%[in_ptr]]!\n"
          "vld1.32 {d16-d17}, [%[bias_ptr]]!\n"
          "vld1.32 {d18-d19}, [%[scale_ptr]]!\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vcvt.f32.s32 q11, q7\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vmla.f32 q8, q11, q9\n"
          "vadd.f32 q7, q5, %q[voffset]\n"
          "vmul.f32 q9, q5, %q[valpha]\n"
          "vmax.f32 q7, q7, %q[vzero]\n"
          "subs %[cnt], #1\n"
          "vadd.f32 q10, q8, %q[voffset]\n"
          "vmul.f32 q11, q8, %q[valpha]\n"
          "vmin.f32 q7, q7, %q[vthreshold]\n"
          "vmax.f32 q10, q10, %q[vzero]\n"
          "vmul.f32 q5,  q7, q9\n"
          "vmin.f32 q10, q10, %q[vthreshold]\n"
          "vst1.32 {d10-d11}, [%[out_ptr]]!\n"
          "vmul.f32 q8,  q10, q11\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vst1.32 {d16-d17}, [%[out_ptr]]!\n"
          "bne 1b\n"
          "2: \n"
          "cmp %[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vadd.f32 q7, q5, %q[voffset]\n"
          "vmul.f32 q9, q5, %q[valpha]\n"
          "vmax.f32 q7, q7, %q[vzero]\n"
          "vmin.f32 q7, q7, %q[vthreshold]\n"
          "vmul.f32 q5,  q7, q9\n"
          "vst1.32 {d10-d11}, [%[out_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4),
            [cnt_remain] "r"(cnt_remain),
            [vzero] "w"(vzero),
            [voffset] "w"(voffset),
            [vthreshold] "w"(vthreshold),
            [valpha] "w"(valpha)
          : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
      in -= 4;
      scale -= 4;
      bias_ptr -= 4;
      for (int i = 0; i < cnt_remain; i++) {
        out[0] = *(bias_ptr++) + *(in++) * *(scale)++;
        auto tmp0 = std::min(std::max(out[0] + offset, 0.f), threshold);
        auto tmp1 = out[0] * alpha;
        out[0] = tmp0 * tmp1;
        out++;
      }
    } else {
      LOG(FATAL) << "it doesn't support act_type: " << flag_act;
      return;
    }
  } else {
#ifdef __aarch64__
    asm volatile(
        "cmp %w[cnt], #1\n"
        "ldr q2, [%[in_ptr]], #0x10\n"
        "ldr q0, [%[bias_ptr]], #0x10\n"
        "ldr q1, [%[scale_ptr]], #0x10\n"
        "blt 2f\n"
        "1: \n"
        "ldr q5, [%[in_ptr]], #0x10\n"
        "ldr q3, [%[bias_ptr]], #0x10\n"
        "ldr q4, [%[scale_ptr]], #0x10\n"
        // int32 -> fp32
        "scvtf v6.4s, v2.4s\n"
        "ldr q2, [%[in_ptr]], #0x10\n"
        "scvtf v7.4s, v5.4s\n"
        // din * scale + bias
        "fmla v0.4s, v6.4s, v1.4s\n"
        "ldr q1, [%[scale_ptr]], #0x10\n"
        "fmla v3.4s, v7.4s, v4.4s\n"
        "subs %w[cnt], %w[cnt], #1\n"
        "str q0, [%[out_ptr]], #0x10\n"
        "ldr q0, [%[bias_ptr]], #0x10\n"
        "str q3, [%[out_ptr]], #0x10\n"
        "bne 1b\n"
        "2: \n"
        "cmp %w[cnt_4], #1\n"
        "blt 3f\n"
        // int32 -> fp32
        "scvtf v6.4s, v2.4s\n"
        "ldr q2, [%[in_ptr]], #0x10\n"
        // din * scale + bias
        "fmla v0.4s, v6.4s, v1.4s\n"
        "ldr q1, [%[scale_ptr]], #0x10\n"
        "str q0, [%[out_ptr]], #0x10\n"
        "ldr q0, [%[bias_ptr]], #0x10\n"
        "3: \n"
        : [in_ptr] "+r"(in),
          [out_ptr] "+r"(out),
          [scale_ptr] "+r"(scale),
          [bias_ptr] "+r"(bias_ptr),
          [cnt] "+r"(cnt)
        : [cnt_4] "r"(cnt_4), [cnt_remain] "r"(cnt_remain), [vzero] "w"(vzero)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8");
#else
    asm volatile(
        "cmp %[cnt], #1\n"
        "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
        "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
        "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
        "blt 2f\n"
        "1: \n"
        "vld1.32 {d14-d15}, [%[in_ptr]]!\n"
        "vld1.32 {d16-d17}, [%[bias_ptr]]!\n"
        "vld1.32 {d18-d19}, [%[scale_ptr]]!\n"
        // int32 -> fp32
        "vcvt.f32.s32 q10, q4\n"
        "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
        "vcvt.f32.s32 q11, q7\n"
        // din * scale + bias
        "vmla.f32 q5, q10, q6\n"
        "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
        "vmla.f32 q8, q11, q9\n"
        "subs %[cnt], #1\n"
        "vst1.32 {d10-d11}, [%[out_ptr]]!\n"
        "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
        "vst1.32 {d16-d17}, [%[out_ptr]]!\n"
        "bne 1b\n"
        "2: \n"
        "cmp %[cnt_4], #1\n"
        "blt 3f\n"
        // int32 -> fp32
        "vcvt.f32.s32 q10, q4\n"
        "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
        // din * scale + bias
        "vmla.f32 q5, q10, q6\n"
        "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
        "vst1.32 {d10-d11}, [%[out_ptr]]!\n"
        "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
        "3: \n"
        : [in_ptr] "+r"(in),
          [out_ptr] "+r"(out),
          [scale_ptr] "+r"(scale),
          [bias_ptr] "+r"(bias_ptr),
          [cnt] "+r"(cnt)
        : [cnt_4] "r"(cnt_4), [cnt_remain] "r"(cnt_remain), [vzero] "w"(vzero)

        : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
    in -= 4;
    scale -= 4;
    bias_ptr -= 4;
    for (int i = 0; i < cnt_remain; i++) {
      out[0] = *(bias_ptr++) + *(in++) * *(scale)++;
      out++;
    }
  }
}

template <>
inline void write_gemv_out(const int* in,
                           signed char* out,
                           const float* scale,
                           const float* bias_ptr,
                           int size,
                           bool flag_act,
                           lite_api::ActivationType act,
                           float six,
                           float alpha,
                           float offset,
                           float threshold) {
  int cnt = size >> 3;
  int remain = size & 7;
  float32x4_t vzero = vdupq_n_f32(0.f);
  int cnt_4 = remain >> 2;
  int cnt_remain = remain & 3;
#ifdef __aarch64__
#else
  float32x4_t vfive = vdupq_n_f32(-0.5f);
#endif
  if (flag_act) {
    if (act == lite_api::ActivationType::kRelu) {
#ifdef __aarch64__
      asm volatile(
          "cmp %w[cnt], #1\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "blt 2f\n"
          "1: \n"
          "ldr q5, [%[in_ptr]], #0x10\n"
          "ldr q3, [%[bias_ptr]], #0x10\n"
          "ldr q4, [%[scale_ptr]], #0x10\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "scvtf v7.4s, v5.4s\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "fmla v3.4s, v7.4s, v4.4s\n"
          "fmax v0.4s, v0.4s, %[vzero].4s\n"
          "subs %w[cnt], %w[cnt], #1\n"
          "fmax v3.4s, v3.4s, %[vzero].4s\n"
          // fp32 - int32
          "fcvtas  v4.4s, v0.4s\n"
          "fcvtas  v5.4s, v3.4s\n"
          // int32 - int16
          "sqxtn   v0.4h, v4.4s\n"
          "sqxtn   v3.4h, v5.4s\n"
          // int16-int8
          "sqxtn  v4.8b, v0.8h\n"
          "sqxtn  v5.8b, v3.8h\n"
          "str s4, [%[out_ptr]], #0x04\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "str s5, [%[out_ptr]], #0x04\n"
          "bne 1b\n"
          "2: \n"
          "cmp %w[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "fmax v0.4s, v0.4s, %[vzero].4s\n"
          // fp32 - int32
          "fcvtas  v4.4s, v0.4s\n"
          // int32 - int16
          "sqxtn   v0.4h, v4.4s\n"
          // int16-int8
          "sqxtn  v4.8b, v0.8h\n"
          "str s4, [%[out_ptr]], #0x04\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4), [cnt_remain] "r"(cnt_remain), [vzero] "w"(vzero)
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
            "v8");
#else
      asm volatile(
          "cmp %[cnt], #1\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "blt 2f\n"
          "1: \n"
          "vld1.32 {d14-d15}, [%[in_ptr]]!\n"
          "vld1.32 {d16-d17}, [%[bias_ptr]]!\n"
          "vld1.32 {d18-d19}, [%[scale_ptr]]!\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vcvt.f32.s32 q11, q7\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vmla.f32 q8, q11, q9\n"
          "vmov.f32 q10, #0.5\n"
          "vmov.f32 q11, #0.5\n"
          "vmax.f32 q5, q5, %q[vzero]\n"
          "vmax.f32 q8, q8, %q[vzero]\n"
          // data >= -127
          "vadd.f32 q5, q5, q10\n"
          "vadd.f32 q8, q8, q11\n"
          // fp32 -> int32
          "vcvt.s32.f32  q7, q5\n"
          "vcvt.s32.f32  q9, q8\n"
          // int32 -> int16
          "vqmovn.s32 d10, q7\n"
          "vqmovn.s32 d16, q9\n"
          // int16 -> int8
          "vqmovn.s16 d14, q5\n"
          "vqmovn.s16 d18, q8\n"
          "subs %[cnt], #1\n"
          "vst1.32 {d14[0]}, [%[out_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vst1.32 {d18[0]}, [%[out_ptr]]!\n"
          "bne 1b\n"
          "2: \n"
          "cmp %[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vmov.f32 q10, #0.5\n"
          "vmax.f32 q5, q5, %q[vzero]\n"
          "vadd.f32 q5, q5, q10\n"
          // fp32 -> int32
          "vcvt.s32.f32  q7, q5\n"
          // int32 -> int16
          "vqmovn.s32 d10, q7\n"
          // int16 -> int8
          "vqmovn.s16 d14, q5\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vst1.32 {d14[0]}, [%[out_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4), [cnt_remain] "r"(cnt_remain), [vzero] "w"(vzero)
          : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
      in -= 4;
      scale -= 4;
      bias_ptr -= 4;
      for (int i = 0; i < cnt_remain; i++) {
        float tmp = *(bias_ptr++) + *(in++) * *(scale)++;
        tmp = tmp > 0.f ? tmp : 0.f;
        out[0] = saturate_cast<signed char>(roundf(tmp));
        // out[0] = out[0] < -127 ? -127 : out[0];  // -127 - 127
        out++;
      }
    } else if (act == lite_api::ActivationType::kRelu6) {
      float32x4_t vsix = vdupq_n_f32(six);
#ifdef __aarch64__
      asm volatile(
          "cmp %w[cnt], #1\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "blt 2f\n"
          "1: \n"
          "ldr q5, [%[in_ptr]], #0x10\n"
          "ldr q3, [%[bias_ptr]], #0x10\n"
          "ldr q4, [%[scale_ptr]], #0x10\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "scvtf v7.4s, v5.4s\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "fmla v3.4s, v7.4s, v4.4s\n"
          "fmax v0.4s, v0.4s, %[vzero].4s\n"
          "fmax v3.4s, v3.4s, %[vzero].4s\n"
          "subs %w[cnt], %w[cnt], #1\n"
          "fmin v0.4s, v0.4s, %[vsix].4s\n"
          "fmin v3.4s, v3.4s, %[vsix].4s\n"
          // fp32 - int32
          "fcvtas  v4.4s, v0.4s\n"
          "fcvtas  v5.4s, v3.4s\n"
          // int32 - int16
          "sqxtn   v0.4h, v4.4s\n"
          "sqxtn   v3.4h, v5.4s\n"
          // int16-int8
          "sqxtn  v4.8b, v0.8h\n"
          "sqxtn  v5.8b, v3.8h\n"
          "str s4, [%[out_ptr]], #0x04\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "str s5, [%[out_ptr]], #0x04\n"
          "bne 1b\n"
          "2: \n"
          "cmp %w[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "fmax v0.4s, v0.4s, %[vzero].4s\n"
          "fmin v0.4s, v0.4s, %[vsix].4s\n"
          // fp32 - int32
          "fcvtas  v4.4s, v0.4s\n"
          // int32 - int16
          "sqxtn   v0.4h, v4.4s\n"
          // int16-int8
          "sqxtn  v4.8b, v0.8h\n"
          "str s4, [%[out_ptr]], #0x04\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4),
            [cnt_remain] "r"(cnt_remain),
            [vzero] "w"(vzero),
            [vsix] "w"(vsix)
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
            "v8");
#else
      asm volatile(
          "cmp %[cnt], #1\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "blt 2f\n"
          "1: \n"
          "vld1.32 {d14-d15}, [%[in_ptr]]!\n"
          "vld1.32 {d16-d17}, [%[bias_ptr]]!\n"
          "vld1.32 {d18-d19}, [%[scale_ptr]]!\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vcvt.f32.s32 q11, q7\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vmla.f32 q8, q11, q9\n"
          "vmov.f32 q10, #0.5\n"
          "vmov.f32 q11, #0.5\n"
          "vmax.f32 q5, q5, %q[vzero]\n"
          "vmax.f32 q8, q8, %q[vzero]\n"
          "subs %[cnt], #1\n"
          "vmin.f32 q5, q5, %q[vsix]\n"
          "vmin.f32 q8, q8, %q[vsix]\n"
          // data >= -127
          "vadd.f32 q5, q5, q10\n"
          "vadd.f32 q8, q8, q11\n"
          // fp32 -> int32
          "vcvt.s32.f32  q7, q5\n"
          "vcvt.s32.f32  q9, q8\n"
          // int32 -> int16
          "vqmovn.s32 d10, q7\n"
          "vqmovn.s32 d16, q9\n"
          // int16 -> int8
          "vqmovn.s16 d14, q5\n"
          "vqmovn.s16 d18, q8\n"
          "vst1.32 {d14[0]}, [%[out_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vst1.32 {d18[0]}, [%[out_ptr]]!\n"
          "bne 1b\n"
          "2: \n"
          "cmp %[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vmov.f32 q10, #0.5\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vmax.f32 q5, q5, %q[vzero]\n"
          "vmin.f32 q5, q5, %q[vsix]\n"
          // data >= -127
          "vadd.f32 q5, q5, q10\n"
          // fp32 -> int32
          "vcvt.s32.f32  q7, q5\n"
          // int32 -> int16
          "vqmovn.s32 d10, q7\n"
          // int16 -> int8
          "vqmovn.s16 d14, q5\n"
          "vst1.32 {d14[0]}, [%[out_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4),
            [cnt_remain] "r"(cnt_remain),
            [vzero] "w"(vzero),
            [vsix] "w"(vsix)
          : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
      in -= 4;
      scale -= 4;
      bias_ptr -= 4;
      for (int i = 0; i < cnt_remain; i++) {
        float tmp = *(bias_ptr++) + *(in++) * *(scale)++;
        tmp = tmp > 0.f ? (tmp < six ? tmp : six) : 0.f;
        out[0] = saturate_cast<signed char>(roundf(tmp));
        out++;
      }
    } else if (act == lite_api::ActivationType::kLeakyRelu) {
      float32x4_t vmax = vdupq_n_f32(-127.f);
      float32x4_t valpha = vdupq_n_f32(alpha);
#ifdef __aarch64__
      asm volatile(
          "cmp %w[cnt], #1\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "blt 2f\n"
          "1: \n"
          "ldr q5, [%[in_ptr]], #0x10\n"
          "ldr q3, [%[bias_ptr]], #0x10\n"
          "ldr q4, [%[scale_ptr]], #0x10\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "scvtf v7.4s, v5.4s\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "fmla v3.4s, v7.4s, v4.4s\n"
          "fcmge v4.4s, v0.4s,  %[vzero].4s\n"
          "fmul v5.4s, v0.4s,  %[valpha].4s\n"
          "subs %w[cnt], %w[cnt], #1\n"
          "fcmge v6.4s, v3.4s,  %[vzero].4s\n"
          "fmul v7.4s, v3.4s,  %[valpha].4s\n"
          "bif v0.16b, v5.16b, v4.16b\n"
          "bif v3.16b, v7.16b, v6.16b\n"
          // out >= -127
          "fcmge v4.4s, v0.4s, %[vmax].4s\n"
          "fcmge v5.4s, v3.4s, %[vmax].4s\n"
          "bif v0.16b, %[vmax].16b, v4.16b\n"
          "bif v3.16b, %[vmax].16b, v5.16b\n"
          // fp32 - int32
          "fcvtas  v4.4s, v0.4s\n"
          "fcvtas  v5.4s, v3.4s\n"
          // int32 - int16
          "sqxtn   v0.4h, v4.4s\n"
          "sqxtn   v3.4h, v5.4s\n"
          // int16-int8
          "sqxtn  v4.8b, v0.8h\n"
          "sqxtn  v5.8b, v3.8h\n"
          "str s4, [%[out_ptr]], #0x04\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "str s5, [%[out_ptr]], #0x04\n"
          "bne 1b\n"
          "2: \n"
          "cmp %w[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "fcmge v4.4s, v0.4s,  %[vzero].4s\n"
          "fmul v5.4s, v0.4s,  %[valpha].4s\n"
          "bif v0.16b, v5.16b, v4.16b\n"
          // out >= -127
          "fcmge v4.4s, v0.4s, %[vmax].4s\n"
          "bif v0.16b, %[vmax].16b, v4.16b\n"
          // fp32 - int32
          "fcvtas  v4.4s, v0.4s\n"
          // int32 - int16
          "sqxtn   v0.4h, v4.4s\n"
          // int16-int8
          "sqxtn  v4.8b, v0.8h\n"
          "str s4, [%[out_ptr]], #0x04\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4),
            [cnt_remain] "r"(cnt_remain),
            [vzero] "w"(vzero),
            [valpha] "w"(valpha),
            [vmax] "w"(vmax)
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
            "v8");
#else
      asm volatile(
          "cmp %[cnt], #1\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "blt 2f\n"
          "1: \n"
          "vld1.32 {d14-d15}, [%[in_ptr]]!\n"
          "vld1.32 {d16-d17}, [%[bias_ptr]]!\n"
          "vld1.32 {d18-d19}, [%[scale_ptr]]!\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vcvt.f32.s32 q11, q7\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vmov.f32 q12, #0.5\n"
          "vmov.f32 q13, #0.5\n"
          "vmla.f32 q8, q11, q9\n"
          "vcge.f32 q7, q5, %q[vzero]\n"
          "vmul.f32 q9, q5, %q[valpha]\n"
          "subs %[cnt], #1\n"
          "vcge.f32 q10, q8, %q[vzero]\n"
          "vmul.f32 q11, q8, %q[valpha]\n"
          "vbif q5, q9, q7\n"
          "vbif q8, q11, q10\n"
          "vcge.f32 q7, q5, %q[vzero]\n"
          "vcge.f32 q10, q8, %q[vzero]\n"
          // +/-0.5
          "vbif q12, %q[vfive], q7\n"
          "vbif q13, %q[vfive], q10\n"
          "vadd.f32 q5, q5, q12\n"
          "vadd.f32 q8, q8, q13\n"
          // data >= -127
          "vcge.f32 q7, q5, %q[vmax]\n"
          "vcge.f32 q9, q8, %q[vmax]\n"
          "vbif q5, %q[vmax], q7\n"
          "vbif q8, %q[vmax], q9\n"
          // fp32 -> int32
          "vcvt.s32.f32  q7, q5\n"
          "vcvt.s32.f32  q9, q8\n"
          // int32 -> int16
          "vqmovn.s32 d10, q7\n"
          "vqmovn.s32 d16, q9\n"
          // int16 -> int8
          "vqmovn.s16 d14, q5\n"
          "vqmovn.s16 d18, q8\n"
          "vst1.32 {d14[0]}, [%[out_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vst1.32 {d18[0]}, [%[out_ptr]]!\n"
          "bne 1b\n"
          "2: \n"
          "cmp %[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vmov.f32 q12, #0.5\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vcge.f32 q7, q5, %q[vzero]\n"
          "vmul.f32 q9, q5, %q[valpha]\n"
          "vbif q5, q9, q7\n"
          "vbif q12, %q[vfive], q7\n"
          "vadd.f32 q5, q5, q12\n"
          // data >= -127
          "vcge.f32 q7, q5, %q[vmax]\n"
          "vbif q5, %q[vmax], q7\n"
          // fp32 -> int32
          "vcvt.s32.f32  q7, q5\n"
          // int32 -> int16
          "vqmovn.s32 d10, q7\n"
          // int16 -> int8
          "vqmovn.s16 d14, q5\n"
          "vst1.32 {d14[0]}, [%[out_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4),
            [cnt_remain] "r"(cnt_remain),
            [vzero] "w"(vzero),
            [valpha] "w"(valpha),
            [vmax] "w"(vmax),
            [vfive] "w"(vfive)
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
            "q13");
#endif
      in -= 4;
      scale -= 4;
      bias_ptr -= 4;
      for (int i = 0; i < cnt_remain; i++) {
        float tmp = *(bias_ptr++) + *(in++) * *(scale)++;
        tmp = tmp > 0.f ? tmp : tmp * alpha;
        out[0] = saturate_cast<signed char>(roundf(tmp));
        out[0] = out[0] < -127 ? -127 : out[0];  // -127 - 127
        out++;
      }
    } else if (act == lite_api::ActivationType::kHardSwish) {
      float32x4_t valpha = vdupq_n_f32(alpha);
      float32x4_t voffset = vdupq_n_f32(offset);
      float32x4_t vthreshold = vdupq_n_f32(threshold);
      float32x4_t vmax = vdupq_n_f32(-127.f);

#ifdef __aarch64__
      asm volatile(
          "cmp %w[cnt], #1\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "blt 2f\n"
          "1: \n"
          "ldr q5, [%[in_ptr]], #0x10\n"
          "ldr q3, [%[bias_ptr]], #0x10\n"
          "ldr q4, [%[scale_ptr]], #0x10\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          "scvtf v7.4s, v5.4s\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr   q1, [%[scale_ptr]], #0x10\n"
          "fmla  v3.4s, v7.4s, v4.4s\n"
          "fadd  v4.4s, v0.4s,  %[voffset].4s\n"
          "fmul  v5.4s, v0.4s,  %[valpha].4s\n"
          "fmax  v4.4s, v4.4s,  %[vzero].4s\n"
          "subs %w[cnt], %w[cnt], #1\n"
          "fadd  v6.4s, v3.4s,  %[voffset].4s\n"
          "fmul  v7.4s, v3.4s,  %[valpha].4s\n"
          "fmax  v6.4s, v6.4s,  %[vzero].4s\n"
          "fmin  v4.4s, v4.4s,  %[vthreshold].4s\n"
          "fmul  v0.4s, v4.4s,  v5.4s\n"
          "fmin  v6.4s, v6.4s,  %[vthreshold].4s\n"
          "fmul  v3.4s, v6.4s,  v7.4s\n"
          // out >= -127
          "fcmge v4.4s, v0.4s, %[vmax].4s\n"
          "fcmge v5.4s, v3.4s, %[vmax].4s\n"
          "bif v0.16b, %[vmax].16b, v4.16b\n"
          "bif v3.16b, %[vmax].16b, v5.16b\n"
          // fp32 - int32
          "fcvtas  v4.4s, v0.4s\n"
          "fcvtas  v5.4s, v3.4s\n"
          // int32 - int16
          "sqxtn   v0.4h, v4.4s\n"
          "sqxtn   v3.4h, v5.4s\n"
          // int16-int8
          "sqxtn  v4.8b, v0.8h\n"
          "sqxtn  v5.8b, v3.8h\n"
          "str s4, [%[out_ptr]], #0x04\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "str s5, [%[out_ptr]], #0x04\n"
          "bne 1b\n"
          "2: \n"
          "cmp %w[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "scvtf v6.4s, v2.4s\n"
          "ldr q2, [%[in_ptr]], #0x10\n"
          // din * scale + bias
          "fmla v0.4s, v6.4s, v1.4s\n"
          "ldr q1, [%[scale_ptr]], #0x10\n"
          "fadd  v4.4s, v0.4s,  %[voffset].4s\n"
          "fmul  v5.4s, v0.4s,  %[valpha].4s\n"
          "fmax  v4.4s, v4.4s,  %[vzero].4s\n"
          "fmin  v4.4s, v4.4s,  %[vthreshold].4s\n"
          "fmul  v0.4s, v4.4s,  v5.4s\n"
          // out >= -127
          "fcmge v4.4s, v0.4s, %[vmax].4s\n"
          "bif v0.16b, %[vmax].16b, v4.16b\n"
          // fp32 - int32
          "fcvtas  v4.4s, v0.4s\n"
          // int32 - int16
          "sqxtn   v0.4h, v4.4s\n"
          // int16-int8
          "sqxtn  v4.8b, v0.8h\n"
          "str s4, [%[out_ptr]], #0x04\n"
          "ldr q0, [%[bias_ptr]], #0x10\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4),
            [cnt_remain] "r"(cnt_remain),
            [vzero] "w"(vzero),
            [valpha] "w"(valpha),
            [voffset] "w"(voffset),
            [vthreshold] "w"(vthreshold),
            [vmax] "w"(vmax)
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
            "v8");
#else
      asm volatile(
          "cmp %[cnt], #1\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "blt 2f\n"
          "1: \n"
          "vld1.32 {d14-d15}, [%[in_ptr]]!\n"
          "vld1.32 {d16-d17}, [%[bias_ptr]]!\n"
          "vld1.32 {d18-d19}, [%[scale_ptr]]!\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          "vcvt.f32.s32 q11, q7\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vmov.f32 q12, #0.5\n"
          "vmov.f32 q13, #0.5\n"
          "vmla.f32 q8, q11, q9\n"
          "vadd.f32 q7, q5, %q[voffset]\n"
          "vmul.f32 q9, q5, %q[valpha]\n"
          "vmax.f32 q7, q7, %q[vzero]\n"
          "subs %[cnt], #1\n"
          "vadd.f32 q10, q8, %q[voffset]\n"
          "vmul.f32 q11, q8, %q[valpha]\n"
          "vmax.f32 q10, q10, %q[vzero]\n"
          "vmin.f32 q7, q7, %q[vthreshold]\n"
          "vmul.f32 q5, q7, q9\n"
          "vcge.f32 q7, q5, %q[vzero]\n"
          "vmin.f32 q10, q10, %q[vthreshold]\n"
          "vmul.f32 q8, q10, q11\n"
          "vcge.f32 q10, q8, %q[vzero]\n"
          // +/-0.5
          "vbif q12, %q[vfive], q7\n"
          "vbif q13, %q[vfive], q10\n"
          "vadd.f32 q5, q5, q12\n"
          "vadd.f32 q8, q8, q13\n"
          // data >= -127
          "vcge.f32 q7, q5, %q[vmax]\n"
          "vcge.f32 q9, q8, %q[vmax]\n"
          "vbif q5, %q[vmax], q7\n"
          "vbif q8, %q[vmax], q9\n"
          // fp32 -> int32
          "vcvt.s32.f32  q7, q5\n"
          "vcvt.s32.f32  q9, q8\n"
          // int32 -> int16
          "vqmovn.s32 d10, q7\n"
          "vqmovn.s32 d16, q9\n"
          // int16 -> int8
          "vqmovn.s16 d14, q5\n"
          "vqmovn.s16 d18, q8\n"
          "vst1.32 {d14[0]}, [%[out_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "vst1.32 {d18[0]}, [%[out_ptr]]!\n"
          "bne 1b\n"
          "2: \n"
          "cmp %[cnt_4], #1\n"
          "blt 3f\n"
          // int32 -> fp32
          "vcvt.f32.s32 q10, q4\n"
          "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
          // din * scale + bias
          "vmla.f32 q5, q10, q6\n"
          "vmov.f32 q12, #0.5\n"
          "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
          "vadd.f32 q7, q5, %q[voffset]\n"
          "vmul.f32 q9, q5, %q[valpha]\n"
          "vmax.f32 q7, q7, %q[vzero]\n"
          "vmin.f32 q7, q7, %q[vthreshold]\n"
          "vmul.f32 q5, q7, q9\n"
          "vcge.f32 q7, q5, %q[vzero]\n"
          "vbif q12, %q[vfive], q7\n"
          "vadd.f32 q5, q5, q12\n"
          // data >= -127
          "vcge.f32 q7, q5, %q[vmax]\n"
          "vbif q5, %q[vmax], q7\n"
          // fp32 -> int32
          "vcvt.s32.f32  q7, q5\n"
          // int32 -> int16
          "vqmovn.s32 d10, q7\n"
          // int16 -> int8
          "vqmovn.s16 d14, q5\n"
          "vst1.32 {d14[0]}, [%[out_ptr]]!\n"
          "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
          "3: \n"
          : [in_ptr] "+r"(in),
            [out_ptr] "+r"(out),
            [scale_ptr] "+r"(scale),
            [bias_ptr] "+r"(bias_ptr),
            [cnt] "+r"(cnt)
          : [cnt_4] "r"(cnt_4),
            [cnt_remain] "r"(cnt_remain),
            [vzero] "w"(vzero),
            [valpha] "w"(valpha),
            [voffset] "w"(voffset),
            [vthreshold] "w"(vthreshold),
            [vfive] "w"(vfive),
            [vmax] "w"(vmax)
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
            "q13");
#endif
      in -= 4;
      scale -= 4;
      bias_ptr -= 4;
      for (int i = 0; i < cnt_remain; i++) {
        float tmp = *(bias_ptr++) + *(in++) * *(scale)++;
        auto tmp0 = std::min(std::max(tmp + offset, 0.f), threshold);
        auto tmp1 = tmp * alpha;
        auto result = tmp0 * tmp1;
        out[0] = saturate_cast<signed char>(roundf(result));
        out[0] = out[0] < -127 ? -127 : out[0];  // -127 - 127
        out++;
      }
    } else {
      LOG(FATAL) << "it doesn't support act_type: " << flag_act;
      return;
    }
  } else {
    float32x4_t vmax = vdupq_n_f32(-127.f);
#ifdef __aarch64__
    asm volatile(
        "cmp %w[cnt], #1\n"
        "ldr q2, [%[in_ptr]], #0x10\n"
        "ldr q0, [%[bias_ptr]], #0x10\n"
        "ldr q1, [%[scale_ptr]], #0x10\n"
        "blt 2f\n"
        "1: \n"
        "ldr q5, [%[in_ptr]], #0x10\n"
        "ldr q3, [%[bias_ptr]], #0x10\n"
        "ldr q4, [%[scale_ptr]], #0x10\n"
        // int32 -> fp32
        "scvtf v6.4s, v2.4s\n"
        "ldr q2, [%[in_ptr]], #0x10\n"
        "scvtf v7.4s, v5.4s\n"
        // din * scale + bias
        "fmla v0.4s, v6.4s, v1.4s\n"
        "ldr q1, [%[scale_ptr]], #0x10\n"
        "fmla v3.4s, v7.4s, v4.4s\n"
        "subs %w[cnt], %w[cnt], #1\n"
        // out >= -127
        "fcmge v4.4s, v0.4s, %[vmax].4s\n"
        "fcmge v5.4s, v3.4s, %[vmax].4s\n"
        "bif v0.16b, %[vmax].16b, v4.16b\n"
        "bif v3.16b, %[vmax].16b, v5.16b\n"
        // fp32 - int32
        "fcvtas  v4.4s, v0.4s\n"
        "fcvtas  v5.4s, v3.4s\n"
        // int32 - int16
        "sqxtn   v0.4h, v4.4s\n"
        "sqxtn   v3.4h, v5.4s\n"
        // int16-int8
        "sqxtn  v4.8b, v0.8h\n"
        "sqxtn  v5.8b, v3.8h\n"
        "str s4, [%[out_ptr]], #0x04\n"
        "ldr q0, [%[bias_ptr]], #0x10\n"
        "str s5, [%[out_ptr]], #0x04\n"
        "bne 1b\n"
        "2: \n"
        "cmp %w[cnt_4], #1\n"
        "blt 3f\n"
        // int32 -> fp32
        "scvtf v6.4s, v2.4s\n"
        "ldr q2, [%[in_ptr]], #0x10\n"
        // din * scale + bias
        "fmla v0.4s, v6.4s, v1.4s\n"
        "ldr q1, [%[scale_ptr]], #0x10\n"
        // out >= -127
        "fcmge v4.4s, v0.4s, %[vmax].4s\n"
        "bif v0.16b, %[vmax].16b, v4.16b\n"
        // fp32 - int32
        "fcvtas  v4.4s, v0.4s\n"
        // int32 - int16
        "sqxtn   v0.4h, v4.4s\n"
        // int16-int8
        "sqxtn  v4.8b, v0.8h\n"
        "str s4, [%[out_ptr]], #0x04\n"
        "ldr q0, [%[bias_ptr]], #0x10\n"
        "3: \n"
        : [in_ptr] "+r"(in),
          [out_ptr] "+r"(out),
          [scale_ptr] "+r"(scale),
          [bias_ptr] "+r"(bias_ptr),
          [cnt] "+r"(cnt)
        : [cnt_4] "r"(cnt_4),
          [cnt_remain] "r"(cnt_remain),
          [vzero] "w"(vzero),
          [vmax] "w"(vmax)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8");
#else
    asm volatile(
        "cmp %[cnt], #1\n"
        "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
        "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
        "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
        "blt 2f\n"
        "1: \n"
        "vld1.32 {d14-d15}, [%[in_ptr]]!\n"
        "vld1.32 {d16-d17}, [%[bias_ptr]]!\n"
        "vld1.32 {d18-d19}, [%[scale_ptr]]!\n"
        // int32 -> fp32
        "vcvt.f32.s32 q10, q4\n"
        "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
        "vcvt.f32.s32 q11, q7\n"
        // din * scale + bias
        "vmla.f32 q5, q10, q6\n"
        "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
        "vmla.f32 q8, q11, q9\n"
        "vmov.f32 q10, #0.5\n"
        "vmov.f32 q11, #0.5\n"
        "vcge.f32 q7, q5, %q[vzero]\n"
        "vcge.f32 q9, q8, %q[vzero]\n"
        // +/-0.5
        "vbif q10, %q[vfive], q7\n"
        "vbif q11, %q[vfive], q9\n"
        "vadd.f32 q5, q5, q10\n"
        "vadd.f32 q8, q8, q11\n"
        // data >= -127
        "vcge.f32 q7, q5, %q[vmax]\n"
        "vcge.f32 q9, q8, %q[vmax]\n"
        "vbif q5, %q[vmax], q7\n"
        "vbif q8, %q[vmax], q9\n"
        // fp32 -> int32
        "vcvt.s32.f32  q7, q5\n"
        "vcvt.s32.f32  q9, q8\n"
        // int32 -> int16
        "vqmovn.s32 d10, q7\n"
        "vqmovn.s32 d16, q9\n"
        // int16 -> int8
        "vqmovn.s16 d14, q5\n"
        "vqmovn.s16 d18, q8\n"
        "subs %[cnt], #1\n"
        "vst1.32 {d14[0]}, [%[out_ptr]]!\n"
        "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
        "vst1.32 {d18[0]}, [%[out_ptr]]!\n"
        "bne 1b\n"
        "2: \n"
        "cmp %[cnt_4], #1\n"
        "blt 3f\n"
        // int32 -> fp32
        "vcvt.f32.s32 q10, q4\n"
        "vld1.32 {d8-d9}, [%[in_ptr]]!\n"
        // din * scale + bias
        "vmla.f32 q5, q10, q6\n"
        "vmov.f32 q11, #0.5\n"
        "vcge.f32 q7, q5, %q[vzero]\n"
        // +/-0.5
        "vbif q11, %q[vfive], q7\n"
        "vadd.f32 q5, q5, q11\n"
        // data >= -127
        "vcge.f32 q7, q5, %q[vmax]\n"
        "vbif q5, %q[vmax], q7\n"
        // fp32 -> int32
        "vcvt.s32.f32  q7, q5\n"
        // int32 -> int16
        "vqmovn.s32 d10, q7\n"
        // int16 -> int8
        "vqmovn.s16 d14, q5\n"
        "vld1.32 {d12-d13}, [%[scale_ptr]]!\n"
        "vst1.32 {d14[0]}, [%[out_ptr]]!\n"
        "vld1.32 {d10-d11}, [%[bias_ptr]]!\n"
        "3: \n"
        : [in_ptr] "+r"(in),
          [out_ptr] "+r"(out),
          [scale_ptr] "+r"(scale),
          [bias_ptr] "+r"(bias_ptr),
          [cnt] "+r"(cnt)
        : [cnt_4] "r"(cnt_4),
          [cnt_remain] "r"(cnt_remain),
          [vzero] "w"(vzero),
          [vmax] "w"(vmax),
          [vfive] "w"(vfive)
        : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
#endif
    in -= 4;
    scale -= 4;
    bias_ptr -= 4;
    for (int i = 0; i < cnt_remain; i++) {
      float tmp = *(bias_ptr++) + *(in++) * *(scale)++;
      out[0] = saturate_cast<signed char>(roundf(tmp));
      out[0] = out[0] < -127 ? -127 : out[0];  // -127 - 127
      out++;
    }
  }
}

template <typename dtype>
bool gemv_int8_trans_oth(const int8_t* A,
                         const int8_t* x,
                         dtype* y,
                         int M,
                         int N,
                         const float* scale,
                         bool is_bias,
                         const float* bias,
                         bool flag_act,
                         lite_api::ActivationType act,
                         float alpha,
                         float offset,
                         float threshold,
                         ARMContext* ctx) {
  dtype* data_out = y;
  const int8_t* data_in = A;
  const int8_t* weights_ptr = x;
  int out_cnt = M >> 4;
  int out_remain = M & 15;
  int* zero_ptr = new int[M + 16];
  float* zerobuf = new float[M + 16];
  memset(zero_ptr, 0, sizeof(int) * (M + 16));
  memset(zerobuf, 0, sizeof(float) * (M + 16));
  const float* bias_ptr = is_bias ? bias : zerobuf;
  float six = alpha;

#ifdef __aarch64__
  int cnt = N >> 3;
  int tail = N & 7;
  int cnt_4 = tail >> 2;
  int tail_4 = tail & 3;
  int stride_in = M << 3;
  // #pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const int8_t* in_ptr0 = data_in;
    const int8_t* in_ptr1 = in_ptr0 + M;
    const int8_t* in_ptr2 = in_ptr1 + M;
    const int8_t* in_ptr3 = in_ptr2 + M;
    const int8_t* in_ptr4 = in_ptr3 + M;
    const int8_t* in_ptr5 = in_ptr4 + M;
    const int8_t* in_ptr6 = in_ptr5 + M;
    const int8_t* in_ptr7 = in_ptr6 + M;
    const int8_t* wei_ptr = weights_ptr;
    int32_t* out_ptr = zero_ptr;
    int cnt_col = out_cnt;
    asm volatile(
        "prfm  pldl1keep, [%[wc]]        \n"
        "prfm  pldl1keep, [%[in_ptr0]]   \n"
        "prfm  pldl1keep, [%[in_ptr1]]   \n"
        "prfm  pldl1keep, [%[in_ptr2]]   \n"
        "prfm  pldl1keep, [%[in_ptr3]]   \n"
        "prfm  pldl1keep, [%[in_ptr4]]   \n"
        "prfm  pldl1keep, [%[in_ptr5]]   \n"
        "prfm  pldl1keep, [%[in_ptr6]]   \n"
        "prfm  pldl1keep, [%[in_ptr7]]   \n"
        "cmp %w[cnt], #1\n"
        "ldr q8, [%[wc]]\n"
        "ldr q9, [%[in_ptr0]], #0x10\n"
        "ldr q10, [%[in_ptr1]], #0x10\n"
        "ldr q11, [%[in_ptr2]], #0x10\n"
        "ldr q12, [%[in_ptr3]], #0x10\n"
        "sxtl  v0.8h, v8.8b\n"
        "blt 2f\n"
        "1: \n"
        "ldr q13, [%[out_ptr]]\n"
        "sxtl  v1.8h, v9.8b\n"
        "sxtl2 v2.8h, v9.16b\n"
        "ldr q14, [%[out_ptr], #0x10]\n"
        "sxtl  v3.8h, v10.8b\n"
        "sxtl2 v4.8h, v10.16b\n"
        "ldr q15, [%[out_ptr], #0x20]\n"
        "sxtl  v5.8h, v11.8b\n"
        "sxtl2 v6.8h, v11.16b\n"
        "ldr q16, [%[out_ptr], #0x30]\n"
        "ldr q9, [%[in_ptr4]], #0x10\n"
        // r0
        "smlal v13.4s, v1.4h, v0.h[0]\n"
        "smlal2 v14.4s, v1.8h, v0.h[0]\n"
        "sxtl  v7.8h, v12.8b\n"
        "sxtl2 v8.8h, v12.16b\n"
        "ldr q10, [%[in_ptr5]], #0x10\n"
        "smlal v15.4s, v2.4h, v0.h[0]\n"
        "smlal2 v16.4s, v2.8h, v0.h[0]\n"
        "sxtl  v17.8h, v9.8b\n"
        "sxtl2 v18.8h, v9.16b\n"
        "ldr q11, [%[in_ptr6]], #0x10\n"
        // r1
        "smlal v13.4s, v3.4h, v0.h[1]\n"
        "smlal2 v14.4s, v3.8h, v0.h[1]\n"
        "sxtl  v19.8h, v10.8b\n"
        "sxtl2 v20.8h, v10.16b\n"
        "ldr q12, [%[in_ptr7]], #0x10\n"
        "smlal v15.4s, v4.4h, v0.h[1]\n"
        "smlal2 v16.4s, v4.8h, v0.h[1]\n"
        "sxtl  v1.8h, v11.8b\n"
        "sxtl2 v2.8h, v11.16b\n"
        // r2
        "smlal v13.4s, v5.4h, v0.h[2]\n"
        "smlal2 v14.4s, v5.8h, v0.h[2]\n"
        "sxtl  v3.8h, v12.8b\n"
        "sxtl2 v4.8h, v12.16b\n"
        "smlal v15.4s, v6.4h, v0.h[2]\n"
        "smlal2 v16.4s, v6.8h, v0.h[2]\n"
        // r3
        "smlal v13.4s, v7.4h, v0.h[3]\n"
        "smlal2 v14.4s, v7.8h, v0.h[3]\n"
        "ldr q9, [%[in_ptr0]], #0x10\n"
        "smlal v15.4s, v8.4h, v0.h[3]\n"
        "smlal2 v16.4s, v8.8h, v0.h[3]\n"
        // r4
        "smlal v13.4s, v17.4h, v0.h[4]\n"
        "smlal2 v14.4s, v17.8h, v0.h[4]\n"
        "ldr q10, [%[in_ptr1]], #0x10\n"
        "smlal v15.4s, v18.4h, v0.h[4]\n"
        "smlal2 v16.4s, v18.8h, v0.h[4]\n"
        // r5
        "smlal v13.4s, v19.4h, v0.h[5]\n"
        "smlal2 v14.4s, v19.8h, v0.h[5]\n"
        "ldr q11, [%[in_ptr2]], #0x10\n"
        "smlal v15.4s, v20.4h, v0.h[5]\n"
        "smlal2 v16.4s, v20.8h, v0.h[5]\n"
        // r6
        "smlal v13.4s, v1.4h, v0.h[6]\n"
        "smlal2 v14.4s, v1.8h, v0.h[6]\n"
        "ldr q12, [%[in_ptr3]], #0x10\n"
        "smlal v15.4s, v2.4h, v0.h[6]\n"
        "smlal2 v16.4s, v2.8h, v0.h[6]\n"
        // r7
        "smlal v13.4s, v3.4h, v0.h[7]\n"
        "smlal2 v14.4s, v3.8h, v0.h[7]\n"
        "smlal v15.4s, v4.4h, v0.h[7]\n"
        "smlal2 v16.4s, v4.8h, v0.h[7]\n"
        "subs %w[cnt], %w[cnt], #1\n"
        "str q13, [%[out_ptr]], #0x10\n"
        "str q14, [%[out_ptr]], #0x10\n"
        "str q15, [%[out_ptr]], #0x10\n"
        "str q16, [%[out_ptr]], #0x10\n"
        "bne 1b\n"
        "2: \n"
        : [out_ptr] "+r"(out_ptr),
          [in_ptr0] "+r"(in_ptr0),
          [in_ptr1] "+r"(in_ptr1),
          [in_ptr2] "+r"(in_ptr2),
          [in_ptr3] "+r"(in_ptr3),
          [in_ptr4] "+r"(in_ptr4),
          [in_ptr5] "+r"(in_ptr5),
          [in_ptr6] "+r"(in_ptr6),
          [in_ptr7] "+r"(in_ptr7),
          [cnt] "+r"(cnt_col)
        : [wc] "r"(wei_ptr)
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
          "v15",
          "v16",
          "v17",
          "v18",
          "v19",
          "v20");
    in_ptr0 -= 16;
    in_ptr1 -= 16;
    in_ptr2 -= 16;
    in_ptr3 -= 16;
    for (int j = 0; j < out_remain; j++) {
      *out_ptr += *in_ptr0++ * wei_ptr[0];
      *out_ptr += *in_ptr1++ * wei_ptr[1];
      *out_ptr += *in_ptr2++ * wei_ptr[2];
      *out_ptr += *in_ptr3++ * wei_ptr[3];
      *out_ptr += *in_ptr4++ * wei_ptr[4];
      *out_ptr += *in_ptr5++ * wei_ptr[5];
      *out_ptr += *in_ptr6++ * wei_ptr[6];
      *out_ptr += *in_ptr7++ * wei_ptr[7];
      out_ptr++;
    }
    data_in += stride_in;
    weights_ptr += 8;
  }
  if (cnt_4) {
    const int8_t* in_ptr0 = data_in;
    const int8_t* in_ptr1 = in_ptr0 + M;
    const int8_t* in_ptr2 = in_ptr1 + M;
    const int8_t* in_ptr3 = in_ptr2 + M;
    int32_t* out_ptr = zero_ptr;
    int cnt_col = out_cnt;
    const int8_t* wei_ptr = weights_ptr;
    asm volatile(
        "prfm  pldl1keep, [%[wc]]        \n"
        "prfm  pldl1keep, [%[in_ptr0]]   \n"
        "prfm  pldl1keep, [%[in_ptr1]]   \n"
        "prfm  pldl1keep, [%[in_ptr2]]   \n"
        "prfm  pldl1keep, [%[in_ptr3]]   \n"
        "cmp %w[cnt], #1\n"
        "ldr q8, [%[wc]]\n"
        "ldr q9, [%[in_ptr0]], #0x10\n"
        "ldr q10, [%[in_ptr1]], #0x10\n"
        "ldr q11, [%[in_ptr2]], #0x10\n"
        "sxtl  v0.8h, v8.8b\n"
        "ldr q12, [%[in_ptr3]], #0x10\n"
        "blt 2f\n"
        "1: \n"
        "ldr q13, [%[out_ptr]]\n"
        "sxtl  v1.8h, v9.8b\n"
        "sxtl2 v2.8h, v9.16b\n"
        "ldr q14, [%[out_ptr], #0x10]\n"
        "sxtl  v3.8h, v10.8b\n"
        "sxtl2 v4.8h, v10.16b\n"
        "ldr q15, [%[out_ptr], #0x20]\n"
        "sxtl  v5.8h, v11.8b\n"
        "sxtl2 v6.8h, v11.16b\n"
        "ldr q16, [%[out_ptr], #0x30]\n"
        // r0
        "smlal v13.4s, v1.4h, v0.h[0]\n"
        "smlal2 v14.4s, v1.8h, v0.h[0]\n"
        "sxtl  v7.8h, v12.8b\n"
        "sxtl2 v8.8h, v12.16b\n"
        "ldr q9, [%[in_ptr0]], #0x10\n"
        "smlal v15.4s, v2.4h, v0.h[0]\n"
        "smlal2 v16.4s, v2.8h, v0.h[0]\n"
        // r1
        "smlal v13.4s, v3.4h, v0.h[1]\n"
        "smlal2 v14.4s, v3.8h, v0.h[1]\n"
        "ldr q10, [%[in_ptr1]], #0x10\n"
        "smlal v15.4s, v4.4h, v0.h[1]\n"
        "smlal2 v16.4s, v4.8h, v0.h[1]\n"
        // r2
        "smlal v13.4s, v5.4h, v0.h[2]\n"
        "smlal2 v14.4s, v5.8h, v0.h[2]\n"
        "ldr q11, [%[in_ptr2]], #0x10\n"
        "smlal v15.4s, v6.4h, v0.h[2]\n"
        "smlal2 v16.4s, v6.8h, v0.h[2]\n"
        // r3
        "smlal v13.4s, v7.4h, v0.h[3]\n"
        "smlal2 v14.4s, v7.8h, v0.h[3]\n"
        "ldr q12, [%[in_ptr3]], #0x10\n"
        "smlal v15.4s, v8.4h, v0.h[3]\n"
        "smlal2 v16.4s, v8.8h, v0.h[3]\n"
        "subs %w[cnt], %w[cnt], #1\n"
        "str q13, [%[out_ptr]], #0x10\n"
        "str q14, [%[out_ptr]], #0x10\n"
        "str q15, [%[out_ptr]], #0x10\n"
        "str q16, [%[out_ptr]], #0x10\n"
        "bne 1b\n"
        "2: \n"
        : [out_ptr] "+r"(out_ptr),
          [in_ptr0] "+r"(in_ptr0),
          [in_ptr1] "+r"(in_ptr1),
          [in_ptr2] "+r"(in_ptr2),
          [in_ptr3] "+r"(in_ptr3),
          [cnt] "+r"(cnt_col)
        : [wc] "r"(wei_ptr)
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
          "v15",
          "v16");
    in_ptr0 -= 16;
    in_ptr1 -= 16;
    in_ptr2 -= 16;
    in_ptr3 -= 16;
    for (int j = 0; j < out_remain; j++) {
      *out_ptr += *in_ptr0++ * wei_ptr[0];
      *out_ptr += *in_ptr1++ * wei_ptr[1];
      *out_ptr += *in_ptr2++ * wei_ptr[2];
      *out_ptr += *in_ptr3++ * wei_ptr[3];
      out_ptr++;
    }
    data_in += 4 * M;
    weights_ptr += 4;
  }
  for (int i = 0; i < tail_4; i++) {
    const int8_t* in_ptr = data_in;
    const int8_t* wei_ptr = weights_ptr;
    int32_t* out_ptr = zero_ptr;
    int cnt_col = out_cnt;
    asm volatile(
        "prfm  pldl1keep, [%[wc]]        \n"
        "prfm  pldl1keep, [%[out_ptr]]   \n"
        "prfm  pldl1keep, [%[in_ptr0]]   \n"
        "cmp %w[cnt], #1\n"
        "ldr q8, [%[wc]]\n"
        "ldr q9, [%[in_ptr0]], #0x10\n"
        "ldr q13, [%[out_ptr]]\n"
        "ldr q14, [%[out_ptr], #0x10]\n"
        "sxtl  v0.8h, v8.8b\n"
        "blt 2f\n"
        "1: \n"
        "sxtl  v1.8h, v9.8b\n"
        "sxtl2 v2.8h, v9.16b\n"
        "ldr q15, [%[out_ptr], #0x20]\n"
        "ldr q16, [%[out_ptr], #0x30]\n"
        // r0
        "smlal v13.4s, v1.4h, v0.h[0]\n"
        "ldr q9, [%[in_ptr0]], #0x10\n"
        "smlal2 v14.4s, v1.8h, v0.h[0]\n"
        "smlal v15.4s, v2.4h, v0.h[0]\n"
        "smlal2 v16.4s, v2.8h, v0.h[0]\n"

        "subs %w[cnt], %w[cnt], #1\n"
        "str q13, [%[out_ptr]], #0x10\n"
        "str q14, [%[out_ptr]], #0x10\n"
        "str q15, [%[out_ptr]], #0x10\n"
        "str q16, [%[out_ptr]], #0x10\n"
        "ldr q13, [%[out_ptr]]\n"
        "ldr q14, [%[out_ptr], #0x10]\n"
        "bne 1b\n"
        "2: \n"
        : [out_ptr] "+r"(out_ptr), [in_ptr0] "+r"(in_ptr), [cnt] "+r"(cnt_col)
        : [wc] "r"(wei_ptr)
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
          "v15",
          "v16");
    in_ptr -= 16;
    for (int j = 0; j < out_remain; j++) {
      *out_ptr += *in_ptr++ * wei_ptr[0];
      out_ptr++;
    }
    data_in += M;
    weights_ptr++;
  }
#else
  int cnt = N >> 2;
  int tail = N & 3;
  int stride_in = M << 2;
  // #pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const int8_t* in_ptr0 = data_in;
    const int8_t* in_ptr1 = in_ptr0 + M;
    const int8_t* in_ptr2 = in_ptr1 + M;
    const int8_t* in_ptr3 = in_ptr2 + M;
    const int8_t* wei_ptr = weights_ptr;
    int32_t* out_ptr = zero_ptr;
    int cnt_col = out_cnt;
    asm volatile(
        "pld [%[wc]]        \n"
        "pld [%[in_ptr0]]   \n"
        "pld [%[in_ptr1]]   \n"
        "pld [%[in_ptr2]]   \n"
        "pld [%[in_ptr3]]   \n"
        "pld [%[out_ptr]]   \n"
        "cmp %[cnt], #1\n"
        "vld1.8 {d22}, [%[wc]]\n"
        "blt 2f\n"
        "1: \n"
        "vld1.8 {d10-d11}, [%[in_ptr0]]!\n"
        "vmovl.s8 q0, d22\n"
        "vld1.8 {d12-d13}, [%[in_ptr1]]!\n"
        "vld1.32 {d24-d25}, [%[out_ptr]]\n"
        "vld1.8 {d14-d15}, [%[in_ptr2]]!\n"
        "vldr d26, [%[out_ptr], #0x10]\n"
        "vld1.8 {d16-d17}, [%[in_ptr3]]!\n"
        "vmovl.s8 q1, d10\n"
        "vmovl.s8 q2, d11\n"
        "vldr d27, [%[out_ptr], #0x18]\n"
        "vmovl.s8 q3, d12\n"
        "vmovl.s8 q4, d13\n"
        "vldr d28, [%[out_ptr], #0x20]\n"
        "vldr d29, [%[out_ptr], #0x28]\n"
        "vmovl.s8 q9, d14\n"
        "vmovl.s8 q10, d15\n"
        "vldr d30, [%[out_ptr], #0x30]\n"
        "vldr d31, [%[out_ptr], #0x38]\n"
        // r0
        "vmlal.s16 q12, d2, d0[0]\n"
        "vmlal.s16 q13, d3, d0[0]\n"
        "vmovl.s8 q5, d16\n"
        "vmlal.s16 q14, d4, d0[0]\n"
        "vmlal.s16 q15, d5, d0[0]\n"
        "vmovl.s8 q6, d17\n"
        // r1
        "vmlal.s16 q12, d6, d0[1]\n"
        "vmlal.s16 q13, d7, d0[1]\n"
        "vmlal.s16 q14, d8, d0[1]\n"
        "vmlal.s16 q15, d9, d0[1]\n"
        // r2
        "vmlal.s16 q12, d18, d0[2]\n"
        "vmlal.s16 q13, d19, d0[2]\n"
        "vmlal.s16 q14, d20, d0[2]\n"
        "vmlal.s16 q15, d21, d0[2]\n"
        // r3
        "vmlal.s16 q12, d10, d0[3]\n"
        "vmlal.s16 q13, d11, d0[3]\n"
        "vmlal.s16 q14, d12, d0[3]\n"
        "vmlal.s16 q15, d13, d0[3]\n"

        "subs %[cnt], #1\n"
        "vst1.32 {d24-d25}, [%[out_ptr]]!\n"
        "vst1.32 {d26-d27}, [%[out_ptr]]!\n"
        "vst1.32 {d28-d29}, [%[out_ptr]]!\n"
        "vst1.32 {d30-d31}, [%[out_ptr]]!\n"
        "bne 1b\n"
        "2: \n"
        : [out_ptr] "+r"(out_ptr),
          [in_ptr0] "+r"(in_ptr0),
          [in_ptr1] "+r"(in_ptr1),
          [in_ptr2] "+r"(in_ptr2),
          [in_ptr3] "+r"(in_ptr3),
          [cnt] "+r"(cnt_col)
        : [wc] "r"(wei_ptr)
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
          "q12",
          "q13",
          "q14",
          "q15");
    for (int j = 0; j < out_remain; j++) {
      *out_ptr += *in_ptr0++ * wei_ptr[0];
      *out_ptr += *in_ptr1++ * wei_ptr[1];
      *out_ptr += *in_ptr2++ * wei_ptr[2];
      *out_ptr += *in_ptr3++ * wei_ptr[3];
      out_ptr++;
    }
    data_in += stride_in;
    weights_ptr += 4;
  }
  for (int i = 0; i < tail; i++) {
    const int8_t* in_ptr = data_in;
    const int8_t* wei_ptr = weights_ptr;
    int32_t* out_ptr = zero_ptr;
    int cnt_col = out_cnt;
    asm volatile(
        "pld [%[wc]]        \n"
        "pld [%[in_ptr0]]   \n"
        "pld [%[out_ptr]]   \n"
        "cmp %[cnt], #1\n"
        "vld1.8 {d8}, [%[wc]]\n"
        "vld1.8 {d10-d11}, [%[in_ptr0]]!\n"
        "vld1.32 {d24-d25}, [%[out_ptr]]\n"
        "vldr d26, [%[out_ptr], #0x10]\n"
        "vmovl.s8 q0, d8\n"
        "blt 2f\n"
        "1: \n"
        "vldr d27, [%[out_ptr], #0x18]\n"
        "vmovl.s8 q1, d10\n"
        "vldr d28, [%[out_ptr], #0x20]\n"
        "vmovl.s8 q2, d11\n"
        "vldr d29, [%[out_ptr], #0x28]\n"
        // r0
        "vmlal.s16 q12, d2, d0[0]\n"
        "vldr d30, [%[out_ptr], #0x30]\n"
        "vldr d31, [%[out_ptr], #0x38]\n"
        "vmlal.s16 q13, d3, d0[0]\n"
        "vld1.8 {d10-d11}, [%[in_ptr0]]!\n"
        "vmlal.s16 q14, d4, d0[0]\n"
        "vmlal.s16 q15, d5, d0[0]\n"

        "subs %[cnt], #1\n"
        "vst1.32 {d24-d25}, [%[out_ptr]]!\n"
        "vst1.32 {d26-d27}, [%[out_ptr]]!\n"
        "vst1.32 {d28-d29}, [%[out_ptr]]!\n"
        "vst1.32 {d30-d31}, [%[out_ptr]]!\n"
        "vld1.32 {d24-d25}, [%[out_ptr]]\n"
        "vldr d26, [%[out_ptr], #0x10]\n"
        "bne 1b\n"
        "2: \n"
        : [out_ptr] "+r"(out_ptr), [in_ptr0] "+r"(in_ptr), [cnt] "+r"(cnt_col)
        : [wc] "r"(wei_ptr)
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
          "q12",
          "q13",
          "q14",
          "q15");
    in_ptr -= 16;
    for (int j = 0; j < out_remain; j++) {
      *out_ptr += *in_ptr++ * wei_ptr[0];
      out_ptr++;
    }
    data_in += M;
    weights_ptr++;
  }
#endif
  // write output
  write_gemv_out(zero_ptr,
                 y,
                 scale,
                 bias_ptr,
                 M,
                 flag_act,
                 act,
                 six,
                 alpha,
                 offset,
                 threshold);
  delete[] zero_ptr;
  delete[] zerobuf;
  return true;
}

#if defined(__aarch64__)
// clang-format off
#define GEMV_COMPUTE_INIT                                                 \
    "prfm  pldl1keep, [%[in]]     \n"   /* preload din */                 \
    "prfm  pldl1keep, [%[w0]]     \n"   /* preload w0 */                  \
    "prfm  pldl1keep, [%[w1]]     \n"   /* preload w1 */                  \
    "prfm  pldl1keep, [%[w2]]     \n"   /* preload w2 */                  \
    "prfm  pldl1keep, [%[w3]]     \n"   /* preload w3 */                  \
    "prfm  pldl1keep, [%[w4]]     \n"   /* preload w4 */                  \
    "prfm  pldl1keep, [%[w5]]     \n"   /* preload w5 */                  \
    "prfm  pldl1keep, [%[w6]]     \n"   /* preload w6 */                  \
    "prfm  pldl1keep, [%[w7]]     \n"   /* preload w7 */                  \
    "movi   v0.4s,  #0            \n"   /* set out0 to 0 */               \
    "movi   v1.4s,  #0            \n"   /* set out1 to 0 */               \
    "movi   v2.4s,  #0            \n"   /* set out2 to 0 */               \
    "movi   v3.4s,  #0            \n"   /* set out3 to 0 */               \
    "movi   v4.4s,  #0            \n"   /* set out4 to 0 */               \
    "movi   v5.4s,  #0            \n"   /* set out5 to 0 */               \
    "movi   v6.4s,  #0            \n"   /* set out6 to 0 */               \
    "movi   v7.4s,  #0            \n"   /* set out7 to 0 */               \
    /* check main loop */                                                 \
    "cmp %w[cnt], #1              \n"   /* check whether has main loop */ \
    "blt  2f                      \n"   /* jump to tail */                \
    /* main loop */                                                       \
    "1:                           \n"   /* main loop */                   \
    "ldr    q8,     [%[in]], #16  \n"   /* load input, 16 int8 */         \
    "ldr    q9,     [%[w0]], #16  \n"   /* load w0, 16 int8 */            \
    "ldr    q10,    [%[w1]], #16  \n"   /* load w1, 16 int8 */            \
    "ldr    q11,    [%[w2]], #16  \n"   /* load w2, 16 int8 */            \
    "ldr    q12,    [%[w3]], #16  \n"   /* load w3, 16 int8 */            \
    "ldr    q13,    [%[w4]], #16  \n"   /* load w4, 16 int8 */            \
    "ldr    q14,    [%[w5]], #16  \n"   /* load w5, 16 int8 */            \
    "ldr    q15,    [%[w6]], #16  \n"   /* load w6, 16 int8 */            \
    "ldr    q16,    [%[w7]], #16  \n"   /* load w7, 16 int8 */

#define GEMV_COMPUTE                                                      \
    /* mul, lower 8 int8 * int8 = int16 */                                \
    "smull  v18.8h, v8.8b, v9.8b  \n"   /* mul in * w0, low, 8 int8 */    \
    "smull  v19.8h, v8.8b, v10.8b \n"   /* mul in * w1, low, 8 int8 */    \
    "smull  v20.8h, v8.8b, v11.8b \n"   /* mul in * w2, low, 8 int8 */    \
    "smull  v21.8h, v8.8b, v12.8b \n"   /* mul in * w3, low, 8 int8 */    \
    "smull  v22.8h, v8.8b, v13.8b \n"   /* mul in * w4, low, 8 int8 */    \
    "smull  v23.8h, v8.8b, v14.8b \n"   /* mul in * w5, low, 8 int8 */    \
    "smull  v24.8h, v8.8b, v15.8b \n"   /* mul in * w6, low, 8 int8 */    \
    "smull  v25.8h, v8.8b, v16.8b \n"   /* mul in * w7, low, 8 int8 */    \
    /* mul, higher 8 int8 * int8 + int16 = int16 */                       \
    "smlal2 v18.8h,v8.16b,v9.16b  \n"   /* mul in * w0, high, 8 int8 */   \
    "smlal2 v19.8h,v8.16b,v10.16b \n"   /* mul in * w1, high, 8 int8 */   \
    "smlal2 v20.8h,v8.16b,v11.16b \n"   /* mul in * w2, high, 8 int8 */   \
    "smlal2 v21.8h,v8.16b,v12.16b \n"   /* mul in * w2, high, 8 int8 */   \
    "smlal2 v22.8h,v8.16b,v13.16b \n"   /* mul in * w2, high, 8 int8 */   \
    "smlal2 v23.8h,v8.16b,v14.16b \n"   /* mul in * w2, high, 8 int8 */   \
    "smlal2 v24.8h,v8.16b,v15.16b \n"   /* mul in * w2, high, 8 int8 */   \
    "smlal2 v25.8h,v8.16b,v16.16b \n"   /* mul in * w2, high, 8 int8 */   \
    "subs %w[cnt], %w[cnt], #1    \n"   /* sub main loop count */         \
    /* add int16 to int32 */                                              \
    "sadalp v0.4s, v18.8h         \n"   /* pair acc, 8 int16 -> 4 int32 */\
    "sadalp v1.4s, v19.8h         \n"   /* pair acc, 8 int16 -> 4 int32 */\
    "sadalp v2.4s, v20.8h         \n"   /* pair acc, 8 int16 -> 4 int32 */\
    "sadalp v3.4s, v21.8h         \n"   /* pair acc, 8 int16 -> 4 int32 */\
    "sadalp v4.4s, v22.8h         \n"   /* pair acc, 8 int16 -> 4 int32 */\
    "sadalp v5.4s, v23.8h         \n"   /* pair acc, 8 int16 -> 4 int32 */\
    "sadalp v6.4s, v24.8h         \n"   /* pair acc, 8 int16 -> 4 int32 */\
    "sadalp v7.4s, v25.8h         \n"   /* pair acc, 8 int16 -> 4 int32 */\
    "bne 1b                       \n"   /* jump to main loop */

#define GEMV_COMPUTE_ACT                                                  \
    /* pair add to final result */                                        \
    "2:                           \n"   /* reduce to scale */             \
    "ldp  q17,    q18, [%[scale]] \n"   /* load scale */                  \
    "movi   v19.4s, #0            \n"                                     \
    "movi   v20.4s, #0            \n"                                     \
    "cmp    %w[bias],   #0        \n"                                     \
    "beq    9f                    \n"                                     \
    "ldp  q19,    q20, [%[bias]]  \n"   /* load bias */                   \
    "9:                           \n"                                     \
    "addp v9.4s,  v2.4s, v3.4s    \n"   /* pair add to 4 int32*/          \
    "addp v8.4s,  v0.4s, v1.4s    \n"   /* pair add to 4 int32*/          \
    "addp v10.4s, v4.4s, v5.4s    \n"   /* pair add to 4 int32*/          \
    "addp v11.4s, v6.4s, v7.4s    \n"   /* pair add to 4 int32*/          \
    "addp v12.4s, v8.4s , v9.4s   \n"   /* pair add to 4 int32*/          \
    "addp v13.4s, v10.4s, v11.4s  \n"   /* pair add to 4 int32*/          \
    "scvtf  v21.4s, v12.4s        \n"   /* convert to fp32 */             \
    "scvtf  v22.4s, v13.4s        \n"   /* convert to fp32 */             \
    "fmla v19.4s, v21.4s, v17.4s  \n"   /* mul scale to get result */     \
    "fmla v20.4s, v22.4s, v18.4s  \n"   /* mul scale to get  result */    \
    "cmp    %w[relu],   #0        \n"                                     \
    "movi   v0.4s, #0             \n"                                     \
    "beq    12f                   \n"                                     \
    "cmp    %w[relu],    #1       \n"                                     \
    "beq    15f                   \n"                                     \
    "cmp    %w[relu],    #2       \n"                                     \
    "beq    13f                   \n"                                     \
    "cmp    %w[relu],    #4       \n"                                     \
    "beq    14f                   \n"                                     \
    "ldr    q2,    [%[alpha], #16]\n"                                     \
    "ldr    q1,    [%[alpha]]     \n"                                     \
    "ldr    q3,    [%[alpha], #32]\n"                                     \
    "fadd   v21.4s, v19.4s, v2.4s \n"                                     \
    "fmul   v22.4s, v19.4s, v1.4s \n"                                     \
    "fmax   v21.4s, v21.4s, v0.4s \n"                                     \
    "fmin   v21.4s, v21.4s, v3.4s \n"                                     \
    "fmul   v19.4s, v21.4s, v22.4s\n"                                     \
    "fadd   v21.4s, v20.4s, v2.4s \n"                                     \
    "fmul   v22.4s, v20.4s, v1.4s \n"                                     \
    "fmax   v21.4s, v21.4s, v0.4s \n"                                     \
    "fmin   v21.4s, v21.4s, v3.4s \n"                                     \
    "fmul   v20.4s, v21.4s, v22.4s\n"                                     \
    "b      12f                   \n"                                     \
    "15:                          \n"                                     \
    "fmax   v19.4s, v19.4s, v0.4s \n"                                     \
    "fmax   v20.4s, v20.4s, v0.4s \n"                                     \
    "b      12f                   \n"                                     \
    "13:                          \n"                                     \
    "ldr    q1,    [%[alpha]]     \n"                                     \
    "fmax   v19.4s, v19.4s, v0.4s \n"                                     \
    "fmax   v20.4s, v20.4s, v0.4s \n"                                     \
    "fmin   v19.4s, v19.4s, v1.4s \n"                                     \
    "fmin   v20.4s, v20.4s, v1.4s \n"                                     \
    "b      12f                   \n"                                     \
    "14:                          \n"                                     \
    "ldr    q1,    [%[alpha]]     \n"                                     \
    "fcmge  v21.4s, v19.4s, v0.4s \n"                                     \
    "fmul   v22.4s, v19.4s, v1.4s \n"                                     \
    "bif    v19.16b,v22.16b,v21.16b\n"                                    \
    "fcmge  v21.4s, v20.4s, v0.4s \n"                                     \
    "fmul   v22.4s, v20.4s, v1.4s \n"                                     \
    "bif    v20.16b,v22.16b,v21.16b\n"

#define GEMV_ST_INT8                                                      \
    "12:                          \n"                                     \
    "dup    v8.4s,  %w[vmax]      \n"                                     \
    "fcmge  v0.4s,  v19.4s, v8.4s \n"                                     \
    "fcmge  v1.4s,  v20.4s, v8.4s \n"                                     \
    "bif  v19.16b,  v8.16b, v0.16b\n"                                     \
    "bif  v20.16b,  v8.16b, v1.16b\n"                                     \
    "fcvtas v0.4s,  v19.4s        \n"                                     \
    "fcvtas v1.4s,  v20.4s        \n"                                     \
    "sqxtn  v19.4h, v0.4s         \n"                                     \
    "sqxtn2 v19.8h, v1.4s         \n"                                     \
    "sqxtn  v0.8b,  v19.8h        \n"                                     \
    "st1  {v0.8b},  [%[out]]      \n"

#define GEMV_ST_FP32                                                      \
    "12:                          \n"                                     \
    "stp  q19,  q20,  [%[out]]    \n"

#define GEMV_ASM_PARAMS                                                   \
    [in] "+r"(ptr_in),                                                    \
    [w0] "+r"(ptr_w0),                                                    \
    [w1] "+r"(ptr_w1),                                                    \
    [w2] "+r"(ptr_w2),                                                    \
    [w3] "+r"(ptr_w3),                                                    \
    [w4] "+r"(ptr_w4),                                                    \
    [w5] "+r"(ptr_w5),                                                    \
    [w6] "+r"(ptr_w6),                                                    \
    [w7] "+r"(ptr_w7),                                                    \
    [cnt] "+r"(cnt),                                                      \
    [scale] "+r"(scale_ptr),                                              \
    [bias] "+r"(bias_ptr),                                                \
    [relu] "+r"(act)                                                      \
  : [out] "r"(out_ptr), [alpha] "r"(tmp_ptr)                              \
  : "cc", "memory",                                                       \
    "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",           \
    "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",               \
    "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25"

#define GEMV_ASM_FUN_PARAMS(dtype)                                        \
  const int8_t *ptr_in, const int8_t *ptr_w0,                             \
  const int8_t *ptr_w1, const int8_t *ptr_w2,                             \
  const int8_t *ptr_w3, const int8_t *ptr_w4,                             \
  const int8_t *ptr_w5, const int8_t *ptr_w6,                             \
  const int8_t *ptr_w7, int cnt,                                          \
  const float *scale_ptr, const float *bias_ptr,                          \
  int act, float alpha,  float offset, float threshold,                   \
  dtype *out_ptr

#define GEMV_DOT_COMPUTE                                                  \
  ".word 0x4e899500 //sdot v0.4s, v8.16b, v9.16b  \n" /* out0~3*/         \
  ".word 0x4e8a9501 //sdot v1.4s, v8.16b, v10.16b \n" /* out4~7*/         \
  ".word 0x4e8b9502 //sdot v2.4s, v8.16b, v11.16b \n" /* out0~3*/         \
  ".word 0x4e8c9503 //sdot v3.4s, v8.16b, v12.16b \n" /* out4~7*/         \
  "subs  %w[cnt], %w[cnt], #1                     \n"                     \
  ".word 0x4e8d9504 //sdot v4.4s, v8.16b, v13.16b \n" /* out0~3*/         \
  ".word 0x4e8e9505 //sdot v5.4s, v8.16b, v14.16b \n" /* out4~7*/         \
  ".word 0x4e8f9506 //sdot v6.4s, v8.16b, v15.16b \n" /* out0~3*/         \
  ".word 0x4e909507 //sdot v7.4s, v8.16b, v16.16b \n" /* out4~7*/         \
  "bne 1b                                         \n"

// clang-format on
template <typename dtype>
inline void gemv_int8_asm(GEMV_ASM_FUN_PARAMS(dtype));

template <>
inline void gemv_int8_asm(GEMV_ASM_FUN_PARAMS(float)) {
  float tmp_ptr[12] = {alpha,
                       alpha,
                       alpha,
                       alpha,
                       offset,
                       offset,
                       offset,
                       offset,
                       threshold,
                       threshold,
                       threshold,
                       threshold};
  asm volatile(GEMV_COMPUTE_INIT GEMV_COMPUTE GEMV_COMPUTE_ACT GEMV_ST_FP32
               : GEMV_ASM_PARAMS);
}
template <>
inline void gemv_int8_asm(GEMV_ASM_FUN_PARAMS(int8_t)) {
  float vmax = -127.f;
  float tmp_ptr[12] = {alpha,
                       alpha,
                       alpha,
                       alpha,
                       offset,
                       offset,
                       offset,
                       offset,
                       threshold,
                       threshold,
                       threshold,
                       threshold};
  asm volatile(GEMV_COMPUTE_INIT GEMV_COMPUTE GEMV_COMPUTE_ACT GEMV_ST_INT8
               : [vmax] "+r"(vmax), GEMV_ASM_PARAMS);
}
template <typename dtype>
inline void gemv_int8_dot_asm(GEMV_ASM_FUN_PARAMS(dtype));

template <>
inline void gemv_int8_dot_asm(GEMV_ASM_FUN_PARAMS(int8_t)) {
  float vmax = -127.f;
  float tmp_ptr[12] = {alpha,
                       alpha,
                       alpha,
                       alpha,
                       offset,
                       offset,
                       offset,
                       offset,
                       threshold,
                       threshold,
                       threshold,
                       threshold};
  asm volatile(GEMV_COMPUTE_INIT GEMV_DOT_COMPUTE GEMV_COMPUTE_ACT GEMV_ST_INT8
               : [vmax] "+r"(vmax), GEMV_ASM_PARAMS);
}

template <>
inline void gemv_int8_dot_asm(GEMV_ASM_FUN_PARAMS(float)) {
  float tmp_ptr[12] = {alpha,
                       alpha,
                       alpha,
                       alpha,
                       offset,
                       offset,
                       offset,
                       offset,
                       threshold,
                       threshold,
                       threshold,
                       threshold};
  asm volatile(GEMV_COMPUTE_INIT GEMV_DOT_COMPUTE GEMV_COMPUTE_ACT GEMV_ST_FP32
               : GEMV_ASM_PARAMS);
}

#undef GEMV_COMPUTE_INIT
#undef GEMV_DOT_COMPUTE
#undef GEMV_COMPUTE
#undef GEMV_COMPUTE_ACT
#undef GEMV_ST_FP32
#undef GEMV_ST_INT8
#undef GEMV_ASM_PARAMS
#undef GEMV_ASM_FUN_PARAMS
#else
// clang-format off
#define GEMV_COMPUTE                    \
  "pld [%[in]]                    \n"   \
  "pld [%[w0]]                    \n"   \
  "pld [%[w1]]                    \n"   \
  "pld [%[w2]]                    \n"   \
  "pld [%[w3]]                    \n"   \
  "vmov.u32 q0, #0                \n"   \
  "vmov.u32 q1, #0                \n"   \
  "vmov.u32 q2, #0                \n"   \
  "vmov.u32 q3, #0                \n"   \
  "cmp %[cnt], #1                 \n"   \
  "blt  2f                        \n"   \
  "1:                             \n"   \
  "vld1.8 {d8-d9},    [%[in]]!    \n"   \
  "vld1.8 {d12-d13},  [%[w0]]!    \n"   \
  "vld1.8 {d14-d15},  [%[w1]]!    \n"   \
  "vld1.8 {d16-d17},  [%[w2]]!    \n"   \
  "vld1.8 {d18-d19},  [%[w3]]!    \n"   \
  "vmull.s8 q12, d8, d12          \n"   \
  "vmull.s8 q13, d8, d14          \n"   \
  "vmull.s8 q14, d8, d16          \n"   \
  "vmull.s8 q15, d8, d18          \n"   \
  "vmlal.s8 q12,  d9, d13         \n"   \
  "vmlal.s8 q13,  d9, d15         \n"   \
  "vmlal.s8 q14,  d9, d17         \n"   \
  "vmlal.s8 q15,  d9, d19         \n"   \
  "vpadal.s16   q0,   q12         \n"   \
  "vpadal.s16   q1,   q13         \n"   \
  "vpadal.s16   q2,   q14         \n"   \
  "vpadal.s16   q3,   q15         \n"   \
  "subs %[cnt], #1                \n"   \
  "bne 1b                         \n"   \
  "2:                             \n"   \
  "vld1.8 {d20-d21}, [%[scale]]!  \n"   \
  "vmov.f32   q11, #0.0           \n"   \
  "cmp    %[bias],   #0           \n"   \
  "beq    9f                      \n"   \
  "vld1.8 {d22-d23}, [%[bias]]!   \n"   \
  "9:                             \n"   \
  "vpadd.s32 d8,  d0, d1          \n"   \
  "vpadd.s32 d9,  d2, d3          \n"   \
  "vpadd.s32 d10, d4, d5          \n"   \
  "vpadd.s32 d11, d6, d7          \n"   \
  "vpadd.s32 d0,  d8, d9          \n"   \
  "vpadd.s32 d1,  d10,d11         \n"   \
  "vcvt.f32.s32   q1, q0          \n"   \
  "vmla.f32  q11, q1, q10         \n"   \
  "cmp    %[relu],  #0            \n"   \
  "beq    12f                     \n"   \
  "cmp    %[relu],  #1            \n"   \
  "vmov.f32    q0,  #0.0          \n"   \
  "beq    15f                     \n"   \
  "cmp    %[relu],   #2           \n"   \
  "beq    13f                     \n"   \
  "cmp    %[relu],   #4           \n"   \
  "beq    14f                     \n"   \
  "vld1.32    {d2-d5}, [%[alpha]] \n"   \
  "vldr       d6,   [%[alpha], #32]\n"  \
  "vldr       d7,   [%[alpha], #40]\n"  \
  "vadd.f32   q4,   q11,  q2      \n"   \
  "vmul.f32   q5,   q11,  q1      \n"   \
  "vmax.f32   q4,   q4,   q0      \n"   \
  "vmin.f32   q4,   q4,   q3      \n"   \
  "vmul.f32   q11,  q4,   q5      \n"   \
  "b      12f                     \n"   \
  "15:                            \n"   \
  "vmax.f32   q11,  q11,  q0      \n"   \
  "b      12f                     \n"   \
  "13:                            \n"   \
  "vld1.32    {d2-d3}, [%[alpha]] \n"   \
  "vmax.f32   q11,  q11,  q0      \n"   \
  "vmin.f32   q11,  q11,  q1      \n"   \
  "b      12f                     \n"   \
  "14:                            \n"   \
  "vmov.f32   q0,   #0.0          \n"   \
  "vld1.32    {d2-d3}, [%[alpha]] \n"   \
  "vcge.f32   q2,   q11,  q0      \n"   \
  "vmul.f32   q3,   q11,  q1      \n"   \
  "vbif       q11,  q3,   q2      \n"

#define GEMV_ST_INT8                  \
  "12:                            \n" \
  "vdup.32    q0,   %[vmax]       \n" \
  "vmov.f32   q1,   #0.5          \n" \
  "vmov.f32   q2,   #-0.5         \n" \
  "vcgt.f32   q3,   q11,  #0      \n" \
  "vbif.f32   q1,   q2,   q3      \n" \
  "vadd.f32   q11,  q1,   q11     \n" \
  "vcge.f32   q12,  q11,  q0      \n" \
  "vbif q11,  q0,   q12           \n" \
  "vcvt.s32.f32     q1,   q11     \n" \
  "vqmovn.s32 d8,   q1            \n" \
  "vqmovn.s16 d22,  q4            \n" \
  "vmov.32    %[vmax], d22[0]     \n" \
  "str      %[vmax], [%[out]]     \n"

#define GEMV_ST_FP32                 \
  "12:                           \n" \
  "vst1.32 {d22-d23}, [%[out]]   \n"

#define GEMV_ASM_PARAMS                                                 \
  [in] "+r"(ptr_in), [w0] "+r"(ptr_w0),                                 \
  [w1] "+r"(ptr_w1), [w2] "+r"(ptr_w2),                                 \
  [w3] "+r"(ptr_w3), [cnt] "+r"(cnt),                                   \
  [scale] "+r"(scale_ptr), [bias] "+r"(bias_ptr)                        \
  : [out] "r"(out_ptr), [relu] "r"(act), [alpha] "r"(tmp_ptr)           \
  : "cc", "memory",                                                     \
    "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",                     \
    "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"

#define GEMV_ASM_FUN_PARAMS(dtype)                                      \
  const int8_t *ptr_in, const int8_t *ptr_w0,                           \
  const int8_t *ptr_w1, const int8_t *ptr_w2,                           \
  const int8_t *ptr_w3, int cnt,                                        \
  const float *scale_ptr, const float *bias_ptr,                        \
  int act, float alpha,  float offset, float threshold,                 \
  dtype *out_ptr
// clang-format on

template <typename dtype>
inline void gemv_int8_asm(GEMV_ASM_FUN_PARAMS(dtype));

template <>
inline void gemv_int8_asm(GEMV_ASM_FUN_PARAMS(float)) {
  float tmp_ptr[12] = {alpha,
                       alpha,
                       alpha,
                       alpha,
                       offset,
                       offset,
                       offset,
                       offset,
                       threshold,
                       threshold,
                       threshold,
                       threshold};
  asm volatile(GEMV_COMPUTE GEMV_ST_FP32 : GEMV_ASM_PARAMS);
}

template <>
inline void gemv_int8_asm(GEMV_ASM_FUN_PARAMS(int8_t)) {
  float vmax = -127.f;
  float tmp_ptr[12] = {alpha,
                       alpha,
                       alpha,
                       alpha,
                       offset,
                       offset,
                       offset,
                       offset,
                       threshold,
                       threshold,
                       threshold,
                       threshold};
  asm volatile(GEMV_COMPUTE GEMV_ST_INT8 : [vmax] "+r"(vmax), GEMV_ASM_PARAMS);
}
#undef GEMV_COMPUTE_INIT
#undef GEMV_COMPUTE
#undef GEMV_ST_FP32
#undef GEMV_ST_INT8
#undef GEMV_ASM_PARAMS
#undef GEMV_ASM_FUN_PARAMS
#endif

template <typename dtype>
void gemv_int8_oth(const int8_t* A,
                   const int8_t* x,
                   dtype* data_out,
                   int M,
                   int N,
                   const float* scale,
                   bool is_bias,
                   const float* bias,
                   bool fact,
                   lite_api::ActivationType act,
                   float alpha,
                   float offset,
                   float threshold,
                   ARMContext* ctx) {
  int cnt = N >> 4;
  int tail = N & 15;
  int Nup = (N + 15) / 16 * 16;
  cnt = Nup >> 4;
  int8_t* ptr_zero = ctx->workspace_data<int8_t>();
  memset(ptr_zero, 0, Nup * 3);
  int8_t* data_in = ptr_zero + Nup;
  lite::TargetWrapperHost::MemcpySync(data_in, x, N);
  int8_t* ptr_w = data_in + Nup;
  lite::TargetWrapperHost::MemcpySync(ptr_w, A + (M - 1) * N, N);

#ifdef __aarch64__
  int out_cnt = M >> 3;
  int remain = M & 7;
  if (remain > 0) out_cnt++;
  LITE_PARALLEL_BEGIN(j, tid, out_cnt) {
    int out_idx = j * 8;
    dtype* out_ptr = data_out + out_idx;
    dtype* out_p = out_ptr;
    const float* scale_ptr = scale + out_idx;
    dtype out_temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t* ptr_in = data_in;
    const int8_t* ptr_w0 = A + (N * out_idx);
    const int8_t* ptr_w1 = ptr_w0 + N;
    const int8_t* ptr_w2 = ptr_w1 + N;
    const int8_t* ptr_w3 = ptr_w2 + N;
    const int8_t* ptr_w4 = ptr_w3 + N;
    const int8_t* ptr_w5 = ptr_w4 + N;
    const int8_t* ptr_w6 = ptr_w5 + N;
    const int8_t* ptr_w7 = ptr_w6 + N;
    float scale_v[8] = {0.f};
    float bias_v[8] = {0.f};
    auto bias_ptr = is_bias ? bias + out_idx : bias_v;
    if (j == out_cnt - 1 && remain) {
      for (int p = 0; p < remain; p++) {
        scale_v[p] = scale_ptr[p];
        bias_v[p] = bias_ptr[p];
      }
      switch (8 - remain) {
        case 7:
          ptr_w1 = ptr_zero;
        case 6:
          ptr_w2 = ptr_zero;
        case 5:
          ptr_w3 = ptr_zero;
        case 4:
          ptr_w4 = ptr_zero;
        case 3:
          ptr_w5 = ptr_zero;
        case 2:
          ptr_w6 = ptr_zero;
        case 1:
          ptr_w7 = ptr_zero;
          out_p = out_temp;
          break;
        default:
          break;
      }
      switch (8 - remain) {
        case 7:
          ptr_w0 = ptr_w;
          break;
        case 6:
          ptr_w1 = ptr_w;
          break;
        case 5:
          ptr_w2 = ptr_w;
          break;
        case 4:
          ptr_w3 = ptr_w;
          break;
        case 3:
          ptr_w4 = ptr_w;
          break;
        case 2:
          ptr_w5 = ptr_w;
          break;
        case 1:
          ptr_w6 = ptr_w;
          break;
        default:
          break;
      }
    } else {
      for (int p = 0; p < 8; p++) {
        scale_v[p] = scale_ptr[p];
        bias_v[p] = bias_ptr[p];
      }
    }
    gemv_int8_asm<dtype>(ptr_in,
                         ptr_w0,
                         ptr_w1,
                         ptr_w2,
                         ptr_w3,
                         ptr_w4,
                         ptr_w5,
                         ptr_w6,
                         ptr_w7,
                         cnt,
                         scale_v,
                         bias_v,
                         static_cast<int>(act),
                         alpha,
                         offset,
                         threshold,
                         out_p);
    if (remain > 0) {
      for (int i = 0; i < remain; i++) {
        out_ptr[i] = out_p[i];
      }
    }
  }
  LITE_PARALLEL_END();
#else  //  __aarch64__

  int out_cnt = M >> 2;
  int remain = M & 3;
  if (remain > 0) out_cnt++;
  LITE_PARALLEL_BEGIN(j, tid, out_cnt) {
    int out_idx = j * 4;
    dtype* out_ptr = data_out + out_idx;
    dtype* out_p = out_ptr;
    const float* scale_ptr = scale + out_idx;
    dtype out_temp[4] = {0, 0, 0, 0};
    const int8_t* ptr_in = data_in;
    const int8_t* ptr_w0 = A + (N * out_idx);
    const int8_t* ptr_w1 = ptr_w0 + N;
    const int8_t* ptr_w2 = ptr_w1 + N;
    const int8_t* ptr_w3 = ptr_w2 + N;
    float scale_v[4] = {0.f};
    float bias_v[4] = {0.f};
    auto bias_ptr = is_bias ? bias + out_idx : bias_v;
    if (j == out_cnt - 1 && remain) {
      for (int p = 0; p < remain; p++) {
        scale_v[p] = scale_ptr[p];
        bias_v[p] = bias_ptr[p];
      }
      switch (4 - remain) {
        case 3:
          ptr_w1 = ptr_zero;
        case 2:
          ptr_w2 = ptr_zero;
        case 1:
          ptr_w3 = ptr_zero;
          out_p = out_temp;
          break;
        default:
          break;
      }
      switch (4 - remain) {
        case 3:
          ptr_w0 = ptr_w;
          break;
        case 2:
          ptr_w1 = ptr_w;
          break;
        case 1:
          ptr_w2 = ptr_w;
          break;
        default:
          break;
      }
    } else {
      for (int p = 0; p < 4; p++) {
        scale_v[p] = scale_ptr[p];
        bias_v[p] = bias_ptr[p];
      }
    }
    gemv_int8_asm<dtype>(ptr_in,
                         ptr_w0,
                         ptr_w1,
                         ptr_w2,
                         ptr_w3,
                         cnt,
                         scale_v,
                         bias_v,
                         static_cast<int>(act),
                         alpha,
                         offset,
                         threshold,
                         out_p);
    if (remain > 0) {
      for (int i = 0; i < remain; i++) {
        out_ptr[i] = out_p[i];
      }
    }
  }
  LITE_PARALLEL_END();

#endif  //  __aarch64__
}

#if defined(__aarch64__) && defined(WITH_ARM_DOTPROD)
template <typename dtype>
void gemv_int8_sdot(const int8_t* A,
                    const int8_t* x,
                    dtype* data_out,
                    int M,
                    int N,
                    const float* scale,
                    bool is_bias,
                    const float* bias,
                    bool fact,
                    lite_api::ActivationType act,
                    float alpha,
                    float offset,
                    float threshold,
                    ARMContext* ctx) {
  int Nup = (N + 15) / 16 * 16;
  int cnt = Nup >> 4;
  int tail = N & 15;
  int out_cnt = M >> 3;
  int remain = M & 7;
  if (remain > 0) out_cnt++;
  int8_t* ptr_zero = ctx->workspace_data<int8_t>();
  memset(ptr_zero, 0, Nup * 3);
  int8_t* data_in = ptr_zero + Nup;
  lite::TargetWrapperHost::MemcpySync(data_in, x, N);
  int8_t* ptr_w = data_in + Nup;
  lite::TargetWrapperHost::MemcpySync(ptr_w, A + (M - 1) * N, N);

  LITE_PARALLEL_BEGIN(j, tid, out_cnt) {
    int out_idx = j * 8;
    dtype* out_ptr = data_out + out_idx;
    dtype* out_p = out_ptr;
    const float* scale_ptr = scale + out_idx;
    dtype out_temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float scale_v[8] = {0.f};
    float bias_v[8] = {0.f};
    auto bias_ptr = is_bias ? bias + out_idx : bias_v;
    const int8_t* ptr_in = data_in;
    const int8_t* ptr_w0 = A + (N * out_idx);
    const int8_t* ptr_w1 = ptr_w0 + N;
    const int8_t* ptr_w2 = ptr_w1 + N;
    const int8_t* ptr_w3 = ptr_w2 + N;
    const int8_t* ptr_w4 = ptr_w3 + N;
    const int8_t* ptr_w5 = ptr_w4 + N;
    const int8_t* ptr_w6 = ptr_w5 + N;
    const int8_t* ptr_w7 = ptr_w6 + N;
    if (j == out_cnt - 1 && remain) {
      for (int p = 0; p < remain; p++) {
        scale_v[p] = scale_ptr[p];
        bias_v[p] = bias_ptr[p];
      }
      switch (8 - remain) {
        case 7:
          ptr_w1 = ptr_zero;
        case 6:
          ptr_w2 = ptr_zero;
        case 5:
          ptr_w3 = ptr_zero;
        case 4:
          ptr_w4 = ptr_zero;
        case 3:
          ptr_w5 = ptr_zero;
        case 2:
          ptr_w6 = ptr_zero;
        case 1:
          ptr_w7 = ptr_zero;
          out_p = out_temp;
          break;
        default:
          break;
      }
      switch (8 - remain) {
        case 7:
          ptr_w0 = ptr_w;
          break;
        case 6:
          ptr_w1 = ptr_w;
          break;
        case 5:
          ptr_w2 = ptr_w;
          break;
        case 4:
          ptr_w3 = ptr_w;
          break;
        case 3:
          ptr_w4 = ptr_w;
          break;
        case 2:
          ptr_w5 = ptr_w;
          break;
        case 1:
          ptr_w6 = ptr_w;
          break;
        default:
          break;
      }
    } else {
      for (int p = 0; p < 8; p++) {
        scale_v[p] = scale_ptr[p];
        bias_v[p] = bias_ptr[p];
      }
    }

    if (cnt > 0) {
      gemv_int8_dot_asm<dtype>(ptr_in,
                               ptr_w0,
                               ptr_w1,
                               ptr_w2,
                               ptr_w3,
                               ptr_w4,
                               ptr_w5,
                               ptr_w6,
                               ptr_w7,
                               cnt,
                               scale_v,
                               bias_v,
                               static_cast<int>(act),
                               alpha,
                               offset,
                               threshold,
                               out_p);
      if (remain > 0) {
        for (int i = 0; i < remain; i++) {
          out_ptr[i] = out_p[i];
        }
      }
    }
  }
  LITE_PARALLEL_END();
}
#endif  // __aarch64__ && sdot

template <typename dtype>
void gemv_int8(const int8_t* A,
               const int8_t* x,
               dtype* y,
               bool transA,
               int M,
               int N,
               const float* scale,
               bool is_bias,
               const float* bias,
               const operators::ActivationParam act_param,
               ARMContext* ctx) {
#define IN_PARAMS                                            \
  A, x, y, M, N, scale, is_bias, bias, act_param.has_active, \
      act_param.active_type, alpha, offset, threshold, ctx

  float alpha = 1.f;
  float offset = 3.f;
  float threshold = 6.f;
  if (act_param.has_active) {
    if (act_param.active_type == lite_api::ActivationType::kRelu6) {
      alpha = act_param.Relu_clipped_coef;
    } else if (act_param.active_type == lite_api::ActivationType::kLeakyRelu) {
      alpha = act_param.Leaky_relu_alpha;
    } else if (act_param.active_type == lite_api::ActivationType::kHardSwish) {
      alpha = 1.0 / act_param.hard_swish_scale;
      offset = act_param.hard_swish_offset;
      threshold = act_param.hard_swish_threshold;
    }
  }
  if (transA) {
    gemv_int8_trans_oth(IN_PARAMS);
    return;
  }

#ifdef __aarch64__
  if (ctx->has_dot()) {
#ifdef WITH_ARM_DOTPROD
    gemv_int8_sdot<dtype>(IN_PARAMS);
#endif
  } else {
    gemv_int8_oth<dtype>(IN_PARAMS);
  }
#else
  gemv_int8_oth<dtype>(IN_PARAMS);
#endif
#undef IN_PARAMS
}

#define GEMV_INT8_FUN(dtype)                                                 \
  template void gemv_int8<dtype>(const int8_t* A,                            \
                                 const int8_t* x,                            \
                                 dtype* y,                                   \
                                 bool transA,                                \
                                 int M,                                      \
                                 int N,                                      \
                                 const float* scale,                         \
                                 bool is_bias,                               \
                                 const float* bias,                          \
                                 const operators::ActivationParam act_param, \
                                 ARMContext* ctx);

GEMV_INT8_FUN(int8_t);
GEMV_INT8_FUN(float);
#undef GEMV_INT8_FUN
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
