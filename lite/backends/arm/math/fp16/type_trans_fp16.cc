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
#include "lite/backends/arm/math/fp16/type_trans_fp16.h"

#include <arm_neon.h>

#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
void fp16_to_fp32(const float16_t* in, float* out, int size) {
#ifdef __aarch64__
  int cnt = size >> 6;
  int remain = size & 63;
#else
  int cnt = size >> 5;
  int remain = size & 31;
#endif
  int remain_cnt = remain >> 3;
  int remain_remain = remain & 7;
#ifdef __aarch64__
  asm volatile(
      "cmp %w[cnt], #1\n"
      "blt 1f\n"
      "0: \n"
      "ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [%[in]], #64\n"
      "ld1 {v4.8h, v5.8h, v6.8h, v7.8h}, [%[in]], #64\n"
      // 16bit->32bit
      "fcvtl v8.4s, v0.4h\n"
      "fcvtl2 v9.4s, v0.8h\n"
      "fcvtl v10.4s, v1.4h\n"
      "fcvtl2 v11.4s, v1.8h\n"
      "fcvtl v12.4s, v2.4h\n"
      "fcvtl2 v13.4s, v2.8h\n"
      "fcvtl v14.4s, v3.4h\n"
      "fcvtl2 v15.4s, v3.8h\n"
      "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[out]], #64\n"
      "fcvtl v16.4s, v4.4h\n"
      "fcvtl2 v17.4s, v4.8h\n"
      "fcvtl v18.4s, v5.4h\n"
      "fcvtl2 v19.4s, v5.8h\n"
      "st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[out]], #64\n"
      "subs %w[cnt], %w[cnt], #1\n"
      "fcvtl v20.4s, v6.4h\n"
      "fcvtl2 v21.4s, v6.8h\n"
      "st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[out]], #64\n"
      "fcvtl v22.4s, v7.4h\n"
      "fcvtl2 v23.4s, v7.8h\n"
      "st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [%[out]], #64\n"
      "bne 0b\n"
      "1: \n"
      "cmp %w[remain_cnt], #1\n"
      "blt 2f\n"
      "4: \n"
      "ld1 {v0.8h}, [%[in]], #16\n"
      "subs %w[remain_cnt], %w[remain_cnt], #1\n"
      // 16bit->32bit
      "fcvtl v8.4s, v0.4h\n"
      "fcvtl2 v9.4s, v0.8h\n"
      "st1 {v8.4s, v9.4s}, [%[out]], #32\n"
      "bne 4b\n"
      "2: \n"
      "cmp %w[remain_remain], #1\n"
      "blt 3f\n"
      "5: \n"
      "ldr h0, [%[in]], #2\n"
      "subs %w[remain_remain], %w[remain_remain], #1\n"
      "fcvt s0, h0\n"
      "str s0, [%[out]], #4\n"
      "bne 5b\n"
      "3: \n"
      : [in] "+r"(in),
        [out] "+r"(out),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_cnt),
        [remain_remain] "+r"(remain_remain)
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
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23");
#else
  asm volatile(
      "cmp %[cnt], #1\n"
      "blt 1f\n"
      "0: \n"
      "vld1.32 {d0-d3}, [%[in]]!\n"
      "vld1.32 {d4-d7}, [%[in]]!\n"
      // 16->32
      "vcvt.f32.f16 q4, d0\n"
      "vcvt.f32.f16 q5, d1\n"
      "vcvt.f32.f16 q6, d2\n"
      "vcvt.f32.f16 q7, d3\n"
      "vcvt.f32.f16 q8, d4\n"
      "vst1.32 {d8-d11}, [%[out]]!\n"
      "vcvt.f32.f16 q9, d5\n"
      "subs %[cnt], #1\n"
      "vst1.32 {d12-d15}, [%[out]]!\n"
      "vcvt.f32.f16 q10, d6\n"
      "vst1.32 {d16-d19}, [%[out]]!\n"
      "vcvt.f32.f16 q11, d7\n"
      "vst1.32 {d20-d23}, [%[out]]!\n"
      "bne 0b\n"
      "1: \n"
      "cmp %[remain_cnt], #1\n"
      "blt 2f\n"
      "4: \n"
      "vld1.32 {d0-d1}, [%[in]]!\n"
      "subs %[remain_cnt], #1\n"
      // 16bit->32bit
      "vcvt.f32.f16 q2, d0\n"
      "vcvt.f32.f16 q3, d1\n"
      "vst1.32 {d4-d7}, [%[out]]!\n"
      "bne 4b\n"
      "2: \n"
      "cmp %[remain_remain], #1\n"
      "blt 3f\n"
      "5: \n"
      "vld1.16 d0[0], [%[in]]!\n"
      "subs %[remain_remain], #1\n"
      "vcvt.f32.f16 q2, d0\n"
      "vst1.32 d4[0], [%[out]]!\n"
      "bne 5b\n"
      "3: \n"
      : [in] "+r"(in),
        [out] "+r"(out),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_cnt),
        [remain_remain] "+r"(remain_remain)
      :
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
        "q11");
#endif
}

void fp32_to_fp16(const float* in, float16_t* out, int size) {
#ifdef __aarch64__
  int cnt = size >> 6;
  int remain = size & 63;
#else
  int cnt = size >> 5;
  int remain = size & 31;
#endif
  int remain_cnt = remain >> 3;
  int remain_remain = remain & 7;
#ifdef __aarch64__
  asm volatile(
      "cmp %w[cnt], #1\n"
      "blt 1f\n"
      "0: \n"
      "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[in]], #64\n"
      "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[in]], #64\n"
      "ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[in]], #64\n"
      "ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[in]], #64\n"
      // 32bit->16bit
      "fcvtn v16.4h, v0.4s\n"
      "fcvtn2 v16.8h, v1.4s\n"
      "fcvtn v17.4h, v2.4s\n"
      "fcvtn2 v17.8h, v3.4s\n"
      "fcvtn v18.4h, v4.4s\n"
      "fcvtn2 v18.8h, v5.4s\n"
      "fcvtn v19.4h, v6.4s\n"
      "fcvtn2 v19.8h, v7.4s\n"
      "fcvtn v20.4h, v8.4s\n"
      "fcvtn2 v20.8h, v9.4s\n"
      "fcvtn v21.4h, v10.4s\n"
      "fcvtn2 v21.8h, v11.4s\n"
      "subs %w[cnt], %w[cnt], #1\n"
      "st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [%[out]], #64\n"
      "fcvtn v22.4h, v12.4s\n"
      "fcvtn2 v22.8h, v13.4s\n"
      "fcvtn v23.4h, v14.4s\n"
      "fcvtn2 v23.8h, v15.4s\n"
      "st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [%[out]], #64\n"
      "bne 0b\n"
      "1: \n"
      "cmp %w[remain_cnt], #1\n"
      "blt 2f\n"
      "4: \n"
      "ld1 {v0.4s, v1.4s}, [%[in]], #32\n"
      "subs %w[remain_cnt], %w[remain_cnt], #1\n"
      // 32bit->16bit
      "fcvtn v16.4h, v0.4s\n"
      "fcvtn2 v16.8h, v1.4s\n"
      "st1 {v16.8h}, [%[out]], #16\n"
      "bne 4b\n"
      "2: \n"
      "cmp %w[remain_remain], #1\n"
      "blt 3f\n"
      "5: \n"
      "ldr s0, [%[in]], #4\n"
      "subs %w[remain_remain], %w[remain_remain], #1\n"
      "fcvt h0, s0\n"
      "str h0, [%[out]], #2\n"
      "bne 5b\n"
      "3: \n"
      : [in] "+r"(in),
        [out] "+r"(out),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_cnt),
        [remain_remain] "+r"(remain_remain)
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
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
        "v21",
        "v22",
        "v23");
#else
  asm volatile(
      "cmp %[cnt], #1\n"
      "blt 1f\n"
      "0: \n"
      "vld1.32 {d0-d3}, [%[in]]!\n"
      "vld1.32 {d4-d7}, [%[in]]!\n"
      "vld1.32 {d8-d11}, [%[in]]!\n"
      "vld1.32 {d12-d15}, [%[in]]!\n"
      // 32->16
      "vcvt.f16.f32 d16, q0\n"
      "vcvt.f16.f32 d17, q1\n"
      "vcvt.f16.f32 d18, q2\n"
      "vcvt.f16.f32 d19, q3\n"
      "vcvt.f16.f32 d20, q4\n"
      "vcvt.f16.f32 d21, q5\n"
      "vst1.32 {d16-d19}, [%[out]]!\n"
      "subs %[cnt], #1\n"
      "vcvt.f16.f32 d22, q6\n"
      "vcvt.f16.f32 d23, q7\n"
      "vst1.32 {d20-d23}, [%[out]]!\n"
      "bne 0b\n"
      "1: \n"
      "cmp %[remain_cnt], #1\n"
      "blt 2f\n"
      "4: \n"
      "vld1.32 {d0-d3}, [%[in]]!\n"
      "subs %[remain_cnt], #1\n"
      // 32->16
      "vcvt.f16.f32 d16, q0\n"
      "vcvt.f16.f32 d17, q1\n"
      "vst1.32 {d16-d17}, [%[out]]!\n"
      "bne 4b\n"
      "2: \n"
      : [in] "+r"(in),
        [out] "+r"(out),
        [cnt] "+r"(cnt),
        [remain_cnt] "+r"(remain_cnt),
        [remain_remain] "+r"(remain_remain)
      :
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
        "q11");
  for (int i = 0; i < remain_remain; i++) {
    *out = static_cast<float16_t>(*in);
    out++;
    in++;
  }
#endif
}

// Safe vectorized dequant: load only within numel (no pipelined overread).
void int8_to_fp16(const int8_t* in,
                  float16_t* out,
                  const float* scale,
                  int axis_size,
                  int64_t outer_size,
                  int64_t inner_size) {
  int cnt = static_cast<int>(inner_size / 16);
  int remain = static_cast<int>(inner_size & 15);
  int64_t loop_size = axis_size * outer_size;

  LITE_PARALLEL_BEGIN(n, tid, loop_size) {
    float in_scale = scale[n % axis_size];
    const int8_t* din = in + n * inner_size;
    float16_t* dout = out + n * inner_size;
    float32x4_t vscale = vdupq_n_f32(in_scale);
    for (int i = 0; i < cnt; ++i) {
      int8x16_t vin = vld1q_s8(din);
      din += 16;
      int16x8_t v16_lo = vmovl_s8(vget_low_s8(vin));
      int16x8_t v16_hi = vmovl_s8(vget_high_s8(vin));
      float32x4_t f0 =
          vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_lo))), vscale);
      float32x4_t f1 =
          vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_lo))), vscale);
      float32x4_t f2 =
          vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v16_hi))), vscale);
      float32x4_t f3 =
          vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v16_hi))), vscale);
#ifdef __aarch64__
      float16x4_t h0 = vcvt_f16_f32(f0);
      float16x4_t h1 = vcvt_f16_f32(f1);
      float16x4_t h2 = vcvt_f16_f32(f2);
      float16x4_t h3 = vcvt_f16_f32(f3);
      vst1_f16(dout, h0);
      vst1_f16(dout + 4, h1);
      vst1_f16(dout + 8, h2);
      vst1_f16(dout + 12, h3);
#else
      // ARMv7: no native f16 store helper in all toolchains — scalar tail path
      // for the 16-wide block via temporary float.
      float tmp[16];
      vst1q_f32(tmp, f0);
      vst1q_f32(tmp + 4, f1);
      vst1q_f32(tmp + 8, f2);
      vst1q_f32(tmp + 12, f3);
      for (int k = 0; k < 16; ++k) {
        dout[k] = static_cast<float16_t>(tmp[k]);
      }
#endif
      dout += 16;
    }
    for (int i = 0; i < remain; ++i) {
      dout[i] = static_cast<float16_t>(in_scale * static_cast<float>(din[i]));
    }
  }
  LITE_PARALLEL_END()
}

void uint8_to_fp16(const uint8_t* in,
                   float16_t* out,
                   const float* scale,
                   int axis_size,
                   int64_t outer_size,
                   int64_t inner_size) {
  int cnt = static_cast<int>(inner_size / 16);
  int remain = static_cast<int>(inner_size & 15);
  int64_t loop_size = axis_size * outer_size;
  const int16x8_t v128 = vdupq_n_s16(128);

  LITE_PARALLEL_BEGIN(n, tid, loop_size) {
    float in_scale = scale[n % axis_size];
    const uint8_t* din = in + n * inner_size;
    float16_t* dout = out + n * inner_size;
    float32x4_t vscale = vdupq_n_f32(in_scale);
    for (int i = 0; i < cnt; ++i) {
      uint8x16_t vu = vld1q_u8(din);
      din += 16;
      int16x8_t s_lo =
          vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(vu))), v128);
      int16x8_t s_hi =
          vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(vu))), v128);
      float32x4_t f0 =
          vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s_lo))), vscale);
      float32x4_t f1 =
          vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s_lo))), vscale);
      float32x4_t f2 =
          vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s_hi))), vscale);
      float32x4_t f3 =
          vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s_hi))), vscale);
#ifdef __aarch64__
      vst1_f16(dout, vcvt_f16_f32(f0));
      vst1_f16(dout + 4, vcvt_f16_f32(f1));
      vst1_f16(dout + 8, vcvt_f16_f32(f2));
      vst1_f16(dout + 12, vcvt_f16_f32(f3));
#else
      float tmp[16];
      vst1q_f32(tmp, f0);
      vst1q_f32(tmp + 4, f1);
      vst1q_f32(tmp + 8, f2);
      vst1q_f32(tmp + 12, f3);
      for (int k = 0; k < 16; ++k) {
        dout[k] = static_cast<float16_t>(tmp[k]);
      }
#endif
      dout += 16;
    }
    for (int i = 0; i < remain; ++i) {
      dout[i] = static_cast<float16_t>(in_scale *
                                       (static_cast<float>(din[i]) - 128.f));
    }
  }
  LITE_PARALLEL_END()
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
