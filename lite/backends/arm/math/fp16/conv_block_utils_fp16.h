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

#pragma once
#include <arm_neon.h>
#include <cmath>
#include "lite/backends/arm/math/fp16/gemm_fp16.h"
#include "lite/core/target_wrapper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/log/cp_logging.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
inline void trans_gemm_weights_fp16(const Tensor& tin,
                                    Tensor& tout,  // NOLINT
                                    int group,
                                    ARMContext* ctx) {
  CHECK_EQ(tin.dims().size(), 4) << "conv weights dims size must = 4";
  int m = tin.dims()[0] / group;
  int k = tin.dims().count(1, 4);
  int hblock = lite::arm::math::fp16::get_hblock_fp16(ctx);
  int m_roundup = hblock * ((m + hblock - 1) / hblock);
  int group_size_round_up = ((m_roundup * k + 15) / 16) * 16;
  float16_t* w_trans_ptr = nullptr;
  tout.Resize({group_size_round_up * group});
  w_trans_ptr = tout.mutable_data<float16_t>();
  const auto* w_data = tin.data<float16_t>();
  for (int g = 0; g < group; ++g) {
    const float16_t* weights_group = w_data + g * m * k;
    float16_t* weights_trans_ptr = w_trans_ptr + g * group_size_round_up;
    lite::arm::math::fp16::prepackA_fp16(
        weights_trans_ptr, weights_group, 1.f, k, 0, m, 0, k, false, ctx);
  }
}

inline bool prepack_input_nxw(const float16_t* din,
                              float16_t* dout,
                              int cs,
                              int ce,
                              int hs,
                              int he,
                              int ws,
                              int we,
                              int channel,
                              int width,
                              int height,
                              float16_t* zero_ptr) {
  int n = he - hs;
  if (n <= 0) {
    LOG(ERROR) << "input height is more than zero";
    return false;
  }
  int w0 = ws < 0 ? 0 : ws;
  int w1 = we > width ? width : we;

  int size_w = we - ws;
  int size_wc_len = size_w * channel;
  int size_c = width * height;

  int valid_w = w1 - w0;
  size_t valid_w_byte = valid_w * sizeof(float16_t);

  float16_t* out_array[n];
  out_array[0] = dout;
  for (int i = 1; i < n; i++) {
    out_array[i] = out_array[i - 1] + size_wc_len;
  }

  for (int c = 0; c < channel; ++c) {
    int j = 0;
    // valid height
    for (int i = hs; i < he; i++) {
      // get address
      const float16_t* in_array;
      if (i < 0 || i >= height) {
        in_array = zero_ptr;
      } else {
        in_array = din + i * width;
      }

      for (int w = ws; w < w0; ++w) {
        *(out_array[j]++) = 0.f;
      }
      lite::TargetWrapperHost::MemcpySync(out_array[j], in_array, valid_w_byte);
      out_array[j] += valid_w;
      for (int w = w1; w < we; ++w) {
        *(out_array[j]++) = 0.f;
      }
      j++;
    }
    din += size_c;
  }
  return true;
}

/**
 * @brief C4 transpose process
 * input: N 4 H W, output: N 1 H W 4
 * @param din input ptr
 * @param dout output ptr
 * @param cs channel start
 * @param ce channel end
 * @param hs height start
 * @param he height end
 * @param ws width start
 * @param we width end
 * @param channel  number of channel, require: channel <=4
 * @param width number of width
 * @param height number of height
 * @param zero_ptr
 */
inline void prepack_input_nxwc4(const float16_t* din,
                                float16_t* dout,
                                int cs,
                                int ce,
                                int hs,
                                int he,
                                int ws,
                                int we,
                                int channel,
                                int width,
                                int height,
                                float16_t* zero_ptr) {
  int n = he - hs;
  if (n <= 0) {
    LOG(ERROR) << "input height is more than zero";
  }
  int size_w = we - ws;
  int w0 = ws < 0 ? 0 : ws;
  int w1 = we > width ? width : we;
  int valid_w = w1 - w0;
  int pad_l = ws < 0 ? -ws : 0;
  int pad_r = we > width ? we - width : 0;
  int size_c = width * height;

  int valid_cnt = valid_w >> 3;
  int remain = valid_w & 7;

  for (int h = hs; h < he; ++h) {
    const float16_t* ptr_c0 = din + h * width + cs * size_c;
    const float16_t* ptr_c1 = ptr_c0 + size_c;
    const float16_t* ptr_c2 = ptr_c1 + size_c;
    const float16_t* ptr_c3 = ptr_c2 + size_c;
    if (h < 0 || h >= height) {
      memset(dout, 0, 4 * size_w * sizeof(float16_t));
      dout += size_w * 4;
      continue;
    } else if (cs + 4 > channel) {
      switch (cs + 4 - channel) {
        case 3:
          ptr_c1 = zero_ptr;
        case 2:
          ptr_c2 = zero_ptr;
        case 1:
          ptr_c3 = zero_ptr;
        default:
          break;
      }
    }
    if (pad_l) {
      memset(dout, 0, pad_l * 4 * sizeof(float16_t));
      dout += pad_l * 4;
    }
    if (valid_cnt) {
      int cnt = valid_cnt;
#ifdef __aarch64__
      asm volatile(
          /* main loop */
          "1:\n"
          "ldr q0,    [%[r0]], #16\n"
          "ldr q1,    [%[r1]], #16\n"
          "ldr q2,    [%[r2]], #16\n"
          "ldr q3,    [%[r3]], #16\n"
          // a0b0a2b2a4b4a6b6
          "trn1 v8.8h,  v0.8h, v1.8h\n"
          // a1b1a3b3a5b5a7b7
          "trn2 v9.8h,  v0.8h, v1.8h\n"
          "trn1 v10.8h, v2.8h, v3.8h\n"
          "trn2 v11.8h, v2.8h, v3.8h\n"

          // a0b0c0d0a4b4c4d4
          "trn1 v0.4s,  v8.4s, v10.4s\n"
          // a2b2c2d2a6b6c6d6
          "trn2 v1.4s,  v8.4s, v10.4s\n"
          // a1b1c1d1
          "trn1 v2.4s,  v9.4s, v11.4s\n"
          // a3b3c3d3
          "trn2 v3.4s,  v9.4s, v11.4s\n"

          "trn1 v8.2d,  v0.2d, v2.2d\n"  // 0 1
          "trn1 v9.2d,  v1.2d, v3.2d\n"  // 2 3
          "trn2 v10.2d, v0.2d, v2.2d\n"  // 4 5
          "trn2 v11.2d, v1.2d, v3.2d\n"  // 6 7
          "str q8, [%[ptr_out]], #16\n"
          "subs %w[cnt], %w[cnt], #1\n"
          "str q9, [%[ptr_out]], #16\n"
          "str q10, [%[ptr_out]], #16\n"
          "str q11, [%[ptr_out]], #16\n"
          "bne    1b\n"
          : [cnt] "+r"(cnt),
            [r0] "+r"(ptr_c0),
            [r1] "+r"(ptr_c1),
            [r2] "+r"(ptr_c2),
            [r3] "+r"(ptr_c3),
            [ptr_out] "+r"(dout)
          :
          : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11");
#else
      asm volatile(
          /* main loop */
          "1:\n"
          "vld1.32 {d0-d1},  [%[r0]]!\n"
          "vld1.32 {d2-d3},  [%[r1]]!\n"
          "vld1.32 {d4-d5},  [%[r2]]!\n"
          "vld1.32 {d6-d7},  [%[r3]]!\n"
          "vtrn.16   q0, q1\n"
          "vtrn.16   q2, q3\n"
          "vtrn.32   q0, q2\n"
          "vtrn.32   q1, q3\n"

          "vswp      d1, d2\n"
          "vswp      d5, d6\n"

          "subs %[cnt], #1\n"
          "vst1.16 {d0-d1}, [%[ptr_out]]!\n"
          "vst1.16 {d4-d5}, [%[ptr_out]]!\n"
          "vst1.16 {d2-d3}, [%[ptr_out]]!\n"
          "vst1.16 {d6-d7}, [%[ptr_out]]!\n"
          "bne    1b\n"
          : [cnt] "+r"(cnt),
            [r0] "+r"(ptr_c0),
            [r1] "+r"(ptr_c1),
            [r2] "+r"(ptr_c2),
            [r3] "+r"(ptr_c3),
            [ptr_out] "+r"(dout)
          :
          : "cc", "memory", "q0", "q1", "q2", "q3");
#endif  // __aarch64__
    }
    for (int i = 0; i < remain; ++i) {
      dout[0] = *(ptr_c0++);
      dout[1] = *(ptr_c1++);
      dout[2] = *(ptr_c2++);
      dout[3] = *(ptr_c3++);
      dout += 4;
    }
    if (pad_r) {
      memset(dout, 0, pad_r * 4 * sizeof(float16_t));
      dout += pad_r * 4;
    }
  }
}

// clang-format off
#ifdef __aarch64__
#define INIT_C8                       \
  "cmp %w[cnt], #1\n"                 \
  "ldp  q0, q1, [%[din_ptr]], #32\n"  \
  "ldp  q2, q3, [%[din_ptr]], #32\n"  \
  "prfm pldl1keep, [%[din_ptr]]\n"    \
  "blt 2f\n"

#define PROCESS_C8                    \
  "1: \n"                             \
  "ldp  q4, q5, [%[din_ptr]], #32\n"  \
  "ldp  q6, q7, [%[din_ptr]], #32\n"  \
  "prfm pldl1keep, [%[din_ptr]]\n"

#define PROCESS_C8_REMAIN             \
  "2: \n"                             \
  "ldp  q4, q5, [%[din_ptr]], #32\n"  \
  "ldp  q6, q7, [%[din_ptr]], #32\n"

#define TRANS_C8                      \
  "fadd v0.8h, v0.8h, %[vbias].8h\n"  \
  "fadd v1.8h, v1.8h, %[vbias].8h\n"  \
  "fadd v2.8h, v2.8h, %[vbias].8h\n"  \
  "fadd v3.8h, v3.8h, %[vbias].8h\n"  \
  "fadd v4.8h, v4.8h, %[vbias].8h\n"  \
  "fadd v5.8h, v5.8h, %[vbias].8h\n"  \
  /* v8=a0b0a2b2a4b4a6b6 */           \
  "trn1 v8.8h, v0.8h, v1.8h\n"        \
  "fadd v6.8h, v6.8h, %[vbias].8h\n"  \
  /* v9=a1b1a3b3a5b5a7b7 */           \
  "trn2 v9.8h, v0.8h, v1.8h\n"        \
  "fadd v7.8h, v7.8h, %[vbias].8h\n"  \
  /* v10=c0d0c2d2c4d4c6d6 */          \
  "trn1 v10.8h, v2.8h, v3.8h\n"       \
  /* v11=c1d1c3d3c5d5c7d7 */          \
  "trn2 v11.8h, v2.8h, v3.8h\n"       \
  /* v12=e0f0e2f2e4f4e6f6 */          \
  "trn1 v12.8h, v4.8h, v5.8h\n"       \
  /* v13=e1f1e3f3e5f5e7f7 */          \
  "trn2 v13.8h, v4.8h, v5.8h\n"       \
  /* v0=a0b0c0d0a4b4c4d4 */           \
  "trn1 v0.4s, v8.4s, v10.4s\n"       \
  /* v1=a2b2c2d2a6b6c6d6 */           \
  "trn2 v1.4s, v8.4s, v10.4s\n"       \
  /* v14=g0h0g2h2e4f4e6f6 */          \
  "trn1 v14.8h, v6.8h, v7.8h\n"       \
  /* v15=g1h1g3h3e5f5e7f7 */          \
  "trn2 v15.8h, v6.8h, v7.8h\n"       \
  /* v2=a1b1c1d1a5b4c4d4 */           \
  "trn1 v2.4s, v9.4s, v11.4s\n"       \
  /* v3=a3b3c3d3a7b6c6d6 */           \
  "trn2 v3.4s, v9.4s, v11.4s\n"       \
  /* v4=e0f0g0h0e4b4c4d4 */           \
  "trn1 v4.4s, v12.4s, v14.4s\n"      \
  /* v5=e2f2g2h2e6b6c6d6 */           \
  "trn2 v5.4s, v12.4s, v14.4s\n"      \
  /* v6=e1f1g1h1e5b4c4d4 */           \
  "trn1 v6.4s, v13.4s, v15.4s\n"      \
  /* v7=e3f3g3h3e7b6c6d6 */           \
  "trn2 v7.4s, v13.4s, v15.4s\n"      \
  /* v8=a0b0c0d0e0f0g0h0 */           \
  "trn1 v8.2d, v0.2d, v4.2d\n"        \
  /* v9=a4b4c4d4e4f4g4h4 */           \
  "trn2 v9.2d, v0.2d, v4.2d\n"        \
  /* v10=a2b2c2d2e2f2g2h2 */          \
  "trn1 v10.2d, v1.2d, v5.2d\n"       \
  /* v11=a6b6c6d6e6b6c6d6 */          \
  "trn2 v11.2d, v1.2d, v5.2d\n"       \
  /* v12=a1b1c1d1e1f1g1h1 */          \
  "trn1 v12.2d, v2.2d, v6.2d\n"       \
  /* v13=a5b5c5d5e5f5g5h5 */          \
  "trn2 v13.2d, v2.2d, v6.2d\n"       \
  /* v14=a3b3c3d3e3f3g3h3 */          \
  "trn1 v14.2d, v3.2d, v7.2d\n"       \
  /* v14=a7b7c7d7e7f7g7h7 */          \
  "trn2 v15.2d, v3.2d, v7.2d\n"

#define RELU_C8                       \
  "fmax v8.8h, v8.8h, %[vzero].8h\n"  \
  "fmax v9.8h, v9.8h, %[vzero].8h\n"  \
  "fmax v10.8h, v10.8h, %[vzero].8h\n"\
  "fmax v11.8h, v11.8h, %[vzero].8h\n"\
  "fmax v12.8h, v12.8h, %[vzero].8h\n"\
  "fmax v13.8h, v13.8h, %[vzero].8h\n"\
  "fmax v14.8h, v14.8h, %[vzero].8h\n"\
  "fmax v15.8h, v15.8h, %[vzero].8h\n"

#define RELU6_C8                       \
  "fmin v8.8h, v8.8h, %[valpha].8h\n"  \
  "fmin v9.8h, v9.8h, %[valpha].8h\n"  \
  "fmin v10.8h, v10.8h, %[valpha].8h\n"\
  "fmin v11.8h, v11.8h, %[valpha].8h\n"\
  "fmin v12.8h, v12.8h, %[valpha].8h\n"\
  "fmin v13.8h, v13.8h, %[valpha].8h\n"\
  "fmin v14.8h, v14.8h, %[valpha].8h\n"\
  "fmin v15.8h, v15.8h, %[valpha].8h\n"

#define LEAKYRELU_C8                   \
  "fcmge v0.8h, v8.8h, %[vzero].8h\n"  \
  "fmul v1.8h, v8.8h, %[valpha].8h\n"  \
  "fcmge v2.8h, v9.8h, %[vzero].8h\n"  \
  "fmul v3.8h, v9.8h, %[valpha].8h\n"  \
  "fcmge v4.8h, v10.8h, %[vzero].8h\n" \
  "fmul v5.8h, v10.8h, %[valpha].8h\n" \
  "bif v8.16b, v1.16b, v0.16b\n"       \
  "fcmge v6.8h, v11.8h, %[vzero].8h\n" \
  "fmul v7.8h, v11.8h, %[valpha].8h\n" \
  "bif v9.16b, v3.16b, v2.16b\n"       \
  "fcmge v0.8h, v12.8h, %[vzero].8h\n" \
  "fmul v1.8h, v12.8h, %[valpha].8h\n" \
  "bif v10.16b, v5.16b, v4.16b\n"      \
  "fcmge v2.8h, v13.8h, %[vzero].8h\n" \
  "fmul v3.8h, v13.8h, %[valpha].8h\n" \
  "bif v11.16b, v7.16b, v6.16b\n"      \
  "fcmge v4.8h, v14.8h, %[vzero].8h\n" \
  "fmul v5.8h, v14.8h, %[valpha].8h\n" \
  "bif v12.16b, v1.16b, v0.16b\n"      \
  "fcmge v6.8h, v15.8h, %[vzero].8h\n" \
  "fmul v7.8h, v15.8h, %[valpha].8h\n" \
  "bif v13.16b, v3.16b, v2.16b\n"      \
  "bif v14.16b, v5.16b, v4.16b\n"      \
  "bif v15.16b, v7.16b, v6.16b\n"

#define HARD_SWISH_C8                  \
  "fadd v0.8h, v8.8h, %[voffset].8h\n" \
  "fadd v1.8h, v9.8h, %[voffset].8h\n" \
  "fadd v2.8h, v10.8h, %[voffset].8h\n"\
  "fadd v3.8h, v11.8h, %[voffset].8h\n"\
  "fadd v4.8h, v12.8h, %[voffset].8h\n"\
  "fadd v5.8h, v13.8h, %[voffset].8h\n"\
  "fadd v6.8h, v14.8h, %[voffset].8h\n"\
  "fadd v7.8h, v15.8h, %[voffset].8h\n"\
  "fmul v16.8h, v8.8h, %[valpha].8h\n" \
  "fmul v17.8h, v9.8h, %[valpha].8h\n" \
  "fmul v18.8h, v10.8h, %[valpha].8h\n"\
  "fmul v19.8h, v11.8h, %[valpha].8h\n"\
  "fmul v20.8h, v12.8h, %[valpha].8h\n"\
  "fmul v21.8h, v13.8h, %[valpha].8h\n"\
  "fmul v22.8h, v14.8h, %[valpha].8h\n"\
  "fmul v23.8h, v15.8h, %[valpha].8h\n"\
  "fmax v0.8h,  v0.8h, %[vzero].8h\n"  \
  "fmax v1.8h,  v1.8h, %[vzero].8h\n"  \
  "fmax v2.8h,  v2.8h, %[vzero].8h\n"  \
  "fmax v3.8h,  v3.8h, %[vzero].8h\n"  \
  "fmax v4.8h,  v4.8h, %[vzero].8h\n"  \
  "fmax v5.8h,  v5.8h, %[vzero].8h\n"  \
  "fmax v6.8h,  v6.8h, %[vzero].8h\n"  \
  "fmax v7.8h,  v7.8h, %[vzero].8h\n"  \
  "fmin v0.8h,  v0.8h, %[vthreshold].8h\n"\
  "fmin v1.8h,  v1.8h, %[vthreshold].8h\n"\
  "fmin v2.8h,  v2.8h, %[vthreshold].8h\n"\
  "fmin v3.8h,  v3.8h, %[vthreshold].8h\n"\
  "fmin v4.8h,  v4.8h, %[vthreshold].8h\n"\
  "fmin v5.8h,  v5.8h, %[vthreshold].8h\n"\
  "fmin v6.8h,  v6.8h, %[vthreshold].8h\n"\
  "fmin v7.8h,  v7.8h, %[vthreshold].8h\n"\
  "fmul v8.8h,  v0.8h,  v16.8h\n"      \
  "fmul v9.8h,  v1.8h,  v17.8h\n"      \
  "fmul v10.8h, v2.8h,  v18.8h\n"      \
  "fmul v11.8h, v3.8h,  v19.8h\n"      \
  "fmul v12.8h, v4.8h,  v20.8h\n"      \
  "fmul v13.8h, v5.8h,  v21.8h\n"      \
  "fmul v14.8h, v6.8h,  v22.8h\n"      \
  "fmul v15.8h, v7.8h,  v23.8h\n"
#define STORE_C8                       \
  "str q8, [%[doutc0r0]], #16\n"       \
  "ldp  q0, q1, [%[din_ptr]], #32\n"   \
  "str q9, [%[doutc4r0]], #16\n"       \
  "ldp  q2, q3, [%[din_ptr]], #32\n"   \
  "str q10, [%[doutc2r0]], #16\n"      \
  "prfm pldl1keep, [%[din_ptr]]\n"     \
  "str q11, [%[doutc6r0]], #16\n"      \
  "subs %w[cnt], %w[cnt], #1\n"        \
  "str q12, [%[doutc1r0]], #16\n"      \
  "str q13, [%[doutc5r0]], #16\n"      \
  "str q14, [%[doutc3r0]], #16\n"      \
  "str q15, [%[doutc7r0]], #16\n"      \
  "bne 1b\n"

#define STORE_C8_REMAIN                \
  "str q8,  [%[tmp0]]\n"               \
  "str q12, [%[tmp0], 0x10]\n"         \
  "str q10, [%[tmp0], 0x20]\n"         \
  "str q14, [%[tmp0], 0x30]\n"         \
  "str q9,  [%[tmp0], 0x40]\n"         \
  "str q13, [%[tmp0], 0x50]\n"         \
  "str q11, [%[tmp0], 0x60]\n"         \
  "str q15, [%[tmp0], 0x70]\n"

#define ASM_PARAM                     \
  :  [doutc0r0] "+r"(doutc0_ptr),    \
    [doutc1r0] "+r"(doutc1_ptr),     \
    [doutc2r0] "+r"(doutc2_ptr),     \
    [doutc3r0] "+r"(doutc3_ptr),     \
    [doutc4r0] "+r"(doutc4_ptr),     \
    [doutc5r0] "+r"(doutc5_ptr),     \
    [doutc6r0] "+r"(doutc6_ptr),     \
    [doutc7r0] "+r"(doutc7_ptr),     \
    [cnt] "+r"(cnt),                 \
    [din_ptr] "+r"(din_hei_ptr)      \
  : [vbias] "w"(vbias),              \
    [tmp0] "r"(tmp0),                \
    [vzero] "w"(vzero),              \
    [valpha] "w"(valpha)             \
  : "cc", "memory", "v0", "v1", "v2", \
    "v3", "v4", "v5", "v6", "v7",     \
    "v8", "v9", "v10", "v11", "v12",  \
    "v13", "v14", "v15", "v16"
#define ASM_PARAM_1                  \
  :  [doutc0r0] "+r"(doutc0_ptr),    \
    [doutc1r0] "+r"(doutc1_ptr),     \
    [doutc2r0] "+r"(doutc2_ptr),     \
    [doutc3r0] "+r"(doutc3_ptr),     \
    [doutc4r0] "+r"(doutc4_ptr),     \
    [doutc5r0] "+r"(doutc5_ptr),     \
    [doutc6r0] "+r"(doutc6_ptr),     \
    [doutc7r0] "+r"(doutc7_ptr),     \
    [cnt] "+r"(cnt),                 \
    [din_ptr] "+r"(din_hei_ptr)      \
  : [vbias] "w"(vbias),              \
    [tmp0] "r"(tmp0),                \
    [vzero] "w"(vzero),              \
    [voffset] "w"(voffset),          \
    [vthreshold] "w"(vthreshold),    \
    [valpha] "w"(valpha)             \
  : "cc", "memory", "v0", "v1", "v2", \
    "v3", "v4", "v5", "v6", "v7",     \
    "v8", "v9", "v10", "v11", "v12",  \
    "v13", "v14", "v15", "v16", "v16", \
    "v17", "v18", "v19", "v20", "v21", \
    "v22", "v23"
#define C8_OUT_REMAIN                \
  for (int j = 0; j < remain; j++) { \
    *doutc0_ptr++ = tmp0[j];         \
    *doutc1_ptr++ = tmp0[j + 8];     \
    *doutc2_ptr++ = tmp0[j + 16];    \
    *doutc3_ptr++ = tmp0[j + 24];    \
    *doutc4_ptr++ = tmp0[j + 32];    \
    *doutc5_ptr++ = tmp0[j + 40];    \
    *doutc6_ptr++ = tmp0[j + 48];    \
    *doutc7_ptr++ = tmp0[j + 56];    \
  }
#else
#define INIT_C8                       \
  "cmp %[cnt], #1\n"                  \
  "vld1.16 {d0-d3}, [%[din_ptr]]!\n"   \
  "vld1.16 {d4-d7}, [%[din_ptr]]!\n"   \
  "pld [%[din_ptr]]\n"                \
  "blt 2f\n"

#define PROCESS_C8                    \
  "1: \n"                             \
  "vld1.16 {d8-d11}, [%[din_ptr]]!\n"  \
  "vld1.16 {d12-d15}, [%[din_ptr]]!\n" \
  "pld [%[din_ptr]]\n"

#define PROCESS_C8_REMAIN             \
  "2: \n"                             \
  "vld1.16 {d8-d11}, [%[din_ptr]]!\n"  \
  "vld1.16 {d12-d15}, [%[din_ptr]]!\n"

#define TRANS_C8                      \
  "vadd.f16 q0, q0, %q[vbias]    \n"  \
  "vadd.f16 q1, q1, %q[vbias]    \n"  \
  "vadd.f16 q2, q2, %q[vbias]    \n"  \
  "vadd.f16 q3, q3, %q[vbias]    \n"  \
  "vadd.f16 q4, q4, %q[vbias]    \n"  \
  "vadd.f16 q5, q5, %q[vbias]    \n"  \
  /* q0=a0b0a2b2a4b4a6b6 q1=a1b1a3b3a5b5a7b7*/ \
  "vtrn.16 q0, q1                \n"  \
  "vadd.f16 q6, q6, %q[vbias]    \n"  \
  /* v2=c0d0c2d2c4d4c6d6 v3=c1d1c3d3c5d5c7d7*/ \
  "vtrn.16 q2, q3                \n"  \
  "vadd.f16 q7, q7, %q[vbias]    \n"  \
  /* v4=e0f0e2f2e4f4e6f6 v5=e1f1e3f3e5f5e7f7*/ \
  "vtrn.16 q4, q5                \n"  \
  /* v6=g0h0g2h2e4f4e6f6 v7=g1h1g3h3e5f5e7f7*/ \
  "vtrn.16 q6, q7                \n"  \
  /* q0=a0b0c0d0a4b4c4d4 v2=a2b2c2d2a6b6c6d6*/ \
  "vtrn.32 q0, q2                \n"  \
  /* v1=a1b1c1d1a5b4c4d4 v3=a3b3c3d3a7b6c6d6*/ \
  "vtrn.32 q1, q3                \n"  \
  /* v4=e0f0g0h0e4b4c4d4 v6=e2f2g2h2e6b6c6d6*/ \
  "vtrn.32 q4, q6                \n"  \
  /* v5=e1f1g1h1e5b4c4d4 v7=e3f3g3h3e7b6c6d6*/ \
  "vtrn.32 q5, q7                \n"  \
  /* v0=a0b0c0d0e0f0g0h0 v4=a4b4c4d4e4f4g4h4*/ \
  "vswp    d1, d8                \n"   \
  /* v2=a2b2c2d2e2f2g2h2 v6=a6b6c6d6e6b6c6d6*/ \
  "vswp    d5, d12               \n"  \
  /* v1=a1b1c1d1e1f1g1h1 v5=a5b5c5d5e5f5g5h5*/ \
  "vswp    d3, d10               \n"  \
  /* v3=a3b3c3d3e3f3g3h3 v7=a7b7c7d7e7f7g7h7*/ \
  "vswp    d7, d14               \n"

#define RELU_C8                       \
  "vmax.f16 q0,  q0, %q[vzero]\n"     \
  "vmax.f16 q1,  q1, %q[vzero]\n"     \
  "vmax.f16 q2,  q2, %q[vzero]\n"     \
  "vmax.f16 q3,  q3, %q[vzero]\n"     \
  "vmax.f16 q4,  q4, %q[vzero]\n"     \
  "vmax.f16 q5,  q5, %q[vzero]\n"     \
  "vmax.f16 q6,  q6, %q[vzero]\n"     \
  "vmax.f16 q7,  q7, %q[vzero]\n"

#define RELU6_C8                       \
  "vmin.f16 q0,  q0, %q[valpha]\n"     \
  "vmin.f16 q1,  q1, %q[valpha]\n"     \
  "vmin.f16 q2,  q2, %q[valpha]\n"     \
  "vmin.f16 q3,  q3, %q[valpha]\n"     \
  "vmin.f16 q4,  q4, %q[valpha]\n"     \
  "vmin.f16 q5,  q5, %q[valpha]\n"     \
  "vmin.f16 q6,  q6, %q[valpha]\n"     \
  "vmin.f16 q7,  q7, %q[valpha]\n"

#define LEAKYRELU_C8                   \
  "vcge.f16  q8,  q0, %q[vzero] \n"    \
  "vmul.f16  q9,  q0, %q[valpha]\n"    \
  "vcge.f16  q10, q1, %q[vzero] \n"    \
  "vmul.f16  q11, q1, %q[valpha]\n"    \
  "vbif      q0,  q9,  q8        \n"   \
  "vbif      q1,  q11, q10       \n"   \
  "vcge.f16  q8,  q2, %q[vzero]  \n"   \
  "vmul.f16  q9,  q2, %q[valpha] \n"   \
  "vcge.f16  q10, q3,  %q[vzero] \n"   \
  "vmul.f16  q11, q3,  %q[valpha]\n"   \
  "vbif      q2,  q9,  q8        \n"   \
  "vbif      q3,  q11, q10       \n"   \
  "vcge.f16  q8,  q4, %q[vzero] \n"    \
  "vmul.f16  q9,  q4, %q[valpha]\n"    \
  "vcge.f16  q10, q5, %q[vzero] \n"    \
  "vmul.f16  q11, q5, %q[valpha]\n"    \
  "vbif      q4,  q9, q8        \n"    \
  "vbif      q5,  q11,q10       \n"    \
  "vcge.f16  q8,  q6, %q[vzero] \n"    \
  "vmul.f16  q9,  q6, %q[valpha]\n"    \
  "vcge.f16  q10, q7, %q[vzero] \n"    \
  "vmul.f16  q11, q7, %q[valpha]\n"    \
  "vbif      q6,  q9, q8        \n"    \
  "vbif      q7,  q11,q10       \n"

#define HARD_SWISH_C8                   \
  "vld1.16 {d16-d17}, [%[voffset]]\n"   \
  "vldr     d18, [%[voffset], #16]\n"   \
  "vldr     d19, [%[voffset], #24]\n"   \
  "vmul.f16 q10, q0, %q[valpha]\n"      \
  "vadd.f16 q11, q0,  q8\n"             \
  "vmul.f16 q12, q1, %q[valpha]\n"      \
  "vmax.f16 q11, q11, %q[vzero]\n"      \
  "vmin.f16 q11, q11, q9\n"             \
  "vmul.f16 q0,  q10, q11\n"            \
  "vadd.f16 q11, q1,  q8\n"             \
  "vmul.f16 q10, q2, %q[valpha]\n"      \
  "vmax.f16 q11, q11, %q[vzero]\n"      \
  "vmin.f16 q11, q11, q9\n"             \
  "vmul.f16 q1,  q12, q11\n"            \
  "vadd.f16 q11, q2,  q8\n"             \
  "vmul.f16 q12, q3, %q[valpha]\n"      \
  "vmax.f16 q11, q11, %q[vzero]\n"      \
  "vmin.f16 q11, q11, q9\n"             \
  "vmul.f16 q2,  q10, q11\n"            \
  "vadd.f16 q11, q3,  q8\n"             \
  "vmul.f16 q10, q4, %q[valpha]\n"      \
  "vmax.f16 q11, q11, %q[vzero]\n"      \
  "vmin.f16 q11, q11, q9\n"             \
  "vmul.f16 q3,  q12, q11\n"            \
  "vadd.f16 q11, q4,  q8\n"             \
  "vmul.f16 q12, q5, %q[valpha]\n"      \
  "vmax.f16 q11, q11, %q[vzero]\n"      \
  "vmin.f16 q11, q11, q9\n"             \
  "vmul.f16 q4,  q10, q11\n"            \
  "vadd.f16 q11, q5,  q8\n"             \
  "vmul.f16 q10, q6, %q[valpha]\n"      \
  "vmax.f16 q11, q11, %q[vzero]\n"      \
  "vmin.f16 q11, q11, q9\n"             \
  "vmul.f16 q5,  q12, q11\n"            \
  "vadd.f16 q11, q6,  q8\n"             \
  "vmul.f16 q12, q7, %q[valpha]\n"      \
  "vmax.f16 q11, q11, %q[vzero]\n"      \
  "vmin.f16 q11, q11, q9\n"             \
  "vmul.f16 q6,  q10, q11\n"            \
  "vadd.f16 q11, q7,  q8\n"             \
  "vmax.f16 q11, q11, %q[vzero]\n"      \
  "vmin.f16 q11, q11, q9\n"             \
  "vmul.f16 q7,  q12, q11\n"

#define STORE_C8                       \
  "vst1.16 {d0-d1},   [%[doutc0r0]]!\n"  \
  "vst1.16 {d2-d3},   [%[doutc1r0]]!\n"  \
  "vst1.16 {d4-d5},   [%[doutc2r0]]!\n"  \
  "vst1.16 {d6-d7},   [%[doutc3r0]]!\n"  \
  "vld1.16 {d0-d3},   [%[din_ptr]]!  \n"  \
  "subs    %[cnt],    #1            \n"  \
  "vst1.16 {d8-d9},   [%[doutc4r0]]!\n"  \
  "vst1.16 {d10-d11}, [%[doutc5r0]]!\n"  \
  "vst1.16 {d12-d13}, [%[doutc6r0]]!\n"  \
  "vst1.16 {d14-d15}, [%[doutc7r0]]!\n"  \
  "vld1.16 {d4-d7},   [%[din_ptr]]!  \n"  \
  "pld    [%[din_ptr]]              \n"  \
  "bne 1b\n"

#define STORE_C8_REMAIN                 \
  "vstr    d0,        [%[tmp0], #0x00]\n"\
  "vstr    d1,        [%[tmp0], #0x08]\n"\
  "vstr    d2,        [%[tmp0], #0x10]\n"\
  "vstr    d3,        [%[tmp0], #0x18]\n"\
  "vstr    d4,        [%[tmp0], #0x20]\n"\
  "vstr    d5,        [%[tmp0], #0x28]\n"\
  "vstr    d6,        [%[tmp0], #0x30]\n"\
  "vstr    d7,        [%[tmp0], #0x38]\n"\
  "vstr    d8,        [%[tmp0], #0x40]\n"\
  "vstr    d9,        [%[tmp0], #0x48]\n"\
  "vstr    d10,       [%[tmp0], #0x50]\n"\
  "vstr    d11,       [%[tmp0], #0x58]\n"\
  "vstr    d12,       [%[tmp0], #0x60]\n"\
  "vstr    d13,       [%[tmp0], #0x68]\n"\
  "vstr    d14,       [%[tmp0], #0x70]\n"\
  "vstr    d15,       [%[tmp0], #0x78]\n"

#define ASM_PARAM                    \
  : [doutc0r0] "+r"(doutc0_ptr),     \
    [doutc1r0] "+r"(doutc1_ptr),     \
    [doutc2r0] "+r"(doutc2_ptr),     \
    [doutc3r0] "+r"(doutc3_ptr),     \
    [doutc4r0] "+r"(doutc4_ptr),     \
    [doutc5r0] "+r"(doutc5_ptr),     \
    [doutc6r0] "+r"(doutc6_ptr),     \
    [doutc7r0] "+r"(doutc7_ptr),     \
    [cnt] "+r"(cnt),                 \
    [din_ptr] "+r"(din_hei_ptr)      \
  : [vbias] "w"(vbias),              \
    [tmp0] "r"(tmp0),                \
    [vzero] "w"(vzero),              \
    [valpha] "w"(valpha)             \
  : "cc", "memory", "q0", "q1", "q2", \
    "q3", "q4", "q5", "q6", "q7",     \
    "q8", "q9", "q10", "q11", "q12"
#define ASM_PARAM_1                  \
  : [doutc0r0] "+r"(doutc0_ptr),     \
    [doutc1r0] "+r"(doutc1_ptr),     \
    [doutc2r0] "+r"(doutc2_ptr),     \
    [doutc3r0] "+r"(doutc3_ptr),     \
    [doutc4r0] "+r"(doutc4_ptr),     \
    [doutc5r0] "+r"(doutc5_ptr),     \
    [doutc6r0] "+r"(doutc6_ptr),     \
    [doutc7r0] "+r"(doutc7_ptr),     \
    [cnt] "+r"(cnt),                 \
    [din_ptr] "+r"(din_hei_ptr)      \
  : [vbias] "w"(vbias),              \
    [tmp0] "r"(tmp0),                \
    [voffset] "r"(voffset),          \
    [vzero] "w"(vzero),              \
    [valpha] "w"(valpha)             \
  : "cc", "memory", "q0", "q1", "q2", \
    "q3", "q4", "q5", "q6", "q7",     \
    "q8", "q9", "q10", "q11", "q12"
#define C8_OUT_REMAIN                \
  for (int j = 0; j < remain; j++) { \
    *doutc0_ptr++ = tmp0[j];         \
    *doutc1_ptr++ = tmp0[j + 8];     \
    *doutc2_ptr++ = tmp0[j + 16];    \
    *doutc3_ptr++ = tmp0[j + 24];    \
    *doutc4_ptr++ = tmp0[j + 32];    \
    *doutc5_ptr++ = tmp0[j + 40];    \
    *doutc6_ptr++ = tmp0[j + 48];    \
    *doutc7_ptr++ = tmp0[j + 56];    \
  }
#endif
// clang-format on

#define C8_OUT_PARAM                  \
  float16_t* doutc0_ptr = doutc0r0;   \
  float16_t* doutc1_ptr = doutc1r0;   \
  float16_t* doutc2_ptr = doutc2r0;   \
  float16_t* doutc3_ptr = doutc3r0;   \
  float16_t* doutc4_ptr = doutc4r0;   \
  float16_t* doutc5_ptr = doutc5r0;   \
  float16_t* doutc6_ptr = doutc6r0;   \
  float16_t* doutc7_ptr = doutc7r0;   \
  const float16_t* din_hei_ptr = din; \
  int cnt = cnt_col;

#define PTR_ADD      \
  doutc0r0 += width; \
  doutc1r0 += width; \
  doutc2r0 += width; \
  doutc3r0 += width; \
  doutc4r0 += width; \
  doutc5r0 += width; \
  doutc6r0 += width; \
  doutc7r0 += width;

/*wirte result in outputs
* input din: [n, c / 8, h, w * 8], output dout: [n, c, h, w]
*/
static void write_to_oc8_fp16(const float16_t* din,
                              float16_t* dout,
                              int cs,
                              int ce,
                              int hs,
                              int he,
                              int ws,
                              int we,
                              int channel,
                              int height,
                              int width,
                              int flag_act,
                              float16_t alpha,
                              const float16_t* bias,
                              bool flag_bias,
                              float16_t offset = 0.f,
                              float16_t threshold = 6.f) {
  int size_c_out = width * height;

  float16_t* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
  float16_t* doutc1r0 = doutc0r0 + size_c_out;
  float16_t* doutc2r0 = doutc1r0 + size_c_out;
  float16_t* doutc3r0 = doutc2r0 + size_c_out;
  float16_t* doutc4r0 = doutc3r0 + size_c_out;
  float16_t* doutc5r0 = doutc4r0 + size_c_out;
  float16_t* doutc6r0 = doutc5r0 + size_c_out;
  float16_t* doutc7r0 = doutc6r0 + size_c_out;
  float16_t ptr_zero[size_c_out];  // NOLINT

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n
  int w_round = we - ws;

  int valid_we = we > width ? width : we;
  int win = valid_we - ws;
  int w_in_stride = w_round << 3;
  int cnt_col = win >> 3;
  int remain = win & 7;
  float16x8_t vbias = vdupq_n_f16(0.f);
  float16x8_t vzero = vdupq_n_f16(0.f);
  float16x8_t valpha = vdupq_n_f16(alpha);
  if (flag_bias) {
    if (ce <= channel) {
      vbias = vld1q_f16(bias);
    } else {
      float16_t bias_val[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int i = 0; i < 8 && cs + i < channel; i++) {
        bias_val[i] = bias[i];
      }
      vbias = vld1q_f16(bias_val);
    }
  }
#ifdef __aarch64__
  float16_t tmp0[64] = {0.f};
  float16x8_t voffset = vdupq_n_f16(offset);
  float16x8_t vthreshold = vdupq_n_f16(threshold);
#else
  float16_t tmp0[64] = {0.f};
  float16_t voffset[16] = {offset,
                           offset,
                           offset,
                           offset,
                           offset,
                           offset,
                           offset,
                           offset,
                           threshold,
                           threshold,
                           threshold,
                           threshold,
                           threshold,
                           threshold,
                           threshold,
                           threshold};
#endif
  if (ce > channel) {
    switch (7 - (channel - cs)) {
      case 6:
        doutc1r0 = ptr_zero;
      case 5:
        doutc2r0 = ptr_zero;
      case 4:
        doutc3r0 = ptr_zero;
      case 3:
        doutc4r0 = ptr_zero;
      case 2:
        doutc5r0 = ptr_zero;
      case 1:
        doutc6r0 = ptr_zero;
      case 0:
        doutc7r0 = ptr_zero;
      default:
        break;
    }
  }
  switch (flag_act) {
    case 0:  // no act
      for (int i = 0; i < size_h; i++) {
        C8_OUT_PARAM
        asm volatile(INIT_C8 PROCESS_C8 TRANS_C8 STORE_C8 PROCESS_C8_REMAIN
                         TRANS_C8 STORE_C8_REMAIN ASM_PARAM);
        PTR_ADD
        if (remain) {
          C8_OUT_REMAIN
        }
        din += w_in_stride;
      }
      break;
    case 1:  // relu
      for (int i = 0; i < size_h; i++) {
        C8_OUT_PARAM
        asm volatile(
            INIT_C8 PROCESS_C8 TRANS_C8 RELU_C8 STORE_C8 PROCESS_C8_REMAIN
                TRANS_C8 RELU_C8 STORE_C8_REMAIN ASM_PARAM);
        PTR_ADD
        if (remain) {
          C8_OUT_REMAIN
        }
        din += w_in_stride;
      }
      break;
    case 2:  // relu6
      for (int i = 0; i < size_h; i++) {
        C8_OUT_PARAM
        asm volatile(INIT_C8 PROCESS_C8 TRANS_C8 RELU_C8 RELU6_C8 STORE_C8
                         PROCESS_C8_REMAIN TRANS_C8 RELU_C8 RELU6_C8
                             STORE_C8_REMAIN ASM_PARAM);
        PTR_ADD
        if (remain) {
          C8_OUT_REMAIN
        }
        din += w_in_stride;
      }
      break;
    case 3:  // leakyrelu
      for (int i = 0; i < size_h; i++) {
        C8_OUT_PARAM
        asm volatile(
            INIT_C8 PROCESS_C8 TRANS_C8 LEAKYRELU_C8 STORE_C8 PROCESS_C8_REMAIN
                TRANS_C8 LEAKYRELU_C8 STORE_C8_REMAIN ASM_PARAM);
        PTR_ADD
        if (remain) {
          C8_OUT_REMAIN
        }
        din += w_in_stride;
      }
      break;
    case 4:  // hardswish
      for (int i = 0; i < size_h; i++) {
        C8_OUT_PARAM
        asm volatile(
            INIT_C8 PROCESS_C8 TRANS_C8 HARD_SWISH_C8 STORE_C8 PROCESS_C8_REMAIN
                TRANS_C8 HARD_SWISH_C8 STORE_C8_REMAIN ASM_PARAM_1);
        PTR_ADD
        if (remain) {
          C8_OUT_REMAIN
        }
        din += w_in_stride;
      }
      break;
  }
}

inline void prepack_input_nxwc8_fp16_dw(const float16_t* din,
                                        float16_t* dout,
                                        int cs,
                                        int hs,
                                        int he,
                                        int ws,
                                        int we,
                                        int channel,
                                        int width,
                                        int height,
                                        float16_t* zero_ptr) {
  int n = he - hs;
  if (n <= 0) {
    LOG(FATAL) << "prepack_dw_input_int8, valid height must > zero";
  }
  int size_w = we - ws;
  int w0 = ws < 0 ? 0 : ws;
  int w1 = we > width ? width : we;
  int valid_w = w1 - w0;
  int pad_l = ws < 0 ? -ws : 0;
  int pad_r = we > width ? we - width : 0;
  int size_c = width * height;

  int valid_cnt = valid_w >> 3;
  int remain = valid_w & 7;
  int stride = size_w << 3;
  int pad_l_stride = pad_l << 3;
  int pad_r_stride = pad_r << 3;
  for (int h = hs; h < he; ++h) {
    const float16_t* ptr_c0 = din + h * width + cs * size_c;
    const float16_t* ptr_c1 = ptr_c0 + size_c;
    const float16_t* ptr_c2 = ptr_c1 + size_c;
    const float16_t* ptr_c3 = ptr_c2 + size_c;
    const float16_t* ptr_c4 = ptr_c3 + size_c;
    const float16_t* ptr_c5 = ptr_c4 + size_c;
    const float16_t* ptr_c6 = ptr_c5 + size_c;
    const float16_t* ptr_c7 = ptr_c6 + size_c;
    if (h < 0 || h >= height) {
      memset(dout, 0.f, stride * sizeof(float16_t));
      dout += stride;
      continue;
    } else if (cs + 8 > channel) {
      switch (cs + 8 - channel) {
        case 7:
          ptr_c1 = zero_ptr;
        case 6:
          ptr_c2 = zero_ptr;
        case 5:
          ptr_c3 = zero_ptr;
        case 4:
          ptr_c4 = zero_ptr;
        case 3:
          ptr_c5 = zero_ptr;
        case 2:
          ptr_c6 = zero_ptr;
        case 1:
          ptr_c7 = zero_ptr;
        default:
          break;
      }
    }
    if (pad_l) {
      memset(dout, 0.f, pad_l_stride * sizeof(float16_t));
      dout += pad_l_stride;
    }
    if (valid_cnt) {
      int cnt = valid_cnt;
#ifdef __aarch64__
      asm volatile(
          /* main loop */
          "1:\n"
          "ldr q0,    [%[r0]], #16\n"
          "ldr q1,    [%[r1]], #16\n"
          "ldr q2,    [%[r2]], #16\n"
          "ldr q3,    [%[r3]], #16\n"
          "ldr q4,    [%[r4]], #16\n"
          "ldr q5,    [%[r5]], #16\n"
          "ldr q6,    [%[r6]], #16\n"
          "ldr q7,    [%[r7]], #16\n"
          "trn1 v8.8h,  v0.8h, v1.8h\n"
          "trn2 v9.8h,  v0.8h, v1.8h\n"
          "trn1 v10.8h, v2.8h, v3.8h\n"
          "trn2 v11.8h, v2.8h, v3.8h\n"
          "trn1 v12.8h, v4.8h, v5.8h\n"
          "trn2 v13.8h, v4.8h, v5.8h\n"
          "trn1 v14.8h, v6.8h, v7.8h\n"
          "trn2 v15.8h, v6.8h, v7.8h\n"
          "trn1 v0.4s,  v8.4s, v10.4s\n"
          "trn2 v1.4s,  v8.4s, v10.4s\n"
          "trn1 v2.4s,  v9.4s, v11.4s\n"
          "trn2 v3.4s,  v9.4s, v11.4s\n"
          "trn1 v4.4s,  v12.4s, v14.4s\n"
          "trn2 v5.4s,  v12.4s, v14.4s\n"
          "trn1 v6.4s,  v13.4s, v15.4s\n"
          "trn2 v7.4s,  v13.4s, v15.4s\n"
          "trn1 v8.2d,  v0.2d, v4.2d\n"
          "trn1 v9.2d,  v2.2d, v6.2d\n"
          "trn1 v10.2d, v1.2d, v5.2d\n"
          "trn1 v11.2d, v3.2d, v7.2d\n"
          "trn2 v12.2d, v0.2d, v4.2d\n"
          "str q8, [%[ptr_out]], #16\n"
          "trn2 v13.2d, v2.2d, v6.2d\n"
          "str q9, [%[ptr_out]], #16\n"
          "trn2 v14.2d, v1.2d, v5.2d\n"
          "str q10, [%[ptr_out]], #16\n"
          "subs %w[cnt], %w[cnt], #1\n"
          "trn2 v15.2d, v3.2d, v7.2d\n"
          "str q11, [%[ptr_out]], #16\n"
          "stp q12, q13, [%[ptr_out]], #32\n"
          "stp q14, q15, [%[ptr_out]], #32\n"
          "bne    1b\n"
          : [cnt] "+r"(cnt),
            [r0] "+r"(ptr_c0),
            [r1] "+r"(ptr_c1),
            [r2] "+r"(ptr_c2),
            [r3] "+r"(ptr_c3),
            [r4] "+r"(ptr_c4),
            [r5] "+r"(ptr_c5),
            [r6] "+r"(ptr_c6),
            [r7] "+r"(ptr_c7),
            [ptr_out] "+r"(dout)
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
      asm volatile(
          /* main loop */
          "1:\n"
          "vld1.32 {d0-d1},  [%[r0]]!\n"
          "vld1.32 {d2-d3},  [%[r1]]!\n"
          "vld1.32 {d4-d5},  [%[r2]]!\n"
          "vld1.32 {d6-d7},  [%[r3]]!\n"
          "vld1.32 {d8-d9},  [%[r4]]!\n"
          "vld1.32 {d10-d11},  [%[r5]]!\n"
          "vld1.32 {d12-d13},  [%[r6]]!\n"
          "vld1.32 {d14-d15},  [%[r7]]!\n"
          "vtrn.16   q0, q1\n"
          "vtrn.16   q2, q3\n"
          "vtrn.16   q4, q5\n"
          "vtrn.16   q6, q7\n"
          //
          "vtrn.32  q0, q2\n"
          "vtrn.32  q1, q3\n"
          "vtrn.32  q4, q6\n"
          "vtrn.32  q5, q7\n"

          "vswp    d1, d8\n"
          "vswp    d3, d10\n"
          "vswp    d5, d12\n"
          "vswp    d7, d14\n"

          "subs %[cnt], #1\n"
          "vst1.16 {d0-d3}, [%[ptr_out]]!\n"
          "vst1.16 {d4-d7}, [%[ptr_out]]!\n"
          "vst1.16 {d8-d11}, [%[ptr_out]]!\n"
          "vst1.16 {d12-d15}, [%[ptr_out]]!\n"
          "bne    1b\n"
          : [cnt] "+r"(cnt),
            [r0] "+r"(ptr_c0),
            [r1] "+r"(ptr_c1),
            [r2] "+r"(ptr_c2),
            [r3] "+r"(ptr_c3),
            [r4] "+r"(ptr_c4),
            [r5] "+r"(ptr_c5),
            [r6] "+r"(ptr_c6),
            [r7] "+r"(ptr_c7),
            [ptr_out] "+r"(dout)
          :
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
#endif  // __aarch64__
    }
    for (int i = 0; i < remain; ++i) {
      dout[0] = *(ptr_c0++);
      dout[1] = *(ptr_c1++);
      dout[2] = *(ptr_c2++);
      dout[3] = *(ptr_c3++);
      dout[4] = *(ptr_c4++);
      dout[5] = *(ptr_c5++);
      dout[6] = *(ptr_c6++);
      dout[7] = *(ptr_c7++);
      dout += 8;
    }
    if (pad_r) {
      memset(dout, 0.f, pad_r_stride * sizeof(float16_t));
      dout += pad_r_stride;
    }
  }
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
