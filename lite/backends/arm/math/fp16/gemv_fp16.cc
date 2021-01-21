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

#include "lite/backends/arm/math/fp16/gemv_fp16.h"
#include <arm_neon.h>
namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
// clang-format off
#ifdef __aarch64__
#define GEMV_INIT                       \
  "prfm  pldl1keep, [%[in]]     \n"     \
  "prfm  pldl1keep, [%[ptr_w0]] \n"     \
  "prfm  pldl1keep, [%[ptr_w1]] \n"     \
  "prfm  pldl1keep, [%[ptr_w2]] \n"     \
  "prfm  pldl1keep, [%[ptr_w3]] \n"     \
  "dup v9.8h, %[vbias].h[0]     \n"     \
  "prfm  pldl1keep, [%[ptr_w4]] \n"     \
  "dup v10.8h, %[vbias].h[1]    \n"     \
  "prfm  pldl1keep, [%[ptr_w5]] \n"     \
  "dup v11.8h, %[vbias].h[2]    \n"     \
  "prfm  pldl1keep, [%[ptr_w6]] \n"     \
  "dup v12.8h, %[vbias].h[3]    \n"     \
  "prfm  pldl1keep, [%[ptr_w7]] \n"     \
  "dup v13.8h, %[vbias].h[4]    \n"     \
  "cmp %w[cnt], #1              \n"     \
  "dup v14.8h, %[vbias].h[5]    \n"     \
  "dup v15.8h, %[vbias].h[6]    \n"     \
  "dup v16.8h, %[vbias].h[7]    \n"     \
  "blt 1f                       \n"     \
  "0:                           \n"     \
  "ld1 {v0.8h}, [%[ptr_in]], #16\n"     \
  "ld1 {v1.8h}, [%[ptr_w0]], #16\n"     \
  "ld1 {v2.8h}, [%[ptr_w1]], #16\n"     \
  "ld1 {v3.8h}, [%[ptr_w2]], #16\n"     \
  "ld1 {v4.8h}, [%[ptr_w3]], #16\n"     \
  "ld1 {v5.8h}, [%[ptr_w4]], #16\n"

#define GEMV_TRANS_COMPUTE              \
  "fmla v9.8h, v1.8h, v0.h[0]   \n"     \
  "ld1 {v6.8h}, [%[ptr_w5]], #16\n"     \
  "fmla v10.8h, v2.8h, v0.h[1]  \n"     \
  "ld1 {v7.8h}, [%[ptr_w6]], #16\n"     \
  "fmla v11.8h, v3.8h, v0.h[2]  \n"     \
  "ld1 {v8.8h}, [%[ptr_w7]], #16\n"     \
  "fmla v12.8h, v4.8h, v0.h[3]  \n"     \
  "add %[ptr_w0], %[ptr_w0], %[stride]\n"\
  "fmla v13.8h, v5.8h, v0.h[4]  \n"     \
  "add %[ptr_w1], %[ptr_w1], %[stride]\n"\
  "fmla v14.8h, v6.8h, v0.h[5]  \n"     \
  "add %[ptr_w2], %[ptr_w2], %[stride]\n"\
  "fmla v15.8h, v7.8h, v0.h[6]  \n"     \
  "add %[ptr_w3], %[ptr_w3], %[stride]\n"\
  "fmla v16.8h, v8.8h, v0.h[7]  \n"     \
  "add %[ptr_w4], %[ptr_w4], %[stride]\n"\
  "subs %w[cnt], %w[cnt], #1    \n"     \
  "add %[ptr_w5], %[ptr_w5], %[stride]\n"\
  "add %[ptr_w6], %[ptr_w6], %[stride]\n"\
  "add %[ptr_w7], %[ptr_w7], %[stride]\n"\
  "bne 0b                       \n"

#define GEMV_COMPUTE                     \
  "fmla v9.8h, v1.8h, v0.8h     \n"      \
  "ld1 {v6.8h}, [%[ptr_w5]], #16\n"      \
  "fmla v10.8h, v2.8h, v0.8h    \n"      \
  "ld1 {v7.8h}, [%[ptr_w6]], #16\n"      \
  "fmla v11.8h, v3.8h, v0.8h    \n"      \
  "ld1 {v8.8h}, [%[ptr_w7]], #16\n"      \
  "fmla v12.8h, v4.8h, v0.8h    \n"      \
  "ld1 {v17.8h}, [%[ptr_in]], #16\n"     \
  "fmla v13.8h, v5.8h, v0.8h    \n"      \
  "ld1 {v1.8h}, [%[ptr_w0]], #16\n"      \
  "fmla v14.8h, v6.8h, v0.8h    \n"      \
  "ld1 {v2.8h}, [%[ptr_w1]], #16\n"      \
  "fmla v15.8h, v7.8h, v0.8h    \n"      \
  "ld1 {v3.8h}, [%[ptr_w2]], #16\n"      \
  "fmla v16.8h, v8.8h, v0.8h    \n"      \
  "ld1 {v4.8h}, [%[ptr_w3]], #16\n"      \
  "fmla v9.8h, v1.8h, v17.8h    \n"      \
  "ld1 {v5.8h}, [%[ptr_w4]], #16\n"      \
  "fmla v10.8h, v2.8h, v17.8h   \n"      \
  "ld1 {v6.8h}, [%[ptr_w5]], #16\n"      \
  "fmla v11.8h, v3.8h, v17.8h   \n"      \
  "ld1 {v7.8h}, [%[ptr_w6]], #16\n"      \
  "fmla v12.8h, v4.8h, v17.8h   \n"      \
  "ld1 {v8.8h}, [%[ptr_w7]], #16\n"      \
  "fmla v13.8h, v5.8h, v17.8h   \n"      \
  "subs %w[cnt], %w[cnt], #1    \n"      \
  "fmla v14.8h, v6.8h, v17.8h   \n"      \
  "fmla v15.8h, v7.8h, v17.8h   \n"      \
  "fmla v16.8h, v8.8h, v17.8h   \n"      \
  "bne 0b                       \n"

#define STORE                            \
  "1:                           \n"      \
  "cmp %w[flag_act], #0         \n"      \
  "faddp v0.8h, v9.8h, v10.8h   \n"      \
  "faddp v1.8h, v11.8h, v12.8h  \n"      \
  "faddp v2.8h, v13.8h, v14.8h  \n"      \
  "faddp v3.8h, v15.8h, v16.8h  \n"      \
  "beq 2f                       \n"      \
  "cmp %w[flag_act], #1         \n"      \
  "faddp v4.8h, v0.8h, v1.8h    \n"      \
  "faddp v5.8h, v2.8h, v3.8h    \n"      \
  "faddp v6.8h, v4.8h, v5.8h    \n"      \
  "beq 3f                       \n"      \
  "cmp %w[flag_act], #2         \n"      \
  "beq 4f                       \n"      \
  /* leakyrelu */                        \
  "fmul v7.8h, v6.8h, %[valpha].8h\n"    \
  "fcmge v8.8h, v6.8h, %[vzero].8h\n"    \
  "bif  v6.16b, v7.16b, v8.16b  \n"      \
  "b 5f                         \n"      \
  /* relu6 */                            \
  "4:                           \n"      \
  "fmax v6.8h, v6.8h, %[vzero].8h\n"     \
  "fmin v6.8h, v6.8h, %[valpha].8h\n"    \
  "b 5f                         \n"      \
  /* relu */                             \
  "3:                           \n"      \
  "fmax v6.8h, v6.8h, %[vzero].8h\n"     \
  "b 5f                         \n"      \
  /* no act */                           \
  "2:                           \n"      \
  "faddp v4.8h, v0.8h, v1.8h    \n"      \
  "faddp v5.8h, v2.8h, v3.8h    \n"      \
  "faddp v6.8h, v4.8h, v5.8h    \n"      \
  /* store */                            \
  "5:                           \n"      \
  "st1 {v6.8h}, [%[outptr]], #16\n"

#define GEMV_ASM_PARAMS                  \
      [ptr_in] "+r"(ptr_in),             \
      [ptr_w0] "+r"(ptr_w0),             \
      [ptr_w1] "+r"(ptr_w1),             \
      [ptr_w2] "+r"(ptr_w2),             \
      [ptr_w3] "+r"(ptr_w3),             \
      [ptr_w4] "+r"(ptr_w4),             \
      [ptr_w5] "+r"(ptr_w5),             \
      [ptr_w6] "+r"(ptr_w6),             \
      [ptr_w7] "+r"(ptr_w7),             \
      [cnt] "+"(cnt_col)                 \
    : [vbias] "w"(vbias),                \
      [vzero] "w"(vzero),                \
      [valpha] "w"(valpha),              \
      [flag_act] "r"(flag_act),          \
      [stride] "r"(stride)               \
    : "cc", "memory", "v0", "v1", "v2",  \
      "v3", "v4", "v5", "v6", "v7",      \
      "v8", "v9", "v10", "v11", "v12",   \
      "v13", "v14", "v15", "v16", "v17"

#endif
// clang-format on
#define PTR_ACQUIRE_PARAM(dtype)                                     \
  const dtype *ptr_zero, const dtype *ptr_w0, const dtype *ptr_w1,   \
      const dtype *ptr_w2, const dtype *ptr_w3, const dtype *ptr_w4, \
      const dtype *ptr_w5, const dtype *ptr_w6, const dtype *ptr_w7, \
      const dtype *ptr_w, dtype *out_p, dtype *out_tmp, int remain

inline void act_acquire(lite_api::ActivationType act,
                        int &flag_act,      // NOLINT
                        float &local_alpha  // NOLINT
                        ) {
  if (act == lite_api::ActivationType::kRelu) {
    flag_act = 0x01;
  } else if (act == lite_api::ActivationType::kRelu6) {
    flag_act = 0x02;
    local_alpha = six;
  } else if (act == lite_api::ActivationType::kLeakyRelu) {
    flag_act = 0x03;
    local_alpha = alpha;
  }
}

inline void ptr_acquire(PTR_ACQUIRE_PARAM(float16_t)) {
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
}

bool gemv_fp16_trans(const float16_t *A,
                     const float16_t *x,
                     float16_t *y,
                     int M,
                     int N,
                     float16_t beta,
                     bool is_bias,
                     const float16_t *bias,
                     bool flag_act,
                     lite_api::ActivationType act,
                     const ARMContext *ctx,
                     float16_t six,
                     float16_t alpha) {
  int Nup = (N + 7) / 8 * 8;
  float16_t *ptr_zero = ctx->workspace_data<float16_t>();
  memset(ptr_zero, 0, Nup * sizeof(float16_t));
  float16_t *ptr_w = ptr_zero + Nup;
  lite::TargetWrapperHost::MemcpySync(ptr_w, A, N);
  float16_t *data_in = ptr_w + Nup;
  lite::TargetWrapperHost::MemcpySync(data_in, x + (M - 1) * N, N);
  int cnt = Nup >> 3;
  float local_alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (flag_act) {
    act_acquire(act, &flag_act, &local_alpha);
  }

#ifdef __aarch64__
  int out_cnt = M >> 3;
  int remain = M & 7;
  if (remain > 0) out_cnt++;
  float16x8_t vzero = vdupq_n_f16(0.f);
  float16x8_t valpha = vdupq_n_f16(local_alpha);
  int stride = 16 * (M - 1);  // (8 * M - 8) * 2
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 8;
    float16_t out_temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float16_t *ptr_out = y + out_idx;
    const float16_t *ptr_in = ptr_w;
    const float16_t *ptr_w0 = x + out_idx;
    const float16_t *ptr_w1 = ptr_w0 + M;
    const float16_t *ptr_w2 = ptr_w1 + M;
    const float16_t *ptr_w3 = ptr_w2 + M;
    const float16_t *ptr_w4 = ptr_w3 + M;
    const float16_t *ptr_w5 = ptr_w4 + M;
    const float16_t *ptr_w6 = ptr_w5 + M;
    const float16_t *ptr_w7 = ptr_w6 + M;
    float16x8_t vbias = is_bias ? vld1q_f16(bias + out_idx) : vdupq_n_f16(0.f);
    if (j == out_cnt - 1 && remain) {
      ptr_acquire(ptr_zero,
                  ptr_w0,
                  ptr_w1,
                  ptr_w2,
                  ptr_w3,
                  ptr_w4,
                  ptr_w5,
                  ptr_w6,
                  ptr_w7,
                  data_in,
                  out_p,
                  out_temp,
                  remain);
    }
    // 8x8
    int cnt_col = cnt;
    asm volatile(GEMV_INIT GEMV_TRANS_COMPUTE STORE : GEMV_ASM_PARAMS);
    if (remain > 0) {
      for (int i = 0; i < remain; i++) {
        out_ptr[i] = out_p[i];
      }
    }
  }
#else
#endif
}

bool gemv_fp16(const float16_t *A,
               const float16_t *x,
               float16_t *y,
               bool transA,
               int M,
               int N,
               float16_t beta,
               bool is_bias,
               const float16_t *bias,
               bool flag_act,
               lite_api::ActivationType act,
               const ARMContext *ctx,
               float16_t six,
               float16_t alpha) {
  if (transA) {
    // 8x16
    gemv_fp16_trans(
        A, x, y, M, N, beta, is_bias, bias, flag_act, act, six, alpha);
    return;
  }
  int Nup = (N + 15) / 16 * 16;
  float16_t *ptr_zero = ctx->workspace_data<float16_t>();
  memset(ptr_zero, 0, Nup * sizeof(float16_t));
  float16_t *data_in = ptr_zero + Nup;
  lite::TargetWrapperHost::MemcpySync(data_in, x, N);
  float16_t *ptr_w = data_in + Nup;
  lite::TargetWrapperHost::MemcpySync(ptr_w, A + (M - 1) * N, N);
  int cnt = Nup >> 4;
  float local_alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (flag_act) {
    if (act == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      local_alpha = six;
    } else if (act == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      local_alpha = alpha;
    }
  }

#ifdef __aarch64__
  int out_cnt = M >> 3;
  int remain = M & 7;
  if (remain > 0) out_cnt++;
  float16x8_t vzero = vdupq_n_f16(0.f);
  float16x8_t valpha = vdupq_n_f16(local_alpha);
  int stride = 1;
#pragma omp parallel for
  for (int j = 0; j < out_cnt; j++) {
    int out_idx = j * 8;
    float16_t out_temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float16_t *ptr_out = y + out_idx;
    const float16_t *ptr_in = data_in;
    const float16_t *ptr_w0 = A + (N * out_idx);
    const float16_t *ptr_w1 = ptr_w0 + N;
    const float16_t *ptr_w2 = ptr_w1 + N;
    const float16_t *ptr_w3 = ptr_w2 + N;
    const float16_t *ptr_w4 = ptr_w3 + N;
    const float16_t *ptr_w5 = ptr_w4 + N;
    const float16_t *ptr_w6 = ptr_w5 + N;
    const float16_t *ptr_w7 = ptr_w6 + N;
    float16x8_t vbias = is_bias ? vld1q_f16(bias + out_idx) : vdupq_n_f16(0.f);
    if (j == out_cnt - 1 && remain) {
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
    }
    // 8x16
    int cnt_col = cnt;
    asm volatile(GEMV_INIT GEMV_TRANS_COMPUTE STORE : GEMV_ASM_PARAMS);
    if (remain > 0) {
      for (int i = 0; i < remain; i++) {
        out_ptr[i] = out_p[i];
      }
    }
  }
#else
#endif
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
