// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/backends/arm/math/packed_sgemm.h"
#include "lite/core/context.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
#ifdef __aarch64__
// loadb: 6x8 b=6
void loadb_6x8(
    float *out, const float *in, int ldin, int k0, int kmax, int n0, int nmax) {
  auto outptr = reinterpret_cast<uint32_t *>(out);
  auto inptr = reinterpret_cast<const uint32_t *>(in) + k0 * ldin + n0;
  uint32_t mask_buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int x_len = nmax - n0;
  int y_len = kmax - k0;
  int right_remain = x_len - 6 * (x_len / 6);
  int right_pad = 6 - right_remain;

  uint32_t *outptr_row = outptr;
  int stride_out = 6 * y_len;
  int cnt_y = 4 * (y_len / 4);

  uint32x4_t vzero = vdupq_n_u32(0);
  uint32x4_t vmask1 =
      vcltq_u32(vld1q_u32(mask_buffer), vdupq_n_u32(right_remain));
  uint32x4_t vmask2 =
      vcltq_u32(vld1q_u32(mask_buffer + 4), vdupq_n_u32(right_remain));

  LITE_PARALLEL_COMMON_BEGIN(y, tid, y_len - 3, 0, 4) {
    const uint32_t *ptr0 = inptr + y * ldin;
    const uint32_t *ptr1 = ptr0 + ldin;
    const uint32_t *ptr2 = ptr1 + ldin;
    const uint32_t *ptr3 = ptr2 + ldin;
    uint32_t *outptr_row_col = outptr_row + y * 6;
    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]                \n"
        "prfm   pldl1keep, [%[ptr0], #64]   \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr1], #64]   \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr2], #64]   \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        "prfm   pldl1keep, [%[ptr3], #64]   \n"
        :
        : [ptr0] "r"(ptr0), [ptr1] "r"(ptr1), [ptr2] "r"(ptr2), [ptr3] "r"(ptr3)
        : "memory");
    int i = 0;
    for (; i < x_len - 5; i += 6) {
      uint32_t *ptr_out = outptr_row_col;
      asm volatile(
          "ldr q0, [%[ptr0]], #16\n"
          "ld1 {v1.2s}, [%[ptr0]], #8\n"
          "ldr q2, [%[ptr1]], #16\n"
          "ld1 {v3.2s}, [%[ptr1]], #8\n"
          "ldr q4, [%[ptr2]], #16\n"
          "str q0, [%[outptr]], #16\n"
          "ld1 {v5.2s}, [%[ptr2]], #8\n"
          "st1 {v1.2s}, [%[outptr]], #8\n"
          "ldr q6, [%[ptr3]], #16\n"
          "str q2, [%[outptr]], #16\n"
          "ld1 {v7.2s}, [%[ptr3]], #8\n"
          "st1 {v3.2s}, [%[outptr]], #8\n"
          "str q4, [%[outptr]], #16\n"
          "st1 {v5.2s}, [%[outptr]], #8\n"
          "str q6, [%[outptr]], #16\n"
          "st1 {v7.2s}, [%[outptr]], #8\n"
          : [outptr] "+r"(ptr_out),
            [ptr0] "+r"(ptr0),
            [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2),
            [ptr3] "+r"(ptr3)
          :
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "cc", "memory");
      outptr_row_col += stride_out;
    }
    if (right_remain > 0) {
      uint32_t *ptr_out = outptr_row_col;
      asm volatile(
          "ldp q0, q1, [%[ptr0]], #32\n"
          "ldp q2, q3, [%[ptr1]], #32\n"
          "bif v0.16b, %[vzero].16b, %[vmask1].16b\n"
          "bif v1.16b, %[vzero].16b, %[vmask2].16b\n"
          "bif v2.16b, %[vzero].16b, %[vmask1].16b\n"
          "bif v3.16b, %[vzero].16b, %[vmask2].16b\n"
          "str q0, [%[outptr]], #16\n"
          "ldp q4, q5, [%[ptr2]], #32\n"
          "st1 {v1.2s}, [%[outptr]], #8\n"
          "ldp q6, q7, [%[ptr3]], #32\n"
          "str q2, [%[outptr]], #16\n"
          "bif v4.16b, %[vzero].16b, %[vmask1].16b\n"
          "bif v5.16b, %[vzero].16b, %[vmask2].16b\n"
          "st1 {v3.2s}, [%[outptr]], #8\n"
          "bif v6.16b, %[vzero].16b, %[vmask1].16b\n"
          "bif v7.16b, %[vzero].16b, %[vmask2].16b\n"
          "str q4, [%[outptr]], #16\n"
          "st1 {v5.2s}, [%[outptr]], #8\n"
          "str q6, [%[outptr]], #16\n"
          "st1 {v7.2s}, [%[outptr]], #8\n"
          : [outptr] "+r"(ptr_out),
            [ptr0] "+r"(ptr0),
            [ptr1] "+r"(ptr1),
            [ptr2] "+r"(ptr2),
            [ptr3] "+r"(ptr3)
          : [vmask1] "w"(vmask1), [vmask2] "w"(vmask2), [vzero] "w"(vzero)
          : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "cc", "memory");
    }
  }
  LITE_PARALLEL_COMMON_END()
  LITE_PARALLEL_COMMON_BEGIN(y, tid, y_len, cnt_y, 1) {
    const uint32_t *ptr0 = inptr + y * ldin;
    uint32_t *outptr_row_col = outptr_row + y * 6;
    int i = 0;
    for (; i < x_len - 5; i += 6) {
      uint32_t *ptr_out = outptr_row_col;
      asm volatile(
          "ldr q0, [%[ptr0]], #16\n"
          "ld1 {v1.2s}, [%[ptr0]], #8\n"
          "str q0, [%[outptr]], #16\n"
          "st1 {v1.2s}, [%[outptr]], #8\n"
          : [ptr0] "+r"(ptr0), [outptr] "+r"(ptr_out)
          :
          : "v0", "v1", "cc", "memory");
      outptr_row_col += stride_out;
    }
    if (right_remain > 0) {
      uint32_t *ptr_out = outptr_row_col;
      asm volatile(
          "ldp q0, q1, [%[ptr0]], #32\n"
          "bif v0.16b, %[vzero].16b, %[vmask1].16b\n"
          "bif v1.16b, %[vzero].16b, %[vmask2].16b\n"
          "str q0, [%[outptr]], #16\n"
          "st1 {v1.2s}, [%[outptr]], #8\n"
          : [ptr0] "+r"(ptr0), [outptr] "+r"(ptr_out)
          : [vmask1] "w"(vmask1), [vmask2] "w"(vmask2), [vzero] "w"(vzero)
          : "v0", "v1", "cc", "memory");
    }
  }
  LITE_PARALLEL_COMMON_END()
}
// loadb_trans: 8x6 b=6
void loadb_trans_6x8(
    float *out, const float *in, int ldin, int k0, int kmax, int n0, int nmax) {
  int x_len = kmax - k0;
  uint32_t zerobuff[x_len];  // NOLINT
  memset(zerobuff, 0, sizeof(uint32_t) * x_len);

  auto outptr = reinterpret_cast<uint32_t *>(out);
  auto inptr = reinterpret_cast<const uint32_t *>(in);
  //! data B is not transposed, transpose B to k * 6
  for (int y = n0; y < nmax; y += 6) {
    const uint32_t *inptr0 = inptr + y * ldin + k0;
    const uint32_t *inptr1 = inptr0 + ldin;
    const uint32_t *inptr2 = inptr1 + ldin;
    const uint32_t *inptr3 = inptr2 + ldin;
    const uint32_t *inptr4 = inptr3 + ldin;
    const uint32_t *inptr5 = inptr4 + ldin;

    int x = x_len;
    asm volatile(
        "prfm   pldl1keep, [%[ptr0]]        \n"
        "prfm   pldl1keep, [%[ptr0], #64]   \n"
        "prfm   pldl1keep, [%[ptr1]]        \n"
        "prfm   pldl1keep, [%[ptr1], #64]   \n"
        "prfm   pldl1keep, [%[ptr2]]        \n"
        "prfm   pldl1keep, [%[ptr2], #64]   \n"
        "prfm   pldl1keep, [%[ptr3]]        \n"
        "prfm   pldl1keep, [%[ptr3], #64]   \n"
        "prfm   pldl1keep, [%[ptr4]]        \n"
        "prfm   pldl1keep, [%[ptr4], #64]   \n"
        "prfm   pldl1keep, [%[ptr5]]        \n"
        "prfm   pldl1keep, [%[ptr5], #64]   \n"
        :
        : [ptr0] "r"(inptr0),
          [ptr1] "r"(inptr1),
          [ptr2] "r"(inptr2),
          [ptr3] "r"(inptr3),
          [ptr4] "r"(inptr4),
          [ptr5] "r"(inptr5)
        : "memory");

    //! cope with row index exceed real size, set to zero buffer
    if ((y + 5) >= nmax) {
      switch ((y + 5) - nmax) {
        case 4:
          inptr1 = zerobuff;
        case 3:
          inptr2 = zerobuff;
        case 2:
          inptr3 = zerobuff;
        case 1:
          inptr4 = zerobuff;
        case 0:
          inptr5 = zerobuff;
        default:
          break;
      }
    }

    for (; x > 3; x -= 4) {
      // clang-format off
      //! zip load 8 elements (2 neon Q registers) from each of 8 rows
      asm volatile(
          "ldr q0, [%[inptr0]], #16\n"
          "ldr q1, [%[inptr1]], #16\n"
          "ldr q2, [%[inptr2]], #16\n"
          "ldr q3, [%[inptr3]], #16\n"
          "ldr q4, [%[inptr4]], #16\n" // e0e1e2e3
          "ldr q5, [%[inptr5]], #16\n" // f0f1f2f3

          "trn1 v8.4s, v0.4s, v1.4s\n" // a0b0a2b2
          "trn2 v9.4s, v0.4s, v1.4s\n" // a1b1a3b3
          "trn1 v10.4s, v2.4s, v3.4s\n"// c0d0c2d2
          "trn2 v11.4s, v2.4s, v3.4s\n"// c1d1c3d3

          "trn1 v6.4s, v4.4s, v5.4s\n" // e0f0e2f2
          "trn2 v7.4s, v4.4s, v5.4s\n" // e1f1e3f3
         
          "trn1 v0.2d, v8.2d, v10.2d\n" // a0b0c0d0
          "trn1 v1.2d, v9.2d, v11.2d\n" // a1b1c1d1
          "trn2 v2.2d, v8.2d, v10.2d\n" // a2b2c2d2
          "trn2 v3.2d, v9.2d, v11.2d\n" // a3b3c3d3

          "trn1 v4.2d, v6.2d, v7.2d\n" // e0f0e1f1
          "trn2 v5.2d, v6.2d, v7.2d\n" // e2f2e3f3

          "str q0, [%[outptr]], #16\n" // save q0, a0~d0
          "trn1 v6.2d, v4.2d, v1.2d\n" // e0f0a1b1
          "st1 {v4.2s}, [%[outptr]], #8\n"
          "trn2 v7.2d, v4.2d, v1.2d\n" // e1f1c1d1
          "str q1, [%[outptr]], #16\n" // save q0, a1~d1
          "trn1 v8.2d, v5.2d, v3.2d\n" // e2f2a3b3
          "trn2 v9.2d, v5.2d, v3.2d\n" // e3f3c3d3
          "st1 {v7.2s}, [%[outptr]], #8\n"
          "str q2, [%[outptr]], #16\n" // save q0, a2~d2
          "st1 {v5.2s}, [%[outptr]], #8\n"
          "str q3, [%[outptr]], #16\n" // save q0, a3~d3
          "st1 {v9.2s}, [%[outptr]], #8\n"
          : [inptr0] "+r"(inptr0),
            [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3),
            [inptr4] "+r"(inptr4),
            [inptr5] "+r"(inptr5),
            [outptr] "+r"(outptr)
          :
          : "v0","v1","v2","v3","v4","v5",
            "v6","v7","v8","v9","v10","v11","cc","memory");
      // clang-format on
    }

    for (; x > 0; x--) {
      *outptr++ = *inptr0++;
      *outptr++ = *inptr1++;
      *outptr++ = *inptr2++;
      *outptr++ = *inptr3++;
      *outptr++ = *inptr4++;
      *outptr++ = *inptr5++;
    }
  }
}

#define FMLA_N8X6                  \
  "fmla v12.2s, v5.2s, v1.2s\n"    \
  "fmla v14.2s, v4.2s, v2.2s\n"    \
  "fmla v15.2s, v5.2s, v2.2s\n"    \
  "ld1r {v3.2s}, [%[a_ptr]], #4\n" \
  "fmla v10.2s, v6.2s, v0.2s\n"    \
  "fmla v13.2s, v6.2s, v1.2s\n"    \
  "fmla v16.2s, v6.2s, v2.2s\n"    \
  "ld1r {v0.2s}, [%[a_ptr]], #4\n" \
  "fmla v17.2s, v4.2s, v3.2s\n"    \
  "fmla v18.2s, v5.2s, v3.2s\n"    \
  "fmla v19.2s, v6.2s, v3.2s\n"    \
  "ld1r {v1.2s}, [%[a_ptr]], #4\n" \
  "fmla v20.2s, v4.2s, v0.2s\n"    \
  "fmla v21.2s, v5.2s, v0.2s\n"    \
  "fmla v22.2s, v6.2s, v0.2s\n"    \
  "ld1r {v2.2s}, [%[a_ptr]], #4\n" \
  "fmla v23.2s, v4.2s, v1.2s\n"    \
  "fmla v24.2s, v5.2s, v1.2s\n"    \
  "fmla v25.2s, v6.2s, v1.2s\n"    \
  "ld1r {v3.2s}, [%[a_ptr]], #4\n" \
  "fmla v26.2s, v4.2s, v2.2s\n"    \
  "fmla v27.2s, v5.2s, v2.2s\n"    \
  "fmla v28.2s, v6.2s, v2.2s\n"    \
  "ld1r {v0.2s}, [%[a_ptr]], #4\n" \
  "fmla v29.2s, v4.2s, v3.2s\n"    \
  "fmla v30.2s, v5.2s, v3.2s\n"    \
  "fmla v31.2s, v6.2s, v3.2s\n"

void sgemm_prepacked_8x6_a35(bool is_transB,
                             int M,
                             int N,
                             int K,
                             const float *A_packed,
                             const float *B,
                             int ldb,
                             float beta,
                             float *C,
                             int ldc,
                             const float *bias,
                             bool has_bias,
                             const operators::ActivationParam act_param,
                             ARMContext *ctx) {
  size_t l2_cache = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  auto workspace = ctx->workspace_data<float>();
  int threads = ctx->threads();

  auto act_type = act_param.active_type;
  float alpha[4] = {0.f, 0.f, 0.f, 0.f};
  float offset[4] = {0.f, 0.f, 0.f, 0.f};
  float threshold[4] = {0.f, 0.f, 0.f, 0.f};
  float local_alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3, hardswish: 4
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      local_alpha = act_param.Relu_clipped_coef;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      local_alpha = act_param.Leaky_relu_alpha;
    } else if (act_type == lite_api::ActivationType::kHardSwish) {
      flag_act = 0x04;
      local_alpha = 1.0 / act_param.hard_swish_scale;
      for (int i = 0; i < 4; i++) {
        offset[i] = act_param.hard_swish_offset;
        threshold[i] = act_param.hard_swish_threshold;
      }
    }
  }
  alpha[0] = local_alpha;
  alpha[1] = local_alpha;
  alpha[2] = local_alpha;
  alpha[3] = local_alpha;
  // NBLOCK = 6
  int n_block = 6;
  int k_block = 2;
  X_BLOCK_COMPUTE(l2_cache, MBLOCK, n_block, M, N, K)

  // unroll 2 loop
  int tail_pre = (K & (k_block - 1));
  int k_pre = ((K + k_block - 1) / k_block) - 1;

  bool flag_p_remain = false;
  int remain = 0;
  if (tail_pre == 0) {
    tail_pre = k_block;
  }

  int has_beta = fabsf(beta) > 1e-8f ? 1 : 0;
  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + n_block - 1) / n_block;
    remain = xmax - x0 - (bblocks - 1) * n_block;
    if (remain > 0) {
      flag_p_remain = true;
    }
    //! load bpanel
    float *b_pannel = workspace;
    if (is_transB) {
      loadb_trans_6x8(b_pannel, B, ldb, 0, K, x0, xmax);
    } else {
      loadb_6x8(b_pannel, B, ldb, 0, K, x0, xmax);
    }

    LITE_PARALLEL_COMMON_BEGIN(y, tid, M, 0, MBLOCK) {
      unsigned int ymax = y + MBLOCK;
      if (ymax > M) {
        ymax = M;
      }

      float bias_local[8] = {0};
      if (has_bias) {
        int i = 0;
        for (; i < 8 && y + i < ymax; i++) {
          bias_local[i] = bias[y + i];
        }
      }

      float cout0[n_block];  // NOLINT
      float cout1[n_block];  // NOLINT
      float cout2[n_block];  // NOLINT
      float cout3[n_block];  // NOLINT
      float cout4[n_block];  // NOLINT
      float cout5[n_block];  // NOLINT
      float cout6[n_block];  // NOLINT
      float cout7[n_block];  // NOLINT

      float *c_ptr0 = C + y * ldc + x0;
      float *c_ptr1 = c_ptr0 + ldc;
      float *c_ptr2 = c_ptr1 + ldc;
      float *c_ptr3 = c_ptr2 + ldc;
      float *c_ptr4 = c_ptr3 + ldc;
      float *c_ptr5 = c_ptr4 + ldc;
      float *c_ptr6 = c_ptr5 + ldc;
      float *c_ptr7 = c_ptr6 + ldc;

      float *pout0 = c_ptr0;
      float *pout1 = c_ptr1;
      float *pout2 = c_ptr2;
      float *pout3 = c_ptr3;
      float *pout4 = c_ptr4;
      float *pout5 = c_ptr5;
      float *pout6 = c_ptr6;
      float *pout7 = c_ptr7;

      const float *a_ptr_l = A_packed + y * K;
      const float *b_ptr = b_pannel;
      for (int xb = 0; xb < bblocks; xb++) {
        if ((y + 7) >= ymax) {
          switch ((y + 7) - ymax) {
            case 6:
              c_ptr1 = cout1;
            case 5:
              c_ptr2 = cout2;
            case 4:
              c_ptr3 = cout3;
            case 3:
              c_ptr4 = cout4;
            case 2:
              c_ptr5 = cout5;
            case 1:
              c_ptr6 = cout6;
            case 0:
              c_ptr7 = cout7;
            default:
              break;
          }
        }
        if (flag_p_remain && (xb == bblocks - 1)) {
          pout0 = c_ptr0;
          pout1 = c_ptr1;
          pout2 = c_ptr2;
          pout3 = c_ptr3;
          pout4 = c_ptr4;
          pout5 = c_ptr5;
          pout6 = c_ptr6;
          pout7 = c_ptr7;

          c_ptr0 = cout0;
          c_ptr1 = cout1;
          c_ptr2 = cout2;
          c_ptr3 = cout3;
          c_ptr4 = cout4;
          c_ptr5 = cout5;
          c_ptr6 = cout6;
          c_ptr7 = cout7;
        }
        const float *a_ptr = a_ptr_l;
        int tail = tail_pre;
        int k = k_pre;
        // clang-format off
          asm volatile(
            "prfum   pldl1keep, [%[b_ptr]]\n"
            "ldp	q2, q3, [%[bias_ptr]]\n"
            "prfm   pldl1keep, [%[a_ptr]]\n"
            "dup v8.2s, v2.s[0]\n"
            "dup v9.2s, v2.s[0]\n"
            "dup v10.2s, v2.s[0]\n"
            "prfm   pldl1keep, [%[b_ptr], #64]\n"
            "dup v11.2s, v2.s[1]\n"
            "dup v12.2s, v2.s[1]\n"
            "dup v13.2s, v2.s[1]\n"
            "prfum   pldl1keep, [%[a_ptr], #52]\n"
            "dup v14.2s, v2.s[2]\n"
            "dup v15.2s, v2.s[2]\n"
            "dup v16.2s, v2.s[2]\n"
            "prfum   pldl1keep, [%[a_ptr], #116]\n"
            "dup v17.2s, v2.s[3]\n"
            "dup v18.2s, v2.s[3]\n"
            "dup v19.2s, v2.s[3]\n"
            "prfm   pldl1keep, [%[b_ptr], #128]\n"
            "dup v20.2s, v3.s[0]\n"
            "dup v21.2s, v3.s[0]\n"
            "dup v22.2s, v3.s[0]\n"
            "dup v23.2s, v3.s[1]\n"
            "dup v24.2s, v3.s[1]\n"
            "dup v25.2s, v3.s[1]\n"
            "cmp %w[has_beta], #1\n"
            "dup v26.2s, v3.s[2]\n"
            "dup v27.2s, v3.s[2]\n"
            "dup v28.2s, v3.s[2]\n"
            "dup v29.2s, v3.s[3]\n"
            "dup v30.2s, v3.s[3]\n"
            "dup v31.2s, v3.s[3]\n"
            "blt 0f\n" /* check beta == 0? */
            /* process beta */
            "dup v7.2s, %w[beta]\n"
            "ld1 {v0.2s, v1.2s, v2.2s}, [%[c_ptr0]]\n"
            "ld1 {v3.2s, v4.2s, v5.2s}, [%[c_ptr1]]\n"
            "fmla v8.2s, v0.2s, v7.2s\n"
            "fmla v9.2s, v1.2s, v7.2s\n"
            "fmla v10.2s, v2.2s, v7.2s\n"
            "ld1 {v0.2s, v1.2s, v2.2s}, [%[c_ptr2]]\n"
            "fmla v11.2s, v3.2s, v7.2s\n"
            "fmla v12.2s, v4.2s, v7.2s\n"
            "fmla v13.2s, v5.2s, v7.2s\n"
            "ld1 {v3.2s, v4.2s, v5.2s}, [%[c_ptr3]]\n"
            "fmla v14.2s, v0.2s, v7.2s\n"
            "fmla v15.2s, v1.2s, v7.2s\n"
            "fmla v16.2s, v2.2s, v7.2s\n"
            "ld1 {v0.2s, v1.2s, v2.2s}, [%[c_ptr4]]\n"
            "fmla v17.2s, v3.2s, v7.2s\n"
            "fmla v18.2s, v4.2s, v7.2s\n"
            "fmla v19.2s, v5.2s, v7.2s\n"
            "ld1 {v3.2s, v4.2s, v5.2s}, [%[c_ptr5]]\n"
            "fmla v20.2s, v0.2s, v7.2s\n"
            "fmla v21.2s, v1.2s, v7.2s\n"
            "fmla v22.2s, v2.2s, v7.2s\n"
            "ld1 {v0.2s, v1.2s, v2.2s}, [%[c_ptr6]]\n"
            "fmla v23.2s, v3.2s, v7.2s\n"
            "fmla v24.2s, v4.2s, v7.2s\n"
            "fmla v25.2s, v5.2s, v7.2s\n"
            "ld1 {v3.2s, v4.2s, v5.2s}, [%[c_ptr7]]\n"
            "fmla v26.2s, v0.2s, v7.2s\n"
            "fmla v27.2s, v1.2s, v7.2s\n"
            "fmla v28.2s, v2.2s, v7.2s\n"
            "fmla v29.2s, v3.2s, v7.2s\n"
            "fmla v30.2s, v4.2s, v7.2s\n"
            "fmla v31.2s, v5.2s, v7.2s\n"
            "0: \n"
            "cmp %w[k], #1\n"
            "ld1r {v0.2s}, [%[a_ptr]], #4\n"
            "ldr d4, [%[b_ptr], #0]\n"
            "ld1r	{v1.2s}, [%[a_ptr]], #4\n"
            "blt 2f\n"
            /* main loop */
            "1: \n"
            /* unrool 0*/
            "ldr d5, [%[b_ptr], #8]\n"
            "ld1r {v2.2s}, [%[a_ptr]], #4\n"
            "fmla v8.2s, v4.2s, v0.2s\n"
            "fmla v11.2s, v4.2s, v1.2s\n"
            "fmla v9.2s, v5.2s, v0.2s\n"
            
            "ldr d6, [%[b_ptr], #16]\n"
            FMLA_N8X6
            /* unrool 1*/
            "ldr	d4, [%[b_ptr], #24]\n"
            "prfm   pldl1keep, [%[b_ptr], #128]\n"
            "ld1r {v1.2s}, [%[a_ptr]], #4\n" 
            "subs	%w[k], %w[k], #1\n"
            "prfum   pldl1keep, [%[a_ptr], #156]\n"
            
            "ldr d5, [%[b_ptr], #32]\n"
            "ld1r {v2.2s}, [%[a_ptr]], #4\n"
            "fmla v8.2s, v4.2s, v0.2s\n"
            "fmla v11.2s, v4.2s, v1.2s\n"
            "fmla v9.2s, v5.2s, v0.2s\n"
            
            "ldr d6, [%[b_ptr], #40]\n"
            FMLA_N8X6
            "ld1r {v1.2s}, [%[a_ptr]], #4\n" 
            "add	%[b_ptr], %[b_ptr], #48\n"
            "prfum   pldl1keep, [%[a_ptr], #188]\n"
            "ldr d4, [%[b_ptr], #0]\n"
            "bne	1b\n"
            // tail
            "2: \n"
            "cmp %w[tail], #1\n"
            "ld1r {v2.2s}, [%[a_ptr]], #4\n"
            "ldr d5, [%[b_ptr], #8]\n"
            "beq 3f \n"
            // tail=2
            "fmla v8.2s, v4.2s, v0.2s\n"
            "fmla v11.2s, v4.2s, v1.2s\n"
            "fmla v9.2s, v5.2s, v0.2s\n"
            "ldr d6, [%[b_ptr], #16]\n"
            FMLA_N8X6
            /* unrool 1*/
            "ldr	d4, [%[b_ptr], #24]\n"
            "prfm   pldl1keep, [%[b_ptr], #128]\n"
            "ld1r {v1.2s}, [%[a_ptr]], #4\n" 
            "ldr d5, [%[b_ptr], #32]\n"

            "ld1r {v2.2s}, [%[a_ptr]], #4\n"
            "fmla v8.2s, v4.2s, v0.2s\n"
            "fmla v11.2s, v4.2s, v1.2s\n"
            "fmla v9.2s, v5.2s, v0.2s\n"
            "ldr d6, [%[b_ptr], #40]\n"
            FMLA_N8X6
            "add	%[b_ptr], %[b_ptr], #48\n"
            "b 4f\n"
            "3: \n"
            // tail=1
            "fmla v8.2s, v4.2s, v0.2s\n"
            "fmla v11.2s, v4.2s, v1.2s\n"
            "fmla v9.2s, v5.2s, v0.2s\n"
            "ldr d6, [%[b_ptr], #16]\n"
            FMLA_N8X6
            "add	%[b_ptr], %[b_ptr], #24\n"
            "4: \n"
            "cmp %w[flag_act], #0\n" // no act
            "beq 5f\n"
            "cmp %w[flag_act], #1\n" // relu
            "movi v0.2s, #0\n"
            "beq 6f\n"
            "cmp %w[flag_act], #2\n" // relu6
            "ld1    {v1.2s}, [%[alpha]]\n"
            "beq 7f\n"
            "cmp %w[flag_act], #3\n" // relu6
            "ld1    {v2.2s}, [%[offset]]\n"
            "ld1    {v3.2s}, [%[threshold]]\n"
            "beq 8f\n"
            // hard_swish
            "fadd  v4.2s,  v8.2s,  v2.2s\n"
            "fadd  v5.2s,  v9.2s,  v2.2s\n"
            "fmul  v6.2s,  v8.2s,  v1.2s\n"
            "fmul  v7.2s,  v9.2s,  v1.2s\n"
            "fmax  v4.2s,  v4.2s,  v0.2s\n"
            "fmax  v5.2s,  v5.2s,  v0.2s\n"
            "fmin  v4.2s,  v4.2s,  v3.2s\n"
            "fmin  v5.2s,  v5.2s,  v3.2s\n"
            "fmul  v8.2s,  v4.2s,  v6.2s\n"
            "fmul  v9.2s,  v5.2s,  v7.2s\n"
            "fadd  v4.2s,  v10.2s, v2.2s\n"
            "fadd  v5.2s,  v11.2s, v2.2s\n"
            "fmul  v6.2s,  v10.2s, v1.2s\n"
            "fmul  v7.2s,  v11.2s, v1.2s\n"
            "fmax  v4.2s,  v4.2s,  v0.2s\n"
            "fmax  v5.2s,  v5.2s,  v0.2s\n"
            "fmin  v4.2s,  v4.2s,  v3.2s\n"
            "fmin  v5.2s,  v5.2s,  v3.2s\n"
            "fmul  v10.2s, v4.2s,  v6.2s\n"
            "fmul  v11.2s, v5.2s,  v7.2s\n"
            "fadd  v4.2s,  v12.2s,  v2.2s\n"
            "fadd  v5.2s,  v13.2s,  v2.2s\n"
            "fmul  v6.2s,  v12.2s,  v1.2s\n"
            "fmul  v7.2s,  v13.2s,  v1.2s\n"
            "fmax  v4.2s,  v4.2s,  v0.2s\n"
            "fmax  v5.2s,  v5.2s,  v0.2s\n"
            "fmin  v4.2s,  v4.2s,  v3.2s\n"
            "fmin  v5.2s,  v5.2s,  v3.2s\n"
            "fmul  v12.2s, v4.2s,  v6.2s\n"
            "fmul  v13.2s, v5.2s,  v7.2s\n"
            "fadd  v4.2s,  v14.2s,  v2.2s\n"
            "fadd  v5.2s,  v15.2s,  v2.2s\n"
            "fmul  v6.2s,  v14.2s,  v1.2s\n"
            "fmul  v7.2s,  v15.2s,  v1.2s\n"
            "fmax  v4.2s,  v4.2s,  v0.2s\n"
            "fmax  v5.2s,  v5.2s,  v0.2s\n"
            "fmin  v4.2s,  v4.2s,  v3.2s\n"
            "fmin  v5.2s,  v5.2s,  v3.2s\n"
            "fmul  v14.2s, v4.2s,  v6.2s\n"
            "fmul  v15.2s, v5.2s,  v7.2s\n"
            "fadd  v4.2s,  v16.2s,  v2.2s\n"
            "fadd  v5.2s,  v17.2s,  v2.2s\n"
            "fmul  v6.2s,  v16.2s,  v1.2s\n"
            "fmul  v7.2s,  v17.2s,  v1.2s\n"
            "fmax  v4.2s,  v4.2s,  v0.2s\n"
            "fmax  v5.2s,  v5.2s,  v0.2s\n"
            "fmin  v4.2s,  v4.2s,  v3.2s\n"
            "fmin  v5.2s,  v5.2s,  v3.2s\n"
            "fmul  v16.2s, v4.2s,  v6.2s\n"
            "fmul  v17.2s, v5.2s,  v7.2s\n"
            "fadd  v4.2s,  v18.2s,  v2.2s\n"
            "fadd  v5.2s,  v19.2s,  v2.2s\n"
            "fmul  v6.2s,  v18.2s,  v1.2s\n"
            "fmul  v7.2s,  v19.2s,  v1.2s\n"
            "fmax  v4.2s,  v4.2s,  v0.2s\n"
            "fmax  v5.2s,  v5.2s,  v0.2s\n"
            "fmin  v4.2s,  v4.2s,  v3.2s\n"
            "fmin  v5.2s,  v5.2s,  v3.2s\n"
            "fmul  v18.2s, v4.2s,  v6.2s\n"
            "fmul  v19.2s, v5.2s,  v7.2s\n"
            "fadd  v4.2s,  v20.2s,  v2.2s\n"
            "fadd  v5.2s,  v21.2s,  v2.2s\n"
            "fmul  v6.2s,  v20.2s,  v1.2s\n"
            "fmul  v7.2s,  v21.2s,  v1.2s\n"
            "fmax  v4.2s,  v4.2s,  v0.2s\n"
            "fmax  v5.2s,  v5.2s,  v0.2s\n"
            "fmin  v4.2s,  v4.2s,  v3.2s\n"
            "fmin  v5.2s,  v5.2s,  v3.2s\n"
            "fmul  v20.2s, v4.2s,  v6.2s\n"
            "fmul  v21.2s, v5.2s,  v7.2s\n"
            "fadd  v4.2s,  v22.2s,  v2.2s\n"
            "fadd  v5.2s,  v23.2s,  v2.2s\n"
            "fmul  v6.2s,  v22.2s,  v1.2s\n"
            "fmul  v7.2s,  v23.2s,  v1.2s\n"
            "fmax  v4.2s,  v4.2s,  v0.2s\n"
            "fmax  v5.2s,  v5.2s,  v0.2s\n"
            "fmin  v4.2s,  v4.2s,  v3.2s\n"
            "fmin  v5.2s,  v5.2s,  v3.2s\n"
            "fmul  v22.2s, v4.2s,  v6.2s\n"
            "fmul  v23.2s, v5.2s,  v7.2s\n"
            "fadd  v4.2s,  v24.2s,  v2.2s\n"
            "fadd  v5.2s,  v25.2s,  v2.2s\n"
            "fmul  v6.2s,  v24.2s,  v1.2s\n"
            "fmul  v7.2s,  v25.2s,  v1.2s\n"
            "fmax  v4.2s,  v4.2s,  v0.2s\n"
            "fmax  v5.2s,  v5.2s,  v0.2s\n"
            "fmin  v4.2s,  v4.2s,  v3.2s\n"
            "fmin  v5.2s,  v5.2s,  v3.2s\n"
            "fmul  v24.2s, v4.2s,  v6.2s\n"
            "fmul  v25.2s, v5.2s,  v7.2s\n"
            "fadd  v4.2s,  v26.2s,  v2.2s\n"
            "fadd  v5.2s,  v27.2s,  v2.2s\n"
            "fmul  v6.2s,  v26.2s,  v1.2s\n"
            "fmul  v7.2s,  v27.2s,  v1.2s\n"
            "fmax  v4.2s,  v4.2s,  v0.2s\n"
            "fmax  v5.2s,  v5.2s,  v0.2s\n"
            "fmin  v4.2s,  v4.2s,  v3.2s\n"
            "fmin  v5.2s,  v5.2s,  v3.2s\n"
            "fmul  v26.2s, v4.2s,  v6.2s\n"
            "fmul  v27.2s, v5.2s,  v7.2s\n"
            "fadd  v4.2s,  v28.2s,  v2.2s\n"
            "fadd  v5.2s,  v29.2s,  v2.2s\n"
            "fmul  v6.2s,  v28.2s,  v1.2s\n"
            "fmul  v7.2s,  v29.2s,  v1.2s\n"
            "fmax  v4.2s,  v4.2s,  v0.2s\n"
            "fmax  v5.2s,  v5.2s,  v0.2s\n"
            "fmin  v4.2s,  v4.2s,  v3.2s\n"
            "fmin  v5.2s,  v5.2s,  v3.2s\n"
            "fmul  v28.2s, v4.2s,  v6.2s\n"
            "fmul  v29.2s, v5.2s,  v7.2s\n"
            "fadd  v4.2s,  v30.2s,  v2.2s\n"
            "fadd  v5.2s,  v31.2s,  v2.2s\n"
            "fmul  v6.2s,  v30.2s,  v1.2s\n"
            "fmul  v7.2s,  v31.2s,  v1.2s\n"
            "fmax  v4.2s,  v4.2s,  v0.2s\n"
            "fmax  v5.2s,  v5.2s,  v0.2s\n"
            "fmin  v4.2s,  v4.2s,  v3.2s\n"
            "fmin  v5.2s,  v5.2s,  v3.2s\n"
            "fmul  v30.2s, v4.2s,  v6.2s\n"
            "fmul  v31.2s, v5.2s,  v7.2s\n"
            "b 5f\n"
            "8: \n"// leakyrelu
            "fcmge v2.2s, v8.2s, v0.2s\n"
            "fmul v3.2s, v8.2s, v1.2s\n"
            "fcmge v4.2s, v9.2s, v0.2s\n"
            "fmul v5.2s, v9.2s, v1.2s\n"
            "fcmge v6.2s, v10.2s, v0.2s\n"
            "fmul v7.2s, v10.2s, v1.2s\n"
            "bif v8.16b, v3.16b, v2.16b\n"
            "fcmge v2.2s, v11.2s, v0.2s\n"
            "fmul v3.2s, v11.2s, v1.2s\n"
            "bif v9.16b, v5.16b, v4.16b\n"
            "fcmge v4.2s, v12.2s, v0.2s\n"
            "fmul v5.2s, v12.2s, v1.2s\n"
            "bif v10.16b, v7.16b, v6.16b\n"
            "fcmge v6.2s, v13.2s, v0.2s\n"
            "fmul v7.2s, v13.2s, v1.2s\n"
            "bif v11.16b, v3.16b, v2.16b\n"
            "fcmge v2.2s, v14.2s, v0.2s\n"
            "fmul v3.2s, v14.2s, v1.2s\n"
            "bif v12.16b, v5.16b, v4.16b\n"
            "fcmge v4.2s, v15.2s, v0.2s\n"
            "fmul v5.2s, v15.2s, v1.2s\n"
            "bif v13.16b, v7.16b, v6.16b\n"
            "fcmge v6.2s, v16.2s, v0.2s\n"
            "fmul v7.2s, v16.2s, v1.2s\n"
            "bif v14.16b, v3.16b, v2.16b\n"
            "fcmge v2.2s, v17.2s, v0.2s\n"
            "fmul v3.2s, v17.2s, v1.2s\n"
            "bif v15.16b, v5.16b, v4.16b\n"
            "fcmge v4.2s, v18.2s, v0.2s\n"
            "fmul v5.2s, v18.2s, v1.2s\n"
            "bif v16.16b, v7.16b, v6.16b\n"
            "fcmge v6.2s, v19.2s, v0.2s\n"
            "fmul v7.2s, v19.2s, v1.2s\n"
            "bif v17.16b, v3.16b, v2.16b\n"
            "fcmge v2.2s, v20.2s, v0.2s\n"
            "fmul v3.2s, v20.2s, v1.2s\n"
            "bif v18.16b, v5.16b, v4.16b\n"
            "fcmge v4.2s, v21.2s, v0.2s\n"
            "fmul v5.2s, v21.2s, v1.2s\n"
            "bif v19.16b, v7.16b, v6.16b\n"
            "fcmge v6.2s, v22.2s, v0.2s\n"
            "fmul v7.2s, v22.2s, v1.2s\n"
            "bif v20.16b, v3.16b, v2.16b\n"
            "fcmge v2.2s, v23.2s, v0.2s\n"
            "fmul v3.2s, v23.2s, v1.2s\n"
            "bif v21.16b, v5.16b, v4.16b\n"
            "fcmge v4.2s, v24.2s, v0.2s\n"
            "fmul v5.2s, v24.2s, v1.2s\n"
            "bif v22.16b, v7.16b, v6.16b\n"
            "fcmge v6.2s, v25.2s, v0.2s\n"
            "fmul v7.2s, v25.2s, v1.2s\n"
            "bif v23.16b, v3.16b, v2.16b\n"
            "fcmge v2.2s, v26.2s, v0.2s\n"
            "fmul v3.2s, v26.2s, v1.2s\n"
            "bif v24.16b, v5.16b, v4.16b\n"
            "fcmge v4.2s, v27.2s, v0.2s\n"
            "fmul v5.2s, v27.2s, v1.2s\n"
            "bif v25.16b, v7.16b, v6.16b\n"
            "fcmge v6.2s, v28.2s, v0.2s\n"
            "fmul v7.2s, v28.2s, v1.2s\n"
            "bif v26.16b, v3.16b, v2.16b\n"
            "fcmge v2.2s, v29.2s, v0.2s\n"
            "fmul v3.2s, v29.2s, v1.2s\n"
            "bif v27.16b, v5.16b, v4.16b\n"
            "fcmge v4.2s, v30.2s, v0.2s\n"
            "fmul v5.2s, v30.2s, v1.2s\n"
            "bif v28.16b, v7.16b, v6.16b\n"
            "fcmge v6.2s, v31.2s, v0.2s\n"
            "fmul v7.2s, v31.2s, v1.2s\n"
            "bif v29.16b, v3.16b, v2.16b\n"
            "bif v30.16b, v5.16b, v4.16b\n"
            "bif v31.16b, v7.16b, v6.16b\n"
            "b 5f\n"
            "6: \n" // relu
            "fmax v8.2s, v8.2s, v0.2s\n"
            "fmax v9.2s, v9.2s, v0.2s\n"
            "fmax v10.2s, v10.2s, v0.2s\n"
            "fmax v11.2s, v11.2s, v0.2s\n"
            "fmax v12.2s, v12.2s, v0.2s\n"
            "fmax v13.2s, v13.2s, v0.2s\n"
            "fmax v14.2s, v14.2s, v0.2s\n"
            "fmax v15.2s, v15.2s, v0.2s\n"
            "fmax v16.2s, v16.2s, v0.2s\n"
            "fmax v17.2s, v17.2s, v0.2s\n"
            "fmax v18.2s, v18.2s, v0.2s\n"
            "fmax v19.2s, v19.2s, v0.2s\n"
            "fmax v20.2s, v20.2s, v0.2s\n"
            "fmax v21.2s, v21.2s, v0.2s\n"
            "fmax v22.2s, v22.2s, v0.2s\n"
            "fmax v23.2s, v23.2s, v0.2s\n"
            "fmax v24.2s, v24.2s, v0.2s\n"
            "fmax v25.2s, v25.2s, v0.2s\n"
            "fmax v26.2s, v26.2s, v0.2s\n"
            "fmax v27.2s, v27.2s, v0.2s\n"
            "fmax v28.2s, v28.2s, v0.2s\n"
            "fmax v29.2s, v29.2s, v0.2s\n"
            "fmax v30.2s, v30.2s, v0.2s\n"
            "fmax v31.2s, v31.2s, v0.2s\n"
            "b 5f\n"
            "7: \n" // relu6
            "fmax v8.2s, v8.2s, v0.2s\n"
            "fmax v9.2s, v9.2s, v0.2s\n"
            "fmax v10.2s, v10.2s, v0.2s\n"
            "fmax v11.2s, v11.2s, v0.2s\n"
            "fmax v12.2s, v12.2s, v0.2s\n"
            "fmax v13.2s, v13.2s, v0.2s\n"
            "fmax v14.2s, v14.2s, v0.2s\n"
            "fmax v15.2s, v15.2s, v0.2s\n"
            "fmax v16.2s, v16.2s, v0.2s\n"
            "fmax v17.2s, v17.2s, v0.2s\n"
            "fmax v18.2s, v18.2s, v0.2s\n"
            "fmax v19.2s, v19.2s, v0.2s\n"
            "fmax v20.2s, v20.2s, v0.2s\n"
            "fmax v21.2s, v21.2s, v0.2s\n"
            "fmax v22.2s, v22.2s, v0.2s\n"
            "fmax v23.2s, v23.2s, v0.2s\n"
            "fmax v24.2s, v24.2s, v0.2s\n"
            "fmax v25.2s, v25.2s, v0.2s\n"
            "fmax v26.2s, v26.2s, v0.2s\n"
            "fmax v27.2s, v27.2s, v0.2s\n"
            "fmax v28.2s, v28.2s, v0.2s\n"
            "fmax v29.2s, v29.2s, v0.2s\n"
            "fmax v30.2s, v30.2s, v0.2s\n"
            "fmax v31.2s, v31.2s, v0.2s\n"
            "fmin v8.2s, v8.2s, v1.2s\n"
            "fmin v9.2s, v9.2s, v1.2s\n"
            "fmin v10.2s, v10.2s, v1.2s\n"
            "fmin v11.2s, v11.2s, v1.2s\n"
            "fmin v12.2s, v12.2s, v1.2s\n"
            "fmin v13.2s, v13.2s, v1.2s\n"
            "fmin v14.2s, v14.2s, v1.2s\n"
            "fmin v15.2s, v15.2s, v1.2s\n"
            "fmin v16.2s, v16.2s, v1.2s\n"
            "fmin v17.2s, v17.2s, v1.2s\n"
            "fmin v18.2s, v18.2s, v1.2s\n"
            "fmin v19.2s, v19.2s, v1.2s\n"
            "fmin v20.2s, v20.2s, v1.2s\n"
            "fmin v21.2s, v21.2s, v1.2s\n"
            "fmin v22.2s, v22.2s, v1.2s\n"
            "fmin v23.2s, v23.2s, v1.2s\n"
            "fmin v24.2s, v24.2s, v1.2s\n"
            "fmin v25.2s, v25.2s, v1.2s\n"
            "fmin v26.2s, v26.2s, v1.2s\n"
            "fmin v27.2s, v27.2s, v1.2s\n"
            "fmin v28.2s, v28.2s, v1.2s\n"
            "fmin v29.2s, v29.2s, v1.2s\n"
            "fmin v30.2s, v30.2s, v1.2s\n"
            "fmin v31.2s, v31.2s, v1.2s\n"
            "b 5f\n"
            // no act
            "5: \n"
            "str d8, [%[c_ptr0]], #8\n"
            "str d11, [%[c_ptr1]], #8\n"
            "str d14, [%[c_ptr2]], #8\n"
            "str d17, [%[c_ptr3]], #8\n"
            "str d20, [%[c_ptr4]], #8\n"
            "str d23, [%[c_ptr5]], #8\n"
            "str d26, [%[c_ptr6]], #8\n"
            "str d29, [%[c_ptr7]], #8\n"
            "str d9, [%[c_ptr0]], #8\n"
            "str d12, [%[c_ptr1]], #8\n"
            "str d15, [%[c_ptr2]], #8\n"
            "str d18, [%[c_ptr3]], #8\n"
            "str d21, [%[c_ptr4]], #8\n"
            "str d24, [%[c_ptr5]], #8\n"
            "str d27, [%[c_ptr6]], #8\n"
            "str d30, [%[c_ptr7]], #8\n"
            "str d10, [%[c_ptr0]], #8\n"
            "str d13, [%[c_ptr1]], #8\n"
            "str d16, [%[c_ptr2]], #8\n"
            "str d19, [%[c_ptr3]], #8\n"
            "str d22, [%[c_ptr4]], #8\n"
            "str d25, [%[c_ptr5]], #8\n"
            "str d28, [%[c_ptr6]], #8\n"
            "str d31, [%[c_ptr7]], #8\n"
            : [a_ptr] "+r"(a_ptr),
              [b_ptr] "+r"(b_ptr),
              [k] "+r"(k),
              [c_ptr0] "+r"(c_ptr0),
              [c_ptr1] "+r"(c_ptr1),
              [c_ptr2] "+r"(c_ptr2),
              [c_ptr3] "+r"(c_ptr3),
              [c_ptr4] "+r"(c_ptr4),
              [c_ptr5] "+r"(c_ptr5),
              [c_ptr6] "+r"(c_ptr6),
              [c_ptr7] "+r"(c_ptr7)
            : [bias_ptr] "r"(bias_local),
              [has_beta] "r"(has_beta),
              [beta] "r"(beta),
              [alpha] "r"(alpha),
              [offset] "r"(offset),
              [threshold] "r"(threshold),
              [tail] "r"(tail),
              [flag_act] "r"(flag_act)
            : "cc","memory",
              "v0","v1","v2","v3","v4","v5","v6","v7",
              "v8","v9","v10","v11","v12","v13",
              "v14","v15","v16","v17","v18","v19",
              "v20","v21","v22","v23","v24","v25",
              "v26","v27","v28","v29","v30","v31");
        // clang-format on
        if (flag_p_remain && (xb == bblocks - 1)) {
          for (int i = 0; i < remain; ++i) {
            *pout0++ = cout0[i];
            *pout1++ = cout1[i];
            *pout2++ = cout2[i];
            *pout3++ = cout3[i];
            *pout4++ = cout4[i];
            *pout5++ = cout5[i];
            *pout6++ = cout6[i];
            *pout7++ = cout7[i];
          }
        }
      }
    }
    LITE_PARALLEL_COMMON_END()
  }
}
#undef FMLA_N8X6
#else
void sgemm_prepacked_4x8_a35(bool is_transB,
                             int M,
                             int N,
                             int K,
                             const float* A_packed,
                             const float* B,
                             int ldb,
                             float beta,
                             float* C,
                             int ldc,
                             const float* bias,
                             bool has_bias,
                             const operators::ActivationParam act_param,
                             ARMContext* ctx) {
  size_t l2_cache = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  auto* workspace = ctx->workspace_data<float>();
  int threads = ctx->threads();
  auto act_type = act_param.active_type;
  float alpha[12] = {0.f};
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3, hardswish: 4
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kHardSwish) {
      flag_act = 0x04;
      for (int i = 0; i < 4; i++) {
        alpha[i] = act_param.hard_swish_offset;
        alpha[i + 4] = 1.0 / act_param.hard_swish_scale;
        alpha[i + 8] = act_param.hard_swish_threshold;
      }
    }
  }
  X_BLOCK_COMPUTE(l2_cache, MBLOCK_A73, NBLOCK, M, N, K)
  int k_num = 2;

  int k_pre = ((K + k_num - 1) / k_num) - 1;
  int tail_pre = (K & (k_num - 1));
  if (tail_pre == 0) {
    tail_pre = k_num;
  }

  bool flag_p_remain = false;
  int remain = 0;
  //! merge tail_pre and flag_act
  tail_pre = (tail_pre * 5 + flag_act);

  int has_beta = fabsf(beta) > 1e-8f ? 1 : 0;

  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + NBLOCK - 1) / NBLOCK;
    remain = xmax - x0 - (bblocks - 1) * NBLOCK;
    if (remain > 0) {
      flag_p_remain = true;
    }
    //! load bpanel
    auto b_pannel = static_cast<float*>(workspace);
    if (is_transB) {
      loadb_trans(b_pannel, B, ldb, 0, K, x0, xmax);
    } else {
      loadb(b_pannel, B, ldb, 0, K, x0, xmax);
    }

    LITE_PARALLEL_COMMON_BEGIN(y, tid, M, 0, MBLOCK_A73) {
      unsigned int ymax = y + MBLOCK_A73;
      if (ymax > M) {
        ymax = M;
      }

      float cout0[NBLOCK];
      float cout1[NBLOCK];
      float cout2[NBLOCK];
      float cout3[NBLOCK];

      float bias_local[4] = {0};
      if (has_bias) {
        int i = 0;
        for (; i < 4 && y + i < ymax; i++) {
          bias_local[i] = bias[y + i];
        }
      }

      float* c_ptr0 = C + y * ldc + x0;
      float* c_ptr1 = c_ptr0 + ldc;
      float* c_ptr2 = c_ptr1 + ldc;
      float* c_ptr3 = c_ptr2 + ldc;

      float* pout0 = c_ptr0;
      float* pout1 = c_ptr1;
      float* pout2 = c_ptr2;
      float* pout3 = c_ptr3;

      const float* a_ptr_l = A_packed + y * K;
      const float* b_ptr = b_pannel;
      for (int xb = 0; xb < bblocks; xb++) {
        if ((y + 3) >= ymax) {
          switch ((y + 3) - ymax) {
            case 2:
              c_ptr1 = cout1;
            case 1:
              c_ptr2 = cout1;
            case 0:
              c_ptr3 = cout1;
            default:
              break;
          }
        }
        if (flag_p_remain && (xb == bblocks - 1)) {
          pout0 = c_ptr0;
          pout1 = c_ptr1;
          pout2 = c_ptr2;
          pout3 = c_ptr3;

          c_ptr0 = cout0;
          c_ptr1 = cout1;
          c_ptr2 = cout2;
          c_ptr3 = cout3;

          if (has_beta) {
            for (int i = 0; i < remain; ++i) {
              cout0[i] = pout0[i];
              cout1[i] = pout1[i];
              cout2[i] = pout2[i];
              cout3[i] = pout3[i];
            }
          }
        }
        const float* a_ptr = a_ptr_l;
        int tails = tail_pre;
        int k = k_pre;
        // clang-format off
        asm volatile(
            "pld [%[a_ptr]]                         @ preload a, 64byte\n"
            "vld1.32    {d6-d7}, [%[bias_ptr]]      @ load bias\n"
            "pld [%[b_ptr]]                         @ preload b\n"
            "vldr d0, [%[a_ptr]]                    \n"
            "vdup.32    q8, d6[0]                   @ add bias to out00\n"
            "vldr d4, [%[b_ptr]]                    \n"
            "vdup.32    q9, d6[0]                   @ add bias to out01\n"
            "vldr d1, [%[a_ptr], #0x08]             \n"
            "vdup.32    q10, d6[1]                  @ add bias to out10\n"
            "vldr d5, [%[b_ptr], #0x08]             \n"
            "vdup.32    q11, d6[1]                  @ add bias to out11\n"
            "pld [%[a_ptr], #64]                    @ preload a\n"
            "vdup.32    q12, d7[0]                  @ add bias to out20\n"
            "pld [%[b_ptr], #64]                    @ preload b\n"
            "vdup.32    q13, d7[0]                  @ add bias to out21\n"
            "pld [%[a_ptr], #128]                   @ preload a\n"
            "cmp %[beta], #0\n"                     //  check beta == 0
            "vdup.32    q14, d7[1]                  @ add bias to out30\n"
            "pld [%[b_ptr], #128]                   @ preload b\n"
            "vdup.32    q15, d7[1]                  @ add bias to out31\n"
            "pld [%[b_ptr], #192]                   @ preload b\n"
            "beq    11f\n"
            // process beta
            "vdup.32    q4, %[beta]\n"
            "vld1.32    {d0-d3}, [%[c_ptr0]]\n"
            "vld1.32    {d4-d7}, [%[c_ptr1]]\n"
            "vmla.f32   q8, q0, q4\n"           
            "vmla.f32   q9, q1, q4\n"          
            "vld1.32    {d0-d3}, [%[c_ptr2]]\n"
            "vmla.f32   q10, q2, q4\n"          
            "vmla.f32   q11, q3, q4\n"         
            "vld1.32    {d4-d7}, [%[c_ptr3]]\n" 
            "vmla.f32   q12, q0, q4\n"         
            "vmla.f32   q13, q1, q4\n"         
            "vmla.f32   q14, q2, q4\n"        
            "vmla.f32   q15, q3, q4\n"
            "vld1.32	  {d0-d1}, [%[a_ptr] :128]   @ load a0~a3\n"
            "vld1.32    {d4-d5}, [%[b_ptr] :128]  @ load b1\n"
            "11: \n"                            /* check loop count */
            "cmp %[k], #0                         @ check k==0 \n"
            "beq 0f                               @ jump to tail\n"
            "1:                                   @ main loop for k\n"
            /* Unroll 0*/
            "vldr d6, [%[b_ptr], #0x10]           \n"
            "vmla.f32 q8, q2, d0[0]\n"
            "ldr  r0, [%[b_ptr], #0x18]           \n"
            "vmla.f32 q10, q2, d0[1]\n"
            "ldr  r1, [%[b_ptr], #0x1c]           \n"
            "vmla.f32 q12, q2, d1[0]\n"
            "vldr d2, [%[a_ptr], #0x10]           \n"
            "vmov d7, r0, r1                      \n"
            "vmla.f32 q14, q2, d1[1]\n"
            "ldr  r0, [%[a_ptr], #0x18]           \n"

            "vmla.f32 q9, q3, d0[0]\n"
            "ldr  r1, [%[a_ptr], #0x1c]           \n"
            "vmla.f32 q11, q3, d0[1]\n"
            "vldr d8, [%[b_ptr], #0x20]           \n"
            "vmov d3, r0, r1                      \n"
            "ldr  r0, [%[b_ptr], #0x28]           \n"
            "vmla.f32 q13, q3, d1[0]\n"
            "ldr  r1, [%[b_ptr], #0x2c]           \n"
            "vmla.f32 q15, q3, d1[1]\n"
            "vldr d10, [%[b_ptr], #0x30]          \n"
            "vmov d9, r0, r1                      \n"
            /* Unroll 1 */
            "vmla.f32 q8, q4, d2[0]\n"
            "ldr  r0, [%[b_ptr], #0x38]           \n"
            "vmla.f32 q10, q4, d2[1]\n"
            "ldr  r1, [%[b_ptr], #0x3c]           \n"
            "pld [%[a_ptr], #128]                   @ preload b\n"
            "vmla.f32 q12, q4, d3[0]\n"
            "vldr d0, [%[a_ptr], #0x20]           \n"
            "vmov d11, r0, r1                     \n"
            "ldr  r0, [%[a_ptr], #0x28]           \n"
            "vmla.f32 q14, q4, d3[1]\n"
            "ldr  r1, [%[a_ptr], #0x2c]           \n"
            "pld [%[b_ptr], #192]                   @ preload b\n"

            "vmla.f32 q9, q5, d2[0]\n"
            "vldr d4, [%[b_ptr], #0x40]           \n"
            "vmov d1, r0, r1                      \n"
            "vmla.f32 q11, q5, d2[1]\n"
            "ldr  r0, [%[b_ptr], #0x48]           \n"
            "vmla.f32 q13, q5, d3[0]\n"
            "ldr  r1, [%[b_ptr], #0x4c]           \n"
            "subs		%[k], %[k], #1                @ k--\n"
            "add  %[b_ptr], %[b_ptr], #0x40       \n"
            "vmla.f32 q15, q5, d3[1]\n"
            "vmov d5, r0, r1                      \n"
            "add  %[a_ptr], %[a_ptr], #0x20       \n"
            "bne		1b                            @ jump to main loop\n"
            "0:                                   @ process tail\n"
            "cmp    %[tails], #10                  @ cmp with act bits\n"
            "blt		3f                            @ jump to tail = 1\n"
            /* Unroll 0*/
            "vldr d6, [%[b_ptr], #0x10]           \n"
            "vmla.f32 q8,  q2, d0[0]\n"
            "ldr  r0, [%[b_ptr], #0x18]           \n"
            "vmla.f32 q10, q2, d0[1]\n"
            "ldr  r1, [%[b_ptr], #0x1c]           \n"
            "vmla.f32 q12, q2, d1[0]\n"
            "vldr d2, [%[a_ptr], #0x10]           \n"
            "vmov d7, r0, r1                      \n"
            "vmla.f32 q14, q2, d1[1]\n"
            "ldr  r0, [%[a_ptr], #0x18]           \n"

            "vmla.f32 q9,  q3, d0[0]\n"
            "ldr  r1, [%[a_ptr], #0x1c]           \n"
            "vmla.f32 q11, q3, d0[1]\n"
            "vldr d8, [%[b_ptr], #0x20]           \n"
            "vmov d3, r0, r1                      \n"
            "vmla.f32 q13, q3, d1[0]\n"
            "ldr  r0, [%[b_ptr], #0x28]           \n"
            "vmla.f32 q15, q3, d1[1]\n"
            "ldr  r1, [%[b_ptr], #0x2c]           \n"
            "vldr d10, [%[b_ptr], #0x30]          \n"
            "vmov d9, r0, r1                      \n"
            /* Unroll 1 */
            "vmla.f32 q8,  q4, d2[0]\n"
            "ldr  r0, [%[b_ptr], #0x38]           \n"
            "vmla.f32 q10, q4, d2[1]\n"
            "ldr  r1, [%[b_ptr], #0x3c]           \n"
            "vmla.f32 q12, q4, d3[0]\n"
            "vmov d11, r0, r1                     \n"
            "vmla.f32 q14, q4, d3[1]\n"
            "sub		%[tails], %[tails], #10       @ tail--\n"

            "vmla.f32 q9,  q5, d2[0]\n"
            "add  %[b_ptr], %[b_ptr], #0x40       \n"
            "vmla.f32 q11, q5, d2[1]\n"
            "vmla.f32 q13, q5, d3[0]\n"
            "add  %[a_ptr], %[a_ptr], #0x20       \n"
            "vmla.f32 q15, q5, d3[1]\n"
            "b		2f\n"
            /* tails==1 final tail */
            "3:                                   @ tail=1\n"
            "vmla.f32 q8, q2, d0[0]\n"
            "vldr     d6, [%[b_ptr], #0x10]\n"
            "vmla.f32 q10, q2, d0[1]\n"
            "ldr  r0, [%[b_ptr], #0x18]           \n"
            "vmla.f32 q12, q2, d1[0]\n"
            "ldr  r1, [%[b_ptr], #0x1c]           \n"
            "vmla.f32 q14, q2, d1[1]\n"
            "vmov d7, r0, r1                      \n"
            "add  %[a_ptr], %[a_ptr], #0x10       \n"
            "sub	%[tails], %[tails], #5       @ tail--\n"

            "vmla.f32 q9,  q3, d0[0]\n"
            "vmla.f32 q11, q3, d0[1]\n"
            "add  %[b_ptr], %[b_ptr], #0x20       \n"
            "vmla.f32 q13, q3, d1[0]\n"
            "vmla.f32 q15, q3, d1[1]\n"
            "2:                                   @ check relu\n"
            //!   relu
            "cmp        %[tails], #1              @ check if has relu\n"
            "bne        6f                        @ jump if not relu \n"
            "vmov.u32   q0, #0                    @ for relu\n"
            "vmax.f32   q8, q8, q0                @ for relu\n"
            "vmax.f32   q9, q9, q0                @ for relu\n"
            "vmax.f32   q10, q10, q0              @ for relu\n"
            "vmax.f32   q11, q11, q0              @ for relu\n"
            "vmax.f32   q12, q12, q0              @ for relu\n"
            "vmax.f32   q13, q13, q0              @ for relu\n"
            "vmax.f32   q14, q14, q0              @ for relu\n"
            "vmax.f32   q15, q15, q0              @ for relu\n"
            "b          10f                       @ relu end\n"
            "6:                                   @ no relu \n"
            "cmp        %[tails], #0              @ check no act\n"
            "beq        10f                       @ no act end  \n"
            //!   relu6
            "cmp        %[tails], #2              @ check if has relu6\n"
            "bne        7f                        @ jump if no relu6 \n"
            "vmov.u32   q0, #0                    @ for relu6\n"
            "vld1.f32   {d2-d3}, [%[alpha]]       @ load relu6 alpha\n"
            "vmax.f32   q8, q8, q0                @ for relu6\n"
            "vmax.f32   q9, q9, q0                @ for relu6\n"
            "vmax.f32   q10, q10, q0              @ for relu6\n"
            "vmax.f32   q11, q11, q0              @ for relu6\n"
            "vmax.f32   q12, q12, q0              @ for relu6\n"
            "vmax.f32   q13, q13, q0              @ for relu6\n"
            "vmax.f32   q14, q14, q0              @ for relu6\n"
            "vmax.f32   q15, q15, q0              @ for relu6\n"

            "vmin.f32   q8, q8, q1                @ for relu6\n"
            "vmin.f32   q9, q9, q1                @ for relu6\n"
            "vmin.f32   q10, q10, q1              @ for relu6\n"
            "vmin.f32   q11, q11, q1              @ for relu6\n"
            "vmin.f32   q12, q12, q1              @ for relu6\n"
            "vmin.f32   q13, q13, q1              @ for relu6\n"
            "vmin.f32   q14, q14, q1              @ for relu6\n"
            "vmin.f32   q15, q15, q1              @ for relu6\n"
            "b          10f                       @ relu6 end \n"
            "7:                                   @ otherwise is leaky relu\n"
            "cmp        %[tails], #3              @ check if has leaky relu\n"
            "bne        8f                        @ jump if no leaky relu \n"
            //! leaky relu
            "vmov.u32   q0,   #0                  @ for leakey relu \n"
            "vld1.f32   {d2-d3}, [%[alpha]]       @ load leakey relu alpha\n"
            "vcge.f32   q2, q8, q0                @ vcgeq_u32 \n"
            "vmul.f32   q3, q8, q1                @ vmulq_f32 \n"
            "vcge.f32   q4, q9, q0                @ vcgeq_u32 \n"
            "vmul.f32   q5, q9, q1                @ vmulq_f32 \n"
            "vcge.f32   q6, q10, q0               @ vcgeq_u32 \n"
            "vmul.f32   q7, q10, q1               @ vmulq_f32 \n"
            "vbif       q8, q3, q2                @ choose    \n"
            "vbif       q9, q5, q4                @ choose    \n"
            "vbif       q10, q7, q6               @ choose    \n"
            "vcge.f32   q2, q11, q0               @ vcgeq_u32 \n"
            "vmul.f32   q3, q11, q1               @ vmulq_f32 \n"
            "vcge.f32   q4, q12, q0               @ vcgeq_u32 \n"
            "vmul.f32   q5, q12, q1               @ vmulq_f32 \n"
            "vcge.f32   q6, q13, q0               @ vcgeq_u32 \n"
            "vmul.f32   q7, q13, q1               @ vmulq_f32 \n"
            "vbif       q11, q3, q2               @ choose    \n"
            "vbif       q12, q5, q4               @ choose    \n"
            "vbif       q13, q7, q6               @ choose    \n"
            "vcge.f32   q2, q14, q0               @ vcgeq_u32 \n"
            "vmul.f32   q3, q14, q1               @ vmulq_f32 \n"
            "vcge.f32   q4, q15, q0               @ vcgeq_u32 \n"
            "vmul.f32   q5, q15, q1               @ vmulq_f32 \n"
            "vbif       q14, q3, q2               @ choose    \n"
            "vbif       q15, q5, q4               @ choose    \n"
            "b          10f                       @ leaky relu end \n"
            //! hard swish
            "8:                                   \n"
            "vmov.u32   q0,   #0                  @ for leakey relu \n"
            "vld1.f32   {d2-d5}, [%[alpha]]!      @ load scale and offset\n"
            "vadd.f32   q4,  q8, q1               \n"
            "vadd.f32   q6,  q9, q1               \n"
            "vmul.f32   q5,  q8, q2               \n"
            "vmul.f32   q7,  q9, q2               \n"
            "vld1.f32   {d6-d7}, [%[alpha]]       @ load threshold\n"
            "vmax.f32   q4,  q4, q0               \n"
            "vmax.f32   q6,  q6, q0               \n"
            "sub        %[alpha], #32             \n"
            "vmin.f32   q4,  q4, q3               \n"
            "vmin.f32   q6,  q6, q3               \n"
            "vmul.f32   q8,  q4, q5               \n"
            "vmul.f32   q9,  q6, q7               \n"
            "vadd.f32   q4,  q10, q1              \n"
            "vadd.f32   q6,  q11, q1              \n"
            "vmul.f32   q5,  q10, q2              \n"
            "vmul.f32   q7,  q11, q2              \n"
            "vmax.f32   q4,  q4, q0               \n"
            "vmax.f32   q6,  q6, q0               \n"
            "vmin.f32   q4,  q4, q3               \n"
            "vmin.f32   q6,  q6, q3               \n"
            "vmul.f32   q10, q4, q5               \n"
            "vmul.f32   q11, q6, q7               \n"
            "vadd.f32   q4,  q12, q1              \n"
            "vadd.f32   q6,  q13, q1              \n"
            "vmul.f32   q5,  q12, q2              \n"
            "vmul.f32   q7,  q13, q2              \n"
            "vmax.f32   q4,  q4, q0               \n"
            "vmax.f32   q6,  q6, q0               \n"
            "vmin.f32   q4,  q4, q3               \n"
            "vmin.f32   q6,  q6, q3               \n"
            "vmul.f32   q12, q4, q5               \n"
            "vmul.f32   q13, q6, q7               \n"
            "vadd.f32   q4,  q14, q1              \n"
            "vadd.f32   q6,  q15, q1              \n"
            "vmul.f32   q5,  q14, q2              \n"
            "vmul.f32   q7,  q15, q2              \n"
            "vmax.f32   q4,  q4, q0               \n"
            "vmax.f32   q6,  q6, q0               \n"
            "vmin.f32   q4,  q4, q3               \n"
            "vmin.f32   q6,  q6, q3               \n"
            "vmul.f32   q14, q4, q5               \n"
            "vmul.f32   q15, q6, q7               \n"
            "10:                                  @ act end  \n"
            "vst1.32    {d16-d19},  [%[c_ptr0]]!    @ store r0\n"
            "vst1.32    {d20-d23},  [%[c_ptr1]]!    @ store r1\n"
            "vst1.32    {d24-d27},  [%[c_ptr2]]!    @ store r2\n"
            "vst1.32    {d28-d31},  [%[c_ptr3]]!    @ store r3\n"
            : [a_ptr] "+r"(a_ptr),
              [b_ptr] "+r"(b_ptr),
              [c_ptr0] "+r"(c_ptr0),
              [c_ptr1] "+r"(c_ptr1),
              [c_ptr2] "+r"(c_ptr2),
              [c_ptr3] "+r"(c_ptr3),
              [k] "+r"(k),
              [tails] "+r"(tails)
            : [bias_ptr] "r"(bias_local),
              [beta] "r"(beta),
              [alpha] "r"(alpha)
            : "r0", "r1", "q0","q1","q2","q3",
              "q4","q5","q6","q7","q8","q9","q10",
              "q11","q12","q13","q14","q15","cc","memory");
        // clang-format on
        if (flag_p_remain && (xb == bblocks - 1)) {
          for (int i = 0; i < remain; ++i) {
            *pout0++ = cout0[i];
            *pout1++ = cout1[i];
            *pout2++ = cout2[i];
            *pout3++ = cout3[i];
          }
        }
      }
    }
    LITE_PARALLEL_COMMON_END()
  }
}
#endif
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
