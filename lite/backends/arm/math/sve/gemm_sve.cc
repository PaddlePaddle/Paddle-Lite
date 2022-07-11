// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/arm/math/sve/gemm_sve.h"
#include <arm_neon.h>
#include <arm_sve.h>
#include "lite/backends/arm/math/packed_sgemm.h"
#include "lite/backends/arm/math/sve/funcs_sve.h"
#include "lite/core/parallel_defines.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace sve {
#define GEMM_FUNCS_PARAM(dtype)                                        \
  bool is_transB, int M, int N, int K, const dtype *A_packed, int lda, \
      const dtype *B, int ldb, dtype beta, dtype *C, int ldc,          \
      const dtype *bias, bool has_bias,                                \
      const operators::ActivationParam act_param, ARMContext *ctx

#define GEMM_ACTUAL_PARAM                                             \
  is_transB, M, N, K, A_packed, B, ldb, beta, C, ldc, bias, has_bias, \
      act_param, ctx

#define X_BLOCK_COMPUTE_SVE(l2_cache, MBLOCK_SVE, NBLOCK_SVE_FP32, M, N, K) \
  int x_block = (l2_cache - (MBLOCK_SVE * K * sizeof(float))) /             \
                (sizeof(float) * (K + MBLOCK_SVE));                         \
  x_block = (x_block == 0) ? 1 : x_block;                                   \
  x_block /= NBLOCK_SVE_FP32;                                               \
  x_block *= NBLOCK_SVE_FP32;                                               \
  int x_num = (N + (x_block - 1)) / x_block;                                \
  x_block = (N + x_num - 1) / x_num;                                        \
  x_block = (x_block + NBLOCK_SVE_FP32 - 1) / NBLOCK_SVE_FP32;              \
  x_block *= NBLOCK_SVE_FP32;                                               \
  x_block = x_block < NBLOCK_SVE_FP32 ? NBLOCK_SVE_FP32 : x_block;

#define ACT_INIT                                                   \
  if (act_param.has_active) {                                      \
    if (act_type == lite_api::ActivationType::kRelu) {             \
      flag_act = 0x01;                                             \
    } else if (act_type == lite_api::ActivationType::kRelu6) {     \
      flag_act = 0x02;                                             \
      local_alpha = act_param.Relu_clipped_coef;                   \
      alpha[0] = local_alpha;                                      \
      alpha[1] = local_alpha;                                      \
      alpha[2] = local_alpha;                                      \
      alpha[3] = local_alpha;                                      \
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) { \
      flag_act = 0x03;                                             \
      local_alpha = act_param.Leaky_relu_alpha;                    \
      alpha[0] = local_alpha;                                      \
      alpha[1] = local_alpha;                                      \
      alpha[2] = local_alpha;                                      \
      alpha[3] = local_alpha;                                      \
    } else if (act_type == lite_api::ActivationType::kHardSwish) { \
      flag_act = 0x04;                                             \
      local_alpha = 1.0 / act_param.hard_swish_scale;              \
      for (int i = 0; i < 4; i++) {                                \
        alpha[i] = act_param.hard_swish_offset;                    \
        alpha[i + 4] = 1.0 / act_param.hard_swish_scale;           \
        alpha[i + 8] = act_param.hard_swish_threshold;             \
      }                                                            \
    }                                                              \
  }

#define COMPUTE_ASM_PARAM(a, b, c, d, e, f, g, h, i) \
  "fmla z" #a ".s, z" #i                             \
  ".s, z0.s[0]\n"                                    \
  "fmla z" #b ".s, z" #i                             \
  ".s, z0.s[1]\n"                                    \
  "fmla z" #c ".s, z" #i                             \
  ".s, z0.s[2]\n"                                    \
  "fmla z" #d ".s, z" #i                             \
  ".s, z0.s[3]\n"                                    \
  "fmla z" #e ".s, z" #i                             \
  ".s, z1.s[0]\n"                                    \
  "fmla z" #f ".s, z" #i                             \
  ".s, z1.s[1]\n"                                    \
  "fmla z" #g ".s, z" #i                             \
  ".s, z1.s[2]\n"                                    \
  "fmla z" #h ".s, z" #i ".s, z1.s[3]\n"

#define VMAXMIN_ASM(inst, a, b, c, d, e, f, g, h)                              \
  #inst " z" #a ".s, p0/m, z" #a ".s, z0.s\n" #inst " z" #b ".s, p0/m, z" #b   \
        ".s, z0.s\n" #inst " z" #c ".s, p0/m, z" #c ".s, z0.s\n" #inst " z" #d \
        ".s, p0/m, z" #d ".s, z0.s\n" #inst " z" #e ".s, p0/m, z" #e           \
        ".s, z0.s\n" #inst " z" #f ".s, p0/m, z" #f ".s, z0.s\n" #inst " z" #g \
        ".s, p0/m, z" #g ".s, z0.s\n" #inst " z" #h ".s, p0/m, z" #h           \
        ".s, z0.s\n"

#define VLEAKY_ASM(a, b, c, d, e, f, g, h) \
  "movprfx z2, z" #a                       \
  "\n  fmin z2.s,  p0/m, z2.s, z0.s\n"     \
  "movprfx z3, z" #b                       \
  "\n fmin z3.s,  p0/m, z3.s, z0.s\n"      \
  "movprfx z4, z" #c                       \
  "\n fmin z4.s,  p0/m, z4.s, z0.s\n"      \
  "movprfx z5, z" #d                       \
  "\n fmin z5.s,  p0/m, z5.s, z0.s\n"      \
  "fmax z" #a ".s, p0/m, z" #a             \
  ".s,  z0.s\n"                            \
  "fmax z" #b ".s, p0/m, z" #b             \
  ".s, z0.s\n"                             \
  "fmax z" #c ".s, p0/m, z" #c             \
  ".s, z0.s\n"                             \
  "fmax z" #d ".s, p0/m, z" #d             \
  ".s, z0.s\n"                             \
  "fmul z2.s,  p0/m, z2.s,  z1.s\n"        \
  "fmul z3.s,  p0/m, z3.s,  z1.s\n"        \
  "fmul z4.s,  p0/m, z4.s,  z1.s\n"        \
  "fmul z5.s,  p0/m, z5.s,  z1.s\n"        \
  "fadd z" #a ".s, p0/m, z" #a             \
  ".s, z2.s\n"                             \
  "fadd z" #b ".s, p0/m, z" #b             \
  ".s, z3.s\n"                             \
  "fadd z" #c ".s, p0/m, z" #c             \
  ".s, z4.s\n"                             \
  "fadd z" #d ".s, p0/m, z" #d             \
  ".s, z5.s\n"                             \
  "movprfx z2, z" #e                       \
  "\n  fmin z2.s,  p0/m, z2.s, z0.s\n"     \
  "movprfx z3, z" #f                       \
  "\n fmin z3.s,  p0/m, z3.s, z0.s\n"      \
  "movprfx z4, z" #g                       \
  "\n fmin z4.s,  p0/m, z4.s, z0.s\n"      \
  "movprfx z5, z" #h                       \
  "\n fmin z5.s,  p0/m, z5.s, z0.s\n"      \
  "fmax z" #e ".s, p0/m, z" #e             \
  ".s,  z0.s\n"                            \
  "fmax z" #f ".s, p0/m, z" #f             \
  ".s, z0.s\n"                             \
  "fmax z" #g ".s, p0/m, z" #g             \
  ".s, z0.s\n"                             \
  "fmax z" #h ".s, p0/m, z" #h             \
  ".s, z0.s\n"                             \
  "fmul z2.s,  p0/m, z2.s,  z1.s\n"        \
  "fmul z3.s,  p0/m, z3.s,  z1.s\n"        \
  "fmul z4.s,  p0/m, z4.s,  z1.s\n"        \
  "fmul z5.s,  p0/m, z5.s,  z1.s\n"        \
  "fadd z" #e ".s, p0/m, z" #e             \
  ".s, z2.s\n"                             \
  "fadd z" #f ".s, p0/m, z" #f             \
  ".s, z3.s\n"                             \
  "fadd z" #g ".s, p0/m, z" #g             \
  ".s, z4.s\n"                             \
  "fadd z" #h ".s, p0/m, z" #h ".s, z5.s\n"

// z1=offset z2=scale z3=threshold
#define VHARDSWISH_ASM(a, b, c, d, e, f, g, h) \
  "movprfx z4, z" #a                           \
  "\n fadd z4.s,  p0/m, z4.s, z1.s\n"          \
  "movprfx z5, z" #b                           \
  "\n fadd z5.s,  p0/m, z5.s, z1.s\n"          \
  "movprfx z6, z" #c                           \
  "\n fadd z6.s,  p0/m, z6.s, z1.s\n"          \
  "movprfx z7, z" #d                           \
  "\n fadd z7.s,  p0/m, z7.s, z1.s\n"          \
  "fmul z" #a ".s, p0/m, z" #a                 \
  ".s,  z2.s\n"                                \
  "fmul z" #b ".s, p0/m, z" #b                 \
  ".s,  z2.s\n"                                \
  "fmul z" #c ".s, p0/m, z" #c                 \
  ".s,  z2.s\n"                                \
  "fmul z" #d ".s, p0/m, z" #d                 \
  ".s,  z2.s\n"                                \
  "fmax z4.s,  p0/m, z4.s,   z0.s\n"           \
  "fmax z5.s,  p0/m, z5.s,   z0.s\n"           \
  "fmax z6.s,  p0/m, z6.s,   z0.s\n"           \
  "fmax z7.s,  p0/m, z7.s,   z0.s\n"           \
  "fmin z4.s,  p0/m, z4.s,   z3.s\n"           \
  "fmin z5.s,  p0/m, z5.s,   z3.s\n"           \
  "fmin z6.s,  p0/m, z6.s,   z3.s\n"           \
  "fmin z7.s,  p0/m, z7.s,   z3.s\n"           \
  "fmul z" #a ".s, p0/m, z" #a                 \
  ".s,  z4.s\n"                                \
  "fmul z" #b ".s, p0/m, z" #b                 \
  ".s,  z5.s\n"                                \
  "fmul z" #c ".s, p0/m, z" #c                 \
  ".s,  z6.s\n"                                \
  "fmul z" #d ".s, p0/m, z" #d                 \
  ".s,  z7.s\n"                                \
  "movprfx z4, z" #e                           \
  "\n fadd z4.s,  p0/m, z4.s, z1.s\n"          \
  "movprfx z5, z" #f                           \
  "\n fadd z5.s,  p0/m, z5.s, z1.s\n"          \
  "movprfx z6, z" #g                           \
  "\n fadd z6.s,  p0/m, z6.s, z1.s\n"          \
  "movprfx z7, z" #h                           \
  "\n fadd z7.s,  p0/m, z7.s, z1.s\n"          \
  "fmul z" #e ".s, p0/m, z" #e                 \
  ".s,  z2.s\n"                                \
  "fmul z" #f ".s, p0/m, z" #f                 \
  ".s,  z2.s\n"                                \
  "fmul z" #g ".s, p0/m, z" #g                 \
  ".s,  z2.s\n"                                \
  "fmul z" #h ".s, p0/m, z" #h                 \
  ".s,  z2.s\n"                                \
  "fmax z4.s,  p0/m, z4.s,   z0.s\n"           \
  "fmax z5.s,  p0/m, z5.s,   z0.s\n"           \
  "fmax z6.s,  p0/m, z6.s,   z0.s\n"           \
  "fmax z7.s,  p0/m, z7.s,   z0.s\n"           \
  "fmin z4.s,  p0/m, z4.s,   z3.s\n"           \
  "fmin z5.s,  p0/m, z5.s,   z3.s\n"           \
  "fmin z6.s,  p0/m, z6.s,   z3.s\n"           \
  "fmin z7.s,  p0/m, z7.s,   z3.s\n"           \
  "fmul z" #e ".s, p0/m, z" #e                 \
  ".s,  z4.s\n"                                \
  "fmul z" #f ".s, p0/m, z" #f                 \
  ".s,  z5.s\n"                                \
  "fmul z" #g ".s, p0/m, z" #g                 \
  ".s,  z6.s\n"                                \
  "fmul z" #h ".s, p0/m, z" #h ".s,  z7.s\n"

void sgemm_prepacked_8x12_sve(bool is_transB,
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
  float alpha[12] = {0.f};
  float local_alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  ACT_INIT
  // l2 = (MBLOCK * k + k * x_block + MBLOCK * x_block) * sizeof(float)
  // x_block = (l2 - MBLOCK * k * sizeof(float)) / (sizeof(float) * (k +
  // MBLOCK))
  X_BLOCK_COMPUTE_SVE(l2_cache, MBLOCK_SVE, NBLOCK_SVE_FP32, M, N, K)

  int has_beta = fabsf(beta) > 1e-8f ? 1 : 0;
  const auto all_true_pg = svptrue<float>();
  bool flag_p_remain = false;
  int remain = 0;

  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < N; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > N) {
      xmax = N;
    }
    int bblocks = (xmax - x0 + NBLOCK_SVE_FP32 - 1) / NBLOCK_SVE_FP32;
    remain = xmax - x0 - (bblocks - 1) * NBLOCK_SVE_FP32;
    if (remain > 0 && remain != 12) {
      flag_p_remain = true;
    }
    //! load bpanel
    float *b_pannel = workspace;
    if (is_transB) {
      loadb_trans(b_pannel, B, ldb, 0, K, x0, xmax);
    } else {
      loadb(b_pannel, B, ldb, 0, K, x0, xmax);
    }
    LITE_PARALLEL_COMMON_BEGIN(y, tid, M, 0, MBLOCK_SVE) {
      unsigned int ymax = y + MBLOCK_SVE;
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

      float cout0[NBLOCK_SVE_FP32];
      float cout1[NBLOCK_SVE_FP32];
      float cout2[NBLOCK_SVE_FP32];
      float cout3[NBLOCK_SVE_FP32];
      float cout4[NBLOCK_SVE_FP32];
      float cout5[NBLOCK_SVE_FP32];
      float cout6[NBLOCK_SVE_FP32];
      float cout7[NBLOCK_SVE_FP32];

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
          int cnt_rem = remain >> 2;
          int rem_rem = remain & 3;
          for (int i = 0; i < cnt_rem; i++) {
            const float *a_ptr = a_ptr_l;
            int k_cnt = K;
            asm volatile(
              "ptrue p0.b \n"
              "prfm   pldl1keep, [%[a_ptr]]\n" 
              "ld1rqw {z0.s}, p0/Z, [%x[bias]]\n"
              "ld1rqw {z1.s}, p0/Z, [%x[bias], #0x10]\n"
              "prfm   pldl1keep, [%[b_ptr]]\n" 
              "dup	  z8.s,  z0.s[0]\n"
              "prfm   pldl1keep, [%[b_ptr], #64]\n"
              "dup	  z11.s,  z0.s[1]\n"
              "prfm   pldl1keep, [%[a_ptr], #64]\n"
              "dup	  z14.s,  z0.s[2]\n"
              "prfm   pldl1keep, [%[b_ptr], #128]\n"
              "dup	  z17.s,  z0.s[3]\n"
              "prfm   pldl1keep, [%[a_ptr], #192]\n"
              "dup	  z20.s,  z1.s[0]\n"
              "prfm   pldl1keep, [%[b_ptr], #192]\n"
              "dup	  z23.s,  z1.s[1]\n"
              "prfm   pldl1keep, [%[a_ptr], #256]\n"
              "dup	  z26.s,  z1.s[2]\n"
              "prfm   pldl1keep, [%[b_ptr], #256]\n"
              "dup	  z29.s,  z1.s[2]\n"
              "cbz    %x[has_beta], 0f\n"
              /* process beta */
              "0: \n"
              "cbz	%x[k], 2f\n" 
              "1: \n"
              "ld1rqw {z0.s}, p0/Z, [%x[a_ptr]]\n"
              "ld1rqw {z1.s}, p0/Z, [%x[a_ptr], #0x10]\n"
              "ld1w   {z4.s}, p0/Z, [%x[b_ptr]]\n"
              "add %x[a_ptr], %x[a_ptr], #0x20\n"
              "add %x[b_ptr], %x[b_ptr], #0x10\n"
              "subs %x[k], %x[k], #1\n"
              COMPUTE_ASM_PARAM(8, 11, 14, 17, 20, 23, 26, 29, 4)
              "bne 1b\n"
              "2: \n"
              "mov z0.s, #0x0    \n"
              "cmp %x[flag_act], #1\n"
              "beq 3f\n"
              "cmp %x[flag_act], #0\n"
              "beq 10f\n"
              "cmp %x[flag_act], #2\n"
              "beq 4f\n"
              "cmp %x[flag_act], #3\n"
              "beq 5f\n"
              // hard_swish
              "ld1rqw {z1.s}, p0/Z, [%x[alpha]]\n"
              "ld1rqw {z2.s}, p0/Z, [%x[alpha], #0x10]\n"
              "ld1rqw {z3.s}, p0/Z, [%x[alpha], #0x20]\n"
              VHARDSWISH_ASM(8, 11, 14, 17, 20, 23, 26, 29)
              "b 10f\n"
              // relu
              "3: \n"
              VMAXMIN_ASM(fmax, 8, 11, 14, 17, 20, 23, 26, 29)
              "b 10f\n"
              // relu6
              "4: \n"
              "mov z0.s, #0x0    \n"
              VMAXMIN_ASM(fmax, 8, 11, 14, 17, 20, 23, 26, 29)
              "ld1rqw {z0.s}, p0/Z, [%x[alpha]]\n"
              VMAXMIN_ASM(fmin, 8, 11, 14, 17, 20, 23, 26, 29)
              "b 10f\n"
              // leakyrelu
              "5: \n"
              "ld1rqw {z1.s}, p0/Z, [%x[alpha]]\n"
              VLEAKY_ASM(8, 11, 14, 17, 20, 23, 26, 29)
              // no act
              "10: \n"
              "st1w {z8.s},  p0, [%x[c_ptr0]]\n"
              "st1w {z11.s}, p0, [%x[c_ptr1]]\n"
              "st1w {z14.s}, p0, [%x[c_ptr2]]\n"
              "st1w {z17.s}, p0, [%x[c_ptr3]]\n"
              "st1w {z20.s}, p0, [%x[c_ptr4]]\n"
              "st1w {z23.s}, p0, [%x[c_ptr5]]\n"
              "st1w {z26.s}, p0, [%x[c_ptr6]]\n"
              "st1w {z29.s}, p0, [%x[c_ptr7]]\n"
              "add %x[c_ptr0], %x[c_ptr0], #0x10\n"
              "add %x[c_ptr1], %x[c_ptr1], #0x10\n"
              "add %x[c_ptr2], %x[c_ptr2], #0x10\n"
              "add %x[c_ptr3], %x[c_ptr3], #0x10\n"
              "add %x[c_ptr4], %x[c_ptr4], #0x10\n"
              "add %x[c_ptr5], %x[c_ptr5], #0x10\n"
              "add %x[c_ptr6], %x[c_ptr6], #0x10\n"
              "add %x[c_ptr7], %x[c_ptr7], #0x10\n"
              : [a_ptr] "+r"(a_ptr),
                [b_ptr] "+r"(b_ptr),
                [k] "+r"(k_cnt),
                [c_ptr0] "+r"(c_ptr0),
                [c_ptr1] "+r"(c_ptr1),
                [c_ptr2] "+r"(c_ptr2),
                [c_ptr3] "+r"(c_ptr3),
                [c_ptr4] "+r"(c_ptr4),
                [c_ptr5] "+r"(c_ptr5),
                [c_ptr6] "+r"(c_ptr6),
                [c_ptr7] "+r"(c_ptr7)
              : [bias] "r"(bias_local),
                [has_beta] "r"(has_beta),
                [beta] "r"(beta),
                [alpha] "r"(alpha),
                [flag_act] "r"(flag_act)
              : "cc","memory", "p0",
                "z0","z1","z2","z3","z4","z5","z6","z7",
                "z8","z9","z10","z11","z12","z13",
                "z14","z15","z16","z17","z18","z19",
                "z20","z21","z22","z23","z24","z25",
                "z26","z27","z28","z29","z30","z31"
            );
          }
          auto vzero = svdup_n_f32(0.f);
          auto valpha = svdup_n_f32(local_alpha);
          for (int i = 0; i < rem_rem; i++) {
            const float *a_ptr = a_ptr_l;
            auto vout0 = svld1_f32(all_true_pg, bias_local);
            auto vout1 = svld1_f32(all_true_pg, bias_local + 4);
            for (int j = 0; j < K; j++) {
              auto va0 = svld1_f32(all_true_pg, a_ptr);
              auto va1 = svld1_f32(all_true_pg, a_ptr + 4);
              a_ptr += 8;
              vout0 = svmla_n_f32_z(all_true_pg, vout0, va0, b_ptr[0]);
              vout1 = svmla_n_f32_z(all_true_pg, vout1, va1, b_ptr[0]);
              b_ptr++;
            }
            if (flag_act == 0) {
            } else if (flag_act == 1) {
              vout0 = svmax_m(all_true_pg, vout0, vzero);
              vout1 = svmax_m(all_true_pg, vout1, vzero);
            } else if (flag_act == 2) {
              vout0 = svmin_m(
                  all_true_pg, svmax_m(all_true_pg, vout0, vzero), valpha);
              vout1 = svmin_m(
                  all_true_pg, svmax_m(all_true_pg, vout1, vzero), valpha);
            } else if (flag_act == 3) {
              vout0 = svadd_z(
                  all_true_pg,
                  svmul_z(
                      all_true_pg, svmin_z(all_true_pg, vout0, vzero), valpha),
                  svmax_z(all_true_pg, vout0, vzero));
              vout1 = svadd_z(
                  all_true_pg,
                  svmul_z(
                      all_true_pg, svmin_z(all_true_pg, vout1, vzero), valpha),
                  svmax_z(all_true_pg, vout1, vzero));
            } else {  // hard_swish
              auto voffset = svdup_n_f32(act_param.hard_swish_offset);
              auto vthreshold = svdup_n_f32(act_param.hard_swish_threshold);
              auto voff0 = svadd_m(all_true_pg, vout0, voffset);
              auto voff1 = svadd_m(all_true_pg, vout1, voffset);
              auto vscale0 = svmul_m(all_true_pg, vout0, valpha);
              auto vscale1 = svmul_m(all_true_pg, vout1, valpha);
              auto vmax0 = svmin_m(
                  all_true_pg, svmax_m(all_true_pg, voff0, vzero), vthreshold);
              auto vmax1 = svmin_m(
                  all_true_pg, svmax_m(all_true_pg, voff1, vzero), vthreshold);
              vout0 = svmul_m(all_true_pg, vscale0, vmax0);
              vout1 = svmul_m(all_true_pg, vscale1, vmax1);
            }
            float tmp[8] = {0};
            svst1_f32(all_true_pg, tmp, vout0);
            svst1_f32(all_true_pg, tmp + 4, vout1);
            *c_ptr0++ = tmp[0];
            *c_ptr1++ = tmp[1];
            *c_ptr2++ = tmp[2];
            *c_ptr3++ = tmp[3];
            *c_ptr4++ = tmp[4];
            *c_ptr5++ = tmp[5];
            *c_ptr6++ = tmp[6];
            *c_ptr7++ = tmp[7];
          }
        } else {
          const float *a_ptr = a_ptr_l;
          int k_cnt = K;
          asm volatile(
            "ptrue p0.b \n"
            "prfm   pldl1keep, [%[a_ptr]]\n" 
            "ld1rqw {z0.s}, p0/Z, [%x[bias]]\n"
            "ld1rqw {z1.s}, p0/Z, [%x[bias], #0x10]\n"
            "prfm   pldl1keep, [%[b_ptr]]\n" 
            "dup	  z8.s,  z0.s[0]\n"
            "dup	  z9.s,  z0.s[0]\n"
            "dup	  z10.s,  z0.s[0]\n"
            "prfm   pldl1keep, [%[b_ptr], #64]\n"
            "dup	  z11.s,  z0.s[1]\n"
            "dup	  z12.s,  z0.s[1]\n"
            "dup	  z13.s,  z0.s[1]\n"
            "prfm   pldl1keep, [%[a_ptr], #64]\n"
            "dup	  z14.s,  z0.s[2]\n"
            "dup	  z15.s,  z0.s[2]\n"
            "dup	  z16.s,  z0.s[2]\n"
            "prfm   pldl1keep, [%[b_ptr], #128]\n"
            "dup	  z17.s,  z0.s[3]\n"
            "dup	  z18.s,  z0.s[3]\n"
            "dup	  z19.s,  z0.s[3]\n"
            "prfm   pldl1keep, [%[a_ptr], #192]\n"
            "dup	  z20.s,  z1.s[0]\n"
            "dup	  z21.s,  z1.s[0]\n"
            "dup	  z22.s,  z1.s[0]\n"
            "prfm   pldl1keep, [%[b_ptr], #192]\n"
            "dup	  z23.s,  z1.s[1]\n"
            "dup	  z24.s,  z1.s[1]\n"
            "dup	  z25.s,  z1.s[1]\n"
            "prfm   pldl1keep, [%[a_ptr], #256]\n"
            "dup	  z26.s,  z1.s[2]\n"
            "dup	  z27.s,  z1.s[2]\n"
            "dup	  z28.s,  z1.s[2]\n"
            "prfm   pldl1keep, [%[b_ptr], #256]\n"
            "dup	  z29.s,  z1.s[2]\n"
            "dup	  z30.s,  z1.s[2]\n"
            "dup	  z31.s,  z1.s[2]\n"
            "prfm   pldl1keep, [%[a_ptr], #320]\n"
            "cbz    %x[has_beta], 0f\n"
            /* process beta */
            "0: \n"
            "cbz	%x[k], 2f\n" 
            "1: \n"
            "ld1rqw {z0.s}, p0/Z, [%x[a_ptr]]\n"
            "ld1rqw {z1.s}, p0/Z, [%x[a_ptr], #0x10]\n"
            "ld1w   {z4.s}, p0/Z, [%x[b_ptr]]\n"
            "ld1w   {z5.s}, p0/Z, [%x[b_ptr],  #1, MUL VL]\n"
            "add %x[a_ptr], %x[a_ptr], #0x20\n"
            "ld1w   {z6.s}, p0/Z, [%x[b_ptr],  #2, MUL VL]\n"
            "add %x[b_ptr], %x[b_ptr], #0x30\n"
            COMPUTE_ASM_PARAM(8, 11, 14, 17, 20, 23, 26, 29, 4)
            COMPUTE_ASM_PARAM(9, 12, 15, 18, 21, 24, 27, 30, 5)
            "subs %x[k], %x[k], #1\n"
            COMPUTE_ASM_PARAM(10, 13, 16, 19, 22, 25, 28, 31, 6)
            "bne 1b\n"
            "2: \n"
            "mov z0.s, #0x0    \n"
            "cmp %x[flag_act], #1\n"
            "beq 3f\n"
            "cmp %x[flag_act], #0\n"
            "beq 10f\n"
            "cmp %x[flag_act], #2\n"
            "beq 4f\n"
            "cmp %x[flag_act], #3\n"
            "beq 5f\n"
            // hard_swish
            "ld1rqw {z1.s}, p0/Z, [%x[alpha]]\n"
            "ld1rqw {z2.s}, p0/Z, [%x[alpha], #0x10]\n"
            "ld1rqw {z3.s}, p0/Z, [%x[alpha], #0x20]\n"
            VHARDSWISH_ASM(8, 11, 14, 17, 20, 23, 26, 29)
            VHARDSWISH_ASM(9,  12, 15, 18, 21, 24, 27, 30)
            VHARDSWISH_ASM(10, 13, 16, 19, 22, 25, 28, 31)
            "b 10f\n"
            // relu
            "3: \n"
            VMAXMIN_ASM(fmax, 8,  11, 14, 17, 20, 23, 26, 29)
            VMAXMIN_ASM(fmax, 9,  12, 15, 18, 21, 24, 27, 30)
            VMAXMIN_ASM(fmax, 10, 13, 16, 19, 22, 25, 28, 31)
            "b 10f\n"
            // relu6
            "4: \n"
            "mov z0.s, #0x0    \n"
            VMAXMIN_ASM(fmax, 8,  11, 14, 17, 20, 23, 26, 29)
            VMAXMIN_ASM(fmax, 9,  12, 15, 18, 21, 24, 27, 30)
            VMAXMIN_ASM(fmax, 10, 13, 16, 19, 22, 25, 28, 31)
            "ld1rqw {z0.s}, p0/Z, [%x[alpha]]\n"
            VMAXMIN_ASM(fmin, 8,  11, 14, 17, 20, 23, 26, 29)
            VMAXMIN_ASM(fmin, 9,  12, 15, 18, 21, 24, 27, 30)
            VMAXMIN_ASM(fmin, 10, 13, 16, 19, 22, 25, 28, 31)
            "b 10f\n"
            // leakyrelu
            "5: \n"
            "ld1rqw {z1.s}, p0/Z, [%x[alpha]]\n"
            VLEAKY_ASM(8, 11, 14, 17, 20, 23, 26, 29)
            VLEAKY_ASM(9,  12, 15, 18, 21, 24, 27, 30)
            VLEAKY_ASM(10, 13, 16, 19, 22, 25, 28, 31)
            // no act
            "10: \n"
            "st1w {z8.s},  p0, [%x[c_ptr0]]\n"
            "st1w {z11.s}, p0, [%x[c_ptr1]]\n"
            "st1w {z14.s}, p0, [%x[c_ptr2]]\n"
            "st1w {z17.s}, p0, [%x[c_ptr3]]\n"
            "st1w {z20.s}, p0, [%x[c_ptr4]]\n"
            "st1w {z23.s}, p0, [%x[c_ptr5]]\n"
            "st1w {z26.s}, p0, [%x[c_ptr6]]\n"
            "st1w {z29.s}, p0, [%x[c_ptr7]]\n"
            "st1w {z9.s},  p0, [%x[c_ptr0], #1, MUL VL]\n"
            "st1w {z12.s}, p0, [%x[c_ptr1], #1, MUL VL]\n"
            "st1w {z15.s}, p0, [%x[c_ptr2], #1, MUL VL]\n"
            "st1w {z18.s}, p0, [%x[c_ptr3], #1, MUL VL]\n"
            "st1w {z21.s}, p0, [%x[c_ptr4], #1, MUL VL]\n"
            "st1w {z24.s}, p0, [%x[c_ptr5], #1, MUL VL]\n"
            "st1w {z27.s}, p0, [%x[c_ptr6], #1, MUL VL]\n"
            "st1w {z30.s}, p0, [%x[c_ptr7], #1, MUL VL]\n"
            "st1w {z10.s}, p0, [%x[c_ptr0], #2, MUL VL]\n"
            "st1w {z13.s}, p0, [%x[c_ptr1], #2, MUL VL]\n"
            "st1w {z16.s}, p0, [%x[c_ptr2], #2, MUL VL]\n"
            "st1w {z19.s}, p0, [%x[c_ptr3], #2, MUL VL]\n"
            "st1w {z22.s}, p0, [%x[c_ptr4], #2, MUL VL]\n"
            "st1w {z25.s}, p0, [%x[c_ptr5], #2, MUL VL]\n"
            "st1w {z28.s}, p0, [%x[c_ptr6], #2, MUL VL]\n"
            "st1w {z31.s}, p0, [%x[c_ptr7], #2, MUL VL]\n"
            "add %x[c_ptr0], %x[c_ptr0], #0x30\n"
            "add %x[c_ptr1], %x[c_ptr1], #0x30\n"
            "add %x[c_ptr2], %x[c_ptr2], #0x30\n"
            "add %x[c_ptr3], %x[c_ptr3], #0x30\n"
            "add %x[c_ptr4], %x[c_ptr4], #0x30\n"
            "add %x[c_ptr5], %x[c_ptr5], #0x30\n"
            "add %x[c_ptr6], %x[c_ptr6], #0x30\n"
            "add %x[c_ptr7], %x[c_ptr7], #0x30\n"
            : [a_ptr] "+r"(a_ptr),
              [b_ptr] "+r"(b_ptr),
              [k] "+r"(k_cnt),
              [c_ptr0] "+r"(c_ptr0),
              [c_ptr1] "+r"(c_ptr1),
              [c_ptr2] "+r"(c_ptr2),
              [c_ptr3] "+r"(c_ptr3),
              [c_ptr4] "+r"(c_ptr4),
              [c_ptr5] "+r"(c_ptr5),
              [c_ptr6] "+r"(c_ptr6),
              [c_ptr7] "+r"(c_ptr7)
            : [bias] "r"(bias_local),
              [has_beta] "r"(has_beta),
              [beta] "r"(beta),
              [alpha] "r"(alpha),
              [flag_act] "r"(flag_act)
            : "cc","memory", "p0", "p1", "p2", "p3", "p4",
              "z0","z1","z2","z3","z4","z5","z6","z7",
              "z8","z9","z10","z11","z12","z13",
              "z14","z15","z16","z17","z18","z19",
              "z20","z21","z22","z23","z24","z25",
              "z26","z27","z28","z29","z30","z31"
          );
        }
      }
    }
    LITE_PARALLEL_COMMON_END();
  }
}

void sgemm_prepacked_4x8_sve(bool is_transB,
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
  auto *workspace = ctx->workspace_data<float>();
  int threads = ctx->threads();
  auto act_type = act_param.active_type;
  float alpha[12] = {0.f};
  float local_alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leaky: 4
  const int n_block = 8;
  const int m_block = 4;
  ACT_INIT
  X_BLOCK_COMPUTE_SVE(l2_cache, m_block, n_block, M, N, K)
  bool flag_p_remain = false;
  int remain = 0;
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
    auto b_pannel = static_cast<float *>(workspace);
    if (is_transB) {
      loadb_trans_eight(b_pannel, B, ldb, 0, K, x0, xmax);
    } else {
      loadb_eight(b_pannel, B, ldb, 0, K, x0, xmax);
    }
    LITE_PARALLEL_COMMON_BEGIN(y, tid, M, 0, m_block) {
      unsigned int ymax = y + m_block;
      if (ymax > M) {
        ymax = M;
      }
      float cout0[n_block];  // NOLINT
      float cout1[n_block];  // NOLINT
      float cout2[n_block];  // NOLINT;
      float cout3[n_block];  // NOLINT

      float bias_local[4] = {0};
      if (has_bias) {
        int i = 0;
        for (; i < 4 && y + i < ymax; i++) {
          bias_local[i] = bias[y + i];
        }
      }

      float *c_ptr0 = C + y * ldc + x0;
      float *c_ptr1 = c_ptr0 + ldc;
      float *c_ptr2 = c_ptr1 + ldc;
      float *c_ptr3 = c_ptr2 + ldc;

      float *pout0 = c_ptr0;
      float *pout1 = c_ptr1;
      float *pout2 = c_ptr2;
      float *pout3 = c_ptr3;

      const float *a_ptr_l = A_packed + y * K;
      const float *b_ptr = b_pannel;
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
        const float *a_ptr = a_ptr_l;
        int k_cnt = K;
        // clang-format off
       asm volatile(
              "ptrue p0.b \n"
              "prfm   pldl1keep, [%[a_ptr]]\n" 
              "ld1rqw {z0.s}, p0/Z, [%x[bias]]\n"
              "prfm   pldl1keep, [%[b_ptr]]\n" 
              "dup	  z8.s,  z0.s[0]\n"
              "dup	  z9.s,  z0.s[0]\n"
              "prfm   pldl1keep, [%[b_ptr], #64]\n"
              "dup	  z11.s,  z0.s[1]\n"
              "dup	  z12.s,  z0.s[1]\n"
              "prfm   pldl1keep, [%[a_ptr], #64]\n"
              "dup	  z14.s,  z0.s[2]\n"
              "dup	  z15.s,  z0.s[2]\n"
              "prfm   pldl1keep, [%[b_ptr], #128]\n"
              "dup	  z17.s,  z0.s[3]\n"
              "dup	  z18.s,  z0.s[3]\n"
              "prfm   pldl1keep, [%[a_ptr], #128]\n"
              "prfm   pldl1keep, [%[b_ptr], #192]\n"
              "cbz    %x[has_beta], 0f\n"
              /* process beta */
              "0: \n"
              "cbz	%x[k], 2f\n" 
              "1: \n"
              "ld1rqw {z0.s}, p0/Z, [%x[a_ptr]]\n"
              "ld1w   {z4.s}, p0/Z, [%x[b_ptr]]\n"
              "ld1rqw {z5.s}, p0/Z, [%x[b_ptr], #0x10]\n"
              "add %x[a_ptr], %x[a_ptr], #0x10\n"
              "add %x[b_ptr], %x[b_ptr], #0x20\n"
              "subs %x[k], %x[k], #1\n"
              "fmla z8.s,  z4.s, z0.s[0]\n"
              "fmla z11.s, z4.s, z0.s[1]\n"
              "fmla z14.s, z4.s, z0.s[2]\n"
              "fmla z17.s, z4.s, z0.s[3]\n"
              "fmla z9.s,  z5.s, z0.s[0]\n"
              "fmla z12.s, z5.s, z0.s[1]\n"
              "fmla z15.s, z5.s, z0.s[2]\n"
              "fmla z18.s, z5.s, z0.s[3]\n"
              "bne 1b\n"
              "2: \n"
              "mov z0.s, #0x0    \n"
              "cmp %x[flag_act], #1\n"
              "beq 3f\n"
              "cmp %x[flag_act], #0\n"
              "beq 10f\n"
              "cmp %x[flag_act], #2\n"
              "beq 4f\n"
              "cmp %x[flag_act], #3\n"
              "beq 5f\n"
              // hard_swish
              "ld1rqw {z1.s}, p0/Z, [%x[alpha]]\n"
              "ld1rqw {z2.s}, p0/Z, [%x[alpha], #0x10]\n"
              "ld1rqw {z3.s}, p0/Z, [%x[alpha], #0x20]\n"
              "movprfx z4, z8\n  fadd z4.s,    p0/m, z4.s, z1.s\n"
              "movprfx z5, z11\n fadd z5.s,    p0/m, z5.s, z1.s\n"
              "movprfx z6, z14\n fadd z6.s,    p0/m, z6.s, z1.s\n"
              "movprfx z7, z17\n fadd z7.s,    p0/m, z7.s, z1.s\n"
              "movprfx z10, z9\n  fadd z10.s,  p0/m, z10.s, z1.s\n"
              "movprfx z13, z12\n fadd z13.s,  p0/m, z13.s, z1.s\n"
              "movprfx z16, z15\n fadd z16.s,  p0/m, z16.s, z1.s\n"
              "movprfx z19, z18\n fadd z19.s,  p0/m, z19.s, z1.s\n"

              "fmul z8.s,  p0/m, z8.s,   z2.s\n" 
              "fmul z11.s, p0/m, z11.s,  z2.s\n"
              "fmul z14.s, p0/m, z14.s,  z2.s\n"
              "fmul z17.s, p0/m, z17.s,  z2.s\n"
              "fmul z9.s,  p0/m, z9.s,   z2.s\n" 
              "fmul z12.s, p0/m, z12.s,  z2.s\n"
              "fmul z15.s, p0/m, z15.s,  z2.s\n"
              "fmul z18.s, p0/m, z18.s,  z2.s\n"
              "fmax z4.s,  p0/m, z4.s,   z0.s\n"
              "fmax z5.s,  p0/m, z5.s,   z0.s\n" 
              "fmax z6.s,  p0/m, z6.s,   z0.s\n"
              "fmax z7.s,  p0/m, z7.s,   z0.s\n"
              "fmax z10.s, p0/m, z10.s,  z0.s\n"
              "fmax z13.s, p0/m, z13.s,  z0.s\n" 
              "fmax z16.s, p0/m, z16.s,  z0.s\n"
              "fmax z19.s, p0/m, z19.s,  z0.s\n"
              "fmin z4.s,  p0/m, z4.s,   z3.s\n"
              "fmin z5.s,  p0/m, z5.s,   z3.s\n"
              "fmin z6.s,  p0/m, z6.s,   z3.s\n"
              "fmin z7.s,  p0/m, z7.s,   z3.s\n"
              "fmin z10.s, p0/m, z10.s,  z0.s\n"
              "fmin z13.s, p0/m, z13.s,  z0.s\n" 
              "fmin z16.s, p0/m, z16.s,  z0.s\n"
              "fmin z19.s, p0/m, z19.s,  z0.s\n"

              "fmul z8.s,  p0/m, z8.s,  z4.s\n"
              "fmul z11.s, p0/m, z11.s, z5.s\n"
              "fmul z14.s, p0/m, z14.s, z6.s\n"
              "fmul z17.s, p0/m, z17.s, z7.s\n"
              "fmul z9.s,  p0/m, z9.s,  z10.s\n"
              "fmul z12.s, p0/m, z12.s, z13.s\n"
              "fmul z15.s, p0/m, z15.s, z16.s\n"
              "fmul z18.s, p0/m, z18.s, z19.s\n"
              "b 10f\n"
              // relu
              "3: \n"
              "fmax z8.s,  p0/m, z8.s,   z0.s\n"
              "fmax z11.s, p0/m, z11.s,  z0.s\n" 
              "fmax z14.s, p0/m, z14.s,  z0.s\n"
              "fmax z17.s, p0/m, z17.s,  z0.s\n"
              "fmax z9.s,  p0/m, z9.s,  z0.s\n"
              "fmax z12.s, p0/m, z12.s,  z0.s\n" 
              "fmax z15.s, p0/m, z15.s,  z0.s\n"
              "fmax z18.s, p0/m, z18.s,  z0.s\n"
              "b 10f\n"
              // relu6
              "4: \n"
              "ld1rqw {z1.s}, p0/Z, [%x[alpha]]\n"
              "fmax z8.s,  p0/m, z8.s,   z0.s\n"
              "fmax z11.s, p0/m, z11.s,  z0.s\n" 
              "fmax z14.s, p0/m, z14.s,  z0.s\n"
              "fmax z17.s, p0/m, z17.s,  z0.s\n"
              "fmax z9.s,  p0/m, z9.s,   z0.s\n"
              "fmax z12.s, p0/m, z12.s,  z0.s\n" 
              "fmax z15.s, p0/m, z15.s,  z0.s\n"
              "fmax z18.s, p0/m, z18.s,  z0.s\n"
              "fmin z8.s,  p0/m, z8.s,   z1.s\n"
              "fmin z11.s, p0/m, z11.s,  z1.s\n" 
              "fmin z14.s, p0/m, z14.s,  z1.s\n"
              "fmin z17.s, p0/m, z17.s,  z1.s\n"
              "fmin z9.s,  p0/m, z9.s,   z1.s\n"
              "fmin z12.s, p0/m, z12.s,  z1.s\n" 
              "fmin z15.s, p0/m, z15.s,  z1.s\n"
              "fmin z18.s, p0/m, z18.s,  z1.s\n"
              "b 10f\n"
              // leakyrelu
              "5: \n"
              "ld1rqw {z1.s}, p0/Z, [%x[alpha]]\n"
              "movprfx z2,  z8\n  fmin z2.s,   p0/m, z2.s,  z0.s\n" 
              "movprfx z3,  z11\n fmin z3.s,   p0/m, z3.s,  z0.s\n" 
              "movprfx z4,  z14\n fmin z4.s,   p0/m, z4.s,  z0.s\n" 
              "movprfx z5,  z17\n fmin z5.s,   p0/m, z5.s,  z0.s\n"
              "movprfx z10, z9\n  fmin z10.s,  p0/m, z10.s, z0.s\n" 
              "movprfx z13, z12\n fmin z13.s,  p0/m, z13.s, z0.s\n" 
              "movprfx z16, z15\n fmin z16.s,  p0/m, z16.s, z0.s\n" 
              "movprfx z19, z18\n fmin z19.s,  p0/m, z19.s, z0.s\n" 
              "fmax z8.s,  p0/m, z8.s,  z0.s\n"
              "fmax z11.s, p0/m, z11.s, z0.s\n"
              "fmax z14.s, p0/m, z14.s, z0.s\n"
              "fmax z17.s, p0/m, z17.s, z0.s\n"
              "fmax z9.s,  p0/m, z9.s,  z0.s\n"
              "fmax z12.s, p0/m, z12.s, z0.s\n"
              "fmax z15.s, p0/m, z15.s, z0.s\n"
              "fmax z18.s, p0/m, z18.s, z0.s\n"

              "fmul z2.s,   p0/m, z2.s,   z1.s\n"
              "fmul z3.s,   p0/m, z3.s,   z1.s\n"
              "fmul z4.s,   p0/m, z4.s,   z1.s\n"
              "fmul z5.s,   p0/m, z5.s,   z1.s\n"
              "fmul z10.s,  p0/m, z10.s,  z1.s\n"
              "fmul z13.s,  p0/m, z13.s,  z1.s\n"
              "fmul z16.s,  p0/m, z16.s,  z1.s\n"
              "fmul z19.s,  p0/m, z19.s,  z1.s\n"

              "fadd z8.s,  p0/m, z8.s,  z2.s\n"
              "fadd z11.s, p0/m, z11.s, z3.s\n" 
              "fadd z14.s, p0/m, z14.s, z4.s\n"
              "fadd z17.s, p0/m, z17.s, z5.s\n"
              "fadd z9.s,  p0/m, z9.s,  z10.s\n"
              "fadd z12.s, p0/m, z12.s, z13.s\n" 
              "fadd z15.s, p0/m, z15.s, z16.s\n"
              "fadd z18.s, p0/m, z18.s, z19.s\n"
              
              // no act
              "10: \n"
              "st1w {z8.s},  p0, [%x[c_ptr0]]\n"
              "st1w {z11.s}, p0, [%x[c_ptr1]]\n"
              "st1w {z14.s}, p0, [%x[c_ptr2]]\n"
              "st1w {z17.s}, p0, [%x[c_ptr3]]\n"
              "st1w {z9.s},  p0, [%x[c_ptr0], #1, MUL VL]\n"
              "st1w {z12.s}, p0, [%x[c_ptr1], #1, MUL VL]\n"
              "st1w {z15.s}, p0, [%x[c_ptr2], #1, MUL VL]\n"
              "st1w {z18.s}, p0, [%x[c_ptr3], #1, MUL VL]\n"
              "add %x[c_ptr0], %x[c_ptr0], #0x20\n"
              "add %x[c_ptr1], %x[c_ptr1], #0x20\n"
              "add %x[c_ptr2], %x[c_ptr2], #0x20\n"
              "add %x[c_ptr3], %x[c_ptr3], #0x20\n"
              : [a_ptr] "+r"(a_ptr),
                [b_ptr] "+r"(b_ptr),
                [k] "+r"(k_cnt),
                [c_ptr0] "+r"(c_ptr0),
                [c_ptr1] "+r"(c_ptr1),
                [c_ptr2] "+r"(c_ptr2),
                [c_ptr3] "+r"(c_ptr3)
              : [bias] "r"(bias_local),
                [has_beta] "r"(has_beta),
                [beta] "r"(beta),
                [alpha] "r"(alpha),
                [flag_act] "r"(flag_act)
              : "cc","memory", "p0",
                "z0","z1","z2","z3","z4","z5","z6","z7",
                "z8","z9","z10","z11","z12","z13",
                "z14","z15","z16","z17","z18","z19"
            );
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
    LITE_PARALLEL_COMMON_END();
  }
}

/// a: m*k  b: k*n  c: m*n
template <>
void sgemm_prepack_sve<float>(GEMM_FUNCS_PARAM(float)) {
  if (M <= 4) {
    sgemm_prepacked_4x8_sve(GEMM_ACTUAL_PARAM);
  } else {
    sgemm_prepacked_8x12_sve(GEMM_ACTUAL_PARAM);
  }
}

#undef GEMM_FUNCS_PARAM
#undef GEMM_ACTUAL_PARAM
#undef X_BLOCK_COMPUTE_SVE
#undef ACT_INIT
#undef COMPUTE_ASM_PARAM
#undef VMAXMIN_ASM
#undef VLEAKY_ASM
#undef VHARDSWISH_ASM
}  // namespace sve
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
