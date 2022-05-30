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
#define PREPACKA_FUNCS_PARAM(dtype)                                     \
  dtype *out, const dtype *in, dtype alpha, int ldin, int m0, int mmax, \
      int k0, int kmax

#define PREPACKA_ACTUAL_PARAM out, in, alpha, ldin, m0, mmax, k0, kmax

#define GEMM_FUNCS_PARAM(dtype)                                        \
  bool is_transB, int M, int N, int K, const dtype *A_packed, int lda, \
      const dtype *B, int ldb, dtype beta, dtype *C, int ldc,          \
      const dtype *bias, bool has_bias,                                \
      const operators::ActivationParam act_param, ARMContext *ctx

#define GEMM_ACTUAL_PARAM                                                  \
  is_transB, M, N, K, A_packed, lda, B, ldb, beta, C, ldc, bias, has_bias, \
      act_param, ctx

template <typename Dtype>
void sgemm_prepacked_m8_sve(bool is_transB,
                            int M,
                            int N,
                            int K,
                            const Dtype *A,
                            int lda,
                            const Dtype *B,
                            int ldb,
                            Dtype beta,
                            Dtype *C,
                            int ldc,
                            const Dtype *bias,
                            bool has_bias,
                            const operators::ActivationParam act_param,
                            ARMContext *ctx) {
  size_t l2_cache = ctx->llc_size() > 0 ? ctx->llc_size() : 512 * 1024;
  auto workspace = ctx->workspace_data<Dtype>();
  int threads = ctx->threads();

  auto act_type = act_param.active_type;
  Dtype alpha[12] = {0.f};
  Dtype local_alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 0x01;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 0x02;
      local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 0x03;
      local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kHardSwish) {
      flag_act = 0x04;
      local_alpha = 1.0 / act_param.hard_swish_scale;
      for (int i = 0; i < 4; i++) {
        alpha[i] = act_param.hard_swish_offset;
        alpha[i + 4] = 1.0 / act_param.hard_swish_scale;
        alpha[i + 8] = act_param.hard_swish_threshold;
      }
    }
  }
  // l2 = (x_block * k + k * n + x_block * n) * sizeof(Dtype)
  // x_block = (l2 - k * n * sizeof(Dtype)) / (sizeof(Dtype) * (k + n))
  int x_block = (l2_cache - K * N * sizeof(Dtype)) / (sizeof(Dtype) * (K + N));
  x_block = (x_block <= 0) ? 1 : x_block;
  x_block /= MBLOCK_SVE;
  x_block *= MBLOCK_SVE;
  int x_num = (M + (x_block - 1)) / x_block;
  x_block = (M + (x_block - 1)) / x_num;
  x_block = (x_block + MBLOCK_SVE - 1) / MBLOCK_SVE;
  x_block *= MBLOCK_SVE;
  x_block = x_block < MBLOCK_SVE ? MBLOCK_SVE : x_block;

  // unroll 2 loop
  int tail_pre = (K & (KBLOCK_SVE - 1));
  int k_pre = ((K + KBLOCK_SVE - 1) / KBLOCK_SVE) - 1;

  bool flag_p_remain = false;
  int remain = 0;
  if (tail_pre == 0) {
    tail_pre = KBLOCK_SVE;
  }

  int has_beta = fabsf(beta) > 1e-8f ? 1 : 0;
  auto vzero = svdup_n(static_cast<Dtype>(0.f));
  auto valpha = svdup_n(static_cast<Dtype>(local_alpha));

  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < M; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > M) {
      xmax = M;
    }
    int bblocks = (xmax - x0 + MBLOCK_SVE - 1) / MBLOCK_SVE;
    remain = xmax - x0 - (bblocks - 1) * MBLOCK_SVE;
    if (remain > 0 && remain != 8) {
      flag_p_remain = true;
    }
    if (flag_p_remain) {
      bblocks -= 1;
    }

    LITE_PARALLEL_COMMON_BEGIN(y, tid, bblocks, 0, 1) {
      auto index_y = y * MBLOCK_SVE;
      const Dtype *a_ptr_l = A + index_y * lda;
      const Dtype *b_ptr = B;
      Dtype *c_ptr0 = C + index_y * ldc;
      Dtype *c_ptr1 = c_ptr0 + ldc;
      Dtype *c_ptr2 = c_ptr1 + ldc;
      Dtype *c_ptr3 = c_ptr2 + ldc;
      Dtype *c_ptr4 = c_ptr3 + ldc;
      Dtype *c_ptr5 = c_ptr4 + ldc;
      Dtype *c_ptr6 = c_ptr5 + ldc;
      Dtype *c_ptr7 = c_ptr6 + ldc;
      const Dtype *a_ptr0 = a_ptr_l;
      const Dtype *a_ptr1 = a_ptr0 + lda;
      const Dtype *a_ptr2 = a_ptr1 + lda;
      const Dtype *a_ptr3 = a_ptr2 + lda;
      const Dtype *a_ptr4 = a_ptr3 + lda;
      const Dtype *a_ptr5 = a_ptr4 + lda;
      const Dtype *a_ptr6 = a_ptr5 + lda;
      const Dtype *a_ptr7 = a_ptr6 + lda;
      auto vbias0 = svdup_n(static_cast<Dtype>(0.f));
      auto vbias1 = svdup_n(static_cast<Dtype>(0.f));
      auto vbias2 = svdup_n(static_cast<Dtype>(0.f));
      auto vbias3 = svdup_n(static_cast<Dtype>(0.f));
      auto vbias4 = svdup_n(static_cast<Dtype>(0.f));
      auto vbias5 = svdup_n(static_cast<Dtype>(0.f));
      auto vbias6 = svdup_n(static_cast<Dtype>(0.f));
      auto vbias7 = svdup_n(static_cast<Dtype>(0.f));
      if (has_bias) {
        vbias0 = svdup_n(static_cast<Dtype>(bias[index_y]));
        vbias1 = svdup_n(static_cast<Dtype>(bias[index_y + 1]));
        vbias2 = svdup_n(static_cast<Dtype>(bias[index_y + 2]));
        vbias3 = svdup_n(static_cast<Dtype>(bias[index_y + 3]));
        vbias4 = svdup_n(static_cast<Dtype>(bias[index_y + 4]));
        vbias5 = svdup_n(static_cast<Dtype>(bias[index_y + 5]));
        vbias6 = svdup_n(static_cast<Dtype>(bias[index_y + 6]));
        vbias7 = svdup_n(static_cast<Dtype>(bias[index_y + 7]));
      }

      for (int x = 0; x < N; x += svcnt<Dtype>()) {
        svbool_t pg = svwhilelt<Dtype>(x, N);
        const Dtype *b_ptr_l = b_ptr;
        auto vout0 = vbias0;
        auto vout1 = vbias1;
        auto vout2 = vbias2;
        auto vout3 = vbias3;
        auto vout4 = vbias4;
        auto vout5 = vbias5;
        auto vout6 = vbias6;
        auto vout7 = vbias7;
        for (int k = 0; k < K; k++) {
          auto vb = svld1(pg, b_ptr_l);
          auto va0 = svdup_n(static_cast<Dtype>(*a_ptr0));
          auto va1 = svdup_n(static_cast<Dtype>(*a_ptr1));
          auto va2 = svdup_n(static_cast<Dtype>(*a_ptr2));
          auto va3 = svdup_n(static_cast<Dtype>(*a_ptr3));
          auto va4 = svdup_n(static_cast<Dtype>(*a_ptr4));
          auto va5 = svdup_n(static_cast<Dtype>(*a_ptr5));
          auto va6 = svdup_n(static_cast<Dtype>(*a_ptr6));
          auto va7 = svdup_n(static_cast<Dtype>(*a_ptr7));
          vout0 = svmla_m(pg, vout0, vb, va0);
          vout1 = svmla_m(pg, vout1, vb, va1);
          vout2 = svmla_m(pg, vout2, vb, va2);
          vout3 = svmla_m(pg, vout3, vb, va3);
          vout4 = svmla_m(pg, vout4, vb, va4);
          vout5 = svmla_m(pg, vout5, vb, va5);
          vout6 = svmla_m(pg, vout6, vb, va6);
          vout7 = svmla_m(pg, vout7, vb, va7);
          b_ptr_l += ldb;
          a_ptr0++;
          a_ptr1++;
          a_ptr2++;
          a_ptr3++;
          a_ptr4++;
          a_ptr5++;
          a_ptr6++;
          a_ptr7++;
        }
        if (flag_act == 0) {
        } else if (flag_act == 1) {  // relu
          vout0 = svmax_m(pg, vout0, vzero);
          vout1 = svmax_m(pg, vout1, vzero);
          vout2 = svmax_m(pg, vout2, vzero);
          vout3 = svmax_m(pg, vout3, vzero);
          vout4 = svmax_m(pg, vout4, vzero);
          vout5 = svmax_m(pg, vout5, vzero);
          vout6 = svmax_m(pg, vout6, vzero);
          vout7 = svmax_m(pg, vout7, vzero);
        } else if (flag_act == 2) {  // relu6
          vout0 = svmin_m(pg, svmax_m(pg, vout0, vzero), valpha);
          vout1 = svmin_m(pg, svmax_m(pg, vout1, vzero), valpha);
          vout2 = svmin_m(pg, svmax_m(pg, vout2, vzero), valpha);
          vout3 = svmin_m(pg, svmax_m(pg, vout3, vzero), valpha);
          vout4 = svmin_m(pg, svmax_m(pg, vout4, vzero), valpha);
          vout5 = svmin_m(pg, svmax_m(pg, vout5, vzero), valpha);
          vout6 = svmin_m(pg, svmax_m(pg, vout6, vzero), valpha);
          vout7 = svmin_m(pg, svmax_m(pg, vout7, vzero), valpha);
        } else if (flag_act == 3) {  // leakyrelu
          vout0 = svadd_z(pg,
                          svmul_z(pg, svmin_z(pg, vout0, vzero), valpha),
                          svmax_z(pg, vout0, vzero));
          vout1 = svadd_z(pg,
                          svmul_z(pg, svmin_z(pg, vout1, vzero), valpha),
                          svmax_z(pg, vout1, vzero));
          vout2 = svadd_z(pg,
                          svmul_z(pg, svmin_z(pg, vout2, vzero), valpha),
                          svmax_z(pg, vout2, vzero));
          vout3 = svadd_z(pg,
                          svmul_z(pg, svmin_z(pg, vout3, vzero), valpha),
                          svmax_z(pg, vout3, vzero));
          vout4 = svadd_z(pg,
                          svmul_z(pg, svmin_z(pg, vout4, vzero), valpha),
                          svmax_z(pg, vout4, vzero));
          vout5 = svadd_z(pg,
                          svmul_z(pg, svmin_z(pg, vout5, vzero), valpha),
                          svmax_z(pg, vout5, vzero));
          vout6 = svadd_z(pg,
                          svmul_z(pg, svmin_z(pg, vout6, vzero), valpha),
                          svmax_z(pg, vout6, vzero));
          vout0 = svadd_z(pg,
                          svmul_z(pg, svmin_z(pg, vout7, vzero), valpha),
                          svmax_z(pg, vout7, vzero));
        } else if (flag_act == 4) {  // hardswish
          auto voffset =
              svdup_n(static_cast<Dtype>(act_param.hard_swish_offset));
          auto vthreshold =
              svdup_n(static_cast<Dtype>(act_param.hard_swish_threshold));
          auto voff0 = svadd_m(pg, vout0, voffset);
          auto voff1 = svadd_m(pg, vout1, voffset);
          auto voff2 = svadd_m(pg, vout2, voffset);
          auto voff3 = svadd_m(pg, vout3, voffset);
          auto voff4 = svadd_m(pg, vout4, voffset);
          auto voff5 = svadd_m(pg, vout5, voffset);
          auto voff6 = svadd_m(pg, vout6, voffset);
          auto voff7 = svadd_m(pg, vout7, voffset);
          auto vscale0 = svadd_m(pg, vout0, valpha);
          auto vscale1 = svadd_m(pg, vout1, valpha);
          auto vscale2 = svadd_m(pg, vout2, valpha);
          auto vscale3 = svadd_m(pg, vout3, valpha);
          auto vscale4 = svadd_m(pg, vout4, valpha);
          auto vscale5 = svadd_m(pg, vout5, valpha);
          auto vscale6 = svadd_m(pg, vout6, valpha);
          auto vscale7 = svadd_m(pg, vout7, valpha);
          auto vmax0 = svmin_m(pg, svmax_m(pg, voff0, vzero), vthreshold);
          auto vmax1 = svmin_m(pg, svmax_m(pg, voff1, vzero), vthreshold);
          auto vmax2 = svmin_m(pg, svmax_m(pg, voff2, vzero), vthreshold);
          auto vmax3 = svmin_m(pg, svmax_m(pg, voff3, vzero), vthreshold);
          auto vmax4 = svmin_m(pg, svmax_m(pg, voff4, vzero), vthreshold);
          auto vmax5 = svmin_m(pg, svmax_m(pg, voff5, vzero), vthreshold);
          auto vmax6 = svmin_m(pg, svmax_m(pg, voff6, vzero), vthreshold);
          auto vmax7 = svmin_m(pg, svmax_m(pg, voff7, vzero), vthreshold);
          vout0 = svmul_m(pg, vscale0, vmax0);
          vout1 = svmul_m(pg, vscale1, vmax1);
          vout2 = svmul_m(pg, vscale2, vmax2);
          vout3 = svmul_m(pg, vscale3, vmax3);
          vout4 = svmul_m(pg, vscale4, vmax0);
          vout5 = svmul_m(pg, vscale5, vmax1);
          vout6 = svmul_m(pg, vscale6, vmax2);
          vout7 = svmul_m(pg, vscale7, vmax3);
        }
        svst1(pg, c_ptr0, vout0);
        svst1(pg, c_ptr1, vout1);
        svst1(pg, c_ptr2, vout2);
        svst1(pg, c_ptr3, vout3);
        svst1(pg, c_ptr4, vout4);
        svst1(pg, c_ptr5, vout5);
        svst1(pg, c_ptr6, vout6);
        svst1(pg, c_ptr7, vout7);
        b_ptr += svcnt<Dtype>();
        a_ptr7 = a_ptr6;
        a_ptr6 = a_ptr5;
        a_ptr5 = a_ptr4;
        a_ptr4 = a_ptr3;
        a_ptr3 = a_ptr2;
        a_ptr2 = a_ptr1;
        a_ptr1 = a_ptr0;
        a_ptr0 -= K;
        c_ptr0 += svcnt<Dtype>();
        c_ptr1 += svcnt<Dtype>();
        c_ptr2 += svcnt<Dtype>();
        c_ptr3 += svcnt<Dtype>();
        c_ptr4 += svcnt<Dtype>();
        c_ptr5 += svcnt<Dtype>();
        c_ptr6 += svcnt<Dtype>();
        c_ptr7 += svcnt<Dtype>();
      }
    }
    LITE_PARALLEL_COMMON_END();
    // m < 8
    auto index_y = bblocks * MBLOCK_SVE;
    if (flag_p_remain) {
      const Dtype *a_ptr = A + index_y * lda;
      Dtype *c_ptr = C + index_y * ldc;
      if (remain >= 4) {
        const Dtype *a_ptr_l = a_ptr;
        const Dtype *b_ptr = B;
        Dtype *c_ptr0 = c_ptr;
        Dtype *c_ptr1 = c_ptr0 + ldc;
        Dtype *c_ptr2 = c_ptr1 + ldc;
        Dtype *c_ptr3 = c_ptr2 + ldc;
        const Dtype *a_ptr0 = a_ptr_l;
        const Dtype *a_ptr1 = a_ptr0 + lda;
        const Dtype *a_ptr2 = a_ptr1 + lda;
        const Dtype *a_ptr3 = a_ptr2 + lda;
        auto vbias0 = svdup_n(static_cast<Dtype>(0.f));
        auto vbias1 = svdup_n(static_cast<Dtype>(0.f));
        auto vbias2 = svdup_n(static_cast<Dtype>(0.f));
        auto vbias3 = svdup_n(static_cast<Dtype>(0.f));
        if (has_bias) {
          vbias0 = svdup_n(static_cast<Dtype>(bias[index_y]));
          vbias1 = svdup_n(static_cast<Dtype>(bias[index_y + 1]));
          vbias2 = svdup_n(static_cast<Dtype>(bias[index_y + 2]));
          vbias3 = svdup_n(static_cast<Dtype>(bias[index_y + 3]));
        }

        for (int x = 0; x < N; x += svcnt<Dtype>()) {
          svbool_t pg = svwhilelt<Dtype>(x, N);
          const Dtype *b_ptr_l = b_ptr;
          auto vout0 = vbias0;
          auto vout1 = vbias1;
          auto vout2 = vbias2;
          auto vout3 = vbias3;
          for (int k = 0; k < K; k++) {
            auto vb = svld1(pg, b_ptr_l);
            auto va0 = svdup_n(static_cast<Dtype>(*a_ptr0));
            auto va1 = svdup_n(static_cast<Dtype>(*a_ptr1));
            auto va2 = svdup_n(static_cast<Dtype>(*a_ptr2));
            auto va3 = svdup_n(static_cast<Dtype>(*a_ptr3));
            vout0 = svmla_m(pg, vout0, vb, va0);
            vout1 = svmla_m(pg, vout1, vb, va1);
            vout2 = svmla_m(pg, vout2, vb, va2);
            vout3 = svmla_m(pg, vout3, vb, va3);
            b_ptr_l += ldb;
            a_ptr0++;
            a_ptr1++;
            a_ptr2++;
            a_ptr3++;
          }
          if (flag_act == 0) {
          } else if (flag_act == 1) {  // relu
            vout0 = svmax_m(pg, vout0, vzero);
            vout1 = svmax_m(pg, vout1, vzero);
            vout2 = svmax_m(pg, vout2, vzero);
            vout3 = svmax_m(pg, vout3, vzero);
          } else if (flag_act == 2) {  // relu6
            vout0 = svmax_m(pg, vout0, vzero);
            vout1 = svmax_m(pg, vout1, vzero);
            vout2 = svmax_m(pg, vout2, vzero);
            vout3 = svmax_m(pg, vout3, vzero);
            vout0 = svmin_m(pg, vout0, valpha);
            vout1 = svmin_m(pg, vout1, valpha);
            vout2 = svmin_m(pg, vout2, valpha);
            vout3 = svmin_m(pg, vout3, valpha);
          } else if (flag_act == 3) {  // leakyrelu
            auto vres0_0 = svcmpge(pg, vout0, vzero);
            auto vres0_1 = svmul_m(pg, vout0, valpha);
            auto vres1_0 = svcmpge(pg, vout1, vzero);
            auto vres1_1 = svmul_m(pg, vout1, valpha);
            auto vres2_0 = svcmpge(pg, vout2, vzero);
            auto vres2_1 = svmul_m(pg, vout2, valpha);
            auto vres3_0 = svcmpge(pg, vout3, vzero);
            auto vres3_1 = svmul_m(pg, vout3, valpha);
          } else if (flag_act == 4) {  // hardswish
            auto voffset =
                svdup_n(static_cast<Dtype>(act_param.hard_swish_offset));
            auto vthreshold =
                svdup_n(static_cast<Dtype>(act_param.hard_swish_threshold));
            auto voff0 = svadd_m(pg, vout0, voffset);
            auto voff1 = svadd_m(pg, vout1, voffset);
            auto voff2 = svadd_m(pg, vout2, voffset);
            auto voff3 = svadd_m(pg, vout3, voffset);
            auto vscale0 = svadd_m(pg, vout0, valpha);
            auto vscale1 = svadd_m(pg, vout1, valpha);
            auto vscale2 = svadd_m(pg, vout2, valpha);
            auto vscale3 = svadd_m(pg, vout3, valpha);
            auto vmax0 = svmin_m(pg, svmax_m(pg, voff0, vzero), vthreshold);
            auto vmax1 = svmin_m(pg, svmax_m(pg, voff1, vzero), vthreshold);
            auto vmax2 = svmin_m(pg, svmax_m(pg, voff2, vzero), vthreshold);
            auto vmax3 = svmin_m(pg, svmax_m(pg, voff3, vzero), vthreshold);
            vout0 = svmul_m(pg, vscale0, vmax0);
            vout1 = svmul_m(pg, vscale1, vmax1);
            vout2 = svmul_m(pg, vscale2, vmax2);
            vout3 = svmul_m(pg, vscale3, vmax3);
          }
          svst1(pg, c_ptr0, vout0);
          svst1(pg, c_ptr1, vout1);
          svst1(pg, c_ptr2, vout2);
          svst1(pg, c_ptr3, vout3);
          a_ptr3 = a_ptr2;
          a_ptr2 = a_ptr1;
          a_ptr1 = a_ptr0;
          a_ptr0 -= K;
          b_ptr += svcnt<Dtype>();
          c_ptr0 += svcnt<Dtype>();
          c_ptr1 += svcnt<Dtype>();
          c_ptr2 += svcnt<Dtype>();
          c_ptr3 += svcnt<Dtype>();
        }
        a_ptr += 4 * lda;
        c_ptr += 4 * ldc;
        index_y += 4;
        remain -= 4;
      }
      if (remain >= 2) {
        const Dtype *a_ptr_l = a_ptr;
        const Dtype *b_ptr = B;
        Dtype *c_ptr0 = c_ptr;
        Dtype *c_ptr1 = c_ptr0 + ldc;
        const Dtype *a_ptr0 = a_ptr_l;
        const Dtype *a_ptr1 = a_ptr0 + lda;
        auto vbias0 = svdup_n(static_cast<Dtype>(0.f));
        auto vbias1 = svdup_n(static_cast<Dtype>(0.f));
        if (has_bias) {
          vbias0 = svdup_n(static_cast<Dtype>(bias[index_y]));
          vbias1 = svdup_n(static_cast<Dtype>(bias[index_y + 1]));
        }

        for (int x = 0; x < N; x += svcnt<Dtype>()) {
          svbool_t pg = svwhilelt<Dtype>(x, N);
          const Dtype *b_ptr_l = b_ptr;
          auto vout0 = vbias0;
          auto vout1 = vbias1;
          for (int k = 0; k < K; k++) {
            auto vb = svld1(pg, b_ptr_l);
            auto va0 = svdup_n(static_cast<Dtype>(*a_ptr0));
            auto va1 = svdup_n(static_cast<Dtype>(*a_ptr1));
            vout0 = svmla_m(pg, vout0, vb, va0);
            vout1 = svmla_m(pg, vout1, vb, va1);
            b_ptr_l += ldb;
            a_ptr0++;
            a_ptr1++;
          }
          if (flag_act == 0) {
          } else if (flag_act == 1) {  // relu
            vout0 = svmax_m(pg, vout0, vzero);
            vout1 = svmax_m(pg, vout1, vzero);
          } else if (flag_act == 2) {  // relu6
            vout0 = svmax_m(pg, vout0, vzero);
            vout1 = svmax_m(pg, vout1, vzero);
            vout0 = svmin_m(pg, vout0, valpha);
            vout1 = svmin_m(pg, vout1, valpha);
          } else if (flag_act == 3) {  // leakyrelu
            auto vres0_0 = svcmpge(pg, vout0, vzero);
            auto vres0_1 = svmul_m(pg, vout0, valpha);
            auto vres1_0 = svcmpge(pg, vout1, vzero);
            auto vres1_1 = svmul_m(pg, vout1, valpha);
          } else if (flag_act == 4) {  // hardswish
            auto voffset =
                svdup_n(static_cast<Dtype>(act_param.hard_swish_offset));
            auto vthreshold =
                svdup_n(static_cast<Dtype>(act_param.hard_swish_threshold));
            auto voff0 = svadd_m(pg, vout0, voffset);
            auto voff1 = svadd_m(pg, vout1, voffset);
            auto vscale0 = svadd_m(pg, vout0, valpha);
            auto vscale1 = svadd_m(pg, vout1, valpha);
            auto vmax0 = svmin_m(pg, svmax_m(pg, voff0, vzero), vthreshold);
            auto vmax1 = svmin_m(pg, svmax_m(pg, voff1, vzero), vthreshold);
            vout0 = svmul_m(pg, vscale0, vmax0);
            vout1 = svmul_m(pg, vscale1, vmax1);
          }
          a_ptr1 = a_ptr0;
          a_ptr0 -= K;
          svst1(pg, c_ptr0, vout0);
          svst1(pg, c_ptr1, vout1);
          b_ptr += svcnt<Dtype>();
          c_ptr0 += svcnt<Dtype>();
          c_ptr1 += svcnt<Dtype>();
        }
        a_ptr += 2 * lda;
        c_ptr += 2 * ldc;
        index_y += 2;
        remain -= 2;
      }
      if (remain > 0) {
        const Dtype *a_ptr_l = a_ptr;
        const Dtype *b_ptr = B;
        Dtype *c_ptr0 = c_ptr;
        const Dtype *a_ptr0 = a_ptr_l;
        auto vbias0 = svdup_n(static_cast<Dtype>(0.f));
        if (has_bias) {
          vbias0 = svdup_n(static_cast<Dtype>(bias[index_y]));
        }

        for (int x = 0; x < N; x += svcnt<Dtype>()) {
          svbool_t pg = svwhilelt<Dtype>(x, N);
          const Dtype *b_ptr_l = b_ptr;
          auto vout0 = vbias0;
          for (int k = 0; k < K; k++) {
            auto vb = svld1(pg, b_ptr_l);
            auto va0 = svdup_n(static_cast<Dtype>(*a_ptr0));
            vout0 = svmla_m(pg, vout0, vb, va0);
            b_ptr_l += ldb;
            a_ptr0++;
          }
          if (flag_act == 0) {
          } else if (flag_act == 1) {  // relu
            vout0 = svmax_m(pg, vout0, vzero);
          } else if (flag_act == 2) {  // relu6
            vout0 = svmax_m(pg, vout0, vzero);
            vout0 = svmin_m(pg, vout0, valpha);
          } else if (flag_act == 3) {  // leakyrelu
            auto vres0_0 = svcmpge(pg, vout0, vzero);
            auto vres0_1 = svmul_m(pg, vout0, valpha);
          } else if (flag_act == 4) {  // hardswish
            auto voffset =
                svdup_n(static_cast<Dtype>(act_param.hard_swish_offset));
            auto vthreshold =
                svdup_n(static_cast<Dtype>(act_param.hard_swish_threshold));
            auto voff0 = svadd_m(pg, vout0, voffset);
            auto vscale0 = svadd_m(pg, vout0, valpha);
            auto vmax0 = svmin_m(pg, svmax_m(pg, voff0, vzero), vthreshold);
            vout0 = svmul_m(pg, vscale0, vmax0);
          }
          a_ptr0 -= K;
          svst1(pg, c_ptr0, vout0);
          b_ptr += svcnt<Dtype>();
          c_ptr0 += svcnt<Dtype>();
        }
        a_ptr += lda;
        c_ptr += ldc;
      }
    }
  }
}

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
#define OUT_INIT(index, bias_local)                   \
  auto vout0_##index##_ = svdup_n_f32(bias_local[0]); \
  auto vout1_##index##_ = svdup_n_f32(bias_local[1]); \
  auto vout2_##index##_ = svdup_n_f32(bias_local[2]); \
  auto vout3_##index##_ = svdup_n_f32(bias_local[3]); \
  auto vout4_##index##_ = svdup_n_f32(bias_local[4]); \
  auto vout5_##index##_ = svdup_n_f32(bias_local[5]); \
  auto vout6_##index##_ = svdup_n_f32(bias_local[6]); \
  auto vout7_##index##_ = svdup_n_f32(bias_local[7]);

#define COMPUTE_MLA(index, vb)                                    \
  vout0_##index##_ =                                              \
      svmla_n_f32_z(all_true_pg, vout0_##index##_, vb, a_ptr[0]); \
  vout1_##index##_ =                                              \
      svmla_n_f32_z(all_true_pg, vout1_##index##_, vb, a_ptr[1]); \
  vout2_##index##_ =                                              \
      svmla_n_f32_z(all_true_pg, vout2_##index##_, vb, a_ptr[2]); \
  vout3_##index##_ =                                              \
      svmla_n_f32_z(all_true_pg, vout3_##index##_, vb, a_ptr[3]); \
  vout4_##index##_ =                                              \
      svmla_n_f32_z(all_true_pg, vout4_##index##_, vb, a_ptr[4]); \
  vout5_##index##_ =                                              \
      svmla_n_f32_z(all_true_pg, vout5_##index##_, vb, a_ptr[5]); \
  vout6_##index##_ =                                              \
      svmla_n_f32_z(all_true_pg, vout6_##index##_, vb, a_ptr[6]); \
  vout7_##index##_ = svmla_n_f32_z(all_true_pg, vout7_##index##_, vb, a_ptr[7]);

#define STORE_OUT(offset, index)                             \
  svst1_f32(all_true_pg, c_ptr0 + offset, vout0_##index##_); \
  svst1_f32(all_true_pg, c_ptr1 + offset, vout1_##index##_); \
  svst1_f32(all_true_pg, c_ptr2 + offset, vout2_##index##_); \
  svst1_f32(all_true_pg, c_ptr3 + offset, vout3_##index##_); \
  svst1_f32(all_true_pg, c_ptr4 + offset, vout4_##index##_); \
  svst1_f32(all_true_pg, c_ptr5 + offset, vout5_##index##_); \
  svst1_f32(all_true_pg, c_ptr6 + offset, vout6_##index##_); \
  svst1_f32(all_true_pg, c_ptr7 + offset, vout7_##index##_);

#define VMAXMIN_OUT(prefix, index, vzero)                                  \
  vout0_##index##_ = sv##prefix##_m(all_true_pg, vout0_##index##_, vzero); \
  vout1_##index##_ = sv##prefix##_m(all_true_pg, vout1_##index##_, vzero); \
  vout2_##index##_ = sv##prefix##_m(all_true_pg, vout2_##index##_, vzero); \
  vout3_##index##_ = sv##prefix##_m(all_true_pg, vout3_##index##_, vzero); \
  vout4_##index##_ = sv##prefix##_m(all_true_pg, vout4_##index##_, vzero); \
  vout5_##index##_ = sv##prefix##_m(all_true_pg, vout5_##index##_, vzero); \
  vout6_##index##_ = sv##prefix##_m(all_true_pg, vout6_##index##_, vzero); \
  vout7_##index##_ = sv##prefix##_m(all_true_pg, vout7_##index##_, vzero);

#define COMPUTE_ASM_0           \
  "fmla z8.s,  z4.s, z0.s[0]\n" \
  "fmla z11.s, z4.s, z0.s[1]\n" \
  "fmla z14.s, z4.s, z0.s[2]\n" \
  "fmla z17.s, z4.s, z0.s[3]\n" \
  "fmla z20.s, z4.s, z1.s[0]\n" \
  "fmla z23.s, z4.s, z1.s[1]\n" \
  "fmla z26.s, z4.s, z1.s[2]\n" \
  "fmla z29.s, z4.s, z1.s[3]\n"

#define COMPUTE_ASM_1           \
  "fmla z9.s,  z5.s, z0.s[0]\n" \
  "fmla z12.s, z5.s, z0.s[1]\n" \
  "fmla z15.s, z5.s, z0.s[2]\n" \
  "fmla z18.s, z5.s, z0.s[3]\n" \
  "fmla z21.s, z5.s, z1.s[0]\n" \
  "fmla z24.s, z5.s, z1.s[1]\n" \
  "fmla z27.s, z5.s, z1.s[2]\n" \
  "fmla z30.s, z5.s, z1.s[3]\n"

#define COMPUTE_ASM_2           \
  "fmla z10.s, z6.s, z0.s[0]\n" \
  "fmla z13.s, z6.s, z0.s[1]\n" \
  "fmla z16.s, z6.s, z0.s[2]\n" \
  "fmla z19.s, z6.s, z0.s[3]\n" \
  "fmla z22.s, z6.s, z1.s[0]\n" \
  "fmla z25.s, z6.s, z1.s[1]\n" \
  "fmla z28.s, z6.s, z1.s[2]\n" \
  "fmla z31.s, z6.s, z1.s[3]\n"

#define VMAXMIN_ASM_0(inst)                                                   \
  #inst                                                                       \
      " z8.s,  p0/m, z8.s, z0.s\n" #inst " z11.s, p0/m, z11.s, z0.s\n" #inst  \
      " z14.s, p0/m, z14.s, z0.s\n" #inst " z17.s, p0/m, z17.s, z0.s\n" #inst \
      " z20.s, p0/m, z20.s, z0.s\n" #inst " z23.s, p0/m, z23.s, z0.s\n" #inst \
      " z26.s, p0/m, z26.s, z0.s\n" #inst " z29.s, p0/m, z29.s, z0.s\n"

#define VMAXMIN_ASM_1(inst)                                                   \
  #inst                                                                       \
      " z9.s,  p0/m, z9.s, z0.s\n" #inst " z12.s, p0/m, z12.s, z0.s\n" #inst  \
      " z15.s, p0/m, z15.s, z0.s\n" #inst " z18.s, p0/m, z18.s, z0.s\n" #inst \
      " z21.s, p0/m, z21.s, z0.s\n" #inst " z24.s, p0/m, z24.s, z0.s\n" #inst \
      " z27.s, p0/m, z27.s, z0.s\n" #inst " z30.s, p0/m, z30.s, z0.s\n"

#define VMAXMIN_ASM_2(inst)                                                   \
  #inst                                                                       \
      " z10.s, p0/m, z10.s, z0.s\n" #inst " z13.s, p0/m, z13.s, z0.s\n" #inst \
      " z16.s, p0/m, z16.s, z0.s\n" #inst " z19.s, p0/m, z19.s, z0.s\n" #inst \
      " z22.s, p0/m, z22.s, z0.s\n" #inst " z25.s, p0/m, z25.s, z0.s\n" #inst \
      " z28.s, p0/m, z28.s, z0.s\n" #inst " z31.s, p0/m, z31.s, z0.s\n"

#define VLEAKY_ASM_0                                 \
  "movprfx z2, z8\n  fmin z2.s,  p0/m, z2.s, z0.s\n" \
  "movprfx z3, z11\n fmin z3.s,  p0/m, z3.s, z0.s\n" \
  "movprfx z4, z14\n fmin z4.s,  p0/m, z4.s, z0.s\n" \
  "movprfx z5, z17\n fmin z5.s,  p0/m, z5.s, z0.s\n" \
  "fmax z8.s,  p0/m, z8.s,  z0.s\n"                  \
  "fmax z11.s, p0/m, z11.s, z0.s\n"                  \
  "fmax z14.s, p0/m, z14.s, z0.s\n"                  \
  "fmax z17.s, p0/m, z17.s, z0.s\n"                  \
  "fmul z2.s,  p0/m, z2.s,  z1.s\n"                  \
  "fmul z3.s,  p0/m, z3.s,  z1.s\n"                  \
  "fmul z4.s,  p0/m, z4.s,  z1.s\n"                  \
  "fmul z5.s,  p0/m, z5.s,  z1.s\n"                  \
  "fadd z8.s,  p0/m, z8.s,  z2.s\n"                  \
  "fadd z11.s, p0/m, z11.s, z3.s\n"                  \
  "fadd z14.s, p0/m, z14.s, z4.s\n"                  \
  "fadd z17.s, p0/m, z17.s, z5.s\n"                  \
  "movprfx z2, z20\n fmin z2.s,  p0/m, z2.s, z0.s\n" \
  "movprfx z3, z23\n fmin z3.s,  p0/m, z3.s, z0.s\n" \
  "movprfx z4, z26\n fmin z4.s,  p0/m, z4.s, z0.s\n" \
  "movprfx z5, z29\n fmin z5.s,  p0/m, z5.s, z0.s\n" \
  "fmax z20.s, p0/m, z20.s, z0.s\n"                  \
  "fmax z23.s, p0/m, z23.s, z0.s\n"                  \
  "fmax z26.s, p0/m, z26.s, z0.s\n"                  \
  "fmax z29.s, p0/m, z29.s, z0.s\n"                  \
  "fmul z2.s,  p0/m, z2.s,  z1.s\n"                  \
  "fmul z3.s,  p0/m, z3.s,  z1.s\n"                  \
  "fmul z4.s,  p0/m, z4.s,  z1.s\n"                  \
  "fmul z5.s,  p0/m, z5.s,  z1.s\n"                  \
  "fadd z20.s, p0/m, z20.s, z2.s\n"                  \
  "fadd z23.s, p0/m, z23.s, z3.s\n"                  \
  "fadd z26.s, p0/m, z26.s, z4.s\n"                  \
  "fadd z29.s, p0/m, z29.s, z5.s\n"

#define VLEAKY_ASM_1                                 \
  "movprfx z2, z9\n  fmax z2.s,  p0/m, z2.s, z0.s\n" \
  "movprfx z3, z12\n fmax z3.s,  p0/m, z3.s, z0.s\n" \
  "movprfx z4, z15\n fmax z4.s,  p0/m, z4.s, z0.s\n" \
  "movprfx z5, z18\n fmax z5.s,  p0/m, z5.s, z0.s\n" \
  "fmin z9.s,  p0/m, z9.s,  z0.s\n"                  \
  "fmin z12.s, p0/m, z12.s, z0.s\n"                  \
  "fmin z15.s, p0/m, z15.s, z0.s\n"                  \
  "fmin z18.s, p0/m, z18.s, z0.s\n"                  \
  "fmul z9.s,  p0/m, z9.s,  z1.s\n"                  \
  "fmul z12.s, p0/m, z12.s, z1.s\n"                  \
  "fmul z15.s, p0/m, z15.s, z1.s\n"                  \
  "fmul z18.s, p0/m, z18.s, z1.s\n"                  \
  "fadd z9.s,  p0/m, z9.s,  z2.s\n"                  \
  "fadd z12.s, p0/m, z12.s, z3.s\n"                  \
  "fadd z15.s, p0/m, z15.s, z4.s\n"                  \
  "fadd z18.s, p0/m, z18.s, z5.s\n"                  \
  "movprfx z2, z21\n fmax z2.s,  p0/m, z2.s, z0.s\n" \
  "movprfx z3, z24\n fmax z3.s,  p0/m, z3.s, z0.s\n" \
  "movprfx z4, z27\n fmax z4.s,  p0/m, z4.s, z0.s\n" \
  "movprfx z5, z30\n fmax z5.s,  p0/m, z5.s, z0.s\n" \
  "fmin z21.s, p0/m, z21.s, z0.s\n"                  \
  "fmin z24.s, p0/m, z24.s, z0.s\n"                  \
  "fmin z27.s, p0/m, z27.s, z0.s\n"                  \
  "fmin z30.s, p0/m, z30.s, z0.s\n"                  \
  "fmul z21.s, p0/m, z21.s, z1.s\n"                  \
  "fmul z24.s, p0/m, z24.s, z1.s\n"                  \
  "fmul z27.s, p0/m, z27.s, z1.s\n"                  \
  "fmul z30.s, p0/m, z30.s, z1.s\n"                  \
  "fadd z21.s, p0/m, z21.s, z2.s\n"                  \
  "fadd z24.s, p0/m, z24.s, z3.s\n"                  \
  "fadd z27.s, p0/m, z27.s, z4.s\n"                  \
  "fadd z30.s, p0/m, z30.s, z5.s\n"

#define VLEAKY_ASM_2                                 \
  "movprfx z2, z10\n fmax z2.s,  p0/m, z2.s, z0.s\n" \
  "movprfx z3, z13\n fmax z3.s,  p0/m, z3.s, z0.s\n" \
  "movprfx z4, z16\n fmax z4.s,  p0/m, z4.s, z0.s\n" \
  "movprfx z5, z19\n fmax z5.s,  p0/m, z5.s, z0.s\n" \
  "fmin z10.s, p0/m, z10.s, z0.s\n"                  \
  "fmin z13.s, p0/m, z13.s, z0.s\n"                  \
  "fmin z16.s, p0/m, z16.s, z0.s\n"                  \
  "fmin z19.s, p0/m, z19.s, z0.s\n"                  \
  "fmul z10.s, p0/m, z10.s, z1.s\n"                  \
  "fmul z13.s, p0/m, z13.s, z1.s\n"                  \
  "fmul z16.s, p0/m, z16.s, z1.s\n"                  \
  "fmul z19.s, p0/m, z19.s, z1.s\n"                  \
  "fadd z10.s, p0/m, z10.s, z2.s\n"                  \
  "fadd z13.s, p0/m, z13.s, z3.s\n"                  \
  "fadd z16.s, p0/m, z16.s, z4.s\n"                  \
  "fadd z19.s, p0/m, z19.s, z5.s\n"                  \
  "movprfx z2, z22\n fmax z2.s,  p0/m, z2.s, z0.s\n" \
  "movprfx z3, z25\n fmax z3.s,  p0/m, z3.s, z0.s\n" \
  "movprfx z4, z28\n fmax z4.s,  p0/m, z4.s, z0.s\n" \
  "movprfx z5, z31\n fmax z5.s,  p0/m, z5.s, z0.s\n" \
  "fmin z22.s, p0/m, z22.s, z0.s\n"                  \
  "fmin z25.s, p0/m, z25.s, z0.s\n"                  \
  "fmin z28.s, p0/m, z28.s, z0.s\n"                  \
  "fmin z31.s, p0/m, z31.s, z0.s\n"                  \
  "fmul z22.s, p0/m, z22.s, z1.s\n"                  \
  "fmul z25.s, p0/m, z25.s, z1.s\n"                  \
  "fmul z28.s, p0/m, z28.s, z1.s\n"                  \
  "fmul z31.s, p0/m, z31.s, z1.s\n"                  \
  "fadd z22.s, p0/m, z22.s, z2.s\n"                  \
  "fadd z25.s, p0/m, z25.s, z3.s\n"                  \
  "fadd z28.s, p0/m, z28.s, z4.s\n"                  \
  "fadd z31.s, p0/m, z31.s, z5.s\n"

// z1=offset z2=scale z3=threshold
#define VHARDSWISH_ASM_0                               \
  "movprfx z4, z8\n  fadd z4.s,  p0/m, z4.s, z1.s\n"   \
  "movprfx z5, z11\n fadd z5.s,  p0/m, z5.s, z1.s\n"   \
  "movprfx z6, z14\n fadd z6.s,  p0/m, z6.s, z1.s\n"   \
  "movprfx z7, z17\n fadd z7.s,  p0/m, z7.s, z1.s\n"   \
  "fmul z8.s,  p0/m, z8.s,   z2.s\n"                   \
  "fmul z11.s, p0/m, z11.s,  z2.s\n"                   \
  "fmul z14.s, p0/m, z14.s,  z2.s\n"                   \
  "fmul z17.s, p0/m, z17.s,  z2.s\n"                   \
  "fmax z4.s,  p0/m, z4.s,   z0.s\n"                   \
  "fmax z5.s,  p0/m, z5.s,   z0.s\n"                   \
  "fmax z6.s,  p0/m, z6.s,   z0.s\n"                   \
  "fmax z7.s,  p0/m, z7.s,   z0.s\n"                   \
  "fmin z4.s,  p0/m, z4.s,   z3.s\n"                   \
  "fmin z5.s,  p0/m, z5.s,   z3.s\n"                   \
  "fmin z6.s,  p0/m, z6.s,   z3.s\n"                   \
  "fmin z7.s,  p0/m, z7.s,   z3.s\n"                   \
  "fmul z8.s,  p0/m, z8.s,  z4.s\n"                    \
  "fmul z11.s, p0/m, z11.s, z5.s\n"                    \
  "fmul z14.s, p0/m, z14.s, z6.s\n"                    \
  "fmul z17.s, p0/m, z17.s, z7.s\n"                    \
  "movprfx z4, z20\n fadd z4.s,  p0/m, z4.s, z1.s\n"   \
  "movprfx z5, z23\n fadd z5.s,  p0/m, z5.s, z1.s\n"   \
  "movprfx z6, z26\n fadd z6.s,  p0/m, z6.s, z1.s\n"   \
  "movprfx z7, z29\n fadd z7.s,  p0/m, z7.s,   z1.s\n" \
  "fmul z20.s, p0/m, z20.s,  z2.s\n"                   \
  "fmul z23.s, p0/m, z23.s,  z2.s\n"                   \
  "fmul z26.s, p0/m, z26.s,  z2.s\n"                   \
  "fmul z29.s, p0/m, z29.s,  z2.s\n"                   \
  "fmax z4.s,  p0/m, z4.s,   z0.s\n"                   \
  "fmax z5.s,  p0/m, z5.s,   z0.s\n"                   \
  "fmax z6.s,  p0/m, z6.s,   z0.s\n"                   \
  "fmax z7.s,  p0/m, z7.s,   z0.s\n"                   \
  "fmin z4.s,  p0/m, z4.s,   z3.s\n"                   \
  "fmin z5.s,  p0/m, z5.s,   z3.s\n"                   \
  "fmin z6.s,  p0/m, z6.s,   z3.s\n"                   \
  "fmin z7.s,  p0/m, z7.s,   z3.s\n"                   \
  "fmul z20.s, p0/m, z20.s,  z4.s\n"                   \
  "fmul z23.s, p0/m, z23.s, z5.s\n"                    \
  "fmul z26.s, p0/m, z26.s, z6.s\n"                    \
  "fmul z29.s, p0/m, z29.s, z7.s\n"

#define VHARDSWISH_ASM_1                               \
  "movprfx z4, z9\n  fadd z4.s,  p0/m, z4.s, z1.s\n"   \
  "movprfx z5, z12\n fadd z5.s,  p0/m, z5.s, z1.s\n"   \
  "movprfx z6, z15\n fadd z6.s,  p0/m, z6.s, z1.s\n"   \
  "movprfx z7, z18\n fadd z7.s,  p0/m, z7.s,   z1.s\n" \
  "fmul z9.s,  p0/m, z9.s,   z2.s\n"                   \
  "fmul z12.s, p0/m, z12.s,  z2.s\n"                   \
  "fmul z15.s, p0/m, z15.s,  z2.s\n"                   \
  "fmul z18.s, p0/m, z18.s,  z2.s\n"                   \
  "fmax z4.s,  p0/m, z4.s,   z0.s\n"                   \
  "fmax z5.s,  p0/m, z5.s,   z0.s\n"                   \
  "fmax z6.s,  p0/m, z6.s,   z0.s\n"                   \
  "fmax z7.s,  p0/m, z7.s,   z0.s\n"                   \
  "fmin z4.s,  p0/m, z4.s,   z3.s\n"                   \
  "fmin z5.s,  p0/m, z5.s,   z3.s\n"                   \
  "fmin z6.s,  p0/m, z6.s,   z3.s\n"                   \
  "fmin z7.s,  p0/m, z7.s,   z3.s\n"                   \
  "fmul z9.s,  p0/m, z9.s,  z4.s\n"                    \
  "fmul z12.s, p0/m, z12.s, z5.s\n"                    \
  "fmul z15.s, p0/m, z15.s, z6.s\n"                    \
  "fmul z18.s, p0/m, z18.s, z7.s\n"                    \
  "movprfx z4, z21\n fadd z4.s,  p0/m, z4.s, z1.s\n"   \
  "movprfx z5, z24\n fadd z5.s,  p0/m, z5.s, z1.s\n"   \
  "movprfx z6, z27\n fadd z6.s,  p0/m, z6.s, z1.s\n"   \
  "movprfx z7, z30\n fadd z7.s,  p0/m, z7.s,   z1.s\n" \
  "fmul z21.s, p0/m, z21.s,   z2.s\n"                  \
  "fmul z24.s, p0/m, z24.s,  z2.s\n"                   \
  "fmul z27.s, p0/m, z27.s,  z2.s\n"                   \
  "fmul z30.s, p0/m, z30.s,  z2.s\n"                   \
  "fmax z4.s,  p0/m, z4.s,   z0.s\n"                   \
  "fmax z5.s,  p0/m, z5.s,   z0.s\n"                   \
  "fmax z6.s,  p0/m, z6.s,   z0.s\n"                   \
  "fmax z7.s,  p0/m, z7.s,   z0.s\n"                   \
  "fmin z4.s,  p0/m, z4.s,   z3.s\n"                   \
  "fmin z5.s,  p0/m, z5.s,   z3.s\n"                   \
  "fmin z6.s,  p0/m, z6.s,   z3.s\n"                   \
  "fmin z7.s,  p0/m, z7.s,   z3.s\n"                   \
  "fmul z21.s, p0/m, z21.s,  z4.s\n"                   \
  "fmul z24.s, p0/m, z24.s, z5.s\n"                    \
  "fmul z27.s, p0/m, z27.s, z6.s\n"                    \
  "fmul z30.s, p0/m, z30.s, z7.s\n"

#define VHARDSWISH_ASM_2                               \
  "movprfx z4, z10\n fadd z4.s,  p0/m, z4.s, z1.s\n"   \
  "movprfx z5, z13\n fadd z5.s,  p0/m, z5.s, z1.s\n"   \
  "movprfx z6, z16\n fadd z6.s,  p0/m, z6.s, z1.s\n"   \
  "movprfx z7, z19\n fadd z7.s,  p0/m, z7.s,   z1.s\n" \
  "fmul z10.s, p0/m, z10.s,   z2.s\n"                  \
  "fmul z13.s, p0/m, z13.s,  z2.s\n"                   \
  "fmul z16.s, p0/m, z16.s,  z2.s\n"                   \
  "fmul z19.s, p0/m, z19.s,  z2.s\n"                   \
  "fmax z4.s,  p0/m, z4.s,   z0.s\n"                   \
  "fmax z5.s,  p0/m, z5.s,   z0.s\n"                   \
  "fmax z6.s,  p0/m, z6.s,   z0.s\n"                   \
  "fmax z7.s,  p0/m, z7.s,   z0.s\n"                   \
  "fmin z4.s,  p0/m, z4.s,   z3.s\n"                   \
  "fmin z5.s,  p0/m, z5.s,   z3.s\n"                   \
  "fmin z6.s,  p0/m, z6.s,   z3.s\n"                   \
  "fmin z7.s,  p0/m, z7.s,   z3.s\n"                   \
  "fmul z10.s, p0/m, z10.s,  z4.s\n"                   \
  "fmul z13.s, p0/m, z13.s, z5.s\n"                    \
  "fmul z16.s, p0/m, z16.s, z6.s\n"                    \
  "fmul z19.s, p0/m, z19.s, z7.s\n"                    \
  "movprfx z4, z22\n fadd z4.s,  p0/m, z4.s, z1.s\n"   \
  "movprfx z5, z25\n fadd z5.s,  p0/m, z5.s, z1.s\n"   \
  "movprfx z6, z28\n fadd z6.s,  p0/m, z6.s, z1.s\n"   \
  "movprfx z7, z31\n fadd z7.s,  p0/m, z7.s,   z1.s\n" \
  "fmul z22.s, p0/m, z22.s,   z2.s\n"                  \
  "fmul z25.s, p0/m, z25.s,  z2.s\n"                   \
  "fmul z28.s, p0/m, z28.s,  z2.s\n"                   \
  "fmul z31.s, p0/m, z31.s,  z2.s\n"                   \
  "fmax z4.s,  p0/m, z4.s,   z0.s\n"                   \
  "fmax z5.s,  p0/m, z5.s,   z0.s\n"                   \
  "fmax z6.s,  p0/m, z6.s,   z0.s\n"                   \
  "fmax z7.s,  p0/m, z7.s,   z0.s\n"                   \
  "fmin z4.s,  p0/m, z4.s,   z3.s\n"                   \
  "fmin z5.s,  p0/m, z5.s,   z3.s\n"                   \
  "fmin z6.s,  p0/m, z6.s,   z3.s\n"                   \
  "fmin z7.s,  p0/m, z7.s,   z3.s\n"                   \
  "fmul z22.s, p0/m, z22.s,  z4.s\n"                   \
  "fmul z25.s, p0/m, z25.s, z5.s\n"                    \
  "fmul z28.s, p0/m, z28.s, z6.s\n"                    \
  "fmul z31.s, p0/m, z31.s, z7.s\n"

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
  int x_block = (l2_cache - (MBLOCK_SVE * K * sizeof(float))) /
                (sizeof(float) * (K + MBLOCK_SVE));
  x_block = (x_block == 0) ? 1 : x_block;
  x_block /= NBLOCK_SVE_FP32;
  x_block *= NBLOCK_SVE_FP32;
  int x_num = (N + (x_block - 1)) / x_block;
  x_block = (N + x_num - 1) / x_num;
  x_block = (x_block + NBLOCK_SVE_FP32 - 1) / NBLOCK_SVE_FP32;
  x_block *= NBLOCK_SVE_FP32;
  x_block = x_block < NBLOCK_SVE_FP32 ? NBLOCK_SVE_FP32 : x_block;

  // unroll 2 loop
  int tail_pre = (K & (KBLOCK_SVE - 1));
  int k_pre = ((K + KBLOCK_SVE - 1) / KBLOCK_SVE) - 1;

  bool flag_p_remain = false;
  int remain = 0;
  if (tail_pre == 0) {
    tail_pre = KBLOCK_SVE;
  }

  int has_beta = fabsf(beta) > 1e-8f ? 1 : 0;
  auto vzero = svdup_n_f32(0.f);
  auto valpha = svdup_n_f32(local_alpha);
  const auto all_true_pg = svptrue<float>();

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
            OUT_INIT(0, bias_local)
            for (int i = 0; i < K; i++) {
              auto vb0 = svld1_f32(all_true_pg, b_ptr);
              b_ptr += 4;
              COMPUTE_MLA(0, vb0)
              a_ptr += 8;
            }
            if (flag_act == 0) {
            } else if (flag_act == 1) {
              VMAXMIN_OUT(max, 0, vzero)
            } else if (flag_act == 2) {
              VMAXMIN_OUT(max, 0, vzero)
              VMAXMIN_OUT(min, 0, valpha)
            }
            STORE_OUT(0, 0)
            c_ptr0 += 4;
            c_ptr1 += 4;
            c_ptr2 += 4;
            c_ptr3 += 4;
            c_ptr4 += 4;
            c_ptr5 += 4;
            c_ptr6 += 4;
            c_ptr7 += 4;
          }
          for (int i = 0; i < rem_rem; i++) {
            const float *a_ptr = a_ptr_l;
            auto vout0 = svld1_f32(all_true_pg, bias_local);
            auto vout1 = svld1_f32(all_true_pg, bias_local + 4);
            for (int i = 0; i < K; i++) {
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
              vout0 = svmax_m(all_true_pg, vout0, vzero);
              vout1 = svmax_m(all_true_pg, vout1, vzero);
              vout0 = svmin_m(all_true_pg, vout0, valpha);
              vout1 = svmin_m(all_true_pg, vout1, valpha);
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
          /*OUT_INIT(0, bias_local)
          OUT_INIT(1, bias_local)
          OUT_INIT(2, bias_local)
          for (int i = 0; i < K; i++) {
            auto vb0 = svld1_f32(all_true_pg, b_ptr);
            auto vb1 = svld1_f32(all_true_pg, b_ptr + 4);
            auto vb2 = svld1_f32(all_true_pg, b_ptr + 8);
            b_ptr += 12;
            COMPUTE_MLA(0, vb0)
            COMPUTE_MLA(1, vb1)
            COMPUTE_MLA(2, vb2)
            a_ptr += 8;
          }
          */
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
            COMPUTE_ASM_0
            COMPUTE_ASM_1
            "subs %x[k], %x[k], #1\n"
            COMPUTE_ASM_2
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
            VHARDSWISH_ASM_0
            VHARDSWISH_ASM_1
            VHARDSWISH_ASM_2
            "b 10f\n"
            // relu
            "3: \n"
            VMAXMIN_ASM_0(fmax)
            VMAXMIN_ASM_1(fmax)
            VMAXMIN_ASM_2(fmax)
            "b 10f\n"
            // relu6
            "4: \n"
            "mov z0.s, #0x0    \n"
            VMAXMIN_ASM_0(fmax)
            VMAXMIN_ASM_1(fmax)
            VMAXMIN_ASM_2(fmax)
            "ld1rqw {z0.s}, p0/Z, [%x[alpha]]\n"
            VMAXMIN_ASM_0(fmin)
            VMAXMIN_ASM_1(fmin)
            VMAXMIN_ASM_2(fmin)
            "b 10f\n"
            // leakyrelu
            "5: \n"
            "ld1rqw {z1.s}, p0/Z, [%x[alpha]]\n"
            VLEAKY_ASM_0
            VLEAKY_ASM_1
            VLEAKY_ASM_2
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

          /*
          if (flag_act == 0) {
          } else if (flag_act == 1) {
            VMAXMIN_OUT(max, 0, vzero)
            VMAXMIN_OUT(max, 1, vzero)
            VMAXMIN_OUT(max, 2, vzero)
          } else if (flag_act == 2) {
            VMAXMIN_OUT(max, 0, vzero)
            VMAXMIN_OUT(max, 1, vzero)
            VMAXMIN_OUT(max, 2, vzero)
            VMAXMIN_OUT(min, 0, vzero)
            VMAXMIN_OUT(min, 1, vzero)
            VMAXMIN_OUT(min, 2, vzero)
          }
          STORE_OUT(0, 0)
          STORE_OUT(4, 1)
          STORE_OUT(8, 2)
          c_ptr0 += 12;
          c_ptr1 += 12;
          c_ptr2 += 12;
          c_ptr3 += 12;
          c_ptr4 += 12;
          c_ptr5 += 12;
          c_ptr6 += 12;
          c_ptr7 += 12;
          */
        }
      }
    }
    LITE_PARALLEL_COMMON_END();
  }
}

/// a: m*k  b: k*n  c: m*n
template <typename Dtype>
void sgemm_prepack_sve(GEMM_FUNCS_PARAM(Dtype)) {
  sgemm_prepacked_m8_sve<Dtype>(GEMM_ACTUAL_PARAM);
}

template void sgemm_prepack_sve<float>(GEMM_FUNCS_PARAM(float));
template void sgemm_prepacked_m8_sve<float>(GEMM_FUNCS_PARAM(float));
#ifdef ENABLE_ARM_FP16
template void sgemm_prepack_sve<float16_t>(GEMM_FUNCS_PARAM(float16_t));
template void sgemm_prepacked_m8_sve<float16_t>(GEMM_FUNCS_PARAM(float16_t));
#endif
#undef PREPACKA_FUNCS_PARAM
#undef GEMM_FUNCS_PARAM
#undef PREPACKA_ACTUAL_PARAM
#undef GEMM_ACTUAL_PARAM
}  // namespace sve
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
