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
      auto vout0 = svdup_n(static_cast<Dtype>(0.f));
      auto vout1 = svdup_n(static_cast<Dtype>(0.f));
      auto vout2 = svdup_n(static_cast<Dtype>(0.f));
      auto vout3 = svdup_n(static_cast<Dtype>(0.f));
      auto vout4 = svdup_n(static_cast<Dtype>(0.f));
      auto vout5 = svdup_n(static_cast<Dtype>(0.f));
      auto vout6 = svdup_n(static_cast<Dtype>(0.f));
      auto vout7 = svdup_n(static_cast<Dtype>(0.f));
      if (has_bias) {
        vout0 = svdup_n(static_cast<Dtype>(bias[index_y]));
        vout1 = svdup_n(static_cast<Dtype>(bias[index_y + 1]));
        vout2 = svdup_n(static_cast<Dtype>(bias[index_y + 2]));
        vout3 = svdup_n(static_cast<Dtype>(bias[index_y + 3]));
        vout4 = svdup_n(static_cast<Dtype>(bias[index_y + 4]));
        vout5 = svdup_n(static_cast<Dtype>(bias[index_y + 5]));
        vout6 = svdup_n(static_cast<Dtype>(bias[index_y + 6]));
        vout7 = svdup_n(static_cast<Dtype>(bias[index_y + 7]));
      }

      for (int x = 0; x < N; x += svcnt<Dtype>()) {
        svbool_t pg = svwhilelt<Dtype>(x, N);
        const Dtype *b_ptr_l = b_ptr;
        for (int k = 0; k < K; k++) {
          const Dtype *b_ptr_k = b_ptr_l;
          auto vb = svld1(pg, b_ptr_k);
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
          b_ptr_k += ldb;
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
          vout0 = svmax_m(pg, vout0, vzero);
          vout1 = svmax_m(pg, vout1, vzero);
          vout2 = svmax_m(pg, vout2, vzero);
          vout3 = svmax_m(pg, vout3, vzero);
          vout4 = svmax_m(pg, vout4, vzero);
          vout5 = svmax_m(pg, vout5, vzero);
          vout6 = svmax_m(pg, vout6, vzero);
          vout7 = svmax_m(pg, vout7, vzero);
          vout0 = svmin_m(pg, vout0, valpha);
          vout1 = svmin_m(pg, vout1, valpha);
          vout2 = svmin_m(pg, vout2, valpha);
          vout3 = svmin_m(pg, vout3, valpha);
          vout4 = svmin_m(pg, vout4, valpha);
          vout5 = svmin_m(pg, vout5, valpha);
          vout6 = svmin_m(pg, vout6, valpha);
          vout7 = svmin_m(pg, vout7, valpha);
        } else if (flag_act == 3) {  // leakyrelu
          auto vres0_0 = svcmpge(pg, vout0, vzero);
          auto vres0_1 = svmul_m(pg, vout0, valpha);
          auto vres1_0 = svcmpge(pg, vout1, vzero);
          auto vres1_1 = svmul_m(pg, vout1, valpha);
          auto vres2_0 = svcmpge(pg, vout2, vzero);
          auto vres2_1 = svmul_m(pg, vout2, valpha);
          auto vres3_0 = svcmpge(pg, vout3, vzero);
          auto vres3_1 = svmul_m(pg, vout3, valpha);
          auto vres4_0 = svcmpge(pg, vout4, vzero);
          auto vres4_1 = svmul_m(pg, vout4, valpha);
          auto vres5_0 = svcmpge(pg, vout5, vzero);
          auto vres5_1 = svmul_m(pg, vout5, valpha);
          auto vres6_0 = svcmpge(pg, vout6, vzero);
          auto vres6_1 = svmul_m(pg, vout6, valpha);
          auto vres7_0 = svcmpge(pg, vout7, vzero);
          auto vres7_1 = svmul_m(pg, vout7, valpha);
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
        auto vout0 = svdup_n(static_cast<Dtype>(0.f));
        auto vout1 = svdup_n(static_cast<Dtype>(0.f));
        auto vout2 = svdup_n(static_cast<Dtype>(0.f));
        auto vout3 = svdup_n(static_cast<Dtype>(0.f));
        if (has_bias) {
          vout0 = svdup_n(static_cast<Dtype>(bias[index_y]));
          vout1 = svdup_n(static_cast<Dtype>(bias[index_y + 1]));
          vout2 = svdup_n(static_cast<Dtype>(bias[index_y + 2]));
          vout3 = svdup_n(static_cast<Dtype>(bias[index_y + 3]));
        }

        for (int x = 0; x < N; x += svcnt<Dtype>()) {
          svbool_t pg = svwhilelt<Dtype>(x, N);
          const Dtype *b_ptr_l = b_ptr;
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
          b_ptr += svcnt<Dtype>();
          c_ptr0 += svcnt<Dtype>();
          c_ptr1 += svcnt<Dtype>();
          c_ptr2 += svcnt<Dtype>();
          c_ptr3 += svcnt<Dtype>();
        }
        a_ptr += 4 * lda;
        c_ptr += 4 * ldc;
        index_y += 4;
      } else if (remain >= 2) {
        const Dtype *a_ptr_l = a_ptr;
        const Dtype *b_ptr = B;
        Dtype *c_ptr0 = c_ptr;
        Dtype *c_ptr1 = c_ptr0 + ldc;
        const Dtype *a_ptr0 = a_ptr_l;
        const Dtype *a_ptr1 = a_ptr0 + lda;
        auto vout0 = svdup_n(static_cast<Dtype>(0.f));
        auto vout1 = svdup_n(static_cast<Dtype>(0.f));
        if (has_bias) {
          vout0 = svdup_n(static_cast<Dtype>(bias[index_y]));
          vout1 = svdup_n(static_cast<Dtype>(bias[index_y + 1]));
        }

        for (int x = 0; x < N; x += svcnt<Dtype>()) {
          svbool_t pg = svwhilelt<Dtype>(x, N);
          const Dtype *b_ptr_l = b_ptr;
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
          svst1(pg, c_ptr0, vout0);
          svst1(pg, c_ptr1, vout1);
          b_ptr += svcnt<Dtype>();
          c_ptr0 += svcnt<Dtype>();
          c_ptr1 += svcnt<Dtype>();
        }
        a_ptr += 2 * lda;
        c_ptr += 2 * ldc;
        index_y += 2;
      } else {
        const Dtype *a_ptr_l = a_ptr;
        const Dtype *b_ptr = B;
        Dtype *c_ptr0 = c_ptr;
        const Dtype *a_ptr0 = a_ptr_l;
        auto vout0 = svdup_n(static_cast<Dtype>(0.f));
        if (has_bias) {
          vout0 = svdup_n(static_cast<Dtype>(bias[index_y]));
        }

        for (int x = 0; x < N; x += svcnt<Dtype>()) {
          svbool_t pg = svwhilelt<Dtype>(x, N);
          const Dtype *b_ptr_l = b_ptr;
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

template <typename Dtype>
void sgemm_prepacked_m4_sve(bool is_transB,
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
  int mblock_sve = 4;
  x_block /= mblock_sve;
  x_block *= mblock_sve;
  int x_num = (M + (x_block - 1)) / x_block;
  x_block = (M + (x_block - 1)) / x_num;
  x_block = (x_block + mblock_sve - 1) / mblock_sve;
  x_block *= mblock_sve;
  x_block = x_block < mblock_sve ? mblock_sve : x_block;

  // unroll 2 loop
  int tail_pre = (K & (KBLOCK_SVE - 1));
  int k_pre = ((K + KBLOCK_SVE - 1) / KBLOCK_SVE) - 1;

  bool flag_p_remain = false;
  int remain = 0;
  if (tail_pre == 0) {
    tail_pre = KBLOCK_SVE;
  }
  auto vzero = svdup_n(static_cast<Dtype>(0.f));
  auto valpha = svdup_n(static_cast<Dtype>(local_alpha));

  int has_beta = fabsf(beta) > 1e-8f ? 1 : 0;
  //! apanel is pre_compute outside gemm
  for (unsigned int x0 = 0; x0 < M; x0 += x_block) {
    unsigned int xmax = x0 + x_block;
    if (xmax > M) {
      xmax = M;
    }
    int bblocks = (xmax - x0 + mblock_sve - 1) / mblock_sve;
    remain = xmax - x0 - (bblocks - 1) * mblock_sve;
    if (remain > 0 && remain != 8) {
      flag_p_remain = true;
    }
    if (flag_p_remain) {
      bblocks -= 1;
    }

    LITE_PARALLEL_COMMON_BEGIN(y, tid, bblocks, 0, 1) {
      auto index_y = y * mblock_sve;
      const Dtype *a_ptr_l = A + index_y * lda;
      const Dtype *b_ptr = B;
      Dtype *c_ptr0 = C + index_y * ldc;
      Dtype *c_ptr1 = c_ptr0 + ldc;
      Dtype *c_ptr2 = c_ptr1 + ldc;
      Dtype *c_ptr3 = c_ptr2 + ldc;
      const Dtype *a_ptr0 = a_ptr_l;
      const Dtype *a_ptr1 = a_ptr0 + lda;
      const Dtype *a_ptr2 = a_ptr1 + lda;
      const Dtype *a_ptr3 = a_ptr2 + lda;
      auto vout0 = svdup_n(static_cast<Dtype>(0.f));
      auto vout1 = svdup_n(static_cast<Dtype>(0.f));
      auto vout2 = svdup_n(static_cast<Dtype>(0.f));
      auto vout3 = svdup_n(static_cast<Dtype>(0.f));
      if (has_bias) {
        vout0 = svdup_n(static_cast<Dtype>(bias[index_y]));
        vout1 = svdup_n(static_cast<Dtype>(bias[index_y + 1]));
        vout2 = svdup_n(static_cast<Dtype>(bias[index_y + 2]));
        vout3 = svdup_n(static_cast<Dtype>(bias[index_y + 3]));
      }

      for (int x = 0; x < N; x += svcnt<Dtype>()) {
        svbool_t pg = svwhilelt<Dtype>(x, N);
        const Dtype *b_ptr_l = b_ptr;
        for (int k = 0; k < K; k++) {
          const Dtype *b_ptr_k = b_ptr_l;
          auto vb = svld1(pg, b_ptr_k);
          auto va0 = svdup_n(static_cast<Dtype>(*a_ptr0));
          auto va1 = svdup_n(static_cast<Dtype>(*a_ptr1));
          auto va2 = svdup_n(static_cast<Dtype>(*a_ptr2));
          auto va3 = svdup_n(static_cast<Dtype>(*a_ptr3));
          vout0 = svmla_m(pg, vout0, vb, va0);
          vout1 = svmla_m(pg, vout1, vb, va1);
          vout2 = svmla_m(pg, vout2, vb, va2);
          vout3 = svmla_m(pg, vout3, vb, va3);
          b_ptr_k += ldb;
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
        b_ptr += svcnt<Dtype>();
        c_ptr0 += svcnt<Dtype>();
        c_ptr1 += svcnt<Dtype>();
        c_ptr2 += svcnt<Dtype>();
        c_ptr3 += svcnt<Dtype>();
      }
    }
    LITE_PARALLEL_COMMON_END();
    // m < 4
    auto index_y = bblocks * mblock_sve;
    if (flag_p_remain) {
      const Dtype *a_ptr = A + index_y * lda;
      Dtype *c_ptr = C + index_y * ldc;
      if (remain >= 2) {
        const Dtype *a_ptr_l = a_ptr;
        const Dtype *b_ptr = B;
        Dtype *c_ptr0 = c_ptr;
        Dtype *c_ptr1 = c_ptr0 + ldc;
        const Dtype *a_ptr0 = a_ptr_l;
        const Dtype *a_ptr1 = a_ptr0 + lda;
        auto vout0 = svdup_n(static_cast<Dtype>(0.f));
        auto vout1 = svdup_n(static_cast<Dtype>(0.f));
        if (has_bias) {
          vout0 = svdup_n(static_cast<Dtype>(bias[index_y]));
          vout1 = svdup_n(static_cast<Dtype>(bias[index_y + 1]));
        }

        for (int x = 0; x < N; x += svcnt<Dtype>()) {
          svbool_t pg = svwhilelt<Dtype>(x, N);
          const Dtype *b_ptr_l = b_ptr;
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
          svst1(pg, c_ptr0, vout0);
          svst1(pg, c_ptr1, vout1);
          b_ptr += svcnt<Dtype>();
          c_ptr0 += svcnt<Dtype>();
          c_ptr1 += svcnt<Dtype>();
        }
        a_ptr += 2 * lda;
        c_ptr += 2 * ldc;
        index_y += 2;
      } else {
        const Dtype *a_ptr_l = a_ptr;
        const Dtype *b_ptr = B;
        Dtype *c_ptr0 = c_ptr;
        const Dtype *a_ptr0 = a_ptr_l;
        auto vout0 = svdup_n(static_cast<Dtype>(0.f));
        if (has_bias) {
          vout0 = svdup_n(static_cast<Dtype>(bias[index_y]));
        }

        for (int x = 0; x < N; x += svcnt<Dtype>()) {
          svbool_t pg = svwhilelt<Dtype>(x, N);
          const Dtype *b_ptr_l = b_ptr;
          for (int k = 0; k < K; k++) {
            auto vb = svld1(pg, b_ptr_l);
            auto va0 = svdup_n(static_cast<Dtype>(*a_ptr0));
            vout0 = svmla_m(pg, vout0, vb, va0);
            b_ptr_l += ldb;
            a_ptr0++;
          }
          svst1(pg, c_ptr0, vout0);
          b_ptr += svcnt<Dtype>();
          c_ptr0 += svcnt<Dtype>();
          index_y++;
        }
        a_ptr += lda;
        c_ptr += ldc;
      }
    }
  }
}
/// a: m*k  b: k*n  c: m*n
template <typename Dtype>
void sgemm_prepack_sve(GEMM_FUNCS_PARAM(Dtype)) {
  if (M <= 4) {
    sgemm_prepacked_m4_sve<Dtype>(GEMM_ACTUAL_PARAM);
  } else {
    sgemm_prepacked_m8_sve<Dtype>(GEMM_ACTUAL_PARAM);
  }
}

template void sgemm_prepack_sve<float>(GEMM_FUNCS_PARAM(float));
template void sgemm_prepacked_m8_sve<float>(GEMM_FUNCS_PARAM(float));
template void sgemm_prepacked_m4_sve<float>(GEMM_FUNCS_PARAM(float));
#ifdef ENABLE_ARM_FP16
template void sgemm_prepack_sve<float16_t>(GEMM_FUNCS_PARAM(float16_t));
template void sgemm_prepacked_m8_sve<float16_t>(GEMM_FUNCS_PARAM(float16_t));
template void sgemm_prepacked_m4_sve<float16_t>(GEMM_FUNCS_PARAM(float16_t));
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
