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

#include <cmath>
#include "lite/core/context.h"
#include "lite/core/device_info.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
typedef __fp16 float16_t;
#define ROUNDUP(a, b) ((((a) + (b)-1) / (b)) * (b))
#define PTR_ACQUIRE_PARAM(dtype)                                        \
  const dtype *ptr_zero, const dtype **ptr_w0, const dtype **ptr_w1,    \
      const dtype **ptr_w2, const dtype **ptr_w3, const dtype **ptr_w4, \
      const dtype **ptr_w5, const dtype **ptr_w6, const dtype **ptr_w7, \
      int remain

#define PTR_ACQUIRE_PARAM_4(dtype)                                   \
  const dtype *ptr_zero, const dtype **ptr_w0, const dtype **ptr_w1, \
      const dtype **ptr_w2, const dtype **ptr_w3, int remain

#define PTR_ACQUIRE_PARAM_A8(dtype)                                     \
  const dtype *zerobuff, const dtype **inptr1, const dtype **inptr2,    \
      const dtype **inptr3, const dtype **inptr4, const dtype **inptr5, \
      const dtype **inptr6, const dtype **inptr7, int numa, int numb

#define PTR_ACQUIRE_PARAM_B8(dtype) \
  dtype *bias_local, const dtype *bias, int y, int numa, int numb

#define PTR_ACQUIRE_PARAM_A16(dtype)                                       \
  const dtype *zerobuff, const dtype **inptr1, const dtype **inptr2,       \
      const dtype **inptr3, const dtype **inptr4, const dtype **inptr5,    \
      const dtype **inptr6, const dtype **inptr7, const dtype **inptr8,    \
      const dtype **inptr9, const dtype **inptr10, const dtype **inptr11,  \
      const dtype **inptr12, const dtype **inptr13, const dtype **inptr14, \
      const dtype **inptr15, int numa, int numb

#define X_BLOCK_COMPUTE_FP16(llc_size, MBLOCK, NBLOCK, KBLOCK, beta)  \
  /* MBLOCK * x (result) + MBLOCK * k (A) + x * k (B) = l2*/          \
  int x_block =                                                       \
      (llc_size - (MBLOCK * K)) / (sizeof(float16_t) * (K + MBLOCK)); \
  x_block /= NBLOCK;                                                  \
  x_block = (x_block == 0) ? 1 : x_block;                             \
  x_block *= NBLOCK;                                                  \
  int x_num = (N + (x_block - 1)) / x_block;                          \
  x_block = (N + x_num - 1) / x_num;                                  \
  x_block = (x_block + NBLOCK - 1) / NBLOCK;                          \
  x_block *= NBLOCK;                                                  \
  x_block = x_block < NBLOCK ? NBLOCK : x_block;                      \
  int tail_pre = (K & (KBLOCK - 1));                                  \
  int k_pre = ((K + KBLOCK - 1) / KBLOCK) - 1;                        \
  bool flag_p_remain = false;                                         \
  int remain = 0;                                                     \
  if (tail_pre == 0) {                                                \
    tail_pre = KBLOCK;                                                \
  }                                                                   \
  int has_beta = fabsf(beta) > 1e-8f ? 1 : 0;

#define DIRECT_WORKSPACE_COMPUTE(                                              \
    ctx, kernel_w, stride_w, ow, oh, ic, OUT_C_BLOCK, OUT_H_BLOCK)             \
  const int threads = ctx->threads();                                          \
  int llc_size = ctx->llc_size() / sizeof(float16_t);                          \
  const int wout_round = ROUNDUP(ow, OUT_W_BLOCK);                             \
  const int win_round = (wout_round - 1) * stride_w + kernel_w;                \
  /* get h block */                                                            \
  /* win_round * ic * hin_r_block + wout_round * OUT_C_BLOCK * hout_r_block */ \
  /* * threads = llc_size */                                                   \
  /* win_round = (wout_round - 1) * stride_w + kernel_w*/                      \
  /* hin_r_block = (hout_r_block - 1) * stride_w + kernel_w*/                  \
  int a = kernel_w * stride_w;                                                 \
  int b = kernel_w * kernel_w;                                                 \
  int c = stride_w * stride_w;                                                 \
  int hout_r_block =                                                           \
      (llc_size - ic * (a * (wout_round - 2) + b - c * (wout_round - 1))) /    \
      ((ic * ((wout_round - 1) * c + a)) +                                     \
       wout_round * OUT_C_BLOCK * threads);                                    \
  hout_r_block = hout_r_block > oh ? oh : hout_r_block;                        \
  hout_r_block = (hout_r_block / OUT_H_BLOCK) * OUT_H_BLOCK;                   \
  hout_r_block = hout_r_block < OUT_H_BLOCK ? OUT_H_BLOCK : hout_r_block;      \
  const int hin_r_block = (hout_r_block - 1) * stride_w + kernel_w;            \
  int in_len = win_round * ic;                                                 \
  int pre_in_size = hin_r_block * in_len;                                      \
  int pre_out_size = OUT_C_BLOCK * hout_r_block * wout_round;

#define GEMM_PREPARE_C(dtype, NBLOCK) \
  dtype cout0[NBLOCK];                \
  dtype cout1[NBLOCK];                \
  dtype cout2[NBLOCK];                \
  dtype cout3[NBLOCK];                \
  dtype cout4[NBLOCK];                \
  dtype cout5[NBLOCK];                \
  dtype cout6[NBLOCK];                \
  dtype cout7[NBLOCK];                \
  dtype *c_ptr0 = C + y * ldc + x0;   \
  dtype *c_ptr1 = c_ptr0 + ldc;       \
  dtype *c_ptr2 = c_ptr1 + ldc;       \
  dtype *c_ptr3 = c_ptr2 + ldc;       \
  dtype *c_ptr4 = c_ptr3 + ldc;       \
  dtype *c_ptr5 = c_ptr4 + ldc;       \
  dtype *c_ptr6 = c_ptr5 + ldc;       \
  dtype *c_ptr7 = c_ptr6 + ldc;       \
  dtype *pout0 = c_ptr0;              \
  dtype *pout1 = c_ptr1;              \
  dtype *pout2 = c_ptr2;              \
  dtype *pout3 = c_ptr3;              \
  dtype *pout4 = c_ptr4;              \
  dtype *pout5 = c_ptr5;              \
  dtype *pout6 = c_ptr6;              \
  dtype *pout7 = c_ptr7;

#define GEMM_REMAIN_C_PREPARE          \
  pout0 = c_ptr0;                      \
  pout1 = c_ptr1;                      \
  pout2 = c_ptr2;                      \
  pout3 = c_ptr3;                      \
  pout4 = c_ptr4;                      \
  pout5 = c_ptr5;                      \
  pout6 = c_ptr6;                      \
  pout7 = c_ptr7;                      \
  c_ptr0 = cout0;                      \
  c_ptr1 = cout1;                      \
  c_ptr2 = cout2;                      \
  c_ptr3 = cout3;                      \
  c_ptr4 = cout4;                      \
  c_ptr5 = cout5;                      \
  c_ptr6 = cout6;                      \
  c_ptr7 = cout7;                      \
  if (has_beta) {                      \
    for (int i = 0; i < remain; ++i) { \
      cout0[i] = pout0[i];             \
      cout1[i] = pout1[i];             \
      cout2[i] = pout2[i];             \
      cout3[i] = pout3[i];             \
      cout4[i] = pout4[i];             \
      cout5[i] = pout5[i];             \
      cout6[i] = pout6[i];             \
      cout7[i] = pout7[i];             \
    }                                  \
  }

#define GEMM_REMAIN_C_PROCESS        \
  for (int i = 0; i < remain; ++i) { \
    *pout0++ = cout0[i];             \
    *pout1++ = cout1[i];             \
    *pout2++ = cout2[i];             \
    *pout3++ = cout3[i];             \
    *pout4++ = cout4[i];             \
    *pout5++ = cout5[i];             \
    *pout6++ = cout6[i];             \
    *pout7++ = cout7[i];             \
  }

inline void act_acquire(lite_api::ActivationType act,
                        int &flag_act,           // NOLINT
                        float16_t &local_alpha,  // NOLINT
                        float16_t &offset,       // NOLINT
                        float16_t &threshold,    // NOLINT
                        const operators::ActivationParam act_param) {
  switch (act) {
    case lite_api::ActivationType::kRelu:
      flag_act = 0x01;
      break;
    case lite_api::ActivationType::kRelu6:
      flag_act = 0x02;
      local_alpha = static_cast<float16_t>(act_param.Relu_clipped_coef);
      break;
    case lite_api::ActivationType::kLeakyRelu:
      flag_act = 0x03;
      local_alpha = static_cast<float16_t>(act_param.Leaky_relu_alpha);
      break;
    case lite_api::ActivationType::kHardSwish:
      flag_act = 0x04;
      local_alpha = static_cast<float16_t>(1.0 / act_param.hard_swish_scale);
      offset = static_cast<float16_t>(act_param.hard_swish_offset);
      threshold = static_cast<float16_t>(act_param.hard_swish_threshold);
      break;
    default:
      break;
  }
}

template <typename dtype>
inline void ptr_acquire_remain(PTR_ACQUIRE_PARAM(dtype)) {
  switch (8 - remain) {
    case 7:
      *ptr_w0 = ptr_zero;
      break;
    case 6:
      *ptr_w1 = ptr_zero;
      break;
    case 5:
      *ptr_w2 = ptr_zero;
      break;
    case 4:
      *ptr_w3 = ptr_zero;
      break;
    case 3:
      *ptr_w4 = ptr_zero;
      break;
    case 2:
      *ptr_w5 = ptr_zero;
      break;
    case 1:
      *ptr_w6 = ptr_zero;
      break;
    default:
      break;
  }
}

template <typename dtype>
inline void ptr_acquire_remain_four(PTR_ACQUIRE_PARAM_4(dtype)) {
  switch (4 - remain) {
    case 3:
      *ptr_w0 = ptr_zero;
      break;
    case 2:
      *ptr_w1 = ptr_zero;
      break;
    case 1:
      *ptr_w2 = ptr_zero;
      break;
    default:
      break;
  }
}

template <typename dtype>
inline void ptr_acquire_norm(PTR_ACQUIRE_PARAM(dtype)) {
  switch (8 - remain) {
    case 7:
      *ptr_w1 = ptr_zero;
    case 6:
      *ptr_w2 = ptr_zero;
    case 5:
      *ptr_w3 = ptr_zero;
    case 4:
      *ptr_w4 = ptr_zero;
    case 3:
      *ptr_w5 = ptr_zero;
    case 2:
      *ptr_w6 = ptr_zero;
    case 1:
      *ptr_w7 = ptr_zero;
      break;
    default:
      break;
  }
}

template <typename dtype>
inline void ptr_acquire_norm_four(PTR_ACQUIRE_PARAM_4(dtype)) {
  switch (4 - remain) {
    case 3:
      *ptr_w1 = ptr_zero;
    case 2:
      *ptr_w2 = ptr_zero;
    case 1:
      *ptr_w3 = ptr_zero;
      break;
    default:
      break;
  }
}
template <typename dtype>
inline void ptr_acquire_a8(PTR_ACQUIRE_PARAM_A8(dtype)) {
  switch (numa - numb) {
    case 6:
      *inptr1 = zerobuff;
    case 5:
      *inptr2 = zerobuff;
    case 4:
      *inptr3 = zerobuff;
    case 3:
      *inptr4 = zerobuff;
    case 2:
      *inptr5 = zerobuff;
    case 1:
      *inptr6 = zerobuff;
    case 0:
      *inptr7 = zerobuff;
    default:
      break;
  }
}

template <typename dtype>
inline void ptr_acquire_b8(PTR_ACQUIRE_PARAM_B8(dtype)) {
  switch (numa - numb) {
    case 0:
      bias_local[6] = bias[y + 6];
    case 1:
      bias_local[5] = bias[y + 5];
    case 2:
      bias_local[4] = bias[y + 4];
    case 3:
      bias_local[3] = bias[y + 3];
    case 4:
      bias_local[2] = bias[y + 2];
    case 5:
      bias_local[1] = bias[y + 1];
    case 6:
      bias_local[0] = bias[y];
    default:
      break;
  }
}

template <typename dtype>
inline void ptr_acquire_a16(PTR_ACQUIRE_PARAM_A16(dtype)) {
  switch (numa - numb) {
    case 14:
      *inptr1 = zerobuff;
    case 13:
      *inptr2 = zerobuff;
    case 12:
      *inptr3 = zerobuff;
    case 11:
      *inptr4 = zerobuff;
    case 10:
      *inptr5 = zerobuff;
    case 9:
      *inptr6 = zerobuff;
    case 8:
      *inptr7 = zerobuff;
    case 7:
      *inptr8 = zerobuff;
    case 6:
      *inptr9 = zerobuff;
    case 5:
      *inptr10 = zerobuff;
    case 4:
      *inptr11 = zerobuff;
    case 3:
      *inptr12 = zerobuff;
    case 2:
      *inptr13 = zerobuff;
    case 1:
      *inptr14 = zerobuff;
    case 0:
      *inptr15 = zerobuff;
    default:
      break;
  }
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
