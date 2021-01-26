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

#include "lite/backends/arm/math/fp16/conv_block_utils_fp16.h"
#include "lite/backends/arm/math/fp16/conv_impl_fp16.h"
#include "lite/core/context.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

const int OUT_C_BLOCK = 8;
const int OUT_H_BLOCK = 2;
const int OUT_W_BLOCK = 8;

#define DIRECT_WORKSPACE_COMPUTE                                               \
  const int threads = ctx->threads();                                          \
  int llc_size = ctx->llc_size() / sizeof(float);                              \
  const int wout_round = ROUNDUP(ow, OUT_W_BLOCK);                             \
  const int win_round = wout_round * 2 /*stride_w*/ + 1;                       \
  /* get h block */                                                            \
  /* win_round * ic * hin_r_block + wout_round * OUT_C_BLOCK * hout_r_block */ \
  /* * threads = llc_size */                                                   \
  /* win_round = 2 * wout_round + 1 */                                         \
  /* hin_r_block = 2 * hout_r_block + 1 */                                     \
  int hout_r_block =                                                           \
      (llc_size - 2 * wout_round * ic - ic) /                                  \
      ((4 * wout_round + 2) * ic + wout_round * OUT_C_BLOCK * threads);        \
  hout_r_block = hout_r_block > oh ? oh : hout_r_block;                        \
  hout_r_block = (hout_r_block / OUT_H_BLOCK) * OUT_H_BLOCK;                   \
  hout_r_block = hout_r_block < OUT_H_BLOCK ? OUT_H_BLOCK : hout_r_block;      \
  const int hin_r_block = hout_r_block * 2 /*stride_h*/ + 1;                   \
  int in_len = win_round * ic;                                                 \
  int pre_in_size = hin_r_block * in_len;                                      \
  int pre_out_size = OUT_C_BLOCK * hout_r_block * wout_round;

#define COMPUT_INIT                     \
  float16_t* ptr_out0 = pre_out0;       \
  float16_t* ptr_out1 = pre_out1;       \
  float16x8_t w0 = vld1q_f16(wc0);      \
  float16x8_t w1 = vld1q_f16(wc0 + 8);  \
  float16x8_t w2 = vld1q_f16(wc0 + 16); \
  float16x8_t w3 = vld1q_f16(wc0 + 24); \
  float16x8_t w4 = vld1q_f16(wc0 + 32); \
  float16x8_t w5 = vld1q_f16(wc0 + 40); \
  float16x8_t w6 = vld1q_f16(wc0 + 48); \
  float16x8_t w7 = vld1q_f16(wc0 + 56); \
  float16x8_t w8 = vld1q_f16(wc0 + 64); \
  const float16_t* r0 = inr0;           \
  const float16_t* r1 = inr1;           \
  const float16_t* r2 = inr2;           \
  const float16_t* r3 = inr3;           \
  const float16_t* r4 = inr4;

size_t conv3x3s2_direct_workspace_size(const operators::ConvParam& param,
                                       ARMContext* ctx) {
  auto dim_in = param.x->dims();
  auto dim_out = param.output->dims();
  auto paddings = *param.paddings;
  int ow = dim_out[3];
  int oh = dim_out[2];
  int ic = dim_in[1];
  DIRECT_WORKSPACE_COMPUTE
  return sizeof(float) * (pre_in_size + ctx->threads() * pre_out_size);
}

#ifdef __aarch64__
#define INIT_FIRST                   \
  "2:\n"                             \
  "ldp q0, q1, [%[r0]], #32\n"       \
  "ldp q4, q5, [%[r2]], #32\n"       \
  "ldr d10, [%[r0]]\n"               \
  "ldr d12, [%[r2]]\n"               \
  "fmul v16.8h, %[w0].8h, v0.h[0]\n" \
  "fmul v17.8h, %[w0].8h, v0.h[2]\n" \
  "fmul v18.8h, %[w0].8h, v0.h[4]\n" \
  "fmul v19.8h, %[w0].8h, v0.h[6]\n" \
  "fmul v20.8h, %[w0].8h, v1.h[0]\n" \
  "fmul v21.8h, %[w0].8h, v1.h[2]\n" \
  "fmul v22.8h, %[w0].8h, v1.h[4]\n" \
  "fmul v23.8h, %[w0].8h, v1.h[6]\n" \
  "fmul v24.8h, %[w0].8h, v4.h[0]\n" \
  "fmul v25.8h, %[w0].8h, v4.h[2]\n" \
  "fmul v26.8h, %[w0].8h, v4.h[4]\n" \
  "fmul v27.8h, %[w0].8h, v4.h[6]\n" \
  "fmul v28.8h, %[w0].8h, v5.h[0]\n" \
  "fmul v29.8h, %[w0].8h, v5.h[2]\n" \
  "fmul v30.8h, %[w0].8h, v5.h[4]\n" \
  "fmul v31.8h, %[w0].8h, v5.h[6]\n"

#define INIT                          \
  "2:\n"                              \
  "ldp q16, q17 [%[ptr_out0]]\n"      \
  "ldp q0, q1, [%[r0]], #32\n"        \
  "ldp q18, q19 [%[ptr_out0], #32]\n" \
  "ldp q4, q5, [%[r2]], #32\n"        \
  "ldp q20, q21 [%[ptr_out0], #64]\n" \
  "ldr d10, [%[r0]]\n"                \
  "ldp q22, q23 [%[ptr_out0], #96]\n" \
  "ldr d12, [%[r2]]\n"                \
  "ldp q24, q25 [%[ptr_out1]]\n"      \
  "fmla v16.8h, %[w0].8h, v0.h[0]\n"  \
  "ldp q26, q27 [%[ptr_out1], #32]\n" \
  "fmla v17.8h, %[w0].8h, v0.h[2]\n"  \
  "ldp q28, q29 [%[ptr_out1], #64]\n" \
  "fmla v18.8h, %[w0].8h, v0.h[4]\n"  \
  "ldp q30, q31 [%[ptr_out1], #96]\n" \
  "fmla v19.8h, %[w0].8h, v0.h[6]\n"  \
  "fmla v20.8h, %[w0].8h, v1.h[0]\n"  \
  "fmla v21.8h, %[w0].8h, v1.h[2]\n"  \
  "fmla v22.8h, %[w0].8h, v1.h[4]\n"  \
  "fmla v23.8h, %[w0].8h, v1.h[6]\n"  \
  "fmla v24.8h, %[w0].8h, v4.h[0]\n"  \
  "fmla v25.8h, %[w0].8h, v4.h[2]\n"  \
  "fmla v26.8h, %[w0].8h, v4.h[4]\n"  \
  "fmla v27.8h, %[w0].8h, v4.h[6]\n"  \
  "fmla v28.8h, %[w0].8h, v5.h[0]\n"  \
  "fmla v29.8h, %[w0].8h, v5.h[2]\n"  \
  "fmla v30.8h, %[w0].8h, v5.h[4]\n"  \
  "fmla v31.8h, %[w0].8h, v5.h[6]\n"

#define COMPUTE                        \
  "ldp q2, q3, [%[r1]], #32\n"         \
  "fmla v16.8h, %[w6].8h, v4.h[0]\n"   \
  "fmla v17.8h, %[w6].8h, v4.h[2]\n"   \
  "fmla v18.8h, %[w6].8h, v4.h[4]\n"   \
  "fmla v19.8h, %[w6].8h, v4.h[6]\n"   \
  "fmla v20.8h, %[w6].8h, v5.h[0]\n"   \
  "fmla v21.8h, %[w6].8h, v5.h[2]\n"   \
  "fmla v22.8h, %[w6].8h, v5.h[4]\n"   \
  "fmla v23.8h, %[w6].8h, v5.h[6]\n"   \
  "ldr d11, [%[r1]]\n"                 \
  "fmla v16.8h, %[w1].8h, v0.h[1]\n"   \
  "fmla v17.8h, %[w1].8h, v0.h[3]\n"   \
  "fmla v18.8h, %[w1].8h, v0.h[5]\n"   \
  "fmla v19.8h, %[w1].8h, v0.h[7]\n"   \
  "fmla v20.8h, %[w1].8h, v1.h[1]\n"   \
  "fmla v21.8h, %[w1].8h, v1.h[3]\n"   \
  "fmla v22.8h, %[w1].8h, v1.h[5]\n"   \
  "fmla v23.8h, %[w1].8h, v1.h[7]\n"   \
  "fmla v24.8h, %[w1].8h, v4.h[1]\n"   \
  "fmla v25.8h, %[w1].8h, v4.h[3]\n"   \
  "fmla v26.8h, %[w1].8h, v4.h[5]\n"   \
  "fmla v27.8h, %[w1].8h, v4.h[7]\n"   \
  "fmla v28.8h, %[w1].8h, v5.h[1]\n"   \
  "fmla v29.8h, %[w1].8h, v5.h[3]\n"   \
  "fmla v30.8h, %[w1].8h, v5.h[5]\n"   \
  "fmla v31.8h, %[w1].8h, v5.h[7]\n"   \
  "ldp q6, q7, [%[r3]], #32\n"         \
  "fmla v16.8h, %[w7].8h, v4.h[1]\n"   \
  "fmla v17.8h, %[w7].8h, v4.h[3]\n"   \
  "fmla v18.8h, %[w7].8h, v4.h[5]\n"   \
  "fmla v19.8h, %[w7].8h, v4.h[7]\n"   \
  "fmla v20.8h, %[w7].8h, v5.h[1]\n"   \
  "fmla v21.8h, %[w7].8h, v5.h[3]\n"   \
  "fmla v22.8h, %[w7].8h, v5.h[5]\n"   \
  "fmla v23.8h, %[w7].8h, v5.h[7]\n"   \
  "ldr d13, [%[r3]]\n"                 \
  "fmla v16.8h, %[w2].8h, v0.h[2]\n"   \
  "fmla v17.8h, %[w2].8h, v0.h[4]\n"   \
  "fmla v18.8h, %[w2].8h, v0.h[6]\n"   \
  "fmla v19.8h, %[w2].8h, v1.h[0]\n"   \
  "fmla v20.8h, %[w2].8h, v1.h[2]\n"   \
  "fmla v21.8h, %[w2].8h, v1.h[4]\n"   \
  "fmla v22.8h, %[w2].8h, v1.h[6]\n"   \
  "fmla v23.8h, %[w2].8h, v10.h[0]\n"  \
  "fmla v24.8h, %[w2].8h, v4.h[2]\n"   \
  "fmla v25.8h, %[w2].8h, v4.h[4]\n"   \
  "fmla v26.8h, %[w2].8h, v4.h[6]\n"   \
  "fmla v27.8h, %[w2].8h, v4.h[0]\n"   \
  "fmla v28.8h, %[w2].8h, v5.h[2]\n"   \
  "fmla v29.8h, %[w2].8h, v5.h[4]\n"   \
  "fmla v30.8h, %[w2].8h, v5.h[6]\n"   \
  "fmla v31.8h, %[w2].8h, v12.h[0]\n"  \
  "ldp q8, q9, [%[r4]], #32\n"         \
  "fmla v16.8h, %[w8].8h, v4.h[2]\n"   \
  "fmla v17.8h, %[w8].8h, v4.h[4]\n"   \
  "fmla v18.8h, %[w8].8h, v4.h[6]\n"   \
  "fmla v19.8h, %[w8].8h, v5.h[0]\n"   \
  "fmla v20.8h, %[w8].8h, v5.h[2]\n"   \
  "fmla v21.8h, %[w8].8h, v5.h[4]\n"   \
  "fmla v22.8h, %[w8].8h, v5.h[6]\n"   \
  "fmla v23.8h, %[w8].8h, v12.h[0]\n"  \
  "ldr d14, [%[r4]]\n"                 \
  "fmla v16.8h, %[w3].8h, v2.h[0]\n"   \
  "fmla v17.8h, %[w3].8h, v2.h[2]\n"   \
  "fmla v18.8h, %[w3].8h, v2.h[4]\n"   \
  "fmla v19.8h, %[w3].8h, v2.h[6]\n"   \
  "fmla v20.8h, %[w3].8h, v3.h[0]\n"   \
  "fmla v21.8h, %[w3].8h, v3.h[2]\n"   \
  "fmla v22.8h, %[w3].8h, v3.h[4]\n"   \
  "fmla v23.8h, %[w3].8h, v3.h[6]\n"   \
  "fmla v24.8h, %[w3].8h, v6.h[0]\n"   \
  "fmla v25.8h, %[w3].8h, v6.h[2]\n"   \
  "fmla v26.8h, %[w3].8h, v6.h[4]\n"   \
  "fmla v27.8h, %[w3].8h, v6.h[6]\n"   \
  "fmla v28.8h, %[w3].8h, v7.h[0]\n"   \
  "fmla v29.8h, %[w3].8h, v7.h[2]\n"   \
  "fmla v30.8h, %[w3].8h, v7.h[4]\n"   \
  "fmla v31.8h, %[w3].8h, v7.h[6]\n"   \
  "fmla v16.8h, %[w4].8h, v2.h[1]\n"   \
  "fmla v17.8h, %[w4].8h, v2.h[3]\n"   \
  "fmla v18.8h, %[w4].8h, v2.h[5]\n"   \
  "fmla v19.8h, %[w4].8h, v2.h[7]\n"   \
  "fmla v20.8h, %[w4].8h, v3.h[1]\n"   \
  "fmla v21.8h, %[w4].8h, v3.h[3]\n"   \
  "fmla v22.8h, %[w4].8h, v3.h[5]\n"   \
  "fmla v23.8h, %[w4].8h, v3.h[7]\n"   \
  "fmla v24.8h, %[w4].8h, v6.h[1]\n"   \
  "fmla v25.8h, %[w4].8h, v6.h[3]\n"   \
  "fmla v26.8h, %[w4].8h, v6.h[5]\n"   \
  "fmla v27.8h, %[w4].8h, v6.h[7]\n"   \
  "fmla v28.8h, %[w4].8h, v7.h[1]\n"   \
  "fmla v29.8h, %[w4].8h, v7.h[3]\n"   \
  "fmla v30.8h, %[w4].8h, v7.h[5]\n"   \
  "fmla v31.8h, %[w4].8h, v7.h[7]\n"   \
  "fmla v16.8h, %[w5].8h, v2.h[2]\n"   \
  "fmla v17.8h, %[w5].8h, v2.h[4]\n"   \
  "fmla v18.8h, %[w5].8h, v2.h[6]\n"   \
  "fmla v19.8h, %[w5].8h, v3.h[0]\n"   \
  "fmla v20.8h, %[w5].8h, v3.h[2]\n"   \
  "fmla v21.8h, %[w5].8h, v3.h[4]\n"   \
  "fmla v22.8h, %[w5].8h, v3.h[6]\n"   \
  "fmla v23.8h, %[w5].8h, v11.h[0]\n"  \
  "fmla v24.8h, %[w5].8h, v6.h[2]\n"   \
  "fmla v25.8h, %[w5].8h, v6.h[4]\n"   \
  "fmla v26.8h, %[w5].8h, v6.h[6]\n"   \
  "fmla v27.8h, %[w5].8h, v7.h[0]\n"   \
  "fmla v28.8h, %[w5].8h, v7.h[2]\n"   \
  "fmla v29.8h, %[w5].8h, v7.h[4]\n"   \
  "fmla v30.8h, %[w5].8h, v7.h[6]\n"   \
  "fmla v31.8h, %[w5].8h, v13.h[0]\n"  \
  "fmla v24.8h, %[w6].8h, v8.h[0]\n"   \
  "fmla v25.8h, %[w6].8h, v8.h[2]\n"   \
  "fmla v26.8h, %[w6].8h, v8.h[4]\n"   \
  "fmla v27.8h, %[w6].8h, v8.h[6]\n"   \
  "stp q16, q17, [%[ptr_out0]], #32\n" \
  "fmla v28.8h, %[w6].8h, v9.h[0]\n"   \
  "stp q18, q19, [%[ptr_out0]], #32\n" \
  "fmla v29.8h, %[w6].8h, v9.h[2]\n"   \
  "stp q20, q21, [%[ptr_out0]], #32\n" \
  "fmla v30.8h, %[w6].8h, v9.h[4]\n"   \
  "stp q22, q23, [%[ptr_out0]], #32\n" \
  "fmla v31.8h, %[w6].8h, v9.h[6]\n"   \
  "fmla v24.8h, %[w7].8h, v8.h[1]\n"   \
  "fmla v25.8h, %[w7].8h, v8.h[3]\n"   \
  "fmla v26.8h, %[w7].8h, v8.h[5]\n"   \
  "fmla v27.8h, %[w7].8h, v8.h[7]\n"   \
  "fmla v28.8h, %[w7].8h, v9.h[1]\n"   \
  "fmla v29.8h, %[w7].8h, v9.h[3]\n"   \
  "fmla v30.8h, %[w7].8h, v9.h[5]\n"   \
  "fmla v31.8h, %[w7].8h, v9.h[7]\n"   \
  "fmla v24.8h, %[w8].8h, v8.h[2]\n"   \
  "fmla v25.8h, %[w8].8h, v8.h[4]\n"   \
  "fmla v26.8h, %[w8].8h, v8.h[6]\n"   \
  "subs   %w[cnt], %w[cnt], #1\n"      \
  "fmla v27.8h, %[w8].8h, v9.h[0]\n"   \
  "stp q24, q25, [%[ptr_out1]], #32\n" \
  "fmla v28.8h, %[w8].8h, v9.h[2]\n"   \
  "fmla v29.8h, %[w8].8h, v9.h[4]\n"   \
  "stp q26, q27, [%[ptr_out1]], #32\n" \
  "fmla v30.8h, %[w8].8h, v9.h[6]\n"   \
  "fmla v31.8h, %[w8].8h, v14.h[0]\n"  \
  "stp q28, q29, [%[ptr_out1]], #32\n" \
  "stp q30, q31, [%[ptr_out1]], #32\n" \
  "bne    2b\n"
#else
#endif

void conv_3x3s2_direct_fp16(const float16_t* i_data,
                            float16_t* o_data,
                            int bs,
                            int oc,
                            int oh,
                            int ow,
                            int ic,
                            int ih,
                            int win,
                            const float16_t* weights,
                            const float16_t* bias,
                            const operators::ConvParam& param,
                            ARMContext* ctx) {
  auto paddings = *param.paddings;
  auto act_param = param.activation_param;
  const int pad_w = paddings[2];
  const int pad_h = paddings[0];
  //   const int threads = ctx->threads();
  //   int llc_size = ctx->llc_size() / sizeof(float);
  //   const int wout_round = ROUNDUP(ow, OUT_W_BLOCK);
  //   const int win_round = wout_round * 2 /*stride_w*/ + 1;
  //   int hout_r_block =
  //       (llc_size - 2 * wout_round * ic - ic) /
  //       ((4 * wout_round + 2) * ic + wout_round * OUT_C_BLOCK * threads);
  //   hout_r_block = hout_r_block > oh ? oh : hout_r_block;
  //   hout_r_block = (hout_r_block / OUT_H_BLOCK) * OUT_H_BLOCK;
  //   hout_r_block = hout_r_block < OUT_H_BLOCK ? OUT_H_BLOCK : hout_r_block;
  //   const int hin_r_block = hout_r_block * 2 /*stride_h*/ + 1;

  //   int in_len = win_round * ic;
  //   int pre_in_size = hin_r_block * in_len;
  //   int pre_out_size = OUT_C_BLOCK * hout_r_block * wout_round;
  DIRECT_WORKSPACE_COMPUTE

  float16_t* tmp_work_space = ctx->workspace_data<float16_t>();
  float16_t ptr_zero[win_round];  // NOLINT
  memset(ptr_zero, 0, sizeof(float16_t) * win_round);
  float16_t ptr_write[wout_round];  // NOLINT

  //! l2_cache start
  float16_t* pre_din = tmp_work_space;

  int size_in_channel = win * ih;
  int size_out_channel = ow * oh;
  int w_stride = ic * 9; /*kernel_w * kernel_h*/
  int w_stride_chin = OUT_C_BLOCK * 9;

  int ws = -pad_w;
  int we = ws + win_round;
  int w_loop = wout_round / 4;

  int c_remain = oc - (oc / OUT_C_BLOCK) * OUT_C_BLOCK;
  int c_round_down = (oc / OUT_C_BLOCK) * OUT_C_BLOCK;
  if (c_remain) {
    c_round_down++;
  }

  int out_row_stride = OUT_C_BLOCK * wout_round;
  auto act_type = act_param.active_type;
  bool flag_bias = param.bias != nullptr;
  float alpha = 0.f;
  int flag_act = 0x00;  // relu: 1, relu6: 2, leakey: 3
  if (act_param.has_active) {
    act_acquire(act_type,
                flag_act,
                alpha,
                act_param.Relu_clipped_coef,
                act_param.Leaky_relu_alpha);
  }

  for (int n = 0; n < bs; ++n) {
    const float16_t* din_batch = i_data + n * ic * size_in_channel;
    float16_t* dout_batch = o_data + n * oc * size_out_channel;
    for (int h = 0; h < oh; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > oh) {
        h_kernel = oh - h;
      }

      int hs = h * 2 /*stride_h*/ - pad_h;
      int he = hs + h_kernel * 2 /*stride_h*/ + 1;

      prepack_input_nxw(
          din_batch, pre_din, 0, ic, hs, he, ws, we, ic, win, ih, ptr_zero);

      const float16_t* cblock_inr0 = pre_din;
      const float16_t* cblock_inr1 = cblock_inr0 + in_len;
      const float16_t* cblock_inr2 = cblock_inr1 + in_len;
      const float16_t* cblock_inr3 = cblock_inr2 + in_len;
      const float16_t* cblock_inr4 = cblock_inr3 + in_len;

#pragma omp parallel for num_threads(threads)
      for (int c = 0; c < c_round_down; c += OUT_C_BLOCK) {
#ifdef ARM_WITH_OMP
        float16_t* pre_out =
            pre_din + pre_in_size + omp_get_thread_num() * pre_out_size;
#else
        float16_t* pre_out = pre_din + pre_in_size;
#endif
        const float16_t* block_inr0 = cblock_inr0;
        const float16_t* block_inr1 = cblock_inr1;
        const float16_t* block_inr2 = cblock_inr2;
        const float16_t* block_inr3 = cblock_inr3;
        const float16_t* block_inr4 = cblock_inr4;

        const float16_t* weight_c = weights + c * w_stride;
        const float16_t* bias_ptr = ptr_zero;
        if (flag_bias) {
          bias_ptr = bias + c;
        }

        for (int hk = 0; hk < h_kernel; hk += OUT_H_BLOCK) {
          const float16_t* wc0 = weight_c;

          const float16_t* inr0 = block_inr0;
          const float16_t* inr1 = block_inr1;
          const float16_t* inr2 = block_inr2;
          const float16_t* inr3 = block_inr3;
          const float16_t* inr4 = block_inr4;

          float16_t* pre_out0 = pre_out + hk * out_row_stride;
          float16_t* pre_out1 = pre_out0 + out_row_stride;
#ifdef __aarch64__
          // first
          if (1) {
            float16_t* ptr_out0 = pre_out0;
            float16_t* ptr_out1 = pre_out1;
            float16x8_t w0 = vld1q_f16(wc0);       // w0, v23
            float16x8_t w1 = vld1q_f16(wc0 + 8);   // w1, v24
            float16x8_t w2 = vld1q_f16(wc0 + 16);  // w2, v25
            float16x8_t w3 = vld1q_f16(wc0 + 24);  // w3, v26
            float16x8_t w4 = vld1q_f16(wc0 + 32);  // w4, v27
            float16x8_t w5 = vld1q_f16(wc0 + 40);  // w5, v28
            float16x8_t w6 = vld1q_f16(wc0 + 48);  // w6, v29
            float16x8_t w7 = vld1q_f16(wc0 + 56);  // w7, v30
            float16x8_t w8 = vld1q_f16(wc0 + 64);  // w8, v31
            const float16_t* r0 = inr0;
            const float16_t* r1 = inr1;
            const float16_t* r2 = inr2;
            const float16_t* r3 = inr3;
            const float16_t* r4 = inr4;

            int cnt = w_loop;
            // clang-format off
            asm volatile(
                INIT_FIRST COMPUTE RESULT_FIRST
                : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1),
                [r2] "+r"(r2),[r3] "+r"(r3), [r4] "+r"(r4),
                [ptr_out0] "+r"(ptr_out0),
                [ptr_out1] "+r"(ptr_out1)
                : [w0] "w"(w0),
                [w1] "w"(w1), [w2] "w"(w2),
                [w3] "w"(w3), [w4] "w"(w4),
                [w5] "w"(w5), [w6] "w"(w6),
                [w7] "w"(w7), [w8] "w"(w8)
                : "cc","memory","v0","v1","v2","v3","v4",
                "v5","v6","v7","v8","v9","v10","v11","v12","v13",
                "v14","v15","v16","v17","v18","v19","v20","v21","v22");
            // clang-format on
            wc0 += 9 * OUT_C_BLOCK;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
            inr4 += win_round;
          }
          for (int i = 0; i < ic - 1; ++i) {
            float16_t* ptr_out0 = pre_out0;
            float16_t* ptr_out1 = pre_out1;

            float16x8_t w0 = vld1q_f16(wc0);       // w0, v23
            float16x8_t w1 = vld1q_f16(wc0 + 4);   // w1, v24
            float16x8_t w2 = vld1q_f16(wc0 + 8);   // w2, v25
            float16x8_t w3 = vld1q_f16(wc0 + 12);  // w3, v26
            float16x8_t w4 = vld1q_f16(wc0 + 16);  // w4, v27
            float16x8_t w5 = vld1q_f16(wc0 + 20);  // w5, v28
            float16x8_t w6 = vld1q_f16(wc0 + 24);  // w6, v29
            float16x8_t w7 = vld1q_f16(wc0 + 28);  // w7, v30
            float16x8_t w8 = vld1q_f16(wc0 + 32);  // w8, v31

            const float16_t* r0 = inr0;
            const float16_t* r1 = inr1;
            const float16_t* r2 = inr2;
            const float16_t* r3 = inr3;
            const float16_t* r4 = inr4;

            int cnt = w_loop;
            // clang-format off
            asm volatile(
            INIT COMPUTE RESULT
            : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1),
              [r2] "+r"(r2),[r3] "+r"(r3), [r4] "+r"(r4),
              [ptr_out0] "+r"(ptr_out0),
              [ptr_out1] "+r"(ptr_out1)
            : [w0] "w"(w0),
              [w1] "w"(w1), [w2] "w"(w2),
              [w3] "w"(w3), [w4] "w"(w4),
              [w5] "w"(w5), [w6] "w"(w6),
              [w7] "w"(w7), [w8] "w"(w8)
            : "cc","memory","v0","v1","v2","v3","v4",
              "v5","v6","v7","v8","v9","v10","v11","v12","v13",
              "v14","v15","v16","v17","v18","v19","v20","v21","v22");
            // clang-format on
            wc0 += 9 * OUT_C_BLOCK;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
            inr4 += win_round;
          }
#else   // not __aarch64__
#endif  // __aarch64__
          block_inr0 = block_inr4;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }
        write_to_oc8_fp16(pre_out,
                          dout_batch,
                          c,
                          c + OUT_C_BLOCK,
                          h,
                          h + h_kernel,
                          0,
                          wout_round,
                          oc,
                          oh,
                          ow,
                          flag_act,
                          alpha,
                          bias_ptr);
      }
    }
  }
}
}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
