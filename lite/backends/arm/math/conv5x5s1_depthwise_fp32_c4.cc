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

#include <arm_neon.h>
#include "lite/backends/arm/math/conv_block_utils.h"
#include "lite/backends/arm/math/conv_depthwise.h"
#include "lite/core/context.h"
#include "lite/operators/op_params.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {

#define ROUNDUP(a, b) ((((a) + (b)-1) / (b)) * (b))
#ifdef __aarch64__
void conv_depthwise_5x5s1_fp32(float* dout,
                               const float* din,
                               const float* weights,
                               const float* bias,
                               bool flag_bias,
                               bool flag_relu,
                               int num,
                               int chin,
                               int hin,
                               int win,
                               int hout,
                               int wout,
                               int padw,
                               int padh,
                               const operators::ConvParam& param,
                               ARMContext* ctx) {
  const int threads = ctx->threads();
  int llc_size = ctx->llc_size() / 4;
  auto act_param = param.activation_param;
  const int hout_c_block = 4;
  const int hout_r_kernel = 2;
  const int wout_block = 4;
  const int wout_round = ((wout + wout_block - 1) / wout_block) * wout_block;
  const int win_round = wout_round + 4;

  //! get h block
  //! llc_size = threads * win_round * hout_c_block * hin_r_block *
  //! sizeof(float)
  //! + wout_round * hout_c_block * hout_r_block * threads * sizeof(float)
  //! win_round = wout_round + 4
  //! hin_r_block = hout_r_block + 4
  int hout_r_block = (llc_size - 16 * win_round * hout_c_block * threads) /
                     (win_round * hout_c_block * threads * 4 +
                      hout_c_block * wout_round * threads * 4);
  hout_r_block = hout_r_block > hout ? hout : hout_r_block;
  hout_r_block =
      ((hout_r_block + hout_r_kernel - 1) / hout_r_kernel) * hout_r_kernel;
  hout_r_block = hout_r_block < hout_r_kernel ? hout_r_kernel : hout_r_block;

  const int hin_r_block = hout_r_block + 4;

  float* tmp_work_space = ctx->workspace_data<float>();
  float ptr_zero[win_round];  // NOLINT
  memset(ptr_zero, 0, sizeof(float) * win_round);
  float ptr_write[wout_round];  // NOLINT

  int in_len = win_round * hout_c_block;
  int pre_in_size = hin_r_block * in_len;
  pre_in_size = ROUNDUP(pre_in_size, 4);
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  float* tmp_din = tmp_work_space;

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = 25;  // kernel_w * kernel_h;

  int ws = -padw;
  int we = ws + win_round;
  int w_loop = wout_round / 4;
  int chout = chin;

  int out_row_stride = hout_c_block * wout_round;
  for (int n = 0; n < num; ++n) {
    const float* din_batch = din + n * chin * size_in_channel;
    float* dout_batch = dout + n * chout * size_out_channel;
    for (int h = 0; h < hout; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > hout) {
        h_kernel = hout - h;
      }
      int hs = h - padh;
      int he = hs + h_kernel + 4;

#pragma omp parallel for num_threads(threads)
      for (int c = 0; c < chout; c += hout_c_block) {
#ifdef ARM_WITH_OMP
        float* pre_din =
            tmp_din + omp_get_thread_num() * (pre_in_size + pre_out_size);
        float* pre_out = pre_din + pre_in_size;
#else
        float* pre_din = tmp_din;
        float* pre_out = pre_din + pre_in_size;
#endif
        prepack_input_nxwc4_dw(
            din_batch, pre_din, c, hs, he, ws, we, chin, win, hin, ptr_zero);
        const float* block_inr0 = pre_din;
        const float* block_inr1 = block_inr0 + in_len;
        const float* block_inr2 = block_inr1 + in_len;
        const float* block_inr3 = block_inr2 + in_len;
        const float* block_inr4 = block_inr3 + in_len;
        const float* block_inr5 = block_inr4 + in_len;

        const float* weight_c = weights + c * w_stride;
        float bias_local[4] = {0, 0, 0, 0};
        if (flag_bias) {
          bias_local[0] = bias[c];
          bias_local[1] = bias[c + 1];
          bias_local[2] = bias[c + 2];
          bias_local[3] = bias[c + 3];
        }
        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          int cnt = w_loop;
          const float* inr0 = block_inr0;
          const float* inr1 = block_inr1;
          const float* inr2 = block_inr2;
          const float* inr3 = block_inr3;
          const float* inr4 = block_inr4;
          const float* inr5 = block_inr5;

          float* ptr_out0 = pre_out + hk * out_row_stride;
          float* ptr_out1 = ptr_out0 + out_row_stride;
          // clang-format off
          auto wptr = weight_c;
          asm volatile(
              "ldr  q24,  [%[bias]]   \n" /* load bias to out00 */
              "ld1  {v0.4s,  v1.4s,  v2.4s,  v3.4s},  [%[wc]],    #64 \n" /* load w0-w3 */
              "ld1  {v8.4s,  v9.4s,  v10.4s, v11.4s}, [%[inr0]],  #64 \n" /* load inr0, 0-3 */
              "1:\n"
              "ld1  {v16.4s, v17.4s, v18.4s, v19.4s}, [%[inr1]],  #64 \n" /* load inr1, 0-3 */
              "mov  v25.16b,  v24.16b  \n" /* mov bias to out01 */
              "mov  v26.16b,  v24.16b  \n" /* mov bias to out02 */
              "mov  v27.16b,  v24.16b  \n" /* mov bias to out03 */
              "mov  v28.16b,  v24.16b  \n" /* mov bias to out10 */
              "mov  v29.16b,  v24.16b  \n" /* mov bias to out11 */
              "mov  v30.16b,  v24.16b  \n" /* mov bias to out12 */
              "mov  v31.16b,  v24.16b  \n" /* mov bias to out13 */
              //   out row0
              "fmla v24.4s, v8.4s,  v0.4s  \n"  /* out00 = w0 * inr00 */
              "fmla v25.4s, v9.4s,  v0.4s  \n"  /* out01 = w0 * inr01 */
              "ldp  q12,  q13,  [%[inr0]]  \n"  /* load inr0, 4-5 */
              "fmla v26.4s, v10.4s, v0.4s  \n"  /* out02 = w0 * inr02 */
              "fmla v27.4s, v11.4s, v0.4s  \n"  /* out03 = w0 * inr03 */
              "fmla v28.4s, v16.4s, v0.4s  \n"  /* out10 = w0 * inr10 */
              "fmla v29.4s, v17.4s, v0.4s  \n"  /* out11 = w0 * inr11 */
              "ldp  q20,  q21,  [%[inr1]]  \n"  /* load inr1, 4-5 */
              "fmla v30.4s, v18.4s, v0.4s  \n"  /* out12 = w0 * inr12 */
              "fmla v31.4s, v19.4s, v0.4s  \n"  /* out13 = w0 * inr13 */
              "fmla v24.4s, v9.4s,  v1.4s  \n"  /* out00 = w1 * inr01 */
              "fmla v25.4s, v10.4s, v1.4s  \n"  /* out01 = w1 * inr02 */
              "fmla v26.4s, v11.4s, v1.4s  \n"  /* out02 = w1 * inr03 */
              "fmla v27.4s, v12.4s, v1.4s  \n"  /* out03 = w1 * inr04 */
              "ldp  q14,  q15,  [%[inr0], #32]  \n" /* load inr0, 6-7 */
              "fmla v28.4s, v17.4s, v1.4s  \n"  /* out10 = w1 * inr11 */
              "fmla v29.4s, v18.4s, v1.4s  \n"  /* out11 = w1 * inr12 */
              "fmla v30.4s, v19.4s, v1.4s  \n"  /* out12 = w1 * inr13 */
              "fmla v31.4s, v20.4s, v1.4s  \n"  /* out13 = w1 * inr14 */
              "fmla v24.4s, v10.4s, v2.4s  \n"  /* out00 = w2 * inr02 */
              "fmla v25.4s, v11.4s, v2.4s  \n"  /* out01 = w2 * inr03 */
              "fmla v26.4s, v12.4s, v2.4s  \n"  /* out02 = w2 * inr04 */
              "fmla v27.4s, v13.4s, v2.4s  \n"  /* out03 = w2 * inr05 */
              "ldp  q22,  q23,  [%[inr1], #32]  \n" /* load inr1, 6-7 */
              "fmla v28.4s, v18.4s, v2.4s  \n"  /* out10 = w2 * inr12 */
              "fmla v29.4s, v19.4s, v2.4s  \n"  /* out11 = w2 * inr13 */
              "fmla v30.4s, v20.4s, v2.4s  \n"  /* out12 = w2 * inr14 */
              "fmla v31.4s, v21.4s, v2.4s  \n"  /* out13 = w2 * inr15 */
              "ldp  q4, q5, [%[wc]],  #32  \n"  /* load w4-w5 */
              "fmla v24.4s, v11.4s, v3.4s  \n"  /* out00 = w3 * inr03 */
              "fmla v25.4s, v12.4s, v3.4s  \n"  /* out01 = w3 * inr04 */
              "fmla v26.4s, v13.4s, v3.4s  \n"  /* out02 = w3 * inr05 */
              "fmla v27.4s, v14.4s, v3.4s  \n"  /* out03 = w3 * inr06 */
              "ldp  q6, q7, [%[wc]],  #32  \n"  /* load w6-w7 */
              "fmla v28.4s, v19.4s, v3.4s  \n"  /* out10 = w3 * inr13 */
              "fmla v29.4s, v20.4s, v3.4s  \n"  /* out11 = w3 * inr14 */
              "fmla v30.4s, v21.4s, v3.4s  \n"  /* out12 = w3 * inr15 */
              "fmla v31.4s, v22.4s, v3.4s  \n"  /* out13 = w3 * inr16 */
              "fmla v24.4s, v12.4s, v4.4s  \n"  /* out00 = w4 * inr04 */
              "fmla v25.4s, v13.4s, v4.4s  \n"  /* out01 = w4 * inr05 */
              "fmla v26.4s, v14.4s, v4.4s  \n"  /* out02 = w4 * inr06 */
              "fmla v27.4s, v15.4s, v4.4s  \n"  /* out03 = w4 * inr07 */
              "ldp  q8, q9, [%[inr2]], #32 \n"  /* load inr2, 0-1 */
              "fmla v28.4s, v20.4s, v4.4s  \n"  /* out10 = w4 * inr14 */
              "fmla v29.4s, v21.4s, v4.4s  \n"  /* out11 = w4 * inr15 */
              "fmla v30.4s, v22.4s, v4.4s  \n"  /* out12 = w4 * inr16 */
              "fmla v31.4s, v23.4s, v4.4s  \n"  /* out13 = w4 * inr17 */
              "ldp q10, q11, [%[inr2]], #32\n"  /* load inr2, 2-3 */
              //   out row1
              "fmla v24.4s, v16.4s, v5.4s  \n"  /* out00 = w5 * inr10 */
              "fmla v25.4s, v17.4s, v5.4s  \n"  /* out01 = w5 * inr11 */
              "fmla v26.4s, v18.4s, v5.4s  \n"  /* out02 = w5 * inr12 */
              "fmla v27.4s, v19.4s, v5.4s  \n"  /* out03 = w5 * inr13 */
              "ldp  q12,  q13,  [%[inr2]]  \n"  /* load inr2, 4-5 */
              "fmla v28.4s, v8.4s,  v5.4s  \n"  /* out10 = w5 * inr20 */
              "fmla v29.4s, v9.4s,  v5.4s  \n"  /* out11 = w5 * inr21 */
              "fmla v30.4s, v10.4s, v5.4s  \n"  /* out12 = w5 * inr22 */
              "fmla v31.4s, v11.4s, v5.4s  \n"  /* out13 = w5 * inr23 */
              "fmla v24.4s, v17.4s, v6.4s  \n"  /* out00 = w6 * inr11 */
              "fmla v25.4s, v18.4s, v6.4s  \n"  /* out01 = w6 * inr12 */
              "fmla v26.4s, v19.4s, v6.4s  \n"  /* out02 = w6 * inr13 */
              "fmla v27.4s, v20.4s, v6.4s  \n"  /* out03 = w6 * inr14 */
              "ldp q14, q15, [%[inr2], #32]\n"  /* load inr2, 6-7 */
              "fmla v28.4s, v9.4s,  v6.4s  \n"  /* out10 = w6 * inr21 */
              "fmla v29.4s, v10.4s, v6.4s  \n"  /* out11 = w6 * inr22 */
              "fmla v30.4s, v11.4s, v6.4s  \n"  /* out12 = w6 * inr23 */
              "fmla v31.4s, v12.4s, v6.4s  \n"  /* out13 = w6 * inr24 */
              "fmla v24.4s, v18.4s, v7.4s  \n"  /* out00 = w7 * inr12 */
              "fmla v25.4s, v19.4s, v7.4s  \n"  /* out01 = w7 * inr13 */
              "fmla v26.4s, v20.4s, v7.4s  \n"  /* out02 = w7 * inr14 */
              "fmla v27.4s, v21.4s, v7.4s  \n"  /* out03 = w7 * inr15 */
              "ldp  q0, q1, [%[wc]],  #32  \n"  /* load w8-w9 */
              "fmla v28.4s, v10.4s, v7.4s  \n"  /* out10 = w7 * inr22 */
              "fmla v29.4s, v11.4s, v7.4s  \n"  /* out11 = w7 * inr23 */
              "fmla v30.4s, v12.4s, v7.4s  \n"  /* out12 = w7 * inr24 */
              "fmla v31.4s, v13.4s, v7.4s  \n"  /* out13 = w7 * inr25 */
              "fmla v24.4s, v19.4s, v0.4s  \n"  /* out00 = w8 * inr13 */
              "fmla v25.4s, v20.4s, v0.4s  \n"  /* out01 = w8 * inr14 */
              "fmla v26.4s, v21.4s, v0.4s  \n"  /* out02 = w8 * inr15 */
              "fmla v27.4s, v22.4s, v0.4s  \n"  /* out03 = w8 * inr16 */
              "ldp  q2, q3, [%[wc]],  #32  \n"  /* load w10-w11 */
              "fmla v28.4s, v11.4s, v0.4s  \n"  /* out10 = w8 * inr23 */
              "fmla v29.4s, v12.4s, v0.4s  \n"  /* out11 = w8 * inr24 */
              "fmla v30.4s, v13.4s, v0.4s  \n"  /* out12 = w8 * inr25 */
              "fmla v31.4s, v14.4s, v0.4s  \n"  /* out13 = w8 * inr26 */
              "ldp q16, q17, [%[inr3]], #32\n"  /* load inr3, 0-1 */
              "fmla v24.4s, v20.4s, v1.4s  \n"  /* out00 = w9 * inr14 */
              "fmla v25.4s, v21.4s, v1.4s  \n"  /* out01 = w9 * inr15 */
              "fmla v26.4s, v22.4s, v1.4s  \n"  /* out02 = w9 * inr16 */
              "fmla v27.4s, v23.4s, v1.4s  \n"  /* out03 = w9 * inr17 */
              "ldp q18, q19, [%[inr3]], #32\n"  /* load inr3, 2-3 */
              "fmla v28.4s, v12.4s, v1.4s  \n"  /* out10 = w9 * inr24 */
              "fmla v29.4s, v13.4s, v1.4s  \n"  /* out11 = w9 * inr25 */
              "fmla v30.4s, v14.4s, v1.4s  \n"  /* out12 = w9 * inr26 */
              "fmla v31.4s, v15.4s, v1.4s  \n"  /* out13 = w9 * inr27 */
              //   out row2
              "fmla v24.4s, v8.4s,  v2.4s  \n"  /* out00 = w10 * inr20 */
              "fmla v25.4s, v9.4s,  v2.4s  \n"  /* out01 = w10 * inr21 */
              "fmla v26.4s, v10.4s, v2.4s  \n"  /* out02 = w10 * inr22 */
              "fmla v27.4s, v11.4s, v2.4s  \n"  /* out03 = w10 * inr23 */
              "ldp  q4, q5, [%[wc]],  #32  \n"  /* load w12-w13 */
              "fmla v28.4s, v16.4s, v2.4s  \n"  /* out10 = w10 * inr30 */
              "fmla v29.4s, v17.4s, v2.4s  \n"  /* out11 = w10 * inr31 */
              "fmla v30.4s, v18.4s, v2.4s  \n"  /* out12 = w10 * inr32 */
              "fmla v31.4s, v19.4s, v2.4s  \n"  /* out13 = w10 * inr33 */
              "ldp  q20,  q21,  [%[inr3]]  \n"  /* load inr3, 4-5 */
              "fmla v24.4s, v9.4s,  v3.4s  \n"  /* out00 = w11 * inr21 */
              "fmla v25.4s, v10.4s, v3.4s  \n"  /* out01 = w11 * inr22 */
              "fmla v26.4s, v11.4s, v3.4s  \n"  /* out02 = w11 * inr23 */
              "fmla v27.4s, v12.4s, v3.4s  \n"  /* out03 = w11 * inr24 */
              "ldp q22, q23, [%[inr3], #32]\n"  /* load inr3, 6-7 */
              "fmla v28.4s, v17.4s, v3.4s  \n"  /* out10 = w11 * inr31 */
              "fmla v29.4s, v18.4s, v3.4s  \n"  /* out11 = w11 * inr32 */
              "fmla v30.4s, v19.4s, v3.4s  \n"  /* out12 = w11 * inr33 */
              "fmla v31.4s, v20.4s, v3.4s  \n"  /* out13 = w11 * inr34 */
              "fmla v24.4s, v10.4s, v4.4s  \n"  /* out00 = w12 * inr22 */
              "fmla v25.4s, v11.4s, v4.4s  \n"  /* out01 = w12 * inr23 */
              "fmla v26.4s, v12.4s, v4.4s  \n"  /* out02 = w12 * inr24 */
              "fmla v27.4s, v13.4s, v4.4s  \n"  /* out03 = w12 * inr25 */
              "ldp  q6, q7, [%[wc]],  #32  \n"  /* load w14-w15 */
              "fmla v28.4s, v18.4s, v4.4s  \n"  /* out10 = w12 * inr32 */
              "fmla v29.4s, v19.4s, v4.4s  \n"  /* out11 = w12 * inr33 */
              "fmla v30.4s, v20.4s, v4.4s  \n"  /* out12 = w12 * inr34 */
              "fmla v31.4s, v21.4s, v4.4s  \n"  /* out13 = w12 * inr35 */
              "fmla v24.4s, v11.4s, v5.4s  \n"  /* out00 = w13 * inr23 */
              "fmla v25.4s, v12.4s, v5.4s  \n"  /* out01 = w13 * inr24 */
              "fmla v26.4s, v13.4s, v5.4s  \n"  /* out02 = w13 * inr25 */
              "fmla v27.4s, v14.4s, v5.4s  \n"  /* out03 = w13 * inr26 */
              "ldp  q8, q9, [%[inr4]], #32 \n"  /* load inr4, 0-1 */
              "fmla v28.4s, v19.4s, v5.4s  \n"  /* out10 = w13 * inr33 */
              "fmla v29.4s, v20.4s, v5.4s  \n"  /* out11 = w13 * inr34 */
              "fmla v30.4s, v21.4s, v5.4s  \n"  /* out12 = w13 * inr35 */
              "fmla v31.4s, v22.4s, v5.4s  \n"  /* out13 = w13 * inr36 */
              "fmla v24.4s, v12.4s, v6.4s  \n"  /* out00 = w14 * inr24 */
              "fmla v25.4s, v13.4s, v6.4s  \n"  /* out01 = w14 * inr25 */
              "fmla v26.4s, v14.4s, v6.4s  \n"  /* out02 = w14 * inr26 */
              "fmla v27.4s, v15.4s, v6.4s  \n"  /* out03 = w14 * inr27 */
              "ldp q10, q11, [%[inr4]], #32\n"  /* load inr4, 2-3 */
              "fmla v28.4s, v20.4s, v6.4s  \n"  /* out10 = w14 * inr34 */
              "fmla v29.4s, v21.4s, v6.4s  \n"  /* out11 = w14 * inr35 */
              "fmla v30.4s, v22.4s, v6.4s  \n"  /* out12 = w14 * inr36 */
              "fmla v31.4s, v23.4s, v6.4s  \n"  /* out13 = w14 * inr37 */
              "ldp  q0, q1, [%[wc]],  #32  \n"  /* load w16-w17 */
              //   out row3
              "fmla v24.4s, v16.4s, v7.4s  \n"  /* out00 = w15 * inr30 */
              "fmla v25.4s, v17.4s, v7.4s  \n"  /* out01 = w15 * inr31 */
              "fmla v26.4s, v18.4s, v7.4s  \n"  /* out02 = w15 * inr32 */
              "fmla v27.4s, v19.4s, v7.4s  \n"  /* out03 = w15 * inr33 */
              "ldp  q12,  q13,  [%[inr4]]  \n"  /* load inr4, 4-5 */
              "fmla v28.4s, v8.4s,  v7.4s  \n"  /* out10 = w15 * inr40 */
              "fmla v29.4s, v9.4s,  v7.4s  \n"  /* out11 = w15 * inr41 */
              "fmla v30.4s, v10.4s, v7.4s  \n"  /* out12 = w15 * inr42 */
              "fmla v31.4s, v11.4s, v7.4s  \n"  /* out13 = w15 * inr42 */
              "ldp  q2, q3, [%[wc]],  #32  \n"  /* load w18-w19 */
              "fmla v24.4s, v17.4s, v0.4s  \n"  /* out00 = w16 * inr31 */
              "fmla v25.4s, v18.4s, v0.4s  \n"  /* out01 = w16 * inr32 */
              "fmla v26.4s, v19.4s, v0.4s  \n"  /* out02 = w16 * inr33 */
              "fmla v27.4s, v20.4s, v0.4s  \n"  /* out03 = w16 * inr34 */
              "ldp q14, q15, [%[inr4], #32]\n"  /* load inr4, 6-7 */
              "fmla v28.4s, v9.4s,  v0.4s  \n"  /* out10 = w16 * inr41 */
              "fmla v29.4s, v10.4s, v0.4s  \n"  /* out11 = w16 * inr42 */
              "fmla v30.4s, v11.4s, v0.4s  \n"  /* out12 = w16 * inr43 */
              "fmla v31.4s, v12.4s, v0.4s  \n"  /* out13 = w16 * inr44 */
              "fmla v24.4s, v18.4s, v1.4s  \n"  /* out00 = w17 * inr32 */
              "fmla v25.4s, v19.4s, v1.4s  \n"  /* out01 = w17 * inr33 */
              "fmla v26.4s, v20.4s, v1.4s  \n"  /* out02 = w17 * inr34 */
              "fmla v27.4s, v21.4s, v1.4s  \n"  /* out03 = w17 * inr35 */
              "ldp  q4, q5, [%[wc]],  #32  \n"  /* load w20-w21 */
              "fmla v28.4s, v10.4s, v1.4s  \n"  /* out10 = w17 * inr42 */
              "fmla v29.4s, v11.4s, v1.4s  \n"  /* out11 = w17 * inr43 */
              "fmla v30.4s, v12.4s, v1.4s  \n"  /* out12 = w17 * inr44 */
              "fmla v31.4s, v13.4s, v1.4s  \n"  /* out13 = w17 * inr45 */
              "fmla v24.4s, v19.4s, v2.4s  \n"  /* out00 = w18 * inr33 */
              "fmla v25.4s, v20.4s, v2.4s  \n"  /* out01 = w18 * inr34 */
              "fmla v26.4s, v21.4s, v2.4s  \n"  /* out02 = w18 * inr35 */
              "fmla v27.4s, v22.4s, v2.4s  \n"  /* out03 = w18 * inr36 */
              "ldp q16, q17, [%[inr5]], #32\n"  /* load inr5, 0-1 */
              "fmla v28.4s, v11.4s, v2.4s  \n"  /* out10 = w18 * inr43 */
              "fmla v29.4s, v12.4s, v2.4s  \n"  /* out11 = w18 * inr44 */
              "fmla v30.4s, v13.4s, v2.4s  \n"  /* out12 = w18 * inr45 */
              "fmla v31.4s, v14.4s, v2.4s  \n"  /* out13 = w18 * inr46 */
              "fmla v24.4s, v20.4s, v3.4s  \n"  /* out00 = w19 * inr34 */
              "fmla v25.4s, v21.4s, v3.4s  \n"  /* out01 = w19 * inr35 */
              "fmla v26.4s, v22.4s, v3.4s  \n"  /* out02 = w19 * inr36 */
              "fmla v27.4s, v23.4s, v3.4s  \n"  /* out03 = w19 * inr37 */
              "ldp q18, q19, [%[inr5]], #32\n"  /* load inr5, 2-3 */
              "fmla v28.4s, v12.4s, v3.4s  \n"  /* out10 = w19 * inr44 */
              "fmla v29.4s, v13.4s, v3.4s  \n"  /* out11 = w19 * inr45 */
              "fmla v30.4s, v14.4s, v3.4s  \n"  /* out12 = w19 * inr46 */
              "fmla v31.4s, v15.4s, v3.4s  \n"  /* out13 = w19 * inr47 */
              //   out row4
              "fmla v24.4s, v8.4s,  v4.4s  \n"  /* out00 = w20 * inr40 */
              "fmla v25.4s, v9.4s,  v4.4s  \n"  /* out01 = w20 * inr41 */
              "fmla v26.4s, v10.4s, v4.4s  \n"  /* out02 = w20 * inr42 */
              "fmla v27.4s, v11.4s, v4.4s  \n"  /* out03 = w20 * inr43 */
              "ldp  q20,  q21,  [%[inr5]]  \n"  /* load inr5, 4-5 */
              "fmla v28.4s, v16.4s, v4.4s  \n"  /* out10 = w20 * inr50 */
              "fmla v29.4s, v17.4s, v4.4s  \n"  /* out11 = w20 * inr51 */
              "fmla v30.4s, v18.4s, v4.4s  \n"  /* out12 = w20 * inr52 */
              "fmla v31.4s, v19.4s, v4.4s  \n"  /* out13 = w20 * inr53 */
              "ldp  q6, q7, [%[wc]],  #32  \n"  /* load w22-w23 */
              "fmla v24.4s, v9.4s,  v5.4s  \n"  /* out00 = w21 * inr41 */
              "fmla v25.4s, v10.4s, v5.4s  \n"  /* out01 = w21 * inr42 */
              "fmla v26.4s, v11.4s, v5.4s  \n"  /* out02 = w21 * inr43 */
              "fmla v27.4s, v12.4s, v5.4s  \n"  /* out03 = w21 * inr44 */
              "ldp q22, q23, [%[inr5], #32]\n"  /* load inr5, 6-7 */
              "fmla v28.4s, v17.4s, v5.4s  \n"  /* out10 = w21 * inr51 */
              "fmla v29.4s, v18.4s, v5.4s  \n"  /* out11 = w21 * inr52 */
              "fmla v30.4s, v19.4s, v5.4s  \n"  /* out12 = w21 * inr53 */
              "fmla v31.4s, v20.4s, v5.4s  \n"  /* out13 = w21 * inr54 */
              "ldp q8, q9, [%[inr0]], #32  \n"  /* load inr0, 0-1 */
              "fmla v24.4s, v10.4s, v6.4s  \n"  /* out00 = w22 * inr42 */
              "fmla v25.4s, v11.4s, v6.4s  \n"  /* out01 = w22 * inr43 */
              "fmla v26.4s, v12.4s, v6.4s  \n"  /* out02 = w22 * inr44 */
              "fmla v27.4s, v13.4s, v6.4s  \n"  /* out03 = w22 * inr45 */
              "ldp q4, q5, [%[wc]], #-384  \n"  /* load w24 */
              "fmla v28.4s, v18.4s, v6.4s  \n"  /* out10 = w22 * inr52 */
              "fmla v29.4s, v19.4s, v6.4s  \n"  /* out11 = w22 * inr53 */
              "fmla v30.4s, v20.4s, v6.4s  \n"  /* out12 = w22 * inr54 */
              "fmla v31.4s, v21.4s, v6.4s  \n"  /* out13 = w22 * inr55 */
              "ldp  q0, q1, [%[wc]],  #32  \n"  /* load w0-w1 */
              "fmla v24.4s, v11.4s, v7.4s  \n"  /* out00 = w23 * inr43 */
              "fmla v25.4s, v12.4s, v7.4s  \n"  /* out01 = w23 * inr44 */
              "fmla v26.4s, v13.4s, v7.4s  \n"  /* out02 = w23 * inr45 */
              "fmla v27.4s, v14.4s, v7.4s  \n"  /* out03 = w23 * inr46 */
              "ldp  q2, q3, [%[wc]],  #32  \n"  /* load w1-w2 */
              "fmla v28.4s, v19.4s, v7.4s  \n"  /* out10 = w23 * inr53 */
              "fmla v29.4s, v20.4s, v7.4s  \n"  /* out11 = w23 * inr54 */
              "fmla v30.4s, v21.4s, v7.4s  \n"  /* out12 = w23 * inr55 */
              "fmla v31.4s, v22.4s, v7.4s  \n"  /* out13 = w23 * inr56 */
              "ldp q10, q11, [%[inr0]], #32\n"  /* load inr0, 2-3 */
              "fmla v24.4s, v12.4s, v4.4s  \n"  /* out00 = w24 * inr44 */
              "fmla v25.4s, v13.4s, v4.4s  \n"  /* out01 = w24 * inr45 */
              "fmla v26.4s, v14.4s, v4.4s  \n"  /* out02 = w24 * inr46 */
              "fmla v27.4s, v15.4s, v4.4s  \n"  /* out03 = w24 * inr47 */
              "stp q24, q25, [%[out0]], #32\n"  /* store outr0, 0-1 */
              "fmla v28.4s, v20.4s, v4.4s  \n"  /* out10 = w24 * inr54 */
              "fmla v29.4s, v21.4s, v4.4s  \n"  /* out11 = w24 * inr55 */
              "stp q26, q27, [%[out0]], #32\n"  /* store outr0, 2-3 */
              "fmla v30.4s, v22.4s, v4.4s  \n"  /* out12 = w24 * inr56 */
              "fmla v31.4s, v23.4s, v4.4s  \n"  /* out13 = w24 * inr57 */
              "ldr  q24,  [%[bias]]        \n"  /* load bias to out00 */
              "subs   %w[cnt], %w[cnt], #1\n"   /*  cnt = cnt - 1   */
              "stp q28, q29, [%[out1]], #32\n"  /* store outr1, 0-1 */
              "stp q30, q31, [%[out1]], #32\n"  /* store outr1, 2-3 */
              "bne    1b\n"
              : [cnt] "+r"(cnt),
                [inr0] "+r"(inr0),
                [inr1] "+r"(inr1),
                [inr2] "+r"(inr2),
                [inr3] "+r"(inr3),
                [inr4] "+r"(inr4),
                [inr5] "+r"(inr5),
                [wc] "+r"(wptr),
                [out0] "+r"(ptr_out0),
                [out1] "+r"(ptr_out1)
              : [bias] "r"(bias_local)
              : "cc","memory",
                "v0","v1","v2","v3","v4","v5","v6","v7",
                "v8","v9","v10","v11","v12","v13",
                "v14","v15","v16","v17","v18","v19",
                "v20","v21","v22","v23","v24","v25",
                "v26","v27","v28","v29","v30","v31"
              );
          // clang-format on
          block_inr0 = block_inr2;
          block_inr1 = block_inr3;
          block_inr2 = block_inr4;
          block_inr3 = block_inr5;
          block_inr4 = block_inr3 + in_len;
          block_inr5 = block_inr4 + in_len;
        }
        write_to_output_c4_fp32(pre_out,
                                dout_batch,
                                c,
                                c + hout_c_block,
                                h,
                                h + h_kernel,
                                0,
                                wout_round,
                                chout,
                                hout,
                                wout,
                                flag_relu,
                                ptr_write,
                                &act_param);
      }
    }
  }
}
#else  // __aarch64__
void conv_depthwise_5x5s1_fp32(float* dout,
                               const float* din,
                               const float* weights,
                               const float* bias,
                               bool flag_bias,
                               bool flag_relu,
                               int num,
                               int chin,
                               int hin,
                               int win,
                               int hout,
                               int wout,
                               int padw,
                               int padh,
                               const operators::ConvParam& param,
                               ARMContext* ctx) {
  const int threads = ctx->threads();
  int llc_size = ctx->llc_size() / 4;
  auto act_param = param.activation_param;
  const int hout_c_block = 4;
  const int hout_r_kernel = 1;
  const int wout_block = 4;
  const int wout_round = ((wout + wout_block - 1) / wout_block) * wout_block;
  const int win_round = wout_round + 4;

  //! get h block
  //! llc_size = threads * win_round * hout_c_block * hin_r_block *
  //! sizeof(float)
  //! + wout_round * hout_c_block * hout_r_block * threads * sizeof(float)
  //! win_round = wout_round + 4
  //! hin_r_block = hout_r_block + 4
  int hout_r_block = (llc_size - 16 * win_round * hout_c_block * threads) /
                     (win_round * hout_c_block * threads * 4 +
                      hout_c_block * wout_round * threads * 4);
  hout_r_block = hout_r_block > hout ? hout : hout_r_block;
  hout_r_block =
      ((hout_r_block + hout_r_kernel - 1) / hout_r_kernel) * hout_r_kernel;
  hout_r_block = hout_r_block < hout_r_kernel ? hout_r_kernel : hout_r_block;

  const int hin_r_block = hout_r_block + 4;

  float* tmp_work_space = ctx->workspace_data<float>();
  float ptr_zero[win_round];  // NOLINT
  memset(ptr_zero, 0, sizeof(float) * win_round);
  float ptr_write[wout_round];  // NOLINT

  int in_len = win_round * hout_c_block;
  int pre_in_size = hin_r_block * in_len;
  pre_in_size = ROUNDUP(pre_in_size, 4);
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  float* tmp_din = tmp_work_space;

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = 25;  // kernel_w * kernel_h;

  int ws = -padw;
  int we = ws + win_round;
  int w_loop = wout_round / 4;
  int chout = chin;

  int out_row_stride = hout_c_block * wout_round;
  for (int n = 0; n < num; ++n) {
    const float* din_batch = din + n * chin * size_in_channel;
    float* dout_batch = dout + n * chout * size_out_channel;
    for (int h = 0; h < hout; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > hout) {
        h_kernel = hout - h;
      }
      int hs = h - padh;
      int he = hs + h_kernel + 4;

#pragma omp parallel for num_threads(threads)
      for (int c = 0; c < chout; c += hout_c_block) {
#ifdef ARM_WITH_OMP
        float* pre_din =
            tmp_din + omp_get_thread_num() * (pre_in_size + pre_out_size);
        float* pre_out = pre_din + pre_in_size;
#else
        float* pre_din = tmp_din;
        float* pre_out = pre_din + pre_in_size;
#endif
        prepack_input_nxwc4_dw(
            din_batch, pre_din, c, hs, he, ws, we, chin, win, hin, ptr_zero);
        const float* block_inr0 = pre_din;
        const float* block_inr1 = block_inr0 + in_len;
        const float* block_inr2 = block_inr1 + in_len;
        const float* block_inr3 = block_inr2 + in_len;
        const float* block_inr4 = block_inr3 + in_len;

        const float* weight_c = weights + c * w_stride;
        float bias_local[4] = {0, 0, 0, 0};
        if (flag_bias) {
          bias_local[0] = bias[c];
          bias_local[1] = bias[c + 1];
          bias_local[2] = bias[c + 2];
          bias_local[3] = bias[c + 3];
        }
        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          int cnt = w_loop;
          const float* inr0 = block_inr0;
          const float* inr1 = block_inr1;
          const float* inr2 = block_inr2;
          const float* inr3 = block_inr3;
          const float* inr4 = block_inr4;

          float* ptr_out0 = pre_out + hk * out_row_stride;
          // clang-format off
          auto wptr = weight_c;
          asm volatile(
              "vld1.32  {d24-d25},  [%[bias]]   \n" /* load bias to out00 */
              "vld1.32  {d0-d3},    [%[wc]]!    \n" /* load w0-w1 */
              "vld1.32  {d4-d7},    [%[wc]]!    \n" /* load w2-w3 */
              "vld1.32  {d8-d11},   [%[inr0]]!  \n" /* load inr0, 0-1 */
              "vld1.32  {d12-d15},  [%[inr0]]!  \n" /* load inr0, 2-3 */
              "1:\n"
              "vld1.32  {d16-d19},  [%[inr0]]!  \n" /* load inr0, 4-5 */
              "vmov.u32 q13,  q12 \n" /* mov bias to out01 */
              "vmov.u32 q14,  q12 \n" /* mov bias to out02 */
              "vmov.u32 q15,  q12 \n" /* mov bias to out03 */
              //  out row0
              "vmla.f32 q12,  q4,   q0  \n"   /* out00 = w0 * inr00 */
              "vmla.f32 q13,  q5,   q0  \n"   /* out01 = w0 * inr01 */
              "vmla.f32 q14,  q6,   q0  \n"   /* out02 = w0 * inr02 */
              "vmla.f32 q15,  q7,   q0  \n"   /* out03 = w0 * inr03 */
              "vld1.32  {d20-d23},  [%[inr0]]!  \n" /* load inr0, 6-7 */
              "sub    %[inr0], %[inr0], #64   \n" /* inr0 -= 64 */
              "vmla.f32 q12,  q5,   q1  \n"   /* out00 = w1 * inr01 */
              "vmla.f32 q13,  q6,   q1  \n"   /* out01 = w1 * inr02 */
              "vmla.f32 q14,  q7,   q1  \n"   /* out02 = w1 * inr03 */
              "vmla.f32 q15,  q8,   q1  \n"   /* out03 = w1 * inr04 */
              "vld1.32  {d8-d11},   [%[inr1]]!\n" /* load inr1, 0-1 */
              "vmla.f32 q12,  q6,   q2  \n"   /* out00 = w2 * inr02 */
              "vmla.f32 q13,  q7,   q2  \n"   /* out01 = w2 * inr03 */
              "vmla.f32 q14,  q8,   q2  \n"   /* out02 = w2 * inr04 */
              "vmla.f32 q15,  q9,   q2  \n"   /* out03 = w2 * inr05 */
              "vld1.32   {d0-d3},   [%[wc]]!  \n" /* load w4-w5 */
              "vmla.f32 q12,  q7,   q3  \n"   /* out00 = w3 * inr03 */
              "vmla.f32 q13,  q8,   q3  \n"   /* out01 = w3 * inr04 */
              "vmla.f32 q14,  q9,   q3  \n"   /* out02 = w3 * inr05 */
              "vmla.f32 q15,  q10,  q3  \n"   /* out03 = w3 * inr06 */
              "vld1.32  {d12-d15},  [%[inr1]]!\n" /* load inr1, 2-3 */
              "vmla.f32 q12,  q8,   q0  \n"   /* out00 = w4 * inr04 */
              "vmla.f32 q13,  q9,   q0  \n"   /* out01 = w4 * inr05 */
              "vmla.f32 q14,  q10,  q0  \n"   /* out02 = w4 * inr06 */
              "vmla.f32 q15,  q11,  q0  \n"   /* out03 = w4 * inr07 */
              "vld1.32   {d4-d7},   [%[wc]]!  \n" /* load w6-w7 */
              //  out row1
              "vmla.f32 q12,  q4,   q1  \n"   /* out00 = w5 * inr10 */
              "vmla.f32 q13,  q5,   q1  \n"   /* out01 = w5 * inr11 */
              "vmla.f32 q14,  q6,   q1  \n"   /* out02 = w5 * inr12 */
              "vmla.f32 q15,  q7,   q1  \n"   /* out03 = w5 * inr13 */
              "vld1.32  {d16-d19},  [%[inr1]]!\n" /* load inr1, 4-5 */
              "vmla.f32 q12,  q5,   q2  \n"   /* out00 = w6 * inr11 */
              "vmla.f32 q13,  q6,   q2  \n"   /* out01 = w6 * inr12 */
              "vmla.f32 q14,  q7,   q2  \n"   /* out02 = w6 * inr13 */
              "vmla.f32 q15,  q8,   q2  \n"   /* out03 = w6 * inr14 */
              "vld1.32   {d0-d3},   [%[wc]]!  \n" /* load w8-w9 */
              "vmla.f32 q12,  q6,   q3  \n"   /* out00 = w7 * inr12 */
              "vmla.f32 q13,  q7,   q3  \n"   /* out01 = w7 * inr13 */
              "vld1.32  {d20-d23},  [%[inr1]]!\n" /* load inr1, 6-7 */
              "vmla.f32 q14,  q8,   q3  \n"   /* out02 = w7 * inr14 */
              "vmla.f32 q15,  q9,   q3  \n"   /* out03 = w7 * inr15 */
              "sub    %[inr1], %[inr1], #64   \n" /* inr1 -= 64 */
              "vmla.f32 q12,  q7,   q0  \n"   /* out00 = w8 * inr13 */
              "vmla.f32 q13,  q8,   q0  \n"   /* out01 = w8 * inr14 */
              "vld1.32  {d8-d11},   [%[inr2]]!\n" /* load inr2, 0-1 */
              "vmla.f32 q14,  q9,   q0  \n"   /* out02 = w8 * inr15 */
              "vmla.f32 q15,  q10,  q0  \n"   /* out03 = w8 * inr16 */
              "vld1.32  {d4-d7},    [%[wc]]!  \n" /* load w10-w11 */
              "vmla.f32 q12,  q8,   q1  \n"   /* out00 = w9 * inr14 */
              "vmla.f32 q13,  q9,   q1  \n"   /* out01 = w9 * inr15 */
              "vld1.32  {d12-d15},  [%[inr2]]!\n" /* load inr2, 2-3 */
              "vmla.f32 q14,  q10,  q1  \n"   /* out02 = w9 * inr16 */
              "vmla.f32 q15,  q11,  q1  \n"   /* out03 = w9 * inr17 */
              //  out row3
              "vmla.f32 q12,  q4,   q2  \n"   /* out00 = w10 * inr20 */
              "vmla.f32 q13,  q5,   q2  \n"   /* out01 = w10 * inr21 */
              "vld1.32  {d16-d19},  [%[inr2]]!\n" /* load inr2, 4-5 */
              "vmla.f32 q14,  q6,   q2  \n"   /* out02 = w10 * inr22 */
              "vmla.f32 q15,  q7,   q2  \n"   /* out03 = w10 * inr23 */
              "vld1.32  {d0-d3},    [%[wc]]!  \n" /* load w12-w13 */
              "vmla.f32 q12,  q5,   q3  \n"   /* out00 = w11 * inr21 */
              "vmla.f32 q13,  q6,   q3  \n"   /* out01 = w11 * inr22 */
              "vld1.32  {d20-d23},  [%[inr2]]!\n" /* load inr2, 6-7 */
              "vmla.f32 q14,  q7,   q3  \n"   /* out02 = w11 * inr23 */
              "vmla.f32 q15,  q8,   q3  \n"   /* out03 = w11 * inr24 */
              "vld1.32  {d4-d7},    [%[wc]]!  \n" /* load w14-w15 */
              "sub    %[inr2], %[inr2], #64   \n" /* inr2 -= 64 */
              "vmla.f32 q12,  q6,   q0  \n"   /* out00 = w12 * inr22 */
              "vmla.f32 q13,  q7,   q0  \n"   /* out01 = w12 * inr23 */
              "vmla.f32 q14,  q8,   q0  \n"   /* out02 = w12 * inr24 */
              "vmla.f32 q15,  q9,   q0  \n"   /* out03 = w12 * inr25 */
              "vld1.32  {d8-d11},   [%[inr3]]!\n" /* load inr3, 0-1 */
              "vmla.f32 q12,  q7,   q1  \n"   /* out00 = w13 * inr23 */
              "vmla.f32 q13,  q8,   q1  \n"   /* out01 = w13 * inr24 */
              "vmla.f32 q14,  q9,   q1  \n"   /* out02 = w13 * inr25 */
              "vmla.f32 q15,  q10,  q1  \n"   /* out03 = w13 * inr26 */
              "vld1.32  {d0-d3},    [%[wc]]!  \n" /* load w16-w17 */
              "vmla.f32 q12,  q8,   q2  \n"   /* out00 = w14 * inr24 */
              "vmla.f32 q13,  q9,   q2  \n"   /* out01 = w14 * inr25 */
              "vld1.32  {d12-d15},  [%[inr3]]!\n" /* load inr3, 2-3 */
              "vmla.f32 q14,  q10,  q2  \n"   /* out02 = w14 * inr26 */
              "vmla.f32 q15,  q11,  q2  \n"   /* out03 = w14 * inr27 */
              //  out row3
              "vmla.f32 q12,  q4,   q3  \n"   /* out00 = w15 * inr30 */
              "vmla.f32 q13,  q5,   q3  \n"   /* out01 = w15 * inr31 */
              "vld1.32  {d16-d19},  [%[inr3]]!\n" /* load inr3, 4-5 */
              "vmla.f32 q14,  q6,   q3  \n"   /* out02 = w15 * inr32 */
              "vmla.f32 q15,  q7,   q3  \n"   /* out03 = w15 * inr33 */
              "vld1.32  {d4-d7},    [%[wc]]!  \n" /* load w18-w19 */
              "vmla.f32 q12,  q5,   q0  \n"   /* out00 = w16 * inr31 */
              "vmla.f32 q13,  q6,   q0  \n"   /* out01 = w16 * inr32 */
              "vld1.32  {d20-d23},  [%[inr3]]!\n" /* load inr3, 6-7 */
              "vmla.f32 q14,  q7,   q0  \n"   /* out02 = w16 * inr33 */
              "vmla.f32 q15,  q8,   q0  \n"   /* out03 = w16 * inr34 */
              "sub    %[inr3], %[inr3], #64   \n" /* inr3 -= 64 */
              "vmla.f32 q12,  q6,   q1  \n"   /* out00 = w17 * inr32 */
              "vmla.f32 q13,  q7,   q1  \n"   /* out01 = w17 * inr33 */
              "vmla.f32 q14,  q8,   q1  \n"   /* out02 = w17 * inr34 */
              "vmla.f32 q15,  q9,   q1  \n"   /* out03 = w17 * inr35 */
              "vld1.32  {d0-d3},    [%[wc]]!  \n" /* load w20-w21 */
              "vmla.f32 q12,  q7,   q2  \n"   /* out00 = w18 * inr33 */
              "vmla.f32 q13,  q8,   q2  \n"   /* out01 = w18 * inr34 */
              "vmla.f32 q14,  q9,   q2  \n"   /* out02 = w18 * inr35 */
              "vmla.f32 q15,  q10,  q2  \n"   /* out03 = w18 * inr36 */
              "vld1.32  {d8-d11},  [%[inr4]]!\n" /* load inr4, 0-1 */
              "vmla.f32 q12,  q8,   q3  \n"   /* out00 = w19 * inr34 */
              "vmla.f32 q13,  q9,   q3  \n"   /* out01 = w19 * inr35 */
              "vld1.32  {d12-d15},  [%[inr4]]!\n" /* load inr4, 2-3 */
              "vmla.f32 q14,  q10,  q3  \n"   /* out02 = w19 * inr36 */
              "vmla.f32 q15,  q11,  q3  \n"   /* out03 = w19 * inr37 */
              //  out row4
              "vmla.f32 q12,  q4,   q0  \n"   /* out00 = w20 * inr40 */
              "vmla.f32 q13,  q5,   q0  \n"   /* out01 = w20 * inr41 */
              "vld1.32  {d16-d19},  [%[inr4]]!\n" /* load inr4, 4-5 */
              "vmla.f32 q14,  q6,   q0  \n"   /* out02 = w20 * inr42 */
              "vmla.f32 q15,  q7,   q0  \n"   /* out03 = w20 * inr43 */
              "vld1.32  {d4-d7},    [%[wc]]!  \n" /* load w22-w23 */
              "vmla.f32 q12,  q5,   q1  \n"   /* out00 = w21 * inr41 */
              "vmla.f32 q13,  q6,   q1  \n"   /* out01 = w21 * inr42 */
              "vmla.f32 q14,  q7,   q1  \n"   /* out02 = w21 * inr43 */
              "vmla.f32 q15,  q8,   q1  \n"   /* out03 = w21 * inr44 */
              "vld1.32  {d20-d23},  [%[inr4]]!\n" /* load inr4, 6-7 */
              "vmla.f32 q12,  q6,   q2  \n"   /* out00 = w22 * inr42 */
              "vmla.f32 q13,  q7,   q2  \n"   /* out01 = w22 * inr43 */
              "vmla.f32 q14,  q8,   q2  \n"   /* out02 = w22 * inr44 */
              "vmla.f32 q15,  q9,   q2  \n"   /* out03 = w22 * inr45 */
              "vld1.32  {d4-d5},    [%[wc]]   \n" /* load w24 */
              "sub    %[inr4], %[inr4], #64   \n" /* inr4 -= 64 */
              "vmla.f32 q12,  q7,   q3  \n"   /* out00 = w23 * inr43 */
              "vmla.f32 q13,  q8,   q3  \n"   /* out01 = w23 * inr44 */
              "vld1.32  {d8-d11},   [%[inr0]]!\n" /* load inr0, 0-1 */
              "sub  %[wc],  %[wc], #384 \n"   /* wptr = wptr - 384 */
              "vmla.f32 q14,  q9,   q3  \n"   /* out02 = w23 * inr45 */
              "vmla.f32 q15,  q10,  q3  \n"   /* out03 = w23 * inr46 */
              "vld1.32  {d0-d3},    [%[wc]]!  \n" /* load w0-w1 */
              "vmla.f32 q12,  q8,   q2  \n"   /* out00 = w24 * inr44 */
              "vmla.f32 q13,  q9,   q2  \n"   /* out01 = w24 * inr45 */
              "vld1.32  {d12-d15},  [%[inr0]]!\n" /* load inr0, 2-3 */
              "vmla.f32 q14,  q10,  q2  \n"   /* out02 = w24 * inr46 */
              "vmla.f32 q15,  q11,  q2  \n"   /* out03 = w24 * inr47 */
              "vst1.32  {d24-d27},  [%[out0]]!\n" /* store out00, out01 */
              "vld1.32  {d4-d7},    [%[wc]]!  \n" /* load w2-w3 */
              "subs     %[cnt],   %[cnt], #1  \n" /* cnt = cnt - 1 */
              "vst1.32  {d28-d31},  [%[out0]]!\n" /* store out02, out03 */
              "vld1.32  {d24-d25},  [%[bias]] \n" /* load bias to out00 */
              "bne    1b\n"
              : [cnt] "+r"(cnt),
                [inr0] "+r"(inr0),
                [inr1] "+r"(inr1),
                [inr2] "+r"(inr2),
                [inr3] "+r"(inr3),
                [inr4] "+r"(inr4),
                [wc] "+r"(wptr),
                [out0] "+r"(ptr_out0)
              : [bias] "r"(bias_local)
              : "cc","memory",
                "q0", "q1", "q2", "q3", "q4", "q5",
                "q6", "q7", "q8", "q9", "q10", "q11",
                "q12", "q13", "q14", "q15"
              );
          // clang-format on
          block_inr0 = block_inr1;
          block_inr1 = block_inr2;
          block_inr2 = block_inr3;
          block_inr3 = block_inr4;
          block_inr4 = block_inr3 + in_len;
        }
        write_to_output_c4_fp32(pre_out,
                                dout_batch,
                                c,
                                c + hout_c_block,
                                h,
                                h + h_kernel,
                                0,
                                wout_round,
                                chout,
                                hout,
                                wout,
                                flag_relu,
                                ptr_write,
                                &act_param);
      }
    }
  }
}
#endif  // __aarch64__
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
