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
#include "lite/core/parallel_defines.h"
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
void conv_depthwise_5x5s1_fp32(float *dout,
                               const float *din,
                               const float *weights,
                               const float *bias,
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
                               const operators::ConvParam &param,
                               ARMContext *ctx) {
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

  float *tmp_work_space = ctx->workspace_data<float>();
  float ptr_zero[win_round];  // NOLINT
  memset(ptr_zero, 0, sizeof(float) * win_round);
  float ptr_write[wout_round];  // NOLINT

  int in_len = win_round * hout_c_block;
  int pre_in_size = hin_r_block * in_len;
  pre_in_size = ROUNDUP(pre_in_size, 4);
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  float *tmp_din = tmp_work_space;

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = 25;  // kernel_w * kernel_h;

  int ws = -padw;
  int we = ws + win_round;
  int w_loop = wout_round / 4;
  int chout = chin;

  int out_row_stride = hout_c_block * wout_round;
  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * chin * size_in_channel;
    float *dout_batch = dout + n * chout * size_out_channel;
    for (int h = 0; h < hout; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > hout) {
        h_kernel = hout - h;
      }
      int hs = h - padh;
      int he = hs + h_kernel + 4;
      LITE_PARALLEL_COMMON_BEGIN(c, tid, chout, 0, hout_c_block) {
#ifdef LITE_USE_THREAD_POOL
        float *pre_din = tmp_din + tid * (pre_in_size + pre_out_size);
        float *pre_out = pre_din + pre_in_size;
#elif defined(ARM_WITH_OMP)
        float *pre_din =
            tmp_din + omp_get_thread_num() * (pre_in_size + pre_out_size);
        float *pre_out = pre_din + pre_in_size;
#else
        float *pre_din = tmp_din;
        float *pre_out = pre_din + pre_in_size;
#endif
        prepack_input_nxwc4_dw(
            din_batch, pre_din, c, hs, he, ws, we, chin, win, hin, ptr_zero);
        const float *block_inr0 = pre_din;
        const float *block_inr1 = block_inr0 + in_len;
        const float *block_inr2 = block_inr1 + in_len;
        const float *block_inr3 = block_inr2 + in_len;
        const float *block_inr4 = block_inr3 + in_len;
        const float *block_inr5 = block_inr4 + in_len;

        const float *weight_c = weights + c * w_stride;
        float bias_local[4] = {0, 0, 0, 0};
        if (flag_bias) {
          if (c + hout_c_block < chout) {
            bias_local[0] = bias[c];
            bias_local[1] = bias[c + 1];
            bias_local[2] = bias[c + 2];
            bias_local[3] = bias[c + 3];
          } else {
            for (int k = 0; k < 4 && k + c < chout; k++) {
              bias_local[k] = bias[c + k];
            }
          }
        }
        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          int cnt = w_loop;
          const float *inr0 = block_inr0;
          const float *inr1 = block_inr1;
          const float *inr2 = block_inr2;
          const float *inr3 = block_inr3;
          const float *inr4 = block_inr4;
          const float *inr5 = block_inr5;

          float *ptr_out0 = pre_out + hk * out_row_stride;
          float *ptr_out1 = ptr_out0 + out_row_stride;
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
      LITE_PARALLEL_END();
    }
  }
}
#else  // __aarch64__
void conv_depthwise_5x5s1_fp32(float *dout,
                               const float *din,
                               const float *weights,
                               const float *bias,
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
                               const operators::ConvParam &param,
                               ARMContext *ctx) {
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

  float *tmp_work_space = ctx->workspace_data<float>();
  float ptr_zero[win_round];  // NOLINT
  memset(ptr_zero, 0, sizeof(float) * win_round);
  float ptr_write[wout_round];  // NOLINT

  int in_len = win_round * hout_c_block;
  int pre_in_size = hin_r_block * in_len;
  pre_in_size = ROUNDUP(pre_in_size, 4);
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  float *tmp_din = tmp_work_space;

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = 25;  // kernel_w * kernel_h;

  int ws = -padw;
  int we = ws + win_round;
  int w_loop = wout_round / 4;
  int chout = chin;

  int out_row_stride = hout_c_block * wout_round;
  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * chin * size_in_channel;
    float *dout_batch = dout + n * chout * size_out_channel;
    for (int h = 0; h < hout; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > hout) {
        h_kernel = hout - h;
      }
      int hs = h - padh;
      int he = hs + h_kernel + 4;

      LITE_PARALLEL_COMMON_BEGIN(c, tid, chout, 0, hout_c_block) {
#ifdef LITE_USE_THREAD_POOL
        float *pre_din = tmp_din + tid * (pre_in_size + pre_out_size);
        float *pre_out = pre_din + pre_in_size;
#elif defined(ARM_WITH_OMP)
        float *pre_din =
            tmp_din + omp_get_thread_num() * (pre_in_size + pre_out_size);
        float *pre_out = pre_din + pre_in_size;
#else
        float *pre_din = tmp_din;
        float *pre_out = pre_din + pre_in_size;
#endif
        prepack_input_nxwc4_dw(
            din_batch, pre_din, c, hs, he, ws, we, chin, win, hin, ptr_zero);
        const float *block_inr0 = pre_din;
        const float *block_inr1 = block_inr0 + in_len;
        const float *block_inr2 = block_inr1 + in_len;
        const float *block_inr3 = block_inr2 + in_len;
        const float *block_inr4 = block_inr3 + in_len;

        const float *weight_c = weights + c * w_stride;
        float bias_local[4] = {0, 0, 0, 0};
        if (flag_bias) {
          if (c + hout_c_block < chout) {
            bias_local[0] = bias[c];
            bias_local[1] = bias[c + 1];
            bias_local[2] = bias[c + 2];
            bias_local[3] = bias[c + 3];
          } else {
            for (int k = 0; k < 4 && k + c < chout; k++) {
              bias_local[k] = bias[c + k];
            }
          }
        }
        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          int cnt = w_loop;
          const float *inr0 = block_inr0;
          const float *inr1 = block_inr1;
          const float *inr2 = block_inr2;
          const float *inr3 = block_inr3;
          const float *inr4 = block_inr4;

          float *ptr_out0 = pre_out + hk * out_row_stride;
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
      LITE_PARALLEL_END();
    }
  }
}
#endif  // __aarch64__
#define ACTUAL_PARAM                                                         \
  dout, din, weights, bias, flag_bias, flag_relu, num, chin, hin, win, hout, \
      wout

#define IN_PARAM                                                           \
  float *dout, const float *din, const float *weights, const float *bias,  \
      bool flag_bias, bool flag_relu, int num, int chin, int hin, int win, \
      int hout, int wout
// clang-format off
#ifdef __aarch64__
inline std::pair<uint32_t, uint32_t> right_mask_5x5s1p2_fp32(int win,
                                                             int wout,
                                                             uint32_t* vmask) {
  uint32_t right_pad_idx[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  uint32_t cnt_col = ((wout >> 3) - 2);
  uint32_t size_right_remain = static_cast<uint32_t>(win - (6 + cnt_col * 8));
  if (size_right_remain >= 12) {
    cnt_col++;
    size_right_remain -= 8;
  }
  uint32_t cnt_remain = (size_right_remain >= 9 && wout % 8 == 0)
                            ? 8
                            : static_cast<uint32_t>(wout % 8);
  size_right_remain = (cnt_remain == 8) ? size_right_remain :
                      (size_right_remain + 8 - cnt_remain);
  uint32x4_t vmask_rp0 =
      vcgtq_u32(vdupq_n_u32(size_right_remain), vld1q_u32(right_pad_idx));
  uint32x4_t vmask_rp1 =
      vcgtq_u32(vdupq_n_u32(size_right_remain), vld1q_u32(right_pad_idx + 4));
  uint32x4_t vmask_rp2 =
      vcgtq_u32(vdupq_n_u32(size_right_remain), vld1q_u32(right_pad_idx + 8));
  vst1q_u32(vmask, vmask_rp0);
  vst1q_u32(vmask + 4, vmask_rp1);
  vst1q_u32(vmask + 8, vmask_rp2);
  return std::make_pair(cnt_col, cnt_remain);
}
#define DIN_PTR_INIT               \
  const float *din_ptr0 = dr0;     \
  const float *din_ptr1 = dr1;     \
  const float *din_ptr2 = dr2;     \
  const float *din_ptr3 = dr3;     \
  const float *din_ptr4 = dr4;     \
  const float *din_ptr5 = dr5;     \
  /* h - 2 + 6 = h - 4 */          \
  if (h + 4 > hin) {               \
    switch (h + 4 - hin) {         \
      case 5:                      \
        din_ptr1 = zero_ptr;       \
      case 4:                      \
        din_ptr2 = zero_ptr;       \
      case 3:                      \
        din_ptr3 = zero_ptr;       \
      case 2:                      \
        din_ptr4 = zero_ptr;       \
      case 1:                      \
        din_ptr5 = zero_ptr;       \
      default:                     \
        break;                     \
    }                              \
  }                                \
  float *doutr0 = dout_ptr;        \
  float *doutr1 = dout_ptr + wout; \
  if (h + 2 > hout) {              \
    doutr1 = write_ptr;            \
  }                                \
  /* update in_address */          \
  dr0 = dr2;                       \
  dr1 = dr3;                       \
  dr2 = dr4;                       \
  dr3 = dr5;                       \
  dr4 = dr3 + win;                 \
  dr5 = dr4 + win;
#define LEFT_COMPUTE_S1                             \
  "PRFM PLDL1KEEP, [%[din_ptr0]]\n"                 \
  "PRFM PLDL1KEEP, [%[din_ptr1]]\n"                 \
  "PRFM PLDL1KEEP, [%[din_ptr2]]\n"                 \
  "PRFM PLDL1KEEP, [%[din_ptr3]]\n"                 \
  "PRFM PLDL1KEEP, [%[din_ptr4]]\n"                 \
  "PRFM PLDL1KEEP, [%[din_ptr5]]\n"                 \
  "ld1 {v0.4s, v1.4s}, [%[din_ptr0]], #32\n"        \
  "ld1 {v3.4s, v4.4s}, [%[din_ptr1]], #32\n"        \
  "ld1 {v28.4s}, [%[bias_val]]\n"                   \
  "ld1 {v29.4s}, [%[bias_val]]\n"                   \
  "ld1 {v30.4s}, [%[bias_val]]\n"                   \
  "ld1 {v31.4s}, [%[bias_val]]\n"                   \
  "ld1 {v2.4s}, [%[din_ptr0]]\n"                    \
  "ld1 {v5.4s}, [%[din_ptr1]]\n"                    \
  /* line 0 */                                      \
  "ext  v6.16b,  %[vzero].16b, v0.16b, #8\n"        \
  "ext  v9.16b,  v0.16b,  v1.16b, #8\n"             \
  "ext  v13.16b, %[vzero].16b, v3.16b, #8\n"        \
  "ext  v16.16b, v3.16b,  v4.16b, #8\n"             \
  "ext  v7.16b,  %[vzero].16b, v0.16b, #12\n"       \
  "ext  v10.16b,  v0.16b,  v1.16b, #12\n"           \
  "ext  v14.16b, %[vzero].16b, v3.16b, #12\n"       \
  "ext  v17.16b, v3.16b,  v4.16b, #12\n"            \
  "fmla v28.4s, v6.4s,  %[w0].s[0]\n"               \
  "fmla v29.4s, v9.4s,  %[w0].s[0]\n"               \
  "fmla v30.4s, v13.4s, %[w0].s[0]\n"               \
  "fmla v31.4s, v16.4s, %[w0].s[0]\n"               \
  "ext  v8.16b,  v0.16b,  v1.16b, #4\n"             \
  "ext  v11.16b, v1.16b,  v2.16b, #4\n"             \
  "ext  v15.16b, v3.16b,  v4.16b, #4\n"             \
  "ext  v18.16b, v4.16b,  v5.16b, #4\n"             \
  "fmla v28.4s, v7.4s,  %[w0].s[1]\n"               \
  "fmla v29.4s, v10.4s, %[w0].s[1]\n"               \
  "fmla v30.4s, v14.4s, %[w0].s[1]\n"               \
  "fmla v31.4s, v17.4s, %[w0].s[1]\n"               \
  "ext  v12.16b, v1.16b,  v2.16b, #8\n"             \
  "ext  v19.16b, v4.16b,  v5.16b, #8\n"             \
  "fmla v28.4s, v0.4s,  %[w0].s[2]\n"               \
  "fmla v29.4s, v1.4s,  %[w0].s[2]\n"               \
  "fmla v30.4s, v3.4s,  %[w0].s[2]\n"               \
  "fmla v31.4s, v4.4s,  %[w0].s[2]\n"               \
  "ld1 {v0.4s, v1.4s}, [%[din_ptr2]], #32\n"        \
  "fmla v28.4s, v8.4s,  %[w0].s[3]\n"               \
  "fmla v29.4s, v11.4s, %[w0].s[3]\n"               \
  "fmla v30.4s, v15.4s, %[w0].s[3]\n"               \
  "fmla v31.4s, v18.4s, %[w0].s[3]\n"               \
  "ld1 {v2.4s}, [%[din_ptr2]]\n"                    \
  "fmla v28.4s, v9.4s,  %[w1].s[0]\n"               \
  "fmla v29.4s, v12.4s, %[w1].s[0]\n"               \
  "fmla v30.4s, v16.4s, %[w1].s[0]\n"               \
  "fmla v31.4s, v19.4s, %[w1].s[0]\n"               \
  "ext  v6.16b,  %[vzero].16b, v0.16b, #8\n"        \
  "ext  v9.16b,  v0.16b,  v1.16b, #8\n"             \
  "ext  v7.16b,  %[vzero].16b, v0.16b, #12\n"       \
  "ext  v10.16b, v0.16b,  v1.16b, #12\n"            \
  "ext  v8.16b,  v0.16b,  v1.16b, #4\n"             \
  /* line 1 */                                      \
  "fmla v28.4s, v13.4s, %[w1].s[1]\n"               \
  "fmla v29.4s, v16.4s, %[w1].s[1]\n"               \
  "fmla v30.4s, v6.4s,  %[w1].s[1]\n"               \
  "fmla v31.4s, v9.4s,  %[w1].s[1]\n"               \
  "ext  v11.16b, v1.16b,  v2.16b, #4\n"             \
  "ext  v12.16b, v1.16b,  v2.16b, #8\n"             \
  "fmla v28.4s, v14.4s, %[w1].s[2]\n"               \
  "fmla v29.4s, v17.4s, %[w1].s[2]\n"               \
  "fmla v30.4s, v7.4s,  %[w1].s[2]\n"               \
  "fmla v31.4s, v10.4s, %[w1].s[2]\n"               \
  "fmla v28.4s, v3.4s,  %[w1].s[3]\n"               \
  "fmla v29.4s, v4.4s,  %[w1].s[3]\n"               \
  "fmla v30.4s, v0.4s,  %[w1].s[3]\n"               \
  "fmla v31.4s, v1.4s,  %[w1].s[3]\n"               \
  "ld1 {v3.4s, v4.4s}, [%[din_ptr3]], #32\n"        \
  "fmla v28.4s, v15.4s, %[w2].s[0]\n"               \
  "fmla v29.4s, v18.4s, %[w2].s[0]\n"               \
  "fmla v30.4s, v8.4s,  %[w2].s[0]\n"               \
  "fmla v31.4s, v11.4s, %[w2].s[0]\n"               \
  "ld1 {v5.4s}, [%[din_ptr3]]\n"                    \
  "fmla v28.4s, v16.4s, %[w2].s[1]\n"               \
  "fmla v29.4s, v19.4s, %[w2].s[1]\n"               \
  "fmla v30.4s, v9.4s,  %[w2].s[1]\n"               \
  "fmla v31.4s, v12.4s, %[w2].s[1]\n"               \
  "ext  v13.16b, %[vzero].16b, v3.16b, #8\n"        \
  "ext  v16.16b, v3.16b,  v4.16b, #8\n"             \
  "ext  v14.16b, %[vzero].16b, v3.16b, #12\n"       \
  "ext  v17.16b, v3.16b,  v4.16b, #12\n"            \
  "ext  v15.16b, v3.16b,  v4.16b, #4\n"             \
  /* line 2 */                                      \
  "fmla v28.4s, v6.4s,  %[w2].s[2]\n"               \
  "fmla v29.4s, v9.4s,  %[w2].s[2]\n"               \
  "fmla v30.4s, v13.4s, %[w2].s[2]\n"               \
  "fmla v31.4s, v16.4s, %[w2].s[2]\n"               \
  "ext  v18.16b, v4.16b,  v5.16b, #4\n"             \
  "ext  v19.16b, v4.16b,  v5.16b, #8\n"             \
  "fmla v28.4s, v7.4s,  %[w2].s[3]\n"               \
  "fmla v29.4s, v10.4s, %[w2].s[3]\n"               \
  "fmla v30.4s, v14.4s, %[w2].s[3]\n"               \
  "fmla v31.4s, v17.4s, %[w2].s[3]\n"               \
  "fmla v28.4s, v0.4s,  %[w3].s[0]\n"               \
  "fmla v29.4s, v1.4s,  %[w3].s[0]\n"               \
  "fmla v30.4s, v3.4s,  %[w3].s[0]\n"               \
  "fmla v31.4s, v4.4s,  %[w3].s[0]\n"               \
  "ld1 {v0.4s, v1.4s}, [%[din_ptr4]], #32\n"        \
  "fmla v28.4s, v8.4s,  %[w3].s[1]\n"               \
  "fmla v29.4s, v11.4s, %[w3].s[1]\n"               \
  "fmla v30.4s, v15.4s, %[w3].s[1]\n"               \
  "fmla v31.4s, v18.4s, %[w3].s[1]\n"               \
  "ld1 {v2.4s}, [%[din_ptr4]]\n"                    \
  "fmla v28.4s, v9.4s,  %[w3].s[2]\n"               \
  "fmla v29.4s, v12.4s, %[w3].s[2]\n"               \
  "fmla v30.4s, v16.4s, %[w3].s[2]\n"               \
  "fmla v31.4s, v19.4s, %[w3].s[2]\n"               \
  "ext  v6.16b,  %[vzero].16b, v0.16b, #8\n"        \
  "ext  v9.16b,  v0.16b,  v1.16b, #8\n"             \
  "ext  v7.16b,  %[vzero].16b, v0.16b, #12\n"       \
  "ext  v10.16b, v0.16b,  v1.16b, #12\n"            \
  "ext  v8.16b,  v0.16b,  v1.16b, #4\n"             \
  /* line 3 */                                      \
  "fmla v28.4s, v13.4s, %[w3].s[3]\n"               \
  "fmla v29.4s, v16.4s, %[w3].s[3]\n"               \
  "fmla v30.4s, v6.4s,  %[w3].s[3]\n"               \
  "fmla v31.4s, v9.4s,  %[w3].s[3]\n"               \
  "ext  v11.16b, v1.16b,  v2.16b, #4\n"             \
  "ext  v12.16b, v1.16b,  v2.16b, #8\n"             \
  "fmla v28.4s, v14.4s, %[w4].s[0]\n"               \
  "fmla v29.4s, v17.4s, %[w4].s[0]\n"               \
  "fmla v30.4s, v7.4s,  %[w4].s[0]\n"               \
  "fmla v31.4s, v10.4s, %[w4].s[0]\n"               \
  "fmla v28.4s, v3.4s,  %[w4].s[1]\n"               \
  "fmla v29.4s, v4.4s,  %[w4].s[1]\n"               \
  "fmla v30.4s, v0.4s,  %[w4].s[1]\n"               \
  "fmla v31.4s, v1.4s,  %[w4].s[1]\n"               \
  "ld1 {v3.4s, v4.4s}, [%[din_ptr5]], #32\n"        \
  "fmla v28.4s, v15.4s, %[w4].s[2]\n"               \
  "fmla v29.4s, v18.4s, %[w4].s[2]\n"               \
  "fmla v30.4s, v8.4s,  %[w4].s[2]\n"               \
  "fmla v31.4s, v11.4s, %[w4].s[2]\n"               \
  "ld1 {v5.4s}, [%[din_ptr5]]\n"                    \
  "fmla v28.4s, v16.4s, %[w4].s[3]\n"               \
  "fmla v29.4s, v19.4s, %[w4].s[3]\n"               \
  "fmla v30.4s, v9.4s,  %[w4].s[3]\n"               \
  "fmla v31.4s, v12.4s, %[w4].s[3]\n"               \
  "ext  v13.16b, %[vzero].16b, v3.16b, #8\n"        \
  "ext  v16.16b, v3.16b,  v4.16b, #8\n"             \
  "ext  v14.16b, %[vzero].16b, v3.16b, #12\n"       \
  "ext  v17.16b, v3.16b,  v4.16b, #12\n"            \
  "ext  v15.16b, v3.16b,  v4.16b, #4\n"             \
  /* line 4 */                                      \
  "fmla v28.4s, v6.4s,  %[w5].s[0]\n"               \
  "fmla v29.4s, v9.4s,  %[w5].s[0]\n"               \
  "fmla v30.4s, v13.4s, %[w5].s[0]\n"               \
  "fmla v31.4s, v16.4s, %[w5].s[0]\n"               \
  "ext  v18.16b, v4.16b,  v5.16b, #4\n"             \
  "ext  v19.16b, v4.16b,  v5.16b, #8\n"             \
  "fmla v28.4s, v7.4s,  %[w5].s[1]\n"               \
  "fmla v29.4s, v10.4s, %[w5].s[1]\n"               \
  "fmla v30.4s, v14.4s, %[w5].s[1]\n"               \
  "fmla v31.4s, v17.4s, %[w5].s[1]\n"               \
  "fmla v28.4s, v0.4s,  %[w5].s[2]\n"               \
  "fmla v29.4s, v1.4s,  %[w5].s[2]\n"               \
  "fmla v30.4s, v3.4s,  %[w5].s[2]\n"               \
  "fmla v31.4s, v4.4s,  %[w5].s[2]\n"               \
  "sub %[din_ptr0], %[din_ptr0], #8\n"              \
  "sub %[din_ptr1], %[din_ptr1], #8\n"              \
  "fmla v28.4s, v8.4s,  %[w5].s[3]\n"               \
  "fmla v29.4s, v11.4s, %[w5].s[3]\n"               \
  "fmla v30.4s, v15.4s, %[w5].s[3]\n"               \
  "fmla v31.4s, v18.4s, %[w5].s[3]\n"               \
  "sub %[din_ptr2], %[din_ptr2], #8\n"              \
  "sub %[din_ptr3], %[din_ptr3], #8\n"              \
  "fmla v28.4s, v9.4s,  %[w6].s[0]\n"               \
  "fmla v29.4s, v12.4s, %[w6].s[0]\n"               \
  "fmla v30.4s, v16.4s, %[w6].s[0]\n"               \
  "fmla v31.4s, v19.4s, %[w6].s[0]\n"               \
  "sub %[din_ptr4], %[din_ptr4], #8\n"              \
  "sub %[din_ptr5], %[din_ptr5], #8\n"

#define LEFT_RESULT_S1                        \
  "cmp %w[cnt], #16                       \n" \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"  \
  "st1 {v30.4s, v31.4s}, [%[doutr1]], #32\n"  \
  "blt 2f                         \n"

#define LEFT_RESULT_S1_RELU            \
  "fmax v28.4s, v28.4s, %[vzero].4s\n" \
  "fmax v29.4s, v29.4s, %[vzero].4s\n" \
  "fmax v30.4s, v30.4s, %[vzero].4s\n" \
  "fmax v31.4s, v31.4s, %[vzero].4s\n" \
  "cmp %w[cnt], #16               \n"  \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"\
  "st1 {v30.4s, v31.4s}, [%[doutr1]], #32\n"\
  "blt 2f                         \n"

#define LEFT_RESULT_S1_RELU6          \
  "ld1 {v1.4s}, [%[six_ptr]]   \n"    \
  "fmax v28.4s, v28.4s, %[vzero].4s\n"\
  "fmax v29.4s, v29.4s, %[vzero].4s\n"\
  "fmax v30.4s, v30.4s, %[vzero].4s\n"\
  "fmax v31.4s, v31.4s, %[vzero].4s\n"\
  "cmp %w[cnt], #16              \n"  \
  "fmin v28.4s, v28.4s, v1.4s\n"      \
  "fmin v29.4s, v29.4s, v1.4s\n"      \
  "fmin v30.4s, v30.4s, v1.4s\n"      \
  "fmin v31.4s, v31.4s, v1.4s\n"      \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"\
  "st1 {v30.4s, v31.4s}, [%[doutr1]], #32\n"\
  "blt 2f                         \n"

#define MID_COMPITE_S1                              \
  "1:                        \n"                    \
  "ld1 {v0.4s, v1.4s}, [%[din_ptr0]], #32\n"        \
  "ld1 {v3.4s, v4.4s}, [%[din_ptr1]], #32\n"        \
  "ld1 {v28.4s}, [%[bias_val]]\n"                   \
  "ld1 {v29.4s}, [%[bias_val]]\n"                   \
  "ld1 {v30.4s}, [%[bias_val]]\n"                   \
  "ld1 {v31.4s}, [%[bias_val]]\n"                   \
  "ld1 {v2.4s}, [%[din_ptr0]]\n"                    \
  "ld1 {v5.4s}, [%[din_ptr1]]\n"                    \
  /* line 0 */                                      \
  "ext v6.16b,  v0.16b, v1.16b, #4\n"               \
  "ext v9.16b,  v1.16b, v2.16b, #4\n"               \
  "ext v12.16b, v3.16b, v4.16b, #4\n"               \
  "ext v15.16b, v4.16b, v5.16b, #4\n"               \
  "fmla v28.4s, v0.4s,  %[w0].s[0]\n"               \
  "fmla v29.4s, v1.4s,  %[w0].s[0]\n"               \
  "fmla v30.4s, v3.4s,  %[w0].s[0]\n"               \
  "fmla v31.4s, v4.4s,  %[w0].s[0]\n"               \
  "ext v7.16b,  v0.16b, v1.16b, #8\n"               \
  "ext v10.16b, v1.16b, v2.16b, #8\n"               \
  "ext v13.16b, v3.16b, v4.16b, #8\n"               \
  "ext v16.16b, v4.16b, v5.16b, #8\n"               \
  "fmla v28.4s, v6.4s,  %[w0].s[1]\n"               \
  "fmla v29.4s, v9.4s,  %[w0].s[1]\n"               \
  "fmla v30.4s, v12.4s, %[w0].s[1]\n"               \
  "fmla v31.4s, v15.4s, %[w0].s[1]\n"               \
  "ext v8.16b,  v0.16b, v1.16b, #12\n"              \
  "ext v11.16b, v1.16b, v2.16b, #12\n"              \
  "ext v14.16b, v3.16b, v4.16b, #12\n"              \
  "ext v17.16b, v4.16b, v5.16b, #12\n"              \
  "fmla v28.4s, v1.4s,  %[w1].s[0]\n"               \
  "fmla v29.4s, v2.4s,  %[w1].s[0]\n"               \
  "fmla v30.4s, v4.4s,  %[w1].s[0]\n"               \
  "fmla v31.4s, v5.4s,  %[w1].s[0]\n"               \
  "ld1 {v0.4s, v1.4s}, [%[din_ptr2]], #32\n"        \
  "fmla v28.4s, v7.4s,  %[w0].s[2]\n"               \
  "fmla v29.4s, v10.4s, %[w0].s[2]\n"               \
  "fmla v30.4s, v13.4s, %[w0].s[2]\n"               \
  "fmla v31.4s, v16.4s, %[w0].s[2]\n"               \
  "ld1 {v2.4s}, [%[din_ptr2]]\n"                    \
  "fmla v28.4s, v8.4s,  %[w0].s[3]\n"               \
  "fmla v29.4s, v11.4s, %[w0].s[3]\n"               \
  "fmla v30.4s, v14.4s, %[w0].s[3]\n"               \
  "fmla v31.4s, v17.4s, %[w0].s[3]\n"               \
  "ext v6.16b,  v0.16b, v1.16b, #4\n"               \
  "ext v9.16b,  v1.16b, v2.16b, #4\n"               \
  "ext v7.16b,  v0.16b, v1.16b, #8\n"               \
  "ext v10.16b, v1.16b, v2.16b, #8\n"               \
  /* line 1 */                                      \
  "fmla v28.4s, v3.4s,  %[w1].s[1]\n"               \
  "fmla v29.4s, v4.4s,  %[w1].s[1]\n"               \
  "fmla v30.4s, v0.4s,  %[w1].s[1]\n"               \
  "fmla v31.4s, v1.4s,  %[w1].s[1]\n"               \
  "ext v8.16b,  v0.16b, v1.16b, #12\n"              \
  "ext v11.16b, v1.16b, v2.16b, #12\n"              \
  "fmla v28.4s, v4.4s,  %[w2].s[1]\n"               \
  "fmla v29.4s, v5.4s,  %[w2].s[1]\n"               \
  "fmla v30.4s, v1.4s,  %[w2].s[1]\n"               \
  "fmla v31.4s, v2.4s,  %[w2].s[1]\n"               \
  "ld1 {v3.4s, v4.4s}, [%[din_ptr3]], #32\n"        \
  "fmla v28.4s, v12.4s, %[w1].s[2]\n"               \
  "fmla v29.4s, v15.4s, %[w1].s[2]\n"               \
  "fmla v30.4s, v6.4s,  %[w1].s[2]\n"               \
  "fmla v31.4s, v9.4s,  %[w1].s[2]\n"               \
  "ld1 {v5.4s}, [%[din_ptr3]]\n"                    \
  "fmla v28.4s, v13.4s, %[w1].s[3]\n"               \
  "fmla v29.4s, v16.4s, %[w1].s[3]\n"               \
  "fmla v30.4s, v7.4s,  %[w1].s[3]\n"               \
  "fmla v31.4s, v10.4s, %[w1].s[3]\n"               \
  "ext v12.16b, v3.16b, v4.16b, #4\n"               \
  "ext v15.16b, v4.16b, v5.16b, #4\n"               \
  "fmla v28.4s, v14.4s, %[w2].s[0]\n"               \
  "fmla v29.4s, v17.4s, %[w2].s[0]\n"               \
  "fmla v30.4s, v8.4s,  %[w2].s[0]\n"               \
  "fmla v31.4s, v11.4s, %[w2].s[0]\n"               \
  "ext v13.16b, v3.16b, v4.16b, #8\n"               \
  "ext v16.16b, v4.16b, v5.16b, #8\n"               \
  "ext v14.16b, v3.16b, v4.16b, #12\n"              \
  "ext v17.16b, v4.16b, v5.16b, #12\n"              \
  /* line 2 */                                      \
  "fmla v28.4s, v0.4s,  %[w2].s[2]\n"               \
  "fmla v29.4s, v1.4s,  %[w2].s[2]\n"               \
  "fmla v30.4s, v3.4s,  %[w2].s[2]\n"               \
  "fmla v31.4s, v4.4s,  %[w2].s[2]\n"               \
  "fmla v28.4s, v1.4s,  %[w3].s[2]\n"               \
  "fmla v29.4s, v2.4s,  %[w3].s[2]\n"               \
  "fmla v30.4s, v4.4s,  %[w3].s[2]\n"               \
  "fmla v31.4s, v5.4s,  %[w3].s[2]\n"               \
  "ld1 {v0.4s, v1.4s}, [%[din_ptr4]], #32\n"        \
  "fmla v28.4s, v6.4s,  %[w2].s[3]\n"               \
  "fmla v29.4s, v9.4s,  %[w2].s[3]\n"               \
  "fmla v30.4s, v12.4s, %[w2].s[3]\n"               \
  "fmla v31.4s, v15.4s, %[w2].s[3]\n"               \
  "ld1 {v2.4s}, [%[din_ptr4]]\n"                    \
  "fmla v28.4s, v7.4s,  %[w3].s[0]\n"               \
  "fmla v29.4s, v10.4s, %[w3].s[0]\n"               \
  "fmla v30.4s, v13.4s, %[w3].s[0]\n"               \
  "fmla v31.4s, v16.4s, %[w3].s[0]\n"               \
  "ext v6.16b,  v0.16b, v1.16b, #4\n"               \
  "ext v9.16b,  v1.16b, v2.16b, #4\n"               \
  "fmla v28.4s, v8.4s,  %[w3].s[1]\n"               \
  "fmla v29.4s, v11.4s, %[w3].s[1]\n"               \
  "fmla v30.4s, v14.4s, %[w3].s[1]\n"               \
  "fmla v31.4s, v17.4s, %[w3].s[1]\n"               \
  "ext v7.16b,  v0.16b, v1.16b, #8\n"               \
  "ext v10.16b, v1.16b, v2.16b, #8\n"               \
  /* line 3 */                                      \
  "fmla v28.4s, v3.4s,  %[w3].s[3]\n"               \
  "fmla v29.4s, v4.4s,  %[w3].s[3]\n"               \
  "fmla v30.4s, v0.4s,  %[w3].s[3]\n"               \
  "fmla v31.4s, v1.4s,  %[w3].s[3]\n"               \
  "ext v8.16b,  v0.16b, v1.16b, #12\n"              \
  "ext v11.16b, v1.16b, v2.16b, #12\n"              \
  "fmla v28.4s, v4.4s,  %[w4].s[3]\n"               \
  "fmla v29.4s, v5.4s,  %[w4].s[3]\n"               \
  "fmla v30.4s, v1.4s,  %[w4].s[3]\n"               \
  "fmla v31.4s, v2.4s,  %[w4].s[3]\n"               \
  "ld1 {v3.4s, v4.4s}, [%[din_ptr5]], #32\n"        \
  "fmla v28.4s, v12.4s, %[w4].s[0]\n"               \
  "fmla v29.4s, v15.4s, %[w4].s[0]\n"               \
  "fmla v30.4s, v6.4s,  %[w4].s[0]\n"               \
  "fmla v31.4s, v9.4s,  %[w4].s[0]\n"               \
  "ld1 {v5.4s}, [%[din_ptr5]]\n"                    \
  "fmla v28.4s, v13.4s, %[w4].s[1]\n"               \
  "fmla v29.4s, v16.4s, %[w4].s[1]\n"               \
  "fmla v30.4s, v7.4s,  %[w4].s[1]\n"               \
  "fmla v31.4s, v10.4s, %[w4].s[1]\n"               \
  "ext v12.16b, v3.16b, v4.16b, #4\n"               \
  "ext v15.16b, v4.16b, v5.16b, #4\n"               \
  "fmla v28.4s, v14.4s, %[w4].s[2]\n"               \
  "fmla v29.4s, v17.4s, %[w4].s[2]\n"               \
  "fmla v30.4s, v8.4s,  %[w4].s[2]\n"               \
  "fmla v31.4s, v11.4s, %[w4].s[2]\n"               \
  "ext v13.16b, v3.16b, v4.16b, #8\n"               \
  "ext v16.16b, v4.16b, v5.16b, #8\n"               \
  "ext v14.16b, v3.16b, v4.16b, #12\n"              \
  "ext v17.16b, v4.16b, v5.16b, #12\n"              \
  /* line 4 */                                      \
  "fmla v28.4s, v0.4s,  %[w5].s[0]\n"               \
  "fmla v29.4s, v1.4s,  %[w5].s[0]\n"               \
  "fmla v30.4s, v3.4s,  %[w5].s[0]\n"               \
  "fmla v31.4s, v4.4s,  %[w5].s[0]\n"               \
  "fmla v28.4s, v1.4s,  %[w6].s[0]\n"               \
  "fmla v29.4s, v2.4s,  %[w6].s[0]\n"               \
  "fmla v30.4s, v4.4s,  %[w6].s[0]\n"               \
  "fmla v31.4s, v5.4s,  %[w6].s[0]\n"               \
  "fmla v28.4s, v6.4s,  %[w5].s[1]\n"               \
  "fmla v29.4s, v9.4s,  %[w5].s[1]\n"               \
  "fmla v30.4s, v12.4s, %[w5].s[1]\n"               \
  "fmla v31.4s, v15.4s, %[w5].s[1]\n"               \
  "fmla v28.4s, v7.4s,  %[w5].s[2]\n"               \
  "fmla v29.4s, v10.4s, %[w5].s[2]\n"               \
  "fmla v30.4s, v13.4s, %[w5].s[2]\n"               \
  "fmla v31.4s, v16.4s, %[w5].s[2]\n"               \
  "subs %w[cnt], %w[cnt], #16\n"                    \
  "fmla v28.4s, v8.4s,  %[w5].s[3]\n"               \
  "fmla v29.4s, v11.4s, %[w5].s[3]\n"               \
  "fmla v30.4s, v14.4s, %[w5].s[3]\n"               \
  "fmla v31.4s, v17.4s, %[w5].s[3]\n"

#define MID_RESULT_S1                         \
  "cmp %w[cnt], #16\n"                        \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"  \
  "st1 {v30.4s, v31.4s}, [%[doutr1]], #32\n"  \
  "bge 1b                         \n"

#define MID_RESULT_S1_RELU             \
  "cmp %w[cnt], #16\n"                 \
  "fmax  v28.4s, v28.4s, %[vzero].4s\n"\
  "fmax  v29.4s, v29.4s, %[vzero].4s\n"\
  "fmax  v30.4s, v30.4s, %[vzero].4s\n"\
  "fmax  v31.4s, v31.4s, %[vzero].4s\n"\
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"\
  "st1 {v30.4s, v31.4s}, [%[doutr1]], #32\n"\
  "bge 1b                         \n"

#define MID_RESULT_S1_RELU6           \
  "ld1 {v1.4s}, [%[six_ptr]]   \n"    \
  "fmax  v28.4s, v28.4s, %[vzero].4s\n"\
  "fmax  v29.4s, v29.4s, %[vzero].4s\n"\
  "fmax  v30.4s, v30.4s, %[vzero].4s\n"\
  "fmax  v31.4s, v31.4s, %[vzero].4s\n"\
  "cmp %w[cnt], #16\n"                 \
  "fmin  v28.4s, v28.4s, v1.4s\n"      \
  "fmin  v29.4s, v29.4s, v1.4s\n"      \
  "fmin  v30.4s, v30.4s, v1.4s\n"      \
  "fmin  v31.4s, v31.4s, v1.4s\n"      \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"\
  "st1 {v30.4s, v31.4s}, [%[doutr1]], #32\n"\
  "bge 1b                         \n"

#define RIGHT_COMPUTE_S1                            \
  "2:                             \n"               \
  "cmp %w[cnt], #1                \n"               \
  "sub %[din_ptr0], %[din_ptr0], %[right_pad_num]\n"\
  "sub %[din_ptr1], %[din_ptr1], %[right_pad_num]\n"\
  "sub %[din_ptr2], %[din_ptr2], %[right_pad_num]\n"\
  "sub %[din_ptr3], %[din_ptr3], %[right_pad_num]\n"\
  "sub %[din_ptr4], %[din_ptr4], %[right_pad_num]\n"\
  "sub %[din_ptr5], %[din_ptr5], %[right_pad_num]\n"\
  "blt 3f                         \n"               \
  "ld1 {v0.4s, v1.4s, v2.4s}, [%[din_ptr0]]\n"      \
  "ld1 {v3.4s, v4.4s, v5.4s}, [%[din_ptr1]]\n"      \
  "sub %[doutr0], %[doutr0], %[right_pad_num]\n"    \
  "sub %[doutr1], %[doutr1], %[right_pad_num]\n"    \
  "ld1 {v17.4s, v18.4s, v19.4s}, [%[vmask]]\n"      \
  "ld1 {v28.4s}, [%[bias_val]]\n"                   \
  "ld1 {v29.4s}, [%[bias_val]]\n"                   \
  "ld1 {v30.4s}, [%[bias_val]]\n"                   \
  "ld1 {v31.4s}, [%[bias_val]]\n"                   \
  "bif v0.16b, %[vzero].16b, v17.16b\n"             \
  "bif v1.16b, %[vzero].16b, v18.16b\n"             \
  "bif v2.16b, %[vzero].16b, v19.16b\n"             \
  "bif v3.16b, %[vzero].16b, v17.16b\n"             \
  "bif v4.16b, %[vzero].16b, v18.16b\n"             \
  "bif v5.16b, %[vzero].16b, v19.16b\n"             \
  /* line 0 */                                      \
  "ext v6.16b,  v0.16b, v1.16b, #4\n"               \
  "ext v9.16b,  v1.16b, v2.16b, #4\n"               \
  "ext v12.16b, v3.16b, v4.16b, #4\n"               \
  "ext v15.16b, v4.16b, v5.16b, #4\n"               \
  "fmla v28.4s, v0.4s,  %[w0].s[0]\n"               \
  "fmla v29.4s, v1.4s,  %[w0].s[0]\n"               \
  "fmla v30.4s, v3.4s,  %[w0].s[0]\n"               \
  "fmla v31.4s, v4.4s,  %[w0].s[0]\n"               \
  "ext v7.16b,  v0.16b, v1.16b, #8\n"               \
  "ext v10.16b, v1.16b, v2.16b, #8\n"               \
  "ext v13.16b, v3.16b, v4.16b, #8\n"               \
  "ext v16.16b, v4.16b, v5.16b, #8\n"               \
  "fmla v28.4s, v6.4s,  %[w0].s[1]\n"               \
  "fmla v29.4s, v9.4s,  %[w0].s[1]\n"               \
  "fmla v30.4s, v12.4s, %[w0].s[1]\n"               \
  "fmla v31.4s, v15.4s, %[w0].s[1]\n"               \
  "ext v8.16b,  v0.16b, v1.16b, #12\n"              \
  "ext v11.16b, v1.16b, v2.16b, #12\n"              \
  "ext v14.16b, v3.16b, v4.16b, #12\n"              \
  "ext v17.16b, v4.16b, v5.16b, #12\n"              \
  "fmla v28.4s, v1.4s,  %[w1].s[0]\n"               \
  "fmla v29.4s, v2.4s,  %[w1].s[0]\n"               \
  "fmla v30.4s, v4.4s,  %[w1].s[0]\n"               \
  "fmla v31.4s, v5.4s,  %[w1].s[0]\n"               \
  "ld1 {v0.4s, v1.4s, v2.4s}, [%[din_ptr2]]\n"      \
  "fmla v28.4s, v7.4s,  %[w0].s[2]\n"               \
  "fmla v29.4s, v10.4s, %[w0].s[2]\n"               \
  "fmla v30.4s, v13.4s, %[w0].s[2]\n"               \
  "fmla v31.4s, v16.4s, %[w0].s[2]\n"               \
  "ld1  {v6.4s}, [%[vmask]]\n"                      \
  "fmla v28.4s, v8.4s,  %[w0].s[3]\n"               \
  "fmla v29.4s, v11.4s, %[w0].s[3]\n"               \
  "fmla v30.4s, v14.4s, %[w0].s[3]\n"               \
  "fmla v31.4s, v17.4s, %[w0].s[3]\n"               \
  "bif  v0.16b, %[vzero].16b, v6.16b\n"             \
  "bif  v1.16b, %[vzero].16b, v18.16b\n"            \
  "bif  v2.16b, %[vzero].16b, v19.16b\n"            \
  /* line 1 */                                      \
  "fmla v28.4s, v3.4s,  %[w1].s[1]\n"               \
  "fmla v29.4s, v4.4s,  %[w1].s[1]\n"               \
  "fmla v30.4s, v0.4s,  %[w1].s[1]\n"               \
  "fmla v31.4s, v1.4s,  %[w1].s[1]\n"               \
  "ext v6.16b,  v0.16b, v1.16b, #4\n"               \
  "ext v9.16b,  v1.16b, v2.16b, #4\n"               \
  "ext v7.16b,  v0.16b, v1.16b, #8\n"               \
  "ext v10.16b, v1.16b, v2.16b, #8\n"               \
  "fmla v28.4s, v4.4s,  %[w2].s[1]\n"               \
  "fmla v29.4s, v5.4s,  %[w2].s[1]\n"               \
  "fmla v30.4s, v1.4s,  %[w2].s[1]\n"               \
  "fmla v31.4s, v2.4s,  %[w2].s[1]\n"               \
  "ld1 {v3.4s, v4.4s, v5.4s}, [%[din_ptr3]]\n"      \
  "ext v8.16b,  v0.16b, v1.16b, #12\n"              \
  "ext v11.16b, v1.16b, v2.16b, #12\n"              \
  "fmla v28.4s, v12.4s, %[w1].s[2]\n"               \
  "fmla v29.4s, v15.4s, %[w1].s[2]\n"               \
  "fmla v30.4s, v6.4s,  %[w1].s[2]\n"               \
  "fmla v31.4s, v9.4s,  %[w1].s[2]\n"               \
  "ld1 {v12.4s}, [%[vmask]]\n"                      \
  "fmla v28.4s, v13.4s, %[w1].s[3]\n"               \
  "fmla v29.4s, v16.4s, %[w1].s[3]\n"               \
  "fmla v30.4s, v7.4s,  %[w1].s[3]\n"               \
  "fmla v31.4s, v10.4s, %[w1].s[3]\n"               \
  "bif  v3.16b, %[vzero].16b, v12.16b\n"            \
  "bif  v4.16b, %[vzero].16b, v18.16b\n"            \
  "bif  v5.16b, %[vzero].16b, v19.16b\n"            \
  "fmla v28.4s, v14.4s, %[w2].s[0]\n"               \
  "fmla v29.4s, v17.4s, %[w2].s[0]\n"               \
  "fmla v30.4s, v8.4s,  %[w2].s[0]\n"               \
  "fmla v31.4s, v11.4s, %[w2].s[0]\n"               \
  "ext v12.16b, v3.16b, v4.16b, #4\n"               \
  "ext v15.16b, v4.16b, v5.16b, #4\n"               \
  "ext v13.16b, v3.16b, v4.16b, #8\n"               \
  "ext v16.16b, v4.16b, v5.16b, #8\n"               \
  /* line 2 */                                      \
  "fmla v28.4s, v0.4s,  %[w2].s[2]\n"               \
  "fmla v29.4s, v1.4s,  %[w2].s[2]\n"               \
  "fmla v30.4s, v3.4s,  %[w2].s[2]\n"               \
  "fmla v31.4s, v4.4s,  %[w2].s[2]\n"               \
  "ext v14.16b, v3.16b, v4.16b, #12\n"              \
  "ext v17.16b, v4.16b, v5.16b, #12\n"              \
  "fmla v28.4s, v1.4s,  %[w3].s[2]\n"               \
  "fmla v29.4s, v2.4s,  %[w3].s[2]\n"               \
  "fmla v30.4s, v4.4s,  %[w3].s[2]\n"               \
  "fmla v31.4s, v5.4s,  %[w3].s[2]\n"               \
  "ld1 {v0.4s, v1.4s, v2.4s}, [%[din_ptr4]]\n"      \
  "fmla v28.4s, v6.4s,  %[w2].s[3]\n"               \
  "fmla v29.4s, v9.4s,  %[w2].s[3]\n"               \
  "fmla v30.4s, v12.4s, %[w2].s[3]\n"               \
  "fmla v31.4s, v15.4s, %[w2].s[3]\n"               \
  "ld1 {v6.4s}, [%[vmask]]\n"                       \
  "fmla v28.4s, v7.4s,  %[w3].s[0]\n"               \
  "fmla v29.4s, v10.4s, %[w3].s[0]\n"               \
  "fmla v30.4s, v13.4s, %[w3].s[0]\n"               \
  "fmla v31.4s, v16.4s, %[w3].s[0]\n"               \
  "bif  v0.16b, %[vzero].16b, v6.16b\n"             \
  "bif  v1.16b, %[vzero].16b, v18.16b\n"            \
  "bif  v2.16b, %[vzero].16b, v19.16b\n"            \
  "fmla v28.4s, v8.4s,  %[w3].s[1]\n"               \
  "fmla v29.4s, v11.4s, %[w3].s[1]\n"               \
  "fmla v30.4s, v14.4s, %[w3].s[1]\n"               \
  "fmla v31.4s, v17.4s, %[w3].s[1]\n"               \
  "ext v6.16b,  v0.16b, v1.16b, #4\n"               \
  "ext v9.16b,  v1.16b, v2.16b, #4\n"               \
  "ext v7.16b,  v0.16b, v1.16b, #8\n"               \
  "ext v10.16b, v1.16b, v2.16b, #8\n"               \
  /* line 3 */                                      \
  "fmla v28.4s, v3.4s,  %[w3].s[3]\n"               \
  "fmla v29.4s, v4.4s,  %[w3].s[3]\n"               \
  "fmla v30.4s, v0.4s,  %[w3].s[3]\n"               \
  "fmla v31.4s, v1.4s,  %[w3].s[3]\n"               \
  "ext v8.16b,  v0.16b, v1.16b, #12\n"              \
  "ext v11.16b, v1.16b, v2.16b, #12\n"              \
  "fmla v28.4s, v4.4s,  %[w4].s[3]\n"               \
  "fmla v29.4s, v5.4s,  %[w4].s[3]\n"               \
  "fmla v30.4s, v1.4s,  %[w4].s[3]\n"               \
  "fmla v31.4s, v2.4s,  %[w4].s[3]\n"               \
  "ld1 {v3.4s, v4.4s, v5.4s}, [%[din_ptr5]]\n"      \
  "fmla v28.4s, v12.4s, %[w4].s[0]\n"               \
  "fmla v29.4s, v15.4s, %[w4].s[0]\n"               \
  "fmla v30.4s, v6.4s,  %[w4].s[0]\n"               \
  "fmla v31.4s, v9.4s,  %[w4].s[0]\n"               \
  "ld1 {v12.4s}, [%[vmask]]\n"                      \
  "fmla v28.4s, v13.4s, %[w4].s[1]\n"               \
  "fmla v29.4s, v16.4s, %[w4].s[1]\n"               \
  "fmla v30.4s, v7.4s,  %[w4].s[1]\n"               \
  "fmla v31.4s, v10.4s, %[w4].s[1]\n"               \
  "bif  v3.16b, %[vzero].16b, v12.16b\n"            \
  "bif  v4.16b, %[vzero].16b, v18.16b\n"            \
  "bif  v5.16b, %[vzero].16b, v19.16b\n"            \
  "fmla v28.4s, v14.4s, %[w4].s[2]\n"               \
  "fmla v29.4s, v17.4s, %[w4].s[2]\n"               \
  "fmla v30.4s, v8.4s,  %[w4].s[2]\n"               \
  "fmla v31.4s, v11.4s, %[w4].s[2]\n"               \
  "ext v12.16b, v3.16b, v4.16b, #4\n"               \
  "ext v15.16b, v4.16b, v5.16b, #4\n"               \
  "ext v13.16b, v3.16b, v4.16b, #8\n"               \
  "ext v16.16b, v4.16b, v5.16b, #8\n"               \
  /* line 4 */                                      \
  "fmla v28.4s, v0.4s,  %[w5].s[0]\n"               \
  "fmla v29.4s, v1.4s,  %[w5].s[0]\n"               \
  "fmla v30.4s, v3.4s,  %[w5].s[0]\n"               \
  "fmla v31.4s, v4.4s,  %[w5].s[0]\n"               \
  "ext v14.16b, v3.16b, v4.16b, #12\n"              \
  "ext v17.16b, v4.16b, v5.16b, #12\n"              \
  "fmla v28.4s, v1.4s,  %[w6].s[0]\n"               \
  "fmla v29.4s, v2.4s,  %[w6].s[0]\n"               \
  "fmla v30.4s, v4.4s,  %[w6].s[0]\n"               \
  "fmla v31.4s, v5.4s,  %[w6].s[0]\n"               \
  "fmla v28.4s, v6.4s,  %[w5].s[1]\n"               \
  "fmla v29.4s, v9.4s,  %[w5].s[1]\n"               \
  "fmla v30.4s, v12.4s, %[w5].s[1]\n"               \
  "fmla v31.4s, v15.4s, %[w5].s[1]\n"               \
  "fmla v28.4s, v7.4s,  %[w5].s[2]\n"               \
  "fmla v29.4s, v10.4s, %[w5].s[2]\n"               \
  "fmla v30.4s, v13.4s, %[w5].s[2]\n"               \
  "fmla v31.4s, v16.4s, %[w5].s[2]\n"               \
  "fmla v28.4s, v8.4s,  %[w5].s[3]\n"               \
  "fmla v29.4s, v11.4s, %[w5].s[3]\n"               \
  "fmla v30.4s, v14.4s, %[w5].s[3]\n"               \
  "fmla v31.4s, v17.4s, %[w5].s[3]\n"

#define RIGHT_RESULT_S1                       \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"  \
  "st1 {v30.4s, v31.4s}, [%[doutr1]], #32\n"  \
  "3:                             \n"

#define RIGHT_RESULT_S1_RELU                  \
  "fmax  v28.4s, v28.4s, %[vzero].4s\n"       \
  "fmax  v29.4s, v29.4s, %[vzero].4s\n"       \
  "fmax  v30.4s, v30.4s, %[vzero].4s\n"       \
  "fmax  v31.4s, v31.4s, %[vzero].4s\n"       \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"  \
  "st1 {v30.4s, v31.4s}, [%[doutr1]], #32\n"  \
  "3:                             \n"

#define RIGHT_RESULT_S1_RELU6                 \
  "ld1 {v8.4s}, [%[six_ptr]]   \n"            \
  "fmax  v28.4s, v28.4s, %[vzero].4s\n"       \
  "fmax  v29.4s, v29.4s, %[vzero].4s\n"       \
  "fmax  v30.4s, v30.4s, %[vzero].4s\n"       \
  "fmax  v31.4s, v31.4s, %[vzero].4s\n"       \
  "fmin v28.4s, v28.4s, v8.4s\n"              \
  "fmin v29.4s, v29.4s, v8.4s\n"              \
  "fmin v30.4s, v30.4s, v8.4s\n"              \
  "fmin v31.4s, v31.4s, v8.4s\n"              \
  "st1 {v28.4s, v29.4s}, [%[doutr0]], #32\n"  \
  "st1 {v30.4s, v31.4s}, [%[doutr1]], #32\n"  \
  "3:                             \n"

#else
inline std::pair<uint32_t, uint32_t> right_mask_5x5s1p2_fp32(int win,
                                                             int wout,
                                                             uint32_t* vmask) {
  uint32_t right_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint32_t cnt_col = ((wout >> 2) - 2);
  uint32_t size_right_remain = static_cast<uint32_t>(win - (2 + cnt_col * 4));
  if (size_right_remain >= 8) {
    cnt_col++;
    size_right_remain -= 4;
  }
  uint32_t cnt_remain = (size_right_remain >= 5 && wout % 4 == 0)
                            ? 4
                            : static_cast<uint32_t>(wout % 4);
  size_right_remain = (cnt_remain == 4) ? size_right_remain :
                      (size_right_remain + 4 - cnt_remain);
  uint32x4_t vmask_rp1 =
      vcgtq_u32(vdupq_n_u32(size_right_remain), vld1q_u32(right_pad_idx));
  uint32x4_t vmask_rp2 =
      vcgtq_u32(vdupq_n_u32(size_right_remain), vld1q_u32(right_pad_idx + 4));
  vst1q_u32(vmask, vmask_rp1);
  vst1q_u32(vmask + 4, vmask_rp2);
  return std::make_pair(cnt_col, cnt_remain);
}
#define DIN_PTR_INIT           \
  const float *din_ptr0 = dr0; \
  const float *din_ptr1 = dr1; \
  const float *din_ptr2 = dr2; \
  const float *din_ptr3 = dr3; \
  const float *din_ptr4 = dr4; \
  float *doutr0 = dout_ptr;    \
  /* h - 2 + 5 = h + 3 */      \
  if (h + 3 > hin) {           \
    switch (h + 3 - hin) {     \
      case 4:                  \
        din_ptr1 = zero_ptr;   \
      case 3:                  \
        din_ptr2 = zero_ptr;   \
      case 2:                  \
        din_ptr3 = zero_ptr;   \
      case 1:                  \
        din_ptr4 = zero_ptr;   \
      default:                 \
        break;                 \
    }                          \
  }                            \
  /* update in_address */      \
  dr0 = dr1;                   \
  dr1 = dr2;                   \
  dr2 = dr3;                   \
  dr3 = dr4;                   \
  dr4 = dr3 + win;

#define LEFT_COMPUTE_S1               \
  "pld  [%[wei_ptr]]  \n"             \
  "pld  [%[din_ptr0]] \n"             \
  "pld  [%[din_ptr1]] \n"             \
  "pld  [%[din_ptr2]] \n"             \
  "pld  [%[din_ptr3]] \n"             \
  "vld1.32 {d0-d3}, [%[wei_ptr]]!\n"  \
  "pld  [%[din_ptr4]] \n"             \
  "vmov.u32  q7, #0   \n"             \
  "vld1.32 {d16-d19}, [%[din_ptr0]]!\n"\
  "vld1.32 {d30-d31}, [%[bias_val]] \n"\
  "vld1.32 {d4-d7}, [%[wei_ptr]]!\n"   \
  /* line 0 */                         \
  "sub     %[din_ptr0], #24\n"         \
  "vext.32 q10, q7, q8, #2\n"          \
  "vext.32 q11, q7, q8, #3\n"          \
  "vld1.32 {d8-d11}, [%[wei_ptr]]!\n"  \
  "vext.32  q12, q8,  q9, #1\n"        \
  "vext.32  q13, q8,  q9, #2\n"        \
  "vmla.f32 q15, q10, d0[0]\n"         \
  "vmul.f32 q14, q11, d0[1]\n"         \
  "vld1.32 {d20-d23}, [%[din_ptr1]]!\n"\
  "vld1.32 {d12},     [%[wei_ptr]]!\n" \
  "vmla.f32 q15, q8,  d1[0]\n"         \
  "vmla.f32 q14, q12, d1[1]\n"         \
  "sub      %[din_ptr1], #24\n"        \
  "vmla.f32 q15, q13, d2[0]\n"         \
  /* line 1 */                         \
  "vext.32  q8,  q7,  q10, #2\n"       \
  "vext.32  q9,  q7,  q10, #3\n"       \
  "vmla.f32 q14, q8,  d2[1]\n"         \
  "vmla.f32 q15, q9,  d3[0]\n"         \
  "vld1.32 {d16-d19}, [%[din_ptr2]]!\n"\
  "vext.32  q12, q10, q11, #1\n"       \
  "vext.32  q13, q10, q11, #2\n"       \
  "vmla.f32 q14, q10, d3[1]\n"         \
  "vmla.f32 q15, q12, d4[0]\n"         \
  "vext.32  q10,  q7,  q8, #2\n"       \
  "vext.32  q11,  q7,  q8, #3\n"       \
  "vmla.f32 q14, q13, d4[1]\n"         \
  /* line 2 */                         \
  "sub      %[din_ptr2], #24\n"        \
  "vmla.f32 q15, q10, d5[0]\n"         \
  "vmla.f32 q14, q11, d5[1]\n"         \
  "vld1.32 {d20-d23}, [%[din_ptr3]]!\n"\
  "vext.32  q12, q8,  q9, #1\n"        \
  "vext.32  q13, q8,  q9, #2\n"        \
  "vmla.f32 q15, q8,  d6[0]\n"         \
  "vmla.f32 q14, q12, d6[1]\n"         \
  "vext.32  q8,  q7,  q10, #2\n"       \
  "vext.32  q9,  q7,  q10, #3\n"       \
  "vmla.f32 q15, q13, d7[0]\n"         \
  /* line 3 */                         \
  "sub      %[din_ptr3], #24\n"        \
  "vmla.f32 q14, q8,  d7[1]\n"         \
  "vmla.f32 q15, q9,  d8[0]\n"         \
  "vld1.32 {d16-d19}, [%[din_ptr4]]!\n"\
  "vext.32  q12, q10, q11, #1\n"       \
  "vext.32  q13, q10, q11, #2\n"       \
  "vmla.f32 q14, q10, d8[1]\n"         \
  "vmla.f32 q15, q12, d9[0]\n"         \
  "vext.32  q10,  q7,  q8, #2\n"       \
  "vext.32  q11,  q7,  q8, #3\n"       \
  "vmla.f32 q14, q13, d9[1]\n"         \
  /* line 4 */                         \
  "sub      %[din_ptr4], #24\n"        \
  "vmla.f32 q15, q10, d10[0]\n"        \
  "vmla.f32 q14, q11, d10[1]\n"        \
  "vext.32  q12, q8,  q9, #1\n"        \
  "vext.32  q13, q8,  q9, #2\n"        \
  "vmla.f32 q15, q8,  d11[0]\n"        \
  "vmla.f32 q14, q12, d11[1]\n"        \
  "vld1.32 {d16-d19}, [%[din_ptr0]]!\n"\
  "vmla.f32 q15, q13, d12[0]\n"

#define LEFT_RESULT_S1                 \
  "cmp %[cnt], #16\n"                  \
  "vadd.f32 q13, q14, q15\n"           \
  "sub      %[din_ptr0], #16\n"        \
  "vld1.32 {d30-d31}, [%[bias_val]]\n" \
  "vext.32  q10,  q8,  q9, #1\n"       \
  "vext.32  q11,  q8,  q9, #2\n"       \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n"  \
  "blt 2f\n"
#define LEFT_RESULT_S1_RELU            \
  "cmp %[cnt], #16\n"                  \
  "vadd.f32 q13, q14, q15\n"           \
  "sub      %[din_ptr0], #16\n"        \
  "vmax.f32 q13, q13, q7\n"            \
  "vld1.32 {d30-d31}, [%[bias_val]]\n" \
  "vext.32  q10,  q8,  q9, #1\n"       \
  "vext.32  q11,  q8,  q9, #2\n"       \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n"  \
  "blt 2f\n"

#define LEFT_RESULT_S1_RELU6           \
  "cmp %[cnt], #16\n"                  \
  "vldr d20, [%[bias_val], #16]\n"     \
  "vldr d21, [%[bias_val], #24]\n"     \
  "vadd.f32 q13, q14, q15\n"           \
  "sub      %[din_ptr0], #16\n"        \
  "vmax.f32 q13, q13, q7\n"            \
  "vld1.32 {d30-d31}, [%[bias_val]]\n" \
  "vmin.f32 q13, q13, q10\n"           \
  "vext.32  q10,  q8,  q9, #1\n"       \
  "vext.32  q11,  q8,  q9, #2\n"       \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n"  \
  "blt 2f\n"

#define MID_COMPITE_S1                 \
  "1: \n"                              \
  /* line 0 */                         \
  "vmla.f32 q15,  q8,  d0[0]\n"        \
  "vmul.f32 q14,  q9,  d2[0]\n"        \
  "vext.32  q12,  q8,  q9, #3\n"       \
  "vld1.32 {d16-d19}, [%[din_ptr1]]!\n"\
  "vmul.f32 q13, q10,  d0[1]\n"        \
  "vmla.f32 q15, q11,  d1[0]\n"        \
  "vmla.f32 q14, q12,  d1[1]\n"        \
  /* line 1 */                         \
  "sub      %[din_ptr1], #16\n"        \
  "vext.32  q10,  q8,  q9, #1\n"       \
  "vext.32  q11,  q8,  q9, #2\n"       \
  "vmla.f32 q13,  q8,  d2[1]\n"        \
  "vmla.f32 q15,  q9,  d4[1]\n"        \
  "vext.32  q12,  q8,  q9, #3\n"       \
  "vld1.32 {d16-d19}, [%[din_ptr2]]!\n"\
  "vmla.f32 q14, q10,  d3[0]\n"        \
  "vmla.f32 q13, q11,  d3[1]\n"        \
  "vmla.f32 q15, q12,  d4[0]\n"        \
  /* line 2 */                         \
  "sub      %[din_ptr2], #16\n"        \
  "vext.32  q10,  q8,  q9, #1\n"       \
  "vext.32  q11,  q8,  q9, #2\n"       \
  "vmla.f32 q14,  q8,  d5[0]\n"        \
  "vmla.f32 q13,  q9,  d7[0]\n"        \
  "vext.32  q12,  q8,  q9, #3\n"       \
  "vld1.32 {d16-d19}, [%[din_ptr3]]!\n"\
  "vmla.f32 q15, q10,  d5[1]\n"        \
  "vmla.f32 q14, q11,  d6[0]\n"        \
  "vmla.f32 q13, q12,  d6[1]\n"        \
  /* line 3 */                         \
  "sub      %[din_ptr3], #16\n"        \
  "vext.32  q10,  q8,  q9, #1\n"       \
  "vext.32  q11,  q8,  q9, #2\n"       \
  "vmla.f32 q15,  q8,  d7[1]\n"        \
  "vmla.f32 q14,  q9,  d9[1]\n"        \
  "vext.32  q12,  q8,  q9, #3\n"       \
  "vld1.32 {d16-d19}, [%[din_ptr4]]!\n"\
  "vmla.f32 q13, q10,  d8[0]\n"        \
  "vmla.f32 q15, q11,  d8[1]\n"        \
  "vmla.f32 q14, q12,  d9[0]\n"        \
  /* line 4 */                         \
  "sub      %[din_ptr4], #16\n"        \
  "vext.32  q10,  q8,  q9, #1\n"       \
  "vext.32  q11,  q8,  q9, #2\n"       \
  "vmla.f32 q13,  q8,  d10[0]\n"       \
  "vmla.f32 q15,  q9,  d12[0]\n"       \
  "vext.32  q12,  q8,  q9, #3\n"       \
  "vld1.32 {d16-d19}, [%[din_ptr0]]!\n"\
  "sub      %[cnt],   #16\n"           \
  "vmla.f32 q14, q10,  d10[1]\n"       \
  "vmla.f32 q13, q11,  d11[0]\n"       \
  "vmla.f32 q15, q12,  d11[1]\n"

#define MID_RESULT_S1                 \
  "sub      %[din_ptr0], #16\n"       \
  "vadd.f32 q12, q14, q13\n"          \
  "cmp     %[cnt], #16\n"             \
  "vext.32  q10,  q8,  q9, #1\n"      \
  "vext.32  q11,  q8,  q9, #2\n"      \
  "vadd.f32 q13, q12, q15\n"          \
  "vld1.32 {d30-d31}, [%[bias_val]]\n"\
  "vst1.32 {d26-d27}, [%[doutr0]]!\n" \
  "bge 1b\n"

#define MID_RESULT_S1_RELU            \
  "sub      %[din_ptr0], #16\n"       \
  "vadd.f32 q12, q14, q13\n"          \
  "cmp     %[cnt], #16\n"             \
  "vext.32  q10,  q8,  q9, #1\n"      \
  "vext.32  q11,  q8,  q9, #2\n"      \
  "vadd.f32 q13, q12, q15\n"          \
  "vld1.32 {d30-d31}, [%[bias_val]]\n"\
  "vmax.f32 q13, q13, q7\n"           \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n" \
  "bge 1b\n"
#define MID_RESULT_S1_RELU6           \
  "sub      %[din_ptr0], #16\n"       \
  "vadd.f32 q12, q14, q13\n"          \
  "cmp     %[cnt], #16\n"             \
  "vldr     d28, [%[bias_val], #16]\n"\
  "vldr     d29, [%[bias_val], #24]\n"\
  "vadd.f32 q13, q12, q15\n"          \
  "vext.32  q10,  q8,  q9, #1\n"      \
  "vext.32  q11,  q8,  q9, #2\n"      \
  "vmax.f32 q13, q13, q7\n"           \
  "vld1.32 {d30-d31}, [%[bias_val]]\n"\
  "vmin.f32 q13, q13, q14\n"          \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n" \
  "bge 1b\n"

#define RIGHT_COMPUTE_S1               \
  "2: \n"                              \
  "sub     %[din_ptr0], #16\n"         \
  "cmp     %[cnt], #1\n"               \
  "vld1.32 {d24-d27}, [%[vmask]]\n"    \
  "sub     %[din_ptr0], %[right_pad_num]\n"\
  "sub     %[din_ptr1], %[right_pad_num]\n"\
  "sub     %[din_ptr2], %[right_pad_num]\n"\
  "sub     %[din_ptr3], %[right_pad_num]\n"\
  "sub     %[din_ptr4], %[right_pad_num]\n"\
  "blt 3f\n"                           \
  "vld1.32 {d16-d19}, [%[din_ptr0]]\n" \
  "sub     %[doutr0], %[right_pad_num]\n"\
  "vbif q8, q7, q12\n"                 \
  "vbif q9, q7, q13\n"                 \
  "vld1.32 {d30-d31}, [%[bias_val]]\n"\
  /* line 0 */                         \
  "vext.32  q10,  q8,  q9, #1\n"       \
  "vext.32  q11,  q8,  q9, #2\n"       \
  "vmla.f32 q15,  q8,  d0[0]\n"        \
  "vmul.f32 q14,  q9,  d2[0]\n"        \
  "vext.32  q12,  q8,  q9, #3\n"       \
  "vld1.32 {d16-d19}, [%[din_ptr1]]\n" \
  "vmla.f32 q15, q10,  d0[1]\n"        \
  "vld1.32 {d20-d21}, [%[vmask]]\n"    \
  "vmla.f32 q14, q11,  d1[0]\n"        \
  "vbif q9, q7, q13\n"                 \
  "vbif q8, q7, q10\n"                 \
  "vmla.f32 q15, q12,  d1[1]\n"        \
  /* line 1 */                         \
  "vext.32  q10,  q8,  q9, #1\n"       \
  "vext.32  q11,  q8,  q9, #2\n"       \
  "vmla.f32 q14,  q8,  d2[1]\n"        \
  "vmla.f32 q15,  q9,  d4[1]\n"        \
  "vext.32  q12,  q8,  q9, #3\n"       \
  "vld1.32 {d16-d19}, [%[din_ptr2]]\n" \
  "vmla.f32 q14, q10,  d3[0]\n"        \
  "vld1.32 {d20-d21}, [%[vmask]]\n"    \
  "vmla.f32 q15, q11,  d3[1]\n"        \
  "vbif q9, q7, q13\n"                 \
  "vbif q8, q7, q10\n"                 \
  "vmla.f32 q14, q12,  d4[0]\n"        \
  /* line 2 */                         \
  "vext.32  q10,  q8,  q9, #1\n"       \
  "vext.32  q11,  q8,  q9, #2\n"       \
  "vmla.f32 q15,  q8,  d5[0]\n"        \
  "vmla.f32 q14,  q9,  d7[0]\n"        \
  "vext.32  q12,  q8,  q9, #3\n"       \
  "vld1.32 {d16-d19}, [%[din_ptr3]] \n"\
  "vmla.f32 q15, q10,  d5[1]\n"        \
  "vld1.32 {d20-d21}, [%[vmask]]\n"    \
  "vmla.f32 q14, q11,  d6[0]\n"        \
  "vbif q9, q7, q13\n"                 \
  "vbif q8, q7, q10\n"                 \
  "vmla.f32 q15, q12,  d6[1]\n"        \
  /* line 3 */                         \
  "vext.32  q10,  q8,  q9, #1\n"       \
  "vext.32  q11,  q8,  q9, #2\n"       \
  "vmla.f32 q14,  q8,  d7[1]\n"        \
  "vmla.f32 q15,  q9,  d9[1]\n"        \
  "vext.32  q12,  q8,  q9, #3\n"       \
  "vld1.32 {d16-d19}, [%[din_ptr4]] \n"\
  "vmla.f32 q14, q10,  d8[0]\n"        \
  "vld1.32 {d20-d21}, [%[vmask]]\n"    \
  "vmla.f32 q15, q11,  d8[1]\n"        \
  "vbif q9, q7, q13\n"                 \
  "vbif q8, q7, q10\n"                 \
  "vmla.f32 q14, q12,  d9[0]\n"        \
  /* line 4 */                         \
  "vext.32  q10,  q8,  q9, #1\n"       \
  "vext.32  q11,  q8,  q9, #2\n"       \
  "vmla.f32 q15,  q8,  d10[0]\n"       \
  "vmla.f32 q14,  q9,  d12[0]\n"       \
  "vext.32  q12,  q8,  q9, #3\n"       \
  "vmla.f32 q15, q10,  d10[1]\n"       \
  "vmla.f32 q14, q11,  d11[0]\n"       \
  "vmla.f32 q15, q12,  d11[1]\n"

#define RIGHT_RESULT_S1                \
  "vadd.f32  q13, q15, q14\n"          \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n"  \
  "3:                             \n"
#define RIGHT_RESULT_S1_RELU           \
  "vadd.f32  q13, q15, q14\n"          \
  "vmax.f32  q13, q13, q7\n"           \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n"  \
  "3:                             \n"
#define RIGHT_RESULT_S1_RELU6          \
  "vldr d20, [%[bias_val], #16]\n"     \
  "vldr d21, [%[bias_val], #24]\n"     \
  "vadd.f32  q13, q15, q14\n"          \
  "vmax.f32  q13, q13, q7\n"           \
  "vmin.f32  q13, q13, q10\n"          \
  "vst1.32 {d26-d27}, [%[doutr0]]!\n"  \
  "3:                             \n"
#endif

void conv_depthwise_5x5s1p2_fp32_relu(IN_PARAM, ARMContext *ctx) {
  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = 25;
  uint32_t vmask[12];
  auto&& res = right_mask_5x5s1p2_fp32(win, wout, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
#ifdef __aarch64__
    uint32_t right_pad_num = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 4);
#else
    uint32_t right_pad_num = (cnt_remain == 4) ? 0 : ((4 - cnt_remain) * 4);
#endif
  float *zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, (win + 16) * sizeof(float));
  float *write_ptr = zero_ptr + win + 16;
  cnt_col = (cnt_col << 4) + cnt_remain;
  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * chin * size_in_channel;
    float *dout_batch = dout + n * chin * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, chin) {
      float *dout_ptr = dout_batch + c * size_out_channel;
      const float *din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0.f;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
      const float *wei_ptr = weights + c * w_stride;
      const float *dr0 = zero_ptr;
      const float *dr1 = zero_ptr;
      const float *dr2 = din_ch_ptr;
      const float *dr3 = dr2 + win;
      const float *dr4 = dr3 + win;
      const float *dr5 = dr4 + win;
#ifdef __aarch64__
      float32x4_t w0 = vld1q_f32(wei_ptr);
      float32x4_t w1 = vld1q_f32(wei_ptr + 4);
      float32x4_t w2 = vld1q_f32(wei_ptr + 8);
      float32x4_t w3 = vld1q_f32(wei_ptr + 12);
      float32x4_t w4 = vld1q_f32(wei_ptr + 16);
      float32x4_t w5 = vld1q_f32(wei_ptr + 20);
      float32x4_t w6 = vdupq_n_f32(wei_ptr[24]);
      float32x4_t vzero = vdupq_n_f32(0.f);
      for (int h = 0; h < hout; h += 2) {
        DIN_PTR_INIT
        int cnt = cnt_col;
        asm volatile(
          LEFT_COMPUTE_S1 LEFT_RESULT_S1_RELU
          MID_COMPITE_S1 MID_RESULT_S1_RELU
          RIGHT_COMPUTE_S1 RIGHT_RESULT_S1_RELU
          : [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2),
            [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5),
            [doutr0] "+r"(doutr0), [doutr1] "+r"(doutr1), [cnt] "+r"(cnt)
          : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3), [w4] "w"(w4), [w5] "w"(w5),
            [w6] "w"(w6), [vzero] "w"(vzero), [bias_val] "r"(vbias),
            [right_pad_num] "r"(right_pad_num), [vmask] "r"(vmask)
          : "cc","memory", "v0","v1","v2","v3","v4","v5","v6","v7",
            "v8","v9","v10","v11","v12","v13", "v14","v15","v16","v17","v18","v19",
            "v28","v29","v30","v31"
        );
        dout_ptr += 2 * wout;
      }
#else
      for (int h = 0; h < hout; h++) {
        DIN_PTR_INIT
        int cnt = cnt_col;
        auto weight_ptr = wei_ptr;
        asm volatile(
          LEFT_COMPUTE_S1 LEFT_RESULT_S1_RELU
          MID_COMPITE_S1 MID_RESULT_S1_RELU
          RIGHT_COMPUTE_S1 RIGHT_RESULT_S1_RELU
          : [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2),
            [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4),
            [doutr0] "+r"(doutr0), [cnt] "+r"(cnt), [wei_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(vbias), [right_pad_num] "r"(right_pad_num), [vmask] "r"(vmask)
          : "cc","memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
            "q7", "q8","q9","q10","q11","q12","q13", "q14","q15"
        );
        dout_ptr += wout;
      }
#endif
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_5x5s1p2_fp32_relu6(IN_PARAM, float six, ARMContext *ctx) {
  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = 25;
  uint32_t vmask[12];
  auto&& res = right_mask_5x5s1p2_fp32(win, wout, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
#ifdef __aarch64__
    uint32_t right_pad_num = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 4);
#else
    uint32_t right_pad_num = (cnt_remain == 4) ? 0 : ((4 - cnt_remain) * 4);
#endif
  float *zero_ptr = ctx->workspace_data<float>();
  memset(zero_ptr, 0, (win + 16) * sizeof(float));
  float *write_ptr = zero_ptr + win + 16;
  cnt_col = (cnt_col << 4) + cnt_remain;
  float six_val[4] = {six, six, six, six};
  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * chin * size_in_channel;
    float *dout_batch = dout + n * chin * size_out_channel;
    LITE_PARALLEL_BEGIN(c, tid, chin) {
    // for (int c = 0; c < chin; c++) {
      float *dout_ptr = dout_batch + c * size_out_channel;
      const float *din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0.f;
      const float *wei_ptr = weights + c * w_stride;
      const float *dr0 = zero_ptr;
      const float *dr1 = zero_ptr;
      const float *dr2 = din_ch_ptr;
      const float *dr3 = dr2 + win;
      const float *dr4 = dr3 + win;
      const float *dr5 = dr4 + win;
#ifdef __aarch64__
      float32x4_t w0 = vld1q_f32(wei_ptr);
      float32x4_t w1 = vld1q_f32(wei_ptr + 4);
      float32x4_t w2 = vld1q_f32(wei_ptr + 8);
      float32x4_t w3 = vld1q_f32(wei_ptr + 12);
      float32x4_t w4 = vld1q_f32(wei_ptr + 16);
      float32x4_t w5 = vld1q_f32(wei_ptr + 20);
      float32x4_t w6 = vdupq_n_f32(wei_ptr[24]);
      float32x4_t vzero = vdupq_n_f32(0.f);
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
      for (int h = 0; h < hout; h += 2) {
        DIN_PTR_INIT
        int cnt = cnt_col;
        asm volatile(
          LEFT_COMPUTE_S1 LEFT_RESULT_S1_RELU6
          MID_COMPITE_S1 MID_RESULT_S1_RELU6
          RIGHT_COMPUTE_S1 RIGHT_RESULT_S1_RELU6
          : [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2),
            [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5),
            [doutr0] "+r"(doutr0), [doutr1] "+r"(doutr1), [cnt] "+r"(cnt)
          : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3), [w4] "w"(w4), [w5] "w"(w5),
            [w6] "w"(w6), [vzero] "w"(vzero), [bias_val] "r"(vbias),
            [right_pad_num] "r"(right_pad_num), [vmask] "r"(vmask), [six_ptr] "r"(six_val)
          : "cc","memory", "v0","v1","v2","v3","v4","v5","v6","v7",
            "v8","v9","v10","v11","v12","v13", "v14","v15","v16","v17","v18","v19",
            "v28","v29","v30","v31"
        );
        dout_ptr += 2 * wout;
      }
#else
      float vbias[8] = {
          bias_val, bias_val, bias_val, bias_val, six, six, six, six};
      for (int h = 0; h < hout; h++) {
        DIN_PTR_INIT
        int cnt = cnt_col;
        auto weight_ptr = wei_ptr;
        asm volatile(
          LEFT_COMPUTE_S1 LEFT_RESULT_S1_RELU6
          MID_COMPITE_S1 MID_RESULT_S1_RELU6
          RIGHT_COMPUTE_S1 RIGHT_RESULT_S1_RELU6
          : [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2),
            [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4),
            [doutr0] "+r"(doutr0), [cnt] "+r"(cnt), [wei_ptr] "+r"(weight_ptr)
          : [bias_val] "r"(vbias), [right_pad_num] "r"(right_pad_num), [vmask] "r"(vmask)
          : "cc","memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
            "q7", "q8","q9","q10","q11","q12","q13", "q14","q15"
        );
        dout_ptr += wout;
      }
#endif
    }
    LITE_PARALLEL_END();
  }
}

void conv_depthwise_5x5s1p2_fp32(float *dout,
                                 const float *din,
                                 const float *weights,
                                 const float *bias,
                                 bool flag_bias,
                                 bool flag_relu,
                                 int num,
                                 int chin,
                                 int hin,
                                 int win,
                                 int hout,
                                 int wout,
                                 const operators::ConvParam &param,
                                 ARMContext *ctx) {
  auto act_param = param.activation_param;
  bool has_active = act_param.has_active;
  auto act_type = act_param.active_type;
  if (has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      conv_depthwise_5x5s1p2_fp32_relu(ACTUAL_PARAM, ctx);
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      conv_depthwise_5x5s1p2_fp32_relu6(
          ACTUAL_PARAM, act_param.Relu_clipped_coef, ctx);
    } else {
      LOG(FATAL) << "this act_type: " << static_cast<int>(act_type)
                 << " fuse not support";
    }
  } else {
    int size_in_channel = win * hin;
    int size_out_channel = wout * hout;
    int w_stride = 25;
    uint32_t vmask[12];
    auto&& res = right_mask_5x5s1p2_fp32(win, wout, vmask);
    uint32_t cnt_col = res.first;
    uint32_t cnt_remain = res.second;
#ifdef __aarch64__
    uint32_t right_pad_num = (cnt_remain == 8) ? 0 : ((8 - cnt_remain) * 4);
#else
    uint32_t right_pad_num = (cnt_remain == 4) ? 0 : ((4 - cnt_remain) * 4);
#endif
    float *zero_ptr = ctx->workspace_data<float>();
    memset(zero_ptr, 0, (win + 16) * sizeof(float));
    float *write_ptr = zero_ptr + win + 16;
    cnt_col = (cnt_col << 4) + cnt_remain;
    for (int n = 0; n < num; ++n) {
      const float *din_batch = din + n * chin * size_in_channel;
      float *dout_batch = dout + n * chin * size_out_channel;
      LITE_PARALLEL_BEGIN(c, tid, chin) {
        float *dout_ptr = dout_batch + c * size_out_channel;
        const float *din_ch_ptr = din_batch + c * size_in_channel;
        float bias_val = flag_bias ? bias[c] : 0.f;
        float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
        const float *wei_ptr = weights + c * w_stride;
        const float *dr0 = zero_ptr;
        const float *dr1 = zero_ptr;
        const float *dr2 = din_ch_ptr;
        const float *dr3 = dr2 + win;
        const float *dr4 = dr3 + win;
        const float *dr5 = dr4 + win;
#ifdef __aarch64__
        float32x4_t w0 = vld1q_f32(wei_ptr);
        float32x4_t w1 = vld1q_f32(wei_ptr + 4);
        float32x4_t w2 = vld1q_f32(wei_ptr + 8);
        float32x4_t w3 = vld1q_f32(wei_ptr + 12);
        float32x4_t w4 = vld1q_f32(wei_ptr + 16);
        float32x4_t w5 = vld1q_f32(wei_ptr + 20);
        float32x4_t w6 = vdupq_n_f32(wei_ptr[24]);
        float32x4_t vzero = vdupq_n_f32(0.f);
        for (int h = 0; h < hout; h += 2) {
          DIN_PTR_INIT
          int cnt = cnt_col;
          asm volatile(
            LEFT_COMPUTE_S1 LEFT_RESULT_S1
            MID_COMPITE_S1 MID_RESULT_S1
            RIGHT_COMPUTE_S1 RIGHT_RESULT_S1
            : [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), [din_ptr5] "+r"(din_ptr5),
              [doutr0] "+r"(doutr0), [doutr1] "+r"(doutr1), [cnt] "+r"(cnt)
            : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3), [w4] "w"(w4), [w5] "w"(w5),
              [w6] "w"(w6), [vzero] "w"(vzero), [bias_val] "r"(vbias),
              [right_pad_num] "r"(right_pad_num), [vmask] "r"(vmask)
            : "cc","memory", "v0","v1","v2","v3","v4","v5","v6","v7",
              "v8","v9","v10","v11","v12","v13", "v14","v15","v16","v17","v18","v19",
              "v28","v29","v30","v31"
          );
          dout_ptr += 2 * wout;
        }
#else
        for (int h = 0; h < hout; h++) {
          DIN_PTR_INIT
          int cnt = cnt_col;
          auto weight_ptr = wei_ptr;
          asm volatile(
            LEFT_COMPUTE_S1 LEFT_RESULT_S1
            MID_COMPITE_S1 MID_RESULT_S1
            RIGHT_COMPUTE_S1 RIGHT_RESULT_S1
            : [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4),
              [doutr0] "+r"(doutr0), [cnt] "+r"(cnt), [wei_ptr] "+r"(weight_ptr)
            : [bias_val] "r"(vbias), [right_pad_num] "r"(right_pad_num), [vmask] "r"(vmask)
            : "cc","memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
              "q7", "q8","q9","q10","q11","q12","q13", "q14","q15"
          );
          dout_ptr += wout;
        }
#endif
      }
      LITE_PARALLEL_END();
    }
  }
}

#undef LEFT_COMPUTE_S1
#undef LEFT_RESULT_S1
#undef LEFT_RESULT_S1_RELU
#undef LEFT_RESULT_S1_RELU6
#undef MID_COMPITE_S1
#undef MID_RESULT_S1
#undef MID_RESULT_S1_RELU
#undef MID_RESULT_S1_RELU6
#undef RIGHT_RESULT_S1
#undef RIGHT_RESULT_S1_RELU
#undef RIGHT_RESULT_S1_RELU6
#undef DIN_PTR_INIT
#undef IN_PARAM
#undef ACTUAL_PARAM
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
