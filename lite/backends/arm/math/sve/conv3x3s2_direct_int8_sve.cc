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

#include "lite/backends/arm/math/conv_block_utils.h"
#include "lite/backends/arm/math/conv_impl.h"
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
namespace sve {

#if 0
template <typename Dtype>
void conv_3x3s2_direct_int8(const int8_t* din,
                            Dtype* dout,
                            int num,
                            int chout,
                            int hout,
                            int wout,
                            int chin,
                            int hin,
                            int win,
                            const int8_t* weights,
                            const float* bias,
                            const operators::ConvParam& param,
                            Context<TARGET(kARM)>* ctx,
                            const float* scale) {
  //! 3x3s2 int8 convolution, implemented by direct algorithm
  //! prepack input to tmp buffer
  //! write output to tmp buffer
  auto paddings = *param.paddings;
  bool flag_bias = param.bias;
  auto act_param = param.activation_param;
  auto act_type = act_param.active_type;
  int flag_act = 0;  // relu: 1, relu6: 2, leakey: 3
  float alpha[12] = {
      0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (act_param.has_active) {
    if (act_type == lite_api::ActivationType::kRelu) {
      flag_act = 1;
    } else if (act_type == lite_api::ActivationType::kRelu6) {
      flag_act = 2;
      float local_alpha = act_param.Relu_clipped_coef;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kLeakyRelu) {
      flag_act = 3;
      float local_alpha = act_param.Leaky_relu_alpha;
      alpha[0] = local_alpha;
      alpha[1] = local_alpha;
      alpha[2] = local_alpha;
      alpha[3] = local_alpha;
    } else if (act_type == lite_api::ActivationType::kHardSwish) {
      flag_act = 4;
      for (int i = 0; i < 4; i++) {
        alpha[i] = 1.f / act_param.hard_swish_scale;
        alpha[i + 4] = act_param.hard_swish_offset;
        alpha[i + 8] = act_param.hard_swish_threshold;
      }
    }
  }
  int pad_h = paddings[0];
  int pad_w = paddings[2];

  const int threads = ctx->threads();
  int llc_size = ctx->llc_size() / 4;

  const int hout_c_block = 8;
  const int hout_r_kernel = 2;
  const int wout_round = ((wout + 3) / 4) * 4;
  const int win_round = wout_round * 2 /*stride_w*/ + 1;

  //! get h block
  //! win_round * chin * hin_r_block * sizeof(int8_t) + wout_round *
  //! hout_c_block * hout_r_block * threads * sizeof(int32_t)= l2_size
  //! win_round = 2 * wout_round + 1
  //! hin_r_block = 2 * hout_r_block + 1
  int hout_r_block =
      (llc_size - 2 * wout_round * chin - chin) /
      ((4 * wout_round + 2) * chin + wout_round * hout_c_block * threads * 4);
  hout_r_block = hout_r_block > hout ? hout : hout_r_block;
  hout_r_block = (hout_r_block / hout_r_kernel) * hout_r_kernel;
  hout_r_block = hout_r_block < hout_r_kernel ? hout_r_kernel : hout_r_block;

  const int hin_r_block = hout_r_block * 2 + 1;

  auto tmp_work_space = ctx->workspace_data<int8_t>();
  int zero_size = chout > (win_round + 3) / 4 ? chout : (win_round + 3) / 4;
  int32_t ptr_zero[zero_size];  // NOLINT
  memset(ptr_zero, 0, sizeof(int32_t) * zero_size);
  Dtype ptr_write[wout_round];  // NOLINT

  int in_len = win_round * chin;
  int pre_in_size = hin_r_block * in_len;
  pre_in_size = ROUNDUP(pre_in_size, 4);
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  //! l2_cache start
  int8_t* pre_din = tmp_work_space;

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = chin * 9;

  int ws = -pad_w;
  int we = ws + win_round;
  int w_loop = wout_round / 4;

  int out_row_stride = hout_c_block * wout_round;

  for (int n = 0; n < num; ++n) {
    auto din_batch = din + n * chin * size_in_channel;
    auto dout_batch = dout + n * chout * size_out_channel;
    for (int h = 0; h < hout; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > hout) {
        h_kernel = hout - h;
      }
      int hs = h * 2 - pad_h;
      int he = hs + h_kernel * 2 + 1;
      prepack_input_nxw(din_batch,
                        pre_din,
                        0,
                        chin,
                        hs,
                        he,
                        ws,
                        we,
                        chin,
                        win,
                        hin,
                        reinterpret_cast<int8_t*>(ptr_zero));

      const int8_t* cblock_inr0 = pre_din;
      const int8_t* cblock_inr1 = cblock_inr0 + in_len;
      const int8_t* cblock_inr2 = cblock_inr1 + in_len;
      const int8_t* cblock_inr3 = cblock_inr2 + in_len;
      const int8_t* cblock_inr4 = cblock_inr3 + in_len;

      LITE_PARALLEL_COMMON_BEGIN(c, tid, chout, 0, hout_c_block) {
#ifdef LITE_USE_THREAD_POOL
        auto pre_out =
            reinterpret_cast<int*>(pre_din + pre_in_size) + tid * pre_out_size;
#elif defined(ARM_WITH_OMP)
        auto pre_out = reinterpret_cast<int*>(pre_din + pre_in_size) +
                       omp_get_thread_num() * pre_out_size;
#else
        auto pre_out = reinterpret_cast<int32_t*>(pre_din + pre_in_size);
#endif
        const int8_t* block_inr0 = cblock_inr0;
        const int8_t* block_inr1 = cblock_inr1;
        const int8_t* block_inr2 = cblock_inr2;
        const int8_t* block_inr3 = cblock_inr3;
        const int8_t* block_inr4 = cblock_inr4;

        const int8_t* weight_c = weights + c * w_stride;
        memset(pre_out, 0, pre_out_size * sizeof(int32_t));
        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          const int8_t* wc0 = weight_c;

          const int8_t* inr0 = block_inr0;
          const int8_t* inr1 = block_inr1;
          const int8_t* inr2 = block_inr2;
          const int8_t* inr3 = block_inr3;
          const int8_t* inr4 = block_inr4;

          int32_t* pre_out0 = pre_out + hk * out_row_stride;
          int32_t* pre_out1 = pre_out0 + out_row_stride;
          for (int i = 0; i < chin; ++i) {
            int16x8_t v0 = vmovl_s8(vld1_s8(wc0));       // w0
            int16x8_t v1 = vmovl_s8(vld1_s8(wc0 + 8));   // w1
            int16x8_t v2 = vmovl_s8(vld1_s8(wc0 + 16));  // w2,

            int16x8_t v3 = vmovl_s8(vld1_s8(wc0 + 24));  // w3
            int16x8_t v4 = vmovl_s8(vld1_s8(wc0 + 32));  // w4
            int16x8_t v5 = vmovl_s8(vld1_s8(wc0 + 40));  // w5

            int16x8_t v6 = vmovl_s8(vld1_s8(wc0 + 48));  // w6
            int16x8_t v7 = vmovl_s8(vld1_s8(wc0 + 56));  // w7
            int16x8_t v8 = vmovl_s8(vld1_s8(wc0 + 64));  // w8

            const int8_t* r0 = inr0;
            const int8_t* r1 = inr1;
            const int8_t* r2 = inr2;
            const int8_t* r3 = inr3;
            const int8_t* r4 = inr4;

            int32_t* ptr_out0 = pre_out0;
            int32_t* ptr_out1 = pre_out1;
            int cnt = w_loop;
            // clang-format off
            asm volatile(
            "ldr    q0,    [%[r0]], #8  \n" /* load input r0 */
            "ldr    q1,    [%[r2]], #8  \n" /* load input r2 */
            "sshll  v0.8h, v0.8b, #0    \n" /*  r0: int8 -> int16 */
            "sshll  v1.8h, v1.8b, #0    \n" /*  r1: int8 -> int16*/
            "1:                         \n" /* main loop */
            /* r0, r2 mul w00 */
            "smull   v4.4s,   %[v0].4h,  v0.h[0]\n" /* outr00 = v0 * r0[0]*/
            "smull2  v5.4s,   %[v0].8h,  v0.h[0]\n" /* outr00 = v0 * r0[0]*/
            "smull   v6.4s,   %[v0].4h,  v0.h[2]\n" /* outr01 = v0 * r0[2]*/
            "smull2  v7.4s,   %[v0].8h,  v0.h[2]\n" /* outr00 = v0 * r0[0]*/
            "smull   v8.4s,   %[v0].4h,  v0.h[4]\n" /* outr02 = v0 * r0[4]*/
            "smull2  v9.4s,   %[v0].8h,  v0.h[4]\n" /* outr00 = v0 * r0[0]*/
            "smull   v10.4s,  %[v0].4h,  v0.h[6]\n" /* outr03 = v0 * r0[6]*/
            "smull2  v11.4s,  %[v0].8h,  v0.h[6]\n" /* outr00 = v0 * r0[0]*/
            "smull   v12.4s,  %[v0].4h,  v1.h[0]\n" /* outr10 = v0 * r2[0]*/
            "smull2  v13.4s,  %[v0].8h,  v1.h[0]\n" /* outr11 = v0 * r2[2]*/
            "smull   v14.4s,  %[v0].4h,  v1.h[2]\n" /* outr12 = v0 * r2[4]*/
            "smull2  v15.4s,  %[v0].8h,  v1.h[2]\n" /* outr13 = v0 * r2[6]*/
            "smull   v16.4s,  %[v0].4h,  v1.h[4]\n" /* outr10 = v0 * r2[0]*/
            "smull2  v17.4s,  %[v0].8h,  v1.h[4]\n" /* outr11 = v0 * r2[2]*/
            "smull   v18.4s,  %[v0].4h,  v1.h[6]\n" /* outr12 = v0 * r2[4]*/
            "smull2  v19.4s,  %[v0].8h,  v1.h[6]\n" /* outr13 = v0 * r2[6]*/
            /* r2, mul w06 */
            "smlal   v4.4s,   %[v6].4h,  v1.h[0]\n" /* outr00 = v6 * r2[1]*/
            "smlal2  v5.4s,   %[v6].8h,  v1.h[0]\n" /* outr01 = v6 * r2[3]*/
            "smlal   v6.4s,   %[v6].4h,  v1.h[2]\n" /* outr02 = v6 * r2[5]*/
            "smlal2  v7.4s,   %[v6].8h,  v1.h[2]\n" /* outr03 = v6 * r2[7]*/
            "smlal   v8.4s,   %[v6].4h,  v1.h[4]\n" /* outr00 = v6 * r2[1]*/
            "smlal2  v9.4s,   %[v6].8h,  v1.h[4]\n" /* outr01 = v6 * r2[3]*/
            "smlal   v10.4s,  %[v6].4h,  v1.h[6]\n" /* outr02 = v6 * r2[5]*/
            "smlal2  v11.4s,  %[v6].8h,  v1.h[6]\n" /* outr03 = v6 * r2[7]*/
            "ldr    q2,      [%[r0]]\n"     /* load r0, 9th data,v10.s[0] */
            /*  r0, r2, mul w01 */
            "smlal   v4.4s,   %[v1].4h,  v0.h[1]\n" /* outr00 = v0 * r0[0]*/
            "smlal2  v5.4s,   %[v1].8h,  v0.h[1]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v6.4s,   %[v1].4h,  v0.h[3]\n" /* outr01 = v0 * r0[2]*/
            "smlal2  v7.4s,   %[v1].8h,  v0.h[3]\n" /* outr00 = v0 * r0[0]*/
            "sshll   v2.8h,   v2.8b,     #0     \n" /*  r0: int8 -> int16 */
            "smlal   v8.4s,   %[v1].4h,  v0.h[5]\n" /* outr02 = v0 * r0[4]*/
            "smlal2  v9.4s,   %[v1].8h,  v0.h[5]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v10.4s,  %[v1].4h,  v0.h[7]\n" /* outr03 = v0 * r0[6]*/
            "smlal2  v11.4s,  %[v1].8h,  v0.h[7]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v12.4s,  %[v1].4h,  v1.h[1]\n" /* outr10 = v0 * r2[0]*/
            "smlal2  v13.4s,  %[v1].8h,  v1.h[1]\n" /* outr11 = v0 * r2[2]*/
            "smlal   v14.4s,  %[v1].4h,  v1.h[3]\n" /* outr12 = v0 * r2[4]*/
            "smlal2  v15.4s,  %[v1].8h,  v1.h[3]\n" /* outr13 = v0 * r2[6]*/
            "smlal   v16.4s,  %[v1].4h,  v1.h[5]\n" /* outr10 = v0 * r2[0]*/
            "smlal2  v17.4s,  %[v1].8h,  v1.h[5]\n" /* outr11 = v0 * r2[2]*/
            "smlal   v18.4s,  %[v1].4h,  v1.h[7]\n" /* outr12 = v0 * r2[4]*/
            "smlal2  v19.4s,  %[v1].8h,  v1.h[7]\n" /* outr13 = v0 * r2[6]*/
            /* r2, mul w07 */
            "smlal   v4.4s,   %[v7].4h,  v1.h[1]\n" /* outr00 = v6 * r2[1]*/
            "smlal2  v5.4s,   %[v7].8h,  v1.h[1]\n" /* outr01 = v6 * r2[3]*/
            "smlal   v6.4s,   %[v7].4h,  v1.h[3]\n" /* outr02 = v6 * r2[5]*/
            "smlal2  v7.4s,   %[v7].8h,  v1.h[3]\n" /* outr03 = v6 * r2[7]*/
            "smlal   v8.4s,   %[v7].4h,  v1.h[5]\n" /* outr00 = v6 * r2[1]*/
            "smlal2  v9.4s,   %[v7].8h,  v1.h[5]\n" /* outr01 = v6 * r2[3]*/
            "smlal   v10.4s,  %[v7].4h,  v1.h[7]\n" /* outr02 = v6 * r2[5]*/
            "smlal2  v11.4s,  %[v7].8h,  v1.h[7]\n" /* outr03 = v6 * r2[7]*/
            "ldr     q3,      [%[r2]]\n"    /* load r2, 9th data,v11.s[0] */
            /*  r0, r2, mul w02 */
            "smlal   v4.4s,   %[v2].4h,  v0.h[2]\n" /* outr00 = v0 * r0[0]*/
            "smlal2  v5.4s,   %[v2].8h,  v0.h[2]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v6.4s,   %[v2].4h,  v0.h[4]\n" /* outr01 = v0 * r0[2]*/
            "smlal2  v7.4s,   %[v2].8h,  v0.h[4]\n" /* outr00 = v0 * r0[0]*/
            "sshll   v3.8h,   v3.8b,     #0     \n" /* r2: int8 -> int16*/
            "smlal   v8.4s,   %[v2].4h,  v0.h[6]\n" /* outr02 = v0 * r0[4]*/
            "smlal2  v9.4s,   %[v2].8h,  v0.h[6]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v10.4s,  %[v2].4h,  v2.h[0]\n" /* outr03 = v0 * r0[6]*/
            "smlal2  v11.4s,  %[v2].8h,  v2.h[0]\n" /* outr00 = v0 * r0[0]*/
            "ldr     q0, [%[r1]], #8 \n"            /* load input r1 */
            "smlal   v12.4s,  %[v2].4h,  v1.h[2]\n" /* outr10 = v0 * r2[0]*/
            "smlal2  v13.4s,  %[v2].8h,  v1.h[2]\n" /* outr11 = v0 * r2[2]*/
            "smlal   v14.4s,  %[v2].4h,  v1.h[4]\n" /* outr12 = v0 * r2[4]*/
            "smlal2  v15.4s,  %[v2].8h,  v1.h[4]\n" /* outr13 = v0 * r2[6]*/
            "sshll   v0.8h,   v0.8b,     #0     \n" /* r1 : int8 -> int16 */
            "smlal   v16.4s,  %[v2].4h,  v1.h[6]\n" /* outr10 = v0 * r2[0]*/
            "smlal2  v17.4s,  %[v2].8h,  v1.h[6]\n" /* outr11 = v0 * r2[2]*/
            "smlal   v18.4s,  %[v2].4h,  v3.h[0]\n" /* outr12 = v0 * r2[4]*/
            "smlal2  v19.4s,  %[v2].8h,  v3.h[0]\n" /* outr13 = v0 * r2[6]*/
            /* r2, mul w08 */
            "smlal   v4.4s,   %[v8].4h,  v1.h[2]\n" /* outr00 = v6 * r2[1]*/
            "smlal2  v5.4s,   %[v8].8h,  v1.h[2]\n" /* outr01 = v6 * r2[3]*/
            "smlal   v6.4s,   %[v8].4h,  v1.h[4]\n" /* outr02 = v6 * r2[5]*/
            "smlal2  v7.4s,   %[v8].8h,  v1.h[4]\n" /* outr03 = v6 * r2[7]*/
            "smlal   v8.4s,   %[v8].4h,  v1.h[6]\n" /* outr00 = v6 * r2[1]*/
            "smlal2  v9.4s,   %[v8].8h,  v1.h[6]\n" /* outr01 = v6 * r2[3]*/
            "smlal   v10.4s,  %[v8].4h,  v3.h[0]\n" /* outr02 = v6 * r2[5]*/
            "smlal2  v11.4s,  %[v8].8h,  v3.h[0]\n" /* outr03 = v6 * r2[7]*/
            "ldr     q1, [%[r3]], #8 \n"            /* load input r3 */
            /*  r1, r3, mul w03 */
            "smlal   v4.4s,   %[v3].4h,  v0.h[0]\n" /* outr00 = v0 * r0[0]*/
            "smlal2  v5.4s,   %[v3].8h,  v0.h[0]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v6.4s,   %[v3].4h,  v0.h[2]\n" /* outr01 = v0 * r0[2]*/
            "smlal2  v7.4s,   %[v3].8h,  v0.h[2]\n" /* outr00 = v0 * r0[0]*/
            "sshll   v1.8h,   v1.8b,     #0     \n" /* r3: int8 -> int16 */
            "smlal   v8.4s,   %[v3].4h,  v0.h[4]\n" /* outr02 = v0 * r0[4]*/
            "smlal2  v9.4s,   %[v3].8h,  v0.h[4]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v10.4s,  %[v3].4h,  v0.h[6]\n" /* outr03 = v0 * r0[6]*/
            "smlal2  v11.4s,  %[v3].8h,  v0.h[6]\n" /* outr00 = v0 * r0[0]*/
            "ldr     q2,       [%[r1]]\n" /* load r1, 9th data,v10.s[0] */
            "smlal   v12.4s,  %[v3].4h,  v1.h[0]\n" /* outr10 = v0 * r2[0]*/
            "smlal2  v13.4s,  %[v3].8h,  v1.h[0]\n" /* outr11 = v0 * r2[2]*/
            "smlal   v14.4s,  %[v3].4h,  v1.h[2]\n" /* outr12 = v0 * r2[4]*/
            "smlal2  v15.4s,  %[v3].8h,  v1.h[2]\n" /* outr13 = v0 * r2[6]*/
            "ldr     q3,      [%[r3]]\n"  /* load r3, 9th data,v11.s[0] */
            "smlal   v16.4s,  %[v3].4h,  v1.h[4]\n" /* outr10 = v0 * r2[0]*/
            "smlal2  v17.4s,  %[v3].8h,  v1.h[4]\n" /* outr11 = v0 * r2[2]*/
            "smlal   v18.4s,  %[v3].4h,  v1.h[6]\n" /* outr12 = v0 * r2[4]*/
            "smlal2  v19.4s,  %[v3].8h,  v1.h[6]\n" /* outr13 = v0 * r2[6]*/
            "sshll v2.8h, v2.8b, #0 \n"             /* r1 : int8 -> int16 */
            /*  r1, r3, mul w05 */
            "smlal   v4.4s,   %[v5].4h,  v0.h[2]\n" /* outr00 = v0 * r0[0]*/
            "smlal2  v5.4s,   %[v5].8h,  v0.h[2]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v6.4s,   %[v5].4h,  v0.h[4]\n" /* outr01 = v0 * r0[2]*/
            "smlal2  v7.4s,   %[v5].8h,  v0.h[4]\n" /* outr00 = v0 * r0[0]*/
            "sshll   v3.8h,   v3.8b,     #0     \n" /* r3 : int8 -> int16 */
            "smlal   v8.4s,   %[v5].4h,  v0.h[6]\n" /* outr02 = v0 * r0[4]*/
            "smlal2  v9.4s,   %[v5].8h,  v0.h[6]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v10.4s,  %[v5].4h,  v2.h[0]\n" /* outr03 = v0 * r0[6]*/
            "smlal2  v11.4s,  %[v5].8h,  v2.h[0]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v12.4s,  %[v5].4h,  v1.h[2]\n" /* outr10 = v0 * r2[0]*/
            "smlal2  v13.4s,  %[v5].8h,  v1.h[2]\n" /* outr11 = v0 * r2[2]*/
            "smlal   v14.4s,  %[v5].4h,  v1.h[4]\n" /* outr12 = v0 * r2[4]*/
            "smlal2  v15.4s,  %[v5].8h,  v1.h[4]\n" /* outr13 = v0 * r2[6]*/
            "smlal   v16.4s,  %[v5].4h,  v1.h[6]\n" /* outr10 = v0 * r2[0]*/
            "smlal2  v17.4s,  %[v5].8h,  v1.h[6]\n" /* outr11 = v0 * r2[2]*/
            "smlal   v18.4s,  %[v5].4h,  v3.h[0]\n" /* outr12 = v0 * r2[4]*/
            "smlal2  v19.4s,  %[v5].8h,  v3.h[0]\n" /* outr13 = v0 * r2[6]*/
            "subs    %w[cnt], %w[cnt], #1       \n" /* loop count -1 */
            /*  r1, r3, mul w04 */
            "smlal   v4.4s,   %[v4].4h,  v0.h[1]\n" /* outr00 = v0 * r0[0]*/
            "smlal2  v5.4s,   %[v4].8h,  v0.h[1]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v6.4s,   %[v4].4h,  v0.h[3]\n" /* outr01 = v0 * r0[2]*/
            "smlal2  v7.4s,   %[v4].8h,  v0.h[3]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v8.4s,   %[v4].4h,  v0.h[5]\n" /* outr02 = v0 * r0[4]*/
            "smlal2  v9.4s,   %[v4].8h,  v0.h[5]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v10.4s,  %[v4].4h,  v0.h[7]\n" /* outr03 = v0 * r0[6]*/
            "smlal2  v11.4s,  %[v4].8h,  v0.h[7]\n" /* outr00 = v0 * r0[0]*/
            "ldr     q0, [%[r4]], #8            \n" /* load input r4 */
            "smlal   v12.4s,  %[v4].4h,  v1.h[1]\n" /* outr10 = v0 * r2[0]*/
            "smlal2  v13.4s,  %[v4].8h,  v1.h[1]\n" /* outr11 = v0 * r2[2]*/
            "smlal   v14.4s,  %[v4].4h,  v1.h[3]\n" /* outr12 = v0 * r2[4]*/
            "smlal2  v15.4s,  %[v4].8h,  v1.h[3]\n" /* outr13 = v0 * r2[6]*/
            "sshll   v0.8h,   v0.8b,     #0     \n" /* r4 : int8 -> int16 */
            "smlal   v16.4s,  %[v4].4h,  v1.h[5]\n" /* outr10 = v0 * r2[0]*/
            "smlal2  v17.4s,  %[v4].8h,  v1.h[5]\n" /* outr11 = v0 * r2[2]*/
            "smlal   v18.4s,  %[v4].4h,  v1.h[7]\n" /* outr12 = v0 * r2[4]*/
            "smlal2  v19.4s,  %[v4].8h,  v1.h[7]\n" /* outr13 = v0 * r2[6]*/
            "ldr     q2,      [%[r4]]\n" /* load r4, 9th data,v10.s[0] */
            "sshll   v2.8h,   v2.8b, #0\n" /* r4 : int8 -> int16 */
            "ldp     q1, q3, [%[ptr_out0]]\n"        /* load ptr_out */
            "ldp     q20, q21, [%[ptr_out0], #32]\n" /* load ptr_out */
            "add     v4.4s,  v1.4s ,  v4.4s\n"  /* v10 = outr00[0].low + q2 */
            "add     v5.4s,  v3.4s ,  v5.4s\n"  /* v11 = outr00[0].high+ q3 */
            "add     v6.4s,  v20.4s,  v6.4s\n"  /* v12 = outr01[0].low + q4 */
            "add     v7.4s,  v21.4s,  v7.4s\n"  /* v13 = outr01[0].high+ q5 */
            "ldp     q1 , q3 , [%[ptr_out0], #64]\n" /* load ptr_out*/
            "ldp     q20, q21, [%[ptr_out0], #96]\n" /* load ptr_out*/
            "stp     q4,  q5 , [%[ptr_out0]], #32\n" /* store q10, q11*/
            "stp     q6,  q7 , [%[ptr_out0]], #32\n" /* store q10, q11*/
            "add     v8.4s ,  v1.4s ,  v8.4s\n" /* v10 = outr00[0].low+ q2 */
            "add     v9.4s ,  v3.4s ,  v9.4s\n" /* v11 = outr00[0].high+q3 */
            "add     v10.4s,  v20.4s,  v10.4s\n" /* v12 = outr01[0].low+q4 */
            "add     v11.4s,  v21.4s,  v11.4s\n" /* v13 = outr01[0].high+q5 */
            "stp     q8,  q9,  [%[ptr_out0]], #32\n" /* store q14, q15*/
            "stp     q10, q11, [%[ptr_out0]], #32\n" /* store q16, q17*/
            /* r4, mul w08 */
            "smlal   v12.4s,   %[v8].4h,  v0.h[2]\n" /* outr00 = v0 * r0[0]*/
            "smlal2  v13.4s,   %[v8].8h,  v0.h[2]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v14.4s,   %[v8].4h,  v0.h[4]\n" /* outr01 = v0 * r0[2]*/
            "smlal2  v15.4s,   %[v8].8h,  v0.h[4]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v16.4s,   %[v8].4h,  v0.h[6]\n" /* outr02 = v0 * r0[4]*/
            "smlal2  v17.4s,   %[v8].8h,  v0.h[6]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v18.4s,   %[v8].4h,  v2.h[0]\n" /* outr03 = v0 * r0[6]*/
            "smlal2  v19.4s,   %[v8].8h,  v2.h[0]\n" /* outr00 = v0 * r0[0]*/
            /* r4, mul w07 */
            "smlal   v12.4s,   %[v7].4h,  v0.h[1]\n"  /* outr00 = v0 * r0[0]*/
            "smlal2  v13.4s,   %[v7].8h,  v0.h[1]\n"  /* outr00 = v0 * r0[0]*/
            "smlal   v14.4s,   %[v7].4h,  v0.h[3]\n"  /* outr01 = v0 * r0[2]*/
            "smlal2  v15.4s,   %[v7].8h,  v0.h[3]\n"  /* outr00 = v0 * r0[0]*/
            "ldr     q1,   [%[r2]], #8            \n" /* load input r2 */
            "smlal   v16.4s,   %[v7].4h,  v0.h[5]\n"  /* outr02 = v0 * r0[4]*/
            "smlal2  v17.4s,   %[v7].8h,  v0.h[5]\n"  /* outr00 = v0 * r0[0]*/
            "smlal   v18.4s,   %[v7].4h,  v0.h[7]\n"  /* outr03 = v0 * r0[6]*/
            "smlal2  v19.4s,   %[v7].8h,  v0.h[7]\n"  /* outr00 = v0 * r0[0]*/
            "sshll   v1.8h,    v1.8b,     #0     \n"  /*  r2: int8 -> int16*/
            /* r4, mul w06 */
            "ldp     q4,  q5,  [%[ptr_out1]]     \n" /* load ptr_out*/
            "smlal   v12.4s,   %[v6].4h,  v0.h[0]\n" /* outr00 = v0 * r0[0]*/
            "smlal2  v13.4s,   %[v6].8h,  v0.h[0]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v14.4s,   %[v6].4h,  v0.h[2]\n" /* outr01 = v0 * r0[2]*/
            "ldp     q8,  q9,  [%[ptr_out1], #64]\n" /* load ptr_out*/
            "smlal2  v15.4s,   %[v6].8h,  v0.h[2]\n" /* outr00 = v0 * r0[0]*/
            "smlal   v16.4s,   %[v6].4h,  v0.h[4]\n" /* outr02 = v0 * r0[4]*/
            "smlal2  v17.4s,   %[v6].8h,  v0.h[4]\n" /* outr00 = v0 * r0[0]*/
            "ldp     q10, q11, [%[ptr_out1], #96]\n" /* load ptr_out*/
            "smlal   v18.4s,   %[v6].4h,  v0.h[6]\n" /* outr03 = v0 * r0[6]*/
            "smlal2  v19.4s,   %[v6].8h,  v0.h[6]\n" /* outr00 = v0 * r0[0]*/
            "ldr     q0,   [%[r0]], #8           \n" /* load input r2 */
            "ldp     q6,   q7, [%[ptr_out1], #32]\n" /* load ptr_out*/
            "sshll   v0.8h, v0.8b, #0            \n" /* r0: int8 -> int16 */
            /* store outr1 */
            "add   v12.4s, v4.4s , v12.4s\n" /* v10 = outr10[0].low  + q2 */
            "add   v13.4s, v5.4s , v13.4s\n" /* v11 = outr10[0].high + q3 */
            "add   v14.4s, v6.4s , v14.4s\n" /* v12 = outr11[0].low  + q4 */
            "add   v15.4s, v7.4s , v15.4s\n" /* v13 = outr11[0].high + q5 */
            "stp   q12, q13, [%[ptr_out1]], #32\n" /* store q10, q11*/
            "add   v16.4s, v8.4s , v16.4s\n" /* v14 = outr12[0].low  + q6 */
            "add   v17.4s, v9.4s , v17.4s\n" /* v15 = outr12[0].high + q7 */
            "stp   q14, q15, [%[ptr_out1]], #32\n" /* store q12, q13*/
            "add   v18.4s, v10.4s, v18.4s\n" /* v16 = outr13[0].low  + q8 */
            "add   v19.4s, v11.4s, v19.4s\n" /* v17 = outr13[0].high + q9 */
            "stp   q16, q17, [%[ptr_out1]], #32\n" /* store q14, q15*/
            "stp   q18, q19, [%[ptr_out1]], #32\n" /* store q16, q17*/
            "bne     1b\n" /* jump to main loop */
            : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1),
              [r2] "+r"(r2), [r3] "+r"(r3), [r4] "+r"(r4),
              [ptr_out0] "+r"(ptr_out0), [ptr_out1] "+r"(ptr_out1)
            : [v0] "w"(v0), [v1] "w"(v1), [v2] "w"(v2),
              [v3] "w"(v3), [v4] "w"(v4), [v5] "w"(v5),
              [v6] "w"(v6), [v7] "w"(v7), [v8] "w"(v8)
            : "cc", "memory", "v0", "v1", "v2", "v3",
              "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16",
              "v17", "v18", "v19", "v20", "v21", "v22"
            );
            // clang-format on
            wc0 += 9 * hout_c_block;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
            inr4 += win_round;
          }
          block_inr0 = block_inr4;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }
        write_int32_nchwc8_to_nchw(pre_out,
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
                                   flag_act,
                                   alpha,
                                   bias + c,
                                   flag_bias,
                                   ptr_write,
                                   scale + c);
      }
      LITE_PARALLEL_COMMON_END();
    }
  }
}
#endif


template void conv_3x3s2_direct_int8_sve(const int8_t* din,
                                     float* dout,
                                     int num,
                                     int chout,
                                     int hout,
                                     int wout,
                                     int chin,
                                     int hin,
                                     int win,
                                     const int8_t* weights,
                                     const float* bias,
                                     const operators::ConvParam& param,
                                     Context<TARGET(kARM)>* ctx,
                                     const float* scale) {
  VLOG(1) << "conv_3x3s2_direct_int8_sve_float";
}

template void conv_3x3s2_direct_int8_sve(const int8_t* din,
                                     int8_t* dout,
                                     int num,
                                     int chout,
                                     int hout,
                                     int wout,
                                     int chin,
                                     int hin,
                                     int win,
                                     const int8_t* weights,
                                     const float* bias,
                                     const operators::ConvParam& param,
                                     Context<TARGET(kARM)>* ctx,
                                     const float* scale) {
  VLOG(1) << "conv_3x3s2_direct_int8_sve_float";
}

}  // namespace sve
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
