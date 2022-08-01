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

template <typename Dtype>
void conv_3x3s1_direct_int8(const int8_t* din,
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

  const int hout_c_block = 4;
  const int hout_r_kernel = 2;
  const int wout_block = 4;
  const int wout_round = ((wout + wout_block - 1) / wout_block) * wout_block;
  const int win_round = wout_round + 2;

  //! get h block
  //! llc_size = win_round * chin * hin_r_block * sizeof(int8_t) + wout_round *
  //! hout_c_block * hout_r_block * threads * sizeof(int32_t)
  //! win_round = wout_round + 2
  //! hin_r_block = hout_r_block + 2
  int hout_r_block =
      (llc_size - 2 * win_round * chin) /
      (win_round * chin + hout_c_block * wout_round * threads * 4);
  hout_r_block = hout_r_block > hout ? hout : hout_r_block;
  hout_r_block =
      ((hout_r_block + hout_r_kernel - 1) / hout_r_kernel) * hout_r_kernel;
  hout_r_block = hout_r_block < hout_r_kernel ? hout_r_kernel : hout_r_block;

  const int hin_r_block = hout_r_block + 2;

  auto tmp_work_space = ctx->workspace_data<int8_t>();
  int8_t ptr_zero[win_round];  // NOLINT
  memset(ptr_zero, 0, sizeof(int8_t) * win_round);
  Dtype ptr_write[wout_round];  // NOLINT

  int in_len = win_round * chin;
  int pre_in_size = hin_r_block * in_len;
  pre_in_size = ROUNDUP(pre_in_size, 4);
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  int8_t* pre_din = tmp_work_space;

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = chin * 9;               // kernel_w * kernel_h;
  int w_stride_chin = hout_c_block * 9;  // kernel_w * kernel_h *

  int ws = -pad_w;
  int we = ws + win_round;
  int w_loop = wout_round / 4;

  int out_row_stride = hout_c_block * wout_round;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * chin * size_in_channel;
    Dtype* dout_batch = dout + n * chout * size_out_channel;
    for (int h = 0; h < hout; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > hout) {
        h_kernel = hout - h;
      }

      int hs = h - pad_h;
      int he = hs + h_kernel + 2;

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
                        ptr_zero);

      LITE_PARALLEL_COMMON_BEGIN(c, tid, chout, 0, hout_c_block) {
#ifdef LITE_USE_THREAD_POOL
        int32_t* pre_out =
            reinterpret_cast<int*>(pre_din + pre_in_size) + tid * pre_out_size;
#elif defined(ARM_WITH_OMP)
        int32_t* pre_out = reinterpret_cast<int*>(pre_din + pre_in_size) +
                           omp_get_thread_num() * pre_out_size;
#else
        auto pre_out = reinterpret_cast<int32_t*>(pre_din + pre_in_size);
#endif
        const int8_t* block_inr0 = pre_din;
        const int8_t* block_inr1 = block_inr0 + in_len;
        const int8_t* block_inr2 = block_inr1 + in_len;
        const int8_t* block_inr3 = block_inr2 + in_len;

        const int8_t* weight_c = weights + c * w_stride;
        memset(pre_out, 0, pre_out_size * sizeof(int32_t));
        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          const int8_t* wc0 = weight_c;

          const int8_t* inr0 = block_inr0;
          const int8_t* inr1 = block_inr1;
          const int8_t* inr2 = block_inr2;
          const int8_t* inr3 = block_inr3;

          int32_t* pre_out0 = pre_out + hk * out_row_stride;
          int32_t* pre_out1 = pre_out0 + out_row_stride;

          for (int i = 0; i < chin; ++i) {
            int32_t* ptr_out0 = pre_out0;
            int32_t* ptr_out1 = pre_out1;

            const signed char* r0 = inr0;
            const signed char* r1 = inr1;
            const signed char* r2 = inr2;
            const signed char* r3 = inr3;

            int cnt = w_loop;
            const int8_t* ptr_wc0 = wc0;
// clang-format off
#ifdef __aarch64__
            asm volatile(
                "ldp   q4, q5, [%[wc0]]\n"
                "ldr   d6, [%[wc0], #32]\n"
                "sxtl  v11.8h, v4.8b\n"
                "sxtl2 v12.8h, v4.16b\n"
                "sxtl  v13.8h, v5.8b\n"
                "sxtl2 v14.8h, v5.16b\n"
                "sxtl  v15.8h, v6.8b\n"
                "ldp    q16, q17,   [%[ptr_out0]]\n"
                "ldp    q18, q19,   [%[ptr_out0], #32]\n"
                "ldr  d0, [%[r1]], #4\n" /* load r1 */
                "ldr  d1, [%[r2]], #4\n" /* load r2 */
                "sxtl  v2.8h,  v0.8b\n"  /* r1, cvt to int16 */
                "sxtl  v3.8h,  v1.8b\n"  /* r2, cvt to int16 */
                "1:\n"
                /* inr1 -> outr0, outr1 */
                "ldp    q20, q21,   [%[ptr_out1]]\n"
                "ldr  d0, [%[r0]], #4\n"            /* load r0 */
                "smlal2  v16.4s, v12.8h, v2.h[0]\n" /* out00, w10 * r10 */
                "smlal2  v17.4s, v12.8h, v2.h[1]\n" /* out01, w10 * r11 */
                "smlal2  v18.4s, v12.8h, v2.h[2]\n" /* out02, w10 * r12 */
                "smlal2  v19.4s, v12.8h, v2.h[3]\n" /* out03, w10 * r13 */
                "ldp    q22, q23,   [%[ptr_out1], #32]\n"
                "smlal   v16.4s, v13.4h, v2.h[1]\n" /* out00, w11 * r11 */
                "smlal   v17.4s, v13.4h, v2.h[2]\n" /* out01, w11 * r12 */
                "smlal   v18.4s, v13.4h, v2.h[3]\n" /* out02, w11 * r13 */
                "smlal   v19.4s, v13.4h, v2.h[4]\n" /* out03, w11 * r14 */
                "smlal2  v16.4s, v13.8h, v2.h[2]\n" /* out00, w12 * r12 */
                "smlal2  v17.4s, v13.8h, v2.h[3]\n" /* out01, w12 * r13 */
                "smlal2  v18.4s, v13.8h, v2.h[4]\n" /* out02, w12 * r14 */
                "smlal2  v19.4s, v13.8h, v2.h[5]\n" /* out03, w12 * r15 */
                "smlal   v20.4s, v11.4h, v2.h[0]\n" /* out10, w00 * r10 */
                "smlal   v21.4s, v11.4h, v2.h[1]\n" /* out11, w00 * r11 */
                "smlal   v22.4s, v11.4h, v2.h[2]\n" /* out12, w00 * r12 */
                "smlal   v23.4s, v11.4h, v2.h[3]\n" /* out13, w00 * r13 */
                "smlal2  v20.4s, v11.8h, v2.h[1]\n" /* out10, w01 * r11 */
                "smlal2  v21.4s, v11.8h, v2.h[2]\n" /* out11, w01 * r12 */
                "smlal2  v22.4s, v11.8h, v2.h[3]\n" /* out12, w01 * r13 */
                "smlal2  v23.4s, v11.8h, v2.h[4]\n" /* out13, w01 * r14 */
                "smlal   v20.4s, v12.4h, v2.h[2]\n" /* out10, w02 * r12 */
                "smlal   v21.4s, v12.4h, v2.h[3]\n" /* out11, w02 * r13 */
                "smlal   v22.4s, v12.4h, v2.h[4]\n" /* out12, w02 * r14 */
                "smlal   v23.4s, v12.4h, v2.h[5]\n" /* out13, w02 * r15 */
                "sxtl  v2.8h,  v0.8b\n"             /* r0, cvt to int16 */
                /* inr2 -> outr0, outr1 */
                "ldr    d1, [%[r3]], #4\n"          /* load r3 */
                "smlal   v16.4s, v14.4h, v3.h[0]\n" /* out00, w20 * r20 */
                "smlal   v17.4s, v14.4h, v3.h[1]\n" /* out01, w20 * r21 */
                "smlal   v18.4s, v14.4h, v3.h[2]\n" /* out02, w20 * r22 */
                "smlal   v19.4s, v14.4h, v3.h[3]\n" /* out03, w20 * r23 */
                "smlal2  v20.4s, v12.8h, v3.h[0]\n" /* out10, w10 * r20 */
                "smlal2  v21.4s, v12.8h, v3.h[1]\n" /* out11, w10 * r21 */
                "smlal2  v22.4s, v12.8h, v3.h[2]\n" /* out12, w10 * r22 */
                "smlal2  v23.4s, v12.8h, v3.h[3]\n" /* out13, w10 * r23 */
                "smlal2  v16.4s, v14.8h, v3.h[1]\n" /* out00, w21 * r21 */
                "smlal2  v17.4s, v14.8h, v3.h[2]\n" /* out01, w21 * r22 */
                "smlal2  v18.4s, v14.8h, v3.h[3]\n" /* out02, w21 * r23 */
                "smlal2  v19.4s, v14.8h, v3.h[4]\n" /* out03, w21 * r24 */
                "smlal   v20.4s, v13.4h, v3.h[1]\n" /* out10, w11 * r21 */
                "smlal   v21.4s, v13.4h, v3.h[2]\n" /* out11, w11 * r22 */
                "smlal   v22.4s, v13.4h, v3.h[3]\n" /* out12, w11 * r23 */
                "smlal   v23.4s, v13.4h, v3.h[4]\n" /* out13, w11 * r24 */
                "smlal   v16.4s, v15.4h, v3.h[2]\n" /* out00, w22 * r22 */
                "smlal   v17.4s, v15.4h, v3.h[3]\n" /* out01, w22 * r23 */
                "smlal   v18.4s, v15.4h, v3.h[4]\n" /* out02, w22 * r24 */
                "smlal   v19.4s, v15.4h, v3.h[5]\n" /* out03, w22 * r25 */
                "smlal2  v20.4s, v13.8h, v3.h[2]\n" /* out10, w12 * r22 */
                "smlal2  v21.4s, v13.8h, v3.h[3]\n" /* out11, w12 * r23 */
                "smlal2  v22.4s, v13.8h, v3.h[4]\n" /* out12, w12 * r24 */
                "smlal2  v23.4s, v13.8h, v3.h[5]\n" /* out13, w12 * r25 */
                "sxtl  v3.8h,  v1.8b\n"             /* r0, cvt to int16 */
                /* inr0 -> outr0 */
                "ldr  d0, [%[r1]], #4\n"            /* load r1 */
                "smlal   v16.4s, v11.4h, v2.h[0]\n" /* out00, w00 * r00 */
                "smlal   v17.4s, v11.4h, v2.h[1]\n" /* out01, w00 * r01 */
                "smlal   v18.4s, v11.4h, v2.h[2]\n" /* out02, w00 * r02 */
                "smlal   v19.4s, v11.4h, v2.h[3]\n" /* out03, w00 * r03 */
                "smlal2  v16.4s, v11.8h, v2.h[1]\n" /* out00, w01 * r01 */
                "smlal2  v17.4s, v11.8h, v2.h[2]\n" /* out01, w01 * r02 */
                "smlal2  v18.4s, v11.8h, v2.h[3]\n" /* out02, w01 * r03 */
                "smlal2  v19.4s, v11.8h, v2.h[4]\n" /* out03, w01 * r04 */
                "smlal   v16.4s, v12.4h, v2.h[2]\n" /* out00, w02 * r02 */
                "smlal   v17.4s, v12.4h, v2.h[3]\n" /* out01, w02 * r03 */
                "smlal   v18.4s, v12.4h, v2.h[4]\n" /* out02, w02 * r04 */
                "smlal   v19.4s, v12.4h, v2.h[5]\n" /* out03, w02 * r05 */
                "sxtl  v2.8h,  v0.8b\n"             /* r0, cvt to int16 */
                /* inr3 -> outr1 */
                "ldr  d1, [%[r2]], #4\n" /* load r2 */
                "stp    q16, q17, [%[ptr_out0]], #32\n"
                "smlal   v20.4s, v14.4h, v3.h[0]\n" /* out10, w20 * r30 */
                "smlal   v21.4s, v14.4h, v3.h[1]\n" /* out11, w20 * r31 */
                "smlal   v22.4s, v14.4h, v3.h[2]\n" /* out12, w20 * r32 */
                "smlal   v23.4s, v14.4h, v3.h[3]\n" /* out13, w20 * r33 */
                "stp    q18, q19, [%[ptr_out0]], #32\n"
                "ldp    q16, q17,   [%[ptr_out0]]\n"
                "smlal2  v20.4s, v14.8h, v3.h[1]\n" /* out10, w21 * r31 */
                "smlal2  v21.4s, v14.8h, v3.h[2]\n" /* out11, w21 * r32 */
                "smlal2  v22.4s, v14.8h, v3.h[3]\n" /* out12, w21 * r33 */
                "smlal2  v23.4s, v14.8h, v3.h[4]\n" /* out13, w21 * r34 */
                "ldp    q18, q19,   [%[ptr_out0], #32]\n"
                "smlal   v20.4s, v15.4h, v3.h[2]\n" /* out10, w22 * r32 */
                "smlal   v21.4s, v15.4h, v3.h[3]\n" /* out11, w22 * r33 */
                "smlal   v22.4s, v15.4h, v3.h[4]\n" /* out12, w22 * r34 */
                "smlal   v23.4s, v15.4h, v3.h[5]\n" /* out13, w22 * r35 */
                "sxtl  v3.8h,  v1.8b\n"             /* r0, cvt to int16 */
                "subs    %w[cnt], %w[cnt], #1\n"
                "stp    q20, q21, [%[ptr_out1]], #32\n"
                "stp    q22, q23, [%[ptr_out1]], #32\n"
                "bne    1b\n"
                : [cnt] "+r"(cnt),
                  [wc0] "+r"(ptr_wc0),
                  [r0] "+r"(r0),
                  [r1] "+r"(r1),
                  [r2] "+r"(r2),
                  [r3] "+r"(r3),
                  [ptr_out0] "+r"(ptr_out0),
                  [ptr_out1] "+r"(ptr_out1)
                :
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4",
                  "v5", "v6", "v11", "v12", "v13", "v14", "v15",
                  "v16", "v17", "v18", "v19", "v20","v21", "v22", "v23"

                );

#else
            asm volatile(
                "vld1.32   {d0-d3}, [%[wc0]]!\n"
                "vld1.32   {d4}, [%[wc0]]!\n"
                "vmovl.s8   q3,   d0\n" /* q3 = w0, w1 */
                "vmovl.s8   q4,   d1\n" /* q4 = w2 ,w3 */
                "vmovl.s8   q5,   d2\n" /* q5 = w4, w5 */
                "vmovl.s8   q6,   d3\n" /* q6 = w6, w7 */
                "vmovl.s8   q7,   d4\n" /* q7 = w8 */
                "vld1.32    d0, [%[r1]]\n"
                "vld1.32    d1, [%[r2]]\n"
                "vld1.32    {d16-d19}, [%[ptr_out0]]!\n"
                "vld1.32    {d20-d23}, [%[ptr_out0]]\n"
                "vmovl.s8   q1,   d0\n"
                "vmovl.s8   q2,   d1\n"
                "1:\n"
                /* inr1 -> outr0, outr1 */
                "vld1.32    {d24-d27}, [%[ptr_out1]]!\n"
                "vld1.32  d0, [%[r0]]\n"      /* load r0 */
                "vmlal.s16  q8,  d9, d2[0]\n" /* out00, w10 * r10 */
                "vmlal.s16  q9,  d9, d2[1]\n" /* out01, w10 * r11 */
                "vmlal.s16  q10, d9, d2[2]\n" /* out02, w10 * r12 */
                "vmlal.s16  q11, d9, d2[3]\n" /* out03, w10 * r13 */
                "vld1.32    {d28-d31}, [%[ptr_out1]]\n"
                "vmlal.s16  q8, d10, d2[1]\n"  /* out00, w11 * r11 */
                "vmlal.s16  q9, d10, d2[2]\n"  /* out01, w11 * r12 */
                "vmlal.s16  q10, d10, d2[3]\n" /* out02, w11 * r13 */
                "vmlal.s16  q11, d10, d3[0]\n" /* out03, w11 * r14 */
                "sub    %[ptr_out0], %[ptr_out0], #32\n"
                "vmlal.s16  q8, d11, d2[2]\n"  /* out00, w12 * r12 */
                "vmlal.s16  q9, d11, d2[3]\n"  /* out01, w12 * r13 */
                "vmlal.s16  q10, d11, d3[0]\n" /* out02, w12 * r14 */
                "vmlal.s16  q11, d11, d3[1]\n" /* out03, w12 * r15 */
                "vmlal.s16  q12, d6, d2[0]\n"  /* out10, w00 * r10 */
                "vmlal.s16  q13, d6, d2[1]\n"  /* out11, w00 * r11 */
                "vmlal.s16  q14, d6, d2[2]\n"  /* out12, w00 * r12 */
                "vmlal.s16  q15, d6, d2[3]\n"  /* out13, w00 * r13 */
                "add    %[r1], %[r1], #4\n"
                "vmlal.s16  q12, d7, d2[1]\n" /* out10, w01 * r11 */
                "vmlal.s16  q13, d7, d2[2]\n" /* out11, w01 * r12 */
                "vmlal.s16  q14, d7, d2[3]\n" /* out12, w01 * r13 */
                "vmlal.s16  q15, d7, d3[0]\n" /* out13, w01 * r14 */
                "sub    %[ptr_out1], %[ptr_out1], #32\n"
                "vmlal.s16  q12, d8, d2[2]\n" /* out10, w02 * r12 */
                "vmlal.s16  q13, d8, d2[3]\n" /* out11, w02 * r13 */
                "vmlal.s16  q14, d8, d3[0]\n" /* out12, w02 * r14 */
                "vmlal.s16  q15, d8, d3[1]\n" /* out13, w02 * r15 */
                "vmovl.s8  q1,  d0\n"         /* r0, cvt to int16 */
                /* inr2 -> outr0, outr1 */
                "vld1.32    d1, [%[r3]]\n"     /* load r3 */
                "vmlal.s16  q8,  d12, d4[0]\n" /* out00, w20 * r20 */
                "vmlal.s16  q9,  d12, d4[1]\n" /* out01, w20 * r21 */
                "vmlal.s16  q10, d12, d4[2]\n" /* out02, w20 * r22 */
                "vmlal.s16  q11, d12, d4[3]\n" /* out03, w20 * r23 */
                "add    %[r2], %[r2], #4\n"
                "vmlal.s16  q12, d9, d4[0]\n"  /* out10, w10 * r20 */
                "vmlal.s16  q13, d9, d4[1]\n"  /* out11, w10 * r21 */
                "vmlal.s16  q14, d9, d4[2]\n"  /* out12, w10 * r22 */
                "vmlal.s16  q15, d9, d4[3]\n"  /* out13, w10 * r23 */
                "vmlal.s16  q8,  d13, d4[1]\n" /* out00, w21 * r21 */
                "vmlal.s16  q9,  d13, d4[2]\n" /* out01, w21 * r22 */
                "vmlal.s16  q10, d13, d4[3]\n" /* out02, w21 * r23 */
                "vmlal.s16  q11, d13, d5[0]\n" /* out03, w21 * r24 */
                "add    %[r0], %[r0], #4\n"
                "vmlal.s16  q12, d10, d4[1]\n" /* out10, w11 * r21 */
                "vmlal.s16  q13, d10, d4[2]\n" /* out11, w11 * r22 */
                "vmlal.s16  q14, d10, d4[3]\n" /* out12, w11 * r23 */
                "vmlal.s16  q15, d10, d5[0]\n" /* out13, w11 * r24 */
                "vmlal.s16  q8,  d14, d4[2]\n" /* out00, w22 * r22 */
                "vmlal.s16  q9,  d14, d4[3]\n" /* out01, w22 * r23 */
                "vmlal.s16  q10, d14, d5[0]\n" /* out02, w22 * r24 */
                "vmlal.s16  q11, d14, d5[1]\n" /* out03, w22 * r25 */
                "add    %[r3], %[r3], #4\n"
                "vmlal.s16  q12, d11, d4[2]\n" /* out10, w12 * r22 */
                "vmlal.s16  q13, d11, d4[3]\n" /* out11, w12 * r23 */
                "vmlal.s16  q14, d11, d5[0]\n" /* out12, w12 * r24 */
                "vmlal.s16  q15, d11, d5[1]\n" /* out13, w12 * r25 */
                "vmovl.s8  q2,  d1\n"          /* r3, cvt to int16 */
                /* inr0 -> outr0 */
                "vld1.32  d0, [%[r1]]\n"      /* load r1 */
                "vmlal.s16  q8,  d6, d2[0]\n" /* out00, w00 * r00 */
                "vmlal.s16  q9,  d6, d2[1]\n" /* out01, w00 * r01 */
                "vmlal.s16  q10, d6, d2[2]\n" /* out02, w00 * r02 */
                "vmlal.s16  q11, d6, d2[3]\n" /* out03, w00 * r03 */
                "vmlal.s16  q8,  d7, d2[1]\n" /* out00, w01 * r01 */
                "vmlal.s16  q9,  d7, d2[2]\n" /* out01, w01 * r02 */
                "vmlal.s16  q10, d7, d2[3]\n" /* out02, w01 * r03 */
                "vmlal.s16  q11, d7, d3[0]\n" /* out03, w01 * r04 */
                "vmlal.s16  q8,  d8, d2[2]\n" /* out00, w02 * r02 */
                "vmlal.s16  q9,  d8, d2[3]\n" /* out01, w02 * r03 */
                "vmlal.s16  q10, d8, d3[0]\n" /* out02, w02 * r04 */
                "vmlal.s16  q11, d8, d3[1]\n" /* out03, w02 * r05 */
                "vmovl.s8  q1,  d0\n"         /* r1, cvt to int16 */
                /* inr3 -> outr1 */
                "vld1.32  {d1}, [%[r2]]\n" /* load r2 */
                "vst1.32  {d16-d19}, [%[ptr_out0]]!\n"
                "vmlal.s16  q12, d12, d4[0]\n" /* out10, w20 * r30 */
                "vmlal.s16  q13, d12, d4[1]\n" /* out11, w20 * r31 */
                "vmlal.s16  q14, d12, d4[2]\n" /* out12, w20 * r32 */
                "vmlal.s16  q15, d12, d4[3]\n" /* out13, w20 * r33 */
                "vst1.32  {d20-d23}, [%[ptr_out0]]!\n"
                "vld1.32  {d16-d19}, [%[ptr_out0]]!\n"
                "vmlal.s16  q12, d13, d4[1]\n" /* out10, w21 * r31 */
                "vmlal.s16  q13, d13, d4[2]\n" /* out11, w21 * r32 */
                "vmlal.s16  q14, d13, d4[3]\n" /* out12, w21 * r33 */
                "vmlal.s16  q15, d13, d5[0]\n" /* out13, w21 * r34 */
                "vld1.32  {d20-d23}, [%[ptr_out0]]\n"
                "vmlal.s16  q12, d14, d4[2]\n" /* out10, w22 * r32 */
                "vmlal.s16  q13, d14, d4[3]\n" /* out11, w22 * r33 */
                "vmlal.s16  q14, d14, d5[0]\n" /* out12, w22 * r34 */
                "vmlal.s16  q15, d14, d5[1]\n" /* out13, w22 * r35 */
                "vmovl.s8  q2,  d1\n"          /* r2, cvt to int16 */
                "subs   %[cnt], #1\n"
                "vst1.32  {d24-d27}, [%[ptr_out1]]!\n"
                "vst1.32  {d28-d31}, [%[ptr_out1]]!\n"
                "bne    1b\n"
                : [cnt] "+r"(cnt),
                  [r0] "+r"(r0),
                  [r1] "+r"(r1),
                  [r2] "+r"(r2),
                  [r3] "+r"(r3),
                  [ptr_out0] "+r"(ptr_out0),
                  [ptr_out1] "+r"(ptr_out1),
                  [wc0] "+r"(ptr_wc0)
                :
                : "cc", "memory", "q0", "q1", "q2", "q3",
                  "q4", "q5", "q6", "q7", "q8", "q9", "q10",
                  "q11", "q12", "q13", "q14", "q15"
                );
#endif  // __aarch64__
            // clang-format on
            wc0 += 9 * hout_c_block;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
          }
          block_inr0 = block_inr2;
          block_inr1 = block_inr3;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
        }
        write_int32_nchwc4_to_nchw(pre_out,
                                   dout_batch,
                                   c,
                                   c + 4,
                                   h,
                                   h + hout_r_block,
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

template void conv_3x3s1_direct_int8(const int8_t* din,
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
                                     const float* scale);

template void conv_3x3s1_direct_int8(const int8_t* din,
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
                                     const float* scale);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
