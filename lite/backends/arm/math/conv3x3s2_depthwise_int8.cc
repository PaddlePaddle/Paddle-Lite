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
#include "lite/backends/arm/math/conv_impl.h"
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

template <typename Dtype>
void conv_depthwise_3x3s2_common_int8(Dtype* dout,
                                      const int8_t* din,
                                      const int8_t* weights,
                                      const float* scale,
                                      const float* bias,
                                      bool flag_bias,
                                      int flag_act,
                                      float* alpha,
                                      int num,
                                      int chin,
                                      int hin,
                                      int win,
                                      int hout,
                                      int wout,
                                      int padw,
                                      int padh,
                                      ARMContext* ctx) {
  const int threads = ctx->threads();
  int llc_size = ctx->llc_size() / 4;

  const int hout_c_block = 8;
  const int hout_r_kernel = 1;
  const int wout_block = 4;
  const int wout_round = ROUNDUP(wout, wout_block);
  const int win_round = wout_round * 2 /*stride*/ + 1;

  //! get h block
  //! llc_size = threads * win_round * hin_r_block * hout_c_block *
  //! sizeof(int8_t)
  //!  + wout_round * hout_c_block * hout_r_block * threads * sizeof(int32_t)
  //! win_round = wout_round + 2
  //! hin_r_block = hout_r_block + 2
  int hout_r_block = (llc_size - 2 * win_round * threads * hout_c_block) /
                     (2 * win_round * threads * hout_c_block +
                      hout_c_block * wout_round * threads * 4);
  hout_r_block = hout_r_block > hout ? hout : hout_r_block;
  hout_r_block =
      ((hout_r_block + hout_r_kernel - 1) / hout_r_kernel) * hout_r_kernel;
  hout_r_block = hout_r_block < hout_r_kernel ? hout_r_kernel : hout_r_block;

  const int hin_r_block = hout_r_block * 2 /*stride*/ + 1;

  auto tmp_work_space = ctx->workspace_data<int8_t>();
  int8_t ptr_zero[win_round];  // NOLINT
  memset(ptr_zero, 0, sizeof(int8_t) * win_round);
  Dtype ptr_write[wout_round];  // NOLINT

  int in_len = win_round * hout_c_block;
  int pre_in_size = hin_r_block * in_len;
  pre_in_size = ROUNDUP(pre_in_size, 4);
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  int8_t* tmp_din = tmp_work_space;

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = 9;  // kernel_w * kernel_h;

  int ws = -padw;
  int we = ws + win_round;
  int w_loop = wout_round / 4;
  int chout = chin;

  int out_row_stride = hout_c_block * wout_round;

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * chin * size_in_channel;
    int8_t* dout_batch = reinterpret_cast<int8_t*>(dout) +
                         n * chout * size_out_channel * sizeof(Dtype);
    for (int h = 0; h < hout; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > hout) {
        h_kernel = hout - h;
      }
      int hs = h * 2 /*stride*/ - padh;
      int he = hs + h_kernel * 2 /*stride*/ + 1;

#pragma omp parallel for num_threads(threads)
      for (int c = 0; c < chout; c += hout_c_block) {
#ifdef ARM_WITH_OMP
        int8_t* pre_din =
            tmp_din + omp_get_thread_num() * (pre_in_size + pre_out_size * 4);
        int32_t* pre_out = reinterpret_cast<int*>(pre_din + pre_in_size);
#else
        int32_t* pre_out = reinterpret_cast<int32_t*>(tmp_din + pre_in_size);
        auto pre_din = tmp_din;
#endif
        prepack_input_nxwc8_int8_dw(
            din_batch, pre_din, c, hs, he, ws, we, chin, win, hin);
        const int8_t* block_inr0 = pre_din;
        const int8_t* block_inr1 = block_inr0 + in_len;
        const int8_t* block_inr2 = block_inr1 + in_len;

        const int8_t* weight_c = weights + c * w_stride;
        float bias_local[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (flag_bias) {
          bias_local[0] = bias[c];
          bias_local[1] = bias[c + 1];
          bias_local[2] = bias[c + 2];
          bias_local[3] = bias[c + 3];
          bias_local[4] = bias[c + 4];
          bias_local[5] = bias[c + 5];
          bias_local[6] = bias[c + 6];
          bias_local[7] = bias[c + 7];
        }
#ifdef __aarch64__
        int8x8_t vw0 = vld1_s8(weight_c);
        int8x8_t vw1 = vld1_s8(weight_c + 8);
        int8x8_t vw2 = vld1_s8(weight_c + 16);
        int8x8_t vw3 = vld1_s8(weight_c + 24);
        int8x8_t vw4 = vld1_s8(weight_c + 32);
        int8x8_t vw5 = vld1_s8(weight_c + 40);
        int8x8_t vw6 = vld1_s8(weight_c + 48);
        int8x8_t vw7 = vld1_s8(weight_c + 56);
        int8x8_t vw8 = vld1_s8(weight_c + 64);
#endif
        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          int cnt = w_loop;
          const int8_t* inr0 = block_inr0;
          const int8_t* inr1 = block_inr1;
          const int8_t* inr2 = block_inr2;
          int32_t* ptr_out0 = pre_out + hk * out_row_stride;
#ifdef __aarch64__
          asm volatile(
              "ld1  {v0.8b, v1.8b, v2.8b, v3.8b}, [%[r0]], #32\n"
              "ld1  {v4.8b, v5.8b, v6.8b, v7.8b}, [%[r0]], #32\n"
              "1:\n"
              /* inr0 -> outr0 */
              "smull v20.8h, v0.8b,  %[w0].8b\n" /* int16, out0 */
              "smull v21.8h, v2.8b,  %[w0].8b\n" /* int16, out1 */
              "smull v22.8h, v4.8b,  %[w0].8b\n" /* int16, out2 */
              "smull v23.8h, v6.8b,  %[w0].8b\n" /* int16, out3 */
              "smlal v20.8h, v1.8b,  %[w1].8b\n" /* int16, out0 */
              "smlal v21.8h, v3.8b,  %[w1].8b\n" /* int16, out1 */
              "smlal v22.8h, v5.8b,  %[w1].8b\n" /* int16, out2 */
              "smlal v23.8h, v7.8b,  %[w1].8b\n" /* int16, out3 */
              "ldr  d8, [%[r0]]\n"               /* load r0, 8 */
              "ldp  d0, d1, [%[r1]], #16\n"      /* load r1, 0,1 */
              "sxtl  v24.4s, v20.4h\n"
              "sxtl2 v25.4s, v20.8h\n"
              "smull v20.8h, v2.8b,  %[w2].8b\n" /* int16, out0 */
              "ldp  d2, d3, [%[r1]], #16\n"      /* load r1, 2,3 */
              "sxtl  v26.4s, v21.4h\n"
              "sxtl2 v27.4s, v21.8h\n"
              "smull v21.8h, v4.8b,  %[w2].8b\n" /* int16, out1 */
              "ldp  d4, d5, [%[r1]], #16\n"      /* load r1, 4,5 */
              "sxtl  v28.4s, v22.4h\n"
              "sxtl2 v29.4s, v22.8h\n"
              "smull v22.8h, v6.8b,  %[w2].8b\n" /* int16, out2 */
              "ldp  d6, d7, [%[r1]], #16\n"      /* load r1, 6,7 */
              "sxtl  v30.4s, v23.4h\n"
              "sxtl2 v31.4s, v23.8h\n"
              "smull v23.8h, v8.8b,  %[w2].8b\n" /* int16, out3 */
              "smlal v20.8h, v0.8b,  %[w3].8b\n" /* int16, out0 */
              "smlal v21.8h, v2.8b,  %[w3].8b\n" /* int16, out1 */
              "smlal v22.8h, v4.8b,  %[w3].8b\n" /* int16, out2 */
              "smlal v23.8h, v6.8b,  %[w3].8b\n" /* int16, out3 */
              "saddw  v24.4s, v24.4s, v20.4h\n"
              "saddw2 v25.4s, v25.4s, v20.8h\n"
              "saddw  v26.4s, v26.4s, v21.4h\n"
              "saddw2 v27.4s, v27.4s, v21.8h\n"
              "ldr  d8, [%[r1]]\n" /* load r1, 8 */
              "saddw  v28.4s, v28.4s, v22.4h\n"
              "saddw2 v29.4s, v29.4s, v22.8h\n"
              "saddw  v30.4s, v30.4s, v23.4h\n"
              "saddw2 v31.4s, v31.4s, v23.8h\n"
              "smull v20.8h, v1.8b,  %[w4].8b\n" /* int16, out0 */
              "smull v21.8h, v3.8b,  %[w4].8b\n" /* int16, out1 */
              "smull v22.8h, v5.8b,  %[w4].8b\n" /* int16, out1 */
              "smull v23.8h, v7.8b,  %[w4].8b\n" /* int16, out1 */
              "ldp  d0, d1, [%[r2]], #16\n"      /* load r2, 0,1 */
              "smlal v20.8h, v2.8b,  %[w5].8b\n" /* int16, out0 */
              "smlal v21.8h, v4.8b,  %[w5].8b\n" /* int16, out1 */
              "ldp  d2, d3, [%[r2]], #16\n"      /* load r2, 2,3 */
              "smlal v22.8h, v6.8b,  %[w5].8b\n" /* int16, out2 */
              "smlal v23.8h, v8.8b,  %[w5].8b\n" /* int16, out3 */
              "ldp  d4, d5, [%[r2]], #16\n"      /* load r2, 4,5 */
              "saddw  v24.4s, v24.4s, v20.4h\n"
              "saddw2 v25.4s, v25.4s, v20.8h\n"
              "saddw  v26.4s, v26.4s, v21.4h\n"
              "saddw2 v27.4s, v27.4s, v21.8h\n"
              "ldp  d6, d7, [%[r2]], #16\n" /* load r2, 6,7 */
              "saddw  v28.4s, v28.4s, v22.4h\n"
              "saddw2 v29.4s, v29.4s, v22.8h\n"
              "saddw  v30.4s, v30.4s, v23.4h\n"
              "saddw2 v31.4s, v31.4s, v23.8h\n"
              "smull v20.8h, v0.8b,  %[w6].8b\n" /* int16, out0 */
              "smull v21.8h, v2.8b,  %[w6].8b\n" /* int16, out1 */
              "smull v22.8h, v4.8b,  %[w6].8b\n" /* int16, out1 */
              "smull v23.8h, v6.8b,  %[w6].8b\n" /* int16, out1 */
              "smlal v20.8h, v1.8b,  %[w7].8b\n" /* int16, out0 */
              "smlal v21.8h, v3.8b,  %[w7].8b\n" /* int16, out1 */
              "smlal v22.8h, v5.8b,  %[w7].8b\n" /* int16, out1 */
              "smlal v23.8h, v7.8b,  %[w7].8b\n" /* int16, out1 */
              "ldp  d0, d1, [%[r0]], #16\n"      /* load r0, 0,1 */
              "saddw  v24.4s, v24.4s, v20.4h\n"
              "saddw2 v25.4s, v25.4s, v20.8h\n"
              "saddw  v26.4s, v26.4s, v21.4h\n"
              "saddw2 v27.4s, v27.4s, v21.8h\n"
              "ldr  d8, [%[r2]]\n" /* load r2 */
              "saddw  v28.4s, v28.4s, v22.4h\n"
              "saddw2 v29.4s, v29.4s, v22.8h\n"
              "saddw  v30.4s, v30.4s, v23.4h\n"
              "saddw2 v31.4s, v31.4s, v23.8h\n"
              "smull v20.8h, v2.8b,  %[w8].8b\n" /* int16, out0 */
              "smull v21.8h, v4.8b,  %[w8].8b\n" /* int16, out1 */
              "ldp  d2, d3, [%[r0]], #16\n"      /* load r0, 2,3 */
              "smull v22.8h, v6.8b,  %[w8].8b\n" /* int16, out1 */
              "smull v23.8h, v8.8b,  %[w8].8b\n" /* int16, out1 */
              "ldp  d4, d5, [%[r0]], #16\n"      /* load r0, 5 */
              "saddw  v24.4s, v24.4s, v20.4h\n"
              "saddw2 v25.4s, v25.4s, v20.8h\n"
              "saddw  v26.4s, v26.4s, v21.4h\n"
              "saddw2 v27.4s, v27.4s, v21.8h\n"
              "ldp  d6, d7, [%[r0]], #16\n" /* load r0, 6 */
              "stp    q24, q25, [%[ptr_out0]], #32\n"
              "saddw  v28.4s, v28.4s, v22.4h\n"
              "saddw2 v29.4s, v29.4s, v22.8h\n"
              "stp    q26, q27, [%[ptr_out0]], #32\n"
              "saddw  v30.4s, v30.4s, v23.4h\n"
              "saddw2 v31.4s, v31.4s, v23.8h\n"
              "subs    %w[cnt], %w[cnt], #1\n"
              "stp    q28, q29, [%[ptr_out0]], #32\n"
              "stp    q30, q31, [%[ptr_out0]], #32\n"
              "bne    1b\n"
              : [cnt] "+r"(cnt),
                [r0] "+r"(inr0),
                [r1] "+r"(inr1),
                [r2] "+r"(inr2),
                [ptr_out0] "+r"(ptr_out0)
              : [w0] "w"(vw0),
                [w1] "w"(vw1),
                [w2] "w"(vw2),
                [w3] "w"(vw3),
                [w4] "w"(vw4),
                [w5] "w"(vw5),
                [w6] "w"(vw6),
                [w7] "w"(vw7),
                [w8] "w"(vw8)
              : "cc",
                "memory",
                "v0",
                "v1",
                "v2",
                "v3",
                "v4",
                "v5",
                "v6",
                "v7",
                "v8",
                "v20",
                "v21",
                "v22",
                "v23",
                "v24",
                "v25",
                "v26",
                "v27",
                "v28",
                "v29",
                "v30",
                "v31"

              );
#else
          auto wptr = weight_c;
          asm volatile(
              "vld1.32    {d0-d3}, [%[r0]]!\n"   /* load r0, 0-3 */
              "vld1.32    {d6-d7}, [%[wptr]]!\n" /* load w0-w1 */
              "vld1.32    {d4-d5}, [%[r0]]!\n"   /* load r0, 4-5 */
              "1:\n"
              /* inr0 -> outr0 */
              "vmull.s8 q4, d1,  d7\n"      /* int16, out0 */
              "vld1.32    {d1}, [%[r0]]!\n" /* load r0, 6 */
              "vmull.s8 q5, d3,  d7\n"      /* int16, out1 */
              "vld1.32    {d3}, [%[r0]]!\n" /* load r0, 7 */
              "vmull.s8 q6, d5,  d7\n"      /* int16, out2 */
              "vld1.32    {d5}, [%[r0]]\n"  /* load r0, 8 */
              "vmull.s8 q7, d1,  d6\n"      /* int16, out0 */
              "vmlal.s8 q4, d0,  d6\n"      /* int16, out3 */
              "vmlal.s8 q5, d2,  d6\n"      /* int16, out1 */
              "vmlal.s8 q6, d4,  d6\n"      /* int16, out2 */
              "vmlal.s8 q7, d3,  d7\n"      /* int16, out3 */
              "vmovl.s16  q8, d8\n"
              "vmovl.s16  q9, d9\n"
              "vld1.32    {d6-d7}, [%[wptr]]!\n" /* load w2-w3 */
              "vmovl.s16  q10, d10\n"
              "vmovl.s16  q11, d11\n"
              "vmovl.s16  q12, d12\n"
              "vmovl.s16  q13, d13\n"
              "vmovl.s16  q14, d14\n"
              "vmovl.s16  q15, d15\n"
              "vmull.s8 q4, d2,  d6\n"         /* int16, out0 */
              "vmull.s8 q6, d1,  d6\n"         /* int16, out2 */
              "vld1.32    {d0-d3}, [%[r1]]!\n" /* load r1, 0-3 */
              "vmull.s8 q5, d4,  d6\n"         /* int16, out1 */
              "vmull.s8 q7, d5,  d6\n"         /* int16, out3 */
              "vld1.32    {d4-d5}, [%[r1]]!\n" /* load r1, 4,5 */
              /* inr1 -> outr0 */
              "vmlal.s8 q4, d0,  d7\n"        /* int16, out0 */
              "vld1.32    {d0},   [%[r1]]!\n" /* load r1, 6 */
              "vmlal.s8 q5, d2,  d7\n"        /* int16, out1 */
              "vmlal.s8 q6, d4,  d7\n"        /* int16, out2 */
              "vaddw.s16  q8, q8, d8\n"
              "vaddw.s16  q9, q9, d9\n"
              "vmlal.s8 q7, d0,  d7\n" /* int16, out3 */
              "vaddw.s16  q10, q10, d10\n"
              "vaddw.s16  q11, q11, d11\n"
              "vld1.32    {d6-d7}, [%[wptr]]!\n" /* load w4-w5 */
              "vaddw.s16  q12, q12, d12\n"
              "vaddw.s16  q13, q13, d13\n"
              "vaddw.s16  q14, q14, d14\n"
              "vaddw.s16  q15, q15, d15\n"
              "vmull.s8 q4, d1,  d6\n"         /* int16, out0 */
              "vld1.32    {d1},   [%[r1]]!\n"  /* load r1, 7 */
              "vmull.s8 q5, d3,  d6\n"         /* int16, out1 */
              "vld1.32    {d3},   [%[r1]]\n"   /* load r1, 8 */
              "vmull.s8 q6, d5,  d6\n"         /* int16, out2 */
              "vmull.s8 q7, d1,  d6\n"         /* int16, out3 */
              "vmlal.s8 q4, d2,  d7\n"         /* int16, out0 */
              "vmlal.s8 q5, d4,  d7\n"         /* int16, out2 */
              "vmlal.s8 q6, d0,  d7\n"         /* int16, out1 */
              "vmlal.s8 q7, d3,  d7\n"         /* int16, out3 */
              "vld1.32    {d0-d3}, [%[r2]]!\n" /* load r2, 0-3 */
              "vaddw.s16  q8, q8, d8\n"
              "vaddw.s16  q9, q9, d9\n"
              "vaddw.s16  q10, q10, d10\n"
              "vaddw.s16  q11, q11, d11\n"
              "vld1.32    {d6-d7}, [%[wptr]]!\n" /* load w6-w7 */
              "vaddw.s16  q12, q12, d12\n"
              "vaddw.s16  q13, q13, d13\n"
              "vaddw.s16  q14, q14, d14\n"
              "vaddw.s16  q15, q15, d15\n"
              "vld1.32    {d4-d5}, [%[r2]]!\n" /* load r2, 4-5 */
              /* inr2 -> outr0 */
              "vmull.s8 q4, d1,  d7\n"          /* int16, out0 */
              "vld1.32    {d1},   [%[r2]]!\n"   /* load r2, 6 */
              "vmull.s8 q5, d3,  d7\n"          /* int16, out1 */
              "vld1.32    {d3},   [%[r2]]!\n"   /* load r2, 7 */
              "vmull.s8 q6, d5,  d7\n"          /* int16, out2 */
              "vld1.32    {d5},   [%[r2]]\n"    /* load r2, 8 */
              "vmull.s8 q7, d1,  d6\n"          /* int16, out3 */
              "vmlal.s8 q4, d0,  d6\n"          /* int16, out0 */
              "vmlal.s8 q5, d2,  d6\n"          /* int16, out1 */
              "vmlal.s8 q6, d4,  d6\n"          /* int16, out2 */
              "vmlal.s8 q7, d3,  d7\n"          /* int16, out3 */
              "vld1.32    {d6},   [%[wptr]]!\n" /* load w8 */
              "vaddw.s16  q8, q8, d8\n"
              "vaddw.s16  q9, q9, d9\n"
              "vaddw.s16  q10, q10, d10\n"
              "vaddw.s16  q11, q11, d11\n"
              "vaddw.s16  q12, q12, d12\n"
              "vaddw.s16  q13, q13, d13\n"
              "vaddw.s16  q14, q14, d14\n"
              "vaddw.s16  q15, q15, d15\n"
              "sub %[wptr],   %[wptr],    #72\n"
              "vmull.s8 q4, d2,  d6\n"         /* int16, out0 */
              "vmull.s8 q5, d4,  d6\n"         /* int16, out1 */
              "vmull.s8 q6, d1,  d6\n"         /* int16, out2 */
              "vmull.s8 q7, d5,  d6\n"         /* int16, out3 */
              "vld1.32    {d0-d3}, [%[r0]]!\n" /* load r0, 0-3 */
              "vaddw.s16  q8, q8, d8\n"
              "vaddw.s16  q9, q9, d9\n"
              "vld1.32    {d4-d5}, [%[r0]]!\n" /* load r0, 4-5 */
              "vaddw.s16  q10, q10, d10\n"
              "vaddw.s16  q11, q11, d11\n"
              "vst1.32    {d16-d19},  [%[ptr_out0]]!\n"
              "vld1.32    {d6-d7},   [%[wptr]]!\n" /* load w0-w1 */
              "vaddw.s16  q12, q12, d12\n"
              "vaddw.s16  q13, q13, d13\n"
              "vst1.32    {d20-d23},  [%[ptr_out0]]!\n"
              "vaddw.s16  q14, q14, d14\n"
              "vaddw.s16  q15, q15, d15\n"
              "subs    %[cnt], #1\n"
              "vst1.32    {d24-d27},  [%[ptr_out0]]!\n"
              "vst1.32    {d28-d31},  [%[ptr_out0]]!\n"
              "bne    1b\n"
              : [cnt] "+r"(cnt),
                [r0] "+r"(inr0),
                [r1] "+r"(inr1),
                [r2] "+r"(inr2),
                [ptr_out0] "+r"(ptr_out0),
                [wptr] "+r"(wptr)
              :
              : "cc",
                "memory",
                "q0",
                "q1",
                "q2",
                "q3",
                "q4",
                "q5",
                "q6",
                "q7",
                "q8",
                "q9",
                "q10",
                "q11",
                "q12",
                "q13",
                "q14",
                "q15");
#endif
          block_inr0 = block_inr2;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
        }
        write_int32_nchwc8_to_nchw<Dtype>(pre_out,
                                          reinterpret_cast<Dtype*>(dout_batch),
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
                                          bias_local,
                                          flag_bias,
                                          ptr_write,
                                          scale + c);
      }
    }
  }
}
#define FILL_WEIGHTS_BIAS_INT8(weight_ptr, bias_val) \
  int8x8_t wr00 = vdup_n_s8(weight_ptr[0]);          \
  int8x8_t wr10 = vdup_n_s8(weight_ptr[3]);          \
  int8x8_t wr20 = vdup_n_s8(weight_ptr[6]);          \
  int8x8_t wr01 = vdup_n_s8(weight_ptr[1]);          \
  int8x8_t wr11 = vdup_n_s8(weight_ptr[4]);          \
  int8x8_t wr21 = vdup_n_s8(weight_ptr[7]);          \
  int8x8_t wr02 = vdup_n_s8(weight_ptr[2]);          \
  int8x8_t wr12 = vdup_n_s8(weight_ptr[5]);          \
  int8x8_t wr22 = vdup_n_s8(weight_ptr[8]);          \
  float v_bias[8] = {bias_val,                       \
                     bias_val,                       \
                     bias_val,                       \
                     bias_val,                       \
                     bias_val,                       \
                     bias_val,                       \
                     bias_val,                       \
                     bias_val};

#define INIT_PTR_3x3_S2_INT8(Dtype, din, w_in) \
  Dtype* doutr0 = nullptr;                     \
  Dtype* doutr1 = nullptr;                     \
  const int8_t* dr0 = din;                     \
  const int8_t* dr1 = dr0 + w_in;              \
  const int8_t* dr2 = dr1 + w_in;              \
  const int8_t* dr3 = dr2 + w_in;              \
  const int8_t* dr4 = dr3 + w_in;              \
  const int8_t* din_ptr0 = nullptr;            \
  const int8_t* din_ptr1 = nullptr;            \
  const int8_t* din_ptr2 = nullptr;            \
  const int8_t* din_ptr3 = nullptr;            \
  const int8_t* din_ptr4 = nullptr;

#define ASSIGN_PTR_3x3_S2_INT8(w_out) \
  din_ptr0 = dr0;                     \
  din_ptr1 = dr1;                     \
  din_ptr2 = dr2;                     \
  din_ptr3 = dr3;                     \
  din_ptr4 = dr4;                     \
  doutr0 = dout_ptr;                  \
  doutr1 = doutr0 + w_out;

#define TOP_BOTTOM_BORDER_3x3_S2P1_INT8(w_in, h_in, h_out) \
  if (i == 0) {                                            \
    din_ptr0 = zero_ptr;                                   \
    din_ptr1 = dr0;                                        \
    din_ptr2 = dr1;                                        \
    din_ptr3 = dr2;                                        \
    din_ptr4 = dr3;                                        \
    dr0 = dr3;                                             \
    dr1 = dr4;                                             \
  } else {                                                 \
    dr0 = dr4;                                             \
    dr1 = dr0 + w_in;                                      \
  }                                                        \
  dr2 = dr1 + w_in;                                        \
  dr3 = dr2 + w_in;                                        \
  dr4 = dr3 + w_in;                                        \
  if (i + 4 > h_in) {                                      \
    switch (i + 4 - h_in) {                                \
      case 4:                                              \
        din_ptr1 = zero_ptr;                               \
      case 3:                                              \
        din_ptr2 = zero_ptr;                               \
      case 2:                                              \
        din_ptr3 = zero_ptr;                               \
      case 1:                                              \
        din_ptr4 = zero_ptr;                               \
      default:                                             \
        break;                                             \
    }                                                      \
  }                                                        \
  if (i / 2 + 2 > h_out) {                                 \
    doutr1 = write_ptr;                                    \
  }

#define TOP_BOTTOM_BORDER_3x3_S2P0_INT8(w_in, h_in, h_out) \
  dr0 = dr4;                                               \
  dr1 = dr0 + w_in;                                        \
  dr2 = dr1 + w_in;                                        \
  dr3 = dr2 + w_in;                                        \
  dr4 = dr3 + w_in;                                        \
  if (i * 2 + 5 > h_in) {                                  \
    switch (i * 2 + 5 - h_in) {                            \
      case 4:                                              \
        din_ptr1 = zero_ptr;                               \
      case 3:                                              \
        din_ptr2 = zero_ptr;                               \
      case 2:                                              \
        din_ptr3 = zero_ptr;                               \
      case 1:                                              \
        din_ptr4 = zero_ptr;                               \
      case 0:                                              \
        din_ptr4 = zero_ptr;                               \
      default:                                             \
        break;                                             \
    }                                                      \
  }                                                        \
  if (i + 2 > h_out) {                                     \
    doutr1 = write_ptr;                                    \
  }

inline std::pair<uint32_t, uint32_t> right_mask_3x3s2p1_int8(int w_in,
                                                             int w_out,
                                                             uint8_t* vmask) {
  const uint8_t right_pad_idx[16] = {
      0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
  uint32_t cnt_col = ((w_out >> 3) - 2);
  uint8_t size_right_remain = static_cast<uint8_t>(w_in - (15 + cnt_col * 16));
  if (size_right_remain >= 17) {
    cnt_col++;
    size_right_remain -= 16;
  }
  uint32_t cnt_remain = (size_right_remain == 16 && w_out % 8 == 0)
                            ? 8
                            : static_cast<uint32_t>(w_out % 8);
  uint8x8_t vmask_rp1 =
      vcgt_u8(vdup_n_u8(size_right_remain), vld1_u8(right_pad_idx));
  uint8x8_t vmask_rp2 =
      vcgt_u8(vdup_n_u8(size_right_remain), vld1_u8(right_pad_idx + 8));
  vst1_u8(vmask, vmask_rp1);
  vst1_u8(vmask + 8, vmask_rp2);
  return std::make_pair(cnt_col, cnt_remain);
}
#ifdef __aarch64__
#define INIT_INT8_S2                                    \
  "PRFM PLDL1KEEP, [%[din_ptr0]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr1]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr2]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr3]]                    \n" \
  "PRFM PLDL1KEEP, [%[din_ptr4]]                    \n" \
  "ld2    {v0.8b, v1.8b}, [%[din_ptr0]], #16        \n" \
  "ld2    {v4.8b, v5.8b}, [%[din_ptr2]], #16        \n" \
  "ld2    {v2.8b, v3.8b}, [%[din_ptr1]], #16        \n" \
  "ld2    {v6.8b, v7.8b}, [%[din_ptr3]], #16        \n" \
  "ld2    {v8.8b, v9.8b}, [%[din_ptr4]], #16        \n"

#define LEFT_COMPUTE_INT8_S2                                              \
  "movi    v16.4s, #0                               \n"                   \
  "movi    v17.4s, #0                               \n"                   \
  "ext    v10.8b, %[vzero].8b, v1.8b, #7            \n"                   \
  "ext    v11.8b, %[vzero].8b, v5.8b, #7            \n"                   \
  "smull  v12.8h, v0.8b, %[wr01].8b                 \n"                   \
  "smull  v14.8h, v4.8b, %[wr01].8b                 \n"                   \
  "smull  v13.8h, v1.8b, %[wr02].8b                 \n"                   \
  "smull  v15.8h, v5.8b, %[wr02].8b                 \n"                   \
  "movi    v18.4s, #0                               \n"                   \
  "movi    v19.4s, #0                               \n"                   \
  "sub    %[din_ptr0], %[din_ptr0], #1              \n"                   \
  "sub    %[din_ptr2], %[din_ptr2], #1              \n"                   \
  "smlal  v12.8h, v10.8b, %[wr00].8b                \n"                   \
  "smlal  v14.8h, v11.8b, %[wr00].8b                \n" /* line 2 */      \
  "smlal  v13.8h, v4.8b, %[wr21].8b                 \n"                   \
  "saddw  v16.4s, v16.4s, v12.4h                    \n"                   \
  "saddw2 v17.4s, v17.4s, v12.8h                    \n"                   \
  "saddw  v18.4s, v18.4s, v14.4h                    \n"                   \
  "saddw2  v19.4s, v19.4s, v14.8h                   \n"                   \
  "smull  v12.8h, v5.8b, %[wr22].8b                 \n"                   \
  "saddw  v16.4s, v16.4s, v13.4h                    \n"                   \
  "saddw2 v17.4s, v17.4s, v13.8h                    \n"                   \
  "smull  v13.8h, v11.8b, %[wr20].8b                \n" /* line 1 */      \
  "ext    v10.8b, %[vzero].8b, v3.8b, #7            \n"                   \
  "ext    v11.8b, %[vzero].8b, v7.8b, #7            \n"                   \
  "smlal  v15.8h, v6.8b, %[wr11].8b                 \n"                   \
  "smlal  v12.8h, v2.8b, %[wr11].8b                 \n"                   \
  "smull  v14.8h, v7.8b, %[wr12].8b                 \n"                   \
  "smlal  v13.8h, v3.8b, %[wr12].8b                 \n"                   \
  "sub    %[din_ptr1], %[din_ptr1], #1              \n"                   \
  "sub    %[din_ptr3], %[din_ptr3], #1              \n"                   \
  "saddw  v18.4s, v18.4s, v15.4h                    \n"                   \
  "saddw2 v19.4s, v19.4s, v15.8h                    \n"                   \
  "saddw  v16.4s, v16.4s, v12.4h                    \n"                   \
  "saddw2 v17.4s, v17.4s, v12.8h                    \n"                   \
  "smlal  v14.8h, v11.8b, %[wr10].8b                \n"                   \
  "smull  v12.8h, v10.8b, %[wr10].8b                \n"                   \
  "saddw  v16.4s, v16.4s, v13.4h                    \n"                   \
  "saddw2 v17.4s, v17.4s, v13.8h                    \n" /* line 2 */      \
  "ext    v11.8b, %[vzero].8b, v9.8b, #7            \n"                   \
  "saddw  v18.4s, v18.4s, v14.4h                    \n"                   \
  "saddw2 v19.4s, v19.4s, v14.8h                    \n"                   \
  "saddw  v16.4s, v16.4s, v12.4h                    \n"                   \
  "saddw2 v17.4s, v17.4s, v12.8h                    \n"                   \
  "smull  v15.8h, v9.8b, %[wr22].8b                 \n"                   \
  "smull  v14.8h, v8.8b, %[wr21].8b                 \n"                   \
  "sub    %[din_ptr4], %[din_ptr4], #1              \n"                   \
  "ld2    {v0.8b, v1.8b}, [%[din_ptr0]], #16        \n"                   \
  "ld2    {v4.8b, v5.8b}, [%[din_ptr2]], #16        \n"                   \
  "saddw  v18.4s, v18.4s, v15.4h                    \n"                   \
  "saddw2  v19.4s, v19.4s, v15.8h                   \n"                   \
  "smlal  v14.8h, v11.8b, %[wr20].8b                \n"                   \
  "ld2    {v2.8b, v3.8b}, [%[din_ptr1]], #16        \n"                   \
  "ld2    {v6.8b, v7.8b}, [%[din_ptr3]], #16        \n"                   \
  "ld2    {v8.8b, v9.8b}, [%[din_ptr4]], #16        \n"                   \
  "saddw  v18.4s, v18.4s, v14.4h                    \n"                   \
  "saddw2  v19.4s, v19.4s, v14.8h                   \n" /* int32->fp32 */ \
  "ld1     {v14.4s}, [%[scale_val]]                 \n"                   \
  "scvtf   v10.4s, v16.4s                           \n"                   \
  "scvtf   v11.4s, v17.4s                           \n"                   \
  "ld1    {v16.4s}, [%[bias_val]]                   \n"                   \
  "ld1    {v17.4s}, [%[bias_val]]                   \n"                   \
  "scvtf   v12.4s, v18.4s                           \n"                   \
  "scvtf   v13.4s, v19.4s                           \n"                   \
  "ld1    {v18.4s}, [%[bias_val]]                   \n"                   \
  "ld1    {v19.4s}, [%[bias_val]]                   \n"                   \
  "fmla   v16.4s, v10.4s, v14.4s                    \n"                   \
  "fmla   v17.4s, v11.4s, v14.4s                    \n"                   \
  "fmla   v18.4s, v12.4s, v14.4s                    \n"                   \
  "fmla   v19.4s, v13.4s, v14.4s                    \n"                   \
  "cmp    %w[cnt], #1                               \n"

#define RESULT_INT8_S2_RELU                             \
  "fmax   v16.4s,  v16.4s, %[vzero].4s              \n" \
  "fmax   v17.4s,  v17.4s, %[vzero].4s              \n" \
  "fmax   v18.4s,  v18.4s, %[vzero].4s              \n" \
  "fmax   v19.4s,  v19.4s, %[vzero].4s              \n"

#define RESULT_INT8_S2_RELU6                            \
  "ld1    {v14.4s}, [%[alpha_val]]                  \n" \
  "fmin   v16.4s,  v16.4s, v14.4s                   \n" \
  "fmin   v17.4s,  v17.4s, v14.4s                   \n" \
  "fmin   v18.4s,  v18.4s, v14.4s                   \n" \
  "fmin   v19.4s,  v19.4s, v14.4s                   \n"

#define RESULT_INT8_S2_LEAKY_RELU                       \
  "ld1    {v14.4s}, [%[alpha_val]]                  \n" \
  "fcmge  v10.4s,  v16.4s, %[vzero].4s              \n" \
  "fmul   v20.4s,  v16.4s, v14.4s                   \n" \
  "fcmge  v11.4s,  v17.4s, %[vzero].4s              \n" \
  "fmul   v21.4s,  v17.4s, v14.4s                   \n" \
  "bif    v16.16b, v20.16b, v10.16b                 \n" \
  "bif    v17.16b, v21.16b, v11.16b                 \n" \
  "fcmge  v10.4s,  v18.4s, %[vzero].4s              \n" \
  "fmul   v20.4s,  v18.4s, v14.4s                   \n" \
  "fcmge  v11.4s,  v19.4s, %[vzero].4s              \n" \
  "fmul   v21.4s,  v19.4s, v14.4s                   \n" \
  "bif    v18.16b, v20.16b, v10.16b                 \n" \
  "bif    v19.16b, v21.16b, v11.16b                 \n"

#define RESULT_INT8_INT8_S2                                                \
  /*fp32->mul scale->int32->int16->int8*/                                  \
  "ld1 {v14.4s}, [%[max_val]]                       \n" /* data >= -127 */ \
  "fcmge v10.4s, v16.4s, v14.4s                     \n"                    \
  "fcmge v11.4s, v17.4s, v14.4s                     \n"                    \
  "fcmge v12.4s, v18.4s, v14.4s                     \n"                    \
  "fcmge v13.4s, v19.4s, v14.4s                     \n" /* choose data */  \
  "bif v16.16b, v14.16b, v10.16b                    \n"                    \
  "bif v17.16b, v14.16b, v11.16b                    \n"                    \
  "bif v18.16b, v14.16b, v12.16b                    \n"                    \
  "bif v19.16b, v14.16b, v13.16b                    \n" /* fp32 - int32 */ \
  "fcvtas  v10.4s, v16.4s                           \n"                    \
  "fcvtas  v11.4s, v17.4s                           \n"                    \
  "fcvtas  v12.4s, v18.4s                           \n"                    \
  "fcvtas  v13.4s, v19.4s                           \n" /* int32-int16 */  \
  "sqxtn   v16.4h, v10.4s                           \n"                    \
  "sqxtn2  v16.8h, v11.4s                           \n"                    \
  "sqxtn   v18.4h, v12.4s                           \n"                    \
  "sqxtn2  v18.8h, v13.4s                           \n" /* int16-int8 */   \
  "sqxtn  v17.8b, v16.8h                            \n"                    \
  "sqxtn  v19.8b, v18.8h                            \n"                    \
  "st1    {v17.8b}, [%[ptr_out0]], #8               \n"                    \
  "st1    {v19.8b}, [%[ptr_out1]], #8               \n"

#define RESULT_INT8_FP32_S2                             \
  "st1    {v16.4s}, [%[ptr_out0]], #16              \n" \
  "st1    {v18.4s}, [%[ptr_out1]], #16              \n" \
  "st1    {v17.4s}, [%[ptr_out0]], #16              \n" \
  "st1    {v19.4s}, [%[ptr_out1]], #16              \n"

#define MID_COMPUTE_INT8_S2                                               \
  "blt    1f                                        \n"                   \
  "2:                                               \n"                   \
  "ld1    {v12.8b}, [%[din_ptr0]]                   \n"                   \
  "ld1    {v13.8b}, [%[din_ptr2]]                   \n"                   \
  "movi    v18.4s, #0                               \n"                   \
  "movi    v19.4s, #0                               \n"                   \
  "movi    v20.4s, #0                               \n"                   \
  "movi    v21.4s, #0                               \n"                   \
  "ext    v10.8b, v0.8b, v12.8b, #1                \n"                    \
  "ext    v11.8b, v4.8b, v13.8b, #1                \n"                    \
  "smull  v14.8h, v0.8b, %[wr00].8b                 \n"                   \
  "smull  v16.8h, v4.8b, %[wr00].8b                 \n"                   \
  "smull  v15.8h, v1.8b, %[wr01].8b                 \n"                   \
  "smull  v17.8h, v5.8b, %[wr01].8b                 \n"                   \
  "ld1    {v12.8h}, [%[din_ptr1]]                   \n"                   \
  "ld1    {v13.8h}, [%[din_ptr3]]                   \n"                   \
  "smlal  v14.8h, v10.8b, %[wr02].8b                \n"                   \
  "smlal  v16.8h, v11.8b, %[wr02].8b                \n"                   \
  "smlal  v15.8h, v4.8b, %[wr20].8b                 \n"                   \
  "saddw  v18.4s,  v18.4s, v14.4h                   \n"                   \
  "saddw2 v19.4s,  v19.4s, v14.8h                   \n"                   \
  "saddw  v20.4s,  v20.4s, v16.4h                   \n"                   \
  "saddw2 v21.4s,  v21.4s, v16.8h                   \n"                   \
  "ext    v10.8b, v2.8b, v12.8b, #1                \n"                    \
  "saddw  v18.4s,  v18.4s, v15.4h                   \n"                   \
  "saddw2 v19.4s,  v19.4s, v15.8h                   \n"                   \
  "smull  v14.8h, v5.8b, %[wr21].8b                 \n"                   \
  "smull  v15.8h, v11.8b, %[wr22].8b                \n"                   \
  "ext    v11.8b, v6.8b, v13.8b, #1                \n" /* line 1 */       \
  "smlal  v17.8h,  v6.8b,  %[wr10].8b               \n"                   \
  "smlal  v14.8h,  v2.8b,  %[wr10].8b               \n"                   \
  "smull  v16.8h,  v7.8b,  %[wr11].8b               \n"                   \
  "smlal  v15.8h,  v3.8b,  %[wr11].8b               \n"                   \
  "saddw  v20.4s,  v20.4s, v17.4h                   \n"                   \
  "saddw2 v21.4s,  v21.4s, v17.8h                   \n"                   \
  "saddw  v18.4s,  v18.4s, v14.4h                   \n"                   \
  "saddw2 v19.4s,  v19.4s, v14.8h                   \n"                   \
  "ld1    {v12.8h}, [%[din_ptr4]]                   \n"                   \
  "smull  v14.8h,  v10.8b,  %[wr12].8b              \n"                   \
  "smlal  v16.8h,  v11.8b,  %[wr12].8b              \n" /* line 2 */      \
  "saddw  v18.4s,  v18.4s, v15.4h                   \n"                   \
  "saddw2 v19.4s,  v19.4s, v15.8h                   \n"                   \
  "ext    v11.8b, v8.8b, v12.8b, #1                \n"                    \
  "smull  v17.8h,  v8.8b,  %[wr20].8b               \n"                   \
  "saddw  v20.4s,  v20.4s, v16.4h                   \n"                   \
  "saddw2 v21.4s,  v21.4s, v16.8h                   \n"                   \
  "saddw  v18.4s,  v18.4s, v14.4h                   \n"                   \
  "saddw2 v19.4s,  v19.4s, v14.8h                   \n"                   \
  "smull  v16.8h,  v9.8b,  %[wr21].8b               \n"                   \
  "smlal  v17.8h,  v11.8b,  %[wr22].8b              \n"                   \
  "ld2    {v0.8b, v1.8b}, [%[din_ptr0]], #16        \n"                   \
  "ld2    {v4.8b, v5.8b}, [%[din_ptr2]], #16        \n"                   \
  "saddw  v20.4s,  v20.4s, v16.4h                   \n"                   \
  "saddw2 v21.4s,  v21.4s, v16.8h                   \n"                   \
  "ld2    {v2.8b, v3.8b}, [%[din_ptr1]], #16        \n"                   \
  "ld2    {v6.8b, v7.8b}, [%[din_ptr3]], #16        \n"                   \
  "saddw  v20.4s,  v20.4s, v17.4h                   \n"                   \
  "saddw2 v21.4s,  v21.4s, v17.8h                   \n"                   \
  "ld2    {v8.8b, v9.8b}, [%[din_ptr4]], #16        \n" /* int32->fp32 */ \
  "ld1     {v14.4s}, [%[scale_val]]                 \n"                   \
  "scvtf   v10.4s, v18.4s                           \n"                   \
  "scvtf   v11.4s, v19.4s                           \n"                   \
  "ld1    {v16.4s}, [%[bias_val]]                   \n"                   \
  "ld1    {v17.4s}, [%[bias_val]]                   \n"                   \
  "scvtf   v12.4s, v20.4s                           \n"                   \
  "scvtf   v13.4s, v21.4s                           \n"                   \
  "ld1    {v18.4s}, [%[bias_val]]                   \n"                   \
  "ld1    {v19.4s}, [%[bias_val]]                   \n"                   \
  "fmla   v16.4s, v10.4s, v14.4s                    \n"                   \
  "fmla   v17.4s, v11.4s, v14.4s                    \n"                   \
  "fmla   v18.4s, v12.4s, v14.4s                    \n"                   \
  "fmla   v19.4s, v13.4s, v14.4s                    \n"                   \
  "subs   %w[cnt], %w[cnt], #1                      \n"

#define RIGHT_COMPUTE_INT8_S2                                        \
  "bne    2b                                        \n"              \
  "1:                                               \n"              \
  "cmp    %w[remain], #1                            \n"              \
  "blt    4f                                        \n"              \
  "3:                                               \n"              \
  "ld1    {v12.8b}, [%[vmask]], #8                  \n"              \
  "ld1    {v13.8b}, [%[vmask]]                      \n"              \
  "movi    v18.4s, #0                               \n"              \
  "movi    v19.4s, #0                               \n"              \
  "bif    v0.8b, %[vzero].8b, v12.8b                \n"              \
  "bif    v1.8b, %[vzero].8b, v13.8b                \n"              \
  "bif    v4.8b, %[vzero].8b, v12.8b                \n"              \
  "bif    v5.8b, %[vzero].8b, v13.8b                \n"              \
  "bif    v2.8b, %[vzero].8b, v12.8b                \n"              \
  "bif    v3.8b, %[vzero].8b, v13.8b                \n"              \
  "ext    v10.8b, v0.8b, %[vzero].8b, #1            \n"              \
  "ext    v11.8b, v4.8b, %[vzero].8b, #1            \n"              \
  "bif    v6.16b, %[vzero].16b, v12.16b             \n"              \
  "bif    v7.16b, %[vzero].16b, v13.16b             \n"              \
  "movi    v20.4s, #0                               \n"              \
  "movi    v21.4s, #0                               \n"              \
  "smull  v14.8h, v0.8b, %[wr00].8b                 \n"              \
  "smull  v16.8h, v4.8b, %[wr00].8b                 \n"              \
  "smull  v15.8h, v1.8b, %[wr01].8b                 \n"              \
  "smull  v17.8h, v5.8b, %[wr01].8b                 \n"              \
  "smlal  v14.8h, v10.8b, %[wr02].8b                \n"              \
  "smlal  v16.8h, v11.8b, %[wr02].8b                \n"              \
  "smlal  v15.8h, v4.8b, %[wr20].8b                 \n"              \
  "saddw  v18.4s, v18.4s, v14.4h                    \n"              \
  "saddw2 v19.4s, v19.4s, v14.8h                    \n"              \
  "saddw  v20.4s, v20.4s, v16.4h                    \n"              \
  "saddw2 v21.4s, v21.4s, v16.8h                    \n"              \
  "bif    v8.16b, %[vzero].16b, v12.16b             \n"              \
  "bif    v9.16b, %[vzero].16b, v13.16b             \n"              \
  "saddw  v18.4s, v18.4s, v15.4h                    \n"              \
  "saddw2 v19.4s, v19.4s, v15.8h                    \n"              \
  "smull  v14.8h, v5.8b, %[wr21].8b                 \n"              \
  "smull  v15.8h, v11.8b, %[wr22].8b                \n"              \
  "ext    v10.8b, v2.8b, %[vzero].8b, #1            \n"              \
  "ext    v11.8b, v6.8b, %[vzero].8b, #1            \n" /* line 1 */ \
  "smlal  v17.8h, v6.8b, %[wr10].8b                 \n"              \
  "smlal  v14.8h, v2.8b, %[wr10].8b                 \n"              \
  "smull  v16.8h, v7.8b, %[wr11].8b                 \n"              \
  "smlal  v15.8h, v3.8b, %[wr11].8b                 \n"              \
  "saddw  v20.4s, v20.4s, v17.4h                    \n"              \
  "saddw2 v21.4s, v21.4s, v17.8h                    \n"              \
  "saddw  v18.4s, v18.4s, v14.4h                    \n"              \
  "saddw2 v19.4s, v19.4s, v14.8h                    \n"              \
  "smlal v16.8h, v11.8b, %[wr12].8b                 \n"              \
  "smull v14.8h, v10.8b, %[wr12].8b                 \n"              \
  "saddw  v18.4s, v18.4s, v15.4h                    \n"              \
  "saddw2 v19.4s, v19.4s, v15.8h                    \n" /* line 2 */ \
  "ext    v11.8b, v8.8b, %[vzero].8b, #1         \n"                 \
  "saddw  v20.4s, v20.4s, v16.4h                    \n"              \
  "saddw2 v21.4s, v21.4s, v16.8h                    \n"              \
  "smull  v17.8h, v8.8b, %[wr20].8b                 \n"              \
  "smull  v16.8h, v9.8b, %[wr21].8b                 \n"              \
  "saddw  v20.4s, v20.4s, v16.4h                    \n"              \
  "saddw2 v21.4s, v21.4s, v16.8h                    \n"              \
  "smlal  v17.8h, v11.8b, %[wr22].8b                \n"              \
  "scvtf   v10.4s, v18.4s                           \n"              \
  "scvtf   v11.4s, v19.4s                           \n"              \
  "saddw  v20.4s, v20.4s, v17.4h                    \n"              \
  "saddw2 v21.4s, v21.4s, v17.8h                    \n"

#define RIGHT_RESULT_INT8_FP32_S2                                         \
  "ld1     {v4.4s}, [%[scale_val]]                  \n"                   \
  "ldp    q12, q13, [%[rmask]]                      \n" /* int32->fp32 */ \
  "ld1    {v16.4s}, [%[bias_val]]                   \n"                   \
  "ld1    {v17.4s}, [%[bias_val]]                   \n"                   \
  "scvtf   v14.4s, v20.4s                           \n"                   \
  "ldp    q0, q1,   [%[ptr_out0]]                   \n"                   \
  "scvtf   v15.4s, v21.4s                           \n"                   \
  "ld1    {v18.4s}, [%[bias_val]]                   \n"                   \
  "ld1    {v19.4s}, [%[bias_val]]                   \n"                   \
  "ldp    q2, q3,   [%[ptr_out1]]                   \n"                   \
  "fmla   v16.4s, v10.4s, v4.4s                     \n"                   \
  "fmla   v17.4s, v11.4s, v4.4s                     \n"                   \
  "fmla   v18.4s, v14.4s, v4.4s                     \n"                   \
  "fmla   v19.4s, v15.4s, v4.4s                     \n"

#define RIGHT_RESULT_INT8_FP32_ST                       \
  "bif    v16.16b, v0.16b, v12.16b                  \n" \
  "bif    v17.16b, v1.16b, v13.16b                  \n" \
  "bif    v18.16b, v2.16b, v12.16b                  \n" \
  "bif    v19.16b, v3.16b, v13.16b                  \n" \
  "st1    {v16.4s}, [%[ptr_out0]], #16              \n" \
  "st1    {v18.4s}, [%[ptr_out1]], #16              \n" \
  "st1    {v17.4s}, [%[ptr_out0]], #16              \n" \
  "st1    {v19.4s}, [%[ptr_out1]], #16              \n" \
  "4:                                               \n"

#define RIGHT_RESULT_INT8_INT8_S2                                         \
  "ld1     {v4.4s}, [%[scale_val]]                  \n" /* int32->fp32 */ \
  "ld1    {v16.4s}, [%[bias_val]]                   \n"                   \
  "ld1    {v17.4s}, [%[bias_val]]                   \n"                   \
  "scvtf   v14.4s, v20.4s                           \n"                   \
  "scvtf   v15.4s, v21.4s                           \n"                   \
  "ld1 {v0.4s}, [%[max_val]]                       \n"                    \
  "ld1    {v18.4s}, [%[bias_val]]                   \n"                   \
  "ld1    {v19.4s}, [%[bias_val]]                   \n"                   \
  "fmla   v16.4s, v10.4s, v4.4s                     \n"                   \
  "fmla   v17.4s, v11.4s, v4.4s                     \n"                   \
  "fmla   v18.4s, v14.4s, v4.4s                     \n"                   \
  "fmla   v19.4s, v15.4s, v4.4s                     \n"

#define RIGHT_RESULT_INT8_INT8_ST                                           \
  /* data >= -127 */                                                        \
  "fcmge v10.4s, v16.4s, v0.4s                       \n"                    \
  "fcmge v11.4s, v17.4s, v0.4s                       \n"                    \
  "fcmge v14.4s, v18.4s, v0.4s                       \n"                    \
  "fcmge v15.4s, v19.4s, v0.4s                       \n" /* choose data */  \
  "bif v16.16b, v0.16b, v10.16b                     \n"                     \
  "bif v17.16b, v0.16b, v11.16b                     \n"                     \
  "bif v18.16b, v0.16b, v14.16b                     \n"                     \
  "bif v19.16b, v0.16b, v15.16b                      \n" /* fp32 - int32 */ \
  "fcvtas  v10.4s, v16.4s                           \n"                     \
  "fcvtas  v11.4s, v17.4s                           \n"                     \
  "fcvtas  v12.4s, v18.4s                           \n"                     \
  "fcvtas  v13.4s, v19.4s                           \n"                     \
  "ld1     {v0.8b}, [%[ptr_out0]]                   \n"                     \
  "ld1     {v1.8b}, [%[ptr_out1]]                   \n" /* int32-int16 */   \
  "sqxtn   v16.4h, v10.4s                           \n"                     \
  "sqxtn2  v16.8h, v11.4s                           \n"                     \
  "sqxtn   v18.4h, v12.4s                           \n"                     \
  "sqxtn2  v18.8h, v13.4s                           \n"                     \
  "ld1    {v12.8b}, [%[rmask]]                      \n" /* int16-int8 */    \
  "sqxtn  v17.8b, v16.8h                            \n"                     \
  "sqxtn  v19.8b, v18.8h                            \n"                     \
  "bif    v17.8b, v0.8b, v12.8b                     \n"                     \
  "bif    v19.8b, v1.8b, v12.8b                     \n"                     \
  "st1    {v17.8b}, [%[ptr_out0]]                   \n"                     \
  "st1    {v19.8b}, [%[ptr_out1]]                   \n"                     \
  "4:                                               \n"

#else
#endif

void conv_3x3s2p1_depthwise_int8(int8_t* dout,
                                 const int8_t* din,
                                 const int8_t* weights,
                                 const float* scale,
                                 const float* bias,
                                 bool flag_bias,
                                 float* alpha,
                                 int num,
                                 int chin,
                                 int hin,
                                 int win,
                                 int hout,
                                 int wout,
                                 ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, win * sizeof(int8_t));
  int8_t* write_ptr =
      reinterpret_cast<int8_t*>(ctx->workspace_data<int8_t>() + win);
  int threads = ctx->threads();
  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  const uint8_t out_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint8_t vmask[16];
  uint8_t rmask[8];
  auto&& res = right_mask_3x3s2p1_int8(win, wout, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint8x8_t vrmask_rp = vcgt_u8(vdup_n_u8(cnt_remain), vld1_u8(out_pad_idx));
  float max_val[4] = {-127.f, -127.f, -127.f, -127.f};
  vst1_u8(rmask, vrmask_rp);
  float32x4_t vzero = vdupq_n_f32(0);
  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * chin * size_in_channel;
    int8_t* dout_batch = dout + n * chin * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < chin; c++) {
      int8_t* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? static_cast<const float>(bias[c]) : 0;
      const int8_t* weight_ptr = weights + c * 9;
      float scale_val[4] = {scale[c], scale[c], scale[c], scale[c]};
      // clang-format off
      FILL_WEIGHTS_BIAS_INT8(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_INT8(int8_t, din_ch_ptr, win)
      for (int i = 0; i < hin; i += 4) {
        ASSIGN_PTR_3x3_S2_INT8(wout)
        TOP_BOTTOM_BORDER_3x3_S2P1_INT8(win, hin, hout)
        uint32_t cnt = cnt_col;
        uint8_t* vmask_ptr = vmask;
#ifdef __aarch64__
        asm volatile(
          INIT_INT8_S2 LEFT_COMPUTE_INT8_S2 RESULT_INT8_INT8_S2
          MID_COMPUTE_INT8_S2 RESULT_INT8_INT8_S2
          RIGHT_COMPUTE_INT8_S2 RIGHT_RESULT_INT8_INT8_S2 RIGHT_RESULT_INT8_INT8_ST
            : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
              [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (vmask_ptr)
            : [vzero] "w"(vzero), [wr00]"w"(wr00), [wr01]"w"(wr01), [wr02]"w"(wr02), \
              [wr10]"w"(wr10), [wr11]"w"(wr11), [wr12]"w"(wr12), [wr20]"w"(wr20), \
              [wr21]"w"(wr21), [wr22] "w" (wr22), [bias_val] "r"(v_bias), \
              [remain] "r"(cnt_remain), [rmask] "r"(rmask), \
              [max_val] "r" (max_val), [scale_val] "r"(scale_val)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * wout;
      }
    }
  }
}

void conv_3x3s2p1_depthwise_int8(float* dout,
                                 const int8_t* din,
                                 const int8_t* weights,
                                 const float* scale,
                                 const float* bias,
                                 bool flag_bias,
                                 float* alpha,
                                 int num,
                                 int chin,
                                 int hin,
                                 int win,
                                 int hout,
                                 int wout,
                                 ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, win * sizeof(int8_t));
  float* write_ptr =
      reinterpret_cast<float*>(ctx->workspace_data<float>() + win);
  int threads = ctx->threads();
  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  const uint32_t out_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint8_t vmask[16];
  uint32_t rmask[8];
  auto&& res = right_mask_3x3s2p1_int8(win, wout, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32x4_t vrmask_rp1 =
      vcgtq_u32(vdupq_n_u32(cnt_remain), vld1q_u32(out_pad_idx));
  uint32x4_t vrmask_rp2 =
      vcgtq_u32(vdupq_n_u32(cnt_remain), vld1q_u32(out_pad_idx + 4));
  float32x4_t vzero = vdupq_n_f32(0.f);
  vst1q_u32(rmask, vrmask_rp1);
  vst1q_u32(rmask + 4, vrmask_rp2);
  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * chin * size_in_channel;
    float* dout_batch = dout + n * chin * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < chin; c++) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? static_cast<const float>(bias[c]) : 0;
      const int8_t* weight_ptr = weights + c * 9;
      float scale_val[4] = {scale[c], scale[c], scale[c], scale[c]};
      // clang-format off
      FILL_WEIGHTS_BIAS_INT8(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_INT8(float, din_ch_ptr, win)
      for (int i = 0; i < hin; i += 4) {
        ASSIGN_PTR_3x3_S2_INT8(wout)
        TOP_BOTTOM_BORDER_3x3_S2P1_INT8(win, hin, hout)
        uint32_t cnt = cnt_col;
        uint8_t* vmask_ptr = vmask;
#ifdef __aarch64__
        asm volatile(
          INIT_INT8_S2 LEFT_COMPUTE_INT8_S2 RESULT_INT8_FP32_S2
          MID_COMPUTE_INT8_S2 RESULT_INT8_FP32_S2
          RIGHT_COMPUTE_INT8_S2 RIGHT_RESULT_INT8_FP32_S2 RIGHT_RESULT_INT8_FP32_ST
            : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
              [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (vmask_ptr)
            : [vzero] "w"(vzero), [wr00]"w"(wr00), [wr01]"w"(wr01), [wr02]"w"(wr02), \
              [wr10]"w"(wr10), [wr11]"w"(wr11), [wr12]"w"(wr12), [wr20]"w"(wr20), \
              [wr21]"w"(wr21), [wr22] "w" (wr22), [bias_val] "r"(v_bias), \
              [remain] "r"(cnt_remain), [rmask] "r"(rmask), \
              [scale_val] "r"(scale_val)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * wout;
      }
    }
  }
}

void conv_3x3s2p1_depthwise_int8_relu(int8_t* dout,
                                      const int8_t* din,
                                      const int8_t* weights,
                                      const float* scale,
                                      const float* bias,
                                      bool flag_bias,
                                      float* alpha,
                                      int num,
                                      int chin,
                                      int hin,
                                      int win,
                                      int hout,
                                      int wout,
                                      ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, win * sizeof(int8_t));
  int8_t* write_ptr =
      reinterpret_cast<int8_t*>(ctx->workspace_data<int8_t>() + win);
  int threads = ctx->threads();
  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  const uint8_t out_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint8_t vmask[16];
  uint8_t rmask[8];
  auto&& res = right_mask_3x3s2p1_int8(win, wout, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint8x8_t vrmask_rp = vcgt_u8(vdup_n_u8(cnt_remain), vld1_u8(out_pad_idx));
  float max_val[4] = {-127.f, -127.f, -127.f, -127.f};
  vst1_u8(rmask, vrmask_rp);
  float32x4_t vzero = vdupq_n_f32(0);
  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * chin * size_in_channel;
    int8_t* dout_batch = dout + n * chin * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < chin; c++) {
      int8_t* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? static_cast<const float>(bias[c]) : 0;
      const int8_t* weight_ptr = weights + c * 9;
      float scale_val[4] = {scale[c], scale[c], scale[c], scale[c]};
      // clang-format off
      FILL_WEIGHTS_BIAS_INT8(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_INT8(int8_t, din_ch_ptr, win)
      for (int i = 0; i < hin; i += 4) {
        ASSIGN_PTR_3x3_S2_INT8(wout)
        TOP_BOTTOM_BORDER_3x3_S2P1_INT8(win, hin, hout)
        uint32_t cnt = cnt_col;
        uint8_t* vmask_ptr = vmask;
#ifdef __aarch64__
        asm volatile(
          INIT_INT8_S2 LEFT_COMPUTE_INT8_S2 RESULT_INT8_S2_RELU RESULT_INT8_INT8_S2
          MID_COMPUTE_INT8_S2 RESULT_INT8_S2_RELU RESULT_INT8_INT8_S2
          RIGHT_COMPUTE_INT8_S2 RIGHT_RESULT_INT8_INT8_S2 RESULT_INT8_S2_RELU RIGHT_RESULT_INT8_INT8_ST
            : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
              [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (vmask_ptr)
            : [vzero] "w"(vzero), [wr00]"w"(wr00), [wr01]"w"(wr01), [wr02]"w"(wr02), \
              [wr10]"w"(wr10), [wr11]"w"(wr11), [wr12]"w"(wr12), [wr20]"w"(wr20), \
              [wr21]"w"(wr21), [wr22] "w" (wr22), [bias_val] "r"(v_bias), \
              [remain] "r"(cnt_remain), [rmask] "r"(rmask), \
              [max_val] "r" (max_val), [scale_val] "r"(scale_val)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * wout;
      }
    }
  }
}

void conv_3x3s2p1_depthwise_int8_relu(float* dout,
                                      const int8_t* din,
                                      const int8_t* weights,
                                      const float* scale,
                                      const float* bias,
                                      bool flag_bias,
                                      float* alpha,
                                      int num,
                                      int chin,
                                      int hin,
                                      int win,
                                      int hout,
                                      int wout,
                                      ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, win * sizeof(int8_t));
  float* write_ptr =
      reinterpret_cast<float*>(ctx->workspace_data<float>() + win);
  int threads = ctx->threads();
  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  const uint32_t out_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint8_t vmask[16];
  uint32_t rmask[8];
  auto&& res = right_mask_3x3s2p1_int8(win, wout, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32x4_t vrmask_rp1 =
      vcgtq_u32(vdupq_n_u32(cnt_remain), vld1q_u32(out_pad_idx));
  uint32x4_t vrmask_rp2 =
      vcgtq_u32(vdupq_n_u32(cnt_remain), vld1q_u32(out_pad_idx + 4));
  float32x4_t vzero = vdupq_n_f32(0.f);
  vst1q_u32(rmask, vrmask_rp1);
  vst1q_u32(rmask + 4, vrmask_rp2);
  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * chin * size_in_channel;
    float* dout_batch = dout + n * chin * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < chin; c++) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? static_cast<const float>(bias[c]) : 0;
      const int8_t* weight_ptr = weights + c * 9;
      float scale_val[4] = {scale[c], scale[c], scale[c], scale[c]};
      // clang-format off
      FILL_WEIGHTS_BIAS_INT8(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_INT8(float, din_ch_ptr, win)
      for (int i = 0; i < hin; i += 4) {
        ASSIGN_PTR_3x3_S2_INT8(wout)
        TOP_BOTTOM_BORDER_3x3_S2P1_INT8(win, hin, hout)
        uint32_t cnt = cnt_col;
        uint8_t* vmask_ptr = vmask;
#ifdef __aarch64__
        asm volatile(
          INIT_INT8_S2 LEFT_COMPUTE_INT8_S2 RESULT_INT8_S2_RELU RESULT_INT8_FP32_S2
          MID_COMPUTE_INT8_S2 RESULT_INT8_S2_RELU RESULT_INT8_FP32_S2
          RIGHT_COMPUTE_INT8_S2 RIGHT_RESULT_INT8_FP32_S2 RESULT_INT8_S2_RELU RIGHT_RESULT_INT8_FP32_ST
            : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
              [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (vmask_ptr)
            : [vzero] "w"(vzero), [wr00]"w"(wr00), [wr01]"w"(wr01), [wr02]"w"(wr02), \
              [wr10]"w"(wr10), [wr11]"w"(wr11), [wr12]"w"(wr12), [wr20]"w"(wr20), \
              [wr21]"w"(wr21), [wr22] "w" (wr22), [bias_val] "r"(v_bias), \
              [remain] "r"(cnt_remain), [rmask] "r"(rmask), \
              [scale_val] "r"(scale_val)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * wout;
      }
    }
  }
}

void conv_3x3s2p1_depthwise_int8_relu6(int8_t* dout,
                                       const int8_t* din,
                                       const int8_t* weights,
                                       const float* scale,
                                       const float* bias,
                                       bool flag_bias,
                                       float* alpha,
                                       int num,
                                       int chin,
                                       int hin,
                                       int win,
                                       int hout,
                                       int wout,
                                       ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, win * sizeof(int8_t));
  int8_t* write_ptr =
      reinterpret_cast<int8_t*>(ctx->workspace_data<int8_t>() + win);
  int threads = ctx->threads();
  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  const uint8_t out_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint8_t vmask[16];
  uint8_t rmask[8];
  auto&& res = right_mask_3x3s2p1_int8(win, wout, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint8x8_t vrmask_rp = vcgt_u8(vdup_n_u8(cnt_remain), vld1_u8(out_pad_idx));
  float max_val[4] = {-127.f, -127.f, -127.f, -127.f};
  vst1_u8(rmask, vrmask_rp);
  float32x4_t vzero = vdupq_n_f32(0);
  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * chin * size_in_channel;
    int8_t* dout_batch = dout + n * chin * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < chin; c++) {
      int8_t* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? static_cast<const float>(bias[c]) : 0;
      const int8_t* weight_ptr = weights + c * 9;
      float scale_val[4] = {scale[c], scale[c], scale[c], scale[c]};
      // clang-format off
      FILL_WEIGHTS_BIAS_INT8(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_INT8(int8_t, din_ch_ptr, win)
      for (int i = 0; i < hin; i += 4) {
        ASSIGN_PTR_3x3_S2_INT8(wout)
        TOP_BOTTOM_BORDER_3x3_S2P1_INT8(win, hin, hout)
        uint32_t cnt = cnt_col;
        uint8_t* vmask_ptr = vmask;
#ifdef __aarch64__
        asm volatile(
          INIT_INT8_S2 LEFT_COMPUTE_INT8_S2 RESULT_INT8_S2_RELU RESULT_INT8_S2_RELU6 RESULT_INT8_INT8_S2
          MID_COMPUTE_INT8_S2 RESULT_INT8_S2_RELU RESULT_INT8_S2_RELU6 RESULT_INT8_INT8_S2
          RIGHT_COMPUTE_INT8_S2 RIGHT_RESULT_INT8_INT8_S2 RESULT_INT8_S2_RELU RESULT_INT8_S2_RELU6 RIGHT_RESULT_INT8_INT8_ST
            : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
              [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (vmask_ptr)
            : [vzero] "w"(vzero), [wr00]"w"(wr00), [wr01]"w"(wr01), [wr02]"w"(wr02), \
              [wr10]"w"(wr10), [wr11]"w"(wr11), [wr12]"w"(wr12), [wr20]"w"(wr20), \
              [wr21]"w"(wr21), [wr22] "w" (wr22), [bias_val] "r"(v_bias), \
              [remain] "r"(cnt_remain), [rmask] "r"(rmask), \
              [max_val] "r" (max_val), [scale_val] "r"(scale_val), [alpha_val] "r"(alpha)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * wout;
      }
    }
  }
}

void conv_3x3s2p1_depthwise_int8_relu6(float* dout,
                                       const int8_t* din,
                                       const int8_t* weights,
                                       const float* scale,
                                       const float* bias,
                                       bool flag_bias,
                                       float* alpha,
                                       int num,
                                       int chin,
                                       int hin,
                                       int win,
                                       int hout,
                                       int wout,
                                       ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, win * sizeof(int8_t));
  float* write_ptr =
      reinterpret_cast<float*>(ctx->workspace_data<float>() + win);
  int threads = ctx->threads();
  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  const uint32_t out_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint8_t vmask[16];
  uint32_t rmask[8];
  auto&& res = right_mask_3x3s2p1_int8(win, wout, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32x4_t vrmask_rp1 =
      vcgtq_u32(vdupq_n_u32(cnt_remain), vld1q_u32(out_pad_idx));
  uint32x4_t vrmask_rp2 =
      vcgtq_u32(vdupq_n_u32(cnt_remain), vld1q_u32(out_pad_idx + 4));
  float32x4_t vzero = vdupq_n_f32(0.f);
  vst1q_u32(rmask, vrmask_rp1);
  vst1q_u32(rmask + 4, vrmask_rp2);
  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * chin * size_in_channel;
    float* dout_batch = dout + n * chin * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < chin; c++) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? static_cast<const float>(bias[c]) : 0;
      const int8_t* weight_ptr = weights + c * 9;
      float scale_val[4] = {scale[c], scale[c], scale[c], scale[c]};
      // clang-format off
      FILL_WEIGHTS_BIAS_INT8(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_INT8(float, din_ch_ptr, win)
      for (int i = 0; i < hin; i += 4) {
        ASSIGN_PTR_3x3_S2_INT8(wout)
        TOP_BOTTOM_BORDER_3x3_S2P1_INT8(win, hin, hout)
        uint32_t cnt = cnt_col;
        uint8_t* vmask_ptr = vmask;
#ifdef __aarch64__
        asm volatile(
          INIT_INT8_S2 LEFT_COMPUTE_INT8_S2 RESULT_INT8_S2_RELU RESULT_INT8_S2_RELU6 RESULT_INT8_FP32_S2
          MID_COMPUTE_INT8_S2 RESULT_INT8_S2_RELU RESULT_INT8_S2_RELU6 RESULT_INT8_FP32_S2
          RIGHT_COMPUTE_INT8_S2 RIGHT_RESULT_INT8_FP32_S2 RESULT_INT8_S2_RELU RESULT_INT8_S2_RELU6 RIGHT_RESULT_INT8_FP32_ST
            : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
              [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (vmask_ptr)
            : [vzero] "w"(vzero), [wr00]"w"(wr00), [wr01]"w"(wr01), [wr02]"w"(wr02), \
              [wr10]"w"(wr10), [wr11]"w"(wr11), [wr12]"w"(wr12), [wr20]"w"(wr20), \
              [wr21]"w"(wr21), [wr22] "w" (wr22), [bias_val] "r"(v_bias), \
              [remain] "r"(cnt_remain), [rmask] "r"(rmask), \
              [scale_val] "r"(scale_val), [alpha_val] "r"(alpha)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * wout;
      }
    }
  }
}

void conv_3x3s2p1_depthwise_int8_leaky_relu(int8_t* dout,
                                            const int8_t* din,
                                            const int8_t* weights,
                                            const float* scale,
                                            const float* bias,
                                            bool flag_bias,
                                            float* alpha,
                                            int num,
                                            int chin,
                                            int hin,
                                            int win,
                                            int hout,
                                            int wout,
                                            ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, win * sizeof(int8_t));
  int8_t* write_ptr =
      reinterpret_cast<int8_t*>(ctx->workspace_data<int8_t>() + win);
  int threads = ctx->threads();
  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  const uint8_t out_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint8_t vmask[16];
  uint8_t rmask[8];
  auto&& res = right_mask_3x3s2p1_int8(win, wout, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint8x8_t vrmask_rp = vcgt_u8(vdup_n_u8(cnt_remain), vld1_u8(out_pad_idx));
  float max_val[4] = {-127.f, -127.f, -127.f, -127.f};
  vst1_u8(rmask, vrmask_rp);
  float32x4_t vzero = vdupq_n_f32(0);
  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * chin * size_in_channel;
    int8_t* dout_batch = dout + n * chin * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < chin; c++) {
      int8_t* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? static_cast<const float>(bias[c]) : 0;
      const int8_t* weight_ptr = weights + c * 9;
      float scale_val[4] = {scale[c], scale[c], scale[c], scale[c]};
      // clang-format off
      FILL_WEIGHTS_BIAS_INT8(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_INT8(int8_t, din_ch_ptr, win)
      for (int i = 0; i < hin; i += 4) {
        ASSIGN_PTR_3x3_S2_INT8(wout)
        TOP_BOTTOM_BORDER_3x3_S2P1_INT8(win, hin, hout)
        uint32_t cnt = cnt_col;
        uint8_t* vmask_ptr = vmask;
#ifdef __aarch64__
        asm volatile(
          INIT_INT8_S2 LEFT_COMPUTE_INT8_S2 RESULT_INT8_S2_LEAKY_RELU RESULT_INT8_INT8_S2
          MID_COMPUTE_INT8_S2 RESULT_INT8_S2_LEAKY_RELU RESULT_INT8_INT8_S2
          RIGHT_COMPUTE_INT8_S2 RIGHT_RESULT_INT8_INT8_S2 RESULT_INT8_S2_LEAKY_RELU RIGHT_RESULT_INT8_INT8_ST
            : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
              [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (vmask_ptr)
            : [vzero] "w"(vzero), [wr00]"w"(wr00), [wr01]"w"(wr01), [wr02]"w"(wr02), \
              [wr10]"w"(wr10), [wr11]"w"(wr11), [wr12]"w"(wr12), [wr20]"w"(wr20), \
              [wr21]"w"(wr21), [wr22] "w" (wr22), [bias_val] "r"(v_bias), \
              [remain] "r"(cnt_remain), [rmask] "r"(rmask), \
              [max_val] "r" (max_val), [scale_val] "r"(scale_val), [alpha_val] "r"(alpha)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * wout;
      }
    }
  }
}

void conv_3x3s2p1_depthwise_int8_leaky_relu(float* dout,
                                            const int8_t* din,
                                            const int8_t* weights,
                                            const float* scale,
                                            const float* bias,
                                            bool flag_bias,
                                            float* alpha,
                                            int num,
                                            int chin,
                                            int hin,
                                            int win,
                                            int hout,
                                            int wout,
                                            ARMContext* ctx) {
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, win * sizeof(int8_t));
  float* write_ptr =
      reinterpret_cast<float*>(ctx->workspace_data<float>() + win);
  int threads = ctx->threads();
  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  const uint32_t out_pad_idx[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint8_t vmask[16];
  uint32_t rmask[8];
  auto&& res = right_mask_3x3s2p1_int8(win, wout, vmask);
  uint32_t cnt_col = res.first;
  uint32_t cnt_remain = res.second;
  uint32x4_t vrmask_rp1 =
      vcgtq_u32(vdupq_n_u32(cnt_remain), vld1q_u32(out_pad_idx));
  uint32x4_t vrmask_rp2 =
      vcgtq_u32(vdupq_n_u32(cnt_remain), vld1q_u32(out_pad_idx + 4));
  float32x4_t vzero = vdupq_n_f32(0.f);
  vst1q_u32(rmask, vrmask_rp1);
  vst1q_u32(rmask + 4, vrmask_rp2);
  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * chin * size_in_channel;
    float* dout_batch = dout + n * chin * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < chin; c++) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? static_cast<const float>(bias[c]) : 0;
      const int8_t* weight_ptr = weights + c * 9;
      float scale_val[4] = {scale[c], scale[c], scale[c], scale[c]};
      // clang-format off
      FILL_WEIGHTS_BIAS_INT8(weight_ptr, bias_val)
      INIT_PTR_3x3_S2_INT8(float, din_ch_ptr, win)
      for (int i = 0; i < hin; i += 4) {
        ASSIGN_PTR_3x3_S2_INT8(wout)
        TOP_BOTTOM_BORDER_3x3_S2P1_INT8(win, hin, hout)
        uint32_t cnt = cnt_col;
        uint8_t* vmask_ptr = vmask;
#ifdef __aarch64__
        asm volatile(
          INIT_INT8_S2 LEFT_COMPUTE_INT8_S2 RESULT_INT8_S2_LEAKY_RELU RESULT_INT8_FP32_S2
          MID_COMPUTE_INT8_S2 RESULT_INT8_S2_LEAKY_RELU RESULT_INT8_FP32_S2
          RIGHT_COMPUTE_INT8_S2 RIGHT_RESULT_INT8_FP32_S2 RESULT_INT8_S2_LEAKY_RELU RIGHT_RESULT_INT8_FP32_ST
            : [cnt] "+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), \
              [din_ptr2] "+r"(din_ptr2), [din_ptr3] "+r"(din_ptr3), [din_ptr4] "+r"(din_ptr4), \
              [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), [vmask] "+r" (vmask_ptr)
            : [vzero] "w"(vzero), [wr00]"w"(wr00), [wr01]"w"(wr01), [wr02]"w"(wr02), \
              [wr10]"w"(wr10), [wr11]"w"(wr11), [wr12]"w"(wr12), [wr20]"w"(wr20), \
              [wr21]"w"(wr21), [wr22] "w" (wr22), [bias_val] "r"(v_bias), \
              [remain] "r"(cnt_remain), [rmask] "r"(rmask), \
              [scale_val] "r"(scale_val), [alpha_val] "r"(alpha)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * wout;
      }
    }
  }
}

template <typename Dtype>
void conv_depthwise_3x3s2_int8(Dtype* dout,
                               const int8_t* din,
                               const int8_t* weights,
                               const float* scale,
                               const float* bias,
                               bool flag_bias,
                               int flag_act,
                               float* alpha,
                               int num,
                               int chin,
                               int hin,
                               int win,
                               int hout,
                               int wout,
                               int padw,
                               int padh,
                               ARMContext* ctx) {
#ifdef __aarch64__
  if (padh == padw && padw == 1 && win > 18) {
    switch (flag_act) {
      case 0:  // no act
        conv_3x3s2p1_depthwise_int8(dout,
                                    din,
                                    weights,
                                    scale,
                                    bias,
                                    flag_bias,
                                    alpha,
                                    num,
                                    chin,
                                    hin,
                                    win,
                                    hout,
                                    wout,
                                    ctx);
        break;
      case 1:  // relu
        conv_3x3s2p1_depthwise_int8_relu(dout,
                                         din,
                                         weights,
                                         scale,
                                         bias,
                                         flag_bias,
                                         alpha,
                                         num,
                                         chin,
                                         hin,
                                         win,
                                         hout,
                                         wout,
                                         ctx);
        break;
      case 2:  // relu6
        conv_3x3s2p1_depthwise_int8_relu6(dout,
                                          din,
                                          weights,
                                          scale,
                                          bias,
                                          flag_bias,
                                          alpha,
                                          num,
                                          chin,
                                          hin,
                                          win,
                                          hout,
                                          wout,
                                          ctx);
        break;
      case 3:  // leakyrelu
        conv_3x3s2p1_depthwise_int8_leaky_relu(dout,
                                               din,
                                               weights,
                                               scale,
                                               bias,
                                               flag_bias,
                                               alpha,
                                               num,
                                               chin,
                                               hin,
                                               win,
                                               hout,
                                               wout,
                                               ctx);
        break;
      default:
        LOG(FATAL) << "this act_type: " << flag_act << " fuse not support";
    }
  } else {
    conv_depthwise_3x3s2_common_int8(dout,
                                     din,
                                     weights,
                                     scale,
                                     bias,
                                     flag_bias,
                                     flag_act,
                                     alpha,
                                     num,
                                     chin,
                                     hin,
                                     win,
                                     hout,
                                     wout,
                                     padw,
                                     padh,
                                     ctx);
  }
#else
  conv_depthwise_3x3s2_common_int8(dout,
                                   din,
                                   weights,
                                   scale,
                                   bias,
                                   flag_bias,
                                   flag_act,
                                   alpha,
                                   num,
                                   chin,
                                   hin,
                                   win,
                                   hout,
                                   wout,
                                   padw,
                                   padh,
                                   ctx);
#endif
}
template void conv_depthwise_3x3s2_int8<int8_t>(int8_t* dout,
                                                const int8_t* din,
                                                const int8_t* weights,
                                                const float* scale,
                                                const float* bias,
                                                bool flag_bias,
                                                int flag_act,
                                                float* alpha,
                                                int num,
                                                int chin,
                                                int hin,
                                                int win,
                                                int hout,
                                                int wout,
                                                int padw,
                                                int padh,
                                                ARMContext* ctx);

template void conv_depthwise_3x3s2_int8<float>(float* dout,
                                               const int8_t* din,
                                               const int8_t* weights,
                                               const float* scale,
                                               const float* bias,
                                               bool flag_bias,
                                               int flag_act,
                                               float* alpha,
                                               int num,
                                               int chin,
                                               int hin,
                                               int win,
                                               int hout,
                                               int wout,
                                               int padw,
                                               int padh,
                                               ARMContext* ctx);

template void conv_depthwise_3x3s2_common_int8<int8_t>(int8_t* dout,
                                                       const int8_t* din,
                                                       const int8_t* weights,
                                                       const float* scale,
                                                       const float* bias,
                                                       bool flag_bias,
                                                       int flag_act,
                                                       float* alpha,
                                                       int num,
                                                       int chin,
                                                       int hin,
                                                       int win,
                                                       int hout,
                                                       int wout,
                                                       int padw,
                                                       int padh,
                                                       ARMContext* ctx);

template void conv_depthwise_3x3s2_common_int8<float>(float* dout,
                                                      const int8_t* din,
                                                      const int8_t* weights,
                                                      const float* scale,
                                                      const float* bias,
                                                      bool flag_bias,
                                                      int flag_act,
                                                      float* alpha,
                                                      int num,
                                                      int chin,
                                                      int hin,
                                                      int win,
                                                      int hout,
                                                      int wout,
                                                      int padw,
                                                      int padh,
                                                      ARMContext* ctx);

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
