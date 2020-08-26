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
void conv_depthwise_3x3s1_int8(Dtype* dout,
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
  const int wout_round = ((wout + wout_block - 1) / wout_block) * wout_block;
  const int win_round = wout_round + 2;

  //! get h block
  //! llc_size = threads * win_round *  hout_c_block * hin_r_block *
  //! sizeof(int8_t)
  //!  + wout_round * hout_c_block * hout_r_block * threads * sizeof(int32_t)
  //! win_round = wout_round + 2
  //! hin_r_block = hout_r_block + 2
  int hout_r_block = (llc_size - 2 * win_round * threads * hout_c_block) /
                     (win_round * threads * hout_c_block +
                      hout_c_block * wout_round * threads * 4);
  hout_r_block = hout_r_block > hout ? hout : hout_r_block;
  hout_r_block =
      ((hout_r_block + hout_r_kernel - 1) / hout_r_kernel) * hout_r_kernel;
  hout_r_block = hout_r_block < hout_r_kernel ? hout_r_kernel : hout_r_block;

  const int hin_r_block = hout_r_block + 2;

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
      int hs = h - padh;
      int he = hs + h_kernel + 2;

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
              "1:\n"
              /* inr0 -> outr0 */
              "ldp  d4, d5, [%[r0]]\n"           /* load r0, 4 */
              "smull v20.8h, v0.8b,  %[w0].8b\n" /* int16, out0 */
              "smull v21.8h, v1.8b,  %[w0].8b\n" /* int16, out1 */
              "smull v22.8h, v2.8b,  %[w0].8b\n" /* int16, out2 */
              "smull v23.8h, v3.8b,  %[w0].8b\n" /* int16, out3 */
              "smlal v20.8h, v1.8b,  %[w1].8b\n" /* int16, out0 */
              "smlal v21.8h, v2.8b,  %[w1].8b\n" /* int16, out1 */
              "smlal v22.8h, v3.8b,  %[w1].8b\n" /* int16, out2 */
              "smlal v23.8h, v4.8b,  %[w1].8b\n" /* int16, out3 */
              "ldp  d0, d1, [%[r1]], #16\n"      /* load r1, 0,1 */
              "sxtl  v24.4s, v20.4h\n"
              "sxtl2 v25.4s, v20.8h\n"
              "sxtl  v26.4s, v21.4h\n"
              "sxtl2 v27.4s, v21.8h\n"
              "sxtl  v28.4s, v22.4h\n"
              "sxtl2 v29.4s, v22.8h\n"
              "sxtl  v30.4s, v23.4h\n"
              "sxtl2 v31.4s, v23.8h\n"
              "smull v20.8h, v2.8b,  %[w2].8b\n" /* int16, out0 */
              "smull v21.8h, v3.8b,  %[w2].8b\n" /* int16, out1 */
              "smull v22.8h, v4.8b,  %[w2].8b\n" /* int16, out2 */
              "smull v23.8h, v5.8b,  %[w2].8b\n" /* int16, out3 */
              "ldp  d2, d3, [%[r1]], #16\n"      /* load r1, 2,3 */
              "smlal v20.8h, v0.8b,  %[w3].8b\n" /* int16, out0 */
              "smlal v21.8h, v1.8b,  %[w3].8b\n" /* int16, out1 */
              "smlal v22.8h, v2.8b,  %[w3].8b\n" /* int16, out2 */
              "smlal v23.8h, v3.8b,  %[w3].8b\n" /* int16, out3 */
              "saddw  v24.4s, v24.4s, v20.4h\n"
              "saddw2 v25.4s, v25.4s, v20.8h\n"
              "saddw  v26.4s, v26.4s, v21.4h\n"
              "saddw2 v27.4s, v27.4s, v21.8h\n"
              "ldp  d4, d5, [%[r1]]\n" /* load r1, 4,5 */
              "saddw  v28.4s, v28.4s, v22.4h\n"
              "saddw2 v29.4s, v29.4s, v22.8h\n"
              "saddw  v30.4s, v30.4s, v23.4h\n"
              "saddw2 v31.4s, v31.4s, v23.8h\n"
              "smull v20.8h, v1.8b,  %[w4].8b\n" /* int16, out0 */
              "smull v21.8h, v2.8b,  %[w4].8b\n" /* int16, out1 */
              "smull v22.8h, v3.8b,  %[w4].8b\n" /* int16, out1 */
              "smull v23.8h, v4.8b,  %[w4].8b\n" /* int16, out1 */
              "ldp  d0, d1, [%[r2]], #16\n"      /* load r2, 0,1 */
              "smlal v20.8h, v2.8b,  %[w5].8b\n" /* int16, out0 */
              "smlal v21.8h, v3.8b,  %[w5].8b\n" /* int16, out1 */
              "smlal v22.8h, v4.8b,  %[w5].8b\n" /* int16, out2 */
              "smlal v23.8h, v5.8b,  %[w5].8b\n" /* int16, out3 */
              "ldp  d2, d3, [%[r2]], #16\n"      /* load r2, 2,3 */
              "saddw  v24.4s, v24.4s, v20.4h\n"
              "saddw2 v25.4s, v25.4s, v20.8h\n"
              "saddw  v26.4s, v26.4s, v21.4h\n"
              "saddw2 v27.4s, v27.4s, v21.8h\n"
              "ldp  d4, d5, [%[r2]]\n" /* load r2 */
              "saddw  v28.4s, v28.4s, v22.4h\n"
              "saddw2 v29.4s, v29.4s, v22.8h\n"
              "saddw  v30.4s, v30.4s, v23.4h\n"
              "saddw2 v31.4s, v31.4s, v23.8h\n"
              "smull v20.8h, v0.8b,  %[w6].8b\n" /* int16, out0 */
              "smull v21.8h, v1.8b,  %[w6].8b\n" /* int16, out1 */
              "smull v22.8h, v2.8b,  %[w6].8b\n" /* int16, out1 */
              "smull v23.8h, v3.8b,  %[w6].8b\n" /* int16, out1 */
              "smlal v20.8h, v1.8b,  %[w7].8b\n" /* int16, out0 */
              "smlal v21.8h, v2.8b,  %[w7].8b\n" /* int16, out1 */
              "smlal v22.8h, v3.8b,  %[w7].8b\n" /* int16, out1 */
              "smlal v23.8h, v4.8b,  %[w7].8b\n" /* int16, out1 */
              "ldp  d0, d1, [%[r0]], #16\n"      /* load r0, 0,1 */
              "saddw  v24.4s, v24.4s, v20.4h\n"
              "saddw2 v25.4s, v25.4s, v20.8h\n"
              "saddw  v26.4s, v26.4s, v21.4h\n"
              "saddw2 v27.4s, v27.4s, v21.8h\n"
              "saddw  v28.4s, v28.4s, v22.4h\n"
              "saddw2 v29.4s, v29.4s, v22.8h\n"
              "saddw  v30.4s, v30.4s, v23.4h\n"
              "saddw2 v31.4s, v31.4s, v23.8h\n"
              "smull v20.8h, v2.8b,  %[w8].8b\n" /* int16, out0 */
              "smull v21.8h, v3.8b,  %[w8].8b\n" /* int16, out1 */
              "smull v22.8h, v4.8b,  %[w8].8b\n" /* int16, out1 */
              "smull v23.8h, v5.8b,  %[w8].8b\n" /* int16, out1 */
              "ldp  d2, d3, [%[r0]], #16\n"      /* load r0, 2,3 */
              "saddw  v24.4s, v24.4s, v20.4h\n"
              "saddw2 v25.4s, v25.4s, v20.8h\n"
              "saddw  v26.4s, v26.4s, v21.4h\n"
              "saddw2 v27.4s, v27.4s, v21.8h\n"
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
              "vld1.32    {d0-d3}, [%[r0]]!\n"   /* load r0, 0-4 */
              "vld1.32    {d6-d7}, [%[wptr]]!\n" /* load w0-w1 */
              "1:\n"
              /* inr0 -> outr0 */
              "vld1.32    {d4-d5}, [%[r0]]\n"   /* load r0, 5-6 */
              "vmull.s8 q4, d0,  d6\n"          /* int16, out0 */
              "vmull.s8 q5, d1,  d6\n"          /* int16, out1 */
              "vmull.s8 q6, d2,  d6\n"          /* int16, out2 */
              "vmull.s8 q7, d3,  d6\n"          /* int16, out3 */
              "vld1.32    {d6},   [%[wptr]]!\n" /* load w2 */
              "vmlal.s8 q4, d1,  d7\n"          /* int16, out0 */
              "vmlal.s8 q5, d2,  d7\n"          /* int16, out1 */
              "vmlal.s8 q6, d3,  d7\n"          /* int16, out2 */
              "vmlal.s8 q7, d4,  d7\n"          /* int16, out3 */
              "vld1.32    {d7},   [%[wptr]]!\n" /* load w3 */
              "vmovl.s16  q8, d8\n"
              "vmovl.s16  q9, d9\n"
              "vmovl.s16  q10, d10\n"
              "vmovl.s16  q11, d11\n"
              "vld1.32    {d0-d1}, [%[r1]]!\n" /* load r1, 0-1 */
              "vmovl.s16  q12, d12\n"
              "vmovl.s16  q13, d13\n"
              "vmovl.s16  q14, d14\n"
              "vmovl.s16  q15, d15\n"
              "vmull.s8 q4, d2,  d6\n"          /* int16, out0 */
              "vmull.s8 q5, d3,  d6\n"          /* int16, out1 */
              "vld1.32    {d2-d3}, [%[r1]]!\n"  /* load r1, 2-3 */
              "vmull.s8 q6, d4,  d6\n"          /* int16, out2 */
              "vmull.s8 q7, d5,  d6\n"          /* int16, out3 */
              "vld1.32    {d6},   [%[wptr]]!\n" /* load w4 */
              /* inr1 -> outr0 */
              "vmlal.s8 q4, d0,  d7\n"        /* int16, out0 */
              "vmlal.s8 q5, d1,  d7\n"        /* int16, out1 */
              "vmlal.s8 q6, d2,  d7\n"        /* int16, out2 */
              "vmlal.s8 q7, d3,  d7\n"        /* int16, out3 */
              "vld1.32    {d4-d5}, [%[r1]]\n" /* load r1, 4-5 */
              "vaddw.s16  q8, q8, d8\n"
              "vaddw.s16  q9, q9, d9\n"
              "vaddw.s16  q10, q10, d10\n"
              "vaddw.s16  q11, q11, d11\n"
              "vld1.32    {d7},   [%[wptr]]!\n" /* load w5 */
              "vaddw.s16  q12, q12, d12\n"
              "vaddw.s16  q13, q13, d13\n"
              "vaddw.s16  q14, q14, d14\n"
              "vaddw.s16  q15, q15, d15\n"
              "vmull.s8 q4, d1,  d6\n"          /* int16, out0 */
              "vmull.s8 q5, d2,  d6\n"          /* int16, out1 */
              "vmull.s8 q6, d3,  d6\n"          /* int16, out2 */
              "vmull.s8 q7, d4,  d6\n"          /* int16, out3 */
              "vld1.32    {d6},   [%[wptr]]!\n" /* load w6 */
              "vld1.32    {d0-d1}, [%[r2]]!\n"  /* load r2, 0-1 */
              "vmlal.s8 q4, d2,  d7\n"          /* int16, out0 */
              "vmlal.s8 q5, d3,  d7\n"          /* int16, out1 */
              "vmlal.s8 q6, d4,  d7\n"          /* int16, out2 */
              "vmlal.s8 q7, d5,  d7\n"          /* int16, out3 */
              "vld1.32    {d7},   [%[wptr]]!\n" /* load w7 */
              "vaddw.s16  q8, q8, d8\n"
              "vaddw.s16  q9, q9, d9\n"
              "vaddw.s16  q10, q10, d10\n"
              "vaddw.s16  q11, q11, d11\n"
              "vld1.32    {d2-d3}, [%[r2]]!\n" /* load r2, 2-3 */
              "vaddw.s16  q12, q12, d12\n"
              "vaddw.s16  q13, q13, d13\n"
              "vaddw.s16  q14, q14, d14\n"
              "vaddw.s16  q15, q15, d15\n"
              "vld1.32    {d4-d5}, [%[r2]]\n" /* load r2, 4-5 */
              /* inr2 -> outr0 */
              "vmull.s8 q4, d0,  d6\n"          /* int16, out0 */
              "vmull.s8 q5, d1,  d6\n"          /* int16, out1 */
              "vmull.s8 q6, d2,  d6\n"          /* int16, out2 */
              "vmull.s8 q7, d3,  d6\n"          /* int16, out3 */
              "vld1.32    {d6},   [%[wptr]]!\n" /* load w8 */
              "vmlal.s8 q4, d1,  d7\n"          /* int16, out0 */
              "vmlal.s8 q5, d2,  d7\n"          /* int16, out1 */
              "vmlal.s8 q6, d3,  d7\n"          /* int16, out2 */
              "vmlal.s8 q7, d4,  d7\n"          /* int16, out3 */
              "vaddw.s16  q8, q8, d8\n"
              "vaddw.s16  q9, q9, d9\n"
              "vaddw.s16  q10, q10, d10\n"
              "vaddw.s16  q11, q11, d11\n"
              "vld1.32    {d0-d1}, [%[r0]]!\n" /* load r0, 0-1 */
              "vaddw.s16  q12, q12, d12\n"
              "vaddw.s16  q13, q13, d13\n"
              "vaddw.s16  q14, q14, d14\n"
              "vaddw.s16  q15, q15, d15\n"
              "sub %[wptr],   %[wptr],    #72\n"
              "vmull.s8 q4, d2,  d6\n"         /* int16, out0 */
              "vmull.s8 q5, d3,  d6\n"         /* int16, out1 */
              "vmull.s8 q6, d4,  d6\n"         /* int16, out2 */
              "vmull.s8 q7, d5,  d6\n"         /* int16, out3 */
              "vld1.32    {d2-d3}, [%[r0]]!\n" /* load r0, 2-3 */
              "vaddw.s16  q8, q8, d8\n"
              "vaddw.s16  q9, q9, d9\n"
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
          block_inr0 = block_inr1;
          block_inr1 = block_inr2;
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

template void conv_depthwise_3x3s1_int8<int8_t>(int8_t* dout,
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

template void conv_depthwise_3x3s1_int8<float>(float* dout,
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

void conv_depthwise_3x3s1p1_bias_int8_float(float* dout,
                                            const int8_t* din,
                                            const int8_t* weights,
                                            const float* scale,
                                            const float* bias,
                                            bool flag_bias,
                                            int flag_act,
                                            float* alpha,
                                            int num,
                                            int ch_in,
                                            int h_in,
                                            int w_in,
                                            int h_out,
                                            int w_out,
                                            ARMContext* ctx) {
  const unsigned char right_pad_idx[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, w_in * sizeof(int8_t));
  float* write_ptr =
      reinterpret_cast<float*>(ctx->workspace_data<int8_t>() + w_in);
  int threads = ctx->threads();

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  int tile_w = (w_in + 7) >> 3;
  int cnt_col = tile_w - 2;

  unsigned int size_pad_right = (unsigned int)(w_in - 7 - (cnt_col << 3));

  unsigned int rst_remain = (unsigned int)(w_out - ((cnt_col + 1) << 3));
  uint32x4_t vmask_result1 =
      vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
  uint32x4_t vmask_result2 =
      vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

  int8x8_t vzero = vdup_n_s8(0);
  uint8x16_t vmask_rp =
      vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
  unsigned char vmask[16];
  vst1q_u8(vmask, vmask_rp);

  unsigned int rmask[8];
  vst1q_u32(rmask, vmask_result1);
  vst1q_u32(rmask + 4, vmask_result2);

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;

      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
      float vscale[4] = {scale_val, scale_val, scale_val, scale_val};

#ifdef __aarch64__
      int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
      int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
      int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

      int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
      int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
      int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

      int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
      int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
      int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

      float32x4_t v_bias = vdupq_n_f32(vbias[0]);
      float32x4_t v_scale = vdupq_n_f32(vscale[0]);
#endif

      float* doutr0 = nullptr;
      float* doutr1 = nullptr;

      const int8_t* dr0 = din_ch_ptr;
      const int8_t* dr1 = dr0 + w_in;
      const int8_t* dr2 = dr1 + w_in;
      const int8_t* dr3 = dr2 + w_in;

      const int8_t* din_ptr0 = nullptr;
      const int8_t* din_ptr1 = nullptr;
      const int8_t* din_ptr2 = nullptr;
      const int8_t* din_ptr3 = nullptr;

      for (int i = 0; i < h_in; i += 2) {
        //! process top pad pad_h = 1
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;
        unsigned int* rst_mask = rmask;
        unsigned char* val_mask = vmask;

        if (i == 0) {
          din_ptr0 = zero_ptr;
          din_ptr1 = dr0;
          din_ptr2 = dr1;
          din_ptr3 = dr2;
          dr0 = dr1;
          dr1 = dr2;
          dr2 = dr3;
          dr3 = dr2 + w_in;
        } else {
          dr0 = dr2;
          dr1 = dr3;
          dr2 = dr1 + w_in;
          dr3 = dr2 + w_in;
        }
        //! process bottom pad
        if (i + 3 > h_in) {
          switch (i + 3 - h_in) {
            case 3:
              din_ptr1 = zero_ptr;
            case 2:
              din_ptr2 = zero_ptr;
            case 1:
              din_ptr3 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 2 > h_out) {
          doutr1 = write_ptr;
        }
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr3]] \n"
                    "movi   v21.4s, #0x0\n"                                     /* v21 = 0 */
                // left
                    "ld1    {v0.8b}, [%[din_ptr0]], #8               \n"   /* load a(0,0)-a(0,7) to q0*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8               \n"   /* load a(1,0)-a(1.7) to q2*/
                    "ld1    {v1.8b}, [%[din_ptr0]]                   \n"   /* load a(0,8)-a(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]                   \n"   /* load a(1,8)-a(1,15) to q3*/
                    "movi   v10.4s, #0x0\n"                 /* init int32 acc v10 to 0 */
                    "movi   v11.4s, #0x0\n"                 /* init int32 acc v11 to 0 */
                    "movi   v12.4s, #0x0\n"                   /* init int32 acc v12 to 0 */
                    "movi   v13.4s, #0x0\n"                   /* init int32 acc v13 to 0 */
                    //r0
                    "smull  v18.8h,  %[v1].8b,  v0.8b   \n"   /* acc_r(0,0) = r(0,01234567) * w01 */
                    "ext v4.8b, v21.8b, v0.8b, #7       \n"   /* vext_s8(vzero, vin_r0, 7); r(0,00123456) */
                    "ext v5.8b, v0.8b, v1.8b, #1        \n"   /* vext_s8(vin_r0, vin_r0_h, 1); r(0,12345678) */
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a(2,0)-a(2,7) to q6*/
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a(3,0)-a(3,7) to q8*/
                    "smlal  v18.8h,  %[v0].8b,  v4.8b   \n"   /* acc_r(0,0) += r(0,00123456) * w00 */
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a(2,8)-a(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a(3,8)-a(3,15) to q9*/
                    "sub   %[din_ptr0], %[din_ptr0], #1 \n"
                    "sub   %[din_ptr1], %[din_ptr1], #1 \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v2].8b,  v5.8b \n"   /* acc_r00 += r(0,12345678) * w02 */
                    "ext v14.8b, v21.8b, v2.8b, #7       \n"   /* vext_s8(vzero, vin_r1, 7); r(1,00123456) */
                    "ext v15.8b, v2.8b, v3.8b, #1        \n"   /* vext_s8(vin_r1, vin_r1_h, 1); r(1,12345678) */
                    //r1
                    "sub   %[din_ptr2], %[din_ptr2], #1 \n"
                    "sub   %[din_ptr3], %[din_ptr3], #1 \n"
                    "smull  v19.8h,  %[v1].8b,  v2.8b   \n"   /* acc_r10 += r(1,01234567) * w01 */
                    "smlal  v18.8h,  %[v4].8b,  v2.8b   \n"   /* acc_r00 += r(1,01234567) * w11 */
                    "ext v4.8b, v21.8b, v6.8b, #7      \n"
                    "ext v5.8b, v6.8b, v7.8b, #1       \n"
                    "smlal  v19.8h,  %[v0].8b,  v14.8b   \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v3].8b,  v14.8b   \n"   /* acc_r00 += r(1, 00123456) * w10 */
                    "ld1    {v0.8b}, [%[din_ptr0]], #8  \n"   /* load a'(0,0)-a'(0,7) to q0 for prefetch*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8  \n"   /* load a'(1,0)-a'(1,7) to q2 for prefetch*/
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v2].8b,  v15.8b   \n"   /* acc_r10 += r(1,12345678) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b   \n"   /* acc_r00 += r(1,12345678) * w12 */
                    //r2
                    "ld1    {v1.8b}, [%[din_ptr0]]      \n"   /* load a'(0,8)-a'(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]      \n"   /* load a'(1,8)-a'(1,15) to q3*/
                    "smlal  v19.8h,  %[v4].8b,  v6.8b   \n"   /* acc_r10 += r(2,01234567) * w11 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v7].8b,  v6.8b   \n"   /* acc_r00 += r(2,01234567) * w21 */
                    "ext v14.8b, v21.8b, v8.8b, #7       \n"
                    "ext v15.8b, v8.8b, v9.8b, #1        \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v3].8b,  v4.8b  \n"   /* acc_r10 += r(2, 00123456) * w10 */
                    "smlal  v18.8h,  %[v6].8b,  v4.8b  \n"
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a'(2,0)-a'(2,7) to q6 for prefetch*/
                    "smlal  v19.8h,  %[v5].8b,  v5.8b  \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b  \n"   /* acc_r00 += r(2, 12345678) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v12 += outr10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v13 += outr10.high*/
                    //r3
                    "smull  v19.8h,  %[v7].8b,  v8.8b   \n"   /* acc_r10 += r(3, 01234567) * w21 */
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a'(3,0)-a'(3,7) to q8 for prefetch*/
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a'(2,7)-a'(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a'(3,7)-a'(3,15) to q9*/
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v6].8b,  v14.8b   \n"   /* acc_r10 += r(3, 00123456) * w20) */
                    "dup   v20.4s, %[v_bias].s[0] \n "         /* dup bias to v20 */
                    "dup   v22.4s, %[v_bias].s[0] \n "         /* dup bias to v22 */
                    "scvtf  v10.4s, v10.4s               \n"   /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s               \n"   /* int32-> float32*/
                    "fmla   v20.4s, v10.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "fmla   v22.4s, v11.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "stp    q20, q22, [%[ptr_out0]], #32 \n"   /* store q10, q11 -> ptr_out 0 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"    /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"    /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b   \n"   /* acc_r10 += r(3, 12345678) * w22 */
                    "movi   v10.4s, #0x0\n"         /* clear 0 for v10 */
                    "movi   v11.4s, #0x0\n"         /* clear 0 for v11 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"    /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"    /* v13 += acc_r10.high*/
                    "scvtf  v12.4s, v12.4s                \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                \n"  /* int32-> float32*/
                    "dup   v14.4s, %[v_bias].s[0] \n "         /* dup bias to v14 */
                    "dup   v15.4s, %[v_bias].s[0] \n "         /* dup bias to v15 */
                    "fmla   v14.4s, v12.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "fmla   v15.4s, v13.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "stp     q14, q15, [%[ptr_out1]], #32   \n" /* store q10, q11 -> ptr_out  1 */
                    "movi   v12.4s, #0x0\n"
                    "movi   v13.4s, #0x0\n"
                    "cmp  %[cnt], #1                \n"
                    "blt 3f                         \n"
                //mid
                    "1:                             \n"
                    "ext v4.8b, v0.8b, v1.8b, #1       \n"      /* vext_s8(vin_r0, vin_r0_h, 1); r(0, 12345678) */
                    "ext v5.8b, v0.8b, v1.8b, #2       \n"      /* vext_s8(vin_r0, vin_r0_h, 2); r(0, 23456789) */
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b  \n"      /* acc_r00 = r(0, 01234567) * w00 */
                    "ext v14.8b, v2.8b, v3.8b, #1      \n"      /* vext_s8(vin_r1, vin_r1_h, 1); r(1, 12345678) */
                    "ext v15.8b, v2.8b, v3.8b, #2      \n"      /* vext_s8(vin_r1, vin_r1_h, 1); r(1, 23456789) */
                    "smlal  v18.8h,  %[v1].8b,  v4.8b  \n"      /* acc_r00 += r(0, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v2].8b,  v5.8b   \n"
                    //r1
                    "ext v4.8b, v6.8b, v7.8b, #1       \n"     /* vext_s8(vin_r2, vin_r2_h, 1); r(2, 12345678) */
                    "ext v5.8b, v6.8b, v7.8b, #2       \n"     /* vext_s8(vin_r2, vin_r2_h, 2); r(2, 23456789) */
                    "smull  v19.8h,  %[v0].8b,  v2.8b   \n"     /* acc_r10 += r(1, 01234567) * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b   \n"     /* acc_r00 += r(1, 01234567) * w10 */
                    "ld1    {v0.8b}, [%[din_ptr0]], #8  \n"     /* load a'(0,7)-a'(0,15) to q0 for prefetch*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8  \n"     /* load a'(1,7)-a'(1,15) to q2 for prefetch*/
                    "smlal  v19.8h,  %[v1].8b,  v14.8b\n"       /* acc_r10 += r(1, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v4].8b,  v14.8b  \n"     /* acc_r00 += r(1, 12345678) * w11 */
                    "ld1    {v1.8b}, [%[din_ptr0]]      \n"     /* load a'(0,7)-a'(0,15) to q1 for prefetch */
                    "ld1    {v3.8b}, [%[din_ptr1]]      \n"     /* load a'(1,7)-a'(1,15) to q3 for prefetch */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v2].8b,  v15.8b  \n"       /* acc_r10 += r(1, 23456789) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b  \n"       /* acc_r00 += r(1, 23456789) * w12 */
                    //r2
                    "ext v14.8b, v8.8b, v9.8b, #1       \n"      /* vext_s8(vin_r3, vin_r3_h, 1); r(3, 12345678) */
                    "ext v15.8b, v8.8b, v9.8b, #2       \n"      /* vext_s8(vin_r3, vin_r3_h, 1); r(3, 23456789) */
                    "smlal  v19.8h,  %[v3].8b,  v6.8b   \n"     /* acc_r10 += r(2, 01234567) * w10 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r10.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r10.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b   \n"     /* acc_r00 += r(2, 01234567) * w20 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b\n"       /* acc_r10 += r(2, 12345678) * w11 */
                    "smlal  v18.8h,  %[v7].8b,  v4.8b\n"       /* acc_r00 += r(2, 12345678) * w21 */
                    "smlal  v19.8h,  %[v5].8b,  v5.8b\n"       /* acc_r10 += r(2, 23456789) * w12 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b  \n"     /* acc_r00 += r(2, 23456789) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high */
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b   \n"     /* acc_r10 = r(3, 01234567) * w20 */
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"     /* load a'(2,0)-a'(2,7) to q6 for prefetch */
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"     /* load a'(3,0)-a'(3,7) to q8 for prefetch */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high */
                    "smlal  v19.8h,  %[v7].8b,  v14.8b   \n"     /* acc_r10 += r(3, 12345678) * w21 */
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"     /* load a'(2,8)-a'(2,15) to q7 for prefetch */
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"     /* load a'(3,8)-a'(3,15) to q9 for prefetch */
                    "scvtf  v10.4s, v10.4s                 \n"  /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s                 \n"  /* int32-> float32*/
                    "dup   v20.4s, %[v_bias].s[0]          \n "
                    "dup   v22.4s, %[v_bias].s[0]          \n "
                    "fmla   v20.4s, v10.4s, %[v_scale].4s  \n"
                    "fmla   v22.4s, v11.4s, %[v_scale].4s  \n"
                    "saddw   v12.4s, v12.4s, v19.4h        \n"  /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h       \n"  /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b     \n"  /* acc_r10 += r(3, 23456789) * w22 */
                    "stp     q20, q22, [%[ptr_out0]], #32  \n"  /* store q10, q11 -> ptr_out 0  */
                    "movi   v10.4s, #0x0                   \n"
                    "movi   v11.4s, #0x0                   \n"
                    "saddw   v12.4s, v12.4s, v19.4h        \n"  /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h       \n"  /* v13 += acc_r10.high */
                    "subs %[cnt], %[cnt], #1                \n"
                    "scvtf  v12.4s, v12.4s                  \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                  \n"  /* int32-> float32*/
                    "dup   v18.4s, %[v_bias].s[0]           \n "
                    "dup   v19.4s, %[v_bias].s[0]           \n "
                    "fmla   v18.4s, v12.4s, %[v_scale].4s   \n"
                    "fmla   v19.4s, v13.4s, %[v_scale].4s   \n"
                    "stp     q18, q19, [%[ptr_out1]], #32   \n"   /* store q12, q13 -> ptr_out 1  */
                    "movi   v12.4s, #0x0                    \n"
                    "movi   v13.4s, #0x0                    \n"
                    "bne 1b                                 \n"
                //right
                    "3:                                      \n" 
                    "ld1 {v20.8b}, [%[vmask]], #8            \n"
                    "ld1 {v22.8b}, [%[vmask]]                \n"
                    "bif v0.8b, v21.8b, v20.8b               \n"  /* a(0, 0) to a(0, 7) */
                    "bif v1.8b, v21.8b, v22.8b               \n"  /* a(0, 8) to a(0, 15) */
                    "bif v2.8b, v21.8b, v20.8b               \n"  /* a(1, 0) to a(1, 7) */
                    "bif v3.8b, v21.8b, v22.8b               \n"  /* a(1, 8) to a(1, 15) */
                    "ext v4.8b, v0.8b, v1.8b, #1             \n"  /* r(0, 12345678) */
                    "ext v5.8b, v0.8b, v1.8b, #2             \n"  /* r(0, 23456789) */
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b        \n"  /* acc_r00 += r(0, 01234567) * w00 */
                    "ext v14.8b, v2.8b, v3.8b, #1            \n"  /* r(1, 12345678) */
                    "ext v15.8b, v2.8b, v3.8b, #2            \n"  /* r(1, 23456789) */
                    "bif v6.8b, v21.8b, v20.8b               \n"  /* a(2, 0) to a(0, 7) */
                    "bif v7.8b, v21.8b, v22.8b               \n"  /* a(2, 8) to a(2, 15) */
                    "smlal  v18.8h,  %[v1].8b,  v4.8b        \n"  /* acc_r00 += r(0, 12345678) * w01 */
                    "bif v8.8b, v21.8b, v20.8b               \n"  /* a(3, 0) to a(3, 7) */ 
                    "bif v9.8b, v21.8b, v22.8b               \n"  /* a(3, 8) to a(3, 15) */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v2].8b,  v5.8b        \n"  /* acc_r00 = r(0, 23456789) * w02 */
                    //r1
                    "ext v4.8b, v6.8b, v7.8b, #1            \n"
                    "ext v5.8b, v6.8b, v7.8b, #2            \n"
                    "smull  v19.8h,  %[v0].8b,  v2.8b        \n"  /* acc_r10 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b        \n"  /* acc_r00 = 01234567 * w00 */
                    "ld1 {v20.4s}, [%[rmask]], #16           \n"
                    "ld1 {v22.4s}, [%[rmask]]                \n"
                    "smlal  v19.8h,  %[v1].8b,  v14.8b       \n"  /* acc_r10 += r(1, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v4].8b,  v14.8b       \n"  /* acc_r00 += r(1, 12345678) * w11 */
                    "ld1 {v0.4s}, [%[ptr_out0]], #16         \n"  /* load original ptr_out0 low */
                    "ld1 {v2.4s}, [%[ptr_out1]], #16         \n"  /* load original ptr_out1 low */
                    "saddw   v12.4s, v12.4s, v19.4h          \n"  /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h         \n"  /* v13 += acc_r10.high */
                    "smull  v19.8h,  %[v2].8b,  v15.8b       \n"  /* acc_r10 = r(1, 23456789) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b       \n"  /* acc_r00 = r(1, 23456789) * w12 */
                    "ld1 {v1.4s}, [%[ptr_out0]]              \n"  /* load original ptr_out0 high */
                    "ld1 {v3.4s}, [%[ptr_out1]]              \n"  /* load original ptr_out1 high */
                    //r2
                    "ext v14.8b, v8.8b, v9.8b, #1             \n"  /* r(3, 12345678) */
                    "ext v15.8b, v8.8b, v9.8b, #2             \n"  /* r(3, 23456789) */
                    "smlal  v19.8h,  %[v3].8b,  v6.8b        \n"  /* acc_r10 += r(2, 01234567) * w10 */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b        \n"  /* acc_r00 = r(2, 01234567) * w20 */
                    "sub %[ptr_out0], %[ptr_out0], #16      \n"
                    "sub %[ptr_out1], %[ptr_out1], #16      \n"
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v13 += outr00.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b       \n"   /* acc_r10 += r(2, 12345678) * w11 */
                    "smlal  v18.8h,  %[v7].8b,  v4.8b       \n"   /* acc_r00 += r(2, 12345678) * w21 */
                    "smlal  v19.8h,  %[v5].8b,  v5.8b       \n"   /* acc_r10 += r(2, 23456789) * w12 */
                    "saddw   v10.4s, v10.4s, v18.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h        \n"   /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b       \n"   /* acc_r00 += r(2, 23456789) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += acc_r00.high*/
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b       \n"   /* acc_r10 += r(3, 01234567) * w20 */
                    "saddw   v10.4s, v10.4s, v18.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h        \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v7].8b,  v14.8b       \n"   /* acc_r10 += r(3, 12345678) * w21 */
                    "scvtf  v10.4s, v10.4s                  \n"  /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s                  \n"  /* int32-> float32*/
                    "dup   v6.4s, %[v_bias].s[0]                 \n "
                    "dup   v7.4s, %[v_bias].s[0]                 \n "
                    "fmla   v6.4s, v10.4s, %[v_scale].4s         \n"
                    "fmla   v7.4s, v11.4s, %[v_scale].4s         \n"
                    "bif v6.16b, v0.16b, v20.16b            \n"   /* select bit of ptr_out 0 according to rmask */
                    "bif v7.16b, v1.16b, v22.16b            \n"   /* select bit of ptr_out 0 high according to rmask */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += acc_r00.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b      \n"   /* acc_r10 = r(3, 23456789) * w22 */
                    "stp     q6, q7, [%[ptr_out0]], #32     \n"   /* store q10, q11 -> ptr_out 0   */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += outr00.high*/
                    "scvtf  v12.4s, v12.4s                  \n"   /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                  \n"   /* int32-> float32*/
                    "dup   v8.4s, %[v_bias].s[0]            \n "
                    "dup   v9.4s, %[v_bias].s[0]            \n "
                    "fmla   v8.4s, v12.4s, %[v_scale].4s    \n"
                    "fmla   v9.4s, v13.4s, %[v_scale].4s    \n"
                    "bif v8.16b, v2.16b, v20.16b            \n"   /* select bit of ptr_out 1 according to rmask */
                    "bif v9.16b, v3.16b, v22.16b            \n"   /* select bit of ptr_out 1 hihg according to rmask */
                    "stp   q8, q9, [%[ptr_out1]], #32       \n"   /* store q12, q13 -> ptr_out 1  */
                    :   [cnt]"+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [din_ptr3] "+r"(din_ptr3), [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
                        [vmask] "+r" (val_mask), [rmask] "+r" (rst_mask)
                    :   [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), \
                        //[bias_val] "r" (vbias), [scale_val] "r"(vscale),
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22), \
                        [v_bias]"w"(v_bias), [v_scale] "w"(v_scale)
                    :"cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
                      "v10", "v11", "v12","v13","v14","v15", \
                      "v18",\
                      "v19", "v20", "v21", "v22"
                );
#else
                // store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                \n"
                    "pld [%[din_ptr1]]                \n"
                    "pld [%[din_ptr2]]                \n"
                    "pld [%[din_ptr3]]                \n"
                    "vdup.s8     d2, d0[0]            \n"
                    "vdup.s8     d3, d0[1]            \n"
                    "vdup.s8     d4, d0[2]            \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]  \n"
                    "vmov.u32 d11, #0                  \n"
                    "vmov.u32 q8, #0                   \n"
                    "vmov.u32 q9, #0                   \n"
                    "vmov.u32 q10, #0                  \n"
                    "vmov.u32 q11, #0                  \n"
                    //r0
                    "vmull.s8 q12, d12, d3                 \n" // q12 = d12 * w01
                    "vext.8     d30, d11, d12, #7          \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          \n" //d11 = 12345678
                    "vld1.8 {d12-d13}, [%[din_ptr1]]    \n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]    \n"
                    "vdup.s8     d5, d0[3]              \n"
                    "vdup.s8     d6, d0[4]              \n"
                    "vmlal.s8 q12, d30, d2              \n" // q12 += d10 * w00
                    "vdup.s8     d7, d0[5]               \n"
                    "add %[din_ptr0], #7                   \n"
                    "add %[din_ptr1], #7                   \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                 \n" // q12 += d11 * w02
                    //r1
                    "vext.8     d30, d11, d12, #7     \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d3                 \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d12, d6                 \n" // q12 = d12 * w11
                    "vld1.8 {d12-d13}, [%[din_ptr3]]     \n"
                    "vdup.s8     d8, d0[6]               \n"
                    "vdup.s8     d9, d0[7]               \n"
                    "vdup.s8     d10, d1[0]               \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d30, d2                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d5                 \n" // q12 += d10 * w00
                    "add %[din_ptr2], #7                   \n"
                    "add %[din_ptr3], #7                   \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 \n" // q12 += d10 * w00
                    //r2
                    "vext.8     d30, d11, d14, #7     \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #1          \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d14, d6                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d9                 \n" // q13 = d12 * w01
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d5                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d8                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                 \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d11, d12, #7           \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1           \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d12, d9                 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                     \n"
                    "pld [%[din_ptr1]]                     \n"
                    "vmlal.s8 q13, d30, d8                 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                \n"
                    "pld [%[din_ptr3]]                \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                 \n" // q12 += d10 * w00
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q7, %[scale]                            \n"
                    "vdup.32 q14, %[bias]                            \n"
                    "vdup.32 q15, %[bias]                            \n"
                    "vcvt.f32.s32   q8, q8                      \n"
                    "vcvt.f32.s32   q9, q9                      \n"
                    "vmla.f32 q14, q8, q7                     \n"
                    "vmla.f32 q15, q9, q7                     \n"
                    "vst1.32 {d28-d29}, [%[dout_ptr1]]!             \n"
                    "vst1.32 {d30-d31}, [%[dout_ptr1]]!             \n"
                    "vdup.32 q12, %[bias]                            \n"
                    "vdup.32 q13, %[bias]                            \n"
                    "vcvt.f32.s32   q10, q10                      \n"
                    "vcvt.f32.s32   q11, q11                      \n"
                    "vmla.f32 q12, q10, q7                     \n"
                    "vmla.f32 q13, q11, q7                     \n"
                    "vst1.32 {d24-d25}, [%[dout_ptr2]]!         \n"
                    "cmp %[cnt], #1                                 \n"
                    "vst1.32 {d26-d27}, [%[dout_ptr2]]!         \n"
                    "blt 1f                                         \n"
                //mid
                    "2:                                          \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    \n"
                    "vmov.u32 q8, #0                            \n"
                    "vmov.u32 q9, #0                            \n"
                    "vmov.u32 q10, #0                            \n"
                    "vmov.u32 q11, #0                            \n"
                     //r0
                    "vmull.s8 q12, d12, d2                 \n" // q12 = d12 * w01
                    "vext.8     d30, d12, d13, #1          \n" //d10 = 12345678
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr1]]    \n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]    \n"
                    "vmlal.s8 q12, d30, d3                 \n" // q12 += d10 * w00
                    "add %[din_ptr0], #8                   \n"
                    "add %[din_ptr1], #8                   \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                 \n" // q12 += d11 * w02
                    //r1
                    "vext.8     d30, d12, d13, #1     \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d2                 \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d12, d5                 \n" // q12 = d12 * w11
                    "vld1.8 {d12-d13}, [%[din_ptr3]]    \n"
                    "vmlal.s8 q13, d30, d3                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d30, d6                 \n" // q12 += d10 * w00
                    "add %[din_ptr2], #8                   \n"
                    "add %[din_ptr3], #8                   \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 \n" // q12 += d10 * w00
                    //r2
                    "vext.8     d30, d14, d15, #1     \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d14, d5                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d8                 \n" // q13 = d12 * w01
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 \n" // q12 += d10 * w00
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d10                 \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d12, d13, #1     \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d12, d8                 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                \n"
                    "pld [%[din_ptr1]]                \n"
                    "vmlal.s8 q13, d30, d9                 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                \n"
                    "pld [%[din_ptr3]]                \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                 \n" // q12 += d10 * w00
                    "vdup.32 q7, %[scale]                            \n"
                    "vdup.32 q14, %[bias]                            \n"
                    "vdup.32 q15, %[bias]                            \n"
                    "vcvt.f32.s32   q8, q8                      \n"
                    "vcvt.f32.s32   q9, q9                      \n"
                    "vmla.f32 q14, q8, q7                            \n"
                    "vmla.f32 q15, q9, q7                            \n"
                    "vst1.32 {d28-d29}, [%[dout_ptr1]]!              \n"
                    "vst1.32 {d30-d31}, [%[dout_ptr1]]!              \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q12, %[bias]                            \n"
                    "vdup.32 q13, %[bias]                            \n"
                    "vcvt.f32.s32   q10, q10                      \n"
                    "vcvt.f32.s32   q11, q11                      \n"
                    "vmla.f32 q12, q10, q7                     \n"
                    "vmla.f32 q13, q11, q7                     \n"
                    "vst1.32 {d24-d25}, [%[dout_ptr2]]!         \n"
                    "subs %[cnt], #1                                \n"
                    "vst1.32 {d26-d27}, [%[dout_ptr2]]!         \n"
                    "bne  2b                                        \n"
                //right
                    "1:                                          \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    \n"
                    "vld1.8 {d28-d29}, [%[mask]]        \n"
                    "vmov.u32 q8, #0                            \n"
                    "vmov.u32 q9, #0                            \n"
                    "vmov.u32 q10, #0                           \n"
                    "vmov.u32 q11, #0                           \n"
                    "vbif.8 d12, d11, d28        \n"
                    "vbif.8 d13, d11, d29        \n"
                    "vld1.8 {d14-d15}, [%[din_ptr1]]    \n"
                     //r0
                    "vmull.s8 q12, d12, d2                 \n" // q12 = d12 * w01
                    "vext.8 d30, d12, d13, #1               \n" //d10 = 12345678
                    "vext.8 d31, d12, d13, #2               \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr2]]    \n"
                    "vbif.8 d14, d11, d28        \n"
                    "vbif.8 d15, d11, d29        \n"
                    "vmlal.s8 q12, d30, d3                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                 \n" // q12 += d11 * w02
                    //r1
                    "vext.8 d30, d14, d15, #1           \n" //d10 = 00123456
                    "vext.8 d31, d14, d15, #2          \n" //d11 = 12345678
                    "vmull.s8 q13, d14, d2                 \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d14, d5                 \n" // q12 = d12 * w11
                    "vld1.8 {d14-d15}, [%[din_ptr3]]    \n"
                    "vbif.8 d12, d11, d28                 \n"
                    "vbif.8 d13, d11, d29                 \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d30, d3                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d6                 \n" // q12 += d10 * w00
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 \n" // q12 += d10 * w00
                    //r2
                    "vext.8 d30, d12, d13, #1               \n" //d10 = 00123456
                    "vext.8 d31, d12, d13, #2               \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d12, d5                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d12, d8                 \n" // q13 = d12 * w01
                    "vbif.8 d14, d11, d28                     \n"
                    "vbif.8 d15, d11, d29                     \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 \n" // q12 += d10 * w00
                    "vld1.32 {d28-d29}, [%[dout_ptr1]]!    \n"
                    "vld1.32 {d12-d13}, [%[dout_ptr1]]    \n"
                    "vld1.32 {d2-d3}, [%[rs_mask]]!     \n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]    \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                 \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d14, d15, #1           \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d14, d8                 \n" // q13 = d12 * w01
                    "vld1.32 {d14-d15}, [%[dout_ptr2]]!    \n"
                    "vld1.32 {d24-d25}, [%[dout_ptr2]]     \n"
                    "vmlal.s8 q13, d30, d9                 \n" // q13 += d10 * w00
                    "sub %[dout_ptr1], #16                  \n"
                    "sub %[dout_ptr2], #16                  \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                 \n" // q12 += d10 * w00
                    "vdup.32 q3, %[scale]                    \n"
                    "vdup.32 q4, %[bias]                     \n"
                    "vdup.32 q5, %[bias]                     \n"
                    "vcvt.f32.s32   q8, q8                      \n"
                    "vcvt.f32.s32   q9, q9                      \n"
                    "vmla.f32 q4, q8, q3                     \n"
                    "vmla.f32 q5, q9, q3                     \n"
                    "vbif q4, q14, q1                   \n"
                    "vbif q5, q6, q2                    \n"
                    "vst1.32 {d8-d9}, [%[dout_ptr1]]!         \n"
                    "vst1.32 {d10-d11}, [%[dout_ptr1]]!         \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q13, %[bias]                            \n"
                    "vdup.32 q14, %[bias]                            \n"
                    "vcvt.f32.s32   q10, q10                      \n"
                    "vcvt.f32.s32   q11, q11                      \n"
                    "vmla.f32 q13, q10, q3                     \n"
                    "vmla.f32 q14, q11, q3                     \n"
                    "vbif q13, q7, q1        \n"
                    "vbif q14, q12, q2       \n"
                    "vst1.32 {d26-d27}, [%[dout_ptr2]]!         \n"
                    "vst1.32 {d28-d29}, [%[dout_ptr2]]!         \n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2), \
                      [din_ptr3] "+r" (din_ptr3), \
                      [dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [scale] "+r" (scale_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask)
                    :"cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                      "q11", "q12", "q13", "q14", "q15"
                );
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
  }
}

void conv_depthwise_3x3s1p1_bias_int8_int8(int8_t* dout,
                                           const int8_t* din,
                                           const int8_t* weights,
                                           const float* scale,
                                           const float* bias,
                                           bool flag_bias,
                                           int flag_act,
                                           float* alpha,
                                           int num,
                                           int ch_in,
                                           int h_in,
                                           int w_in,
                                           int h_out,
                                           int w_out,
                                           ARMContext* ctx) {
  const unsigned char right_pad_idx[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const unsigned char right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, w_in * sizeof(int8_t));
  int8_t* write_ptr =
      reinterpret_cast<int8_t*>(ctx->workspace_data<int8_t>() + w_in);
  int threads = ctx->threads();

  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  int tile_w = (w_in + 7) >> 3;
  int cnt_col = tile_w - 2;

  unsigned int size_pad_right = (unsigned int)(w_in - 7 - (cnt_col << 3));

  int size_pad_bottom = h_out % 2;

  unsigned char rst_remain = (unsigned char)(w_out - ((cnt_col + 1) << 3));
  uint8x8_t vmask_result =
      vcgt_u8(vdup_n_u8(rst_remain), vld1_u8(right_pad_rst));
  uint8x16_t vmask_rp =
      vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
  unsigned char vmask[16];
  vst1q_u8(vmask, vmask_rp);

  unsigned char rmask[8];
  vst1_u8(rmask, vmask_result);

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    int8_t* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      int8_t* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;

      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
      float vscale[4] = {scale_val, scale_val, scale_val, scale_val};
      float vmax[4] = {-127.f, -127.f, -127.f, -127.f};
      float* vmax_ptr = vmax;
#ifdef __aarch64__
      int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
      int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
      int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

      int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
      int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
      int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

      int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
      int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
      int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

      float32x4_t v_bias = vdupq_n_f32(vbias[0]);
      float32x4_t v_scale = vdupq_n_f32(vscale[0]);
#endif

      int8_t* doutr0 = nullptr;
      int8_t* doutr1 = nullptr;

      const int8_t* dr0 = din_ch_ptr;
      const int8_t* dr1 = dr0 + w_in;
      const int8_t* dr2 = dr1 + w_in;
      const int8_t* dr3 = dr2 + w_in;

      const int8_t* din_ptr0 = nullptr;
      const int8_t* din_ptr1 = nullptr;
      const int8_t* din_ptr2 = nullptr;
      const int8_t* din_ptr3 = nullptr;

      for (int i = 0; i < h_in; i += 2) {
        //! process top pad pad_h = 1
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;
        unsigned char* rst_mask = rmask;
        unsigned char* val_mask = vmask;

        if (i == 0) {
          din_ptr0 = zero_ptr;
          din_ptr1 = dr0;
          din_ptr2 = dr1;
          din_ptr3 = dr2;
          dr0 = dr1;
          dr1 = dr2;
          dr2 = dr3;
          dr3 = dr2 + w_in;
        } else {
          dr0 = dr2;
          dr1 = dr3;
          dr2 = dr1 + w_in;
          dr3 = dr2 + w_in;
        }
        //! process bottom pad
        if (i + 3 > h_in) {
          switch (i + 3 - h_in) {
            case 3:
              din_ptr1 = zero_ptr;
            case 2:
              din_ptr2 = zero_ptr;
            case 1:
              din_ptr3 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 2 > h_out) {
          doutr1 = write_ptr;
        }
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr3]] \n"
                    "movi   v21.4s, #0x0\n"                                     /* v21 = 0 */
                // left
                    "ld1    {v0.8b}, [%[din_ptr0]], #8               \n"   /* load a(0,0)-a(0,7) to q0*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8               \n"   /* load a(1,0)-a(1.7) to q2*/
                    "ld1    {v1.8b}, [%[din_ptr0]]                   \n"   /* load a(0,8)-a(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]                   \n"   /* load a(1,8)-a(1,15) to q3*/
                    "movi   v10.4s, #0x0\n"                             /* init int32 acc v10 to 0 */
                    "movi   v11.4s, #0x0\n"                             /* init int32 acc v11 to 0 */
                    "movi   v12.4s, #0x0\n"                             /* init int32 acc v12 to 0 */
                    "movi   v13.4s, #0x0\n"                             /* init int32 acc v13 to 0 */
                    //r0
                    "smull  v18.8h,  %[v1].8b,  v0.8b   \n"   /* acc_r(0,0) = r(0,01234567) * w01 */
                    "ext v4.8b, v21.8b, v0.8b, #7       \n"   /* vext_s8(vzero, vin_r0, 7); r(0,00123456) */
                    "ext v5.8b, v0.8b, v1.8b, #1        \n"   /* vext_s8(vin_r0, vin_r0_h, 1); r(0,12345678) */
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a(2,0)-a(2,7) to q6*/
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a(3,0)-a(3,7) to q8*/
                    "smlal  v18.8h,  %[v0].8b,  v4.8b   \n"   /* acc_r(0,0) += r(0,00123456) * w00 */
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a(2,8)-a(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a(3,8)-a(3,15) to q9*/
                    "sub   %[din_ptr0], %[din_ptr0], #1 \n"
                    "sub   %[din_ptr1], %[din_ptr1], #1 \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v2].8b,  v5.8b \n"   /* acc_r00 += r(0,12345678) * w02 */
                    "ext v14.8b, v21.8b, v2.8b, #7       \n"   /* vext_s8(vzero, vin_r1, 7); r(1,00123456) */
                    "ext v15.8b, v2.8b, v3.8b, #1        \n"   /* vext_s8(vin_r1, vin_r1_h, 1); r(1,12345678) */
                    //r1
                    "sub   %[din_ptr2], %[din_ptr2], #1 \n"
                    "sub   %[din_ptr3], %[din_ptr3], #1 \n"
                    "smull  v19.8h,  %[v1].8b,  v2.8b   \n"   /* acc_r10 += r(1,01234567) * w01 */
                    "smlal  v18.8h,  %[v4].8b,  v2.8b   \n"   /* acc_r00 += r(1,01234567) * w11 */
                    "ext v4.8b, v21.8b, v6.8b, #7      \n"
                    "ext v5.8b, v6.8b, v7.8b, #1       \n"
                    "smlal  v19.8h,  %[v0].8b,  v14.8b   \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v3].8b,  v14.8b   \n"   /* acc_r00 += r(1, 00123456) * w10 */
                    "ld1    {v0.8b}, [%[din_ptr0]], #8  \n"   /* load a'(0,0)-a'(0,7) to q0 for prefetch*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8  \n"   /* load a'(1,0)-a'(1,7) to q2 for prefetch*/
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v2].8b,  v15.8b   \n"   /* acc_r10 += r(1,12345678) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b   \n"   /* acc_r00 += r(1,12345678) * w12 */
                    //r2
                    "ld1    {v1.8b}, [%[din_ptr0]]      \n"   /* load a'(0,8)-a'(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]      \n"   /* load a'(1,8)-a'(1,15) to q3*/
                    "smlal  v19.8h,  %[v4].8b,  v6.8b   \n"   /* acc_r10 += r(2,01234567) * w11 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v7].8b,  v6.8b   \n"   /* acc_r00 += r(2,01234567) * w21 */
                    "ext v14.8b, v21.8b, v8.8b, #7       \n"
                    "ext v15.8b, v8.8b, v9.8b, #1        \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v3].8b,  v4.8b  \n"   /* acc_r10 += r(2, 00123456) * w10 */
                    "smlal  v18.8h,  %[v6].8b,  v4.8b  \n"
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a'(2,0)-a'(2,7) to q6 for prefetch*/
                    "smlal  v19.8h,  %[v5].8b,  v5.8b  \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b  \n"   /* acc_r00 += r(2, 12345678) * w22 */
                    "ld1 {v4.4s}, [%[vmax_ptr]]        \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v12 += outr10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v13 += outr10.high*/
                    //r3
                    "smull  v19.8h,  %[v7].8b,  v8.8b   \n"   /* acc_r10 += r(3, 01234567) * w21 */
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a'(3,0)-a'(3,7) to q8 for prefetch*/
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a'(2,7)-a'(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a'(3,7)-a'(3,15) to q9*/
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v6].8b,  v14.8b   \n"   /* acc_r10 += r(3, 00123456) * w20) */
                    "dup   v20.4s, %[v_bias].s[0] \n "                /* dup bias to v20 */
                    "dup   v22.4s, %[v_bias].s[0] \n "                /* dup bias to v22 */
                    "scvtf  v10.4s, v10.4s               \n"   /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s               \n"   /* int32-> float32*/
                    "fmla   v20.4s, v10.4s, %[v_scale].4s       \n"  /* mul scale, add bias */
                    "fmla   v22.4s, v11.4s, %[v_scale].4s       \n"  /* mul scale, add bias */
                    "fcmge v11.4s, v20.4s, v4.4s             \n"
                    "bif v20.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v22.4s, v4.4s             \n"
                    "bif v22.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v20.4s, v20.4s                   \n" /* fp32 - int32 */
                    "fcvtas  v22.4s, v22.4s                   \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v20.4s                   \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v22.4s                   \n" /* int32-int16 */
                    "sqxtn   v20.8b, v21.8h                   \n" /* int16-int8 */
                    "st1    {v20.8b}, [%[ptr_out0]], #8     \n"  /* store q10, q11 -> ptr_out 0 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"    /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"    /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b   \n"    /* acc_r10 += r(3, 12345678) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"    /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"    /* v13 += acc_r10.high*/
                    "scvtf  v12.4s, v12.4s                \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                \n"  /* int32-> float32*/
                    "dup   v14.4s, %[v_bias].s[0] \n "    /* dup bias to v14 */
                    "dup   v15.4s, %[v_bias].s[0] \n "    /* dup bias to v15 */
                    "fmla   v14.4s, v12.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "fmla   v15.4s, v13.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "fcmge v11.4s, v14.4s, v4.4s             \n"
                    "bif v14.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v15.4s, v4.4s             \n"
                    "bif v15.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v14.4s, v14.4s                   \n" /* fp32 - int32 */
                    "fcvtas  v15.4s, v15.4s                   \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v14.4s                   \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v15.4s                   \n" /* int32-int16 */
                    "sqxtn   v14.8b, v21.8h                   \n" /* int16-int8 */
                    "st1    {v14.8b}, [%[ptr_out1]], #8     \n"  /* store q10, q11 -> ptr_out 0 */
                    "movi   v10.4s, #0x0\n"         /* clear 0 for v10 */
                    "movi   v11.4s, #0x0\n"         /* clear 0 for v11 */
                    "movi   v12.4s, #0x0\n"
                    "movi   v13.4s, #0x0\n"
                    "movi   v21.4s, #0x0\n"
                    "cmp  %[cnt], #1                \n"
                    "blt 3f                         \n"
                //mid
                    "1:                             \n"
                    "ext v4.8b, v0.8b, v1.8b, #1       \n"      /* vext_s8(vin_r0, vin_r0_h, 1); r(0, 12345678) */
                    "ext v5.8b, v0.8b, v1.8b, #2       \n"      /* vext_s8(vin_r0, vin_r0_h, 2); r(0, 23456789) */
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b  \n"      /* acc_r00 = r(0, 01234567) * w00 */
                    "ext v14.8b, v2.8b, v3.8b, #1      \n"      /* vext_s8(vin_r1, vin_r1_h, 1); r(1, 12345678) */
                    "ext v15.8b, v2.8b, v3.8b, #2      \n"      /* vext_s8(vin_r1, vin_r1_h, 1); r(1, 23456789) */
                    "smlal  v18.8h,  %[v1].8b,  v4.8b  \n"      /* acc_r00 += r(0, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v2].8b,  v5.8b   \n"
                    //r1
                    "ext v4.8b, v6.8b, v7.8b, #1       \n"     /* vext_s8(vin_r2, vin_r2_h, 1); r(2, 12345678) */
                    "ext v5.8b, v6.8b, v7.8b, #2       \n"     /* vext_s8(vin_r2, vin_r2_h, 2); r(2, 23456789) */
                    "smull  v19.8h,  %[v0].8b,  v2.8b   \n"     /* acc_r10 += r(1, 01234567) * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b   \n"     /* acc_r00 += r(1, 01234567) * w10 */
                    "ld1    {v0.8b}, [%[din_ptr0]], #8  \n"        /* load a'(0,7)-a'(0,15) to q0 for prefetch*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8  \n"        /* load a'(1,7)-a'(1,15) to q2 for prefetch*/
                    "smlal  v19.8h,  %[v1].8b,  v14.8b\n"       /* acc_r10 += r(1, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v4].8b,  v14.8b  \n"     /* acc_r00 += r(1, 12345678) * w11 */
                    "ld1    {v1.8b}, [%[din_ptr0]]      \n"     /* load a'(0,7)-a'(0,15) to q1 for prefetch */
                    "ld1    {v3.8b}, [%[din_ptr1]]      \n"     /* load a'(1,7)-a'(1,15) to q3 for prefetch */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v2].8b,  v15.8b  \n"       /* acc_r10 += r(1, 23456789) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b  \n"       /* acc_r00 += r(1, 23456789) * w12 */
                    //r2
                    "ext v14.8b, v8.8b, v9.8b, #1       \n"      /* vext_s8(vin_r3, vin_r3_h, 1); r(3, 12345678) */
                    "ext v15.8b, v8.8b, v9.8b, #2       \n"      /* vext_s8(vin_r3, vin_r3_h, 1); r(3, 23456789) */
                    "smlal  v19.8h,  %[v3].8b,  v6.8b   \n"     /* acc_r10 += r(2, 01234567) * w10 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r10.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r10.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b   \n"     /* acc_r00 += r(2, 01234567) * w20 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b\n"       /* acc_r10 += r(2, 12345678) * w11 */
                    "smlal  v18.8h,  %[v7].8b,  v4.8b\n"       /* acc_r00 += r(2, 12345678) * w21 */
                    "smlal  v19.8h,  %[v5].8b,  v5.8b\n"       /* acc_r10 += r(2, 23456789) * w12 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b  \n"     /* acc_r00 += r(2, 23456789) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high */
                    "ld1 {v4.4s}, [%[vmax_ptr]]                \n"
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b   \n"     /* acc_r10 = r(3, 01234567) * w20 */
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"     /* load a'(2,0)-a'(2,7) to q6 for prefetch */
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"     /* load a'(3,0)-a'(3,7) to q8 for prefetch */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high */
                    "smlal  v19.8h,  %[v7].8b,  v14.8b   \n"     /* acc_r10 += r(3, 12345678) * w21 */
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"     /* load a'(2,8)-a'(2,15) to q7 for prefetch */
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"     /* load a'(3,8)-a'(3,15) to q9 for prefetch */
                    "scvtf  v10.4s, v10.4s                 \n"  /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s                 \n"  /* int32-> float32*/
                    "dup   v20.4s, %[v_bias].s[0]                \n "
                    "dup   v22.4s, %[v_bias].s[0]                \n "
                    "fmla   v20.4s, v10.4s, %[v_scale].4s        \n"
                    "fmla   v22.4s, v11.4s, %[v_scale].4s        \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b   \n"     /* acc_r10 += r(3, 23456789) * w22 */
                    "fcmge v11.4s, v20.4s, v4.4s             \n"
                    "bif v20.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v22.4s, v4.4s             \n"
                    "bif v22.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v20.4s, v20.4s                  \n" /* fp32 - int32 */
                    "fcvtas  v22.4s, v22.4s                  \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v20.4s                  \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v22.4s                 \n" /* int32-int16 */
                    "sqxtn   v20.8b, v21.8h                  \n" /* int16-int8 */
                    "st1    {v20.8b}, [%[ptr_out0]], #8     \n"  /* store q10, q11 -> ptr_out 0 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high */
                    "subs %[cnt], %[cnt], #1                \n"
                    "scvtf  v12.4s, v12.4s                  \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                  \n"  /* int32-> float32*/
                    "dup   v18.4s, %[v_bias].s[0]                 \n "
                    "dup   v19.4s, %[v_bias].s[0]                 \n "
                    "fmla   v18.4s, v12.4s, %[v_scale].4s          \n"
                    "fmla   v19.4s, v13.4s, %[v_scale].4s          \n"
                    "fcmge v11.4s, v18.4s, v4.4s             \n"
                    "bif v18.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v19.4s, v4.4s             \n"
                    "bif v19.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v18.4s, v18.4s                   \n" /* fp32 - int32 */
                    "fcvtas  v19.4s, v19.4s                   \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v18.4s                   \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v19.4s                   \n" /* int32-int16 */
                    "sqxtn   v18.8b, v21.8h                   \n" /* int16-int8 */
                    "st1    {v18.8b}, [%[ptr_out1]], #8     \n"  /* store q10, q11 -> ptr_out 0 */
                    "movi   v10.4s, #0x0                    \n"
                    "movi   v11.4s, #0x0                    \n"
                    "movi   v12.4s, #0x0                    \n"
                    "movi   v13.4s, #0x0                    \n"
                    "movi   v21.4s, #0x0                    \n"
                    "bne 1b                                 \n"
                //right
                    "3:                                      \n" 
                    "ld1 {v20.8b}, [%[vmask]], #8            \n"
                    "ld1 {v22.8b}, [%[vmask]]                \n"
                    "bif v0.8b, v21.8b, v20.8b               \n"  /* a(0, 0) to a(0, 7) */
                    "bif v1.8b, v21.8b, v22.8b               \n"  /* a(0, 8) to a(0, 15) */
                    "bif v2.8b, v21.8b, v20.8b               \n"  /* a(1, 0) to a(1, 7) */
                    "bif v3.8b, v21.8b, v22.8b               \n"  /* a(1, 8) to a(1, 15) */
                    "ext v4.8b, v0.8b, v1.8b, #1             \n"  /* r(0, 12345678) */
                    "ext v5.8b, v0.8b, v1.8b, #2             \n"  /* r(0, 23456789) */
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b        \n"  /* acc_r00 += r(0, 01234567) * w00 */
                    "ext v14.8b, v2.8b, v3.8b, #1            \n"  /* r(1, 12345678) */
                    "ext v15.8b, v2.8b, v3.8b, #2            \n"  /* r(1, 23456789) */
                    "bif v6.8b, v21.8b, v20.8b               \n"  /* a(2, 0) to a(0, 7) */
                    "bif v7.8b, v21.8b, v22.8b               \n"  /* a(2, 8) to a(2, 15) */
                    "smlal  v18.8h,  %[v1].8b,  v4.8b        \n"  /* acc_r00 += r(0, 12345678) * w01 */
                    "bif v8.8b, v21.8b, v20.8b               \n"  /* a(3, 0) to a(3, 7) */ 
                    "bif v9.8b, v21.8b, v22.8b               \n"  /* a(3, 8) to a(3, 15) */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v2].8b,  v5.8b        \n"  /* acc_r00 = r(0, 23456789) * w02 */
                    //r1
                    "ext v4.8b, v6.8b, v7.8b, #1            \n"
                    "ext v5.8b, v6.8b, v7.8b, #2            \n"
                    "smull  v19.8h,  %[v0].8b,  v2.8b        \n"  /* acc_r10 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b        \n"  /* acc_r00 = 01234567 * w00 */
                    "ld1 {v20.8b}, [%[rmask]], #8           \n"
                    "smlal  v19.8h,  %[v1].8b,  v14.8b       \n"  /* acc_r10 += r(1, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v4].8b,  v14.8b       \n"  /* acc_r00 += r(1, 12345678) * w11 */
                    "ld1 {v0.8b}, [%[ptr_out0]]         \n"  /* load original ptr_out0 low */
                    "ld1 {v2.8b}, [%[ptr_out1]]         \n"  /* load original ptr_out1 low */
                    "saddw   v12.4s, v12.4s, v19.4h          \n"  /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h         \n"  /* v13 += acc_r10.high */
                    "smull  v19.8h,  %[v2].8b,  v15.8b       \n"  /* acc_r10 = r(1, 23456789) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b       \n"  /* acc_r00 = r(1, 23456789) * w12 */
                    //r2
                    "ext v14.8b, v8.8b, v9.8b, #1             \n"  /* r(3, 12345678) */
                    "ext v15.8b, v8.8b, v9.8b, #2             \n"  /* r(3, 23456789) */
                    "smlal  v19.8h,  %[v3].8b,  v6.8b        \n"  /* acc_r10 += r(2, 01234567) * w10 */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b        \n"  /* acc_r00 = r(2, 01234567) * w20 */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v13 += outr00.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b      \n"   /* acc_r10 += r(2, 12345678) * w11 */
                    "smlal  v18.8h,  %[v7].8b,  v4.8b      \n"   /* acc_r00 += r(2, 12345678) * w21 */
                    "smlal  v19.8h,  %[v5].8b,  v5.8b      \n"   /* acc_r10 += r(2, 23456789) * w12 */
                    "saddw   v10.4s, v10.4s, v18.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h        \n"   /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b      \n"   /* acc_r00 += r(2, 23456789) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += acc_r00.high*/
                    "ld1 {v4.4s}, [%[vmax_ptr]]                \n"
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b       \n"   /* acc_r10 += r(3, 01234567) * w20 */
                    "saddw   v10.4s, v10.4s, v18.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h        \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v7].8b,  v14.8b       \n"   /* acc_r10 += r(3, 12345678) * w21 */
                    "scvtf  v10.4s, v10.4s                  \n"  /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s                  \n"  /* int32-> float32*/
                    "dup   v6.4s, %[v_bias].s[0]                 \n "
                    "dup   v7.4s, %[v_bias].s[0]                 \n "
                    "fmla   v6.4s, v10.4s, %[v_scale].4s         \n"
                    "fmla   v7.4s, v11.4s, %[v_scale].4s         \n"
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += acc_r00.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b       \n"   /* acc_r10 = r(3, 23456789) * w22 */
                    "fcmge v11.4s, v6.4s, v4.4s             \n"
                    "bif v6.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v7.4s, v4.4s             \n"
                    "bif v7.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v6.4s, v6.4s                   \n" /* fp32 - int32 */
                    "fcvtas  v7.4s, v7.4s                   \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v6.4s                   \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v7.4s                   \n" /* int32-int16 */
                    "sqxtn   v6.8b, v21.8h                   \n" /* int16-int8 */
                    "bif v6.8b, v0.8b, v20.8b           \n"
                    "st1    {v6.8b}, [%[ptr_out0]], #8     \n"  /* store q10, q11 -> ptr_out 0 */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += outr00.high*/
                    "scvtf  v12.4s, v12.4s                 \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                 \n"  /* int32-> float32*/
                    "dup   v8.4s, %[v_bias].s[0]                 \n"
                    "dup   v9.4s, %[v_bias].s[0]                 \n"
                    "fmla   v8.4s, v12.4s, %[v_scale].4s  \n"
                    "fmla   v9.4s, v13.4s, %[v_scale].4s  \n"
                    "fcmge v11.4s, v8.4s, v4.4s             \n"
                    "bif v8.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v9.4s, v4.4s             \n"
                    "bif v9.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v8.4s, v8.4s                   \n" /* fp32 - int32 */
                    "fcvtas  v9.4s, v9.4s                   \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v8.4s                   \n" /* int32-int16 */
                    "sqxtn2  v21.8h, v9.4s                   \n" /* int32-int16 */
                    "sqxtn   v8.8b, v21.8h                   \n" /* int16-int8 */
                    "bif v8.8b, v2.8b, v20.8b                \n"
                    "st1    {v8.8b}, [%[ptr_out1]], #8       \n"  /* store q10, q11 -> ptr_out 0 */
                    :   [cnt]"+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [din_ptr3] "+r"(din_ptr3), [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
                        [vmask] "+r" (val_mask), [rmask] "+r" (rst_mask)
                    :   [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), \
                        [vmax_ptr]"r"(vmax_ptr), \
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22), \
                        [v_bias]"w"(v_bias), [v_scale] "w"(v_scale)
                    :"cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
                      "v10", "v11", "v12","v13","v14","v15", \
                      "v18",\
                      "v19", "v20", "v21", "v22"
                );
#else
                // store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                     \n"
                    "pld [%[din_ptr1]]                     \n"
                    "pld [%[din_ptr2]]                     \n"
                    "pld [%[din_ptr3]]                     \n"
                    "vdup.s8     d2, d0[0]                 \n"
                    "vdup.s8     d3, d0[1]                 \n"
                    "vdup.s8     d4, d0[2]                 \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]       \n"
                    "vmov.u32 d11, #0                      \n"
                    "vmov.u32 q8, #0                       \n"
                    "vmov.u32 q9, #0                       \n"
                    "vmov.u32 q10, #0                      \n"
                    "vmov.u32 q11, #0                      \n"
                    //r0
                    "vmull.s8 q12, d12, d3                 \n" // q12 = d12 * w01
                    "vext.8     d30, d11, d12, #7          \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          \n" //d11 = 12345678
                    "vld1.8 {d12-d13}, [%[din_ptr1]]       \n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]       \n"
                    "vdup.s8     d5, d0[3]                 \n"
                    "vdup.s8     d6, d0[4]                 \n"
                    "vmlal.s8 q12, d30, d2                 \n" // q12 += d10 * w00
                    "vdup.s8     d7, d0[5]                 \n"
                    "add %[din_ptr0], #7                   \n"
                    "add %[din_ptr1], #7                   \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                 \n" // q12 += d11 * w02
                    //r1
                    "vext.8     d30, d11, d12, #7          \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d3                 \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d12, d6                 \n" // q12 = d12 * w11
                    "vld1.8 {d12-d13}, [%[din_ptr3]]       \n"
                    "vdup.s8     d8, d0[6]                 \n"
                    "vdup.s8     d9, d0[7]                 \n"
                    "vdup.s8     d10, d1[0]                \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d30, d2                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d5                 \n" // q12 += d10 * w00
                    "add %[din_ptr2], #7                   \n"
                    "add %[din_ptr3], #7                   \n"
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 \n" // q12 += d10 * w00
                    //r2
                    "vext.8     d30, d11, d14, #7          \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #1          \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d14, d6                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d9                 \n" // q13 = d12 * w01
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d5                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d8                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d11, d12, #7          \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d12, d9                 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                     \n"
                    "pld [%[din_ptr1]]                     \n"
                    "vmlal.s8 q13, d30, d8                 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                     \n"
                    "pld [%[din_ptr3]]                     \n"
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                \n" // q12 += d10 * w00
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q7, %[scale]                 \n"
                    "vdup.32 q14, %[bias]                 \n"
                    "vdup.32 q15, %[bias]                 \n"
                    "vcvt.f32.s32   q8, q8                \n"
                    "vcvt.f32.s32   q9, q9                \n"
                    "vmla.f32 q14, q8, q7                 \n"
                    "vmla.f32 q15, q9, q7                 \n"
                    "vmov.f32 q8, #-0.5                   \n"
                    "vmov.f32 q9, #0.5                    \n"
                    "vcgt.f32   q1, q14, q8               \n"
                    "vbif.f32   q9, q8, q1                \n"
                    "vadd.f32   q14, q14, q9              \n"
                    "vmov.f32   q9, #0.5                  \n"
                    "vcgt.f32   q2, q15, q8               \n"
                    "vbif.f32   q9, q8, q2                \n"
                    "vadd.f32   q15, q15, q9              \n"
                    "vld1.32 {d2-d3}, [%[vmax]]           \n"
                    "vcge.f32 q3, q14, q1                 \n" /* data >= -127 */
                    "vbif q14, q1, q3                     \n" /* choose data */
                    "vcge.f32 q3, q15, q1                 \n" /* data >= -127 */
                    "vbif q15, q1, q3                     \n" /* choose data */
                    "vcvt.s32.f32  q1, q14                \n" /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q15                \n" /* fp32 to int32 */
                    "vqmovn.s32 d28, q1                   \n" /* int32 to int16 */
                    "vqmovn.s32 d29, q2                   \n" /* int32 to int16 */
                    "vqmovn.s16 d10, q14                  \n" /* int16 to int8 */
                    "vst1.8    {d10}, [%[dout_ptr1]]! \n"
                    "vdup.32 q12, %[bias]             \n"
                    "vdup.32 q13, %[bias]             \n"
                    "vcvt.f32.s32   q10, q10          \n"
                    "vcvt.f32.s32   q11, q11          \n"
                    "vmla.f32 q12, q10, q7            \n"
                    "vmla.f32 q13, q11, q7            \n"
                    "vmov.f32 q8, #-0.5               \n"
                    "vmov.f32 q9, #0.5                \n"
                    "vcgt.f32   q1, q12, q8           \n"
                    "vbif.f32   q9, q8, q1            \n"
                    "vadd.f32   q12, q12, q9          \n"
                    "vmov.f32   q9, #0.5              \n"
                    "vcgt.f32   q2, q13, q8           \n"
                    "vbif.f32   q9, q8, q2            \n"
                    "vadd.f32   q13, q13, q9          \n"
                    "vld1.32 {d2-d3}, [%[vmax]]       \n"
                    "vcge.f32 q3, q12, q1             \n" /* data >= -127 */
                    "vbif q12, q1, q3                 \n" /* choose data */
                    "vcge.f32 q3, q13, q1             \n" /* data >= -127 */
                    "vbif q13, q1, q3                 \n" /* choose data */
                    "vcvt.s32.f32  q1, q12            \n" /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q13            \n" /* fp32 to int32 */
                    "vqmovn.s32 d24, q1               \n" /* int32 to int16 */
                    "vqmovn.s32 d25, q2               \n" /* int32 to int16 */
                    "vqmovn.s16 d9, q12               \n" /* int16 to int8 */
                    "vst1.8    {d9}, [%[dout_ptr2]]!  \n"
                    "cmp %[cnt], #1                   \n"
                    "blt 1f                           \n"
                //mid
                    "2:                                  \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]     \n"
                    "vdup.s8     d2, d0[0]               \n"
                    "vdup.s8     d3, d0[1]               \n"
                    "vdup.s8     d4, d0[2]               \n"
                    "vdup.s8     d5, d0[3]               \n"
                    "vdup.s8     d6, d0[4]               \n"
                    "vdup.s8     d7, d0[5]               \n"
                    "vdup.s8     d8, d0[6]               \n"
                    "vdup.s8     d9, d0[7]               \n"
                    "vdup.s8     d10, d1[0]              \n"
                    "vmov.u32 q8, #0                     \n"
                    "vmov.u32 q9, #0                     \n"
                    "vmov.u32 q10, #0                    \n"
                    "vmov.u32 q11, #0                    \n"
                     //r0
                    "vmull.s8 q12, d12, d2                 \n" // q12 = d12 * w01
                    "vext.8     d30, d12, d13, #1          \n" //d10 = 12345678
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr1]]       \n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]       \n"
                    "vmlal.s8 q12, d30, d3                 \n" // q12 += d10 * w00
                    "add %[din_ptr0], #8                   \n"
                    "add %[din_ptr1], #8                   \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                 \n" // q12 += d11 * w02
                    //r1
                    "vext.8     d30, d12, d13, #1          \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d2                 \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d12, d5                 \n" // q12 = d12 * w11
                    "vld1.8 {d12-d13}, [%[din_ptr3]]       \n"
                    "vmlal.s8 q13, d30, d3                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d30, d6                 \n" // q12 += d10 * w00
                    "add %[din_ptr2], #8                   \n"
                    "add %[din_ptr3], #8                   \n"
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 \n" // q12 += d10 * w00
                    //r2
                    "vext.8     d30, d14, d15, #1          \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d14, d5                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d8                 \n" // q13 = d12 * w01
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 \n" // q12 += d10 * w00
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d10                \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d12, d13, #1          \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d12, d8                 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                     \n"
                    "pld [%[din_ptr1]]                     \n"
                    "vmlal.s8 q13, d30, d9                 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                     \n"
                    "pld [%[din_ptr3]]                     \n"
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                \n" // q12 += d10 * w00
                    "vdup.32 q7, %[scale]                  \n"
                    "vdup.32 q14, %[bias]                  \n"
                    "vdup.32 q15, %[bias]                  \n"
                    "vcvt.f32.s32   q8, q8                 \n"
                    "vcvt.f32.s32   q9, q9                 \n"
                    "vmla.f32 q14, q8, q7                  \n"
                    "vmla.f32 q15, q9, q7                  \n"
                    "vmov.f32 q8, #-0.5                    \n"
                    "vmov.f32 q9, #0.5                     \n"
                    "vcgt.f32   q1, q14, q8                \n"
                    "vbif.f32   q9, q8, q1                 \n"
                    "vadd.f32   q14, q14, q9               \n"
                    "vmov.f32   q9, #0.5                   \n"
                    "vcgt.f32   q2, q15, q8                \n"
                    "vbif.f32   q9, q8, q2                 \n"
                    "vadd.f32   q15, q15, q9               \n"
                    "vld1.32 {d2-d3}, [%[vmax]]            \n"
                    "vcge.f32 q3, q14, q1                  \n" /* data >= -127 */
                    "vbif q14, q1, q3                      \n" /* choose data */
                    "vcge.f32 q3, q15, q1                  \n" /* data >= -127 */
                    "vbif q15, q1, q3                      \n" /* choose data */
                    "vcvt.s32.f32  q1, q14                 \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q15                 \n"  /* fp32 to int32 */
                    "vqmovn.s32 d28, q1                    \n"  /* int32 to int16 */
                    "vqmovn.s32 d29, q2                    \n"  /* int32 to int16 */
                    "vqmovn.s16 d10, q14                   \n"   /* int16 to int8 */
                    "vst1.8    {d10}, [%[dout_ptr1]]!      \n"
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q12, %[bias]                  \n"
                    "vdup.32 q13, %[bias]                  \n"
                    "vcvt.f32.s32   q10, q10               \n"
                    "vcvt.f32.s32   q11, q11               \n"
                    "vmla.f32 q12, q10, q7                 \n"
                    "vmla.f32 q13, q11, q7                 \n"
                    "vmov.f32 q8, #-0.5                    \n"
                    "vmov.f32 q9, #0.5                     \n"
                    "vcgt.f32   q1, q12, q8                \n"
                    "vbif.f32   q9, q8, q1                 \n"
                    "vadd.f32   q12, q12, q9               \n"
                    "vmov.f32   q9, #0.5                   \n"
                    "vcgt.f32   q2, q13, q8                \n"
                    "vbif.f32   q9, q8, q2                 \n"
                    "vadd.f32   q13, q13, q9               \n"
                    "vld1.32 {d2-d3}, [%[vmax]]            \n"
                    "vcge.f32 q3, q12, q1                  \n" /* data >= -127 */
                    "vbif q12, q1, q3                      \n" /* choose data */                    
                    "vcge.f32 q3, q13, q1                  \n" /* data >= -127 */
                    "vbif q13, q1, q3                      \n" /* choose data */
                    "vcvt.s32.f32  q1, q12                 \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q13                 \n"  /* fp32 to int32 */
                    "vqmovn.s32 d24, q1                    \n"  /* int32 to int16 */
                    "vqmovn.s32 d25, q2                    \n"  /* int32 to int16 */
                    "vqmovn.s16 d9, q12                    \n"   /* int16 to int8 */
                    "vst1.8    {d9}, [%[dout_ptr2]]!       \n"
                    "subs %[cnt], #1                       \n"
                    "bne  2b                               \n"
                //right
                    "1:                                  \n"
                    "vdup.s8     d2, d0[0]               \n"
                    "vdup.s8     d3, d0[1]               \n"
                    "vdup.s8     d4, d0[2]               \n"
                    "vdup.s8     d5, d0[3]               \n"
                    "vdup.s8     d6, d0[4]               \n"
                    "vdup.s8     d7, d0[5]               \n"
                    "vdup.s8     d8, d0[6]               \n"
                    "vdup.s8     d9, d0[7]               \n"
                    "vdup.s8     d10, d1[0]              \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]     \n"
                    "vld1.8 {d28-d29}, [%[mask]]         \n"
                    "vmov.u32 q8, #0                     \n"
                    "vmov.u32 q9, #0                     \n"
                    "vmov.u32 q10, #0                    \n"
                    "vmov.u32 q11, #0                    \n"
                    "vbif.8 d12, d11, d28                \n"
                    "vbif.8 d13, d11, d29                \n"
                    "vld1.8 {d14-d15}, [%[din_ptr1]]     \n"
                     //r0
                    "vmull.s8 q12, d12, d2              \n" // q12 = d12 * w01
                    "vext.8 d30, d12, d13, #1           \n" //d10 = 12345678
                    "vext.8 d31, d12, d13, #2           \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr2]]    \n"
                    "vbif.8 d14, d11, d28               \n"
                    "vbif.8 d15, d11, d29               \n"
                    "vmlal.s8 q12, d30, d3              \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24              \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25              \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4              \n" // q12 += d11 * w02
                    //r1
                    "vext.8 d30, d14, d15, #1           \n" //d10 = 00123456
                    "vext.8 d31, d14, d15, #2           \n" //d11 = 12345678
                    "vmull.s8 q13, d14, d2              \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d14, d5              \n" // q12 = d12 * w11
                    "vld1.8 {d14-d15}, [%[din_ptr3]]    \n"
                    "vbif.8 d12, d11, d28               \n"
                    "vbif.8 d13, d11, d29               \n"
                    "vaddw.s16 q8, q8, d24              \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25              \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d30, d3              \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d6              \n" // q12 += d10 * w00
                    "vaddw.s16 q10, q10, d26            \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27            \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4              \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7              \n" // q12 += d10 * w00
                    //r2
                    "vext.8 d30, d12, d13, #1           \n" //d10 = 00123456
                    "vext.8 d31, d12, d13, #2           \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24              \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25              \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d12, d5              \n" // q13 = d12 * w01
                    "vmull.s8 q12, d12, d8              \n" // q13 = d12 * w01
                    "vbif.8 d14, d11, d28               \n"
                    "vbif.8 d15, d11, d29               \n"
                    "vaddw.s16 q10, q10, d26            \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27            \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6              \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9              \n" // q12 += d10 * w00
                    "vld1.8 d12, [%[rs_mask]]!          \n"
                    "vaddw.s16 q8, q8, d24              \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25              \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d31, d7              \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10             \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d14, d15, #1          \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d14, d8                 \n" // q13 = d12 * w01
                    "vmlal.s8 q13, d30, d9                 \n" // q13 += d10 * w00
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                \n" // q12 += d10 * w00
                    "vdup.32 q3, %[scale]                  \n"
                    "vdup.32 q4, %[bias]                   \n"
                    "vdup.32 q5, %[bias]                   \n"
                    "vcvt.f32.s32   q8, q8                 \n"
                    "vcvt.f32.s32   q9, q9                   \n"
                    "vmla.f32 q4, q8, q3                     \n"
                    "vmla.f32 q5, q9, q3                     \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q13, %[bias]                    \n"
                    "vdup.32 q14, %[bias]                    \n"
                    "vcvt.f32.s32   q10, q10                 \n"
                    "vcvt.f32.s32   q11, q11                 \n"
                    "vmla.f32 q13, q10, q3                   \n"
                    "vmla.f32 q14, q11, q3                   \n"
                    "vmov.f32 q8, #-0.5                      \n"
                    "vmov.f32 q9, #0.5                       \n"
                    "vcgt.f32   q1, q4, q8                   \n"
                    "vbif.f32   q9, q8, q1                    \n"
                    "vadd.f32   q4, q4, q9                    \n"
                    "vmov.f32   q9, #0.5                      \n"
                    "vcgt.f32   q2, q5, q8                    \n"
                    "vbif.f32   q9, q8, q2                    \n"
                    "vadd.f32   q5, q5, q9                    \n"
                    "vld1.32 {d2-d3}, [%[vmax]]               \n"
                    "vcge.f32 q3, q4, q1                      \n" /* data >= -127 */
                    "vbif q4, q1, q3                          \n" /* choose data */
                    "vcge.f32 q3, q5, q1                      \n" /* data >= -127 */
                    "vbif q5, q1, q3                          \n" /* choose data */
                    "vcvt.s32.f32  q1, q4                     \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q5                     \n"  /* fp32 to int32 */
                    "vqmovn.s32 d8, q1                        \n"  /* int32 to int16 */
                    "vqmovn.s32 d9, q2                        \n"  /* int32 to int16 */
                    "vqmovn.s16 d7, q4                        \n"   /* int16 to int8 */
                    "vld1.8 d10, [%[dout_ptr1]]               \n"
                    "vbif.8 d7, d10, d12                      \n"
                    "vst1.8    {d7}, [%[dout_ptr1]]!          \n"
                    "vmov.f32 q8, #-0.5                       \n"
                    "vmov.f32 q9, #0.5                        \n"
                    "vcgt.f32   q1, q13, q8                   \n"
                    "vbif.f32   q9, q8, q1                    \n"
                    "vadd.f32   q13, q13, q9                  \n"
                    "vmov.f32   q9, #0.5                      \n"
                    "vcgt.f32   q2, q14, q8                   \n"
                    "vbif.f32   q9, q8, q2                    \n"
                    "vadd.f32   q14, q14, q9                  \n"
                    "vld1.32 {d2-d3}, [%[vmax]]               \n"
                    "vcge.f32 q3, q13, q1                     \n" /* data >= -127 */
                    "vbif q13, q1, q3                         \n" /* choose data */
                    "vcge.f32 q3, q14, q1                     \n" /* data >= -127 */
                    "vbif q14, q1, q3                         \n" /* choose data */
                    "vcvt.s32.f32  q1, q13                    \n" /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q14                    \n" /* fp32 to int32 */
                    "vqmovn.s32 d26, q1                       \n" /* int32 to int16 */
                    "vqmovn.s32 d27, q2                       \n" /* int32 to int16 */
                    "vqmovn.s16 d10, q13                      \n" /* int16 to int8 */
                    "vld1.8 d14, [%[dout_ptr2]]               \n"
                    "vbif.8 d10, d14, d12                     \n"
                    "vst1.8    {d10}, [%[dout_ptr2]]!         \n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2), \
                      [din_ptr3] "+r" (din_ptr3), \
                      [dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [scale] "+r" (scale_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask), [vmax]"r"(vmax_ptr)
                    :"cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                      "q11", "q12", "q13", "q14", "q15"
                );
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
  }
}

void conv_depthwise_3x3s1p0_bias_int8_float(float* dout,
                                            const int8_t* din,
                                            const int8_t* weights,
                                            const float* scale,
                                            const float* bias,
                                            bool flag_bias,
                                            int flag_act,
                                            float* alpha,
                                            int num,
                                            int ch_in,
                                            int h_in,
                                            int w_in,
                                            int h_out,
                                            int w_out,
                                            ARMContext* ctx) {
  const unsigned char right_pad_idx[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, w_in * sizeof(int8_t));
  float* write_ptr =
      reinterpret_cast<float*>(ctx->workspace_data<int8_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;

  int tile_w = w_out >> 3;
  int cnt_col = tile_w;
  unsigned int size_pad_right = (unsigned int)(w_in - (cnt_col << 3));
  unsigned int rst_remain = (w_out - ((cnt_col) << 3));
  uint32x4_t vmask_result1 =
      vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
  uint32x4_t vmask_result2 =
      vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

  uint8x16_t vmask_rp =
      vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
  unsigned char vmask[16];
  vst1q_u8(vmask, vmask_rp);
  unsigned int rmask[8];
  vst1q_u32(rmask, vmask_result1);
  vst1q_u32(rmask + 4, vmask_result2);

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
      float vscale[4] = {scale_val, scale_val, scale_val, scale_val};

#ifdef __aarch64__
      int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
      int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
      int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

      int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
      int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
      int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

      int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
      int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
      int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

      float32x4_t v_bias = vdupq_n_f32(vbias[0]);
      float32x4_t v_scale = vdupq_n_f32(vscale[0]);
#endif

      float* doutr0 = nullptr;
      float* doutr1 = nullptr;

      const int8_t* dr0 = din_ch_ptr;
      const int8_t* dr1 = dr0 + w_in;
      const int8_t* dr2 = dr1 + w_in;
      const int8_t* dr3 = dr2 + w_in;

      const int8_t* din_ptr0 = nullptr;
      const int8_t* din_ptr1 = nullptr;
      const int8_t* din_ptr2 = nullptr;
      const int8_t* din_ptr3 = nullptr;

      for (int i = 0; i < h_out; i += 2) {
        //! process top pad pad_h = 1
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;

        unsigned int* rst_mask = rmask;
        unsigned char* val_mask = vmask;

        dr0 = dr2;
        dr1 = dr3;
        dr2 = dr1 + w_in;
        dr3 = dr2 + w_in;

        //! process bottom pad
        if (i + 3 > h_in) {
          switch (i + 3 - h_in) {
            case 3:
              din_ptr1 = zero_ptr;
            case 2:
              din_ptr2 = zero_ptr;
            case 1:
              din_ptr3 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 2 > h_out) {
          doutr1 = write_ptr;
        }
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
                asm volatile(
                    
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr3]] \n"
                    "movi   v21.4s, #0x0\n"                                     /* v21 = 0 */
                // middle
                    "ld1    {v0.8b}, [%[din_ptr0]], #8               \n"   /* load a(0,0)-a(0,7) to q0*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8               \n"   /* load a(1,0)-a(1.7) to q2*/
                    "ld1    {v1.8b}, [%[din_ptr0]]                   \n"   /* load a(0,8)-a(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]                   \n"   /* load a(1,8)-a(1,15) to q3*/
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a(2,0)-a(2,7) to q6*/
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a(3,0)-a(3,7) to q8*/
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a(2,8)-a(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a(3,8)-a(3,15) to q9*/
                    "movi   v10.4s, #0x0\n"                 /* init int32 acc v10 to 0 */
                    "movi   v11.4s, #0x0\n"                 /* init int32 acc v11 to 0 */
                    "movi   v12.4s, #0x0\n"                   /* init int32 acc v12 to 0 */
                    "movi   v13.4s, #0x0\n"                   /* init int32 acc v13 to 0 */
                    "1:                            \n"
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b   \n"
                    "ext v4.8b, v0.8b, v1.8b, #1        \n"
                    "ext v5.8b, v0.8b, v1.8b, #2        \n"
                    "smlal  v18.8h,  %[v1].8b,  v4.8b   \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v2].8b,  v5.8b \n"   /* acc_r00 += r(0,12345678) * w02 */
                    "ext v14.8b, v2.8b, v3.8b, #1       \n"
                    "ext v15.8b, v2.8b, v3.8b, #2       \n"
                    //r1
                    "smull  v19.8h,  %[v0].8b,  v2.8b   \n"
                    "smlal  v18.8h,  %[v3].8b,  v2.8b   \n"
                    "ext v4.8b, v6.8b, v7.8b, #1       \n"
                    "ext v5.8b, v6.8b, v7.8b, #2       \n"
                    "smlal  v19.8h,  %[v1].8b,  v14.8b   \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v4].8b,  v14.8b   \n"
                    "ld1    {v0.8b}, [%[din_ptr0]], #8  \n"   /* load a'(0,0)-a'(0,7) to q0 for prefetch*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8  \n"   /* load a'(1,0)-a'(1,7) to q2 for prefetch*/
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v2].8b,  v15.8b   \n"   /* acc_r10 += r(1,12345678) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b   \n"   /* acc_r00 += r(1,12345678) * w12 */
                    //r2
                    "ld1    {v1.8b}, [%[din_ptr0]]      \n"   /* load a'(0,8)-a'(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]      \n"   /* load a'(1,8)-a'(1,15) to q3*/
                    "smlal  v19.8h,  %[v3].8b,  v6.8b   \n"  /* acc_r10 += r(2,01234567) * w11 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b   \n"
                    "ext v14.8b, v8.8b, v9.8b, #1        \n"
                    "ext v15.8b, v8.8b, v9.8b, #2        \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b  \n"
                    "smlal  v18.8h,  %[v7].8b,  v4.8b  \n"
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a'(2,0)-a'(2,7) to q6 for prefetch*/
                    "smlal  v19.8h,  %[v5].8b,  v5.8b  \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b  \n"   /* acc_r00 += r(2, 12345678) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v12 += outr10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v13 += outr10.high*/
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b   \n"   /* acc_r10 += r(3, 01234567) * w21 */
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a'(3,0)-a'(3,7) to q8 for prefetch*/
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a'(2,7)-a'(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a'(3,7)-a'(3,15) to q9*/
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v7].8b,  v14.8b   \n"   /* acc_r10 += r(3, 00123456) * w20) */
                    "dup   v20.4s, %[v_bias].s[0] \n "                /* dup bias to v20 */
                    "dup   v22.4s, %[v_bias].s[0] \n "                /* dup bias to v22 */
                    "scvtf  v10.4s, v10.4s               \n"   /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s               \n"   /* int32-> float32*/
                    "fmla   v20.4s, v10.4s, %[v_scale].4s       \n"  /* mul scale, add bias */
                    "fmla   v22.4s, v11.4s, %[v_scale].4s       \n"  /* mul scale, add bias */
                    "stp    q20, q22, [%[ptr_out0]], #32 \n"  /* store q10, q11 -> ptr_out 0 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"    /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"    /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b   \n"    /* acc_r10 += r(3, 12345678) * w22 */
                    "movi   v10.4s, #0x0\n"         /* clear 0 for v10 */
                    "movi   v11.4s, #0x0\n"         /* clear 0 for v11 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"    /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"    /* v13 += acc_r10.high*/
                    "scvtf  v12.4s, v12.4s                \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                \n"  /* int32-> float32*/
                    "dup   v14.4s, %[v_bias].s[0] \n "    /* dup bias to v14 */
                    "dup   v15.4s, %[v_bias].s[0] \n "    /* dup bias to v15 */
                    "fmla   v14.4s, v12.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "fmla   v15.4s, v13.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "stp     q14, q15, [%[ptr_out1]], #32   \n" /* store q10, q11 -> ptr_out  1 */
                    "movi   v12.4s, #0x0\n"
                    "movi   v13.4s, #0x0\n"
                    "subs %[cnt], %[cnt], #1                \n"
                    "bne 1b                                 \n"
                //right
                    "3:                                      \n" 
                    "ld1 {v20.8b}, [%[vmask]], #8            \n"
                    "ld1 {v22.8b}, [%[vmask]]                \n"
                    "bif v0.8b, v21.8b, v20.8b               \n"  /* a(0, 0) to a(0, 7) */
                    "bif v1.8b, v21.8b, v22.8b               \n"  /* a(0, 8) to a(0, 15) */
                    "bif v2.8b, v21.8b, v20.8b               \n"  /* a(1, 0) to a(1, 7) */
                    "bif v3.8b, v21.8b, v22.8b               \n"  /* a(1, 8) to a(1, 15) */
                    "ext v4.8b, v0.8b, v1.8b, #1             \n"  /* r(0, 12345678) */
                    "ext v5.8b, v0.8b, v1.8b, #2             \n"  /* r(0, 23456789) */
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b        \n"  /* acc_r00 += r(0, 01234567) * w00 */
                    "ext v14.8b, v2.8b, v3.8b, #1            \n"  /* r(1, 12345678) */
                    "ext v15.8b, v2.8b, v3.8b, #2            \n"  /* r(1, 23456789) */
                    "bif v6.8b, v21.8b, v20.8b               \n"  /* a(2, 0) to a(0, 7) */  /* r(2, 12345678) */
                    "bif v7.8b, v21.8b, v22.8b               \n"  /* a(2, 8) to a(2, 15) */ /* r(2, 23456789) */
                    "smlal  v18.8h,  %[v1].8b,  v4.8b        \n"  /* acc_r00 += r(0, 12345678) * w01 */
                    "bif v8.8b, v21.8b, v20.8b               \n"  /* a(3, 0) to a(3, 7) */ 
                    "bif v9.8b, v21.8b, v22.8b               \n"  /* a(3, 8) to a(3, 15) */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v2].8b,  v5.8b        \n"  /* acc_r00 = r(0, 23456789) * w02 */
                    //r1
                    "ext v4.8b, v6.8b, v7.8b, #1            \n"
                    "ext v5.8b, v6.8b, v7.8b, #2            \n"
                    "smull  v19.8h,  %[v0].8b,  v2.8b        \n"  /* acc_r10 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b        \n"  /* acc_r00 = 01234567 * w00 */
                    "ld1 {v20.4s}, [%[rmask]], #16           \n"
                    "ld1 {v22.4s}, [%[rmask]]                \n"
                    "smlal  v19.8h,  %[v1].8b,  v14.8b       \n"  /* acc_r10 += r(1, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v4].8b,  v14.8b       \n"  /* acc_r00 += r(1, 12345678) * w11 */
                    "ld1 {v0.4s}, [%[ptr_out0]], #16         \n"  /* load original ptr_out0 low */
                    "ld1 {v2.4s}, [%[ptr_out1]], #16         \n"  /* load original ptr_out1 low */
                    "saddw   v12.4s, v12.4s, v19.4h          \n"  /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h         \n"  /* v13 += acc_r10.high */
                    "smull  v19.8h,  %[v2].8b,  v15.8b       \n"  /* acc_r10 = r(1, 23456789) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b       \n"  /* acc_r00 = r(1, 23456789) * w12 */
                    "ld1 {v1.4s}, [%[ptr_out0]]              \n"  /* load original ptr_out0 high */
                    "ld1 {v3.4s}, [%[ptr_out1]]              \n"  /* load original ptr_out1 high */
                    //r2
                    "ext v14.8b, v8.8b, v9.8b, #1             \n"  /* r(3, 12345678) */
                    "ext v15.8b, v8.8b, v9.8b, #2             \n"  /* r(3, 23456789) */
                    "smlal  v19.8h,  %[v3].8b,  v6.8b        \n"  /* acc_r10 += r(2, 01234567) * w10 */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b        \n"  /* acc_r00 = r(2, 01234567) * w20 */
                    "sub %[ptr_out0], %[ptr_out0], #16      \n"
                    "sub %[ptr_out1], %[ptr_out1], #16      \n"
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v13 += outr00.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b      \n"   /* acc_r10 += r(2, 12345678) * w11 */
                    "smlal  v18.8h,  %[v7].8b,  v4.8b      \n"   /* acc_r00 += r(2, 12345678) * w21 */
                    "smlal  v19.8h,  %[v5].8b,  v5.8b      \n"   /* acc_r10 += r(2, 23456789) * w12 */
                    "saddw   v10.4s, v10.4s, v18.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h        \n"   /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b      \n"   /* acc_r00 += r(2, 23456789) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += acc_r00.high*/
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b       \n"   /* acc_r10 += r(3, 01234567) * w20 */
                    "saddw   v10.4s, v10.4s, v18.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h        \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v7].8b,  v14.8b       \n"   /* acc_r10 += r(3, 12345678) * w21 */
                    "scvtf  v10.4s, v10.4s                  \n"  /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s                  \n"  /* int32-> float32*/
                    "dup   v6.4s, %[v_bias].s[0]                 \n "
                    "dup   v7.4s, %[v_bias].s[0]                 \n "
                    "fmla   v6.4s, v10.4s, %[v_scale].4s         \n"
                    "fmla   v7.4s, v11.4s, %[v_scale].4s         \n"
                    "bif v6.16b, v0.16b, v20.16b           \n"   /* select bit of ptr_out 0 according to rmask */
                    "bif v7.16b, v1.16b, v22.16b           \n"   /* select bit of ptr_out 0 high according to rmask */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += acc_r00.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b       \n"   /* acc_r10 = r(3, 23456789) * w22 */
                    "stp     q6, q7, [%[ptr_out0]], #32   \n"   /* store q10, q11 -> ptr_out 0   */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += outr00.high*/
                    "scvtf  v12.4s, v12.4s                 \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                 \n"  /* int32-> float32*/
                    "dup   v8.4s, %[v_bias].s[0]                 \n "
                    "dup   v9.4s, %[v_bias].s[0]                 \n "
                    "fmla   v8.4s, v12.4s, %[v_scale].4s  \n"
                    "fmla   v9.4s, v13.4s, %[v_scale].4s  \n"
                    "bif v8.16b, v2.16b, v20.16b           \n"   /* select bit of ptr_out 1 according to rmask */
                    "bif v9.16b, v3.16b, v22.16b           \n"   /* select bit of ptr_out 1 hihg according to rmask */
                    "stp   q8, q9, [%[ptr_out1]], #32   \n"   /* store q12, q13 -> ptr_out 1  */
                    :   [cnt]"+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [din_ptr3] "+r"(din_ptr3), [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
                        [vmask] "+r" (val_mask), [rmask] "+r" (rst_mask)
                    :   [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), \
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22), \
                        [v_bias]"w"(v_bias), [v_scale] "w"(v_scale)
                    :"cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
                      "v10", "v11", "v12","v13","v14","v15", \
                      "v18",\
                      "v19", "v20", "v21", "v22"
                );
#else
                // store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                    "pld [%[din_ptr0]]                \n"
                    "pld [%[din_ptr1]]                \n"
                    "pld [%[din_ptr2]]                \n"
                    "pld [%[din_ptr3]]                \n"
                    "vmov.u32 d11, #0                 \n"
                //mid
                    "1:                                          \n"
                    "vdup.s8     d2, d0[0]            \n"
                    "vdup.s8     d3, d0[1]            \n"
                    "vdup.s8     d4, d0[2]            \n"
                    "vdup.s8     d5, d0[3]               \n"
                    "vdup.s8     d6, d0[4]               \n"
                    "vdup.s8     d7, d0[5]               \n"
                    "vdup.s8     d8, d0[6]               \n"
                    "vdup.s8     d9, d0[7]               \n"
                    "vdup.s8     d10, d1[0]              \n"
                    "vmov.u32 q8, #0                  \n"
                    "vmov.u32 q9, #0                  \n"
                    "vmov.u32 q10, #0                 \n"
                    "vmov.u32 q11, #0                 \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    \n"
                     //r0
                    "vmull.s8 q12, d12, d2                 \n" // q12 = d12 * w01
                    "vext.8     d30, d12, d13, #1     \n" //d10 = 12345678
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr1]]    \n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]    \n"
                    "vmlal.s8 q12, d30, d3                 \n" // q12 += d10 * w00
                    "add %[din_ptr0], #8                   \n"
                    "add %[din_ptr1], #8                   \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                 \n" // q12 += d11 * w02
                    //r1
                    "vext.8     d30, d12, d13, #1     \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d2                 \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d12, d5                 \n" // q12 = d12 * w11
                    "vld1.8 {d12-d13}, [%[din_ptr3]]    \n"
                    "vmlal.s8 q13, d30, d3                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d30, d6                 \n" // q12 += d10 * w00
                    "add %[din_ptr2], #8                   \n"
                    "add %[din_ptr3], #8                   \n"
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 \n" // q12 += d10 * w00
                    //r2
                    "vext.8     d30, d14, d15, #1          \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d14, d5                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d8                 \n" // q13 = d12 * w01
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 \n" // q12 += d10 * w00
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d10                \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d12, d13, #1          \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d12, d8                 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                     \n"
                    "pld [%[din_ptr1]]                     \n"
                    "vmlal.s8 q13, d30, d9                 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                     \n"
                    "pld [%[din_ptr3]]                     \n"
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                 \n" // q12 += d10 * w00
                    "vdup.32 q7, %[scale]                            \n"
                    "vdup.32 q14, %[bias]                            \n"
                    "vdup.32 q15, %[bias]                            \n"
                    "vcvt.f32.s32   q8, q8                           \n"
                    "vcvt.f32.s32   q9, q9                           \n"
                    "vmla.f32 q14, q8, q7                            \n"
                    "vmla.f32 q15, q9, q7                            \n"
                    "vst1.32 {d28-d29}, [%[dout_ptr1]]!              \n"
                    "vst1.32 {d30-d31}, [%[dout_ptr1]]!              \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q12, %[bias]                            \n"
                    "vdup.32 q13, %[bias]                            \n"
                    "vcvt.f32.s32   q10, q10                         \n"
                    "vcvt.f32.s32   q11, q11                         \n"
                    "vmla.f32 q12, q10, q7                           \n"
                    "vmla.f32 q13, q11, q7                           \n"
                    "vst1.32 {d24-d25}, [%[dout_ptr2]]!              \n"
                    "subs %[cnt], %[cnt], #1                         \n"
                    "vst1.32 {d26-d27}, [%[dout_ptr2]]!              \n"
                    "bne  1b                                         \n"
                //right
                    "2:                                              \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    \n"
                    "vld1.8 {d28-d29}, [%[mask]]        \n"
                    "vmov.u32 q8, #0                            \n"
                    "vmov.u32 q9, #0                            \n"
                    "vmov.u32 q10, #0                           \n"
                    "vmov.u32 q11, #0                           \n"
                    "vbif.8 d12, d11, d28               \n"
                    "vbif.8 d13, d11, d29               \n"
                    "vld1.8 {d14-d15}, [%[din_ptr1]]    \n"
                     //r0
                    "vmull.s8 q12, d12, d2                 \n" // q12 = d12 * w01
                    "vext.8 d30, d12, d13, #1               \n" //d10 = 12345678
                    "vext.8 d31, d12, d13, #2               \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr2]]       \n"
                    "vbif.8 d14, d11, d28                  \n"
                    "vbif.8 d15, d11, d29                  \n"
                    "vmlal.s8 q12, d30, d3                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                 \n" // q12 += d11 * w02
                    //r1
                    "vext.8 d30, d14, d15, #1              \n" //d10 = 00123456
                    "vext.8 d31, d14, d15, #2              \n" //d11 = 12345678
                    "vmull.s8 q13, d14, d2                 \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d14, d5                 \n" // q12 = d12 * w11
                    "vld1.8 {d14-d15}, [%[din_ptr3]]       \n"
                    "vbif.8 d12, d11, d28                  \n"
                    "vbif.8 d13, d11, d29                  \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d30, d3                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d6                 \n" // q12 += d10 * w00
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 \n" // q12 += d10 * w00
                    //r2
                    "vext.8 d30, d12, d13, #1               \n" //d10 = 00123456
                    "vext.8 d31, d12, d13, #2               \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d12, d5                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d12, d8                 \n" // q13 = d12 * w01
                    "vbif.8 d14, d11, d28                    \n"
                    "vbif.8 d15, d11, d29                    \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 \n" // q12 += d10 * w00
                    "vld1.32 {d28-d29}, [%[dout_ptr1]]!    \n"
                    "vld1.32 {d12-d13}, [%[dout_ptr1]]     \n"
                    "vld1.32 {d2-d3}, [%[rs_mask]]!        \n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]         \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                 \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d14, d15, #1           \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2           \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d14, d8                 \n" // q13 = d12 * w01
                    "vld1.32 {d14-d15}, [%[dout_ptr2]]!    \n"
                    "vld1.32 {d24-d25}, [%[dout_ptr2]]     \n"
                    "vmlal.s8 q13, d30, d9                 \n" // q13 += d10 * w00
                    "sub %[dout_ptr1], #16                 \n"
                    "sub %[dout_ptr2], #16                 \n"
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                \n" // q12 += d10 * w00
                    "vdup.32 q3, %[scale]                  \n"
                    "vdup.32 q4, %[bias]                   \n"
                    "vdup.32 q5, %[bias]                   \n"
                    "vcvt.f32.s32   q8, q8                 \n"
                    "vcvt.f32.s32   q9, q9                 \n"
                    "vmla.f32 q4, q8, q3                   \n"
                    "vmla.f32 q5, q9, q3                   \n"
                    "vbif q4, q14, q1                      \n"
                    "vbif q5, q6, q2                       \n"
                    "vst1.32 {d8-d9}, [%[dout_ptr1]]!      \n"
                    "vst1.32 {d10-d11}, [%[dout_ptr1]]!    \n"
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q13, %[bias]                  \n"
                    "vdup.32 q14, %[bias]                  \n"
                    "vcvt.f32.s32   q10, q10               \n"
                    "vcvt.f32.s32   q11, q11               \n"
                    "vmla.f32 q13, q10, q3                 \n"
                    "vmla.f32 q14, q11, q3                 \n"
                    "vbif q13, q7, q1                      \n"
                    "vbif q14, q12, q2                     \n"
                    "vst1.32 {d26-d27}, [%[dout_ptr2]]!    \n"
                    "vst1.32 {d28-d29}, [%[dout_ptr2]]!    \n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2), \
                      [din_ptr3] "+r" (din_ptr3), \
                      [dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [scale] "+r" (scale_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask)
                    :"cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                      "q11", "q12", "q13", "q14", "q15"
                );
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
  }
}

void conv_depthwise_3x3s1p0_bias_int8_int8(int8_t* dout,
                                           const int8_t* din,
                                           const int8_t* weights,
                                           const float* scale,
                                           const float* bias,
                                           bool flag_bias,
                                           int flag_act,
                                           float* alpha,
                                           int num,
                                           int ch_in,
                                           int h_in,
                                           int w_in,
                                           int h_out,
                                           int w_out,
                                           ARMContext* ctx) {
  const unsigned char right_pad_idx[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const unsigned char right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, w_in * sizeof(int8_t));
  int8_t* write_ptr =
      reinterpret_cast<int8_t*>(ctx->workspace_data<int8_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;
  int tile_w = w_out >> 3;
  int cnt_col = tile_w;
  unsigned int size_pad_right = (unsigned int)(w_in - (cnt_col << 3));
  uint8x8_t vmask_rp1 =
      vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
  uint8x8_t vmask_rp2 =
      vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
  unsigned char rst_remain = (w_out - ((cnt_col) << 3));
  uint8x8_t vmask_result =
      vcgt_u8(vdup_n_u8(rst_remain), vld1_u8(right_pad_rst));
  uint8x16_t vmask_rp =
      vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
  unsigned char vmask[16];
  vst1q_u8(vmask, vmask_rp);
  unsigned char rmask[8];
  vst1_u8(rmask, vmask_result);

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    int8_t* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      int8_t* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
      float vscale[4] = {scale_val, scale_val, scale_val, scale_val};
      float vmax[4] = {-127.f, -127.f, -127.f, -127.f};
      float* vmax_ptr = vmax;
#ifdef __aarch64__
      int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
      int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
      int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

      int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
      int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
      int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

      int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
      int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
      int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

      float32x4_t v_bias = vdupq_n_f32(vbias[0]);
      float32x4_t v_scale = vdupq_n_f32(vscale[0]);
#endif

      int8_t* doutr0 = nullptr;
      int8_t* doutr1 = nullptr;

      const int8_t* dr0 = din_ch_ptr;
      const int8_t* dr1 = dr0 + w_in;
      const int8_t* dr2 = dr1 + w_in;
      const int8_t* dr3 = dr2 + w_in;

      const int8_t* din_ptr0 = nullptr;
      const int8_t* din_ptr1 = nullptr;
      const int8_t* din_ptr2 = nullptr;
      const int8_t* din_ptr3 = nullptr;

      for (int i = 0; i < h_out; i += 2) {
        //! process top pad pad_h = 1
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;

        unsigned char* rst_mask = rmask;
        unsigned char* val_mask = vmask;

        dr0 = dr2;
        dr1 = dr3;
        dr2 = dr1 + w_in;
        dr3 = dr2 + w_in;
        //! process bottom pad
        if (i + 3 > h_in) {
          switch (i + 3 - h_in) {
            case 3:
              din_ptr1 = zero_ptr;
            case 2:
              din_ptr2 = zero_ptr;
            case 1:
              din_ptr3 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 2 > h_out) {
          doutr1 = write_ptr;
        }
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr3]] \n"
                    "movi   v21.4s, #0x0\n"                                     /* v21 = 0 */
                // middle
                    "ld1    {v0.8b}, [%[din_ptr0]], #8               \n"   /* load a(0,0)-a(0,7) to q0*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8               \n"   /* load a(1,0)-a(1.7) to q2*/
                    "ld1    {v1.8b}, [%[din_ptr0]]                   \n"   /* load a(0,8)-a(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]                   \n"   /* load a(1,8)-a(1,15) to q3*/
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a(2,0)-a(2,7) to q6*/
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a(3,0)-a(3,7) to q8*/
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a(2,8)-a(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a(3,8)-a(3,15) to q9*/
                    "movi   v10.4s, #0x0\n"                 /* init int32 acc v10 to 0 */
                    "movi   v11.4s, #0x0\n"                 /* init int32 acc v11 to 0 */
                    "movi   v12.4s, #0x0\n"                   /* init int32 acc v12 to 0 */
                    "movi   v13.4s, #0x0\n"                   /* init int32 acc v13 to 0 */
                    "1:                            \n"
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b   \n"
                    "ext v4.8b, v0.8b, v1.8b, #1        \n"
                    "ext v5.8b, v0.8b, v1.8b, #2        \n"
                    "smlal  v18.8h,  %[v1].8b,  v4.8b   \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v2].8b,  v5.8b \n"   /* acc_r00 += r(0,12345678) * w02 */
                    "ext v14.8b, v2.8b, v3.8b, #1       \n"
                    "ext v15.8b, v2.8b, v3.8b, #2       \n"
                    //r1
                    "smull  v19.8h,  %[v0].8b,  v2.8b   \n"
                    "smlal  v18.8h,  %[v3].8b,  v2.8b   \n"
                    "ext v4.8b, v6.8b, v7.8b, #1       \n"
                    "ext v5.8b, v6.8b, v7.8b, #2       \n"
                    "smlal  v19.8h,  %[v1].8b,  v14.8b   \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v4].8b,  v14.8b   \n"
                    "ld1    {v0.8b}, [%[din_ptr0]], #8  \n"   /* load a'(0,0)-a'(0,7) to q0 for prefetch*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8  \n"   /* load a'(1,0)-a'(1,7) to q2 for prefetch*/
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v2].8b,  v15.8b   \n"   /* acc_r10 += r(1,12345678) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b   \n"   /* acc_r00 += r(1,12345678) * w12 */
                    //r2
                    "ld1    {v1.8b}, [%[din_ptr0]]      \n"   /* load a'(0,8)-a'(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]      \n"   /* load a'(1,8)-a'(1,15) to q3*/
                    "smlal  v19.8h,  %[v3].8b,  v6.8b   \n"  /* acc_r10 += r(2,01234567) * w11 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b   \n"
                    "ext v14.8b, v8.8b, v9.8b, #1        \n"
                    "ext v15.8b, v8.8b, v9.8b, #2        \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b  \n"
                    "smlal  v18.8h,  %[v7].8b,  v4.8b  \n"
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a'(2,0)-a'(2,7) to q6 for prefetch*/
                    "smlal  v19.8h,  %[v5].8b,  v5.8b  \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b  \n"   /* acc_r00 += r(2, 12345678) * w22 */
                    "ld1 {v4.4s}, [%[vmax_ptr]]        \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v12 += outr10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v13 += outr10.high*/
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b   \n"   /* acc_r10 += r(3, 01234567) * w21 */
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a'(3,0)-a'(3,7) to q8 for prefetch*/
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a'(2,7)-a'(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a'(3,7)-a'(3,15) to q9*/
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v7].8b,  v14.8b   \n"   /* acc_r10 += r(3, 00123456) * w20) */
                    "dup   v20.4s, %[v_bias].s[0] \n "                /* dup bias to v20 */
                    "dup   v22.4s, %[v_bias].s[0] \n "                /* dup bias to v22 */
                    "scvtf  v10.4s, v10.4s               \n"   /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s               \n"   /* int32-> float32*/
                    "fmla   v20.4s, v10.4s, %[v_scale].4s       \n"  /* mul scale, add bias */
                    "fmla   v22.4s, v11.4s, %[v_scale].4s       \n"  /* mul scale, add bias */
                    "fcmge v11.4s, v20.4s, v4.4s             \n"
                    "bif v20.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v22.4s, v4.4s             \n"
                    "bif v22.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v20.4s, v20.4s                  \n" /* fp32 - int32 */
                    "fcvtas  v22.4s, v22.4s                  \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v20.4s                  \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v22.4s                 \n" /* int32-int16 */
                    "sqxtn   v20.8b, v21.8h                  \n" /* int16-int8 */
                    "st1    {v20.8b}, [%[ptr_out0]], #8     \n"  /* store q10, q11 -> ptr_out 0 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"    /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"    /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b   \n"    /* acc_r10 += r(3, 12345678) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"    /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"    /* v13 += acc_r10.high*/
                    "scvtf  v12.4s, v12.4s                \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                \n"  /* int32-> float32*/
                    "dup   v14.4s, %[v_bias].s[0] \n "    /* dup bias to v14 */
                    "dup   v15.4s, %[v_bias].s[0] \n "    /* dup bias to v15 */
                    "fmla   v14.4s, v12.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "fmla   v15.4s, v13.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "fcmge v11.4s, v14.4s, v4.4s             \n"
                    "bif v14.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v15.4s, v4.4s             \n"
                    "bif v15.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v14.4s, v14.4s                   \n" /* fp32 - int32 */
                    "fcvtas  v15.4s, v15.4s                   \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v14.4s                   \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v15.4s                   \n" /* int32-int16 */
                    "sqxtn   v14.8b, v21.8h                   \n" /* int16-int8 */
                    "st1    {v14.8b}, [%[ptr_out1]], #8     \n"  /* store q10, q11 -> ptr_out 0 */
                    "movi   v10.4s, #0x0\n"         /* clear 0 for v10 */
                    "movi   v11.4s, #0x0\n"         /* clear 0 for v11 */
                    "movi   v12.4s, #0x0\n"
                    "movi   v13.4s, #0x0\n"
                    "movi   v21.4s, #0x0\n"
                    "subs %[cnt], %[cnt], #1                \n"
                    "bne 1b                                 \n"
                //right
                    "3:                                      \n" 
                    "ld1 {v20.8b}, [%[vmask]], #8            \n"
                    "ld1 {v22.8b}, [%[vmask]]                \n"
                    "bif v0.8b, v21.8b, v20.8b               \n"  /* a(0, 0) to a(0, 7) */
                    "bif v1.8b, v21.8b, v22.8b               \n"  /* a(0, 8) to a(0, 15) */
                    "bif v2.8b, v21.8b, v20.8b               \n"  /* a(1, 0) to a(1, 7) */
                    "bif v3.8b, v21.8b, v22.8b               \n"  /* a(1, 8) to a(1, 15) */
                    "ext v4.8b, v0.8b, v1.8b, #1             \n"  /* r(0, 12345678) */
                    "ext v5.8b, v0.8b, v1.8b, #2             \n"  /* r(0, 23456789) */
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b        \n"  /* acc_r00 += r(0, 01234567) * w00 */
                    "ext v14.8b, v2.8b, v3.8b, #1            \n"  /* r(1, 12345678) */
                    "ext v15.8b, v2.8b, v3.8b, #2            \n"  /* r(1, 23456789) */
                    "bif v6.8b, v21.8b, v20.8b               \n"  /* a(2, 0) to a(0, 7) */  /* r(2, 12345678) */
                    "bif v7.8b, v21.8b, v22.8b               \n"  /* a(2, 8) to a(2, 15) */ /* r(2, 23456789) */
                    "smlal  v18.8h,  %[v1].8b,  v4.8b        \n"  /* acc_r00 += r(0, 12345678) * w01 */
                    "bif v8.8b, v21.8b, v20.8b               \n"  /* a(3, 0) to a(3, 7) */ 
                    "bif v9.8b, v21.8b, v22.8b               \n"  /* a(3, 8) to a(3, 15) */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v2].8b,  v5.8b        \n"  /* acc_r00 = r(0, 23456789) * w02 */
                    //r1
                    "ext v4.8b, v6.8b, v7.8b, #1            \n"
                    "ext v5.8b, v6.8b, v7.8b, #2            \n"
                    "smull  v19.8h,  %[v0].8b,  v2.8b        \n"  /* acc_r10 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b        \n"  /* acc_r00 = 01234567 * w00 */
                    "ld1 {v20.8b}, [%[rmask]], #8               \n"
                    "smlal  v19.8h,  %[v1].8b,  v14.8b       \n"  /* acc_r10 += r(1, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v4].8b,  v14.8b       \n"  /* acc_r00 += r(1, 12345678) * w11 */
                    "ld1 {v0.8b}, [%[ptr_out0]]         \n"  /* load original ptr_out0 low */
                    "ld1 {v2.8b}, [%[ptr_out1]]         \n"  /* load original ptr_out1 low */
                    "saddw   v12.4s, v12.4s, v19.4h          \n"  /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h         \n"  /* v13 += acc_r10.high */
                    "smull  v19.8h,  %[v2].8b,  v15.8b       \n"  /* acc_r10 = r(1, 23456789) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b       \n"  /* acc_r00 = r(1, 23456789) * w12 */
                    //r2
                    "ext v14.8b, v8.8b, v9.8b, #1             \n"  /* r(3, 12345678) */
                    "ext v15.8b, v8.8b, v9.8b, #2             \n"  /* r(3, 23456789) */
                    "smlal  v19.8h,  %[v3].8b,  v6.8b        \n"  /* acc_r10 += r(2, 01234567) * w10 */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b        \n"  /* acc_r00 = r(2, 01234567) * w20 */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v13 += outr00.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b      \n"   /* acc_r10 += r(2, 12345678) * w11 */
                    "smlal  v18.8h,  %[v7].8b,  v4.8b      \n"   /* acc_r00 += r(2, 12345678) * w21 */
                    "smlal  v19.8h,  %[v5].8b,  v5.8b      \n"   /* acc_r10 += r(2, 23456789) * w12 */
                    "saddw   v10.4s, v10.4s, v18.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h        \n"   /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b      \n"   /* acc_r00 += r(2, 23456789) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += acc_r00.high*/
                    "ld1 {v4.4s}, [%[vmax_ptr]]                \n"
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b       \n"   /* acc_r10 += r(3, 01234567) * w20 */
                    "saddw   v10.4s, v10.4s, v18.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h        \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v7].8b,  v14.8b       \n"   /* acc_r10 += r(3, 12345678) * w21 */
                    "scvtf  v10.4s, v10.4s                  \n"  /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s                  \n"  /* int32-> float32*/
                    "dup   v6.4s, %[v_bias].s[0]                 \n "
                    "dup   v7.4s, %[v_bias].s[0]                 \n "
                    "fmla   v6.4s, v10.4s, %[v_scale].4s         \n"
                    "fmla   v7.4s, v11.4s, %[v_scale].4s         \n"
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += acc_r00.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b       \n"   /* acc_r10 = r(3, 23456789) * w22 */
                    "fcmge v11.4s, v6.4s, v4.4s             \n"
                    "bif v6.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v7.4s, v4.4s             \n"
                    "bif v7.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v6.4s, v6.4s                   \n" /* fp32 - int32 */
                    "fcvtas  v7.4s, v7.4s                   \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v6.4s                   \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v7.4s                   \n" /* int32-int16 */
                    "sqxtn   v6.8b, v21.8h                   \n" /* int16-int8 */
                    "bif v6.8b, v0.8b, v20.8b           \n"
                    "st1    {v6.8b}, [%[ptr_out0]], #8     \n"  /* store q10, q11 -> ptr_out 0 */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += outr00.high*/
                    "scvtf  v12.4s, v12.4s                 \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                 \n"  /* int32-> float32*/
                    "dup   v8.4s, %[v_bias].s[0]                 \n "
                    "dup   v9.4s, %[v_bias].s[0]                 \n "
                    "fmla   v8.4s, v12.4s, %[v_scale].4s  \n"
                    "fmla   v9.4s, v13.4s, %[v_scale].4s  \n"
                    "fcmge v11.4s, v8.4s, v4.4s             \n"
                    "bif v8.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v9.4s, v4.4s             \n"
                    "bif v9.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v8.4s, v8.4s                   \n" /* fp32 - int32 */
                    "fcvtas  v9.4s, v9.4s                   \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v8.4s                   \n" /* int32-int16 */
                    "sqxtn2  v21.8h, v9.4s                   \n" /* int32-int16 */
                    "sqxtn   v8.8b, v21.8h                   \n" /* int16-int8 */
                    "bif v8.8b, v2.8b, v20.8b           \n"
                    "st1    {v8.8b}, [%[ptr_out1]], #8     \n"  /* store q10, q11 -> ptr_out 0 */
                    :   [cnt]"+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [din_ptr3] "+r"(din_ptr3), [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
                        [vmask] "+r" (val_mask), [rmask] "+r" (rst_mask)
                    :   [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), \
                        [vmax_ptr]"r"(vmax_ptr), \
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22), \
                        [v_bias]"w"(v_bias), [v_scale] "w"(v_scale)
                    :"cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
                      "v10", "v11", "v12","v13","v14","v15", \
                      "v18",\
                      "v19", "v20", "v21", "v22"
                );
#else
                // store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                    "pld [%[din_ptr0]]                \n"
                    "pld [%[din_ptr1]]                \n"
                    "pld [%[din_ptr2]]                \n"
                    "pld [%[din_ptr3]]                \n"
                    "vmov.u32 d11, #0                 \n"
                //mid
                    "1:                                          \n"
                    "vdup.s8     d2, d0[0]               \n"
                    "vdup.s8     d3, d0[1]               \n"
                    "vdup.s8     d4, d0[2]               \n"
                    "vdup.s8     d5, d0[3]               \n"
                    "vdup.s8     d6, d0[4]               \n"
                    "vdup.s8     d7, d0[5]               \n"
                    "vdup.s8     d8, d0[6]               \n"
                    "vdup.s8     d9, d0[7]               \n"
                    "vdup.s8     d10, d1[0]              \n"
                    "vmov.u32 q8, #0                  \n"
                    "vmov.u32 q9, #0                  \n"
                    "vmov.u32 q10, #0                 \n"
                    "vmov.u32 q11, #0                 \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    \n"
                     //r0
                    "vmull.s8 q12, d12, d2                 \n" // q12 = d12 * w01
                    "vext.8     d30, d12, d13, #1     \n" //d10 = 12345678
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr1]]    \n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]    \n"
                    "vmlal.s8 q12, d30, d3                 \n" // q12 += d10 * w00
                    "add %[din_ptr0], #8                   \n"
                    "add %[din_ptr1], #8                   \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                 \n" // q12 += d11 * w02
                    //r1
                    "vext.8     d30, d12, d13, #1     \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d2                 \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d12, d5                 \n" // q12 = d12 * w11
                    "vld1.8 {d12-d13}, [%[din_ptr3]]    \n"
                    "vmlal.s8 q13, d30, d3                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d30, d6                 \n" // q12 += d10 * w00
                    "add %[din_ptr2], #8                   \n"
                    "add %[din_ptr3], #8                   \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 \n" // q12 += d10 * w00
                    //r2
                    "vext.8     d30, d14, d15, #1          \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d14, d5                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d8                 \n" // q13 = d12 * w01
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 \n" // q12 += d10 * w00
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d10                 \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d12, d13, #1     \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d12, d8                 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                \n"
                    "pld [%[din_ptr1]]                \n"
                    "vmlal.s8 q13, d30, d9                 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                \n"
                    "pld [%[din_ptr3]]                \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                  \n" // q12 += d10 * w00
                    "vdup.32 q7, %[scale]                    \n"
                    "vdup.32 q14, %[bias]                    \n"
                    "vdup.32 q15, %[bias]                    \n"
                    "vcvt.f32.s32   q8, q8                   \n"
                    "vcvt.f32.s32   q9, q9                   \n"
                    "vmla.f32 q14, q8, q7                    \n"
                    "vmla.f32 q15, q9, q7                    \n"
                    "vmov.f32 q8, #-0.5                      \n"
                    "vmov.f32 q9, #0.5                       \n"
                    "vcgt.f32   q1, q14, q8                  \n"
                    "vbif.f32   q9, q8, q1                   \n"
                    "vadd.f32   q14, q14, q9                 \n"
                    "vmov.f32   q9, #0.5                     \n"
                    "vcgt.f32   q2, q15, q8                  \n"
                    "vbif.f32   q9, q8, q2                   \n"
                    "vadd.f32   q15, q15, q9                 \n"
                    "vld1.32 {d2-d3}, [%[vmax]]              \n"
                    "vcge.f32 q3, q14, q1                    \n" /* data >= -127 */
                    "vbif q14, q1, q3                        \n" /* choose data */
                    "vcge.f32 q3, q15, q1                    \n" /* data >= -127 */
                    "vbif q15, q1, q3                        \n" /* choose data */
                    "vcvt.s32.f32  q1, q14    \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q15    \n"  /* fp32 to int32 */
                    "vqmovn.s32 d28, q1       \n"  /* int32 to int16 */
                    "vqmovn.s32 d29, q2       \n"  /* int32 to int16 */
                    "vqmovn.s16 d10, q14      \n"   /* int16 to int8 */
                    "vst1.8    {d10}, [%[dout_ptr1]]!        \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q12, %[bias]                    \n"
                    "vdup.32 q13, %[bias]                    \n"
                    "vcvt.f32.s32   q10, q10                 \n"
                    "vcvt.f32.s32   q11, q11                 \n"
                    "vmla.f32 q12, q10, q7                   \n"
                    "vmla.f32 q13, q11, q7                   \n"
                    "vmov.f32 q8, #-0.5                      \n"
                    "vmov.f32 q9, #0.5                       \n"
                    "vcgt.f32   q1, q12, q8                  \n"
                    "vbif.f32   q9, q8, q1                   \n"
                    "vadd.f32   q12, q12, q9                 \n"
                    "vmov.f32   q9, #0.5                     \n"
                    "vcgt.f32   q2, q13, q8                  \n"
                    "vbif.f32   q9, q8, q2                   \n"
                    "vadd.f32   q13, q13, q9                 \n"
                    "vld1.32 {d2-d3}, [%[vmax]]              \n"
                    "vcge.f32 q3, q12, q1                    \n" /* data >= -127 */
                    "vbif q12, q1, q3                        \n" /* choose data */
                    "vcge.f32 q3, q13, q1                    \n" /* data >= -127 */
                    "vbif q13, q1, q3                        \n" /* choose data */
                    "vcvt.s32.f32  q1, q12    \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q13    \n"  /* fp32 to int32 */
                    "vqmovn.s32 d24, q1       \n"  /* int32 to int16 */
                    "vqmovn.s32 d25, q2       \n"  /* int32 to int16 */
                    "vqmovn.s16 d9, q12       \n"   /* int16 to int8 */
                    "vst1.8    {d9}, [%[dout_ptr2]]!\n"
                    "subs %[cnt], %[cnt], #1                        \n"
                    "bne  1b                                        \n"
                //right
                    "2:                                             \n"
                    "vdup.s8     d2, d0[0]               \n"
                    "vdup.s8     d3, d0[1]               \n"
                    "vdup.s8     d4, d0[2]               \n"
                    "vdup.s8     d5, d0[3]               \n"
                    "vdup.s8     d6, d0[4]               \n"
                    "vdup.s8     d7, d0[5]               \n"
                    "vdup.s8     d8, d0[6]               \n"
                    "vdup.s8     d9, d0[7]               \n"
                    "vdup.s8     d10, d1[0]              \n"
                    "vmov.u32 d11, #0                    \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]     \n"
                    "vld1.8 {d28-d29}, [%[mask]]         \n"
                    "vmov.u32 q8, #0                            \n"
                    "vmov.u32 q9, #0                            \n"
                    "vmov.u32 q10, #0                           \n"
                    "vmov.u32 q11, #0                           \n"
                    "vbif.8 d12, d11, d28               \n"
                    "vbif.8 d13, d11, d29               \n"
                    "vld1.8 {d14-d15}, [%[din_ptr1]]    \n"
                     //r0
                    "vmull.s8 q12, d12, d2                  \n" // q12 = d12 * w01
                    "vext.8 d30, d12, d13, #1               \n" //d10 = 12345678
                    "vext.8 d31, d12, d13, #2               \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr2]]        \n"
                    "vbif.8 d14, d11, d28                   \n"
                    "vbif.8 d15, d11, d29                   \n"
                    "vmlal.s8 q12, d30, d3                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                 \n" // q12 += d11 * w02
                    //r1
                    "vext.8 d30, d14, d15, #1           \n" //d10 = 00123456
                    "vext.8 d31, d14, d15, #2           \n" //d11 = 12345678
                    "vmull.s8 q13, d14, d2                 \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d14, d5                 \n" // q12 = d12 * w11
                    "vld1.8 {d14-d15}, [%[din_ptr3]]    \n"
                    "vbif.8 d12, d11, d28                 \n"
                    "vbif.8 d13, d11, d29                 \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d30, d3                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d6                 \n" // q12 += d10 * w00
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 \n" // q12 += d10 * w00
                    //r2
                    "vext.8 d30, d12, d13, #1               \n" //d10 = 00123456
                    "vext.8 d31, d12, d13, #2               \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d12, d5                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d12, d8                 \n" // q13 = d12 * w01
                    "vbif.8 d14, d11, d28                     \n"
                    "vbif.8 d15, d11, d29                     \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 \n" // q12 += d10 * w00
                    "vld1.8 d12, [%[rs_mask]]!            \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                 \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d14, d15, #1     \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d14, d8                 \n" // q13 = d12 * w01
                    "vmlal.s8 q13, d30, d9                 \n" // q13 += d10 * w00
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                 \n" // q12 += d10 * w00
                    "vdup.32 q3, %[scale]                    \n"
                    "vdup.32 q4, %[bias]                     \n"
                    "vdup.32 q5, %[bias]                     \n"
                    "vcvt.f32.s32   q8, q8                      \n"
                    "vcvt.f32.s32   q9, q9                      \n"
                    "vmla.f32 q4, q8, q3                     \n"
                    "vmla.f32 q5, q9, q3                     \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q13, %[bias]                    \n"
                    "vdup.32 q14, %[bias]                    \n"
                    "vcvt.f32.s32   q10, q10                 \n"
                    "vcvt.f32.s32   q11, q11                 \n"
                    "vmla.f32 q13, q10, q3                   \n"
                    "vmla.f32 q14, q11, q3                   \n"
                    "vmov.f32 q8, #-0.5\n"
                    "vmov.f32 q9, #0.5\n"
                    "vcgt.f32   q1, q4, q8   \n"
                    "vbif.f32   q9, q8, q1   \n"
                    "vadd.f32   q4, q4, q9   \n"
                    "vmov.f32   q9, #0.5     \n"
                    "vcgt.f32   q2, q5, q8   \n"
                    "vbif.f32   q9, q8, q2   \n"
                    "vadd.f32   q5, q5, q9\n"
                    "vld1.32 {d2-d3}, [%[vmax]] \n"
                    "vcge.f32 q3, q4, q1        \n" /* data >= -127 */
                    "vbif q4, q1, q3            \n" /* choose data */
                    "vcge.f32 q3, q5, q1        \n" /* data >= -127 */
                    "vbif q5, q1, q3            \n" /* choose data */
                    "vcvt.s32.f32  q1, q4    \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q5    \n"  /* fp32 to int32 */
                    "vqmovn.s32 d8, q1       \n"  /* int32 to int16 */
                    "vqmovn.s32 d9, q2       \n"  /* int32 to int16 */
                    "vqmovn.s16 d7, q4      \n"   /* int16 to int8 */
                    "vld1.8 d10, [%[dout_ptr1]]    \n"
                    "vbif.8 d7, d10, d12                   \n"
                    "vst1.8    {d7}, [%[dout_ptr1]]!\n"
                    "vmov.f32 q8, #-0.5       \n"
                    "vmov.f32 q9, #0.5        \n"
                    "vcgt.f32   q1, q13, q8   \n"
                    "vbif.f32   q9, q8, q1    \n"
                    "vadd.f32   q13, q13, q9  \n"
                    "vmov.f32   q9, #0.5      \n"
                    "vcgt.f32   q2, q14, q8   \n"
                    "vbif.f32   q9, q8, q2    \n"
                    "vadd.f32   q14, q14, q9  \n"
                    "vld1.32 {d2-d3}, [%[vmax]] \n"
                    "vcge.f32 q3, q13, q1       \n" /* data >= -127 */
                    "vbif q13, q1, q3    \n" /* choose data */
                    "vcge.f32 q3, q14, q1         \n" /* data >= -127 */
                    "vbif q14, q1, q3    \n" /* choose data */
                    "vcvt.s32.f32  q1, q13    \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q14    \n"  /* fp32 to int32 */
                    "vqmovn.s32 d26, q1       \n"  /* int32 to int16 */
                    "vqmovn.s32 d27, q2       \n"  /* int32 to int16 */
                    "vqmovn.s16 d10, q13      \n"   /* int16 to int8 */
                    "vld1.8 d14, [%[dout_ptr2]]   \n"
                    "vbif.8 d10, d14, d12            \n"
                    "vst1.8    {d10}, [%[dout_ptr2]]!\n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2), \
                      [din_ptr3] "+r" (din_ptr3), \
                      [dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [scale] "+r" (scale_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask), [vmax]"r"(vmax_ptr)
                    :"cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                      "q11", "q12", "q13", "q14", "q15"
                );
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
  }
}

void conv_depthwise_3x3s1p1_bias_relu_int8_float(float* dout,
                                                 const int8_t* din,
                                                 const int8_t* weights,
                                                 const float* scale,
                                                 const float* bias,
                                                 bool flag_bias,
                                                 int flag_act,
                                                 float* alpha,
                                                 int num,
                                                 int ch_in,
                                                 int h_in,
                                                 int w_in,
                                                 int h_out,
                                                 int w_out,
                                                 ARMContext* ctx) {
  const unsigned char right_pad_idx[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, w_in * sizeof(int8_t));
  float* write_ptr =
      reinterpret_cast<float*>(ctx->workspace_data<int8_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;
  int tile_w = (w_in + 7) >> 3;
  int cnt_col = tile_w - 2;
  unsigned int size_pad_right = (unsigned int)(w_in - 7 - (cnt_col << 3));
  unsigned int rst_remain = (unsigned int)(w_out - ((cnt_col + 1) << 3));
  uint32x4_t vmask_result1 =
      vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
  uint32x4_t vmask_result2 =
      vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));

  uint8x16_t vmask_rp =
      vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
  unsigned char vmask[16];
  vst1q_u8(vmask, vmask_rp);
  unsigned int rmask[8];
  vst1q_u32(rmask, vmask_result1);
  vst1q_u32(rmask + 4, vmask_result2);

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
      float vscale[4] = {scale_val, scale_val, scale_val, scale_val};

#ifdef __aarch64__
      int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
      int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
      int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

      int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
      int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
      int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

      int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
      int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
      int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

      float32x4_t v_bias = vdupq_n_f32(vbias[0]);
      float32x4_t v_scale = vdupq_n_f32(vscale[0]);
#endif

      float* doutr0 = nullptr;
      float* doutr1 = nullptr;

      const int8_t* dr0 = din_ch_ptr;
      const int8_t* dr1 = dr0 + w_in;
      const int8_t* dr2 = dr1 + w_in;
      const int8_t* dr3 = dr2 + w_in;

      const int8_t* din_ptr0 = nullptr;
      const int8_t* din_ptr1 = nullptr;
      const int8_t* din_ptr2 = nullptr;
      const int8_t* din_ptr3 = nullptr;

      for (int i = 0; i < h_in; i += 2) {
        //! process top pad pad_h = 1
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;
        unsigned int* rst_mask = rmask;
        unsigned char* val_mask = vmask;

        if (i == 0) {
          din_ptr0 = zero_ptr;
          din_ptr1 = dr0;
          din_ptr2 = dr1;
          din_ptr3 = dr2;
          dr0 = dr1;
          dr1 = dr2;
          dr2 = dr3;
          dr3 = dr2 + w_in;
        } else {
          dr0 = dr2;
          dr1 = dr3;
          dr2 = dr1 + w_in;
          dr3 = dr2 + w_in;
        }
        //! process bottom pad
        if (i + 3 > h_in) {
          switch (i + 3 - h_in) {
            case 3:
              din_ptr1 = zero_ptr;
            case 2:
              din_ptr2 = zero_ptr;
            case 1:
              din_ptr3 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 2 > h_out) {
          doutr1 = write_ptr;
        }
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr3]] \n"
                    "movi   v21.4s, #0x0\n"                                     /* v21 = 0 */
                // left
                    "ld1    {v0.8b}, [%[din_ptr0]], #8               \n"   /* load a(0,0)-a(0,7) to q0*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8               \n"   /* load a(1,0)-a(1.7) to q2*/
                    "ld1    {v1.8b}, [%[din_ptr0]]                   \n"   /* load a(0,8)-a(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]                   \n"   /* load a(1,8)-a(1,15) to q3*/
                    "movi   v10.4s, #0x0\n"                 /* init int32 acc v10 to 0 */
                    "movi   v11.4s, #0x0\n"                 /* init int32 acc v11 to 0 */
                    "movi   v12.4s, #0x0\n"                   /* init int32 acc v12 to 0 */
                    "movi   v13.4s, #0x0\n"                   /* init int32 acc v13 to 0 */
                    //r0
                    "smull  v18.8h,  %[v1].8b,  v0.8b   \n"   /* acc_r(0,0) = r(0,01234567) * w01 */
                    "ext v4.8b, v21.8b, v0.8b, #7       \n"   /* vext_s8(vzero, vin_r0, 7); r(0,00123456) */
                    "ext v5.8b, v0.8b, v1.8b, #1        \n"   /* vext_s8(vin_r0, vin_r0_h, 1); r(0,12345678) */
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a(2,0)-a(2,7) to q6*/
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a(3,0)-a(3,7) to q8*/
                    "smlal  v18.8h,  %[v0].8b,  v4.8b   \n"   /* acc_r(0,0) += r(0,00123456) * w00 */
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a(2,8)-a(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a(3,8)-a(3,15) to q9*/
                    "sub   %[din_ptr0], %[din_ptr0], #1 \n"
                    "sub   %[din_ptr1], %[din_ptr1], #1 \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v2].8b,  v5.8b \n"   /* acc_r00 += r(0,12345678) * w02 */
                    "ext v14.8b, v21.8b, v2.8b, #7       \n"   /* vext_s8(vzero, vin_r1, 7); r(1,00123456) */
                    "ext v15.8b, v2.8b, v3.8b, #1        \n"   /* vext_s8(vin_r1, vin_r1_h, 1); r(1,12345678) */
                    //r1
                    "sub   %[din_ptr2], %[din_ptr2], #1 \n"
                    "sub   %[din_ptr3], %[din_ptr3], #1 \n"
                    "smull  v19.8h,  %[v1].8b,  v2.8b   \n"   /* acc_r10 += r(1,01234567) * w01 */
                    "smlal  v18.8h,  %[v4].8b,  v2.8b   \n"   /* acc_r00 += r(1,01234567) * w11 */
                    "ext v4.8b, v21.8b, v6.8b, #7      \n"
                    "ext v5.8b, v6.8b, v7.8b, #1       \n"
                    "smlal  v19.8h,  %[v0].8b,  v14.8b   \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v3].8b,  v14.8b   \n"   /* acc_r00 += r(1, 00123456) * w10 */
                    "ld1    {v0.8b}, [%[din_ptr0]], #8  \n"   /* load a'(0,0)-a'(0,7) to q0 for prefetch*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8  \n"   /* load a'(1,0)-a'(1,7) to q2 for prefetch*/
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v2].8b,  v15.8b   \n"   /* acc_r10 += r(1,12345678) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b   \n"   /* acc_r00 += r(1,12345678) * w12 */
                    //r2
                    "ld1    {v1.8b}, [%[din_ptr0]]      \n"   /* load a'(0,8)-a'(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]      \n"   /* load a'(1,8)-a'(1,15) to q3*/
                    "smlal  v19.8h,  %[v4].8b,  v6.8b   \n"   /* acc_r10 += r(2,01234567) * w11 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v7].8b,  v6.8b   \n"   /* acc_r00 += r(2,01234567) * w21 */
                    "ext v14.8b, v21.8b, v8.8b, #7       \n"
                    "ext v15.8b, v8.8b, v9.8b, #1        \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v3].8b,  v4.8b  \n"   /* acc_r10 += r(2, 00123456) * w10 */
                    "smlal  v18.8h,  %[v6].8b,  v4.8b  \n"
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a'(2,0)-a'(2,7) to q6 for prefetch*/
                    "smlal  v19.8h,  %[v5].8b,  v5.8b  \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b  \n"   /* acc_r00 += r(2, 12345678) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v12 += outr10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v13 += outr10.high*/
                    //r3
                    "smull  v19.8h,  %[v7].8b,  v8.8b   \n"   /* acc_r10 += r(3, 01234567) * w21 */
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a'(3,0)-a'(3,7) to q8 for prefetch*/
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a'(2,7)-a'(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a'(3,7)-a'(3,15) to q9*/
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v6].8b,  v14.8b   \n"   /* acc_r10 += r(3, 00123456) * w20) */
                    "dup   v20.4s, %[v_bias].s[0] \n "                /* dup bias to v20 */
                    "dup   v22.4s, %[v_bias].s[0] \n "                /* dup bias to v22 */
                    "scvtf  v10.4s, v10.4s               \n"   /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s               \n"   /* int32-> float32*/
                    "fmla   v20.4s, v10.4s, %[v_scale].4s       \n"  /* mul scale, add bias */
                    "fmla   v22.4s, v11.4s, %[v_scale].4s       \n"  /* mul scale, add bias */
                    "fmax  v20.4s, v20.4s, v21.4s        \n"  /* relu */
                    "fmax  v22.4s, v22.4s, v21.4s        \n"  /* relu */
                    "stp    q20, q22, [%[ptr_out0]], #32 \n"  /* store q10, q11 -> ptr_out 0 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"    /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"    /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b   \n"    /* acc_r10 += r(3, 12345678) * w22 */
                    "movi   v10.4s, #0x0\n"         /* clear 0 for v10 */
                    "movi   v11.4s, #0x0\n"         /* clear 0 for v11 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"    /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"    /* v13 += acc_r10.high*/
                    "scvtf  v12.4s, v12.4s                \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                \n"  /* int32-> float32*/
                    "dup   v14.4s, %[v_bias].s[0] \n "    /* dup bias to v14 */
                    "dup   v15.4s, %[v_bias].s[0] \n "    /* dup bias to v15 */
                    "fmla   v14.4s, v12.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "fmla   v15.4s, v13.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "fmax  v14.4s, v14.4s, v21.4s        \n"   /* relu*/
                    "fmax  v15.4s, v15.4s, v21.4s        \n"   /* relu*/
                    "stp     q14, q15, [%[ptr_out1]], #32   \n" /* store q10, q11 -> ptr_out  1 */
                    "movi   v12.4s, #0x0\n"
                    "movi   v13.4s, #0x0\n"
                    "cmp  %[cnt], #1                \n"
                    "blt 3f                         \n"
                //mid
                    "1:                             \n"
                    "ext v4.8b, v0.8b, v1.8b, #1       \n"      /* vext_s8(vin_r0, vin_r0_h, 1); r(0, 12345678) */
                    "ext v5.8b, v0.8b, v1.8b, #2       \n"      /* vext_s8(vin_r0, vin_r0_h, 2); r(0, 23456789) */
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b  \n"      /* acc_r00 = r(0, 01234567) * w00 */
                    "ext v14.8b, v2.8b, v3.8b, #1      \n"      /* vext_s8(vin_r1, vin_r1_h, 1); r(1, 12345678) */
                    "ext v15.8b, v2.8b, v3.8b, #2      \n"      /* vext_s8(vin_r1, vin_r1_h, 1); r(1, 23456789) */
                    "smlal  v18.8h,  %[v1].8b,  v4.8b  \n"      /* acc_r00 += r(0, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v2].8b,  v5.8b   \n"
                    //r1
                    "ext v4.8b, v6.8b, v7.8b, #1       \n"     /* vext_s8(vin_r2, vin_r2_h, 1); r(2, 12345678) */
                    "ext v5.8b, v6.8b, v7.8b, #2       \n"     /* vext_s8(vin_r2, vin_r2_h, 2); r(2, 23456789) */
                    "smull  v19.8h,  %[v0].8b,  v2.8b   \n"     /* acc_r10 += r(1, 01234567) * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b   \n"     /* acc_r00 += r(1, 01234567) * w10 */
                    "ld1    {v0.8b}, [%[din_ptr0]], #8  \n"        /* load a'(0,7)-a'(0,15) to q0 for prefetch*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8  \n"        /* load a'(1,7)-a'(1,15) to q2 for prefetch*/
                    "smlal  v19.8h,  %[v1].8b,  v14.8b\n"       /* acc_r10 += r(1, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v4].8b,  v14.8b  \n"     /* acc_r00 += r(1, 12345678) * w11 */
                    "ld1    {v1.8b}, [%[din_ptr0]]      \n"     /* load a'(0,7)-a'(0,15) to q1 for prefetch */
                    "ld1    {v3.8b}, [%[din_ptr1]]      \n"     /* load a'(1,7)-a'(1,15) to q3 for prefetch */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v2].8b,  v15.8b  \n"       /* acc_r10 += r(1, 23456789) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b  \n"       /* acc_r00 += r(1, 23456789) * w12 */
                    //r2
                    "ext v14.8b, v8.8b, v9.8b, #1       \n"      /* vext_s8(vin_r3, vin_r3_h, 1); r(3, 12345678) */
                    "ext v15.8b, v8.8b, v9.8b, #2       \n"      /* vext_s8(vin_r3, vin_r3_h, 1); r(3, 23456789) */
                    "smlal  v19.8h,  %[v3].8b,  v6.8b   \n"     /* acc_r10 += r(2, 01234567) * w10 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r10.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r10.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b   \n"     /* acc_r00 += r(2, 01234567) * w20 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b\n"       /* acc_r10 += r(2, 12345678) * w11 */
                    "smlal  v18.8h,  %[v7].8b,  v4.8b\n"       /* acc_r00 += r(2, 12345678) * w21 */
                    "smlal  v19.8h,  %[v5].8b,  v5.8b\n"       /* acc_r10 += r(2, 23456789) * w12 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b  \n"     /* acc_r00 += r(2, 23456789) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high */
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b   \n"     /* acc_r10 = r(3, 01234567) * w20 */
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"     /* load a'(2,0)-a'(2,7) to q6 for prefetch */
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"     /* load a'(3,0)-a'(3,7) to q8 for prefetch */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high */
                    "smlal  v19.8h,  %[v7].8b,  v14.8b   \n"     /* acc_r10 += r(3, 12345678) * w21 */
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"     /* load a'(2,8)-a'(2,15) to q7 for prefetch */
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"     /* load a'(3,8)-a'(3,15) to q9 for prefetch */
                    "scvtf  v10.4s, v10.4s                 \n"  /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s                 \n"  /* int32-> float32*/
                    "dup   v20.4s, %[v_bias].s[0]                \n "
                    "dup   v22.4s, %[v_bias].s[0]                \n "
                    "fmla   v20.4s, v10.4s, %[v_scale].4s        \n"
                    "fmla   v22.4s, v11.4s, %[v_scale].4s        \n"
                    "fmax  v20.4s, v20.4s, v21.4s         \n"  /* relu */
                    "fmax  v22.4s, v22.4s, v21.4s         \n"  /* relu */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b   \n"     /* acc_r10 += r(3, 23456789) * w22 */
                    "stp     q20, q22, [%[ptr_out0]], #32 \n"   /* store q10, q11 -> ptr_out 0  */
                    "movi   v10.4s, #0x0                    \n"
                    "movi   v11.4s, #0x0                    \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high */
                    "subs %[cnt], %[cnt], #1                \n"
                    "scvtf  v12.4s, v12.4s                  \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                  \n"  /* int32-> float32*/
                    "dup   v18.4s, %[v_bias].s[0]                 \n "
                    "dup   v19.4s, %[v_bias].s[0]                 \n "
                    "fmla   v18.4s, v12.4s, %[v_scale].4s          \n"
                    "fmla   v19.4s, v13.4s, %[v_scale].4s          \n"
                    "fmax  v18.4s, v18.4s, v21.4s           \n"    /* relu*/
                    "fmax  v19.4s, v19.4s, v21.4s           \n"    /* relu*/
                    "stp     q18, q19, [%[ptr_out1]], #32   \n"   /* store q12, q13 -> ptr_out 1  */
                    "movi   v12.4s, #0x0                    \n"
                    "movi   v13.4s, #0x0                    \n"
                    "bne 1b                                 \n"
                //right
                    "3:                                      \n" 
                    "ld1 {v20.8b}, [%[vmask]], #8            \n"
                    "ld1 {v22.8b}, [%[vmask]]                \n"
                    "bif v0.8b, v21.8b, v20.8b               \n"  /* a(0, 0) to a(0, 7) */
                    "bif v1.8b, v21.8b, v22.8b               \n"  /* a(0, 8) to a(0, 15) */
                    "bif v2.8b, v21.8b, v20.8b               \n"  /* a(1, 0) to a(1, 7) */
                    "bif v3.8b, v21.8b, v22.8b               \n"  /* a(1, 8) to a(1, 15) */
                    "ext v4.8b, v0.8b, v1.8b, #1             \n"  /* r(0, 12345678) */
                    "ext v5.8b, v0.8b, v1.8b, #2             \n"  /* r(0, 23456789) */
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b        \n"  /* acc_r00 += r(0, 01234567) * w00 */
                    "ext v14.8b, v2.8b, v3.8b, #1            \n"  /* r(1, 12345678) */
                    "ext v15.8b, v2.8b, v3.8b, #2            \n"  /* r(1, 23456789) */
                    "bif v6.8b, v21.8b, v20.8b               \n"  /* a(2, 0) to a(0, 7) */  /* r(2, 12345678) */
                    "bif v7.8b, v21.8b, v22.8b               \n"  /* a(2, 8) to a(2, 15) */ /* r(2, 23456789) */
                    "smlal  v18.8h,  %[v1].8b,  v4.8b        \n"  /* acc_r00 += r(0, 12345678) * w01 */
                    "bif v8.8b, v21.8b, v20.8b               \n"  /* a(3, 0) to a(3, 7) */ 
                    "bif v9.8b, v21.8b, v22.8b               \n"  /* a(3, 8) to a(3, 15) */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v2].8b,  v5.8b        \n"  /* acc_r00 = r(0, 23456789) * w02 */
                    //r1
                    "ext v4.8b, v6.8b, v7.8b, #1            \n"
                    "ext v5.8b, v6.8b, v7.8b, #2            \n"
                    "smull  v19.8h,  %[v0].8b,  v2.8b        \n"  /* acc_r10 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b        \n"  /* acc_r00 = 01234567 * w00 */
                    "ld1 {v20.4s}, [%[rmask]], #16           \n"
                    "ld1 {v22.4s}, [%[rmask]]                \n"
                    "smlal  v19.8h,  %[v1].8b,  v14.8b       \n"  /* acc_r10 += r(1, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v4].8b,  v14.8b       \n"  /* acc_r00 += r(1, 12345678) * w11 */
                    "ld1 {v0.4s}, [%[ptr_out0]], #16         \n"  /* load original ptr_out0 low */
                    "ld1 {v2.4s}, [%[ptr_out1]], #16         \n"  /* load original ptr_out1 low */
                    "saddw   v12.4s, v12.4s, v19.4h          \n"  /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h         \n"  /* v13 += acc_r10.high */
                    "smull  v19.8h,  %[v2].8b,  v15.8b       \n"  /* acc_r10 = r(1, 23456789) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b       \n"  /* acc_r00 = r(1, 23456789) * w12 */
                    "ld1 {v1.4s}, [%[ptr_out0]]              \n"  /* load original ptr_out0 high */
                    "ld1 {v3.4s}, [%[ptr_out1]]              \n"  /* load original ptr_out1 high */
                    //r2
                    "ext v14.8b, v8.8b, v9.8b, #1             \n"  /* r(3, 12345678) */
                    "ext v15.8b, v8.8b, v9.8b, #2             \n"  /* r(3, 23456789) */
                    "smlal  v19.8h,  %[v3].8b,  v6.8b        \n"  /* acc_r10 += r(2, 01234567) * w10 */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b        \n"  /* acc_r00 = r(2, 01234567) * w20 */
                    "sub %[ptr_out0], %[ptr_out0], #16      \n"
                    "sub %[ptr_out1], %[ptr_out1], #16      \n"
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v13 += outr00.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b      \n"   /* acc_r10 += r(2, 12345678) * w11 */
                    "smlal  v18.8h,  %[v7].8b,  v4.8b      \n"   /* acc_r00 += r(2, 12345678) * w21 */
                    "smlal  v19.8h,  %[v5].8b,  v5.8b      \n"   /* acc_r10 += r(2, 23456789) * w12 */
                    "saddw   v10.4s, v10.4s, v18.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h        \n"   /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b       \n"   /* acc_r00 += r(2, 23456789) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += acc_r00.high*/
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b       \n"   /* acc_r10 += r(3, 01234567) * w20 */
                    "saddw   v10.4s, v10.4s, v18.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h        \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v7].8b,  v14.8b      \n"   /* acc_r10 += r(3, 12345678) * w21 */
                    "scvtf  v10.4s, v10.4s                  \n"  /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s                  \n"  /* int32-> float32*/
                    "dup   v6.4s, %[v_bias].s[0]            \n "
                    "dup   v7.4s, %[v_bias].s[0]            \n "
                    "fmla   v6.4s, v10.4s, %[v_scale].4s    \n"
                    "fmla   v7.4s, v11.4s, %[v_scale].4s    \n"
                    "fmax  v6.4s, v6.4s, v21.4s             \n"   /* relu */
                    "fmax  v7.4s, v7.4s, v21.4s             \n"   /* relu */
                    "bif v6.16b, v0.16b, v20.16b            \n"   /* select bit of ptr_out 0 according to rmask */
                    "bif v7.16b, v1.16b, v22.16b            \n"   /* select bit of ptr_out 0 high according to rmask */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += acc_r00.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b      \n"   /* acc_r10 = r(3, 23456789) * w22 */
                    "stp     q6, q7, [%[ptr_out0]], #32     \n"   /* store q10, q11 -> ptr_out 0   */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += outr00.high*/
                    "scvtf  v12.4s, v12.4s                  \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                  \n"  /* int32-> float32*/
                    "dup   v8.4s, %[v_bias].s[0]            \n "
                    "dup   v9.4s, %[v_bias].s[0]            \n "
                    "fmla   v8.4s, v12.4s, %[v_scale].4s    \n"
                    "fmla   v9.4s, v13.4s, %[v_scale].4s    \n"
                    "fmax  v8.4s, v8.4s, v21.4s             \n"   /* relu */
                    "fmax  v9.4s, v9.4s, v21.4s             \n"   /* relu */
                    "bif v8.16b, v2.16b, v20.16b            \n"   /* select bit of ptr_out 1 according to rmask */
                    "bif v9.16b, v3.16b, v22.16b            \n"   /* select bit of ptr_out 1 hihg according to rmask */
                    "stp   q8, q9, [%[ptr_out1]], #32       \n"   /* store q12, q13 -> ptr_out 1  */
                    :   [cnt]"+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [din_ptr3] "+r"(din_ptr3), [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
                        [vmask] "+r" (val_mask), [rmask] "+r" (rst_mask)
                    :   [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), \
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22), \
                        [v_bias]"w"(v_bias), [v_scale] "w"(v_scale)
                    :"cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
                      "v10", "v11", "v12","v13","v14","v15", \
                      "v18",\
                      "v19", "v20", "v21", "v22"
                );
#else
                // store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                \n"
                    "pld [%[din_ptr1]]                \n"
                    "pld [%[din_ptr2]]                \n"
                    "pld [%[din_ptr3]]                \n"
                    "vdup.s8     d2, d0[0]            \n"
                    "vdup.s8     d3, d0[1]            \n"
                    "vdup.s8     d4, d0[2]            \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]  \n"
                    "vmov.u32 d11, #0                 \n"
                    "vmov.u32 q8, #0                  \n"
                    "vmov.u32 q9, #0                  \n"
                    "vmov.u32 q10, #0                 \n"
                    "vmov.u32 q11, #0                 \n"
                    //r0
                    "vmull.s8 q12, d12, d3                 \n" // q12 = d12 * w01
                    "vext.8     d30, d11, d12, #7          \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          \n" //d11 = 12345678
                    "vld1.8 {d12-d13}, [%[din_ptr1]]        \n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]        \n"
                    "vdup.s8     d5, d0[3]                  \n"
                    "vdup.s8     d6, d0[4]                  \n"
                    "vmlal.s8 q12, d30, d2                  \n" // q12 += d10 * w00
                    "vdup.s8     d7, d0[5]                  \n"
                    "add %[din_ptr0], #7                    \n"
                    "add %[din_ptr1], #7                    \n"
                    "vaddw.s16 q8, q8, d24                  \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                  \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                  \n" // q12 += d11 * w02
                    //r1
                    "vext.8     d30, d11, d12, #7           \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1           \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d3                  \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d12, d6                  \n" // q12 = d12 * w11
                    "vld1.8 {d12-d13}, [%[din_ptr3]]        \n"
                    "vdup.s8     d8, d0[6]                  \n"
                    "vdup.s8     d9, d0[7]                  \n"
                    "vdup.s8     d10, d1[0]                 \n"
                    "vaddw.s16 q8, q8, d24                  \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                  \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d30, d2                  \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d5                  \n" // q12 += d10 * w00
                    "add %[din_ptr2], #7                    \n"
                    "add %[din_ptr3], #7                    \n"
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                  \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                  \n" // q12 += d10 * w00
                    //r2
                    "vext.8     d30, d11, d14, #7           \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #1           \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                  \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                  \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d14, d6                  \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d9                  \n" // q13 = d12 * w01
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d5                  \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d8                  \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                  \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                  \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d31, d7                  \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                 \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d11, d12, #7           \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1           \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                  \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                  \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d12, d9                  \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                      \n"
                    "pld [%[din_ptr1]]                      \n"
                    "vmlal.s8 q13, d30, d8                  \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                      \n"
                    "pld [%[din_ptr3]]                      \n"
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                 \n" // q12 += d10 * w00
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q7, %[scale]                   \n"
                    "vdup.32 q14, %[bias]                   \n"
                    "vdup.32 q15, %[bias]                   \n"
                    "vmov.f32 q1, #0.0                      \n"
                    "vcvt.f32.s32   q8, q8                  \n"
                    "vcvt.f32.s32   q9, q9                  \n"
                    "vmla.f32 q14, q8, q7                   \n"
                    "vmla.f32 q15, q9, q7                   \n"
                    "vmax.f32 q14, q14, q1                  \n"
                    "vmax.f32 q15, q15, q1                  \n"
                    "vst1.32 {d28-d29}, [%[dout_ptr1]]!     \n"
                    "vst1.32 {d30-d31}, [%[dout_ptr1]]!     \n"
                    "vdup.32 q12, %[bias]                   \n"
                    "vdup.32 q13, %[bias]                   \n"
                    "vcvt.f32.s32   q10, q10                \n"
                    "vcvt.f32.s32   q11, q11                \n"
                    "vmla.f32 q12, q10, q7                  \n"
                    "vmla.f32 q13, q11, q7                  \n"
                    "vmax.f32 q12, q12, q1                  \n"
                    "vmax.f32 q13, q13, q1                  \n"
                    "vst1.32 {d24-d25}, [%[dout_ptr2]]!     \n"
                    "cmp %[cnt], #1                         \n"
                    "vst1.32 {d26-d27}, [%[dout_ptr2]]!     \n"
                    "blt 1f                                 \n"
                // mid
                    "2:                                     \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]        \n"                    
                    "vdup.s8     d2, d0[0]                  \n"
                    "vdup.s8     d3, d0[1]                  \n"
                    "vmov.u32 q8, #0                        \n"
                    "vmov.u32 q9, #0                        \n"
                    "vmov.u32 q10, #0                       \n"
                    "vmov.u32 q11, #0                       \n"
                     // r0
                    "vmull.s8 q12, d12, d2                  \n" // q12 = d12 * w01
                    "vext.8     d30, d12, d13, #1           \n" //d10 = 12345678
                    "vext.8     d31, d12, d13, #2           \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr1]]        \n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]        \n"
                    "vmlal.s8 q12, d30, d3                  \n" // q12 += d10 * w00
                    "add %[din_ptr0], #8                    \n"
                    "add %[din_ptr1], #8                    \n"
                    "vaddw.s16 q8, q8, d24                  \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                  \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                  \n" // q12 += d11 * w02
                    // r1
                    "vext.8     d30, d12, d13, #1           \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2           \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d2                  \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d12, d5                  \n" // q12 = d12 * w11
                    "vld1.8 {d12-d13}, [%[din_ptr3]]        \n"
                    "vmlal.s8 q13, d30, d3                  \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                  \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                  \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d30, d6                  \n" // q12 += d10 * w00
                    "add %[din_ptr2], #8                    \n"
                    "add %[din_ptr3], #8                    \n"
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                  \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                  \n" // q12 += d10 * w00
                    //r2
                    "vext.8     d30, d14, d15, #1           \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2           \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                  \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                  \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d14, d5                  \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d8                  \n" // q13 = d12 * w01
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6                  \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                  \n" // q12 += d10 * w00
                    "vmlal.s8 q13, d31, d7                  \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                  \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                  \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d10                 \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d12, d13, #1           \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2           \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                  \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                  \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d12, d8                  \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                      \n"
                    "pld [%[din_ptr1]]                      \n"
                    "vmlal.s8 q13, d30, d9                  \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                      \n"
                    "pld [%[din_ptr3]]                      \n"
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                 \n" // q12 += d10 * w00
                    "vdup.32 q7, %[scale]                   \n"
                    "vdup.32 q14, %[bias]                   \n"
                    "vdup.32 q15, %[bias]                   \n"
                    "vmov.f32 q1, #0.0                      \n"
                    "vcvt.f32.s32   q8, q8                  \n"
                    "vcvt.f32.s32   q9, q9                  \n"
                    "vmla.f32 q14, q8, q7                   \n"
                    "vmla.f32 q15, q9, q7                   \n"
                    "vmax.f32 q14, q14, q1                  \n"
                    "vmax.f32 q15, q15, q1                  \n"
                    "vst1.32 {d28-d29}, [%[dout_ptr1]]!     \n"
                    "vst1.32 {d30-d31}, [%[dout_ptr1]]!     \n"
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q12, %[bias]                   \n"
                    "vdup.32 q13, %[bias]                   \n"
                    "vcvt.f32.s32   q10, q10                \n"
                    "vcvt.f32.s32   q11, q11                \n"
                    "vmla.f32 q12, q10, q7                  \n"
                    "vmla.f32 q13, q11, q7                  \n"
                    "vmax.f32 q12, q12, q1                  \n"
                    "vmax.f32 q13, q13, q1                  \n"
                    "vdup.s8     d2, d0[0]                  \n"
                    "vdup.s8     d3, d0[1]                  \n"
                    "vst1.32 {d24-d25}, [%[dout_ptr2]]!     \n"
                    "subs %[cnt], #1                        \n"
                    "vst1.32 {d26-d27}, [%[dout_ptr2]]!     \n"
                    "bne  2b                                \n"
                //right
                    "1:                                     \n"
                    "vdup.s8     d2, d0[0]                  \n"
                    "vdup.s8     d3, d0[1]                  \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]        \n"
                    "vld1.8 {d28-d29}, [%[mask]]            \n"
                    "vmov.u32 q8, #0                        \n"
                    "vmov.u32 q9, #0                        \n"
                    "vmov.u32 q10, #0                       \n"
                    "vmov.u32 q11, #0                       \n"
                    "vbif.8 d12, d11, d28                   \n"
                    "vbif.8 d13, d11, d29                   \n"
                    "vld1.8 {d14-d15}, [%[din_ptr1]]        \n"
                     //r0
                    "vmull.s8 q12, d12, d2                  \n" // q12 = d12 * w01
                    "vext.8 d30, d12, d13, #1               \n" //d10 = 12345678
                    "vext.8 d31, d12, d13, #2               \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr2]]        \n"
                    "vbif.8 d14, d11, d28                   \n"
                    "vbif.8 d15, d11, d29                   \n"
                    "vmlal.s8 q12, d30, d3                  \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                  \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                  \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                  \n" // q12 += d11 * w02
                    //r1
                    "vext.8 d30, d14, d15, #1               \n" //d10 = 00123456
                    "vext.8 d31, d14, d15, #2               \n" //d11 = 12345678
                    "vmull.s8 q13, d14, d2                  \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d14, d5                  \n" // q12 = d12 * w11
                    "vld1.8 {d14-d15}, [%[din_ptr3]]        \n"
                    "vbif.8 d12, d11, d28                   \n"
                    "vbif.8 d13, d11, d29                   \n"
                    "vaddw.s16 q8, q8, d24                  \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                  \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d30, d3                  \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d6                  \n" // q12 += d10 * w00
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                  \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                  \n" // q12 += d10 * w00
                    //r2
                    "vext.8 d30, d12, d13, #1               \n" //d10 = 00123456
                    "vext.8 d31, d12, d13, #2               \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                  \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                  \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d12, d5                  \n" // q13 = d12 * w01
                    "vmull.s8 q12, d12, d8                  \n" // q13 = d12 * w01
                    "vbif.8 d14, d11, d28                   \n"
                    "vbif.8 d15, d11, d29                   \n"
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6                  \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                  \n" // q12 += d10 * w00
                    "vld1.32 {d28-d29}, [%[dout_ptr1]]!     \n"
                    "vld1.32 {d12-d13}, [%[dout_ptr1]]      \n"
                    "vld1.32 {d2-d3}, [%[rs_mask]]!         \n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]          \n"
                    "vaddw.s16 q8, q8, d24                  \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                  \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d31, d7                  \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                 \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d14, d15, #1           \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2           \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                  \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                  \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d14, d8                  \n" // q13 = d12 * w01
                    "vld1.32 {d14-d15}, [%[dout_ptr2]]!     \n"
                    "vld1.32 {d24-d25}, [%[dout_ptr2]]      \n"
                    "vmlal.s8 q13, d30, d9                  \n" // q13 += d10 * w00
                    "sub %[dout_ptr1], #16                  \n"
                    "sub %[dout_ptr2], #16                  \n"
                    "vaddw.s16 q10, q10, d26                \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                 \n" // q12 += d10 * w00
                    "vdup.32 q3, %[scale]                   \n"
                    "vdup.32 q4, %[bias]                    \n"
                    "vdup.32 q5, %[bias]                     \n"
                    "vmov.f32 q15, #0.0                      \n"
                    "vcvt.f32.s32   q8, q8                   \n"
                    "vcvt.f32.s32   q9, q9                   \n"
                    "vmla.f32 q4, q8, q3                     \n"
                    "vmla.f32 q5, q9, q3                     \n"
                    "vmax.f32 q4, q4, q15                    \n"
                    "vmax.f32 q5, q5, q15                    \n"
                    "vbif q4, q14, q1                        \n"
                    "vbif q5, q6, q2                         \n"
                    "vst1.32 {d8-d9}, [%[dout_ptr1]]!        \n"
                    "vst1.32 {d10-d11}, [%[dout_ptr1]]!      \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q13, %[bias]                    \n"
                    "vdup.32 q14, %[bias]                    \n"
                    "vcvt.f32.s32   q10, q10                 \n"
                    "vcvt.f32.s32   q11, q11                 \n"
                    "vmla.f32 q13, q10, q3                   \n"
                    "vmla.f32 q14, q11, q3                   \n"
                    "vmax.f32 q13, q13, q15                  \n"
                    "vmax.f32 q14, q14, q15                  \n"
                    "vbif q13, q7, q1                        \n"
                    "vbif q14, q12, q2                       \n"
                    "vst1.32 {d26-d27}, [%[dout_ptr2]]!      \n"
                    "vst1.32 {d28-d29}, [%[dout_ptr2]]!      \n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2), \
                      [din_ptr3] "+r" (din_ptr3), \
                      [dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [scale] "+r" (scale_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask)
                    :"cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                      "q11", "q12", "q13", "q14", "q15"
                );
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
  }
}

void conv_depthwise_3x3s1p1_bias_relu_int8_int8(int8_t* dout,
                                                const int8_t* din,
                                                const int8_t* weights,
                                                const float* scale,
                                                const float* bias,
                                                bool flag_bias,
                                                int flag_act,
                                                float* alpha,
                                                int num,
                                                int ch_in,
                                                int h_in,
                                                int w_in,
                                                int h_out,
                                                int w_out,
                                                ARMContext* ctx) {
  const unsigned char right_pad_idx[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const unsigned char right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, w_in * sizeof(int8_t));
  int8_t* write_ptr =
      reinterpret_cast<int8_t*>(ctx->workspace_data<int8_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;
  int tile_w = (w_in + 7) >> 3;
  int cnt_col = tile_w - 2;
  unsigned int size_pad_right = (unsigned int)(w_in - 7 - (cnt_col << 3));
  unsigned char rst_remain = (unsigned char)(w_out - ((cnt_col + 1) << 3));
  uint8x8_t vmask_result =
      vcgt_u8(vdup_n_u8(rst_remain), vld1_u8(right_pad_rst));
  uint8x16_t vmask_rp =
      vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
  unsigned char vmask[16];
  vst1q_u8(vmask, vmask_rp);
  unsigned char rmask[8];
  vst1_u8(rmask, vmask_result);

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    int8_t* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      int8_t* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;

      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
      float vscale[4] = {scale_val, scale_val, scale_val, scale_val};
      float vmax[4] = {-127.f, -127.f, -127.f, -127.f};
      float* vmax_ptr = vmax;
#ifdef __aarch64__
      int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
      int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
      int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

      int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
      int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
      int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

      int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
      int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
      int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

      float32x4_t v_bias = vdupq_n_f32(vbias[0]);
      float32x4_t v_scale = vdupq_n_f32(vscale[0]);
#endif
      int8_t* doutr0 = nullptr;
      int8_t* doutr1 = nullptr;
      const int8_t* dr0 = din_ch_ptr;
      const int8_t* dr1 = dr0 + w_in;
      const int8_t* dr2 = dr1 + w_in;
      const int8_t* dr3 = dr2 + w_in;

      const int8_t* din_ptr0 = nullptr;
      const int8_t* din_ptr1 = nullptr;
      const int8_t* din_ptr2 = nullptr;
      const int8_t* din_ptr3 = nullptr;

      for (int i = 0; i < h_in; i += 2) {
        //! process top pad pad_h = 1
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;
        unsigned char* rst_mask = rmask;
        unsigned char* val_mask = vmask;

        if (i == 0) {
          din_ptr0 = zero_ptr;
          din_ptr1 = dr0;
          din_ptr2 = dr1;
          din_ptr3 = dr2;
          dr0 = dr1;
          dr1 = dr2;
          dr2 = dr3;
          dr3 = dr2 + w_in;
        } else {
          dr0 = dr2;
          dr1 = dr3;
          dr2 = dr1 + w_in;
          dr3 = dr2 + w_in;
        }
        //! process bottom pad
        if (i + 3 > h_in) {
          switch (i + 3 - h_in) {
            case 3:
              din_ptr1 = zero_ptr;
            case 2:
              din_ptr2 = zero_ptr;
            case 1:
              din_ptr3 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 2 > h_out) {
          doutr1 = write_ptr;
        }
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr3]] \n"
                    "movi   v21.4s, #0x0\n"                                     /* v21 = 0 */
                // left
                    "ld1    {v0.8b}, [%[din_ptr0]], #8               \n"   /* load a(0,0)-a(0,7) to q0*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8               \n"   /* load a(1,0)-a(1.7) to q2*/
                    "ld1    {v1.8b}, [%[din_ptr0]]                   \n"   /* load a(0,8)-a(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]                   \n"   /* load a(1,8)-a(1,15) to q3*/
                    "movi   v10.4s, #0x0\n"                             /* init int32 acc v10 to 0 */
                    "movi   v11.4s, #0x0\n"                             /* init int32 acc v11 to 0 */
                    "movi   v12.4s, #0x0\n"                             /* init int32 acc v12 to 0 */
                    "movi   v13.4s, #0x0\n"                             /* init int32 acc v13 to 0 */
                    //r0
                    "smull  v18.8h,  %[v1].8b,  v0.8b   \n"   /* acc_r(0,0) = r(0,01234567) * w01 */
                    "ext v4.8b, v21.8b, v0.8b, #7       \n"   /* vext_s8(vzero, vin_r0, 7); r(0,00123456) */
                    "ext v5.8b, v0.8b, v1.8b, #1        \n"   /* vext_s8(vin_r0, vin_r0_h, 1); r(0,12345678) */
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a(2,0)-a(2,7) to q6*/
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a(3,0)-a(3,7) to q8*/
                    "smlal  v18.8h,  %[v0].8b,  v4.8b   \n"   /* acc_r(0,0) += r(0,00123456) * w00 */
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a(2,8)-a(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a(3,8)-a(3,15) to q9*/
                    "sub   %[din_ptr0], %[din_ptr0], #1 \n"
                    "sub   %[din_ptr1], %[din_ptr1], #1 \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v2].8b,  v5.8b \n"   /* acc_r00 += r(0,12345678) * w02 */
                    "ext v14.8b, v21.8b, v2.8b, #7       \n"   /* vext_s8(vzero, vin_r1, 7); r(1,00123456) */
                    "ext v15.8b, v2.8b, v3.8b, #1        \n"   /* vext_s8(vin_r1, vin_r1_h, 1); r(1,12345678) */
                    //r1
                    "sub   %[din_ptr2], %[din_ptr2], #1 \n"
                    "sub   %[din_ptr3], %[din_ptr3], #1 \n"
                    "smull  v19.8h,  %[v1].8b,  v2.8b   \n"   /* acc_r10 += r(1,01234567) * w01 */
                    "smlal  v18.8h,  %[v4].8b,  v2.8b   \n"   /* acc_r00 += r(1,01234567) * w11 */
                    "ext v4.8b, v21.8b, v6.8b, #7      \n"
                    "ext v5.8b, v6.8b, v7.8b, #1       \n"
                    "smlal  v19.8h,  %[v0].8b,  v14.8b   \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v3].8b,  v14.8b   \n"   /* acc_r00 += r(1, 00123456) * w10 */
                    "ld1    {v0.8b}, [%[din_ptr0]], #8  \n"   /* load a'(0,0)-a'(0,7) to q0 for prefetch*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8  \n"   /* load a'(1,0)-a'(1,7) to q2 for prefetch*/
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v2].8b,  v15.8b   \n"   /* acc_r10 += r(1,12345678) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b   \n"   /* acc_r00 += r(1,12345678) * w12 */
                    //r2
                    "ld1    {v1.8b}, [%[din_ptr0]]      \n"   /* load a'(0,8)-a'(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]      \n"   /* load a'(1,8)-a'(1,15) to q3*/
                    "smlal  v19.8h,  %[v4].8b,  v6.8b   \n"   /* acc_r10 += r(2,01234567) * w11 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v7].8b,  v6.8b   \n"   /* acc_r00 += r(2,01234567) * w21 */
                    "ext v14.8b, v21.8b, v8.8b, #7       \n"
                    "ext v15.8b, v8.8b, v9.8b, #1        \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v3].8b,  v4.8b  \n"   /* acc_r10 += r(2, 00123456) * w10 */
                    "smlal  v18.8h,  %[v6].8b,  v4.8b  \n"
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a'(2,0)-a'(2,7) to q6 for prefetch*/
                    "smlal  v19.8h,  %[v5].8b,  v5.8b  \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b  \n"   /* acc_r00 += r(2, 12345678) * w22 */
                    "ld1 {v4.4s}, [%[vmax_ptr]]        \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v12 += outr10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v13 += outr10.high*/
                    //r3
                    "smull  v19.8h,  %[v7].8b,  v8.8b   \n"   /* acc_r10 += r(3, 01234567) * w21 */
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a'(3,0)-a'(3,7) to q8 for prefetch*/
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a'(2,7)-a'(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a'(3,7)-a'(3,15) to q9*/
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v6].8b,  v14.8b   \n"   /* acc_r10 += r(3, 00123456) * w20) */
                    "dup   v20.4s, %[v_bias].s[0]        \n "  /* dup bias to v20 */
                    "dup   v22.4s, %[v_bias].s[0]        \n "  /* dup bias to v22 */
                    "scvtf  v10.4s, v10.4s               \n"   /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s               \n"   /* int32-> float32*/
                    "fmla   v20.4s, v10.4s, %[v_scale].4s   \n"  /* mul scale, add bias */
                    "fmla   v22.4s, v11.4s, %[v_scale].4s   \n"  /* mul scale, add bias */
                    "fmax  v20.4s, v20.4s, v21.4s        \n"  /* relu */
                    "fmax  v22.4s, v22.4s, v21.4s        \n"  /* relu */
                    "fcmge v11.4s, v20.4s, v4.4s             \n"
                    "bif v20.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v22.4s, v4.4s             \n"
                    "bif v22.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v20.4s, v20.4s                   \n" /* fp32 - int32 */
                    "fcvtas  v22.4s, v22.4s                   \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v20.4s                   \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v22.4s                  \n" /* int32-int16 */
                    "sqxtn   v20.8b, v21.8h                   \n" /* int16-int8 */
                    "st1    {v20.8b}, [%[ptr_out0]], #8       \n" /* store q10, q11 -> ptr_out 0 */
                    "saddw   v12.4s, v12.4s, v19.4h           \n" /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h          \n" /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b        \n" /* acc_r10 += r(3, 12345678) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h           \n" /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h          \n" /* v13 += acc_r10.high*/
                    "movi   v21.4s, #0x0                      \n" /* v21 = 0 */
                    "scvtf  v12.4s, v12.4s                    \n" /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                    \n" /* int32-> float32*/
                    "dup   v14.4s, %[v_bias].s[0]             \n" /* dup bias to v14 */
                    "dup   v15.4s, %[v_bias].s[0]             \n" /* dup bias to v15 */
                    "fmla   v14.4s, v12.4s, %[v_scale].4s     \n" /* mul scale, add bias */
                    "fmla   v15.4s, v13.4s, %[v_scale].4s     \n" /* mul scale, add bias */
                    "fmax  v14.4s, v14.4s, v21.4s             \n" /* relu*/
                    "fmax  v15.4s, v15.4s, v21.4s             \n" /* relu*/
                    "fcmge v11.4s, v14.4s, v4.4s              \n"
                    "bif v14.16b, v4.16b, v11.16b             \n" /* choose data */
                    "fcmge v11.4s, v15.4s, v4.4s              \n"
                    "bif v15.16b, v4.16b, v11.16b             \n" /* choose data */
                    "fcvtas  v14.4s, v14.4s                   \n" /* fp32 - int32 */
                    "fcvtas  v15.4s, v15.4s                   \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v14.4s                   \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v15.4s                  \n" /* int32-int16 */
                    "sqxtn   v14.8b, v21.8h                   \n" /* int16-int8 */
                    "st1    {v14.8b}, [%[ptr_out1]], #8       \n" /* store q10, q11 -> ptr_out 0 */
                    "movi   v10.4s, #0x0                      \n" /* clear 0 for v10 */
                    "movi   v11.4s, #0x0                      \n" /* clear 0 for v11 */
                    "movi   v12.4s, #0x0                      \n"
                    "movi   v13.4s, #0x0                      \n"
                    "movi   v21.4s, #0x0                      \n"
                    "cmp  %[cnt], #1                          \n"
                    "blt 3f                                   \n"
                //mid
                    "1:                                       \n"
                    "ext v4.8b, v0.8b, v1.8b, #1              \n" /* vext_s8(vin_r0, vin_r0_h, 1); r(0, 12345678) */
                    "ext v5.8b, v0.8b, v1.8b, #2              \n" /* vext_s8(vin_r0, vin_r0_h, 2); r(0, 23456789) */
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b         \n" /* acc_r00 = r(0, 01234567) * w00 */
                    "ext v14.8b, v2.8b, v3.8b, #1             \n" /* vext_s8(vin_r1, vin_r1_h, 1); r(1, 12345678) */
                    "ext v15.8b, v2.8b, v3.8b, #2       \n"      /* vext_s8(vin_r1, vin_r1_h, 1); r(1, 23456789) */
                    "smlal  v18.8h,  %[v1].8b,  v4.8b   \n"      /* acc_r00 += r(0, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v2].8b,  v5.8b   \n"
                    //r1
                    "ext v4.8b, v6.8b, v7.8b, #1        \n"     /* vext_s8(vin_r2, vin_r2_h, 1); r(2, 12345678) */
                    "ext v5.8b, v6.8b, v7.8b, #2        \n"     /* vext_s8(vin_r2, vin_r2_h, 2); r(2, 23456789) */
                    "smull  v19.8h,  %[v0].8b,  v2.8b   \n"     /* acc_r10 += r(1, 01234567) * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b   \n"     /* acc_r00 += r(1, 01234567) * w10 */
                    "ld1    {v0.8b}, [%[din_ptr0]], #8  \n"     /* load a'(0,7)-a'(0,15) to q0 for prefetch*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8  \n"     /* load a'(1,7)-a'(1,15) to q2 for prefetch*/
                    "smlal  v19.8h,  %[v1].8b,  v14.8b  \n"     /* acc_r10 += r(1, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v4].8b,  v14.8b  \n"     /* acc_r00 += r(1, 12345678) * w11 */
                    "ld1    {v1.8b}, [%[din_ptr0]]      \n"     /* load a'(0,7)-a'(0,15) to q1 for prefetch */
                    "ld1    {v3.8b}, [%[din_ptr1]]      \n"     /* load a'(1,7)-a'(1,15) to q3 for prefetch */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v2].8b,  v15.8b  \n"     /* acc_r10 += r(1, 23456789) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b  \n"     /* acc_r00 += r(1, 23456789) * w12 */
                    //r2
                    "ext v14.8b, v8.8b, v9.8b, #1       \n"     /* vext_s8(vin_r3, vin_r3_h, 1); r(3, 12345678) */
                    "ext v15.8b, v8.8b, v9.8b, #2       \n"     /* vext_s8(vin_r3, vin_r3_h, 1); r(3, 23456789) */
                    "smlal  v19.8h,  %[v3].8b,  v6.8b   \n"     /* acc_r10 += r(2, 01234567) * w10 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r10.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r10.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b   \n"     /* acc_r00 += r(2, 01234567) * w20 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b   \n"     /* acc_r10 += r(2, 12345678) * w11 */
                    "smlal  v18.8h,  %[v7].8b,  v4.8b   \n"     /* acc_r00 += r(2, 12345678) * w21 */
                    "smlal  v19.8h,  %[v5].8b,  v5.8b   \n"     /* acc_r10 += r(2, 23456789) * w12 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b   \n"     /* acc_r00 += r(2, 23456789) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"     /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"     /* v13 += acc_r10.high */
                    "ld1 {v4.4s}, [%[vmax_ptr]]         \n"
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b   \n"     /* acc_r10 = r(3, 01234567) * w20 */
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"     /* load a'(2,0)-a'(2,7) to q6 for prefetch */
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"     /* load a'(3,0)-a'(3,7) to q8 for prefetch */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"     /* v10 += acc_r00.low */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"     /* v11 += acc_r00.high */
                    "smlal  v19.8h,  %[v7].8b,  v14.8b  \n"     /* acc_r10 += r(3, 12345678) * w21 */
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"     /* load a'(2,8)-a'(2,15) to q7 for prefetch */
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"     /* load a'(3,8)-a'(3,15) to q9 for prefetch */
                    "scvtf  v10.4s, v10.4s              \n"  /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s              \n"  /* int32-> float32*/
                    "dup   v20.4s, %[v_bias].s[0]                \n "
                    "dup   v22.4s, %[v_bias].s[0]                \n "
                    "fmla   v20.4s, v10.4s, %[v_scale].4s        \n"
                    "fmla   v22.4s, v11.4s, %[v_scale].4s        \n"
                    "fmax  v20.4s, v20.4s, v21.4s         \n"  /* relu */
                    "fmax  v22.4s, v22.4s, v21.4s         \n"  /* relu */
                    "saddw   v12.4s, v12.4s, v19.4h       \n"  /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h      \n"  /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b    \n"  /* acc_r10 += r(3, 23456789) * w22 */
                    "fcmge v11.4s, v20.4s, v4.4s             \n"
                    "bif v20.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v22.4s, v4.4s             \n"
                    "bif v22.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v20.4s, v20.4s                  \n" /* fp32 - int32 */
                    "fcvtas  v22.4s, v22.4s                  \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v20.4s                  \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v22.4s                 \n" /* int32-int16 */
                    "sqxtn   v20.8b, v21.8h                  \n" /* int16-int8 */
                    "st1    {v20.8b}, [%[ptr_out0]], #8      \n" /* store q10, q11 -> ptr_out 0 */
                    "saddw   v12.4s, v12.4s, v19.4h          \n" /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h         \n" /* v13 += acc_r10.high */
                    "subs %[cnt], %[cnt], #1                 \n"
                    "movi   v21.4s, #0x0                     \n"
                    "scvtf  v12.4s, v12.4s                   \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                   \n"  /* int32-> float32*/
                    "dup   v18.4s, %[v_bias].s[0]            \n "
                    "dup   v19.4s, %[v_bias].s[0]            \n "
                    "fmla   v18.4s, v12.4s, %[v_scale].4s    \n"
                    "fmla   v19.4s, v13.4s, %[v_scale].4s    \n"
                    "fmax  v18.4s, v18.4s, v21.4s            \n"    /* relu*/
                    "fmax  v19.4s, v19.4s, v21.4s            \n"    /* relu*/
                    "fcmge v11.4s, v18.4s, v4.4s             \n"
                    "bif v18.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v19.4s, v4.4s             \n"
                    "bif v19.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v18.4s, v18.4s                  \n" /* fp32 - int32 */
                    "fcvtas  v19.4s, v19.4s                  \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v18.4s                  \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v19.4s                 \n" /* int32-int16 */
                    "sqxtn   v18.8b, v21.8h                  \n" /* int16-int8 */
                    "st1    {v18.8b}, [%[ptr_out1]], #8      \n"  /* store q10, q11 -> ptr_out 0 */
                    "movi   v10.4s, #0x0                     \n"
                    "movi   v11.4s, #0x0                     \n"
                    "movi   v12.4s, #0x0                     \n"
                    "movi   v13.4s, #0x0                     \n"
                    "movi   v21.4s, #0x0                     \n"
                    "bne 1b                                  \n"
                //right
                    "3:                                      \n" 
                    "ld1 {v20.8b}, [%[vmask]], #8            \n"
                    "ld1 {v22.8b}, [%[vmask]]                \n"
                    "bif v0.8b, v21.8b, v20.8b               \n"  /* a(0, 0) to a(0, 7) */
                    "bif v1.8b, v21.8b, v22.8b               \n"  /* a(0, 8) to a(0, 15) */
                    "bif v2.8b, v21.8b, v20.8b               \n"  /* a(1, 0) to a(1, 7) */
                    "bif v3.8b, v21.8b, v22.8b               \n"  /* a(1, 8) to a(1, 15) */
                    "ext v4.8b, v0.8b, v1.8b, #1             \n"  /* r(0, 12345678) */
                    "ext v5.8b, v0.8b, v1.8b, #2             \n"  /* r(0, 23456789) */
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b        \n"  /* acc_r00 += r(0, 01234567) * w00 */
                    "ext v14.8b, v2.8b, v3.8b, #1            \n"  /* r(1, 12345678) */
                    "ext v15.8b, v2.8b, v3.8b, #2            \n"  /* r(1, 23456789) */
                    "bif v6.8b, v21.8b, v20.8b               \n"  /* a(2, 0) to a(0, 7) */  /* r(2, 12345678) */
                    "bif v7.8b, v21.8b, v22.8b               \n"  /* a(2, 8) to a(2, 15) */ /* r(2, 23456789) */
                    "smlal  v18.8h,  %[v1].8b,  v4.8b        \n"  /* acc_r00 += r(0, 12345678) * w01 */
                    "bif v8.8b, v21.8b, v20.8b               \n"  /* a(3, 0) to a(3, 7) */ 
                    "bif v9.8b, v21.8b, v22.8b               \n"  /* a(3, 8) to a(3, 15) */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v2].8b,  v5.8b        \n"  /* acc_r00 = r(0, 23456789) * w02 */
                    //r1
                    "ext v4.8b, v6.8b, v7.8b, #1             \n"
                    "ext v5.8b, v6.8b, v7.8b, #2             \n"
                    "smull  v19.8h,  %[v0].8b,  v2.8b        \n"  /* acc_r10 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b        \n"  /* acc_r00 = 01234567 * w00 */
                    "ld1 {v20.8b}, [%[rmask]], #8            \n"
                    "smlal  v19.8h,  %[v1].8b,  v14.8b       \n"  /* acc_r10 += r(1, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v4].8b,  v14.8b       \n"  /* acc_r00 += r(1, 12345678) * w11 */
                    "ld1 {v0.8b}, [%[ptr_out0]]              \n"  /* load original ptr_out0 low */
                    "ld1 {v2.8b}, [%[ptr_out1]]              \n"  /* load original ptr_out1 low */
                    "saddw   v12.4s, v12.4s, v19.4h          \n"  /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h         \n"  /* v13 += acc_r10.high */
                    "smull  v19.8h,  %[v2].8b,  v15.8b       \n"  /* acc_r10 = r(1, 23456789) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b       \n"  /* acc_r00 = r(1, 23456789) * w12 */
                    //r2
                    "ext v14.8b, v8.8b, v9.8b, #1             \n"  /* r(3, 12345678) */
                    "ext v15.8b, v8.8b, v9.8b, #2             \n"  /* r(3, 23456789) */
                    "smlal  v19.8h,  %[v3].8b,  v6.8b         \n"  /* acc_r10 += r(2, 01234567) * w10 */
                    "saddw   v10.4s, v10.4s, v18.4h           \n"  /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h          \n"  /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b         \n"  /* acc_r00 = r(2, 01234567) * w20 */
                    "saddw   v12.4s, v12.4s, v19.4h           \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h          \n"   /* v13 += outr00.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b         \n"   /* acc_r10 += r(2, 12345678) * w11 */
                    "smlal  v18.8h,  %[v7].8b,  v4.8b         \n"   /* acc_r00 += r(2, 12345678) * w21 */
                    "smlal  v19.8h,  %[v5].8b,  v5.8b         \n"   /* acc_r10 += r(2, 23456789) * w12 */
                    "saddw   v10.4s, v10.4s, v18.4h           \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h          \n"   /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b         \n"   /* acc_r00 += r(2, 23456789) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h           \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h          \n"   /* v11 += acc_r00.high*/
                    "ld1 {v4.4s}, [%[vmax_ptr]]               \n"
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b         \n"   /* acc_r10 += r(3, 01234567) * w20 */
                    "saddw   v10.4s, v10.4s, v18.4h           \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h          \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v7].8b,  v14.8b        \n"   /* acc_r10 += r(3, 12345678) * w21 */
                    "scvtf  v10.4s, v10.4s                    \n"  /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s                    \n"  /* int32-> float32*/
                    "dup   v6.4s, %[v_bias].s[0]              \n "
                    "dup   v7.4s, %[v_bias].s[0]              \n "
                    "fmla   v6.4s, v10.4s, %[v_scale].4s      \n"
                    "fmla   v7.4s, v11.4s, %[v_scale].4s      \n"
                    "fmax  v6.4s, v6.4s, v21.4s               \n"   /* relu */
                    "fmax  v7.4s, v7.4s, v21.4s               \n"   /* relu */
                    "saddw   v12.4s, v12.4s, v19.4h           \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h          \n"   /* v11 += acc_r00.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b        \n"   /* acc_r10 = r(3, 23456789) * w22 */
                    "fcmge v11.4s, v6.4s, v4.4s               \n"
                    "bif v6.16b, v4.16b, v11.16b              \n" /* choose data */
                    "fcmge v11.4s, v7.4s, v4.4s               \n"
                    "bif v7.16b, v4.16b, v11.16b              \n" /* choose data */
                    "fcvtas  v6.4s, v6.4s                     \n" /* fp32 - int32 */
                    "fcvtas  v7.4s, v7.4s                     \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v6.4s                    \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v7.4s                   \n" /* int32-int16 */
                    "sqxtn   v6.8b, v21.8h                    \n" /* int16-int8 */
                    "bif v6.8b, v0.8b, v20.8b                 \n"
                    "st1    {v6.8b}, [%[ptr_out0]], #8        \n"  /* store q10, q11 -> ptr_out 0 */
                    "saddw   v12.4s, v12.4s, v19.4h           \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h          \n"   /* v11 += outr00.high*/
                    "movi   v21.4s, #0x0                      \n"
                    "scvtf  v12.4s, v12.4s                    \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                    \n"  /* int32-> float32*/
                    "dup   v8.4s, %[v_bias].s[0]              \n "
                    "dup   v9.4s, %[v_bias].s[0]              \n "
                    "fmla   v8.4s, v12.4s, %[v_scale].4s      \n"
                    "fmla   v9.4s, v13.4s, %[v_scale].4s      \n"
                    "fmax  v8.4s, v8.4s, v21.4s               \n"   /* relu */
                    "fmax  v9.4s, v9.4s, v21.4s               \n"   /* relu */
                    "fcmge v11.4s, v8.4s, v4.4s               \n"
                    "bif v8.16b, v4.16b, v11.16b              \n" /* choose data */
                    "fcmge v11.4s, v9.4s, v4.4s               \n"
                    "bif v9.16b, v4.16b, v11.16b              \n" /* choose data */
                    "fcvtas  v8.4s, v8.4s                     \n" /* fp32 - int32 */
                    "fcvtas  v9.4s, v9.4s                     \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v8.4s                    \n" /* int32-int16 */
                    "sqxtn2  v21.8h, v9.4s                    \n" /* int32-int16 */
                    "sqxtn   v8.8b, v21.8h                    \n" /* int16-int8 */
                    "bif v8.8b, v2.8b, v20.8b                 \n"
                    "st1    {v8.8b}, [%[ptr_out1]], #8        \n"  /* store q10, q11 -> ptr_out 0 */
                    :   [cnt]"+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [din_ptr3] "+r"(din_ptr3), [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
                        [vmask] "+r" (val_mask), [rmask] "+r" (rst_mask)
                    :   [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), \
                        [vmax_ptr]"r"(vmax_ptr), \
                        //[bias_val] "r" (vbias), [scale_val] "r"(vscale),
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22), \
                        [v_bias]"w"(v_bias), [v_scale] "w"(v_scale)
                    :"cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
                      "v10", "v11", "v12","v13","v14","v15", \
                      "v18",\
                      "v19", "v20", "v21", "v22"
                );
#else
                // store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                //left
                    "pld [%[din_ptr0]]                    \n"
                    "pld [%[din_ptr1]]                    \n"
                    "pld [%[din_ptr2]]                    \n"
                    "pld [%[din_ptr3]]                    \n"
                    "vdup.s8     d2, d0[0]                \n"
                    "vdup.s8     d3, d0[1]                \n"
                    "vdup.s8     d4, d0[2]                \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]      \n"
                    "vmov.u32 d11, #0                     \n"
                    "vmov.u32 q8, #0                      \n"
                    "vmov.u32 q9, #0                      \n"
                    "vmov.u32 q10, #0                     \n"
                    "vmov.u32 q11, #0                     \n"
                    //r0
                    "vmull.s8 q12, d12, d3                \n" // q12 = d12 * w01
                    "vext.8     d30, d11, d12, #7         \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1         \n" //d11 = 12345678
                    "vld1.8 {d12-d13}, [%[din_ptr1]]      \n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]      \n"
                    "vdup.s8     d5, d0[3]                \n"
                    "vdup.s8     d6, d0[4]                \n"
                    "vmlal.s8 q12, d30, d2                \n" // q12 += d10 * w00
                    "vdup.s8     d7, d0[5]                \n"
                    "add %[din_ptr0], #7                  \n"
                    "add %[din_ptr1], #7                  \n"
                    "vaddw.s16 q8, q8, d24                \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                \n" // q12 += d11 * w02
                    //r1
                    "vext.8     d30, d11, d12, #7         \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1         \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d3                \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d12, d6                \n" // q12 = d12 * w11
                    "vld1.8 {d12-d13}, [%[din_ptr3]]      \n"
                    "vdup.s8     d8, d0[6]                \n"
                    "vdup.s8     d9, d0[7]                \n"
                    "vdup.s8     d10, d1[0]               \n"
                    "vaddw.s16 q8, q8, d24                \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d30, d2                \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d5                \n" // q12 += d10 * w00
                    "add %[din_ptr2], #7                  \n"
                    "add %[din_ptr3], #7                  \n"
                    "vaddw.s16 q10, q10, d26              \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27              \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                \n" // q12 += d10 * w00
                    //r2
                    "vext.8     d30, d11, d14, #7         \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #1          \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d14, d6                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d9                 \n" // q13 = d12 * w01
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d5                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d8                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d11, d12, #7          \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #1          \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d12, d9                 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                     \n"
                    "pld [%[din_ptr1]]                     \n"
                    "vmlal.s8 q13, d30, d8                 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                     \n"
                    "pld [%[din_ptr3]]                     \n"
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                \n" // q12 += d10 * w00
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q7, %[scale]                  \n"
                    "vdup.32 q14, %[bias]                  \n"
                    "vdup.32 q15, %[bias]                  \n"
                    "vmov.f32 q1, #0.0                     \n"
                    "vcvt.f32.s32   q8, q8                 \n"
                    "vcvt.f32.s32   q9, q9                 \n"
                    "vmla.f32 q14, q8, q7                  \n"
                    "vmla.f32 q15, q9, q7                  \n"
                    "vmax.f32 q14, q14, q1                 \n"
                    "vmax.f32 q15, q15, q1                 \n"
                    "vmov.f32 q8, #-0.5                    \n"
                    "vmov.f32 q9, #0.5                     \n"
                    "vcgt.f32   q1, q14, q8                \n"
                    "vbif.f32   q9, q8, q1                 \n"
                    "vadd.f32   q14, q14, q9               \n"
                    "vmov.f32   q9, #0.5                   \n"
                    "vcgt.f32   q2, q15, q8                \n"
                    "vbif.f32   q9, q8, q2                 \n"
                    "vadd.f32   q15, q15, q9               \n"
                    "vld1.32 {d2-d3}, [%[vmax]]            \n"
                    "vcge.f32 q3, q14, q1                  \n" /* data >= -127 */
                    "vbif q14, q1, q3                      \n" /* choose data */
                    "vcge.f32 q3, q15, q1                  \n" /* data >= -127 */
                    "vbif q15, q1, q3                      \n" /* choose data */
                    "vcvt.s32.f32  q1, q14    \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q15    \n"  /* fp32 to int32 */
                    "vqmovn.s32 d28, q1       \n"  /* int32 to int16 */
                    "vqmovn.s32 d29, q2       \n"  /* int32 to int16 */
                    "vqmovn.s16 d10, q14      \n"   /* int16 to int8 */
                    "vst1.8    {d10}, [%[dout_ptr1]]!\n"
                    "vdup.32 q12, %[bias]                   \n"
                    "vdup.32 q13, %[bias]                   \n"
                    "vmov.f32 q1, #0.0                      \n"
                    "vcvt.f32.s32   q10, q10                \n"
                    "vcvt.f32.s32   q11, q11                \n"
                    "vmla.f32 q12, q10, q7                  \n"
                    "vmla.f32 q13, q11, q7                  \n"
                    "vmax.f32 q12, q12, q1                  \n"
                    "vmax.f32 q13, q13, q1                  \n"
                    "vmov.f32 q8, #-0.5                     \n"
                    "vmov.f32 q9, #0.5                      \n"
                    "vcgt.f32   q1, q12, q8                 \n"
                    "vbif.f32   q9, q8, q1                  \n"
                    "vadd.f32   q12, q12, q9                \n"
                    "vmov.f32   q9, #0.5                    \n"
                    "vcgt.f32   q2, q13, q8                 \n"
                    "vbif.f32   q9, q8, q2                  \n"
                    "vadd.f32   q13, q13, q9                \n"
                    "vld1.32 {d2-d3}, [%[vmax]]             \n"
                    "vcge.f32 q3, q12, q1                   \n" /* data >= -127 */
                    "vbif q12, q1, q3                       \n" /* choose data */
                    "vcge.f32 q3, q13, q1                   \n" /* data >= -127 */
                    "vbif q13, q1, q3                       \n" /* choose data */
                    "vcvt.s32.f32  q1, q12                  \n" /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q13                  \n" /* fp32 to int32 */
                    "vqmovn.s32 d24, q1                     \n" /* int32 to int16 */
                    "vqmovn.s32 d25, q2                     \n" /* int32 to int16 */
                    "vqmovn.s16 d9, q12                     \n" /* int16 to int8 */
                    "vst1.8    {d9}, [%[dout_ptr2]]!        \n"
                    "cmp %[cnt], #1                         \n"
                    "blt 1f                                 \n"
                //mid
                    "2:                                     \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]        \n"
                    "vdup.s8     d2, d0[0]               \n"
                    "vdup.s8     d3, d0[1]               \n"
                    "vdup.s8     d4, d0[2]               \n"
                    "vdup.s8     d5, d0[3]               \n"
                    "vdup.s8     d6, d0[4]               \n"
                    "vdup.s8     d7, d0[5]               \n"
                    "vdup.s8     d8, d0[6]               \n"
                    "vdup.s8     d9, d0[7]               \n"
                    "vdup.s8     d10, d1[0]              \n"
                    "vmov.u32 q8, #0                        \n"
                    "vmov.u32 q9, #0                        \n"
                    "vmov.u32 q10, #0                       \n"
                    "vmov.u32 q11, #0                       \n"                    
                     //r0
                    "vmull.s8 q12, d12, d2                 \n" // q12 = d12 * w01
                    "vext.8     d30, d12, d13, #1          \n" //d10 = 12345678
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr1]]       \n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]       \n"
                    "vmlal.s8 q12, d30, d3                 \n" // q12 += d10 * w00
                    "add %[din_ptr0], #8                   \n"
                    "add %[din_ptr1], #8                   \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                 \n" // q12 += d11 * w02
                    //r1
                    "vext.8     d30, d12, d13, #1          \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d2                 \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d12, d5                 \n" // q12 = d12 * w11
                    "vld1.8 {d12-d13}, [%[din_ptr3]]       \n"
                    "vmlal.s8 q13, d30, d3                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d30, d6                 \n" // q12 += d10 * w00
                    "add %[din_ptr2], #8                   \n"
                    "add %[din_ptr3], #8                   \n"
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 \n" // q12 += d10 * w00
                    //r2
                    "vext.8     d30, d14, d15, #1          \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d14, d5                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d8                 \n" // q13 = d12 * w01
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 \n" // q12 += d10 * w00
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d10                \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d12, d13, #1          \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d12, d8                 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                     \n"
                    "pld [%[din_ptr1]]                     \n"
                    "vmlal.s8 q13, d30, d9                 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                     \n"
                    "pld [%[din_ptr3]]                     \n"
                    "vaddw.s16 q10, q10, d26               \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27               \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                \n" // q12 += d10 * w00
                    "vdup.32 q7, %[scale]                            \n"
                    "vdup.32 q14, %[bias]                            \n"
                    "vdup.32 q15, %[bias]                            \n"
                    "vmov.f32 q1, #0.0                       \n"
                    "vcvt.f32.s32   q8, q8                                 \n"
                    "vcvt.f32.s32   q9, q9                                 \n"
                    "vmla.f32 q14, q8, q7                            \n"
                    "vmla.f32 q15, q9, q7                            \n"
                    "vmax.f32 q14, q14, q1              \n"
                    "vmax.f32 q15, q15, q1              \n"
                    "vmov.f32 q8, #-0.5\n"
                    "vmov.f32 q9, #0.5\n"
                    "vcgt.f32   q1, q14, q8   \n"
                    "vbif.f32   q9, q8, q1   \n"
                    "vadd.f32   q14, q14, q9\n"
                    "vmov.f32   q9, #0.5\n"
                    "vcgt.f32   q2, q15, q8   \n"
                    "vbif.f32   q9, q8, q2   \n"
                    "vadd.f32   q15, q15, q9\n"
                    "vld1.32 {d2-d3}, [%[vmax]] \n"//"vld1.32 {d22-d23}, [%[vmax]] \n"
                    "vcge.f32 q3, q14, q1         \n" /* data >= -127 */
                    "vbif q14, q1, q3    \n" /* choose data */
                    "vcge.f32 q3, q15, q1         \n" /* data >= -127 */
                    "vbif q15, q1, q3    \n" /* choose data */
                    "vcvt.s32.f32  q1, q14    \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q15    \n"  /* fp32 to int32 */
                    "vqmovn.s32 d28, q1       \n"  /* int32 to int16 */
                    "vqmovn.s32 d29, q2       \n"  /* int32 to int16 */
                    "vqmovn.s16 d10, q14      \n"   /* int16 to int8 */
                    "vst1.8    {d10}, [%[dout_ptr1]]!\n"        //???? vst1.8 or vst1.32
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q12, %[bias]                            \n"
                    "vdup.32 q13, %[bias]                            \n"
                    "vmov.f32 q1, #0.0                       \n"
                    "vcvt.f32.s32   q10, q10                      \n"
                    "vcvt.f32.s32   q11, q11                      \n"
                    "vmla.f32 q12, q10, q7                     \n"
                    "vmla.f32 q13, q11, q7                     \n"
                    "vmax.f32 q12, q12, q1              \n"
                    "vmax.f32 q13, q13, q1              \n"
                    "vmov.f32 q8, #-0.5\n"
                    "vmov.f32 q9, #0.5\n"
                    "vcgt.f32   q1, q12, q8   \n"
                    "vbif.f32   q9, q8, q1    \n"
                    "vadd.f32   q12, q12, q9                    \n"
                    "vmov.f32   q9, #0.5\n"
                    "vcgt.f32   q2, q13, q8   \n"
                    "vbif.f32   q9, q8, q2   \n"
                    "vadd.f32   q13, q13, q9\n"
                    "vld1.32 {d2-d3}, [%[vmax]] \n"//"vld1.32 {d22-d23}, [%[vmax]] \n"
                    "vcge.f32 q3, q12, q1         \n" /* data >= -127 */
                    "vbif q12, q1, q3    \n" /* choose data */
                    "vcge.f32 q3, q13, q1         \n" /* data >= -127 */
                    "vbif q13, q1, q3    \n" /* choose data */
                    "vcvt.s32.f32  q1, q12    \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q13    \n"  /* fp32 to int32 */
                    "vqmovn.s32 d24, q1       \n"  /* int32 to int16 */
                    "vqmovn.s32 d25, q2       \n"  /* int32 to int16 */
                    "vqmovn.s16 d9, q12      \n"   /* int16 to int8 */
                    "vst1.8    {d9}, [%[dout_ptr2]]!\n"        //???? vst1.8 or vst1.32                    
                    "subs %[cnt], #1                                \n"
                    "bne  2b                                        \n"
                //right
                    "1:                                          \n"
                    "vdup.s8     d2, d0[0]               \n"
                    "vdup.s8     d3, d0[1]               \n"
                    "vdup.s8     d4, d0[2]               \n"
                    "vdup.s8     d5, d0[3]               \n"
                    "vdup.s8     d6, d0[4]               \n"
                    "vdup.s8     d7, d0[5]               \n"
                    "vdup.s8     d8, d0[6]               \n"
                    "vdup.s8     d9, d0[7]               \n"
                    "vdup.s8     d10, d1[0]              \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    \n"
                    "vld1.8 {d28-d29}, [%[mask]]        \n"
                    "vmov.u32 q8, #0                            \n"
                    "vmov.u32 q9, #0                            \n"
                    "vmov.u32 q10, #0                           \n"
                    "vmov.u32 q11, #0                           \n"
                    "vbif.8 d12, d11, d28        \n"
                    "vbif.8 d13, d11, d29        \n"
                    "vld1.8 {d14-d15}, [%[din_ptr1]]    \n"
                     //r0
                    "vmull.s8 q12, d12, d2                 \n" // q12 = d12 * w01
                    "vext.8 d30, d12, d13, #1               \n" //d10 = 12345678
                    "vext.8 d31, d12, d13, #2               \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr2]]    \n"
                    "vbif.8 d14, d11, d28        \n"
                    "vbif.8 d15, d11, d29        \n"
                    "vmlal.s8 q12, d30, d3                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                 \n" // q12 += d11 * w02
                    //r1
                    "vext.8 d30, d14, d15, #1           \n" //d10 = 00123456
                    "vext.8 d31, d14, d15, #2          \n" //d11 = 12345678
                    "vmull.s8 q13, d14, d2                 \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d14, d5                 \n" // q12 = d12 * w11
                    "vld1.8 {d14-d15}, [%[din_ptr3]]    \n"
                    "vbif.8 d12, d11, d28                 \n"
                    "vbif.8 d13, d11, d29                 \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d30, d3                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d6                 \n" // q12 += d10 * w00
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 \n" // q12 += d10 * w00
                    //r2
                    "vext.8 d30, d12, d13, #1               \n" //d10 = 00123456
                    "vext.8 d31, d12, d13, #2               \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d12, d5                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d12, d8                 \n" // q13 = d12 * w01
                    "vbif.8 d14, d11, d28                     \n"
                    "vbif.8 d15, d11, d29                     \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 \n" // q12 += d10 * w00
                    "vld1.8 d12, [%[rs_mask]]!            \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                 \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d14, d15, #1           \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d14, d8                 \n" // q13 = d12 * w01
                    "vmlal.s8 q13, d30, d9                 \n" // q13 += d10 * w00
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                 \n" // q12 += d10 * w00
                    "vdup.32 q3, %[scale]                    \n"
                    "vdup.32 q4, %[bias]                     \n"
                    "vdup.32 q5, %[bias]                     \n"
                    "vmov.f32 q1, #0.0                       \n"
                    "vcvt.f32.s32   q8, q8                      \n"
                    "vcvt.f32.s32   q9, q9                      \n"
                    "vmla.f32 q4, q8, q3                     \n"
                    "vmla.f32 q5, q9, q3                     \n"
                    "vmax.f32 q4, q4, q1              \n"
                    "vmax.f32 q5, q5, q1              \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q13, %[bias]                            \n"
                    "vdup.32 q14, %[bias]                            \n"
                    "vcvt.f32.s32   q10, q10                      \n"
                    "vcvt.f32.s32   q11, q11                      \n"
                    "vmla.f32 q13, q10, q3                     \n"
                    "vmla.f32 q14, q11, q3                     \n"
                    "vmax.f32 q13, q13, q1              \n"
                    "vmax.f32 q14, q14, q1              \n"
                    "vmov.f32 q8, #-0.5\n"
                    "vmov.f32 q9, #0.5\n"
                    "vcgt.f32   q1, q4, q8   \n"
                    "vbif.f32   q9, q8, q1    \n"
                    "vadd.f32   q4, q4, q9                    \n"
                    "vmov.f32   q9, #0.5\n"
                    "vcgt.f32   q2, q5, q8   \n"
                    "vbif.f32   q9, q8, q2   \n"
                    "vadd.f32   q5, q5, q9\n"
                    "vld1.32 {d2-d3}, [%[vmax]] \n"//"vld1.32 {d22-d23}, [%[vmax]] \n"
                    "vcge.f32 q3, q4, q1         \n" /* data >= -127 */
                    "vbif q4, q1, q3    \n" /* choose data */
                    "vcge.f32 q3, q5, q1         \n" /* data >= -127 */
                    "vbif q5, q1, q3    \n" /* choose data */
                    "vcvt.s32.f32  q1, q4    \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q5    \n"  /* fp32 to int32 */
                    "vqmovn.s32 d8, q1       \n"  /* int32 to int16 */
                    "vqmovn.s32 d9, q2       \n"  /* int32 to int16 */
                    "vqmovn.s16 d7, q4      \n"   /* int16 to int8 */
                    "vld1.8 d10, [%[dout_ptr1]]    \n"
                    "vbif.8 d7, d10, d12                   \n"
                    "vst1.8    {d7}, [%[dout_ptr1]]!\n" 
                    "vmov.f32 q8, #-0.5\n"
                    "vmov.f32 q9, #0.5\n"
                    "vcgt.f32   q1, q13, q8   \n"
                    "vbif.f32   q9, q8, q1    \n"
                    "vadd.f32   q13, q13, q9                    \n"
                    "vmov.f32   q9, #0.5\n"
                    "vcgt.f32   q2, q14, q8   \n"
                    "vbif.f32   q9, q8, q2   \n"
                    "vadd.f32   q14, q14, q9\n"
                    "vld1.32 {d2-d3}, [%[vmax]] \n"
                    "vcge.f32 q3, q13, q1         \n" /* data >= -127 */
                    "vbif q13, q1, q3    \n" /* choose data */
                    "vcge.f32 q3, q14, q1         \n" /* data >= -127 */
                    "vbif q14, q1, q3    \n" /* choose data */
                    "vcvt.s32.f32  q1, q13    \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q14    \n"  /* fp32 to int32 */
                    "vqmovn.s32 d26, q1       \n"  /* int32 to int16 */
                    "vqmovn.s32 d27, q2       \n"  /* int32 to int16 */
                    "vqmovn.s16 d10, q13      \n"   /* int16 to int8 */
                    "vld1.8 d14, [%[dout_ptr2]]   \n"
                    "vbif.8 d10, d14, d12                   \n"
                    "vst1.8    {d10}, [%[dout_ptr2]]!\n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2), \
                      [din_ptr3] "+r" (din_ptr3), \
                      [dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [scale] "+r" (scale_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask), [vmax]"r"(vmax_ptr)
                    :"cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                      "q11", "q12", "q13", "q14", "q15"
                );
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
  }
}

void conv_depthwise_3x3s1p0_bias_relu_int8_float(float* dout,
                                                 const int8_t* din,
                                                 const int8_t* weights,
                                                 const float* scale,
                                                 const float* bias,
                                                 bool flag_bias,
                                                 int flag_act,
                                                 float* alpha,
                                                 int num,
                                                 int ch_in,
                                                 int h_in,
                                                 int w_in,
                                                 int h_out,
                                                 int w_out,
                                                 ARMContext* ctx) {
  const unsigned char right_pad_idx[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const unsigned int right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, w_in * sizeof(int8_t));
  float* write_ptr =
      reinterpret_cast<float*>(ctx->workspace_data<int8_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;
  int tile_w = w_out >> 3;
  int cnt_col = tile_w;
  unsigned int size_pad_right = (unsigned int)(w_in - (cnt_col << 3));
  unsigned int rst_remain = (w_out - ((cnt_col) << 3));
  uint32x4_t vmask_result1 =
      vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst));
  uint32x4_t vmask_result2 =
      vcgtq_u32(vdupq_n_u32(rst_remain), vld1q_u32(right_pad_rst + 4));
  uint8x16_t vmask_rp =
      vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
  unsigned char vmask[16];
  vst1q_u8(vmask, vmask_rp);
  unsigned int rmask[8];
  vst1q_u32(rmask, vmask_result1);
  vst1q_u32(rmask + 4, vmask_result2);

  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    float* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
      float vscale[4] = {scale_val, scale_val, scale_val, scale_val};
#ifdef __aarch64__
      int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
      int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
      int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

      int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
      int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
      int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

      int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
      int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
      int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

      float32x4_t v_bias = vdupq_n_f32(vbias[0]);
      float32x4_t v_scale = vdupq_n_f32(vscale[0]);
#endif

      float* doutr0 = nullptr;
      float* doutr1 = nullptr;

      const int8_t* dr0 = din_ch_ptr;
      const int8_t* dr1 = dr0 + w_in;
      const int8_t* dr2 = dr1 + w_in;
      const int8_t* dr3 = dr2 + w_in;

      const int8_t* din_ptr0 = nullptr;
      const int8_t* din_ptr1 = nullptr;
      const int8_t* din_ptr2 = nullptr;
      const int8_t* din_ptr3 = nullptr;

      for (int i = 0; i < h_out; i += 2) {
        //! process top pad pad_h = 1
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;

        unsigned int* rst_mask = rmask;
        unsigned char* val_mask = vmask;

        dr0 = dr2;
        dr1 = dr3;
        dr2 = dr1 + w_in;
        dr3 = dr2 + w_in;
        //! process bottom pad
        if (i + 3 > h_in) {
          switch (i + 3 - h_in) {
            case 3:
              din_ptr1 = zero_ptr;
            case 2:
              din_ptr2 = zero_ptr;
            case 1:
              din_ptr3 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 2 > h_out) {
          doutr1 = write_ptr;
        }
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr3]] \n"
                    "movi   v21.4s, #0x0\n"                                     /* v21 = 0 */
                // middle
                    "ld1    {v0.8b}, [%[din_ptr0]], #8               \n"   /* load a(0,0)-a(0,7) to q0*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8               \n"   /* load a(1,0)-a(1.7) to q2*/
                    "ld1    {v1.8b}, [%[din_ptr0]]                   \n"   /* load a(0,8)-a(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]                   \n"   /* load a(1,8)-a(1,15) to q3*/
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a(2,0)-a(2,7) to q6*/
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a(3,0)-a(3,7) to q8*/
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a(2,8)-a(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a(3,8)-a(3,15) to q9*/
                    "movi   v10.4s, #0x0\n"                 /* init int32 acc v10 to 0 */
                    "movi   v11.4s, #0x0\n"                 /* init int32 acc v11 to 0 */
                    "movi   v12.4s, #0x0\n"                   /* init int32 acc v12 to 0 */
                    "movi   v13.4s, #0x0\n"                   /* init int32 acc v13 to 0 */
                    "1:                            \n"
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b   \n"
                    "ext v4.8b, v0.8b, v1.8b, #1        \n"
                    "ext v5.8b, v0.8b, v1.8b, #2        \n"
                    "smlal  v18.8h,  %[v1].8b,  v4.8b   \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v2].8b,  v5.8b \n"   /* acc_r00 += r(0,12345678) * w02 */
                    "ext v14.8b, v2.8b, v3.8b, #1       \n"
                    "ext v15.8b, v2.8b, v3.8b, #2       \n"
                    //r1
                    "smull  v19.8h,  %[v0].8b,  v2.8b   \n"
                    "smlal  v18.8h,  %[v3].8b,  v2.8b   \n"
                    "ext v4.8b, v6.8b, v7.8b, #1       \n"
                    "ext v5.8b, v6.8b, v7.8b, #2       \n"
                    "smlal  v19.8h,  %[v1].8b,  v14.8b   \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v4].8b,  v14.8b   \n"
                    "ld1    {v0.8b}, [%[din_ptr0]], #8  \n"   /* load a'(0,0)-a'(0,7) to q0 for prefetch*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8  \n"   /* load a'(1,0)-a'(1,7) to q2 for prefetch*/
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v2].8b,  v15.8b   \n"   /* acc_r10 += r(1,12345678) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b   \n"   /* acc_r00 += r(1,12345678) * w12 */
                    //r2
                    "ld1    {v1.8b}, [%[din_ptr0]]      \n"   /* load a'(0,8)-a'(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]      \n"   /* load a'(1,8)-a'(1,15) to q3*/
                    "smlal  v19.8h,  %[v3].8b,  v6.8b   \n"  /* acc_r10 += r(2,01234567) * w11 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b   \n"
                    "ext v14.8b, v8.8b, v9.8b, #1        \n"
                    "ext v15.8b, v8.8b, v9.8b, #2        \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b  \n"
                    "smlal  v18.8h,  %[v7].8b,  v4.8b  \n"
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a'(2,0)-a'(2,7) to q6 for prefetch*/
                    "smlal  v19.8h,  %[v5].8b,  v5.8b  \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b  \n"   /* acc_r00 += r(2, 12345678) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v12 += outr10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v13 += outr10.high*/
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b   \n"   /* acc_r10 += r(3, 01234567) * w21 */
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a'(3,0)-a'(3,7) to q8 for prefetch*/
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a'(2,7)-a'(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a'(3,7)-a'(3,15) to q9*/
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v7].8b,  v14.8b   \n"   /* acc_r10 += r(3, 00123456) * w20) */
                    "dup   v20.4s, %[v_bias].s[0] \n "                /* dup bias to v20 */
                    "dup   v22.4s, %[v_bias].s[0] \n "                /* dup bias to v22 */
                    "scvtf  v10.4s, v10.4s               \n"   /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s               \n"   /* int32-> float32*/
                    "fmla   v20.4s, v10.4s, %[v_scale].4s       \n"  /* mul scale, add bias */
                    "fmla   v22.4s, v11.4s, %[v_scale].4s       \n"  /* mul scale, add bias */
                    "fmax  v20.4s, v20.4s, v21.4s        \n"  /* relu */
                    "fmax  v22.4s, v22.4s, v21.4s        \n"  /* relu */
                    "stp    q20, q22, [%[ptr_out0]], #32 \n"  /* store q10, q11 -> ptr_out 0 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"    /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"    /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b   \n"    /* acc_r10 += r(3, 12345678) * w22 */
                    "movi   v10.4s, #0x0\n"         /* clear 0 for v10 */
                    "movi   v11.4s, #0x0\n"         /* clear 0 for v11 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"    /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"    /* v13 += acc_r10.high*/
                    "scvtf  v12.4s, v12.4s                \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                \n"  /* int32-> float32*/
                    "dup   v14.4s, %[v_bias].s[0] \n "    /* dup bias to v14 */
                    "dup   v15.4s, %[v_bias].s[0] \n "    /* dup bias to v15 */
                    "fmla   v14.4s, v12.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "fmla   v15.4s, v13.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "fmax  v14.4s, v14.4s, v21.4s        \n"   /* relu*/
                    "fmax  v15.4s, v15.4s, v21.4s        \n"   /* relu*/
                    "stp     q14, q15, [%[ptr_out1]], #32   \n" /* store q10, q11 -> ptr_out  1 */
                    "movi   v12.4s, #0x0\n"
                    "movi   v13.4s, #0x0\n"
                    "subs %[cnt], %[cnt], #1                \n"
                    "bne 1b                                 \n"
                //right
                    "3:                                      \n" 
                    "ld1 {v20.8b}, [%[vmask]], #8            \n"
                    "ld1 {v22.8b}, [%[vmask]]                \n"
                    "bif v0.8b, v21.8b, v20.8b               \n"  /* a(0, 0) to a(0, 7) */
                    "bif v1.8b, v21.8b, v22.8b               \n"  /* a(0, 8) to a(0, 15) */
                    "bif v2.8b, v21.8b, v20.8b               \n"  /* a(1, 0) to a(1, 7) */
                    "bif v3.8b, v21.8b, v22.8b               \n"  /* a(1, 8) to a(1, 15) */
                    "ext v4.8b, v0.8b, v1.8b, #1             \n"  /* r(0, 12345678) */
                    "ext v5.8b, v0.8b, v1.8b, #2             \n"  /* r(0, 23456789) */
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b        \n"  /* acc_r00 += r(0, 01234567) * w00 */
                    "ext v14.8b, v2.8b, v3.8b, #1            \n"  /* r(1, 12345678) */
                    "ext v15.8b, v2.8b, v3.8b, #2            \n"  /* r(1, 23456789) */
                    "bif v6.8b, v21.8b, v20.8b               \n"  /* a(2, 0) to a(0, 7) */  /* r(2, 12345678) */
                    "bif v7.8b, v21.8b, v22.8b               \n"  /* a(2, 8) to a(2, 15) */ /* r(2, 23456789) */
                    "smlal  v18.8h,  %[v1].8b,  v4.8b        \n"  /* acc_r00 += r(0, 12345678) * w01 */
                    "bif v8.8b, v21.8b, v20.8b               \n"  /* a(3, 0) to a(3, 7) */ 
                    "bif v9.8b, v21.8b, v22.8b               \n"  /* a(3, 8) to a(3, 15) */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v2].8b,  v5.8b        \n"  /* acc_r00 = r(0, 23456789) * w02 */
                    //r1
                    "ext v4.8b, v6.8b, v7.8b, #1            \n"
                    "ext v5.8b, v6.8b, v7.8b, #2            \n"
                    "smull  v19.8h,  %[v0].8b,  v2.8b        \n"  /* acc_r10 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b        \n"  /* acc_r00 = 01234567 * w00 */
                    "ld1 {v20.4s}, [%[rmask]], #16           \n"
                    "ld1 {v22.4s}, [%[rmask]]                \n"
                    "smlal  v19.8h,  %[v1].8b,  v14.8b       \n"  /* acc_r10 += r(1, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v4].8b,  v14.8b       \n"  /* acc_r00 += r(1, 12345678) * w11 */
                    "ld1 {v0.4s}, [%[ptr_out0]], #16         \n"  /* load original ptr_out0 low */
                    "ld1 {v2.4s}, [%[ptr_out1]], #16         \n"  /* load original ptr_out1 low */
                    "saddw   v12.4s, v12.4s, v19.4h          \n"  /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h         \n"  /* v13 += acc_r10.high */
                    "smull  v19.8h,  %[v2].8b,  v15.8b       \n"  /* acc_r10 = r(1, 23456789) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b       \n"  /* acc_r00 = r(1, 23456789) * w12 */
                    "ld1 {v1.4s}, [%[ptr_out0]]              \n"  /* load original ptr_out0 high */
                    "ld1 {v3.4s}, [%[ptr_out1]]              \n"  /* load original ptr_out1 high */
                    //r2
                    "ext v14.8b, v8.8b, v9.8b, #1             \n"  /* r(3, 12345678) */
                    "ext v15.8b, v8.8b, v9.8b, #2             \n"  /* r(3, 23456789) */
                    "smlal  v19.8h,  %[v3].8b,  v6.8b        \n"  /* acc_r10 += r(2, 01234567) * w10 */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b        \n"  /* acc_r00 = r(2, 01234567) * w20 */
                    "sub %[ptr_out0], %[ptr_out0], #16      \n"
                    "sub %[ptr_out1], %[ptr_out1], #16      \n"
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v13 += outr00.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b      \n"   /* acc_r10 += r(2, 12345678) * w11 */
                    "smlal  v18.8h,  %[v7].8b,  v4.8b      \n"   /* acc_r00 += r(2, 12345678) * w21 */
                    "smlal  v19.8h,  %[v5].8b,  v5.8b      \n"   /* acc_r10 += r(2, 23456789) * w12 */
                    "saddw   v10.4s, v10.4s, v18.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h        \n"   /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b      \n"   /* acc_r00 += r(2, 23456789) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += acc_r00.high*/
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b       \n"   /* acc_r10 += r(3, 01234567) * w20 */
                    "saddw   v10.4s, v10.4s, v18.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h        \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v7].8b,  v14.8b       \n"   /* acc_r10 += r(3, 12345678) * w21 */
                    "scvtf  v10.4s, v10.4s                  \n"  /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s                  \n"  /* int32-> float32*/
                    "dup   v6.4s, %[v_bias].s[0]                 \n "
                    "dup   v7.4s, %[v_bias].s[0]                 \n "
                    "fmla   v6.4s, v10.4s, %[v_scale].4s         \n"
                    "fmla   v7.4s, v11.4s, %[v_scale].4s         \n"
                    "fmax  v6.4s, v6.4s, v21.4s           \n"   /* relu */
                    "fmax  v7.4s, v7.4s, v21.4s           \n"   /* relu */
                    "bif v6.16b, v0.16b, v20.16b           \n"   /* select bit of ptr_out 0 according to rmask */
                    "bif v7.16b, v1.16b, v22.16b           \n"   /* select bit of ptr_out 0 high according to rmask */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += acc_r00.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b       \n"   /* acc_r10 = r(3, 23456789) * w22 */
                    "stp     q6, q7, [%[ptr_out0]], #32   \n"   /* store q10, q11 -> ptr_out 0   */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += outr00.high*/
                    "scvtf  v12.4s, v12.4s                 \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                 \n"  /* int32-> float32*/
                    "dup   v8.4s, %[v_bias].s[0]                 \n "
                    "dup   v9.4s, %[v_bias].s[0]                 \n "
                    "fmla   v8.4s, v12.4s, %[v_scale].4s  \n"
                    "fmla   v9.4s, v13.4s, %[v_scale].4s  \n"
                    "fmax  v8.4s, v8.4s, v21.4s           \n"   /* relu */
                    "fmax  v9.4s, v9.4s, v21.4s           \n"   /* relu */
                    "bif v8.16b, v2.16b, v20.16b           \n"   /* select bit of ptr_out 1 according to rmask */
                    "bif v9.16b, v3.16b, v22.16b           \n"   /* select bit of ptr_out 1 hihg according to rmask */
                    "stp   q8, q9, [%[ptr_out1]], #32   \n"   /* store q12, q13 -> ptr_out 1  */
                    :   [cnt]"+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [din_ptr3] "+r"(din_ptr3), [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
                        [vmask] "+r" (val_mask), [rmask] "+r" (rst_mask)
                    :   [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), \
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22), \
                        [v_bias]"w"(v_bias), [v_scale] "w"(v_scale)
                    :"cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
                      "v10", "v11", "v12","v13","v14","v15", \
                      "v18",\
                      "v19", "v20", "v21", "v22"
                );
#else
                // store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                
                    "pld [%[din_ptr0]]                \n"
                    "pld [%[din_ptr1]]                \n"
                    "pld [%[din_ptr2]]                \n"
                    "pld [%[din_ptr3]]                \n"
                    "vmov.u32 d11, #0                 \n"
                //mid
                    "1:                                          \n"
                    "vdup.s8     d2, d0[0]            \n"
                    "vdup.s8     d3, d0[1]            \n"
                    "vdup.s8     d4, d0[2]            \n"
                    "vdup.s8     d5, d0[3]               \n"
                    "vdup.s8     d6, d0[4]               \n"
                    "vdup.s8     d7, d0[5]               \n"
                    "vdup.s8     d8, d0[6]               \n"
                    "vdup.s8     d9, d0[7]               \n"
                    "vdup.s8     d10, d1[0]              \n"
                    "vmov.u32 q8, #0                  \n"
                    "vmov.u32 q9, #0                  \n"
                    "vmov.u32 q10, #0                 \n"
                    "vmov.u32 q11, #0                 \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    \n"
                     //r0
                    "vmull.s8 q12, d12, d2                 \n" // q12 = d12 * w01
                    "vext.8     d30, d12, d13, #1     \n" //d10 = 12345678
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr1]]    \n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]    \n"
                    "vmlal.s8 q12, d30, d3                 \n" // q12 += d10 * w00
                    "add %[din_ptr0], #8                   \n"
                    "add %[din_ptr1], #8                   \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                 \n" // q12 += d11 * w02
                    //r1
                    "vext.8     d30, d12, d13, #1     \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d2                 \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d12, d5                 \n" // q12 = d12 * w11
                    "vld1.8 {d12-d13}, [%[din_ptr3]]    \n"
                    "vmlal.s8 q13, d30, d3                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d30, d6                 \n" // q12 += d10 * w00
                    "add %[din_ptr2], #8                   \n"
                    "add %[din_ptr3], #8                   \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 \n" // q12 += d10 * w00
                    //r2
                    "vext.8     d30, d14, d15, #1     \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d14, d5                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d8                 \n" // q13 = d12 * w01
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 \n" // q12 += d10 * w00
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d10                 \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d12, d13, #1     \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d12, d8                 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                \n"
                    "pld [%[din_ptr1]]                \n"
                    "vmlal.s8 q13, d30, d9                 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                \n"
                    "pld [%[din_ptr3]]                \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                 \n" // q12 += d10 * w00
                    "vdup.32 q7, %[scale]                            \n"
                    "vdup.32 q14, %[bias]                            \n"
                    "vdup.32 q15, %[bias]                            \n"
                    "vcvt.f32.s32   q8, q8                      \n"
                    "vcvt.f32.s32   q9, q9                      \n"
                    "vmov.f32 q1, #0.0                            \n"
                    "vmla.f32 q14, q8, q7                            \n"
                    "vmla.f32 q15, q9, q7                            \n"
                    "vmax.f32 q14, q14, q1              \n"
                    "vmax.f32 q15, q15, q1              \n"
                    "vst1.32 {d28-d29}, [%[dout_ptr1]]!              \n"
                    "vst1.32 {d30-d31}, [%[dout_ptr1]]!              \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q12, %[bias]                            \n"
                    "vdup.32 q13, %[bias]                            \n"
                    "vcvt.f32.s32   q10, q10                      \n"
                    "vcvt.f32.s32   q11, q11                      \n"
                    "vmla.f32 q12, q10, q7                     \n"
                    "vmla.f32 q13, q11, q7                     \n"
                    "vmax.f32 q12, q12, q1              \n"
                    "vmax.f32 q13, q13, q1              \n"
                    "vst1.32 {d24-d25}, [%[dout_ptr2]]!         \n"
                    "subs %[cnt], %[cnt], #1                                \n"
                    "vst1.32 {d26-d27}, [%[dout_ptr2]]!         \n"
                    "bne  1b                                        \n"
                //right
                    "2:                                          \n"
                    "vdup.s8     d2, d0[0]            \n"
                    "vdup.s8     d3, d0[1]            \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    \n"
                    "vld1.8 {d28-d29}, [%[mask]]        \n"
                    "vmov.u32 q8, #0                            \n"
                    "vmov.u32 q9, #0                            \n"
                    "vmov.u32 q10, #0                           \n"
                    "vmov.u32 q11, #0                           \n"
                    "vbif.8 d12, d11, d28                   \n"
                    "vbif.8 d13, d11, d29                   \n"
                    "vld1.8 {d14-d15}, [%[din_ptr1]]        \n"
                     //r0
                    "vmull.s8 q12, d12, d2                  \n" // q12 = d12 * w01
                    "vext.8 d30, d12, d13, #1               \n" //d10 = 12345678
                    "vext.8 d31, d12, d13, #2               \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr2]]        \n"
                    "vbif.8 d14, d11, d28                   \n"
                    "vbif.8 d15, d11, d29                   \n"
                    "vmlal.s8 q12, d30, d3                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                 \n" // q12 += d11 * w02
                    //r1
                    "vext.8 d30, d14, d15, #1              \n" //d10 = 00123456
                    "vext.8 d31, d14, d15, #2              \n" //d11 = 12345678
                    "vmull.s8 q13, d14, d2                 \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d14, d5                 \n" // q12 = d12 * w11
                    "vld1.8 {d14-d15}, [%[din_ptr3]]       \n"
                    "vbif.8 d12, d11, d28                  \n"
                    "vbif.8 d13, d11, d29                  \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d30, d3                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d6                 \n" // q12 += d10 * w00
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 \n" // q12 += d10 * w00
                    //r2
                    "vext.8 d30, d12, d13, #1               \n" //d10 = 00123456
                    "vext.8 d31, d12, d13, #2               \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d12, d5                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d12, d8                 \n" // q13 = d12 * w01
                    "vbif.8 d14, d11, d28                     \n"
                    "vbif.8 d15, d11, d29                     \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 \n" // q12 += d10 * w00
                    "vld1.32 {d28-d29}, [%[dout_ptr1]]!    \n"
                    "vld1.32 {d12-d13}, [%[dout_ptr1]]    \n"
                    "vld1.32 {d2-d3}, [%[rs_mask]]!     \n"
                    "vld1.32 {d4-d5}, [%[rs_mask]]    \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                 \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d14, d15, #1     \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d14, d8                 \n" // q13 = d12 * w01
                    "vld1.32 {d14-d15}, [%[dout_ptr2]]!    \n"
                    "vld1.32 {d24-d25}, [%[dout_ptr2]]     \n"
                    "vmlal.s8 q13, d30, d9                 \n" // q13 += d10 * w00
                    "sub %[dout_ptr1], #16                  \n"
                    "sub %[dout_ptr2], #16                  \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                 \n" // q12 += d10 * w00
                    "vdup.32 q3, %[scale]                    \n"
                    "vdup.32 q4, %[bias]                     \n"
                    "vdup.32 q5, %[bias]                     \n"
                    "vmov.f32   q15, #0.0\n"
                    "vcvt.f32.s32   q8, q8                      \n"
                    "vcvt.f32.s32   q9, q9                      \n"
                    "vmla.f32 q4, q8, q3                     \n"
                    "vmla.f32 q5, q9, q3                     \n"
                    "vmax.f32 q4, q4, q15              \n"
                    "vmax.f32 q5, q5, q15              \n"
                    "vbif q4, q14, q1                   \n"
                    "vbif q5, q6, q2                    \n"
                    "vst1.32 {d8-d9}, [%[dout_ptr1]]!         \n"
                    "vst1.32 {d10-d11}, [%[dout_ptr1]]!         \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q13, %[bias]                            \n"
                    "vdup.32 q14, %[bias]                            \n"
                    "vcvt.f32.s32   q10, q10                      \n"
                    "vcvt.f32.s32   q11, q11                      \n"
                    "vmla.f32 q13, q10, q3                     \n"
                    "vmla.f32 q14, q11, q3                     \n"
                    "vmax.f32 q13, q13, q15              \n"
                    "vmax.f32 q14, q14, q15              \n"
                    "vbif q13, q7, q1        \n"
                    "vbif q14, q12, q2       \n"
                    "vst1.32 {d26-d27}, [%[dout_ptr2]]!         \n"
                    "vst1.32 {d28-d29}, [%[dout_ptr2]]!         \n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2), \
                      [din_ptr3] "+r" (din_ptr3), \
                      [dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [scale] "+r" (scale_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask)
                    :"cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                      "q11", "q12", "q13", "q14", "q15"
                );
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
  }
}

void conv_depthwise_3x3s1p0_bias_relu_int8_int8(int8_t* dout,
                                                const int8_t* din,
                                                const int8_t* weights,
                                                const float* scale,
                                                const float* bias,
                                                bool flag_bias,
                                                int flag_act,
                                                float* alpha,
                                                int num,
                                                int ch_in,
                                                int h_in,
                                                int w_in,
                                                int h_out,
                                                int w_out,
                                                ARMContext* ctx) {
  const unsigned char right_pad_idx[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const unsigned char right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  int8_t* zero_ptr = ctx->workspace_data<int8_t>();
  memset(zero_ptr, 0, w_in * sizeof(int8_t));
  int8_t* write_ptr =
      reinterpret_cast<int8_t*>(ctx->workspace_data<int8_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;
  int tile_w = w_out >> 3;
  int cnt_col = tile_w;
  unsigned int size_pad_right = (unsigned int)(w_in - (cnt_col << 3));
  uint8x8_t vmask_rp1 =
      vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx));
  uint8x8_t vmask_rp2 =
      vcgt_u8(vdup_n_u8(size_pad_right), vld1_u8(right_pad_idx + 8));
  unsigned char rst_remain = (w_out - ((cnt_col) << 3));
  uint8x8_t vmask_result =
      vcgt_u8(vdup_n_u8(rst_remain), vld1_u8(right_pad_rst));
  uint8x16_t vmask_rp =
      vcgtq_u8(vdupq_n_u8(size_pad_right), vld1q_u8(right_pad_idx));
  unsigned char vmask[16];
  vst1q_u8(vmask, vmask_rp);
  unsigned char rmask[8];
  vst1_u8(rmask, vmask_result);
  for (int n = 0; n < num; ++n) {
    const int8_t* din_batch = din + n * ch_in * size_in_channel;
    int8_t* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      int8_t* dout_ptr = dout_batch + c * size_out_channel;
      const int8_t* din_ch_ptr = din_batch + c * size_in_channel;
      float bias_val = flag_bias ? bias[c] : 0;
      float scale_val = scale[c];
      const int8_t* wei_ptr = weights + c * w_stride;
      float vbias[4] = {bias_val, bias_val, bias_val, bias_val};
      float vscale[4] = {scale_val, scale_val, scale_val, scale_val};
      float vmax[4] = {-127.f, -127.f, -127.f, -127.f};
      float* vmax_ptr = vmax;
#ifdef __aarch64__
      int8x8_t wr00 = vdup_n_s8(wei_ptr[0]);
      int8x8_t wr10 = vdup_n_s8(wei_ptr[3]);
      int8x8_t wr20 = vdup_n_s8(wei_ptr[6]);

      int8x8_t wr01 = vdup_n_s8(wei_ptr[1]);
      int8x8_t wr11 = vdup_n_s8(wei_ptr[4]);
      int8x8_t wr21 = vdup_n_s8(wei_ptr[7]);

      int8x8_t wr02 = vdup_n_s8(wei_ptr[2]);
      int8x8_t wr12 = vdup_n_s8(wei_ptr[5]);
      int8x8_t wr22 = vdup_n_s8(wei_ptr[8]);

      float32x4_t v_bias = vdupq_n_f32(vbias[0]);
      float32x4_t v_scale = vdupq_n_f32(vscale[0]);
#endif

      int8_t* doutr0 = nullptr;
      int8_t* doutr1 = nullptr;

      const int8_t* dr0 = din_ch_ptr;
      const int8_t* dr1 = dr0 + w_in;
      const int8_t* dr2 = dr1 + w_in;
      const int8_t* dr3 = dr2 + w_in;

      const int8_t* din_ptr0 = nullptr;
      const int8_t* din_ptr1 = nullptr;
      const int8_t* din_ptr2 = nullptr;
      const int8_t* din_ptr3 = nullptr;

      for (int i = 0; i < h_out; i += 2) {
        //! process top pad pad_h = 1
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;
        unsigned char* rst_mask = rmask;
        unsigned char* val_mask = vmask;

        dr0 = dr2;
        dr1 = dr3;
        dr2 = dr1 + w_in;
        dr3 = dr2 + w_in;

        //! process bottom pad
        if (i + 3 > h_in) {
          switch (i + 3 - h_in) {
            case 3:
              din_ptr1 = zero_ptr;
            case 2:
              din_ptr2 = zero_ptr;
            case 1:
              din_ptr3 = zero_ptr;
            default:
              break;
          }
        }
        //! process bottom remain
        if (i + 2 > h_out) {
          doutr1 = write_ptr;
        }
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
                asm volatile(
                    "PRFM PLDL1KEEP, [%[din_ptr0]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr1]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr2]] \n"
                    "PRFM PLDL1KEEP, [%[din_ptr3]] \n"
                    "movi   v21.4s, #0x0\n"                                     /* v21 = 0 */
                // middle
                    "ld1    {v0.8b}, [%[din_ptr0]], #8               \n"   /* load a(0,0)-a(0,7) to q0*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8               \n"   /* load a(1,0)-a(1.7) to q2*/
                    "ld1    {v1.8b}, [%[din_ptr0]]                   \n"   /* load a(0,8)-a(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]                   \n"   /* load a(1,8)-a(1,15) to q3*/
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a(2,0)-a(2,7) to q6*/
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a(3,0)-a(3,7) to q8*/
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a(2,8)-a(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a(3,8)-a(3,15) to q9*/
                    "movi   v10.4s, #0x0\n"                 /* init int32 acc v10 to 0 */
                    "movi   v11.4s, #0x0\n"                 /* init int32 acc v11 to 0 */
                    "movi   v12.4s, #0x0\n"                   /* init int32 acc v12 to 0 */
                    "movi   v13.4s, #0x0\n"                   /* init int32 acc v13 to 0 */
                    "1:                            \n"
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b   \n"
                    "ext v4.8b, v0.8b, v1.8b, #1        \n"
                    "ext v5.8b, v0.8b, v1.8b, #2        \n"
                    "smlal  v18.8h,  %[v1].8b,  v4.8b   \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v2].8b,  v5.8b \n"   /* acc_r00 += r(0,12345678) * w02 */
                    "ext v14.8b, v2.8b, v3.8b, #1       \n"
                    "ext v15.8b, v2.8b, v3.8b, #2       \n"
                    //r1
                    "smull  v19.8h,  %[v0].8b,  v2.8b   \n"
                    "smlal  v18.8h,  %[v3].8b,  v2.8b   \n"
                    "ext v4.8b, v6.8b, v7.8b, #1       \n"
                    "ext v5.8b, v6.8b, v7.8b, #2       \n"
                    "smlal  v19.8h,  %[v1].8b,  v14.8b   \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low  */
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high */
                    "smull  v18.8h,  %[v4].8b,  v14.8b   \n"
                    "ld1    {v0.8b}, [%[din_ptr0]], #8  \n"   /* load a'(0,0)-a'(0,7) to q0 for prefetch*/
                    "ld1    {v2.8b}, [%[din_ptr1]], #8  \n"   /* load a'(1,0)-a'(1,7) to q2 for prefetch*/
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v2].8b,  v15.8b   \n"   /* acc_r10 += r(1,12345678) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b   \n"   /* acc_r00 += r(1,12345678) * w12 */
                    //r2
                    "ld1    {v1.8b}, [%[din_ptr0]]      \n"   /* load a'(0,8)-a'(0,15) to q1*/
                    "ld1    {v3.8b}, [%[din_ptr1]]      \n"   /* load a'(1,8)-a'(1,15) to q3*/
                    "smlal  v19.8h,  %[v3].8b,  v6.8b   \n"  /* acc_r10 += r(2,01234567) * w11 */
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b   \n"
                    "ext v14.8b, v8.8b, v9.8b, #1        \n"
                    "ext v15.8b, v8.8b, v9.8b, #2        \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v11 += outr00.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b  \n"
                    "smlal  v18.8h,  %[v7].8b,  v4.8b  \n"
                    "ld1    {v6.8b}, [%[din_ptr2]], #8  \n"   /* load a'(2,0)-a'(2,7) to q6 for prefetch*/
                    "smlal  v19.8h,  %[v5].8b,  v5.8b  \n"
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b  \n"   /* acc_r00 += r(2, 12345678) * w22 */
                    "ld1 {v4.4s}, [%[vmax_ptr]]        \n"
                    "saddw   v12.4s, v12.4s, v19.4h     \n"   /* v12 += outr10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"   /* v13 += outr10.high*/
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b   \n"   /* acc_r10 += r(3, 01234567) * w21 */
                    "ld1    {v8.8b}, [%[din_ptr3]], #8  \n"   /* load a'(3,0)-a'(3,7) to q8 for prefetch*/
                    "ld1    {v7.8b}, [%[din_ptr2]]      \n"   /* load a'(2,7)-a'(2,15) to q7*/
                    "ld1    {v9.8b}, [%[din_ptr3]]      \n"   /* load a'(3,7)-a'(3,15) to q9*/
                    "saddw   v10.4s, v10.4s, v18.4h     \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h    \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v7].8b,  v14.8b   \n"   /* acc_r10 += r(3, 00123456) * w20) */
                    "dup   v20.4s, %[v_bias].s[0] \n "                /* dup bias to v20 */
                    "dup   v22.4s, %[v_bias].s[0] \n "                /* dup bias to v22 */
                    "scvtf  v10.4s, v10.4s               \n"   /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s               \n"   /* int32-> float32*/
                    "fmla   v20.4s, v10.4s, %[v_scale].4s       \n"  /* mul scale, add bias */
                    "fmla   v22.4s, v11.4s, %[v_scale].4s       \n"  /* mul scale, add bias */
                    "fmax  v20.4s, v20.4s, v21.4s        \n"  /* relu */
                    "fmax  v22.4s, v22.4s, v21.4s        \n"  /* relu */
                    "fcmge v11.4s, v20.4s, v4.4s             \n"
                    "bif v20.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v22.4s, v4.4s             \n"
                    "bif v22.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v20.4s, v20.4s                  \n" /* fp32 - int32 */
                    "fcvtas  v22.4s, v22.4s                  \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v20.4s                  \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v22.4s                 \n" /* int32-int16 */
                    "sqxtn   v20.8b, v21.8h                  \n" /* int16-int8 */
                    "st1    {v20.8b}, [%[ptr_out0]], #8     \n"  /* store q10, q11 -> ptr_out 0 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"    /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"    /* v13 += acc_r10.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b   \n"    /* acc_r10 += r(3, 12345678) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h     \n"    /* v12 += acc_r10.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h    \n"    /* v13 += acc_r10.high*/
                    "movi   v21.4s, #0x0\n"                                     /* v21 = 0 */
                    "scvtf  v12.4s, v12.4s                \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                \n"  /* int32-> float32*/
                    "dup   v14.4s, %[v_bias].s[0] \n "    /* dup bias to v14 */
                    "dup   v15.4s, %[v_bias].s[0] \n "    /* dup bias to v15 */
                    "fmla   v14.4s, v12.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "fmla   v15.4s, v13.4s, %[v_scale].4s \n"  /* mul scale, add bias */
                    "fmax  v14.4s, v14.4s, v21.4s        \n"   /* relu*/
                    "fmax  v15.4s, v15.4s, v21.4s        \n"   /* relu*/
                    "fcmge v11.4s, v14.4s, v4.4s             \n"
                    "bif v14.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v15.4s, v4.4s             \n"
                    "bif v15.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v14.4s, v14.4s                   \n" /* fp32 - int32 */
                    "fcvtas  v15.4s, v15.4s                   \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v14.4s                   \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v15.4s                   \n" /* int32-int16 */
                    "sqxtn   v14.8b, v21.8h                   \n" /* int16-int8 */
                    "st1    {v14.8b}, [%[ptr_out1]], #8     \n"  /* store q10, q11 -> ptr_out 0 */
                    "movi   v10.4s, #0x0\n"         /* clear 0 for v10 */
                    "movi   v11.4s, #0x0\n"         /* clear 0 for v11 */
                    "movi   v12.4s, #0x0\n"
                    "movi   v13.4s, #0x0\n"
                    "movi   v21.4s, #0x0\n"
                    "subs %[cnt], %[cnt], #1                \n"
                    "bne 1b                                 \n"                
                //right
                    "3:                                      \n" 
                    "ld1 {v20.8b}, [%[vmask]], #8            \n"
                    "ld1 {v22.8b}, [%[vmask]]                \n"
                    "bif v0.8b, v21.8b, v20.8b               \n"  /* a(0, 0) to a(0, 7) */
                    "bif v1.8b, v21.8b, v22.8b               \n"  /* a(0, 8) to a(0, 15) */
                    "bif v2.8b, v21.8b, v20.8b               \n"  /* a(1, 0) to a(1, 7) */
                    "bif v3.8b, v21.8b, v22.8b               \n"  /* a(1, 8) to a(1, 15) */
                    "ext v4.8b, v0.8b, v1.8b, #1             \n"  /* r(0, 12345678) */
                    "ext v5.8b, v0.8b, v1.8b, #2             \n"  /* r(0, 23456789) */
                    //r0
                    "smull  v18.8h,  %[v0].8b,  v0.8b        \n"  /* acc_r00 += r(0, 01234567) * w00 */
                    "ext v14.8b, v2.8b, v3.8b, #1            \n"  /* r(1, 12345678) */
                    "ext v15.8b, v2.8b, v3.8b, #2            \n"  /* r(1, 23456789) */
                    "bif v6.8b, v21.8b, v20.8b               \n"  /* a(2, 0) to a(0, 7) */  /* r(2, 12345678) */
                    "bif v7.8b, v21.8b, v22.8b               \n"  /* a(2, 8) to a(2, 15) */ /* r(2, 23456789) */
                    "smlal  v18.8h,  %[v1].8b,  v4.8b        \n"  /* acc_r00 += r(0, 12345678) * w01 */
                    "bif v8.8b, v21.8b, v20.8b               \n"  /* a(3, 0) to a(3, 7) */ 
                    "bif v9.8b, v21.8b, v22.8b               \n"  /* a(3, 8) to a(3, 15) */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v2].8b,  v5.8b        \n"  /* acc_r00 = r(0, 23456789) * w02 */
                    //r1
                    "ext v4.8b, v6.8b, v7.8b, #1            \n"
                    "ext v5.8b, v6.8b, v7.8b, #2            \n"
                    "smull  v19.8h,  %[v0].8b,  v2.8b        \n"  /* acc_r10 = 01234567 * w00 */
                    "smlal  v18.8h,  %[v3].8b,  v2.8b        \n"  /* acc_r00 = 01234567 * w00 */
                    "ld1 {v20.8b}, [%[rmask]], #8               \n"
                    "smlal  v19.8h,  %[v1].8b,  v14.8b       \n"  /* acc_r10 += r(1, 12345678) * w01 */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v4].8b,  v14.8b       \n"  /* acc_r00 += r(1, 12345678) * w11 */
                    "ld1 {v0.8b}, [%[ptr_out0]]         \n"  /* load original ptr_out0 low */
                    "ld1 {v2.8b}, [%[ptr_out1]]         \n"  /* load original ptr_out1 low */
                    "saddw   v12.4s, v12.4s, v19.4h          \n"  /* v12 += acc_r10.low */
                    "saddw2   v13.4s, v13.4s, v19.8h         \n"  /* v13 += acc_r10.high */
                    "smull  v19.8h,  %[v2].8b,  v15.8b       \n"  /* acc_r10 = r(1, 23456789) * w02 */
                    "smlal  v18.8h,  %[v5].8b,  v15.8b       \n"  /* acc_r00 = r(1, 23456789) * w12 */
                    //r2
                    "ext v14.8b, v8.8b, v9.8b, #1             \n"  /* r(3, 12345678) */
                    "ext v15.8b, v8.8b, v9.8b, #2             \n"  /* r(3, 23456789) */
                    "smlal  v19.8h,  %[v3].8b,  v6.8b        \n"  /* acc_r10 += r(2, 01234567) * w10 */
                    "saddw   v10.4s, v10.4s, v18.4h          \n"  /* v10 += outr00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h         \n"  /* v11 += outr00.high*/
                    "smull  v18.8h,  %[v6].8b,  v6.8b        \n"  /* acc_r00 = r(2, 01234567) * w20 */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v13 += outr00.high*/
                    "smull  v19.8h,  %[v4].8b,  v4.8b      \n"   /* acc_r10 += r(2, 12345678) * w11 */
                    "smlal  v18.8h,  %[v7].8b,  v4.8b      \n"   /* acc_r00 += r(2, 12345678) * w21 */
                    "smlal  v19.8h,  %[v5].8b,  v5.8b      \n"   /* acc_r10 += r(2, 23456789) * w12 */
                    "saddw   v10.4s, v10.4s, v18.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h        \n"   /* v11 += acc_r00.high*/
                    "smull  v18.8h,  %[v8].8b,  v5.8b      \n"   /* acc_r00 += r(2, 23456789) * w22 */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += acc_r00.high*/
                    "ld1 {v4.4s}, [%[vmax_ptr]]                \n"
                    //r3
                    "smull  v19.8h,  %[v6].8b,  v8.8b       \n"   /* acc_r10 += r(3, 01234567) * w20 */
                    "saddw   v10.4s, v10.4s, v18.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v11.4s, v11.4s, v18.8h        \n"   /* v11 += acc_r00.high*/
                    "smlal  v19.8h,  %[v7].8b,  v14.8b       \n"   /* acc_r10 += r(3, 12345678) * w21 */
                    "scvtf  v10.4s, v10.4s                  \n"  /* int32-> float32*/
                    "scvtf  v11.4s, v11.4s                  \n"  /* int32-> float32*/
                    "dup   v6.4s, %[v_bias].s[0]                 \n "
                    "dup   v7.4s, %[v_bias].s[0]                 \n "
                    "fmla   v6.4s, v10.4s, %[v_scale].4s         \n"
                    "fmla   v7.4s, v11.4s, %[v_scale].4s         \n"
                    "fmax  v6.4s, v6.4s, v21.4s           \n"   /* relu */
                    "fmax  v7.4s, v7.4s, v21.4s           \n"   /* relu */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v10 += acc_r00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += acc_r00.high*/
                    "smull  v19.8h,  %[v8].8b,  v15.8b       \n"   /* acc_r10 = r(3, 23456789) * w22 */
                    "fcmge v11.4s, v6.4s, v4.4s             \n"
                    "bif v6.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v7.4s, v4.4s             \n"
                    "bif v7.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v6.4s, v6.4s                   \n" /* fp32 - int32 */
                    "fcvtas  v7.4s, v7.4s                   \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v6.4s                   \n" /* int32-int16 */
                    "sqxtn2   v21.8h, v7.4s                   \n" /* int32-int16 */
                    "sqxtn   v6.8b, v21.8h                   \n" /* int16-int8 */
                    "bif v6.8b, v0.8b, v20.8b           \n"
                    "st1    {v6.8b}, [%[ptr_out0]], #8     \n"  /* store q10, q11 -> ptr_out 0 */
                    "saddw   v12.4s, v12.4s, v19.4h         \n"   /* v12 += outr00.low*/
                    "saddw2   v13.4s, v13.4s, v19.8h        \n"   /* v11 += outr00.high*/
                    "movi   v21.4s, #0x0\n"                                     /* v21 = 0 */
                    "scvtf  v12.4s, v12.4s                 \n"  /* int32-> float32*/
                    "scvtf  v13.4s, v13.4s                 \n"  /* int32-> float32*/
                    "dup   v8.4s, %[v_bias].s[0]                 \n "
                    "dup   v9.4s, %[v_bias].s[0]                 \n "
                    "fmla   v8.4s, v12.4s, %[v_scale].4s  \n"
                    "fmla   v9.4s, v13.4s, %[v_scale].4s  \n"
                    "fmax  v8.4s, v8.4s, v21.4s           \n"   /* relu */
                    "fmax  v9.4s, v9.4s, v21.4s           \n"   /* relu */
                    "fcmge v11.4s, v8.4s, v4.4s             \n"
                    "bif v8.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcmge v11.4s, v9.4s, v4.4s             \n"
                    "bif v9.16b, v4.16b, v11.16b            \n" /* choose data */
                    "fcvtas  v8.4s, v8.4s                   \n" /* fp32 - int32 */
                    "fcvtas  v9.4s, v9.4s                   \n" /* fp32 - int32 */
                    "sqxtn   v21.4h, v8.4s                   \n" /* int32-int16 */
                    "sqxtn2  v21.8h, v9.4s                   \n" /* int32-int16 */
                    "sqxtn   v8.8b, v21.8h                   \n" /* int16-int8 */
                    "bif v8.8b, v2.8b, v20.8b           \n"
                    "st1    {v8.8b}, [%[ptr_out1]], #8     \n"  /* store q10, q11 -> ptr_out 0 */
                    :   [cnt]"+r"(cnt), [din_ptr0] "+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
                        [din_ptr3] "+r"(din_ptr3), [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
                        [vmask] "+r" (val_mask), [rmask] "+r" (rst_mask)
                    :   [v0]"w"(wr00), [v1]"w"(wr01), [v2]"w"(wr02), [v3]"w"(wr10), \
                        [vmax_ptr]"r"(vmax_ptr), \
                        [v4]"w"(wr11), [v5]"w"(wr12), [v6]"w"(wr20), [v7]"w"(wr21), [v8] "w" (wr22), \
                        [v_bias]"w"(v_bias), [v_scale] "w"(v_scale)
                    :"cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",\
                      "v10", "v11", "v12","v13","v14","v15", \
                      "v18",\
                      "v19", "v20", "v21", "v22"
                );
#else
                // store weights
                asm volatile(
                    "vld1.8    {d0-d1}, [%[wei_ptr]]    \n"
                    :
                    :[wei_ptr] "r" (wei_ptr)
                    : "memory"
                );
                asm volatile(
                    "pld [%[din_ptr0]]                \n"
                    "pld [%[din_ptr1]]                \n"
                    "pld [%[din_ptr2]]                \n"
                    "pld [%[din_ptr3]]                \n"
                    "vmov.u32 d11, #0                 \n"
                //mid
                    "1:                                          \n"
                    "vdup.s8     d2, d0[0]               \n"
                    "vdup.s8     d3, d0[1]               \n"
                    "vdup.s8     d4, d0[2]               \n"
                    "vdup.s8     d5, d0[3]               \n"
                    "vdup.s8     d6, d0[4]               \n"
                    "vdup.s8     d7, d0[5]               \n"
                    "vdup.s8     d8, d0[6]               \n"
                    "vdup.s8     d9, d0[7]               \n"
                    "vdup.s8     d10, d1[0]              \n"
                    "vmov.u32 q8, #0                  \n"
                    "vmov.u32 q9, #0                  \n"
                    "vmov.u32 q10, #0                 \n"
                    "vmov.u32 q11, #0                 \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    \n"
                     //r0
                    "vmull.s8 q12, d12, d2                 \n" // q12 = d12 * w01
                    "vext.8     d30, d12, d13, #1     \n" //d10 = 12345678
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr1]]    \n"
                    "vld1.8 {d14-d15}, [%[din_ptr2]]    \n"
                    "vmlal.s8 q12, d30, d3                 \n" // q12 += d10 * w00
                    "add %[din_ptr0], #8                   \n"
                    "add %[din_ptr1], #8                   \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                 \n" // q12 += d11 * w02
                    //r1
                    "vext.8     d30, d12, d13, #1     \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 12345678
                    "vmull.s8 q13, d12, d2                 \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d12, d5                 \n" // q12 = d12 * w11
                    "vld1.8 {d12-d13}, [%[din_ptr3]]    \n"
                    "vmlal.s8 q13, d30, d3                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d30, d6                 \n" // q12 += d10 * w00
                    "add %[din_ptr2], #8                   \n"
                    "add %[din_ptr3], #8                   \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 \n" // q12 += d10 * w00
                    //r2
                    "vext.8     d30, d14, d15, #1          \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d14, d5                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d14, d8                 \n" // q13 = d12 * w01
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 \n" // q12 += d10 * w00
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d10                 \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d12, d13, #1          \n" //d10 = 00123456
                    "vext.8     d31, d12, d13, #2          \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d12, d8                 \n" // q13 = d12 * w01
                    "pld [%[din_ptr0]]                \n"
                    "pld [%[din_ptr1]]                \n"
                    "vmlal.s8 q13, d30, d9                 \n" // q13 += d10 * w00
                    "pld [%[din_ptr2]]                \n"
                    "pld [%[din_ptr3]]                \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                 \n" // q12 += d10 * w00
                    "vdup.32 q7, %[scale]                            \n"
                    "vdup.32 q14, %[bias]                            \n"
                    "vdup.32 q15, %[bias]                            \n"
                    "vmov.f32  q1, #0.0\n"
                    "vcvt.f32.s32   q8, q8                      \n"
                    "vcvt.f32.s32   q9, q9                      \n"
                    "vmla.f32 q14, q8, q7                            \n"
                    "vmla.f32 q15, q9, q7                            \n"
                    "vmax.f32 q14, q14, q1              \n"
                    "vmax.f32 q15, q15, q1              \n"
                    "vmov.f32 q8, #-0.5\n"
                    "vmov.f32 q9, #0.5\n"
                    "vcgt.f32   q1, q14, q8   \n"
                    "vbif.f32   q9, q8, q1   \n"
                    "vadd.f32   q14, q14, q9\n"
                    "vmov.f32   q9, #0.5\n"
                    "vcgt.f32   q2, q15, q8   \n"
                    "vbif.f32   q9, q8, q2   \n"
                    "vadd.f32   q15, q15, q9\n"
                    "vld1.32 {d2-d3}, [%[vmax]] \n"
                    "vcge.f32 q3, q14, q1         \n" /* data >= -127 */
                    "vbif q14, q1, q3    \n" /* choose data */
                    "vcge.f32 q3, q15, q1         \n" /* data >= -127 */
                    "vbif q15, q1, q3    \n" /* choose data */
                    "vcvt.s32.f32  q1, q14    \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q15    \n"  /* fp32 to int32 */
                    "vqmovn.s32 d28, q1       \n"  /* int32 to int16 */
                    "vqmovn.s32 d29, q2       \n"  /* int32 to int16 */
                    "vqmovn.s16 d10, q14      \n"   /* int16 to int8 */
                    "vst1.8    {d10}, [%[dout_ptr1]]!\n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q12, %[bias]                            \n"
                    "vdup.32 q13, %[bias]                            \n"
                    "vmov.f32  q1, #0.0\n"
                    "vcvt.f32.s32   q10, q10                      \n"
                    "vcvt.f32.s32   q11, q11                      \n"
                    "vmla.f32 q12, q10, q7                     \n"
                    "vmla.f32 q13, q11, q7                     \n"
                    "vmax.f32 q12, q12, q1              \n"
                    "vmax.f32 q13, q13, q1              \n"
                    "vmov.f32 q8, #-0.5\n"
                    "vmov.f32 q9, #0.5\n"
                    "vcgt.f32   q1, q12, q8   \n"
                    "vbif.f32   q9, q8, q1    \n"
                    "vadd.f32   q12, q12, q9                    \n"
                    "vmov.f32   q9, #0.5\n"
                    "vcgt.f32   q2, q13, q8   \n"
                    "vbif.f32   q9, q8, q2   \n"
                    "vadd.f32   q13, q13, q9\n"
                    "vld1.32 {d2-d3}, [%[vmax]]   \n"
                    "vcge.f32 q3, q12, q1         \n" /* data >= -127 */
                    "vbif q12, q1, q3    \n" /* choose data */
                    "vcge.f32 q3, q13, q1         \n" /* data >= -127 */
                    "vbif q13, q1, q3    \n" /* choose data */
                    "vcvt.s32.f32  q1, q12    \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q13    \n"  /* fp32 to int32 */
                    "vqmovn.s32 d24, q1       \n"  /* int32 to int16 */
                    "vqmovn.s32 d25, q2       \n"  /* int32 to int16 */
                    "vqmovn.s16 d9, q12       \n"   /* int16 to int8 */
                    "vst1.8    {d9}, [%[dout_ptr2]]! \n"
                    "subs %[cnt], %[cnt], #1                                \n"
                    "bne  1b                                        \n"
                //right
                    "2:                                          \n"
                    "vdup.s8     d2, d0[0]            \n"
                    "vdup.s8     d3, d0[1]            \n"
                    "vdup.s8     d4, d0[2]            \n"
                    "vdup.s8     d5, d0[3]               \n"
                    "vdup.s8     d6, d0[4]               \n"
                    "vdup.s8     d7, d0[5]               \n"
                    "vdup.s8     d8, d0[6]               \n"
                    "vdup.s8     d9, d0[7]               \n"
                    "vdup.s8     d10, d1[0]              \n"
                    "vmov.u32 d11, #0                 \n"
                    "vld1.8 {d12-d13}, [%[din_ptr0]]    \n"
                    "vld1.8 {d28-d29}, [%[mask]]        \n"
                    "vmov.u32 q8, #0                            \n"
                    "vmov.u32 q9, #0                            \n"
                    "vmov.u32 q10, #0                           \n"
                    "vmov.u32 q11, #0                           \n"
                    "vbif.8 d12, d11, d28        \n"
                    "vbif.8 d13, d11, d29        \n"
                    "vld1.8 {d14-d15}, [%[din_ptr1]]    \n"
                     //r0
                    "vmull.s8 q12, d12, d2                 \n" // q12 = d12 * w01
                    "vext.8 d30, d12, d13, #1               \n" //d10 = 12345678
                    "vext.8 d31, d12, d13, #2               \n" //d11 = 23456789
                    "vld1.8 {d12-d13}, [%[din_ptr2]]    \n"
                    "vbif.8 d14, d11, d28        \n"
                    "vbif.8 d15, d11, d29        \n"
                    "vmlal.s8 q12, d30, d3                 \n" // q12 += d10 * w00
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q12, d31, d4                 \n" // q12 += d11 * w02
                    //r1
                    "vext.8 d30, d14, d15, #1           \n" //d10 = 00123456
                    "vext.8 d31, d14, d15, #2          \n" //d11 = 12345678
                    "vmull.s8 q13, d14, d2                 \n" // q13 = d12 * w01
                    "vmlal.s8 q12, d14, d5                 \n" // q12 = d12 * w11
                    "vld1.8 {d14-d15}, [%[din_ptr3]]    \n"
                    "vbif.8 d12, d11, d28                 \n"
                    "vbif.8 d13, d11, d29                 \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d30, d3                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d30, d6                 \n" // q12 += d10 * w00
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d4                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d31, d7                 \n" // q12 += d10 * w00
                    //r2
                    "vext.8 d30, d12, d13, #1               \n" //d10 = 00123456
                    "vext.8 d31, d12, d13, #2               \n" //d11 = 12345678
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d12, d5                 \n" // q13 = d12 * w01
                    "vmull.s8 q12, d12, d8                 \n" // q13 = d12 * w01
                    "vbif.8 d14, d11, d28                     \n"
                    "vbif.8 d15, d11, d29                     \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d30, d6                 \n" // q12 += d10 * w00
                    "vmlal.s8 q12, d30, d9                 \n" // q12 += d10 * w00
                    "vld1.8 d12, [%[rs_mask]]!            \n"
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmlal.s8 q13, d31, d7                 \n" // q12 += d10 * w00
                    "vmull.s8 q12, d31, d10                 \n" // q12 += d10 * w00
                    //r3
                    "vext.8     d30, d14, d15, #1     \n" //d10 = 00123456
                    "vext.8     d31, d14, d15, #2          \n" //d11 = 12345678
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vaddw.s16 q8, q8, d24                 \n" // out0 += vget_low_s16(out00)
                    "vaddw.s16 q9, q9, d25                 \n" // out0_1 += vget_high_s16(out00)
                    "vmull.s8 q13, d14, d8                 \n" // q13 = d12 * w01
                    "vmlal.s8 q13, d30, d9                 \n" // q13 += d10 * w00
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vmull.s8 q13, d31, d10                 \n" // q12 += d10 * w00
                    "vdup.32 q3, %[scale]                    \n"
                    "vdup.32 q4, %[bias]                     \n"
                    "vdup.32 q5, %[bias]                     \n"
                    "vmov.f32  q1, #0.0\n"
                    "vcvt.f32.s32   q8, q8                      \n"
                    "vcvt.f32.s32   q9, q9                      \n"
                    "vmla.f32 q4, q8, q3                     \n"
                    "vmla.f32 q5, q9, q3                     \n"
                    "vmax.f32 q4, q4, q1              \n"
                    "vmax.f32 q5, q5, q1              \n"
                    "vaddw.s16 q10, q10, d26                 \n" // out1 += vget_low_s16(out10)
                    "vaddw.s16 q11, q11, d27                 \n" // out1_1 += vget_high_s16(out10)
                    "vdup.32 q13, %[bias]                            \n"
                    "vdup.32 q14, %[bias]                            \n"
                    "vcvt.f32.s32   q10, q10                      \n"
                    "vcvt.f32.s32   q11, q11                      \n"
                    "vmla.f32 q13, q10, q3                     \n"
                    "vmla.f32 q14, q11, q3                     \n"
                    "vmax.f32 q13, q13, q1              \n"
                    "vmax.f32 q14, q14, q1              \n"
                    "vmov.f32 q8, #-0.5\n"
                    "vmov.f32 q9, #0.5\n"
                    "vcgt.f32   q1, q4, q8   \n"
                    "vbif.f32   q9, q8, q1    \n"
                    "vadd.f32   q4, q4, q9                    \n"
                    "vmov.f32   q9, #0.5\n"
                    "vcgt.f32   q2, q5, q8   \n"
                    "vbif.f32   q9, q8, q2   \n"
                    "vadd.f32   q5, q5, q9\n"
                    "vld1.32 {d2-d3}, [%[vmax]] \n"
                    "vcge.f32 q3, q4, q1         \n" /* data >= -127 */
                    "vbif q4, q1, q3    \n" /* choose data */
                    "vcge.f32 q3, q5, q1         \n" /* data >= -127 */
                    "vbif q5, q1, q3    \n" /* choose data */
                    "vcvt.s32.f32  q1, q4    \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q5    \n"  /* fp32 to int32 */
                    "vqmovn.s32 d8, q1       \n"  /* int32 to int16 */
                    "vqmovn.s32 d9, q2       \n"  /* int32 to int16 */
                    "vqmovn.s16 d7, q4      \n"   /* int16 to int8 */
                    "vld1.8 d10, [%[dout_ptr1]]    \n"
                    "vbif.8 d7, d10, d12                   \n"
                    "vst1.8    {d7}, [%[dout_ptr1]]!\n"
                    "vmov.f32 q8, #-0.5\n"
                    "vmov.f32 q9, #0.5\n"
                    "vcgt.f32   q1, q13, q8   \n"
                    "vbif.f32   q9, q8, q1    \n"
                    "vadd.f32   q13, q13, q9                    \n"
                    "vmov.f32   q9, #0.5\n"
                    "vcgt.f32   q2, q14, q8   \n"
                    "vbif.f32   q9, q8, q2   \n"
                    "vadd.f32   q14, q14, q9\n"
                    "vld1.32 {d2-d3}, [%[vmax]] \n"
                    "vcge.f32 q3, q13, q1         \n" /* data >= -127 */
                    "vbif q13, q1, q3    \n" /* choose data */
                    "vcge.f32 q3, q14, q1         \n" /* data >= -127 */
                    "vbif q14, q1, q3    \n" /* choose data */
                    "vcvt.s32.f32  q1, q13    \n"  /* fp32 to int32 */
                    "vcvt.s32.f32  q2, q14    \n"  /* fp32 to int32 */
                    "vqmovn.s32 d26, q1       \n"  /* int32 to int16 */
                    "vqmovn.s32 d27, q2       \n"  /* int32 to int16 */
                    "vqmovn.s16 d10, q13      \n"   /* int16 to int8 */
                    "vld1.8 d14, [%[dout_ptr2]]   \n"
                    "vbif.8 d10, d14, d12         \n"
                    "vst1.8    {d10}, [%[dout_ptr2]]!\n"
                    : [din_ptr0] "+r" (din_ptr0), [din_ptr1] "+r" (din_ptr1), [din_ptr2] "+r" (din_ptr2), \
                      [din_ptr3] "+r" (din_ptr3), \
                      [dout_ptr1] "+r"(doutr0), [dout_ptr2] "+r"(doutr1), [cnt] "+r" (cnt), \
                      [bias] "+r" (bias_val), [scale] "+r" (scale_val), [rs_mask] "+r" (rst_mask)
                    :[mask] "r" (vmask), [vmax]"r"(vmax_ptr)
                    :"cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", \
                      "q11", "q12", "q13", "q14", "q15"
                );
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
  }
}

void conv_depthwise_3x3s1_int8_float_impl(float* dout,
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
  if (padw == 1 && padh == 1) {
    if (flag_act == 0) {
      conv_depthwise_3x3s1p1_bias_int8_float(dout,
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
                                             ctx);
    } else if (flag_act == 1) {
      conv_depthwise_3x3s1p1_bias_relu_int8_float(dout,
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
                                                  ctx);
    }
  } else if (padw == 0 && padh == 0) {
    if (flag_act == 0) {
      conv_depthwise_3x3s1p0_bias_int8_float(dout,
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
                                             ctx);
    } else if (flag_act == 1) {
      conv_depthwise_3x3s1p0_bias_relu_int8_float(dout,
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
                                                  ctx);
    }
  }
}

void conv_depthwise_3x3s1_int8_int8_impl(int8_t* dout,
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
  if (padw == 1 && padh == 1) {
    if (flag_act == 0) {
      conv_depthwise_3x3s1p1_bias_int8_int8(dout,
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
                                            ctx);
    } else if (flag_act == 1) {
      conv_depthwise_3x3s1p1_bias_relu_int8_int8(dout,
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
                                                 ctx);
    }
  } else if (padw == 0 && padh == 0) {
    if (flag_act == 0) {
      conv_depthwise_3x3s1p0_bias_int8_int8(dout,
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
                                            ctx);
    } else if (flag_act == 1) {
      conv_depthwise_3x3s1p0_bias_relu_int8_int8(dout,
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
                                                 ctx);
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
