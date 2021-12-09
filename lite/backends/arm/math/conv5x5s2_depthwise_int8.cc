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

template <typename Dtype>
void conv_depthwise_5x5s2_int8(Dtype* dout,
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
  const int win_round = wout_round * 2 + 3;

  //! get h block
  //! llc_size = threads * win_round * hout_c_block * hin_r_block *
  //! sizeof(int8_t)
  //! + wout_round * hout_c_block * hout_r_block * threads * sizeof(int32_t)
  //! win_round = wout_round * 2 + 3
  //! hin_r_block = hout_r_block * 2 + 3
  int hout_r_block = (llc_size - 3 * win_round * hout_c_block * threads) /
                     (2 * win_round * hout_c_block * threads +
                      hout_c_block * wout_round * threads * 4);
  hout_r_block = hout_r_block > hout ? hout : hout_r_block;
  hout_r_block =
      ((hout_r_block + hout_r_kernel - 1) / hout_r_kernel) * hout_r_kernel;
  hout_r_block = hout_r_block < hout_r_kernel ? hout_r_kernel : hout_r_block;

  const int hin_r_block = hout_r_block * 2 + 3;

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
  int w_stride = 25;  // kernel_w * kernel_h;

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
      int hs = h * 2 - padh;
      int he = hs + h_kernel * 2 + 3;

      LITE_PARALLEL_COMMON_BEGIN(c, tid, chout, 0, hout_c_block) {
#ifdef LITE_USE_THREAD_POOL
        int8_t* pre_din = tmp_din + tid * (pre_in_size + pre_out_size * 4);
        int32_t* pre_out = reinterpret_cast<int*>(pre_din + pre_in_size);
#elif defined(ARM_WITH_OMP)
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
        const int8_t* block_inr3 = block_inr2 + in_len;
        const int8_t* block_inr4 = block_inr3 + in_len;

        const int8_t* weight_c = weights + c * w_stride;
        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          int cnt = w_loop;
          const int8_t* inr0 = block_inr0;
          const int8_t* inr1 = block_inr1;
          const int8_t* inr2 = block_inr2;
          const int8_t* inr3 = block_inr3;
          const int8_t* inr4 = block_inr4;

          int32_t* ptr_out0 = pre_out + hk * out_row_stride;
// clang-format off
#ifdef __aarch64__
          auto wptr = weight_c;
          asm volatile(
              "ld1  {v0.8b, v1.8b, v2.8b, v3.8b}, [%[r0]], #32\n"   /* load r0 0-3 */
              "ld1  {v4.8b, v5.8b, v6.8b, v7.8b}, [%[r0]], #32\n"   /* load r0 4-7 */
              "ld1  {v12.8b, v13.8b, v14.8b, v15.8b}, [%[wc]], #32\n" /* load wc 0-3 */
              "1:\n"
              /* in r0 */
              "smull  v20.8h, v0.8b,  v12.8b\n" /* w0, int16, out0 */
              "smull  v21.8h, v2.8b,  v12.8b\n" /* w0, int16, out1 */
              "smull  v22.8h, v4.8b,  v12.8b\n" /* w0, int16, out2 */
              "smull  v23.8h, v6.8b,  v12.8b\n" /* w0, int16, out3 */
              "ld1  {v8.8b, v9.8b, v10.8b, v11.8b}, [%[r0]]\n" /* load r0 8-11 */
              "smlal  v20.8h, v1.8b,  v13.8b\n" /* w1, int16, out0 */
              "smlal  v21.8h, v3.8b,  v13.8b\n" /* w1, int16, out1 */
              "smlal  v22.8h, v5.8b,  v13.8b\n" /* w1, int16, out2 */
              "smlal  v23.8h, v7.8b,  v13.8b\n" /* w1, int16, out3 */
              "sxtl   v24.4s, v20.4h\n" /* mov to out0 low */
              "sxtl2  v25.4s, v20.8h\n" /* mov to out0 hig */
              "sxtl   v26.4s, v21.4h\n" /* mov to out1 low */
              "sxtl2  v27.4s, v21.8h\n" /* mov to out1 hig */
              "sxtl   v28.4s, v22.4h\n" /* mov to out2 low */
              "sxtl2  v29.4s, v22.8h\n" /* mov to out2 hig */
              "sxtl   v30.4s, v23.4h\n" /* mov to out3 low */
              "sxtl2  v31.4s, v23.8h\n" /* mov to out3 hig */
              "ld1  {v16.8b, v17.8b, v18.8b, v19.8b}, [%[wc]], #32\n" /* load wc 4-7 */

              "smull  v20.8h, v2.8b,  v14.8b\n" /* w2, int16, out0 */
              "smull  v21.8h, v4.8b,  v14.8b\n" /* w2, int16, out1 */
              "smull  v22.8h, v6.8b,  v14.8b\n" /* w2, int16, out2 */
              "smull  v23.8h, v8.8b,  v14.8b\n" /* w2, int16, out3 */
              "smlal  v20.8h, v3.8b,  v15.8b\n" /* w3, int16, out0 */
              "smlal  v21.8h, v5.8b,  v15.8b\n" /* w3, int16, out1 */
              "smlal  v22.8h, v7.8b,  v15.8b\n" /* w3, int16, out2 */
              "smlal  v23.8h, v9.8b,  v15.8b\n" /* w3, int16, out3 */
              "saddw  v24.4s, v24.4s, v20.4h\n" /* add to out0 low */
              "saddw2 v25.4s, v25.4s, v20.8h\n" /* add to out0 hig */
              "saddw  v26.4s, v26.4s, v21.4h\n" /* add to out1 low */
              "saddw2 v27.4s, v27.4s, v21.8h\n" /* add to out1 hig */
              "ld1  {v0.8b, v1.8b, v2.8b, v3.8b}, [%[r1]], #32\n" /* load r1 0-3 */
              "saddw  v28.4s, v28.4s, v22.4h\n" /* add to out2 low */
              "saddw2 v29.4s, v29.4s, v22.8h\n" /* add to out2 hig */
              "saddw  v30.4s, v30.4s, v23.4h\n" /* add to out3 low */
              "saddw2 v31.4s, v31.4s, v23.8h\n" /* add to out3 hig */

              "smull  v20.8h, v4.8b,  v16.8b\n" /* w4, int16, out0 */
              "smull  v21.8h, v6.8b,  v16.8b\n" /* w4, int16, out1 */
              "smull  v22.8h, v8.8b,  v16.8b\n" /* w4, int16, out2 */
              "smull  v23.8h, v10.8b, v16.8b\n" /* w4, int16, out3 */
              "ld1  {v4.8b, v5.8b, v6.8b, v7.8b}, [%[r1]], #32\n" /* load r1 4-7 */
              /* in r1 */
              "smlal  v20.8h, v0.8b,  v17.8b\n" /* w5, int16, out0 */
              "smlal  v21.8h, v2.8b,  v17.8b\n" /* w5, int16, out1 */
              "smlal  v22.8h, v4.8b,  v17.8b\n" /* w5, int16, out2 */
              "smlal  v23.8h, v6.8b,  v17.8b\n" /* w5, int16, out3 */
              "ld1  {v8.8b, v9.8b, v10.8b, v11.8b}, [%[r1]]\n" /* load r1 8-11 */
              "saddw  v24.4s, v24.4s, v20.4h\n" /* add to out0 low */
              "saddw2 v25.4s, v25.4s, v20.8h\n" /* add to out0 hig */
              "saddw  v26.4s, v26.4s, v21.4h\n" /* add to out1 low */
              "saddw2 v27.4s, v27.4s, v21.8h\n" /* add to out1 hig */
              "saddw  v28.4s, v28.4s, v22.4h\n" /* add to out2 low */
              "saddw2 v29.4s, v29.4s, v22.8h\n" /* add to out2 hig */
              "saddw  v30.4s, v30.4s, v23.4h\n" /* add to out3 low */
              "saddw2 v31.4s, v31.4s, v23.8h\n" /* add to out3 hig */

              "smull  v20.8h, v1.8b,  v18.8b\n" /* w6, int16, out0 */
              "smull  v21.8h, v3.8b,  v18.8b\n" /* w6, int16, out1 */
              "smull  v22.8h, v5.8b,  v18.8b\n" /* w6, int16, out2 */
              "smull  v23.8h, v7.8b,  v18.8b\n" /* w6, int16, out3 */
              "ld1  {v12.8b, v13.8b, v14.8b, v15.8b}, [%[wc]], #32\n" /* load wc 8-11 */
              "smlal  v20.8h, v2.8b,  v19.8b\n" /* w7, int16, out0 */
              "smlal  v21.8h, v4.8b,  v19.8b\n" /* w7, int16, out1 */
              "smlal  v22.8h, v6.8b,  v19.8b\n" /* w7, int16, out2 */
              "smlal  v23.8h, v8.8b,  v19.8b\n" /* w7, int16, out3 */
              "saddw  v24.4s, v24.4s, v20.4h\n" /* add to out0 low */
              "saddw2 v25.4s, v25.4s, v20.8h\n" /* add to out0 hig */
              "saddw  v26.4s, v26.4s, v21.4h\n" /* add to out1 low */
              "saddw2 v27.4s, v27.4s, v21.8h\n" /* add to out1 hig */
              "ld1  {v16.8b, v17.8b, v18.8b, v19.8b}, [%[wc]], #32\n" /* load wc 12-15 */
              "saddw  v28.4s, v28.4s, v22.4h\n" /* add to out2 low */
              "saddw2 v29.4s, v29.4s, v22.8h\n" /* add to out2 hig */
              "saddw  v30.4s, v30.4s, v23.4h\n" /* add to out3 low */
              "saddw2 v31.4s, v31.4s, v23.8h\n" /* add to out3 hig */

              "smull  v20.8h, v3.8b,  v12.8b\n" /* w8, int16, out0 */
              "smull  v21.8h, v5.8b,  v12.8b\n" /* w8, int16, out1 */
              "smull  v22.8h, v7.8b,  v12.8b\n" /* w8, int16, out2 */
              "smull  v23.8h, v9.8b,  v12.8b\n" /* w8, int16, out3 */
              "ld1  {v0.8b, v1.8b, v2.8b, v3.8b}, [%[r2]], #32\n" /* load r2 0-3 */
              "smlal  v20.8h, v4.8b,  v13.8b\n" /* w9, int16, out0 */
              "smlal  v21.8h, v6.8b,  v13.8b\n" /* w9, int16, out1 */
              "smlal  v22.8h, v8.8b,  v13.8b\n" /* w9, int16, out2 */
              "smlal  v23.8h, v10.8b, v13.8b\n" /* w9, int16, out3 */
              "saddw  v24.4s, v24.4s, v20.4h\n" /* add to out0 low */
              "saddw2 v25.4s, v25.4s, v20.8h\n" /* add to out0 hig */
              "saddw  v26.4s, v26.4s, v21.4h\n" /* add to out1 low */
              "saddw2 v27.4s, v27.4s, v21.8h\n" /* add to out1 hig */
              "ld1  {v4.8b, v5.8b, v6.8b, v7.8b}, [%[r2]], #32\n" /* load r2 4-7 */
              "saddw  v28.4s, v28.4s, v22.4h\n" /* add to out2 low */
              "saddw2 v29.4s, v29.4s, v22.8h\n" /* add to out2 hig */
              "saddw  v30.4s, v30.4s, v23.4h\n" /* add to out3 low */
              "saddw2 v31.4s, v31.4s, v23.8h\n" /* add to out3 hig */

               /* in r2 */
              "smull  v20.8h, v0.8b,  v14.8b\n" /* w10, int16, out0 */
              "smull  v21.8h, v2.8b,  v14.8b\n" /* w10, int16, out1 */
              "smull  v22.8h, v4.8b,  v14.8b\n" /* w10, int16, out2 */
              "smull  v23.8h, v6.8b,  v14.8b\n" /* w10, int16, out3 */
              "ld1  {v8.8b, v9.8b, v10.8b, v11.8b}, [%[r2]]\n" /* load r2 8-11 */
              "smlal  v20.8h, v1.8b,  v15.8b\n" /* w11, int16, out0 */
              "smlal  v21.8h, v3.8b,  v15.8b\n" /* w11, int16, out1 */
              "smlal  v22.8h, v5.8b,  v15.8b\n" /* w11, int16, out2 */
              "smlal  v23.8h, v7.8b,  v15.8b\n" /* w11, int16, out3 */

              "saddw  v24.4s, v24.4s, v20.4h\n" /* add to out0 low */
              "saddw2 v25.4s, v25.4s, v20.8h\n" /* add to out0 hig */
              "saddw  v26.4s, v26.4s, v21.4h\n" /* add to out1 low */
              "saddw2 v27.4s, v27.4s, v21.8h\n" /* add to out1 hig */
              "ld1  {v12.8b, v13.8b, v14.8b, v15.8b}, [%[wc]], #32\n" /* load wc 16-19 */
              "saddw  v28.4s, v28.4s, v22.4h\n" /* add to out2 low */
              "saddw2 v29.4s, v29.4s, v22.8h\n" /* add to out2 hig */
              "saddw  v30.4s, v30.4s, v23.4h\n" /* add to out3 low */
              "saddw2 v31.4s, v31.4s, v23.8h\n" /* add to out3 hig */

              "smull  v20.8h, v2.8b,  v16.8b\n" /* w12, int16, out0 */
              "smull  v21.8h, v4.8b,  v16.8b\n" /* w12, int16, out1 */
              "smull  v22.8h, v6.8b,  v16.8b\n" /* w12, int16, out2 */
              "smull  v23.8h, v8.8b,  v16.8b\n" /* w12, int16, out3 */
              "smlal  v20.8h, v3.8b,  v17.8b\n" /* w13, int16, out0 */
              "smlal  v21.8h, v5.8b,  v17.8b\n" /* w13, int16, out1 */
              "smlal  v22.8h, v7.8b,  v17.8b\n" /* w13, int16, out2 */
              "smlal  v23.8h, v9.8b,  v17.8b\n" /* w13, int16, out3 */
              "ld1  {v0.8b, v1.8b, v2.8b, v3.8b}, [%[r3]], #32\n" /* load r3 0-3 */
              "saddw  v24.4s, v24.4s, v20.4h\n" /* add to out0 low */
              "saddw2 v25.4s, v25.4s, v20.8h\n" /* add to out0 hig */
              "saddw  v26.4s, v26.4s, v21.4h\n" /* add to out1 low */
              "saddw2 v27.4s, v27.4s, v21.8h\n" /* add to out1 hig */
              "saddw  v28.4s, v28.4s, v22.4h\n" /* add to out2 low */
              "saddw2 v29.4s, v29.4s, v22.8h\n" /* add to out2 hig */
              "saddw  v30.4s, v30.4s, v23.4h\n" /* add to out3 low */
              "saddw2 v31.4s, v31.4s, v23.8h\n" /* add to out3 hig */
              "smull  v20.8h, v4.8b,  v18.8b\n" /* w14, int16, out0 */
              "smull  v21.8h, v6.8b,  v18.8b\n" /* w14, int16, out1 */
              "ld1  {v4.8b, v5.8b, v6.8b, v7.8b}, [%[r3]], #32\n" /* load r3 4-7 */
              "smull  v22.8h, v8.8b,  v18.8b\n" /* w14, int16, out2 */
              "smull  v23.8h, v10.8b, v18.8b\n" /* w14, int16, out3 */
              /* in r3 */
              "smlal  v20.8h, v0.8b,  v19.8b\n" /* w15, int16, out0 */
              "smlal  v21.8h, v2.8b,  v19.8b\n" /* w15, int16, out1 */
              "smlal  v22.8h, v4.8b,  v19.8b\n" /* w15, int16, out2 */
              "smlal  v23.8h, v6.8b,  v19.8b\n" /* w15, int16, out3 */
              "saddw  v24.4s, v24.4s, v20.4h\n" /* add to out0 low */
              "saddw2 v25.4s, v25.4s, v20.8h\n" /* add to out0 hig */
              "saddw  v26.4s, v26.4s, v21.4h\n" /* add to out1 low */
              "saddw2 v27.4s, v27.4s, v21.8h\n" /* add to out1 hig */
              "ld1  {v8.8b, v9.8b, v10.8b, v11.8b}, [%[r3]]\n" /* load r3 8-11 */
              "saddw  v28.4s, v28.4s, v22.4h\n" /* add to out2 low */
              "saddw2 v29.4s, v29.4s, v22.8h\n" /* add to out2 hig */
              "saddw  v30.4s, v30.4s, v23.4h\n" /* add to out3 low */
              "saddw2 v31.4s, v31.4s, v23.8h\n" /* add to out3 hig */

              "smull  v20.8h, v1.8b,  v12.8b\n" /* w16, int16, out0 */
              "smull  v21.8h, v3.8b,  v12.8b\n" /* w16, int16, out1 */
              "smull  v22.8h, v5.8b,  v12.8b\n" /* w16, int16, out2 */
              "smull  v23.8h, v7.8b,  v12.8b\n" /* w16, int16, out3 */
              "ld1  {v16.8b, v17.8b, v18.8b, v19.8b}, [%[wc]], #32\n" /* load wc 20-23 */
              "smlal  v20.8h, v2.8b,  v13.8b\n" /* w17, int16, out0 */
              "smlal  v21.8h, v4.8b,  v13.8b\n" /* w17, int16, out1 */
              "smlal  v22.8h, v6.8b,  v13.8b\n" /* w17, int16, out2 */
              "smlal  v23.8h, v8.8b,  v13.8b\n" /* w17, int16, out3 */
              "saddw  v24.4s, v24.4s, v20.4h\n" /* add to out0 low */
              "saddw2 v25.4s, v25.4s, v20.8h\n" /* add to out0 hig */
              "saddw  v26.4s, v26.4s, v21.4h\n" /* add to out1 low */
              "saddw2 v27.4s, v27.4s, v21.8h\n" /* add to out1 hig */
              "saddw  v28.4s, v28.4s, v22.4h\n" /* add to out2 low */
              "saddw2 v29.4s, v29.4s, v22.8h\n" /* add to out2 hig */
              "saddw  v30.4s, v30.4s, v23.4h\n" /* add to out3 low */
              "saddw2 v31.4s, v31.4s, v23.8h\n" /* add to out3 hig */

              "smull  v20.8h, v3.8b,  v14.8b\n" /* w18, int16, out0 */
              "smull  v21.8h, v5.8b,  v14.8b\n" /* w18, int16, out1 */
              "smull  v22.8h, v7.8b,  v14.8b\n" /* w18, int16, out2 */
              "smull  v23.8h, v9.8b,  v14.8b\n" /* w18, int16, out3 */
              "ld1  {v0.8b, v1.8b, v2.8b, v3.8b}, [%[r4]], #32\n" /* load r4 0-3 */
              "smlal  v20.8h, v4.8b,  v15.8b\n" /* w19, int16, out0 */
              "smlal  v21.8h, v6.8b,  v15.8b\n" /* w19, int16, out1 */
              "smlal  v22.8h, v8.8b,  v15.8b\n" /* w19, int16, out2 */
              "smlal  v23.8h, v10.8b, v15.8b\n" /* w19, int16, out3 */
              "saddw  v24.4s, v24.4s, v20.4h\n" /* add to out0 low */
              "saddw2 v25.4s, v25.4s, v20.8h\n" /* add to out0 hig */
              "saddw  v26.4s, v26.4s, v21.4h\n" /* add to out1 low */
              "saddw2 v27.4s, v27.4s, v21.8h\n" /* add to out1 hig */
              "ld1  {v4.8b, v5.8b, v6.8b, v7.8b}, [%[r4]], #32\n" /* load r4 4-7 */
              "saddw  v28.4s, v28.4s, v22.4h\n" /* add to out2 low */
              "saddw2 v29.4s, v29.4s, v22.8h\n" /* add to out2 hig */
              "saddw  v30.4s, v30.4s, v23.4h\n" /* add to out3 low */
              "saddw2 v31.4s, v31.4s, v23.8h\n" /* add to out3 hig */

              /* in r4 */
              "smull  v20.8h, v0.8b,  v16.8b\n" /* w20, int16, out0 */
              "smull  v21.8h, v2.8b,  v16.8b\n" /* w20, int16, out1 */
              "smull  v22.8h, v4.8b,  v16.8b\n" /* w20, int16, out2 */
              "smull  v23.8h, v6.8b,  v16.8b\n" /* w20, int16, out3 */
              "ld1  {v8.8b, v9.8b, v10.8b, v11.8b}, [%[r4]]\n" /* load r4 8-11 */
              "smlal  v20.8h, v1.8b,  v17.8b\n" /* w21, int16, out0 */
              "smlal  v21.8h, v3.8b,  v17.8b\n" /* w21, int16, out1 */
              "smlal  v22.8h, v5.8b,  v17.8b\n" /* w21, int16, out2 */
              "smlal  v23.8h, v7.8b,  v17.8b\n" /* w21, int16, out3 */
              "saddw  v24.4s, v24.4s, v20.4h\n" /* add to out0 low */
              "saddw2 v25.4s, v25.4s, v20.8h\n" /* add to out0 hig */
              "saddw  v26.4s, v26.4s, v21.4h\n" /* add to out1 low */
              "saddw2 v27.4s, v27.4s, v21.8h\n" /* add to out1 hig */
              "saddw  v28.4s, v28.4s, v22.4h\n" /* add to out2 low */
              "saddw2 v29.4s, v29.4s, v22.8h\n" /* add to out2 hig */
              "saddw  v30.4s, v30.4s, v23.4h\n" /* add to out3 low */
              "saddw2 v31.4s, v31.4s, v23.8h\n" /* add to out3 hig */
              "ld1  {v16.8b}, [%[wc]], #8\n" /* load wc 24 */
              "smull  v20.8h, v2.8b,  v18.8b\n" /* w22, int16, out0 */
              "smull  v21.8h, v4.8b,  v18.8b\n" /* w22, int16, out1 */
              "smull  v22.8h, v6.8b,  v18.8b\n" /* w22, int16, out2 */
              "smull  v23.8h, v8.8b,  v18.8b\n" /* w22, int16, out3 */
              "sub    %[wc], %[wc], #200 \n"
              "smlal  v20.8h, v3.8b,  v19.8b\n" /* w23, int16, out0 */
              "smlal  v21.8h, v5.8b,  v19.8b\n" /* w23, int16, out1 */
              "smlal  v22.8h, v7.8b,  v19.8b\n" /* w23, int16, out2 */
              "smlal  v23.8h, v9.8b,  v19.8b\n" /* w23, int16, out3 */
              "ld1  {v0.8b, v1.8b, v2.8b, v3.8b}, [%[r0]], #32\n" /* load r0 0-3 */
              "saddw  v24.4s, v24.4s, v20.4h\n" /* add to out0 low */
              "saddw2 v25.4s, v25.4s, v20.8h\n" /* add to out0 hig */
              "saddw  v26.4s, v26.4s, v21.4h\n" /* add to out1 low */
              "saddw2 v27.4s, v27.4s, v21.8h\n" /* add to out1 hig */
              "saddw  v28.4s, v28.4s, v22.4h\n" /* add to out2 low */
              "saddw2 v29.4s, v29.4s, v22.8h\n" /* add to out2 hig */
              "ld1  {v12.8b, v13.8b, v14.8b, v15.8b}, [%[wc]], #32\n" /* load wc 0-3 */
              "saddw  v30.4s, v30.4s, v23.4h\n" /* add to out3 low */
              "saddw2 v31.4s, v31.4s, v23.8h\n" /* add to out3 hig */

              "smull  v20.8h, v4.8b,  v16.8b\n" /* w24, int16, out0 */
              "smull  v21.8h, v6.8b,  v16.8b\n" /* w24, int16, out1 */
              "smull  v22.8h, v8.8b,  v16.8b\n" /* w24, int16, out2 */
              "smull  v23.8h, v10.8b, v16.8b\n" /* w24, int16, out3 */
              "ld1  {v4.8b, v5.8b, v6.8b, v7.8b}, [%[r0]], #32\n" /* load r0 4-7 */
              "saddw  v24.4s, v24.4s, v20.4h\n" /* add to out0 low */
              "saddw2 v25.4s, v25.4s, v20.8h\n" /* add to out0 hig */
              "saddw  v26.4s, v26.4s, v21.4h\n" /* add to out1 low */
              "saddw2 v27.4s, v27.4s, v21.8h\n" /* add to out1 hig */
              "stp    q24, q25, [%[ptr_out0]], #32\n"
              "saddw  v28.4s, v28.4s, v22.4h\n" /* add to out2 low */
              "saddw2 v29.4s, v29.4s, v22.8h\n" /* add to out2 hig */
              "stp    q26, q27, [%[ptr_out0]], #32\n"
              "saddw  v30.4s, v30.4s, v23.4h\n" /* add to out3 low */
              "saddw2 v31.4s, v31.4s, v23.8h\n" /* add to out3 hig */
              "subs   %w[cnt], %w[cnt], #1\n"
              "stp    q28, q29, [%[ptr_out0]], #32\n"
              "stp    q30, q31, [%[ptr_out0]], #32\n"
              "bne    1b\n"
              : [cnt] "+r"(cnt),
                [r0] "+r"(inr0),
                [r1] "+r"(inr1),
                [r2] "+r"(inr2),
                [r3] "+r"(inr3),
                [r4] "+r"(inr4),
                [wc] "+r"(wptr),
                [ptr_out0] "+r"(ptr_out0)
              :
              : "cc","memory",
                "v0","v1","v2","v3","v4","v5","v6","v7",
                "v8","v9","v10","v11","v12","v13",
                "v14","v15","v16","v17","v18","v19",
                "v20","v21","v22","v23","v24","v25",
                "v26","v27","v28","v29","v30","v31"
              );
#else
          auto wptr = weight_c;
          asm volatile(
              "vld1.32    {d0-d3}, [%[r0]]!\n"    /* load r0, 0-3 */
              "vld1.32    {d4-d5}, [%[r0]]!\n"    /* load r0, 4-5 */
              "vld1.32    {d6-d7},  [%[wptr]]!\n" /* load w0-w1 */
              "1:\n"
              /* inr0 */
              "vmull.s8   q4, d0, d6\n"           /* int16, out0 */
              "vmull.s8   q5, d2, d6\n"           /* int16, out1 */
              "vmull.s8   q6, d4, d6\n"           /* int16, out2 */
              "vmlal.s8   q4, d1, d7\n"           /* int16, out0 */
              "vld1.32    {d0-d1}, [%[r0]]!\n"    /* load r0, 6-7 */
              "vmlal.s8   q5, d3, d7\n"           /* int16, out1 */
              "vmlal.s8   q6, d5, d7\n"           /* int16, out2 */
              "vmovl.s16  q8, d8\n"               /* mov to out0 low */
              "vmull.s8   q7, d0, d6\n"           /* int16, out3 */
              "vmovl.s16  q9, d9\n"               /* mov to out0 hig */
              "vmovl.s16  q10, d10\n"             /* mov to out1 low */
              "vmovl.s16  q11, d11\n"             /* mov to out1 hig */
              "vmlal.s8   q7, d1, d7\n"           /* int16, out3 */
              "vld1.32    {d6-d7}, [%[wptr]]!\n"  /* load w2-w3 */
              "vmovl.s16  q12, d12\n"             /* mov to out2 low */
              "vmovl.s16  q13, d13\n"             /* mov to out2 hig */
              "vmovl.s16  q14, d14\n"             /* mov to out3 low */
              "vmovl.s16  q15, d15\n"             /* mov to out3 hig */

              "vmull.s8   q4, d2, d6\n"           /* w2, int16, out0 */
              "vmull.s8   q5, d4, d6\n"           /* w2, int16, out1 */
              "vmull.s8   q6, d0, d6\n"           /* w2, int16, out2 */
              "vmlal.s8   q4, d3, d7\n"           /* w3, int16, out0 */
              "vld1.32    {d2-d3}, [%[r0]]!\n"    /* load r0, 8-9 */
              "vmlal.s8   q5, d5, d7\n"           /* w3, int16, out1 */
              "vmlal.s8   q6, d1, d7\n"           /* w3, int16, out2 */
              "vaddw.s16  q8, q8, d8\n"           /* add to out0 low */
              "vmull.s8   q7, d2, d6\n"           /* w2, int16, out3 */
              "vaddw.s16  q9, q9, d9\n"           /* add to out0 hig */
              "vaddw.s16  q10, q10, d10\n"        /* add to out1 low */
              "vaddw.s16  q11, q11, d11\n"        /* add to out1 hig */
              "vmlal.s8   q7, d3, d7\n"           /* w3, int16, out3 */
              "vld1.32    {d6-d7}, [%[wptr]]!\n"  /* load w4-w5 */
              "vld1.32    {d5}, [%[r0]]\n"        /* load r0, 10 */
              "sub %[r0], %[r0], #16\n"           /* r0 = r0 - 16 */
              "vaddw.s16  q12, q12, d12\n"        /* add to out2 low */
              "vaddw.s16  q13, q13, d13\n"        /* add to out2 hig */
              "vaddw.s16  q14, q14, d14\n"        /* add to out3 low */
              "vaddw.s16  q15, q15, d15\n"        /* add to out3 hig */

              "vmull.s8   q4, d4, d6\n"           /* w4, int16, out0 */
              "vmull.s8   q5, d0, d6\n"           /* w4, int16, out1 */
              "vmull.s8   q6, d2, d6\n"           /* w4, int16, out2 */
              "vmull.s8   q7, d5, d6\n"           /* w4, int16, out3 */
              "vld1.32    {d0-d3}, [%[r1]]!\n"    /* load r1, 0-3 */
              "vld1.32    {d4-d5}, [%[r1]]!\n"    /* load r1, 4-5 */
              /* inr1 */
              "vmlal.s8   q4, d0, d7\n"           /* w5, int16, out0 */
              "vmlal.s8   q5, d2, d7\n"           /* w5, int16, out1 */
              "vmlal.s8   q6, d4, d7\n"           /* w5, int16, out2 */
              "vld1.32    {d0}, [%[r1]]!\n"       /* load r1, 6 */
              "vaddw.s16  q8, q8, d8\n"           /* add to out0 low */
              "vaddw.s16  q9, q9, d9\n"           /* add to out0 hig */
              "vaddw.s16  q10, q10, d10\n"        /* add to out1 low */
              "vaddw.s16  q11, q11, d11\n"        /* add to out1 hig */
              "vmlal.s8   q7, d0, d7\n"           /* w5, int16, out3 */
              "vld1.32    {d6-d7}, [%[wptr]]!\n"  /* load w6-w7 */
              "vaddw.s16  q12, q12, d12\n"        /* add to out2 low */
              "vaddw.s16  q13, q13, d13\n"        /* add to out2 hig */
              "vaddw.s16  q14, q14, d14\n"        /* add to out3 low */
              "vaddw.s16  q15, q15, d15\n"        /* add to out3 hig */

              "vmull.s8   q4, d1, d6\n"           /* w6, int16, out0 */
              "vld1.32    {d1}, [%[r1]]!\n"       /* load r1, 7 */
              "vmull.s8   q5, d3, d6\n"           /* w6, int16, out1 */
              "vmull.s8   q6, d5, d6\n"           /* w6, int16, out2 */
              "vmlal.s8   q4, d2, d7\n"           /* w7, int16, out0 */
              "vmlal.s8   q5, d4, d7\n"           /* w7, int16, out1 */
              "vmlal.s8   q6, d0, d7\n"           /* w7, int16, out2 */
              "vmull.s8   q7, d1, d6\n"           /* w6, int16, out3 */
              "vld1.32    {d2}, [%[r1]]!\n"       /* load r1, 8 */
              "vaddw.s16  q8, q8, d8\n"           /* add to out0 low */
              "vaddw.s16  q9, q9, d9\n"           /* add to out0 hig */
              "vaddw.s16  q10, q10, d10\n"        /* add to out1 low */
              "vaddw.s16  q11, q11, d11\n"        /* add to out1 hig */
              "vmlal.s8   q7, d2, d7\n"           /* w7, int16, out3 */
              "vld1.32    {d6-d7}, [%[wptr]]!\n"  /* load w8-w9 */
              "vaddw.s16  q12, q12, d12\n"        /* add to out2 low */
              "vaddw.s16  q13, q13, d13\n"        /* add to out2 hig */
              "vaddw.s16  q14, q14, d14\n"        /* add to out3 low */
              "vaddw.s16  q15, q15, d15\n"        /* add to out3 hig */

              "vmull.s8   q4, d3, d6\n"           /* w8, int16, out0 */
              "vld1.32    {d3}, [%[r1]]!\n"       /* load r1, 9 */
              "vmull.s8   q5, d5, d6\n"           /* w8, int16, out1 */
              "vmull.s8   q6, d1, d6\n"           /* w8, int16, out2 */
              "vld1.32    {d5}, [%[r1]]\n"        /* load r1, 10 */
              "vmlal.s8   q4, d4, d7\n"           /* w9, int16, out0 */
              "vmlal.s8   q5, d0, d7\n"           /* w9, int16, out1 */
              "vmlal.s8   q6, d2, d7\n"           /* w9, int16, out2 */
              "vmull.s8   q7, d3, d6\n"           /* w8, int16, out3 */
              "vaddw.s16  q8, q8, d8\n"           /* add to out0 low */
              "vaddw.s16  q9, q9, d9\n"           /* add to out0 hig */
              "vaddw.s16  q10, q10, d10\n"        /* add to out1 low */
              "vaddw.s16  q11, q11, d11\n"        /* add to out1 hig */
              "vmlal.s8   q7, d5, d7\n"           /* w9, int16, out3 */
              "sub %[r1], %[r1], #16\n"           /* r1 = r1 - 16 */
              "vld1.32    {d6-d7}, [%[wptr]]!\n"  /* load w10-w11 */
              "vaddw.s16  q12, q12, d12\n"        /* add to out2 low */
              "vaddw.s16  q13, q13, d13\n"        /* add to out2 hig */
              "vaddw.s16  q14, q14, d14\n"        /* add to out3 low */
              "vaddw.s16  q15, q15, d15\n"        /* add to out3 hig */
              "vld1.32    {d0-d3}, [%[r2]]!\n"    /* load r2, 0-3 */
              "vld1.32    {d4-d5}, [%[r2]]!\n"    /* load r2, 4-5 */

              /* inr2 */
              "vmull.s8   q4, d0, d6\n"           /* w10, int16, out0 */
              "vmull.s8   q5, d2, d6\n"           /* w10, int16, out1 */
              "vmull.s8   q6, d4, d6\n"           /* w10, int16, out2 */
              "vmlal.s8   q4, d1, d7\n"           /* w11, int16, out0 */
              "vld1.32    {d0-d1}, [%[r2]]!\n"    /* load r2, 6-7 */
              "vmlal.s8   q5, d3, d7\n"           /* w11, int16, out1 */
              "vmlal.s8   q6, d5, d7\n"           /* w11, int16, out2 */
              "vaddw.s16  q8, q8, d8\n"           /* add to out0 low */
              "vmull.s8   q7, d0, d6\n"           /* w10, int16, out3 */
              "vaddw.s16  q9, q9, d9\n"           /* add to out0 hig */
              "vaddw.s16  q10, q10, d10\n"        /* add to out1 low */
              "vaddw.s16  q11, q11, d11\n"        /* add to out1 hig */
              "vmlal.s8   q7, d1, d7\n"           /* w11, int16, out3 */
              "vld1.32    {d6-d7}, [%[wptr]]!\n"  /* load w12-w13 */
              "vaddw.s16  q12, q12, d12\n"        /* add to out2 low */
              "vaddw.s16  q13, q13, d13\n"        /* add to out2 hig */
              "vaddw.s16  q14, q14, d14\n"        /* add to out3 low */
              "vaddw.s16  q15, q15, d15\n"        /* add to out3 hig */

              "vmull.s8   q4, d2, d6\n"           /* w12, int16, out0 */
              "vmull.s8   q5, d4, d6\n"           /* w12, int16, out1 */
              "vmull.s8   q6, d0, d6\n"           /* w12, int16, out2 */
              "vmlal.s8   q4, d3, d7\n"           /* w13, int16, out0 */
              "vld1.32    {d2-d3}, [%[r2]]!\n"    /* load r2, 8-9 */
              "vmlal.s8   q5, d5, d7\n"           /* w13, int16, out1 */
              "vmlal.s8   q6, d1, d7\n"           /* w13, int16, out2 */
              "vaddw.s16  q8, q8, d8\n"           /* add to out0 low */
              "vmull.s8   q7, d2, d6\n"           /* w12, int16, out3 */
              "vaddw.s16  q9, q9, d9\n"           /* add to out0 hig */
              "vaddw.s16  q10, q10, d10\n"        /* add to out1 low */
              "vaddw.s16  q11, q11, d11\n"        /* add to out1 hig */
              "vmlal.s8   q7, d3, d7\n"           /* w13, int16, out3 */
              "vld1.32    {d6-d7}, [%[wptr]]!\n"  /* load w14-w15 */
              "vld1.32    {d5}, [%[r2]]\n"        /* load r2, 10 */
              "sub %[r2], %[r2], #16\n"           /* r2 = r2 - 16 */
              "vaddw.s16  q12, q12, d12\n"        /* add to out2 low */
              "vaddw.s16  q13, q13, d13\n"        /* add to out2 hig */
              "vaddw.s16  q14, q14, d14\n"        /* add to out3 low */
              "vaddw.s16  q15, q15, d15\n"        /* add to out3 hig */

              "vmull.s8   q4, d4, d6\n"           /* w14, int16, out0 */
              "vmull.s8   q5, d0, d6\n"           /* w14, int16, out1 */
              "vmull.s8   q6, d2, d6\n"           /* w14, int16, out2 */
              "vmull.s8   q7, d5, d6\n"           /* w14, int16, out3 */
              "vld1.32    {d0-d3}, [%[r3]]!\n"    /* load r3, 0-3 */
              "vld1.32    {d4-d5}, [%[r3]]!\n"    /* load r3, 4-5 */
              /* inr3 */
              "vmlal.s8   q4, d0, d7\n"           /* w15, int16, out0 */
              "vmlal.s8   q5, d2, d7\n"           /* w15, int16, out1 */
              "vmlal.s8   q6, d4, d7\n"           /* w15, int16, out2 */
              "vld1.32    {d0}, [%[r3]]!\n"       /* load r3, 6 */
              "vaddw.s16  q8, q8, d8\n"           /* add to out0 low */
              "vaddw.s16  q9, q9, d9\n"           /* add to out0 hig */
              "vaddw.s16  q10, q10, d10\n"        /* add to out1 low */
              "vaddw.s16  q11, q11, d11\n"        /* add to out1 hig */
              "vmlal.s8   q7, d0, d7\n"           /* w15, int16, out3 */
              "vld1.32    {d6-d7}, [%[wptr]]!\n"  /* load w16-w17 */
              "vaddw.s16  q12, q12, d12\n"        /* add to out2 low */
              "vaddw.s16  q13, q13, d13\n"        /* add to out2 hig */
              "vaddw.s16  q14, q14, d14\n"        /* add to out3 low */
              "vaddw.s16  q15, q15, d15\n"        /* add to out3 hig */

              "vmull.s8   q4, d1, d6\n"           /* w16, int16, out0 */
              "vld1.32    {d1}, [%[r3]]!\n"       /* load r3, 7 */
              "vmull.s8   q5, d3, d6\n"           /* w16, int16, out1 */
              "vmull.s8   q6, d5, d6\n"           /* w16, int16, out2 */
              "vmlal.s8   q4, d2, d7\n"           /* w17, int16, out0 */
              "vmlal.s8   q5, d4, d7\n"           /* w17, int16, out1 */
              "vmlal.s8   q6, d0, d7\n"           /* w17, int16, out2 */
              "vmull.s8   q7, d1, d6\n"           /* w16, int16, out3 */
              "vld1.32    {d2}, [%[r3]]!\n"       /* load r3, 8 */
              "vaddw.s16  q8, q8, d8\n"           /* add to out0 low */
              "vaddw.s16  q9, q9, d9\n"           /* add to out0 hig */
              "vaddw.s16  q10, q10, d10\n"        /* add to out1 low */
              "vaddw.s16  q11, q11, d11\n"        /* add to out1 hig */
              "vmlal.s8   q7, d2, d7\n"           /* w17, int16, out3 */
              "vld1.32    {d6-d7}, [%[wptr]]!\n"  /* load w18-w19 */
              "vaddw.s16  q12, q12, d12\n"        /* add to out2 low */
              "vaddw.s16  q13, q13, d13\n"        /* add to out2 hig */
              "vaddw.s16  q14, q14, d14\n"        /* add to out3 low */
              "vaddw.s16  q15, q15, d15\n"        /* add to out3 hig */

              "vmull.s8   q4, d3, d6\n"           /* w18, int16, out0 */
              "vld1.32    {d3}, [%[r3]]!\n"       /* load r3, 9 */
              "vmull.s8   q5, d5, d6\n"           /* w18, int16, out1 */
              "vmull.s8   q6, d1, d6\n"           /* w18, int16, out2 */
              "vld1.32    {d5}, [%[r3]]\n"        /* load r3, 10 */
              "vmlal.s8   q4, d4, d7\n"           /* w19, int16, out0 */
              "vmlal.s8   q5, d0, d7\n"           /* w19, int16, out1 */
              "vmlal.s8   q6, d2, d7\n"           /* w19, int16, out2 */
              "vmull.s8   q7, d3, d6\n"           /* w18, int16, out3 */
              "vaddw.s16  q8, q8, d8\n"           /* add to out0 low */
              "vaddw.s16  q9, q9, d9\n"           /* add to out0 hig */
              "vaddw.s16  q10, q10, d10\n"        /* add to out1 low */
              "vaddw.s16  q11, q11, d11\n"        /* add to out1 hig */
              "vmlal.s8   q7, d5, d7\n"           /* w19, int16, out3 */
              "sub %[r3], %[r3], #16\n"           /* r3 = r3 - 16 */
              "vld1.32    {d6-d7}, [%[wptr]]!\n"  /* load w20-w21 */
              "vaddw.s16  q12, q12, d12\n"        /* add to out2 low */
              "vaddw.s16  q13, q13, d13\n"        /* add to out2 hig */
              "vaddw.s16  q14, q14, d14\n"        /* add to out3 low */
              "vaddw.s16  q15, q15, d15\n"        /* add to out3 hig */
              "vld1.32    {d0-d3}, [%[r4]]!\n"    /* load r4, 0-3 */
              "vld1.32    {d4-d5}, [%[r4]]!\n"    /* load r4, 4-5 */

              /* inr4 */
              "vmull.s8   q4, d0, d6\n"           /* w20, int16, out0 */
              "vmull.s8   q5, d2, d6\n"           /* w20, int16, out1 */
              "vmull.s8   q6, d4, d6\n"           /* w20, int16, out2 */
              "vmlal.s8   q4, d1, d7\n"           /* w21, int16, out0 */
              "vld1.32    {d0-d1}, [%[r4]]!\n"    /* load r4, 6-7 */
              "vmlal.s8   q5, d3, d7\n"           /* w21, int16, out1 */
              "vmlal.s8   q6, d5, d7\n"           /* w21, int16, out2 */
              "vaddw.s16  q8, q8, d8\n"           /* add to out0 low */
              "vmull.s8   q7, d0, d6\n"           /* w20, int16, out3 */
              "vaddw.s16  q9, q9, d9\n"           /* add to out0 hig */
              "vaddw.s16  q10, q10, d10\n"        /* add to out1 low */
              "vaddw.s16  q11, q11, d11\n"        /* add to out1 hig */
              "vmlal.s8   q7, d1, d7\n"           /* w21, int16, out3 */
              "vld1.32    {d6-d7}, [%[wptr]]!\n"  /* load w22-w23 */
              "vaddw.s16  q12, q12, d12\n"        /* add to out2 low */
              "vaddw.s16  q13, q13, d13\n"        /* add to out2 hig */
              "vaddw.s16  q14, q14, d14\n"        /* add to out3 low */
              "vaddw.s16  q15, q15, d15\n"        /* add to out3 hig */

              "vmull.s8   q4, d2, d6\n"           /* w22, int16, out0 */
              "vmull.s8   q5, d4, d6\n"           /* w22, int16, out1 */
              "vmull.s8   q6, d0, d6\n"           /* w22, int16, out2 */
              "vmlal.s8   q4, d3, d7\n"           /* w23, int16, out0 */
              "vld1.32    {d2-d3}, [%[r4]]!\n"    /* load r4, 7-8 */
              "vmlal.s8   q5, d5, d7\n"           /* w23, int16, out1 */
              "vmlal.s8   q6, d1, d7\n"           /* w23, int16, out2 */
              "vaddw.s16  q8, q8, d8\n"           /* add to out0 low */
              "vmull.s8   q7, d2, d6\n"           /* w22, int16, out3 */
              "vaddw.s16  q9, q9, d9\n"           /* add to out0 hig */
              "vaddw.s16  q10, q10, d10\n"        /* add to out1 low */
              "vaddw.s16  q11, q11, d11\n"        /* add to out1 hig */
              "vmlal.s8   q7, d3, d7\n"           /* w23, int16, out3 */
              "vld1.32    {d6}, [%[wptr]]!\n"     /* load w24 */
              "vld1.32    {d5}, [%[r4]]\n"        /* load r4, 10 */
              "sub %[r4], %[r4], #16\n"           /* r4 = r4 - 16 */
              "vaddw.s16  q12, q12, d12\n"        /* add to out2 low */
              "vaddw.s16  q13, q13, d13\n"        /* add to out2 hig */
              "vaddw.s16  q14, q14, d14\n"        /* add to out3 low */
              "vaddw.s16  q15, q15, d15\n"        /* add to out3 hig */
              "sub %[wptr], %[wptr], #200 \n"     /*  wptr = wptr - 200 */

              "vmull.s8   q4, d4, d6\n"           /* w22, int16, out0 */
              "vmull.s8   q5, d0, d6\n"           /* w22, int16, out1 */
              "vmull.s8   q6, d2, d6\n"           /* w22, int16, out2 */
              "vmull.s8   q7, d5, d6\n"           /* w22, int16, out3 */
              "vld1.32    {d0-d3}, [%[r0]]!\n"    /* load r0, 0-3 */
              "vld1.32    {d6-d7}, [%[wptr]]!\n"  /* load w0-w1 */
              "vaddw.s16  q8, q8, d8\n"           /* add to out0 low */
              "vaddw.s16  q9, q9, d9\n"           /* add to out0 hig */
              "vld1.32    {d4-d5}, [%[r0]]!\n"    /* load r0, 0-3 */
              "vaddw.s16  q10, q10, d10\n"        /* add to out1 low */
              "vaddw.s16  q11, q11, d11\n"        /* add to out1 hig */
              "vst1.32    {d16-d19},  [%[ptr_out0]]!\n"/* store out0 */
              "vaddw.s16  q12, q12, d12\n"        /* add to out2 low */
              "vaddw.s16  q13, q13, d13\n"        /* add to out2 hig */
              "vst1.32    {d20-d23},  [%[ptr_out0]]!\n"/*store out1 */
              "vaddw.s16  q14, q14, d14\n"        /* add to out3 low */
              "vaddw.s16  q15, q15, d15\n"        /* add to out3 hig */
              "subs       %[cnt], #1\n"           /* cnt = cnt - 1 */
              "vst1.32    {d24-d27},  [%[ptr_out0]]!\n"/* store out2 */
              "vst1.32    {d28-d31},  [%[ptr_out0]]!\n"/* store out3 */
              "bne 1b\n"                          /* branch main loop */
              : [cnt] "+r"(cnt),
                [r0] "+r"(inr0),
                [r1] "+r"(inr1),
                [r2] "+r"(inr2),
                [r3] "+r"(inr3),
                [r4] "+r"(inr4),
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
          // clang-format on
          block_inr0 = block_inr2;
          block_inr1 = block_inr3;
          block_inr2 = block_inr4;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
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
                                          bias + c,
                                          flag_bias,
                                          ptr_write,
                                          scale + c);
      }
      LITE_PARALLEL_END();
    }
  }
}

template void conv_depthwise_5x5s2_int8<int8_t>(int8_t* dout,
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

template void conv_depthwise_5x5s2_int8<float>(float* dout,
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
