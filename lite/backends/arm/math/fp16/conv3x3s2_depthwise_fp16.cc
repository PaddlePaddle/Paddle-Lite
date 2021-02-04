// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "lite/backends/arm/math/fp16/conv3x3_depthwise_fp16.h"
#include "lite/core/context.h"
#include "lite/operators/op_params.h"
#ifdef ARM_WITH_OMP
#include <omp.h>
#endif

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {
void conv_depthwise_3x3s2p1_bias_fp16_fp16(float16_t* dout,
                                           const float16_t* din,
                                           const float16_t* weights,
                                           const float16_t* bias,
                                           bool flag_bias,
                                           int num,
                                           int ch_in,
                                           int h_in,
                                           int w_in,
                                           int h_out,
                                           int w_out,
                                           ARMContext* ctx) {
  const uint16_t right_pad_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  const uint16_t out_pad_idx[4] = {0, 1, 2, 3};
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;
  uint16_t cnt_col = static_cast<uint16_t>((w_out >> 2) - 2);
  uint16_t size_right_remain = static_cast<uint16_t>(w_in - (7 + cnt_col * 8));
  if (size_right_remain >= 9) {
    cnt_col++;
    size_right_remain -= 8;
  }
  uint16_t cnt_remain = (size_right_remain == 8 && w_out % 4 == 0)
                            ? 4
                            : static_cast<uint16_t>(w_out % 4);
  uint16x8_t vmask_rp = vcgtq_u16(vdupq_n_u16(size_right_remain),
                                  vld1q_u16(right_pad_idx));  // 0 2 4 6 1 3 5 7
  uint16x4_t rmask_rp =
      vcgt_u16(vdup_n_u16(cnt_remain), vld1_u16(out_pad_idx));  // 0 1 2 3
  uint16_t vmask[8];
  vst1q_u16(vmask, vmask_rp);
  uint16_t rmask[4];
  vst1_u16(rmask, rmask_rp);
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * w_stride;
      float16_t v_bias[4] = {bias_val, bias_val, bias_val, bias_val};
      float16x4_t wr00 = vdup_n_f16(weight_ptr[0]);
      float16x4_t wr10 = vdup_n_f16(weight_ptr[3]);
      float16x4_t wr20 = vdup_n_f16(weight_ptr[6]);
      float16x4_t wr01 = vdup_n_f16(weight_ptr[1]);
      float16x4_t wr11 = vdup_n_f16(weight_ptr[4]);
      float16x4_t wr21 = vdup_n_f16(weight_ptr[7]);
      float16x4_t wr02 = vdup_n_f16(weight_ptr[2]);
      float16x4_t wr12 = vdup_n_f16(weight_ptr[5]);
      float16x4_t wr22 = vdup_n_f16(weight_ptr[8]);
      float16x4_t vzero = vdup_n_f16(0.f);

      float16_t* doutr0 = nullptr;
      float16_t* doutr1 = nullptr;

      const float16_t* dr0 = din_ch_ptr;
      const float16_t* dr1 = dr0 + w_in;
      const float16_t* dr2 = dr1 + w_in;
      const float16_t* dr3 = dr2 + w_in;
      const float16_t* dr4 = dr3 + w_in;

      const float16_t* din_ptr0 = nullptr;
      const float16_t* din_ptr1 = nullptr;
      const float16_t* din_ptr2 = nullptr;
      const float16_t* din_ptr3 = nullptr;
      const float16_t* din_ptr4 = nullptr;

      for (int i = 0; i < h_in; i += 4) {
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;
        din_ptr4 = dr4;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;

        uint16_t* rst_mask = rmask;
        uint16_t* val_mask = vmask;
        if (i == 0) {
          din_ptr0 = zero_ptr;
          din_ptr1 = dr0;
          din_ptr2 = dr1;
          din_ptr3 = dr2;
          din_ptr4 = dr3;
          dr0 = dr3;
          dr1 = dr4;
        } else {
          dr0 = dr4;
          dr1 = dr0 + w_in;
        }
        dr2 = dr1 + w_in;
        dr3 = dr2 + w_in;
        dr4 = dr3 + w_in;
        //! process bottom pad
        if (i + 4 > h_in) {
          switch (i + 4 - h_in) {
            case 4:
              din_ptr1 = zero_ptr;
            case 3:
              din_ptr2 = zero_ptr;
            case 2:
              din_ptr3 = zero_ptr;
            case 1:
              din_ptr4 = zero_ptr;
            default:
              break;
          }
        }
        if (i / 2 + 2 > h_out) {
          doutr1 = write_ptr;
        }
        int cnt = cnt_col;
// clang-format off
#ifdef __aarch64__
        asm volatile(
            "PRFM PLDL1KEEP, [%[din_ptr0]]                    \n"
            "PRFM PLDL1KEEP, [%[din_ptr1]]                    \n"
            "PRFM PLDL1KEEP, [%[din_ptr2]]                    \n"
            "PRFM PLDL1KEEP, [%[din_ptr3]]                    \n"
            "PRFM PLDL1KEEP, [%[din_ptr4]]                    \n"
          // left
            "ld2    {v0.4h, v1.4h}, [%[din_ptr0]], #16        \n"
            "ld2    {v2.4h, v3.4h}, [%[din_ptr1]], #16        \n"
            "ld2    {v4.4h, v5.4h}, [%[din_ptr2]], #16        \n"
            "ld2    {v6.4h, v7.4h}, [%[din_ptr3]], #16        \n"
            "ld2    {v8.4h, v9.4h}, [%[din_ptr4]], #16        \n"
            "ld1    {v16.4h}, [%[bias_val]]                   \n"
            "ld1    {v17.4h}, [%[bias_val]]                   \n"
            // r0
            "ext    v10.8b, %[vzero].8b, v1.8b, #6            \n"
            "fmul   v11.4h, v0.4h, %[wr01].4h                 \n"
            "fmul   v12.4h, v1.4h, %[wr02].4h                 \n"
            "fmla   v16.4h, v10.4h, %[wr00].4h                \n"
            "ext    v10.8b, %[vzero].8b, v3.8b, #6            \n"
            "sub    %[din_ptr0], %[din_ptr0], #2              \n"
            "sub    %[din_ptr1], %[din_ptr1], #2              \n"
            // r1
            "fmla   v11.4h, v2.4h, %[wr11].4h                 \n"
            "fmla   v12.4h, v3.4h, %[wr12].4h                 \n"
            "fmla   v16.4h, v10.4h, %[wr10].4h                \n"
            "ext    v10.8b, %[vzero].8b, v5.8b, #6            \n"
            // r2
            "sub    %[din_ptr2], %[din_ptr2], #2              \n"
            "sub    %[din_ptr3], %[din_ptr3], #2              \n"
            "fmul   v13.4h, v4.4h, %[wr01].4h                 \n"
            "fmla   v11.4h, v4.4h, %[wr21].4h                 \n"
            "fmul   v14.4h, v5.4h, %[wr02].4h                 \n"
            "fmla   v12.4h, v5.4h, %[wr22].4h                 \n"
            "fmla   v17.4h, v10.4h, %[wr00].4h                \n"
            "fmla   v16.4h, v10.4h, %[wr20].4h                \n"
            "ext    v10.8b, %[vzero].8b, v7.8b, #6            \n"
            // r3
            "sub    %[din_ptr4], %[din_ptr4], #2              \n"
            "fmla   v13.4h, v6.4h, %[wr11].4h                 \n"
            "fmla   v14.4h, v7.4h, %[wr12].4h                 \n"
            "fmla   v17.4h, v10.4h, %[wr10].4h                \n"
            "ext    v10.8b, %[vzero].8b, v9.8b, #6            \n"
            "fadd   v16.4h, v16.4h, v11.4h                    \n"
            "fadd   v16.4h, v16.4h, v12.4h                    \n"
            // r4
            "fmla   v13.4h, v8.4h, %[wr21].4h                 \n"
            "fmla   v14.4h, v9.4h, %[wr22].4h                 \n"
            "fmla   v17.4h, v10.4h, %[wr20].4h                \n"
            "st1    {v16.4h}, [%[ptr_out0]], #8               \n"
            "ld2    {v0.4h, v1.4h}, [%[din_ptr0]], #16        \n"
            "ld2    {v2.4h, v3.4h}, [%[din_ptr1]], #16        \n"
            "ld2    {v4.4h, v5.4h}, [%[din_ptr2]], #16        \n"
            "fadd   v17.4h, v17.4h, v13.4h                    \n"
            "ld2    {v6.4h, v7.4h}, [%[din_ptr3]], #16        \n"
            "ld2    {v8.4h, v9.4h}, [%[din_ptr4]], #16        \n"
            "ld1    {v15.4h}, [%[din_ptr0]]                   \n"
            "ld1    {v16.4h}, [%[bias_val]]                   \n"
            "fadd   v17.4h, v17.4h, v14.4h                    \n"
            "ld1    {v18.4h}, [%[din_ptr1]]                   \n"
            "ld1    {v19.4h}, [%[din_ptr2]]                   \n"
            "ext    v10.8b, v0.8b, v15.8b, #2                 \n"
            "ld1    {v20.4h}, [%[din_ptr3]]                   \n"
            "ld1    {v21.4h}, [%[din_ptr4]]                   \n"
            "st1    {v17.4h}, [%[ptr_out1]], #8               \n"
            "cmp    %w[cnt], #1                               \n"
            "ld1    {v17.4h}, [%[bias_val]]                   \n"
            "blt    1f                                        \n"
          // mid (for loop)
            "2:                                               \n"
            // r0
            "fmul   v11.4h, v0.4h, %[wr00].4h                \n"
            "fmul   v12.4h, v1.4h, %[wr01].4h                \n"
            "fmla   v16.4h, v10.4h, %[wr02].4h               \n"
            // r1
            "ext    v10.8b, v2.8b, v18.8b, #2                \n"
            "ld2    {v0.4h, v1.4h}, [%[din_ptr0]], #16       \n"
            "fmla   v11.4h, v2.4h, %[wr10].4h                \n"
            "fmla   v12.4h, v3.4h, %[wr11].4h                \n"
            "fmla   v16.4h, v10.4h, %[wr12].4h               \n"
            // r2
            "ext    v10.8b, v4.8b, v19.8b, #2                \n"
            "ld2    {v2.4h, v3.4h}, [%[din_ptr1]], #16       \n"
            "fmul   v13.4h, v4.4h, %[wr00].4h                \n"
            "fmla   v11.4h, v4.4h, %[wr20].4h                \n"
            "fmul   v14.4h, v5.4h, %[wr01].4h                \n"
            "fmla   v12.4h, v5.4h, %[wr21].4h                \n"
            "fmla   v17.4h, v10.4h, %[wr02].4h               \n"
            "fmla   v16.4h, v10.4h, %[wr22].4h               \n"
            // r3
            "ext    v10.8b, v6.8b, v20.8b, #2                 \n"
            "ld2    {v4.4h, v5.4h}, [%[din_ptr2]], #16        \n"
            "fmla   v13.4h, v6.4h, %[wr10].4h                 \n"
            "fmla   v14.4h, v7.4h, %[wr11].4h                 \n"
            "fmla   v17.4h, v10.4h, %[wr12].4h                \n"
            "ext    v10.8b, v8.8b, v21.8b, #2                 \n"
            "ld2    {v6.4h, v7.4h}, [%[din_ptr3]], #16        \n"
            "fadd   v16.4h, v16.4h, v11.4h                    \n"
            "fadd   v16.4h, v16.4h, v12.4h                    \n"
            // r4
            "fmla   v13.4h, v8.4h, %[wr20].4h                 \n"
            "fmla   v14.4h, v9.4h, %[wr21].4h                 \n"
            "fmla   v17.4h, v10.4h, %[wr22].4h                \n"
            "ld2    {v8.4h, v9.4h}, [%[din_ptr4]], #16        \n"
            "ld1    {v15.4h}, [%[din_ptr0]]                   \n"
            "ld1    {v18.4h}, [%[din_ptr1]]                   \n"
            "st1    {v16.4h}, [%[ptr_out0]], #8               \n"
            "fadd   v17.4h, v17.4h, v13.4h                    \n"
            "ld1    {v19.4h}, [%[din_ptr2]]                   \n"
            "ld1    {v20.4h}, [%[din_ptr3]]                   \n"
            "ld1    {v21.4h}, [%[din_ptr4]]                   \n"
            "fadd   v17.4h, v17.4h, v14.4h                    \n"
            "ext    v10.8b, v0.8b, v15.8b, #2                 \n"
            "ld1    {v16.4h}, [%[bias_val]]                   \n"
            "subs   %w[cnt], %w[cnt], #1                      \n"
            "st1    {v17.4h}, [%[ptr_out1]], #8               \n"
            "ld1    {v17.4h}, [%[bias_val]]                   \n"
            "bne    2b                                        \n"
          // right
            "1:                                               \n"
            "cmp    %w[remain], #1                            \n"
            "blt    4f                                        \n"
            "3:                                               \n"
            "ld1    {v18.4h}, [%[vmask]], #8                  \n"
            "ld1    {v19.4h}, [%[vmask]], #8                  \n"
            "ld1    {v20.4h}, [%[rmask]], #8                  \n"
            "bif    v0.8b, %[vzero].8b, v18.8b                \n"
            "bif    v1.8b, %[vzero].8b, v19.8b                \n"
            "bif    v2.8b, %[vzero].8b, v18.8b                \n"
            "bif    v3.8b, %[vzero].8b, v19.8b                \n"
            "bif    v4.8b, %[vzero].8b, v18.8b                \n"
            "bif    v5.8b, %[vzero].8b, v19.8b                \n"
            // r0
            "ext    v10.8b, v0.8b, %[vzero].8b, #2            \n"
            "bif    v6.8b, %[vzero].8b, v18.8b                \n"
            "bif    v7.8b, %[vzero].8b, v19.8b                \n"
            "fmul   v11.4h, v0.4h, %[wr00].4h                 \n"
            "fmul   v12.4h, v1.4h, %[wr01].4h                 \n"
            "fmla   v16.4h, v10.4h, %[wr02].4h                \n"
            // r1
            "ext    v10.8b, v2.8b, %[vzero].8b, #2            \n"
            "bif    v8.8b, %[vzero].8b, v18.8b                \n"
            "bif    v9.8b, %[vzero].8b, v19.8b                \n"
            "fmla   v11.4h, v2.4h, %[wr10].4h                 \n"
            "fmla   v12.4h, v3.4h, %[wr11].4h                 \n"
            "fmla   v16.4h, v10.4h, %[wr12].4h                \n"
            // r2
            "ext    v10.8b, v4.8b, %[vzero].8b, #2            \n"
            "fmul   v13.4h, v4.4h, %[wr00].4h                 \n"
            "fmla   v11.4h, v4.4h, %[wr20].4h                 \n"
            "fmul   v14.4h, v5.4h, %[wr01].4h                 \n"
            "fmla   v12.4h, v5.4h, %[wr21].4h                 \n"
            "fmla   v17.4h, v10.4h, %[wr02].4h                \n"
            "fmla   v16.4h, v10.4h, %[wr22].4h                \n"
            // r3
            "ext    v10.8b, v6.8b, %[vzero].8b, #2            \n"
            "fmla   v13.4h, v6.4h, %[wr10].4h                 \n"
            "fmla   v14.4h, v7.4h, %[wr11].4h                 \n"
            "fmla   v17.4h, v10.4h, %[wr12].4h                \n"
            "ext    v10.8b, v8.8b, %[vzero].8b, #2            \n"
            "ld1    {v0.4h}, [%[ptr_out0]]                    \n"
            "fadd   v16.4h, v16.4h, v11.4h                    \n"
            "fadd   v16.4h, v16.4h, v12.4h                    \n"
            "ld1    {v1.4h}, [%[ptr_out1]]                    \n"
            // r4
            "fmla   v13.4h, v8.4h, %[wr20].4h                 \n"
            "fmla   v14.4h, v9.4h, %[wr21].4h                 \n"
            "fmla   v17.4h, v10.4h, %[wr22].4h                \n"
            "bif    v16.8b, v0.8b, v20.8b                     \n"
            "fadd   v17.4h, v17.4h, v13.4h                    \n"
            "st1    {v16.4h}, [%[ptr_out0]], #8               \n"
            "fadd   v17.4h, v17.4h, v14.4h                    \n"
            "bif    v17.8b, v1.8b, v20.8b                     \n"
            "st1    {v17.4h}, [%[ptr_out1]], #8               \n"
            "4:                                               \n"
            : [cnt] "+r"(cnt),
              [din_ptr0] "+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1),
              [vmask] "+r" (val_mask),
              [rmask] "+r" (rst_mask)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias),
              [remain] "r"(cnt_remain)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
  }
}

void conv_depthwise_3x3s2p0_bias_fp16_fp16(float16_t* dout,
                                           const float16_t* din,
                                           const float16_t* weights,
                                           const float16_t* bias,
                                           bool flag_bias,
                                           int num,
                                           int ch_in,
                                           int h_in,
                                           int w_in,
                                           int h_out,
                                           int w_out,
                                           ARMContext* ctx) {
  const uint16_t right_pad_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};
  const uint16_t out_pad_idx[4] = {0, 1, 2, 3};
  int tile_w = w_out >> 2;
  int cnt_remain = w_out % 4;
  uint16_t size_right_remain = (uint16_t)(8 + (tile_w << 3) - w_in);
  size_right_remain = 8 - size_right_remain;
  if (cnt_remain == 0 && size_right_remain == 0) {
    cnt_remain = 4;
    tile_w -= 1;
    size_right_remain = 8;
  }
  uint16x8_t vmask_rp = vcgtq_u16(vdupq_n_u16(size_right_remain),
                                  vld1q_u16(right_pad_idx));  // 0 2 4 6 1 3 5 7
  uint16x4_t rmask_rp =
      vcgt_u16(vdup_n_u16(cnt_remain), vld1_u16(out_pad_idx));  // 0 1 2 3
  uint16_t vmask[8];
  vst1q_u16(vmask, vmask_rp);
  uint16_t rmask[4];
  vst1_u16(rmask, rmask_rp);
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* weight_ptr = weights + c * w_stride;
      float16_t v_bias[4] = {bias_val, bias_val, bias_val, bias_val};
      float16x4_t wr00 = vdup_n_f16(weight_ptr[0]);
      float16x4_t wr10 = vdup_n_f16(weight_ptr[3]);
      float16x4_t wr20 = vdup_n_f16(weight_ptr[6]);
      float16x4_t wr01 = vdup_n_f16(weight_ptr[1]);
      float16x4_t wr11 = vdup_n_f16(weight_ptr[4]);
      float16x4_t wr21 = vdup_n_f16(weight_ptr[7]);
      float16x4_t wr02 = vdup_n_f16(weight_ptr[2]);
      float16x4_t wr12 = vdup_n_f16(weight_ptr[5]);
      float16x4_t wr22 = vdup_n_f16(weight_ptr[8]);
      float16x4_t vzero = vdup_n_f16(0.f);

      float16_t* doutr0 = nullptr;
      float16_t* doutr1 = nullptr;

      const float16_t* dr0 = din_ch_ptr;
      const float16_t* dr1 = dr0 + w_in;
      const float16_t* dr2 = dr1 + w_in;
      const float16_t* dr3 = dr2 + w_in;
      const float16_t* dr4 = dr3 + w_in;

      const float16_t* din_ptr0 = dr0;
      const float16_t* din_ptr1 = dr1;
      const float16_t* din_ptr2 = dr2;
      const float16_t* din_ptr3 = dr3;
      const float16_t* din_ptr4 = dr4;

      for (int i = 0; i < h_out; i += 2) {
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;
        din_ptr4 = dr4;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;

        uint16_t* rst_mask = rmask;
        uint16_t* val_mask = vmask;
        dr0 = dr4;
        dr1 = dr0 + w_in;
        dr2 = dr1 + w_in;
        dr3 = dr2 + w_in;
        dr4 = dr3 + w_in;
        //! process bottom pad
        if (i * 2 + 5 > h_in) {
          switch (i * 2 + 5 - h_in) {
            case 4:
              din_ptr1 = zero_ptr;
            case 3:
              din_ptr2 = zero_ptr;
            case 2:
              din_ptr3 = zero_ptr;
            case 1:
              din_ptr4 = zero_ptr;
            case 0:
              din_ptr4 = zero_ptr;
            default:
              break;
          }
        }
        if (i + 2 > h_out) {
          doutr1 = write_ptr;
        }
        int cnt = tile_w;
// clang-format off
#ifdef __aarch64__
        asm volatile(
            "PRFM PLDL1KEEP, [%[din_ptr0]]                    \n"
            "PRFM PLDL1KEEP, [%[din_ptr1]]                    \n"
            "PRFM PLDL1KEEP, [%[din_ptr2]]                    \n"
            "PRFM PLDL1KEEP, [%[din_ptr3]]                    \n"
            "PRFM PLDL1KEEP, [%[din_ptr4]]                    \n"
            "ld2    {v0.4h, v1.4h}, [%[din_ptr0]], #16        \n"
            "ld2    {v2.4h, v3.4h}, [%[din_ptr1]], #16        \n"
            "ld2    {v4.4h, v5.4h}, [%[din_ptr2]], #16        \n"
            "ld2    {v6.4h, v7.4h}, [%[din_ptr3]], #16        \n"
            "ld2    {v8.4h, v9.4h}, [%[din_ptr4]], #16        \n"
            "ld1    {v16.4h}, [%[bias_val]]                   \n"
            "ld1    {v17.4h}, [%[bias_val]]                   \n"
            "ld1    {v15.4h}, [%[din_ptr0]]                   \n"
            "ld1    {v18.4h}, [%[din_ptr1]]                   \n"
            "ld1    {v19.4h}, [%[din_ptr2]]                   \n"
            "ld1    {v20.4h}, [%[din_ptr3]]                   \n"
            "ld1    {v21.4h}, [%[din_ptr4]]                   \n"
            "ext    v10.8b, v0.8b, v15.8b, #2                 \n"
            "2:                                               \n"
            // r0
            "fmul   v11.4h, v0.4h, %[wr00].4h                 \n"
            "fmul   v12.4h, v1.4h, %[wr01].4h                 \n"
            "fmla   v16.4h, v10.4h, %[wr02].4h                \n"
            "ext    v10.8b, v2.8b, v18.8b, #2                 \n"
            // r1
            "ld2    {v0.4h, v1.4h}, [%[din_ptr0]], #16        \n"
            "fmla   v11.4h, v2.4h, %[wr10].4h                 \n"
            "fmla   v12.4h, v3.4h, %[wr11].4h                 \n"
            "fmla   v16.4h, v10.4h, %[wr12].4h                \n"
            "ext    v10.8b, v4.8b, v19.8b, #2                 \n"
            // r2
            "ld2    {v2.4h, v3.4h}, [%[din_ptr1]], #16        \n"
            "fmul   v13.4h, v4.4h, %[wr00].4h                 \n"
            "fmla   v11.4h, v4.4h, %[wr20].4h                 \n"
            "fmul   v14.4h, v5.4h, %[wr01].4h                 \n"
            "fmla   v12.4h, v5.4h, %[wr21].4h                 \n"
            "fmla   v17.4h, v10.4h, %[wr02].4h                \n"
            "fmla   v16.4h, v10.4h, %[wr22].4h                \n"
            "ext    v10.8b, v6.8b, v20.8b, #2                 \n"
            // r3
            "ld2    {v4.4h, v5.4h}, [%[din_ptr2]], #16        \n"
            "fmla   v13.4h, v6.4h, %[wr10].4h                 \n"
            "fmla   v14.4h, v7.4h, %[wr11].4h                 \n"
            "fmla   v17.4h, v10.4h, %[wr12].4h                \n"
            "ext    v10.8b, v8.8b, v21.8b, #2                 \n"
            "ld2    {v6.4h, v7.4h}, [%[din_ptr3]], #16        \n"
            "fadd   v16.4h, v16.4h, v11.4h                    \n"
            "fadd   v16.4h, v16.4h, v12.4h                    \n"
            // r4
            "fmla   v13.4h, v8.4h, %[wr20].4h                 \n"
            "fmla   v14.4h, v9.4h, %[wr21].4h                 \n"
            "fmla   v17.4h, v10.4h, %[wr22].4h                \n"
            "ld2    {v8.4h, v9.4h}, [%[din_ptr4]], #16        \n"
            "ld1    {v15.4h}, [%[din_ptr0]]                   \n"
            "ld1    {v18.4h}, [%[din_ptr1]]                   \n"
            "st1    {v16.4h}, [%[ptr_out0]], #8               \n"
            "fadd   v17.4h, v17.4h, v13.4h                    \n"
            "ld1    {v19.4h}, [%[din_ptr2]]                   \n"
            "ld1    {v20.4h}, [%[din_ptr3]]                   \n"
            "ld1    {v21.4h}, [%[din_ptr4]]                   \n"
            "fadd   v17.4h, v17.4h, v14.4h                    \n"
            "ext    v10.8b, v0.8b, v15.8b, #2                 \n"
            "ld1    {v16.4h}, [%[bias_val]]                   \n"
            "subs   %w[cnt], %w[cnt], #1                      \n"
            "st1    {v17.4h}, [%[ptr_out1]], #8               \n"
            "ld1    {v17.4h}, [%[bias_val]]                   \n"
            "bne    2b                                        \n"
            "1:                                               \n"
            "cmp    %w[remain], #1                            \n"
            "blt    4f                                        \n"
            // right
            "3:                                               \n"
            "ld1    {v18.4h}, [%[vmask]], #8                  \n"
            "ld1    {v19.4h}, [%[vmask]], #8                  \n"
            "ld1    {v20.4h}, [%[rmask]], #8                  \n"
            "bif    v0.8b, %[vzero].8b, v18.8b                \n"
            "bif    v1.8b, %[vzero].8b, v19.8b                \n"
            "bif    v2.8b, %[vzero].8b, v18.8b                \n"
            "bif    v3.8b, %[vzero].8b, v19.8b                \n"
            "bif    v4.8b, %[vzero].8b, v18.8b                \n"
            "bif    v5.8b, %[vzero].8b, v19.8b                \n"
            "ext    v10.8b, v0.8b, %[vzero].8b, #2            \n"
            "bif    v6.8b, %[vzero].8b, v18.8b                \n"
            // r0
            "bif    v7.8b, %[vzero].8b, v19.8b                \n"
            "fmul   v11.4h, v0.4h, %[wr00].4h                 \n"
            "fmul   v12.4h, v1.4h, %[wr01].4h                 \n"
            "fmla   v16.4h, v10.4h, %[wr02].4h                \n"
            "ext    v10.8b, v2.8b, %[vzero].8b, #2            \n"
            "bif    v8.8b, %[vzero].8b, v18.8b                \n"
            // r1
            "bif    v9.8b, %[vzero].8b, v19.8b                \n"
            "fmla   v11.4h, v2.4h, %[wr10].4h                 \n"
            "fmla   v12.4h, v3.4h, %[wr11].4h                 \n"
            "fmla   v16.4h, v10.4h, %[wr12].4h                \n"
            // r2
            "ext    v10.8b, v4.8b, %[vzero].8b, #2            \n"
            "fmul   v13.4h, v4.4h, %[wr00].4h                 \n"
            "fmla   v11.4h, v4.4h, %[wr20].4h                 \n"
            "fmul   v14.4h, v5.4h, %[wr01].4h                 \n"
            "fmla   v12.4h, v5.4h, %[wr21].4h                 \n"
            "fmla   v17.4h, v10.4h, %[wr02].4h                \n"
            "fmla   v16.4h, v10.4h, %[wr22].4h                \n"
            // r3
            "ext    v10.8b, v6.8b, %[vzero].8b, #2            \n"
            "fmla   v13.4h, v6.4h, %[wr10].4h                 \n"
            "fmla   v14.4h, v7.4h, %[wr11].4h                 \n"
            "fmla   v17.4h, v10.4h, %[wr12].4h                \n"
            "ext    v10.8b, v8.8b, %[vzero].8b, #2            \n"
            "ld1    {v0.4h}, [%[ptr_out0]]                    \n"
            "fadd   v16.4h, v16.4h, v11.4h                    \n"
            "fadd   v16.4h, v16.4h, v12.4h                    \n"
            "ld1    {v1.4h}, [%[ptr_out1]]                    \n"
            // r4
            "fmla   v13.4h, v8.4h, %[wr20].4h                 \n"
            "fmla   v14.4h, v9.4h, %[wr21].4h                 \n"
            "fmla   v17.4h, v10.4h, %[wr22].4h                \n"
            "bif    v16.8b, v0.8b, v20.8b                     \n"
            "fadd   v17.4h, v17.4h, v13.4h                    \n"
            "st1    {v16.4h}, [%[ptr_out0]], #8               \n"
            "fadd   v17.4h, v17.4h, v14.4h                    \n"
            "bif    v17.8b, v1.8b, v20.8b                     \n"
            "st1    {v17.4h}, [%[ptr_out1]], #8               \n"
            "4:                                               \n"
            : [cnt]"+r"(cnt),
              [din_ptr0]"+r"(din_ptr0),
              [din_ptr1] "+r"(din_ptr1),
              [din_ptr2] "+r"(din_ptr2),
              [din_ptr3] "+r"(din_ptr3),
              [din_ptr4] "+r"(din_ptr4),
              [ptr_out0] "+r"(doutr0),
              [ptr_out1] "+r"(doutr1),
              [vmask] "+r" (val_mask),
              [rmask] "+r" (rst_mask)
            : [vzero] "w"(vzero),
              [wr00]"w"(wr00),
              [wr01]"w"(wr01),
              [wr02]"w"(wr02),
              [wr10]"w"(wr10),
              [wr11]"w"(wr11),
              [wr12]"w"(wr12),
              [wr20]"w"(wr20),
              [wr21]"w"(wr21),
              [wr22] "w" (wr22),
              [bias_val] "r"(v_bias),
              [remain] "r"(cnt_remain)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",\
              "v17", "v18", "v19", "v20", "v21"
        );
#else
#endif
        // clang-format on
        dout_ptr += 2 * w_out;
      }
    }
  }
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
