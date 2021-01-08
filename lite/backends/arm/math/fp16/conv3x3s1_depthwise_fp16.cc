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

#include "lite/backends/arm/math/fp16/conv3x3s1_depthwise_fp16.h"
#include <arm_neon.h>
#include "lite/backends/arm/math/conv_block_utils.h"
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
void conv_depthwise_3x3s1p1_bias_fp16_fp16(float16_t* dout,
                                           const float16_t* din,
                                           const float16_t* weights,
                                           const float16_t* bias,
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
  const uint16_t right_pad_idx[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const uint16_t right_pad_rst[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  float16_t* zero_ptr = ctx->workspace_data<float16_t>();
  memset(zero_ptr, 0, w_in * sizeof(float16_t));
  float16_t* write_ptr =
      reinterpret_cast<float16_t*>(ctx->workspace_data<float16_t>() + w_in);
  int threads = ctx->threads();
  int size_in_channel = w_in * h_in;
  int size_out_channel = w_out * h_out;
  int w_stride = 9;
  int tile_w = (w_in + 7) >> 3;
  int cnt_col = tile_w - 2;
  uint16_t size_pad_right = static_cast<uint16_t>(w_in - 7 - (cnt_col << 3));
  uint16_t rst_remain = static_cast<uint16_t>(w_out - ((cnt_col + 1) << 3));
  uint16x8_t vmask_result =
      vcgtq_u16(vdupq_n_u16(rst_remain), vld1q_u16(right_pad_rst));
  // float16x8_t vzero = vdupq_n_f16(0);
  uint8x16_t vmask_rp1 =
      vcgtq_u16(vdupq_n_u16(size_pad_right), vld1q_u16(right_pad_idx));
  uint8x16_t vmask_rp2 =
      vcgtq_u16(vdupq_n_u16(size_pad_right), vld1q_u16(right_pad_idx + 8));
  uint16_t vmask[16];
  vst1q_u16(vmask, vmask_rp1);
  vst1q_u16(vmask + 8, vmask_rp2);
  uint16_t rmask[8];
  vst1q_u16(rmask, vmask_result);
  for (int n = 0; n < num; ++n) {
    const float16_t* din_batch = din + n * ch_in * size_in_channel;
    float16_t* dout_batch = dout + n * ch_in * size_out_channel;
#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < ch_in; c++) {
      float16_t* dout_ptr = dout_batch + c * size_out_channel;
      const float16_t* din_ch_ptr = din_batch + c * size_in_channel;
      float16_t bias_val =
          flag_bias ? static_cast<const float16_t>(bias[c]) : 0;
      const float16_t* wei_ptr = weights + c * w_stride;
      float16_t v_bias[8] = {bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val,
                             bias_val};
#ifdef __aarch64__
      float16x8_t wr00 = vdupq_n_f16(wei_ptr[0]);
      float16x8_t wr10 = vdupq_n_f16(wei_ptr[3]);
      float16x8_t wr20 = vdupq_n_f16(wei_ptr[6]);
      float16x8_t wr01 = vdupq_n_f16(wei_ptr[1]);
      float16x8_t wr11 = vdupq_n_f16(wei_ptr[4]);
      float16x8_t wr21 = vdupq_n_f16(wei_ptr[7]);
      float16x8_t wr02 = vdupq_n_f16(wei_ptr[2]);
      float16x8_t wr12 = vdupq_n_f16(wei_ptr[5]);
      float16x8_t wr22 = vdupq_n_f16(wei_ptr[8]);
#endif
      float16_t* doutr0 = nullptr;
      float16_t* doutr1 = nullptr;

      const float16_t* dr0 = din_ch_ptr;
      const float16_t* dr1 = dr0 + w_in;
      const float16_t* dr2 = dr1 + w_in;
      const float16_t* dr3 = dr2 + w_in;

      const float16_t* din_ptr0 = nullptr;
      const float16_t* din_ptr1 = nullptr;
      const float16_t* din_ptr2 = nullptr;
      const float16_t* din_ptr3 = nullptr;

      for (int i = 0; i < h_in; i += 2) {
        din_ptr0 = dr0;
        din_ptr1 = dr1;
        din_ptr2 = dr2;
        din_ptr3 = dr3;

        doutr0 = dout_ptr;
        doutr1 = doutr0 + w_out;
        uint16_t* rst_mask = rmask;
        uint16_t* val_mask = vmask;
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
        if (i + 2 > h_out) {
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
            "movi   v21.8h, #0x0                              \n"
          // left
            "ld1    {v0.8h}, [%[din_ptr0]], #16               \n"
            "ld1    {v2.8h}, [%[din_ptr1]], #16               \n"
            "ld1    {v1.8h}, [%[din_ptr0]]                    \n"
            "ld1    {v3.8h}, [%[din_ptr1]]                    \n"
            "ld1    {v18.8h}, [%[bias_val]]                   \n"
            "ld1    {v19.8h}, [%[bias_val]]                   \n"
            // r0
            "fmla   v18.8h, %[ww1].8h, v0.8h                  \n"//v18: output0 mla result
            "ext    v8.16b, v21.16b, v0.16b, #14              \n"//v8: row0 left 8 data
            "ext    v9.16b, v0.16b, v1.16b, #2                \n"//v9: row0 right 8 data
            "ld1    {v4.8h}, [%[din_ptr2]], #16               \n"
            "ld1    {v6.8h}, [%[din_ptr3]], #16               \n"
            "fmla   v18.8h, %[ww0].8h, v8.8h                  \n"
            "ld1    {v5.8h}, [%[din_ptr2]]                    \n"
            "ld1    {v7.8h}, [%[din_ptr3]]                    \n"
            "fmla   v18.8h, %[ww2].8h, v9.8h                  \n"
            "sub    %[din_ptr0], %[din_ptr0], #2              \n"
            "sub    %[din_ptr1], %[din_ptr1], #2              \n"
            "ext    v10.16b, v21.16b, v2.16b, #14             \n"//v10: row1 left 8 data
            "ext    v11.16b, v2.16b, v3.16b, #2               \n"//v11: row1 right 8 data
            // r1
            "sub    %[din_ptr2], %[din_ptr2], #2              \n"
            "sub    %[din_ptr3], %[din_ptr3], #2              \n"
            "fmla   v19.8h, %[ww1].8h, v2.8h                  \n"//v19: output1 mla result
            "fmla   v18.8h, %[ww4].8h, v2.8h                  \n"
            "ext    v8.16b, v21.16b, v4.16b, #14              \n"//v8: row2 left 8 data
            "ext    v9.16b, v4.16b, v5.16b, #2                \n"//v9: row2 right 8 data
            "fmla   v19.8h, %[ww0].8h, v10.8h                 \n"
            "fmla   v18.8h, %[ww3].8h, v10.8h                 \n"
            "ld1    {v0.8h}, [%[din_ptr0]], #16               \n"//prefetch load row0
            "fmla   v19.8h, %[ww2].8h, v11.8h                 \n"
            "ld1    {v2.8h}, [%[din_ptr1]], #16               \n"//prefetch load row1
            "fmla   v18.8h, %[ww5].8h, v11.8h                 \n"
            // r2
            "ld1    {v1.8h}, [%[din_ptr0]]                    \n"
            "fmla   v19.8h, %[ww4].8h, v4.8h                  \n"
            "ld1    {v3.8h}, [%[din_ptr1]]                    \n"
            "fmla   v18.8h, %[ww7].8h, v4.8h                  \n"
            "ext    v10.16b, v21.16b, v6.16b, #14             \n"//v10: row3 left 8 data
            "ext    v11.16b, v6.16b, v7.16b, #2               \n"//v11: row3 right 8 data
            "fmla   v19.8h, %[ww3].8h, v8.8h                  \n"
            "fmla   v18.8h, %[ww6].8h, v8.8h                  \n"
            "ld1    {v4.8h}, [%[din_ptr2]], #16               \n"//prefetch load row0
            "fmla   v19.8h, %[ww5].8h, v9.8h                  \n"
            "fmla   v18.8h, %[ww8].8h, v9.8h                  \n"
            // r3
            "fmla   v19.8h, %[ww7].8h, v6.8h                  \n"
            "fmla   v19.8h, %[ww6].8h, v10.8h                 \n"
            "ld1    {v6.8h}, [%[din_ptr3]], #16               \n"//prefetch load row1
            "fmla   v19.8h, %[ww8].8h, v11.8h                 \n"
            "ld1    {v5.8h}, [%[din_ptr2]]                    \n"
            "ld1    {v7.8h}, [%[din_ptr3]]                    \n"
            "st1    {v18.8h}, [%[ptr_out0]], #16              \n"
            "st1    {v19.8h}, [%[ptr_out1]], #16              \n"
            "cmp    %w[cnt], #1                                \n"
            "blt    3f                                        \n"
          // mid (for loop)
            "1:                                               \n"
            "ext    v8.16b, v0.16b, v1.16b, #2                \n"
            "ext    v9.16b, v0.16b, v1.16b, #4                \n"
            "ld1    {v18.8h}, [%[bias_val]]                   \n"
            // r0
            "fmla   v18.8h, %[ww0].8h,  v0.8h                 \n"
            "ld1    {v19.8h}, [%[bias_val]]                   \n"
            "ext    v10.16b, v2.16b, v3.16b, #2               \n"
            "ext    v11.16b, v2.16b, v3.16b, #4               \n"
            "fmla   v18.8h, %[ww1].8h,  v8.8h                 \n"
            "fmla   v18.8h, %[ww2].8h,  v9.8h                 \n"
            // r1
            "ext    v8.16b, v4.16b, v5.16b, #2                \n"// row2
            "ext    v9.16b, v4.16b, v5.16b, #4                \n"// v4, v8, v9
            "fmla   v19.8h, %[ww0].8h,  v2.8h                 \n"
            "fmla   v18.8h, %[ww3].8h,  v2.8h                 \n"
            "ld1    {v0.8h}, [%[din_ptr0]], #16               \n"
            "fmla   v19.8h, %[ww1].8h,  v10.8h                \n"
            "ld1    {v2.8h}, [%[din_ptr1]], #16               \n"
            "fmla   v18.8h, %[ww4].8h,  v10.8h                \n"
            "ld1    {v1.8h}, [%[din_ptr0]]                    \n"
            "fmla   v19.8h, %[ww2].8h,  v11.8h                \n"
            "ld1    {v3.8h}, [%[din_ptr1]]                    \n"
            "fmla   v18.8h, %[ww5].8h,  v11.8h                \n"
            // r2
            "ext    v10.16b, v6.16b, v7.16b, #2               \n"
            "ext    v11.16b, v6.16b, v7.16b, #4               \n"
            "fmla   v19.8h, %[ww3].8h,  v4.8h                 \n"
            "fmla   v18.8h, %[ww6].8h,  v4.8h                 \n"
            "fmla   v19.8h, %[ww4].8h,  v8.8h                 \n"
            "fmla   v18.8h, %[ww7].8h,  v8.8h                 \n"
            "fmla   v19.8h, %[ww5].8h,  v9.8h                 \n"
            "ld1    {v4.8h}, [%[din_ptr2]], #16               \n"
            "fmla   v18.8h, %[ww8].8h,  v9.8h                 \n"
            "ld1    {v5.8h}, [%[din_ptr2]]                    \n"
            // r3
            "fmla   v19.8h, %[ww6].8h,  v6.8h                 \n"
            "ld1    {v6.8h}, [%[din_ptr3]], #16               \n"
            "fmla   v19.8h, %[ww7].8h,  v10.8h                \n"
            "ld1    {v7.8h}, [%[din_ptr3]]                    \n"
            "fmla   v19.8h, %[ww8].8h,  v11.8h                \n"
            "st1    {v18.8h}, [%[ptr_out0]], #16              \n"
            "subs %w[cnt], %w[cnt], #1                          \n"
            "st1    {v19.8h}, [%[ptr_out1]], #16              \n"
            "bne 1b                                           \n"
          // right
            "3:                                               \n"
            "ld1    {v20.8h}, [%[vmask]], #16                 \n"
            "ld1    {v22.8h}, [%[vmask]]                      \n"
            "ld1    {v18.8h}, [%[bias_val]]                   \n"
            "ld1    {v19.8h}, [%[bias_val]]                   \n"
            "bif    v0.16b, v21.16b, v20.16b                  \n"
            "bif    v1.16b, v21.16b, v22.16b                  \n"
            "bif    v2.16b, v21.16b, v20.16b                  \n"
            "bif    v3.16b, v21.16b, v22.16b                  \n"
            "ext    v8.16b, v0.16b, v1.16b, #2                \n"
            "ext    v9.16b, v0.16b, v1.16b, #4                \n"
            //r0
            "fmla   v18.8h,  %[ww0].8h,  v0.8h                \n"
            "ext    v10.16b, v2.16b, v3.16b, #2               \n"
            "ext    v11.16b, v2.16b, v3.16b, #4               \n"
            "bif    v4.16b, v21.16b, v20.16b                  \n"
            "bif    v5.16b, v21.16b, v22.16b                  \n"
            "fmla   v18.8h,  %[ww1].8h,  v8.8h                \n"
            "bif    v6.16b, v21.16b, v20.16b                  \n"
            "bif    v7.16b, v21.16b, v22.16b                  \n"
            "fmla   v18.8h,  %[ww2].8h,  v9.8h                \n"
            //r1
            "ext    v8.16b, v4.16b, v5.16b, #2                \n"
            "ext    v9.16b, v4.16b, v5.16b, #4                \n"
            "fmla   v19.8h,  %[ww0].8h,  v2.8h                \n"
            "fmla   v18.8h,  %[ww3].8h,  v2.8h                \n"
            "ld1    {v20.16b}, [%[rmask]], #16                \n"
            "fmla   v19.8h,  %[ww1].8h,  v10.8h               \n"
            "ld1    {v0.8h}, [%[ptr_out0]]                    \n"
            "fmla   v18.8h,  %[ww4].8h,  v10.8h               \n"
            "ld1    {v2.8h}, [%[ptr_out1]]                    \n"
            "fmla   v19.8h,  %[ww2].8h,  v11.8h               \n"
            "fmla   v18.8h,  %[ww5].8h,  v11.8h               \n"
            //r2
            "ext    v10.16b, v6.16b, v7.16b, #2               \n"
            "ext    v11.16b, v6.16b, v7.16b, #4               \n"
            "fmla   v19.8h,  %[ww3].8h,  v4.8h                \n"
            "fmla   v18.8h,  %[ww6].8h,  v4.8h                \n"
            "fmla   v19.8h,  %[ww4].8h,  v8.8h                \n"
            "fmla   v18.8h,  %[ww7].8h,  v8.8h                \n"
            "fmla   v19.8h,  %[ww5].8h,  v9.8h                \n"
            "fmla   v18.8h,  %[ww8].8h,  v9.8h                \n"
            //r3
            "fmla   v19.8h,  %[ww6].8h,  v6.8h                \n"
            "fmla   v19.8h,  %[ww7].8h,  v10.8h               \n"
            "fmla   v19.8h,  %[ww8].8h,  v11.8h               \n"
            "bif    v18.16b, v0.16b, v20.16b                  \n"
            "bif    v19.16b, v2.16b, v20.16b                  \n"
            "st1    {v18.8h}, [%[ptr_out0]], #16             \n"
            "st1    {v19.8h}, [%[ptr_out1]], #16             \n"
            : [cnt]"+r"(cnt), [din_ptr0]"+r"(din_ptr0), [din_ptr1] "+r"(din_ptr1), [din_ptr2] "+r"(din_ptr2), \
              [din_ptr3] "+r"(din_ptr3), [ptr_out0] "+r"(doutr0), [ptr_out1] "+r"(doutr1), \
              [vmask] "+r" (val_mask), [rmask] "+r" (rst_mask)
            : [ww0]"w"(wr00), [ww1]"w"(wr01), [ww2]"w"(wr02), [ww3]"w"(wr10), \
              [ww4]"w"(wr11), [ww5]"w"(wr12), [ww6]"w"(wr20), [ww7]"w"(wr21), [ww8] "w" (wr22), \
              [bias_val] "r"(v_bias)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",\
              "v8", "v9", "v10", "v11", "v18", \
              "v19", "v20", "v21", "v22"
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
