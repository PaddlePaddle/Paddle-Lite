/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "operators/math/slidingwindow_utils.h"

namespace paddle_mobile {
namespace operators {
namespace math {

void slidingwindow_fill_bias(float* dout, const float* bias, int ch_num,
                             int ch_size) {
  for (int j = 0; j < ch_num; j++) {
    float32x4_t vb = vdupq_n_f32(bias[j]);
    int i = 0;
    for (; i < ch_size - 3; i += 4) {
      vst1q_f32(dout + i, vb);
    }
    for (; i < ch_size; i++) {
      dout[i] = bias[j];
    }
    dout += ch_size;
  }
}

/* write result in outputs
 * input din: [n, c, h, w], output dout: [n, c, h, w]
 */
void slidingwindow_writeout_c1_fp32(const float* din, float* dout, int cs,
                                    int ce, int hs, int he, int ws, int we,
                                    int channel, int height, int width,
                                    bool flag_relu, float* trash_ptr) {
  if (cs > channel) {
    return;
  }

  const int c1 = 1;
  const int w4 = 4;

  int size_c_out = width * height;

  float* doutc0r0 = dout + cs * size_c_out + hs * width + ws;

  const float* ptr_din = din;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int w_round = we - ws;
  int cnt = (width - ws) / w4;

  for (int i = 0; i < size_h; i++) {
    int size_w = i * width;
    float* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
    const float* din_hei_ptr = ptr_din + i * w_round * c1;
    if (cnt > 0) {
      int cnt_loop = cnt;
      if (flag_relu) {
#ifdef __aarch64__
        asm volatile(
            "ldr q0, [%[ptr_din]], #16      \n" /* load data, c0r0, c0r1, c0r2,
                                                   c0r3 */
            "movi v20.4s, #0                \n" /* for relu */
            "1:                             \n" /* main loop */
            "fmax   v1.4s, v0.4s, v20.4s    \n" /* relu */
            "ldr q0, [%[ptr_din]], #16      \n" /* load data, c0r0, c0r1, c0r2,
                                                   c0r3 */
            "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1 */
            "str    q1, [%[doutc0r0]], #16  \n" /* store c0r0 */
            "bne    1b                      \n" /* jump to main loop */
            : [doutc0r0] "+r"(doutc0_ptr), [cnt] "+r"(cnt_loop),
              [ptr_din] "+r"(din_hei_ptr)
            :
            : "v0", "v1", "v20");
#else
        asm volatile(
            "vld1.32 {d0-d1}, [%[ptr_din]]!         @ load data, c0r0, c1r0, "
            "c0r1, c1r1, , c0r2, c1r2, c0r3, c1r3\n"
            "vmov.u32 q15, #0                       @ dump zero\n"
            "1:                                     @ main loop\n"

            "vmax.f32   q1, q0, q15                 @ relu\n"
            "vld1.32 {d0-d1}, [%[ptr_din]]!         @ load data \n"

            "vst1.32  {d2-d3}, [%[doutc0r0]]!       @ store result, add "
            "pointer\n"

            "subs   %[cnt], %[cnt], #1              @ loop count - 1\n"

            "bne    1b                              @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [ptr_din] "+r"(din_hei_ptr),
              [cnt] "+r"(cnt_loop)
            :
            : "q0", "q1", "q15");
#endif
      } else {
#ifdef __aarch64__
        asm volatile(
            "ldr q0, [%[ptr_din]], #16      \n" /* load data, c0r0, c0r1, c0r2,
                                                   c0r3 */
            "1:                             \n" /* main loop */
            "str    q0, [%[doutc0r0]], #16  \n" /* store c2r0 */
            "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1 */
            "ldr q0, [%[ptr_din]], #16      \n" /* load data, c0r0, c0r1, c0r2,
                                                   c0r3 */
            "bne    1b                      \n" /* jump to main loop */

            : [doutc0r0] "+r"(doutc0_ptr), [cnt] "+r"(cnt_loop),
              [ptr_din] "+r"(din_hei_ptr)
            :
            : "v0");
#else
        asm volatile(
            "vld1.32 {d0-d1}, [%[ptr_din]]!         @ load data, c0r0, c0r1, "
            "c0r2, c0r3\n"
            "1:                                     @ main loop\n"
            "vst1.32  {d0-d1}, [%[doutc0r0]]!       @ store result, add "
            "pointer\n"
            "subs   %[cnt], %[cnt], #1              @ loop count - 1\n"
            "vld1.32 {d0-d1}, [%[ptr_din]]!         @ load data \n"
            "bne    1b                              @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [ptr_din] "+r"(din_hei_ptr),
              [cnt] "+r"(cnt_loop)
            :
            : "q0");
#endif
      }
    }
    if (we > width) {
      int offset = i * w_round * c1 + c1 * w4 * cnt;
      din_hei_ptr = ptr_din + offset;
      int j = we - w4;
      if (flag_relu) {
        for (; j < width; ++j) {
          *(doutc0_ptr++) = std::max(din_hei_ptr[0], 0.f);
          din_hei_ptr++;
        }
      } else {
        for (; j < width; ++j) {
          *(doutc0_ptr++) = *(din_hei_ptr++);
        }
      }
    }
  }
}

/* write result in outputs
 * input din: [n, c / 4, h, w * 4], output dout: [n, c, h, w]
 */
void slidingwindow_writeout_c4_fp32(const float* din, float* dout, int cs,
                                    int ce, int hs, int he, int ws, int we,
                                    int channel, int height, int width,
                                    bool flag_relu, float* trash_ptr) {
  const int c4 = 4;
  const int w4 = 4;
  const int w_round = we - ws;
  const int ch_n = ce - cs;
  int size_c_out = width * height;

  float* doutc0r0 = dout + cs * size_c_out + hs * width + ws;
  float* doutc1r0 = doutc0r0 + size_c_out;
  float* doutc2r0 = doutc1r0 + size_c_out;
  float* doutc3r0 = doutc2r0 + size_c_out;

  const float* ptr_din = din;

  int size_h = (he > height ? height : he) - hs;  // size_h == hei_n

  int cnt = (width - ws) / w4;

  for (int i = 0; i < size_h; i++) {
    int size_w = i * width;
    float* doutc0_ptr = doutc0r0 + size_w;  // doutc0r0 + width;
    float* doutc1_ptr = doutc1r0 + size_w;
    float* doutc2_ptr = doutc2r0 + size_w;
    float* doutc3_ptr = doutc3r0 + size_w;
    if (ce > channel) {
      switch (ce - channel) {
        case 3:
          doutc1_ptr = trash_ptr;
        case 2:
          doutc2_ptr = trash_ptr;
        case 1:
          doutc3_ptr = trash_ptr;
        default:
          break;
      }
    }
    const float* din_hei_ptr = ptr_din + i * w_round * ch_n;
    if (cnt > 0) {
      int cnt_loop = cnt;
      if (flag_relu) {
#ifdef __aarch64__
        asm volatile(
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "movi v20.4s, #0                \n" /* for relu */
            "1:                             \n" /* main loop */
            "trn1   v8.4s, v0.4s, v1.4s     \n" /* trans q0, q1 */
            "trn2   v9.4s, v0.4s, v1.4s     \n" /* trans q0, q1 */
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "trn1   v10.4s, v2.4s, v3.4s    \n" /* trans q2, q3 */
            "trn2   v11.4s, v2.4s, v3.4s    \n" /* trans q2, q3 */
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "trn1   v16.2d, v8.2d, v10.2d   \n" /* trans q8, q10 */
            "trn2   v17.2d, v8.2d, v10.2d   \n" /* trans q8, q10 */
            "trn1   v18.2d, v9.2d, v11.2d   \n" /* trans q9, q11 */
            "trn2   v19.2d, v9.2d, v11.2d   \n" /* trans q9, q11 */
            "fmax   v16.4s, v16.4s, v20.4s  \n" /* relu */
            "fmax   v17.4s, v17.4s, v20.4s  \n" /* relu */
            "fmax   v18.4s, v18.4s, v20.4s  \n" /* relu */
            "fmax   v19.4s, v19.4s, v20.4s  \n" /* relu */
            "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0 */
            "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0 */
            "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0 */
            "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0 */

            "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1 */
            "bne    1b                      \n" /* jump to main loop */

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v16", "v17", "v18", "v19", "v20");
#else
        asm volatile(
            "vld1.32 {d0-d3}, [%[ptr_din]]!       @ load data \n"
            "vld1.32 {d4-d7}, [%[ptr_din]]!       @ load data \n"
            "vmov.u32 q15, #0                     @ dump zero \n"
            "1:                                   @ main loop \n"
            "vtrn.32 q0, q1                       @ trans data:c00c01c20c21 "
            "\n"
            "vtrn.32 q2, q3                       @ trans data:c02c03c22c23 "
            "\n"

            "vswp   d1, d4                        @ swap data\n"
            "vswp   d3, d6                        @ swap data\n"

            "vmax.f32   q0, q0, q15               @ relu\n"
            "vmax.f32   q1, q1, q15               @ relu\n"
            "vmax.f32   q2, q2, q15               @ relu\n"
            "vmax.f32   q3, q3, q15               @ relu\n"

            "vst1.32  {d0-d1}, [%[doutc0r0]]!     @ store result, add pointer\n"
            "vst1.32  {d2-d3}, [%[doutc1r0]]!     @ store result, add pointer\n"
            "vst1.32  {d4-d5}, [%[doutc2r0]]!     @ store result, add pointer\n"
            "vst1.32  {d6-d7}, [%[doutc3r0]]!     @ store result, add pointer\n"

            "subs   %[cnt], %[cnt], #1            @ loop count - 1\n"

            "vld1.32 {d0-d3}, [%[ptr_din]]!       @ load data \n"
            "vld1.32 {d4-d7}, [%[ptr_din]]!       @ load data \n"

            "bne    1b                            @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
            :
            : "q0", "q1", "q2", "q3", "q15");
#endif
      } else {
#ifdef __aarch64__
        asm volatile(
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "1:                             \n" /* main loop */
            "trn1   v8.4s, v0.4s, v1.4s     \n" /* trans q0, q1 */
            "trn2   v9.4s, v0.4s, v1.4s     \n" /* trans q0, q1 */
            "ldp q0, q1, [%[ptr_din]], #32  \n" /* load r00, r01 to q0, q1 */
            "trn1   v10.4s, v2.4s, v3.4s    \n" /* trans q2, q3 */
            "trn2   v11.4s, v2.4s, v3.4s    \n" /* trans q2, q3 */
            "ldp q2, q3, [%[ptr_din]], #32  \n" /* load r02, r03 to q2, q3 */
            "trn1   v16.2d, v8.2d, v10.2d   \n" /* trans q8, q10 */
            "trn2   v17.2d, v8.2d, v10.2d   \n" /* trans q8, q10 */
            "trn1   v18.2d, v9.2d, v11.2d   \n" /* trans q9, q11 */
            "trn2   v19.2d, v9.2d, v11.2d   \n" /* trans q9, q11 */
            "str    q16, [%[doutc0r0]], #16 \n" /* store c0r0 */
            "str    q17, [%[doutc2r0]], #16 \n" /* store c2r0 */
            "str    q18, [%[doutc1r0]], #16 \n" /* store c1r0 */
            "str    q19, [%[doutc3r0]], #16 \n" /* store c3r0 */

            "subs   %w[cnt], %w[cnt], #1    \n" /* loop count -1 */
            "bne    1b                      \n" /* jump to main loop */

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [cnt] "+r"(cnt_loop), [ptr_din] "+r"(din_hei_ptr)
            :
            : "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v16", "v17",
              "v18", "v19");
#else
        asm volatile(
            "vld1.32 {d0-d3}, [%[ptr_din]]!       @ load data \n"
            "vld1.32 {d4-d7}, [%[ptr_din]]!       @ load data \n"
            "1:                                   @ main loop \n"
            "vtrn.32 q0, q1                       @ trans data:c00c01c20c21 "
            "\n"
            "vtrn.32 q2, q3                       @ trans data:c02c03c22c23 "
            "\n"

            "vswp   d1, d4                        @ swap data\n"
            "vswp   d3, d6                        @ swap data\n"

            "vst1.32  {d0-d1}, [%[doutc0r0]]!     @ store result, add pointer\n"
            "vst1.32  {d2-d3}, [%[doutc1r0]]!     @ store result, add pointer\n"
            "vst1.32  {d4-d5}, [%[doutc2r0]]!     @ store result, add pointer\n"
            "vst1.32  {d6-d7}, [%[doutc3r0]]!     @ store result, add pointer\n"

            "subs   %[cnt], %[cnt], #1            @ loop count - 1\n"

            "vld1.32 {d0-d3}, [%[ptr_din]]!       @ load data \n"
            "vld1.32 {d4-d7}, [%[ptr_din]]!       @ load data \n"

            "bne    1b                            @ jump to main loop\n"

            : [doutc0r0] "+r"(doutc0_ptr), [doutc1r0] "+r"(doutc1_ptr),
              [doutc2r0] "+r"(doutc2_ptr), [doutc3r0] "+r"(doutc3_ptr),
              [ptr_din] "+r"(din_hei_ptr), [cnt] "+r"(cnt_loop)
            :
            : "q0", "q1", "q2", "q3");
#endif
      }
    }
    if (we > width) {
      int offset = i * w_round * c4 + c4 * w4 * cnt;
      din_hei_ptr = ptr_din + offset;
      int j = we - w4;
      if (flag_relu) {
        for (; j < width; ++j) {
          *(doutc0_ptr++) = std::max(din_hei_ptr[0], 0.f);
          *(doutc1_ptr++) = std::max(din_hei_ptr[1], 0.f);
          *(doutc2_ptr++) = std::max(din_hei_ptr[2], 0.f);
          *(doutc3_ptr++) = std::max(din_hei_ptr[3], 0.f);
          din_hei_ptr += w4;
        }
      } else {
        for (; j < width; ++j) {
          *(doutc0_ptr++) = din_hei_ptr[0];
          *(doutc1_ptr++) = din_hei_ptr[1];
          *(doutc2_ptr++) = din_hei_ptr[2];
          *(doutc3_ptr++) = din_hei_ptr[3];
          din_hei_ptr += w4;
        }
      }
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
