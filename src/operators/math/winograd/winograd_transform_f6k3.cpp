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

// Inspired by https://arxiv.org/abs/1509.09308 and
// https://github.com/andravin/wincnn
// Refered from nnpack and ncnn project

#include "operators/math/pad.h"
#include "operators/math/winograd/winograd_transform.h"

namespace paddle_mobile {
namespace operators {
namespace math {

template <>
void winograd_transform_weight<8, 3>(const framework::Tensor &weight,
                                     framework::Tensor *output) {
  /*
   * w0 = g0
   * w1 = ((g0 + g2) + g1) * (-2.0 / 9)
   * w2 = ((g0 + g2) - g1) * (-2.0 / 9)
   * w3 = ((g0 + 4 * g2) + 2 * g1) * (1.0 / 90)
   * w4 = ((g0 + 4 * g2) - 2 * g1) * (1.0 / 90)
   * w5 = ((g2 + 4 * g0) + 2 * g1) * (1.0 / 180)
   * w6 = ((g2 + 4 * g0) - 2 * g1) * (1.0 / 180)
   * w7 = g2
   */
  // weight shape is [out_channel, in_channel, kernel_h, kernel_w]
  int out_channel = weight.dims()[0];
  int in_channel = weight.dims()[1];
  // reshape and alloc transformed weight
  framework::DDim transformed_shape = framework::make_ddim(
      std::vector<int>{(out_channel + 3) / 4, 64, in_channel, 4});
  float *trans_outptr = output->mutable_data<float>(transformed_shape);
  memset(trans_outptr, 0, output->numel() * sizeof(float));

  const float transform_matrix[8] = {2.f, -2.f / 9, 1.f / 90, 1.f / 180};
  const float *inptr = weight.data<float>();
  int remain_start = out_channel & 0xFFFC;
#ifdef __aarch64__
  remain_start = 0;
#else
  for (int oc = 0; oc < out_channel; oc += 4) {
    float gw[96];                                       // gw[3][8][4]
    const float *inptr0 = inptr + oc * in_channel * 9;  //
    const float *inptr1 = inptr + (oc + 1) * in_channel * 9;
    const float *inptr2 = inptr + (oc + 2) * in_channel * 9;
    const float *inptr3 = inptr + (oc + 3) * in_channel * 9;
    // oc * 64 * in_channel
    float *outptr = trans_outptr + ((oc * in_channel) << 6);
    for (int ic = 0; ic < in_channel; ++ic) {
      float *gw_ptr = gw;
      asm volatile(
          "vld1.32    {d0-d1}, [%[tm_ptr]]          \n"

          "mov        r0, #6                        \n"
          "vld1.32    {d2-d5}, [%[inptr0]], r0      \n"
          "vld1.32    {d6-d9}, [%[inptr1]], r0      \n"
          "vld1.32    {d10-d13}, [%[inptr2]], r0    \n"
          "vld1.32    {d14-d17}, [%[inptr3]], r0    \n"
          "vtrn.32    q1, q3                        \n"
          "vtrn.32    q2, q4                        \n"
          "vtrn.32    q5, q7                        \n"
          "vtrn.32    q6, q8                        \n"
          "vswp.32    d3, d10                       \n"
          "vswp.32    d7, d14                       \n"
          "vswp.32    d5, d12                       \n"
          "vswp.32    d9, d16                       \n"

          // q1: g0, q3: g1, q5: g2
          "vst1.32    {d2-d3}, [%[gw_ptr]]!         \n"
          "vadd.f32   q9, q1, q5                    \n"
          "vadd.f32   q10, q9, q3                   \n"
          "vsub.f32   q11, q9, q3                   \n"
          "vmul.f32   q10, q10, d0[1]               \n"
          "vst1.32    {d20-d21}, [%[gw_ptr]]!       \n"
          "vmul.f32   q11, q11, d0[1]               \n"
          "vst1.32    {d22-d23}, [%[gw_ptr]]!       \n"

          "vmul.f32   q9, q1, d0[0]                 \n"
          "vmul.f32   q9, q9, d0[0]                 \n"  // 4 * g0
          "vmul.f32   q10, q3, d0[0]                \n"  // 2 * g1
          "vmul.f32   q11, q5, d0[0]                \n"
          "vmul.f32   q11, q11, d0[0]               \n"  // 4 * g2

          "vadd.f32   q12, q1, q11                  \n"
          "vadd.f32   q13, q12, q10                 \n"
          "vmul.f32   q13, q13, d1[0]               \n"
          "vst1.32    {d26-d27}, [%[gw_ptr]]!       \n"
          "vsub.f32   q13, q12, q10                 \n"
          "vmul.f32   q13, q13, d1[0]               \n"
          "vst1.32    {d26-d27}, [%[gw_ptr]]!       \n"

          "vadd.f32   q12, q5, q9                   \n"
          "vadd.f32   q13, q12, q10                 \n"
          "vmul.f32   q13, q13, d1[1]               \n"
          "vst1.32    {d26-d27}, [%[gw_ptr]]!       \n"
          "vsub.f32   q13, q12, q10                 \n"
          "vmul.f32   q13, q13, d1[1]               \n"
          "vst1.32    {d26-d27}, [%[gw_ptr]]!       \n"

          "vst1.32    {d10-d11}, [%[gw_ptr]]!       \n"

          // q7: g0, q2: g1, q4: g2
          "vst1.32    {d14-d15}, [%[gw_ptr]]!       \n"
          "vadd.f32   q9, q7, q4                    \n"
          "vadd.f32   q10, q9, q2                   \n"
          "vsub.f32   q11, q9, q2                   \n"
          "vmul.f32   q10, q10, d0[1]               \n"
          "vst1.32    {d20-d21}, [%[gw_ptr]]!       \n"
          "vmul.f32   q11, q11, d0[1]               \n"
          "vst1.32    {d22-d23}, [%[gw_ptr]]!       \n"

          "vmul.f32   q9, q7, d0[0]                 \n"
          "vmul.f32   q9, q9, d0[0]                 \n"  // 4 * g0
          "vmul.f32   q10, q2, d0[0]                \n"  // 2 * g1
          "vmul.f32   q11, q4, d0[0]                \n"
          "vmul.f32   q11, q11, d0[0]               \n"  // 4 * g2

          "vadd.f32   q12, q7, q11                  \n"
          "vadd.f32   q13, q12, q10                 \n"
          "vmul.f32   q13, q13, d1[0]               \n"
          "vst1.32    {d26-d27}, [%[gw_ptr]]!       \n"
          "vsub.f32   q13, q12, q10                 \n"
          "vmul.f32   q13, q13, d1[0]               \n"
          "vst1.32    {d26-d27}, [%[gw_ptr]]!       \n"

          "vadd.f32   q12, q4, q9                   \n"
          "vadd.f32   q13, q12, q10                 \n"
          "vmul.f32   q13, q13, d1[1]               \n"
          "vst1.32    {d26-d27}, [%[gw_ptr]]!       \n"
          "vsub.f32   q13, q12, q10                 \n"
          "vmul.f32   q13, q13, d1[1]               \n"
          "vst1.32    {d26-d27}, [%[gw_ptr]]!       \n"

          "vst1.32    {d8-d9}, [%[gw_ptr]]!         \n"

          "mov        r0, #3                        \n"
          "vld1.32    {d2-d5}, [%[inptr0]], r0      \n"
          "vld1.32    {d6-d9}, [%[inptr1]], r0      \n"
          "vld1.32    {d10-d13}, [%[inptr2]], r0    \n"
          "vld1.32    {d14-d17}, [%[inptr3]], r0    \n"
          "vtrn.32    q1, q3                        \n"
          "vtrn.32    q2, q4                        \n"
          "vtrn.32    q5, q7                        \n"
          "vtrn.32    q6, q8                        \n"
          "vswp.32    d3, d10                       \n"
          "vswp.32    d7, d14                       \n"
          "vswp.32    d5, d12                       \n"
          "vswp.32    d9, d16                       \n"

          // q1: g0, q3: g1, q5: g2
          "vst1.32    {d2-d3}, [%[gw_ptr]]!         \n"
          "vadd.f32   q9, q1, q5                    \n"
          "vadd.f32   q10, q9, q3                   \n"
          "vsub.f32   q11, q9, q3                   \n"
          "vmul.f32   q10, q10, d0[1]               \n"
          "vst1.32    {d20-d21}, [%[gw_ptr]]!       \n"
          "vmul.f32   q11, q11, d0[1]               \n"
          "vst1.32    {d22-d23}, [%[gw_ptr]]!       \n"

          "vmul.f32   q9, q1, d0[0]                 \n"
          "vmul.f32   q9, q9, d0[0]                 \n"  // 4 * g0
          "vmul.f32   q10, q3, d0[0]                \n"  // 2 * g1
          "vmul.f32   q11, q5, d0[0]                \n"
          "vmul.f32   q11, q11, d0[0]               \n"  // 4 * g2

          "vadd.f32   q12, q1, q11                  \n"
          "vadd.f32   q13, q12, q10                 \n"
          "vmul.f32   q13, q13, d1[0]               \n"
          "vst1.32    {d26-d27}, [%[gw_ptr]]!       \n"
          "vsub.f32   q13, q12, q10                 \n"
          "vmul.f32   q13, q13, d1[0]               \n"
          "vst1.32    {d26-d27}, [%[gw_ptr]]!       \n"

          "vadd.f32   q12, q5, q9                   \n"
          "vadd.f32   q13, q12, q10                 \n"
          "vmul.f32   q13, q13, d1[1]               \n"
          "vst1.32    {d26-d27}, [%[gw_ptr]]!       \n"
          "vsub.f32   q13, q12, q10                 \n"
          "vmul.f32   q13, q13, d1[1]               \n"
          "vst1.32    {d26-d27}, [%[gw_ptr]]!       \n"

          "vst1.32    {d10-d11}, [%[gw_ptr]]!       \n"
          : [gw_ptr] "+r"(gw_ptr), [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
            [inptr2] "+r"(inptr0), [inptr3] "+r"(inptr3)
          : [tm_ptr] "r"((float *)transform_matrix)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "q8", "q9", "q10", "q11", "q12", "q13", "r0");

      float *gw_ptr0 = gw;
      float *gw_ptr1 = gw + 32;
      float *gw_ptr2 = gw + 64;
      float *outptr0 = outptr + (ic << 2);  // ic * 4
      int steps = (in_channel << 2);        // in_channel * 4
      asm volatile(
          "vld1.32    {d0-d1}, [%[tm_ptr]]               \n"
          "mov        r0, #8                             \n"

          "loop_8_%=:                                    \n"
          "vld1.32    {d2-d3}, [%[gw_ptr0]]!             \n"
          "vld1.32    {d4-d5}, [%[gw_ptr1]]!             \n"
          "vld1.32    {d6-d7}, [%[gw_ptr2]]!             \n"

          // q1: g0, q2: g1, q3: g2
          "vst1.32    {d2-d3}, [%[outptr0]], %[steps]    \n"
          "vadd.f32   q9, q1, q3                         \n"
          "vadd.f32   q10, q9, q2                        \n"
          "vsub.f32   q11, q9, q2                        \n"
          "vmul.f32   q10, q10, d0[1]                    \n"
          "vst1.32    {d20-d21}, [%[outptr0]], %[steps]  \n"
          "vmul.f32   q11, q11, d0[1]                    \n"
          "vst1.32    {d22-d23}, [%[outptr0]], %[steps]  \n"

          "vmul.f32   q9, q1, d0[0]                      \n"
          "vmul.f32   q9, q9, d0[0]                      \n"  // 4 * g0
          "vmul.f32   q10, q2, d0[0]                     \n"  // 2 * g1
          "vmul.f32   q11, q3, d0[0]                     \n"
          "vmul.f32   q11, q11, d0[0]                    \n"  // 4 * g2

          "vadd.f32   q12, q1, q11                       \n"
          "vadd.f32   q13, q12, q10                      \n"
          "vmul.f32   q13, q13, d1[0]                    \n"
          "vst1.32    {d26-d27}, [%[outptr0]], %[steps]  \n"
          "vsub.f32   q13, q12, q10                      \n"
          "vmul.f32   q13, q13, d1[0]                    \n"
          "vst1.32    {d26-d27}, [%[outptr0]], %[steps]  \n"

          "vadd.f32   q12, q2, q9                        \n"
          "vadd.f32   q13, q12, q10                      \n"
          "vmul.f32   q13, q13, d1[1]                    \n"
          "vst1.32    {d26-d27}, [%[outptr0]], %[steps]  \n"
          "vsub.f32   q13, q12, q10                      \n"
          "vmul.f32   q13, q13, d1[1]                    \n"
          "vst1.32    {d26-d27}, [%[outptr0]], %[steps]  \n"

          "vst1.32    {d6-d7}, [%[outptr0]], %[steps]    \n"

          "subs       r0, #1                             \n"
          "bne        loop_8_%=                          \n"
          : [outptr0] "+r"(outptr0), [gw_ptr0] "+r"(gw_ptr0),
            [gw_ptr1] "+r"(gw_ptr1), [gw_ptr2] "+r"(gw_ptr2)
          : [tm_ptr] "r"((float *)transform_matrix), [steps] "r"(steps)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q9", "q10", "q11", "q12",
            "q13", "r0");
    }
  }
#endif  // __aarch64__

  // remain output channel
  for (int oc = remain_start; oc < out_channel; ++oc) {
    float gw[8][3];                                     // gw[3][8]
    const float *inptr0 = inptr + oc * in_channel * 9;  //
    // oc * 64 * in_channel + oc % 4
    int offset = ((oc * in_channel) << 6) + oc & 0x3;
    int steps = (in_channel << 2);  // in_channel * 4
    float *outptr = trans_outptr + offset;
    for (int ic = 0; ic < in_channel; ++ic) {
      for (int i = 0; i < 3; ++i) {
        float g0 = inptr0[i];
        float g1 = inptr0[3 + i];
        float g2 = inptr0[6 + i];
        float d0 = g0 + g2;
        float d1 = g0 + 4 * g2;
        float d2 = g2 + 4 * g0;
        float d3 = 2 * g1;
        gw[0][i] = g0;
        gw[1][i] = -2.f / 9 * (d0 + g1);  // -2.f/9 * (g0 + g1 + g2)
        gw[2][i] = -2.f / 9 * (d0 - g1);  // -2.f/9 * (g0 - g1 + g2)
        gw[3][i] = 1.f / 90 * (d1 + d3);  // 1.f/90 * (g0 + 2 * g1 + 4 * g2)
        gw[4][i] = 1.f / 90 * (d1 - d3);  // 1.f/90 * (g0 - 2 * g1 + 4 * g2)
        gw[5][i] = 8.f / 45 * (d2 + d3);  // 8.f/45 * (4 * g0 + 2 * g1 + g2)
        gw[6][i] = 8.f / 45 * (d2 - d3);  // 8.f/45 * (4 * g0 - 2 * g1 + g2)
        gw[7][i] = g2;
      }
      inptr0 += 9;
      outptr += ic * 4;
      for (int i = 0; i < 8; ++i) {
        float g0 = gw[i][0];
        float g1 = gw[i][1];
        float g2 = gw[i][2];
        float d0 = g0 + g2;
        float d1 = g0 + 4 * g2;
        float d2 = g2 + 4 * g0;
        float d3 = 2 * g1;
        int offset = i * 8 * steps;
        outptr[offset] = gw[i][0];
        outptr[offset + 1 * steps] = -2.f / 9 * (d0 + g1);
        outptr[offset + 2 * steps] = -2.f / 9 * (d0 - g1);
        outptr[offset + 3 * steps] = 1.f / 90 * (d1 + d3);
        outptr[offset + 4 * steps] = 1.f / 90 * (d1 - d3);
        outptr[offset + 5 * steps] = 8.f / 45 * (d2 + d3);
        outptr[offset + 6 * steps] = 8.f / 45 * (d2 - d3);
        outptr[offset + 7 * steps] = gw[i][2];
      }
    }
  }
}

template <>
void winograd_transform_input<8, 3>(const framework::Tensor &input,
                                    framework::Tensor *output) {
  // pack input to [8 * roundup(h/6), 8 * roundup(w/6), channel] tiles
  int channel = input.dims()[1];
  int height = input.dims()[2];
  int width = input.dims()[3];
  int h_tiles = (height + 3) / 6;  // (height + 5 - 2) / 6
  int w_tiles = (width + 3) / 6;   // (width + 5 - 2) / 6

  int tiles = (h_tiles * w_tiles + 7) / 8;
  framework::DDim transformed_shape =
      framework::make_ddim(std::vector<int>{tiles, 64, channel, 8});
  float *outptr = output->mutable_data<float>(transformed_shape);
  memset(outptr, 0, output->numel() * sizeof(float));

  const float *inptr = input.data<float>();
  int inter_h = (height - 2) / 6;
  int inter_w = (width - 2) / 6;
  int remain_h = height - (inter_h * 6);
  int remain_w = width - (inter_w * 6);
  framework::Tensor input_pad;
  if (remain_h > 2 || remain_w > 2) {
    inter_h += (remain_h > 2);
    inter_w += (remain_w > 2);
    height = (inter_h - 1) * 6 + 8;
    width = (inter_w - 1) * 6 + 8;
    framework::DDim input_shape =
        framework::make_ddim(std::vector<int>{1, channel, height, width});
    PadFunctor<CPU, float> pad;
    inptr = input_pad.mutable_data<float>(input_shape);
    pad(input, 0, height - input.dims()[2], 0, width - input.dims()[3],
        &input_pad);
  }

  size_t image_size = height * width;
  const float transform_matrix[8] = {5.25f, -5.f,   -4.25f, -2.5f,
                                     2.f,   -1.25f, 0.5f,   0.25f};
  #pragma omp parallel for
  for (int c = 0; c < channel; c += 4) {
    const float *in = inptr + c * image_size;
    float d_bt[64 * 4];  // d * B_t
    for (int h = 0; h < h_tiles; ++h) {
      for (int w = 0; w < w_tiles; ++w) {
        const float *in0 = in + (h * width + w) * 6;
        const float *in1 = in0 + image_size;
        const float *in2 = in1 + image_size;
        const float *in3 = in2 + image_size;
        float *d_bt_ptr = d_bt;
        asm volatile(
            "mov        r0, #8                          \n"
            "vld1.32    {d0-d3}, [%[tm_ptr]]            \n"
            // row loop
            "loop_r_%=:                                 \n"
            "vld1.32    {d4-d7}, [%[in0]], %[width]     \n"
            "vld1.32    {d8-d11}, [%[in1]], %[width]    \n"
            "vld1.32    {d12-d15}, [%[in2]], %[width]   \n"
            "vld1.32    {d16-d19}, [%[in3]], %[width]   \n"
            "vtrn.32    q2, q4                          \n"
            "vtrn.32    q3, q5                          \n"
            "vtrn.32    q6, q8                          \n"
            "vtrn.32    q7, q9                          \n"
            "vswp.32    d5, d12                         \n"
            "vswp.32    d9, d16                         \n"
            "vswp.32    d7, d14                         \n"
            "vswp.32    d11, d18                        \n"  // q2: d0, q3: d1,
                                                             // ..., q9: d7
            "vsub.f32   q10, q2, q8                     \n"
            "vsub.f32   q11, q6, q4                     \n"
            "vmla.f32   q10, q11, d0[0]                 \n"  // d0 - d6 + (d4 -
                                                             // d2) * 5.25
            "vst1.32    {d20-d21}, [%[d_bt]]!           \n"

            "vadd.f32   q10, q4, q8                     \n"
            "vadd.f32   q11, q3, q7                     \n"
            "vmla.f32   q10, q6, d1[0]                  \n"  // d2 - 4.25 * d4 +
                                                             // d6
            "vmla.f32   q11, q5, d1[0]                  \n"  // d1 - 4.25 * d3 +
                                                             // d5
            "vadd.f32   q12, q10, q11                   \n"
            "vsub.f32   q13, q10, q11                   \n"
            "vst1.32    {d24-d27}, [%[d_bt]]!           \n"

            "vmul.f32   q10, q4, d3[1]                  \n"  // 0.25 * d2
            "vmul.f32   q11, q3, d3[0]                  \n"  // 0.5 * d1
            "vadd.f32   q10, q10, q8                    \n"  // 0.25 * d2 + d6
            "vmla.f32   q11, q7, d2[0]                  \n"  // 0.5 * d1 + 2 *
                                                             // d5
            "vmla.f32   q10, q6, d2[1]                  \n"  // 0.25 * d2 + d6
                                                             // - 1.25 * d4
            "vmla.f32   q11, q5, d1[1]                  \n"  // 0.5 * d1 + 2 *
                                                             // d5 - 2.5 * d3
            "vadd.f32   q12, q10, q11                   \n"
            "vsub.f32   q13, q10, q11                   \n"
            "vst1.32    {d24-d27}, [%[d_bt]]!           \n"

            "vmul.f32   q10, q4, d2[0]                  \n"  // 2 * d2
            "vmul.f32   q11, q3, d2[0]                  \n"  // 2 * d1
            "vmla.f32   q10, q6, d1[1]                  \n"  // 2 * d2 - 2.5 *
                                                             // d4
            "vmla.f32   q11, q5, d1[1]                  \n"  // 2 * d1 - 2.5 *
                                                             // d3
            "vmla.f32   q10, q8, d3[0]                  \n"  // 2 * d1 - 2.5 *
                                                             // d3 + 0.5 * d6
            "vmla.f32   q11, q7, d3[0]                  \n"  // 2 * d2 - 2.5 *
                                                             // d4 + 0.5 * d5
            "vmul.f32   q10, q10, d2[0]                 \n"  // 4 * d1 - 5 * d3
                                                             // + d6
            "vadd.f32   q12, q10, q11                   \n"
            "vsub.f32   q13, q10, q11                   \n"
            "vst1.32    {d24-d27}, [%[d_bt]]!           \n"

            "vsub.f32   q10, q9, q3                     \n"
            "vsub.f32   q11, q5, q7                     \n"
            "vmla.f32   q10, q11, d0[0]                 \n"
            "vst1.32    {d20-d21}, [%[d_bt]]!           \n"

            "subs       r0, #1                          \n"
            "bne        loop_r_%=                       \n"
            : [d_bt] "+r"(d_bt_ptr), [in0] "+r"(in0), [in1] "+r"(in1),
              [in2] "+r"(in2), [in3] "+r"(in3)
            : [tm_ptr] "r"((float *)transform_matrix), [width] "r"(width)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12", "q13", "r0");

        float *ptr0 = d_bt;
        float *ptr1 = ptr0 + 32;
        float *ptr2 = ptr1 + 32;
        float *ptr3 = ptr2 + 32;
        float *ptr4 = ptr3 + 32;
        float *ptr5 = ptr4 + 32;
        float *ptr6 = ptr5 + 32;
        float *ptr7 = ptr6 + 32;
        int tile_id = h * w_tiles + w;
        int block_id = tile_id >> 3;
        int pack_id = tile_id & 0x7;
        // (tiles / 8, 64, channel, 8)
        float *out0 = outptr + (block_id * 64 * channel + c) * 8 + pack_id;
        int steps = channel * 8;
        asm volatile(
            "vld1.32    {d0-d3}, [%[tm_ptr]]            \n"
            "mov        r0, 4                           \n"
            "loop_col_%=:                               \n"
            // col 0:
            "vld1.32    {d4-d5}, [%[ptr0]]!             \n"  // q2: d0
            "vld1.32    {d6-d7}, [%[ptr1]]!             \n"  // q3: d1
            "vld1.32    {d8-d9}, [%[ptr2]]!             \n"  // q4: d2
            "vld1.32    {d10-d11}, [%[ptr3]]!           \n"  // q5: d3
            "vld1.32    {d12-d13}, [%[ptr4]]!           \n"  // q6: d4
            "vld1.32    {d14-d15}, [%[ptr5]]!           \n"  // q7: d5
            "vld1.32    {d16-d17}, [%[ptr6]]!           \n"  // q8: d6
            "vld1.32    {d18-d19}, [%[ptr7]]!           \n"  // q9: d7

            "vsub.f32  q10, q2, q8                      \n"  // d0 - d6
            "vsub.f32  q11, q6, q4                      \n"  // d4 - d2
            "vmla.f32  q10, q11, d0[0]                  \n"  // d0 - d6 + (d4 -
                                                             // d2) * 5.25
            "vst1.32   {d20-d21}, [%[out0]], %[steps]   \n"

            "vadd.f32   q10, q4, q8                     \n"
            "vadd.f32   q11, q3, q7                     \n"
            "vmla.f32   q10, q6, d1[0]                  \n"  // d2 - 4.25 * d4 +
                                                             // d6
            "vmla.f32   q11, q5, d1[0]                  \n"  // d1 - 4.25 * d3 +
                                                             // d5
            "vadd.f32   q12, q10, q11                   \n"
            "vst1.32    {d24-d25}, [%[out0]], %[steps]  \n"
            "vsub.f32   q12, q10, q11                   \n"
            "vst1.32    {d24-d25}, [%[out0]], %[steps]  \n"

            "vmul.f32   q10, q4, d3[1]                  \n"  // 0.25 * d2
            "vmul.f32   q11, q3, d3[0]                  \n"  // 0.5 * d1
            "vadd.f32   q10, q10, q8                    \n"  // 0.25 * d2 + d6
            "vmla.f32   q11, q7, d2[0]                  \n"  // 0.5 * d1 + 2 *
                                                             // d5
            "vmla.f32   q10, q6, d2[1]                  \n"  // 0.25 * d2 + d6
                                                             // - 1.25 * d4
            "vmla.f32   q11, q5, d1[1]                  \n"  // 0.5 * d1 + 2 *
                                                             // d5 - 2.5 * d3
            "vadd.f32   q12, q10, q11                   \n"
            "vst1.32    {d24-d25}, [%[out0]], %[steps]  \n"
            "vsub.f32   q12, q10, q11                   \n"
            "vst1.32    {d24-d25}, [%[out0]], %[steps]  \n"

            "vmul.f32   q10, q4, d2[0]                  \n"  // 2 * d2
            "vmul.f32   q11, q3, d2[0]                  \n"  // 2 * d1
            "vmla.f32   q10, q6, d1[1]                  \n"  // 2 * d2 - 2.5 *
                                                             // d4
            "vmla.f32   q11, q5, d1[1]                  \n"  // 2 * d1 - 2.5 *
                                                             // d3
            "vmla.f32   q10, q8, d3[0]                  \n"  // 2 * d1 - 2.5 *
                                                             // d3 + 0.5 * d6
            "vmla.f32   q11, q7, d3[0]                  \n"  // 2 * d2 - 2.5 *
                                                             // d4 + 0.5 * d5
            "vmul.f32   q10, q10, d2[0]                 \n"  // 4 * d1 - 5 * d3
                                                             // + d6
            "vadd.f32   q12, q10, q11                   \n"
            "vst1.32    {d24-d25}, [%[out0]], %[steps]  \n"
            "vsub.f32   q12, q10, q11                   \n"
            "vst1.32    {d24-d25}, [%[out0]], %[steps]  \n"

            "vsub.f32   q10, q9, q3                     \n"
            "vsub.f32   q11, q5, q7                     \n"
            "vmla.f32   q10, q11, d0[0]                 \n"
            "vst1.32    {d24-d25}, [%[out0]], %[steps]  \n"

            // col 1:
            "vld1.32    {d4-d5}, [%[ptr0]]!             \n"  // q2: d0
            "vld1.32    {d6-d7}, [%[ptr1]]!             \n"  // q3: d1
            "vld1.32    {d8-d9}, [%[ptr2]]!             \n"  // q4: d2
            "vld1.32    {d10-d11}, [%[ptr3]]!           \n"  // q5: d3
            "vld1.32    {d12-d13}, [%[ptr4]]!           \n"  // q6: d4
            "vld1.32    {d14-d15}, [%[ptr5]]!           \n"  // q7: d5
            "vld1.32    {d16-d17}, [%[ptr6]]!           \n"  // q8: d6
            "vld1.32    {d18-d19}, [%[ptr7]]!           \n"  // q9: d7

            "vsub.f32  q10, q2, q8                      \n"  // d0 - d6
            "vsub.f32  q11, q6, q4                      \n"  // d4 - d2
            "vmla.f32  q10, q11, d0[0]                  \n"  // d0 - d6 + (d4 -
                                                             // d2) * 5.25
            "vst1.32   {d20-d21}, [%[out0]], %[steps]   \n"

            "vadd.f32   q10, q4, q8                     \n"
            "vadd.f32   q11, q3, q7                     \n"
            "vmla.f32   q10, q6, d1[0]                  \n"  // d2 - 4.25 * d4 +
                                                             // d6
            "vmla.f32   q11, q5, d1[0]                  \n"  // d1 - 4.25 * d3 +
                                                             // d5
            "vadd.f32   q12, q10, q11                   \n"
            "vst1.32    {d24-d25}, [%[out0]], %[steps]  \n"
            "vsub.f32   q12, q10, q11                   \n"
            "vst1.32    {d24-d25}, [%[out0]], %[steps]  \n"

            "vmul.f32   q10, q4, d3[1]                  \n"  // 0.25 * d2
            "vmul.f32   q11, q3, d3[0]                  \n"  // 0.5 * d1
            "vadd.f32   q10, q10, q8                    \n"  // 0.25 * d2 + d6
            "vmla.f32   q11, q7, d2[0]                  \n"  // 0.5 * d1 + 2 *
                                                             // d5
            "vmla.f32   q10, q6, d2[1]                  \n"  // 0.25 * d2 + d6
                                                             // - 1.25 * d4
            "vmla.f32   q11, q5, d1[1]                  \n"  // 0.5 * d1 + 2 *
                                                             // d5 - 2.5 * d3
            "vadd.f32   q12, q10, q11                   \n"
            "vst1.32    {d24-d25}, [%[out0]], %[steps]  \n"
            "vsub.f32   q12, q10, q11                   \n"
            "vst1.32    {d24-d25}, [%[out0]], %[steps]  \n"

            "vmul.f32   q10, q4, d2[0]                  \n"  // 2 * d2
            "vmul.f32   q11, q3, d2[0]                  \n"  // 2 * d1
            "vmla.f32   q10, q6, d1[1]                  \n"  // 2 * d2 - 2.5 *
                                                             // d4
            "vmla.f32   q11, q5, d1[1]                  \n"  // 2 * d1 - 2.5 *
                                                             // d3
            "vmla.f32   q10, q8, d3[0]                  \n"  // 2 * d1 - 2.5 *
                                                             // d3 + 0.5 * d6
            "vmla.f32   q11, q7, d3[0]                  \n"  // 2 * d2 - 2.5 *
                                                             // d4 + 0.5 * d5
            "vmul.f32   q10, q10, d2[0]                 \n"  // 4 * d1 - 5 * d3
                                                             // + d6
            "vadd.f32   q12, q10, q11                   \n"
            "vst1.32    {d24-d25}, [%[out0]], %[steps]  \n"
            "vsub.f32   q12, q10, q11                   \n"
            "vst1.32    {d24-d25}, [%[out0]], %[steps]  \n"

            "vsub.f32   q10, q9, q3                     \n"
            "vsub.f32   q11, q5, q7                     \n"
            "vmla.f32   q10, q11, d0[0]                 \n"
            "vst1.32    {d24-d25}, [%[out0]], %[steps]  \n"

            "subs       r0, #1                          \n"
            "bne        loop_col_%=                     \n"
            : [out0] "+r"(out0), [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1),
              [ptr2] "+r"(ptr2), [ptr3] "+r"(ptr3), [ptr4] "+r"(ptr4),
              [ptr5] "+r"(ptr5), [ptr6] "+r"(ptr6), [ptr7] "+r"(ptr7)
            : [tm_ptr] "r"((float *)transform_matrix), [steps] "r"(steps)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12", "q13", "r0");
      }
    }
  }

  // remainer channels
  int remain_c_start = ((channel >> 2) << 2);
  for (int c = remain_c_start; c < channel; ++c) {
    const float *in = inptr + c * image_size;
    float d_bt[64];  // d * B_t
    for (int h = 0; h < h_tiles; ++h) {
      for (int w = 0; w < w_tiles; ++w) {
        const float *in0 = in + (h * width + w) * 6;
        const float *in1 = in0 + width;
        const float *in2 = in1 + width;
        const float *in3 = in2 + width;
        float *d_bt_ptr = d_bt;
        int steps = 4 * width;
        asm volatile(
            "vld1.32    {d0-d3}, [%[tm_ptr]]            \n"
            "mov        r0, #2                          \n"
            // row loop
            "loop_r_%=:                                 \n"
            "vld1.32    {d4-d7}, [%[in0]], %[steps]     \n"
            "vld1.32    {d8-d11}, [%[in1]], %[steps]    \n"
            "vld1.32    {d12-d15}, [%[in2]], %[steps]   \n"
            "vld1.32    {d16-d19}, [%[in3]], %[steps]   \n"
            "vtrn.32    q2, q4                          \n"
            "vtrn.32    q3, q5                          \n"
            "vtrn.32    q6, q8                          \n"
            "vtrn.32    q7, q9                          \n"
            "vswp.32    d5, d12                         \n"
            "vswp.32    d9, d16                         \n"
            "vswp.32    d7, d14                         \n"
            "vswp.32    d11, d18                        \n"  // q2: d0, q3: d1,
                                                             // ..., q9: d7
            "vsub.f32   q10, q2, q8                     \n"
            "vsub.f32   q11, q6, q4                     \n"
            "vmla.f32   q10, q11, d0[0]                 \n"  // d0 - d6 + (d4 -
                                                             // d2) * 5.25"
            "vst1.32    {d20-d21}, [%[d_bt]]!           \n"

            "vadd.f32   q10, q4, q8                     \n"
            "vadd.f32   q11, q3, q7                     \n"
            "vmla.f32   q10, q6, d1[0]                  \n"  // d2 - 4.25 * d4 +
                                                             // d6
            "vmla.f32   q11, q5, d1[0]                  \n"  // d1 - 4.25 * d3 +
                                                             // d5
            "vadd.f32   q12, q10, q11                   \n"
            "vsub.f32   q13, q10, q11                   \n"
            "vst1.32    {d24-d27}, [%[d_bt]]!           \n"

            "vmul.f32   q10, q4, d3[1]                  \n"  // 0.25 * d2
            "vmul.f32   q11, q3, d3[0]                  \n"  // 0.5 * d1
            "vadd.f32   q10, q10, q8                    \n"  // 0.25 * d2 + d6
            "vmla.f32   q11, q7, d2[0]                  \n"  // 0.5 * d1 + 2 *
                                                             // d5
            "vmla.f32   q10, q6, d2[1]                  \n"  // 0.25 * d2 + d6
                                                             // - 1.25 * d4
            "vmla.f32   q11, q5, d1[1]                  \n"  // 0.5 * d1 + 2 *
                                                             // d5 - 2.5 * d3
            "vadd.f32   q12, q10, q11                   \n"
            "vsub.f32   q13, q10, q11                   \n"
            "vst1.32    {d24-d27}, [%[d_bt]]!           \n"

            "vmul.f32   q10, q4, d2[0]                  \n"  // 2 * d2
            "vmul.f32   q11, q3, d2[0]                  \n"  // 2 * d1
            "vmla.f32   q10, q6, d1[1]                  \n"  // 2 * d2 - 2.5 *
                                                             // d4
            "vmla.f32   q11, q5, d1[1]                  \n"  // 2 * d1 - 2.5 *
                                                             // d3
            "vmla.f32   q10, q8, d3[0]                  \n"  // 2 * d1 - 2.5 *
                                                             // d3 + 0.5 * d6
            "vmla.f32   q11, q7, d3[0]                  \n"  // 2 * d2 - 2.5 *
                                                             // d4 + 0.5 * d5
            "vmul.f32   q10, q10, d2[0]                 \n"  // 4 * d1 - 5 * d3
                                                             // + d6
            "vadd.f32   q12, q10, q11                   \n"
            "vsub.f32   q13, q10, q11                   \n"
            "vst1.32    {d24-d27}, [%[d_bt]]!           \n"

            "vsub.f32   q10, q9, q3                     \n"
            "vsub.f32   q11, q5, q7                     \n"
            "vmla.f32   q10, q11, d0[0]                 \n"
            "vst1.32    {d20-d21}, [%[d_bt]]!           \n"

            "subs       r0, #1                          \n"
            "bne        loop_r_%=                       \n"
            : [d_bt] "+r"(d_bt_ptr), [in0] "+r"(in0), [in1] "+r"(in1),
              [in2] "+r"(in2), [in3] "+r"(in3)
            : [tm_ptr] "r"((float *)transform_matrix), [steps] "r"(steps)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12", "q13", "r0");

        float *ptr0 = d_bt;
        float *ptr1 = ptr0 + 32;
        int tile_id = h * w_tiles + w;
        int block_id = tile_id >> 3;
        int pack_id = tile_id & 0x7;
        // (tiles / 8, 64, channel, 8)
        float *out0 = outptr + (block_id * 64 * channel + c) * 8 + pack_id;
        steps = channel * 8;
        asm volatile(
            "mov        r0, #2                          \n"
            "vld1.32    {d0-d3}, [%[tm_ptr]]            \n"
            // row loop
            "loop_r_%=:                                 \n"
            "vld1.32    {d4-d5}, [%[ptr0]]!             \n"  // q2: d0
            "vld1.32    {d6-d7}, [%[ptr0]]!             \n"  // q3: d1
            "vld1.32    {d8-d9}, [%[ptr0]]!             \n"  // q4: d2
            "vld1.32    {d10-d11}, [%[ptr0]]!           \n"  // q5: d3
            "vld1.32    {d12-d13}, [%[ptr1]]!           \n"  // q6: d4
            "vld1.32    {d14-d15}, [%[ptr1]]!           \n"  // q7: d5
            "vld1.32    {d16-d17}, [%[ptr1]]!           \n"  // q8: d6
            "vld1.32    {d18-d19}, [%[ptr1]]!           \n"  // q9: d7

            "vtrn.32    q2, q3                          \n"
            "vtrn.32    q4, q5                          \n"
            "vtrn.32    q6, q7                          \n"
            "vtrn.32    q8, q9                          \n"
            "vswp.32    d5, d8                          \n"
            "vswp.32    d7, d10                         \n"
            "vswp.32    d13, d16                        \n"
            "vswp.32    d15, d18                        \n"

            "vsub.f32   q10, q2, q8                     \n"  // d0 - d6
            "vsub.f32   q11, q6, q4                     \n"  // d4 - d2
            "vmla.f32   q10, q11, d0[0]                 \n"  // d0 - d6 + (d4 -
                                                             // d2) * 5.25
            "vst1.32    {d20[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d20[1]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d21[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d21[1]}, [%[out0]], %[steps]   \n"

            "vadd.f32   q10, q4, q8                     \n"
            "vadd.f32   q11, q3, q7                     \n"
            "vmla.f32   q10, q6, d1[0]                  \n"  // d2 - 4.25 * d4 +
                                                             // d6
            "vmla.f32   q11, q5, d1[0]                  \n"  // d1 - 4.25 * d3 +
                                                             // d5
            "vadd.f32   q12, q10, q11                   \n"
            "vst1.32    {d24[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d24[1]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d25[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d25[1]}, [%[out0]], %[steps]   \n"
            "vsub.f32   q12, q10, q11                   \n"
            "vst1.32    {d24[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d24[1]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d25[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d25[1]}, [%[out0]], %[steps]   \n"

            "vmul.f32   q10, q4, d3[1]                  \n"  // 0.25 * d2
            "vmul.f32   q11, q3, d3[0]                  \n"  // 0.5 * d1
            "vadd.f32   q10, q10, q8                    \n"  // 0.25 * d2 + d6
            "vmla.f32   q11, q7, d2[0]                  \n"  // 0.5 * d1 + 2 *
                                                             // d5
            "vmla.f32   q10, q6, d2[1]                  \n"  // 0.25 * d2 + d6
                                                             // - 1.25 * d4
            "vmla.f32   q11, q5, d1[1]                  \n"  // 0.5 * d1 + 2 *
                                                             // d5 - 2.5 * d3
            "vadd.f32   q12, q10, q11                   \n"
            "vst1.32    {d24[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d24[1]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d25[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d25[1]}, [%[out0]], %[steps]   \n"
            "vsub.f32   q12, q10, q11                   \n"
            "vst1.32    {d24[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d24[1]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d25[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d25[1]}, [%[out0]], %[steps]   \n"

            "vmul.f32   q10, q4, d2[0]                  \n"  // 2 * d2
            "vmul.f32   q11, q3, d2[0]                  \n"  // 2 * d1
            "vmla.f32   q10, q6, d1[1]                  \n"  // 2 * d2 - 2.5 *
                                                             // d4
            "vmla.f32   q11, q5, d1[1]                  \n"  // 2 * d1 - 2.5 *
                                                             // d3
            "vmla.f32   q10, q8, d3[0]                  \n"  // 2 * d1 - 2.5 *
                                                             // d3 + 0.5 * d6
            "vmla.f32   q11, q7, d3[0]                  \n"  // 2 * d2 - 2.5 *
                                                             // d4 + 0.5 * d5
            "vmul.f32   q10, q10, d2[0]                 \n"  // 4 * d1 - 5 * d3
                                                             // + d6
            "vadd.f32   q12, q10, q11                   \n"
            "vst1.32    {d24[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d24[1]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d25[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d25[1]}, [%[out0]], %[steps]   \n"
            "vsub.f32   q12, q10, q11                   \n"
            "vst1.32    {d24[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d24[1]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d25[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d25[1]}, [%[out0]], %[steps]   \n"

            "vsub.f32   q10, q9, q3                     \n"
            "vsub.f32   q11, q5, q7                     \n"
            "vmla.f32   q10, q11, d0[0]                 \n"
            "vst1.32    {d20[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d20[1]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d21[0]}, [%[out0]], %[steps]   \n"
            "vst1.32    {d21[1]}, [%[out0]], %[steps]   \n"

            "subs       r0, #1                          \n"
            "bne        loop_r_%=                       \n"
            : [out0] "+r"(out0), [ptr0] "+r"(ptr0), [ptr1] "+r"(ptr1)
            : [tm_ptr] "r"((float *)transform_matrix), [steps] "r"(steps)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12", "q13", "r0");
      }
    }
  }
}

template <>
void winograd_transform_output<8, 3>(const framework::Tensor &input,
                                     const framework::Tensor &weight,
                                     framework::Tensor *output) {
  // weight shape is [out_channel/4, 64, in_channel, 4],
  // input shape is [hw/8, 64, in_channel, 8]
  int in_channel = input.dims()[2];
  int tiles = input.dims()[0];
  int out_channel = weight.dims()[0];

  // compute U*V first
  framework::Tensor uv_trans;
  framework::DDim shape =
      framework::make_ddim(std::vector<int>{4 * out_channel, 8 * tiles, 64});
  float *uv_trans_ptr = uv_trans.mutable_data<float>(shape);
  memset(uv_trans_ptr, 0, uv_trans.numel() * sizeof(float));
  const float *input_ptr = input.data<float>();
  const float *weight_ptr = weight.data<float>();

  for (int i = 0; i < out_channel; ++i) {
    float *uv_ptr = uv_trans_ptr + (i * tiles * 64 * 32);
    for (int k = 0; k < 64; ++k) {
      const float *w_ptr = weight_ptr + (i * 64 + k) * in_channel * 4;
      for (int j = 0; j < tiles; ++j) {
        const float *in_ptr = input_ptr + (k * tiles + j) * in_channel * 8;
        float *out0 = uv_ptr + (8 * j) * 64 + k;  // out channel 0
        float *out1 = out0 + 8 * tiles * 64;      // out channel 1
        float *out2 = out1 + 8 * tiles * 64;      // out channel 2
        float *out3 = out2 + 8 * tiles * 64;      // out channel 3
        int inter_channel = in_channel >> 1;
        int remain_channel = in_channel & 0x1;
        int steps = 64;
        asm volatile(
            "veor       q8, q8, q8                     \n"
            "veor       q9, q9, q9                     \n"
            "veor       q10, q10, q10                  \n"
            "veor       q11, q11, q11                  \n"
            "veor       q12, q12, q12                  \n"
            "veor       q13, q13, q13                  \n"
            "veor       q14, q14, q14                  \n"
            "veor       q15, q15, q15                  \n"
            // loop 2 channels
            "cmp        %[inter_channel], #0           \n"
            "ble        cmp_remain_%=                  \n"

            "loop_4c_%=:                               \n"
            "vld1.32    {d0-d3}, [%[w_ptr]]!           \n"
            "vld1.32    {d4-d7}, [%[in_ptr]]!          \n"
            "vmla.f32   q8, q2, d0[0]                  \n"
            "vmla.f32   q9, q3, d0[0]                  \n"
            "vmla.f32   q10, q2, d0[1]                 \n"
            "vmla.f32   q11, q3, d0[1]                 \n"
            "vmla.f32   q12, q2, d1[0]                 \n"
            "vmla.f32   q13, q3, d1[0]                 \n"
            "vmla.f32   q14, q2, d1[1]                 \n"
            "vmla.f32   q15, q3, d1[1]                 \n"

            "vld1.32    {d8-d11}, [%[in_ptr]]!         \n"
            "vmla.f32   q8, q4, d2[0]                  \n"
            "vmla.f32   q9, q5, d2[0]                  \n"
            "vmla.f32   q10, q4, d2[1]                 \n"
            "vmla.f32   q11, q5, d2[1]                 \n"
            "vmla.f32   q12, q4, d3[0]                 \n"
            "vmla.f32   q13, q5, d3[0]                 \n"
            "vmla.f32   q14, q4, d3[1]                 \n"
            "vmla.f32   q15, q5, d3[1]                 \n"

            "subs      %[inter_channel], #1            \n"
            "bne       loop_4c_%=                      \n"

            // cmp remain channel > 0
            "cmp_remain_%=:                            \n"
            "cmp        %[remain_channel], #0          \n"
            "ble        store_res_%=                   \n"

            // loop 1 channel
            "loop_c_%=:                                \n"
            "vld1.32    {d0-d1}, [%[w_ptr]]!           \n"
            "vld1.32    {d4-d7}, [%[in_ptr]]!          \n"

            "vmla.f32   q8, q2, d0[0]                  \n"
            "vmla.f32   q9, q3, d0[0]                  \n"
            "vmla.f32   q10, q2, d0[1]                 \n"
            "vmla.f32   q11, q3, d0[1]                 \n"
            "vmla.f32   q12, q2, d1[0]                 \n"
            "vmla.f32   q13, q3, d1[0]                 \n"
            "vmla.f32   q14, q2, d1[1]                 \n"
            "vmla.f32   q15, q3, d1[1]                 \n"

            "subs       %[remain_channel], #1          \n"
            "bne        loop_c_%=                      \n"

            "store_res_%=:                             \n"
            "vst1.32    {d16[0]},  [%[out0]], %[steps] \n"
            "vst1.32    {d16[1]},  [%[out0]], %[steps] \n"
            "vst1.32    {d17[0]},  [%[out0]], %[steps] \n"
            "vst1.32    {d17[1]},  [%[out0]], %[steps] \n"
            "vst1.32    {d18[0]},  [%[out0]], %[steps] \n"
            "vst1.32    {d18[1]},  [%[out0]], %[steps] \n"
            "vst1.32    {d19[0]},  [%[out0]], %[steps] \n"
            "vst1.32    {d19[1]},  [%[out0]], %[steps] \n"

            "vst1.32    {d20[0]},  [%[out1]], %[steps] \n"
            "vst1.32    {d20[1]},  [%[out1]], %[steps] \n"
            "vst1.32    {d21[0]},  [%[out1]], %[steps] \n"
            "vst1.32    {d21[1]},  [%[out1]], %[steps] \n"
            "vst1.32    {d22[0]},  [%[out1]], %[steps] \n"
            "vst1.32    {d22[1]},  [%[out1]], %[steps] \n"
            "vst1.32    {d23[0]},  [%[out1]], %[steps] \n"
            "vst1.32    {d23[1]},  [%[out1]], %[steps] \n"

            "vst1.32    {d24[0]},  [%[out2]], %[steps] \n"
            "vst1.32    {d24[1]},  [%[out2]], %[steps] \n"
            "vst1.32    {d25[0]},  [%[out2]], %[steps] \n"
            "vst1.32    {d25[1]},  [%[out2]], %[steps] \n"
            "vst1.32    {d26[0]},  [%[out2]], %[steps] \n"
            "vst1.32    {d26[1]},  [%[out2]], %[steps] \n"
            "vst1.32    {d27[0]},  [%[out2]], %[steps] \n"
            "vst1.32    {d27[1]},  [%[out2]], %[steps] \n"

            "vst1.32    {d28[0]},  [%[out3]], %[steps] \n"
            "vst1.32    {d28[1]},  [%[out3]], %[steps] \n"
            "vst1.32    {d29[0]},  [%[out3]], %[steps] \n"
            "vst1.32    {d29[1]},  [%[out3]], %[steps] \n"
            "vst1.32    {d30[0]},  [%[out3]], %[steps] \n"
            "vst1.32    {d30[1]},  [%[out3]], %[steps] \n"
            "vst1.32    {d31[0]},  [%[out3]], %[steps] \n"
            "vst1.32    {d31[1]},  [%[out3]], %[steps] \n"
            : [w_ptr] "+r"(w_ptr), [in_ptr] "+r"(in_ptr), [out0] "+r"(out0),
              [out1] "+r"(out1), [out2] "+r"(out2), [out3] "+r"(out3),
              [remain_channel] "+r"(remain_channel),
              [inter_channel] "+r"(inter_channel)
            : [steps] "r"(steps)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
      }
    }
  }
  /*
   * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
   * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
   * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
   * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
   * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
   * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
   */
  int out_h = output->dims()[2];
  int out_w = output->dims()[3];
  int h_tiles = (out_h + 5) / 6;
  int w_tiles = (out_w + 5) / 6;
  int remain_h = out_h - out_h / 6 * 6;
  int remain_w = out_w - out_w / 6 * 6;
  float *output_ptr = output->mutable_data<float>();
  out_channel = output->dims()[1];
  float transform_matrix[8] = {2.f, 4.f, 8.f, 16.f};

  for (int oc = 0; oc < out_channel; ++oc) {
    float at_m[48];        // [6][8]
    float output_tmp[36];  // [6][6], temporarily restore results
    const float *uv_ptr = uv_trans_ptr + oc * h_tiles * w_tiles * 64;
    for (int tile_h = 0; tile_h < h_tiles; ++tile_h) {
      for (int tile_w = 0; tile_w < w_tiles; ++tile_w) {
        float *at_m_ptr = at_m;
        asm volatile(
            "vld1.32    {d0-d1}, [%[tm_ptr]]          \n"
            "mov        r0, #2                        \n"
            "loop_%=:                                 \n"
            "vld1.32    {d2-d5}, [%[uv_ptr]]!         \n"
            "vld1.32    {d6-d9}, [%[uv_ptr]]!         \n"
            "vld1.32    {d10-d13}, [%[uv_ptr]]!       \n"
            "vld1.32    {d14-d17}, [%[uv_ptr]]!       \n"
            "vtrn.32    q1, q3                        \n"
            "vtrn.32    q2, q4                        \n"
            "vtrn.32    q5, q7                        \n"
            "vtrn.32    q6, q8                        \n"
            "vswp.32    d3, d10                       \n"  // q1: m0, q5: m2
            "vswp.32    d7, d14                       \n"  // q3: m1, q7: m3
            "vswp.32    d5, d12                       \n"  // q2: m4, q6: m6
            "vswp.32    d9, d16                       \n"  // q4: m5, q8: m7

            "vadd.f32   q9, q3, q5                    \n"  // m1 + m2
            "vadd.f32   q10, q7, q2                   \n"  // m3 + m4
            "vadd.f32   q11, q4, q6                   \n"  // m5 + m6
            "vsub.f32   q12, q3, q5                   \n"  // m1 - m2
            "vsub.f32   q13, q7, q2                   \n"  // m3 - m4
            "vsub.f32   q14, q4, q6                   \n"  // m5 - m6
            "vmul.f32   q2, q13, d0[0]                \n"  // 2 * (m3 - m4)
            "vmul.f32   q3, q11, d0[0]                \n"  // 2 * (m5 + m6)

            "vadd.f32   q15, q1, q9                   \n"
            "vadd.f32   q15, q15, q10                 \n"
            "vmla.f32   q15, q3, d1[1]                \n"
            "vst1.32    {d30-d31}, [%[at_m_ptr]]!     \n"

            "vadd.f32   q15, q12, q2                  \n"
            "vmla.f32   q15, q14, d1[1]               \n"
            "vst1.32    {d30-d31}, [%[at_m_ptr]]!     \n"

            "vmov.32    q15, q9                       \n"
            "vmla.f32   q15, q10, d0[1]               \n"
            "vmla.f32   q15, q11, d1[0]               \n"
            "vst1.32    {d30-d31}, [%[at_m_ptr]]!     \n"

            "vmov.32    q15, q12                      \n"
            "vmla.f32   q15, q13, d1[0]               \n"
            "vmla.f32   q15, q14, d0[1]               \n"
            "vst1.32    {d30-d31}, [%[at_m_ptr]]!     \n"

            "vadd.f32   q15, q9, q3                   \n"
            "vmla.f32   q15, q10, d1[1]               \n"
            "vst1.32    {d30-d31}, [%[at_m_ptr]]!     \n"

            "vadd.f32   q15, q12, q8                  \n"
            "vmla.f32   q15, q2, d1[1]                \n"
            "vst1.32    {d30-d31}, [%[at_m_ptr]]!     \n"

            "subs       r0, #1                        \n"
            "bne        loop_%=                       \n"
            : [uv_ptr] "+r"(uv_ptr), [at_m_ptr] "+r"(at_m_ptr)
            : [tm_ptr] "r"((float *)transform_matrix)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

        float *at_m_ptr0 = at_m;
        float *at_m_ptr1 = at_m + 24;
        if ((remain_w > 0 && tile_w == w_tiles - 1) ||
            (remain_h > 0 && tile_h == h_tiles - 1)) {
          float *out_ptr0 = output_tmp;
          float *out_ptr1 = output_tmp + 6;
          float *out_ptr2 = output_tmp + 12;
          float *out_ptr3 = output_tmp + 18;
          float *out_ptr4 = output_tmp + 24;
          float *out_ptr5 = output_tmp + 30;
          asm volatile(
              "vld1.32    {d0-d1}, [%[tm_ptr]]          \n"
              // process 4 rows
              "vld1.32    {d2-d5}, [%[at_m_ptr0]]!      \n"  // q1: m0, q2: m1
              "vld1.32    {d6-d9}, [%[at_m_ptr0]]!      \n"  // q3: m2, q4: m3
              "vld1.32    {d10-d13}, [%[at_m_ptr1]]!    \n"  // q5: m4, q6: m5
              "vld1.32    {d14-d17}, [%[at_m_ptr1]]!    \n"  // q7: m6, q8: m7
              "vtrn.32    q1, q2                        \n"
              "vtrn.32    q3, q4                        \n"
              "vtrn.32    q5, q6                        \n"
              "vtrn.32    q7, q8                        \n"
              "vswp.32    d3, d6                        \n"
              "vswp.32    d5, d8                        \n"
              "vswp.32    d11, d14                      \n"
              "vswp.32    d13, d16                      \n"

              "vadd.f32   q9, q2, q3                    \n"  // m1 + m2
              "vadd.f32   q10, q4, q5                   \n"  // m3 + m4
              "vadd.f32   q11, q6, q7                   \n"  // m5 + m6
              "vsub.f32   q12, q2, q3                   \n"  // m1 - m2
              "vsub.f32   q13, q4, q5                   \n"  // m3 - m4
              "vsub.f32   q14, q6, q7                   \n"  // m5 - m6
              "vmul.f32   q6, q13, d0[0]                \n"  // 2 * (m3 - m4)
              "vmul.f32   q7, q11, d0[0]                \n"  // 2 * (m5 + m6)

              "vadd.f32   q1, q1, q9                   \n"
              "vadd.f32   q1, q1, q10                 \n"
              "vmla.f32   q1, q3, d1[1]                \n"

              "vadd.f32   q2, q12, q6                  \n"
              "vmla.f32   q2, q14, d1[1]               \n"

              "vmov.32    q3, q9                       \n"
              "vmla.f32   q3, q10, d0[1]               \n"
              "vmla.f32   q3, q11, d1[0]               \n"

              "vmov.32    q4, q12                      \n"
              "vmla.f32   q4, q13, d1[0]               \n"
              "vmla.f32   q4, q14, d0[1]               \n"

              "vtrn.32    q1, q2                       \n"
              "vtrn.32    q3, q4                       \n"
              "vswp.32    d3, d6                       \n"
              "vswp.32    d5, d8                       \n"
              "vst1.32    {d2-d3}, [%[out_ptr0]]!      \n"
              "vst1.32    {d4-d5}, [%[out_ptr1]]!      \n"
              "vst1.32    {d6-d7}, [%[out_ptr2]]!      \n"
              "vst1.32    {d8-d9}, [%[out_ptr3]]!      \n"

              "vadd.f32   q1, q9, q7                   \n"
              "vmla.f32   q1, q10, d1[1]               \n"

              "vadd.f32   q2, q12, q8                  \n"
              "vmla.f32   q2, q6, d1[1]                \n"

              "vtrn.32    q1, q2                       \n"
              "vst1.32    {d2}, [%[out_ptr0]]!         \n"
              "vst1.32    {d4}, [%[out_ptr1]]!         \n"
              "vst1.32    {d3}, [%[out_ptr2]]!         \n"
              "vst1.32    {d5}, [%[out_ptr3]]!         \n"

              // remain 2 rows
              "vld1.32    {d2-d5}, [%[at_m_ptr0]]!      \n"  // d2: m0, d3: m2,
                                                             // d4: m1, d5: m3
              "vld1.32    {d6-d9}, [%[at_m_ptr1]]!      \n"  // d6: m4, d7: m6,
                                                             // d8: m5, d9: m7
              "vtrn.32    q1, q2                        \n"
              "vtrn.32    q3, q4                        \n"

              "vadd.f32   d10, d4, d3                   \n"  // m1 + m2
              "vadd.f32   d11, d5, d6                   \n"  // m3 + m4
              "vadd.f32   d12, d8, d7                   \n"  // m5 + m6
              "vsub.f32   d13, d4, d3                   \n"  // m1 - m2
              "vsub.f32   d14, d5, d6                   \n"  // m3 - m4
              "vsub.f32   d15, d8, d7                   \n"  // m5 - m6
              "vmul.f32   d16, d14, d0[0]               \n"  // 2 * (m3 - m4)
              "vmul.f32   d17, d12, d0[0]               \n"  // 2 * (m5 + m6)

              "vadd.f32   d18, d2, d10                  \n"
              "vadd.f32   d18, d18, d11                 \n"
              "vmla.f32   d18, d17, d1[1]               \n"

              "vadd.f32   d20, d13, d16                 \n"
              "vmla.f32   d20, d15, d1[1]               \n"

              "vmov.32    d19, d10                      \n"
              "vmla.f32   d19, d11, d0[1]               \n"
              "vmla.f32   d19, d20, d1[0]               \n"

              "vmov.32    d21, d13                      \n"
              "vmla.f32   d21, d14, d1[0]               \n"
              "vmla.f32   d21, d15, d0[1]               \n"

              "vtrn.32    d18, d19                      \n"
              "vtrn.32    d20, d21                      \n"
              "vst1.32    {d18-d19}, [%[out_ptr4]]!     \n"
              "vst1.32    {d20-d21}, [%[out_ptr5]]!     \n"

              "vadd.f32   d18, d10, d17                 \n"
              "vmla.f32   d18, d11, d1[1]               \n"

              "vadd.f32   d19, d13, d9                  \n"
              "vmla.f32   d19, d16, d1[1]               \n"

              "vtrn.32    d18, d19                      \n"
              "vst1.32    {d18}, [%[out_ptr4]]!         \n"
              "vst1.32    {d19}, [%[out_ptr5]]!         \n"
              : [out_ptr0] "+r"(out_ptr0), [out_ptr1] "+r"(out_ptr1),
                [out_ptr2] "+r"(out_ptr2), [out_ptr3] "+r"(out_ptr3),
                [out_ptr4] "+r"(out_ptr4), [out_ptr5] "+r"(out_ptr5),
                [at_m_ptr0] "+r"(at_m_ptr0), [at_m_ptr1] "+r"(at_m_ptr1)
              : [tm_ptr] "r"((float *)transform_matrix)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
          size_t offset = (oc * out_h + 6 * tile_h) * out_w + 6 * tile_w;
          float *out_ptr = output_ptr + offset;
          int remain_row = (tile_h < h_tiles - 1) ? 6 : remain_h;
          int remain_col = (tile_w < w_tiles - 1) ? 6 : remain_w;
          for (int i = 0; i < remain_row; ++i, out_ptr += out_w) {
            memcpy(out_ptr, output_tmp + i * 6, remain_col * sizeof(float));
          }
        } else {
          size_t offset = (oc * out_h + 6 * tile_h) * out_w + 6 * tile_w;
          float *out_ptr0 = output_ptr + offset;
          float *out_ptr1 = out_ptr0 + out_w;
          float *out_ptr2 = out_ptr1 + out_w;
          float *out_ptr3 = out_ptr2 + out_w;
          float *out_ptr4 = out_ptr3 + out_w;
          float *out_ptr5 = out_ptr4 + out_w;
          asm volatile(
              "vld1.32    {d0-d1}, [%[tm_ptr]]          \n"
              // process 4 rows
              "vld1.32    {d2-d5}, [%[at_m_ptr0]]!      \n"  // q1: m0, q2: m1
              "vld1.32    {d6-d9}, [%[at_m_ptr0]]!      \n"  // q3: m2, q4: m3
              "vld1.32    {d10-d13}, [%[at_m_ptr1]]!    \n"  // q5: m4, q6: m5
              "vld1.32    {d14-d17}, [%[at_m_ptr1]]!    \n"  // q7: m6, q8: m7
              "vtrn.32    q1, q2                        \n"
              "vtrn.32    q3, q4                        \n"
              "vtrn.32    q5, q6                        \n"
              "vtrn.32    q7, q8                        \n"
              "vswp.32    d3, d6                        \n"
              "vswp.32    d5, d8                        \n"
              "vswp.32    d11, d14                      \n"
              "vswp.32    d13, d16                      \n"

              "vadd.f32   q9, q2, q3                    \n"  // m1 + m2
              "vadd.f32   q10, q4, q5                   \n"  // m3 + m4
              "vadd.f32   q11, q6, q7                   \n"  // m5 + m6
              "vsub.f32   q12, q2, q3                   \n"  // m1 - m2
              "vsub.f32   q13, q4, q5                   \n"  // m3 - m4
              "vsub.f32   q14, q6, q7                   \n"  // m5 - m6
              "vmul.f32   q6, q13, d0[0]                \n"  // 2 * (m3 - m4)
              "vmul.f32   q7, q11, d0[0]                \n"  // 2 * (m5 + m6)

              "vadd.f32   q1, q1, q9                   \n"
              "vadd.f32   q1, q1, q10                 \n"
              "vmla.f32   q1, q3, d1[1]                \n"

              "vadd.f32   q2, q12, q6                  \n"
              "vmla.f32   q2, q14, d1[1]               \n"

              "vmov.32    q3, q9                       \n"
              "vmla.f32   q3, q10, d0[1]               \n"
              "vmla.f32   q3, q11, d1[0]               \n"

              "vmov.32    q4, q12                      \n"
              "vmla.f32   q4, q13, d1[0]               \n"
              "vmla.f32   q4, q14, d0[1]               \n"

              "vtrn.32    q1, q2                       \n"
              "vtrn.32    q3, q4                       \n"
              "vswp.32    d3, d6                       \n"
              "vswp.32    d5, d8                       \n"
              "vst1.32    {d2-d3}, [%[out_ptr0]]!      \n"
              "vst1.32    {d4-d5}, [%[out_ptr1]]!      \n"
              "vst1.32    {d6-d7}, [%[out_ptr2]]!      \n"
              "vst1.32    {d8-d9}, [%[out_ptr3]]!      \n"

              "vadd.f32   q1, q9, q7                   \n"
              "vmla.f32   q1, q10, d1[1]               \n"

              "vadd.f32   q2, q12, q8                  \n"
              "vmla.f32   q2, q6, d1[1]                \n"

              "vtrn.32    q1, q2                       \n"
              "vst1.32    {d2}, [%[out_ptr0]]!         \n"
              "vst1.32    {d4}, [%[out_ptr1]]!         \n"
              "vst1.32    {d3}, [%[out_ptr2]]!         \n"
              "vst1.32    {d5}, [%[out_ptr3]]!         \n"

              // remain 2 rows
              "vld1.32    {d2-d5}, [%[at_m_ptr0]]!      \n"  // d2: m0, d3: m2,
                                                             // d4: m1, d5: m3
              "vld1.32    {d6-d9}, [%[at_m_ptr1]]!      \n"  // d6: m4, d7: m6,
                                                             // d8: m5, d9: m7
              "vtrn.32    q1, q2                        \n"
              "vtrn.32    q3, q4                        \n"

              "vadd.f32   d10, d4, d3                   \n"  // m1 + m2
              "vadd.f32   d11, d5, d6                   \n"  // m3 + m4
              "vadd.f32   d12, d8, d7                   \n"  // m5 + m6
              "vsub.f32   d13, d4, d3                   \n"  // m1 - m2
              "vsub.f32   d14, d5, d6                   \n"  // m3 - m4
              "vsub.f32   d15, d8, d7                   \n"  // m5 - m6
              "vmul.f32   d16, d14, d0[0]               \n"  // 2 * (m3 - m4)
              "vmul.f32   d17, d12, d0[0]               \n"  // 2 * (m5 + m6)

              "vadd.f32   d18, d2, d10                  \n"
              "vadd.f32   d18, d18, d11                 \n"
              "vmla.f32   d18, d17, d1[1]               \n"

              "vadd.f32   d20, d13, d16                 \n"
              "vmla.f32   d20, d15, d1[1]               \n"

              "vmov.32    d19, d10                      \n"
              "vmla.f32   d19, d11, d0[1]               \n"
              "vmla.f32   d19, d20, d1[0]               \n"

              "vmov.32    d21, d13                      \n"
              "vmla.f32   d21, d14, d1[0]               \n"
              "vmla.f32   d21, d15, d0[1]               \n"

              "vtrn.32    d18, d19                      \n"
              "vtrn.32    d20, d21                      \n"
              "vst1.32    {d18-d19}, [%[out_ptr4]]!     \n"
              "vst1.32    {d20-d21}, [%[out_ptr5]]!     \n"

              "vadd.f32   d18, d10, d17                 \n"
              "vmla.f32   d18, d11, d1[1]               \n"

              "vadd.f32   d19, d13, d9                  \n"
              "vmla.f32   d19, d16, d1[1]               \n"

              "vtrn.32    d18, d19                      \n"
              "vst1.32    {d18}, [%[out_ptr4]]!         \n"
              "vst1.32    {d19}, [%[out_ptr5]]!         \n"
              : [out_ptr0] "+r"(out_ptr0), [out_ptr1] "+r"(out_ptr1),
                [out_ptr2] "+r"(out_ptr2), [out_ptr3] "+r"(out_ptr3),
                [out_ptr4] "+r"(out_ptr4), [out_ptr5] "+r"(out_ptr5),
                [at_m_ptr0] "+r"(at_m_ptr0), [at_m_ptr1] "+r"(at_m_ptr1)
              : [tm_ptr] "r"((float *)transform_matrix)
              : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
        }
      }
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
