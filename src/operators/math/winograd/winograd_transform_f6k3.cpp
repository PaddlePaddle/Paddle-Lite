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

// Inspired by https://arxiv.org/abs/1509.09308 and refered from nnpack and ncnn
// project.

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#ifdef CONV_OP

#include <arm_neon.h>
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
  // package weight into [roundup(out_channel/4), 64, in_channel, 4] tiles
  int out_channel = weight.dims()[0];
  int in_channel = weight.dims()[1];
  // reshape and alloc transformed weight
  framework::DDim transformed_shape = framework::make_ddim(
      std::vector<int>{(out_channel + 3) / 4, 64, in_channel, 4});
  float *trans_outptr = output->mutable_data<float>(transformed_shape);
  memset(trans_outptr, 0, output->numel() * sizeof(float));

  const float transform_matrix[8] = {2.f, -2.f / 9, 1.f / 90, 1.f / 180};
  const float *inptr = weight.data<float>();

#if __aarch64__
  int remain_start = 0;
#else
  int remain_start = out_channel & 0xFFFC;

  #pragma omp parallel for
  for (int oc = 0; oc < out_channel - 3; oc += 4) {
    float gw[96];  // gw[3][8][4]
    const float *inptr0 = inptr + oc * in_channel * 9;
    const float *inptr1 = inptr + (oc + 1) * in_channel * 9;
    const float *inptr2 = inptr + (oc + 2) * in_channel * 9;
    const float *inptr3 = inptr + (oc + 3) * in_channel * 9;
    // oc * 64 * in_channel
    float *outptr = trans_outptr + ((oc * in_channel) << 6);
    for (int ic = 0; ic < in_channel; ++ic) {
      float *gw_ptr = gw;
      asm volatile(
          "vld1.32    {d0-d1}, [%[tm_ptr]]          \n"

          "mov        r0, #24                       \n"
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

          "mov        r0, #12                       \n"
          "vld1.32    {d2-d3}, [%[inptr0]], r0      \n"
          "vld1.32    {d6-d7}, [%[inptr1]], r0      \n"
          "vld1.32    {d10-d11}, [%[inptr2]], r0    \n"
          "vld1.32    {d14-d15}, [%[inptr3]], r0    \n"
          "vtrn.32    q1, q3                        \n"
          "vtrn.32    q5, q7                        \n"
          "vswp.32    d3, d10                       \n"
          "vswp.32    d7, d14                       \n"

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
            [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3)
          : [tm_ptr] "r"((float *)transform_matrix)
          : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "q8", "q9", "q10", "q11", "q12", "q13", "r0");

      float *gw_ptr0 = gw;
      float *gw_ptr1 = gw + 32;
      float *gw_ptr2 = gw + 64;
      float *outptr0 = outptr + (ic << 2);            // ic * 4
      int steps = (in_channel << 2) * sizeof(float);  // in_channel * 4
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

          // w5 = ((g2 + 4 * g0) + 2 * g1) * (1.0 / 180)
          "vadd.f32   q12, q3, q9                        \n"
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
  #pragma omp parallel for
  for (int oc = remain_start; oc < out_channel; ++oc) {
    float gw[3][8];                                     // gw[3][8]
    const float *inptr0 = inptr + oc * in_channel * 9;  //
    // (oc / 4) * 64 * in_channel * 4 + oc % 4
    int offset = ((oc & 0xFFFC) << 6) * in_channel + (oc & 0x3);
    int steps = (in_channel << 2);  // in_channel * 4
    float *outptr = trans_outptr + offset;
    for (int ic = 0; ic < in_channel; ++ic) {
      for (int i = 0; i < 3; ++i, inptr0 += 3) {
        float g0 = inptr0[0];
        float g1 = inptr0[1];
        float g2 = inptr0[2];
        float d0 = g0 + g2;
        float d1 = g0 + 4 * g2;
        float d2 = g2 + 4 * g0;
        float d3 = 2 * g1;
        gw[i][0] = g0;
        gw[i][1] = -2.f / 9 * (d0 + g1);   // -2.f/9 * (g0 + g1 + g2)
        gw[i][2] = -2.f / 9 * (d0 - g1);   // -2.f/9 * (g0 - g1 + g2)
        gw[i][3] = 1.f / 90 * (d1 + d3);   // 1.f/90 * (g0 + 2 * g1 + 4 * g2)
        gw[i][4] = 1.f / 90 * (d1 - d3);   // 1.f/90 * (g0 - 2 * g1 + 4 * g2)
        gw[i][5] = 1.f / 180 * (d2 + d3);  // 1.f/180 * (4 * g0 + 2 * g1 + g2)
        gw[i][6] = 1.f / 180 * (d2 - d3);  // 1.f/180 * (4 * g0 - 2 * g1 + g2)
        gw[i][7] = g2;
      }
      for (int i = 0; i < 8; ++i) {
        float g0 = gw[0][i];
        float g1 = gw[1][i];
        float g2 = gw[2][i];
        float d0 = g0 + g2;
        float d1 = g0 + 4 * g2;
        float d2 = g2 + 4 * g0;
        float d3 = 2 * g1;
        int offset = i * 8 * steps;
        outptr[offset] = g0;
        outptr[offset + 1 * steps] = -2.f / 9 * (d0 + g1);
        outptr[offset + 2 * steps] = -2.f / 9 * (d0 - g1);
        outptr[offset + 3 * steps] = 1.f / 90 * (d1 + d3);
        outptr[offset + 4 * steps] = 1.f / 90 * (d1 - d3);
        outptr[offset + 5 * steps] = 1.f / 180 * (d2 + d3);
        outptr[offset + 6 * steps] = 1.f / 180 * (d2 - d3);
        outptr[offset + 7 * steps] = g2;
      }
      outptr += 4;
    }
  }
}

template <>
void winograd_transform_input<8, 3>(const framework::Tensor &input,
                                    framework::Tensor *output) {
  /*
   * x0 = (d0 - d6) + (d4 - d2) * 5.25
   * x1 = (d2 + d6) - 4.25 * (d4 + d3) + (d1 + d5)
   * x2 = (d2 + d6) - 4.25 * (d4 - d3) - (d1 + d5)
   * x3 = (0.25 * d2 - 1.25 * d4 + d6) + (0.5 * d1 - 2.5 * d3 + 2 * d5)
   * x4 = (0.25 * d2 - 1.25 * d4 + d6) - (0.5 * d1 - 2.5 * d3 + 2 * d5)
   * x5 = (4 * d2 - 5 * d4 + d6) + (2 * d1 - 2.5 * d3 + 0.5 * d5)
   * x6 = (4 * d2 - 5 * d4 + d6) - (2 * d1 - 2.5 * d3 + 0.5 * d5)
   * x7 = (d7 - d1) + (d3 - d5) * 5.25
   */
  // package input into [roundup(tiles/8), 64, channel, 8] tiles
  int channel = input.dims()[1];
  int height = input.dims()[2];
  int width = input.dims()[3];
  int h_tiles = (height + 3) / 6;  // (height - 2 + 5) / 6
  int w_tiles = (width + 3) / 6;   // (width - 2 + 5) / 6
  int tiles = (h_tiles * w_tiles + 7) / 8;
  framework::DDim transformed_shape =
      framework::make_ddim(std::vector<int>{tiles, 64, channel, 8});
  float *outptr = output->mutable_data<float>(transformed_shape);
  memset(outptr, 0, output->numel() * sizeof(float));

  const float *inptr = input.data<float>();
  height = h_tiles * 6 + 2;
  width = w_tiles * 6 + 2;
  framework::Tensor input_pad;
  if (height > input.dims()[2] || width > input.dims()[3]) {
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
  for (int c = 0; c < channel; ++c) {
    const float *in = inptr + c * image_size;
    float d_bt[64];  // d * B_t
    for (int h = 0; h < h_tiles; ++h) {
      for (int w = 0; w < w_tiles; ++w) {
        const float *in0 = in + (h * width + w) * 6;
        const float *in1 = in0 + width;
        const float *in2 = in1 + width;
        const float *in3 = in2 + width;
        float *d_bt_ptr = d_bt;
#if __aarch64__
        int steps = 4 * width;
        float32x4_t _q0 = vld1q_f32(transform_matrix);
        float32x4_t _q1 = vld1q_f32(transform_matrix + 4);
        for (int l = 0; l < 2; ++l) {
          float32x4x2_t _q23, _q45, _q67, _q89;
          _q23.val[0] = vld1q_f32(in0);
          _q45.val[0] = vld1q_f32(in0 + 4);
          _q23.val[1] = vld1q_f32(in1);
          _q45.val[1] = vld1q_f32(in1 + 4);
          _q67.val[0] = vld1q_f32(in2);
          _q89.val[0] = vld1q_f32(in2 + 4);
          _q67.val[1] = vld1q_f32(in3);
          _q89.val[1] = vld1q_f32(in3 + 4);
          _q23 = vtrnq_f32(_q23.val[0], _q23.val[1]);
          _q45 = vtrnq_f32(_q45.val[0], _q45.val[1]);
          _q67 = vtrnq_f32(_q67.val[0], _q67.val[1]);
          _q89 = vtrnq_f32(_q89.val[0], _q89.val[1]);
          float32x4_t _q2 = vcombine_f32(vget_low_f32(_q23.val[0]),
                                         vget_low_f32(_q67.val[0]));
          float32x4_t _q4 = vcombine_f32(vget_low_f32(_q23.val[1]),
                                         vget_low_f32(_q67.val[1]));
          float32x4_t _q3 = vcombine_f32(vget_low_f32(_q45.val[0]),
                                         vget_low_f32(_q89.val[0]));
          float32x4_t _q5 = vcombine_f32(vget_low_f32(_q45.val[1]),
                                         vget_low_f32(_q89.val[1]));
          float32x4_t _q6 = vcombine_f32(vget_high_f32(_q23.val[0]),
                                         vget_high_f32(_q67.val[0]));
          float32x4_t _q8 = vcombine_f32(vget_high_f32(_q23.val[1]),
                                         vget_high_f32(_q67.val[1]));
          float32x4_t _q7 = vcombine_f32(vget_high_f32(_q45.val[0]),
                                         vget_high_f32(_q89.val[0]));
          float32x4_t _q9 = vcombine_f32(vget_high_f32(_q45.val[1]),
                                         vget_high_f32(_q89.val[1]));

          float32x4_t _q10 = vsubq_f32(_q2, _q7);
          float32x4_t _q11 = vsubq_f32(_q3, _q6);
          _q10 = vmlaq_lane_f32(_q10, _q11, vget_low_f32(_q0), 0);
          vst1q_f32(d_bt_ptr, _q10);

          _q10 = vaddq_f32(_q6, _q7);
          _q11 = vaddq_f32(_q4, _q5);
          _q10 = vmlaq_lane_f32(_q10, _q3, vget_high_f32(_q0), 0);
          _q11 = vmlaq_lane_f32(_q11, _q8, vget_high_f32(_q0), 0);
          float32x4_t _q12 = vaddq_f32(_q10, _q11);
          float32x4_t _q13 = vsubq_f32(_q10, _q11);
          vst1q_f32(d_bt_ptr + 4, _q12);
          vst1q_f32(d_bt_ptr + 8, _q13);

          _q10 = vmulq_lane_f32(_q6, vget_high_f32(_q1), 1);
          _q11 = vmulq_lane_f32(_q4, vget_high_f32(_q1), 0);
          _q10 = vaddq_f32(_q10, _q7);
          _q11 = vmlaq_lane_f32(_q11, _q5, vget_low_f32(_q1), 0);
          _q10 = vmlaq_lane_f32(_q10, _q3, vget_low_f32(_q1), 1);
          _q11 = vmlaq_lane_f32(_q11, _q8, vget_high_f32(_q0), 1);
          _q12 = vaddq_f32(_q10, _q11);
          _q13 = vsubq_f32(_q10, _q11);
          vst1q_f32(d_bt_ptr + 12, _q12);
          vst1q_f32(d_bt_ptr + 16, _q13);

          _q10 = vmulq_lane_f32(_q6, vget_low_f32(_q1), 0);
          _q11 = vmulq_lane_f32(_q4, vget_low_f32(_q1), 0);
          _q10 = vmlaq_lane_f32(_q10, _q3, vget_high_f32(_q0), 1);
          _q11 = vmlaq_lane_f32(_q11, _q8, vget_high_f32(_q0), 1);
          _q10 = vmlaq_lane_f32(_q10, _q7, vget_high_f32(_q1), 0);
          _q11 = vmlaq_lane_f32(_q11, _q5, vget_high_f32(_q1), 0);
          _q10 = vmulq_lane_f32(_q10, vget_low_f32(_q1), 0);
          _q12 = vaddq_f32(_q10, _q11);
          _q13 = vsubq_f32(_q10, _q11);
          vst1q_f32(d_bt_ptr + 20, _q12);
          vst1q_f32(d_bt_ptr + 24, _q13);

          _q10 = vsubq_f32(_q9, _q4);
          _q11 = vsubq_f32(_q8, _q5);
          _q10 = vmlaq_lane_f32(_q10, _q11, vget_low_f32(_q0), 0);
          vst1q_f32(d_bt_ptr + 28, _q10);

          in0 += steps;
          in1 += steps;
          in2 += steps;
          in3 += steps;
          d_bt_ptr += 32;
        }
#else
        int steps = 4 * width * sizeof(float);
        asm volatile(
            "vld1.32    {d0-d3}, [%[tm_ptr]]            \n"
            "mov        r0, #2                          \n"
            // row loop
            "loop_r_%=:                                 \n"
            "vld1.32    {d4-d7}, [%[in0]], %[steps]     \n"
            "vld1.32    {d8-d11}, [%[in1]], %[steps]    \n"
            "vld1.32    {d12-d15}, [%[in2]], %[steps]   \n"
            "vld1.32    {d16-d19}, [%[in3]], %[steps]   \n"
            "vtrn.32    q2, q4                          \n"  // d0: q2
            "vtrn.32    q3, q5                          \n"  // d1: q4
            "vtrn.32    q6, q8                          \n"  // d2: q6
            "vtrn.32    q7, q9                          \n"  // d3: q8
            "vswp.32    d5, d12                         \n"  // d4: q3
            "vswp.32    d9, d16                         \n"  // d5: q5
            "vswp.32    d7, d14                         \n"  // d6: q7
            "vswp.32    d11, d18                        \n"  // d7: q9

            "vsub.f32   q10, q2, q7                     \n"
            "vsub.f32   q11, q3, q6                     \n"
            "vmla.f32   q10, q11, d0[0]                 \n"  // d0 - d6 + (d4 -
                                                             // d2) * 5.25"
            "vst1.32    {d20-d21}, [%[d_bt]]!           \n"

            "vadd.f32   q10, q6, q7                     \n"
            "vadd.f32   q11, q4, q5                     \n"
            "vmla.f32   q10, q3, d1[0]                  \n"  // d2 - 4.25 * d4 +
                                                             // d6
            "vmla.f32   q11, q8, d1[0]                  \n"  // d1 - 4.25 * d3 +
                                                             // d5
            "vadd.f32   q12, q10, q11                   \n"
            "vsub.f32   q13, q10, q11                   \n"
            "vst1.32    {d24-d27}, [%[d_bt]]!           \n"

            "vmul.f32   q10, q6, d3[1]                  \n"  // 0.25 * d2
            "vmul.f32   q11, q4, d3[0]                  \n"  // 0.5 * d1
            "vadd.f32   q10, q10, q7                    \n"  // 0.25 * d2 + d6
            "vmla.f32   q11, q5, d2[0]                  \n"  // 0.5 * d1 + 2 *
                                                             // d5
            "vmla.f32   q10, q3, d2[1]                  \n"  // 0.25 * d2 + d6
                                                             // - 1.25 * d4
            "vmla.f32   q11, q8, d1[1]                  \n"  // 0.5 * d1 + 2 *
                                                             // d5 - 2.5 * d3
            "vadd.f32   q12, q10, q11                   \n"
            "vsub.f32   q13, q10, q11                   \n"
            "vst1.32    {d24-d27}, [%[d_bt]]!           \n"

            "vmul.f32   q10, q6, d2[0]                  \n"  // 2 * d2
            "vmul.f32   q11, q4, d2[0]                  \n"  // 2 * d1
            "vmla.f32   q10, q3, d1[1]                  \n"  // 2 * d2 - 2.5 *
                                                             // d4
            "vmla.f32   q11, q8, d1[1]                  \n"  // 2 * d1 - 2.5 *
                                                             // d3
            "vmla.f32   q10, q7, d3[0]                  \n"  // 2 * d1 - 2.5 *
                                                             // d3 + 0.5 * d6
            "vmla.f32   q11, q5, d3[0]                  \n"  // 2 * d2 - 2.5 *
                                                             // d4 + 0.5 * d5
            "vmul.f32   q10, q10, d2[0]                 \n"  // 4 * d1 - 5 * d3
                                                             // + d6
            "vadd.f32   q12, q10, q11                   \n"
            "vsub.f32   q13, q10, q11                   \n"
            "vst1.32    {d24-d27}, [%[d_bt]]!           \n"

            "vsub.f32   q10, q9, q4                     \n"
            "vsub.f32   q11, q8, q5                     \n"
            "vmla.f32   q10, q11, d0[0]                 \n"
            "vst1.32    {d20-d21}, [%[d_bt]]!           \n"

            "subs       r0, #1                          \n"
            "bne        loop_r_%=                       \n"
            : [d_bt] "+r"(d_bt_ptr), [in0] "+r"(in0), [in1] "+r"(in1),
              [in2] "+r"(in2), [in3] "+r"(in3)
            : [tm_ptr] "r"((float *)transform_matrix), [steps] "r"(steps)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12", "q13", "r0");
#endif  // __aarch64__
        float *ptr0 = d_bt;
        float *ptr1 = ptr0 + 32;
        int tile_indics = h * w_tiles + w;
        int tile_block = tile_indics >> 3;
        int block_indics = tile_indics & 0x7;
        // (tiles / 8, 64, channel, 8)
        float *out0 =
            outptr + (tile_block * 64 * channel + c) * 8 + block_indics;
        float *out1 = out0 + channel * 8;
        float *out2 = out1 + channel * 8;
        float *out3 = out2 + channel * 8;
        float *out4 = out3 + channel * 8;
        float *out5 = out4 + channel * 8;
        float *out6 = out5 + channel * 8;
        float *out7 = out6 + channel * 8;
#if __aarch64__
        steps = 8 * channel * 8;
        for (int l = 0; l < 2; ++l) {
          float32x4x2_t _q23, _q45, _q67, _q89;
          _q23.val[0] = vld1q_f32(ptr0);
          _q23.val[1] = vld1q_f32(ptr0 + 4);
          _q45.val[0] = vld1q_f32(ptr0 + 8);
          _q45.val[1] = vld1q_f32(ptr0 + 12);
          _q67.val[0] = vld1q_f32(ptr1);
          _q67.val[1] = vld1q_f32(ptr1 + 4);
          _q89.val[0] = vld1q_f32(ptr1 + 8);
          _q89.val[1] = vld1q_f32(ptr1 + 12);
          _q23 = vtrnq_f32(_q23.val[0], _q23.val[1]);
          _q45 = vtrnq_f32(_q45.val[0], _q45.val[1]);
          _q67 = vtrnq_f32(_q67.val[0], _q67.val[1]);
          _q89 = vtrnq_f32(_q89.val[0], _q89.val[1]);
          float32x4_t _q2 = vcombine_f32(vget_low_f32(_q23.val[0]),
                                         vget_low_f32(_q45.val[0]));
          float32x4_t _q4 = vcombine_f32(vget_high_f32(_q23.val[0]),
                                         vget_high_f32(_q45.val[0]));
          float32x4_t _q3 = vcombine_f32(vget_low_f32(_q23.val[1]),
                                         vget_low_f32(_q45.val[1]));
          float32x4_t _q5 = vcombine_f32(vget_high_f32(_q23.val[1]),
                                         vget_high_f32(_q45.val[1]));
          float32x4_t _q6 = vcombine_f32(vget_low_f32(_q67.val[0]),
                                         vget_low_f32(_q89.val[0]));
          float32x4_t _q8 = vcombine_f32(vget_high_f32(_q67.val[0]),
                                         vget_high_f32(_q89.val[0]));
          float32x4_t _q7 = vcombine_f32(vget_low_f32(_q67.val[1]),
                                         vget_low_f32(_q89.val[1]));
          float32x4_t _q9 = vcombine_f32(vget_high_f32(_q67.val[1]),
                                         vget_high_f32(_q89.val[1]));

          float32x4_t _q10 = vsubq_f32(_q2, _q8);
          float32x4_t _q11 = vsubq_f32(_q6, _q4);
          _q10 = vmlaq_lane_f32(_q10, _q11, vget_low_f32(_q0), 0);
          vst1q_lane_f32(out0, _q10, 0);
          vst1q_lane_f32(out0 + steps, _q10, 1);
          vst1q_lane_f32(out0 + 2 * steps, _q10, 2);
          vst1q_lane_f32(out0 + 3 * steps, _q10, 3);

          _q10 = vaddq_f32(_q4, _q8);
          _q11 = vaddq_f32(_q3, _q7);
          _q10 = vmlaq_lane_f32(_q10, _q6, vget_high_f32(_q0), 0);
          _q11 = vmlaq_lane_f32(_q11, _q5, vget_high_f32(_q0), 0);
          float32x4_t _q12 = vaddq_f32(_q10, _q11);
          vst1q_lane_f32(out1, _q12, 0);
          vst1q_lane_f32(out1 + steps, _q12, 1);
          vst1q_lane_f32(out1 + 2 * steps, _q12, 2);
          vst1q_lane_f32(out1 + 3 * steps, _q12, 3);

          _q12 = vsubq_f32(_q10, _q11);
          vst1q_lane_f32(out2, _q12, 0);
          vst1q_lane_f32(out2 + steps, _q12, 1);
          vst1q_lane_f32(out2 + 2 * steps, _q12, 2);
          vst1q_lane_f32(out2 + 3 * steps, _q12, 3);

          _q10 = vmulq_lane_f32(_q4, vget_high_f32(_q1), 1);
          _q11 = vmulq_lane_f32(_q3, vget_high_f32(_q1), 0);
          _q10 = vaddq_f32(_q10, _q8);
          _q11 = vmlaq_lane_f32(_q11, _q7, vget_low_f32(_q1), 0);
          _q10 = vmlaq_lane_f32(_q10, _q6, vget_low_f32(_q1), 1);
          _q11 = vmlaq_lane_f32(_q11, _q5, vget_high_f32(_q0), 1);
          _q12 = vaddq_f32(_q10, _q11);
          vst1q_lane_f32(out3, _q12, 0);
          vst1q_lane_f32(out3 + steps, _q12, 1);
          vst1q_lane_f32(out3 + 2 * steps, _q12, 2);
          vst1q_lane_f32(out3 + 3 * steps, _q12, 3);

          _q12 = vsubq_f32(_q10, _q11);
          vst1q_lane_f32(out4, _q12, 0);
          vst1q_lane_f32(out4 + steps, _q12, 1);
          vst1q_lane_f32(out4 + 2 * steps, _q12, 2);
          vst1q_lane_f32(out4 + 3 * steps, _q12, 3);

          _q10 = vmulq_lane_f32(_q4, vget_low_f32(_q1), 0);
          _q11 = vmulq_lane_f32(_q3, vget_low_f32(_q1), 0);
          _q10 = vmlaq_lane_f32(_q10, _q6, vget_high_f32(_q0), 1);
          _q11 = vmlaq_lane_f32(_q11, _q5, vget_high_f32(_q0), 1);
          _q10 = vmlaq_lane_f32(_q10, _q8, vget_high_f32(_q1), 0);
          _q11 = vmlaq_lane_f32(_q11, _q7, vget_high_f32(_q1), 0);
          _q10 = vmulq_lane_f32(_q10, vget_low_f32(_q1), 0);
          _q12 = vaddq_f32(_q10, _q11);
          vst1q_lane_f32(out5, _q12, 0);
          vst1q_lane_f32(out5 + steps, _q12, 1);
          vst1q_lane_f32(out5 + 2 * steps, _q12, 2);
          vst1q_lane_f32(out5 + 3 * steps, _q12, 3);

          _q12 = vsubq_f32(_q10, _q11);
          vst1q_lane_f32(out6, _q12, 0);
          vst1q_lane_f32(out6 + steps, _q12, 1);
          vst1q_lane_f32(out6 + 2 * steps, _q12, 2);
          vst1q_lane_f32(out6 + 3 * steps, _q12, 3);

          _q10 = vsubq_f32(_q9, _q3);
          _q11 = vsubq_f32(_q5, _q7);
          _q10 = vmlaq_lane_f32(_q10, _q11, vget_low_f32(_q0), 0);
          vst1q_lane_f32(out7, _q10, 0);
          vst1q_lane_f32(out7 + steps, _q10, 1);
          vst1q_lane_f32(out7 + 2 * steps, _q10, 2);
          vst1q_lane_f32(out7 + 3 * steps, _q10, 3);

          ptr0 += 16;
          ptr1 += 16;
          out0 += 4 * steps;
          out1 += 4 * steps;
          out2 += 4 * steps;
          out3 += 4 * steps;
          out4 += 4 * steps;
          out5 += 4 * steps;
          out6 += 4 * steps;
          out7 += 4 * steps;
        }
#else
        steps = 8 * channel * 8 * sizeof(float);
        asm volatile(
            "mov        r0, #2                          \n"
            "vld1.32    {d0-d3}, [%[tm_ptr]]            \n"
            // row loop
            "loop_r_%=:                                 \n"
            "vld1.32    {d4-d7}, [%[ptr0]]!             \n"  // q2: d0, q3: d1
            "vld1.32    {d8-d11}, [%[ptr0]]!            \n"  // q4: d2, q5: d3
            "vld1.32    {d12-d15}, [%[ptr1]]!           \n"  // q6: d4, q7: d5
            "vld1.32    {d16-d19}, [%[ptr1]]!           \n"  // q8: d6, q9: d7
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
            "vst1.32    {d24[0]}, [%[out1]], %[steps]   \n"
            "vst1.32    {d24[1]}, [%[out1]], %[steps]   \n"
            "vst1.32    {d25[0]}, [%[out1]], %[steps]   \n"
            "vst1.32    {d25[1]}, [%[out1]], %[steps]   \n"
            "vsub.f32   q12, q10, q11                   \n"
            "vst1.32    {d24[0]}, [%[out2]], %[steps]   \n"
            "vst1.32    {d24[1]}, [%[out2]], %[steps]   \n"
            "vst1.32    {d25[0]}, [%[out2]], %[steps]   \n"
            "vst1.32    {d25[1]}, [%[out2]], %[steps]   \n"

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
            "vst1.32    {d24[0]}, [%[out3]], %[steps]   \n"
            "vst1.32    {d24[1]}, [%[out3]], %[steps]   \n"
            "vst1.32    {d25[0]}, [%[out3]], %[steps]   \n"
            "vst1.32    {d25[1]}, [%[out3]], %[steps]   \n"
            "vsub.f32   q12, q10, q11                   \n"
            "vst1.32    {d24[0]}, [%[out4]], %[steps]   \n"
            "vst1.32    {d24[1]}, [%[out4]], %[steps]   \n"
            "vst1.32    {d25[0]}, [%[out4]], %[steps]   \n"
            "vst1.32    {d25[1]}, [%[out4]], %[steps]   \n"

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
            "vst1.32    {d24[0]}, [%[out5]], %[steps]   \n"
            "vst1.32    {d24[1]}, [%[out5]], %[steps]   \n"
            "vst1.32    {d25[0]}, [%[out5]], %[steps]   \n"
            "vst1.32    {d25[1]}, [%[out5]], %[steps]   \n"
            "vsub.f32   q12, q10, q11                   \n"
            "vst1.32    {d24[0]}, [%[out6]], %[steps]   \n"
            "vst1.32    {d24[1]}, [%[out6]], %[steps]   \n"
            "vst1.32    {d25[0]}, [%[out6]], %[steps]   \n"
            "vst1.32    {d25[1]}, [%[out6]], %[steps]   \n"

            "vsub.f32   q10, q9, q3                     \n"
            "vsub.f32   q11, q5, q7                     \n"
            "vmla.f32   q10, q11, d0[0]                 \n"
            "vst1.32    {d20[0]}, [%[out7]], %[steps]   \n"
            "vst1.32    {d20[1]}, [%[out7]], %[steps]   \n"
            "vst1.32    {d21[0]}, [%[out7]], %[steps]   \n"
            "vst1.32    {d21[1]}, [%[out7]], %[steps]   \n"

            "subs       r0, #1                          \n"
            "bne        loop_r_%=                       \n"
            : [out0] "+r"(out0), [out1] "+r"(out1), [out2] "+r"(out2),
              [out3] "+r"(out3), [out4] "+r"(out4), [out5] "+r"(out5),
              [out6] "+r"(out6), [out7] "+r"(out7), [ptr0] "+r"(ptr0),
              [ptr1] "+r"(ptr1)
            : [tm_ptr] "r"((float *)transform_matrix), [steps] "r"(steps)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12", "q13", "r0");
#endif  // __aarch64__
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
  int tiles = input.dims()[0];
  int in_channel = input.dims()[2];
  int out_channel = weight.dims()[0];

  // compute U*V first
  framework::Tensor uv_trans;
  framework::DDim shape =
      framework::make_ddim(std::vector<int>{out_channel, tiles, 64, 32});
  float *uv_trans_ptr = uv_trans.mutable_data<float>(shape);
  const float *input_ptr = input.data<float>();
  const float *weight_ptr = weight.data<float>();

  #pragma omp parallel for
  for (int i = 0; i < out_channel; ++i) {
    float *uv_ptr = uv_trans_ptr + (i * tiles * 64 * 32);
    for (int j = 0; j < tiles; ++j) {
      for (int k = 0; k < 64; ++k) {
        const float *w_ptr = weight_ptr + (i * 64 + k) * in_channel * 4;
        const float *in_ptr = input_ptr + (j * 64 + k) * in_channel * 8;
        int inter_channel = in_channel >> 1;
        int remain_channel = in_channel & 0x1;
#if __aarch64__
        asm volatile(
            "dup        v8.4s,     wzr                 \n"
            "dup        v9.4s,     wzr                 \n"
            "dup        v10.4s,    wzr                 \n"
            "dup        v11.4s,    wzr                 \n"
            "dup        v12.4s,    wzr                 \n"
            "dup        v13.4s,    wzr                 \n"
            "dup        v14.4s,    wzr                 \n"
            "dup        v15.4s,    wzr                 \n"

            "cmp        %[inter], #0                       \n"
            "ble        2f                                 \n"
            // loop 2 channels
            "1:                                            \n"
            "ld1        {v0.4s, v1.4s}, [%[w_ptr]], #32    \n"
            "ld1        {v2.4s, v3.4s}, [%[in_ptr]], #32   \n"
            "ld1        {v4.4s, v5.4s}, [%[in_ptr]], #32   \n"

            "fmla       v8.4s, v2.4s, v0.s[0]              \n"
            "fmla       v9.4s, v3.4s, v0.s[0]              \n"
            "fmla       v10.4s, v2.4s, v0.s[1]             \n"
            "fmla       v11.4s, v3.4s, v0.s[1]             \n"
            "fmla       v12.4s, v2.4s, v0.s[2]             \n"
            "fmla       v13.4s, v3.4s, v0.s[2]             \n"
            "fmla       v14.4s, v2.4s, v0.s[3]             \n"
            "fmla       v15.4s, v3.4s, v0.s[3]             \n"

            "fmla       v8.4s, v4.4s, v1.s[0]              \n"
            "fmla       v9.4s, v5.4s, v1.s[0]              \n"
            "fmla       v10.4s, v4.4s, v1.s[1]             \n"
            "fmla       v11.4s, v5.4s, v1.s[1]             \n"
            "fmla       v12.4s, v4.4s, v1.s[2]             \n"
            "fmla       v13.4s, v5.4s, v1.s[2]             \n"
            "fmla       v14.4s, v4.4s, v1.s[3]             \n"
            "fmla       v15.4s, v5.4s, v1.s[3]             \n"

            "subs       %[inter], %[inter], #1             \n"
            "bne        1b                                 \n"

            // loop 1 channel
            "2:                                            \n"
            "cmp        %[remain], #0                      \n"
            "ble        3f                                 \n"

            "ld1        {v0.4s, v1.4s}, [%[w_ptr]], #32    \n"
            "ld1        {v2.4s, v3.4s}, [%[in_ptr]], #32   \n"
            "fmla       v8.4s, v2.4s, v0.s[0]              \n"
            "fmla       v9.4s, v3.4s, v0.s[0]              \n"
            "fmla       v10.4s, v2.4s, v0.s[1]             \n"
            "fmla       v11.4s, v3.4s, v0.s[1]             \n"
            "fmla       v12.4s, v2.4s, v0.s[2]             \n"
            "fmla       v13.4s, v3.4s, v0.s[2]             \n"
            "fmla       v14.4s, v2.4s, v0.s[3]             \n"
            "fmla       v15.4s, v3.4s, v0.s[3]             \n"

            "3:                                            \n"
            "st1        {v8.4s, v9.4s, v10.4s, v11.4s}, [%[uv_ptr]], #64 \n"
            "st1        {v12.4s, v13.4s, v14.4s, v15.4s}, [%[uv_ptr]], #64 \n"
            : [w_ptr] "+r"(w_ptr), [in_ptr] "+r"(in_ptr), [uv_ptr] "+r"(uv_ptr),
              [inter] "+r"(inter_channel)
            : [remain] "r"(remain_channel)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
#else
        asm volatile(
            "veor       q8, q8, q8                     \n"
            "veor       q9, q9, q9                     \n"
            "veor       q10, q10, q10                  \n"
            "veor       q11, q11, q11                  \n"
            "veor       q12, q12, q12                  \n"
            "veor       q13, q13, q13                  \n"
            "veor       q14, q14, q14                  \n"
            "veor       q15, q15, q15                  \n"

            "cmp        %[inter_channel], #0           \n"
            "ble        loop_1c_%=                     \n"
            // loop 2 channels
            "loop_2c_%=:                               \n"
            "vld1.32    {d0-d3}, [%[w_ptr]]!           \n"
            "vld1.32    {d4-d7}, [%[in_ptr]]!          \n"
            "vld1.32    {d8-d11}, [%[in_ptr]]!         \n"
            "vmla.f32   q8, q2, d0[0]                  \n"
            "vmla.f32   q9, q3, d0[0]                  \n"
            "vmla.f32   q10, q2, d0[1]                 \n"
            "vmla.f32   q11, q3, d0[1]                 \n"
            "vmla.f32   q12, q2, d1[0]                 \n"
            "vmla.f32   q13, q3, d1[0]                 \n"
            "vmla.f32   q14, q2, d1[1]                 \n"
            "vmla.f32   q15, q3, d1[1]                 \n"

            "vmla.f32   q8, q4, d2[0]                  \n"
            "vmla.f32   q9, q5, d2[0]                  \n"
            "vmla.f32   q10, q4, d2[1]                 \n"
            "vmla.f32   q11, q5, d2[1]                 \n"
            "vmla.f32   q12, q4, d3[0]                 \n"
            "vmla.f32   q13, q5, d3[0]                 \n"
            "vmla.f32   q14, q4, d3[1]                 \n"
            "vmla.f32   q15, q5, d3[1]                 \n"

            "subs       %[inter_channel], #1           \n"
            "bne        loop_2c_%=                     \n"

            // loop 1 channel
            "loop_1c_%=:                               \n"
            "cmp        %[remain_channel], #0          \n"
            "ble        store_res_%=                   \n"

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

            "store_res_%=:                             \n"
            "vst1.32    {d16-d19}, [%[uv_ptr]]!        \n"
            "vst1.32    {d20-d23}, [%[uv_ptr]]!        \n"
            "vst1.32    {d24-d27}, [%[uv_ptr]]!        \n"
            "vst1.32    {d28-d31}, [%[uv_ptr]]!        \n"
            : [w_ptr] "+r"(w_ptr), [in_ptr] "+r"(in_ptr), [uv_ptr] "+r"(uv_ptr),
              [inter_channel] "+r"(inter_channel)
            : [remain_channel] "r"(remain_channel)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif  // __aarch64__
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
  float transform_matrix[8] = {2.f, 4.f, 8.f, 16.f};

  #pragma omp parallel for
  for (int oc = 0; oc < output->dims()[1]; ++oc) {
    float at_m[48];        // [6][8]
    float output_tmp[36];  // [6][6], temporarily restore results
    // (oc / 4) * tiles * 64 * 32 + (oc & 0x3) * 8
    const float *uv_ptr =
        uv_trans_ptr + (oc >> 2) * tiles * 64 * 32 + (oc & 0x3) * 8;
    for (int tile_h = 0; tile_h < h_tiles; ++tile_h) {
      for (int tile_w = 0; tile_w < w_tiles; ++tile_w) {
        float *at_m_ptr = at_m;
        int tile_indics = tile_h * w_tiles + tile_w;
        int tile_block = tile_indics >> 3;
        int block_indics = tile_indics & 0x7;
        const float *uv_ptr0 = uv_ptr + tile_block * 64 * 32 + block_indics;
#if __aarch64__
        float32x4_t _q0 = vld1q_f32(transform_matrix);
        for (int l = 0; l < 2; ++l) {
          float32x4_t _q1, _q2, _q3, _q4, _q5, _q6, _q7, _q8;
          _q1 = vsetq_lane_f32(*uv_ptr0, _q1, 0);
          uv_ptr0 += 32;
          _q3 = vsetq_lane_f32(*uv_ptr0, _q3, 0);
          uv_ptr0 += 32;
          _q5 = vsetq_lane_f32(*uv_ptr0, _q5, 0);
          uv_ptr0 += 32;
          _q7 = vsetq_lane_f32(*uv_ptr0, _q7, 0);
          uv_ptr0 += 32;
          _q2 = vsetq_lane_f32(*uv_ptr0, _q2, 0);
          uv_ptr0 += 32;
          _q4 = vsetq_lane_f32(*uv_ptr0, _q4, 0);
          uv_ptr0 += 32;
          _q6 = vsetq_lane_f32(*uv_ptr0, _q6, 0);
          uv_ptr0 += 32;
          _q8 = vsetq_lane_f32(*uv_ptr0, _q8, 0);
          uv_ptr0 += 32;

          _q1 = vsetq_lane_f32(*uv_ptr0, _q1, 1);
          uv_ptr0 += 32;
          _q3 = vsetq_lane_f32(*uv_ptr0, _q3, 1);
          uv_ptr0 += 32;
          _q5 = vsetq_lane_f32(*uv_ptr0, _q5, 1);
          uv_ptr0 += 32;
          _q7 = vsetq_lane_f32(*uv_ptr0, _q7, 1);
          uv_ptr0 += 32;
          _q2 = vsetq_lane_f32(*uv_ptr0, _q2, 1);
          uv_ptr0 += 32;
          _q4 = vsetq_lane_f32(*uv_ptr0, _q4, 1);
          uv_ptr0 += 32;
          _q6 = vsetq_lane_f32(*uv_ptr0, _q6, 1);
          uv_ptr0 += 32;
          _q8 = vsetq_lane_f32(*uv_ptr0, _q8, 1);
          uv_ptr0 += 32;

          _q1 = vsetq_lane_f32(*uv_ptr0, _q1, 2);
          uv_ptr0 += 32;
          _q3 = vsetq_lane_f32(*uv_ptr0, _q3, 2);
          uv_ptr0 += 32;
          _q5 = vsetq_lane_f32(*uv_ptr0, _q5, 2);
          uv_ptr0 += 32;
          _q7 = vsetq_lane_f32(*uv_ptr0, _q7, 2);
          uv_ptr0 += 32;
          _q2 = vsetq_lane_f32(*uv_ptr0, _q2, 2);
          uv_ptr0 += 32;
          _q4 = vsetq_lane_f32(*uv_ptr0, _q4, 2);
          uv_ptr0 += 32;
          _q6 = vsetq_lane_f32(*uv_ptr0, _q6, 2);
          uv_ptr0 += 32;
          _q8 = vsetq_lane_f32(*uv_ptr0, _q8, 2);
          uv_ptr0 += 32;

          _q1 = vsetq_lane_f32(*uv_ptr0, _q1, 3);
          uv_ptr0 += 32;
          _q3 = vsetq_lane_f32(*uv_ptr0, _q3, 3);
          uv_ptr0 += 32;
          _q5 = vsetq_lane_f32(*uv_ptr0, _q5, 3);
          uv_ptr0 += 32;
          _q7 = vsetq_lane_f32(*uv_ptr0, _q7, 3);
          uv_ptr0 += 32;
          _q2 = vsetq_lane_f32(*uv_ptr0, _q2, 3);
          uv_ptr0 += 32;
          _q4 = vsetq_lane_f32(*uv_ptr0, _q4, 3);
          uv_ptr0 += 32;
          _q6 = vsetq_lane_f32(*uv_ptr0, _q6, 3);
          uv_ptr0 += 32;
          _q8 = vsetq_lane_f32(*uv_ptr0, _q8, 3);
          uv_ptr0 += 32;

          float32x4_t _q9 = vaddq_f32(_q3, _q5);
          float32x4_t _q10 = vaddq_f32(_q7, _q2);
          float32x4_t _q11 = vaddq_f32(_q4, _q6);
          float32x4_t _q12 = vsubq_f32(_q3, _q5);
          float32x4_t _q13 = vsubq_f32(_q7, _q2);
          float32x4_t _q14 = vsubq_f32(_q4, _q6);
          _q2 = vmulq_lane_f32(_q13, vget_low_f32(_q0), 0);
          _q3 = vmulq_lane_f32(_q11, vget_low_f32(_q0), 0);

          float32x4_t _q15 = vaddq_f32(_q1, _q9);
          _q15 = vaddq_f32(_q15, _q10);
          _q15 = vmlaq_lane_f32(_q15, _q3, vget_high_f32(_q0), 1);
          vst1q_f32(at_m_ptr, _q15);

          _q15 = vaddq_f32(_q12, _q2);
          _q15 = vmlaq_lane_f32(_q15, _q14, vget_high_f32(_q0), 1);
          vst1q_f32(at_m_ptr + 4, _q15);

          _q15 = vmlaq_lane_f32(_q9, _q10, vget_low_f32(_q0), 1);
          _q15 = vmlaq_lane_f32(_q15, _q11, vget_high_f32(_q0), 0);
          vst1q_f32(at_m_ptr + 8, _q15);

          _q15 = vmlaq_lane_f32(_q12, _q13, vget_high_f32(_q0), 0);
          _q15 = vmlaq_lane_f32(_q15, _q14, vget_low_f32(_q0), 1);
          vst1q_f32(at_m_ptr + 12, _q15);

          _q15 = vaddq_f32(_q9, _q3);
          _q15 = vmlaq_lane_f32(_q15, _q10, vget_high_f32(_q0), 1);
          vst1q_f32(at_m_ptr + 16, _q15);

          _q15 = vaddq_f32(_q12, _q8);
          _q15 = vaddq_f32(_q15, _q14);
          _q15 = vmlaq_lane_f32(_q15, _q2, vget_high_f32(_q0), 1);
          vst1q_f32(at_m_ptr + 20, _q15);

          at_m_ptr += 24;
        }
#else
        int steps = 32 * sizeof(float);
        asm volatile(
            "vld1.32    {d0-d1}, [%[tm_ptr]]              \n"
            "mov        r0, #2                            \n"

            "loop_%=:                                     \n"
            "vld1.32    {d2[0]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d6[0]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d10[0]}, [%[uv_ptr0]], %[steps]  \n"
            "vld1.32    {d14[0]}, [%[uv_ptr0]], %[steps]  \n"
            "vld1.32    {d4[0]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d8[0]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d12[0]}, [%[uv_ptr0]], %[steps]  \n"
            "vld1.32    {d16[0]}, [%[uv_ptr0]], %[steps]  \n"

            "vld1.32    {d2[1]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d6[1]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d10[1]}, [%[uv_ptr0]], %[steps]  \n"
            "vld1.32    {d14[1]}, [%[uv_ptr0]], %[steps]  \n"
            "vld1.32    {d4[1]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d8[1]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d12[1]}, [%[uv_ptr0]], %[steps]  \n"
            "vld1.32    {d16[1]}, [%[uv_ptr0]], %[steps]  \n"

            "vld1.32    {d3[0]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d7[0]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d11[0]}, [%[uv_ptr0]], %[steps]  \n"
            "vld1.32    {d15[0]}, [%[uv_ptr0]], %[steps]  \n"
            "vld1.32    {d5[0]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d9[0]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d13[0]}, [%[uv_ptr0]], %[steps]  \n"
            "vld1.32    {d17[0]}, [%[uv_ptr0]], %[steps]  \n"

            "vld1.32    {d3[1]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d7[1]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d11[1]}, [%[uv_ptr0]], %[steps]  \n"
            "vld1.32    {d15[1]}, [%[uv_ptr0]], %[steps]  \n"
            "vld1.32    {d5[1]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d9[1]}, [%[uv_ptr0]], %[steps]   \n"
            "vld1.32    {d13[1]}, [%[uv_ptr0]], %[steps]  \n"
            "vld1.32    {d17[1]}, [%[uv_ptr0]], %[steps]  \n"

            "vadd.f32   q9, q3, q5                     \n"  // m1 + m2
            "vadd.f32   q10, q7, q2                    \n"  // m3 + m4
            "vadd.f32   q11, q4, q6                    \n"  // m5 + m6
            "vsub.f32   q12, q3, q5                    \n"  // m1 - m2
            "vsub.f32   q13, q7, q2                    \n"  // m3 - m4
            "vsub.f32   q14, q4, q6                    \n"  // m5 - m6
            "vmul.f32   q2, q13, d0[0]                 \n"  // 2 * (m3 - m4)
            "vmul.f32   q3, q11, d0[0]                 \n"  // 2 * (m5 + m6)

            "vadd.f32   q15, q1, q9                    \n"
            "vadd.f32   q15, q15, q10                  \n"
            "vmla.f32   q15, q3, d1[1]                 \n"
            "vst1.32    {d30-d31}, [%[at_m_ptr]]!      \n"

            "vadd.f32   q15, q12, q2                   \n"
            "vmla.f32   q15, q14, d1[1]                \n"
            "vst1.32    {d30-d31}, [%[at_m_ptr]]!      \n"

            "vmov.32    q15, q9                        \n"
            "vmla.f32   q15, q10, d0[1]                \n"
            "vmla.f32   q15, q11, d1[0]                \n"
            "vst1.32    {d30-d31}, [%[at_m_ptr]]!      \n"

            "vmov.32    q15, q12                       \n"
            "vmla.f32   q15, q13, d1[0]                \n"
            "vmla.f32   q15, q14, d0[1]                \n"
            "vst1.32    {d30-d31}, [%[at_m_ptr]]!      \n"

            "vadd.f32   q15, q9, q3                    \n"
            "vmla.f32   q15, q10, d1[1]                \n"
            "vst1.32    {d30-d31}, [%[at_m_ptr]]!      \n"

            "vadd.f32   q15, q12, q8                   \n"
            "vadd.f32   q15, q15, q14                  \n"
            "vmla.f32   q15, q2, d1[1]                 \n"
            "vst1.32    {d30-d31}, [%[at_m_ptr]]!      \n"

            "subs       r0, #1                         \n"
            "bne        loop_%=                        \n"
            : [uv_ptr0] "+r"(uv_ptr0), [at_m_ptr] "+r"(at_m_ptr)
            : [tm_ptr] "r"((float *)transform_matrix), [steps] "r"(steps)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "r0");
#endif  // __aarch64__

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
#if __aarch64__
          float32x4_t _q0 = vld1q_f32(transform_matrix);
          float32x4x2_t _q23, _q45, _q67, _q89;
          _q23.val[0] = vld1q_f32(at_m_ptr0);
          _q23.val[1] = vld1q_f32(at_m_ptr0 + 4);
          _q45.val[0] = vld1q_f32(at_m_ptr0 + 8);
          _q45.val[1] = vld1q_f32(at_m_ptr0 + 12);
          _q67.val[0] = vld1q_f32(at_m_ptr1);
          _q67.val[1] = vld1q_f32(at_m_ptr1 + 4);
          _q89.val[0] = vld1q_f32(at_m_ptr1 + 8);
          _q89.val[1] = vld1q_f32(at_m_ptr1 + 12);
          _q23 = vtrnq_f32(_q23.val[0], _q23.val[1]);
          _q45 = vtrnq_f32(_q45.val[0], _q45.val[1]);
          _q67 = vtrnq_f32(_q67.val[0], _q67.val[1]);
          _q89 = vtrnq_f32(_q89.val[0], _q89.val[1]);
          float32x4_t _q1 = vcombine_f32(vget_low_f32(_q23.val[0]),
                                         vget_low_f32(_q45.val[0]));
          float32x4_t _q3 = vcombine_f32(vget_high_f32(_q23.val[0]),
                                         vget_high_f32(_q45.val[0]));
          float32x4_t _q2 = vcombine_f32(vget_low_f32(_q23.val[1]),
                                         vget_low_f32(_q45.val[1]));
          float32x4_t _q4 = vcombine_f32(vget_high_f32(_q23.val[1]),
                                         vget_high_f32(_q45.val[1]));
          float32x4_t _q5 = vcombine_f32(vget_low_f32(_q67.val[0]),
                                         vget_low_f32(_q89.val[0]));
          float32x4_t _q7 = vcombine_f32(vget_high_f32(_q67.val[0]),
                                         vget_high_f32(_q89.val[0]));
          float32x4_t _q6 = vcombine_f32(vget_low_f32(_q67.val[1]),
                                         vget_low_f32(_q89.val[1]));
          float32x4_t _q8 = vcombine_f32(vget_high_f32(_q67.val[1]),
                                         vget_high_f32(_q89.val[1]));

          float32x4_t _q9 = vaddq_f32(_q2, _q3);
          float32x4_t _q10 = vaddq_f32(_q4, _q5);
          float32x4_t _q11 = vaddq_f32(_q6, _q7);
          float32x4_t _q12 = vsubq_f32(_q2, _q3);
          float32x4_t _q13 = vsubq_f32(_q4, _q5);
          float32x4_t _q14 = vsubq_f32(_q6, _q7);
          _q6 = vmulq_lane_f32(_q13, vget_low_f32(_q0), 0);
          _q7 = vmulq_lane_f32(_q11, vget_low_f32(_q0), 0);

          _q1 = vaddq_f32(_q1, _q9);
          _q1 = vaddq_f32(_q1, _q10);
          _q1 = vmlaq_lane_f32(_q1, _q7, vget_high_f32(_q0), 1);

          _q2 = vaddq_f32(_q12, _q6);
          _q2 = vmlaq_lane_f32(_q2, _q14, vget_high_f32(_q0), 1);

          _q3 = vmlaq_lane_f32(_q9, _q10, vget_low_f32(_q0), 1);
          _q3 = vmlaq_lane_f32(_q3, _q11, vget_high_f32(_q0), 0);

          _q4 = vmlaq_lane_f32(_q12, _q13, vget_high_f32(_q0), 0);
          _q4 = vmlaq_lane_f32(_q4, _q14, vget_low_f32(_q0), 1);

          _q23 = vtrnq_f32(_q1, _q2);
          _q45 = vtrnq_f32(_q3, _q4);
          vst1_f32(out_ptr0, vget_low_f32(_q23.val[0]));
          vst1_f32(out_ptr0 + 2, vget_low_f32(_q45.val[0]));
          vst1_f32(out_ptr1, vget_low_f32(_q23.val[1]));
          vst1_f32(out_ptr1 + 2, vget_low_f32(_q45.val[1]));
          vst1_f32(out_ptr2, vget_high_f32(_q23.val[0]));
          vst1_f32(out_ptr2 + 2, vget_high_f32(_q45.val[0]));
          vst1_f32(out_ptr3, vget_high_f32(_q23.val[1]));
          vst1_f32(out_ptr3 + 2, vget_high_f32(_q45.val[1]));

          _q1 = vaddq_f32(_q9, _q7);
          _q1 = vmlaq_lane_f32(_q1, _q10, vget_high_f32(_q0), 1);
          _q2 = vaddq_f32(_q12, _q8);
          _q2 = vaddq_f32(_q2, _q14);
          _q2 = vmlaq_lane_f32(_q2, _q6, vget_high_f32(_q0), 1);
          _q23 = vtrnq_f32(_q1, _q2);
          vst1_f32(out_ptr0 + 4, vget_low_f32(_q23.val[0]));
          vst1_f32(out_ptr1 + 4, vget_low_f32(_q23.val[1]));
          vst1_f32(out_ptr2 + 4, vget_high_f32(_q23.val[0]));
          vst1_f32(out_ptr3 + 4, vget_high_f32(_q23.val[1]));

          // remain 2 rows
          _q1 = vld1q_f32(at_m_ptr0 + 16);
          _q2 = vld1q_f32(at_m_ptr0 + 20);
          _q3 = vld1q_f32(at_m_ptr1 + 16);
          _q4 = vld1q_f32(at_m_ptr1 + 20);
          _q23 = vtrnq_f32(_q1, _q2);
          _q45 = vtrnq_f32(_q3, _q4);

          float32x2_t _d2 = vget_low_f32(_q23.val[0]);
          float32x2_t _d3 = vget_high_f32(_q23.val[0]);
          float32x2_t _d4 = vget_low_f32(_q23.val[1]);
          float32x2_t _d5 = vget_high_f32(_q23.val[1]);
          float32x2_t _d6 = vget_low_f32(_q45.val[0]);
          float32x2_t _d7 = vget_high_f32(_q45.val[0]);
          float32x2_t _d8 = vget_low_f32(_q45.val[1]);
          float32x2_t _d9 = vget_high_f32(_q45.val[1]);

          float32x2_t _d10 = vadd_f32(_d4, _d3);
          float32x2_t _d11 = vadd_f32(_d5, _d6);
          float32x2_t _d12 = vadd_f32(_d8, _d7);
          float32x2_t _d13 = vsub_f32(_d4, _d3);
          float32x2_t _d14 = vsub_f32(_d5, _d6);
          float32x2_t _d15 = vsub_f32(_d8, _d7);
          float32x2_t _d16 = vmul_lane_f32(_d14, vget_low_f32(_q0), 0);
          float32x2_t _d17 = vmul_lane_f32(_d12, vget_low_f32(_q0), 0);

          float32x2_t _d18 = vadd_f32(_d2, _d10);
          float32x2_t _d20 = vadd_f32(_d13, _d16);
          float32x2_t _d19 = vmla_lane_f32(_d10, _d11, vget_low_f32(_q0), 1);
          float32x2_t _d21 = vmla_lane_f32(_d13, _d14, vget_high_f32(_q0), 0);
          _d18 = vadd_f32(_d18, _d11);
          _d18 = vmla_lane_f32(_d18, _d17, vget_high_f32(_q0), 1);
          _d20 = vmla_lane_f32(_d20, _d15, vget_high_f32(_q0), 1);
          _d19 = vmla_lane_f32(_d19, _d12, vget_high_f32(_q0), 0);
          _d21 = vmla_lane_f32(_d21, _d15, vget_low_f32(_q0), 1);

          float32x2x2_t _d18d20 = vtrn_f32(_d18, _d20);
          float32x2x2_t _d19d21 = vtrn_f32(_d19, _d21);
          vst1_f32(out_ptr4, _d18d20.val[0]);
          vst1_f32(out_ptr4 + 2, _d19d21.val[0]);
          vst1_f32(out_ptr5, _d18d20.val[1]);
          vst1_f32(out_ptr5 + 2, _d19d21.val[1]);

          _d18 = vadd_f32(_d10, _d17);
          _d18 = vmla_lane_f32(_d18, _d11, vget_high_f32(_q0), 1);
          _d20 = vadd_f32(_d13, _d9);
          _d20 = vadd_f32(_d20, _d15);
          _d20 = vmla_lane_f32(_d20, _d16, vget_high_f32(_q0), 1);
          _d18d20 = vtrn_f32(_d18, _d20);
          vst1_f32(out_ptr4 + 4, _d18d20.val[0]);
          vst1_f32(out_ptr5 + 4, _d18d20.val[1]);
#else
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
              "vadd.f32   q1, q1, q10                  \n"
              "vmla.f32   q1, q7, d1[1]                \n"

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
              "vadd.f32   q2, q2, q14                  \n"
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
              "vmla.f32   d19, d12, d1[0]               \n"

              "vmov.32    d21, d13                      \n"
              "vmla.f32   d21, d14, d1[0]               \n"
              "vmla.f32   d21, d15, d0[1]               \n"

              "vtrn.32    d18, d20                      \n"
              "vtrn.32    d19, d21                      \n"
              "vst1.32    {d18-d19}, [%[out_ptr4]]!     \n"
              "vst1.32    {d20-d21}, [%[out_ptr5]]!     \n"

              "vadd.f32   d18, d10, d17                 \n"
              "vmla.f32   d18, d11, d1[1]               \n"

              "vadd.f32   d19, d13, d9                  \n"
              "vadd.f32   d19, d19, d15                 \n"
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
#endif  // __aarch64__
          size_t offset = (oc * out_h + 6 * tile_h) * out_w + 6 * tile_w;
          float *out_ptr = output_ptr + offset;
          int remain_row = out_h - 6 * tile_h;
          int remain_col = out_w - 6 * tile_w;
          remain_row = (remain_row > 6) ? 6 : remain_row;
          remain_col = (remain_col > 6) ? 6 : remain_col;
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
#if __aarch64__
          float32x4_t _q0 = vld1q_f32(transform_matrix);
          float32x4x2_t _q23, _q45, _q67, _q89;
          _q23.val[0] = vld1q_f32(at_m_ptr0);
          _q23.val[1] = vld1q_f32(at_m_ptr0 + 4);
          _q45.val[0] = vld1q_f32(at_m_ptr0 + 8);
          _q45.val[1] = vld1q_f32(at_m_ptr0 + 12);
          _q67.val[0] = vld1q_f32(at_m_ptr1);
          _q67.val[1] = vld1q_f32(at_m_ptr1 + 4);
          _q89.val[0] = vld1q_f32(at_m_ptr1 + 8);
          _q89.val[1] = vld1q_f32(at_m_ptr1 + 12);
          _q23 = vtrnq_f32(_q23.val[0], _q23.val[1]);
          _q45 = vtrnq_f32(_q45.val[0], _q45.val[1]);
          _q67 = vtrnq_f32(_q67.val[0], _q67.val[1]);
          _q89 = vtrnq_f32(_q89.val[0], _q89.val[1]);
          float32x4_t _q1 = vcombine_f32(vget_low_f32(_q23.val[0]),
                                         vget_low_f32(_q45.val[0]));
          float32x4_t _q3 = vcombine_f32(vget_high_f32(_q23.val[0]),
                                         vget_high_f32(_q45.val[0]));
          float32x4_t _q2 = vcombine_f32(vget_low_f32(_q23.val[1]),
                                         vget_low_f32(_q45.val[1]));
          float32x4_t _q4 = vcombine_f32(vget_high_f32(_q23.val[1]),
                                         vget_high_f32(_q45.val[1]));
          float32x4_t _q5 = vcombine_f32(vget_low_f32(_q67.val[0]),
                                         vget_low_f32(_q89.val[0]));
          float32x4_t _q7 = vcombine_f32(vget_high_f32(_q67.val[0]),
                                         vget_high_f32(_q89.val[0]));
          float32x4_t _q6 = vcombine_f32(vget_low_f32(_q67.val[1]),
                                         vget_low_f32(_q89.val[1]));
          float32x4_t _q8 = vcombine_f32(vget_high_f32(_q67.val[1]),
                                         vget_high_f32(_q89.val[1]));

          float32x4_t _q9 = vaddq_f32(_q2, _q3);
          float32x4_t _q10 = vaddq_f32(_q4, _q5);
          float32x4_t _q11 = vaddq_f32(_q6, _q7);
          float32x4_t _q12 = vsubq_f32(_q2, _q3);
          float32x4_t _q13 = vsubq_f32(_q4, _q5);
          float32x4_t _q14 = vsubq_f32(_q6, _q7);
          _q6 = vmulq_lane_f32(_q13, vget_low_f32(_q0), 0);
          _q7 = vmulq_lane_f32(_q11, vget_low_f32(_q0), 0);

          _q1 = vaddq_f32(_q1, _q9);
          _q1 = vaddq_f32(_q1, _q10);
          _q1 = vmlaq_lane_f32(_q1, _q7, vget_high_f32(_q0), 1);
          _q2 = vaddq_f32(_q12, _q6);
          _q2 = vmlaq_lane_f32(_q2, _q14, vget_high_f32(_q0), 1);
          _q3 = vmlaq_lane_f32(_q9, _q10, vget_low_f32(_q0), 1);
          _q3 = vmlaq_lane_f32(_q3, _q11, vget_high_f32(_q0), 0);
          _q4 = vmlaq_lane_f32(_q12, _q13, vget_high_f32(_q0), 0);
          _q4 = vmlaq_lane_f32(_q4, _q14, vget_low_f32(_q0), 1);

          _q23 = vtrnq_f32(_q1, _q2);
          _q45 = vtrnq_f32(_q3, _q4);
          vst1_f32(out_ptr0, vget_low_f32(_q23.val[0]));
          vst1_f32(out_ptr0 + 2, vget_low_f32(_q45.val[0]));
          vst1_f32(out_ptr1, vget_low_f32(_q23.val[1]));
          vst1_f32(out_ptr1 + 2, vget_low_f32(_q45.val[1]));
          vst1_f32(out_ptr2, vget_high_f32(_q23.val[0]));
          vst1_f32(out_ptr2 + 2, vget_high_f32(_q45.val[0]));
          vst1_f32(out_ptr3, vget_high_f32(_q23.val[1]));
          vst1_f32(out_ptr3 + 2, vget_high_f32(_q45.val[1]));

          _q1 = vaddq_f32(_q9, _q7);
          _q1 = vmlaq_lane_f32(_q1, _q10, vget_high_f32(_q0), 1);
          _q2 = vaddq_f32(_q12, _q8);
          _q2 = vaddq_f32(_q2, _q14);
          _q2 = vmlaq_lane_f32(_q2, _q6, vget_high_f32(_q0), 1);
          _q23 = vtrnq_f32(_q1, _q2);
          vst1_f32(out_ptr0 + 4, vget_low_f32(_q23.val[0]));
          vst1_f32(out_ptr1 + 4, vget_low_f32(_q23.val[1]));
          vst1_f32(out_ptr2 + 4, vget_high_f32(_q23.val[0]));
          vst1_f32(out_ptr3 + 4, vget_high_f32(_q23.val[1]));

          // remain 2 rows
          _q1 = vld1q_f32(at_m_ptr0 + 16);
          _q2 = vld1q_f32(at_m_ptr0 + 20);
          _q3 = vld1q_f32(at_m_ptr1 + 16);
          _q4 = vld1q_f32(at_m_ptr1 + 20);
          _q23 = vtrnq_f32(_q1, _q2);
          _q45 = vtrnq_f32(_q3, _q4);

          float32x2_t _d2 = vget_low_f32(_q23.val[0]);
          float32x2_t _d3 = vget_high_f32(_q23.val[0]);
          float32x2_t _d4 = vget_low_f32(_q23.val[1]);
          float32x2_t _d5 = vget_high_f32(_q23.val[1]);
          float32x2_t _d6 = vget_low_f32(_q45.val[0]);
          float32x2_t _d7 = vget_high_f32(_q45.val[0]);
          float32x2_t _d8 = vget_low_f32(_q45.val[1]);
          float32x2_t _d9 = vget_high_f32(_q45.val[1]);

          float32x2_t _d10 = vadd_f32(_d4, _d3);
          float32x2_t _d11 = vadd_f32(_d5, _d6);
          float32x2_t _d12 = vadd_f32(_d8, _d7);
          float32x2_t _d13 = vsub_f32(_d4, _d3);
          float32x2_t _d14 = vsub_f32(_d5, _d6);
          float32x2_t _d15 = vsub_f32(_d8, _d7);
          float32x2_t _d16 = vmul_lane_f32(_d14, vget_low_f32(_q0), 0);
          float32x2_t _d17 = vmul_lane_f32(_d12, vget_low_f32(_q0), 0);

          float32x2_t _d18 = vadd_f32(_d2, _d10);
          float32x2_t _d20 = vadd_f32(_d13, _d16);
          float32x2_t _d19 = vmla_lane_f32(_d10, _d11, vget_low_f32(_q0), 1);
          float32x2_t _d21 = vmla_lane_f32(_d13, _d14, vget_high_f32(_q0), 0);
          _d18 = vadd_f32(_d18, _d11);
          _d18 = vmla_lane_f32(_d18, _d17, vget_high_f32(_q0), 1);
          _d20 = vmla_lane_f32(_d20, _d15, vget_high_f32(_q0), 1);
          _d19 = vmla_lane_f32(_d19, _d12, vget_high_f32(_q0), 0);
          _d21 = vmla_lane_f32(_d21, _d15, vget_low_f32(_q0), 1);

          float32x2x2_t _d18d20 = vtrn_f32(_d18, _d20);
          float32x2x2_t _d19d21 = vtrn_f32(_d19, _d21);
          vst1_f32(out_ptr4, _d18d20.val[0]);
          vst1_f32(out_ptr4 + 2, _d19d21.val[0]);
          vst1_f32(out_ptr5, _d18d20.val[1]);
          vst1_f32(out_ptr5 + 2, _d19d21.val[1]);

          _d18 = vadd_f32(_d10, _d17);
          _d18 = vmla_lane_f32(_d18, _d11, vget_high_f32(_q0), 1);
          _d20 = vadd_f32(_d13, _d9);
          _d20 = vadd_f32(_d20, _d15);
          _d20 = vmla_lane_f32(_d20, _d16, vget_high_f32(_q0), 1);
          _d18d20 = vtrn_f32(_d18, _d20);
          vst1_f32(out_ptr4 + 4, _d18d20.val[0]);
          vst1_f32(out_ptr5 + 4, _d18d20.val[1]);
#else
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
              "vadd.f32   q1, q1, q10                  \n"
              "vmla.f32   q1, q7, d1[1]                \n"

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
              "vadd.f32   q2, q2, q14                  \n"
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
              "vmla.f32   d19, d12, d1[0]               \n"

              "vmov.32    d21, d13                      \n"
              "vmla.f32   d21, d14, d1[0]               \n"
              "vmla.f32   d21, d15, d0[1]               \n"

              "vtrn.32    d18, d20                      \n"
              "vtrn.32    d19, d21                      \n"
              "vst1.32    {d18-d19}, [%[out_ptr4]]!     \n"
              "vst1.32    {d20-d21}, [%[out_ptr5]]!     \n"

              "vadd.f32   d18, d10, d17                 \n"
              "vmla.f32   d18, d11, d1[1]               \n"

              "vadd.f32   d19, d13, d9                  \n"
              "vadd.f32   d19, d19, d15                 \n"
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
#endif  // __aarch64__
        }
      }
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif  // CONV_OP
#endif  // __ARM_NEON__
