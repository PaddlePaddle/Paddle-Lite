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

#include "operators/math/slidingwindow_conv3x3.h"
#include <vector>
#include "framework/context.h"
#include "operators/math/slidingwindow_utils.h"
#if __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {
template <>
void SlidingwindowConv3x3s1<float, float>(const framework::Tensor *input,
                                          const framework::Tensor *filter,
                                          const std::vector<int> &paddings,
                                          framework::Tensor *output) {
  const int batch = input->dims()[0];
  const int input_ch = input->dims()[1];
  const int input_h = input->dims()[2];
  const int input_w = input->dims()[3];
  const int output_ch = output->dims()[1];
  const int output_h = output->dims()[2];
  const int output_w = output->dims()[3];
  const int padding_h = paddings[0];
  const int padding_w = paddings[1];

  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();
  const float *filter_data = filter->data<float>();

  const int in_ch_size = input_h * input_w;
  const int in_batch_size = input_ch * in_ch_size;
  const int out_ch_size = output_h * output_w;
  const int out_batch_size = output_ch * out_ch_size;
  const int out_size = batch * out_batch_size;
  const int filter_ch_size = 9;
  const int pad_filter_ch_size = (2 * padding_h + 3) * (2 * padding_w + 3);
  const int pad_filter_start =
      2 * padding_h * (2 * padding_w + 3) + 2 * padding_w;
  const int pad_filter_w = 3 + padding_w * 2;
  bool if_nopadding = false;

#if __ARM_NEON
  float *out_ptr = output_data;
  int remain = out_size & 0x3;
  float32x4_t _zero = vdupq_n_f32(0.0);

  for (int i = 0; i < out_size; i += 4) {
    vst1q_f32(out_ptr, _zero);
    out_ptr += 4;
  }
  switch (remain) {
    case 1:
      vst1q_lane_f32(out_ptr, _zero, 0);
      break;
    case 2:
      vst1_f32(out_ptr, vget_low_f32(_zero));
      break;
    case 3:
      vst1_f32(out_ptr, vget_low_f32(_zero));
      vst1q_lane_f32(out_ptr + 2, _zero, 0);
      break;
  }
#else
#pragma omp parallel for
  for (int i = 0; i < out_size; ++i) {
    output_data[i] = 0;
  }
#endif
  if (padding_h == 0 && padding_w == 0) {
    if_nopadding = true;
  }

  for (int b = 0; b < batch; ++b) {
#pragma omp parallel for
    for (int o_c = 0; o_c < output_ch - 1; o_c += 2) {
      bool issamefilter;
      const float *f1;
      const float *f1_c2;
      const float *in_ptr1, *in_ptr2, *in_ptr3, *in_ptr4;
      const float *pad_filter0, *pad_filter1, *pad_filter2, *pad_filter3;
      const float *pad_filter0_c2, *pad_filter1_c2, *pad_filter2_c2,
          *pad_filter3_c2;
      float pad_filter_arr[pad_filter_ch_size];
      float pad_filter_arr_c2[pad_filter_ch_size];

      float *output_data_ch;
      float *output_data_ch_2;
      const float *input_data_ch;
      const float *filter_data_ch;
      const float *filter_data_ch_c2;

      filter_data_ch = filter_data + o_c * filter_ch_size * input_ch;
      filter_data_ch_c2 = filter_data + (o_c + 1) * filter_ch_size * input_ch;

      input_data_ch = input_data;
      output_data_ch = output_data + o_c * out_ch_size;
      output_data_ch_2 = output_data + (o_c + 1) * out_ch_size;

      for (int i_c = 0; i_c < input_ch; ++i_c) {
        f1 = filter_data_ch;
        f1_c2 = filter_data_ch_c2;

        if (!if_nopadding) {
          memset(pad_filter_arr, 0.f, sizeof(pad_filter_arr));
          memset(pad_filter_arr_c2, 0.f, sizeof(pad_filter_arr_c2));
          for (int i = 0; i < 9; i++) {
            int j = i / 3 * (2 * padding_w + 3) + i % 3 + padding_h * 3 +
                    padding_w * (2 * padding_h + 1);
            pad_filter_arr[j] = filter_data_ch[i];
            pad_filter_arr_c2[j] = filter_data_ch_c2[i];
          }
          pad_filter1 = pad_filter_arr;
          pad_filter1 += pad_filter_start;
          pad_filter0 = pad_filter1 - pad_filter_w;
          pad_filter2 = pad_filter1 + pad_filter_w;
          pad_filter3 = pad_filter2 + pad_filter_w;

          pad_filter1_c2 = pad_filter_arr_c2;
          pad_filter1_c2 += pad_filter_start;
          pad_filter0_c2 = pad_filter1_c2 - pad_filter_w;
          pad_filter2_c2 = pad_filter1_c2 + pad_filter_w;
          pad_filter3_c2 = pad_filter2_c2 + pad_filter_w;
        } else {
          pad_filter1 = filter_data_ch;
          pad_filter2 = pad_filter1 + 3;
          pad_filter3 = pad_filter2 + 3;

          pad_filter1_c2 = filter_data_ch_c2;
          pad_filter2_c2 = pad_filter1_c2 + 3;
          pad_filter3_c2 = pad_filter2_c2 + 3;
        }
        float *out_ptr1, *out_ptr2;
        float *out_ptr1_c2, *out_ptr2_c2;

        out_ptr1 = output_data_ch;
        out_ptr2 = out_ptr1 + output_w;
        out_ptr1_c2 = output_data_ch_2;
        out_ptr2_c2 = out_ptr1_c2 + output_w;

        in_ptr1 = input_data_ch;
        in_ptr2 = in_ptr1 + input_w;
        in_ptr3 = in_ptr2 + input_w;
        in_ptr4 = in_ptr3 + input_w;

        int o_h = 0;
        for (; o_h < output_h - 1; o_h = o_h + 2) {
          if (!if_nopadding &&
              (o_h < padding_h || o_h > output_h - padding_h - 2)) {
            issamefilter = false;
          } else {
            issamefilter = true;
          }
          int o_w = 0;
          // pad left
          for (; o_w < padding_w; ++o_w) {
            float sum1 = 0;
            float sum2 = 0;
            float sum1_c2 = 0;
            float sum2_c2 = 0;

            if (issamefilter) {
#if __ARM_NEON
              float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
              float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
              float32x4_t _pad_filter1_c2 = vld1q_f32(pad_filter1_c2);
              float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
              float32x4_t _sum1_c2 = vmulq_f32(_in_ptr1, _pad_filter1_c2);

              float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
              float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
              float32x4_t _pad_filter2_c2 = vld1q_f32(pad_filter2_c2);
              _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
              _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr2, _pad_filter2_c2);
              float32x4_t _sum2 = vmulq_f32(_in_ptr2, _pad_filter1);
              float32x4_t _sum2_c2 = vmulq_f32(_in_ptr2, _pad_filter1_c2);

              float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
              float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
              float32x4_t _pad_filter3_c2 = vld1q_f32(pad_filter3_c2);
              _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
              _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr3, _pad_filter3_c2);
              _sum2 = vmlaq_f32(_sum2, _in_ptr3, _pad_filter2);
              _sum2_c2 = vmlaq_f32(_sum2_c2, _in_ptr3, _pad_filter2_c2);

              float32x4_t _in_ptr4 = vld1q_f32(in_ptr4);
              _sum2 = vmlaq_f32(_sum2, _in_ptr4, _pad_filter3);
              _sum2_c2 = vmlaq_f32(_sum2_c2, _in_ptr4, _pad_filter3_c2);

              _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
              _sum1_c2 = vsetq_lane_f32(sum1_c2, _sum1_c2, 3);
              _sum2 = vsetq_lane_f32(sum2, _sum2, 3);
              _sum2_c2 = vsetq_lane_f32(sum2_c2, _sum2_c2, 3);

              float32x2_t _ss1 =
                  vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
              float32x2_t _ss1_2 =
                  vadd_f32(vget_low_f32(_sum1_c2), vget_high_f32(_sum1_c2));
              float32x2_t _ss2 =
                  vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
              float32x2_t _ss2_2 =
                  vadd_f32(vget_low_f32(_sum2_c2), vget_high_f32(_sum2_c2));
              float32x2_t _ssss1_ssss2 = vpadd_f32(_ss1, _ss2);
              float32x2_t _ssss1_2_ssss2_2 = vpadd_f32(_ss1_2, _ss2_2);

              sum1 += vget_lane_f32(_ssss1_ssss2, 0);
              sum1_c2 += vget_lane_f32(_ssss1_2_ssss2_2, 0);
              sum2 += vget_lane_f32(_ssss1_ssss2, 1);
              sum2_c2 += vget_lane_f32(_ssss1_2_ssss2_2, 1);
#else
              sum1 += in_ptr1[0] * pad_filter1[0];
              sum1 += in_ptr1[1] * pad_filter1[1];
              sum1 += in_ptr1[2] * pad_filter1[2];
              sum1 += in_ptr2[0] * pad_filter2[0];
              sum1 += in_ptr2[1] * pad_filter2[1];
              sum1 += in_ptr2[2] * pad_filter2[2];
              sum1 += in_ptr3[0] * pad_filter3[0];
              sum1 += in_ptr3[1] * pad_filter3[1];
              sum1 += in_ptr3[2] * pad_filter3[2];

              sum2 += in_ptr2[0] * pad_filter1[0];
              sum2 += in_ptr2[1] * pad_filter1[1];
              sum2 += in_ptr2[2] * pad_filter1[2];
              sum2 += in_ptr3[0] * pad_filter2[0];
              sum2 += in_ptr3[1] * pad_filter2[1];
              sum2 += in_ptr3[2] * pad_filter2[2];
              sum2 += in_ptr4[0] * pad_filter3[0];
              sum2 += in_ptr4[1] * pad_filter3[1];
              sum2 += in_ptr4[2] * pad_filter3[2];

              sum1_c2 += in_ptr1[0] * pad_filter1_c2[0];
              sum1_c2 += in_ptr1[1] * pad_filter1_c2[1];
              sum1_c2 += in_ptr1[2] * pad_filter1_c2[2];
              sum1_c2 += in_ptr2[0] * pad_filter2_c2[0];
              sum1_c2 += in_ptr2[1] * pad_filter2_c2[1];
              sum1_c2 += in_ptr2[2] * pad_filter2_c2[2];
              sum1_c2 += in_ptr3[0] * pad_filter3_c2[0];
              sum1_c2 += in_ptr3[1] * pad_filter3_c2[1];
              sum1_c2 += in_ptr3[2] * pad_filter3_c2[2];

              sum2_c2 += in_ptr2[0] * pad_filter1_c2[0];
              sum2_c2 += in_ptr2[1] * pad_filter1_c2[1];
              sum2_c2 += in_ptr2[2] * pad_filter1_c2[2];
              sum2_c2 += in_ptr3[0] * pad_filter2_c2[0];
              sum2_c2 += in_ptr3[1] * pad_filter2_c2[1];
              sum2_c2 += in_ptr3[2] * pad_filter2_c2[2];
              sum2_c2 += in_ptr4[0] * pad_filter3_c2[0];
              sum2_c2 += in_ptr4[1] * pad_filter3_c2[1];
              sum2_c2 += in_ptr4[2] * pad_filter3_c2[2];
#endif
            } else {
#if __ARM_NEON
              float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
              float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
              float32x4_t _pad_filter1_c2 = vld1q_f32(pad_filter1_c2);
              float32x4_t _pad_filter0 = vld1q_f32(pad_filter0);
              float32x4_t _pad_filter0_c2 = vld1q_f32(pad_filter0_c2);

              float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
              float32x4_t _sum1_c2 = vmulq_f32(_in_ptr1, _pad_filter1_c2);
              float32x4_t _sum2 = vmulq_f32(_in_ptr1, _pad_filter0);
              float32x4_t _sum2_c2 = vmulq_f32(_in_ptr1, _pad_filter0_c2);

              float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
              float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
              float32x4_t _pad_filter2_c2 = vld1q_f32(pad_filter2_c2);

              _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
              _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr2, _pad_filter2_c2);
              _sum2 = vmlaq_f32(_sum2, _in_ptr2, _pad_filter1);
              _sum2_c2 = vmlaq_f32(_sum2_c2, _in_ptr2, _pad_filter1_c2);

              float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
              float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
              float32x4_t _pad_filter3_c2 = vld1q_f32(pad_filter3_c2);

              _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
              _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr3, _pad_filter3_c2);
              _sum2 = vmlaq_f32(_sum2, _in_ptr3, _pad_filter2);
              _sum2_c2 = vmlaq_f32(_sum2_c2, _in_ptr3, _pad_filter2_c2);

              _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
              _sum1_c2 = vsetq_lane_f32(sum1_c2, _sum1_c2, 3);
              _sum2 = vsetq_lane_f32(sum2, _sum2, 3);
              _sum2_c2 = vsetq_lane_f32(sum2_c2, _sum2_c2, 3);

              float32x2_t _ss1 =
                  vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
              float32x2_t _ss1_2 =
                  vadd_f32(vget_low_f32(_sum1_c2), vget_high_f32(_sum1_c2));
              float32x2_t _ss2 =
                  vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
              float32x2_t _ss2_2 =
                  vadd_f32(vget_low_f32(_sum2_c2), vget_high_f32(_sum2_c2));
              float32x2_t _ssss1_ssss2 = vpadd_f32(_ss1, _ss2);
              float32x2_t _ssss1_2_ssss2_2 = vpadd_f32(_ss1_2, _ss2_2);

              sum1 += vget_lane_f32(_ssss1_ssss2, 0);
              sum1_c2 += vget_lane_f32(_ssss1_2_ssss2_2, 0);
              sum2 += vget_lane_f32(_ssss1_ssss2, 1);
              sum2_c2 += vget_lane_f32(_ssss1_2_ssss2_2, 1);
#else
              sum1 += in_ptr1[0] * pad_filter1[0];
              sum1 += in_ptr1[1] * pad_filter1[1];
              sum1 += in_ptr1[2] * pad_filter1[2];
              sum1 += in_ptr2[0] * pad_filter2[0];
              sum1 += in_ptr2[1] * pad_filter2[1];
              sum1 += in_ptr2[2] * pad_filter2[2];
              sum1 += in_ptr3[0] * pad_filter3[0];
              sum1 += in_ptr3[1] * pad_filter3[1];
              sum1 += in_ptr3[2] * pad_filter3[2];

              sum2 += in_ptr1[0] * pad_filter0[0];
              sum2 += in_ptr1[1] * pad_filter0[1];
              sum2 += in_ptr1[2] * pad_filter0[2];
              sum2 += in_ptr2[0] * pad_filter1[0];
              sum2 += in_ptr2[1] * pad_filter1[1];
              sum2 += in_ptr2[2] * pad_filter1[2];
              sum2 += in_ptr3[0] * pad_filter2[0];
              sum2 += in_ptr3[1] * pad_filter2[1];
              sum2 += in_ptr3[2] * pad_filter2[2];

              sum1_c2 += in_ptr1[0] * pad_filter1_c2[0];
              sum1_c2 += in_ptr1[1] * pad_filter1_c2[1];
              sum1_c2 += in_ptr1[2] * pad_filter1_c2[2];
              sum1_c2 += in_ptr2[0] * pad_filter2_c2[0];
              sum1_c2 += in_ptr2[1] * pad_filter2_c2[1];
              sum1_c2 += in_ptr2[2] * pad_filter2_c2[2];
              sum1_c2 += in_ptr3[0] * pad_filter3_c2[0];
              sum1_c2 += in_ptr3[1] * pad_filter3_c2[1];
              sum1_c2 += in_ptr3[2] * pad_filter3_c2[2];

              sum2_c2 += in_ptr1[0] * pad_filter0_c2[0];
              sum2_c2 += in_ptr1[1] * pad_filter0_c2[1];
              sum2_c2 += in_ptr1[2] * pad_filter0_c2[2];
              sum2_c2 += in_ptr2[0] * pad_filter1_c2[0];
              sum2_c2 += in_ptr2[1] * pad_filter1_c2[1];
              sum2_c2 += in_ptr2[2] * pad_filter1_c2[2];
              sum2_c2 += in_ptr3[0] * pad_filter2_c2[0];
              sum2_c2 += in_ptr3[1] * pad_filter2_c2[1];
              sum2_c2 += in_ptr3[2] * pad_filter2_c2[2];
#endif
            }
            if (!if_nopadding &&
                (o_w < padding_w || o_w > output_w - padding_w - 2)) {
              pad_filter0--;
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;

              pad_filter0_c2--;
              pad_filter1_c2--;
              pad_filter2_c2--;
              pad_filter3_c2--;
            } else {
              in_ptr1++;
              in_ptr2++;
              in_ptr3++;
              in_ptr4++;
            }
            *out_ptr1 += sum1;
            *out_ptr2 += sum2;
            *out_ptr1_c2 += sum1_c2;
            *out_ptr2_c2 += sum2_c2;

            out_ptr1++;
            out_ptr2++;
            out_ptr1_c2++;
            out_ptr2_c2++;
          }
            // valid
#if __ARM_NEON
#if __aarch64__
          if (issamefilter) {
            int loop = (output_w - 2 * padding_w) >> 2;
            o_w += loop * 4;

            if (loop > 0) {
              asm volatile(
                  "prfm   pldl1keep, [%[f1], #256]          \n\t"
                  "prfm   pldl1keep, [%[f1_c2], #256]       \n\t"

                  "ld1   {v0.4s, v1.4s}, [%[f1]], #32       \n\t"
                  "ld1   {v2.4s, v3.4s}, [%[f1_c2]], #32    \n\t"
                  "ld1   {v4.s}[0], [%[f1]]                 \n\t"

                  "sub        %[f1],%[f1], #32              \n\t"
                  "ld1   {v4.s}[1], [%[f1_c2]]              \n\t"
                  "sub        %[f1_c2],%[f1_c2], #32        \n\t"

                  "prfm   pldl1keep, [%[in_ptr1], #192]     \n\t"
                  "prfm   pldl1keep, [%[in_ptr4], #192]     \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[in_ptr1]]       \n\t"
                  "add        %[in_ptr1],%[in_ptr1], #16    \n\t"

                  "ld1   {v6.d}[1], [%[in_ptr4]]            \n\t"
                  "add        %[in_ptr4],%[in_ptr4], #8     \n\t"
                  "ld1   {v7.4s}, [%[in_ptr4]]              \n\t"
                  "add        %[in_ptr4],%[in_ptr4], #8     \n\t"

                  "0:                                       \n\t"
                  // load out_ptr
                  "prfm   pldl1keep, [%[out_ptr1], #128]    \n\t"
                  "prfm   pldl1keep, [%[out_ptr1_c2], #128] \n\t"
                  "prfm   pldl1keep, [%[out_ptr2], #128]    \n\t"
                  "prfm   pldl1keep, [%[out_ptr2_c2], #128] \n\t"

                  "ld1   {v12.4s}, [%[out_ptr1]]            \n\t"
                  "ld1   {v13.4s}, [%[out_ptr1_c2]]         \n\t"
                  "ld1   {v14.4s}, [%[out_ptr2]]            \n\t"
                  "ld1   {v15.4s}, [%[out_ptr2_c2]]         \n\t"

                  // in_ptr1 and in_ptr4 multiply
                  "ext    v8.16b, v5.16b, v6.16b, #4        \n\t"
                  "fmla   v12.4s, v5.4s, v0.s[0]            \n\t"
                  "fmla   v13.4s, v5.4s, v2.s[0]            \n\t"

                  "ext    v9.16b, v6.16b, v7.16b, #8        \n\t"
                  "fmla   v14.4s, v7.4s, v4.s[0]            \n\t"
                  "fmla   v15.4s, v7.4s, v4.s[1]            \n\t"

                  "ext    v10.16b, v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v0.s[1]            \n\t"
                  "fmla   v13.4s, v8.4s, v2.s[1]            \n\t"

                  "ext    v11.16b, v6.16b, v7.16b, #12      \n\t"
                  "fmla   v14.4s, v9.4s, v1.s[2]            \n\t"
                  "fmla   v15.4s, v9.4s, v3.s[2]            \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[in_ptr2]]       \n\t"
                  "fmla   v12.4s, v10.4s, v0.s[2]           \n\t"
                  "fmla   v13.4s, v10.4s, v2.s[2]           \n\t"

                  "add        %[in_ptr2],%[in_ptr2], #16    \n\t"
                  "fmla   v14.4s, v11.4s, v1.s[3]           \n\t"
                  "fmla   v15.4s, v11.4s, v3.s[3]           \n\t"

                  // in_ptr2 multiply
                  "ext    v8.16b,  v5.16b, v6.16b, #4       \n\t"
                  "fmla   v12.4s, v5.4s, v0.s[3]            \n\t"
                  "fmla   v13.4s, v5.4s, v2.s[3]            \n\t"

                  "fmla   v14.4s, v5.4s, v0.s[0]            \n\t"
                  "fmla   v15.4s, v5.4s, v2.s[0]            \n\t"

                  "ext    v9.16b,  v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v1.s[0]            \n\t"
                  "fmla   v13.4s, v8.4s, v3.s[0]            \n\t"

                  "ld1   {v6.d}[1], [%[in_ptr3]]            \n\t"
                  "add        %[in_ptr3],%[in_ptr3], #8     \n\t"
                  "fmla   v14.4s, v8.4s, v0.s[1]            \n\t"
                  "fmla   v15.4s, v8.4s, v2.s[1]            \n\t"

                  "ld1   {v7.4s}, [%[in_ptr3]]              \n\t"
                  "add        %[in_ptr3],%[in_ptr3], #8     \n\t"

                  "fmla   v12.4s, v9.4s, v1.s[1]            \n\t"
                  "fmla   v13.4s, v9.4s, v3.s[1]            \n\t"

                  "ext    v10.16b, v6.16b, v7.16b, #8       \n\t"
                  "fmla   v14.4s, v9.4s, v0.s[2]            \n\t"
                  "fmla   v15.4s, v9.4s, v2.s[2]            \n\t"

                  // in_ptr3 multiply
                  "fmla   v12.4s, v7.4s, v4.s[0]            \n\t"
                  "fmla   v13.4s, v7.4s, v4.s[1]            \n\t"

                  "ext    v11.16b, v6.16b,  v7.16b, #12     \n\t"
                  "fmla   v14.4s, v7.4s, v1.s[1]            \n\t"
                  "fmla   v15.4s, v7.4s, v3.s[1]            \n\t"

                  "fmla   v12.4s, v10.4s, v1.s[2]           \n\t"
                  "fmla   v13.4s, v10.4s, v3.s[2]           \n\t"

                  "fmla   v14.4s, v10.4s, v0.s[3]           \n\t"
                  "fmla   v15.4s, v10.4s, v2.s[3]           \n\t"

                  "fmla   v12.4s, v11.4s, v1.s[3]           \n\t"
                  "fmla   v13.4s, v11.4s, v3.s[3]           \n\t"

                  "prfm   pldl1keep, [%[in_ptr1], #192]     \n\t"
                  "fmla   v14.4s, v11.4s, v1.s[0]           \n\t"
                  "fmla   v15.4s, v11.4s, v3.s[0]           \n\t"

                  // store out_ptr
                  "prfm   pldl1keep, [%[in_ptr4], #192]     \n\t"
                  "ld1   {v5.4s, v6.4s}, [%[in_ptr1]]       \n\t"
                  "add        %[in_ptr1],%[in_ptr1], #16    \n\t"
                  "st1   {v12.4s}, [%[out_ptr1]], #16       \n\t"

                  "ld1   {v6.d}[1], [%[in_ptr4]]            \n\t"
                  "add        %[in_ptr4],%[in_ptr4], #8     \n\t"
                  "st1   {v13.4s}, [%[out_ptr1_c2]], #16     \n\t"

                  "ld1   {v7.4s}, [%[in_ptr4]]              \n\t"
                  "add        %[in_ptr4],%[in_ptr4], #8     \n\t"
                  "st1   {v14.4s}, [%[out_ptr2]], #16       \n\t"

                  "subs       %[loop],%[loop], #1   \n\t"
                  "st1   {v15.4s}, [%[out_ptr2_c2]], #16    \n\t"

                  // cycle
                  "bne        0b                            \n\t"
                  "sub       %[in_ptr1],%[in_ptr1], #16     \n\t"
                  "sub       %[in_ptr4],%[in_ptr4], #16     \n\t"

                  : [loop] "+r"(loop), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr2] "+r"(out_ptr2), [out_ptr1_c2] "+r"(out_ptr1_c2),
                    [out_ptr2_c2] "+r"(out_ptr2_c2), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3),
                    [in_ptr4] "+r"(in_ptr4)
                  : [f1] "r"(f1), [f1_c2] "r"(f1_c2)
                  : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                    "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
            }
          }
          if (!if_nopadding && o_w == output_w - padding_w) {
            pad_filter0--;
            pad_filter1--;
            pad_filter2--;
            pad_filter3--;

            pad_filter0_c2--;
            pad_filter1_c2--;
            pad_filter2_c2--;
            pad_filter3_c2--;

            in_ptr1--;
            in_ptr2--;
            in_ptr3--;
            in_ptr4--;
          }
#else
          if (issamefilter) {
            int loop = (output_w - 2 * padding_w) >> 2;
            o_w += loop * 4;

            if (loop > 0) {
              asm volatile(
                  "pld        [%[f1], #256]                 \n\t"
                  "pld        [%[f1_c2], #256]              \n\t"

                  "vld1.f32   {d0-d3}, [%[f1]]              \n\t"
                  "add        %[f1], #32                    \n\t"
                  "vld1.f32   {d4-d7}, [%[f1_c2]]           \n\t"
                  "add        %[f1_c2], #32                 \n\t"

                  "vld1.f32   {d8[0]}, [%[f1]]              \n\t"
                  "sub        %[f1], #32                    \n\t"
                  "vld1.f32   {d8[1]}, [%[f1_c2]]           \n\t"
                  "sub        %[f1_c2], #32                 \n\t"

                  "pld        [%[in_ptr1], #192]            \n\t"
                  "pld        [%[in_ptr4], #192]            \n\t"

                  "vld1.f32   {d10-d12}, [%[in_ptr1]]       \n\t"
                  "add        %[in_ptr1], #16               \n\t"

                  "vld1.f32   {d13-d15}, [%[in_ptr4]]       \n\t"
                  "add        %[in_ptr4], #16               \n\t"

                  "0:                                       \n\t"
                  // load out_ptr
                  "pld        [%[out_ptr1], #128]           \n\t"
                  "pld        [%[out_ptr1_c2], #128]        \n\t"
                  "pld        [%[out_ptr2], #128]           \n\t"
                  "pld        [%[out_ptr2_c2], #128]        \n\t"

                  "vld1.f32   {d24, d25}, [%[out_ptr1]]     \n\t"
                  "vld1.f32   {d26, d27}, [%[out_ptr1_c2]]  \n\t"
                  "vld1.f32   {d28, d29}, [%[out_ptr2]]     \n\t"
                  "vld1.f32   {d30, d31}, [%[out_ptr2_c2]]  \n\t"

                  // in_ptr1 + in_ptr4 multiply
                  "vext.32    q8, q5, q6, #1                \n\t"
                  "vmla.f32   q12, q5, d0[0]                \n\t"
                  "vmla.f32   q13, q5, d4[0]                \n\t"

                  "vext.32    q9, q6, q7, #2                \n\t"
                  "vmla.f32   q14, q7, d8[0]                \n\t"
                  "vmla.f32   q15, q7, d8[1]                \n\t"

                  "vext.32    q10, q5, q6, #2               \n\t"
                  "vmla.f32   q12, q8, d0[1]                \n\t"
                  "vmla.f32   q13, q8, d4[1]                \n\t"

                  "vext.32    q11, q6, q7, #3               \n\t"
                  "vmla.f32   q14, q9, d3[0]                \n\t"
                  "vmla.f32   q15, q9, d7[0]                \n\t"

                  "vld1.f32   {d10-d12}, [%[in_ptr2]]       \n\t"
                  "add        %[in_ptr2], #16               \n\t"
                  "vmla.f32   q12, q10, d1[0]               \n\t"
                  "vmla.f32   q13, q10, d5[0]               \n\t"

                  "vmla.f32   q14, q11, d3[1]               \n\t"
                  "vmla.f32   q15, q11, d7[1]               \n\t"

                  // in_ptr2 multiply
                  "vext.32    q8, q5, q6, #1                \n\t"
                  "vmla.f32   q12, q5, d1[1]                \n\t"
                  "vmla.f32   q13, q5, d5[1]                \n\t"

                  "vmla.f32   q14, q5, d0[0]                \n\t"
                  "vmla.f32   q15, q5, d4[0]                \n\t"

                  "vext.32    q9, q5, q6, #2                \n\t"
                  "vmla.f32   q12, q8, d2[0]                \n\t"
                  "vmla.f32   q13, q8, d6[0]                \n\t"

                  "vld1.f32   {d13-d15}, [%[in_ptr3]]       \n\t"
                  "add        %[in_ptr3], #16               \n\t"
                  "vmla.f32   q14, q8, d0[1]                \n\t"
                  "vmla.f32   q15, q8, d4[1]                \n\t"

                  "vmla.f32   q12, q9, d2[1]                \n\t"
                  "vmla.f32   q13, q9, d6[1]                \n\t"

                  "vmla.f32   q14, q9, d1[0]                \n\t"
                  "vmla.f32   q15, q9, d5[0]                \n\t"

                  // in_ptr3 multiply
                  "vext.32    q10, q6, q7, #2               \n\t"
                  "vmla.f32   q12, q7, d8[0]                \n\t"
                  "vmla.f32   q13, q7, d8[1]                \n\t"
                  "vmla.f32   q14, q7, d2[1]                \n\t"
                  "vmla.f32   q15, q7, d6[1]                \n\t"

                  "vext.32    q11, q6, q7, #3               \n\t"
                  "vmla.f32   q12, q10, d3[0]               \n\t"
                  "vmla.f32   q13, q10, d7[0]               \n\t"
                  "vmla.f32   q14, q10, d1[1]               \n\t"
                  "vmla.f32   q15, q10, d5[1]               \n\t"

                  "vmla.f32   q12, q11, d3[1]               \n\t"
                  "vmla.f32   q13, q11, d7[1]               \n\t"
                  "vmla.f32   q14, q11, d2[0]               \n\t"
                  "vmla.f32   q15, q11, d6[0]               \n\t"

                  // store out_ptr
                  "pld        [%[in_ptr1], #192]            \n\t"

                  "pld        [%[in_ptr4], #192]            \n\t"
                  "vld1.f32   {d10-d12}, [%[in_ptr1]]       \n\t"
                  "add        %[in_ptr1], #16               \n\t"

                  "vst1.f32   {d24, d25}, [%[out_ptr1]]!    \n\t"

                  "vst1.f32   {d26, d27}, [%[out_ptr1_c2]]! \n\t"
                  "vld1.f32   {d13-d15}, [%[in_ptr4]]       \n\t"

                  "add        %[in_ptr4], #16               \n\t"
                  "vst1.f32   {d28, d29}, [%[out_ptr2]]!    \n\t"

                  "subs       %[loop], #1               \n\t"
                  "vst1.f32   {d30, d31}, [%[out_ptr2_c2]]! \n\t"

                  // cycle
                  "bne        0b                            \n\t"
                  "sub       %[in_ptr1], #16                \n\t"
                  "sub       %[in_ptr4], #16                \n\t"

                  : [loop] "+r"(loop), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr2] "+r"(out_ptr2), [out_ptr1_c2] "+r"(out_ptr1_c2),
                    [out_ptr2_c2] "+r"(out_ptr2_c2), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3),
                    [in_ptr4] "+r"(in_ptr4)
                  : [f1] "r"(f1), [f1_c2] "r"(f1_c2)
                  : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                    "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
            }
          }
          if (!if_nopadding && o_w == output_w - padding_w) {
            pad_filter0--;
            pad_filter1--;
            pad_filter2--;
            pad_filter3--;

            pad_filter0_c2--;
            pad_filter1_c2--;
            pad_filter2_c2--;
            pad_filter3_c2--;

            in_ptr1--;
            in_ptr2--;
            in_ptr3--;
            in_ptr4--;
          }
#endif  // __aarch64__
#endif  // __ARM_NEON

          // remain output_width
          for (; o_w < output_w; ++o_w) {
            float sum1 = 0;
            float sum2 = 0;
            float sum1_c2 = 0;
            float sum2_c2 = 0;

            if (issamefilter) {
#if __ARM_NEON
              float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
              float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
              float32x4_t _pad_filter1_c2 = vld1q_f32(pad_filter1_c2);
              float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
              float32x4_t _sum1_c2 = vmulq_f32(_in_ptr1, _pad_filter1_c2);

              float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
              float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
              float32x4_t _pad_filter2_c2 = vld1q_f32(pad_filter2_c2);
              _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
              _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr2, _pad_filter2_c2);
              float32x4_t _sum2 = vmulq_f32(_in_ptr2, _pad_filter1);
              float32x4_t _sum2_c2 = vmulq_f32(_in_ptr2, _pad_filter1_c2);

              float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
              float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
              float32x4_t _pad_filter3_c2 = vld1q_f32(pad_filter3_c2);
              _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
              _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr3, _pad_filter3_c2);
              _sum2 = vmlaq_f32(_sum2, _in_ptr3, _pad_filter2);
              _sum2_c2 = vmlaq_f32(_sum2_c2, _in_ptr3, _pad_filter2_c2);

              float32x4_t _in_ptr4 = vld1q_f32(in_ptr4);
              _sum2 = vmlaq_f32(_sum2, _in_ptr4, _pad_filter3);
              _sum2_c2 = vmlaq_f32(_sum2_c2, _in_ptr4, _pad_filter3_c2);

              _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
              _sum1_c2 = vsetq_lane_f32(sum1_c2, _sum1_c2, 3);
              _sum2 = vsetq_lane_f32(sum2, _sum2, 3);
              _sum2_c2 = vsetq_lane_f32(sum2_c2, _sum2_c2, 3);

              float32x2_t _ss1 =
                  vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
              float32x2_t _ss1_2 =
                  vadd_f32(vget_low_f32(_sum1_c2), vget_high_f32(_sum1_c2));
              float32x2_t _ss2 =
                  vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
              float32x2_t _ss2_2 =
                  vadd_f32(vget_low_f32(_sum2_c2), vget_high_f32(_sum2_c2));
              float32x2_t _ssss1_ssss2 = vpadd_f32(_ss1, _ss2);
              float32x2_t _ssss1_2_ssss2_2 = vpadd_f32(_ss1_2, _ss2_2);

              sum1 += vget_lane_f32(_ssss1_ssss2, 0);
              sum1_c2 += vget_lane_f32(_ssss1_2_ssss2_2, 0);
              sum2 += vget_lane_f32(_ssss1_ssss2, 1);
              sum2_c2 += vget_lane_f32(_ssss1_2_ssss2_2, 1);
#else
              sum1 += in_ptr1[0] * pad_filter1[0];
              sum1 += in_ptr1[1] * pad_filter1[1];
              sum1 += in_ptr1[2] * pad_filter1[2];
              sum1 += in_ptr2[0] * pad_filter2[0];
              sum1 += in_ptr2[1] * pad_filter2[1];
              sum1 += in_ptr2[2] * pad_filter2[2];
              sum1 += in_ptr3[0] * pad_filter3[0];
              sum1 += in_ptr3[1] * pad_filter3[1];
              sum1 += in_ptr3[2] * pad_filter3[2];

              sum2 += in_ptr2[0] * pad_filter1[0];
              sum2 += in_ptr2[1] * pad_filter1[1];
              sum2 += in_ptr2[2] * pad_filter1[2];
              sum2 += in_ptr3[0] * pad_filter2[0];
              sum2 += in_ptr3[1] * pad_filter2[1];
              sum2 += in_ptr3[2] * pad_filter2[2];
              sum2 += in_ptr4[0] * pad_filter3[0];
              sum2 += in_ptr4[1] * pad_filter3[1];
              sum2 += in_ptr4[2] * pad_filter3[2];

              sum1_c2 += in_ptr1[0] * pad_filter1_c2[0];
              sum1_c2 += in_ptr1[1] * pad_filter1_c2[1];
              sum1_c2 += in_ptr1[2] * pad_filter1_c2[2];
              sum1_c2 += in_ptr2[0] * pad_filter2_c2[0];
              sum1_c2 += in_ptr2[1] * pad_filter2_c2[1];
              sum1_c2 += in_ptr2[2] * pad_filter2_c2[2];
              sum1_c2 += in_ptr3[0] * pad_filter3_c2[0];
              sum1_c2 += in_ptr3[1] * pad_filter3_c2[1];
              sum1_c2 += in_ptr3[2] * pad_filter3_c2[2];

              sum2_c2 += in_ptr2[0] * pad_filter1_c2[0];
              sum2_c2 += in_ptr2[1] * pad_filter1_c2[1];
              sum2_c2 += in_ptr2[2] * pad_filter1_c2[2];
              sum2_c2 += in_ptr3[0] * pad_filter2_c2[0];
              sum2_c2 += in_ptr3[1] * pad_filter2_c2[1];
              sum2_c2 += in_ptr3[2] * pad_filter2_c2[2];
              sum2_c2 += in_ptr4[0] * pad_filter3_c2[0];
              sum2_c2 += in_ptr4[1] * pad_filter3_c2[1];
              sum2_c2 += in_ptr4[2] * pad_filter3_c2[2];
#endif
            } else {
#if __ARM_NEON
              float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
              float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
              float32x4_t _pad_filter1_c2 = vld1q_f32(pad_filter1_c2);
              float32x4_t _pad_filter0 = vld1q_f32(pad_filter0);
              float32x4_t _pad_filter0_c2 = vld1q_f32(pad_filter0_c2);

              float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
              float32x4_t _sum1_c2 = vmulq_f32(_in_ptr1, _pad_filter1_c2);
              float32x4_t _sum2 = vmulq_f32(_in_ptr1, _pad_filter0);
              float32x4_t _sum2_c2 = vmulq_f32(_in_ptr1, _pad_filter0_c2);

              float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
              float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
              float32x4_t _pad_filter2_c2 = vld1q_f32(pad_filter2_c2);

              _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
              _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr2, _pad_filter2_c2);
              _sum2 = vmlaq_f32(_sum2, _in_ptr2, _pad_filter1);
              _sum2_c2 = vmlaq_f32(_sum2_c2, _in_ptr2, _pad_filter1_c2);

              float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
              float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
              float32x4_t _pad_filter3_c2 = vld1q_f32(pad_filter3_c2);

              _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
              _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr3, _pad_filter3_c2);
              _sum2 = vmlaq_f32(_sum2, _in_ptr3, _pad_filter2);
              _sum2_c2 = vmlaq_f32(_sum2_c2, _in_ptr3, _pad_filter2_c2);

              _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
              _sum1_c2 = vsetq_lane_f32(sum1_c2, _sum1_c2, 3);
              _sum2 = vsetq_lane_f32(sum2, _sum2, 3);
              _sum2_c2 = vsetq_lane_f32(sum2_c2, _sum2_c2, 3);

              float32x2_t _ss1 =
                  vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
              float32x2_t _ss1_2 =
                  vadd_f32(vget_low_f32(_sum1_c2), vget_high_f32(_sum1_c2));
              float32x2_t _ss2 =
                  vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
              float32x2_t _ss2_2 =
                  vadd_f32(vget_low_f32(_sum2_c2), vget_high_f32(_sum2_c2));
              float32x2_t _ssss1_ssss2 = vpadd_f32(_ss1, _ss2);
              float32x2_t _ssss1_2_ssss2_2 = vpadd_f32(_ss1_2, _ss2_2);

              sum1 += vget_lane_f32(_ssss1_ssss2, 0);
              sum1_c2 += vget_lane_f32(_ssss1_2_ssss2_2, 0);
              sum2 += vget_lane_f32(_ssss1_ssss2, 1);
              sum2_c2 += vget_lane_f32(_ssss1_2_ssss2_2, 1);
#else
              sum1 += in_ptr1[0] * pad_filter1[0];
              sum1 += in_ptr1[1] * pad_filter1[1];
              sum1 += in_ptr1[2] * pad_filter1[2];
              sum1 += in_ptr2[0] * pad_filter2[0];
              sum1 += in_ptr2[1] * pad_filter2[1];
              sum1 += in_ptr2[2] * pad_filter2[2];
              sum1 += in_ptr3[0] * pad_filter3[0];
              sum1 += in_ptr3[1] * pad_filter3[1];
              sum1 += in_ptr3[2] * pad_filter3[2];

              sum2 += in_ptr1[0] * pad_filter0[0];
              sum2 += in_ptr1[1] * pad_filter0[1];
              sum2 += in_ptr1[2] * pad_filter0[2];
              sum2 += in_ptr2[0] * pad_filter1[0];
              sum2 += in_ptr2[1] * pad_filter1[1];
              sum2 += in_ptr2[2] * pad_filter1[2];
              sum2 += in_ptr3[0] * pad_filter2[0];
              sum2 += in_ptr3[1] * pad_filter2[1];
              sum2 += in_ptr3[2] * pad_filter2[2];

              sum1_c2 += in_ptr1[0] * pad_filter1_c2[0];
              sum1_c2 += in_ptr1[1] * pad_filter1_c2[1];
              sum1_c2 += in_ptr1[2] * pad_filter1_c2[2];
              sum1_c2 += in_ptr2[0] * pad_filter2_c2[0];
              sum1_c2 += in_ptr2[1] * pad_filter2_c2[1];
              sum1_c2 += in_ptr2[2] * pad_filter2_c2[2];
              sum1_c2 += in_ptr3[0] * pad_filter3_c2[0];
              sum1_c2 += in_ptr3[1] * pad_filter3_c2[1];
              sum1_c2 += in_ptr3[2] * pad_filter3_c2[2];

              sum2_c2 += in_ptr1[0] * pad_filter0_c2[0];
              sum2_c2 += in_ptr1[1] * pad_filter0_c2[1];
              sum2_c2 += in_ptr1[2] * pad_filter0_c2[2];
              sum2_c2 += in_ptr2[0] * pad_filter1_c2[0];
              sum2_c2 += in_ptr2[1] * pad_filter1_c2[1];
              sum2_c2 += in_ptr2[2] * pad_filter1_c2[2];
              sum2_c2 += in_ptr3[0] * pad_filter2_c2[0];
              sum2_c2 += in_ptr3[1] * pad_filter2_c2[1];
              sum2_c2 += in_ptr3[2] * pad_filter2_c2[2];
#endif
            }
            if (!if_nopadding &&
                (o_w < padding_w || o_w > output_w - padding_w - 2)) {
              pad_filter0--;
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;

              pad_filter0_c2--;
              pad_filter1_c2--;
              pad_filter2_c2--;
              pad_filter3_c2--;
            } else {
              in_ptr1++;
              in_ptr2++;
              in_ptr3++;
              in_ptr4++;
            }
            *out_ptr1 += sum1;
            *out_ptr2 += sum2;
            *out_ptr1_c2 += sum1_c2;
            *out_ptr2_c2 += sum2_c2;

            out_ptr1++;
            out_ptr2++;
            out_ptr1_c2++;
            out_ptr2_c2++;
          }
          if (if_nopadding) {
            in_ptr1 += 2 + input_w;
            in_ptr2 += 2 + input_w;
            in_ptr3 += 2 + input_w;
            in_ptr4 += 2 + input_w;
          } else if (o_h == padding_h - 1 || o_h == output_h - padding_h - 2) {
            in_ptr1 += 3;
            in_ptr2 += 3;
            in_ptr3 += 3;
            in_ptr4 += 3;

            pad_filter0 -= 2;
            pad_filter1 -= 2;
            pad_filter2 -= 2;
            pad_filter3 -= 2;

            pad_filter0_c2 -= 2;
            pad_filter1_c2 -= 2;
            pad_filter2_c2 -= 2;
            pad_filter3_c2 -= 2;

          } else if (issamefilter) {
            in_ptr1 += 3 + input_w;
            in_ptr2 += 3 + input_w;
            in_ptr3 += 3 + input_w;
            in_ptr4 += 3 + input_w;

            pad_filter0 += 2 * padding_w + 1;
            pad_filter1 += 2 * padding_w + 1;
            pad_filter2 += 2 * padding_w + 1;
            pad_filter3 += 2 * padding_w + 1;

            pad_filter0_c2 += 2 * padding_w + 1;
            pad_filter1_c2 += 2 * padding_w + 1;
            pad_filter2_c2 += 2 * padding_w + 1;
            pad_filter3_c2 += 2 * padding_w + 1;

          } else {
            pad_filter0 -= 3 + 2 * padding_w + 2;
            pad_filter1 -= 3 + 2 * padding_w + 2;
            pad_filter2 -= 3 + 2 * padding_w + 2;
            pad_filter3 -= 3 + 2 * padding_w + 2;

            pad_filter0_c2 -= 3 + 2 * padding_w + 2;
            pad_filter1_c2 -= 3 + 2 * padding_w + 2;
            pad_filter2_c2 -= 3 + 2 * padding_w + 2;
            pad_filter3_c2 -= 3 + 2 * padding_w + 2;

            in_ptr1 -= input_w - 3;
            in_ptr2 -= input_w - 3;
            in_ptr3 -= input_w - 3;
            in_ptr4 -= input_w - 3;
          }
          out_ptr1 += output_w;
          out_ptr2 += output_w;
          out_ptr1_c2 += output_w;
          out_ptr2_c2 += output_w;
        }
        // remain output_height
        for (; o_h < output_h; ++o_h) {
          int o_w = 0;
          // pad left
          for (; o_w < padding_w; ++o_w) {
            float sum1 = 0;
            float sum1_c2 = 0;
#if __ARM_NEON
            float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
            float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
            float32x4_t _pad_filter1_c2 = vld1q_f32(pad_filter1_c2);
            float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
            float32x4_t _sum1_c2 = vmulq_f32(_in_ptr1, _pad_filter1_c2);

            float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
            float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
            float32x4_t _pad_filter2_c2 = vld1q_f32(pad_filter2_c2);
            _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
            _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr2, _pad_filter2_c2);

            float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
            float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
            float32x4_t _pad_filter3_c2 = vld1q_f32(pad_filter3_c2);
            _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
            _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr3, _pad_filter3_c2);

            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
            _sum1_c2 = vsetq_lane_f32(sum1_c2, _sum1_c2, 3);

            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ss1_2 =
                vadd_f32(vget_low_f32(_sum1_c2), vget_high_f32(_sum1_c2));
            float32x2_t _ssss1_ssss1_2 = vpadd_f32(_ss1, _ss1_2);

            sum1 += vget_lane_f32(_ssss1_ssss1_2, 0);
            sum1_c2 += vget_lane_f32(_ssss1_ssss1_2, 1);
#else
            sum1 += in_ptr1[0] * pad_filter1[0];
            sum1 += in_ptr1[1] * pad_filter1[1];
            sum1 += in_ptr1[2] * pad_filter1[2];
            sum1 += in_ptr2[0] * pad_filter2[0];
            sum1 += in_ptr2[1] * pad_filter2[1];
            sum1 += in_ptr2[2] * pad_filter2[2];
            sum1 += in_ptr3[0] * pad_filter3[0];
            sum1 += in_ptr3[1] * pad_filter3[1];
            sum1 += in_ptr3[2] * pad_filter3[2];

            sum1_c2 += in_ptr1[0] * pad_filter1_c2[0];
            sum1_c2 += in_ptr1[1] * pad_filter1_c2[1];
            sum1_c2 += in_ptr1[2] * pad_filter1_c2[2];
            sum1_c2 += in_ptr2[0] * pad_filter2_c2[0];
            sum1_c2 += in_ptr2[1] * pad_filter2_c2[1];
            sum1_c2 += in_ptr2[2] * pad_filter2_c2[2];
            sum1_c2 += in_ptr3[0] * pad_filter3_c2[0];
            sum1_c2 += in_ptr3[1] * pad_filter3_c2[1];
            sum1_c2 += in_ptr3[2] * pad_filter3_c2[2];
#endif
            if (!if_nopadding &&
                (o_w < padding_w || o_w > output_w - padding_w - 2)) {
              pad_filter0--;
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;

              pad_filter0_c2--;
              pad_filter1_c2--;
              pad_filter2_c2--;
              pad_filter3_c2--;
            } else {
              in_ptr1++;
              in_ptr2++;
              in_ptr3++;
              in_ptr4++;
            }
            *out_ptr1 += sum1;
            *out_ptr1_c2 += sum1_c2;

            out_ptr1++;
            out_ptr1_c2++;
          }
//             valid
#if __ARM_NEON
#if __aarch64__
          if (if_nopadding) {
            int loop = (output_w - 2 * padding_w) >> 2;
            o_w += loop * 4;

            if (loop > 0) {
              asm volatile(
                  "prfm   pldl1keep, [%[f1], #256]          \n\t"
                  "prfm   pldl1keep, [%[f1_c2], #256]        \n\t"

                  "ld1   {v0.4s, v1.4s}, [%[f1]]            \n\t"
                  "add        %[f1], %[f1], #32             \n\t"
                  "ld1   {v2.4s, v3.4s}, [%[f1_c2]]          \n\t"
                  "add        %[f1_c2], %[f1_c2], #32         \n\t"

                  "ld1   {v4.s}[0], [%[f1]]                 \n\t"
                  "sub        %[f1],%[f1], #32              \n\t"
                  "ld1   {v4.s}[1], [%[f1_c2]]               \n\t"
                  "sub        %[f1_c2],%[f1_c2], #32          \n\t"

                  "0:                                       \n\t"
                  // load out_ptr
                  "prfm   pldl1keep, [%[out_ptr1], #128]    \n\t"
                  "prfm   pldl1keep, [%[out_ptr1_c2], #128]  \n\t"

                  "ld1   {v12.4s}, [%[out_ptr1]]            \n\t"
                  "ld1   {v13.4s}, [%[out_ptr1_c2]]          \n\t"

                  // in_ptr1 multiply
                  "prfm   pldl1keep, [%[in_ptr1], #192]     \n\t"
                  "ld1   {v5.4s, v6.4s}, [%[in_ptr1]]       \n\t"
                  "add        %[in_ptr1],%[in_ptr1], #16    \n\t"

                  "ext    v8.16b, v5.16b, v6.16b, #4        \n\t"
                  "fmla   v12.4s, v5.4s, v0.s[0]            \n\t"
                  "fmla   v13.4s, v5.4s, v2.s[0]            \n\t"

                  "ext    v10.16b, v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v0.s[1]            \n\t"
                  "fmla   v13.4s, v8.4s, v2.s[1]            \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[in_ptr2]]       \n\t"
                  "add        %[in_ptr2],%[in_ptr2], #16    \n\t"
                  "fmla   v12.4s, v10.4s, v0.s[2]           \n\t"
                  "fmla   v13.4s, v10.4s, v2.s[2]           \n\t"

                  // in_ptr2 multiply
                  "ext    v8.16b,  v5.16b, v6.16b, #4       \n\t"
                  "fmla   v12.4s, v5.4s, v0.s[3]            \n\t"
                  "fmla   v13.4s, v5.4s, v2.s[3]            \n\t"

                  "ext    v9.16b,  v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v1.s[0]            \n\t"
                  "fmla   v13.4s, v8.4s, v3.s[0]            \n\t"

                  "ld1   {v6.d}[1], [%[in_ptr3]]            \n\t"
                  "add        %[in_ptr3],%[in_ptr3], #8     \n\t"
                  "ld1   {v7.4s}, [%[in_ptr3]]              \n\t"
                  "add        %[in_ptr3],%[in_ptr3], #8     \n\t"

                  "fmla   v12.4s, v9.4s, v1.s[1]            \n\t"
                  "fmla   v13.4s, v9.4s, v3.s[1]            \n\t"

                  // in_ptr3 multiply
                  "ext    v10.16b, v6.16b, v7.16b, #8       \n\t"
                  "fmla   v12.4s, v7.4s, v4.s[0]            \n\t"
                  "fmla   v13.4s, v7.4s, v4.s[1]            \n\t"

                  "ext    v11.16b, v6.16b,  v7.16b, #12     \n\t"
                  "fmla   v12.4s, v10.4s, v1.s[2]           \n\t"
                  "fmla   v13.4s, v10.4s, v3.s[2]           \n\t"

                  "fmla   v12.4s, v11.4s, v1.s[3]           \n\t"
                  "fmla   v13.4s, v11.4s, v3.s[3]           \n\t"

                  // store out_ptr
                  "st1   {v12.4s}, [%[out_ptr1]], #16       \n\t"
                  "st1   {v13.4s}, [%[out_ptr1_c2]], #16     \n\t"

                  // cycle
                  "subs       %[loop],%[loop], #1   \n\t"
                  "bne        0b                            \n\t"

                  : [loop] "+r"(loop), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr2] "+r"(out_ptr2), [out_ptr1_c2] "+r"(out_ptr1_c2),
                    [out_ptr2_c2] "+r"(out_ptr2_c2), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3),
                    [in_ptr4] "+r"(in_ptr4)
                  : [f1] "r"(f1), [f1_c2] "r"(f1_c2)
                  : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                    "v7", "v8", "v9", "v10", "v11", "v12", "v13");
            }
          }
#else
          if (if_nopadding) {
            int loop = (output_w - 2 * padding_w) >> 2;
            o_w += loop * 4;

            if (loop > 0) {
              asm volatile(
                  "pld        [%[f1], #256]                 \n\t"
                  "pld        [%[f1_c2], #256]               \n\t"

                  "vld1.f32   {d0-d3}, [%[f1]]              \n\t"
                  "add        %[f1], #32                    \n\t"
                  "vld1.f32   {d4-d7}, [%[f1_c2]]            \n\t"
                  "add        %[f1_c2], #32                  \n\t"

                  "vld1.f32   {d8[0]}, [%[f1]]              \n\t"
                  "sub        %[f1], #32                    \n\t"
                  "vld1.f32   {d8[1]}, [%[f1_c2]]            \n\t"
                  "sub        %[f1_c2], #32                  \n\t"

                  "0:                                       \n\t"
                  // load out_ptr
                  "pld        [%[out_ptr1], #128]           \n\t"
                  "pld        [%[out_ptr1_c2], #128]         \n\t"

                  "vld1.f32   {d24, d25}, [%[out_ptr1]]     \n\t"
                  "vld1.f32   {d26, d27}, [%[out_ptr1_c2]]   \n\t"

                  // in_ptr1 multiply
                  "pld        [%[in_ptr1], #128]            \n\t"

                  "vld1.f32   {d10-d12}, [%[in_ptr1]]       \n\t"
                  "add        %[in_ptr1], #16               \n\t"
                  "vext.32    q8, q5, q6, #1                \n\t"

                  "pld        [%[in_ptr2], #128]            \n\t"
                  "vmla.f32   q12, q5, d0[0]                \n\t"
                  "vmla.f32   q13, q5, d4[0]                \n\t"

                  "vext.32    q10, q5, q6, #2               \n\t"
                  "vld1.f32   {d10-d12}, [%[in_ptr2]]       \n\t"
                  "add        %[in_ptr2], #16               \n\t"
                  "vmla.f32   q12, q8, d0[1]                \n\t"
                  "vmla.f32   q13, q8, d4[1]                \n\t"

                  "vmla.f32   q12, q10, d1[0]               \n\t"
                  "vmla.f32   q13, q10, d5[0]               \n\t"

                  // in_ptr2 multiply
                  "vext.32    q8, q5, q6, #1                \n\t"
                  "pld        [%[in_ptr3], #128]            \n\t"
                  "vmla.f32   q12, q5, d1[1]                \n\t"
                  "vmla.f32   q13, q5, d5[1]                \n\t"

                  "vext.32    q9, q5, q6, #2                \n\t"
                  "vld1.f32   {d13-d15}, [%[in_ptr3]]       \n\t"
                  "add        %[in_ptr3], #16               \n\t"
                  "vmla.f32   q12, q8, d2[0]                \n\t"
                  "vmla.f32   q13, q8, d6[0]                \n\t"

                  "vmla.f32   q12, q9, d2[1]                \n\t"
                  "vmla.f32   q13, q9, d6[1]                \n\t"

                  // in_ptr3 multiply
                  "vext.32    q10, q6, q7, #2               \n\t"
                  "vmla.f32   q12, q7, d8[0]                \n\t"
                  "vmla.f32   q13, q7, d8[1]                \n\t"

                  "vext.32    q11, q6, q7, #3               \n\t"
                  "vmla.f32   q12, q10, d3[0]               \n\t"
                  "vmla.f32   q13, q10, d7[0]               \n\t"

                  "vmla.f32   q12, q11, d3[1]               \n\t"
                  "vmla.f32   q13, q11, d7[1]               \n\t"

                  // store out_ptr
                  "subs       %[loop], #1               \n\t"
                  "vst1.f32   {d24, d25}, [%[out_ptr1]]!    \n\t"
                  "vst1.f32   {d26, d27}, [%[out_ptr1_c2]]!  \n\t"

                  // cycle
                  "bne        0b                            \n\t"

                  : [loop] "+r"(loop), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr2] "+r"(out_ptr2), [out_ptr1_c2] "+r"(out_ptr1_c2),
                    [out_ptr2_c2] "+r"(out_ptr2_c2), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3),
                    [in_ptr4] "+r"(in_ptr4)
                  : [f1] "r"(f1), [f1_c2] "r"(f1_c2)
                  : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                    "q7", "q8", "q9", "q10", "q11", "q12", "q13");
            }
          }

#endif  // __aarch64__
#endif  // __ARM_NEON

          // remain output_width
          for (; o_w < output_w; ++o_w) {
            float sum1 = 0;
            float sum1_c2 = 0;
#if __ARM_NEON
            float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
            float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
            float32x4_t _pad_filter1_c2 = vld1q_f32(pad_filter1_c2);
            float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
            float32x4_t _sum1_c2 = vmulq_f32(_in_ptr1, _pad_filter1_c2);

            float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
            float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
            float32x4_t _pad_filter2_c2 = vld1q_f32(pad_filter2_c2);
            _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
            _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr2, _pad_filter2_c2);

            float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
            float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
            float32x4_t _pad_filter3_c2 = vld1q_f32(pad_filter3_c2);
            _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
            _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr3, _pad_filter3_c2);

            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
            _sum1_c2 = vsetq_lane_f32(sum1_c2, _sum1_c2, 3);

            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ss1_2 =
                vadd_f32(vget_low_f32(_sum1_c2), vget_high_f32(_sum1_c2));

            float32x2_t _ssss1_ssss1_2 = vpadd_f32(_ss1, _ss1_2);
            sum1 += vget_lane_f32(_ssss1_ssss1_2, 0);
            sum1_c2 += vget_lane_f32(_ssss1_ssss1_2, 1);
#else
            sum1 += in_ptr1[0] * pad_filter1[0];
            sum1 += in_ptr1[1] * pad_filter1[1];
            sum1 += in_ptr1[2] * pad_filter1[2];
            sum1 += in_ptr2[0] * pad_filter2[0];
            sum1 += in_ptr2[1] * pad_filter2[1];
            sum1 += in_ptr2[2] * pad_filter2[2];
            sum1 += in_ptr3[0] * pad_filter3[0];
            sum1 += in_ptr3[1] * pad_filter3[1];
            sum1 += in_ptr3[2] * pad_filter3[2];

            sum1_c2 += in_ptr1[0] * pad_filter1_c2[0];
            sum1_c2 += in_ptr1[1] * pad_filter1_c2[1];
            sum1_c2 += in_ptr1[2] * pad_filter1_c2[2];
            sum1_c2 += in_ptr2[0] * pad_filter2_c2[0];
            sum1_c2 += in_ptr2[1] * pad_filter2_c2[1];
            sum1_c2 += in_ptr2[2] * pad_filter2_c2[2];
            sum1_c2 += in_ptr3[0] * pad_filter3_c2[0];
            sum1_c2 += in_ptr3[1] * pad_filter3_c2[1];
            sum1_c2 += in_ptr3[2] * pad_filter3_c2[2];
#endif
            if (!if_nopadding &&
                (o_w < padding_w || o_w > output_w - padding_w - 2)) {
              pad_filter0--;
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;

              pad_filter0_c2--;
              pad_filter1_c2--;
              pad_filter2_c2--;
              pad_filter3_c2--;
            } else {
              in_ptr1++;
              in_ptr2++;
              in_ptr3++;
              in_ptr4++;
            }
            *out_ptr1 += sum1;
            *out_ptr1_c2 += sum1_c2;

            out_ptr1++;
            out_ptr1_c2++;
          }
          out_ptr1 += output_w;
          out_ptr1_c2 += output_w;
        }
        filter_data_ch += filter_ch_size;
        filter_data_ch_c2 += filter_ch_size;
        input_data_ch += in_ch_size;
      }
    }

    int out_ch_remain_start = output_ch - output_ch % 2;
    // remain output_channel
    for (int o_c = out_ch_remain_start; o_c < output_ch; ++o_c) {
      bool issamefilter;
      const float *in_ptr1, *in_ptr2, *in_ptr3, *in_ptr4;
      const float *f1;
      const float *pad_filter0, *pad_filter1, *pad_filter2, *pad_filter3;
      float pad_filter_arr[pad_filter_ch_size];
      float *output_data_ch;
      const float *input_data_ch;
      const float *filter_data_ch;

      input_data_ch = input_data;
      output_data_ch = output_data + o_c * out_ch_size;
      filter_data_ch = filter_data + o_c * filter_ch_size * input_ch;

      for (int i_c = 0; i_c < input_ch; ++i_c) {
        f1 = filter_data_ch;
        if (!if_nopadding) {
          memset(pad_filter_arr, 0.f, sizeof(pad_filter_arr));
          for (int i = 0; i < 9; ++i) {
            int j = i / 3 * (2 * padding_w + 3) + i % 3 + padding_h * 3 +
                    padding_w * (2 * padding_h + 1);
            pad_filter_arr[j] = filter_data_ch[i];
          }
          pad_filter1 = pad_filter_arr;
          pad_filter1 += pad_filter_start;
          pad_filter0 = pad_filter1 - pad_filter_w;
          pad_filter2 = pad_filter1 + pad_filter_w;
          pad_filter3 = pad_filter2 + pad_filter_w;

        } else {
          pad_filter1 = filter_data_ch;
          pad_filter2 = pad_filter1 + 3;
          pad_filter3 = pad_filter2 + 3;
        }
        float *out_ptr1, *out_ptr2;
        out_ptr1 = output_data_ch;
        out_ptr2 = out_ptr1 + output_w;

        in_ptr1 = input_data_ch;
        in_ptr2 = in_ptr1 + input_w;
        in_ptr3 = in_ptr2 + input_w;
        in_ptr4 = in_ptr3 + input_w;

        int o_h = 0;
        for (; o_h < output_h - 1; o_h = o_h + 2) {
          if (!if_nopadding &&
              (o_h < padding_h || o_h > output_h - padding_h - 2)) {
            issamefilter = false;
          } else {
            issamefilter = true;
          }
          int o_w = 0;
          // pad left
          for (; o_w < padding_w; ++o_w) {
            float sum1 = 0;
            float sum2 = 0;

            if (issamefilter) {
#if __ARM_NEON
              float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
              float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
              float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);

              float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
              float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
              _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
              float32x4_t _sum2 = vmulq_f32(_in_ptr2, _pad_filter1);

              float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
              float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
              _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
              _sum2 = vmlaq_f32(_sum2, _in_ptr3, _pad_filter2);

              float32x4_t _in_ptr4 = vld1q_f32(in_ptr4);
              _sum2 = vmlaq_f32(_sum2, _in_ptr4, _pad_filter3);

              _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
              _sum2 = vsetq_lane_f32(sum2, _sum2, 3);

              float32x2_t _ss1 =
                  vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
              float32x2_t _ss2 =
                  vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
              float32x2_t _ssss1_ssss2 = vpadd_f32(_ss1, _ss2);

              sum1 += vget_lane_f32(_ssss1_ssss2, 0);
              sum2 += vget_lane_f32(_ssss1_ssss2, 1);
#else
              sum1 += in_ptr1[0] * pad_filter1[0];
              sum1 += in_ptr1[1] * pad_filter1[1];
              sum1 += in_ptr1[2] * pad_filter1[2];
              sum1 += in_ptr2[0] * pad_filter2[0];
              sum1 += in_ptr2[1] * pad_filter2[1];
              sum1 += in_ptr2[2] * pad_filter2[2];
              sum1 += in_ptr3[0] * pad_filter3[0];
              sum1 += in_ptr3[1] * pad_filter3[1];
              sum1 += in_ptr3[2] * pad_filter3[2];

              sum2 += in_ptr2[0] * pad_filter1[0];
              sum2 += in_ptr2[1] * pad_filter1[1];
              sum2 += in_ptr2[2] * pad_filter1[2];
              sum2 += in_ptr3[0] * pad_filter2[0];
              sum2 += in_ptr3[1] * pad_filter2[1];
              sum2 += in_ptr3[2] * pad_filter2[2];
              sum2 += in_ptr4[0] * pad_filter3[0];
              sum2 += in_ptr4[1] * pad_filter3[1];
              sum2 += in_ptr4[2] * pad_filter3[2];
#endif
            } else {
#if __ARM_NEON
              float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
              float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
              float32x4_t _pad_filter0 = vld1q_f32(pad_filter0);

              float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
              float32x4_t _sum2 = vmulq_f32(_in_ptr1, _pad_filter0);
              float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
              float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);

              _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
              _sum2 = vmlaq_f32(_sum2, _in_ptr2, _pad_filter1);

              float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
              float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);

              _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
              _sum2 = vmlaq_f32(_sum2, _in_ptr3, _pad_filter2);

              _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
              _sum2 = vsetq_lane_f32(sum2, _sum2, 3);

              float32x2_t _ss1 =
                  vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
              float32x2_t _ss2 =
                  vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
              float32x2_t _ssss1_ssss2 = vpadd_f32(_ss1, _ss2);

              sum1 += vget_lane_f32(_ssss1_ssss2, 0);
              sum2 += vget_lane_f32(_ssss1_ssss2, 1);
#else
              sum1 += in_ptr1[0] * pad_filter1[0];
              sum1 += in_ptr1[1] * pad_filter1[1];
              sum1 += in_ptr1[2] * pad_filter1[2];
              sum1 += in_ptr2[0] * pad_filter2[0];
              sum1 += in_ptr2[1] * pad_filter2[1];
              sum1 += in_ptr2[2] * pad_filter2[2];
              sum1 += in_ptr3[0] * pad_filter3[0];
              sum1 += in_ptr3[1] * pad_filter3[1];
              sum1 += in_ptr3[2] * pad_filter3[2];

              sum2 += in_ptr1[0] * pad_filter0[0];
              sum2 += in_ptr1[1] * pad_filter0[1];
              sum2 += in_ptr1[2] * pad_filter0[2];
              sum2 += in_ptr2[0] * pad_filter1[0];
              sum2 += in_ptr2[1] * pad_filter1[1];
              sum2 += in_ptr2[2] * pad_filter1[2];
              sum2 += in_ptr3[0] * pad_filter2[0];
              sum2 += in_ptr3[1] * pad_filter2[1];
              sum2 += in_ptr3[2] * pad_filter2[2];
#endif
            }
            if (!if_nopadding &&
                (o_w < padding_w || o_w > output_w - padding_w - 2)) {
              pad_filter0--;
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;
            } else {
              in_ptr1++;
              in_ptr2++;
              in_ptr3++;
              in_ptr4++;
            }
            *out_ptr1 += sum1;
            *out_ptr2 += sum2;

            out_ptr1++;
            out_ptr2++;
          }
            // valid
#if __ARM_NEON
#if __aarch64__
          if (issamefilter) {
            int loop = (output_w - 2 * padding_w) >> 2;
            o_w += loop * 4;

            if (loop > 0) {
              asm volatile(
                  "prfm   pldl1keep, [%[f1], #256]          \n\t"

                  "ld1   {v0.4s, v1.4s}, [%[f1]]            \n\t"
                  "add        %[f1], %[f1], #32             \n\t"

                  "ld1   {v4.s}[0], [%[f1]]                 \n\t"
                  "sub        %[f1],%[f1], #32              \n\t"

                  "0:                                       \n\t"
                  // load out_ptr
                  "prfm   pldl1keep, [%[out_ptr1], #128]    \n\t"
                  "prfm   pldl1keep, [%[out_ptr2], #128]    \n\t"

                  "ld1   {v12.4s}, [%[out_ptr1]]            \n\t"
                  "ld1   {v14.4s}, [%[out_ptr2]]            \n\t"

                  // in_ptr1 + in_ptr4 multiply
                  "prfm   pldl1keep, [%[in_ptr1], #192]     \n\t"
                  "prfm   pldl1keep, [%[in_ptr4], #192]     \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[in_ptr1]]       \n\t"
                  "add        %[in_ptr1],%[in_ptr1], #16    \n\t"

                  "ld1   {v6.d}[1], [%[in_ptr4]]            \n\t"
                  "add        %[in_ptr4],%[in_ptr4], #8     \n\t"
                  "ld1   {v7.4s}, [%[in_ptr4]]              \n\t"
                  "add        %[in_ptr4],%[in_ptr4], #8     \n\t"

                  "ext    v8.16b, v5.16b, v6.16b, #4        \n\t"
                  "fmla   v12.4s, v5.4s, v0.s[0]            \n\t"

                  "ext    v9.16b, v6.16b, v7.16b, #8        \n\t"
                  "fmla   v14.4s, v7.4s, v4.s[0]            \n\t"

                  "ext    v10.16b, v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v0.s[1]            \n\t"

                  "ext    v11.16b, v6.16b, v7.16b, #12      \n\t"
                  "fmla   v14.4s, v9.4s, v1.s[2]            \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[in_ptr2]]       \n\t"
                  "add        %[in_ptr2],%[in_ptr2], #16    \n\t"

                  "fmla   v12.4s, v10.4s, v0.s[2]           \n\t"
                  "fmla   v14.4s, v11.4s, v1.s[3]           \n\t"

                  // in_ptr2 multiply
                  "ext    v8.16b,  v5.16b, v6.16b, #4       \n\t"
                  "fmla   v12.4s, v5.4s, v0.s[3]            \n\t"
                  "fmla   v14.4s, v5.4s, v0.s[0]            \n\t"

                  "ext    v9.16b,  v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v1.s[0]            \n\t"
                  "fmla   v14.4s, v8.4s, v0.s[1]            \n\t"

                  "ld1   {v6.d}[1], [%[in_ptr3]]            \n\t"
                  "add        %[in_ptr3],%[in_ptr3], #8     \n\t"
                  "ld1   {v7.4s}, [%[in_ptr3]]              \n\t"

                  "add        %[in_ptr3],%[in_ptr3], #8     \n\t"
                  "fmla   v12.4s, v9.4s, v1.s[1]            \n\t"
                  "fmla   v14.4s, v9.4s, v0.s[2]            \n\t"

                  // in_ptr3 multiply
                  "ext    v10.16b, v6.16b, v7.16b, #8       \n\t"
                  "fmla   v12.4s, v7.4s, v4.s[0]            \n\t"
                  "fmla   v14.4s, v7.4s, v1.s[1]            \n\t"

                  "ext    v11.16b, v6.16b,  v7.16b, #12     \n\t"
                  "fmla   v12.4s, v10.4s, v1.s[2]           \n\t"
                  "fmla   v14.4s, v10.4s, v0.s[3]           \n\t"

                  "fmla   v12.4s, v11.4s, v1.s[3]           \n\t"
                  "fmla   v14.4s, v11.4s, v1.s[0]           \n\t"

                  // store out_ptr
                  "st1   {v12.4s}, [%[out_ptr1]], #16       \n\t"
                  "st1   {v14.4s}, [%[out_ptr2]], #16       \n\t"

                  // cycle
                  "subs       %[loop],%[loop], #1   \n\t"
                  "bne        0b                            \n\t"

                  : [loop] "+r"(loop), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr2] "+r"(out_ptr2), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3),
                    [in_ptr4] "+r"(in_ptr4)
                  : [f1] "r"(f1)
                  : "cc", "memory", "v0", "v1", "v4", "v5", "v6", "v7", "v8",
                    "v9", "v10", "v11", "v12", "v14");
            }
          }
          if (!if_nopadding && o_w == output_w - padding_w) {
            pad_filter0--;
            pad_filter1--;
            pad_filter2--;
            pad_filter3--;

            in_ptr1--;
            in_ptr2--;
            in_ptr3--;
            in_ptr4--;
          }
#else
          if (issamefilter) {
            int loop = (output_w - 2 * padding_w) >> 2;
            o_w += loop * 4;

            if (loop > 0) {
              asm volatile(
                  "pld        [%[f1], #256]                 \n\t"
                  "vld1.f32   {d0-d3}, [%[f1]]              \n\t"
                  "add        %[f1], #32                    \n\t"

                  "vld1.f32   {d8[0]}, [%[f1]]              \n\t"
                  "sub        %[f1], #32                    \n\t"

                  "0:                                       \n\t"
                  // load out_ptr
                  "pld        [%[out_ptr1], #128]           \n\t"
                  "pld        [%[out_ptr2], #128]           \n\t"

                  "vld1.f32   {d24, d25}, [%[out_ptr1]]     \n\t"
                  "vld1.f32   {d28, d29}, [%[out_ptr2]]     \n\t"

                  // in_ptr1 + in_ptr4 multiply
                  "pld        [%[in_ptr1], #192]            \n\t"
                  "pld        [%[in_ptr4], #192]            \n\t"

                  "vld1.f32   {d10-d12}, [%[in_ptr1]]       \n\t"
                  "add        %[in_ptr1], #16               \n\t"

                  "vld1.f32   {d13-d15}, [%[in_ptr4]]       \n\t"
                  "add        %[in_ptr4], #16               \n\t"

                  "vext.32    q8, q5, q6, #1                \n\t"
                  "vmla.f32   q12, q5, d0[0]                \n\t"

                  "vext.32    q9, q6, q7, #2                \n\t"
                  "vmla.f32   q14, q7, d8[0]                \n\t"

                  "vext.32    q10, q5, q6, #2               \n\t"
                  "vmla.f32   q12, q8, d0[1]                \n\t"

                  "vext.32    q11, q6, q7, #3               \n\t"
                  "vmla.f32   q14, q9, d3[0]                \n\t"

                  "vld1.f32   {d10-d12}, [%[in_ptr2]]       \n\t"
                  "add        %[in_ptr2], #16               \n\t"

                  "vmla.f32   q12, q10, d1[0]               \n\t"
                  "vmla.f32   q14, q11, d3[1]               \n\t"

                  // in_ptr2 multiply
                  "vext.32    q8, q5, q6, #1                \n\t"
                  "vmla.f32   q12, q5, d1[1]                \n\t"
                  "vmla.f32   q14, q5, d0[0]                \n\t"

                  "vext.32    q9, q5, q6, #2                \n\t"
                  "vmla.f32   q12, q8, d2[0]                \n\t"
                  "vmla.f32   q14, q8, d0[1]                \n\t"

                  "vld1.f32   {d13-d15}, [%[in_ptr3]]       \n\t"
                  "add        %[in_ptr3], #16               \n\t"

                  "vmla.f32   q12, q9, d2[1]                \n\t"
                  "vmla.f32   q14, q9, d1[0]                \n\t"

                  // in_ptr3 multiply
                  "vext.32    q10, q6, q7, #2               \n\t"
                  "vmla.f32   q12, q7, d8[0]                \n\t"
                  "vmla.f32   q14, q7, d2[1]                \n\t"

                  "vext.32    q11, q6, q7, #3               \n\t"
                  "vmla.f32   q12, q10, d3[0]               \n\t"
                  "vmla.f32   q14, q10, d1[1]               \n\t"

                  "vmla.f32   q12, q11, d3[1]               \n\t"
                  "vmla.f32   q14, q11, d2[0]               \n\t"

                  // store out_ptr
                  "subs       %[loop], #1               \n\t"
                  "vst1.f32   {d24, d25}, [%[out_ptr1]]!    \n\t"
                  "vst1.f32   {d28, d29}, [%[out_ptr2]]!    \n\t"

                  // cycle
                  "bne        0b                            \n\t"

                  : [loop] "+r"(loop), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr2] "+r"(out_ptr2), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3),
                    [in_ptr4] "+r"(in_ptr4)
                  : [f1] "r"(f1)
                  : "cc", "memory", "q0", "q1", "q4", "q5", "q6", "q7", "q8",
                    "q9", "q10", "q11", "q12", "q14");
            }
          }
          if (!if_nopadding && o_w == output_w - padding_w) {
            pad_filter0--;
            pad_filter1--;
            pad_filter2--;
            pad_filter3--;

            in_ptr1--;
            in_ptr2--;
            in_ptr3--;
            in_ptr4--;
          }
#endif  // __aarch64__
#endif  // __ARM_NEON

          // remain output_width
          for (; o_w < output_w; ++o_w) {
            float sum1 = 0;
            float sum2 = 0;

            if (issamefilter) {
#if __ARM_NEON
              float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
              float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
              float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);

              float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
              float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
              _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
              float32x4_t _sum2 = vmulq_f32(_in_ptr2, _pad_filter1);

              float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
              float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
              _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
              _sum2 = vmlaq_f32(_sum2, _in_ptr3, _pad_filter2);

              float32x4_t _in_ptr4 = vld1q_f32(in_ptr4);
              _sum2 = vmlaq_f32(_sum2, _in_ptr4, _pad_filter3);

              _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
              _sum2 = vsetq_lane_f32(sum2, _sum2, 3);

              float32x2_t _ss1 =
                  vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
              float32x2_t _ss2 =
                  vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
              float32x2_t _ssss1_ssss2 = vpadd_f32(_ss1, _ss2);

              sum1 += vget_lane_f32(_ssss1_ssss2, 0);
              sum2 += vget_lane_f32(_ssss1_ssss2, 1);
#else
              sum1 += in_ptr1[0] * pad_filter1[0];
              sum1 += in_ptr1[1] * pad_filter1[1];
              sum1 += in_ptr1[2] * pad_filter1[2];
              sum1 += in_ptr2[0] * pad_filter2[0];
              sum1 += in_ptr2[1] * pad_filter2[1];
              sum1 += in_ptr2[2] * pad_filter2[2];
              sum1 += in_ptr3[0] * pad_filter3[0];
              sum1 += in_ptr3[1] * pad_filter3[1];
              sum1 += in_ptr3[2] * pad_filter3[2];

              sum2 += in_ptr2[0] * pad_filter1[0];
              sum2 += in_ptr2[1] * pad_filter1[1];
              sum2 += in_ptr2[2] * pad_filter1[2];
              sum2 += in_ptr3[0] * pad_filter2[0];
              sum2 += in_ptr3[1] * pad_filter2[1];
              sum2 += in_ptr3[2] * pad_filter2[2];
              sum2 += in_ptr4[0] * pad_filter3[0];
              sum2 += in_ptr4[1] * pad_filter3[1];
              sum2 += in_ptr4[2] * pad_filter3[2];
#endif
            } else {
#if __ARM_NEON
              float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
              float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
              float32x4_t _pad_filter0 = vld1q_f32(pad_filter0);

              float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
              float32x4_t _sum2 = vmulq_f32(_in_ptr1, _pad_filter0);
              float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
              float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);

              _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
              _sum2 = vmlaq_f32(_sum2, _in_ptr2, _pad_filter1);
              float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
              float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);

              _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
              _sum2 = vmlaq_f32(_sum2, _in_ptr3, _pad_filter2);
              _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
              _sum2 = vsetq_lane_f32(sum2, _sum2, 3);

              float32x2_t _ss1 =
                  vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
              float32x2_t _ss2 =
                  vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
              float32x2_t _ssss1_ssss2 = vpadd_f32(_ss1, _ss2);

              sum1 += vget_lane_f32(_ssss1_ssss2, 0);
              sum2 += vget_lane_f32(_ssss1_ssss2, 1);
#else
              sum1 += in_ptr1[0] * pad_filter1[0];
              sum1 += in_ptr1[1] * pad_filter1[1];
              sum1 += in_ptr1[2] * pad_filter1[2];
              sum1 += in_ptr2[0] * pad_filter2[0];
              sum1 += in_ptr2[1] * pad_filter2[1];
              sum1 += in_ptr2[2] * pad_filter2[2];
              sum1 += in_ptr3[0] * pad_filter3[0];
              sum1 += in_ptr3[1] * pad_filter3[1];
              sum1 += in_ptr3[2] * pad_filter3[2];

              sum2 += in_ptr1[0] * pad_filter0[0];
              sum2 += in_ptr1[1] * pad_filter0[1];
              sum2 += in_ptr1[2] * pad_filter0[2];
              sum2 += in_ptr2[0] * pad_filter1[0];
              sum2 += in_ptr2[1] * pad_filter1[1];
              sum2 += in_ptr2[2] * pad_filter1[2];
              sum2 += in_ptr3[0] * pad_filter2[0];
              sum2 += in_ptr3[1] * pad_filter2[1];
              sum2 += in_ptr3[2] * pad_filter2[2];
#endif
            }
            if (!if_nopadding &&
                (o_w < padding_w || o_w > output_w - padding_w - 2)) {
              pad_filter0--;
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;
            } else {
              in_ptr1++;
              in_ptr2++;
              in_ptr3++;
              in_ptr4++;
            }
            *out_ptr1 += sum1;
            *out_ptr2 += sum2;

            out_ptr1++;
            out_ptr2++;
          }
          if (if_nopadding) {
            in_ptr1 += 2 + input_w;
            in_ptr2 += 2 + input_w;
            in_ptr3 += 2 + input_w;
            in_ptr4 += 2 + input_w;
          } else if (o_h == padding_h - 1 || o_h == output_h - padding_h - 2) {
            in_ptr1 += 3;
            in_ptr2 += 3;
            in_ptr3 += 3;
            in_ptr4 += 3;

            pad_filter0 -= 2;
            pad_filter1 -= 2;
            pad_filter2 -= 2;
            pad_filter3 -= 2;

          } else if (issamefilter) {
            in_ptr1 += 3 + input_w;
            in_ptr2 += 3 + input_w;
            in_ptr3 += 3 + input_w;
            in_ptr4 += 3 + input_w;

            pad_filter0 += 2 * padding_w + 1;
            pad_filter1 += 2 * padding_w + 1;
            pad_filter2 += 2 * padding_w + 1;
            pad_filter3 += 2 * padding_w + 1;

          } else {
            pad_filter0 -= 3 + 2 * padding_w + 2;
            pad_filter1 -= 3 + 2 * padding_w + 2;
            pad_filter2 -= 3 + 2 * padding_w + 2;
            pad_filter3 -= 3 + 2 * padding_w + 2;

            in_ptr1 -= input_w - 3;
            in_ptr2 -= input_w - 3;
            in_ptr3 -= input_w - 3;
            in_ptr4 -= input_w - 3;
          }
          out_ptr1 += output_w;
          out_ptr2 += output_w;
        }

        // remain output_height
        for (; o_h < output_h; ++o_h) {
          for (int o_w = 0; o_w < output_w; ++o_w) {
            float sum1 = 0;
#if __ARM_NEON
            float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
            float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
            float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);

            float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
            float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
            _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);

            float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
            float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
            _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);

            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ssss1_ssss1 = vpadd_f32(_ss1, _ss1);
            sum1 += vget_lane_f32(_ssss1_ssss1, 0);
#else
            sum1 += in_ptr1[0] * pad_filter1[0];
            sum1 += in_ptr1[1] * pad_filter1[1];
            sum1 += in_ptr1[2] * pad_filter1[2];
            sum1 += in_ptr2[0] * pad_filter2[0];
            sum1 += in_ptr2[1] * pad_filter2[1];
            sum1 += in_ptr2[2] * pad_filter2[2];
            sum1 += in_ptr3[0] * pad_filter3[0];
            sum1 += in_ptr3[1] * pad_filter3[1];
            sum1 += in_ptr3[2] * pad_filter3[2];
#endif
            if (!if_nopadding &&
                (o_w < padding_w || o_w > output_w - padding_w - 2)) {
              pad_filter0--;
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;

            } else {
              in_ptr1++;
              in_ptr2++;
              in_ptr3++;
              in_ptr4++;
            }
            *out_ptr1 += sum1;
            out_ptr1++;
          }
          out_ptr1 += output_w;
        }
        filter_data_ch += filter_ch_size;
        input_data_ch += in_ch_size;
      }
    }
    input_data += in_batch_size;
    output_data += out_batch_size;
  }
}

template <>
void SlidingwindowConv3x3s2<float, float>(const framework::Tensor *input,
                                          const framework::Tensor *filter,
                                          const std::vector<int> &paddings,
                                          framework::Tensor *output) {
  const int batch = input->dims()[0];
  const int input_ch = input->dims()[1];
  const int input_h = input->dims()[2];
  const int input_w = input->dims()[3];
  const int output_ch = output->dims()[1];
  const int output_h = output->dims()[2];
  const int output_w = output->dims()[3];
  const int padding_h = paddings[0];
  const int padding_w = paddings[1];

  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();
  const float *filter_data = filter->data<float>();

  const int in_ch_size = input_h * input_w;
  const int in_batch_size = input_ch * in_ch_size;
  const int out_ch_size = output_h * output_w;
  const int out_batch_size = output_ch * out_ch_size;
  const int out_size = batch * out_batch_size;
  const int filter_ch_size = 9;
  const int pad_filter_ch_size = (2 * padding_h + 3) * (2 * padding_w + 3);
  const int pad_filter_start =
      2 * padding_h * (2 * padding_w + 3) + 2 * padding_w;
  const int pad_filter_w = 3 + padding_w * 2;

  bool if_nopadding = false;
  const bool if_exact_in_w = (input_w + 2 * padding_w - 3) % 2 == 0;
  const bool if_exact_in_h = (input_h + 2 * padding_h - 3) % 2 == 0;
  const bool if_odd_pad_w = padding_w % 2 == 1;
  const bool if_odd_pad_h = padding_h % 2 == 1;

  int valid_w_start = padding_w >> 1;
  int valid_h_start = padding_h >> 1;
  int valid_w_end = output_w - valid_w_start - 2;
  int valid_h_end = output_h - valid_h_start - 2;
  const int remain_stride_w = input_w + 2 * padding_w - 2 * output_w;
#if __ARM_NEON
  float *out_ptr = output_data;
  int remain = out_size & 0x3;
  float32x4_t _zero = vdupq_n_f32(0.0);

  for (int i = 0; i < out_size; i += 4) {
    vst1q_f32(out_ptr, _zero);
    out_ptr += 4;
  }
  switch (remain) {
    case 1:
      vst1q_lane_f32(out_ptr, _zero, 0);
      break;
    case 2:
      vst1_f32(out_ptr, vget_low_f32(_zero));
      break;
    case 3:
      vst1_f32(out_ptr, vget_low_f32(_zero));
      vst1q_lane_f32(out_ptr + 2, _zero, 0);
      break;
  }
#else
#pragma omp parallel for
  for (int i = 0; i < out_size; ++i) {
    output_data[i] = 0;
  }
#endif

  if (padding_h == 0 && padding_w == 0) {
    if_nopadding = true;
    valid_w_start = -1;
    valid_h_start = -1;
    valid_w_end = output_w;
    valid_h_end = output_h;
  }

  for (int b = 0; b < batch; ++b) {
#pragma omp parallel for
    for (int o_c = 0; o_c < output_ch - 7; o_c += 8) {
      const float *f1;
      const float *in_ptr1, *in_ptr2, *in_ptr3;
      const float *pad_filter1, *pad_filter2, *pad_filter3;
      const float *pad_filter1_c2, *pad_filter2_c2, *pad_filter3_c2;
      const float *pad_filter1_c3, *pad_filter2_c3, *pad_filter3_c3;
      const float *pad_filter1_c4, *pad_filter2_c4, *pad_filter3_c4;
      const float *pad_filter1_c5, *pad_filter2_c5, *pad_filter3_c5;
      const float *pad_filter1_c6, *pad_filter2_c6, *pad_filter3_c6;
      const float *pad_filter1_c7, *pad_filter2_c7, *pad_filter3_c7;
      const float *pad_filter1_c8, *pad_filter2_c8, *pad_filter3_c8;

      float reform_filter_arr[72];
      float pad_filter_arr[pad_filter_ch_size];
      float pad_filter_arr_c2[pad_filter_ch_size];
      float pad_filter_arr_c3[pad_filter_ch_size];
      float pad_filter_arr_c4[pad_filter_ch_size];
      float pad_filter_arr_c5[pad_filter_ch_size];
      float pad_filter_arr_c6[pad_filter_ch_size];
      float pad_filter_arr_c7[pad_filter_ch_size];
      float pad_filter_arr_c8[pad_filter_ch_size];

      float *output_data_ch;
      float *output_data_ch_2;
      float *output_data_ch_3;
      float *output_data_ch_4;
      float *output_data_ch_5;
      float *output_data_ch_6;
      float *output_data_ch_7;
      float *output_data_ch_8;

      const float *input_data_ch;
      const float *filter_data_ch;
      const float *filter_data_ch_c2;
      const float *filter_data_ch_c3;
      const float *filter_data_ch_c4;
      const float *filter_data_ch_c5;
      const float *filter_data_ch_c6;
      const float *filter_data_ch_c7;
      const float *filter_data_ch_c8;

      filter_data_ch = filter_data + o_c * filter_ch_size * input_ch;
      filter_data_ch_c2 = filter_data + (o_c + 1) * filter_ch_size * input_ch;
      filter_data_ch_c3 = filter_data + (o_c + 2) * filter_ch_size * input_ch;
      filter_data_ch_c4 = filter_data + (o_c + 3) * filter_ch_size * input_ch;
      filter_data_ch_c5 = filter_data + (o_c + 4) * filter_ch_size * input_ch;
      filter_data_ch_c6 = filter_data + (o_c + 5) * filter_ch_size * input_ch;
      filter_data_ch_c7 = filter_data + (o_c + 6) * filter_ch_size * input_ch;
      filter_data_ch_c8 = filter_data + (o_c + 7) * filter_ch_size * input_ch;

      input_data_ch = input_data;
      output_data_ch = output_data + o_c * out_ch_size;
      output_data_ch_2 = output_data + (o_c + 1) * out_ch_size;
      output_data_ch_3 = output_data + (o_c + 2) * out_ch_size;
      output_data_ch_4 = output_data + (o_c + 3) * out_ch_size;
      output_data_ch_5 = output_data + (o_c + 4) * out_ch_size;
      output_data_ch_6 = output_data + (o_c + 5) * out_ch_size;
      output_data_ch_7 = output_data + (o_c + 6) * out_ch_size;
      output_data_ch_8 = output_data + (o_c + 7) * out_ch_size;

      for (int i_c = 0; i_c < input_ch; ++i_c) {
        int k = 0;
        for (int i = 0; i < 9; ++i) {
          for (int j = 0; j < 8; ++j) {
            reform_filter_arr[k++] = filter_data_ch[i + input_ch * 9 * j];
          }
        }

        f1 = reform_filter_arr;

        if (!if_nopadding) {
          memset(pad_filter_arr, 0.f, sizeof(pad_filter_arr));
          memset(pad_filter_arr_c2, 0.f, sizeof(pad_filter_arr_c2));
          memset(pad_filter_arr_c3, 0.f, sizeof(pad_filter_arr_c3));
          memset(pad_filter_arr_c4, 0.f, sizeof(pad_filter_arr_c4));
          memset(pad_filter_arr_c5, 0.f, sizeof(pad_filter_arr_c5));
          memset(pad_filter_arr_c6, 0.f, sizeof(pad_filter_arr_c6));
          memset(pad_filter_arr_c7, 0.f, sizeof(pad_filter_arr_c7));
          memset(pad_filter_arr_c8, 0.f, sizeof(pad_filter_arr_c8));

          for (int i = 0; i < 9; ++i) {
            int j = i / 3 * (2 * padding_w + 3) + i % 3 + padding_h * 3 +
                    padding_w * (2 * padding_h + 1);
            pad_filter_arr[j] = filter_data_ch[i];
            pad_filter_arr_c2[j] = filter_data_ch_c2[i];
            pad_filter_arr_c3[j] = filter_data_ch_c3[i];
            pad_filter_arr_c4[j] = filter_data_ch_c4[i];
            pad_filter_arr_c5[j] = filter_data_ch_c5[i];
            pad_filter_arr_c6[j] = filter_data_ch_c6[i];
            pad_filter_arr_c7[j] = filter_data_ch_c7[i];
            pad_filter_arr_c8[j] = filter_data_ch_c8[i];
          }

          pad_filter1 = pad_filter_arr;
          pad_filter1 += pad_filter_start;
          pad_filter2 = pad_filter1 + pad_filter_w;
          pad_filter3 = pad_filter2 + pad_filter_w;

          pad_filter1_c2 = pad_filter_arr_c2;
          pad_filter1_c2 += pad_filter_start;
          pad_filter2_c2 = pad_filter1_c2 + pad_filter_w;
          pad_filter3_c2 = pad_filter2_c2 + pad_filter_w;

          pad_filter1_c3 = pad_filter_arr_c3;
          pad_filter1_c3 += pad_filter_start;
          pad_filter2_c3 = pad_filter1_c3 + pad_filter_w;
          pad_filter3_c3 = pad_filter2_c3 + pad_filter_w;

          pad_filter1_c4 = pad_filter_arr_c4;
          pad_filter1_c4 += pad_filter_start;
          pad_filter2_c4 = pad_filter1_c4 + pad_filter_w;
          pad_filter3_c4 = pad_filter2_c4 + pad_filter_w;

          pad_filter1_c5 = pad_filter_arr_c5;
          pad_filter1_c5 += pad_filter_start;
          pad_filter2_c5 = pad_filter1_c5 + pad_filter_w;
          pad_filter3_c5 = pad_filter2_c5 + pad_filter_w;

          pad_filter1_c6 = pad_filter_arr_c6;
          pad_filter1_c6 += pad_filter_start;
          pad_filter2_c6 = pad_filter1_c6 + pad_filter_w;
          pad_filter3_c6 = pad_filter2_c6 + pad_filter_w;

          pad_filter1_c7 = pad_filter_arr_c7;
          pad_filter1_c7 += pad_filter_start;
          pad_filter2_c7 = pad_filter1_c7 + pad_filter_w;
          pad_filter3_c7 = pad_filter2_c7 + pad_filter_w;

          pad_filter1_c8 = pad_filter_arr_c8;
          pad_filter1_c8 += pad_filter_start;
          pad_filter2_c8 = pad_filter1_c8 + pad_filter_w;
          pad_filter3_c8 = pad_filter2_c8 + pad_filter_w;
        } else {
          pad_filter1 = filter_data_ch;
          pad_filter2 = pad_filter1 + 3;
          pad_filter3 = pad_filter2 + 3;

          pad_filter1_c2 = filter_data_ch_c2;
          pad_filter2_c2 = pad_filter1_c2 + 3;
          pad_filter3_c2 = pad_filter2_c2 + 3;

          pad_filter1_c3 = filter_data_ch_c3;
          pad_filter2_c3 = pad_filter1_c3 + 3;
          pad_filter3_c3 = pad_filter2_c3 + 3;

          pad_filter1_c4 = filter_data_ch_c4;
          pad_filter2_c4 = pad_filter1_c4 + 3;
          pad_filter3_c4 = pad_filter2_c4 + 3;

          pad_filter1_c5 = filter_data_ch_c5;
          pad_filter2_c5 = pad_filter1_c5 + 3;
          pad_filter3_c5 = pad_filter2_c5 + 3;

          pad_filter1_c6 = filter_data_ch_c6;
          pad_filter2_c6 = pad_filter1_c6 + 3;
          pad_filter3_c6 = pad_filter2_c6 + 3;

          pad_filter1_c7 = filter_data_ch_c7;
          pad_filter2_c7 = pad_filter1_c7 + 3;
          pad_filter3_c7 = pad_filter2_c7 + 3;

          pad_filter1_c8 = filter_data_ch_c8;
          pad_filter2_c8 = pad_filter1_c8 + 3;
          pad_filter3_c8 = pad_filter2_c8 + 3;
        }
        float *out_ptr1;
        float *out_ptr1_c2;
        float *out_ptr1_c3;
        float *out_ptr1_c4;
        float *out_ptr1_c5;
        float *out_ptr1_c6;
        float *out_ptr1_c7;
        float *out_ptr1_c8;

        out_ptr1 = output_data_ch;
        out_ptr1_c2 = output_data_ch_2;
        out_ptr1_c3 = output_data_ch_3;
        out_ptr1_c4 = output_data_ch_4;
        out_ptr1_c5 = output_data_ch_5;
        out_ptr1_c6 = output_data_ch_6;
        out_ptr1_c7 = output_data_ch_7;
        out_ptr1_c8 = output_data_ch_8;

        in_ptr1 = input_data_ch;
        in_ptr2 = in_ptr1 + input_w;
        in_ptr3 = in_ptr2 + input_w;

        int o_h = 0;

        for (; o_h < output_h; ++o_h) {
          int o_w = 0;

          // pad left
          for (; o_w <= valid_w_start; ++o_w) {
            float sum1 = 0;
            float sum1_c2 = 0;
            float sum1_c3 = 0;
            float sum1_c4 = 0;
            float sum1_c5 = 0;
            float sum1_c6 = 0;
            float sum1_c7 = 0;
            float sum1_c8 = 0;
#if __ARM_NEON
            float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
            float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
            float32x4_t _pad_filter1_c2 = vld1q_f32(pad_filter1_c2);
            float32x4_t _pad_filter1_c3 = vld1q_f32(pad_filter1_c3);
            float32x4_t _pad_filter1_c4 = vld1q_f32(pad_filter1_c4);
            float32x4_t _pad_filter1_c5 = vld1q_f32(pad_filter1_c5);
            float32x4_t _pad_filter1_c6 = vld1q_f32(pad_filter1_c6);
            float32x4_t _pad_filter1_c7 = vld1q_f32(pad_filter1_c7);
            float32x4_t _pad_filter1_c8 = vld1q_f32(pad_filter1_c8);

            float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
            float32x4_t _sum1_c2 = vmulq_f32(_in_ptr1, _pad_filter1_c2);
            float32x4_t _sum1_c3 = vmulq_f32(_in_ptr1, _pad_filter1_c3);
            float32x4_t _sum1_c4 = vmulq_f32(_in_ptr1, _pad_filter1_c4);
            float32x4_t _sum1_c5 = vmulq_f32(_in_ptr1, _pad_filter1_c5);
            float32x4_t _sum1_c6 = vmulq_f32(_in_ptr1, _pad_filter1_c6);
            float32x4_t _sum1_c7 = vmulq_f32(_in_ptr1, _pad_filter1_c7);
            float32x4_t _sum1_c8 = vmulq_f32(_in_ptr1, _pad_filter1_c8);

            float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
            float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
            float32x4_t _pad_filter2_c2 = vld1q_f32(pad_filter2_c2);
            float32x4_t _pad_filter2_c3 = vld1q_f32(pad_filter2_c3);
            float32x4_t _pad_filter2_c4 = vld1q_f32(pad_filter2_c4);
            float32x4_t _pad_filter2_c5 = vld1q_f32(pad_filter2_c5);
            float32x4_t _pad_filter2_c6 = vld1q_f32(pad_filter2_c6);
            float32x4_t _pad_filter2_c7 = vld1q_f32(pad_filter2_c7);
            float32x4_t _pad_filter2_c8 = vld1q_f32(pad_filter2_c8);

            _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
            _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr2, _pad_filter2_c2);
            _sum1_c3 = vmlaq_f32(_sum1_c3, _in_ptr2, _pad_filter2_c3);
            _sum1_c4 = vmlaq_f32(_sum1_c4, _in_ptr2, _pad_filter2_c4);
            _sum1_c5 = vmlaq_f32(_sum1_c5, _in_ptr2, _pad_filter2_c5);
            _sum1_c6 = vmlaq_f32(_sum1_c6, _in_ptr2, _pad_filter2_c6);
            _sum1_c7 = vmlaq_f32(_sum1_c7, _in_ptr2, _pad_filter2_c7);
            _sum1_c8 = vmlaq_f32(_sum1_c8, _in_ptr2, _pad_filter2_c8);

            float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
            float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
            float32x4_t _pad_filter3_c2 = vld1q_f32(pad_filter3_c2);
            float32x4_t _pad_filter3_c3 = vld1q_f32(pad_filter3_c3);
            float32x4_t _pad_filter3_c4 = vld1q_f32(pad_filter3_c4);
            float32x4_t _pad_filter3_c5 = vld1q_f32(pad_filter3_c5);
            float32x4_t _pad_filter3_c6 = vld1q_f32(pad_filter3_c6);
            float32x4_t _pad_filter3_c7 = vld1q_f32(pad_filter3_c7);
            float32x4_t _pad_filter3_c8 = vld1q_f32(pad_filter3_c8);

            _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
            _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr3, _pad_filter3_c2);
            _sum1_c3 = vmlaq_f32(_sum1_c3, _in_ptr3, _pad_filter3_c3);
            _sum1_c4 = vmlaq_f32(_sum1_c4, _in_ptr3, _pad_filter3_c4);
            _sum1_c5 = vmlaq_f32(_sum1_c5, _in_ptr3, _pad_filter3_c5);
            _sum1_c6 = vmlaq_f32(_sum1_c6, _in_ptr3, _pad_filter3_c6);
            _sum1_c7 = vmlaq_f32(_sum1_c7, _in_ptr3, _pad_filter3_c7);
            _sum1_c8 = vmlaq_f32(_sum1_c8, _in_ptr3, _pad_filter3_c8);

            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
            _sum1_c2 = vsetq_lane_f32(sum1_c2, _sum1_c2, 3);
            _sum1_c3 = vsetq_lane_f32(sum1_c3, _sum1_c3, 3);
            _sum1_c4 = vsetq_lane_f32(sum1_c4, _sum1_c4, 3);
            _sum1_c5 = vsetq_lane_f32(sum1_c5, _sum1_c5, 3);
            _sum1_c6 = vsetq_lane_f32(sum1_c6, _sum1_c6, 3);
            _sum1_c7 = vsetq_lane_f32(sum1_c7, _sum1_c7, 3);
            _sum1_c8 = vsetq_lane_f32(sum1_c8, _sum1_c8, 3);

            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ss1_2 =
                vadd_f32(vget_low_f32(_sum1_c2), vget_high_f32(_sum1_c2));
            float32x2_t _ss1_3 =
                vadd_f32(vget_low_f32(_sum1_c3), vget_high_f32(_sum1_c3));
            float32x2_t _ss1_4 =
                vadd_f32(vget_low_f32(_sum1_c4), vget_high_f32(_sum1_c4));
            float32x2_t _ss1_5 =
                vadd_f32(vget_low_f32(_sum1_c5), vget_high_f32(_sum1_c5));
            float32x2_t _ss1_6 =
                vadd_f32(vget_low_f32(_sum1_c6), vget_high_f32(_sum1_c6));
            float32x2_t _ss1_7 =
                vadd_f32(vget_low_f32(_sum1_c7), vget_high_f32(_sum1_c7));
            float32x2_t _ss1_8 =
                vadd_f32(vget_low_f32(_sum1_c8), vget_high_f32(_sum1_c8));

            float32x2_t _ssss1_ssss1_2 = vpadd_f32(_ss1, _ss1_2);
            float32x2_t _ssss1_3_ssss1_4 = vpadd_f32(_ss1_3, _ss1_4);
            float32x2_t _ssss1_5_ssss1_6 = vpadd_f32(_ss1_5, _ss1_6);
            float32x2_t _ssss1_7_ssss1_8 = vpadd_f32(_ss1_7, _ss1_8);

            sum1 += vget_lane_f32(_ssss1_ssss1_2, 0);
            sum1_c2 += vget_lane_f32(_ssss1_ssss1_2, 1);
            sum1_c3 += vget_lane_f32(_ssss1_3_ssss1_4, 0);
            sum1_c4 += vget_lane_f32(_ssss1_3_ssss1_4, 1);
            sum1_c5 += vget_lane_f32(_ssss1_5_ssss1_6, 0);
            sum1_c6 += vget_lane_f32(_ssss1_5_ssss1_6, 1);
            sum1_c7 += vget_lane_f32(_ssss1_7_ssss1_8, 0);
            sum1_c8 += vget_lane_f32(_ssss1_7_ssss1_8, 1);
#else
            sum1 += in_ptr1[0] * pad_filter1[0];
            sum1 += in_ptr1[1] * pad_filter1[1];
            sum1 += in_ptr1[2] * pad_filter1[2];
            sum1 += in_ptr2[0] * pad_filter2[0];
            sum1 += in_ptr2[1] * pad_filter2[1];
            sum1 += in_ptr2[2] * pad_filter2[2];
            sum1 += in_ptr3[0] * pad_filter3[0];
            sum1 += in_ptr3[1] * pad_filter3[1];
            sum1 += in_ptr3[2] * pad_filter3[2];

            sum1_c2 += in_ptr1[0] * pad_filter1_c2[0];
            sum1_c2 += in_ptr1[1] * pad_filter1_c2[1];
            sum1_c2 += in_ptr1[2] * pad_filter1_c2[2];
            sum1_c2 += in_ptr2[0] * pad_filter2_c2[0];
            sum1_c2 += in_ptr2[1] * pad_filter2_c2[1];
            sum1_c2 += in_ptr2[2] * pad_filter2_c2[2];
            sum1_c2 += in_ptr3[0] * pad_filter3_c2[0];
            sum1_c2 += in_ptr3[1] * pad_filter3_c2[1];
            sum1_c2 += in_ptr3[2] * pad_filter3_c2[2];

            sum1_c3 += in_ptr1[0] * pad_filter1_c3[0];
            sum1_c3 += in_ptr1[1] * pad_filter1_c3[1];
            sum1_c3 += in_ptr1[2] * pad_filter1_c3[2];
            sum1_c3 += in_ptr2[0] * pad_filter2_c3[0];
            sum1_c3 += in_ptr2[1] * pad_filter2_c3[1];
            sum1_c3 += in_ptr2[2] * pad_filter2_c3[2];
            sum1_c3 += in_ptr3[0] * pad_filter3_c3[0];
            sum1_c3 += in_ptr3[1] * pad_filter3_c3[1];
            sum1_c3 += in_ptr3[2] * pad_filter3_c3[2];

            sum1_c4 += in_ptr1[0] * pad_filter1_c4[0];
            sum1_c4 += in_ptr1[1] * pad_filter1_c4[1];
            sum1_c4 += in_ptr1[2] * pad_filter1_c4[2];
            sum1_c4 += in_ptr2[0] * pad_filter2_c4[0];
            sum1_c4 += in_ptr2[1] * pad_filter2_c4[1];
            sum1_c4 += in_ptr2[2] * pad_filter2_c4[2];
            sum1_c4 += in_ptr3[0] * pad_filter3_c4[0];
            sum1_c4 += in_ptr3[1] * pad_filter3_c4[1];
            sum1_c4 += in_ptr3[2] * pad_filter3_c4[2];

            sum1_c5 += in_ptr1[0] * pad_filter1_c5[0];
            sum1_c5 += in_ptr1[1] * pad_filter1_c5[1];
            sum1_c5 += in_ptr1[2] * pad_filter1_c5[2];
            sum1_c5 += in_ptr2[0] * pad_filter2_c5[0];
            sum1_c5 += in_ptr2[1] * pad_filter2_c5[1];
            sum1_c5 += in_ptr2[2] * pad_filter2_c5[2];
            sum1_c5 += in_ptr3[0] * pad_filter3_c5[0];
            sum1_c5 += in_ptr3[1] * pad_filter3_c5[1];
            sum1_c5 += in_ptr3[2] * pad_filter3_c5[2];

            sum1_c6 += in_ptr1[0] * pad_filter1_c6[0];
            sum1_c6 += in_ptr1[1] * pad_filter1_c6[1];
            sum1_c6 += in_ptr1[2] * pad_filter1_c6[2];
            sum1_c6 += in_ptr2[0] * pad_filter2_c6[0];
            sum1_c6 += in_ptr2[1] * pad_filter2_c6[1];
            sum1_c6 += in_ptr2[2] * pad_filter2_c6[2];
            sum1_c6 += in_ptr3[0] * pad_filter3_c6[0];
            sum1_c6 += in_ptr3[1] * pad_filter3_c6[1];
            sum1_c6 += in_ptr3[2] * pad_filter3_c6[2];

            sum1_c7 += in_ptr1[0] * pad_filter1_c7[0];
            sum1_c7 += in_ptr1[1] * pad_filter1_c7[1];
            sum1_c7 += in_ptr1[2] * pad_filter1_c7[2];
            sum1_c7 += in_ptr2[0] * pad_filter2_c7[0];
            sum1_c7 += in_ptr2[1] * pad_filter2_c7[1];
            sum1_c7 += in_ptr2[2] * pad_filter2_c7[2];
            sum1_c7 += in_ptr3[0] * pad_filter3_c7[0];
            sum1_c7 += in_ptr3[1] * pad_filter3_c7[1];
            sum1_c7 += in_ptr3[2] * pad_filter3_c7[2];

            sum1_c8 += in_ptr1[0] * pad_filter1_c8[0];
            sum1_c8 += in_ptr1[1] * pad_filter1_c8[1];
            sum1_c8 += in_ptr1[2] * pad_filter1_c8[2];
            sum1_c8 += in_ptr2[0] * pad_filter2_c8[0];
            sum1_c8 += in_ptr2[1] * pad_filter2_c8[1];
            sum1_c8 += in_ptr2[2] * pad_filter2_c8[2];
            sum1_c8 += in_ptr3[0] * pad_filter3_c8[0];
            sum1_c8 += in_ptr3[1] * pad_filter3_c8[1];
            sum1_c8 += in_ptr3[2] * pad_filter3_c8[2];
#endif
            if (if_nopadding) {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;

            } else if (input_w > 3 &&
                       (if_odd_pad_w && o_w == valid_w_start ||
                        o_w == valid_w_end && if_odd_pad_w && if_exact_in_w ||
                        o_w == valid_w_end + 1 && !if_odd_pad_w &&
                            !if_exact_in_w)) {
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;
              pad_filter1_c2--;
              pad_filter2_c2--;
              pad_filter3_c2--;

              pad_filter1_c3--;
              pad_filter2_c3--;
              pad_filter3_c3--;
              pad_filter1_c4--;
              pad_filter2_c4--;
              pad_filter3_c4--;

              pad_filter1_c5--;
              pad_filter2_c5--;
              pad_filter3_c5--;
              pad_filter1_c6--;
              pad_filter2_c6--;
              pad_filter3_c6--;

              pad_filter1_c7--;
              pad_filter2_c7--;
              pad_filter3_c7--;
              pad_filter1_c8--;
              pad_filter2_c8--;
              pad_filter3_c8--;

              in_ptr1++;
              in_ptr2++;
              in_ptr3++;

            } else if (input_w <= 3 || o_w < valid_w_start ||
                       o_w > valid_w_end) {
              pad_filter1 -= 2;
              pad_filter2 -= 2;
              pad_filter3 -= 2;
              pad_filter1_c2 -= 2;
              pad_filter2_c2 -= 2;
              pad_filter3_c2 -= 2;

              pad_filter1_c3 -= 2;
              pad_filter2_c3 -= 2;
              pad_filter3_c3 -= 2;
              pad_filter1_c4 -= 2;
              pad_filter2_c4 -= 2;
              pad_filter3_c4 -= 2;

              pad_filter1_c5 -= 2;
              pad_filter2_c5 -= 2;
              pad_filter3_c5 -= 2;
              pad_filter1_c6 -= 2;
              pad_filter2_c6 -= 2;
              pad_filter3_c6 -= 2;

              pad_filter1_c7 -= 2;
              pad_filter2_c7 -= 2;
              pad_filter3_c7 -= 2;
              pad_filter1_c8 -= 2;
              pad_filter2_c8 -= 2;
              pad_filter3_c8 -= 2;
            } else {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;
            }
            *out_ptr1 += sum1;
            *out_ptr1_c2 += sum1_c2;
            *out_ptr1_c3 += sum1_c3;
            *out_ptr1_c4 += sum1_c4;
            *out_ptr1_c5 += sum1_c5;
            *out_ptr1_c6 += sum1_c6;
            *out_ptr1_c7 += sum1_c7;
            *out_ptr1_c8 += sum1_c8;

            out_ptr1++;
            out_ptr1_c2++;
            out_ptr1_c3++;
            out_ptr1_c4++;
            out_ptr1_c5++;
            out_ptr1_c6++;
            out_ptr1_c7++;
            out_ptr1_c8++;
          }
            // valid
#if __ARM_NEON
#if __aarch64__
          if (o_h > valid_h_start && o_h <= valid_h_end) {
            int loop = (valid_w_end - valid_w_start - 1) >> 2;
            o_w += loop * 4;

            if (loop > 0) {
              asm volatile(

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "prfm  pldl1keep, [%[in_ptr1], #288]        \n\t"

                  "ld1  {v0.4s, v1.4s}, [%[f1]], #32          \n\t"
                  "ld2   {v4.4s, v5.4s}, [%[in_ptr1]], #32    \n\t"
                  "ld2   {v6.4s, v7.4s}, [%[in_ptr1]]         \n\t"
                  "0:                                         \n\t"
                  // load out_ptr
                  "prfm  pldl1keep, [%[out_ptr1], #128]       \n\t"
                  "prfm  pldl1keep, [%[out_ptr1_c2], #128]     \n\t"
                  "prfm  pldl1keep, [%[out_ptr1_c3], #128]     \n\t"
                  "prfm  pldl1keep, [%[out_ptr1_c4], #128]     \n\t"
                  "prfm  pldl1keep, [%[out_ptr1_c5], #128]     \n\t"
                  "prfm  pldl1keep, [%[out_ptr1_c6], #128]     \n\t"
                  "prfm  pldl1keep, [%[out_ptr1_c7], #128]     \n\t"
                  "prfm  pldl1keep, [%[out_ptr1_c8], #128]     \n\t"

                  "ld1   {v8.4s}, [%[out_ptr1]]               \n\t"
                  "ld1   {v9.4s}, [%[out_ptr1_c2]]             \n\t"
                  "ld1   {v10.4s}, [%[out_ptr1_c3]]            \n\t"
                  "ld1   {v11.4s}, [%[out_ptr1_c4]]            \n\t"
                  "ld1   {v12.4s}, [%[out_ptr1_c5]]            \n\t"
                  "ld1   {v13.4s}, [%[out_ptr1_c6]]            \n\t"
                  "ld1   {v14.4s}, [%[out_ptr1_c7]]            \n\t"
                  "ld1   {v15.4s}, [%[out_ptr1_c8]]            \n\t"

                  // in_ptr1 multiply
                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "ld1   {v2.4s, v3.4s}, [%[f1]], #32         \n\t"
                  "fmla    v8.4s, v4.4s, v0.s[0]              \n\t"
                  "fmla    v9.4s, v4.4s, v0.s[1]              \n\t"
                  "fmla   v10.4s, v4.4s, v0.s[2]              \n\t"
                  "fmla   v11.4s, v4.4s, v0.s[3]              \n\t"

                  "fmla   v12.4s, v4.4s, v1.s[0]              \n\t"
                  "fmla   v13.4s, v4.4s, v1.s[1]              \n\t"
                  "fmla   v14.4s, v4.4s, v1.s[2]              \n\t"
                  "fmla   v15.4s, v4.4s, v1.s[3]              \n\t"

                  "ext    v7.16b, v4.16b, v6.16b, #4          \n\t"
                  "fmla    v8.4s, v5.4s, v2.s[0]              \n\t"
                  "fmla    v9.4s, v5.4s, v2.s[1]              \n\t"
                  "fmla   v10.4s, v5.4s, v2.s[2]              \n\t"
                  "fmla   v11.4s, v5.4s, v2.s[3]              \n\t"

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "ld1   {v0.4s, v1.4s}, [%[f1]], #32         \n\t"
                  "fmla   v12.4s, v5.4s, v3.s[0]              \n\t"
                  "fmla   v13.4s, v5.4s, v3.s[1]              \n\t"
                  "fmla   v14.4s, v5.4s, v3.s[2]              \n\t"
                  "fmla   v15.4s, v5.4s, v3.s[3]              \n\t"

                  "prfm  pldl1keep, [%[in_ptr2], #288]        \n\t"
                  "ld2    {v4.4s, v5.4s}, [%[in_ptr2]], #32   \n\t"
                  "fmla    v8.4s, v7.4s, v0.s[0]              \n\t"
                  "fmla    v9.4s, v7.4s, v0.s[1]              \n\t"
                  "fmla   v10.4s, v7.4s, v0.s[2]              \n\t"
                  "fmla   v11.4s, v7.4s, v0.s[3]              \n\t"

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "ld1   {v2.4s, v3.4s}, [%[f1]], #32         \n\t"

                  "fmla   v12.4s, v7.4s, v1.s[0]              \n\t"
                  "fmla   v13.4s, v7.4s, v1.s[1]              \n\t"
                  "fmla   v14.4s, v7.4s, v1.s[2]              \n\t"
                  "fmla   v15.4s, v7.4s, v1.s[3]              \n\t"

                  // in_ptr2 multiply
                  "ld2    {v6.4s, v7.4s}, [%[in_ptr2]]        \n\t"
                  "fmla    v8.4s, v4.4s, v2.s[0]              \n\t"
                  "fmla    v9.4s, v4.4s, v2.s[1]              \n\t"
                  "fmla   v10.4s, v4.4s, v2.s[2]              \n\t"
                  "fmla   v11.4s, v4.4s, v2.s[3]              \n\t"

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "ld1   {v0.4s, v1.4s}, [%[f1]], #32         \n\t"
                  "fmla   v12.4s, v4.4s, v3.s[0]              \n\t"
                  "fmla   v13.4s, v4.4s, v3.s[1]              \n\t"
                  "fmla   v14.4s, v4.4s, v3.s[2]              \n\t"
                  "fmla   v15.4s, v4.4s, v3.s[3]              \n\t"

                  "ext    v7.16b, v4.16b, v6.16b, #4          \n\t"
                  "fmla    v8.4s, v5.4s, v0.s[0]              \n\t"
                  "fmla    v9.4s, v5.4s, v0.s[1]              \n\t"
                  "fmla   v10.4s, v5.4s, v0.s[2]              \n\t"
                  "fmla   v11.4s, v5.4s, v0.s[3]              \n\t"

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "ld1    {v2.4s, v3.4s}, [%[f1]], #32        \n\t"
                  "fmla   v12.4s, v5.4s, v1.s[0]              \n\t"
                  "fmla   v13.4s, v5.4s, v1.s[1]              \n\t"

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "prfm  pldl1keep, [%[in_ptr3], #288]        \n\t"
                  "fmla   v14.4s, v5.4s, v1.s[2]              \n\t"
                  "fmla   v15.4s, v5.4s, v1.s[3]              \n\t"

                  "ld1  {v0.4s, v1.4s}, [%[f1]], #32          \n\t"
                  "ld2   {v4.4s, v5.4s}, [%[in_ptr3]], #32    \n\t"
                  "fmla    v8.4s, v7.4s, v2.s[0]              \n\t"
                  "fmla    v9.4s, v7.4s, v2.s[1]              \n\t"
                  "fmla   v10.4s, v7.4s, v2.s[2]              \n\t"
                  "fmla   v11.4s, v7.4s, v2.s[3]              \n\t"

                  "fmla   v12.4s, v7.4s, v3.s[0]              \n\t"
                  "fmla   v13.4s, v7.4s, v3.s[1]              \n\t"
                  "fmla   v14.4s, v7.4s, v3.s[2]              \n\t"
                  "fmla   v15.4s, v7.4s, v3.s[3]              \n\t"

                  // in_ptr3 multiply
                  "ld2   {v6.4s, v7.4s}, [%[in_ptr3]]         \n\t"
                  "fmla    v8.4s, v4.4s, v0.s[0]              \n\t"
                  "fmla    v9.4s, v4.4s, v0.s[1]              \n\t"
                  "fmla   v10.4s, v4.4s, v0.s[2]              \n\t"
                  "fmla   v11.4s, v4.4s, v0.s[3]              \n\t"

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "ld1   {v2.4s, v3.4s}, [%[f1]], #32         \n\t"
                  "fmla   v12.4s, v4.4s, v1.s[0]              \n\t"
                  "fmla   v13.4s, v4.4s, v1.s[1]              \n\t"
                  "fmla   v14.4s, v4.4s, v1.s[2]              \n\t"
                  "fmla   v15.4s, v4.4s, v1.s[3]              \n\t"

                  "ext    v7.16b, v4.16b, v6.16b, #4          \n\t"
                  "fmla    v8.4s, v5.4s, v2.s[0]              \n\t"
                  "fmla    v9.4s, v5.4s, v2.s[1]              \n\t"
                  "fmla   v10.4s, v5.4s, v2.s[2]              \n\t"
                  "fmla   v11.4s, v5.4s, v2.s[3]              \n\t"

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "ld1   {v0.4s, v1.4s}, [%[f1]], #32         \n\t"
                  "fmla   v12.4s, v5.4s, v3.s[0]              \n\t"
                  "fmla   v13.4s, v5.4s, v3.s[1]              \n\t"
                  "fmla   v14.4s, v5.4s, v3.s[2]              \n\t"
                  "fmla   v15.4s, v5.4s, v3.s[3]              \n\t"

                  "sub        %[f1], %[f1], #288              \n\t"
                  "fmla    v8.4s, v7.4s, v0.s[0]              \n\t"
                  "fmla    v9.4s, v7.4s, v0.s[1]              \n\t"
                  "fmla   v10.4s, v7.4s, v0.s[2]              \n\t"
                  "fmla   v11.4s, v7.4s, v0.s[3]              \n\t"

                  "fmla   v12.4s, v7.4s, v1.s[0]              \n\t"
                  "fmla   v13.4s, v7.4s, v1.s[1]              \n\t"
                  "fmla   v14.4s, v7.4s, v1.s[2]              \n\t"
                  "fmla   v15.4s, v7.4s, v1.s[3]              \n\t"

                  // store out_ptr
                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "prfm  pldl1keep, [%[in_ptr1], #288]        \n\t"

                  "ld1  {v0.4s, v1.4s}, [%[f1]], #32          \n\t"

                  "ld2   {v4.4s, v5.4s}, [%[in_ptr1]], #32    \n\t"
                  "st1   {v8.4s}, [%[out_ptr1]], #16          \n\t"
                  "st1   {v9.4s}, [%[out_ptr1_c2]], #16        \n\t"

                  "st1   {v10.4s}, [%[out_ptr1_c3]], #16       \n\t"
                  "st1   {v11.4s}, [%[out_ptr1_c4]], #16       \n\t"

                  "st1   {v12.4s}, [%[out_ptr1_c5]], #16       \n\t"
                  "st1   {v13.4s}, [%[out_ptr1_c6]], #16       \n\t"

                  "ld2   {v6.4s, v7.4s}, [%[in_ptr1]]         \n\t"
                  "st1   {v14.4s}, [%[out_ptr1_c7]], #16       \n\t"
                  "subs       %[loop], %[loop], #1    \n\t"
                  "st1   {v15.4s}, [%[out_ptr1_c8]], #16       \n\t"

                  // cycle
                  "bne        0b                              \n\t"
                  "sub       %[f1], %[in_ptr1], #32           \n\t"
                  "sub       %[in_ptr1], %[in_ptr1], #32      \n\t"

                  : [loop] "+r"(loop), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr1_c2] "+r"(out_ptr1_c2),
                    [out_ptr1_c3] "+r"(out_ptr1_c3),
                    [out_ptr1_c4] "+r"(out_ptr1_c4),
                    [out_ptr1_c5] "+r"(out_ptr1_c5),
                    [out_ptr1_c6] "+r"(out_ptr1_c6),
                    [out_ptr1_c7] "+r"(out_ptr1_c7),
                    [out_ptr1_c8] "+r"(out_ptr1_c8), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3)
                  : [f1] "r"(f1)
                  : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                    "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
            }
          }
#else
          if (o_h > valid_h_start && o_h <= valid_h_end) {
            int loop = (valid_w_end - valid_w_start - 1) >> 2;
            o_w += loop * 4;
            int in_stride = (input_w - 8) * 4;

            if (loop > 0) {
              asm volatile(

                  "pld        [%[f1], #256]                   \n\t"
                  "pld        [%[in_ptr1], #288]              \n\t"

                  "vld1.f32   {d0-d3}, [%[f1]]!               \n\t"
                  "vld2.f32   {d8-d11}, [%[in_ptr1]]!         \n\t"
                  "vld2.f32   {d12, d13}, [%[in_ptr1]]        \n\t"
                  "add        %[in_ptr1], %[in_stride]        \n\t"

                  "0:                                         \n\t"
                  // load out_ptr
                  "pld        [%[out_ptr1], #128]             \n\t"
                  "pld        [%[out_ptr1_c2], #128]           \n\t"
                  "pld        [%[out_ptr1_c3], #128]           \n\t"
                  "pld        [%[out_ptr1_c4], #128]           \n\t"
                  "pld        [%[out_ptr1_c5], #128]           \n\t"
                  "pld        [%[out_ptr1_c6], #128]           \n\t"
                  "pld        [%[out_ptr1_c7], #128]           \n\t"
                  "pld        [%[out_ptr1_c8], #128]           \n\t"

                  "vld1.f32   {d16, d17}, [%[out_ptr1]]       \n\t"
                  "vld1.f32   {d18, d19}, [%[out_ptr1_c2]]     \n\t"
                  "vld1.f32   {d20, d21}, [%[out_ptr1_c3]]     \n\t"
                  "vld1.f32   {d22, d23}, [%[out_ptr1_c4]]     \n\t"
                  "vld1.f32   {d24, d25}, [%[out_ptr1_c5]]     \n\t"
                  "vld1.f32   {d26, d27}, [%[out_ptr1_c6]]     \n\t"
                  "vld1.f32   {d28, d29}, [%[out_ptr1_c7]]     \n\t"
                  "vld1.f32   {d30, d31}, [%[out_ptr1_c8]]     \n\t"

                  // in_ptr1 multiply
                  "pld        [%[f1], #256]                   \n\t"
                  "vld1.f32   {d4-d7}, [%[f1]]!               \n\t"
                  "vmla.f32   q8, q4, d0[0]                   \n\t"
                  "vmla.f32   q9, q4, d0[1]                   \n\t"

                  "vmla.f32   q10, q4, d1[0]                  \n\t"
                  "vmla.f32   q11, q4, d1[1]                  \n\t"

                  "vmla.f32   q12, q4, d2[0]                  \n\t"
                  "vmla.f32   q13, q4, d2[1]                  \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vmla.f32   q14, q4, d3[0]                  \n\t"
                  "vmla.f32   q15, q4, d3[1]                  \n\t"

                  "vld1.f32   {d0-d3}, [%[f1]]!               \n\t"
                  "vmla.f32   q8, q5, d4[0]                   \n\t"
                  "vmla.f32   q9, q5, d4[1]                   \n\t"

                  "vext.32    q7, q4, q6, #1                  \n\t"
                  "vmla.f32   q10, q5, d5[0]                  \n\t"
                  "vmla.f32   q11, q5, d5[1]                  \n\t"

                  "vmla.f32   q12, q5, d6[0]                  \n\t"
                  "vmla.f32   q13, q5, d6[1]                  \n\t"

                  "pld        [%[in_ptr1], #288]              \n\t"
                  "vmla.f32   q14, q5, d7[0]                  \n\t"
                  "vmla.f32   q15, q5, d7[1]                  \n\t"

                  "vld2.f32   {d8-d11}, [%[in_ptr1]]!         \n\t"
                  "vmla.f32   q8, q7, d0[0]                   \n\t"
                  "vmla.f32   q9, q7, d0[1]                   \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vld1.f32   {d4-d7}, [%[f1]]!               \n\t"
                  "vmla.f32   q10, q7, d1[0]                  \n\t"
                  "vmla.f32   q11, q7, d1[1]                  \n\t"

                  "vld2.f32   {d12, d13}, [%[in_ptr1]]        \n\t"
                  "add        %[in_ptr1], %[in_stride]        \n\t"
                  "vmla.f32   q12, q7, d2[0]                  \n\t"
                  "vmla.f32   q13, q7, d2[1]                  \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vmla.f32   q14, q7, d3[0]                  \n\t"
                  "vmla.f32   q15, q7, d3[1]                  \n\t"

                  // in_ptr2 multiply
                  "vld1.f32   {d0-d3}, [%[f1]]!               \n\t"
                  "vmla.f32   q8, q4, d4[0]                   \n\t"
                  "vmla.f32   q9, q4, d4[1]                   \n\t"

                  "vmla.f32   q10, q4, d5[0]                  \n\t"
                  "vmla.f32   q11, q4, d5[1]                  \n\t"

                  "vmla.f32   q12, q4, d6[0]                  \n\t"
                  "vmla.f32   q13, q4, d6[1]                  \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vmla.f32   q14, q4, d7[0]                  \n\t"
                  "vmla.f32   q15, q4, d7[1]                  \n\t"

                  "vld1.f32   {d4-d7}, [%[f1]]!               \n\t"
                  "vmla.f32   q8, q5, d0[0]                   \n\t"
                  "vmla.f32   q9, q5, d0[1]                   \n\t"

                  "vext.32    q7, q4, q6, #1                  \n\t"
                  "vmla.f32   q10, q5, d1[0]                  \n\t"
                  "vmla.f32   q11, q5, d1[1]                  \n\t"

                  "vmla.f32   q12, q5, d2[0]                  \n\t"
                  "vmla.f32   q13, q5, d2[1]                  \n\t"

                  "pld        [%[in_ptr1], #288]              \n\t"
                  "vmla.f32   q14, q5, d3[0]                  \n\t"
                  "vmla.f32   q15, q5, d3[1]                  \n\t"

                  "vld2.f32   {d8-d11}, [%[in_ptr1]]!         \n\t"
                  "vmla.f32   q8, q7, d4[0]                   \n\t"
                  "vmla.f32   q9, q7, d4[1]                   \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vld1.f32   {d0-d3}, [%[f1]]!               \n\t"
                  "vmla.f32   q10, q7, d5[0]                  \n\t"
                  "vmla.f32   q11, q7, d5[1]                  \n\t"

                  "vld2.f32   {d12, d13}, [%[in_ptr1]]        \n\t"
                  "sub        %[in_ptr1], %[in_stride]        \n\t"
                  "sub        %[in_ptr1], %[in_stride]        \n\t"
                  "vmla.f32   q12, q7, d6[0]                  \n\t"
                  "vmla.f32   q13, q7, d6[1]                  \n\t"

                  "sub        %[in_ptr1], #64                 \n\t"
                  "pld        [%[f1], #256]                   \n\t"
                  "vmla.f32   q14, q7, d7[0]                  \n\t"
                  "vmla.f32   q15, q7, d7[1]                  \n\t"

                  // in_ptr3 multiply
                  "vld1.f32   {d4-d7}, [%[f1]]!               \n\t"
                  "vmla.f32   q8, q4, d0[0]                   \n\t"
                  "vmla.f32   q9, q4, d0[1]                   \n\t"

                  "vmla.f32   q10, q4, d1[0]                  \n\t"
                  "vmla.f32   q11, q4, d1[1]                  \n\t"

                  "vmla.f32   q12, q4, d2[0]                  \n\t"
                  "vmla.f32   q13, q4, d2[1]                  \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vmla.f32   q14, q4, d3[0]                  \n\t"
                  "vmla.f32   q15, q4, d3[1]                  \n\t"

                  "vld1.f32   {d0-d3}, [%[f1]]!               \n\t"
                  "vmla.f32   q8, q5, d4[0]                   \n\t"
                  "vmla.f32   q9, q5, d4[1]                   \n\t"

                  "vext.32    q7, q4, q6, #1                  \n\t"
                  "vmla.f32   q10, q5, d5[0]                  \n\t"
                  "vmla.f32   q11, q5, d5[1]                  \n\t"

                  "vmla.f32   q12, q5, d6[0]                  \n\t"
                  "vmla.f32   q13, q5, d6[1]                  \n\t"

                  "vmla.f32   q14, q5, d7[0]                  \n\t"
                  "vmla.f32   q15, q5, d7[1]                  \n\t"

                  "sub        %[f1], %[f1], #288              \n\t"
                  "vmla.f32   q8, q7, d0[0]                   \n\t"
                  "vmla.f32   q9, q7, d0[1]                   \n\t"

                  "vmla.f32   q10, q7, d1[0]                  \n\t"
                  "vmla.f32   q11, q7, d1[1]                  \n\t"

                  "vmla.f32   q12, q7, d2[0]                  \n\t"
                  "vmla.f32   q13, q7, d2[1]                  \n\t"

                  "vmla.f32   q14, q7, d3[0]                  \n\t"
                  "vmla.f32   q15, q7, d3[1]                  \n\t"

                  // store out_ptr
                  "pld        [%[f1], #256]                   \n\t"
                  "vld1.f32   {d0-d3}, [%[f1]]!               \n\t"

                  "pld        [%[in_ptr1], #288]              \n\t"
                  "vld2.f32   {d8-d11}, [%[in_ptr1]]!         \n\t"
                  "vst1.f32   {d16, d17}, [%[out_ptr1]]!      \n\t"
                  "vst1.f32   {d18, d19}, [%[out_ptr1_c2]]!    \n\t"

                  "vst1.f32   {d20, d21}, [%[out_ptr1_c3]]!    \n\t"
                  "vst1.f32   {d22, d23}, [%[out_ptr1_c4]]!    \n\t"

                  "vst1.f32   {d24, d25}, [%[out_ptr1_c5]]!    \n\t"
                  "vst1.f32   {d26, d27}, [%[out_ptr1_c6]]!    \n\t"

                  "vld2.f32   {d12, d13}, [%[in_ptr1]]        \n\t"
                  "add        %[in_ptr1], %[in_stride]        \n\t"
                  "vst1.f32   {d28, d29}, [%[out_ptr1_c7]]!    \n\t"

                  "subs       %[loop], #1                 \n\t"
                  "vst1.f32   {d30, d31}, [%[out_ptr1_c8]]!    \n\t"

                  // cycle
                  "bne        0b                              \n\t"
                  "sub        %[f1], %[f1], #32               \n\t"
                  "sub        %[in_ptr1], %[in_ptr1], #32     \n\t"
                  "sub        %[in_ptr1], %[in_stride]        \n\t"

                  : [loop] "+r"(loop), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr1_c2] "+r"(out_ptr1_c2),
                    [out_ptr1_c3] "+r"(out_ptr1_c3),
                    [out_ptr1_c4] "+r"(out_ptr1_c4),
                    [out_ptr1_c5] "+r"(out_ptr1_c5),
                    [out_ptr1_c6] "+r"(out_ptr1_c6),
                    [out_ptr1_c7] "+r"(out_ptr1_c7),
                    [out_ptr1_c8] "+r"(out_ptr1_c8), [in_ptr1] "+r"(in_ptr1)
                  : [f1] "r"(f1), [in_stride] "r"(in_stride)
                  : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                    "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

              in_ptr2 = in_ptr1 + input_w;
              in_ptr3 = in_ptr2 + input_w;
            }
          }
#endif  // __aarch64__
#endif  // __ARM_NEON

          // remain output_width
          for (; o_w < output_w; ++o_w) {
            float sum1 = 0;
            float sum1_c2 = 0;
            float sum1_c3 = 0;
            float sum1_c4 = 0;
            float sum1_c5 = 0;
            float sum1_c6 = 0;
            float sum1_c7 = 0;
            float sum1_c8 = 0;
#if __ARM_NEON
            float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
            float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
            float32x4_t _pad_filter1_c2 = vld1q_f32(pad_filter1_c2);
            float32x4_t _pad_filter1_c3 = vld1q_f32(pad_filter1_c3);
            float32x4_t _pad_filter1_c4 = vld1q_f32(pad_filter1_c4);
            float32x4_t _pad_filter1_c5 = vld1q_f32(pad_filter1_c5);
            float32x4_t _pad_filter1_c6 = vld1q_f32(pad_filter1_c6);
            float32x4_t _pad_filter1_c7 = vld1q_f32(pad_filter1_c7);
            float32x4_t _pad_filter1_c8 = vld1q_f32(pad_filter1_c8);

            float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
            float32x4_t _sum1_c2 = vmulq_f32(_in_ptr1, _pad_filter1_c2);
            float32x4_t _sum1_c3 = vmulq_f32(_in_ptr1, _pad_filter1_c3);
            float32x4_t _sum1_c4 = vmulq_f32(_in_ptr1, _pad_filter1_c4);
            float32x4_t _sum1_c5 = vmulq_f32(_in_ptr1, _pad_filter1_c5);
            float32x4_t _sum1_c6 = vmulq_f32(_in_ptr1, _pad_filter1_c6);
            float32x4_t _sum1_c7 = vmulq_f32(_in_ptr1, _pad_filter1_c7);
            float32x4_t _sum1_c8 = vmulq_f32(_in_ptr1, _pad_filter1_c8);

            float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
            float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
            float32x4_t _pad_filter2_c2 = vld1q_f32(pad_filter2_c2);
            float32x4_t _pad_filter2_c3 = vld1q_f32(pad_filter2_c3);
            float32x4_t _pad_filter2_c4 = vld1q_f32(pad_filter2_c4);
            float32x4_t _pad_filter2_c5 = vld1q_f32(pad_filter2_c5);
            float32x4_t _pad_filter2_c6 = vld1q_f32(pad_filter2_c6);
            float32x4_t _pad_filter2_c7 = vld1q_f32(pad_filter2_c7);
            float32x4_t _pad_filter2_c8 = vld1q_f32(pad_filter2_c8);

            _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
            _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr2, _pad_filter2_c2);
            _sum1_c3 = vmlaq_f32(_sum1_c3, _in_ptr2, _pad_filter2_c3);
            _sum1_c4 = vmlaq_f32(_sum1_c4, _in_ptr2, _pad_filter2_c4);
            _sum1_c5 = vmlaq_f32(_sum1_c5, _in_ptr2, _pad_filter2_c5);
            _sum1_c6 = vmlaq_f32(_sum1_c6, _in_ptr2, _pad_filter2_c6);
            _sum1_c7 = vmlaq_f32(_sum1_c7, _in_ptr2, _pad_filter2_c7);
            _sum1_c8 = vmlaq_f32(_sum1_c8, _in_ptr2, _pad_filter2_c8);

            float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
            float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
            float32x4_t _pad_filter3_c2 = vld1q_f32(pad_filter3_c2);
            float32x4_t _pad_filter3_c3 = vld1q_f32(pad_filter3_c3);
            float32x4_t _pad_filter3_c4 = vld1q_f32(pad_filter3_c4);
            float32x4_t _pad_filter3_c5 = vld1q_f32(pad_filter3_c5);
            float32x4_t _pad_filter3_c6 = vld1q_f32(pad_filter3_c6);
            float32x4_t _pad_filter3_c7 = vld1q_f32(pad_filter3_c7);
            float32x4_t _pad_filter3_c8 = vld1q_f32(pad_filter3_c8);

            _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
            _sum1_c2 = vmlaq_f32(_sum1_c2, _in_ptr3, _pad_filter3_c2);
            _sum1_c3 = vmlaq_f32(_sum1_c3, _in_ptr3, _pad_filter3_c3);
            _sum1_c4 = vmlaq_f32(_sum1_c4, _in_ptr3, _pad_filter3_c4);
            _sum1_c5 = vmlaq_f32(_sum1_c5, _in_ptr3, _pad_filter3_c5);
            _sum1_c6 = vmlaq_f32(_sum1_c6, _in_ptr3, _pad_filter3_c6);
            _sum1_c7 = vmlaq_f32(_sum1_c7, _in_ptr3, _pad_filter3_c7);
            _sum1_c8 = vmlaq_f32(_sum1_c8, _in_ptr3, _pad_filter3_c8);

            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
            _sum1_c2 = vsetq_lane_f32(sum1_c2, _sum1_c2, 3);
            _sum1_c3 = vsetq_lane_f32(sum1_c3, _sum1_c3, 3);
            _sum1_c4 = vsetq_lane_f32(sum1_c4, _sum1_c4, 3);
            _sum1_c5 = vsetq_lane_f32(sum1_c5, _sum1_c5, 3);
            _sum1_c6 = vsetq_lane_f32(sum1_c6, _sum1_c6, 3);
            _sum1_c7 = vsetq_lane_f32(sum1_c7, _sum1_c7, 3);
            _sum1_c8 = vsetq_lane_f32(sum1_c8, _sum1_c8, 3);

            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ss1_2 =
                vadd_f32(vget_low_f32(_sum1_c2), vget_high_f32(_sum1_c2));
            float32x2_t _ss1_3 =
                vadd_f32(vget_low_f32(_sum1_c3), vget_high_f32(_sum1_c3));
            float32x2_t _ss1_4 =
                vadd_f32(vget_low_f32(_sum1_c4), vget_high_f32(_sum1_c4));
            float32x2_t _ss1_5 =
                vadd_f32(vget_low_f32(_sum1_c5), vget_high_f32(_sum1_c5));
            float32x2_t _ss1_6 =
                vadd_f32(vget_low_f32(_sum1_c6), vget_high_f32(_sum1_c6));
            float32x2_t _ss1_7 =
                vadd_f32(vget_low_f32(_sum1_c7), vget_high_f32(_sum1_c7));
            float32x2_t _ss1_8 =
                vadd_f32(vget_low_f32(_sum1_c8), vget_high_f32(_sum1_c8));

            float32x2_t _ssss1_ssss1_2 = vpadd_f32(_ss1, _ss1_2);
            float32x2_t _ssss1_3_ssss1_4 = vpadd_f32(_ss1_3, _ss1_4);
            float32x2_t _ssss1_5_ssss1_6 = vpadd_f32(_ss1_5, _ss1_6);
            float32x2_t _ssss1_7_ssss1_8 = vpadd_f32(_ss1_7, _ss1_8);

            sum1 += vget_lane_f32(_ssss1_ssss1_2, 0);
            sum1_c2 += vget_lane_f32(_ssss1_ssss1_2, 1);
            sum1_c3 += vget_lane_f32(_ssss1_3_ssss1_4, 0);
            sum1_c4 += vget_lane_f32(_ssss1_3_ssss1_4, 1);
            sum1_c5 += vget_lane_f32(_ssss1_5_ssss1_6, 0);
            sum1_c6 += vget_lane_f32(_ssss1_5_ssss1_6, 1);
            sum1_c7 += vget_lane_f32(_ssss1_7_ssss1_8, 0);
            sum1_c8 += vget_lane_f32(_ssss1_7_ssss1_8, 1);
#else
            sum1 += in_ptr1[0] * pad_filter1[0];
            sum1 += in_ptr1[1] * pad_filter1[1];
            sum1 += in_ptr1[2] * pad_filter1[2];
            sum1 += in_ptr2[0] * pad_filter2[0];
            sum1 += in_ptr2[1] * pad_filter2[1];
            sum1 += in_ptr2[2] * pad_filter2[2];
            sum1 += in_ptr3[0] * pad_filter3[0];
            sum1 += in_ptr3[1] * pad_filter3[1];
            sum1 += in_ptr3[2] * pad_filter3[2];

            sum1_c2 += in_ptr1[0] * pad_filter1_c2[0];
            sum1_c2 += in_ptr1[1] * pad_filter1_c2[1];
            sum1_c2 += in_ptr1[2] * pad_filter1_c2[2];
            sum1_c2 += in_ptr2[0] * pad_filter2_c2[0];
            sum1_c2 += in_ptr2[1] * pad_filter2_c2[1];
            sum1_c2 += in_ptr2[2] * pad_filter2_c2[2];
            sum1_c2 += in_ptr3[0] * pad_filter3_c2[0];
            sum1_c2 += in_ptr3[1] * pad_filter3_c2[1];
            sum1_c2 += in_ptr3[2] * pad_filter3_c2[2];

            sum1_c3 += in_ptr1[0] * pad_filter1_c3[0];
            sum1_c3 += in_ptr1[1] * pad_filter1_c3[1];
            sum1_c3 += in_ptr1[2] * pad_filter1_c3[2];
            sum1_c3 += in_ptr2[0] * pad_filter2_c3[0];
            sum1_c3 += in_ptr2[1] * pad_filter2_c3[1];
            sum1_c3 += in_ptr2[2] * pad_filter2_c3[2];
            sum1_c3 += in_ptr3[0] * pad_filter3_c3[0];
            sum1_c3 += in_ptr3[1] * pad_filter3_c3[1];
            sum1_c3 += in_ptr3[2] * pad_filter3_c3[2];

            sum1_c4 += in_ptr1[0] * pad_filter1_c4[0];
            sum1_c4 += in_ptr1[1] * pad_filter1_c4[1];
            sum1_c4 += in_ptr1[2] * pad_filter1_c4[2];
            sum1_c4 += in_ptr2[0] * pad_filter2_c4[0];
            sum1_c4 += in_ptr2[1] * pad_filter2_c4[1];
            sum1_c4 += in_ptr2[2] * pad_filter2_c4[2];
            sum1_c4 += in_ptr3[0] * pad_filter3_c4[0];
            sum1_c4 += in_ptr3[1] * pad_filter3_c4[1];
            sum1_c4 += in_ptr3[2] * pad_filter3_c4[2];

            sum1_c5 += in_ptr1[0] * pad_filter1_c5[0];
            sum1_c5 += in_ptr1[1] * pad_filter1_c5[1];
            sum1_c5 += in_ptr1[2] * pad_filter1_c5[2];
            sum1_c5 += in_ptr2[0] * pad_filter2_c5[0];
            sum1_c5 += in_ptr2[1] * pad_filter2_c5[1];
            sum1_c5 += in_ptr2[2] * pad_filter2_c5[2];
            sum1_c5 += in_ptr3[0] * pad_filter3_c5[0];
            sum1_c5 += in_ptr3[1] * pad_filter3_c5[1];
            sum1_c5 += in_ptr3[2] * pad_filter3_c5[2];

            sum1_c6 += in_ptr1[0] * pad_filter1_c6[0];
            sum1_c6 += in_ptr1[1] * pad_filter1_c6[1];
            sum1_c6 += in_ptr1[2] * pad_filter1_c6[2];
            sum1_c6 += in_ptr2[0] * pad_filter2_c6[0];
            sum1_c6 += in_ptr2[1] * pad_filter2_c6[1];
            sum1_c6 += in_ptr2[2] * pad_filter2_c6[2];
            sum1_c6 += in_ptr3[0] * pad_filter3_c6[0];
            sum1_c6 += in_ptr3[1] * pad_filter3_c6[1];
            sum1_c6 += in_ptr3[2] * pad_filter3_c6[2];

            sum1_c7 += in_ptr1[0] * pad_filter1_c7[0];
            sum1_c7 += in_ptr1[1] * pad_filter1_c7[1];
            sum1_c7 += in_ptr1[2] * pad_filter1_c7[2];
            sum1_c7 += in_ptr2[0] * pad_filter2_c7[0];
            sum1_c7 += in_ptr2[1] * pad_filter2_c7[1];
            sum1_c7 += in_ptr2[2] * pad_filter2_c7[2];
            sum1_c7 += in_ptr3[0] * pad_filter3_c7[0];
            sum1_c7 += in_ptr3[1] * pad_filter3_c7[1];
            sum1_c7 += in_ptr3[2] * pad_filter3_c7[2];

            sum1_c8 += in_ptr1[0] * pad_filter1_c8[0];
            sum1_c8 += in_ptr1[1] * pad_filter1_c8[1];
            sum1_c8 += in_ptr1[2] * pad_filter1_c8[2];
            sum1_c8 += in_ptr2[0] * pad_filter2_c8[0];
            sum1_c8 += in_ptr2[1] * pad_filter2_c8[1];
            sum1_c8 += in_ptr2[2] * pad_filter2_c8[2];
            sum1_c8 += in_ptr3[0] * pad_filter3_c8[0];
            sum1_c8 += in_ptr3[1] * pad_filter3_c8[1];
            sum1_c8 += in_ptr3[2] * pad_filter3_c8[2];
#endif
            if (if_nopadding) {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;
            } else if (input_w > 3 &&
                       (if_odd_pad_w && o_w == valid_w_start ||
                        o_w == valid_w_end && if_odd_pad_w && if_exact_in_w ||
                        o_w == valid_w_end + 1 && !if_odd_pad_w &&
                            !if_exact_in_w)) {
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;
              pad_filter1_c2--;
              pad_filter2_c2--;
              pad_filter3_c2--;

              pad_filter1_c3--;
              pad_filter2_c3--;
              pad_filter3_c3--;
              pad_filter1_c4--;
              pad_filter2_c4--;
              pad_filter3_c4--;

              pad_filter1_c5--;
              pad_filter2_c5--;
              pad_filter3_c5--;
              pad_filter1_c6--;
              pad_filter2_c6--;
              pad_filter3_c6--;

              pad_filter1_c7--;
              pad_filter2_c7--;
              pad_filter3_c7--;
              pad_filter1_c8--;
              pad_filter2_c8--;
              pad_filter3_c8--;

              in_ptr1++;
              in_ptr2++;
              in_ptr3++;
            } else if (input_w <= 3 || o_w < valid_w_start ||
                       o_w > valid_w_end) {
              pad_filter1 -= 2;
              pad_filter2 -= 2;
              pad_filter3 -= 2;
              pad_filter1_c2 -= 2;
              pad_filter2_c2 -= 2;
              pad_filter3_c2 -= 2;

              pad_filter1_c3 -= 2;
              pad_filter2_c3 -= 2;
              pad_filter3_c3 -= 2;
              pad_filter1_c4 -= 2;
              pad_filter2_c4 -= 2;
              pad_filter3_c4 -= 2;

              pad_filter1_c5 -= 2;
              pad_filter2_c5 -= 2;
              pad_filter3_c5 -= 2;
              pad_filter1_c6 -= 2;
              pad_filter2_c6 -= 2;
              pad_filter3_c6 -= 2;

              pad_filter1_c7 -= 2;
              pad_filter2_c7 -= 2;
              pad_filter3_c7 -= 2;
              pad_filter1_c8 -= 2;
              pad_filter2_c8 -= 2;
              pad_filter3_c8 -= 2;
            } else {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;
            }
            *out_ptr1 += sum1;
            *out_ptr1_c2 += sum1_c2;
            *out_ptr1_c3 += sum1_c3;
            *out_ptr1_c4 += sum1_c4;
            *out_ptr1_c5 += sum1_c5;
            *out_ptr1_c6 += sum1_c6;
            *out_ptr1_c7 += sum1_c7;
            *out_ptr1_c8 += sum1_c8;

            out_ptr1++;
            out_ptr1_c2++;
            out_ptr1_c3++;
            out_ptr1_c4++;
            out_ptr1_c5++;
            out_ptr1_c6++;
            out_ptr1_c7++;
            out_ptr1_c8++;
          }
          if (if_nopadding) {
            in_ptr1 += remain_stride_w + input_w;
            in_ptr2 += remain_stride_w + input_w;
            in_ptr3 += remain_stride_w + input_w;

          } else if (input_h > 3 &&
                     (if_odd_pad_h && o_h == valid_h_start ||
                      o_h == valid_h_end && if_odd_pad_h && if_exact_in_h ||
                      o_h == valid_h_end + 1 && !if_odd_pad_h &&
                          !if_exact_in_h)) {
            in_ptr1 += 3;
            in_ptr2 += 3;
            in_ptr3 += 3;

            pad_filter1 -= remain_stride_w;
            pad_filter2 -= remain_stride_w;
            pad_filter3 -= remain_stride_w;
            pad_filter1_c2 -= remain_stride_w;
            pad_filter2_c2 -= remain_stride_w;
            pad_filter3_c2 -= remain_stride_w;

            pad_filter1_c3 -= remain_stride_w;
            pad_filter2_c3 -= remain_stride_w;
            pad_filter3_c3 -= remain_stride_w;
            pad_filter1_c4 -= remain_stride_w;
            pad_filter2_c4 -= remain_stride_w;
            pad_filter3_c4 -= remain_stride_w;

            pad_filter1_c5 -= remain_stride_w;
            pad_filter2_c5 -= remain_stride_w;
            pad_filter3_c5 -= remain_stride_w;
            pad_filter1_c6 -= remain_stride_w;
            pad_filter2_c6 -= remain_stride_w;
            pad_filter3_c6 -= remain_stride_w;

            pad_filter1_c7 -= remain_stride_w;
            pad_filter2_c7 -= remain_stride_w;
            pad_filter3_c7 -= remain_stride_w;
            pad_filter1_c8 -= remain_stride_w;
            pad_filter2_c8 -= remain_stride_w;
            pad_filter3_c8 -= remain_stride_w;
          } else if (input_h <= 3 || o_h < valid_h_start || o_h > valid_h_end) {
            in_ptr1 -= input_w - 3;
            in_ptr2 -= input_w - 3;
            in_ptr3 -= input_w - 3;

            pad_filter1 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter1_c2 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2_c2 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3_c2 -= 3 + 2 * padding_w + remain_stride_w;

            pad_filter1_c3 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2_c3 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3_c3 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter1_c4 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2_c4 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3_c4 -= 3 + 2 * padding_w + remain_stride_w;

            pad_filter1_c5 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2_c5 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3_c5 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter1_c6 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2_c6 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3_c6 -= 3 + 2 * padding_w + remain_stride_w;

            pad_filter1_c7 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2_c7 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3_c7 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter1_c8 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2_c8 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3_c8 -= 3 + 2 * padding_w + remain_stride_w;
          } else {
            pad_filter1 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter1_c2 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2_c2 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3_c2 += 3 + 2 * padding_w - remain_stride_w;

            pad_filter1_c3 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2_c3 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3_c3 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter1_c4 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2_c4 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3_c4 += 3 + 2 * padding_w - remain_stride_w;

            pad_filter1_c5 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2_c5 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3_c5 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter1_c6 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2_c6 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3_c6 += 3 + 2 * padding_w - remain_stride_w;

            pad_filter1_c7 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2_c7 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3_c7 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter1_c8 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2_c8 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3_c8 += 3 + 2 * padding_w - remain_stride_w;

            in_ptr1 += input_w + 3;
            in_ptr2 += input_w + 3;
            in_ptr3 += input_w + 3;
          }
        }

        filter_data_ch += filter_ch_size;
        filter_data_ch_c2 += filter_ch_size;
        filter_data_ch_c3 += filter_ch_size;
        filter_data_ch_c4 += filter_ch_size;
        filter_data_ch_c5 += filter_ch_size;
        filter_data_ch_c6 += filter_ch_size;
        filter_data_ch_c7 += filter_ch_size;
        filter_data_ch_c8 += filter_ch_size;
        input_data_ch += in_ch_size;
      }
    }

    int out_ch_remain_start = output_ch - output_ch % 8;

    // remain output_channel
#pragma omp parallel for
    for (int o_c = out_ch_remain_start; o_c < output_ch; ++o_c) {
      const float *f1, *f9;
      const float *in_ptr1, *in_ptr2, *in_ptr3;
      const float *pad_filter1, *pad_filter2, *pad_filter3;
      float pad_filter_arr[pad_filter_ch_size];
      float *output_data_ch;
      const float *input_data_ch;
      const float *filter_data_ch;

      filter_data_ch = filter_data + o_c * filter_ch_size * input_ch;
      input_data_ch = input_data;
      output_data_ch = output_data + o_c * out_ch_size;

      for (int i_c = 0; i_c < input_ch; ++i_c) {
        f1 = filter_data_ch;
        f9 = f1 + 8;

        if (!if_nopadding) {
          memset(pad_filter_arr, 0.f, sizeof(pad_filter_arr));
          for (int i = 0; i < 9; ++i) {
            int j = i / 3 * (2 * padding_w + 3) + i % 3 + padding_h * 3 +
                    padding_w * (2 * padding_h + 1);
            pad_filter_arr[j] = filter_data_ch[i];
          }
          pad_filter1 = pad_filter_arr;
          pad_filter1 += pad_filter_start;
          pad_filter2 = pad_filter1 + pad_filter_w;
          pad_filter3 = pad_filter2 + pad_filter_w;
        } else {
          pad_filter1 = filter_data_ch;
          pad_filter2 = pad_filter1 + 3;
          pad_filter3 = pad_filter2 + 3;
        }

        float *out_ptr1;
        out_ptr1 = output_data_ch;
        in_ptr1 = input_data_ch;
        in_ptr2 = in_ptr1 + input_w;
        in_ptr3 = in_ptr2 + input_w;

        int o_h = 0;
        for (; o_h < output_h; ++o_h) {
          int o_w = 0;

          // pad left
          for (; o_w <= valid_w_start; ++o_w) {
            float sum1 = 0;
#if __ARM_NEON
            float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
            float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
            float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);

            float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
            float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
            _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);

            float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
            float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
            _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);

            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ssss1_ssss1 = vpadd_f32(_ss1, _ss1);
            sum1 += vget_lane_f32(_ssss1_ssss1, 0);
#else
            sum1 += in_ptr1[0] * pad_filter1[0];
            sum1 += in_ptr1[1] * pad_filter1[1];
            sum1 += in_ptr1[2] * pad_filter1[2];
            sum1 += in_ptr2[0] * pad_filter2[0];
            sum1 += in_ptr2[1] * pad_filter2[1];
            sum1 += in_ptr2[2] * pad_filter2[2];
            sum1 += in_ptr3[0] * pad_filter3[0];
            sum1 += in_ptr3[1] * pad_filter3[1];
            sum1 += in_ptr3[2] * pad_filter3[2];
#endif
            if (if_nopadding) {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;
            } else if (input_w > 3 &&
                       (if_odd_pad_w && o_w == valid_w_start ||
                        o_w == valid_w_end && if_odd_pad_w && if_exact_in_w ||
                        o_w == valid_w_end + 1 && !if_odd_pad_w &&
                            !if_exact_in_w)) {
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;
              in_ptr1++;
              in_ptr2++;
              in_ptr3++;

            } else if (input_w <= 3 || o_w < valid_w_start ||
                       o_w > valid_w_end) {
              pad_filter1 -= 2;
              pad_filter2 -= 2;
              pad_filter3 -= 2;
            } else {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;
            }
            *out_ptr1 += sum1;
            out_ptr1++;
          }
            // valid
#if __ARM_NEON
#if __aarch64__
          if (o_h > valid_h_start && o_h < valid_h_end) {
            int loop = (valid_w_end - valid_w_start - 1) >> 2;
            o_w += loop * 4;

            if (loop > 0) {
              asm volatile(
                  "prfm   pldl1keep, [%[f1], #256]            \n\t"
                  "prfm   pldl1keep, [%[f9], #256]            \n\t"

                  "ld1   {v0.4s, v1.4s}, [%[f1]]              \n\t"
                  "ld1   {v4.s}[0], [%[f9]]                   \n\t"

                  "0:                                         \n\t"
                  // load out_ptr
                  "prfm   pldl1keep, [%[out_ptr1], #128]      \n\t"
                  "ld1   {v12.4s}, [%[out_ptr1]]              \n\t"

                  // in_ptr1 multiply
                  "prfm   pldl1keep, [%[in_ptr1], #256]       \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[in_ptr1]], #32    \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[in_ptr1]]         \n\t"

                  "fmla   v12.4s, v5.4s, v0.s[0]              \n\t"
                  "fmla   v14.4s, v5.4s, v2.s[0]              \n\t"

                  "ext    v8.16b, v5.16b, v7.16b, #4          \n\t"
                  "fmul   v13.4s, v6.4s, v0.s[1]              \n\t"
                  "fmla   v12.4s, v8.4s, v0.s[2]              \n\t"

                  "ld2   {v5.4s, v6.4s}, [%[in_ptr2]], #32    \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[in_ptr2]]         \n\t"

                  // in_ptr2 multiply
                  "fmla   v13.4s, v5.4s, v0.s[3]              \n\t"
                  "ext    v8.16b, v5.16b, v7.16b, #4          \n\t"
                  "fmla   v12.4s, v6.4s, v1.s[0]              \n\t"

                  "fmla   v13.4s, v8.4s, v1.s[1]              \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[in_ptr3]], #32    \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[in_ptr3]]         \n\t"

                  // in_ptr3 multiply
                  "fmla   v12.4s, v5.4s, v1.s[2]              \n\t"
                  "ext    v8.16b, v5.16b, v7.16b, #4          \n\t"

                  "fmla   v13.4s, v6.4s, v1.s[3]              \n\t"
                  "fmla   v12.4s, v8.4s, v4.s[0]              \n\t"

                  // store out_ptr
                  "fadd   v12.4s, v12.4s, v13.4s              \n\t"
                  "st1   {v12.4s}, [%[out_ptr1]], #16         \n\t"

                  // cycle
                  "subs       %[loop], %[loop], #1      \n\t"
                  "bne        0b                              \n\t"

                  : [loop] "+r"(loop), [out_ptr1] "+r"(out_ptr1),
                    [in_ptr1] "+r"(in_ptr1), [in_ptr2] "+r"(in_ptr2),
                    [in_ptr3] "+r"(in_ptr3)
                  : [f1] "r"(f1), [f9] "r"(f9)
                  : "cc", "memory", "v0", "v1", "v4", "v5", "v6", "v7", "v8",
                    "v12", "v13");
            }
          }
#else
          if (o_h > valid_h_start && o_h < valid_h_end) {
            int loop = (valid_w_end - valid_w_start - 1) >> 2;
            o_w += loop * 4;

            if (loop > 0) {
              asm volatile(
                  "pld        [%[f1], #256]                   \n\t"
                  "pld        [%[f9], #256]                   \n\t"

                  "vld1.f32   {d0-d3}, [%[f1]]                \n\t"
                  "vld1.f32   {d8[0]}, [%[f9]]                \n\t"

                  "pld        [%[in_ptr1], #256]              \n\t"
                  "vld2.f32   {d10-d13}, [%[in_ptr1]]!        \n\t"
                  "vld2.f32   {d14, d15}, [%[in_ptr1]]        \n\t"

                  "0:                                         \n\t"
                  // load out_ptr
                  "pld        [%[out_ptr1], #128]             \n\t"
                  "vld1.f32   {d24, d25}, [%[out_ptr1]]       \n\t"

                  // in_ptr1 multiply
                  "pld        [%[in_ptr2], #256]              \n\t"
                  "vld2.f32   {d4-d7}, [%[in_ptr2]]!          \n\t"

                  "vmla.f32   q12, q5, d0[0]                  \n\t"
                  "vld2.f32   {d20, d21}, [%[in_ptr2]]        \n\t"
                  "vext.32    q8, q5, q7, #1                  \n\t"

                  "pld        [%[in_ptr3], #256]              \n\t"
                  "vmul.f32   q13, q6, d0[1]                  \n\t"

                  "vld2.f32   {d10-d13}, [%[in_ptr3]]!        \n\t"
                  "vmul.f32   q14, q8, d1[0]                  \n\t"
                  "vld2.f32   {d14, d15}, [%[in_ptr3]]        \n\t"

                  // in_ptr2 multiply
                  "vmul.f32   q15, q2, d1[1]                  \n\t"
                  "vext.32    q8, q2, q10, #1                 \n\t"

                  "vmla.f32   q12, q3, d2[0]                  \n\t"
                  "vmla.f32   q13, q8, d2[1]                  \n\t"

                  // in_ptr3 multiply
                  "vmla.f32   q14, q5, d3[0]                  \n\t"
                  "vext.32    q8, q5, q7, #1                  \n\t"

                  "pld        [%[in_ptr1], #256]              \n\t"
                  "vmla.f32   q15, q6, d3[1]                  \n\t"

                  "vld2.f32   {d10-d13}, [%[in_ptr1]]!        \n\t"
                  "vmla.f32   q13, q8, d8[0]                  \n\t"

                  // store out_ptr
                  "vld2.f32   {d14, d15}, [%[in_ptr1]]        \n\t"
                  "vadd.f32   q12, q12, q13                   \n\t"
                  "subs       %[loop], #1                 \n\t"

                  "vadd.f32   q14, q14, q15                   \n\t"
                  "vadd.f32   q12, q12, q14                   \n\t"
                  "vst1.f32   {d24, d25}, [%[out_ptr1]]!      \n\t"

                  // cycle
                  "bne        0b                              \n\t"
                  "subs       %[in_ptr1], %[in_ptr1], #32     \n\t"

                  : [loop] "+r"(loop), [out_ptr1] "+r"(out_ptr1),
                    [in_ptr1] "+r"(in_ptr1), [in_ptr2] "+r"(in_ptr2),
                    [in_ptr3] "+r"(in_ptr3)
                  : [f1] "r"(f1), [f9] "r"(f9)
                  : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                    "q7", "q8", "q10", "q12", "q13", "q14", "q15");
            }
          }
#endif  // __aarch64__
#endif  // __ARM_NEON
          out_ptr1 -= 4;
          out_ptr1 += 4;

          // remain output_width
          for (; o_w < output_w; ++o_w) {
            float sum1 = 0;
#if __ARM_NEON
            float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
            float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
            float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);

            float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
            float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
            _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);

            float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
            float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
            _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);

            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ssss1_ssss1 = vpadd_f32(_ss1, _ss1);
            sum1 += vget_lane_f32(_ssss1_ssss1, 0);
#else
            sum1 += in_ptr1[0] * pad_filter1[0];
            sum1 += in_ptr1[1] * pad_filter1[1];
            sum1 += in_ptr1[2] * pad_filter1[2];
            sum1 += in_ptr2[0] * pad_filter2[0];
            sum1 += in_ptr2[1] * pad_filter2[1];
            sum1 += in_ptr2[2] * pad_filter2[2];
            sum1 += in_ptr3[0] * pad_filter3[0];
            sum1 += in_ptr3[1] * pad_filter3[1];
            sum1 += in_ptr3[2] * pad_filter3[2];
#endif
            if (if_nopadding) {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;
            } else if (input_w > 3 &&
                       (if_odd_pad_w && o_w == valid_w_start ||
                        o_w == valid_w_end && if_odd_pad_w && if_exact_in_w ||
                        o_w == valid_w_end + 1 && !if_odd_pad_w &&
                            !if_exact_in_w)) {
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;

              in_ptr1++;
              in_ptr2++;
              in_ptr3++;

            } else if (input_w <= 3 || o_w < valid_w_start ||
                       o_w > valid_w_end) {
              pad_filter1 -= 2;
              pad_filter2 -= 2;
              pad_filter3 -= 2;
            } else {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;
            }
            *out_ptr1 += sum1;
            out_ptr1++;
          }
          if (if_nopadding) {
            in_ptr1 += remain_stride_w + input_w;
            in_ptr2 += remain_stride_w + input_w;
            in_ptr3 += remain_stride_w + input_w;
          } else if (input_h > 3 &&
                     (if_odd_pad_h && o_h == valid_h_start ||
                      o_h == valid_h_end && if_odd_pad_h && if_exact_in_h ||
                      o_h == valid_h_end + 1 && !if_odd_pad_h &&
                          !if_exact_in_h)) {
            in_ptr1 += 3;
            in_ptr2 += 3;
            in_ptr3 += 3;

            pad_filter1 -= remain_stride_w;
            pad_filter2 -= remain_stride_w;
            pad_filter3 -= remain_stride_w;

          } else if (input_h <= 3 || o_h < valid_h_start || o_h > valid_h_end) {
            in_ptr1 -= input_w - 3;
            in_ptr2 -= input_w - 3;
            in_ptr3 -= input_w - 3;

            pad_filter1 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3 -= 3 + 2 * padding_w + remain_stride_w;
          } else {
            pad_filter1 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3 += 3 + 2 * padding_w - remain_stride_w;

            in_ptr1 += input_w + 3;
            in_ptr2 += input_w + 3;
            in_ptr3 += input_w + 3;
          }
        }
        filter_data_ch += filter_ch_size;
        input_data_ch += in_ch_size;
      }
    }
    input_data += in_batch_size;
    output_data += out_batch_size;
  }
}

template <>
void SlidingwindowConv3x3s1Faster<float, float>(
    const framework::Tensor *input, framework::Tensor *filter,
    const std::vector<int> &paddings, framework::Tensor *output) {
  const float *din = input->data<float>();
  float *dout = output->mutable_data<float>();
  const float *weights = filter->mutable_data<float>();
  const float *bias = nullptr;
  bool relu = false;
  const int num = input->dims()[0];
  const int chin = input->dims()[1];
  const int hin = input->dims()[2];
  const int win = input->dims()[3];
  const int chout = output->dims()[1];
  const int hout = output->dims()[2];
  const int wout = output->dims()[3];
  const int pad_h = paddings[0];
  const int pad_w = paddings[1];
  const int threads = framework::CPUContext::Context()->get_thread_num();
  int l2_size =
      framework::CPUContext::Context()->get_l2_cache_size() / sizeof(float);

  const int hout_c_block = 4;
  const int hout_r_kernel = 2;
  const int wout_block = 4;
  const int wout_round = ((wout + wout_block - 1) / wout_block) * wout_block;
  const int win_round = wout_round + 2;

  int hout_r_block = (l2_size - 2 * win_round * chin) /
                     (win_round * chin + hout_c_block * wout_round * threads);
  hout_r_block = hout_r_block > hout ? hout : hout_r_block;
  hout_r_block = (hout_r_block / hout_r_kernel) * hout_r_kernel;
  hout_r_block = hout_r_block < hout_r_kernel ? hout_r_kernel : hout_r_block;

  const int hin_r_block = hout_r_block + 2;

  float ptr_zero[win_round];
  memset(ptr_zero, 0, sizeof(float) * win_round);
  float ptr_write[wout_round];

  int in_len = win_round * chin;
  int pre_in_size = hin_r_block * in_len;
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  float *pre_din =
      static_cast<float *>(framework::CPUContext::Context()->get_work_space(
          (pre_in_size + threads * pre_out_size) * sizeof(float)));

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = chin * 9;               // kernel_w * kernel_h;
  int w_stride_chin = hout_c_block * 9;  // kernel_w * kernel_h *

  int ws = -pad_w;
  int we = ws + win_round;
  int w_loop = wout_round / 4;

  int c_remain = chout - (chout / hout_c_block) * hout_c_block;
  int c_round_down = (chout / hout_c_block) * hout_c_block;

  int out_row_stride = hout_c_block * wout_round;
  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * chin * size_in_channel;
    float *dout_batch = dout + n * chout * size_out_channel;
    for (int h = 0; h < hout; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > hout) {
        h_kernel = hout - h;
      }
      int hs = h - pad_h;
      int he = hs + h_kernel + 2;
      slidingwindow_prepack_input(din_batch, pre_din, 0, chin, hs, he, ws, we,
                                  chin, win, hin, ptr_zero);
#pragma omp parallel for
      for (int c = 0; c < chout - (hout_c_block - 1); c += hout_c_block) {
#ifdef _OPENMP
        float *pre_out =
            pre_din + pre_in_size + omp_get_thread_num() * pre_out_size;
#else
        float *pre_out = pre_din + pre_in_size;
#endif
        const float *block_inr0 = pre_din;
        const float *block_inr1 = block_inr0 + in_len;
        const float *block_inr2 = block_inr1 + in_len;
        const float *block_inr3 = block_inr2 + in_len;

        const float *weight_c = weights + c * w_stride;
        const float *bias_ptr = ptr_zero;
        if (bias != nullptr) {
          bias_ptr = bias + c;
        }
        slidingwindow_fill_bias(pre_out, bias_ptr,
                                wout_round * hout_c_block * h_kernel);

        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          const float *wc0 = weight_c;

          const float *inr0 = block_inr0;
          const float *inr1 = block_inr1;
          const float *inr2 = block_inr2;
          const float *inr3 = block_inr3;

          float *pre_out0 = pre_out + hk * out_row_stride;
          float *pre_out1 = pre_out0 + out_row_stride;
#ifdef __aarch64__
          for (int i = 0; i < chin; ++i) {
            float *ptr_out0 = pre_out0;
            float *ptr_out1 = pre_out1;

            float32x4_t w0 = vld1q_f32(wc0);       // w0, v23
            float32x4_t w1 = vld1q_f32(wc0 + 4);   // w1, v24
            float32x4_t w2 = vld1q_f32(wc0 + 8);   // w2, v25
            float32x4_t w3 = vld1q_f32(wc0 + 12);  // w3, v26
            float32x4_t w4 = vld1q_f32(wc0 + 16);  // w4, v27
            float32x4_t w5 = vld1q_f32(wc0 + 20);  // w5, v28
            float32x4_t w6 = vld1q_f32(wc0 + 24);  // w6, v29
            float32x4_t w7 = vld1q_f32(wc0 + 28);  // w7, v30
            float32x4_t w8 = vld1q_f32(wc0 + 32);  // w8, v31

            const float *r0 = inr0;
            const float *r1 = inr1;
            const float *r2 = inr2;
            const float *r3 = inr3;

            int cnt = w_loop;
            asm volatile(
                "ldp    q15, q16, [%[ptr_out0]]     \n" /* load outr00, outr01*/
                "ldp    q17, q18, [%[ptr_out0], #32]\n" /* load outr02, outr03*/
                "ldp    q19, q20, [%[ptr_out1]]     \n" /* load outr10, outr11*/
                "ldp    q21, q22, [%[ptr_out1], #32]\n" /* load outr10, outr11*/
                "ldp    q0, q1,   [%[r0]], #16      \n" /* load input r0*/
                "ldp    q2, q3,   [%[r1]], #16      \n" /* load input r1*/
                "2:                                 \n" /* main loop*/
                /*  r0, r1, mul w0, get out r0, r1 */
                "fmla   v15.4s ,  %[w0].4s,  v0.s[0]\n" /* outr00 = w0 * r0[0]*/
                "fmla   v16.4s ,  %[w0].4s,  v0.s[1]\n" /* outr01 = w0 * r0[1]*/
                "fmla   v17.4s ,  %[w0].4s,  v0.s[2]\n" /* outr02 = w0 * r0[2]*/
                "fmla   v18.4s ,  %[w0].4s,  v0.s[3]\n" /* outr03 = w0 * r0[3]*/
                "fmla   v19.4s ,  %[w0].4s,  v2.s[0]\n" /* outr10 = w0 * r1[0]*/
                "fmla   v20.4s ,  %[w0].4s,  v2.s[1]\n" /* outr11 = w0 * r1[1]*/
                "fmla   v21.4s ,  %[w0].4s,  v2.s[2]\n" /* outr12 = w0 * r1[2]*/
                "fmla   v22.4s ,  %[w0].4s,  v2.s[3]\n" /* outr13 = w0 * r1[3]*/

                /*  r0, r1, mul w1, get out r0, r1 */
                "fmla   v15.4s ,  %[w1].4s,  v0.s[1]\n" /* outr00 = w1 * r0[1]*/
                "fmla   v16.4s ,  %[w1].4s,  v0.s[2]\n" /* outr01 = w1 * r0[2]*/
                "fmla   v17.4s ,  %[w1].4s,  v0.s[3]\n" /* outr02 = w1 * r0[3]*/
                "fmla   v18.4s ,  %[w1].4s,  v1.s[0]\n" /* outr03 = w1 * r0[4]*/
                "fmla   v19.4s ,  %[w1].4s,  v2.s[1]\n" /* outr10 = w1 * r1[1]*/
                "fmla   v20.4s ,  %[w1].4s,  v2.s[2]\n" /* outr11 = w1 * r1[2]*/
                "fmla   v21.4s ,  %[w1].4s,  v2.s[3]\n" /* outr12 = w1 * r1[3]*/
                "fmla   v22.4s ,  %[w1].4s,  v3.s[0]\n" /* outr13 = w1 * r1[4]*/

                "ldp    q4, q5,   [%[r2]], #16      \n" /* load input r2*/

                /*  r0, r1, mul w2, get out r0, r1 */
                "fmla   v15.4s ,  %[w2].4s,  v0.s[2]\n" /* outr00 = w2 * r0[2]*/
                "fmla   v16.4s ,  %[w2].4s,  v0.s[3]\n" /* outr01 = w2 * r0[3]*/
                "fmla   v17.4s ,  %[w2].4s,  v1.s[0]\n" /* outr02 = w2 * r0[0]*/
                "fmla   v18.4s ,  %[w2].4s,  v1.s[1]\n" /* outr03 = w2 * r0[1]*/
                "fmla   v19.4s ,  %[w2].4s,  v2.s[2]\n" /* outr10 = w2 * r1[2]*/
                "fmla   v20.4s ,  %[w2].4s,  v2.s[3]\n" /* outr11 = w2 * r1[3]*/
                "fmla   v21.4s ,  %[w2].4s,  v3.s[0]\n" /* outr12 = w2 * r1[0]*/
                "fmla   v22.4s ,  %[w2].4s,  v3.s[1]\n" /* outr13 = w2 * r1[1]*/

                /*  r1, r2, mul w3, get out r0, r1 */
                "fmla   v15.4s ,  %[w3].4s,  v2.s[0]\n" /* outr00 = w3 * r1[0]*/
                "fmla   v16.4s ,  %[w3].4s,  v2.s[1]\n" /* outr01 = w3 * r1[1]*/
                "fmla   v17.4s ,  %[w3].4s,  v2.s[2]\n" /* outr02 = w3 * r1[2]*/
                "fmla   v18.4s ,  %[w3].4s,  v2.s[3]\n" /* outr03 = w3 * r1[3]*/
                "fmla   v19.4s ,  %[w3].4s,  v4.s[0]\n" /* outr10 = w3 * r2[0]*/
                "fmla   v20.4s ,  %[w3].4s,  v4.s[1]\n" /* outr11 = w3 * r2[1]*/
                "fmla   v21.4s ,  %[w3].4s,  v4.s[2]\n" /* outr12 = w3 * r2[2]*/
                "fmla   v22.4s ,  %[w3].4s,  v4.s[3]\n" /* outr13 = w3 * r2[3]*/

                "ldp    q0, q1,   [%[r0]], #16      \n" /* load next input r0*/

                /*  r1, r2, mul w4, get out r0, r1 */
                "fmla   v15.4s ,  %[w4].4s,  v2.s[1]\n" /* outr00 = w4 * r1[1]*/
                "fmla   v16.4s ,  %[w4].4s,  v2.s[2]\n" /* outr01 = w4 * r1[2]*/
                "fmla   v17.4s ,  %[w4].4s,  v2.s[3]\n" /* outr02 = w4 * r1[3]*/
                "fmla   v18.4s ,  %[w4].4s,  v3.s[0]\n" /* outr03 = w4 * r1[4]*/
                "fmla   v19.4s ,  %[w4].4s,  v4.s[1]\n" /* outr10 = w4 * r2[1]*/
                "fmla   v20.4s ,  %[w4].4s,  v4.s[2]\n" /* outr11 = w4 * r2[2]*/
                "fmla   v21.4s ,  %[w4].4s,  v4.s[3]\n" /* outr12 = w4 * r2[3]*/
                "fmla   v22.4s ,  %[w4].4s,  v5.s[0]\n" /* outr13 = w4 * r2[4]*/

                "ldp    q6, q7,   [%[r3]], #16      \n" /* load input r3*/

                /*  r1, r2, mul w5, get out r0, r1 */
                "fmla   v15.4s ,  %[w5].4s,  v2.s[2]\n" /* outr00 = w5 * r1[2]*/
                "fmla   v16.4s ,  %[w5].4s,  v2.s[3]\n" /* outr01 = w5 * r1[3]*/
                "fmla   v17.4s ,  %[w5].4s,  v3.s[0]\n" /* outr02 = w5 * r1[0]*/
                "fmla   v18.4s ,  %[w5].4s,  v3.s[1]\n" /* outr03 = w5 * r1[1]*/
                "fmla   v19.4s ,  %[w5].4s,  v4.s[2]\n" /* outr10 = w5 * r2[2]*/
                "fmla   v20.4s ,  %[w5].4s,  v4.s[3]\n" /* outr11 = w5 * r2[3]*/
                "fmla   v21.4s ,  %[w5].4s,  v5.s[0]\n" /* outr12 = w5 * r2[0]*/
                "fmla   v22.4s ,  %[w5].4s,  v5.s[1]\n" /* outr13 = w5 * r2[1]*/

                /*  r2, r3, mul w6, get out r0, r1 */
                "fmla   v15.4s ,  %[w6].4s,  v4.s[0]\n" /* outr00 = w6 * r2[0]*/
                "fmla   v16.4s ,  %[w6].4s,  v4.s[1]\n" /* outr01 = w6 * r2[1]*/
                "fmla   v17.4s ,  %[w6].4s,  v4.s[2]\n" /* outr02 = w6 * r2[2]*/
                "fmla   v18.4s ,  %[w6].4s,  v4.s[3]\n" /* outr03 = w6 * r2[3]*/
                "fmla   v19.4s ,  %[w6].4s,  v6.s[0]\n" /* outr10 = w6 * r3[0]*/
                "fmla   v20.4s ,  %[w6].4s,  v6.s[1]\n" /* outr11 = w6 * r3[1]*/
                "fmla   v21.4s ,  %[w6].4s,  v6.s[2]\n" /* outr12 = w6 * r3[2]*/
                "fmla   v22.4s ,  %[w6].4s,  v6.s[3]\n" /* outr13 = w6 * r3[3]*/

                "ldp    q2, q3,   [%[r1]], #16      \n" /* load next input r1*/

                /*  r2, r3, mul w7, get out r0, r1 */
                "fmla   v15.4s ,  %[w7].4s,  v4.s[1]\n" /* outr00 = w7 * r2[1]*/
                "fmla   v16.4s ,  %[w7].4s,  v4.s[2]\n" /* outr01 = w7 * r2[2]*/
                "fmla   v17.4s ,  %[w7].4s,  v4.s[3]\n" /* outr02 = w7 * r2[3]*/
                "fmla   v18.4s ,  %[w7].4s,  v5.s[0]\n" /* outr03 = w7 * r2[4]*/
                "fmla   v19.4s ,  %[w7].4s,  v6.s[1]\n" /* outr10 = w7 * r3[1]*/
                "fmla   v20.4s ,  %[w7].4s,  v6.s[2]\n" /* outr11 = w7 * r3[2]*/
                "fmla   v21.4s ,  %[w7].4s,  v6.s[3]\n" /* outr12 = w7 * r3[3]*/
                "fmla   v22.4s ,  %[w7].4s,  v7.s[0]\n" /* outr13 = w7 * r3[4]*/

                "subs   %w[cnt], %w[cnt], #1        \n" /*loop count -1*/

                /*  r2, r3, mul w8, get out r0, r1 */
                "fmla   v15.4s ,  %[w8].4s,  v4.s[2]\n" /* outr00 = w8 * r2[2]*/
                "fmla   v16.4s ,  %[w8].4s,  v4.s[3]\n" /* outr01 = w8 * r2[3]*/
                "fmla   v17.4s ,  %[w8].4s,  v5.s[0]\n" /* outr02 = w8 * r2[0]*/
                "fmla   v18.4s ,  %[w8].4s,  v5.s[1]\n" /* outr03 = w8 * r2[1]*/

                "stp    q15, q16, [%[ptr_out0]], #32\n" /* save outr00, outr01*/
                "fmla   v19.4s ,  %[w8].4s,  v6.s[2]\n" /* outr10 = w8 * r3[2]*/
                "stp    q17, q18, [%[ptr_out0]], #32\n" /* save outr02, outr03*/
                "fmla   v20.4s ,  %[w8].4s,  v6.s[3]\n" /* outr11 = w8 * r3[3]*/
                "ldp    q15, q16, [%[ptr_out0]]     \n" /* load outr00, outr01*/
                "fmla   v21.4s ,  %[w8].4s,  v7.s[0]\n" /* outr12 = w8 * r3[0]*/
                "ldp    q17, q18, [%[ptr_out0], #32]\n" /* load outr02, outr03*/
                "fmla   v22.4s ,  %[w8].4s,  v7.s[1]\n" /* outr13 = w8 * r3[1]*/
                "stp    q19, q20, [%[ptr_out1]], #32\n" /* save outr10, outr11*/
                "stp    q21, q22, [%[ptr_out1]], #32\n" /* save outr12, outr13*/
                "ldp    q19, q20, [%[ptr_out1]]     \n" /* load outr10, outr11*/
                "ldp    q21, q22, [%[ptr_out1], #32]\n" /* load outr12, outr13*/
                "bne    2b                          \n" /* jump to main loop*/

                : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2),
                  [r3] "+r"(r3), [ptr_out0] "+r"(ptr_out0),
                  [ptr_out1] "+r"(ptr_out1)
                : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3),
                  [w4] "w"(w4), [w5] "w"(w5), [w6] "w"(w6), [w7] "w"(w7),
                  [w8] "w"(w8)
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22");

            wc0 += 9 * hout_c_block;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
          }
#else   // not __aarch64__
          for (int i = 0; i < chin; ++i) {
            const float *wc0 = weight_c + i * w_stride_chin;

            float *ptr_out0 = pre_out0;
            float *ptr_out1 = pre_out1;

            const float *r0 = inr0;
            const float *r1 = inr1;
            const float *r2 = inr2;
            const float *r3 = inr3;

            int cnt = w_loop;
            asm volatile(
                "vld1.32    {d16-d19}, [%[ptr_out0]]!               @ load "
                "outr0, w0, w1, c0~c3\n"
                "vld1.32    {d20-d23}, [%[ptr_out0]]                @ load "
                "outr0, w2, w3, c0~c3\n"

                /* load weights */
                "vld1.32    {d10-d13}, [%[wc0]]!                    @ load w0, "
                "w1, to q5, q6\n"
                "vld1.32    {d14-d15}, [%[wc0]]!                    @ load w2, "
                "to q7\n"

                /* load r0, r1 */
                "vld1.32    {d0-d1}, [%[r0]]!                       @ load r0, "
                "4 float\n"
                "vld1.32    {d2}, [%[r0]]                           @ load r0, "
                "2 float\n"

                "sub    %[ptr_out0], %[ptr_out0], #32               @ ptr_out0 "
                "- 32, to start address\n"

                /* main loop */
                "0:                                                 @ main "
                "loop\n"
                /* mul r0 with w0, w1, w2, get out r0 */
                "vld1.32    {d24-d27}, [%[ptr_out1]]!               @ load "
                "outr1, w0, w1, c0~c3\n"
                "vmla.f32   q8, q5, d0[0]                           @ w0 * "
                "inr00\n"
                "vld1.32    {d28-d31}, [%[ptr_out1]]                @ load "
                "outr1, w2, w3, c0~c3\n"
                "vmla.f32   q9, q5, d0[1]                           @ w0 * "
                "inr01\n"
                "vmla.f32   q10, q5, d1[0]                          @ w0 * "
                "inr02\n"
                "vmla.f32   q11, q5, d1[1]                          @ w0 * "
                "inr03\n"
                "vld1.32    {d3-d4}, [%[r1]]!                       @ load r1, "
                "4 float\n"
                "vmla.f32   q8, q6, d0[1]                           @ w1 * "
                "inr01\n"
                "vmla.f32   q9, q6, d1[0]                           @ w1 * "
                "inr02\n"
                "vmla.f32   q10, q6, d1[1]                          @ w1 * "
                "inr03\n"
                "vmla.f32   q11, q6, d2[0]                          @ w1 * "
                "inr04\n"
                "vld1.32    {d5}, [%[r1]]                           @ load r0, "
                "2 float\n"
                "vmla.f32   q8, q7, d1[0]                           @ w2 * "
                "inr02\n"
                "vmla.f32   q9, q7, d1[1]                           @ w2 * "
                "inr03\n"
                "vmla.f32   q10, q7, d2[0]                          @ w2 * "
                "inr04\n"
                "vmla.f32   q11, q7, d2[1]                          @ w2 * "
                "inr05\n"

                "sub    %[ptr_out1], %[ptr_out1], #32               @ ptr_out1 "
                "- 32, to start address\n"

                /* mul r1 with w0, w1, w2, get out r1 */
                "vmla.f32   q12, q5, d3[0]                          @ w0 * "
                "inr10\n"
                "vmla.f32   q13, q5, d3[1]                          @ w0 * "
                "inr11\n"
                "vmla.f32   q14, q5, d4[0]                          @ w0 * "
                "inr12\n"
                "vmla.f32   q15, q5, d4[1]                          @ w0 * "
                "inr13\n"
                "vmla.f32   q12, q6, d3[1]                          @ w1 * "
                "inr11\n"
                "vmla.f32   q13, q6, d4[0]                          @ w1 * "
                "inr12\n"
                "vmla.f32   q14, q6, d4[1]                          @ w1 * "
                "inr13\n"
                "vmla.f32   q15, q6, d5[0]                          @ w1 * "
                "inr14\n"
                "vld1.32    {d10-d13}, [%[wc0]]!                    @ load w3, "
                "w4, to q5, q6\n"
                "vmla.f32   q12, q7, d4[0]                          @ w2 * "
                "inr12\n"
                "vmla.f32   q13, q7, d4[1]                          @ w2 * "
                "inr13\n"
                "vmla.f32   q14, q7, d5[0]                          @ w2 * "
                "inr14\n"
                "vmla.f32   q15, q7, d5[1]                          @ w2 * "
                "inr15\n"
                "vld1.32    {d14-d15}, [%[wc0]]!                    @ load w5, "
                "to q7\n"

                /* mul r1 with w3, w4, w5, get out r0 */
                "vmla.f32   q8, q5, d3[0]                           @ w3 * "
                "inr10\n"
                "vmla.f32   q9, q5, d3[1]                           @ w3 * "
                "inr11\n"
                "vmla.f32   q10, q5, d4[0]                          @ w3 * "
                "inr12\n"
                "vmla.f32   q11, q5, d4[1]                          @ w3 * "
                "inr13\n"
                "vld1.32    {d0-d1}, [%[r2]]!                       @ load r2, "
                "4 float\n"
                "vmla.f32   q8, q6, d3[1]                           @ w4 * "
                "inr11\n"
                "vmla.f32   q9, q6, d4[0]                           @ w4 * "
                "inr12\n"
                "vmla.f32   q10, q6, d4[1]                          @ w4 * "
                "inr13\n"
                "vmla.f32   q11, q6, d5[0]                          @ w4 * "
                "inr14\n"
                "vld1.32    {d2}, [%[r2]]                           @ load r2, "
                "2 float\n"
                "vmla.f32   q8, q7, d4[0]                           @ w5 * "
                "inr12\n"
                "vmla.f32   q9, q7, d4[1]                           @ w5 * "
                "inr13\n"
                "vmla.f32   q10, q7, d5[0]                          @ w5 * "
                "inr14\n"
                "vmla.f32   q11, q7, d5[1]                          @ w5 * "
                "inr15\n"

                /* mul r2 with w3, w4, w5, get out r1 */
                "vmla.f32   q12, q5, d0[0]                          @ w3 * "
                "inr20\n"
                "vmla.f32   q13, q5, d0[1]                          @ w3 * "
                "inr21\n"
                "vmla.f32   q14, q5, d1[0]                          @ w3 * "
                "inr22\n"
                "vmla.f32   q15, q5, d1[1]                          @ w3 * "
                "inr23\n"
                "vmla.f32   q12, q6, d0[1]                          @ w4 * "
                "inr21\n"
                "vmla.f32   q13, q6, d1[0]                          @ w4 * "
                "inr22\n"
                "vmla.f32   q14, q6, d1[1]                          @ w4 * "
                "inr23\n"
                "vmla.f32   q15, q6, d2[0]                          @ w4 * "
                "inr24\n"
                "vld1.32    {d10-d13}, [%[wc0]]!                    @ load w6, "
                "w7, to q5, q6\n"
                "vmla.f32   q12, q7, d1[0]                          @ w5 * "
                "inr22\n"
                "vmla.f32   q13, q7, d1[1]                          @ w5 * "
                "inr23\n"
                "vmla.f32   q14, q7, d2[0]                          @ w5 * "
                "inr24\n"
                "vmla.f32   q15, q7, d2[1]                          @ w5 * "
                "inr25\n"
                "vld1.32    {d14-d15}, [%[wc0]]!                    @ load w8, "
                "to q7\n"

                "sub    %[wc0], %[wc0], #144                        @ wc0 - "
                "144 to start address\n"

                /* mul r2 with w6, w7, w8, get out r0 */
                "vmla.f32   q8, q5, d0[0]                           @ w6 * "
                "inr20\n"
                "vmla.f32   q9, q5, d0[1]                           @ w6 * "
                "inr21\n"
                "vld1.32    {d3-d4}, [%[r3]]!                       @ load r3, "
                "4 float\n"
                "vmla.f32   q10, q5, d1[0]                          @ w6 * "
                "inr22\n"
                "vmla.f32   q11, q5, d1[1]                          @ w6 * "
                "inr23\n"
                "vmla.f32   q8, q6, d0[1]                           @ w7 * "
                "inr21\n"
                "vmla.f32   q9, q6, d1[0]                           @ w7 * "
                "inr22\n"
                "vld1.32    {d5}, [%[r3]]                           @ load r3, "
                "2 float\n"
                "vmla.f32   q10, q6, d1[1]                          @ w7 * "
                "inr23\n"
                "vmla.f32   q11, q6, d2[0]                          @ w7 * "
                "inr24\n"
                "vmla.f32   q8, q7, d1[0]                           @ w8 * "
                "inr22\n"
                "vmla.f32   q9, q7, d1[1]                           @ w8 * "
                "inr23\n"
                "vld1.32    {d0-d1}, [%[r0]]!                       @ load r0, "
                "4 float\n"
                "vmla.f32   q10, q7, d2[0]                          @ w8 * "
                "inr24\n"
                "vmla.f32   q11, q7, d2[1]                          @ w8 * "
                "inr25\n"
                "vld1.32    {d2}, [%[r0]]                           @ load r0, "
                "2 float\n"

                /* mul r3 with w6, w7, w8, get out r1 */
                "vmla.f32   q12, q5, d3[0]                          @ w6 * "
                "inr20\n"
                "vmla.f32   q13, q5, d3[1]                          @ w6 * "
                "inr21\n"
                "vst1.32    {d16-d19}, [%[ptr_out0]]!               @ save "
                "r00, r01, c0~c3\n"
                "vmla.f32   q14, q5, d4[0]                          @ w6 * "
                "inr22\n"
                "vmla.f32   q15, q5, d4[1]                          @ w6 * "
                "inr23\n"
                "vst1.32    {d20-d23}, [%[ptr_out0]]!               @ save "
                "r02, r03, c0~c3\n"
                "vmla.f32   q12, q6, d3[1]                          @ w7 * "
                "inr21\n"
                "vmla.f32   q13, q6, d4[0]                          @ w7 * "
                "inr22\n"
                "vld1.32    {d16-d19}, [%[ptr_out0]]!               @ load "
                "outr0, w0, w1, c0~c3\n"
                "vmla.f32   q14, q6, d4[1]                          @ w7 * "
                "inr23\n"
                "vmla.f32   q15, q6, d5[0]                          @ w7 * "
                "inr24\n"
                "vld1.32    {d10-d13}, [%[wc0]]!                    @ load w0, "
                "w1, to q5, q6\n"
                "vmla.f32   q12, q7, d4[0]                          @ w8 * "
                "inr22\n"
                "vmla.f32   q13, q7, d4[1]                          @ w8 * "
                "inr23\n"
                "vld1.32    {d20-d23}, [%[ptr_out0]]                @ load "
                "outr0, w2, w3, c0~c3\n"
                "vmla.f32   q14, q7, d5[0]                          @ w8 * "
                "inr24\n"
                "vmla.f32   q15, q7, d5[1]                          @ w8 * "
                "inr25\n"

                "vst1.32    {d24-d27}, [%[ptr_out1]]!               @ save "
                "r10, r11, c0~c3\n"
                "vst1.32    {d28-d31}, [%[ptr_out1]]!               @ save "
                "r12, r13, c0~c3\n"
                "vld1.32    {d14-d15}, [%[wc0]]!                    @ load w2, "
                "to q7\n"

                "sub    %[ptr_out0], %[ptr_out0], #32               @ ptr_out0 "
                "- 32, to start address\n"

                "subs   %[cnt], #1                                  @ loop "
                "count--\n"
                "bne    0b                                          @ jump to "
                "main loop\n"

                : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2),
                  [r3] "+r"(r3), [ptr_out0] "+r"(ptr_out0),
                  [ptr_out1] "+r"(ptr_out1), [wc0] "+r"(wc0)
                :
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                  "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
          }
#endif  // __aarch64__
          block_inr0 = block_inr2;
          block_inr1 = block_inr3;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
        }
        slidingwindow_writeout_c4_fp32(pre_out, dout_batch, c, c + hout_c_block,
                                       h, h + h_kernel, 0, wout_round, chout,
                                       hout, wout, relu, ptr_write);
      }
      const float *weight_remain_ptr = weights + c_round_down * w_stride;
#pragma omp parallel for
      for (int c = 0; c < c_remain; ++c) {
#ifdef USE_OPENMP
        float *pre_out =
            pre_din + pre_in_size + omp_get_thread_num() * pre_out_size;
#else
        float *pre_out = pre_din + pre_in_size;
#endif

        int c_idx = c_round_down + c;

        int h_kernel = hout_r_block;
        if (h + hout_r_block > hout) {
          h_kernel = hout - h;
        }

        const float *block_inr0 = pre_din;
        const float *block_inr1 = block_inr0 + in_len;
        const float *block_inr2 = block_inr1 + in_len;
        const float *block_inr3 = block_inr2 + in_len;

        const float *bias_ptr = ptr_zero;
        if (bias != nullptr) {
          bias_ptr = bias + c_idx;
        }
        slidingwindow_fill_bias(pre_out, bias_ptr, 1, wout_round * h_kernel);

        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          const float *wc0 = weight_remain_ptr;

          const float *inr0 = block_inr0;
          const float *inr1 = block_inr1;
          const float *inr2 = block_inr2;
          const float *inr3 = block_inr3;

          float *pre_out0 = pre_out + hk * wout_round;
          float *pre_out1 = pre_out0 + wout_round;
#ifdef __aarch64__
          for (int i = 0; i < chin; ++i) {
            float *ptr_out0 = pre_out0;
            float *ptr_out1 = pre_out1;

            float32x4_t w0 = vdupq_n_f32(wc0[c]);       // w0, v23
            float32x4_t w1 = vdupq_n_f32(wc0[4 + c]);   // w1, v24
            float32x4_t w2 = vdupq_n_f32(wc0[8 + c]);   // w2, v25
            float32x4_t w3 = vdupq_n_f32(wc0[12 + c]);  // w3, v26
            float32x4_t w4 = vdupq_n_f32(wc0[16 + c]);  // w4, v27
            float32x4_t w5 = vdupq_n_f32(wc0[20 + c]);  // w5, v28
            float32x4_t w6 = vdupq_n_f32(wc0[24 + c]);  // w6, v29
            float32x4_t w7 = vdupq_n_f32(wc0[28 + c]);  // w7, v30
            float32x4_t w8 = vdupq_n_f32(wc0[32 + c]);  // w8, v31

            const float *r0 = inr0;
            const float *r1 = inr1;
            const float *r2 = inr2;
            const float *r3 = inr3;

            int cnt = w_loop;
            asm volatile(
                "ldr    q21, [%[ptr_out0]]          \n" /* load outr0, w0~w3*/
                "ldr    q22, [%[ptr_out1]]          \n" /* load outr1, w0~w3*/
                "ldp    q0, q1,   [%[r0]], #16      \n" /* load input r0*/
                "ldp    q2, q3,   [%[r1]], #16      \n" /* load input r1*/
                "ldp    q4, q5,   [%[r2]], #16      \n" /* load input r2*/
                "ldp    q6, q7,   [%[r3]], #16      \n" /* load input r3*/
                "2:                                 \n" /* main loop*/

                "fmla   v21.4s ,  %[w0].4s,  v0.4s  \n" /* outr0 = w0 * r0*/
                "fmla   v22.4s ,  %[w0].4s,  v2.4s  \n" /* outr1 = w0 * r1*/

                "ext    v8.16b,  v0.16b,  v1.16b, #4   \n" /* shift r0 left 1*/
                "ext    v10.16b,  v2.16b,  v3.16b, #4  \n" /* shift r1 left 1*/
                "ext    v9.16b,  v0.16b,  v1.16b, #8   \n" /* shift r0 left 2*/
                "ext    v11.16b,  v2.16b,  v3.16b, #8  \n" /* shift r1 left 2*/

                "ldp    q0, q1,   [%[r0]], #16      \n" /* load input r0*/

                "fmla   v21.4s ,  %[w1].4s,  v8.4s  \n" /* outr0 = w1 * r1*/
                "fmla   v22.4s ,  %[w1].4s,  v10.4s \n" /* outr1 = w1 * r2*/

                "fmla   v21.4s ,  %[w2].4s,  v9.4s  \n" /* outr0 = w2 * r1*/
                "fmla   v22.4s ,  %[w2].4s,  v11.4s \n" /* outr1 = w2 * r2*/

                "fmla   v21.4s ,  %[w3].4s,  v2.4s  \n" /* outr0 = w3 * r1*/
                "fmla   v22.4s ,  %[w3].4s,  v4.4s  \n" /* outr1 = w3 * r2*/

                "ext    v12.16b,  v4.16b,  v5.16b, #4\n" /* shift r2 left 1*/
                "ext    v14.16b,  v6.16b,  v7.16b, #4\n" /* shift r3 left 1*/
                "ext    v13.16b,  v4.16b,  v5.16b, #8\n" /* shift r2 left 2*/
                "ext    v15.16b,  v6.16b,  v7.16b, #8\n" /* shift r3 left 2*/

                "fmla   v21.4s ,  %[w4].4s,  v10.4s \n" /* outr0 = w4 * r1*/
                "fmla   v22.4s ,  %[w4].4s,  v12.4s \n" /* outr1 = w4 * r2*/

                "fmla   v21.4s ,  %[w5].4s,  v11.4s \n" /* outr0 = w5 * r1*/
                "fmla   v22.4s ,  %[w5].4s,  v13.4s \n" /* outr1 = w5 * r2*/

                "ldp    q2, q3,   [%[r1]], #16      \n" /* load input r0*/

                "fmla   v21.4s ,  %[w6].4s,  v4.4s  \n" /* outr0 = w6 * r2*/
                "fmla   v22.4s ,  %[w6].4s,  v6.4s  \n" /* outr1 = w6 * r3*/

                "ldp    q4, q5,   [%[r2]], #16      \n" /* load input r2*/

                "fmla   v21.4s ,  %[w7].4s,  v12.4s \n" /* outr0 = w7 * r1*/
                "fmla   v22.4s ,  %[w7].4s,  v14.4s \n" /* outr1 = w7 * r2*/

                "ldp    q6, q7,   [%[r3]], #16      \n" /* load input r3*/

                "fmla   v21.4s ,  %[w8].4s,  v13.4s \n" /* outr0 = w8 * r1*/
                "fmla   v22.4s ,  %[w8].4s,  v15.4s \n" /* outr1 = w8 * r2*/

                "str    q21,    [%[ptr_out0]], #16  \n" /*write output r0*/
                "str    q22,    [%[ptr_out1]], #16  \n" /*write output r1*/

                "subs   %w[cnt], %w[cnt], #1        \n" /*loop count -1*/

                "ldr    q21, [%[ptr_out0]]          \n" /* load outr0, w0~w3*/
                "ldr    q22, [%[ptr_out1]]          \n" /* load outr1, w0~w3*/

                "bne    2b                          \n" /* jump to main loop*/

                : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2),
                  [r3] "+r"(r3), [ptr_out0] "+r"(ptr_out0),
                  [ptr_out1] "+r"(ptr_out1)
                : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3),
                  [w4] "w"(w4), [w5] "w"(w5), [w6] "w"(w6), [w7] "w"(w7),
                  [w8] "w"(w8)
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                  "v21", "v22");

            wc0 += 9 * hout_c_block;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
          }
#else   // not __aarch64__
          for (int i = 0; i < chin; ++i) {
            float *ptr_out0 = pre_out0;
            float *ptr_out1 = pre_out1;

            //! get valid weights of current output channel
            float w_tmp[10] = {
                wc0[c],      wc0[c + 4],  wc0[c + 8],  wc0[c + 12], wc0[c + 16],
                wc0[c + 20], wc0[c + 24], wc0[c + 28], wc0[c + 32], 0.f};
            float32x4_t w0 = vld1q_f32(w_tmp);      // w0, w1, w2, q0
            float32x4_t w1 = vld1q_f32(w_tmp + 3);  // w3, w4, w5, q1
            float32x4_t w2 = vld1q_f32(w_tmp + 6);  // w6, w7, w8, q2

            const float *r0 = inr0;
            const float *r1 = inr1;
            const float *r2 = inr2;
            const float *r3 = inr3;
            int cnt = w_loop / 2;
            if (cnt > 0) {
              asm volatile(
                  "vld1.32    {d24-d27},    [%[ptr_out0]]         @ load or00, "
                  "or01\n"
                  "vld1.32    {d6-d9},      [%[r0]]!              @ load r0, 8 "
                  "float\n"
                  "vld1.32    {d10},        [%[r0]]               @ load r0, 2 "
                  "float\n"
                  /* main loop */
                  "0:                                             @ main loop\n"
                  /* r0 * w0, w1, w2, get out r0*/
                  "vld1.32    {d28-d31},    [%[ptr_out1]]         @ load or10, "
                  "or11\n"
                  "vext.32    q8, q3, q4, #1                      @ r0, shift "
                  "left 1, get 1, 2, 3, 4\n"
                  "vext.32    q9, q4, q5, #1                      @ r0, shift "
                  "left 1, get 5, 6, 7, 8\n"
                  "vmla.f32   q12,    q3, %e[w0][0]               @ w00 * r0, "
                  "0, 1, 2, 3\n"
                  "vmla.f32   q13,    q4, %e[w0][0]               @ w00 * r0, "
                  "4, 5, 6, 7\n"
                  "vext.32    q10, q3, q4, #2                     @ r0, shift "
                  "left 2, get 2, 3, 4, 5\n"
                  "vext.32    q11, q4, q5, #2                     @ r0, shift "
                  "left 2, get 6, 7, 8, 9\n"
                  "vmla.f32   q12,    q8, %e[w0][1]               @ w01 * r0, "
                  "1, 2, 3, 4\n"
                  "vmla.f32   q13,    q9, %e[w0][1]               @ w01 * r0, "
                  "5, 6, 7, 8\n"
                  "vld1.32    {d6-d9},    [%[r1]]!                @ load r1, 8 "
                  "float\n"
                  "vmla.f32   q12,    q10, %f[w0][0]              @ w02 * r0, "
                  "2, 3, 4, 5\n"
                  "vmla.f32   q13,    q11, %f[w0][0]              @ w02 * r0, "
                  "6, 7, 8, 9\n"
                  "vld1.32    {d10},       [%[r1]]                @ load r1, 2 "
                  "float\n"

                  /* r1 * w3, w4, w5, get out r0*/
                  /* r1 * w0, w1, w2, get out r1*/
                  "vmla.f32   q12,    q3, %e[w1][0]               @ w10 * r1, "
                  "0, 1, 2, 3\n"
                  "vmla.f32   q13,    q4, %e[w1][0]               @ w10 * r1, "
                  "4, 5, 6, 7\n"
                  "vext.32    q8, q3, q4, #1                      @ r1, shift "
                  "left 1, get 1, 2, 3, 4\n"
                  "vext.32    q9, q4, q5, #1                      @ r1, shift "
                  "left 1, get 5, 6, 7, 8\n"
                  "vmla.f32   q14,    q3, %e[w0][0]               @ w00 * r1, "
                  "0, 1, 2, 3\n"
                  "vmla.f32   q15,    q4, %e[w0][0]               @ w00 * r1, "
                  "4, 5, 6, 7\n"
                  "vext.32    q10, q3, q4, #2                     @ r1, shift "
                  "left 2, get 2, 3, 4, 5\n"
                  "vext.32    q11, q4, q5, #2                     @ r1, shift "
                  "left 2, get 6, 7, 8, 9\n"
                  "vmla.f32   q12,    q8, %e[w1][1]               @ w11 * r1, "
                  "1, 2, 3, 4\n"
                  "vmla.f32   q13,    q9, %e[w1][1]               @ w11 * r1, "
                  "5, 6, 7, 8\n"
                  "vmla.f32   q14,    q8, %e[w0][1]               @ w01 * r1, "
                  "1, 2, 3, 4\n"
                  "vmla.f32   q15,    q9, %e[w0][1]               @ w01 * r1, "
                  "5, 6, 7, 8\n"
                  "vld1.32    {d6-d9},    [%[r2]]!                @ load r2, 8 "
                  "float\n"
                  "vmla.f32   q12,    q10, %f[w1][0]              @ w12 * r1, "
                  "2, 3, 4, 5\n"
                  "vmla.f32   q13,    q11, %f[w1][0]              @ w12 * r1, "
                  "6, 7, 8, 9\n"
                  "vmla.f32   q14,    q10, %f[w0][0]              @ w02 * r1, "
                  "2, 3, 4, 5\n"
                  "vmla.f32   q15,    q11, %f[w0][0]              @ w02 * r1, "
                  "6, 7, 8, 9\n"
                  "vld1.32    {d10},    [%[r2]]                   @ load r2, 2 "
                  "float\n"

                  /* r2 * w6, w7, w8, get out r0*/
                  /* r2 * w3, w4, w5, get out r1*/
                  "vmla.f32   q12,    q3, %e[w2][0]               @ w20 * r2, "
                  "0, 1, 2, 3\n"
                  "vmla.f32   q13,    q4, %e[w2][0]               @ w20 * r2, "
                  "4, 5, 6, 7\n"
                  "vext.32    q8, q3, q4, #1                      @ r2, shift "
                  "left 1, get 1, 2, 3, 4\n"
                  "vext.32    q9, q4, q5, #1                      @ r2, shift "
                  "left 1, get 5, 6, 7, 8\n"
                  "vmla.f32   q14,    q3, %e[w1][0]               @ w10 * r2, "
                  "0, 1, 2, 3\n"
                  "vmla.f32   q15,    q4, %e[w1][0]               @ w10 * r2, "
                  "4, 5, 6, 7\n"
                  "vext.32    q10, q3, q4, #2                     @ r2, shift "
                  "left 2, get 2, 3, 4, 5\n"
                  "vext.32    q11, q4, q5, #2                     @ r2, shift "
                  "left 2, get 6, 7, 8, 9\n"
                  "vmla.f32   q12,    q8, %e[w2][1]               @ w21 * r2, "
                  "1, 2, 3, 4\n"
                  "vmla.f32   q13,    q9, %e[w2][1]               @ w21 * r2, "
                  "5, 6, 7, 8\n"
                  "vmla.f32   q14,    q8, %e[w1][1]               @ w11 * r2, "
                  "1, 2, 3, 4\n"
                  "vmla.f32   q15,    q9, %e[w1][1]               @ w11 * r2, "
                  "5, 6, 7, 8\n"
                  "vld1.32    {d6-d9},    [%[r3]]!                @ load r3, 8 "
                  "float\n"
                  "vmla.f32   q12,    q10, %f[w2][0]              @ w22 * r2, "
                  "2, 3, 4, 5\n"
                  "vmla.f32   q13,    q11, %f[w2][0]              @ w22 * r2, "
                  "6, 7, 8, 9\n"
                  "vmla.f32   q14,    q10, %f[w1][0]              @ w12 * r2, "
                  "2, 3, 4, 5\n"
                  "vmla.f32   q15,    q11, %f[w1][0]              @ w12 * r2, "
                  "6, 7, 8, 9\n"
                  "vld1.32    {d10},    [%[r3]]                   @ load r3, 2 "
                  "float\n"

                  /* r3 * w6, w7, w8, get out r1*/
                  "vext.32    q8, q3, q4, #1                      @ r3, shift "
                  "left 1, get 1, 2, 3, 4\n"
                  "vext.32    q9, q4, q5, #1                      @ r3, shift "
                  "left 1, get 5, 6, 7, 8\n"
                  "vmla.f32   q14,    q3, %e[w2][0]               @ w20 * r3, "
                  "0, 1, 2, 3\n"
                  "vmla.f32   q15,    q4, %e[w2][0]               @ w20 * r3, "
                  "4, 5, 6, 7\n"
                  "vst1.32    {d24-d27},  [%[ptr_out0]]!          @ save or00, "
                  "or01\n"
                  "vext.32    q10, q3, q4, #2                     @ r3, shift "
                  "left 2, get 2, 3, 4, 5\n"
                  "vext.32    q11, q4, q5, #2                     @ r3, shift "
                  "left 2, get 6, 7, 8, 9\n"
                  "vmla.f32   q14,    q8, %e[w2][1]               @ w21 * r3, "
                  "0, 1, 2, 3\n"
                  "vmla.f32   q15,    q9, %e[w2][1]               @ w21 * r3, "
                  "4, 5, 6, 7\n"
                  "vld1.32    {d24-d27},  [%[ptr_out0]]           @ load or00, "
                  "or01\n"
                  "vld1.32    {d6-d9},    [%[r0]]!                @ load r3, 8 "
                  "float\n"
                  "vmla.f32   q14,    q10, %f[w2][0]              @ w22 * r3, "
                  "2, 3, 4, 5\n"
                  "vmla.f32   q15,    q11, %f[w2][0]              @ w22 * r3, "
                  "6, 7, 8, 9\n"
                  "vld1.32    {d10},    [%[r0]]                   @ load r0, 2 "
                  "float\n"
                  "vst1.32    {d28-d31},  [%[ptr_out1]]!          @ save or10, "
                  "or11\n"

                  "subs   %[cnt], #1                              @loop count "
                  "-1\n"
                  "bne    0b                                      @ jump to "
                  "main loop\n"

                  : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1),
                    [r2] "+r"(r2), [r3] "+r"(r3), [ptr_out0] "+r"(ptr_out0),
                    [ptr_out1] "+r"(ptr_out1)
                  : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2)
                  : "cc", "memory", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "q12", "q13", "q14", "q15");
              r0 -= 8;
            }
            //! deal with remain wout
            if (w_loop & 1) {
              ptr_out0[0] +=
                  r0[0] * w_tmp[0] + r0[1] * w_tmp[1] + r0[2] * w_tmp[2] +
                  r1[0] * w_tmp[3] + r1[1] * w_tmp[4] + r1[2] * w_tmp[5] +
                  r2[0] * w_tmp[6] + r2[1] * w_tmp[7] + r2[2] * w_tmp[8];

              ptr_out0[1] +=
                  r0[1] * w_tmp[0] + r0[2] * w_tmp[1] + r0[3] * w_tmp[2] +
                  r1[1] * w_tmp[3] + r1[2] * w_tmp[4] + r1[3] * w_tmp[5] +
                  r2[1] * w_tmp[6] + r2[2] * w_tmp[7] + r2[3] * w_tmp[8];

              ptr_out0[2] +=
                  r0[2] * w_tmp[0] + r0[3] * w_tmp[1] + r0[4] * w_tmp[2] +
                  r1[2] * w_tmp[3] + r1[3] * w_tmp[4] + r1[4] * w_tmp[5] +
                  r2[2] * w_tmp[6] + r2[3] * w_tmp[7] + r2[4] * w_tmp[8];

              ptr_out0[3] +=
                  r0[3] * w_tmp[0] + r0[4] * w_tmp[1] + r0[5] * w_tmp[2] +
                  r1[3] * w_tmp[3] + r1[4] * w_tmp[4] + r1[5] * w_tmp[5] +
                  r2[3] * w_tmp[6] + r2[4] * w_tmp[7] + r2[5] * w_tmp[8];

              ptr_out1[0] +=
                  r1[0] * w_tmp[0] + r1[1] * w_tmp[1] + r1[2] * w_tmp[2] +
                  r2[0] * w_tmp[3] + r2[1] * w_tmp[4] + r2[2] * w_tmp[5] +
                  r3[0] * w_tmp[6] + r3[1] * w_tmp[7] + r3[2] * w_tmp[8];

              ptr_out1[1] +=
                  r1[1] * w_tmp[0] + r1[2] * w_tmp[1] + r1[3] * w_tmp[2] +
                  r2[1] * w_tmp[3] + r2[2] * w_tmp[4] + r2[3] * w_tmp[5] +
                  r3[1] * w_tmp[6] + r3[2] * w_tmp[7] + r3[3] * w_tmp[8];

              ptr_out1[2] +=
                  r1[2] * w_tmp[0] + r1[3] * w_tmp[1] + r1[4] * w_tmp[2] +
                  r2[2] * w_tmp[3] + r2[3] * w_tmp[4] + r2[4] * w_tmp[5] +
                  r3[2] * w_tmp[6] + r3[3] * w_tmp[7] + r3[4] * w_tmp[8];

              ptr_out1[3] +=
                  r1[3] * w_tmp[0] + r1[4] * w_tmp[1] + r1[5] * w_tmp[2] +
                  r2[3] * w_tmp[3] + r2[4] * w_tmp[4] + r2[5] * w_tmp[5] +
                  r3[3] * w_tmp[6] + r3[4] * w_tmp[7] + r3[5] * w_tmp[8];
            }

            wc0 += 36;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
          }
#endif  // __aarch64__
          block_inr0 = block_inr2;
          block_inr1 = block_inr3;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
        }
        slidingwindow_writeout_c1_fp32(pre_out, dout_batch, c_idx, c_idx + 1, h,
                                       h + h_kernel, 0, wout_round, chout, hout,
                                       wout, relu, ptr_write);
      }
    }
  }
}

template <>
void SlidingwindowConv3x3s2Faster<float, float>(
    const framework::Tensor *input, framework::Tensor *filter,
    const std::vector<int> &paddings, framework::Tensor *output) {
  const float *din = input->data<float>();
  float *dout = output->mutable_data<float>();
  const float *weights = filter->mutable_data<float>();
  const float *bias = nullptr;
  bool relu = false;
  const int num = input->dims()[0];
  const int chin = input->dims()[1];
  const int hin = input->dims()[2];
  const int win = input->dims()[3];
  const int chout = output->dims()[1];
  const int hout = output->dims()[2];
  const int wout = output->dims()[3];
  const int pad_h = paddings[0];
  const int pad_w = paddings[1];
  const int threads = framework::CPUContext::Context()->get_thread_num();
  int l2_size =
      framework::CPUContext::Context()->get_l2_cache_size() / sizeof(float);
  const int hout_c_block = 4;
  const int hout_r_kernel = 2;
  const int wout_block = 4;
  const int wout_round = ((wout + wout_block - 1) / wout_block) * wout_block;
  const int win_round = wout_round * 2 /*stride_w*/ + 1;
  //! get h block
  //! win_round * chin * hin_r_block + wout_round * hout_c_block * hout_r_block
  //! * threads = l2_size win_round = 2 * wout_round + 1 hin_r_block = 2 *
  //! hout_r_block + 1
  int hout_r_block =
      (l2_size - 2 * wout_round * chin - chin) /
      ((4 * wout_round + 2) * chin + wout_round * hout_c_block * threads);
  hout_r_block = hout_r_block > hout ? hout : hout_r_block;
  hout_r_block = (hout_r_block / hout_r_kernel) * hout_r_kernel;
  hout_r_block = hout_r_block < hout_r_kernel ? hout_r_kernel : hout_r_block;

  const int hin_r_block = hout_r_block * 2 /*stride_h*/ + 1;

  float ptr_zero[win_round];
  memset(ptr_zero, 0, sizeof(float) * win_round);
  float ptr_write[wout_round];

  int in_len = win_round * chin;
  int pre_in_size = hin_r_block * in_len;
  int pre_out_size = hout_c_block * hout_r_block * wout_round;

  float *pre_din =
      static_cast<float *>(framework::CPUContext::Context()->get_work_space(
          (pre_in_size + threads * pre_out_size) * sizeof(float)));

  int size_in_channel = win * hin;
  int size_out_channel = wout * hout;
  int w_stride = chin * 9;               /*kernel_w * kernel_h*/
  int w_stride_chin = hout_c_block * 9;  // kernel_w * kernel_h *

  int ws = -pad_w;
  int we = ws + win_round;
  int w_loop = wout_round / 4;

  int c_remain = chout - (chout / hout_c_block) * hout_c_block;
  int c_round_down = (chout / hout_c_block) * hout_c_block;

  int out_row_stride = hout_c_block * wout_round;

  for (int n = 0; n < num; ++n) {
    const float *din_batch = din + n * chin * size_in_channel;
    float *dout_batch = dout + n * chout * size_out_channel;
    for (int h = 0; h < hout; h += hout_r_block) {
      int h_kernel = hout_r_block;
      if (h + hout_r_block > hout) {
        h_kernel = hout - h;
      }

      int hs = h * 2 /*stride_h*/ - pad_h;
      int he = hs + h_kernel * 2 /*stride_h*/ + 1;

      slidingwindow_prepack_input(din_batch, pre_din, 0, chin, hs, he, ws, we,
                                  chin, win, hin, ptr_zero);

      const float *cblock_inr0 = pre_din;
      const float *cblock_inr1 = cblock_inr0 + in_len;
      const float *cblock_inr2 = cblock_inr1 + in_len;
      const float *cblock_inr3 = cblock_inr2 + in_len;
      const float *cblock_inr4 = cblock_inr3 + in_len;

#pragma omp parallel for
      for (int c = 0; c < c_round_down; c += hout_c_block) {
#ifdef _OPENMP
        float *pre_out =
            pre_din + pre_in_size + omp_get_thread_num() * pre_out_size;
#else
        float *pre_out = pre_din + pre_in_size;
#endif
        const float *block_inr0 = cblock_inr0;
        const float *block_inr1 = cblock_inr1;
        const float *block_inr2 = cblock_inr2;
        const float *block_inr3 = cblock_inr3;
        const float *block_inr4 = cblock_inr4;

        const float *weight_c = weights + c * w_stride;
        const float *bias_ptr = ptr_zero;
        if (bias != nullptr) {
          bias_ptr = bias + c;
        }
        slidingwindow_fill_bias(pre_out, bias_ptr,
                                wout_round * hout_c_block * h_kernel);

        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          const float *wc0 = weight_c;

          const float *inr0 = block_inr0;
          const float *inr1 = block_inr1;
          const float *inr2 = block_inr2;
          const float *inr3 = block_inr3;
          const float *inr4 = block_inr4;

          float *pre_out0 = pre_out + hk * out_row_stride;
          float *pre_out1 = pre_out0 + out_row_stride;
#ifdef __aarch64__
          for (int i = 0; i < chin; ++i) {
            float *ptr_out0 = pre_out0;
            float *ptr_out1 = pre_out1;

            float32x4_t w0 = vld1q_f32(wc0);       // w0, v23
            float32x4_t w1 = vld1q_f32(wc0 + 4);   // w1, v24
            float32x4_t w2 = vld1q_f32(wc0 + 8);   // w2, v25
            float32x4_t w3 = vld1q_f32(wc0 + 12);  // w3, v26
            float32x4_t w4 = vld1q_f32(wc0 + 16);  // w4, v27
            float32x4_t w5 = vld1q_f32(wc0 + 20);  // w5, v28
            float32x4_t w6 = vld1q_f32(wc0 + 24);  // w6, v29
            float32x4_t w7 = vld1q_f32(wc0 + 28);  // w7, v30
            float32x4_t w8 = vld1q_f32(wc0 + 32);  // w8, v31

            const float *r0 = inr0;
            const float *r1 = inr1;
            const float *r2 = inr2;
            const float *r3 = inr3;
            const float *r4 = inr4;

            int cnt = w_loop;
            asm volatile(
                "ldp    q15, q16, [%[ptr_out0]]     \n" /* load outr00, outr01*/
                "ldp    q17, q18, [%[ptr_out0], #32]\n" /* load outr02, outr03*/

                "ldp    q0, q1,   [%[r0]], #32      \n" /* load input r0*/
                "ldr    d10,      [%[r0]]           \n" /* load input r0, 9th
                                                           element*/
                "ldp    q4, q5,   [%[r2]], #32      \n" /* load input r2*/
                "ldr    d12,      [%[r2]]           \n" /* load input r2, 9th
                                                           element*/
                "2:                                 \n" /* main loop*/
                /*  r0, r2, mul w0, get out r0, r1 */
                "ldp    q19, q20, [%[ptr_out1]]     \n" /* load outr10, outr11*/
                "ldp    q21, q22, [%[ptr_out1], #32]\n" /* load outr12, outr13*/
                "fmla   v15.4s ,  %[w0].4s,  v0.s[0]\n" /* outr00 = w0 * r0[0]*/
                "fmla   v16.4s ,  %[w0].4s,  v0.s[2]\n" /* outr01 = w0 * r0[2]*/
                "fmla   v17.4s ,  %[w0].4s,  v1.s[0]\n" /* outr02 = w0 * r0[4]*/
                "fmla   v18.4s ,  %[w0].4s,  v1.s[2]\n" /* outr03 = w0 * r0[6]*/
                "fmla   v19.4s ,  %[w0].4s,  v4.s[0]\n" /* outr10 = w0 * r2[0]*/
                "fmla   v20.4s ,  %[w0].4s,  v4.s[2]\n" /* outr11 = w0 * r2[2]*/
                "fmla   v21.4s ,  %[w0].4s,  v5.s[0]\n" /* outr12 = w0 * r2[4]*/
                "fmla   v22.4s ,  %[w0].4s,  v5.s[2]\n" /* outr13 = w0 * r2[6]*/

                "ldp    q2, q3,   [%[r1]], #32      \n" /* load input r1*/

                /* r2 mul w6, get out r0*/
                "fmla   v15.4s ,  %[w6].4s,  v4.s[0]\n" /* outr00 = w6 * r2[0]*/
                "fmla   v16.4s ,  %[w6].4s,  v4.s[2]\n" /* outr01 = w6 * r2[2]*/
                "fmla   v17.4s ,  %[w6].4s,  v5.s[0]\n" /* outr02 = w6 * r2[4]*/
                "fmla   v18.4s ,  %[w6].4s,  v5.s[2]\n" /* outr03 = w6 * r2[6]*/

                "ldr    d11,      [%[r1]]           \n" /* load input r1, 9th
                                                           element*/

                /*  r0, r2, mul w1, get out r0, r1 */
                "fmla   v15.4s ,  %[w1].4s,  v0.s[1]\n" /* outr00 = w1 * r0[1]*/
                "fmla   v16.4s ,  %[w1].4s,  v0.s[3]\n" /* outr01 = w1 * r0[3]*/
                "fmla   v17.4s ,  %[w1].4s,  v1.s[1]\n" /* outr02 = w1 * r0[5]*/
                "fmla   v18.4s ,  %[w1].4s,  v1.s[3]\n" /* outr03 = w1 * r0[7]*/
                "fmla   v19.4s ,  %[w1].4s,  v4.s[1]\n" /* outr10 = w1 * r2[1]*/
                "fmla   v20.4s ,  %[w1].4s,  v4.s[3]\n" /* outr11 = w1 * r2[3]*/
                "fmla   v21.4s ,  %[w1].4s,  v5.s[1]\n" /* outr12 = w1 * r2[5]*/
                "fmla   v22.4s ,  %[w1].4s,  v5.s[3]\n" /* outr13 = w1 * r2[7]*/

                "ldp    q6, q7,   [%[r3]], #32      \n" /* load input r3*/

                /*  r2 mul w7, get out r0 */
                "fmla   v15.4s ,  %[w7].4s,  v4.s[1]\n" /* outr00 = w7 * r2[1]*/
                "fmla   v16.4s ,  %[w7].4s,  v4.s[3]\n" /* outr01 = w7 * r2[3]*/
                "fmla   v17.4s ,  %[w7].4s,  v5.s[1]\n" /* outr02 = w7 * r2[5]*/
                "fmla   v18.4s ,  %[w7].4s,  v5.s[3]\n" /* outr03 = w7 * r2[7]*/

                "ldr    d13,      [%[r3]]           \n" /* load input r3, 9th
                                                           element*/

                /*  r0, r2, mul w2, get out r0, r1 */
                "fmla   v15.4s ,  %[w2].4s,  v0.s[2]\n" /* outr00 = w2 * r0[2]*/
                "fmla   v16.4s ,  %[w2].4s,  v1.s[0]\n" /* outr01 = w2 * r0[4]*/
                "fmla   v17.4s ,  %[w2].4s,  v1.s[2]\n" /* outr02 = w2 * r0[6]*/
                "fmla   v18.4s ,  %[w2].4s,  v10.s[0]\n" /* outr03 = w2 *
                                                            r0[8]*/
                "fmla   v19.4s ,  %[w2].4s,  v4.s[2]\n" /* outr10 = w2 * r2[2]*/
                "fmla   v20.4s ,  %[w2].4s,  v5.s[0]\n" /* outr11 = w2 * r2[4]*/
                "fmla   v21.4s ,  %[w2].4s,  v5.s[2]\n" /* outr12 = w2 * r2[6]*/
                "fmla   v22.4s ,  %[w2].4s,  v12.s[0]\n" /* outr13 = w2 *
                                                            r2[8]*/

                "ldp    q8, q9,   [%[r4]], #32      \n" /* load input r4*/

                /*  r2, mul w8, get out r0 */
                "fmla   v15.4s ,  %[w8].4s,  v4.s[2]\n" /* outr00 = w8 * r2[2]*/
                "fmla   v16.4s ,  %[w8].4s,  v5.s[0]\n" /* outr01 = w8 * r2[4]*/
                "fmla   v17.4s ,  %[w8].4s,  v5.s[2]\n" /* outr02 = w8 * r2[6]*/
                "fmla   v18.4s ,  %[w8].4s,  v12.s[0]\n" /* outr03 = w8 *
                                                            r2[8]*/

                "ldr    d14,      [%[r4]]           \n" /* load input r4, 9th
                                                           element*/

                /* r1, r3, mul w3, get out r0, r1 */
                "fmla   v15.4s ,  %[w3].4s,  v2.s[0]\n" /* outr00 = w3 * r1[0]*/
                "fmla   v16.4s ,  %[w3].4s,  v2.s[2]\n" /* outr01 = w3 * r1[2]*/
                "fmla   v17.4s ,  %[w3].4s,  v3.s[0]\n" /* outr02 = w3 * r1[4]*/
                "fmla   v18.4s ,  %[w3].4s,  v3.s[2]\n" /* outr03 = w3 * r1[6]*/
                "fmla   v19.4s ,  %[w3].4s,  v6.s[0]\n" /* outr10 = w3 * r3[0]*/
                "fmla   v20.4s ,  %[w3].4s,  v6.s[2]\n" /* outr11 = w3 * r3[2]*/
                "fmla   v21.4s ,  %[w3].4s,  v7.s[0]\n" /* outr12 = w3 * r3[4]*/
                "fmla   v22.4s ,  %[w3].4s,  v7.s[2]\n" /* outr13 = w3 * r3[6]*/

                "ldp    q0, q1,   [%[r0]], #32      \n" /* load input r0*/

                /*  r1, r3, mul w4, get out r0, r1 */
                "fmla   v15.4s ,  %[w4].4s,  v2.s[1]\n" /* outr00 = w4 * r1[1]*/
                "fmla   v16.4s ,  %[w4].4s,  v2.s[3]\n" /* outr01 = w4 * r1[3]*/
                "fmla   v17.4s ,  %[w4].4s,  v3.s[1]\n" /* outr02 = w4 * r1[5]*/
                "fmla   v18.4s ,  %[w4].4s,  v3.s[3]\n" /* outr03 = w4 * r1[7]*/
                "fmla   v19.4s ,  %[w4].4s,  v6.s[1]\n" /* outr10 = w4 * r3[1]*/
                "fmla   v20.4s ,  %[w4].4s,  v6.s[3]\n" /* outr11 = w4 * r3[3]*/
                "fmla   v21.4s ,  %[w4].4s,  v7.s[1]\n" /* outr12 = w4 * r3[5]*/
                "fmla   v22.4s ,  %[w4].4s,  v7.s[3]\n" /* outr13 = w4 * r3[7]*/

                "ldr    d10,      [%[r0]]           \n" /* load input r0, 9th
                                                           element*/

                /*  r1, r3, mul w5, get out r0, r1 */
                "fmla   v15.4s ,  %[w5].4s,  v2.s[2]\n" /* outr00 = w5 * r1[2]*/
                "fmla   v16.4s ,  %[w5].4s,  v3.s[0]\n" /* outr01 = w5 * r1[4]*/
                "fmla   v17.4s ,  %[w5].4s,  v3.s[2]\n" /* outr02 = w5 * r1[6]*/
                "fmla   v18.4s ,  %[w5].4s,  v11.s[0]\n" /* outr03 = w5 *
                                                            r1[8]*/

                "ldp    q4, q5,   [%[r2]], #32      \n" /* load input r2*/
                "stp    q15, q16, [%[ptr_out0]], #32\n" /* save outr00, outr01*/

                "fmla   v19.4s ,  %[w5].4s,  v6.s[2]\n" /* outr10 = w5 * r3[2]*/
                "fmla   v20.4s ,  %[w5].4s,  v7.s[0]\n" /* outr11 = w5 * r3[4]*/
                "fmla   v21.4s ,  %[w5].4s,  v7.s[2]\n" /* outr12 = w5 * r3[6]*/
                "fmla   v22.4s ,  %[w5].4s,  v13.s[0]\n" /* outr13 = w5 *
                                                            r3[8]*/

                "ldr    d12,      [%[r2]]           \n" /* load input r2, 9th
                                                           element*/
                "stp    q17, q18, [%[ptr_out0]], #32\n" /* save outr02, outr03*/

                /*  r4, mul w6, get out r1 */
                "fmla   v19.4s ,  %[w6].4s,  v8.s[0]\n" /* outr10 = w6 * r4[0]*/
                "fmla   v20.4s ,  %[w6].4s,  v8.s[2]\n" /* outr11 = w6 * r4[2]*/
                "fmla   v21.4s ,  %[w6].4s,  v9.s[0]\n" /* outr12 = w6 * r4[4]*/
                "fmla   v22.4s ,  %[w6].4s,  v9.s[2]\n" /* outr13 = w6 * r4[6]*/

                "ldp    q15, q16, [%[ptr_out0]]     \n" /* load outr00, outr01*/

                /*  r4, mul w7, get out r1 */
                "fmla   v19.4s ,  %[w7].4s,  v8.s[1]\n" /* outr10 = w7 * r4[1]*/
                "fmla   v20.4s ,  %[w7].4s,  v8.s[3]\n" /* outr11 = w7 * r4[3]*/
                "fmla   v21.4s ,  %[w7].4s,  v9.s[1]\n" /* outr12 = w7 * r4[5]*/
                "fmla   v22.4s ,  %[w7].4s,  v9.s[3]\n" /* outr13 = w7 * r4[7]*/

                "ldp    q17, q18, [%[ptr_out0], #32]\n" /* load outr02, outr03*/

                /*  r4, mul w8, get out r1 */
                "fmla   v19.4s ,  %[w8].4s,  v8.s[2]\n" /* outr10 = w8 * r4[2]*/
                "fmla   v20.4s ,  %[w8].4s,  v9.s[0]\n" /* outr11 = w8 * r4[4]*/
                "fmla   v21.4s ,  %[w8].4s,  v9.s[2]\n" /* outr12 = w8 * r4[6]*/
                "fmla   v22.4s ,  %[w8].4s,  v14.s[0]\n" /* outr13 = w8 *
                                                            r4[8]*/

                "subs   %w[cnt], %w[cnt], #1        \n" /*loop count -1*/

                "stp    q19, q20, [%[ptr_out1]], #32\n" /* save outr10, outr11*/
                "stp    q21, q22, [%[ptr_out1]], #32\n" /* save outr12, outr13*/

                "bne    2b                          \n" /* jump to main loop*/

                : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2),
                  [r3] "+r"(r3), [r4] "+r"(r4), [ptr_out0] "+r"(ptr_out0),
                  [ptr_out1] "+r"(ptr_out1)
                : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3),
                  [w4] "w"(w4), [w5] "w"(w5), [w6] "w"(w6), [w7] "w"(w7),
                  [w8] "w"(w8)
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                  "v16", "v17", "v18", "v19", "v20", "v21", "v22");

            wc0 += 9 * hout_c_block;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
            inr4 += win_round;
          }
#else   // not __aarch64__
          for (int i = 0; i < chin; ++i) {
            const float *wc0 = weight_c + i * w_stride_chin;

            float *ptr_out0 = pre_out0;
            float *ptr_out1 = pre_out1;

            const float *r0 = inr0;
            const float *r1 = inr1;
            const float *r2 = inr2;
            const float *r3 = inr3;
            const float *r4 = inr4;

            int cnt = w_loop;
            asm volatile(
                "vld1.32    {d16-d19}, [%[ptr_out0]]!               @ load "
                "outr0, w0, w1, c0~c3\n"
                "vld1.32    {d20-d23}, [%[ptr_out0]]                @ load "
                "outr0, w2, w3, c0~c3\n"

                /* load weights */
                "vld1.32    {d10-d13}, [%[wc0]]!                    @ load w0, "
                "w1, to q5, q6\n"
                "vld1.32    {d14-d15}, [%[wc0]]!                    @ load w2, "
                "to q7\n"

                /* load r0, r2 */
                "vld1.32    {d0-d3}, [%[r0]]!                       @ load r0, "
                "8 float\n"
                "vld1.32    {d8},   [%[r0]]                         @ load r0, "
                "9th float\n"

                "sub    %[ptr_out0], %[ptr_out0], #32               @ ptr_out0 "
                "- 32, to start address\n"

                /* main loop */
                "0:                                                 @ main "
                "loop\n"
                /* mul r0, with w0, w1, w2 */
                "vld1.32    {d24-d27}, [%[ptr_out1]]!               @ load "
                "outr1, w0, w1, c0~c3\n"
                "vmla.f32   q8, q5, d0[0]                           @ w0 * "
                "inr00\n"
                "vld1.32    {d28-d31}, [%[ptr_out1]]                @ load "
                "outr1, w2, w3, c0~c3\n"
                "vmla.f32   q9, q5, d1[0]                           @ w0 * "
                "inr02\n"
                "vmla.f32   q10, q5, d2[0]                          @ w0 * "
                "inr04\n"
                "vmla.f32   q11, q5, d3[0]                          @ w0 * "
                "inr06\n"
                "vld1.32    {d4-d7}, [%[r2]]!                       @ load r2, "
                "8 float\n"
                "vmla.f32   q8, q6, d0[1]                           @ w1 * "
                "inr01\n"
                "vmla.f32   q9, q6, d1[1]                           @ w1 * "
                "inr03\n"
                "vmla.f32   q10, q6, d2[1]                          @ w1 * "
                "inr05\n"
                "vmla.f32   q11, q6, d3[1]                          @ w1 * "
                "inr07\n"
                "vld1.32    {d9},   [%[r2]]                         @ load r2, "
                "9th float\n"
                "vmla.f32   q8, q7, d1[0]                           @ w2 * "
                "inr02\n"
                "vmla.f32   q9, q7, d2[0]                           @ w2 * "
                "inr04\n"
                "vmla.f32   q10, q7, d3[0]                          @ w2 * "
                "inr06\n"
                "vmla.f32   q11, q7, d8[0]                          @ w2 * "
                "inr08\n"

                "sub    %[r2], %[r2], #32                           @ r2 - 32, "
                "load r2 twice\n"

                /* mul r2, with w0, w1, w2 */
                "vld1.32    {d0-d3}, [%[r1]]!                       @ load r1, "
                "8 float\n"
                "vmla.f32   q12, q5, d4[0]                          @ w0 * "
                "inr20\n"
                "vmla.f32   q13, q5, d5[0]                          @ w0 * "
                "inr22\n"
                "vmla.f32   q14, q5, d6[0]                          @ w0 * "
                "inr24\n"
                "vmla.f32   q15, q5, d7[0]                          @ w0 * "
                "inr26\n"
                "vld1.32    {d8},   [%[r1]]                         @ load r1, "
                "9th float\n"
                "vmla.f32   q12, q6, d4[1]                          @ w1 * "
                "inr21\n"
                "vmla.f32   q13, q6, d5[1]                          @ w1 * "
                "inr23\n"
                "vmla.f32   q14, q6, d6[1]                          @ w1 * "
                "inr25\n"
                "vmla.f32   q15, q6, d7[1]                          @ w1 * "
                "inr27\n"
                "vld1.32    {d10-d13}, [%[wc0]]!                    @ load w3, "
                "w4, to q5, q6\n"
                "vmla.f32   q12, q7, d5[0]                          @ w2 * "
                "inr22\n"
                "vmla.f32   q13, q7, d6[0]                          @ w2 * "
                "inr24\n"
                "vmla.f32   q14, q7, d7[0]                          @ w2 * "
                "inr26\n"
                "vmla.f32   q15, q7, d9[0]                          @ w2 * "
                "inr28\n"
                "vld1.32    {d14-d15}, [%[wc0]]!                    @ load w5, "
                "to q7\n"

                /* mul r1, with w3, w4, w5 */
                "vmla.f32   q8, q5, d0[0]                           @ w3 * "
                "inr10\n"
                "vmla.f32   q9, q5, d1[0]                           @ w3 * "
                "inr12\n"
                "vmla.f32   q10, q5, d2[0]                          @ w3 * "
                "inr14\n"
                "vmla.f32   q11, q5, d3[0]                          @ w3 * "
                "inr16\n"
                "vld1.32    {d4-d7}, [%[r3]]!                       @ load r3, "
                "8 float\n"
                "vmla.f32   q8, q6, d0[1]                           @ w4 * "
                "inr11\n"
                "vmla.f32   q9, q6, d1[1]                           @ w4 * "
                "inr13\n"
                "vmla.f32   q10, q6, d2[1]                          @ w4 * "
                "inr15\n"
                "vmla.f32   q11, q6, d3[1]                          @ w4 * "
                "inr17\n"
                "vld1.32    {d9},   [%[r3]]                         @ load r3, "
                "9th float\n"
                "vmla.f32   q8, q7, d1[0]                           @ w5 * "
                "inr12\n"
                "vmla.f32   q9, q7, d2[0]                           @ w5 * "
                "inr14\n"
                "vmla.f32   q10, q7, d3[0]                          @ w5 * "
                "inr16\n"
                "vmla.f32   q11, q7, d8[0]                          @ w5 * "
                "inr18\n"

                "sub    %[ptr_out1], %[ptr_out1], #32               @ ptr_out1 "
                "- 32, to start address\n"

                /* mul r3, with w3, w4, w5 */
                "vld1.32    {d0-d3}, [%[r2]]!                       @ load r2, "
                "8 float\n"
                "vmla.f32   q12, q5, d4[0]                          @ w3 * "
                "inr30\n"
                "vmla.f32   q13, q5, d5[0]                          @ w3 * "
                "inr32\n"
                "vmla.f32   q14, q5, d6[0]                          @ w3 * "
                "inr34\n"
                "vmla.f32   q15, q5, d7[0]                          @ w3 * "
                "inr36\n"
                "vld1.32    {d8},   [%[r2]]                         @ load r2, "
                "9th float\n"
                "vmla.f32   q12, q6, d4[1]                          @ w4 * "
                "inr31\n"
                "vmla.f32   q13, q6, d5[1]                          @ w4 * "
                "inr33\n"
                "vmla.f32   q14, q6, d6[1]                          @ w4 * "
                "inr35\n"
                "vmla.f32   q15, q6, d7[1]                          @ w4 * "
                "inr37\n"
                "vld1.32    {d10-d13}, [%[wc0]]!                    @ load w6, "
                "w7, to q5, q6\n"
                "vmla.f32   q12, q7, d5[0]                          @ w5 * "
                "inr32\n"
                "vmla.f32   q13, q7, d6[0]                          @ w5 * "
                "inr34\n"
                "vmla.f32   q14, q7, d7[0]                          @ w5 * "
                "inr36\n"
                "vmla.f32   q15, q7, d9[0]                          @ w5 * "
                "inr38\n"
                "vld1.32    {d14-d15}, [%[wc0]]!                    @ load w8, "
                "to q7\n"

                /* mul r2, with w6, w7, w8 */
                "vmla.f32   q8, q5, d0[0]                           @ w6 * "
                "inr20\n"
                "vmla.f32   q9, q5, d1[0]                           @ w6 * "
                "inr22\n"
                "vmla.f32   q10, q5, d2[0]                          @ w6 * "
                "inr24\n"
                "vmla.f32   q11, q5, d3[0]                          @ w6 * "
                "inr26\n"
                "vld1.32    {d4-d7}, [%[r4]]!                       @ load r4, "
                "8 float\n"
                "vmla.f32   q8, q6, d0[1]                           @ w7 * "
                "inr21\n"
                "vmla.f32   q9, q6, d1[1]                           @ w7 * "
                "inr23\n"
                "vmla.f32   q10, q6, d2[1]                          @ w7 * "
                "inr25\n"
                "vmla.f32   q11, q6, d3[1]                          @ w7 * "
                "inr27\n"
                "vld1.32    {d9},   [%[r4]]                         @ load r4, "
                "9th float\n"
                "vmla.f32   q8, q7, d1[0]                           @ w8 * "
                "inr22\n"
                "vmla.f32   q9, q7, d2[0]                           @ w8 * "
                "inr24\n"
                "vmla.f32   q10, q7, d3[0]                          @ w8 * "
                "inr26\n"
                "vmla.f32   q11, q7, d8[0]                          @ w8 * "
                "inr28\n"

                "sub    %[wc0], %[wc0], #144                        @ wc0 - "
                "144 to start address\n"

                /* mul r4, with w6, w7, w8 */
                "vld1.32    {d0-d3}, [%[r0]]!                       @ load r0, "
                "8 float\n"
                "vmla.f32   q12, q5, d4[0]                          @ w3 * "
                "inr40\n"
                "vst1.32    {d16-d19}, [%[ptr_out0]]!               @ save "
                "r00, r01, c0~c3\n"
                "vmla.f32   q13, q5, d5[0]                          @ w3 * "
                "inr42\n"
                "vst1.32    {d20-d23}, [%[ptr_out0]]!               @ save "
                "r02, r03, c0~c3\n"
                "vmla.f32   q14, q5, d6[0]                          @ w3 * "
                "inr44\n"
                "vmla.f32   q15, q5, d7[0]                          @ w3 * "
                "inr46\n"
                "vld1.32    {d8},   [%[r0]]                         @ load r0, "
                "9th float\n"
                "vmla.f32   q12, q6, d4[1]                          @ w4 * "
                "inr41\n"
                "vmla.f32   q13, q6, d5[1]                          @ w4 * "
                "inr43\n"
                "vmla.f32   q14, q6, d6[1]                          @ w4 * "
                "inr45\n"
                "vmla.f32   q15, q6, d7[1]                          @ w4 * "
                "inr47\n"
                "vld1.32    {d10-d13}, [%[wc0]]!                    @ load w0, "
                "w1, to q5, q6\n"
                "vmla.f32   q12, q7, d5[0]                          @ w5 * "
                "inr42\n"
                "vmla.f32   q13, q7, d6[0]                          @ w5 * "
                "inr44\n"
                "vmla.f32   q14, q7, d7[0]                          @ w5 * "
                "inr46\n"
                "vmla.f32   q15, q7, d9[0]                          @ w5 * "
                "inr48\n"
                "vld1.32    {d14-d15}, [%[wc0]]!                    @ load w2, "
                "to q7\n"

                "vst1.32    {d24-d27}, [%[ptr_out1]]!               @ save "
                "r10, r11, c0~c3\n"
                "vst1.32    {d28-d31}, [%[ptr_out1]]!               @ save "
                "r12, r13, c0~c3\n"

                "vld1.32    {d16-d19}, [%[ptr_out0]]!               @ load "
                "outr0, w0, w1, c0~c3\n"
                "vld1.32    {d20-d23}, [%[ptr_out0]]                @ load "
                "outr0, w2, w3, c0~c3\n"

                "sub    %[ptr_out0], %[ptr_out0], #32               @ ptr_out0 "
                "- 32, to start address\n"

                "subs   %[cnt], #1                                  @ loop "
                "count--\n"
                "bne    0b                                          @ jump to "
                "main loop\n"

                : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2),
                  [r3] "+r"(r3), [r4] "+r"(r4), [ptr_out0] "+r"(ptr_out0),
                  [ptr_out1] "+r"(ptr_out1), [wc0] "+r"(wc0)
                :
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6",
                  "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
            inr4 += win_round;
          }
#endif  // __aarch64__
          block_inr0 = block_inr4;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }

        slidingwindow_writeout_c4_fp32(pre_out, dout_batch, c, c + hout_c_block,
                                       h, h + h_kernel, 0, wout_round, chout,
                                       hout, wout, relu, ptr_write);
      }

#pragma omp parallel for
      for (int c = 0; c < c_remain; ++c) {
#ifdef USE_OPENMP
        float *pre_out =
            pre_din + pre_in_size + omp_get_thread_num() * pre_out_size;
#else
        float *pre_out = pre_din + pre_in_size;
#endif

        const float *block_inr0 = cblock_inr0;
        const float *block_inr1 = cblock_inr1;
        const float *block_inr2 = cblock_inr2;
        const float *block_inr3 = cblock_inr3;
        const float *block_inr4 = cblock_inr4;

        //! get weights ptr of remained
        const float *weight_c = weights + c_round_down * w_stride;

        //! fill bias to one channel
        const float *bias_ptr = ptr_zero;
        if (bias != nullptr) {
          bias_ptr = bias + c_round_down + c;
        }
        slidingwindow_fill_bias(pre_out, bias_ptr, 1, wout_round * h_kernel);

        for (int hk = 0; hk < h_kernel; hk += hout_r_kernel) {
          const float *wc0 = weight_c;

          const float *inr0 = block_inr0;
          const float *inr1 = block_inr1;
          const float *inr2 = block_inr2;
          const float *inr3 = block_inr3;
          const float *inr4 = block_inr4;

          float *pre_out0 = pre_out + hk * wout_round;
          float *pre_out1 = pre_out0 + wout_round;
#ifdef __aarch64__
          for (int i = 0; i < chin; ++i) {
            float *ptr_out0 = pre_out0;
            float *ptr_out1 = pre_out1;

            //! get valid weights of current output channel
            float32x4_t w0 = vdupq_n_f32(wc0[c]);       // w0, v23
            float32x4_t w1 = vdupq_n_f32(wc0[c + 4]);   // w1, v24
            float32x4_t w2 = vdupq_n_f32(wc0[c + 8]);   // w2, v25
            float32x4_t w3 = vdupq_n_f32(wc0[c + 12]);  // w3, v26
            float32x4_t w4 = vdupq_n_f32(wc0[c + 16]);  // w4, v27
            float32x4_t w5 = vdupq_n_f32(wc0[c + 20]);  // w5, v28
            float32x4_t w6 = vdupq_n_f32(wc0[c + 24]);  // w6, v29
            float32x4_t w7 = vdupq_n_f32(wc0[c + 28]);  // w7, v30
            float32x4_t w8 = vdupq_n_f32(wc0[c + 32]);  // w8, v31

            const float *r0 = inr0;
            const float *r1 = inr1;
            const float *r2 = inr2;
            const float *r3 = inr3;
            const float *r4 = inr4;

            int cnt = w_loop;
            asm volatile(
                "ldr    q21, [%[ptr_out0]]          \n" /* load outr00, outr01,
                                                           outr02, outr03*/

                "ld2  {v0.4s, v1.4s}, [%[r0]], #32  \n" /* load input r0*/
                "ldr    d10,      [%[r0]]           \n" /* load input r0, 9th
                                                           element*/
                "ld2  {v4.4s, v5.4s}, [%[r2]], #32  \n" /* load input r2*/
                "ldr    d12,      [%[r2]]           \n" /* load input r2, 9th
                                                           element*/
                "2:                                 \n" /* main loop*/
                /*  r0, r2, mul w0, get out r0, r1 */
                "ldr    q22, [%[ptr_out1]]          \n" /* load outr10, outr11,
                                                           outr12, outr13*/

                "fmla   v21.4s ,  %[w0].4s,  v0.4s  \n" /* outr0 = w0 * r0[0, 2,
                                                           4, 6]*/
                "fmla   v22.4s ,  %[w0].4s,  v4.4s  \n" /* outr1 = w0 * r2[0, 2,
                                                           4, 6]*/

                "ld2  {v2.4s, v3.4s}, [%[r1]], #32  \n" /* load input r1*/

                /* r2 mul w6, get out r0*/
                "fmla   v21.4s ,  %[w6].4s,  v4.4s  \n" /* outr0 = w6 * r2[0, 2,
                                                           4, 6]*/
                "ldr    d11,      [%[r1]]           \n" /* load input r1, 9th
                                                           element*/

                /* shift left 1 */
                "ext    v15.16b, v0.16b, v10.16b, #4\n" /* shift left r0 1*/
                "ext    v16.16b, v4.16b, v12.16b, #4\n" /* shift left r2 1*/

                /*  r0, r2, mul w1, get out r0, r1 */
                "fmla   v21.4s ,  %[w1].4s,  v1.4s  \n" /* outr0 = w1 * r0[1, 3,
                                                           5, 7]*/
                "fmla   v22.4s ,  %[w1].4s,  v5.4s  \n" /* outr1 = w1 * r2[1, 3,
                                                           5, 7]*/

                "ld2  {v6.4s, v7.4s}, [%[r3]], #32  \n" /* load input r3*/

                /*  r2 mul w7, get out r0 */
                "fmla   v21.4s ,  %[w7].4s,  v5.4s  \n" /* outr00 = w7 * r2[1,
                                                           3, 5, 7]*/

                "ldr    d13,      [%[r3]]           \n" /* load input r3, 9th
                                                           element*/

                /*  r0, r2, mul w2, get out r0, r1 */
                "fmla   v21.4s ,  %[w2].4s,  v15.4s \n" /* outr0 = w2 * r0[2, 4,
                                                           6, 8]*/
                "fmla   v22.4s ,  %[w2].4s,  v16.4s \n" /* outr1 = w2 * r2[2, 4,
                                                           6, 8]*/

                "ld2  {v8.4s, v9.4s}, [%[r4]], #32  \n" /* load input r4*/

                /*  r2, mul w8, get out r0 */
                "fmla   v21.4s ,  %[w8].4s,  v16.4s \n" /* outr00 = w8 * r2[2,
                                                           4, 6, 8]*/

                "ldr    d14,      [%[r4]]           \n" /* load input r4, 9th
                                                           element*/

                /* r1, r3, mul w3, get out r0, r1 */
                "fmla   v21.4s ,  %[w3].4s,  v2.4s  \n" /* outr0 = w3 * r1[0, 2,
                                                           4, 6]*/
                "fmla   v22.4s ,  %[w3].4s,  v6.4s  \n" /* outr1 = w3 * r3[0, 2,
                                                           4, 6]*/

                /* shift left 1 */
                "ext    v15.16b, v2.16b, v11.16b, #4\n" /* shift left r1 1*/
                "ext    v16.16b, v6.16b, v13.16b, #4\n" /* shift left r3 1*/

                "ld2  {v0.4s, v1.4s}, [%[r0]], #32  \n" /* load input r0*/

                /*  r1, r3, mul w4, get out r0, r1 */
                "fmla   v21.4s ,  %[w4].4s,  v3.4s  \n" /* outr0 = w4 * r1[1, 3,
                                                           5, 7]*/
                "fmla   v22.4s ,  %[w4].4s,  v7.4s  \n" /* outr1 = w4 * r3[1, 3,
                                                           5, 7]*/

                "ldr    d10,      [%[r0]]           \n" /* load input r0, 9th
                                                           element*/

                /*  r1, r3, mul w5, get out r0, r1 */
                "fmla   v21.4s ,  %[w5].4s,  v15.4s \n" /* outr0 = w5 * r1[2]*/
                "fmla   v22.4s ,  %[w5].4s,  v16.4s \n" /* outr1 = w5 * r1[4]*/

                "ld2  {v4.4s, v5.4s}, [%[r2]], #32  \n" /* load input r2*/
                "ldr    d12,      [%[r2]]           \n" /* load input r2, 9th
                                                           element*/
                "str    q21, [%[ptr_out0]], #16     \n" /* save outr00, outr01*/

                /*  r4, mul w6, get out r1 */
                "fmla   v22.4s ,  %[w6].4s,  v8.4s  \n" /* outr1 = w6 * r4[0, 2,
                                                           4, 6]*/

                "ext    v15.16b, v8.16b, v14.16b, #4\n" /* shift left r1 1*/
                "ldr    q21, [%[ptr_out0]]          \n" /* load outr0*/

                /*  r4, mul w7, get out r1 */
                "fmla   v22.4s ,  %[w7].4s,  v9.4s  \n" /* outr1 = w7 * r4[1, 3,
                                                           5, 7]*/

                /*  r4, mul w8, get out r1 */
                "fmla   v22.4s ,  %[w8].4s,  v15.4s \n" /* outr1 = w8 * r4[2, 4,
                                                           6, 8]*/

                "subs   %w[cnt], %w[cnt], #1        \n" /*loop count -1*/
                "str    q22, [%[ptr_out1]], #16     \n" /* save outr1*/
                "bne    2b                          \n" /* jump to main loop*/

                : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1), [r2] "+r"(r2),
                  [r3] "+r"(r3), [r4] "+r"(r4), [ptr_out0] "+r"(ptr_out0),
                  [ptr_out1] "+r"(ptr_out1)
                : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2), [w3] "w"(w3),
                  [w4] "w"(w4), [w5] "w"(w5), [w6] "w"(w6), [w7] "w"(w7),
                  [w8] "w"(w8)
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
                  "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                  "v16", "v21", "v22");

            wc0 += 36;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
            inr4 += win_round;
          }
#else   // not __aarch64__
          for (int i = 0; i < chin; ++i) {
            float *ptr_out0 = pre_out0;
            float *ptr_out1 = pre_out1;

            //! get valid weights of current output channel
            float w_tmp[12] = {wc0[c],      wc0[c + 4],  wc0[c + 8],  0.f,
                               wc0[c + 12], wc0[c + 16], wc0[c + 20], 0.f,
                               wc0[c + 24], wc0[c + 28], wc0[c + 32], 0.f};
            float32x4_t w0 = vld1q_f32(w_tmp);      // w0, w1, w2, q0
            float32x4_t w1 = vld1q_f32(w_tmp + 4);  // w3, w4, w5, q1
            float32x4_t w2 = vld1q_f32(w_tmp + 8);  // w6, w7, w8, q2

            const float *r0 = inr0;
            const float *r1 = inr1;
            const float *r2 = inr2;
            const float *r3 = inr3;
            const float *r4 = inr4;

            int cnt = w_loop / 2;
            if (cnt > 0) {
              asm volatile(
                  /* main loop */
                  "0:                                                     @ "
                  "main loop\n"
                  "vld1.32    {d24-d27},    [%[ptr_out0]]         @ load or00, "
                  "or01\n"
                  "vld1.32    {d28-d31},    [%[ptr_out1]]         @ load or10, "
                  "or11\n"
                  "vld2.32    {d6-d9},    [%[r2]]!                @ load r2, 8 "
                  "float, interleave\n"
                  "vld2.32    {d10-d13},  [%[r2]]!                @ load r2, 8 "
                  "float, interleave\n"
                  "vld1.32    {d22},  [%[r2]]                     @ load 16th "
                  "float\n"

                  /* r2 * w2, r2 * w0, get or0, or1 */
                  "vmla.f32   q12,    q4, %e[w2][1]               @ w21 * r2, "
                  "1, 3, 5, 7\n"
                  "vmla.f32   q13,    q6, %e[w2][1]               @ w21 * r2, "
                  "9, 11, 13, 15\n"
                  "vld2.32    {d14-d17},    [%[r0]]!              @ load r0, 8 "
                  "float, interleave\n"
                  "vmla.f32   q14,    q4, %e[w0][1]               @ w01 * r2, "
                  "1, 3, 5, 7\n"
                  "vmla.f32   q15,    q6, %e[w0][1]               @ w01 * r2, "
                  "9, 11, 13, 15\n"

                  "vext.32    q4, q3, q5, #1                      @ r2, shift "
                  "left 1, get 2, 4, 6, 8\n"
                  "vext.32    q6, q5, q11, #1                     @ r2, shift "
                  "left 1, get 10, 12, 14, 16\n"

                  "vmla.f32   q12,    q3, %e[w2][0]               @ w20 * r2, "
                  "0, 2, 4, 6\n"
                  "vmla.f32   q13,    q5, %e[w2][0]               @ w20 * r2, "
                  "8, 10, 12, 14\n"
                  "vld2.32    {d18-d21},  [%[r0]]!                @ load r0, 8 "
                  "float, interleave\n"
                  "vmla.f32   q14,    q3, %e[w0][0]               @ w00 * r2, "
                  "0, 2, 4, 6\n"
                  "vmla.f32   q15,    q5, %e[w0][0]               @ w00 * r2, "
                  "8, 10, 12, 14\n"

                  "vld1.32    {d22},  [%[r0]]                     @ load 16th "
                  "float\n"

                  "vmla.f32   q12,    q4, %f[w2][0]               @ w22 * r2, "
                  "2, 4, 6, 8\n"
                  "vmla.f32   q14,    q4, %f[w0][0]               @ w02 * r2, "
                  "2, 4, 6, 8\n"
                  "vld2.32    {d6-d9},    [%[r3]]!                @ load r3, 8 "
                  "float, interleave\n"
                  "vmla.f32   q13,    q6, %f[w2][0]               @ w22 * r2, "
                  "10, 12, 14, 16\n"
                  "vmla.f32   q15,    q6, %f[w0][0]               @ w02 * r2, "
                  "10, 12, 14, 16\n"
                  "vld2.32    {d10-d13},  [%[r3]]!                @ load r3, 8 "
                  "float, interleave\n"

                  /* r0 * w0, get or0, r3 * w1, get or1*/
                  "vmla.f32   q12,    q8, %e[w0][1]               @ w01 * r0, "
                  "1, 3, 5, 7\n"
                  "vmla.f32   q13,    q10, %e[w0][1]              @ w01 * r0, "
                  "9, 11, 13, 15\n"
                  "vext.32    q8, q7, q9, #1                      @ r0, shift "
                  "left 1, get 2, 4, 6, 8\n"
                  "vext.32    q10, q9, q11, #1                    @ r0, shift "
                  "left 1, get 10, 12, 14, 16\n"
                  "vld1.32    {d22},  [%[r3]]                     @ load 16th "
                  "float\n"
                  "vmla.f32   q14,    q4, %e[w1][1]               @ w11 * r3, "
                  "1, 3, 5, 7\n"
                  "vmla.f32   q15,    q6, %e[w1][1]               @ w11 * r3, "
                  "9, 11, 13, 15\n"

                  "vmla.f32   q12,    q7, %e[w0][0]               @ w00 * r0, "
                  "0, 2, 4, 6\n"
                  "vmla.f32   q13,    q9, %e[w0][0]               @ w00 * r0, "
                  "8, 10, 12, 14\n"
                  "vext.32    q4, q3, q5, #1                      @ r3, shift "
                  "left 1, get 2, 4, 6, 8\n"
                  "vext.32    q6, q5, q11, #1                     @ r3, shift "
                  "left 1, get 10, 12, 14, 16\n"
                  "vmla.f32   q14,    q3, %e[w1][0]               @ w10 * r3, "
                  "0, 2, 4, 6\n"
                  "vmla.f32   q15,    q5, %e[w1][0]               @ w10 * r3, "
                  "8, 10, 12, 14\n"

                  "vmla.f32   q12,    q8, %f[w0][0]               @ w02 * r0, "
                  "2, 4, 6, 8\n"
                  "vld2.32    {d14-d17},  [%[r1]]!                @ load r1, 8 "
                  "float, interleave\n"
                  "vmla.f32   q13,    q10,%f[w0][0]               @ w02 * r0, "
                  "10, 12, 14, 16\n"
                  "vld2.32    {d18-d21},  [%[r1]]!                @ load r1, 8 "
                  "float, interleave\n"
                  "vmla.f32   q14,    q4, %f[w1][0]               @ w12 * r3, "
                  "2, 4, 6, 8\n"
                  "vld2.32    {d6-d9},    [%[r4]]!                @ load r4, 8 "
                  "float, interleave\n"
                  "vmla.f32   q15,    q6, %f[w1][0]               @ w12 * r3, "
                  "10, 12, 14, 16\n"
                  "vld2.32    {d10-d13},  [%[r4]]!                @ load r4, 8 "
                  "float, interleave\n"

                  "vld1.32    {d22},  [%[r1]]                     @ load 16th "
                  "float\n"

                  /* r1 * w1, get or0, r4 * w2, get or1 */
                  "vmla.f32   q12,    q8, %e[w1][1]               @ w11 * r1, "
                  "1, 3, 5, 7\n"
                  "vmla.f32   q13,    q10, %e[w1][1]              @ w11 * r1, "
                  "9, 11, 13, 15\n"
                  "vext.32    q8, q7, q9, #1                      @ r1, shift "
                  "left 1, get 2, 4, 6, 8\n"
                  "vext.32    q10, q9, q11, #1                    @ r1, shift "
                  "left 1, get 10, 12, 14, 16\n"
                  "vmla.f32   q14,    q4, %e[w2][1]               @ w21 * r4, "
                  "1, 3, 5, 7\n"
                  "vmla.f32   q15,    q6, %e[w2][1]               @ w21 * r4, "
                  "9, 11, 13, 15\n"
                  "vld1.32    {d22},  [%[r4]]                     @ load 16th "
                  "float\n"

                  "vmla.f32   q12,    q7, %e[w1][0]               @ w10 * r1, "
                  "0, 2, 4, 6\n"
                  "vmla.f32   q13,    q9, %e[w1][0]               @ w10 * r1, "
                  "8, 10, 12, 14\n"
                  "vext.32    q4, q3, q5, #1                      @ r1, shift "
                  "left 1, get 2, 4, 6, 8\n"
                  "vext.32    q6, q5, q11, #1                     @ r1, shift "
                  "left 1, get 10, 12, 14, 16\n"
                  "vmla.f32   q14,    q3, %e[w2][0]               @ w20 * r4, "
                  "0, 2, 4, 6\n"
                  "vmla.f32   q15,    q5, %e[w2][0]               @ w20 * r4, "
                  "8, 10, 12, 14\n"

                  "vmla.f32   q12,    q8, %f[w1][0]               @ w12 * r1, "
                  "2, 4, 6, 8\n"
                  "vmla.f32   q13,    q10, %f[w1][0]              @ w12 * r1, "
                  "10, 12, 14, 16\n"
                  "vmla.f32   q14,    q4, %f[w2][0]               @ w22 * r4, "
                  "2, 4, 6, 8\n"
                  "vmla.f32   q15,    q6, %f[w2][0]               @ w22 * r4, "
                  "10, 12, 14, 16\n"

                  "vst1.32    {d24-d27},  [%[ptr_out0]]!          @ save or0\n"
                  "vst1.32    {d28-d31},  [%[ptr_out1]]!          @ save or0\n"

                  "subs   %[cnt], #1                              @loop count "
                  "-1\n"
                  "bne    0b                                      @ jump to "
                  "main loop\n"

                  : [cnt] "+r"(cnt), [r0] "+r"(r0), [r1] "+r"(r1),
                    [r2] "+r"(r2), [r3] "+r"(r3), [r4] "+r"(r4),
                    [ptr_out0] "+r"(ptr_out0), [ptr_out1] "+r"(ptr_out1)
                  : [w0] "w"(w0), [w1] "w"(w1), [w2] "w"(w2)
                  : "cc", "memory", "q3", "q4", "q5", "q6", "q7", "q8", "q9",
                    "q10", "q11", "q12", "q13", "q14", "q15");
            }
            //! deal with remain wout
            if (w_loop & 1) {
              ptr_out0[0] +=
                  r0[0] * w_tmp[0] + r0[1] * w_tmp[1] + r0[2] * w_tmp[2] +
                  r1[0] * w_tmp[4] + r1[1] * w_tmp[5] + r1[2] * w_tmp[6] +
                  r2[0] * w_tmp[8] + r2[1] * w_tmp[9] + r2[2] * w_tmp[10];

              ptr_out0[1] +=
                  r0[2] * w_tmp[0] + r0[3] * w_tmp[1] + r0[4] * w_tmp[2] +
                  r1[2] * w_tmp[4] + r1[3] * w_tmp[5] + r1[4] * w_tmp[6] +
                  r2[2] * w_tmp[8] + r2[3] * w_tmp[9] + r2[4] * w_tmp[10];

              ptr_out0[2] +=
                  r0[4] * w_tmp[0] + r0[5] * w_tmp[1] + r0[6] * w_tmp[2] +
                  r1[4] * w_tmp[4] + r1[5] * w_tmp[5] + r1[6] * w_tmp[6] +
                  r2[4] * w_tmp[8] + r2[5] * w_tmp[9] + r2[6] * w_tmp[10];

              ptr_out0[3] +=
                  r0[6] * w_tmp[0] + r0[7] * w_tmp[1] + r0[8] * w_tmp[2] +
                  r1[6] * w_tmp[4] + r1[7] * w_tmp[5] + r1[8] * w_tmp[6] +
                  r2[6] * w_tmp[8] + r2[7] * w_tmp[9] + r2[8] * w_tmp[10];

              ptr_out1[0] +=
                  r2[0] * w_tmp[0] + r2[1] * w_tmp[1] + r2[2] * w_tmp[2] +
                  r3[0] * w_tmp[4] + r3[1] * w_tmp[5] + r3[2] * w_tmp[6] +
                  r4[0] * w_tmp[8] + r4[1] * w_tmp[9] + r4[2] * w_tmp[10];

              ptr_out1[1] +=
                  r2[2] * w_tmp[0] + r2[3] * w_tmp[1] + r2[4] * w_tmp[2] +
                  r3[2] * w_tmp[4] + r3[3] * w_tmp[5] + r3[4] * w_tmp[6] +
                  r4[2] * w_tmp[8] + r4[3] * w_tmp[9] + r4[4] * w_tmp[10];

              ptr_out1[2] +=
                  r2[4] * w_tmp[0] + r2[5] * w_tmp[1] + r2[6] * w_tmp[2] +
                  r3[4] * w_tmp[4] + r3[5] * w_tmp[5] + r3[6] * w_tmp[6] +
                  r4[4] * w_tmp[8] + r4[5] * w_tmp[9] + r4[6] * w_tmp[10];

              ptr_out1[3] +=
                  r2[6] * w_tmp[0] + r2[7] * w_tmp[1] + r2[8] * w_tmp[2] +
                  r3[6] * w_tmp[4] + r3[7] * w_tmp[5] + r3[8] * w_tmp[6] +
                  r4[6] * w_tmp[8] + r4[7] * w_tmp[9] + r4[8] * w_tmp[10];
            }

            wc0 += 36;
            inr0 += win_round;
            inr1 += win_round;
            inr2 += win_round;
            inr3 += win_round;
            inr4 += win_round;
          }
#endif  // __aarch64__
          block_inr0 = block_inr4;
          block_inr1 = block_inr0 + in_len;
          block_inr2 = block_inr1 + in_len;
          block_inr3 = block_inr2 + in_len;
          block_inr4 = block_inr3 + in_len;
        }
        slidingwindow_writeout_c1_fp32(
            pre_out, dout_batch, c + c_round_down, c + c_round_down + 1, h,
            h + h_kernel, 0, wout_round, chout, hout, wout, relu, ptr_write);
      }
    }
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
