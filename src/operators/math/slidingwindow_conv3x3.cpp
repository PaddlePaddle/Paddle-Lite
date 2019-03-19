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
#endif  //__aarch64__
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

#endif  //__aarch64__
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
#endif  //__aarch64__
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
#endif  //__aarch64__
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
#endif  //__aarch64__
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

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
