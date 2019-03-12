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
#include <float.h>
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

void SlidingwindowConv3x3s1(const framework::Tensor *input,
                            const framework::Tensor *filter,
                            const std::vector<int> &paddings,
                            framework::Tensor *output, framework::Tensor *bias,
                            bool if_bias, bool if_relu) {
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
  const int filter_ch_size = 9;
  const int pad_filter_ch_size = (2 * padding_h + 3) * (2 * padding_w + 3);
  const int pad_filter_start =
      2 * padding_h * (2 * padding_w + 3) + 2 * padding_w;
  const int pad_filter_w = 3 + padding_w * 2;
  const float *bias_data;
  bool if_nopadding = false;

  if (if_bias) {
    bias_data = bias->data<float>();
  }
#if __ARM_NEON
  float *out_ptr = output_data;
  for (int i = 0; i < output_ch; ++i) {
    float32x4_t _bias;
    if (if_bias) {
      _bias = vld1q_dup_f32(bias_data + i);
    } else {
      _bias = vdupq_n_f32(0.0);
    }
    int dim4 = out_ch_size >> 2;
    int lef4 = out_ch_size & 3;

    for (int j = 0; j < dim4; ++j) {
      vst1q_f32(out_ptr, _bias);
      out_ptr += 4;
    }
    if (lef4 != 0) {
      if (lef4 >= 1) {
        vst1q_lane_f32(out_ptr, _bias, 0);
        out_ptr++;
      }
      if (lef4 >= 2) {
        vst1q_lane_f32(out_ptr, _bias, 1);
        out_ptr++;
      }
      if (lef4 >= 3) {
        vst1q_lane_f32(out_ptr, _bias, 2);
        out_ptr++;
      }
    }
  }
#else
  int k = 0;
#pragma omp parallel for
  for (int i = 0; i < output_ch; ++i) {
    for (int j = 0; j < out_ch_size; ++j) {
      if (if_bias) {
        output_data[k++] = bias_data[i];
      } else {
        output_data[k++] = 0;
      }
    }
  }
#endif
  if (padding_h == 0 && padding_w == 0) {
    if_nopadding = true;
  }

  for (int b = 0; b < batch; ++b) {
    int output_ch_d2 = output_ch >> 1;

#pragma omp parallel for
    for (int o_c2 = 0; o_c2 < output_ch_d2; ++o_c2) {
      std::atomic<float> relu_value{0};
      const float *reluptr;
      int o_c = o_c2 * 2;
      bool issamefilter;
      const float *f1;
      const float *f1_2;
      const float *in_ptr1, *in_ptr2, *in_ptr3, *in_ptr4;
      const float *pad_filter0, *pad_filter1, *pad_filter2, *pad_filter3;
      const float *pad_filter0_2, *pad_filter1_2, *pad_filter2_2,
          *pad_filter3_2;
      float pad_filter_arr[pad_filter_ch_size] = {0};
      float pad_filter_arr_2[pad_filter_ch_size] = {0};

      float *output_data_ch;
      float *output_data_ch_2;
      const float *input_data_ch;
      const float *filter_data_ch;
      const float *filter_data_ch_2;

      filter_data_ch = filter_data + o_c * filter_ch_size * input_ch;
      filter_data_ch_2 = filter_data + (o_c + 1) * filter_ch_size * input_ch;

      input_data_ch = input_data;
      output_data_ch = output_data + o_c * out_ch_size;
      output_data_ch_2 = output_data + (o_c + 1) * out_ch_size;

      for (int i_c = 0; i_c < input_ch; ++i_c) {
        float relu_arr[4];
        reluptr = relu_arr;
        if (if_relu && i_c == input_ch - 1) {
          relu_value = 0;
        } else {
          relu_value = -FLT_MAX;
        }
        relu_arr[0] = relu_value;
        relu_arr[1] = relu_value;
        relu_arr[2] = relu_value;
        relu_arr[3] = relu_value;
        f1 = filter_data_ch;
        f1_2 = filter_data_ch_2;

        if (!if_nopadding) {
          pad_filter_arr[pad_filter_ch_size] = {0};
          for (int i = 0; i < 9; i++) {
            int j = i / 3 * (2 * padding_w + 3) + i % 3 + padding_h * 3 +
                    padding_w * (2 * padding_h + 1);
            pad_filter_arr[j] = filter_data_ch[i];
            pad_filter_arr_2[j] = filter_data_ch_2[i];
          }
          pad_filter1 = pad_filter_arr;
          pad_filter1 += pad_filter_start;
          pad_filter0 = pad_filter1 - pad_filter_w;
          pad_filter2 = pad_filter1 + pad_filter_w;
          pad_filter3 = pad_filter2 + pad_filter_w;

          pad_filter1_2 = pad_filter_arr_2;
          pad_filter1_2 += pad_filter_start;
          pad_filter0_2 = pad_filter1_2 - pad_filter_w;
          pad_filter2_2 = pad_filter1_2 + pad_filter_w;
          pad_filter3_2 = pad_filter2_2 + pad_filter_w;
        } else {
          pad_filter1 = filter_data_ch;
          pad_filter2 = pad_filter1 + 3;
          pad_filter3 = pad_filter2 + 3;

          pad_filter1_2 = filter_data_ch_2;
          pad_filter2_2 = pad_filter1_2 + 3;
          pad_filter3_2 = pad_filter2_2 + 3;
        }
        float *out_ptr1, *out_ptr2;
        float *out_ptr1_2, *out_ptr2_2;

        out_ptr1 = output_data_ch;
        out_ptr2 = out_ptr1 + output_w;
        out_ptr1_2 = output_data_ch_2;
        out_ptr2_2 = out_ptr1_2 + output_w;

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
            float sum1_2 = 0;
            float sum2_2 = 0;

            if (issamefilter) {
#if __ARM_NEON
              float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
              float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
              float32x4_t _pad_filter1_2 = vld1q_f32(pad_filter1_2);
              float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
              float32x4_t _sum1_2 = vmulq_f32(_in_ptr1, _pad_filter1_2);

              float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
              float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
              float32x4_t _pad_filter2_2 = vld1q_f32(pad_filter2_2);
              _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
              _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr2, _pad_filter2_2);
              float32x4_t _sum2 = vmulq_f32(_in_ptr2, _pad_filter1);
              float32x4_t _sum2_2 = vmulq_f32(_in_ptr2, _pad_filter1_2);

              float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
              float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
              float32x4_t _pad_filter3_2 = vld1q_f32(pad_filter3_2);
              _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
              _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr3, _pad_filter3_2);
              _sum2 = vmlaq_f32(_sum2, _in_ptr3, _pad_filter2);
              _sum2_2 = vmlaq_f32(_sum2_2, _in_ptr3, _pad_filter2_2);

              float32x4_t _in_ptr4 = vld1q_f32(in_ptr4);
              _sum2 = vmlaq_f32(_sum2, _in_ptr4, _pad_filter3);
              _sum2_2 = vmlaq_f32(_sum2_2, _in_ptr4, _pad_filter3_2);

              _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
              _sum1_2 = vsetq_lane_f32(sum1_2, _sum1_2, 3);
              _sum2 = vsetq_lane_f32(sum2, _sum2, 3);
              _sum2_2 = vsetq_lane_f32(sum2_2, _sum2_2, 3);

              float32x2_t _ss1 =
                  vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
              float32x2_t _ss1_2 =
                  vadd_f32(vget_low_f32(_sum1_2), vget_high_f32(_sum1_2));
              float32x2_t _ss2 =
                  vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
              float32x2_t _ss2_2 =
                  vadd_f32(vget_low_f32(_sum2_2), vget_high_f32(_sum2_2));
              float32x2_t _ssss1_ssss2 = vpadd_f32(_ss1, _ss2);
              float32x2_t _ssss1_2_ssss2_2 = vpadd_f32(_ss1_2, _ss2_2);

              sum1 += vget_lane_f32(_ssss1_ssss2, 0);
              sum1_2 += vget_lane_f32(_ssss1_2_ssss2_2, 0);
              sum2 += vget_lane_f32(_ssss1_ssss2, 1);
              sum2_2 += vget_lane_f32(_ssss1_2_ssss2_2, 1);
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

              sum1_2 += in_ptr1[0] * pad_filter1_2[0];
              sum1_2 += in_ptr1[1] * pad_filter1_2[1];
              sum1_2 += in_ptr1[2] * pad_filter1_2[2];
              sum1_2 += in_ptr2[0] * pad_filter2_2[0];
              sum1_2 += in_ptr2[1] * pad_filter2_2[1];
              sum1_2 += in_ptr2[2] * pad_filter2_2[2];
              sum1_2 += in_ptr3[0] * pad_filter3_2[0];
              sum1_2 += in_ptr3[1] * pad_filter3_2[1];
              sum1_2 += in_ptr3[2] * pad_filter3_2[2];

              sum2_2 += in_ptr2[0] * pad_filter1_2[0];
              sum2_2 += in_ptr2[1] * pad_filter1_2[1];
              sum2_2 += in_ptr2[2] * pad_filter1_2[2];
              sum2_2 += in_ptr3[0] * pad_filter2_2[0];
              sum2_2 += in_ptr3[1] * pad_filter2_2[1];
              sum2_2 += in_ptr3[2] * pad_filter2_2[2];
              sum2_2 += in_ptr4[0] * pad_filter3_2[0];
              sum2_2 += in_ptr4[1] * pad_filter3_2[1];
              sum2_2 += in_ptr4[2] * pad_filter3_2[2];
#endif
            } else {
#if __ARM_NEON
              float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
              float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
              float32x4_t _pad_filter1_2 = vld1q_f32(pad_filter1_2);
              float32x4_t _pad_filter0 = vld1q_f32(pad_filter0);
              float32x4_t _pad_filter0_2 = vld1q_f32(pad_filter0_2);

              float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
              float32x4_t _sum1_2 = vmulq_f32(_in_ptr1, _pad_filter1_2);
              float32x4_t _sum2 = vmulq_f32(_in_ptr1, _pad_filter0);
              float32x4_t _sum2_2 = vmulq_f32(_in_ptr1, _pad_filter0_2);

              float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
              float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
              float32x4_t _pad_filter2_2 = vld1q_f32(pad_filter2_2);

              _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
              _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr2, _pad_filter2_2);
              _sum2 = vmlaq_f32(_sum2, _in_ptr2, _pad_filter1);
              _sum2_2 = vmlaq_f32(_sum2_2, _in_ptr2, _pad_filter1_2);

              float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
              float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
              float32x4_t _pad_filter3_2 = vld1q_f32(pad_filter3_2);

              _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
              _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr3, _pad_filter3_2);
              _sum2 = vmlaq_f32(_sum2, _in_ptr3, _pad_filter2);
              _sum2_2 = vmlaq_f32(_sum2_2, _in_ptr3, _pad_filter2_2);

              _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
              _sum1_2 = vsetq_lane_f32(sum1_2, _sum1_2, 3);
              _sum2 = vsetq_lane_f32(sum2, _sum2, 3);
              _sum2_2 = vsetq_lane_f32(sum2_2, _sum2_2, 3);

              float32x2_t _ss1 =
                  vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
              float32x2_t _ss1_2 =
                  vadd_f32(vget_low_f32(_sum1_2), vget_high_f32(_sum1_2));
              float32x2_t _ss2 =
                  vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
              float32x2_t _ss2_2 =
                  vadd_f32(vget_low_f32(_sum2_2), vget_high_f32(_sum2_2));
              float32x2_t _ssss1_ssss2 = vpadd_f32(_ss1, _ss2);
              float32x2_t _ssss1_2_ssss2_2 = vpadd_f32(_ss1_2, _ss2_2);

              sum1 += vget_lane_f32(_ssss1_ssss2, 0);
              sum1_2 += vget_lane_f32(_ssss1_2_ssss2_2, 0);
              sum2 += vget_lane_f32(_ssss1_ssss2, 1);
              sum2_2 += vget_lane_f32(_ssss1_2_ssss2_2, 1);
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

              sum1_2 += in_ptr1[0] * pad_filter1_2[0];
              sum1_2 += in_ptr1[1] * pad_filter1_2[1];
              sum1_2 += in_ptr1[2] * pad_filter1_2[2];
              sum1_2 += in_ptr2[0] * pad_filter2_2[0];
              sum1_2 += in_ptr2[1] * pad_filter2_2[1];
              sum1_2 += in_ptr2[2] * pad_filter2_2[2];
              sum1_2 += in_ptr3[0] * pad_filter3_2[0];
              sum1_2 += in_ptr3[1] * pad_filter3_2[1];
              sum1_2 += in_ptr3[2] * pad_filter3_2[2];

              sum2_2 += in_ptr1[0] * pad_filter0_2[0];
              sum2_2 += in_ptr1[1] * pad_filter0_2[1];
              sum2_2 += in_ptr1[2] * pad_filter0_2[2];
              sum2_2 += in_ptr2[0] * pad_filter1_2[0];
              sum2_2 += in_ptr2[1] * pad_filter1_2[1];
              sum2_2 += in_ptr2[2] * pad_filter1_2[2];
              sum2_2 += in_ptr3[0] * pad_filter2_2[0];
              sum2_2 += in_ptr3[1] * pad_filter2_2[1];
              sum2_2 += in_ptr3[2] * pad_filter2_2[2];
#endif
            }
            if (!if_nopadding &&
                (o_w < padding_w || o_w > output_w - padding_w - 2)) {
              pad_filter0--;
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;

              pad_filter0_2--;
              pad_filter1_2--;
              pad_filter2_2--;
              pad_filter3_2--;
            } else {
              in_ptr1++;
              in_ptr2++;
              in_ptr3++;
              in_ptr4++;
            }
            *out_ptr1 += sum1;
            *out_ptr2 += sum2;
            *out_ptr1_2 += sum1_2;
            *out_ptr2_2 += sum2_2;

            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }
            if (out_ptr2[0] < relu_value) {
              *out_ptr2 = relu_value;
            }
            if (out_ptr1_2[0] < relu_value) {
              *out_ptr1_2 = relu_value;
            }
            if (out_ptr2_2[0] < relu_value) {
              *out_ptr2_2 = relu_value;
            }
            out_ptr1++;
            out_ptr2++;
            out_ptr1_2++;
            out_ptr2_2++;
          }
            // valid
#if __ARM_NEON
#if __aarch64__
          if (issamefilter) {
            int o_w_dim4 = (output_w - 2 * padding_w) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(
                  "prfm   pldl1keep, [%[f1], #256]          \n\t"
                  "prfm   pldl1keep, [%[f1_2], #256]        \n\t"

                  "ld1   {v0.4s, v1.4s}, [%[f1]]            \n\t"
                  "add        %[f1], %[f1], #32             \n\t"
                  "ld1   {v2.4s, v3.4s}, [%[f1_2]]          \n\t"
                  "add        %[f1_2], %[f1_2], #32         \n\t"

                  "prfm   pldl1keep, [%[reluptr], #64]      \n\t"
                  "ld1   {v16.4s}, [%[reluptr]]             \n\t"
                  "ld1   {v4.s}[0], [%[f1]]                 \n\t"

                  "sub        %[f1],%[f1], #32              \n\t"
                  "ld1   {v4.s}[1], [%[f1_2]]               \n\t"
                  "sub        %[f1_2],%[f1_2], #32          \n\t"

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
                  "prfm   pldl1keep, [%[out_ptr1_2], #128]  \n\t"
                  "prfm   pldl1keep, [%[out_ptr2], #128]    \n\t"
                  "prfm   pldl1keep, [%[out_ptr2_2], #128]  \n\t"

                  "ld1   {v12.4s}, [%[out_ptr1]]            \n\t"
                  "ld1   {v13.4s}, [%[out_ptr1_2]]          \n\t"
                  "ld1   {v14.4s}, [%[out_ptr2]]            \n\t"
                  "ld1   {v15.4s}, [%[out_ptr2_2]]          \n\t"

                  // in_ptr1 and in_ptr4 multiply
                  "ext    v8.16b, v5.16b, v6.16b, #4        \n\t"
                  "fmla   v12.4s, v5.4s, v0.4s[0]           \n\t"
                  "fmla   v13.4s, v5.4s, v2.4s[0]           \n\t"

                  "ext    v9.16b, v6.16b, v7.16b, #8        \n\t"
                  "fmla   v14.4s, v7.4s, v4.4s[0]           \n\t"
                  "fmla   v15.4s, v7.4s, v4.4s[1]           \n\t"

                  "ext    v10.16b, v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v0.4s[1]           \n\t"
                  "fmla   v13.4s, v8.4s, v2.4s[1]           \n\t"

                  "ext    v11.16b, v6.16b, v7.16b, #12      \n\t"
                  "fmla   v14.4s, v9.4s, v1.4s[2]           \n\t"
                  "fmla   v15.4s, v9.4s, v3.4s[2]           \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[in_ptr2]]       \n\t"
                  "fmla   v12.4s, v10.4s, v0.4s[2]          \n\t"
                  "fmla   v13.4s, v10.4s, v2.4s[2]          \n\t"

                  "add        %[in_ptr2],%[in_ptr2], #16    \n\t"
                  "fmla   v14.4s, v11.4s, v1.4s[3]          \n\t"
                  "fmla   v15.4s, v11.4s, v3.4s[3]          \n\t"

                  // in_ptr2 multiply
                  "ext    v8.16b,  v5.16b, v6.16b, #4       \n\t"
                  "fmla   v12.4s, v5.4s, v0.4s[3]           \n\t"
                  "fmla   v13.4s, v5.4s, v2.4s[3]           \n\t"

                  "fmla   v14.4s, v5.4s, v0.4s[0]           \n\t"
                  "fmla   v15.4s, v5.4s, v2.4s[0]           \n\t"

                  "ext    v9.16b,  v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v1.4s[0]           \n\t"
                  "fmla   v13.4s, v8.4s, v3.4s[0]           \n\t"

                  "ld1   {v6.d}[1], [%[in_ptr3]]            \n\t"
                  "add        %[in_ptr3],%[in_ptr3], #8     \n\t"
                  "fmla   v14.4s, v8.4s, v0.4s[1]           \n\t"
                  "fmla   v15.4s, v8.4s, v2.4s[1]           \n\t"

                  "ld1   {v7.4s}, [%[in_ptr3]]              \n\t"
                  "add        %[in_ptr3],%[in_ptr3], #8     \n\t"

                  "fmla   v12.4s, v9.4s, v1.4s[1]           \n\t"
                  "fmla   v13.4s, v9.4s, v3.4s[1]           \n\t"

                  "ext    v10.16b, v6.16b, v7.16b, #8       \n\t"
                  "fmla   v14.4s, v9.4s, v0.4s[2]           \n\t"
                  "fmla   v15.4s, v9.4s, v2.4s[2]           \n\t"

                  // in_ptr3 multiply
                  "fmla   v12.4s, v7.4s, v4.4s[0]           \n\t"
                  "fmla   v13.4s, v7.4s, v4.4s[1]           \n\t"

                  "ext    v11.16b, v6.16b,  v7.16b, #12     \n\t"
                  "fmla   v14.4s, v7.4s, v1.4s[1]           \n\t"
                  "fmla   v15.4s, v7.4s, v3.4s[1]           \n\t"

                  "fmla   v12.4s, v10.4s, v1.4s[2]          \n\t"
                  "fmla   v13.4s, v10.4s, v3.4s[2]          \n\t"

                  "fmla   v14.4s, v10.4s, v0.4s[3]          \n\t"
                  "fmla   v15.4s, v10.4s, v2.4s[3]          \n\t"

                  "fmla   v12.4s, v11.4s, v1.4s[3]          \n\t"
                  "fmla   v13.4s, v11.4s, v3.4s[3]          \n\t"

                  "prfm   pldl1keep, [%[in_ptr1], #192]     \n\t"
                  "fmla   v14.4s, v11.4s, v1.4s[0]          \n\t"
                  "fmla   v15.4s, v11.4s, v3.4s[0]          \n\t"

                  // store out_ptr
                  "prfm   pldl1keep, [%[in_ptr4], #192]     \n\t"
                  "fmax   v12.4s,v12.4s, v16.4s             \n\t"
                  "fmax   v13.4s,v13.4s, v16.4s             \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[in_ptr1]]       \n\t"
                  "add        %[in_ptr1],%[in_ptr1], #16    \n\t"
                  "fmax   v14.4s,v14.4s, v16.4s             \n\t"
                  "fmax   v15.4s,v15.4s, v16.4s             \n\t"
                  "st1   {v12.4s}, [%[out_ptr1]], #16       \n\t"

                  "ld1   {v6.d}[1], [%[in_ptr4]]            \n\t"
                  "add        %[in_ptr4],%[in_ptr4], #8     \n\t"
                  "st1   {v13.4s}, [%[out_ptr1_2]], #16     \n\t"

                  "ld1   {v7.4s}, [%[in_ptr4]]              \n\t"
                  "add        %[in_ptr4],%[in_ptr4], #8     \n\t"
                  "st1   {v14.4s}, [%[out_ptr2]], #16       \n\t"

                  "subs       %[o_w_dim4],%[o_w_dim4], #1   \n\t"
                  "st1   {v15.4s}, [%[out_ptr2_2]], #16     \n\t"

                  // cycle
                  "bne        0b                            \n\t"
                  "sub       %[in_ptr1],%[in_ptr1], #16     \n\t"
                  "sub       %[in_ptr4],%[in_ptr4], #16     \n\t"

                  : [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr2] "+r"(out_ptr2), [out_ptr1_2] "+r"(out_ptr1_2),
                    [out_ptr2_2] "+r"(out_ptr2_2), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3),
                    [in_ptr4] "+r"(in_ptr4)
                  : [f1] "r"(f1), [f1_2] "r"(f1_2), [reluptr] "r"(reluptr)
                  : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                    "v16");
            }
          }
          if (!if_nopadding && o_w == output_w - padding_w) {
            pad_filter0--;
            pad_filter1--;
            pad_filter2--;
            pad_filter3--;

            pad_filter0_2--;
            pad_filter1_2--;
            pad_filter2_2--;
            pad_filter3_2--;

            in_ptr1--;
            in_ptr2--;
            in_ptr3--;
            in_ptr4--;
          }
#else
          if (issamefilter) {
            int o_w_dim4 = (output_w - 2 * padding_w) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(
                  "pld        [%[f1], #256]                 \n\t"
                  "pld        [%[f1_2], #256]               \n\t"

                  "vld1.32   {d0-d3}, [%[f1]]               \n\t"
                  "add        %[f1], #32                    \n\t"
                  "vld1.32   {d4-d7}, [%[f1_2]]             \n\t"
                  "add        %[f1_2], #32                  \n\t"

                  "pld        [%[reluptr], #64]             \n\t"
                  "vld1.32   {d9}, [%[reluptr]]             \n\t"

                  "vld1.32   {d8[0]}, [%[f1]]               \n\t"
                  "sub        %[f1], #32                    \n\t"
                  "vld1.32   {d8[1]}, [%[f1_2]]             \n\t"
                  "sub        %[f1_2], #32                  \n\t"

                  "pld        [%[in_ptr1], #192]            \n\t"
                  "pld        [%[in_ptr4], #192]            \n\t"

                  "vld1.f32   {d10-d12}, [%[in_ptr1]]       \n\t"
                  "add        %[in_ptr1], #16               \n\t"

                  "vld1.f32   {d13-d15}, [%[in_ptr4]]       \n\t"
                  "add        %[in_ptr4], #16               \n\t"

                  "0:                                       \n\t"
                  // load out_ptr
                  "pld        [%[out_ptr1], #128]           \n\t"
                  "pld        [%[out_ptr1_2], #128]         \n\t"
                  "pld        [%[out_ptr2], #128]           \n\t"
                  "pld        [%[out_ptr2_2], #128]         \n\t"

                  "vld1.f32   {d24, d25}, [%[out_ptr1]]     \n\t"
                  "vld1.f32   {d26, d27}, [%[out_ptr1_2]]   \n\t"
                  "vld1.f32   {d28, d29}, [%[out_ptr2]]     \n\t"
                  "vld1.f32   {d30, d31}, [%[out_ptr2_2]]   \n\t"

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
                  "vmax.f32   d24, d24, d9                  \n\t"
                  "vmax.f32   d25, d25, d9                  \n\t"
                  "pld        [%[in_ptr1], #192]            \n\t"

                  "vmax.f32   d26, d26, d9                  \n\t"
                  "pld        [%[in_ptr4], #192]            \n\t"
                  "vmax.f32   d27, d27, d9                  \n\t"

                  "vmax.f32   d28, d28, d9                  \n\t"
                  "vld1.f32   {d10-d12}, [%[in_ptr1]]       \n\t"
                  "vmax.f32   d29, d29, d9                  \n\t"
                  "add        %[in_ptr1], #16               \n\t"

                  "vmax.f32   d30, d30, d9                  \n\t"
                  "vmax.f32   d31, d31, d9                  \n\t"
                  "vst1.f32   {d24, d25}, [%[out_ptr1]]!    \n\t"

                  "vst1.f32   {d26, d27}, [%[out_ptr1_2]]!  \n\t"
                  "vld1.f32   {d13-d15}, [%[in_ptr4]]       \n\t"

                  "add        %[in_ptr4], #16               \n\t"
                  "vst1.f32   {d28, d29}, [%[out_ptr2]]!    \n\t"

                  "subs       %[o_w_dim4], #1               \n\t"
                  "vst1.f32   {d30, d31}, [%[out_ptr2_2]]!  \n\t"

                  // cycle
                  "bne        0b                            \n\t"
                  "sub       %[in_ptr1], #16                \n\t"
                  "sub       %[in_ptr4], #16                \n\t"

                  : [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr2] "+r"(out_ptr2), [out_ptr1_2] "+r"(out_ptr1_2),
                    [out_ptr2_2] "+r"(out_ptr2_2), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3),
                    [in_ptr4] "+r"(in_ptr4)
                  : [f1] "r"(f1), [f1_2] "r"(f1_2), [reluptr] "r"(reluptr)
                  : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                    "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
            }
          }
          if (!if_nopadding && o_w == output_w - padding_w) {
            pad_filter0--;
            pad_filter1--;
            pad_filter2--;
            pad_filter3--;

            pad_filter0_2--;
            pad_filter1_2--;
            pad_filter2_2--;
            pad_filter3_2--;

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
            float sum1_2 = 0;
            float sum2_2 = 0;

            if (issamefilter) {
#if __ARM_NEON
              float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
              float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
              float32x4_t _pad_filter1_2 = vld1q_f32(pad_filter1_2);
              float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
              float32x4_t _sum1_2 = vmulq_f32(_in_ptr1, _pad_filter1_2);

              float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
              float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
              float32x4_t _pad_filter2_2 = vld1q_f32(pad_filter2_2);
              _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
              _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr2, _pad_filter2_2);
              float32x4_t _sum2 = vmulq_f32(_in_ptr2, _pad_filter1);
              float32x4_t _sum2_2 = vmulq_f32(_in_ptr2, _pad_filter1_2);

              float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
              float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
              float32x4_t _pad_filter3_2 = vld1q_f32(pad_filter3_2);
              _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
              _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr3, _pad_filter3_2);
              _sum2 = vmlaq_f32(_sum2, _in_ptr3, _pad_filter2);
              _sum2_2 = vmlaq_f32(_sum2_2, _in_ptr3, _pad_filter2_2);

              float32x4_t _in_ptr4 = vld1q_f32(in_ptr4);
              _sum2 = vmlaq_f32(_sum2, _in_ptr4, _pad_filter3);
              _sum2_2 = vmlaq_f32(_sum2_2, _in_ptr4, _pad_filter3_2);

              _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
              _sum1_2 = vsetq_lane_f32(sum1_2, _sum1_2, 3);
              _sum2 = vsetq_lane_f32(sum2, _sum2, 3);
              _sum2_2 = vsetq_lane_f32(sum2_2, _sum2_2, 3);

              float32x2_t _ss1 =
                  vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
              float32x2_t _ss1_2 =
                  vadd_f32(vget_low_f32(_sum1_2), vget_high_f32(_sum1_2));
              float32x2_t _ss2 =
                  vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
              float32x2_t _ss2_2 =
                  vadd_f32(vget_low_f32(_sum2_2), vget_high_f32(_sum2_2));
              float32x2_t _ssss1_ssss2 = vpadd_f32(_ss1, _ss2);
              float32x2_t _ssss1_2_ssss2_2 = vpadd_f32(_ss1_2, _ss2_2);

              sum1 += vget_lane_f32(_ssss1_ssss2, 0);
              sum1_2 += vget_lane_f32(_ssss1_2_ssss2_2, 0);
              sum2 += vget_lane_f32(_ssss1_ssss2, 1);
              sum2_2 += vget_lane_f32(_ssss1_2_ssss2_2, 1);
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

              sum1_2 += in_ptr1[0] * pad_filter1_2[0];
              sum1_2 += in_ptr1[1] * pad_filter1_2[1];
              sum1_2 += in_ptr1[2] * pad_filter1_2[2];
              sum1_2 += in_ptr2[0] * pad_filter2_2[0];
              sum1_2 += in_ptr2[1] * pad_filter2_2[1];
              sum1_2 += in_ptr2[2] * pad_filter2_2[2];
              sum1_2 += in_ptr3[0] * pad_filter3_2[0];
              sum1_2 += in_ptr3[1] * pad_filter3_2[1];
              sum1_2 += in_ptr3[2] * pad_filter3_2[2];

              sum2_2 += in_ptr2[0] * pad_filter1_2[0];
              sum2_2 += in_ptr2[1] * pad_filter1_2[1];
              sum2_2 += in_ptr2[2] * pad_filter1_2[2];
              sum2_2 += in_ptr3[0] * pad_filter2_2[0];
              sum2_2 += in_ptr3[1] * pad_filter2_2[1];
              sum2_2 += in_ptr3[2] * pad_filter2_2[2];
              sum2_2 += in_ptr4[0] * pad_filter3_2[0];
              sum2_2 += in_ptr4[1] * pad_filter3_2[1];
              sum2_2 += in_ptr4[2] * pad_filter3_2[2];
#endif
            } else {
#if __ARM_NEON
              float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
              float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
              float32x4_t _pad_filter1_2 = vld1q_f32(pad_filter1_2);
              float32x4_t _pad_filter0 = vld1q_f32(pad_filter0);
              float32x4_t _pad_filter0_2 = vld1q_f32(pad_filter0_2);

              float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
              float32x4_t _sum1_2 = vmulq_f32(_in_ptr1, _pad_filter1_2);
              float32x4_t _sum2 = vmulq_f32(_in_ptr1, _pad_filter0);
              float32x4_t _sum2_2 = vmulq_f32(_in_ptr1, _pad_filter0_2);

              float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
              float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
              float32x4_t _pad_filter2_2 = vld1q_f32(pad_filter2_2);

              _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
              _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr2, _pad_filter2_2);
              _sum2 = vmlaq_f32(_sum2, _in_ptr2, _pad_filter1);
              _sum2_2 = vmlaq_f32(_sum2_2, _in_ptr2, _pad_filter1_2);

              float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
              float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
              float32x4_t _pad_filter3_2 = vld1q_f32(pad_filter3_2);

              _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
              _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr3, _pad_filter3_2);
              _sum2 = vmlaq_f32(_sum2, _in_ptr3, _pad_filter2);
              _sum2_2 = vmlaq_f32(_sum2_2, _in_ptr3, _pad_filter2_2);

              _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
              _sum1_2 = vsetq_lane_f32(sum1_2, _sum1_2, 3);
              _sum2 = vsetq_lane_f32(sum2, _sum2, 3);
              _sum2_2 = vsetq_lane_f32(sum2_2, _sum2_2, 3);

              float32x2_t _ss1 =
                  vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
              float32x2_t _ss1_2 =
                  vadd_f32(vget_low_f32(_sum1_2), vget_high_f32(_sum1_2));
              float32x2_t _ss2 =
                  vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
              float32x2_t _ss2_2 =
                  vadd_f32(vget_low_f32(_sum2_2), vget_high_f32(_sum2_2));
              float32x2_t _ssss1_ssss2 = vpadd_f32(_ss1, _ss2);
              float32x2_t _ssss1_2_ssss2_2 = vpadd_f32(_ss1_2, _ss2_2);

              sum1 += vget_lane_f32(_ssss1_ssss2, 0);
              sum1_2 += vget_lane_f32(_ssss1_2_ssss2_2, 0);
              sum2 += vget_lane_f32(_ssss1_ssss2, 1);
              sum2_2 += vget_lane_f32(_ssss1_2_ssss2_2, 1);
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

              sum1_2 += in_ptr1[0] * pad_filter1_2[0];
              sum1_2 += in_ptr1[1] * pad_filter1_2[1];
              sum1_2 += in_ptr1[2] * pad_filter1_2[2];
              sum1_2 += in_ptr2[0] * pad_filter2_2[0];
              sum1_2 += in_ptr2[1] * pad_filter2_2[1];
              sum1_2 += in_ptr2[2] * pad_filter2_2[2];
              sum1_2 += in_ptr3[0] * pad_filter3_2[0];
              sum1_2 += in_ptr3[1] * pad_filter3_2[1];
              sum1_2 += in_ptr3[2] * pad_filter3_2[2];

              sum2_2 += in_ptr1[0] * pad_filter0_2[0];
              sum2_2 += in_ptr1[1] * pad_filter0_2[1];
              sum2_2 += in_ptr1[2] * pad_filter0_2[2];
              sum2_2 += in_ptr2[0] * pad_filter1_2[0];
              sum2_2 += in_ptr2[1] * pad_filter1_2[1];
              sum2_2 += in_ptr2[2] * pad_filter1_2[2];
              sum2_2 += in_ptr3[0] * pad_filter2_2[0];
              sum2_2 += in_ptr3[1] * pad_filter2_2[1];
              sum2_2 += in_ptr3[2] * pad_filter2_2[2];
#endif
            }
            if (!if_nopadding &&
                (o_w < padding_w || o_w > output_w - padding_w - 2)) {
              pad_filter0--;
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;

              pad_filter0_2--;
              pad_filter1_2--;
              pad_filter2_2--;
              pad_filter3_2--;
            } else {
              in_ptr1++;
              in_ptr2++;
              in_ptr3++;
              in_ptr4++;
            }
            *out_ptr1 += sum1;
            *out_ptr2 += sum2;
            *out_ptr1_2 += sum1_2;
            *out_ptr2_2 += sum2_2;
            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }
            if (out_ptr2[0] < relu_value) {
              *out_ptr2 = relu_value;
            }
            if (out_ptr1_2[0] < relu_value) {
              *out_ptr1_2 = relu_value;
            }
            if (out_ptr2_2[0] < relu_value) {
              *out_ptr2_2 = relu_value;
            }
            out_ptr1++;
            out_ptr2++;
            out_ptr1_2++;
            out_ptr2_2++;
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

            pad_filter0_2 -= 2;
            pad_filter1_2 -= 2;
            pad_filter2_2 -= 2;
            pad_filter3_2 -= 2;

          } else if (issamefilter) {
            in_ptr1 += 3 + input_w;
            in_ptr2 += 3 + input_w;
            in_ptr3 += 3 + input_w;
            in_ptr4 += 3 + input_w;

            pad_filter0 += 2 * padding_w + 1;
            pad_filter1 += 2 * padding_w + 1;
            pad_filter2 += 2 * padding_w + 1;
            pad_filter3 += 2 * padding_w + 1;

            pad_filter0_2 += 2 * padding_w + 1;
            pad_filter1_2 += 2 * padding_w + 1;
            pad_filter2_2 += 2 * padding_w + 1;
            pad_filter3_2 += 2 * padding_w + 1;

          } else {
            pad_filter0 -= 3 + 2 * padding_w + 2;
            pad_filter1 -= 3 + 2 * padding_w + 2;
            pad_filter2 -= 3 + 2 * padding_w + 2;
            pad_filter3 -= 3 + 2 * padding_w + 2;

            pad_filter0_2 -= 3 + 2 * padding_w + 2;
            pad_filter1_2 -= 3 + 2 * padding_w + 2;
            pad_filter2_2 -= 3 + 2 * padding_w + 2;
            pad_filter3_2 -= 3 + 2 * padding_w + 2;

            in_ptr1 -= input_w - 3;
            in_ptr2 -= input_w - 3;
            in_ptr3 -= input_w - 3;
            in_ptr4 -= input_w - 3;
          }
          out_ptr1 += output_w;
          out_ptr2 += output_w;
          out_ptr1_2 += output_w;
          out_ptr2_2 += output_w;
        }
        // remain output_height
        for (; o_h < output_h; ++o_h) {
          int o_w = 0;
          // pad left
          for (; o_w < padding_w; ++o_w) {
            float sum1 = 0;
            float sum1_2 = 0;
#if __ARM_NEON
            float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
            float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
            float32x4_t _pad_filter1_2 = vld1q_f32(pad_filter1_2);
            float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
            float32x4_t _sum1_2 = vmulq_f32(_in_ptr1, _pad_filter1_2);

            float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
            float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
            float32x4_t _pad_filter2_2 = vld1q_f32(pad_filter2_2);
            _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
            _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr2, _pad_filter2_2);

            float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
            float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
            float32x4_t _pad_filter3_2 = vld1q_f32(pad_filter3_2);
            _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
            _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr3, _pad_filter3_2);

            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
            _sum1_2 = vsetq_lane_f32(sum1_2, _sum1_2, 3);

            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ss1_2 =
                vadd_f32(vget_low_f32(_sum1_2), vget_high_f32(_sum1_2));
            float32x2_t _ssss1_ssss1_2 = vpadd_f32(_ss1, _ss1_2);

            sum1 += vget_lane_f32(_ssss1_ssss1_2, 0);
            sum1_2 += vget_lane_f32(_ssss1_ssss1_2, 1);
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

            sum1_2 += in_ptr1[0] * pad_filter1_2[0];
            sum1_2 += in_ptr1[1] * pad_filter1_2[1];
            sum1_2 += in_ptr1[2] * pad_filter1_2[2];
            sum1_2 += in_ptr2[0] * pad_filter2_2[0];
            sum1_2 += in_ptr2[1] * pad_filter2_2[1];
            sum1_2 += in_ptr2[2] * pad_filter2_2[2];
            sum1_2 += in_ptr3[0] * pad_filter3_2[0];
            sum1_2 += in_ptr3[1] * pad_filter3_2[1];
            sum1_2 += in_ptr3[2] * pad_filter3_2[2];
#endif
            if (!if_nopadding &&
                (o_w < padding_w || o_w > output_w - padding_w - 2)) {
              pad_filter0--;
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;

              pad_filter0_2--;
              pad_filter1_2--;
              pad_filter2_2--;
              pad_filter3_2--;
            } else {
              in_ptr1++;
              in_ptr2++;
              in_ptr3++;
              in_ptr4++;
            }
            *out_ptr1 += sum1;
            *out_ptr1_2 += sum1_2;

            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }
            if (out_ptr1_2[0] < relu_value) {
              *out_ptr1_2 = relu_value;
            }
            out_ptr1++;
            out_ptr1_2++;
          }
            // valid
#if __ARM_NEON
#if __aarch64__
          if (if_nopadding) {
            int o_w_dim4 = (output_w - 2 * padding_w) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(
                  "prfm   pldl1keep, [%[f1], #256]          \n\t"
                  "prfm   pldl1keep, [%[f1_2], #256]        \n\t"

                  "ld1   {v0.4s, v1.4s}, [%[f1]]            \n\t"
                  "add        %[f1], %[f1], #32             \n\t"
                  "ld1   {v2.4s, v3.4s}, [%[f1_2]]          \n\t"
                  "add        %[f1_2], %[f1_2], #32         \n\t"

                  "prfm   pldl1keep, [%[reluptr], #64]      \n\t"
                  "ld1   {v16.4s}, [%[reluptr]]             \n\t"

                  "ld1   {v4.s}[0], [%[f1]]                 \n\t"
                  "sub        %[f1],%[f1], #32              \n\t"
                  "ld1   {v4.s}[1], [%[f1_2]]               \n\t"
                  "sub        %[f1_2],%[f1_2], #32          \n\t"

                  "0:                                       \n\t"
                  // load out_ptr
                  "prfm   pldl1keep, [%[out_ptr1], #128]    \n\t"
                  "prfm   pldl1keep, [%[out_ptr1_2], #128]  \n\t"

                  "ld1   {v12.4s}, [%[out_ptr1]]            \n\t"
                  "ld1   {v13.4s}, [%[out_ptr1_2]]          \n\t"

                  // in_ptr1 multiply
                  "prfm   pldl1keep, [%[in_ptr1], #192]     \n\t"
                  "ld1   {v5.4s, v6.4s}, [%[in_ptr1]]       \n\t"
                  "add        %[in_ptr1],%[in_ptr1], #16    \n\t"

                  "ext    v8.16b, v5.16b, v6.16b, #4        \n\t"
                  "fmla   v12.4s, v5.4s, v0.4s[0]           \n\t"
                  "fmla   v13.4s, v5.4s, v2.4s[0]           \n\t"

                  "ext    v10.16b, v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v0.4s[1]           \n\t"
                  "fmla   v13.4s, v8.4s, v2.4s[1]           \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[in_ptr2]]       \n\t"
                  "add        %[in_ptr2],%[in_ptr2], #16    \n\t"
                  "fmla   v12.4s, v10.4s, v0.4s[2]          \n\t"
                  "fmla   v13.4s, v10.4s, v2.4s[2]          \n\t"

                  // in_ptr2 multiply
                  "ext    v8.16b,  v5.16b, v6.16b, #4       \n\t"
                  "fmla   v12.4s, v5.4s, v0.4s[3]           \n\t"
                  "fmla   v13.4s, v5.4s, v2.4s[3]           \n\t"

                  "ext    v9.16b,  v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v1.4s[0]           \n\t"
                  "fmla   v13.4s, v8.4s, v3.4s[0]           \n\t"

                  "ld1   {v6.d}[1], [%[in_ptr3]]            \n\t"
                  "add        %[in_ptr3],%[in_ptr3], #8     \n\t"
                  "ld1   {v7.4s}, [%[in_ptr3]]              \n\t"
                  "add        %[in_ptr3],%[in_ptr3], #8     \n\t"

                  "fmla   v12.4s, v9.4s, v1.4s[1]           \n\t"
                  "fmla   v13.4s, v9.4s, v3.4s[1]           \n\t"

                  // in_ptr3 multiply
                  "ext    v10.16b, v6.16b, v7.16b, #8       \n\t"
                  "fmla   v12.4s, v7.4s, v4.4s[0]           \n\t"
                  "fmla   v13.4s, v7.4s, v4.4s[1]           \n\t"

                  "ext    v11.16b, v6.16b,  v7.16b, #12     \n\t"
                  "fmla   v12.4s, v10.4s, v1.4s[2]          \n\t"
                  "fmla   v13.4s, v10.4s, v3.4s[2]          \n\t"

                  "fmla   v12.4s, v11.4s, v1.4s[3]          \n\t"
                  "fmla   v13.4s, v11.4s, v3.4s[3]          \n\t"

                  // store out_ptr
                  "fmax   v12.4s,v12.4s, v16.4s             \n\t"
                  "fmax   v13.4s,v13.4s, v16.4s             \n\t"

                  "st1   {v12.4s}, [%[out_ptr1]], #16       \n\t"
                  "st1   {v13.4s}, [%[out_ptr1_2]], #16     \n\t"

                  // cycle
                  "subs       %[o_w_dim4],%[o_w_dim4], #1   \n\t"
                  "bne        0b                            \n\t"

                  : [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr2] "+r"(out_ptr2), [out_ptr1_2] "+r"(out_ptr1_2),
                    [out_ptr2_2] "+r"(out_ptr2_2), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3),
                    [in_ptr4] "+r"(in_ptr4)
                  : [f1] "r"(f1), [f1_2] "r"(f1_2), [reluptr] "r"(reluptr)
                  : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v16");
            }
          }
#else
          if (if_nopadding) {
            int o_w_dim4 = (output_w - 2 * padding_w) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(
                  "pld        [%[f1], #256]                 \n\t"
                  "pld        [%[f1_2], #256]               \n\t"

                  "vld1.32   {d0-d3}, [%[f1]]               \n\t"
                  "add        %[f1], #32                    \n\t"
                  "vld1.32   {d4-d7}, [%[f1_2]]             \n\t"
                  "add        %[f1_2], #32                  \n\t"

                  "pld        [%[reluptr], #64]             \n\t"
                  "vld1.32   {d9}, [%[reluptr]]             \n\t"

                  "vld1.32   {d8[0]}, [%[f1]]               \n\t"
                  "sub        %[f1], #32                    \n\t"
                  "vld1.32   {d8[1]}, [%[f1_2]]             \n\t"
                  "sub        %[f1_2], #32                  \n\t"

                  "0:                                       \n\t"
                  // load out_ptr
                  "pld        [%[out_ptr1], #128]           \n\t"
                  "pld        [%[out_ptr1_2], #128]         \n\t"

                  "vld1.f32   {d24, d25}, [%[out_ptr1]]     \n\t"
                  "vld1.f32   {d26, d27}, [%[out_ptr1_2]]   \n\t"

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
                  "vmax.f32   d24, d24, d9                  \n\t"
                  "vmax.f32   d25, d25, d9                  \n\t"

                  "vmax.f32   d26, d26, d9                  \n\t"
                  "vmax.f32   d27, d27, d9                  \n\t"

                  "subs       %[o_w_dim4], #1               \n\t"
                  "vst1.f32   {d24, d25}, [%[out_ptr1]]!    \n\t"
                  "vst1.f32   {d26, d27}, [%[out_ptr1_2]]!  \n\t"

                  // cycle
                  "bne        0b                            \n\t"

                  : [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr2] "+r"(out_ptr2), [out_ptr1_2] "+r"(out_ptr1_2),
                    [out_ptr2_2] "+r"(out_ptr2_2), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3),
                    [in_ptr4] "+r"(in_ptr4)
                  : [f1] "r"(f1), [f1_2] "r"(f1_2), [reluptr] "r"(reluptr)
                  : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                    "q8", "q9", "q10", "q11", "q12", "q13");
            }
          }

#endif  //__aarch64__
#endif  // __ARM_NEON

          // remain output_width
          for (; o_w < output_w; ++o_w) {
            float sum1 = 0;
            float sum1_2 = 0;
#if __ARM_NEON
            float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
            float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
            float32x4_t _pad_filter1_2 = vld1q_f32(pad_filter1_2);
            float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
            float32x4_t _sum1_2 = vmulq_f32(_in_ptr1, _pad_filter1_2);

            float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
            float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
            float32x4_t _pad_filter2_2 = vld1q_f32(pad_filter2_2);
            _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
            _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr2, _pad_filter2_2);

            float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
            float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
            float32x4_t _pad_filter3_2 = vld1q_f32(pad_filter3_2);
            _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
            _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr3, _pad_filter3_2);

            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
            _sum1_2 = vsetq_lane_f32(sum1_2, _sum1_2, 3);

            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ss1_2 =
                vadd_f32(vget_low_f32(_sum1_2), vget_high_f32(_sum1_2));

            float32x2_t _ssss1_ssss1_2 = vpadd_f32(_ss1, _ss1_2);
            sum1 += vget_lane_f32(_ssss1_ssss1_2, 0);
            sum1_2 += vget_lane_f32(_ssss1_ssss1_2, 1);
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

            sum1_2 += in_ptr1[0] * pad_filter1_2[0];
            sum1_2 += in_ptr1[1] * pad_filter1_2[1];
            sum1_2 += in_ptr1[2] * pad_filter1_2[2];
            sum1_2 += in_ptr2[0] * pad_filter2_2[0];
            sum1_2 += in_ptr2[1] * pad_filter2_2[1];
            sum1_2 += in_ptr2[2] * pad_filter2_2[2];
            sum1_2 += in_ptr3[0] * pad_filter3_2[0];
            sum1_2 += in_ptr3[1] * pad_filter3_2[1];
            sum1_2 += in_ptr3[2] * pad_filter3_2[2];
#endif
            if (!if_nopadding &&
                (o_w < padding_w || o_w > output_w - padding_w - 2)) {
              pad_filter0--;
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;

              pad_filter0_2--;
              pad_filter1_2--;
              pad_filter2_2--;
              pad_filter3_2--;
            } else {
              in_ptr1++;
              in_ptr2++;
              in_ptr3++;
              in_ptr4++;
            }
            *out_ptr1 += sum1;
            *out_ptr1_2 += sum1_2;

            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }
            if (out_ptr1_2[0] < relu_value) {
              *out_ptr1_2 = relu_value;
            }
            out_ptr1++;
            out_ptr1_2++;
          }
          out_ptr1 += output_w;
          out_ptr1_2 += output_w;
        }
        filter_data_ch += filter_ch_size;
        filter_data_ch_2 += filter_ch_size;
        input_data_ch += in_ch_size;
      }
    }
    int out_ch_remain = output_ch_d2 * 2;
    // remain output_channel
    for (int o_c = out_ch_remain; o_c < output_ch; ++o_c) {
      std::atomic<float> relu_value{0};
      const float *reluptr;
      bool issamefilter;
      const float *in_ptr1, *in_ptr2, *in_ptr3, *in_ptr4;
      const float *f1;
      const float *pad_filter0, *pad_filter1, *pad_filter2, *pad_filter3;
      float pad_filter_arr[pad_filter_ch_size] = {0};
      float *output_data_ch;
      const float *input_data_ch;
      const float *filter_data_ch;

      input_data_ch = input_data;
      output_data_ch = output_data + o_c * out_ch_size;
      filter_data_ch = filter_data + o_c * filter_ch_size * input_ch;

      for (int i_c = 0; i_c < input_ch; ++i_c) {
        float relu_arr[4];
        reluptr = relu_arr;
        if (if_relu && i_c == input_ch - 1) {
          relu_value = 0;
        } else {
          relu_value = -FLT_MAX;
        }
        relu_arr[0] = relu_value;
        relu_arr[1] = relu_value;
        relu_arr[2] = relu_value;
        relu_arr[3] = relu_value;
        f1 = filter_data_ch;
        if (!if_nopadding) {
          pad_filter_arr[pad_filter_ch_size] = {0};
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

            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }
            if (out_ptr2[0] < relu_value) {
              *out_ptr2 = relu_value;
            }
            out_ptr1++;
            out_ptr2++;
          }
            // valid
#if __ARM_NEON
#if __aarch64__
          if (issamefilter) {
            int o_w_dim4 = (output_w - 2 * padding_w) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(
                  "prfm   pldl1keep, [%[f1], #256]          \n\t"

                  "ld1   {v0.4s, v1.4s}, [%[f1]]            \n\t"
                  "add        %[f1], %[f1], #32             \n\t"
                  "prfm   pldl1keep, [%[reluptr], #64]      \n\t"
                  "ld1   {v16.4s}, [%[reluptr]]             \n\t"

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
                  "fmla   v12.4s, v5.4s, v0.4s[0]           \n\t"

                  "ext    v9.16b, v6.16b, v7.16b, #8        \n\t"
                  "fmla   v14.4s, v7.4s, v4.4s[0]           \n\t"

                  "ext    v10.16b, v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v0.4s[1]           \n\t"

                  "ext    v11.16b, v6.16b, v7.16b, #12      \n\t"
                  "fmla   v14.4s, v9.4s, v1.4s[2]           \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[in_ptr2]]       \n\t"
                  "add        %[in_ptr2],%[in_ptr2], #16    \n\t"

                  "fmla   v12.4s, v10.4s, v0.4s[2]          \n\t"
                  "fmla   v14.4s, v11.4s, v1.4s[3]          \n\t"

                  // in_ptr2 multiply
                  "ext    v8.16b,  v5.16b, v6.16b, #4       \n\t"
                  "fmla   v12.4s, v5.4s, v0.4s[3]           \n\t"
                  "fmla   v14.4s, v5.4s, v0.4s[0]           \n\t"

                  "ext    v9.16b,  v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v1.4s[0]           \n\t"
                  "fmla   v14.4s, v8.4s, v0.4s[1]           \n\t"

                  "ld1   {v6.d}[1], [%[in_ptr3]]            \n\t"
                  "add        %[in_ptr3],%[in_ptr3], #8     \n\t"
                  "ld1   {v7.4s}, [%[in_ptr3]]              \n\t"

                  "add        %[in_ptr3],%[in_ptr3], #8     \n\t"
                  "fmla   v12.4s, v9.4s, v1.4s[1]           \n\t"
                  "fmla   v14.4s, v9.4s, v0.4s[2]           \n\t"

                  // in_ptr3 multiply
                  "ext    v10.16b, v6.16b, v7.16b, #8       \n\t"
                  "fmla   v12.4s, v7.4s, v4.4s[0]           \n\t"
                  "fmla   v14.4s, v7.4s, v1.4s[1]           \n\t"

                  "ext    v11.16b, v6.16b,  v7.16b, #12     \n\t"
                  "fmla   v12.4s, v10.4s, v1.4s[2]          \n\t"
                  "fmla   v14.4s, v10.4s, v0.4s[3]          \n\t"

                  "fmla   v12.4s, v11.4s, v1.4s[3]          \n\t"
                  "fmla   v14.4s, v11.4s, v1.4s[0]          \n\t"

                  // store out_ptr
                  "fmax   v12.4s,v12.4s, v16.4s             \n\t"
                  "fmax   v14.4s,v14.4s, v16.4s             \n\t"

                  "st1   {v12.4s}, [%[out_ptr1]], #16       \n\t"
                  "st1   {v14.4s}, [%[out_ptr2]], #16       \n\t"

                  // cycle
                  "subs       %[o_w_dim4],%[o_w_dim4], #1   \n\t"
                  "bne        0b                            \n\t"

                  : [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr2] "+r"(out_ptr2), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3),
                    [in_ptr4] "+r"(in_ptr4)
                  : [f1] "r"(f1), [reluptr] "r"(reluptr)
                  : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v14", "v16");
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
            int o_w_dim4 = (output_w - 2 * padding_w) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(
                  "pld        [%[f1], #256]                 \n\t"
                  "vld1.32   {d0-d3}, [%[f1]]               \n\t"
                  "add        %[f1], #32                    \n\t"

                  "pld        [%[reluptr], #64]             \n\t"
                  "vld1.32   {d9}, [%[reluptr]]             \n\t"

                  "vld1.32   {d8[0]}, [%[f1]]               \n\t"
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
                  "vmax.f32   d24, d24, d9                  \n\t"
                  "vmax.f32   d25, d25, d9                  \n\t"

                  "vmax.f32   d28, d28, d9                  \n\t"
                  "vmax.f32   d29, d29, d9                  \n\t"

                  "subs       %[o_w_dim4], #1               \n\t"
                  "vst1.f32   {d24, d25}, [%[out_ptr1]]!    \n\t"
                  "vst1.f32   {d28, d29}, [%[out_ptr2]]!    \n\t"

                  // cycle
                  "bne        0b                            \n\t"

                  : [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr2] "+r"(out_ptr2), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3),
                    [in_ptr4] "+r"(in_ptr4)
                  : [f1] "r"(f1), [reluptr] "r"(reluptr)
                  : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                    "q8", "q9", "q10", "q11", "q12", "q14");
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

            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }
            if (out_ptr2[0] < relu_value) {
              *out_ptr2 = relu_value;
            }
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
          int o_w = 0;
          // pad left
          for (; o_w < padding_w; ++o_w) {
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

            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }
            out_ptr1++;
          }
            // valid
#if __ARM_NEON
#if __aarch64__
          if (if_nopadding) {
            int o_w_dim4 = (output_w - 2 * padding_w) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(
                  "prfm   pldl1keep, [%[f1], #256]          \n\t"

                  "ld1   {v0.4s, v1.4s}, [%[f1]]            \n\t"
                  "add        %[f1], %[f1], #32             \n\t"
                  "prfm   pldl1keep, [%[reluptr], #64]      \n\t"
                  "ld1   {v16.4s}, [%[reluptr]]             \n\t"

                  "ld1   {v4.s}[0], [%[f1]]                 \n\t"
                  "sub        %[f1],%[f1], #32              \n\t"

                  "0:                                       \n\t"
                  // load out_ptr
                  "prfm   pldl1keep, [%[out_ptr1], #128]    \n\t"
                  "ld1   {v12.4s}, [%[out_ptr1]]            \n\t"

                  // in_ptr1 multiply
                  "prfm   pldl1keep, [%[in_ptr1], #192]     \n\t"
                  "ld1   {v5.4s, v6.4s}, [%[in_ptr1]]       \n\t"
                  "add        %[in_ptr1],%[in_ptr1], #16    \n\t"

                  "ext    v8.16b, v5.16b, v6.16b, #4        \n\t"
                  "fmla   v12.4s, v5.4s, v0.4s[0]           \n\t"

                  "ext    v10.16b, v5.16b, v6.16b, #8       \n\t"
                  "fmul   v13.4s, v8.4s, v0.4s[1]           \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[in_ptr2]]       \n\t"
                  "add        %[in_ptr2],%[in_ptr2], #16    \n\t"
                  "fmla   v12.4s, v10.4s, v0.4s[2]          \n\t"

                  // in_ptr2 multiply
                  "ext    v8.16b,  v5.16b, v6.16b, #4       \n\t"
                  "fmla   v13.4s, v5.4s, v0.4s[3]           \n\t"

                  "ext    v9.16b,  v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v1.4s[0]           \n\t"

                  "ld1   {v6.d}[1], [%[in_ptr3]]            \n\t"
                  "add        %[in_ptr3],%[in_ptr3], #8     \n\t"
                  "ld1   {v7.4s}, [%[in_ptr3]]              \n\t"
                  "add        %[in_ptr3],%[in_ptr3], #8     \n\t"
                  "fmla   v13.4s, v9.4s, v1.4s[1]           \n\t"

                  // in_ptr3 multiply
                  "ext    v10.16b, v6.16b, v7.16b, #8       \n\t"
                  "fmla   v12.4s, v7.4s, v4.4s[0]           \n\t"

                  "ext    v11.16b, v6.16b,  v7.16b, #12     \n\t"
                  "fmla   v13.4s, v10.4s, v1.4s[2]          \n\t"
                  "fmla   v12.4s, v11.4s, v1.4s[3]          \n\t"

                  // store out_ptr
                  "fadd   v12.4s, v13.4s, v12.4s            \n\t"
                  "fmax   v12.4s, v12.4s, v16.4s            \n\t"
                  "st1   {v12.4s}, [%[out_ptr1]], #16       \n\t"

                  // cycle
                  "subs       %[o_w_dim4],%[o_w_dim4], #1     \n\t"
                  "bne        0b                            \n\t"

                  : [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                    [in_ptr1] "+r"(in_ptr1), [in_ptr2] "+r"(in_ptr2),
                    [in_ptr3] "+r"(in_ptr3)
                  : [f1] "r"(f1), [reluptr] "r"(reluptr)
                  : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v16");
            }
          }
#else
          if (if_nopadding) {
            int o_w_dim4 = (output_w - 2 * padding_w) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(
                  "pld        [%[f1], #256]                 \n\t"
                  "vld1.32   {d0-d3}, [%[f1]]               \n\t"
                  "add        %[f1], #32                    \n\t"

                  "pld        [%[reluptr], #64]             \n\t"
                  "vld1.32   {q2}, [%[reluptr]]             \n\t"

                  "vld1.32   {d8[0]}, [%[f1]]               \n\t"
                  "sub        %[f1], #32                    \n\t"

                  "0:                                       \n\t"
                  // load out_ptr
                  "pld        [%[out_ptr1], #128]           \n\t"
                  "vld1.f32   {d24, d25}, [%[out_ptr1]]     \n\t"

                  // in_ptr1 multiply
                  "pld        [%[in_ptr1], #128]            \n\t"
                  "vld1.f32   {d10-d12}, [%[in_ptr1]]       \n\t"
                  "add        %[in_ptr1], #16               \n\t"

                  "vext.32    q8, q5, q6, #1                \n\t"
                  "vmla.f32   q12, q5, d0[0]                \n\t"

                  "vext.32    q9, q6, q7, #2                \n\t"
                  "vext.32    q10, q5, q6, #2               \n\t"
                  "vmul.f32   q13, q8, d0[1]                \n\t"
                  "pld        [%[in_ptr2], #128]            \n\t"

                  "vext.32    q11, q6, q7, #3               \n\t"
                  "vld1.f32   {d10-d12}, [%[in_ptr2]]       \n\t"
                  "add        %[in_ptr2], #16               \n\t"
                  "vmla.f32   q12, q10, d1[0]               \n\t"

                  // in_ptr2 multiply
                  "vext.32    q8, q5, q6, #1                \n\t"
                  "vmla.f32   q13, q5, d1[1]                \n\t"

                  "pld        [%[in_ptr3], #128]            \n\t"
                  "vext.32    q9, q5, q6, #2                \n\t"
                  "vmla.f32   q12, q8, d2[0]                \n\t"

                  "vld1.f32   {d13-d15}, [%[in_ptr3]]       \n\t"
                  "add        %[in_ptr3], #16               \n\t"
                  "vmla.f32   q13, q9, d2[1]                \n\t"

                  // in_ptr3 multiply
                  "vext.32    q10, q6, q7, #2               \n\t"
                  "vmla.f32   q12, q7, d8[0]                \n\t"

                  "vext.32    q11, q6, q7, #3               \n\t"
                  "vmla.f32   q13, q10, d3[0]               \n\t"

                  "vmla.f32   q12, q11, d3[1]               \n\t"

                  // store out_ptr
                  "vadd.f32   q12, q12, q13                 \n\t"
                  "vmax.f32   q12, q12, q2                  \n\t"
                  "vst1.f32   {d24, d25}, [%[out_ptr1]]!    \n\t"

                  // cycle
                  "subs       %[o_w_dim4], #1               \n\t"
                  "bne        0b                            \n\t"

                  : [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr2] "+r"(out_ptr2), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3),
                    [in_ptr4] "+r"(in_ptr4)
                  : [f1] "r"(f1), [reluptr] "r"(reluptr)
                  : "memory", "q0", "q1", "q2", "q4", "q5", "q6", "q7", "q8",
                    "q9", "q10", "q11", "q12", "q13");
            }
          }
#endif  //__aarch64__
#endif  // __ARM_NEON
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

            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }
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

void SlidingwindowConv3x3s2(const framework::Tensor *input,
                            const framework::Tensor *filter,
                            const std::vector<int> &paddings,
                            framework::Tensor *output, framework::Tensor *bias,
                            bool if_bias, bool if_relu) {
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
  const int filter_ch_size = 9;
  const int pad_filter_ch_size = (2 * padding_h + 3) * (2 * padding_w + 3);
  const int pad_filter_start =
      2 * padding_h * (2 * padding_w + 3) + 2 * padding_w;
  const int pad_filter_w = 3 + padding_w * 2;
  const float *bias_data;
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

  if (if_bias) {
    bias_data = bias->data<float>();
  }
#if __ARM_NEON
  float *out_ptr = output_data;
  for (int i = 0; i < output_ch; ++i) {
    float32x4_t _bias;
    if (if_bias) {
      _bias = vld1q_dup_f32(bias_data + i);
    } else {
      _bias = vdupq_n_f32(0.0);
    }
    int dim4 = out_ch_size >> 2;
    int lef4 = out_ch_size & 3;

    for (int j = 0; j < dim4; ++j) {
      vst1q_f32(out_ptr, _bias);
      out_ptr += 4;
    }
    if (lef4 != 0) {
      if (lef4 >= 1) {
        vst1q_lane_f32(out_ptr, _bias, 0);
        out_ptr++;
      }
      if (lef4 >= 2) {
        vst1q_lane_f32(out_ptr, _bias, 1);
        out_ptr++;
      }
      if (lef4 >= 3) {
        vst1q_lane_f32(out_ptr, _bias, 2);
        out_ptr++;
      }
    }
  }
#else
  int k = 0;
#pragma omp parallel for
  for (int i = 0; i < output_ch; ++i) {
    for (int j = 0; j < out_ch_size; ++j) {
      if (if_bias) {
        output_data[k++] = bias_data[i];
      } else {
        output_data[k++] = 0;
      }
    }
  }
#endif
  if (padding_h == 0 && padding_w == 0) {
    if_nopadding = true;
    valid_w_start = 0;
    valid_h_start = 0;
    valid_w_end = output_w;
    valid_h_end = output_h;
  }
  for (int b = 0; b < batch; ++b) {
    int output_ch_d2 = output_ch >> 1;

#pragma omp parallel for
    for (int o_c2 = 0; o_c2 < output_ch_d2; ++o_c2) {
      std::atomic<float> relu_value{0};
      const float *reluptr;
      int o_c = o_c2 * 2;

      const float *f1, *f9;
      const float *f1_2, *f9_2;
      const float *in_ptr1, *in_ptr2, *in_ptr3;
      const float *pad_filter1, *pad_filter2, *pad_filter3;
      const float *pad_filter1_2, *pad_filter2_2, *pad_filter3_2;
      float pad_filter_arr[pad_filter_ch_size] = {0};
      float pad_filter_arr_2[pad_filter_ch_size] = {0};

      float *output_data_ch;
      float *output_data_ch_2;
      const float *input_data_ch;
      const float *filter_data_ch;
      const float *filter_data_ch_2;

      filter_data_ch = filter_data + o_c * filter_ch_size * input_ch;
      filter_data_ch_2 = filter_data + (o_c + 1) * filter_ch_size * input_ch;

      input_data_ch = input_data;
      output_data_ch = output_data + o_c * out_ch_size;
      output_data_ch_2 = output_data + (o_c + 1) * out_ch_size;

      for (int i_c = 0; i_c < input_ch; ++i_c) {
        float relu_arr[4];
        reluptr = relu_arr;
        if (if_relu && i_c == input_ch - 1) {
          relu_value = 0;
        } else {
          relu_value = -FLT_MAX;
        }
        relu_arr[0] = relu_value;
        relu_arr[1] = relu_value;
        relu_arr[2] = relu_value;
        relu_arr[3] = relu_value;

        f1 = filter_data_ch;
        f9 = filter_data_ch + 8;
        f1_2 = filter_data_ch_2;
        f9_2 = filter_data_ch_2 + 8;

        if (!if_nopadding) {
          pad_filter_arr[pad_filter_ch_size] = {0};
          for (int i = 0; i < 9; ++i) {
            int j = i / 3 * (2 * padding_w + 3) + i % 3 + padding_h * 3 +
                    padding_w * (2 * padding_h + 1);
            pad_filter_arr[j] = filter_data_ch[i];
            pad_filter_arr_2[j] = filter_data_ch_2[i];
          }
          pad_filter1 = pad_filter_arr;
          pad_filter1 += pad_filter_start;
          pad_filter2 = pad_filter1 + pad_filter_w;
          pad_filter3 = pad_filter2 + pad_filter_w;

          pad_filter1_2 = pad_filter_arr_2;
          pad_filter1_2 += pad_filter_start;
          pad_filter2_2 = pad_filter1_2 + pad_filter_w;
          pad_filter3_2 = pad_filter2_2 + pad_filter_w;
        } else {
          pad_filter1 = filter_data_ch;
          pad_filter2 = pad_filter1 + 3;
          pad_filter3 = pad_filter2 + 3;

          pad_filter1_2 = filter_data_ch_2;
          pad_filter2_2 = pad_filter1_2 + 3;
          pad_filter3_2 = pad_filter2_2 + 3;
        }

        float *out_ptr1;
        float *out_ptr1_2;
        out_ptr1 = output_data_ch;
        out_ptr1_2 = output_data_ch_2;

        in_ptr1 = input_data_ch;
        in_ptr2 = in_ptr1 + input_w;
        in_ptr3 = in_ptr2 + input_w;

        int o_h = 0;
        for (; o_h < output_h; ++o_h) {
          int o_w = 0;

          // pad left
          for (; o_w <= valid_w_start; ++o_w) {
            float sum1 = 0;
            float sum1_2 = 0;
#if __ARM_NEON
            float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
            float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
            float32x4_t _pad_filter1_2 = vld1q_f32(pad_filter1_2);
            float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
            float32x4_t _sum1_2 = vmulq_f32(_in_ptr1, _pad_filter1_2);

            float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
            float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
            float32x4_t _pad_filter2_2 = vld1q_f32(pad_filter2_2);
            _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
            _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr2, _pad_filter2_2);

            float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
            float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
            float32x4_t _pad_filter3_2 = vld1q_f32(pad_filter3_2);
            _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
            _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr3, _pad_filter3_2);

            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
            _sum1_2 = vsetq_lane_f32(sum1_2, _sum1_2, 3);

            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ss1_2 =
                vadd_f32(vget_low_f32(_sum1_2), vget_high_f32(_sum1_2));
            float32x2_t _ssss1_ssss1_2 = vpadd_f32(_ss1, _ss1_2);

            sum1 += vget_lane_f32(_ssss1_ssss1_2, 0);
            sum1_2 += vget_lane_f32(_ssss1_ssss1_2, 1);
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

            sum1_2 += in_ptr1[0] * pad_filter1_2[0];
            sum1_2 += in_ptr1[1] * pad_filter1_2[1];
            sum1_2 += in_ptr1[2] * pad_filter1_2[2];
            sum1_2 += in_ptr2[0] * pad_filter2_2[0];
            sum1_2 += in_ptr2[1] * pad_filter2_2[1];
            sum1_2 += in_ptr2[2] * pad_filter2_2[2];
            sum1_2 += in_ptr3[0] * pad_filter3_2[0];
            sum1_2 += in_ptr3[1] * pad_filter3_2[1];
            sum1_2 += in_ptr3[2] * pad_filter3_2[2];
#endif
            if (if_nopadding) {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;

            } else if (if_odd_pad_w && o_w == valid_w_start ||
                       o_w == valid_w_end && if_odd_pad_w && if_exact_in_w ||
                       o_w == valid_w_end + 1 && !if_odd_pad_w &&
                           !if_exact_in_w) {
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;
              pad_filter1_2--;
              pad_filter2_2--;
              pad_filter3_2--;

              in_ptr1++;
              in_ptr2++;
              in_ptr3++;

            } else if (o_w < valid_w_start || o_w > valid_w_end) {
              pad_filter1 -= 2;
              pad_filter2 -= 2;
              pad_filter3 -= 2;
              pad_filter1_2 -= 2;
              pad_filter2_2 -= 2;
              pad_filter3_2 -= 2;

            } else {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;
            }
            *out_ptr1 += sum1;
            *out_ptr1_2 += sum1_2;

            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }

            if (out_ptr1_2[0] < relu_value) {
              *out_ptr1_2 = relu_value;
            }
            out_ptr1++;
            out_ptr1_2++;
          }
            // valid
#if __ARM_NEON
#if __aarch64__
          if (o_h > valid_h_start && o_h <= valid_h_end) {
            int o_w_dim4 = (valid_w_end - valid_w_start - 1) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(
                  "prfm   pldl1keep, [%[f1], #256]          \n\t"
                  "prfm   pldl1keep, [%[f9], #256]          \n\t"
                  "prfm   pldl1keep, [%[f1_2], #256]        \n\t"
                  "prfm   pldl1keep, [%[f9_2], #256]        \n\t"
                  "prfm   pldl1keep, [%[reluptr], #256]     \n\t"

                  "ld1   {v0.4s, v1.4s}, [%[f1]]            \n\t"
                  "ld1   {v4.s}[0], [%[f9]]                 \n\t"
                  "ld1   {v2.4s, v3.4s}, [%[f1_2]]          \n\t"
                  "ld1   {v4.s}[1], [%[f9_2]]               \n\t"
                  "ld1   {v16.4s}, [%[reluptr]]             \n\t"

                  "prfm   pldl1keep, [%[in_ptr1], #256]     \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[in_ptr1]], #32  \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[in_ptr1]]       \n\t"

                  "0:                                       \n\t"
                  // load out_ptr
                  "prfm    pldl1keep, [%[out_ptr1], #128]   \n\t"
                  "prfm    pldl1keep, [%[out_ptr1_2], #128] \n\t"

                  "ld1   {v12.4s}, [%[out_ptr1]]            \n\t"
                  "ld1   {v14.4s}, [%[out_ptr1_2]]          \n\t"

                  // in_ptr1

                  "fmla   v12.4s, v5.4s, v0.4s[0]           \n\t"
                  "fmla   v14.4s, v5.4s, v2.4s[0]           \n\t"

                  "ext    v8.16b, v5.16b, v7.16b, #4        \n\t"
                  "fmul   v13.4s, v6.4s, v0.4s[1]           \n\t"
                  "fmul   v15.4s, v6.4s, v2.4s[1]           \n\t"

                  "ld2   {v5.4s, v6.4s}, [%[in_ptr2]], #32  \n\t"
                  "fmla   v12.4s, v8.4s, v0.4s[2]           \n\t"
                  "fmla   v14.4s, v8.4s, v2.4s[2]           \n\t"

                  "ld2   {v7.4s, v8.4s}, [%[in_ptr2]]       \n\t"
                  // in_ptr2

                  "fmla   v13.4s, v5.4s, v0.4s[3]           \n\t"
                  "fmla   v15.4s, v5.4s, v2.4s[3]           \n\t"

                  "ext    v8.16b, v5.16b, v7.16b, #4        \n\t"
                  "fmla   v12.4s, v6.4s, v1.4s[0]           \n\t"
                  "fmla   v14.4s, v6.4s, v3.4s[0]           \n\t"

                  "ld2   {v5.4s, v6.4s}, [%[in_ptr3]], #32  \n\t"
                  "fmla   v13.4s, v8.4s, v1.4s[1]           \n\t"
                  "fmla   v15.4s, v8.4s, v3.4s[1]           \n\t"

                  // in_ptr3
                  "ld2   {v7.4s, v8.4s}, [%[in_ptr3]]       \n\t"
                  "fmla   v12.4s, v5.4s, v1.4s[2]           \n\t"
                  "fmla   v14.4s, v5.4s, v3.4s[2]           \n\t"

                  "ext    v8.16b, v5.16b, v7.16b, #4        \n\t"
                  "fmla   v13.4s, v6.4s, v1.4s[3]           \n\t"
                  "fmla   v15.4s, v6.4s, v3.4s[3]           \n\t"

                  "fmla   v12.4s, v8.4s, v4.4s[0]           \n\t"
                  "fmla   v14.4s, v8.4s, v4.4s[1]           \n\t"

                  // store
                  "prfm   pldl1keep, [%[in_ptr1], #256]     \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[in_ptr1]], #32  \n\t"
                  "fadd   v12.4s, v12.4s, v13.4s            \n\t"
                  "fadd   v14.4s, v14.4s, v15.4s            \n\t"

                  "ld2   {v7.4s, v8.4s}, [%[in_ptr1]]       \n\t"
                  "fmax   v12.4s, v12.4s, v16.4s            \n\t"
                  "fmax   v14.4s, v14.4s, v16.4s            \n\t"

                  "subs       %[o_w_dim4], %[o_w_dim4], #1  \n\t"
                  "st1   {v12.4s}, [%[out_ptr1]], #16       \n\t"
                  "st1   {v14.4s}, [%[out_ptr1_2]], #16     \n\t"

                  // cycle
                  "bne        0b                            \n\t"
                  "sub        %[in_ptr1], %[in_ptr1], #32   \n\t"

                  : [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr1_2] "+r"(out_ptr1_2), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3)
                  : [f1] "r"(f1), [f1_2] "r"(f1_2), [f9] "r"(f9),
                    [f9_2] "r"(f9_2), [reluptr] "r"(reluptr)
                  : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v12", "v13", "v14", "v15", "v16");
            }
          }

#else
          if (o_h > valid_h_start && o_h <= valid_h_end) {
            int o_w_dim4 = (valid_w_end - valid_w_start - 1) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(
                  "pld        [%[f1], #256]                 \n\t"
                  "pld        [%[f9], #256]                 \n\t"
                  "pld        [%[f1_2], #256]               \n\t"
                  "pld        [%[f9_2], #256]               \n\t"
                  "pld        [%[reluptr], #128]            \n\t"

                  "vld1.32   {d0-d3}, [%[f1]]               \n\t"
                  "vld1.32   {d8[0]}, [%[f9]]               \n\t"
                  "vld1.32   {d4-d7}, [%[f1_2]]             \n\t"
                  "vld1.32   {d8[1]}, [%[f9_2]]             \n\t"
                  "vld1.32   {d18, d19}, [%[reluptr]]       \n\t"

                  "vld2.f32   {d10-d13}, [%[in_ptr1]]!      \n\t"
                  "vld2.f32   {d14, d15}, [%[in_ptr1]]      \n\t"

                  "0:                                       \n\t"
                  // load out_ptr
                  "pld        [%[out_ptr1], #128]           \n\t"
                  "pld        [%[out_ptr1_2], #128]         \n\t"
                  "vld1.f32   {d24, d25}, [%[out_ptr1]]     \n\t"
                  "vld1.f32   {d28, d29}, [%[out_ptr1_2]]   \n\t"

                  // in_ptr1 multiply
                  "vmla.f32   q12, q5, d0[0]                \n\t"
                  "vmla.f32   q14, q5, d4[0]                \n\t"

                  "vext.32    q8, q5, q7, #1                \n\t"
                  "vmul.f32   q13, q6, d0[1]                \n\t"
                  "vmul.f32   q15, q6, d4[1]                \n\t"

                  "vld2.f32   {d10-d13}, [%[in_ptr2]]!      \n\t"
                  "vmla.f32   q12, q8, d1[0]                \n\t"
                  "vld2.f32   {d14, d15}, [%[in_ptr2]]      \n\t"
                  "vmla.f32   q14, q8, d5[0]                \n\t"

                  // in_ptr2 multiply
                  "vmla.f32   q13, q5, d1[1]                \n\t"
                  "vmla.f32   q15, q5, d5[1]                \n\t"

                  "vext.32    q8, q5, q7, #1                \n\t"
                  "vmla.f32   q12, q6, d2[0]                \n\t"
                  "vmla.f32   q14, q6, d6[0]                \n\t"

                  "vld2.f32   {d10-d13}, [%[in_ptr3]]!      \n\t"
                  "vmla.f32   q13, q8, d2[1]                \n\t"
                  "vld2.f32   {d14, d15}, [%[in_ptr3]]      \n\t"
                  "vmla.f32   q15, q8, d6[1]                \n\t"

                  // in_ptr3 multiply
                  "vmla.f32   q12, q5, d3[0]                \n\t"
                  "vmla.f32   q14, q5, d7[0]                \n\t"

                  "vext.32    q8, q5, q7, #1                \n\t"
                  "vmla.f32   q13, q6, d3[1]                \n\t"
                  "vmla.f32   q15, q6, d7[1]                \n\t"

                  "vmla.f32   q12, q8, d8[0]                \n\t"
                  "vmla.f32   q14, q8, d8[1]                \n\t"

                  // store
                  "vld2.f32   {d10-d13}, [%[in_ptr1]]!      \n\t"
                  "vadd.f32   q12, q12, q13                 \n\t"
                  "vadd.f32   q14, q14, q15                 \n\t"
                  "vmax.f32   q12, q12, q9                  \n\t"
                  "vmax.f32   q14, q14, q9                  \n\t"

                  "vld2.f32   {d14, d15}, [%[in_ptr1]]      \n\t"
                  "vst1.f32   {d24, d25}, [%[out_ptr1]]!    \n\t"
                  "vst1.f32   {d28, d29}, [%[out_ptr1_2]]!  \n\t"

                  // cycle
                  "subs       %[o_w_dim4], #1               \n\t"
                  "bne        0b                            \n\t"
                  "sub       %[in_ptr1], %[in_ptr1], #32    \n\t"

                  : [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                    [out_ptr1_2] "+r"(out_ptr1_2), [in_ptr1] "+r"(in_ptr1),
                    [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3)
                  : [f1] "r"(f1), [f1_2] "r"(f1_2), [f9] "r"(f9),
                    [f9_2] "r"(f9_2), [reluptr] "r"(reluptr)
                  : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                    "q8", "q9", "q12", "q13", "q14", "q15");
            }
          }
#endif  //__aarch64__
#endif  // __ARM_NEON
        // remain output_width
          for (; o_w < output_w; ++o_w) {
            float sum1 = 0;
            float sum1_2 = 0;
#if __ARM_NEON
            float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
            float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
            float32x4_t _pad_filter1_2 = vld1q_f32(pad_filter1_2);
            float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
            float32x4_t _sum1_2 = vmulq_f32(_in_ptr1, _pad_filter1_2);

            float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
            float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
            float32x4_t _pad_filter2_2 = vld1q_f32(pad_filter2_2);
            _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
            _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr2, _pad_filter2_2);

            float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
            float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
            float32x4_t _pad_filter3_2 = vld1q_f32(pad_filter3_2);
            _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
            _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr3, _pad_filter3_2);

            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
            _sum1_2 = vsetq_lane_f32(sum1_2, _sum1_2, 3);

            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ss1_2 =
                vadd_f32(vget_low_f32(_sum1_2), vget_high_f32(_sum1_2));
            float32x2_t _ssss1_ssss1_2 = vpadd_f32(_ss1, _ss1_2);

            sum1 += vget_lane_f32(_ssss1_ssss1_2, 0);
            sum1_2 += vget_lane_f32(_ssss1_ssss1_2, 1);
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

            sum1_2 += in_ptr1[0] * pad_filter1_2[0];
            sum1_2 += in_ptr1[1] * pad_filter1_2[1];
            sum1_2 += in_ptr1[2] * pad_filter1_2[2];
            sum1_2 += in_ptr2[0] * pad_filter2_2[0];
            sum1_2 += in_ptr2[1] * pad_filter2_2[1];
            sum1_2 += in_ptr2[2] * pad_filter2_2[2];
            sum1_2 += in_ptr3[0] * pad_filter3_2[0];
            sum1_2 += in_ptr3[1] * pad_filter3_2[1];
            sum1_2 += in_ptr3[2] * pad_filter3_2[2];
#endif
            if (if_nopadding) {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;

            } else if (if_odd_pad_w && o_w == valid_w_start ||
                       o_w == valid_w_end && if_odd_pad_w && if_exact_in_w ||
                       o_w == valid_w_end + 1 && !if_odd_pad_w &&
                           !if_exact_in_w) {
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;

              pad_filter1_2--;
              pad_filter2_2--;
              pad_filter3_2--;

              in_ptr1++;
              in_ptr2++;
              in_ptr3++;

            } else if (o_w < valid_w_start || o_w > valid_w_end) {
              pad_filter1 -= 2;
              pad_filter2 -= 2;
              pad_filter3 -= 2;
              pad_filter1_2 -= 2;
              pad_filter2_2 -= 2;
              pad_filter3_2 -= 2;
            } else {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;
            }
            *out_ptr1 += sum1;
            *out_ptr1_2 += sum1_2;

            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }

            if (out_ptr1_2[0] < relu_value) {
              *out_ptr1_2 = relu_value;
            }
            out_ptr1++;
            out_ptr1_2++;
          }
          if (if_nopadding) {
            in_ptr1 += remain_stride_w + input_w;
            in_ptr2 += remain_stride_w + input_w;
            in_ptr3 += remain_stride_w + input_w;

          } else if (if_odd_pad_h && o_h == valid_h_start ||
                     o_h == valid_h_end && if_odd_pad_h && if_exact_in_h ||
                     o_h == valid_h_end + 1 && !if_odd_pad_h &&
                         !if_exact_in_h) {
            in_ptr1 += 3;
            in_ptr2 += 3;
            in_ptr3 += 3;

            pad_filter1 -= remain_stride_w;
            pad_filter2 -= remain_stride_w;
            pad_filter3 -= remain_stride_w;

            pad_filter1_2 -= remain_stride_w;
            pad_filter2_2 -= remain_stride_w;
            pad_filter3_2 -= remain_stride_w;
          } else if (o_h < valid_h_start || o_h > valid_h_end) {
            in_ptr1 -= input_w - 3;
            in_ptr2 -= input_w - 3;
            in_ptr3 -= input_w - 3;

            pad_filter1 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3 -= 3 + 2 * padding_w + remain_stride_w;

            pad_filter1_2 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2_2 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3_2 -= 3 + 2 * padding_w + remain_stride_w;
          } else {
            pad_filter1 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3 += 3 + 2 * padding_w - remain_stride_w;

            pad_filter1_2 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2_2 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3_2 += 3 + 2 * padding_w - remain_stride_w;

            in_ptr1 += input_w + 3;
            in_ptr2 += input_w + 3;
            in_ptr3 += input_w + 3;
          }
        }
        filter_data_ch += filter_ch_size;
        filter_data_ch_2 += filter_ch_size;
        input_data_ch += in_ch_size;
      }
    }

    int out_ch_remain = output_ch_d2 * 2;
    // remain channel
    for (int o_c = out_ch_remain; o_c < output_ch; ++o_c) {
      std::atomic<float> relu_value{0};
      const float *reluptr;
      const float *f1, *f9;
      const float *in_ptr1, *in_ptr2, *in_ptr3;
      const float *pad_filter1, *pad_filter2, *pad_filter3;
      float pad_filter_arr[pad_filter_ch_size] = {0};

      float *output_data_ch;
      const float *input_data_ch;
      const float *filter_data_ch;

      filter_data_ch = filter_data + o_c * filter_ch_size * input_ch;
      input_data_ch = input_data;
      output_data_ch = output_data + o_c * out_ch_size;

      for (int i_c = 0; i_c < input_ch; ++i_c) {
        float relu_arr[4];
        reluptr = relu_arr;
        if (if_relu && i_c == input_ch - 1) {
          relu_value = 0;
        } else {
          relu_value = -FLT_MAX;
        }
        relu_arr[0] = relu_value;
        relu_arr[1] = relu_value;
        relu_arr[2] = relu_value;
        relu_arr[3] = relu_value;
        f1 = filter_data_ch;
        f9 = filter_data_ch + 8;

        if (!if_nopadding) {
          pad_filter_arr[pad_filter_ch_size] = {0};
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

            } else if (if_odd_pad_w && o_w == valid_w_start ||
                       o_w == valid_w_end && if_odd_pad_w && if_exact_in_w ||
                       o_w == valid_w_end + 1 && !if_odd_pad_w &&
                           !if_exact_in_w) {
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;

              in_ptr1++;
              in_ptr2++;
              in_ptr3++;

            } else if (o_w < valid_w_start || o_w > valid_w_end) {
              pad_filter1 -= 2;
              pad_filter2 -= 2;
              pad_filter3 -= 2;

            } else {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;
            }
            *out_ptr1 += sum1;

            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }
            out_ptr1++;
          }

            // valid
#if __ARM_NEON
#if __aarch64__
          if (o_h > valid_h_start && o_h <= valid_h_end) {
            int o_w_dim4 = (valid_w_end - valid_w_start - 1) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(
                  "prfm   pldl1keep, [%[f1], #256]          \n\t"
                  "prfm   pldl1keep, [%[f9], #256]          \n\t"
                  "prfm   pldl1keep, [%[reluptr], #256]     \n\t"

                  "ld1   {v0.4s, v1.4s}, [%[f1]]            \n\t"
                  "ld1   {v4.s}[0], [%[f9]]                 \n\t"
                  "ld1   {v16.4s}, [%[reluptr]]             \n\t"

                  "0:                                       \n\t"
                  // load out_ptr
                  "prfm   pldl1keep, [%[out_ptr1], #128]    \n\t"
                  "ld1   {v12.4s}, [%[out_ptr1]]            \n\t"

                  // in_ptr1 multiply
                  "prfm   pldl1keep, [%[in_ptr1], #256]     \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[in_ptr1]], #32  \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[in_ptr1]]       \n\t"

                  "fmla   v12.4s, v5.4s, v0.4s[0]           \n\t"
                  "fmla   v14.4s, v5.4s, v2.4s[0]           \n\t"

                  "ext    v8.16b, v5.16b, v7.16b, #4        \n\t"
                  "fmul   v13.4s, v6.4s, v0.4s[1]           \n\t"

                  "fmla   v12.4s, v8.4s, v0.4s[2]           \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[in_ptr2]], #32  \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[in_ptr2]]       \n\t"

                  // in_ptr2 multiply
                  "fmla   v13.4s, v5.4s, v0.4s[3]           \n\t"
                  "ext    v8.16b, v5.16b, v7.16b, #4        \n\t"
                  "fmla   v12.4s, v6.4s, v1.4s[0]           \n\t"

                  "fmla   v13.4s, v8.4s, v1.4s[1]           \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[in_ptr3]], #32  \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[in_ptr3]]       \n\t"

                  // in_ptr3 multiply
                  "fmla   v12.4s, v5.4s, v1.4s[2]           \n\t"
                  "ext    v8.16b, v5.16b, v7.16b, #4        \n\t"
                  "fmla   v13.4s, v6.4s, v1.4s[3]           \n\t"
                  "fmla   v12.4s, v8.4s, v4.4s[0]           \n\t"

                  // store out_ptr
                  "fadd   v12.4s, v12.4s, v13.4s            \n\t"
                  "fmax   v12.4s, v12.4s, v16.4s            \n\t"
                  "st1   {v12.4s}, [%[out_ptr1]], #16       \n\t"

                  // cycle
                  "subs       %[o_w_dim4], %[o_w_dim4], #1  \n\t"
                  "bne        0b                            \n\t"

                  : [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                    [in_ptr1] "+r"(in_ptr1), [in_ptr2] "+r"(in_ptr2),
                    [in_ptr3] "+r"(in_ptr3)
                  : [f1] "r"(f1), [f9] "r"(f9), [reluptr] "r"(reluptr)
                  : "memory", "v0", "v1", "v4", "v5", "v6", "v7", "v8", "v9",
                    "v12", "v13", "v16");
            }
          }
#else
          if (o_h > valid_h_start && o_h <= valid_h_end) {
            int o_w_dim4 = (valid_w_end - valid_w_start - 1) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(
                  "pld        [%[f1], #256]                 \n\t"
                  "pld        [%[f9], #256]                 \n\t"
                  "pld        [%[reluptr], #128]            \n\t"

                  "vld1.32   {d0-d3}, [%[f1]]               \n\t"
                  "vld1.32   {d8[0]}, [%[f9]]               \n\t"
                  "vld1.32   {d18, d19}, [%[reluptr]]       \n\t"

                  "0:                                       \n\t"
                  // load out_ptr
                  "pld        [%[out_ptr1], #128]           \n\t"
                  "vld1.f32   {d24, d25}, [%[out_ptr1]]     \n\t"

                  // in_ptr1 multiply
                  "vld2.f32   {d10-d13}, [%[in_ptr1]]!      \n\t"
                  "vld2.f32   {d14, d15}, [%[in_ptr1]]      \n\t"

                  "vmla.f32   q12, q5, d0[0]                \n\t"
                  "vext.32    q8, q5, q7, #1                \n\t"
                  "vmul.f32   q13, q6, d0[1]                \n\t"

                  "vld2.f32   {d10-d13}, [%[in_ptr2]]!      \n\t"
                  "vld2.f32   {d14, d15}, [%[in_ptr2]]      \n\t"
                  "vmla.f32   q12, q8, d1[0]                \n\t"

                  // in_ptr2 multiply
                  "vmla.f32   q13, q5, d1[1]                \n\t"
                  "vext.32    q8, q5, q7, #1                \n\t"
                  "vmla.f32   q12, q6, d2[0]                \n\t"

                  "vld2.f32   {d10-d13}, [%[in_ptr3]]!      \n\t"
                  "vld2.f32   {d14, d15}, [%[in_ptr3]]      \n\t"
                  "vmla.f32   q13, q8, d2[1]                \n\t"

                  // in_ptr3 multiply
                  "vmla.f32   q12, q5, d3[0]                \n\t"
                  "vext.32    q8, q5, q7, #1                \n\t"
                  "vmla.f32   q13, q6, d3[1]                \n\t"
                  "vmla.f32   q12, q8, d8[0]                \n\t"

                  // store out_ptr
                  "vadd.f32   q12, q12, q13                 \n\t"
                  "vmax.f32   q12, q12, q9                  \n\t"
                  "vst1.f32   {d24, d25}, [%[out_ptr1]]!    \n\t"

                  // cycle
                  "subs       %[o_w_dim4], #1               \n\t"
                  "bne        0b                            \n\t"

                  : [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                    [in_ptr1] "+r"(in_ptr1), [in_ptr2] "+r"(in_ptr2),
                    [in_ptr3] "+r"(in_ptr3)
                  : [f1] "r"(f1), [f9] "r"(f9), [reluptr] "r"(reluptr)
                  : "memory", "q0", "q1", "q4", "q5", "q6", "q7", "q8", "q9",
                    "q12", "q13");
            }
          }
#endif  //__aarch64__
#endif  // __ARM_NEON
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

            } else if (if_odd_pad_w && o_w == valid_w_start ||
                       o_w == valid_w_end && if_odd_pad_w && if_exact_in_w ||
                       o_w == valid_w_end + 1 && !if_odd_pad_w &&
                           !if_exact_in_w) {
              pad_filter1--;
              pad_filter2--;
              pad_filter3--;

              in_ptr1++;
              in_ptr2++;
              in_ptr3++;

            } else if (o_w < valid_w_start || o_w > valid_w_end) {
              pad_filter1 -= 2;
              pad_filter2 -= 2;
              pad_filter3 -= 2;

            } else {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;
            }
            *out_ptr1 += sum1;

            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }
            out_ptr1++;
          }
          if (if_nopadding) {
            in_ptr1 += remain_stride_w + input_w;
            in_ptr2 += remain_stride_w + input_w;
            in_ptr3 += remain_stride_w + input_w;
          } else if (if_odd_pad_h && o_h == valid_h_start ||
                     o_h == valid_h_end && if_odd_pad_h && if_exact_in_h ||
                     o_h == valid_h_end + 1 && !if_odd_pad_h &&
                         !if_exact_in_h) {
            in_ptr1 += 3;
            in_ptr2 += 3;
            in_ptr3 += 3;

            pad_filter1 -= remain_stride_w;
            pad_filter2 -= remain_stride_w;
            pad_filter3 -= remain_stride_w;

          } else if (o_h < valid_h_start || o_h > valid_h_end) {
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

void SlidingwindowConv3x3s2_8channel(const framework::Tensor *input,
                                     const framework::Tensor *filter,
                                     const std::vector<int> &paddings,
                                     framework::Tensor *output,
                                     framework::Tensor *bias, bool if_bias,
                                     bool if_relu) {
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
  const int filter_ch_size = 9;
  const int pad_filter_ch_size = (2 * padding_h + 3) * (2 * padding_w + 3);
  const int pad_filter_start =
      2 * padding_h * (2 * padding_w + 3) + 2 * padding_w;
  const int pad_filter_w = 3 + padding_w * 2;

  const float *bias_data;
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

  if (if_bias) {
    bias_data = bias->data<float>();
  }
#if __ARM_NEON
  float *out_ptr = output_data;
  for (int i = 0; i < output_ch; ++i) {
    float32x4_t _bias;
    if (if_bias) {
      _bias = vld1q_dup_f32(bias_data + i);
    } else {
      _bias = vdupq_n_f32(0.0);
    }
    int dim4 = out_ch_size >> 2;
    int lef4 = out_ch_size & 3;

    for (int j = 0; j < dim4; ++j) {
      vst1q_f32(out_ptr, _bias);
      out_ptr += 4;
    }
    if (lef4 != 0) {
      if (lef4 >= 1) {
        vst1q_lane_f32(out_ptr, _bias, 0);
        out_ptr++;
      }
      if (lef4 >= 2) {
        vst1q_lane_f32(out_ptr, _bias, 1);
        out_ptr++;
      }
      if (lef4 >= 3) {
        vst1q_lane_f32(out_ptr, _bias, 2);
        out_ptr++;
      }
    }
  }
#else
  int k = 0;
#pragma omp parallel for
  for (int i = 0; i < output_ch; ++i) {
    for (int j = 0; j < out_ch_size; ++j) {
      if (if_bias) {
        output_data[k++] = bias_data[i];
      } else {
        output_data[k++] = 0;
      }
    }
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
    int output_ch_d8 = output_ch >> 3;

#pragma omp parallel for
    for (int o_c8 = 0; o_c8 < output_ch_d8; ++o_c8) {
      std::atomic<float> relu_value{0};
      int o_c = o_c8 * 8;
      const float *f1;
      const float *in_ptr1, *in_ptr2, *in_ptr3;
      const float *pad_filter1, *pad_filter2, *pad_filter3;
      const float *pad_filter1_2, *pad_filter2_2, *pad_filter3_2;
      const float *pad_filter1_3, *pad_filter2_3, *pad_filter3_3;
      const float *pad_filter1_4, *pad_filter2_4, *pad_filter3_4;
      const float *pad_filter1_5, *pad_filter2_5, *pad_filter3_5;
      const float *pad_filter1_6, *pad_filter2_6, *pad_filter3_6;
      const float *pad_filter1_7, *pad_filter2_7, *pad_filter3_7;
      const float *pad_filter1_8, *pad_filter2_8, *pad_filter3_8;

      float reform_filter_arr[76] = {0};
      float pad_filter_arr[pad_filter_ch_size] = {0};
      float pad_filter_arr_2[pad_filter_ch_size] = {0};
      float pad_filter_arr_3[pad_filter_ch_size] = {0};
      float pad_filter_arr_4[pad_filter_ch_size] = {0};
      float pad_filter_arr_5[pad_filter_ch_size] = {0};
      float pad_filter_arr_6[pad_filter_ch_size] = {0};
      float pad_filter_arr_7[pad_filter_ch_size] = {0};
      float pad_filter_arr_8[pad_filter_ch_size] = {0};

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
      const float *filter_data_ch_2;
      const float *filter_data_ch_3;
      const float *filter_data_ch_4;
      const float *filter_data_ch_5;
      const float *filter_data_ch_6;
      const float *filter_data_ch_7;
      const float *filter_data_ch_8;

      filter_data_ch = filter_data + o_c * filter_ch_size * input_ch;
      filter_data_ch_2 = filter_data + (o_c + 1) * filter_ch_size * input_ch;
      filter_data_ch_3 = filter_data + (o_c + 2) * filter_ch_size * input_ch;
      filter_data_ch_4 = filter_data + (o_c + 3) * filter_ch_size * input_ch;
      filter_data_ch_5 = filter_data + (o_c + 4) * filter_ch_size * input_ch;
      filter_data_ch_6 = filter_data + (o_c + 5) * filter_ch_size * input_ch;
      filter_data_ch_7 = filter_data + (o_c + 6) * filter_ch_size * input_ch;
      filter_data_ch_8 = filter_data + (o_c + 7) * filter_ch_size * input_ch;

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

        if (if_relu && i_c == input_ch - 1) {
          relu_value = 0;
        } else {
          relu_value = -FLT_MAX;
        }
        reform_filter_arr[72] = relu_value;
        reform_filter_arr[73] = relu_value;
        reform_filter_arr[74] = relu_value;
        reform_filter_arr[75] = relu_value;

        f1 = reform_filter_arr;

        if (!if_nopadding) {
          pad_filter_arr[pad_filter_ch_size] = {0};

          for (int i = 0; i < 9; ++i) {
            int j = i / 3 * (2 * padding_w + 3) + i % 3 + padding_h * 3 +
                    padding_w * (2 * padding_h + 1);
            pad_filter_arr[j] = filter_data_ch[i];
            pad_filter_arr_2[j] = filter_data_ch_2[i];
            pad_filter_arr_3[j] = filter_data_ch_3[i];
            pad_filter_arr_4[j] = filter_data_ch_4[i];
            pad_filter_arr_5[j] = filter_data_ch_5[i];
            pad_filter_arr_6[j] = filter_data_ch_6[i];
            pad_filter_arr_7[j] = filter_data_ch_7[i];
            pad_filter_arr_8[j] = filter_data_ch_8[i];
          }

          pad_filter1 = pad_filter_arr;
          pad_filter1 += pad_filter_start;
          pad_filter2 = pad_filter1 + pad_filter_w;
          pad_filter3 = pad_filter2 + pad_filter_w;

          pad_filter1_2 = pad_filter_arr_2;
          pad_filter1_2 += pad_filter_start;
          pad_filter2_2 = pad_filter1_2 + pad_filter_w;
          pad_filter3_2 = pad_filter2_2 + pad_filter_w;

          pad_filter1_3 = pad_filter_arr_3;
          pad_filter1_3 += pad_filter_start;
          pad_filter2_3 = pad_filter1_3 + pad_filter_w;
          pad_filter3_3 = pad_filter2_3 + pad_filter_w;

          pad_filter1_4 = pad_filter_arr_4;
          pad_filter1_4 += pad_filter_start;
          pad_filter2_4 = pad_filter1_4 + pad_filter_w;
          pad_filter3_4 = pad_filter2_4 + pad_filter_w;

          pad_filter1_5 = pad_filter_arr_5;
          pad_filter1_5 += pad_filter_start;
          pad_filter2_5 = pad_filter1_5 + pad_filter_w;
          pad_filter3_5 = pad_filter2_5 + pad_filter_w;

          pad_filter1_6 = pad_filter_arr_6;
          pad_filter1_6 += pad_filter_start;
          pad_filter2_6 = pad_filter1_6 + pad_filter_w;
          pad_filter3_6 = pad_filter2_6 + pad_filter_w;

          pad_filter1_7 = pad_filter_arr_7;
          pad_filter1_7 += pad_filter_start;
          pad_filter2_7 = pad_filter1_7 + pad_filter_w;
          pad_filter3_7 = pad_filter2_7 + pad_filter_w;

          pad_filter1_8 = pad_filter_arr_8;
          pad_filter1_8 += pad_filter_start;
          pad_filter2_8 = pad_filter1_8 + pad_filter_w;
          pad_filter3_8 = pad_filter2_8 + pad_filter_w;
        } else {
          pad_filter1 = filter_data_ch;
          pad_filter2 = pad_filter1 + 3;
          pad_filter3 = pad_filter2 + 3;

          pad_filter1_2 = filter_data_ch_2;
          pad_filter2_2 = pad_filter1_2 + 3;
          pad_filter3_2 = pad_filter2_2 + 3;

          pad_filter1_3 = filter_data_ch_3;
          pad_filter2_3 = pad_filter1_3 + 3;
          pad_filter3_3 = pad_filter2_3 + 3;

          pad_filter1_4 = filter_data_ch_4;
          pad_filter2_4 = pad_filter1_4 + 3;
          pad_filter3_4 = pad_filter2_4 + 3;

          pad_filter1_5 = filter_data_ch_5;
          pad_filter2_5 = pad_filter1_5 + 3;
          pad_filter3_5 = pad_filter2_5 + 3;

          pad_filter1_6 = filter_data_ch_6;
          pad_filter2_6 = pad_filter1_6 + 3;
          pad_filter3_6 = pad_filter2_6 + 3;

          pad_filter1_7 = filter_data_ch_7;
          pad_filter2_7 = pad_filter1_7 + 3;
          pad_filter3_7 = pad_filter2_7 + 3;

          pad_filter1_8 = filter_data_ch_8;
          pad_filter2_8 = pad_filter1_8 + 3;
          pad_filter3_8 = pad_filter2_8 + 3;
        }
        float *out_ptr1;
        float *out_ptr1_2;
        float *out_ptr1_3;
        float *out_ptr1_4;
        float *out_ptr1_5;
        float *out_ptr1_6;
        float *out_ptr1_7;
        float *out_ptr1_8;

        out_ptr1 = output_data_ch;
        out_ptr1_2 = output_data_ch_2;
        out_ptr1_3 = output_data_ch_3;
        out_ptr1_4 = output_data_ch_4;
        out_ptr1_5 = output_data_ch_5;
        out_ptr1_6 = output_data_ch_6;
        out_ptr1_7 = output_data_ch_7;
        out_ptr1_8 = output_data_ch_8;

        in_ptr1 = input_data_ch;
        in_ptr2 = in_ptr1 + input_w;
        in_ptr3 = in_ptr2 + input_w;

        int o_h = 0;

        for (; o_h < output_h; ++o_h) {
          int o_w = 0;

          // pad left
          for (; o_w <= valid_w_start; ++o_w) {
            float sum1 = 0;
            float sum1_2 = 0;
            float sum1_3 = 0;
            float sum1_4 = 0;
            float sum1_5 = 0;
            float sum1_6 = 0;
            float sum1_7 = 0;
            float sum1_8 = 0;
#if __ARM_NEON
            float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
            float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
            float32x4_t _pad_filter1_2 = vld1q_f32(pad_filter1_2);
            float32x4_t _pad_filter1_3 = vld1q_f32(pad_filter1_3);
            float32x4_t _pad_filter1_4 = vld1q_f32(pad_filter1_4);
            float32x4_t _pad_filter1_5 = vld1q_f32(pad_filter1_5);
            float32x4_t _pad_filter1_6 = vld1q_f32(pad_filter1_6);
            float32x4_t _pad_filter1_7 = vld1q_f32(pad_filter1_7);
            float32x4_t _pad_filter1_8 = vld1q_f32(pad_filter1_8);

            float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
            float32x4_t _sum1_2 = vmulq_f32(_in_ptr1, _pad_filter1_2);
            float32x4_t _sum1_3 = vmulq_f32(_in_ptr1, _pad_filter1_3);
            float32x4_t _sum1_4 = vmulq_f32(_in_ptr1, _pad_filter1_4);
            float32x4_t _sum1_5 = vmulq_f32(_in_ptr1, _pad_filter1_5);
            float32x4_t _sum1_6 = vmulq_f32(_in_ptr1, _pad_filter1_6);
            float32x4_t _sum1_7 = vmulq_f32(_in_ptr1, _pad_filter1_7);
            float32x4_t _sum1_8 = vmulq_f32(_in_ptr1, _pad_filter1_8);

            float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
            float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
            float32x4_t _pad_filter2_2 = vld1q_f32(pad_filter2_2);
            float32x4_t _pad_filter2_3 = vld1q_f32(pad_filter2_3);
            float32x4_t _pad_filter2_4 = vld1q_f32(pad_filter2_4);
            float32x4_t _pad_filter2_5 = vld1q_f32(pad_filter2_5);
            float32x4_t _pad_filter2_6 = vld1q_f32(pad_filter2_6);
            float32x4_t _pad_filter2_7 = vld1q_f32(pad_filter2_7);
            float32x4_t _pad_filter2_8 = vld1q_f32(pad_filter2_8);

            _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
            _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr2, _pad_filter2_2);
            _sum1_3 = vmlaq_f32(_sum1_3, _in_ptr2, _pad_filter2_3);
            _sum1_4 = vmlaq_f32(_sum1_4, _in_ptr2, _pad_filter2_4);
            _sum1_5 = vmlaq_f32(_sum1_5, _in_ptr2, _pad_filter2_5);
            _sum1_6 = vmlaq_f32(_sum1_6, _in_ptr2, _pad_filter2_6);
            _sum1_7 = vmlaq_f32(_sum1_7, _in_ptr2, _pad_filter2_7);
            _sum1_8 = vmlaq_f32(_sum1_8, _in_ptr2, _pad_filter2_8);

            float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
            float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
            float32x4_t _pad_filter3_2 = vld1q_f32(pad_filter3_2);
            float32x4_t _pad_filter3_3 = vld1q_f32(pad_filter3_3);
            float32x4_t _pad_filter3_4 = vld1q_f32(pad_filter3_4);
            float32x4_t _pad_filter3_5 = vld1q_f32(pad_filter3_5);
            float32x4_t _pad_filter3_6 = vld1q_f32(pad_filter3_6);
            float32x4_t _pad_filter3_7 = vld1q_f32(pad_filter3_7);
            float32x4_t _pad_filter3_8 = vld1q_f32(pad_filter3_8);

            _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
            _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr3, _pad_filter3_2);
            _sum1_3 = vmlaq_f32(_sum1_3, _in_ptr3, _pad_filter3_3);
            _sum1_4 = vmlaq_f32(_sum1_4, _in_ptr3, _pad_filter3_4);
            _sum1_5 = vmlaq_f32(_sum1_5, _in_ptr3, _pad_filter3_5);
            _sum1_6 = vmlaq_f32(_sum1_6, _in_ptr3, _pad_filter3_6);
            _sum1_7 = vmlaq_f32(_sum1_7, _in_ptr3, _pad_filter3_7);
            _sum1_8 = vmlaq_f32(_sum1_8, _in_ptr3, _pad_filter3_8);

            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
            _sum1_2 = vsetq_lane_f32(sum1_2, _sum1_2, 3);
            _sum1_3 = vsetq_lane_f32(sum1_3, _sum1_3, 3);
            _sum1_4 = vsetq_lane_f32(sum1_4, _sum1_4, 3);
            _sum1_5 = vsetq_lane_f32(sum1_5, _sum1_5, 3);
            _sum1_6 = vsetq_lane_f32(sum1_6, _sum1_6, 3);
            _sum1_7 = vsetq_lane_f32(sum1_7, _sum1_7, 3);
            _sum1_8 = vsetq_lane_f32(sum1_8, _sum1_8, 3);

            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ss1_2 =
                vadd_f32(vget_low_f32(_sum1_2), vget_high_f32(_sum1_2));
            float32x2_t _ss1_3 =
                vadd_f32(vget_low_f32(_sum1_3), vget_high_f32(_sum1_3));
            float32x2_t _ss1_4 =
                vadd_f32(vget_low_f32(_sum1_4), vget_high_f32(_sum1_4));
            float32x2_t _ss1_5 =
                vadd_f32(vget_low_f32(_sum1_5), vget_high_f32(_sum1_5));
            float32x2_t _ss1_6 =
                vadd_f32(vget_low_f32(_sum1_6), vget_high_f32(_sum1_6));
            float32x2_t _ss1_7 =
                vadd_f32(vget_low_f32(_sum1_7), vget_high_f32(_sum1_7));
            float32x2_t _ss1_8 =
                vadd_f32(vget_low_f32(_sum1_8), vget_high_f32(_sum1_8));

            float32x2_t _ssss1_ssss1_2 = vpadd_f32(_ss1, _ss1_2);
            float32x2_t _ssss1_3_ssss1_4 = vpadd_f32(_ss1_3, _ss1_4);
            float32x2_t _ssss1_5_ssss1_6 = vpadd_f32(_ss1_5, _ss1_6);
            float32x2_t _ssss1_7_ssss1_8 = vpadd_f32(_ss1_7, _ss1_8);

            sum1 += vget_lane_f32(_ssss1_ssss1_2, 0);
            sum1_2 += vget_lane_f32(_ssss1_ssss1_2, 1);
            sum1_3 += vget_lane_f32(_ssss1_3_ssss1_4, 0);
            sum1_4 += vget_lane_f32(_ssss1_3_ssss1_4, 1);
            sum1_5 += vget_lane_f32(_ssss1_5_ssss1_6, 0);
            sum1_6 += vget_lane_f32(_ssss1_5_ssss1_6, 1);
            sum1_7 += vget_lane_f32(_ssss1_7_ssss1_8, 0);
            sum1_8 += vget_lane_f32(_ssss1_7_ssss1_8, 1);
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

            sum1_2 += in_ptr1[0] * pad_filter1_2[0];
            sum1_2 += in_ptr1[1] * pad_filter1_2[1];
            sum1_2 += in_ptr1[2] * pad_filter1_2[2];
            sum1_2 += in_ptr2[0] * pad_filter2_2[0];
            sum1_2 += in_ptr2[1] * pad_filter2_2[1];
            sum1_2 += in_ptr2[2] * pad_filter2_2[2];
            sum1_2 += in_ptr3[0] * pad_filter3_2[0];
            sum1_2 += in_ptr3[1] * pad_filter3_2[1];
            sum1_2 += in_ptr3[2] * pad_filter3_2[2];

            sum1_3 += in_ptr1[0] * pad_filter1_3[0];
            sum1_3 += in_ptr1[1] * pad_filter1_3[1];
            sum1_3 += in_ptr1[2] * pad_filter1_3[2];
            sum1_3 += in_ptr2[0] * pad_filter2_3[0];
            sum1_3 += in_ptr2[1] * pad_filter2_3[1];
            sum1_3 += in_ptr2[2] * pad_filter2_3[2];
            sum1_3 += in_ptr3[0] * pad_filter3_3[0];
            sum1_3 += in_ptr3[1] * pad_filter3_3[1];
            sum1_3 += in_ptr3[2] * pad_filter3_3[2];

            sum1_4 += in_ptr1[0] * pad_filter1_4[0];
            sum1_4 += in_ptr1[1] * pad_filter1_4[1];
            sum1_4 += in_ptr1[2] * pad_filter1_4[2];
            sum1_4 += in_ptr2[0] * pad_filter2_4[0];
            sum1_4 += in_ptr2[1] * pad_filter2_4[1];
            sum1_4 += in_ptr2[2] * pad_filter2_4[2];
            sum1_4 += in_ptr3[0] * pad_filter3_4[0];
            sum1_4 += in_ptr3[1] * pad_filter3_4[1];
            sum1_4 += in_ptr3[2] * pad_filter3_4[2];

            sum1_5 += in_ptr1[0] * pad_filter1_5[0];
            sum1_5 += in_ptr1[1] * pad_filter1_5[1];
            sum1_5 += in_ptr1[2] * pad_filter1_5[2];
            sum1_5 += in_ptr2[0] * pad_filter2_5[0];
            sum1_5 += in_ptr2[1] * pad_filter2_5[1];
            sum1_5 += in_ptr2[2] * pad_filter2_5[2];
            sum1_5 += in_ptr3[0] * pad_filter3_5[0];
            sum1_5 += in_ptr3[1] * pad_filter3_5[1];
            sum1_5 += in_ptr3[2] * pad_filter3_5[2];

            sum1_6 += in_ptr1[0] * pad_filter1_6[0];
            sum1_6 += in_ptr1[1] * pad_filter1_6[1];
            sum1_6 += in_ptr1[2] * pad_filter1_6[2];
            sum1_6 += in_ptr2[0] * pad_filter2_6[0];
            sum1_6 += in_ptr2[1] * pad_filter2_6[1];
            sum1_6 += in_ptr2[2] * pad_filter2_6[2];
            sum1_6 += in_ptr3[0] * pad_filter3_6[0];
            sum1_6 += in_ptr3[1] * pad_filter3_6[1];
            sum1_6 += in_ptr3[2] * pad_filter3_6[2];

            sum1_7 += in_ptr1[0] * pad_filter1_7[0];
            sum1_7 += in_ptr1[1] * pad_filter1_7[1];
            sum1_7 += in_ptr1[2] * pad_filter1_7[2];
            sum1_7 += in_ptr2[0] * pad_filter2_7[0];
            sum1_7 += in_ptr2[1] * pad_filter2_7[1];
            sum1_7 += in_ptr2[2] * pad_filter2_7[2];
            sum1_7 += in_ptr3[0] * pad_filter3_7[0];
            sum1_7 += in_ptr3[1] * pad_filter3_7[1];
            sum1_7 += in_ptr3[2] * pad_filter3_7[2];

            sum1_8 += in_ptr1[0] * pad_filter1_8[0];
            sum1_8 += in_ptr1[1] * pad_filter1_8[1];
            sum1_8 += in_ptr1[2] * pad_filter1_8[2];
            sum1_8 += in_ptr2[0] * pad_filter2_8[0];
            sum1_8 += in_ptr2[1] * pad_filter2_8[1];
            sum1_8 += in_ptr2[2] * pad_filter2_8[2];
            sum1_8 += in_ptr3[0] * pad_filter3_8[0];
            sum1_8 += in_ptr3[1] * pad_filter3_8[1];
            sum1_8 += in_ptr3[2] * pad_filter3_8[2];
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
              pad_filter1_2--;
              pad_filter2_2--;
              pad_filter3_2--;

              pad_filter1_3--;
              pad_filter2_3--;
              pad_filter3_3--;
              pad_filter1_4--;
              pad_filter2_4--;
              pad_filter3_4--;

              pad_filter1_5--;
              pad_filter2_5--;
              pad_filter3_5--;
              pad_filter1_6--;
              pad_filter2_6--;
              pad_filter3_6--;

              pad_filter1_7--;
              pad_filter2_7--;
              pad_filter3_7--;
              pad_filter1_8--;
              pad_filter2_8--;
              pad_filter3_8--;

              in_ptr1++;
              in_ptr2++;
              in_ptr3++;

            } else if (input_w <= 3 || o_w < valid_w_start ||
                       o_w > valid_w_end) {
              pad_filter1 -= 2;
              pad_filter2 -= 2;
              pad_filter3 -= 2;
              pad_filter1_2 -= 2;
              pad_filter2_2 -= 2;
              pad_filter3_2 -= 2;

              pad_filter1_3 -= 2;
              pad_filter2_3 -= 2;
              pad_filter3_3 -= 2;
              pad_filter1_4 -= 2;
              pad_filter2_4 -= 2;
              pad_filter3_4 -= 2;

              pad_filter1_5 -= 2;
              pad_filter2_5 -= 2;
              pad_filter3_5 -= 2;
              pad_filter1_6 -= 2;
              pad_filter2_6 -= 2;
              pad_filter3_6 -= 2;

              pad_filter1_7 -= 2;
              pad_filter2_7 -= 2;
              pad_filter3_7 -= 2;
              pad_filter1_8 -= 2;
              pad_filter2_8 -= 2;
              pad_filter3_8 -= 2;
            } else {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;
            }
            *out_ptr1 += sum1;
            *out_ptr1_2 += sum1_2;
            *out_ptr1_3 += sum1_3;
            *out_ptr1_4 += sum1_4;
            *out_ptr1_5 += sum1_5;
            *out_ptr1_6 += sum1_6;
            *out_ptr1_7 += sum1_7;
            *out_ptr1_8 += sum1_8;

            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }
            if (out_ptr1_2[0] < relu_value) {
              *out_ptr1_2 = relu_value;
            }
            if (out_ptr1_3[0] < relu_value) {
              *out_ptr1_3 = relu_value;
            }
            if (out_ptr1_4[0] < relu_value) {
              *out_ptr1_4 = relu_value;
            }
            if (out_ptr1_5[0] < relu_value) {
              *out_ptr1_5 = relu_value;
            }
            if (out_ptr1_6[0] < relu_value) {
              *out_ptr1_6 = relu_value;
            }
            if (out_ptr1_7[0] < relu_value) {
              *out_ptr1_7 = relu_value;
            }
            if (out_ptr1_8[0] < relu_value) {
              *out_ptr1_8 = relu_value;
            }

            out_ptr1++;
            out_ptr1_2++;
            out_ptr1_3++;
            out_ptr1_4++;
            out_ptr1_5++;
            out_ptr1_6++;
            out_ptr1_7++;
            out_ptr1_8++;
          }
            // valid
#if __ARM_NEON
#if __aarch64__
          if (o_h > valid_h_start && o_h <= valid_h_end) {
            int o_w_dim4 = (valid_w_end - valid_w_start - 1) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "prfm  pldl1keep, [%[in_ptr1], #288]        \n\t"

                  "ld1  {v0.4s, v1.4s}, [%[f1]], #32          \n\t"
                  "ld2   {v4.4s, v5.4s}, [%[in_ptr1]], #32    \n\t"
                  "ld2   {v6.4s, v7.4s}, [%[in_ptr1]]         \n\t"
                  "0:                                         \n\t"
                  // load out_ptr
                  "prfm  pldl1keep, [%[out_ptr1], #128]       \n\t"
                  "prfm  pldl1keep, [%[out_ptr1_2], #128]     \n\t"
                  "prfm  pldl1keep, [%[out_ptr1_3], #128]     \n\t"
                  "prfm  pldl1keep, [%[out_ptr1_4], #128]     \n\t"
                  "prfm  pldl1keep, [%[out_ptr1_5], #128]     \n\t"
                  "prfm  pldl1keep, [%[out_ptr1_6], #128]     \n\t"
                  "prfm  pldl1keep, [%[out_ptr1_7], #128]     \n\t"
                  "prfm  pldl1keep, [%[out_ptr1_8], #128]     \n\t"

                  "ld1   {v8.4s}, [%[out_ptr1]]               \n\t"
                  "ld1   {v9.4s}, [%[out_ptr1_2]]             \n\t"
                  "ld1   {v10.4s}, [%[out_ptr1_3]]            \n\t"
                  "ld1   {v11.4s}, [%[out_ptr1_4]]            \n\t"
                  "ld1   {v12.4s}, [%[out_ptr1_5]]            \n\t"
                  "ld1   {v13.4s}, [%[out_ptr1_6]]            \n\t"
                  "ld1   {v14.4s}, [%[out_ptr1_7]]            \n\t"
                  "ld1   {v15.4s}, [%[out_ptr1_8]]            \n\t"

                  // in_ptr1 multiply
                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "ld1   {v2.4s, v3.4s}, [%[f1]], #32         \n\t"
                  "fmla    v8.4s, v4.4s, v0.4s[0]             \n\t"
                  "fmla    v9.4s, v4.4s, v0.4s[1]             \n\t"
                  "fmla   v10.4s, v4.4s, v0.4s[2]             \n\t"
                  "fmla   v11.4s, v4.4s, v0.4s[3]             \n\t"

                  "fmla   v12.4s, v4.4s, v1.4s[0]             \n\t"
                  "fmla   v13.4s, v4.4s, v1.4s[1]             \n\t"
                  "fmla   v14.4s, v4.4s, v1.4s[2]             \n\t"
                  "fmla   v15.4s, v4.4s, v1.4s[3]             \n\t"

                  "ext    v7.16b, v4.16b, v6.16b, #4          \n\t"
                  "fmla    v8.4s, v5.4s, v2.4s[0]             \n\t"
                  "fmla    v9.4s, v5.4s, v2.4s[1]             \n\t"
                  "fmla   v10.4s, v5.4s, v2.4s[2]             \n\t"
                  "fmla   v11.4s, v5.4s, v2.4s[3]             \n\t"

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "ld1   {v0.4s, v1.4s}, [%[f1]], #32         \n\t"
                  "fmla   v12.4s, v5.4s, v3.4s[0]             \n\t"
                  "fmla   v13.4s, v5.4s, v3.4s[1]             \n\t"
                  "fmla   v14.4s, v5.4s, v3.4s[2]             \n\t"
                  "fmla   v15.4s, v5.4s, v3.4s[3]             \n\t"

                  "prfm  pldl1keep, [%[in_ptr2], #288]        \n\t"
                  "ld2    {v4.4s, v5.4s}, [%[in_ptr2]], #32   \n\t"
                  "fmla    v8.4s, v7.4s, v0.4s[0]             \n\t"
                  "fmla    v9.4s, v7.4s, v0.4s[1]             \n\t"
                  "fmla   v10.4s, v7.4s, v0.4s[2]             \n\t"
                  "fmla   v11.4s, v7.4s, v0.4s[3]             \n\t"

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "ld1   {v2.4s, v3.4s}, [%[f1]], #32         \n\t"

                  "fmla   v12.4s, v7.4s, v1.4s[0]             \n\t"
                  "fmla   v13.4s, v7.4s, v1.4s[1]             \n\t"
                  "fmla   v14.4s, v7.4s, v1.4s[2]             \n\t"
                  "fmla   v15.4s, v7.4s, v1.4s[3]             \n\t"

                  // in_ptr2 multiply
                  "ld2    {v6.4s, v7.4s}, [%[in_ptr2]]        \n\t"
                  "fmla    v8.4s, v4.4s, v2.4s[0]             \n\t"
                  "fmla    v9.4s, v4.4s, v2.4s[1]             \n\t"
                  "fmla   v10.4s, v4.4s, v2.4s[2]             \n\t"
                  "fmla   v11.4s, v4.4s, v2.4s[3]             \n\t"

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "ld1   {v0.4s, v1.4s}, [%[f1]], #32         \n\t"
                  "fmla   v12.4s, v4.4s, v3.4s[0]             \n\t"
                  "fmla   v13.4s, v4.4s, v3.4s[1]             \n\t"
                  "fmla   v14.4s, v4.4s, v3.4s[2]             \n\t"
                  "fmla   v15.4s, v4.4s, v3.4s[3]             \n\t"

                  "ext    v7.16b, v4.16b, v6.16b, #4          \n\t"
                  "fmla    v8.4s, v5.4s, v0.4s[0]             \n\t"
                  "fmla    v9.4s, v5.4s, v0.4s[1]             \n\t"
                  "fmla   v10.4s, v5.4s, v0.4s[2]             \n\t"
                  "fmla   v11.4s, v5.4s, v0.4s[3]             \n\t"

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "ld1    {v2.4s, v3.4s}, [%[f1]], #32        \n\t"
                  "fmla   v12.4s, v5.4s, v1.4s[0]             \n\t"
                  "fmla   v13.4s, v5.4s, v1.4s[1]             \n\t"

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "prfm  pldl1keep, [%[in_ptr3], #288]        \n\t"
                  "fmla   v14.4s, v5.4s, v1.4s[2]             \n\t"
                  "fmla   v15.4s, v5.4s, v1.4s[3]             \n\t"

                  "ld1  {v0.4s, v1.4s}, [%[f1]], #32          \n\t"
                  "ld2   {v4.4s, v5.4s}, [%[in_ptr3]], #32    \n\t"
                  "fmla    v8.4s, v7.4s, v2.4s[0]             \n\t"
                  "fmla    v9.4s, v7.4s, v2.4s[1]             \n\t"
                  "fmla   v10.4s, v7.4s, v2.4s[2]             \n\t"
                  "fmla   v11.4s, v7.4s, v2.4s[3]             \n\t"

                  "fmla   v12.4s, v7.4s, v3.4s[0]             \n\t"
                  "fmla   v13.4s, v7.4s, v3.4s[1]             \n\t"
                  "fmla   v14.4s, v7.4s, v3.4s[2]             \n\t"
                  "fmla   v15.4s, v7.4s, v3.4s[3]             \n\t"

                  // in_ptr3 multiply
                  "ld2   {v6.4s, v7.4s}, [%[in_ptr3]]         \n\t"
                  "fmla    v8.4s, v4.4s, v0.4s[0]             \n\t"
                  "fmla    v9.4s, v4.4s, v0.4s[1]             \n\t"
                  "fmla   v10.4s, v4.4s, v0.4s[2]             \n\t"
                  "fmla   v11.4s, v4.4s, v0.4s[3]             \n\t"

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "ld1   {v2.4s, v3.4s}, [%[f1]], #32         \n\t"
                  "fmla   v12.4s, v4.4s, v1.4s[0]             \n\t"
                  "fmla   v13.4s, v4.4s, v1.4s[1]             \n\t"
                  "fmla   v14.4s, v4.4s, v1.4s[2]             \n\t"
                  "fmla   v15.4s, v4.4s, v1.4s[3]             \n\t"

                  "ext    v7.16b, v4.16b, v6.16b, #4          \n\t"
                  "fmla    v8.4s, v5.4s, v2.4s[0]             \n\t"
                  "fmla    v9.4s, v5.4s, v2.4s[1]             \n\t"
                  "fmla   v10.4s, v5.4s, v2.4s[2]             \n\t"
                  "fmla   v11.4s, v5.4s, v2.4s[3]             \n\t"

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "ld1   {v0.4s, v1.4s}, [%[f1]], #32         \n\t"
                  "fmla   v12.4s, v5.4s, v3.4s[0]             \n\t"
                  "fmla   v13.4s, v5.4s, v3.4s[1]             \n\t"
                  "fmla   v14.4s, v5.4s, v3.4s[2]             \n\t"
                  "fmla   v15.4s, v5.4s, v3.4s[3]             \n\t"

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "ld1   {v2.4s}, [%[f1]]                     \n\t"
                  "sub        %[f1], %[f1], #288              \n\t"
                  "fmla    v8.4s, v7.4s, v0.4s[0]             \n\t"
                  "fmla    v9.4s, v7.4s, v0.4s[1]             \n\t"
                  "fmla   v10.4s, v7.4s, v0.4s[2]             \n\t"
                  "fmla   v11.4s, v7.4s, v0.4s[3]             \n\t"

                  "fmla   v12.4s, v7.4s, v1.4s[0]             \n\t"
                  "fmla   v13.4s, v7.4s, v1.4s[1]             \n\t"
                  "fmla   v14.4s, v7.4s, v1.4s[2]             \n\t"
                  "fmla   v15.4s, v7.4s, v1.4s[3]             \n\t"

                  // store out_ptr
                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "fmax    v8.4s,  v8.4s, v2.4s               \n\t"
                  "prfm  pldl1keep, [%[in_ptr1], #288]        \n\t"
                  "fmax    v9.4s,  v9.4s, v2.4s               \n\t"

                  "ld1  {v0.4s, v1.4s}, [%[f1]], #32          \n\t"
                  "fmax   v10.4s, v10.4s, v2.4s               \n\t"

                  "ld2   {v4.4s, v5.4s}, [%[in_ptr1]], #32    \n\t"
                  "st1   {v8.4s}, [%[out_ptr1]], #16          \n\t"
                  "fmax   v11.4s, v11.4s, v2.4s               \n\t"
                  "st1   {v9.4s}, [%[out_ptr1_2]], #16        \n\t"

                  "fmax   v12.4s, v12.4s, v2.4s               \n\t"
                  "st1   {v10.4s}, [%[out_ptr1_3]], #16       \n\t"
                  "fmax   v13.4s, v13.4s, v2.4s               \n\t"
                  "st1   {v11.4s}, [%[out_ptr1_4]], #16       \n\t"

                  "fmax   v14.4s, v14.4s, v2.4s               \n\t"
                  "st1   {v12.4s}, [%[out_ptr1_5]], #16       \n\t"
                  "fmax   v15.4s, v15.4s, v2.4s               \n\t"
                  "st1   {v13.4s}, [%[out_ptr1_6]], #16       \n\t"

                  "ld2   {v6.4s, v7.4s}, [%[in_ptr1]]         \n\t"
                  "st1   {v14.4s}, [%[out_ptr1_7]], #16       \n\t"
                  "subs       %[o_w_dim4], %[o_w_dim4], #1    \n\t"
                  "st1   {v15.4s}, [%[out_ptr1_8]], #16       \n\t"

                  // cycle
                  "bne        0b                              \n\t"
                  "sub       %[f1], %[in_ptr1], #32           \n\t"
                  "sub       %[in_ptr1], %[in_ptr1], #32      \n\t"

                  :
                  [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                  [out_ptr1_2] "+r"(out_ptr1_2), [out_ptr1_3] "+r"(out_ptr1_3),
                  [out_ptr1_4] "+r"(out_ptr1_4), [out_ptr1_5] "+r"(out_ptr1_5),
                  [out_ptr1_6] "+r"(out_ptr1_6), [out_ptr1_7] "+r"(out_ptr1_7),
                  [out_ptr1_8] "+r"(out_ptr1_8), [in_ptr1] "+r"(in_ptr1),
                  [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3)
                  : [f1] "r"(f1)
                  : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v12", "v13", "v14", "v15");
            }
          }
#else
          if (o_h > valid_h_start && o_h <= valid_h_end) {
            int o_w_dim4 = (valid_w_end - valid_w_start - 1) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(

                  "pld        [%[f1], #256]                   \n\t"
                  "pld        [%[in_ptr1], #288]              \n\t"

                  "vld1.32   {d0-d3}, [%[f1]]!                \n\t"
                  "vld2.f32   {d8-d11}, [%[in_ptr1]]!         \n\t"
                  "vld2.f32   {d12, d13}, [%[in_ptr1]]        \n\t"

                  "0:                                         \n\t"
                  // load out_ptr
                  "pld        [%[out_ptr1], #128]             \n\t"
                  "pld        [%[out_ptr1_2], #128]           \n\t"
                  "pld        [%[out_ptr1_3], #128]           \n\t"
                  "pld        [%[out_ptr1_4], #128]           \n\t"
                  "pld        [%[out_ptr1_5], #128]           \n\t"
                  "pld        [%[out_ptr1_6], #128]           \n\t"
                  "pld        [%[out_ptr1_7], #128]           \n\t"
                  "pld        [%[out_ptr1_8], #128]           \n\t"

                  "vld1.f32   {d16, d17}, [%[out_ptr1]]       \n\t"
                  "vld1.f32   {d18, d19}, [%[out_ptr1_2]]     \n\t"
                  "vld1.f32   {d20, d21}, [%[out_ptr1_3]]     \n\t"
                  "vld1.f32   {d22, d23}, [%[out_ptr1_4]]     \n\t"
                  "vld1.f32   {d24, d25}, [%[out_ptr1_5]]     \n\t"
                  "vld1.f32   {d26, d27}, [%[out_ptr1_6]]     \n\t"
                  "vld1.f32   {d28, d29}, [%[out_ptr1_7]]     \n\t"
                  "vld1.f32   {d30, d31}, [%[out_ptr1_8]]     \n\t"

                  // in_ptr1 multiply
                  "pld        [%[f1], #256]                   \n\t"
                  "vld1.32   {d4-d7}, [%[f1]]!                \n\t"
                  "vmla.f32   q8, q4, d0[0]                   \n\t"
                  "vmla.f32   q9, q4, d0[1]                   \n\t"

                  "vmla.f32   q10, q4, d1[0]                  \n\t"
                  "vmla.f32   q11, q4, d1[1]                  \n\t"

                  "vmla.f32   q12, q4, d2[0]                  \n\t"
                  "vmla.f32   q13, q4, d2[1]                  \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vmla.f32   q14, q4, d3[0]                  \n\t"
                  "vmla.f32   q15, q4, d3[1]                  \n\t"

                  "vld1.32   {d0-d3}, [%[f1]]!                \n\t"
                  "vmla.f32   q8, q5, d4[0]                   \n\t"
                  "vmla.f32   q9, q5, d4[1]                   \n\t"

                  "vext.32    q7, q4, q6, #1                  \n\t"
                  "vmla.f32   q10, q5, d5[0]                  \n\t"
                  "vmla.f32   q11, q5, d5[1]                  \n\t"

                  "vmla.f32   q12, q5, d6[0]                  \n\t"
                  "vmla.f32   q13, q5, d6[1]                  \n\t"

                  "pld        [%[in_ptr2], #288]              \n\t"
                  "vmla.f32   q14, q5, d7[0]                  \n\t"
                  "vmla.f32   q15, q5, d7[1]                  \n\t"

                  "vld2.f32   {d8-d11}, [%[in_ptr2]]!         \n\t"
                  "vmla.f32   q8, q7, d0[0]                   \n\t"
                  "vmla.f32   q9, q7, d0[1]                   \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vld1.32   {d4-d7}, [%[f1]]!                \n\t"
                  "vmla.f32   q10, q7, d1[0]                  \n\t"
                  "vmla.f32   q11, q7, d1[1]                  \n\t"

                  "vld2.f32   {d12, d13}, [%[in_ptr2]]        \n\t"
                  "vmla.f32   q12, q7, d2[0]                  \n\t"
                  "vmla.f32   q13, q7, d2[1]                  \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vmla.f32   q14, q7, d3[0]                  \n\t"
                  "vmla.f32   q15, q7, d3[1]                  \n\t"

                  // in_ptr2 multiply
                  "vld1.32   {d0-d3}, [%[f1]]!                \n\t"
                  "vmla.f32   q8, q4, d4[0]                   \n\t"
                  "vmla.f32   q9, q4, d4[1]                   \n\t"

                  "vmla.f32   q10, q4, d5[0]                  \n\t"
                  "vmla.f32   q11, q4, d5[1]                  \n\t"

                  "vmla.f32   q12, q4, d6[0]                  \n\t"
                  "vmla.f32   q13, q4, d6[1]                  \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vmla.f32   q14, q4, d7[0]                  \n\t"
                  "vmla.f32   q15, q4, d7[1]                  \n\t"

                  "vld1.32   {d4-d7}, [%[f1]]!                \n\t"
                  "vmla.f32   q8, q5, d0[0]                   \n\t"
                  "vmla.f32   q9, q5, d0[1]                   \n\t"

                  "vext.32    q7, q4, q6, #1                  \n\t"
                  "vmla.f32   q10, q5, d1[0]                  \n\t"
                  "vmla.f32   q11, q5, d1[1]                  \n\t"

                  "vmla.f32   q12, q5, d2[0]                  \n\t"
                  "vmla.f32   q13, q5, d2[1]                  \n\t"

                  "pld        [%[in_ptr3], #288]              \n\t"
                  "vmla.f32   q14, q5, d3[0]                  \n\t"
                  "vmla.f32   q15, q5, d3[1]                  \n\t"

                  "vld2.f32   {d8-d11}, [%[in_ptr3]]!         \n\t"
                  "vmla.f32   q8, q7, d4[0]                   \n\t"
                  "vmla.f32   q9, q7, d4[1]                   \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vld1.32   {d0-d3}, [%[f1]]!                \n\t"
                  "vmla.f32   q10, q7, d5[0]                  \n\t"
                  "vmla.f32   q11, q7, d5[1]                  \n\t"

                  "vld2.f32   {d12, d13}, [%[in_ptr3]]        \n\t"
                  "vmla.f32   q12, q7, d6[0]                  \n\t"
                  "vmla.f32   q13, q7, d6[1]                  \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vmla.f32   q14, q7, d7[0]                  \n\t"
                  "vmla.f32   q15, q7, d7[1]                  \n\t"

                  // in_ptr3 multiply
                  "vld1.32   {d4-d7}, [%[f1]]!                \n\t"
                  "vmla.f32   q8, q4, d0[0]                   \n\t"
                  "vmla.f32   q9, q4, d0[1]                   \n\t"

                  "vmla.f32   q10, q4, d1[0]                  \n\t"
                  "vmla.f32   q11, q4, d1[1]                  \n\t"

                  "vmla.f32   q12, q4, d2[0]                  \n\t"
                  "vmla.f32   q13, q4, d2[1]                  \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vmla.f32   q14, q4, d3[0]                  \n\t"
                  "vmla.f32   q15, q4, d3[1]                  \n\t"

                  "vld1.32   {d0-d3}, [%[f1]]!                \n\t"
                  "vmla.f32   q8, q5, d4[0]                   \n\t"
                  "vmla.f32   q9, q5, d4[1]                   \n\t"

                  "vext.32    q7, q4, q6, #1                  \n\t"
                  "vmla.f32   q10, q5, d5[0]                  \n\t"
                  "vmla.f32   q11, q5, d5[1]                  \n\t"

                  "vmla.f32   q12, q5, d6[0]                  \n\t"
                  "vmla.f32   q13, q5, d6[1]                  \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vmla.f32   q14, q5, d7[0]                  \n\t"
                  "vmla.f32   q15, q5, d7[1]                  \n\t"

                  "vld1.32   {d4, d5}, [%[f1]]                \n\t"
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
                  "vmax.f32   q8, q8, q2                      \n\t"
                  "vmax.f32   q9, q9, q2                      \n\t"

                  "vmax.f32   q10, q10, q2                    \n\t"
                  "pld        [%[f1], #256]                   \n\t"
                  "vld1.32   {d0-d3}, [%[f1]]!                \n\t"

                  "pld        [%[in_ptr1], #288]              \n\t"
                  "vld2.f32   {d8-d11}, [%[in_ptr1]]!         \n\t"
                  "vst1.f32   {d16, d17}, [%[out_ptr1]]!      \n\t"

                  "vmax.f32   q11, q11, q2                    \n\t"
                  "vst1.f32   {d18, d19}, [%[out_ptr1_2]]!    \n\t"

                  "vmax.f32   q12, q12, q2                    \n\t"
                  "vst1.f32   {d20, d21}, [%[out_ptr1_3]]!    \n\t"

                  "vmax.f32   q13, q13, q2                    \n\t"
                  "vst1.f32   {d22, d23}, [%[out_ptr1_4]]!    \n\t"

                  "vmax.f32   q14, q14, q2                    \n\t"
                  "vst1.f32   {d24, d25}, [%[out_ptr1_5]]!    \n\t"

                  "vmax.f32   q15, q15, q2                    \n\t"
                  "vst1.f32   {d26, d27}, [%[out_ptr1_6]]!    \n\t"

                  "vld2.f32   {d12, d13}, [%[in_ptr1]]        \n\t"
                  "vst1.f32   {d28, d29}, [%[out_ptr1_7]]!    \n\t"

                  "subs       %[o_w_dim4], #1                 \n\t"
                  "vst1.f32   {d30, d31}, [%[out_ptr1_8]]!    \n\t"

                  // cycle
                  "bne        0b                              \n\t"
                  "sub        %[f1], %[f1], #32               \n\t"
                  "sub        %[in_ptr1], %[in_ptr1], #32     \n\t"

                  :
                  [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                  [out_ptr1_2] "+r"(out_ptr1_2), [out_ptr1_3] "+r"(out_ptr1_3),
                  [out_ptr1_4] "+r"(out_ptr1_4), [out_ptr1_5] "+r"(out_ptr1_5),
                  [out_ptr1_6] "+r"(out_ptr1_6), [out_ptr1_7] "+r"(out_ptr1_7),
                  [out_ptr1_8] "+r"(out_ptr1_8), [in_ptr1] "+r"(in_ptr1),
                  [in_ptr2] "+r"(in_ptr2), [in_ptr3] "+r"(in_ptr3)
                  : [f1] "r"(f1)
                  : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                    "q8", "q9", "q12", "q13", "q14", "q15");
            }
          }
#endif  //__aarch64__
#endif  // __ARM_NEON

          // remain output_width
          for (; o_w < output_w; ++o_w) {
            float sum1 = 0;
            float sum1_2 = 0;
            float sum1_3 = 0;
            float sum1_4 = 0;
            float sum1_5 = 0;
            float sum1_6 = 0;
            float sum1_7 = 0;
            float sum1_8 = 0;
#if __ARM_NEON
            float32x4_t _in_ptr1 = vld1q_f32(in_ptr1);
            float32x4_t _pad_filter1 = vld1q_f32(pad_filter1);
            float32x4_t _pad_filter1_2 = vld1q_f32(pad_filter1_2);
            float32x4_t _pad_filter1_3 = vld1q_f32(pad_filter1_3);
            float32x4_t _pad_filter1_4 = vld1q_f32(pad_filter1_4);
            float32x4_t _pad_filter1_5 = vld1q_f32(pad_filter1_5);
            float32x4_t _pad_filter1_6 = vld1q_f32(pad_filter1_6);
            float32x4_t _pad_filter1_7 = vld1q_f32(pad_filter1_7);
            float32x4_t _pad_filter1_8 = vld1q_f32(pad_filter1_8);

            float32x4_t _sum1 = vmulq_f32(_in_ptr1, _pad_filter1);
            float32x4_t _sum1_2 = vmulq_f32(_in_ptr1, _pad_filter1_2);
            float32x4_t _sum1_3 = vmulq_f32(_in_ptr1, _pad_filter1_3);
            float32x4_t _sum1_4 = vmulq_f32(_in_ptr1, _pad_filter1_4);
            float32x4_t _sum1_5 = vmulq_f32(_in_ptr1, _pad_filter1_5);
            float32x4_t _sum1_6 = vmulq_f32(_in_ptr1, _pad_filter1_6);
            float32x4_t _sum1_7 = vmulq_f32(_in_ptr1, _pad_filter1_7);
            float32x4_t _sum1_8 = vmulq_f32(_in_ptr1, _pad_filter1_8);

            float32x4_t _in_ptr2 = vld1q_f32(in_ptr2);
            float32x4_t _pad_filter2 = vld1q_f32(pad_filter2);
            float32x4_t _pad_filter2_2 = vld1q_f32(pad_filter2_2);
            float32x4_t _pad_filter2_3 = vld1q_f32(pad_filter2_3);
            float32x4_t _pad_filter2_4 = vld1q_f32(pad_filter2_4);
            float32x4_t _pad_filter2_5 = vld1q_f32(pad_filter2_5);
            float32x4_t _pad_filter2_6 = vld1q_f32(pad_filter2_6);
            float32x4_t _pad_filter2_7 = vld1q_f32(pad_filter2_7);
            float32x4_t _pad_filter2_8 = vld1q_f32(pad_filter2_8);

            _sum1 = vmlaq_f32(_sum1, _in_ptr2, _pad_filter2);
            _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr2, _pad_filter2_2);
            _sum1_3 = vmlaq_f32(_sum1_3, _in_ptr2, _pad_filter2_3);
            _sum1_4 = vmlaq_f32(_sum1_4, _in_ptr2, _pad_filter2_4);
            _sum1_5 = vmlaq_f32(_sum1_5, _in_ptr2, _pad_filter2_5);
            _sum1_6 = vmlaq_f32(_sum1_6, _in_ptr2, _pad_filter2_6);
            _sum1_7 = vmlaq_f32(_sum1_7, _in_ptr2, _pad_filter2_7);
            _sum1_8 = vmlaq_f32(_sum1_8, _in_ptr2, _pad_filter2_8);

            float32x4_t _in_ptr3 = vld1q_f32(in_ptr3);
            float32x4_t _pad_filter3 = vld1q_f32(pad_filter3);
            float32x4_t _pad_filter3_2 = vld1q_f32(pad_filter3_2);
            float32x4_t _pad_filter3_3 = vld1q_f32(pad_filter3_3);
            float32x4_t _pad_filter3_4 = vld1q_f32(pad_filter3_4);
            float32x4_t _pad_filter3_5 = vld1q_f32(pad_filter3_5);
            float32x4_t _pad_filter3_6 = vld1q_f32(pad_filter3_6);
            float32x4_t _pad_filter3_7 = vld1q_f32(pad_filter3_7);
            float32x4_t _pad_filter3_8 = vld1q_f32(pad_filter3_8);

            _sum1 = vmlaq_f32(_sum1, _in_ptr3, _pad_filter3);
            _sum1_2 = vmlaq_f32(_sum1_2, _in_ptr3, _pad_filter3_2);
            _sum1_3 = vmlaq_f32(_sum1_3, _in_ptr3, _pad_filter3_3);
            _sum1_4 = vmlaq_f32(_sum1_4, _in_ptr3, _pad_filter3_4);
            _sum1_5 = vmlaq_f32(_sum1_5, _in_ptr3, _pad_filter3_5);
            _sum1_6 = vmlaq_f32(_sum1_6, _in_ptr3, _pad_filter3_6);
            _sum1_7 = vmlaq_f32(_sum1_7, _in_ptr3, _pad_filter3_7);
            _sum1_8 = vmlaq_f32(_sum1_8, _in_ptr3, _pad_filter3_8);

            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
            _sum1_2 = vsetq_lane_f32(sum1_2, _sum1_2, 3);
            _sum1_3 = vsetq_lane_f32(sum1_3, _sum1_3, 3);
            _sum1_4 = vsetq_lane_f32(sum1_4, _sum1_4, 3);
            _sum1_5 = vsetq_lane_f32(sum1_5, _sum1_5, 3);
            _sum1_6 = vsetq_lane_f32(sum1_6, _sum1_6, 3);
            _sum1_7 = vsetq_lane_f32(sum1_7, _sum1_7, 3);
            _sum1_8 = vsetq_lane_f32(sum1_8, _sum1_8, 3);

            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ss1_2 =
                vadd_f32(vget_low_f32(_sum1_2), vget_high_f32(_sum1_2));
            float32x2_t _ss1_3 =
                vadd_f32(vget_low_f32(_sum1_3), vget_high_f32(_sum1_3));
            float32x2_t _ss1_4 =
                vadd_f32(vget_low_f32(_sum1_4), vget_high_f32(_sum1_4));
            float32x2_t _ss1_5 =
                vadd_f32(vget_low_f32(_sum1_5), vget_high_f32(_sum1_5));
            float32x2_t _ss1_6 =
                vadd_f32(vget_low_f32(_sum1_6), vget_high_f32(_sum1_6));
            float32x2_t _ss1_7 =
                vadd_f32(vget_low_f32(_sum1_7), vget_high_f32(_sum1_7));
            float32x2_t _ss1_8 =
                vadd_f32(vget_low_f32(_sum1_8), vget_high_f32(_sum1_8));

            float32x2_t _ssss1_ssss1_2 = vpadd_f32(_ss1, _ss1_2);
            float32x2_t _ssss1_3_ssss1_4 = vpadd_f32(_ss1_3, _ss1_4);
            float32x2_t _ssss1_5_ssss1_6 = vpadd_f32(_ss1_5, _ss1_6);
            float32x2_t _ssss1_7_ssss1_8 = vpadd_f32(_ss1_7, _ss1_8);

            sum1 += vget_lane_f32(_ssss1_ssss1_2, 0);
            sum1_2 += vget_lane_f32(_ssss1_ssss1_2, 1);
            sum1_3 += vget_lane_f32(_ssss1_3_ssss1_4, 0);
            sum1_4 += vget_lane_f32(_ssss1_3_ssss1_4, 1);
            sum1_5 += vget_lane_f32(_ssss1_5_ssss1_6, 0);
            sum1_6 += vget_lane_f32(_ssss1_5_ssss1_6, 1);
            sum1_7 += vget_lane_f32(_ssss1_7_ssss1_8, 0);
            sum1_8 += vget_lane_f32(_ssss1_7_ssss1_8, 1);
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

            sum1_2 += in_ptr1[0] * pad_filter1_2[0];
            sum1_2 += in_ptr1[1] * pad_filter1_2[1];
            sum1_2 += in_ptr1[2] * pad_filter1_2[2];
            sum1_2 += in_ptr2[0] * pad_filter2_2[0];
            sum1_2 += in_ptr2[1] * pad_filter2_2[1];
            sum1_2 += in_ptr2[2] * pad_filter2_2[2];
            sum1_2 += in_ptr3[0] * pad_filter3_2[0];
            sum1_2 += in_ptr3[1] * pad_filter3_2[1];
            sum1_2 += in_ptr3[2] * pad_filter3_2[2];

            sum1_3 += in_ptr1[0] * pad_filter1_3[0];
            sum1_3 += in_ptr1[1] * pad_filter1_3[1];
            sum1_3 += in_ptr1[2] * pad_filter1_3[2];
            sum1_3 += in_ptr2[0] * pad_filter2_3[0];
            sum1_3 += in_ptr2[1] * pad_filter2_3[1];
            sum1_3 += in_ptr2[2] * pad_filter2_3[2];
            sum1_3 += in_ptr3[0] * pad_filter3_3[0];
            sum1_3 += in_ptr3[1] * pad_filter3_3[1];
            sum1_3 += in_ptr3[2] * pad_filter3_3[2];

            sum1_4 += in_ptr1[0] * pad_filter1_4[0];
            sum1_4 += in_ptr1[1] * pad_filter1_4[1];
            sum1_4 += in_ptr1[2] * pad_filter1_4[2];
            sum1_4 += in_ptr2[0] * pad_filter2_4[0];
            sum1_4 += in_ptr2[1] * pad_filter2_4[1];
            sum1_4 += in_ptr2[2] * pad_filter2_4[2];
            sum1_4 += in_ptr3[0] * pad_filter3_4[0];
            sum1_4 += in_ptr3[1] * pad_filter3_4[1];
            sum1_4 += in_ptr3[2] * pad_filter3_4[2];

            sum1_5 += in_ptr1[0] * pad_filter1_5[0];
            sum1_5 += in_ptr1[1] * pad_filter1_5[1];
            sum1_5 += in_ptr1[2] * pad_filter1_5[2];
            sum1_5 += in_ptr2[0] * pad_filter2_5[0];
            sum1_5 += in_ptr2[1] * pad_filter2_5[1];
            sum1_5 += in_ptr2[2] * pad_filter2_5[2];
            sum1_5 += in_ptr3[0] * pad_filter3_5[0];
            sum1_5 += in_ptr3[1] * pad_filter3_5[1];
            sum1_5 += in_ptr3[2] * pad_filter3_5[2];

            sum1_6 += in_ptr1[0] * pad_filter1_6[0];
            sum1_6 += in_ptr1[1] * pad_filter1_6[1];
            sum1_6 += in_ptr1[2] * pad_filter1_6[2];
            sum1_6 += in_ptr2[0] * pad_filter2_6[0];
            sum1_6 += in_ptr2[1] * pad_filter2_6[1];
            sum1_6 += in_ptr2[2] * pad_filter2_6[2];
            sum1_6 += in_ptr3[0] * pad_filter3_6[0];
            sum1_6 += in_ptr3[1] * pad_filter3_6[1];
            sum1_6 += in_ptr3[2] * pad_filter3_6[2];

            sum1_7 += in_ptr1[0] * pad_filter1_7[0];
            sum1_7 += in_ptr1[1] * pad_filter1_7[1];
            sum1_7 += in_ptr1[2] * pad_filter1_7[2];
            sum1_7 += in_ptr2[0] * pad_filter2_7[0];
            sum1_7 += in_ptr2[1] * pad_filter2_7[1];
            sum1_7 += in_ptr2[2] * pad_filter2_7[2];
            sum1_7 += in_ptr3[0] * pad_filter3_7[0];
            sum1_7 += in_ptr3[1] * pad_filter3_7[1];
            sum1_7 += in_ptr3[2] * pad_filter3_7[2];

            sum1_8 += in_ptr1[0] * pad_filter1_8[0];
            sum1_8 += in_ptr1[1] * pad_filter1_8[1];
            sum1_8 += in_ptr1[2] * pad_filter1_8[2];
            sum1_8 += in_ptr2[0] * pad_filter2_8[0];
            sum1_8 += in_ptr2[1] * pad_filter2_8[1];
            sum1_8 += in_ptr2[2] * pad_filter2_8[2];
            sum1_8 += in_ptr3[0] * pad_filter3_8[0];
            sum1_8 += in_ptr3[1] * pad_filter3_8[1];
            sum1_8 += in_ptr3[2] * pad_filter3_8[2];
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
              pad_filter1_2--;
              pad_filter2_2--;
              pad_filter3_2--;

              pad_filter1_3--;
              pad_filter2_3--;
              pad_filter3_3--;
              pad_filter1_4--;
              pad_filter2_4--;
              pad_filter3_4--;

              pad_filter1_5--;
              pad_filter2_5--;
              pad_filter3_5--;
              pad_filter1_6--;
              pad_filter2_6--;
              pad_filter3_6--;

              pad_filter1_7--;
              pad_filter2_7--;
              pad_filter3_7--;
              pad_filter1_8--;
              pad_filter2_8--;
              pad_filter3_8--;

              in_ptr1++;
              in_ptr2++;
              in_ptr3++;
            } else if (input_w <= 3 || o_w < valid_w_start ||
                       o_w > valid_w_end) {
              pad_filter1 -= 2;
              pad_filter2 -= 2;
              pad_filter3 -= 2;
              pad_filter1_2 -= 2;
              pad_filter2_2 -= 2;
              pad_filter3_2 -= 2;

              pad_filter1_3 -= 2;
              pad_filter2_3 -= 2;
              pad_filter3_3 -= 2;
              pad_filter1_4 -= 2;
              pad_filter2_4 -= 2;
              pad_filter3_4 -= 2;

              pad_filter1_5 -= 2;
              pad_filter2_5 -= 2;
              pad_filter3_5 -= 2;
              pad_filter1_6 -= 2;
              pad_filter2_6 -= 2;
              pad_filter3_6 -= 2;

              pad_filter1_7 -= 2;
              pad_filter2_7 -= 2;
              pad_filter3_7 -= 2;
              pad_filter1_8 -= 2;
              pad_filter2_8 -= 2;
              pad_filter3_8 -= 2;
            } else {
              in_ptr1 += 2;
              in_ptr2 += 2;
              in_ptr3 += 2;
            }
            *out_ptr1 += sum1;
            *out_ptr1_2 += sum1_2;
            *out_ptr1_3 += sum1_3;
            *out_ptr1_4 += sum1_4;
            *out_ptr1_5 += sum1_5;
            *out_ptr1_6 += sum1_6;
            *out_ptr1_7 += sum1_7;
            *out_ptr1_8 += sum1_8;

            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }
            if (out_ptr1_2[0] < relu_value) {
              *out_ptr1_2 = relu_value;
            }
            if (out_ptr1_3[0] < relu_value) {
              *out_ptr1_3 = relu_value;
            }
            if (out_ptr1_4[0] < relu_value) {
              *out_ptr1_4 = relu_value;
            }
            if (out_ptr1_5[0] < relu_value) {
              *out_ptr1_5 = relu_value;
            }
            if (out_ptr1_6[0] < relu_value) {
              *out_ptr1_6 = relu_value;
            }
            if (out_ptr1_7[0] < relu_value) {
              *out_ptr1_7 = relu_value;
            }
            if (out_ptr1_8[0] < relu_value) {
              *out_ptr1_8 = relu_value;
            }
            out_ptr1++;
            out_ptr1_2++;
            out_ptr1_3++;
            out_ptr1_4++;
            out_ptr1_5++;
            out_ptr1_6++;
            out_ptr1_7++;
            out_ptr1_8++;
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
            pad_filter1_2 -= remain_stride_w;
            pad_filter2_2 -= remain_stride_w;
            pad_filter3_2 -= remain_stride_w;

            pad_filter1_3 -= remain_stride_w;
            pad_filter2_3 -= remain_stride_w;
            pad_filter3_3 -= remain_stride_w;
            pad_filter1_4 -= remain_stride_w;
            pad_filter2_4 -= remain_stride_w;
            pad_filter3_4 -= remain_stride_w;

            pad_filter1_5 -= remain_stride_w;
            pad_filter2_5 -= remain_stride_w;
            pad_filter3_5 -= remain_stride_w;
            pad_filter1_6 -= remain_stride_w;
            pad_filter2_6 -= remain_stride_w;
            pad_filter3_6 -= remain_stride_w;

            pad_filter1_7 -= remain_stride_w;
            pad_filter2_7 -= remain_stride_w;
            pad_filter3_7 -= remain_stride_w;
            pad_filter1_8 -= remain_stride_w;
            pad_filter2_8 -= remain_stride_w;
            pad_filter3_8 -= remain_stride_w;
          } else if (input_h <= 3 || o_h < valid_h_start || o_h > valid_h_end) {
            in_ptr1 -= input_w - 3;
            in_ptr2 -= input_w - 3;
            in_ptr3 -= input_w - 3;

            pad_filter1 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter1_2 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2_2 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3_2 -= 3 + 2 * padding_w + remain_stride_w;

            pad_filter1_3 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2_3 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3_3 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter1_4 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2_4 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3_4 -= 3 + 2 * padding_w + remain_stride_w;

            pad_filter1_5 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2_5 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3_5 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter1_6 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2_6 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3_6 -= 3 + 2 * padding_w + remain_stride_w;

            pad_filter1_7 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2_7 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3_7 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter1_8 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter2_8 -= 3 + 2 * padding_w + remain_stride_w;
            pad_filter3_8 -= 3 + 2 * padding_w + remain_stride_w;
          } else {
            pad_filter1 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter1_2 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2_2 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3_2 += 3 + 2 * padding_w - remain_stride_w;

            pad_filter1_3 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2_3 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3_3 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter1_4 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2_4 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3_4 += 3 + 2 * padding_w - remain_stride_w;

            pad_filter1_5 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2_5 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3_5 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter1_6 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2_6 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3_6 += 3 + 2 * padding_w - remain_stride_w;

            pad_filter1_7 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2_7 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3_7 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter1_8 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter2_8 += 3 + 2 * padding_w - remain_stride_w;
            pad_filter3_8 += 3 + 2 * padding_w - remain_stride_w;

            in_ptr1 += input_w + 3;
            in_ptr2 += input_w + 3;
            in_ptr3 += input_w + 3;
          }
        }

        filter_data_ch += filter_ch_size;
        filter_data_ch_2 += filter_ch_size;
        filter_data_ch_3 += filter_ch_size;
        filter_data_ch_4 += filter_ch_size;
        filter_data_ch_5 += filter_ch_size;
        filter_data_ch_6 += filter_ch_size;
        filter_data_ch_7 += filter_ch_size;
        filter_data_ch_8 += filter_ch_size;
        input_data_ch += in_ch_size;
      }
    }

    int out_ch_remain = output_ch_d8 * 8;

    // remain output_channel
#pragma omp parallel for
    for (int o_c = out_ch_remain; o_c < output_ch; ++o_c) {
      std::atomic<float> relu_value{0};

      const float *reluptr;
      const float *f1, *f9;
      const float *in_ptr1, *in_ptr2, *in_ptr3;
      const float *pad_filter1, *pad_filter2, *pad_filter3;
      float pad_filter_arr[pad_filter_ch_size] = {0};
      float *output_data_ch;
      const float *input_data_ch;
      const float *filter_data_ch;

      filter_data_ch = filter_data + o_c * filter_ch_size * input_ch;
      input_data_ch = input_data;
      output_data_ch = output_data + o_c * out_ch_size;

      for (int i_c = 0; i_c < input_ch; ++i_c) {
        float relu_arr[4];
        reluptr = relu_arr;
        if (if_relu && i_c == input_ch - 1) {
          relu_value = 0;
        } else {
          relu_value = -FLT_MAX;
        }
        relu_arr[0] = relu_value;
        relu_arr[1] = relu_value;
        relu_arr[2] = relu_value;
        relu_arr[3] = relu_value;

        f1 = filter_data_ch;
        f9 = f1 + 8;

        if (!if_nopadding) {
          pad_filter_arr[pad_filter_ch_size] = {0};
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

            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }
            out_ptr1++;
          }
            // valid
#if __ARM_NEON
#if __aarch64__
          if (o_h > valid_h_start && o_h < valid_h_end) {
            int o_w_dim4 = (valid_w_end - valid_w_start - 1) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(
                  "prfm   pldl1keep, [%[f1], #256]            \n\t"
                  "prfm   pldl1keep, [%[f9], #256]            \n\t"
                  "prfm   pldl1keep, [%[reluptr], #256]       \n\t"

                  "ld1   {v0.4s, v1.4s}, [%[f1]]              \n\t"
                  "ld1   {v4.s}[0], [%[f9]]                   \n\t"
                  "ld1   {v16.4s}, [%[reluptr]]               \n\t"

                  "0:                                         \n\t"
                  // load out_ptr
                  "prfm   pldl1keep, [%[out_ptr1], #128]      \n\t"
                  "ld1   {v12.4s}, [%[out_ptr1]]              \n\t"

                  // in_ptr1 multiply
                  "prfm   pldl1keep, [%[in_ptr1], #256]       \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[in_ptr1]], #32    \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[in_ptr1]]         \n\t"

                  "fmla   v12.4s, v5.4s, v0.4s[0]             \n\t"
                  "fmla   v14.4s, v5.4s, v2.4s[0]             \n\t"

                  "ext    v8.16b, v5.16b, v7.16b, #4          \n\t"
                  "fmul   v13.4s, v6.4s, v0.4s[1]             \n\t"
                  "fmla   v12.4s, v8.4s, v0.4s[2]             \n\t"

                  "ld2   {v5.4s, v6.4s}, [%[in_ptr2]], #32    \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[in_ptr2]]         \n\t"

                  // in_ptr2 multiply
                  "fmla   v13.4s, v5.4s, v0.4s[3]             \n\t"
                  "ext    v8.16b, v5.16b, v7.16b, #4          \n\t"
                  "fmla   v12.4s, v6.4s, v1.4s[0]             \n\t"

                  "fmla   v13.4s, v8.4s, v1.4s[1]             \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[in_ptr3]], #32    \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[in_ptr3]]         \n\t"

                  // in_ptr3 multiply
                  "fmla   v12.4s, v5.4s, v1.4s[2]             \n\t"
                  "ext    v8.16b, v5.16b, v7.16b, #4          \n\t"

                  "fmla   v13.4s, v6.4s, v1.4s[3]             \n\t"
                  "fmla   v12.4s, v8.4s, v4.4s[0]             \n\t"

                  // store out_ptr
                  "fadd   v12.4s, v12.4s, v13.4s              \n\t"
                  "fmax   v12.4s, v12.4s, v16.4s              \n\t"
                  "st1   {v12.4s}, [%[out_ptr1]], #16         \n\t"

                  // cycle
                  "subs       %[o_w_dim4], %[o_w_dim4], #1      \n\t"
                  "bne        0b                              \n\t"

                  : [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                    [in_ptr1] "+r"(in_ptr1), [in_ptr2] "+r"(in_ptr2),
                    [in_ptr3] "+r"(in_ptr3)
                  : [f1] "r"(f1), [f9] "r"(f9), [reluptr] "r"(reluptr)
                  : "memory", "v0", "v1", "v4", "v5", "v6", "v7", "v8", "v9",
                    "v12", "v13", "v16");
            }
          }
#else
          if (o_h > valid_h_start && o_h < valid_h_end) {
            int o_w_dim4 = (valid_w_end - valid_w_start - 1) >> 2;
            o_w += o_w_dim4 * 4;

            if (o_w_dim4 > 0) {
              asm volatile(
                  "pld        [%[f1], #256]                   \n\t"
                  "pld        [%[f9], #256]                   \n\t"
                  "pld        [%[reluptr], #128]              \n\t"

                  "vld1.32   {d0-d3}, [%[f1]]                 \n\t"
                  "vld1.32   {d8[0]}, [%[f9]]                 \n\t"
                  "vld1.32   {d18, d19}, [%[reluptr]]         \n\t"

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
                  "subs       %[o_w_dim4], #1                 \n\t"

                  "vadd.f32   q14, q14, q15                   \n\t"
                  "vadd.f32   q12, q12, q14                   \n\t"
                  "vmax.f32   q12, q12, q9                    \n\t"
                  "vst1.f32   {d24, d25}, [%[out_ptr1]]!      \n\t"

                  // cycle
                  "bne        0b                              \n\t"
                  "subs       %[in_ptr1], %[in_ptr1], #32     \n\t"

                  : [o_w_dim4] "+r"(o_w_dim4), [out_ptr1] "+r"(out_ptr1),
                    [in_ptr1] "+r"(in_ptr1), [in_ptr2] "+r"(in_ptr2),
                    [in_ptr3] "+r"(in_ptr3)
                  : [f1] "r"(f1), [f9] "r"(f9), [reluptr] "r"(reluptr)
                  : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                    "q8", "q9", "q10", "q12", "q13", "q14", "q15");
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

            if (out_ptr1[0] < relu_value) {
              *out_ptr1 = relu_value;
            }
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
