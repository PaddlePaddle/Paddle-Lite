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
  const int batch_size = input->dims()[0];
  const int input_channels = input->dims()[1];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const int output_channels = output->dims()[1];
  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  const int padding_height = paddings[0];
  const int padding_width = paddings[1];

  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();
  const float *filter_data = filter->data<float>();

  const int input_channel_stride = input_height * input_width;
  const int input_batch_stride = input_channels * input_channel_stride;
  const int output_channel_stride = output_height * output_width;
  const int output_batch_stride = output_channels * output_channel_stride;
  const int filter_channel_stride = 9;
  const int ffilter_length = (2 * padding_height + 3) * (2 * padding_width + 3);
  const int ffilter_start =
      2 * padding_height * (2 * padding_width + 3) + 2 * padding_width;
  const int ffilter_width = 3 + padding_width * 2;
  const float *bias_data;
  bool isnopadding = false;

  if (if_bias) {
    bias_data = bias->data<float>();
  }
#if __ARM_NEON
  float *outptr = output_data;
  for (int i = 0; i < output_channels; ++i) {
    float32x4_t _bias;
    if (if_bias) {
      _bias = vld1q_dup_f32(bias_data + i);
    } else {
      _bias = vdupq_n_f32(0.0);
    }
    int dim4 = output_channel_stride >> 2;
    int lef4 = output_channel_stride & 3;

    for (int j = 0; j < dim4; ++j) {
      vst1q_f32(outptr, _bias);
      outptr += 4;
    }
    if (lef4 != 0) {
      if (lef4 >= 1) {
        vst1q_lane_f32(outptr, _bias, 0);
        outptr++;
      }
      if (lef4 >= 2) {
        vst1q_lane_f32(outptr, _bias, 1);
        outptr++;
      }
      if (lef4 >= 3) {
        vst1q_lane_f32(outptr, _bias, 2);
        outptr++;
      }
    }
  }
#else
  int k = 0;
#pragma omp parallel for
  for (int i = 0; i < output_channels; ++i) {
    for (int j = 0; j < output_channel_stride; ++j) {
      if (if_bias) {
        output_data[k++] = bias_data[i];
      } else {
        output_data[k++] = 0;
      }
    }
  }
#endif
  if (padding_height == 0 && padding_width == 0) {
    isnopadding = true;
  }

  for (int bs = 0; bs < batch_size; ++bs) {
    int output_channels_d2 = output_channels >> 1;

#pragma omp parallel for
    for (int oc2 = 0; oc2 < output_channels_d2; ++oc2) {
      std::atomic<float> reluvalue{0};
      const float *reluptr;
      int oc = oc2 * 2;
      bool issamefilter;
      const float *f1;
      const float *f1_2;
      const float *inptr1, *inptr2, *inptr3, *inptr4;
      const float *ffilter0, *ffilter1, *ffilter2, *ffilter3;
      const float *ffilter0_2, *ffilter1_2, *ffilter2_2, *ffilter3_2;
      float ffilterarray[ffilter_length] = {0};
      float ffilterarray_2[ffilter_length] = {0};

      float *output_data_ch;
      float *output_data_ch_2;
      const float *input_data_ch;
      const float *filter_data_ch;
      const float *filter_data_ch_2;

      filter_data_ch =
          filter_data + oc * filter_channel_stride * input_channels;
      filter_data_ch_2 =
          filter_data + (oc + 1) * filter_channel_stride * input_channels;

      input_data_ch = input_data;
      output_data_ch = output_data + oc * output_channel_stride;
      output_data_ch_2 = output_data + (oc + 1) * output_channel_stride;

      for (int ic = 0; ic < input_channels; ++ic) {
        float reluarr[4];
        reluptr = reluarr;
        if (if_relu && ic == input_channels - 1) {
          reluvalue = 0;
        } else {
          reluvalue = -FLT_MAX;
        }
        reluarr[0] = reluvalue;
        reluarr[1] = reluvalue;
        reluarr[2] = reluvalue;
        reluarr[3] = reluvalue;
        f1 = filter_data_ch;
        f1_2 = filter_data_ch_2;

        if (!isnopadding) {
          ffilterarray[ffilter_length] = {0};
          for (int i = 0; i < 9; i++) {
            int j = i / 3 * (2 * padding_width + 3) + i % 3 +
                    padding_height * 3 +
                    padding_width * (2 * padding_height + 1);
            ffilterarray[j] = filter_data_ch[i];
            ffilterarray_2[j] = filter_data_ch_2[i];
          }
          ffilter1 = ffilterarray;
          ffilter1 += ffilter_start;
          ffilter0 = ffilter1 - ffilter_width;
          ffilter2 = ffilter1 + ffilter_width;
          ffilter3 = ffilter2 + ffilter_width;

          ffilter1_2 = ffilterarray_2;
          ffilter1_2 += ffilter_start;
          ffilter0_2 = ffilter1_2 - ffilter_width;
          ffilter2_2 = ffilter1_2 + ffilter_width;
          ffilter3_2 = ffilter2_2 + ffilter_width;
        } else {
          ffilter1 = filter_data_ch;
          ffilter2 = ffilter1 + 3;
          ffilter3 = ffilter2 + 3;

          ffilter1_2 = filter_data_ch_2;
          ffilter2_2 = ffilter1_2 + 3;
          ffilter3_2 = ffilter2_2 + 3;
        }
        float *outptr1, *outptr2;
        float *outptr1_2, *outptr2_2;

        outptr1 = output_data_ch;
        outptr2 = outptr1 + output_width;
        outptr1_2 = output_data_ch_2;
        outptr2_2 = outptr1_2 + output_width;

        inptr1 = input_data_ch;
        inptr2 = inptr1 + input_width;
        inptr3 = inptr2 + input_width;
        inptr4 = inptr3 + input_width;

        int oh = 0;
        for (; oh < output_height - 1; oh = oh + 2) {
          if (!isnopadding && (oh < padding_height ||
                               oh > output_height - padding_height - 2)) {
            issamefilter = false;
          } else {
            issamefilter = true;
          }
          int ow = 0;
          for (; ow < padding_width; ++ow) {
            float sum1 = 0;
            float sum2 = 0;
            float sum1_2 = 0;
            float sum2_2 = 0;

            if (issamefilter) {
#if __ARM_NEON
              float32x4_t _inptr1 = vld1q_f32(inptr1);
              float32x4_t _ffilter1 = vld1q_f32(ffilter1);
              float32x4_t _ffilter1_2 = vld1q_f32(ffilter1_2);
              float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);
              float32x4_t _sum1_2 = vmulq_f32(_inptr1, _ffilter1_2);

              float32x4_t _inptr2 = vld1q_f32(inptr2);
              float32x4_t _ffilter2 = vld1q_f32(ffilter2);
              float32x4_t _ffilter2_2 = vld1q_f32(ffilter2_2);
              _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);
              _sum1_2 = vmlaq_f32(_sum1_2, _inptr2, _ffilter2_2);
              float32x4_t _sum2 = vmulq_f32(_inptr2, _ffilter1);
              float32x4_t _sum2_2 = vmulq_f32(_inptr2, _ffilter1_2);

              float32x4_t _inptr3 = vld1q_f32(inptr3);
              float32x4_t _ffilter3 = vld1q_f32(ffilter3);
              float32x4_t _ffilter3_2 = vld1q_f32(ffilter3_2);
              _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
              _sum1_2 = vmlaq_f32(_sum1_2, _inptr3, _ffilter3_2);
              _sum2 = vmlaq_f32(_sum2, _inptr3, _ffilter2);
              _sum2_2 = vmlaq_f32(_sum2_2, _inptr3, _ffilter2_2);

              float32x4_t _inptr4 = vld1q_f32(inptr4);
              _sum2 = vmlaq_f32(_sum2, _inptr4, _ffilter3);
              _sum2_2 = vmlaq_f32(_sum2_2, _inptr4, _ffilter3_2);

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
              sum1 += inptr1[0] * ffilter1[0];
              sum1 += inptr1[1] * ffilter1[1];
              sum1 += inptr1[2] * ffilter1[2];
              sum1 += inptr2[0] * ffilter2[0];
              sum1 += inptr2[1] * ffilter2[1];
              sum1 += inptr2[2] * ffilter2[2];
              sum1 += inptr3[0] * ffilter3[0];
              sum1 += inptr3[1] * ffilter3[1];
              sum1 += inptr3[2] * ffilter3[2];

              sum2 += inptr2[0] * ffilter1[0];
              sum2 += inptr2[1] * ffilter1[1];
              sum2 += inptr2[2] * ffilter1[2];
              sum2 += inptr3[0] * ffilter2[0];
              sum2 += inptr3[1] * ffilter2[1];
              sum2 += inptr3[2] * ffilter2[2];
              sum2 += inptr4[0] * ffilter3[0];
              sum2 += inptr4[1] * ffilter3[1];
              sum2 += inptr4[2] * ffilter3[2];

              sum1_2 += inptr1[0] * ffilter1_2[0];
              sum1_2 += inptr1[1] * ffilter1_2[1];
              sum1_2 += inptr1[2] * ffilter1_2[2];
              sum1_2 += inptr2[0] * ffilter2_2[0];
              sum1_2 += inptr2[1] * ffilter2_2[1];
              sum1_2 += inptr2[2] * ffilter2_2[2];
              sum1_2 += inptr3[0] * ffilter3_2[0];
              sum1_2 += inptr3[1] * ffilter3_2[1];
              sum1_2 += inptr3[2] * ffilter3_2[2];

              sum2_2 += inptr2[0] * ffilter1_2[0];
              sum2_2 += inptr2[1] * ffilter1_2[1];
              sum2_2 += inptr2[2] * ffilter1_2[2];
              sum2_2 += inptr3[0] * ffilter2_2[0];
              sum2_2 += inptr3[1] * ffilter2_2[1];
              sum2_2 += inptr3[2] * ffilter2_2[2];
              sum2_2 += inptr4[0] * ffilter3_2[0];
              sum2_2 += inptr4[1] * ffilter3_2[1];
              sum2_2 += inptr4[2] * ffilter3_2[2];
#endif
            } else {
#if __ARM_NEON
              float32x4_t _inptr1 = vld1q_f32(inptr1);
              float32x4_t _ffilter1 = vld1q_f32(ffilter1);
              float32x4_t _ffilter1_2 = vld1q_f32(ffilter1_2);
              float32x4_t _ffilter0 = vld1q_f32(ffilter0);
              float32x4_t _ffilter0_2 = vld1q_f32(ffilter0_2);

              float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);
              float32x4_t _sum1_2 = vmulq_f32(_inptr1, _ffilter1_2);
              float32x4_t _sum2 = vmulq_f32(_inptr1, _ffilter0);
              float32x4_t _sum2_2 = vmulq_f32(_inptr1, _ffilter0_2);

              float32x4_t _inptr2 = vld1q_f32(inptr2);
              float32x4_t _ffilter2 = vld1q_f32(ffilter2);
              float32x4_t _ffilter2_2 = vld1q_f32(ffilter2_2);

              _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);
              _sum1_2 = vmlaq_f32(_sum1_2, _inptr2, _ffilter2_2);
              _sum2 = vmlaq_f32(_sum2, _inptr2, _ffilter1);
              _sum2_2 = vmlaq_f32(_sum2_2, _inptr2, _ffilter1_2);

              float32x4_t _inptr3 = vld1q_f32(inptr3);
              float32x4_t _ffilter3 = vld1q_f32(ffilter3);
              float32x4_t _ffilter3_2 = vld1q_f32(ffilter3_2);

              _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
              _sum1_2 = vmlaq_f32(_sum1_2, _inptr3, _ffilter3_2);
              _sum2 = vmlaq_f32(_sum2, _inptr3, _ffilter2);
              _sum2_2 = vmlaq_f32(_sum2_2, _inptr3, _ffilter2_2);

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
              sum1 += inptr1[0] * ffilter1[0];
              sum1 += inptr1[1] * ffilter1[1];
              sum1 += inptr1[2] * ffilter1[2];
              sum1 += inptr2[0] * ffilter2[0];
              sum1 += inptr2[1] * ffilter2[1];
              sum1 += inptr2[2] * ffilter2[2];
              sum1 += inptr3[0] * ffilter3[0];
              sum1 += inptr3[1] * ffilter3[1];
              sum1 += inptr3[2] * ffilter3[2];

              sum2 += inptr1[0] * ffilter0[0];
              sum2 += inptr1[1] * ffilter0[1];
              sum2 += inptr1[2] * ffilter0[2];
              sum2 += inptr2[0] * ffilter1[0];
              sum2 += inptr2[1] * ffilter1[1];
              sum2 += inptr2[2] * ffilter1[2];
              sum2 += inptr3[0] * ffilter2[0];
              sum2 += inptr3[1] * ffilter2[1];
              sum2 += inptr3[2] * ffilter2[2];

              sum1_2 += inptr1[0] * ffilter1_2[0];
              sum1_2 += inptr1[1] * ffilter1_2[1];
              sum1_2 += inptr1[2] * ffilter1_2[2];
              sum1_2 += inptr2[0] * ffilter2_2[0];
              sum1_2 += inptr2[1] * ffilter2_2[1];
              sum1_2 += inptr2[2] * ffilter2_2[2];
              sum1_2 += inptr3[0] * ffilter3_2[0];
              sum1_2 += inptr3[1] * ffilter3_2[1];
              sum1_2 += inptr3[2] * ffilter3_2[2];

              sum2_2 += inptr1[0] * ffilter0_2[0];
              sum2_2 += inptr1[1] * ffilter0_2[1];
              sum2_2 += inptr1[2] * ffilter0_2[2];
              sum2_2 += inptr2[0] * ffilter1_2[0];
              sum2_2 += inptr2[1] * ffilter1_2[1];
              sum2_2 += inptr2[2] * ffilter1_2[2];
              sum2_2 += inptr3[0] * ffilter2_2[0];
              sum2_2 += inptr3[1] * ffilter2_2[1];
              sum2_2 += inptr3[2] * ffilter2_2[2];
#endif
            }
            if (!isnopadding &&
                (ow < padding_width || ow > output_width - padding_width - 2)) {
              ffilter0--;
              ffilter1--;
              ffilter2--;
              ffilter3--;

              ffilter0_2--;
              ffilter1_2--;
              ffilter2_2--;
              ffilter3_2--;
            } else {
              inptr1++;
              inptr2++;
              inptr3++;
              inptr4++;
            }
            *outptr1 += sum1;
            *outptr2 += sum2;
            *outptr1_2 += sum1_2;
            *outptr2_2 += sum2_2;

            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }
            if (outptr2[0] < reluvalue) {
              *outptr2 = reluvalue;
            }
            if (outptr1_2[0] < reluvalue) {
              *outptr1_2 = reluvalue;
            }
            if (outptr2_2[0] < reluvalue) {
              *outptr2_2 = reluvalue;
            }
            outptr1++;
            outptr2++;
            outptr1_2++;
            outptr2_2++;
          }
#if __ARM_NEON
#if __aarch64__
          if (issamefilter) {
            int ow_dim4 = (output_width - 2 * padding_width) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
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

                  "prfm   pldl1keep, [%[inptr1], #192]      \n\t"
                  "prfm   pldl1keep, [%[inptr4], #192]      \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[inptr1]]        \n\t"
                  "add        %[inptr1],%[inptr1], #16      \n\t"

                  "ld1   {v6.d}[1], [%[inptr4]]             \n\t"
                  "add        %[inptr4],%[inptr4], #8       \n\t"
                  "ld1   {v7.4s}, [%[inptr4]]               \n\t"
                  "add        %[inptr4],%[inptr4], #8       \n\t"

                  "0:                                       \n\t"
                  // load outptr
                  "prfm   pldl1keep, [%[outptr1], #128]     \n\t"
                  "prfm   pldl1keep, [%[outptr1_2], #128]   \n\t"
                  "prfm   pldl1keep, [%[outptr2], #128]     \n\t"
                  "prfm   pldl1keep, [%[outptr2_2], #128]   \n\t"

                  "ld1   {v12.4s}, [%[outptr1]]             \n\t"
                  "ld1   {v13.4s}, [%[outptr1_2]]           \n\t"
                  "ld1   {v14.4s}, [%[outptr2]]             \n\t"
                  "ld1   {v15.4s}, [%[outptr2_2]]           \n\t"

                  // inptr1 and inptr4 multiply
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

                  "ld1   {v5.4s, v6.4s}, [%[inptr2]]        \n\t"
                  "fmla   v12.4s, v10.4s, v0.4s[2]          \n\t"
                  "fmla   v13.4s, v10.4s, v2.4s[2]          \n\t"

                  "add        %[inptr2],%[inptr2], #16      \n\t"
                  "fmla   v14.4s, v11.4s, v1.4s[3]          \n\t"
                  "fmla   v15.4s, v11.4s, v3.4s[3]          \n\t"

                  // inptr2 multiply
                  "ext    v8.16b,  v5.16b, v6.16b, #4       \n\t"
                  "fmla   v12.4s, v5.4s, v0.4s[3]           \n\t"
                  "fmla   v13.4s, v5.4s, v2.4s[3]           \n\t"

                  "fmla   v14.4s, v5.4s, v0.4s[0]           \n\t"
                  "fmla   v15.4s, v5.4s, v2.4s[0]           \n\t"

                  "ext    v9.16b,  v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v1.4s[0]           \n\t"
                  "fmla   v13.4s, v8.4s, v3.4s[0]           \n\t"

                  "ld1   {v6.d}[1], [%[inptr3]]             \n\t"
                  "add        %[inptr3],%[inptr3], #8       \n\t"
                  "fmla   v14.4s, v8.4s, v0.4s[1]           \n\t"
                  "fmla   v15.4s, v8.4s, v2.4s[1]           \n\t"

                  "ld1   {v7.4s}, [%[inptr3]]               \n\t"
                  "add        %[inptr3],%[inptr3], #8       \n\t"

                  "fmla   v12.4s, v9.4s, v1.4s[1]           \n\t"
                  "fmla   v13.4s, v9.4s, v3.4s[1]           \n\t"

                  "ext    v10.16b, v6.16b, v7.16b, #8       \n\t"
                  "fmla   v14.4s, v9.4s, v0.4s[2]           \n\t"
                  "fmla   v15.4s, v9.4s, v2.4s[2]           \n\t"

                  // inptr3 multiply
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

                  "prfm   pldl1keep, [%[inptr1], #192]      \n\t"
                  "fmla   v14.4s, v11.4s, v1.4s[0]          \n\t"
                  "fmla   v15.4s, v11.4s, v3.4s[0]          \n\t"

                  // store outptr

                  "prfm   pldl1keep, [%[inptr4], #192]      \n\t"
                  "fmax   v12.4s,v12.4s, v16.4s             \n\t"
                  "fmax   v13.4s,v13.4s, v16.4s             \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[inptr1]]        \n\t"
                  "add        %[inptr1],%[inptr1], #16      \n\t"
                  "fmax   v14.4s,v14.4s, v16.4s             \n\t"
                  "fmax   v15.4s,v15.4s, v16.4s             \n\t"
                  "st1   {v12.4s}, [%[outptr1]], #16        \n\t"

                  "ld1   {v6.d}[1], [%[inptr4]]             \n\t"
                  "add        %[inptr4],%[inptr4], #8       \n\t"
                  "st1   {v13.4s}, [%[outptr1_2]], #16      \n\t"

                  "ld1   {v7.4s}, [%[inptr4]]               \n\t"
                  "add        %[inptr4],%[inptr4], #8       \n\t"
                  "st1   {v14.4s}, [%[outptr2]], #16        \n\t"

                  "subs       %[ow_dim4],%[ow_dim4], #1     \n\t"
                  "st1   {v15.4s}, [%[outptr2_2]], #16      \n\t"

                  // cycle
                  "bne        0b                            \n\t"
                  "sub       %[inptr1],%[inptr1], #16       \n\t"
                  "sub       %[inptr4],%[inptr4], #16       \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [outptr2] "+r"(outptr2), [outptr1_2] "+r"(outptr1_2),
                    [outptr2_2] "+r"(outptr2_2), [inptr1] "+r"(inptr1),
                    [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
                    [inptr4] "+r"(inptr4)
                  : [f1] "r"(f1), [f1_2] "r"(f1_2), [reluptr] "r"(reluptr)
                  : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                    "v16");
            }
          }
          if (!isnopadding && ow == output_width - padding_width) {
            ffilter0--;
            ffilter1--;
            ffilter2--;
            ffilter3--;

            ffilter0_2--;
            ffilter1_2--;
            ffilter2_2--;
            ffilter3_2--;

            inptr1--;
            inptr2--;
            inptr3--;
            inptr4--;
          }
#else
          if (issamefilter) {
            int ow_dim4 = (output_width - 2 * padding_width) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
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

                  "pld        [%[inptr1], #192]             \n\t"
                  "pld        [%[inptr4], #192]             \n\t"

                  "vld1.f32   {d10-d12}, [%[inptr1]]        \n\t"
                  "add        %[inptr1], #16                \n\t"

                  "vld1.f32   {d13-d15}, [%[inptr4]]        \n\t"
                  "add        %[inptr4], #16                \n\t"

                  "0:                                       \n\t"
                  // load outptr
                  "pld        [%[outptr1], #128]            \n\t"
                  "pld        [%[outptr1_2], #128]          \n\t"
                  "pld        [%[outptr2], #128]            \n\t"
                  "pld        [%[outptr2_2], #128]          \n\t"

                  "vld1.f32   {d24, d25}, [%[outptr1]]      \n\t"
                  "vld1.f32   {d26, d27}, [%[outptr1_2]]    \n\t"
                  "vld1.f32   {d28, d29}, [%[outptr2]]      \n\t"
                  "vld1.f32   {d30, d31}, [%[outptr2_2]]    \n\t"

                  // inptr1 + inptr4 multiply

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

                  "vld1.f32   {d10-d12}, [%[inptr2]]        \n\t"
                  "add        %[inptr2], #16                \n\t"
                  "vmla.f32   q12, q10, d1[0]               \n\t"
                  "vmla.f32   q13, q10, d5[0]               \n\t"

                  "vmla.f32   q14, q11, d3[1]               \n\t"
                  "vmla.f32   q15, q11, d7[1]               \n\t"

                  // inptr2 multiply
                  "vext.32    q8, q5, q6, #1                \n\t"
                  "vmla.f32   q12, q5, d1[1]                \n\t"
                  "vmla.f32   q13, q5, d5[1]                \n\t"

                  "vmla.f32   q14, q5, d0[0]                \n\t"
                  "vmla.f32   q15, q5, d4[0]                \n\t"

                  "vext.32    q9, q5, q6, #2                \n\t"
                  "vmla.f32   q12, q8, d2[0]                \n\t"
                  "vmla.f32   q13, q8, d6[0]                \n\t"

                  "vld1.f32   {d13-d15}, [%[inptr3]]        \n\t"
                  "add        %[inptr3], #16                \n\t"
                  "vmla.f32   q14, q8, d0[1]                \n\t"
                  "vmla.f32   q15, q8, d4[1]                \n\t"

                  "vmla.f32   q12, q9, d2[1]                \n\t"
                  "vmla.f32   q13, q9, d6[1]                \n\t"

                  "vmla.f32   q14, q9, d1[0]                \n\t"
                  "vmla.f32   q15, q9, d5[0]                \n\t"

                  // inptr3 multiply

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

                  // store outptr
                  "vmax.f32   d24, d24, d9                  \n\t"
                  "vmax.f32   d25, d25, d9                  \n\t"
                  "pld        [%[inptr1], #192]             \n\t"

                  "vmax.f32   d26, d26, d9                  \n\t"
                  "pld        [%[inptr4], #192]             \n\t"
                  "vmax.f32   d27, d27, d9                  \n\t"

                  "vmax.f32   d28, d28, d9                  \n\t"
                  "vld1.f32   {d10-d12}, [%[inptr1]]        \n\t"
                  "vmax.f32   d29, d29, d9                  \n\t"
                  "add        %[inptr1], #16                \n\t"

                  "vmax.f32   d30, d30, d9                  \n\t"
                  "vmax.f32   d31, d31, d9                  \n\t"
                  "vst1.f32   {d24, d25}, [%[outptr1]]!     \n\t"

                  "vst1.f32   {d26, d27}, [%[outptr1_2]]!   \n\t"
                  "vld1.f32   {d13-d15}, [%[inptr4]]        \n\t"

                  "add        %[inptr4], #16                \n\t"
                  "vst1.f32   {d28, d29}, [%[outptr2]]!     \n\t"

                  "subs       %[ow_dim4], #1                \n\t"
                  "vst1.f32   {d30, d31}, [%[outptr2_2]]!   \n\t"

                  // cycle
                  "bne        0b                            \n\t"
                  "sub       %[inptr1], #16                 \n\t"
                  "sub       %[inptr4], #16                 \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [outptr2] "+r"(outptr2), [outptr1_2] "+r"(outptr1_2),
                    [outptr2_2] "+r"(outptr2_2), [inptr1] "+r"(inptr1),
                    [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
                    [inptr4] "+r"(inptr4)
                  : [f1] "r"(f1), [f1_2] "r"(f1_2), [reluptr] "r"(reluptr)
                  : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                    "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
            }
          }
          if (!isnopadding && ow == output_width - padding_width) {
            ffilter0--;
            ffilter1--;
            ffilter2--;
            ffilter3--;

            ffilter0_2--;
            ffilter1_2--;
            ffilter2_2--;
            ffilter3_2--;

            inptr1--;
            inptr2--;
            inptr3--;
            inptr4--;
          }
#endif  //__aarch64__
#endif  // __ARM_NEON

          for (; ow < output_width; ++ow) {
            float sum1 = 0;
            float sum2 = 0;
            float sum1_2 = 0;
            float sum2_2 = 0;

            if (issamefilter) {
#if __ARM_NEON
              float32x4_t _inptr1 = vld1q_f32(inptr1);
              float32x4_t _ffilter1 = vld1q_f32(ffilter1);
              float32x4_t _ffilter1_2 = vld1q_f32(ffilter1_2);
              float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);
              float32x4_t _sum1_2 = vmulq_f32(_inptr1, _ffilter1_2);

              float32x4_t _inptr2 = vld1q_f32(inptr2);
              float32x4_t _ffilter2 = vld1q_f32(ffilter2);
              float32x4_t _ffilter2_2 = vld1q_f32(ffilter2_2);
              _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);
              _sum1_2 = vmlaq_f32(_sum1_2, _inptr2, _ffilter2_2);
              float32x4_t _sum2 = vmulq_f32(_inptr2, _ffilter1);
              float32x4_t _sum2_2 = vmulq_f32(_inptr2, _ffilter1_2);

              float32x4_t _inptr3 = vld1q_f32(inptr3);
              float32x4_t _ffilter3 = vld1q_f32(ffilter3);
              float32x4_t _ffilter3_2 = vld1q_f32(ffilter3_2);
              _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
              _sum1_2 = vmlaq_f32(_sum1_2, _inptr3, _ffilter3_2);
              _sum2 = vmlaq_f32(_sum2, _inptr3, _ffilter2);
              _sum2_2 = vmlaq_f32(_sum2_2, _inptr3, _ffilter2_2);

              float32x4_t _inptr4 = vld1q_f32(inptr4);
              _sum2 = vmlaq_f32(_sum2, _inptr4, _ffilter3);
              _sum2_2 = vmlaq_f32(_sum2_2, _inptr4, _ffilter3_2);

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
              sum1 += inptr1[0] * ffilter1[0];
              sum1 += inptr1[1] * ffilter1[1];
              sum1 += inptr1[2] * ffilter1[2];
              sum1 += inptr2[0] * ffilter2[0];
              sum1 += inptr2[1] * ffilter2[1];
              sum1 += inptr2[2] * ffilter2[2];
              sum1 += inptr3[0] * ffilter3[0];
              sum1 += inptr3[1] * ffilter3[1];
              sum1 += inptr3[2] * ffilter3[2];

              sum2 += inptr2[0] * ffilter1[0];
              sum2 += inptr2[1] * ffilter1[1];
              sum2 += inptr2[2] * ffilter1[2];
              sum2 += inptr3[0] * ffilter2[0];
              sum2 += inptr3[1] * ffilter2[1];
              sum2 += inptr3[2] * ffilter2[2];
              sum2 += inptr4[0] * ffilter3[0];
              sum2 += inptr4[1] * ffilter3[1];
              sum2 += inptr4[2] * ffilter3[2];

              sum1_2 += inptr1[0] * ffilter1_2[0];
              sum1_2 += inptr1[1] * ffilter1_2[1];
              sum1_2 += inptr1[2] * ffilter1_2[2];
              sum1_2 += inptr2[0] * ffilter2_2[0];
              sum1_2 += inptr2[1] * ffilter2_2[1];
              sum1_2 += inptr2[2] * ffilter2_2[2];
              sum1_2 += inptr3[0] * ffilter3_2[0];
              sum1_2 += inptr3[1] * ffilter3_2[1];
              sum1_2 += inptr3[2] * ffilter3_2[2];

              sum2_2 += inptr2[0] * ffilter1_2[0];
              sum2_2 += inptr2[1] * ffilter1_2[1];
              sum2_2 += inptr2[2] * ffilter1_2[2];
              sum2_2 += inptr3[0] * ffilter2_2[0];
              sum2_2 += inptr3[1] * ffilter2_2[1];
              sum2_2 += inptr3[2] * ffilter2_2[2];
              sum2_2 += inptr4[0] * ffilter3_2[0];
              sum2_2 += inptr4[1] * ffilter3_2[1];
              sum2_2 += inptr4[2] * ffilter3_2[2];
#endif
            } else {
#if __ARM_NEON
              float32x4_t _inptr1 = vld1q_f32(inptr1);
              float32x4_t _ffilter1 = vld1q_f32(ffilter1);
              float32x4_t _ffilter1_2 = vld1q_f32(ffilter1_2);
              float32x4_t _ffilter0 = vld1q_f32(ffilter0);
              float32x4_t _ffilter0_2 = vld1q_f32(ffilter0_2);

              float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);
              float32x4_t _sum1_2 = vmulq_f32(_inptr1, _ffilter1_2);
              float32x4_t _sum2 = vmulq_f32(_inptr1, _ffilter0);
              float32x4_t _sum2_2 = vmulq_f32(_inptr1, _ffilter0_2);

              float32x4_t _inptr2 = vld1q_f32(inptr2);
              float32x4_t _ffilter2 = vld1q_f32(ffilter2);
              float32x4_t _ffilter2_2 = vld1q_f32(ffilter2_2);

              _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);
              _sum1_2 = vmlaq_f32(_sum1_2, _inptr2, _ffilter2_2);
              _sum2 = vmlaq_f32(_sum2, _inptr2, _ffilter1);
              _sum2_2 = vmlaq_f32(_sum2_2, _inptr2, _ffilter1_2);

              float32x4_t _inptr3 = vld1q_f32(inptr3);
              float32x4_t _ffilter3 = vld1q_f32(ffilter3);
              float32x4_t _ffilter3_2 = vld1q_f32(ffilter3_2);

              _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
              _sum1_2 = vmlaq_f32(_sum1_2, _inptr3, _ffilter3_2);
              _sum2 = vmlaq_f32(_sum2, _inptr3, _ffilter2);
              _sum2_2 = vmlaq_f32(_sum2_2, _inptr3, _ffilter2_2);

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
              sum1 += inptr1[0] * ffilter1[0];
              sum1 += inptr1[1] * ffilter1[1];
              sum1 += inptr1[2] * ffilter1[2];
              sum1 += inptr2[0] * ffilter2[0];
              sum1 += inptr2[1] * ffilter2[1];
              sum1 += inptr2[2] * ffilter2[2];
              sum1 += inptr3[0] * ffilter3[0];
              sum1 += inptr3[1] * ffilter3[1];
              sum1 += inptr3[2] * ffilter3[2];

              sum2 += inptr1[0] * ffilter0[0];
              sum2 += inptr1[1] * ffilter0[1];
              sum2 += inptr1[2] * ffilter0[2];
              sum2 += inptr2[0] * ffilter1[0];
              sum2 += inptr2[1] * ffilter1[1];
              sum2 += inptr2[2] * ffilter1[2];
              sum2 += inptr3[0] * ffilter2[0];
              sum2 += inptr3[1] * ffilter2[1];
              sum2 += inptr3[2] * ffilter2[2];

              sum1_2 += inptr1[0] * ffilter1_2[0];
              sum1_2 += inptr1[1] * ffilter1_2[1];
              sum1_2 += inptr1[2] * ffilter1_2[2];
              sum1_2 += inptr2[0] * ffilter2_2[0];
              sum1_2 += inptr2[1] * ffilter2_2[1];
              sum1_2 += inptr2[2] * ffilter2_2[2];
              sum1_2 += inptr3[0] * ffilter3_2[0];
              sum1_2 += inptr3[1] * ffilter3_2[1];
              sum1_2 += inptr3[2] * ffilter3_2[2];

              sum2_2 += inptr1[0] * ffilter0_2[0];
              sum2_2 += inptr1[1] * ffilter0_2[1];
              sum2_2 += inptr1[2] * ffilter0_2[2];
              sum2_2 += inptr2[0] * ffilter1_2[0];
              sum2_2 += inptr2[1] * ffilter1_2[1];
              sum2_2 += inptr2[2] * ffilter1_2[2];
              sum2_2 += inptr3[0] * ffilter2_2[0];
              sum2_2 += inptr3[1] * ffilter2_2[1];
              sum2_2 += inptr3[2] * ffilter2_2[2];
#endif
            }
            if (!isnopadding &&
                (ow < padding_width || ow > output_width - padding_width - 2)) {
              ffilter0--;
              ffilter1--;
              ffilter2--;
              ffilter3--;

              ffilter0_2--;
              ffilter1_2--;
              ffilter2_2--;
              ffilter3_2--;
            } else {
              inptr1++;
              inptr2++;
              inptr3++;
              inptr4++;
            }
            *outptr1 += sum1;
            *outptr2 += sum2;
            *outptr1_2 += sum1_2;
            *outptr2_2 += sum2_2;
            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }
            if (outptr2[0] < reluvalue) {
              *outptr2 = reluvalue;
            }
            if (outptr1_2[0] < reluvalue) {
              *outptr1_2 = reluvalue;
            }
            if (outptr2_2[0] < reluvalue) {
              *outptr2_2 = reluvalue;
            }
            outptr1++;
            outptr2++;
            outptr1_2++;
            outptr2_2++;
          }
          if (isnopadding) {
            inptr1 += 2 + input_width;
            inptr2 += 2 + input_width;
            inptr3 += 2 + input_width;
            inptr4 += 2 + input_width;
          } else if (oh == padding_height - 1 ||
                     oh == output_height - padding_height - 2) {
            inptr1 += 3;
            inptr2 += 3;
            inptr3 += 3;
            inptr4 += 3;

            ffilter0 -= 2;
            ffilter1 -= 2;
            ffilter2 -= 2;
            ffilter3 -= 2;

            ffilter0_2 -= 2;
            ffilter1_2 -= 2;
            ffilter2_2 -= 2;
            ffilter3_2 -= 2;

          } else if (issamefilter) {
            inptr1 += 3 + input_width;
            inptr2 += 3 + input_width;
            inptr3 += 3 + input_width;
            inptr4 += 3 + input_width;

            ffilter0 += 2 * padding_width + 1;
            ffilter1 += 2 * padding_width + 1;
            ffilter2 += 2 * padding_width + 1;
            ffilter3 += 2 * padding_width + 1;

            ffilter0_2 += 2 * padding_width + 1;
            ffilter1_2 += 2 * padding_width + 1;
            ffilter2_2 += 2 * padding_width + 1;
            ffilter3_2 += 2 * padding_width + 1;

          } else {
            ffilter0 -= 3 + 2 * padding_width + 2;
            ffilter1 -= 3 + 2 * padding_width + 2;
            ffilter2 -= 3 + 2 * padding_width + 2;
            ffilter3 -= 3 + 2 * padding_width + 2;

            ffilter0_2 -= 3 + 2 * padding_width + 2;
            ffilter1_2 -= 3 + 2 * padding_width + 2;
            ffilter2_2 -= 3 + 2 * padding_width + 2;
            ffilter3_2 -= 3 + 2 * padding_width + 2;

            inptr1 -= input_width - 3;
            inptr2 -= input_width - 3;
            inptr3 -= input_width - 3;
            inptr4 -= input_width - 3;
          }
          outptr1 += output_width;
          outptr2 += output_width;
          outptr1_2 += output_width;
          outptr2_2 += output_width;
        }
        for (; oh < output_height; ++oh) {
          int ow = 0;
          for (; ow < padding_width; ++ow) {
            float sum1 = 0;
            float sum1_2 = 0;
#if __ARM_NEON
            float32x4_t _inptr1 = vld1q_f32(inptr1);
            float32x4_t _ffilter1 = vld1q_f32(ffilter1);
            float32x4_t _ffilter1_2 = vld1q_f32(ffilter1_2);
            float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);
            float32x4_t _sum1_2 = vmulq_f32(_inptr1, _ffilter1_2);

            float32x4_t _inptr2 = vld1q_f32(inptr2);
            float32x4_t _ffilter2 = vld1q_f32(ffilter2);
            float32x4_t _ffilter2_2 = vld1q_f32(ffilter2_2);
            _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);
            _sum1_2 = vmlaq_f32(_sum1_2, _inptr2, _ffilter2_2);

            float32x4_t _inptr3 = vld1q_f32(inptr3);
            float32x4_t _ffilter3 = vld1q_f32(ffilter3);
            float32x4_t _ffilter3_2 = vld1q_f32(ffilter3_2);
            _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
            _sum1_2 = vmlaq_f32(_sum1_2, _inptr3, _ffilter3_2);

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
            sum1 += inptr1[0] * ffilter1[0];
            sum1 += inptr1[1] * ffilter1[1];
            sum1 += inptr1[2] * ffilter1[2];
            sum1 += inptr2[0] * ffilter2[0];
            sum1 += inptr2[1] * ffilter2[1];
            sum1 += inptr2[2] * ffilter2[2];
            sum1 += inptr3[0] * ffilter3[0];
            sum1 += inptr3[1] * ffilter3[1];
            sum1 += inptr3[2] * ffilter3[2];

            sum1_2 += inptr1[0] * ffilter1_2[0];
            sum1_2 += inptr1[1] * ffilter1_2[1];
            sum1_2 += inptr1[2] * ffilter1_2[2];
            sum1_2 += inptr2[0] * ffilter2_2[0];
            sum1_2 += inptr2[1] * ffilter2_2[1];
            sum1_2 += inptr2[2] * ffilter2_2[2];
            sum1_2 += inptr3[0] * ffilter3_2[0];
            sum1_2 += inptr3[1] * ffilter3_2[1];
            sum1_2 += inptr3[2] * ffilter3_2[2];
#endif
            if (!isnopadding &&
                (ow < padding_width || ow > output_width - padding_width - 2)) {
              ffilter0--;
              ffilter1--;
              ffilter2--;
              ffilter3--;

              ffilter0_2--;
              ffilter1_2--;
              ffilter2_2--;
              ffilter3_2--;
            } else {
              inptr1++;
              inptr2++;
              inptr3++;
              inptr4++;
            }
            *outptr1 += sum1;
            *outptr1_2 += sum1_2;

            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }
            if (outptr1_2[0] < reluvalue) {
              *outptr1_2 = reluvalue;
            }
            outptr1++;
            outptr1_2++;
          }
#if __ARM_NEON
#if __aarch64__
          if (isnopadding) {
            int ow_dim4 = (output_width - 2 * padding_width) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
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
                  // load outptr
                  "prfm   pldl1keep, [%[outptr1], #128]     \n\t"
                  "prfm   pldl1keep, [%[outptr1_2], #128]   \n\t"

                  "ld1   {v12.4s}, [%[outptr1]]             \n\t"
                  "ld1   {v13.4s}, [%[outptr1_2]]           \n\t"

                  // inptr1 multiply
                  "prfm   pldl1keep, [%[inptr1], #192]      \n\t"
                  "ld1   {v5.4s, v6.4s}, [%[inptr1]]        \n\t"
                  "add        %[inptr1],%[inptr1], #16      \n\t"

                  "ext    v8.16b, v5.16b, v6.16b, #4        \n\t"
                  "fmla   v12.4s, v5.4s, v0.4s[0]           \n\t"
                  "fmla   v13.4s, v5.4s, v2.4s[0]           \n\t"

                  "ext    v10.16b, v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v0.4s[1]           \n\t"
                  "fmla   v13.4s, v8.4s, v2.4s[1]           \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[inptr2]]        \n\t"
                  "add        %[inptr2],%[inptr2], #16      \n\t"
                  "fmla   v12.4s, v10.4s, v0.4s[2]          \n\t"
                  "fmla   v13.4s, v10.4s, v2.4s[2]          \n\t"

                  // inptr2 multiply
                  "ext    v8.16b,  v5.16b, v6.16b, #4       \n\t"
                  "fmla   v12.4s, v5.4s, v0.4s[3]           \n\t"
                  "fmla   v13.4s, v5.4s, v2.4s[3]           \n\t"

                  "ext    v9.16b,  v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v1.4s[0]           \n\t"
                  "fmla   v13.4s, v8.4s, v3.4s[0]           \n\t"

                  "ld1   {v6.d}[1], [%[inptr3]]             \n\t"
                  "add        %[inptr3],%[inptr3], #8       \n\t"
                  "ld1   {v7.4s}, [%[inptr3]]               \n\t"
                  "add        %[inptr3],%[inptr3], #8       \n\t"

                  "fmla   v12.4s, v9.4s, v1.4s[1]           \n\t"
                  "fmla   v13.4s, v9.4s, v3.4s[1]           \n\t"

                  // inptr3 multiply
                  "ext    v10.16b, v6.16b, v7.16b, #8       \n\t"
                  "fmla   v12.4s, v7.4s, v4.4s[0]           \n\t"
                  "fmla   v13.4s, v7.4s, v4.4s[1]           \n\t"

                  "ext    v11.16b, v6.16b,  v7.16b, #12     \n\t"
                  "fmla   v12.4s, v10.4s, v1.4s[2]          \n\t"
                  "fmla   v13.4s, v10.4s, v3.4s[2]          \n\t"

                  "fmla   v12.4s, v11.4s, v1.4s[3]          \n\t"
                  "fmla   v13.4s, v11.4s, v3.4s[3]          \n\t"

                  // store outptr
                  "fmax   v12.4s,v12.4s, v16.4s             \n\t"
                  "fmax   v13.4s,v13.4s, v16.4s             \n\t"

                  "st1   {v12.4s}, [%[outptr1]], #16        \n\t"
                  "st1   {v13.4s}, [%[outptr1_2]], #16      \n\t"

                  // cycle
                  "subs       %[ow_dim4],%[ow_dim4], #1     \n\t"
                  "bne        0b                            \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [outptr2] "+r"(outptr2), [outptr1_2] "+r"(outptr1_2),
                    [outptr2_2] "+r"(outptr2_2), [inptr1] "+r"(inptr1),
                    [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
                    [inptr4] "+r"(inptr4)
                  : [f1] "r"(f1), [f1_2] "r"(f1_2), [reluptr] "r"(reluptr)
                  : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v16");
            }
          }
#else
          if (isnopadding) {
            int ow_dim4 = (output_width - 2 * padding_width) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
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
                  // load outptr
                  "pld        [%[outptr1], #128]            \n\t"
                  "pld        [%[outptr1_2], #128]          \n\t"

                  "vld1.f32   {d24, d25}, [%[outptr1]]      \n\t"
                  "vld1.f32   {d26, d27}, [%[outptr1_2]]    \n\t"

                  // inptr1 multiply
                  "pld        [%[inptr1], #128]             \n\t"

                  "vld1.f32   {d10-d12}, [%[inptr1]]        \n\t"
                  "add        %[inptr1], #16                \n\t"
                  "vext.32    q8, q5, q6, #1                \n\t"

                  "pld        [%[inptr2], #128]             \n\t"
                  "vmla.f32   q12, q5, d0[0]                \n\t"
                  "vmla.f32   q13, q5, d4[0]                \n\t"

                  "vext.32    q10, q5, q6, #2               \n\t"
                  "vld1.f32   {d10-d12}, [%[inptr2]]        \n\t"
                  "add        %[inptr2], #16                \n\t"
                  "vmla.f32   q12, q8, d0[1]                \n\t"
                  "vmla.f32   q13, q8, d4[1]                \n\t"

                  "vmla.f32   q12, q10, d1[0]               \n\t"
                  "vmla.f32   q13, q10, d5[0]               \n\t"

                  // inptr2 multiply
                  "vext.32    q8, q5, q6, #1                \n\t"
                  "pld        [%[inptr3], #128]            \n\t"
                  "vmla.f32   q12, q5, d1[1]                \n\t"
                  "vmla.f32   q13, q5, d5[1]                \n\t"

                  "vext.32    q9, q5, q6, #2                \n\t"
                  "vld1.f32   {d13-d15}, [%[inptr3]]        \n\t"
                  "add        %[inptr3], #16                \n\t"
                  "vmla.f32   q12, q8, d2[0]                \n\t"
                  "vmla.f32   q13, q8, d6[0]                \n\t"

                  "vmla.f32   q12, q9, d2[1]                \n\t"
                  "vmla.f32   q13, q9, d6[1]                \n\t"

                  // inptr3 multiply
                  "vext.32    q10, q6, q7, #2               \n\t"
                  "vmla.f32   q12, q7, d8[0]                \n\t"
                  "vmla.f32   q13, q7, d8[1]                \n\t"

                  "vext.32    q11, q6, q7, #3               \n\t"
                  "vmla.f32   q12, q10, d3[0]               \n\t"
                  "vmla.f32   q13, q10, d7[0]               \n\t"

                  "vmla.f32   q12, q11, d3[1]               \n\t"
                  "vmla.f32   q13, q11, d7[1]               \n\t"

                  // store outptr
                  "vmax.f32   d24, d24, d9                  \n\t"
                  "vmax.f32   d25, d25, d9                  \n\t"

                  "vmax.f32   d26, d26, d9                  \n\t"
                  "vmax.f32   d27, d27, d9                  \n\t"

                  "subs       %[ow_dim4], #1                \n\t"
                  "vst1.f32   {d24, d25}, [%[outptr1]]!     \n\t"
                  "vst1.f32   {d26, d27}, [%[outptr1_2]]!   \n\t"

                  // cycle
                  "bne        0b                              \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [outptr2] "+r"(outptr2), [outptr1_2] "+r"(outptr1_2),
                    [outptr2_2] "+r"(outptr2_2), [inptr1] "+r"(inptr1),
                    [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
                    [inptr4] "+r"(inptr4)
                  : [f1] "r"(f1), [f1_2] "r"(f1_2), [reluptr] "r"(reluptr)
                  : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                    "q8", "q9", "q10", "q11", "q12", "q13");
            }
          }

#endif  //__aarch64__
#endif  // __ARM_NEON

          for (; ow < output_width; ++ow) {
            float sum1 = 0;
            float sum1_2 = 0;
#if __ARM_NEON
            float32x4_t _inptr1 = vld1q_f32(inptr1);
            float32x4_t _ffilter1 = vld1q_f32(ffilter1);
            float32x4_t _ffilter1_2 = vld1q_f32(ffilter1_2);
            float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);
            float32x4_t _sum1_2 = vmulq_f32(_inptr1, _ffilter1_2);

            float32x4_t _inptr2 = vld1q_f32(inptr2);
            float32x4_t _ffilter2 = vld1q_f32(ffilter2);
            float32x4_t _ffilter2_2 = vld1q_f32(ffilter2_2);
            _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);
            _sum1_2 = vmlaq_f32(_sum1_2, _inptr2, _ffilter2_2);

            float32x4_t _inptr3 = vld1q_f32(inptr3);
            float32x4_t _ffilter3 = vld1q_f32(ffilter3);
            float32x4_t _ffilter3_2 = vld1q_f32(ffilter3_2);
            _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
            _sum1_2 = vmlaq_f32(_sum1_2, _inptr3, _ffilter3_2);

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
            sum1 += inptr1[0] * ffilter1[0];
            sum1 += inptr1[1] * ffilter1[1];
            sum1 += inptr1[2] * ffilter1[2];
            sum1 += inptr2[0] * ffilter2[0];
            sum1 += inptr2[1] * ffilter2[1];
            sum1 += inptr2[2] * ffilter2[2];
            sum1 += inptr3[0] * ffilter3[0];
            sum1 += inptr3[1] * ffilter3[1];
            sum1 += inptr3[2] * ffilter3[2];

            sum1_2 += inptr1[0] * ffilter1_2[0];
            sum1_2 += inptr1[1] * ffilter1_2[1];
            sum1_2 += inptr1[2] * ffilter1_2[2];
            sum1_2 += inptr2[0] * ffilter2_2[0];
            sum1_2 += inptr2[1] * ffilter2_2[1];
            sum1_2 += inptr2[2] * ffilter2_2[2];
            sum1_2 += inptr3[0] * ffilter3_2[0];
            sum1_2 += inptr3[1] * ffilter3_2[1];
            sum1_2 += inptr3[2] * ffilter3_2[2];
#endif
            if (!isnopadding &&
                (ow < padding_width || ow > output_width - padding_width - 2)) {
              ffilter0--;
              ffilter1--;
              ffilter2--;
              ffilter3--;

              ffilter0_2--;
              ffilter1_2--;
              ffilter2_2--;
              ffilter3_2--;
            } else {
              inptr1++;
              inptr2++;
              inptr3++;
              inptr4++;
            }
            *outptr1 += sum1;
            *outptr1_2 += sum1_2;

            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }
            if (outptr1_2[0] < reluvalue) {
              *outptr1_2 = reluvalue;
            }
            outptr1++;
            outptr1_2++;
          }
          outptr1 += output_width;
          outptr1_2 += output_width;
        }
        filter_data_ch += filter_channel_stride;
        filter_data_ch_2 += filter_channel_stride;
        input_data_ch += input_channel_stride;
      }
    }
    int output_channels_left = output_channels_d2 * 2;
    for (int oc = output_channels_left; oc < output_channels; ++oc) {
      std::atomic<float> reluvalue{0};
      const float *reluptr;
      bool issamefilter;
      const float *inptr1, *inptr2, *inptr3, *inptr4;
      const float *f1;
      const float *ffilter0, *ffilter1, *ffilter2, *ffilter3;
      float ffilterarray[ffilter_length] = {0};
      float *output_data_ch;
      const float *input_data_ch;
      const float *filter_data_ch;

      input_data_ch = input_data;
      output_data_ch = output_data + oc * output_channel_stride;
      filter_data_ch =
          filter_data + oc * filter_channel_stride * input_channels;

      for (int ic = 0; ic < input_channels; ++ic) {
        float reluarr[4];
        reluptr = reluarr;
        if (if_relu && ic == input_channels - 1) {
          reluvalue = 0;
        } else {
          reluvalue = -FLT_MAX;
        }
        reluarr[0] = reluvalue;
        reluarr[1] = reluvalue;
        reluarr[2] = reluvalue;
        reluarr[3] = reluvalue;
        f1 = filter_data_ch;
        if (!isnopadding) {
          ffilterarray[ffilter_length] = {0};
          for (int i = 0; i < 9; ++i) {
            int j = i / 3 * (2 * padding_width + 3) + i % 3 +
                    padding_height * 3 +
                    padding_width * (2 * padding_height + 1);
            ffilterarray[j] = filter_data_ch[i];
          }
          ffilter1 = ffilterarray;
          ffilter1 += ffilter_start;
          ffilter0 = ffilter1 - ffilter_width;
          ffilter2 = ffilter1 + ffilter_width;
          ffilter3 = ffilter2 + ffilter_width;

        } else {
          ffilter1 = filter_data_ch;
          ffilter2 = ffilter1 + 3;
          ffilter3 = ffilter2 + 3;
        }
        float *outptr1, *outptr2;
        outptr1 = output_data_ch;
        outptr2 = outptr1 + output_width;

        inptr1 = input_data_ch;
        inptr2 = inptr1 + input_width;
        inptr3 = inptr2 + input_width;
        inptr4 = inptr3 + input_width;

        int oh = 0;
        for (; oh < output_height - 1; oh = oh + 2) {
          if (!isnopadding && (oh < padding_height ||
                               oh > output_height - padding_height - 2)) {
            issamefilter = false;
          } else {
            issamefilter = true;
          }
          int ow = 0;
          for (; ow < padding_width; ++ow) {
            float sum1 = 0;
            float sum2 = 0;

            if (issamefilter) {
#if __ARM_NEON
              float32x4_t _inptr1 = vld1q_f32(inptr1);
              float32x4_t _ffilter1 = vld1q_f32(ffilter1);
              float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);

              float32x4_t _inptr2 = vld1q_f32(inptr2);
              float32x4_t _ffilter2 = vld1q_f32(ffilter2);
              _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);
              float32x4_t _sum2 = vmulq_f32(_inptr2, _ffilter1);

              float32x4_t _inptr3 = vld1q_f32(inptr3);
              float32x4_t _ffilter3 = vld1q_f32(ffilter3);
              _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
              _sum2 = vmlaq_f32(_sum2, _inptr3, _ffilter2);

              float32x4_t _inptr4 = vld1q_f32(inptr4);
              _sum2 = vmlaq_f32(_sum2, _inptr4, _ffilter3);

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
              sum1 += inptr1[0] * ffilter1[0];
              sum1 += inptr1[1] * ffilter1[1];
              sum1 += inptr1[2] * ffilter1[2];
              sum1 += inptr2[0] * ffilter2[0];
              sum1 += inptr2[1] * ffilter2[1];
              sum1 += inptr2[2] * ffilter2[2];
              sum1 += inptr3[0] * ffilter3[0];
              sum1 += inptr3[1] * ffilter3[1];
              sum1 += inptr3[2] * ffilter3[2];

              sum2 += inptr2[0] * ffilter1[0];
              sum2 += inptr2[1] * ffilter1[1];
              sum2 += inptr2[2] * ffilter1[2];
              sum2 += inptr3[0] * ffilter2[0];
              sum2 += inptr3[1] * ffilter2[1];
              sum2 += inptr3[2] * ffilter2[2];
              sum2 += inptr4[0] * ffilter3[0];
              sum2 += inptr4[1] * ffilter3[1];
              sum2 += inptr4[2] * ffilter3[2];
#endif
            } else {
#if __ARM_NEON
              float32x4_t _inptr1 = vld1q_f32(inptr1);
              float32x4_t _ffilter1 = vld1q_f32(ffilter1);
              float32x4_t _ffilter0 = vld1q_f32(ffilter0);

              float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);
              float32x4_t _sum2 = vmulq_f32(_inptr1, _ffilter0);
              float32x4_t _inptr2 = vld1q_f32(inptr2);
              float32x4_t _ffilter2 = vld1q_f32(ffilter2);

              _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);
              _sum2 = vmlaq_f32(_sum2, _inptr2, _ffilter1);

              float32x4_t _inptr3 = vld1q_f32(inptr3);
              float32x4_t _ffilter3 = vld1q_f32(ffilter3);

              _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
              _sum2 = vmlaq_f32(_sum2, _inptr3, _ffilter2);

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
              sum1 += inptr1[0] * ffilter1[0];
              sum1 += inptr1[1] * ffilter1[1];
              sum1 += inptr1[2] * ffilter1[2];
              sum1 += inptr2[0] * ffilter2[0];
              sum1 += inptr2[1] * ffilter2[1];
              sum1 += inptr2[2] * ffilter2[2];
              sum1 += inptr3[0] * ffilter3[0];
              sum1 += inptr3[1] * ffilter3[1];
              sum1 += inptr3[2] * ffilter3[2];

              sum2 += inptr1[0] * ffilter0[0];
              sum2 += inptr1[1] * ffilter0[1];
              sum2 += inptr1[2] * ffilter0[2];
              sum2 += inptr2[0] * ffilter1[0];
              sum2 += inptr2[1] * ffilter1[1];
              sum2 += inptr2[2] * ffilter1[2];
              sum2 += inptr3[0] * ffilter2[0];
              sum2 += inptr3[1] * ffilter2[1];
              sum2 += inptr3[2] * ffilter2[2];
#endif
            }
            if (!isnopadding &&
                (ow < padding_width || ow > output_width - padding_width - 2)) {
              ffilter0--;
              ffilter1--;
              ffilter2--;
              ffilter3--;
            } else {
              inptr1++;
              inptr2++;
              inptr3++;
              inptr4++;
            }
            *outptr1 += sum1;
            *outptr2 += sum2;

            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }
            if (outptr2[0] < reluvalue) {
              *outptr2 = reluvalue;
            }
            outptr1++;
            outptr2++;
          }
#if __ARM_NEON
#if __aarch64__
          if (issamefilter) {
            int ow_dim4 = (output_width - 2 * padding_width) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
              asm volatile(
                  "prfm   pldl1keep, [%[f1], #256]          \n\t"

                  "ld1   {v0.4s, v1.4s}, [%[f1]]            \n\t"
                  "add        %[f1], %[f1], #32             \n\t"
                  "prfm   pldl1keep, [%[reluptr], #64]      \n\t"
                  "ld1   {v16.4s}, [%[reluptr]]             \n\t"

                  "ld1   {v4.s}[0], [%[f1]]                 \n\t"
                  "sub        %[f1],%[f1], #32              \n\t"

                  "0:                                       \n\t"
                  // load outptr
                  "prfm   pldl1keep, [%[outptr1], #128]     \n\t"
                  "prfm   pldl1keep, [%[outptr2], #128]     \n\t"

                  "ld1   {v12.4s}, [%[outptr1]]             \n\t"
                  "ld1   {v14.4s}, [%[outptr2]]             \n\t"

                  // inptr1 + inptr4 multiply
                  "prfm   pldl1keep, [%[inptr1], #192]      \n\t"
                  "prfm   pldl1keep, [%[inptr4], #192]      \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[inptr1]]        \n\t"
                  "add        %[inptr1],%[inptr1], #16      \n\t"

                  "ld1   {v6.d}[1], [%[inptr4]]             \n\t"
                  "add        %[inptr4],%[inptr4], #8       \n\t"
                  "ld1   {v7.4s}, [%[inptr4]]               \n\t"
                  "add        %[inptr4],%[inptr4], #8       \n\t"

                  "ext    v8.16b, v5.16b, v6.16b, #4        \n\t"
                  "fmla   v12.4s, v5.4s, v0.4s[0]           \n\t"

                  "ext    v9.16b, v6.16b, v7.16b, #8        \n\t"
                  "fmla   v14.4s, v7.4s, v4.4s[0]           \n\t"

                  "ext    v10.16b, v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v0.4s[1]           \n\t"

                  "ext    v11.16b, v6.16b, v7.16b, #12      \n\t"
                  "fmla   v14.4s, v9.4s, v1.4s[2]           \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[inptr2]]        \n\t"
                  "add        %[inptr2],%[inptr2], #16      \n\t"

                  "fmla   v12.4s, v10.4s, v0.4s[2]          \n\t"
                  "fmla   v14.4s, v11.4s, v1.4s[3]          \n\t"

                  // inptr2 multiply
                  "ext    v8.16b,  v5.16b, v6.16b, #4       \n\t"
                  "fmla   v12.4s, v5.4s, v0.4s[3]           \n\t"
                  "fmla   v14.4s, v5.4s, v0.4s[0]           \n\t"

                  "ext    v9.16b,  v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v1.4s[0]           \n\t"
                  "fmla   v14.4s, v8.4s, v0.4s[1]           \n\t"

                  "ld1   {v6.d}[1], [%[inptr3]]             \n\t"
                  "add        %[inptr3],%[inptr3], #8       \n\t"
                  "ld1   {v7.4s}, [%[inptr3]]               \n\t"

                  "add        %[inptr3],%[inptr3], #8       \n\t"
                  "fmla   v12.4s, v9.4s, v1.4s[1]           \n\t"
                  "fmla   v14.4s, v9.4s, v0.4s[2]           \n\t"

                  // inptr3 multiply
                  "ext    v10.16b, v6.16b, v7.16b, #8       \n\t"
                  "fmla   v12.4s, v7.4s, v4.4s[0]           \n\t"
                  "fmla   v14.4s, v7.4s, v1.4s[1]           \n\t"

                  "ext    v11.16b, v6.16b,  v7.16b, #12     \n\t"
                  "fmla   v12.4s, v10.4s, v1.4s[2]          \n\t"
                  "fmla   v14.4s, v10.4s, v0.4s[3]          \n\t"

                  "fmla   v12.4s, v11.4s, v1.4s[3]          \n\t"
                  "fmla   v14.4s, v11.4s, v1.4s[0]          \n\t"

                  // store outptr
                  "fmax   v12.4s,v12.4s, v16.4s             \n\t"
                  "fmax   v14.4s,v14.4s, v16.4s             \n\t"

                  "st1   {v12.4s}, [%[outptr1]], #16        \n\t"
                  "st1   {v14.4s}, [%[outptr2]], #16        \n\t"

                  // cycle
                  "subs       %[ow_dim4],%[ow_dim4], #1     \n\t"
                  "bne        0b                            \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [outptr2] "+r"(outptr2), [inptr1] "+r"(inptr1),
                    [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
                    [inptr4] "+r"(inptr4)
                  : [f1] "r"(f1), [reluptr] "r"(reluptr)
                  : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v14", "v16");
            }
          }
          if (!isnopadding && ow == output_width - padding_width) {
            ffilter0--;
            ffilter1--;
            ffilter2--;
            ffilter3--;

            inptr1--;
            inptr2--;
            inptr3--;
            inptr4--;
          }
#else
          if (issamefilter) {
            int ow_dim4 = (output_width - 2 * padding_width) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
              asm volatile(
                  "pld        [%[f1], #256]                 \n\t"
                  "vld1.32   {d0-d3}, [%[f1]]               \n\t"
                  "add        %[f1], #32                    \n\t"

                  "pld        [%[reluptr], #64]             \n\t"
                  "vld1.32   {d9}, [%[reluptr]]             \n\t"

                  "vld1.32   {d8[0]}, [%[f1]]               \n\t"
                  "sub        %[f1], #32                    \n\t"

                  "0:                                       \n\t"
                  // load outptr
                  "pld        [%[outptr1], #128]            \n\t"
                  "pld        [%[outptr2], #128]            \n\t"

                  "vld1.f32   {d24, d25}, [%[outptr1]]      \n\t"
                  "vld1.f32   {d28, d29}, [%[outptr2]]      \n\t"

                  // inptr1 + inptr4 multiply
                  "pld        [%[inptr1], #192]             \n\t"
                  "pld        [%[inptr4], #192]             \n\t"

                  "vld1.f32   {d10-d12}, [%[inptr1]]        \n\t"
                  "add        %[inptr1], #16                \n\t"

                  "vld1.f32   {d13-d15}, [%[inptr4]]        \n\t"
                  "add        %[inptr4], #16                \n\t"

                  "vext.32    q8, q5, q6, #1                \n\t"
                  "vmla.f32   q12, q5, d0[0]                \n\t"

                  "vext.32    q9, q6, q7, #2                \n\t"
                  "vmla.f32   q14, q7, d8[0]                \n\t"

                  "vext.32    q10, q5, q6, #2               \n\t"
                  "vmla.f32   q12, q8, d0[1]                \n\t"

                  "vext.32    q11, q6, q7, #3               \n\t"
                  "vmla.f32   q14, q9, d3[0]                \n\t"

                  "vld1.f32   {d10-d12}, [%[inptr2]]        \n\t"
                  "add        %[inptr2], #16                \n\t"

                  "vmla.f32   q12, q10, d1[0]               \n\t"
                  "vmla.f32   q14, q11, d3[1]               \n\t"

                  // inptr2 multiply
                  "vext.32    q8, q5, q6, #1                \n\t"
                  "vmla.f32   q12, q5, d1[1]                \n\t"
                  "vmla.f32   q14, q5, d0[0]                \n\t"

                  "vext.32    q9, q5, q6, #2                \n\t"
                  "vmla.f32   q12, q8, d2[0]                \n\t"
                  "vmla.f32   q14, q8, d0[1]                \n\t"

                  "vld1.f32   {d13-d15}, [%[inptr3]]        \n\t"
                  "add        %[inptr3], #16                \n\t"

                  "vmla.f32   q12, q9, d2[1]                \n\t"
                  "vmla.f32   q14, q9, d1[0]                \n\t"

                  // inptr3 multiply
                  "vext.32    q10, q6, q7, #2               \n\t"
                  "vmla.f32   q12, q7, d8[0]                \n\t"
                  "vmla.f32   q14, q7, d2[1]                \n\t"

                  "vext.32    q11, q6, q7, #3               \n\t"
                  "vmla.f32   q12, q10, d3[0]               \n\t"
                  "vmla.f32   q14, q10, d1[1]               \n\t"

                  "vmla.f32   q12, q11, d3[1]               \n\t"
                  "vmla.f32   q14, q11, d2[0]               \n\t"

                  // store outptr
                  "vmax.f32   d24, d24, d9                  \n\t"
                  "vmax.f32   d25, d25, d9                  \n\t"

                  "vmax.f32   d28, d28, d9                  \n\t"
                  "vmax.f32   d29, d29, d9                  \n\t"

                  "subs       %[ow_dim4], #1                \n\t"
                  "vst1.f32   {d24, d25}, [%[outptr1]]!     \n\t"
                  "vst1.f32   {d28, d29}, [%[outptr2]]!     \n\t"

                  // cycle
                  "bne        0b                            \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [outptr2] "+r"(outptr2), [inptr1] "+r"(inptr1),
                    [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
                    [inptr4] "+r"(inptr4)
                  : [f1] "r"(f1), [reluptr] "r"(reluptr)
                  : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                    "q8", "q9", "q10", "q11", "q12", "q14");
            }
          }
          if (!isnopadding && ow == output_width - padding_width) {
            ffilter0--;
            ffilter1--;
            ffilter2--;
            ffilter3--;

            inptr1--;
            inptr2--;
            inptr3--;
            inptr4--;
          }
#endif  //__aarch64__
#endif  // __ARM_NEON

          for (; ow < output_width; ++ow) {
            float sum1 = 0;
            float sum2 = 0;

            if (issamefilter) {
#if __ARM_NEON
              float32x4_t _inptr1 = vld1q_f32(inptr1);
              float32x4_t _ffilter1 = vld1q_f32(ffilter1);
              float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);

              float32x4_t _inptr2 = vld1q_f32(inptr2);
              float32x4_t _ffilter2 = vld1q_f32(ffilter2);
              _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);
              float32x4_t _sum2 = vmulq_f32(_inptr2, _ffilter1);

              float32x4_t _inptr3 = vld1q_f32(inptr3);
              float32x4_t _ffilter3 = vld1q_f32(ffilter3);
              _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
              _sum2 = vmlaq_f32(_sum2, _inptr3, _ffilter2);

              float32x4_t _inptr4 = vld1q_f32(inptr4);
              _sum2 = vmlaq_f32(_sum2, _inptr4, _ffilter3);

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
              sum1 += inptr1[0] * ffilter1[0];
              sum1 += inptr1[1] * ffilter1[1];
              sum1 += inptr1[2] * ffilter1[2];
              sum1 += inptr2[0] * ffilter2[0];
              sum1 += inptr2[1] * ffilter2[1];
              sum1 += inptr2[2] * ffilter2[2];
              sum1 += inptr3[0] * ffilter3[0];
              sum1 += inptr3[1] * ffilter3[1];
              sum1 += inptr3[2] * ffilter3[2];

              sum2 += inptr2[0] * ffilter1[0];
              sum2 += inptr2[1] * ffilter1[1];
              sum2 += inptr2[2] * ffilter1[2];
              sum2 += inptr3[0] * ffilter2[0];
              sum2 += inptr3[1] * ffilter2[1];
              sum2 += inptr3[2] * ffilter2[2];
              sum2 += inptr4[0] * ffilter3[0];
              sum2 += inptr4[1] * ffilter3[1];
              sum2 += inptr4[2] * ffilter3[2];
#endif
            } else {
#if __ARM_NEON
              float32x4_t _inptr1 = vld1q_f32(inptr1);
              float32x4_t _ffilter1 = vld1q_f32(ffilter1);
              float32x4_t _ffilter0 = vld1q_f32(ffilter0);

              float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);
              float32x4_t _sum2 = vmulq_f32(_inptr1, _ffilter0);
              float32x4_t _inptr2 = vld1q_f32(inptr2);
              float32x4_t _ffilter2 = vld1q_f32(ffilter2);

              _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);
              _sum2 = vmlaq_f32(_sum2, _inptr2, _ffilter1);
              float32x4_t _inptr3 = vld1q_f32(inptr3);
              float32x4_t _ffilter3 = vld1q_f32(ffilter3);

              _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
              _sum2 = vmlaq_f32(_sum2, _inptr3, _ffilter2);
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
              sum1 += inptr1[0] * ffilter1[0];
              sum1 += inptr1[1] * ffilter1[1];
              sum1 += inptr1[2] * ffilter1[2];
              sum1 += inptr2[0] * ffilter2[0];
              sum1 += inptr2[1] * ffilter2[1];
              sum1 += inptr2[2] * ffilter2[2];
              sum1 += inptr3[0] * ffilter3[0];
              sum1 += inptr3[1] * ffilter3[1];
              sum1 += inptr3[2] * ffilter3[2];

              sum2 += inptr1[0] * ffilter0[0];
              sum2 += inptr1[1] * ffilter0[1];
              sum2 += inptr1[2] * ffilter0[2];
              sum2 += inptr2[0] * ffilter1[0];
              sum2 += inptr2[1] * ffilter1[1];
              sum2 += inptr2[2] * ffilter1[2];
              sum2 += inptr3[0] * ffilter2[0];
              sum2 += inptr3[1] * ffilter2[1];
              sum2 += inptr3[2] * ffilter2[2];
#endif
            }
            if (!isnopadding &&
                (ow < padding_width || ow > output_width - padding_width - 2)) {
              ffilter0--;
              ffilter1--;
              ffilter2--;
              ffilter3--;
            } else {
              inptr1++;
              inptr2++;
              inptr3++;
              inptr4++;
            }
            *outptr1 += sum1;
            *outptr2 += sum2;

            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }
            if (outptr2[0] < reluvalue) {
              *outptr2 = reluvalue;
            }
            outptr1++;
            outptr2++;
          }
          if (isnopadding) {
            inptr1 += 2 + input_width;
            inptr2 += 2 + input_width;
            inptr3 += 2 + input_width;
            inptr4 += 2 + input_width;
          } else if (oh == padding_height - 1 ||
                     oh == output_height - padding_height - 2) {
            inptr1 += 3;
            inptr2 += 3;
            inptr3 += 3;
            inptr4 += 3;

            ffilter0 -= 2;
            ffilter1 -= 2;
            ffilter2 -= 2;
            ffilter3 -= 2;

          } else if (issamefilter) {
            inptr1 += 3 + input_width;
            inptr2 += 3 + input_width;
            inptr3 += 3 + input_width;
            inptr4 += 3 + input_width;

            ffilter0 += 2 * padding_width + 1;
            ffilter1 += 2 * padding_width + 1;
            ffilter2 += 2 * padding_width + 1;
            ffilter3 += 2 * padding_width + 1;

          } else {
            ffilter0 -= 3 + 2 * padding_width + 2;
            ffilter1 -= 3 + 2 * padding_width + 2;
            ffilter2 -= 3 + 2 * padding_width + 2;
            ffilter3 -= 3 + 2 * padding_width + 2;

            inptr1 -= input_width - 3;
            inptr2 -= input_width - 3;
            inptr3 -= input_width - 3;
            inptr4 -= input_width - 3;
          }
          outptr1 += output_width;
          outptr2 += output_width;
        }

        for (; oh < output_height; ++oh) {
          int ow = 0;
          for (; ow < padding_width; ++ow) {
            float sum1 = 0;
#if __ARM_NEON
            float32x4_t _inptr1 = vld1q_f32(inptr1);
            float32x4_t _ffilter1 = vld1q_f32(ffilter1);
            float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);

            float32x4_t _inptr2 = vld1q_f32(inptr2);
            float32x4_t _ffilter2 = vld1q_f32(ffilter2);
            _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);

            float32x4_t _inptr3 = vld1q_f32(inptr3);
            float32x4_t _ffilter3 = vld1q_f32(ffilter3);
            _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);

            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ssss1_ssss1 = vpadd_f32(_ss1, _ss1);
            sum1 += vget_lane_f32(_ssss1_ssss1, 0);
#else
            sum1 += inptr1[0] * ffilter1[0];
            sum1 += inptr1[1] * ffilter1[1];
            sum1 += inptr1[2] * ffilter1[2];
            sum1 += inptr2[0] * ffilter2[0];
            sum1 += inptr2[1] * ffilter2[1];
            sum1 += inptr2[2] * ffilter2[2];
            sum1 += inptr3[0] * ffilter3[0];
            sum1 += inptr3[1] * ffilter3[1];
            sum1 += inptr3[2] * ffilter3[2];
#endif
            if (!isnopadding &&
                (ow < padding_width || ow > output_width - padding_width - 2)) {
              ffilter0--;
              ffilter1--;
              ffilter2--;
              ffilter3--;

            } else {
              inptr1++;
              inptr2++;
              inptr3++;
              inptr4++;
            }
            *outptr1 += sum1;

            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }
            outptr1++;
          }

#if __ARM_NEON
#if __aarch64__
          if (isnopadding) {
            int ow_dim4 = (output_width - 2 * padding_width) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
              asm volatile(
                  "prfm   pldl1keep, [%[f1], #256]          \n\t"

                  "ld1   {v0.4s, v1.4s}, [%[f1]]            \n\t"
                  "add        %[f1], %[f1], #32             \n\t"
                  "prfm   pldl1keep, [%[reluptr], #64]      \n\t"
                  "ld1   {v16.4s}, [%[reluptr]]             \n\t"

                  "ld1   {v4.s}[0], [%[f1]]                 \n\t"
                  "sub        %[f1],%[f1], #32              \n\t"

                  "0:                                       \n\t"
                  // load outptr
                  "prfm   pldl1keep, [%[outptr1], #128]     \n\t"
                  "ld1   {v12.4s}, [%[outptr1]]             \n\t"

                  // inptr1 multiply
                  "prfm   pldl1keep, [%[inptr1], #192]      \n\t"
                  "ld1   {v5.4s, v6.4s}, [%[inptr1]]        \n\t"
                  "add        %[inptr1],%[inptr1], #16      \n\t"

                  "ext    v8.16b, v5.16b, v6.16b, #4        \n\t"
                  "fmla   v12.4s, v5.4s, v0.4s[0]           \n\t"

                  "ext    v10.16b, v5.16b, v6.16b, #8       \n\t"
                  "fmul   v13.4s, v8.4s, v0.4s[1]           \n\t"

                  "ld1   {v5.4s, v6.4s}, [%[inptr2]]        \n\t"
                  "add        %[inptr2],%[inptr2], #16      \n\t"
                  "fmla   v12.4s, v10.4s, v0.4s[2]          \n\t"

                  // inptr2 multiply
                  "ext    v8.16b,  v5.16b, v6.16b, #4       \n\t"
                  "fmla   v13.4s, v5.4s, v0.4s[3]           \n\t"

                  "ext    v9.16b,  v5.16b, v6.16b, #8       \n\t"
                  "fmla   v12.4s, v8.4s, v1.4s[0]           \n\t"

                  "ld1   {v6.d}[1], [%[inptr3]]             \n\t"
                  "add        %[inptr3],%[inptr3], #8       \n\t"
                  "ld1   {v7.4s}, [%[inptr3]]               \n\t"
                  "add        %[inptr3],%[inptr3], #8       \n\t"
                  "fmla   v13.4s, v9.4s, v1.4s[1]           \n\t"

                  // inptr3 multiply
                  "ext    v10.16b, v6.16b, v7.16b, #8       \n\t"
                  "fmla   v12.4s, v7.4s, v4.4s[0]           \n\t"

                  "ext    v11.16b, v6.16b,  v7.16b, #12     \n\t"
                  "fmla   v13.4s, v10.4s, v1.4s[2]          \n\t"
                  "fmla   v12.4s, v11.4s, v1.4s[3]          \n\t"

                  // store outptr
                  "fadd   v12.4s, v13.4s, v12.4s            \n\t"
                  "fmax   v12.4s, v12.4s, v16.4s             \n\t"
                  "st1   {v12.4s}, [%[outptr1]], #16        \n\t"

                  // cycle
                  "subs       %[ow_dim4],%[ow_dim4], #1     \n\t"
                  "bne        0b                            \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
                    [inptr3] "+r"(inptr3)
                  : [f1] "r"(f1), [reluptr] "r"(reluptr)
                  : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v16");
            }
          }
#else
          if (isnopadding) {
            int ow_dim4 = (output_width - 2 * padding_width) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
              asm volatile(
                  "pld        [%[f1], #256]                 \n\t"
                  "vld1.32   {d0-d3}, [%[f1]]               \n\t"
                  "add        %[f1], #32                    \n\t"

                  "pld        [%[reluptr], #64]             \n\t"
                  "vld1.32   {q2}, [%[reluptr]]             \n\t"

                  "vld1.32   {d8[0]}, [%[f1]]               \n\t"
                  "sub        %[f1], #32                    \n\t"

                  "0:                                       \n\t"
                  // load outptr
                  "pld        [%[outptr1], #128]            \n\t"
                  "vld1.f32   {d24, d25}, [%[outptr1]]      \n\t"

                  // inptr1 multiply
                  "pld        [%[inptr1], #128]             \n\t"
                  "vld1.f32   {d10-d12}, [%[inptr1]]        \n\t"
                  "add        %[inptr1], #16                \n\t"

                  "vext.32    q8, q5, q6, #1                \n\t"
                  "vmla.f32   q12, q5, d0[0]                \n\t"

                  "vext.32    q9, q6, q7, #2                \n\t"
                  "vext.32    q10, q5, q6, #2               \n\t"
                  "vmul.f32   q13, q8, d0[1]                \n\t"
                  "pld        [%[inptr2], #128]             \n\t"

                  "vext.32    q11, q6, q7, #3               \n\t"
                  "vld1.f32   {d10-d12}, [%[inptr2]]        \n\t"
                  "add        %[inptr2], #16                \n\t"
                  "vmla.f32   q12, q10, d1[0]               \n\t"

                  // inptr2 multiply
                  "vext.32    q8, q5, q6, #1                \n\t"
                  "vmla.f32   q13, q5, d1[1]                \n\t"

                  "pld        [%[inptr3], #128]             \n\t"
                  "vext.32    q9, q5, q6, #2                \n\t"
                  "vmla.f32   q12, q8, d2[0]                \n\t"

                  "vld1.f32   {d13-d15}, [%[inptr3]]        \n\t"
                  "add        %[inptr3], #16                \n\t"
                  "vmla.f32   q13, q9, d2[1]                \n\t"

                  // inptr3 multiply
                  "vext.32    q10, q6, q7, #2               \n\t"
                  "vmla.f32   q12, q7, d8[0]                \n\t"

                  "vext.32    q11, q6, q7, #3               \n\t"
                  "vmla.f32   q13, q10, d3[0]               \n\t"

                  "vmla.f32   q12, q11, d3[1]               \n\t"

                  // store outptr
                  "vadd.f32   q12, q12, q13                 \n\t"
                  "vmax.f32   q12, q12, q2                  \n\t"
                  "vst1.f32   {d24, d25}, [%[outptr1]]!     \n\t"

                  // cycle
                  "subs       %[ow_dim4], #1                \n\t"
                  "bne        0b                            \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [outptr2] "+r"(outptr2), [inptr1] "+r"(inptr1),
                    [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
                    [inptr4] "+r"(inptr4)
                  : [f1] "r"(f1), [reluptr] "r"(reluptr)
                  : "memory", "q0", "q1", "q2", "q4", "q5", "q6", "q7", "q8",
                    "q9", "q10", "q11", "q12", "q13");
            }
          }
#endif  //__aarch64__
#endif  // __ARM_NEON
          for (; ow < output_width; ++ow) {
            float sum1 = 0;
#if __ARM_NEON
            float32x4_t _inptr1 = vld1q_f32(inptr1);
            float32x4_t _ffilter1 = vld1q_f32(ffilter1);
            float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);

            float32x4_t _inptr2 = vld1q_f32(inptr2);
            float32x4_t _ffilter2 = vld1q_f32(ffilter2);
            _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);

            float32x4_t _inptr3 = vld1q_f32(inptr3);
            float32x4_t _ffilter3 = vld1q_f32(ffilter3);
            _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);

            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ssss1_ssss1 = vpadd_f32(_ss1, _ss1);
            sum1 += vget_lane_f32(_ssss1_ssss1, 0);
#else
            sum1 += inptr1[0] * ffilter1[0];
            sum1 += inptr1[1] * ffilter1[1];
            sum1 += inptr1[2] * ffilter1[2];
            sum1 += inptr2[0] * ffilter2[0];
            sum1 += inptr2[1] * ffilter2[1];
            sum1 += inptr2[2] * ffilter2[2];
            sum1 += inptr3[0] * ffilter3[0];
            sum1 += inptr3[1] * ffilter3[1];
            sum1 += inptr3[2] * ffilter3[2];
#endif
            if (!isnopadding &&
                (ow < padding_width || ow > output_width - padding_width - 2)) {
              ffilter0--;
              ffilter1--;
              ffilter2--;
              ffilter3--;

            } else {
              inptr1++;
              inptr2++;
              inptr3++;
              inptr4++;
            }
            *outptr1 += sum1;

            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }
            outptr1++;
          }
          outptr1 += output_width;
        }
        filter_data_ch += filter_channel_stride;
        input_data_ch += input_channel_stride;
      }
    }
    input_data += input_batch_stride;
    output_data += output_batch_stride;
  }
}

void SlidingwindowConv3x3s2(const framework::Tensor *input,
                            const framework::Tensor *filter,
                            const std::vector<int> &paddings,
                            framework::Tensor *output, framework::Tensor *bias,
                            bool if_bias, bool if_relu) {
  const int batch_size = input->dims()[0];
  const int input_channels = input->dims()[1];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const int output_channels = output->dims()[1];
  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  const int padding_height = paddings[0];
  const int padding_width = paddings[1];

  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();
  const float *filter_data = filter->data<float>();

  const int input_channel_stride = input_height * input_width;
  const int input_batch_stride = input_channels * input_channel_stride;
  const int output_channel_stride = output_height * output_width;
  const int output_batch_stride = output_channels * output_channel_stride;
  const int filter_channel_stride = 9;
  const int ffilter_length = (2 * padding_height + 3) * (2 * padding_width + 3);
  const int ffilter_start =
      2 * padding_height * (2 * padding_width + 3) + 2 * padding_width;
  const int ffilter_width = 3 + padding_width * 2;
  const float *bias_data;
  bool isnopadding = false;

  const bool useallInput_w = (input_width + 2 * padding_width - 3) % 2 == 0;
  const bool useallInput_h = (input_height + 2 * padding_height - 3) % 2 == 0;
  const bool isoddpadding_w = padding_width % 2 == 1;
  const bool isoddpadding_h = padding_height % 2 == 1;
  int position_w1 = padding_width >> 1;
  int position_h1 = padding_height >> 1;
  int position_w2 = output_width - position_w1 - 2;
  int position_h2 = output_height - position_h1 - 2;
  const int left_stride = input_width + 2 * padding_width - 2 * output_width;

  if (if_bias) {
    bias_data = bias->data<float>();
  }
#if __ARM_NEON
  float *outptr = output_data;
  for (int i = 0; i < output_channels; ++i) {
    float32x4_t _bias;
    if (if_bias) {
      _bias = vld1q_dup_f32(bias_data + i);
    } else {
      _bias = vdupq_n_f32(0.0);
    }
    int dim4 = output_channel_stride >> 2;
    int lef4 = output_channel_stride & 3;

    for (int j = 0; j < dim4; ++j) {
      vst1q_f32(outptr, _bias);
      outptr += 4;
    }
    if (lef4 != 0) {
      if (lef4 >= 1) {
        vst1q_lane_f32(outptr, _bias, 0);
        outptr++;
      }
      if (lef4 >= 2) {
        vst1q_lane_f32(outptr, _bias, 1);
        outptr++;
      }
      if (lef4 >= 3) {
        vst1q_lane_f32(outptr, _bias, 2);
        outptr++;
      }
    }
  }
#else
  int k = 0;
#pragma omp parallel for
  for (int i = 0; i < output_channels; ++i) {
    for (int j = 0; j < output_channel_stride; ++j) {
      if (if_bias) {
        output_data[k++] = bias_data[i];
      } else {
        output_data[k++] = 0;
      }
    }
  }
#endif
  if (padding_height == 0 && padding_width == 0) {
    isnopadding = true;
    position_w1 = 0;
    position_h1 = 0;
    position_w2 = output_width;
    position_h2 = output_height;
  }
  for (int bs = 0; bs < batch_size; ++bs) {
    int output_channels_d2 = output_channels >> 1;

#pragma omp parallel for
    for (int oc2 = 0; oc2 < output_channels_d2; ++oc2) {
      std::atomic<float> reluvalue{0};
      const float *reluptr;
      int oc = oc2 * 2;

      const float *f1, *f9;
      const float *f1_2, *f9_2;
      const float *inptr1, *inptr2, *inptr3;
      const float *ffilter1, *ffilter2, *ffilter3;
      const float *ffilter1_2, *ffilter2_2, *ffilter3_2;
      float ffilterarray[ffilter_length] = {0};
      float ffilterarray_2[ffilter_length] = {0};

      float *output_data_ch;
      float *output_data_ch_2;
      const float *input_data_ch;
      const float *filter_data_ch;
      const float *filter_data_ch_2;

      filter_data_ch =
          filter_data + oc * filter_channel_stride * input_channels;
      filter_data_ch_2 =
          filter_data + (oc + 1) * filter_channel_stride * input_channels;

      input_data_ch = input_data;
      output_data_ch = output_data + oc * output_channel_stride;
      output_data_ch_2 = output_data + (oc + 1) * output_channel_stride;

      for (int ic = 0; ic < input_channels; ++ic) {
        float reluarr[4];
        reluptr = reluarr;
        if (if_relu && ic == input_channels - 1) {
          reluvalue = 0;
        } else {
          reluvalue = -FLT_MAX;
        }
        reluarr[0] = reluvalue;
        reluarr[1] = reluvalue;
        reluarr[2] = reluvalue;
        reluarr[3] = reluvalue;

        f1 = filter_data_ch;
        f9 = filter_data_ch + 8;
        f1_2 = filter_data_ch_2;
        f9_2 = filter_data_ch_2 + 8;

        if (!isnopadding) {
          ffilterarray[ffilter_length] = {0};
          for (int i = 0; i < 9; ++i) {
            int j = i / 3 * (2 * padding_width + 3) + i % 3 +
                    padding_height * 3 +
                    padding_width * (2 * padding_height + 1);
            ffilterarray[j] = filter_data_ch[i];
            ffilterarray_2[j] = filter_data_ch_2[i];
          }
          ffilter1 = ffilterarray;
          ffilter1 += ffilter_start;
          ffilter2 = ffilter1 + ffilter_width;
          ffilter3 = ffilter2 + ffilter_width;

          ffilter1_2 = ffilterarray_2;
          ffilter1_2 += ffilter_start;
          ffilter2_2 = ffilter1_2 + ffilter_width;
          ffilter3_2 = ffilter2_2 + ffilter_width;
        } else {
          ffilter1 = filter_data_ch;
          ffilter2 = ffilter1 + 3;
          ffilter3 = ffilter2 + 3;

          ffilter1_2 = filter_data_ch_2;
          ffilter2_2 = ffilter1_2 + 3;
          ffilter3_2 = ffilter2_2 + 3;
        }

        float *outptr1;
        float *outptr1_2;
        outptr1 = output_data_ch;
        outptr1_2 = output_data_ch_2;

        inptr1 = input_data_ch;
        inptr2 = inptr1 + input_width;
        inptr3 = inptr2 + input_width;

        int oh = 0;
        for (; oh < output_height; ++oh) {
          int ow = 0;

          for (; ow <= position_w1; ++ow) {
            float sum1 = 0;
            float sum1_2 = 0;
#if __ARM_NEON
            float32x4_t _inptr1 = vld1q_f32(inptr1);
            float32x4_t _ffilter1 = vld1q_f32(ffilter1);
            float32x4_t _ffilter1_2 = vld1q_f32(ffilter1_2);
            float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);
            float32x4_t _sum1_2 = vmulq_f32(_inptr1, _ffilter1_2);

            float32x4_t _inptr2 = vld1q_f32(inptr2);
            float32x4_t _ffilter2 = vld1q_f32(ffilter2);
            float32x4_t _ffilter2_2 = vld1q_f32(ffilter2_2);
            _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);
            _sum1_2 = vmlaq_f32(_sum1_2, _inptr2, _ffilter2_2);

            float32x4_t _inptr3 = vld1q_f32(inptr3);
            float32x4_t _ffilter3 = vld1q_f32(ffilter3);
            float32x4_t _ffilter3_2 = vld1q_f32(ffilter3_2);
            _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
            _sum1_2 = vmlaq_f32(_sum1_2, _inptr3, _ffilter3_2);

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
            sum1 += inptr1[0] * ffilter1[0];
            sum1 += inptr1[1] * ffilter1[1];
            sum1 += inptr1[2] * ffilter1[2];
            sum1 += inptr2[0] * ffilter2[0];
            sum1 += inptr2[1] * ffilter2[1];
            sum1 += inptr2[2] * ffilter2[2];
            sum1 += inptr3[0] * ffilter3[0];
            sum1 += inptr3[1] * ffilter3[1];
            sum1 += inptr3[2] * ffilter3[2];

            sum1_2 += inptr1[0] * ffilter1_2[0];
            sum1_2 += inptr1[1] * ffilter1_2[1];
            sum1_2 += inptr1[2] * ffilter1_2[2];
            sum1_2 += inptr2[0] * ffilter2_2[0];
            sum1_2 += inptr2[1] * ffilter2_2[1];
            sum1_2 += inptr2[2] * ffilter2_2[2];
            sum1_2 += inptr3[0] * ffilter3_2[0];
            sum1_2 += inptr3[1] * ffilter3_2[1];
            sum1_2 += inptr3[2] * ffilter3_2[2];
#endif
            if (isnopadding) {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;

            } else if (isoddpadding_w && ow == position_w1 ||
                       ow == position_w2 && isoddpadding_w && useallInput_w ||
                       ow == position_w2 + 1 && !isoddpadding_w &&
                           !useallInput_w) {
              ffilter1--;
              ffilter2--;
              ffilter3--;
              ffilter1_2--;
              ffilter2_2--;
              ffilter3_2--;

              inptr1++;
              inptr2++;
              inptr3++;

            } else if (ow < position_w1 || ow > position_w2) {
              ffilter1 -= 2;
              ffilter2 -= 2;
              ffilter3 -= 2;
              ffilter1_2 -= 2;
              ffilter2_2 -= 2;
              ffilter3_2 -= 2;

            } else {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;
            }
            *outptr1 += sum1;
            *outptr1_2 += sum1_2;

            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }

            if (outptr1_2[0] < reluvalue) {
              *outptr1_2 = reluvalue;
            }
            outptr1++;
            outptr1_2++;
          }
#if __ARM_NEON
#if __aarch64__
          if (oh > position_h1 && oh <= position_h2) {
            int ow_dim4 = (position_w2 - position_w1 - 1) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
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

                  "prfm   pldl1keep, [%[inptr1], #256]      \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[inptr1]], #32   \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[inptr1]]        \n\t"

                  "0:                                       \n\t"
                  // load outptr
                  "prfm    pldl1keep, [%[outptr1], #128]    \n\t"
                  "prfm    pldl1keep, [%[outptr1_2], #128]  \n\t"

                  "ld1   {v12.4s}, [%[outptr1]]             \n\t"
                  "ld1   {v14.4s}, [%[outptr1_2]]           \n\t"

                  // inptr1

                  "fmla   v12.4s, v5.4s, v0.4s[0]           \n\t"
                  "fmla   v14.4s, v5.4s, v2.4s[0]           \n\t"

                  "ext    v8.16b, v5.16b, v7.16b, #4        \n\t"
                  "fmul   v13.4s, v6.4s, v0.4s[1]           \n\t"
                  "fmul   v15.4s, v6.4s, v2.4s[1]           \n\t"

                  "ld2   {v5.4s, v6.4s}, [%[inptr2]], #32   \n\t"
                  "fmla   v12.4s, v8.4s, v0.4s[2]           \n\t"
                  "fmla   v14.4s, v8.4s, v2.4s[2]           \n\t"

                  "ld2   {v7.4s, v8.4s}, [%[inptr2]]        \n\t"
                  // inptr2

                  "fmla   v13.4s, v5.4s, v0.4s[3]           \n\t"
                  "fmla   v15.4s, v5.4s, v2.4s[3]           \n\t"

                  "ext    v8.16b, v5.16b, v7.16b, #4        \n\t"
                  "fmla   v12.4s, v6.4s, v1.4s[0]           \n\t"
                  "fmla   v14.4s, v6.4s, v3.4s[0]           \n\t"

                  "ld2   {v5.4s, v6.4s}, [%[inptr3]], #32   \n\t"
                  "fmla   v13.4s, v8.4s, v1.4s[1]           \n\t"
                  "fmla   v15.4s, v8.4s, v3.4s[1]           \n\t"

                  "ld2   {v7.4s, v8.4s}, [%[inptr3]]        \n\t"
                  // inptr3

                  "fmla   v12.4s, v5.4s, v1.4s[2]           \n\t"
                  "fmla   v14.4s, v5.4s, v3.4s[2]           \n\t"

                  "ext    v8.16b, v5.16b, v7.16b, #4        \n\t"
                  "fmla   v13.4s, v6.4s, v1.4s[3]           \n\t"
                  "fmla   v15.4s, v6.4s, v3.4s[3]           \n\t"

                  "fmla   v12.4s, v8.4s, v4.4s[0]           \n\t"
                  "fmla   v14.4s, v8.4s, v4.4s[1]           \n\t"

                  // store

                  "prfm   pldl1keep, [%[inptr1], #256]      \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[inptr1]], #32   \n\t"
                  "fadd   v12.4s, v12.4s, v13.4s            \n\t"
                  "fadd   v14.4s, v14.4s, v15.4s            \n\t"

                  "ld2   {v7.4s, v8.4s}, [%[inptr1]]        \n\t"
                  "fmax   v12.4s, v12.4s, v16.4s            \n\t"
                  "fmax   v14.4s, v14.4s, v16.4s            \n\t"

                  "subs       %[ow_dim4], %[ow_dim4], #1    \n\t"
                  "st1   {v12.4s}, [%[outptr1]], #16        \n\t"
                  "st1   {v14.4s}, [%[outptr1_2]], #16      \n\t"

                  // cycle
                  "bne        0b                            \n\t"
                  "sub        %[inptr1], %[inptr1], #32      \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [outptr1_2] "+r"(outptr1_2), [inptr1] "+r"(inptr1),
                    [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3)
                  : [f1] "r"(f1), [f1_2] "r"(f1_2), [f9] "r"(f9),
                    [f9_2] "r"(f9_2), [reluptr] "r"(reluptr)
                  : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v12", "v13", "v14", "v15", "v16");
            }
          }

#else
          if (oh > position_h1 && oh <= position_h2) {
            int ow_dim4 = (position_w2 - position_w1 - 1) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
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

                  "vld2.f32   {d10-d13}, [%[inptr1]]!       \n\t"
                  "vld2.f32   {d14, d15}, [%[inptr1]]       \n\t"

                  "0:                                       \n\t"
                  // load outptr
                  "pld        [%[outptr1], #128]            \n\t"
                  "pld        [%[outptr1_2], #128]          \n\t"
                  "vld1.f32   {d24, d25}, [%[outptr1]]      \n\t"
                  "vld1.f32   {d28, d29}, [%[outptr1_2]]    \n\t"

                  // inptr1 multiply
                  "vmla.f32   q12, q5, d0[0]                \n\t"
                  "vmla.f32   q14, q5, d4[0]                \n\t"

                  "vext.32    q8, q5, q7, #1                \n\t"
                  "vmul.f32   q13, q6, d0[1]                \n\t"
                  "vmul.f32   q15, q6, d4[1]                \n\t"

                  "vld2.f32   {d10-d13}, [%[inptr2]]!       \n\t"
                  "vmla.f32   q12, q8, d1[0]                \n\t"
                  "vld2.f32   {d14, d15}, [%[inptr2]]       \n\t"
                  "vmla.f32   q14, q8, d5[0]                \n\t"

                  // inptr2 multiply
                  "vmla.f32   q13, q5, d1[1]                \n\t"
                  "vmla.f32   q15, q5, d5[1]                \n\t"

                  "vext.32    q8, q5, q7, #1                \n\t"
                  "vmla.f32   q12, q6, d2[0]                \n\t"
                  "vmla.f32   q14, q6, d6[0]                \n\t"

                  "vld2.f32   {d10-d13}, [%[inptr3]]!       \n\t"
                  "vmla.f32   q13, q8, d2[1]                \n\t"
                  "vld2.f32   {d14, d15}, [%[inptr3]]       \n\t"
                  "vmla.f32   q15, q8, d6[1]                \n\t"

                  // inptr3 multiply
                  "vmla.f32   q12, q5, d3[0]                \n\t"
                  "vmla.f32   q14, q5, d7[0]                \n\t"

                  "vext.32    q8, q5, q7, #1                \n\t"
                  "vmla.f32   q13, q6, d3[1]                \n\t"
                  "vmla.f32   q15, q6, d7[1]                \n\t"

                  "vmla.f32   q12, q8, d8[0]                \n\t"
                  "vmla.f32   q14, q8, d8[1]                \n\t"

                  // store
                  "vld2.f32   {d10-d13}, [%[inptr1]]!       \n\t"
                  "vadd.f32   q12, q12, q13                 \n\t"
                  "vadd.f32   q14, q14, q15                 \n\t"
                  "vmax.f32   q12, q12, q9                  \n\t"
                  "vmax.f32   q14, q14, q9                  \n\t"

                  "vld2.f32   {d14, d15}, [%[inptr1]]       \n\t"
                  "vst1.f32   {d24, d25}, [%[outptr1]]!     \n\t"
                  "vst1.f32   {d28, d29}, [%[outptr1_2]]!   \n\t"

                  // cycle
                  "subs       %[ow_dim4], #1                \n\t"
                  "bne        0b                            \n\t"
                  "sub       %[inptr1], %[inptr1], #32      \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [outptr1_2] "+r"(outptr1_2), [inptr1] "+r"(inptr1),
                    [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3)
                  : [f1] "r"(f1), [f1_2] "r"(f1_2), [f9] "r"(f9),
                    [f9_2] "r"(f9_2), [reluptr] "r"(reluptr)
                  : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                    "q8", "q9", "q12", "q13", "q14", "q15");
            }
          }
#endif  //__aarch64__
#endif  // __ARM_NEON

          for (; ow < output_width; ++ow) {
            float sum1 = 0;
            float sum1_2 = 0;
#if __ARM_NEON
            float32x4_t _inptr1 = vld1q_f32(inptr1);
            float32x4_t _ffilter1 = vld1q_f32(ffilter1);
            float32x4_t _ffilter1_2 = vld1q_f32(ffilter1_2);
            float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);
            float32x4_t _sum1_2 = vmulq_f32(_inptr1, _ffilter1_2);

            float32x4_t _inptr2 = vld1q_f32(inptr2);
            float32x4_t _ffilter2 = vld1q_f32(ffilter2);
            float32x4_t _ffilter2_2 = vld1q_f32(ffilter2_2);
            _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);
            _sum1_2 = vmlaq_f32(_sum1_2, _inptr2, _ffilter2_2);

            float32x4_t _inptr3 = vld1q_f32(inptr3);
            float32x4_t _ffilter3 = vld1q_f32(ffilter3);
            float32x4_t _ffilter3_2 = vld1q_f32(ffilter3_2);
            _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
            _sum1_2 = vmlaq_f32(_sum1_2, _inptr3, _ffilter3_2);

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
            sum1 += inptr1[0] * ffilter1[0];
            sum1 += inptr1[1] * ffilter1[1];
            sum1 += inptr1[2] * ffilter1[2];
            sum1 += inptr2[0] * ffilter2[0];
            sum1 += inptr2[1] * ffilter2[1];
            sum1 += inptr2[2] * ffilter2[2];
            sum1 += inptr3[0] * ffilter3[0];
            sum1 += inptr3[1] * ffilter3[1];
            sum1 += inptr3[2] * ffilter3[2];

            sum1_2 += inptr1[0] * ffilter1_2[0];
            sum1_2 += inptr1[1] * ffilter1_2[1];
            sum1_2 += inptr1[2] * ffilter1_2[2];
            sum1_2 += inptr2[0] * ffilter2_2[0];
            sum1_2 += inptr2[1] * ffilter2_2[1];
            sum1_2 += inptr2[2] * ffilter2_2[2];
            sum1_2 += inptr3[0] * ffilter3_2[0];
            sum1_2 += inptr3[1] * ffilter3_2[1];
            sum1_2 += inptr3[2] * ffilter3_2[2];
#endif
            if (isnopadding) {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;

            } else if (isoddpadding_w && ow == position_w1 ||
                       ow == position_w2 && isoddpadding_w && useallInput_w ||
                       ow == position_w2 + 1 && !isoddpadding_w &&
                           !useallInput_w) {
              ffilter1--;
              ffilter2--;
              ffilter3--;

              ffilter1_2--;
              ffilter2_2--;
              ffilter3_2--;

              inptr1++;
              inptr2++;
              inptr3++;

            } else if (ow < position_w1 || ow > position_w2) {
              ffilter1 -= 2;
              ffilter2 -= 2;
              ffilter3 -= 2;
              ffilter1_2 -= 2;
              ffilter2_2 -= 2;
              ffilter3_2 -= 2;
            } else {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;
            }
            *outptr1 += sum1;
            *outptr1_2 += sum1_2;

            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }

            if (outptr1_2[0] < reluvalue) {
              *outptr1_2 = reluvalue;
            }
            outptr1++;
            outptr1_2++;
          }
          if (isnopadding) {
            inptr1 += left_stride + input_width;
            inptr2 += left_stride + input_width;
            inptr3 += left_stride + input_width;

          } else if (isoddpadding_h && oh == position_h1 ||
                     oh == position_h2 && isoddpadding_h && useallInput_h ||
                     oh == position_h2 + 1 && !isoddpadding_h &&
                         !useallInput_h) {
            inptr1 += 3;
            inptr2 += 3;
            inptr3 += 3;

            ffilter1 -= left_stride;
            ffilter2 -= left_stride;
            ffilter3 -= left_stride;

            ffilter1_2 -= left_stride;
            ffilter2_2 -= left_stride;
            ffilter3_2 -= left_stride;
          } else if (oh < position_h1 || oh > position_h2) {
            inptr1 -= input_width - 3;
            inptr2 -= input_width - 3;
            inptr3 -= input_width - 3;

            ffilter1 -= 3 + 2 * padding_width + left_stride;
            ffilter2 -= 3 + 2 * padding_width + left_stride;
            ffilter3 -= 3 + 2 * padding_width + left_stride;

            ffilter1_2 -= 3 + 2 * padding_width + left_stride;
            ffilter2_2 -= 3 + 2 * padding_width + left_stride;
            ffilter3_2 -= 3 + 2 * padding_width + left_stride;
          } else {
            ffilter1 += 3 + 2 * padding_width - left_stride;
            ffilter2 += 3 + 2 * padding_width - left_stride;
            ffilter3 += 3 + 2 * padding_width - left_stride;

            ffilter1_2 += 3 + 2 * padding_width - left_stride;
            ffilter2_2 += 3 + 2 * padding_width - left_stride;
            ffilter3_2 += 3 + 2 * padding_width - left_stride;

            inptr1 += input_width + 3;
            inptr2 += input_width + 3;
            inptr3 += input_width + 3;
          }
        }
        filter_data_ch += filter_channel_stride;
        filter_data_ch_2 += filter_channel_stride;
        input_data_ch += input_channel_stride;
      }
    }

    int output_channels_left = output_channels_d2 * 2;
    for (int oc = output_channels_left; oc < output_channels; ++oc) {
      std::atomic<float> reluvalue{0};
      const float *reluptr;
      const float *f1, *f9;
      const float *inptr1, *inptr2, *inptr3;
      const float *ffilter1, *ffilter2, *ffilter3;
      float ffilterarray[ffilter_length] = {0};

      float *output_data_ch;
      const float *input_data_ch;
      const float *filter_data_ch;

      filter_data_ch =
          filter_data + oc * filter_channel_stride * input_channels;
      input_data_ch = input_data;
      output_data_ch = output_data + oc * output_channel_stride;

      for (int ic = 0; ic < input_channels; ++ic) {
        float reluarr[4];
        reluptr = reluarr;
        if (if_relu && ic == input_channels - 1) {
          reluvalue = 0;
        } else {
          reluvalue = -FLT_MAX;
        }
        reluarr[0] = reluvalue;
        reluarr[1] = reluvalue;
        reluarr[2] = reluvalue;
        reluarr[3] = reluvalue;
        f1 = filter_data_ch;
        f9 = filter_data_ch + 8;

        if (!isnopadding) {
          ffilterarray[ffilter_length] = {0};
          for (int i = 0; i < 9; ++i) {
            int j = i / 3 * (2 * padding_width + 3) + i % 3 +
                    padding_height * 3 +
                    padding_width * (2 * padding_height + 1);
            ffilterarray[j] = filter_data_ch[i];
          }
          ffilter1 = ffilterarray;
          ffilter1 += ffilter_start;
          ffilter2 = ffilter1 + ffilter_width;
          ffilter3 = ffilter2 + ffilter_width;

        } else {
          ffilter1 = filter_data_ch;
          ffilter2 = ffilter1 + 3;
          ffilter3 = ffilter2 + 3;
        }
        float *outptr1;
        outptr1 = output_data_ch;
        inptr1 = input_data_ch;
        inptr2 = inptr1 + input_width;
        inptr3 = inptr2 + input_width;

        int oh = 0;
        for (; oh < output_height; ++oh) {
          int ow = 0;
          for (; ow <= position_w1; ++ow) {
            float sum1 = 0;
#if __ARM_NEON
            float32x4_t _inptr1 = vld1q_f32(inptr1);
            float32x4_t _ffilter1 = vld1q_f32(ffilter1);
            float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);

            float32x4_t _inptr2 = vld1q_f32(inptr2);
            float32x4_t _ffilter2 = vld1q_f32(ffilter2);
            _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);

            float32x4_t _inptr3 = vld1q_f32(inptr3);
            float32x4_t _ffilter3 = vld1q_f32(ffilter3);
            _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);

            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ssss1_ssss1 = vpadd_f32(_ss1, _ss1);
            sum1 += vget_lane_f32(_ssss1_ssss1, 0);
#else
            sum1 += inptr1[0] * ffilter1[0];
            sum1 += inptr1[1] * ffilter1[1];
            sum1 += inptr1[2] * ffilter1[2];
            sum1 += inptr2[0] * ffilter2[0];
            sum1 += inptr2[1] * ffilter2[1];
            sum1 += inptr2[2] * ffilter2[2];
            sum1 += inptr3[0] * ffilter3[0];
            sum1 += inptr3[1] * ffilter3[1];
            sum1 += inptr3[2] * ffilter3[2];
#endif
            if (isnopadding) {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;

            } else if (isoddpadding_w && ow == position_w1 ||
                       ow == position_w2 && isoddpadding_w && useallInput_w ||
                       ow == position_w2 + 1 && !isoddpadding_w &&
                           !useallInput_w) {
              ffilter1--;
              ffilter2--;
              ffilter3--;

              inptr1++;
              inptr2++;
              inptr3++;

            } else if (ow < position_w1 || ow > position_w2) {
              ffilter1 -= 2;
              ffilter2 -= 2;
              ffilter3 -= 2;

            } else {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;
            }
            *outptr1 += sum1;

            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }
            outptr1++;
          }

#if __ARM_NEON
#if __aarch64__
          if (oh > position_h1 && oh <= position_h2) {
            int ow_dim4 = (position_w2 - position_w1 - 1) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
              asm volatile(
                  "prfm   pldl1keep, [%[f1], #256]          \n\t"
                  "prfm   pldl1keep, [%[f9], #256]          \n\t"
                  "prfm   pldl1keep, [%[reluptr], #256]     \n\t"

                  "ld1   {v0.4s, v1.4s}, [%[f1]]            \n\t"
                  "ld1   {v4.s}[0], [%[f9]]                 \n\t"
                  "ld1   {v16.4s}, [%[reluptr]]             \n\t"

                  "0:                                       \n\t"
                  // load outptr
                  "prfm   pldl1keep, [%[outptr1], #128]     \n\t"
                  "ld1   {v12.4s}, [%[outptr1]]             \n\t"

                  // inptr1 multiply
                  "prfm   pldl1keep, [%[inptr1], #256]      \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[inptr1]], #32   \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[inptr1]]        \n\t"

                  "fmla   v12.4s, v5.4s, v0.4s[0]           \n\t"
                  "fmla   v14.4s, v5.4s, v2.4s[0]           \n\t"

                  "ext    v8.16b, v5.16b, v7.16b, #4        \n\t"
                  "fmul   v13.4s, v6.4s, v0.4s[1]           \n\t"

                  "fmla   v12.4s, v8.4s, v0.4s[2]           \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[inptr2]], #32   \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[inptr2]]        \n\t"

                  // inptr2 multiply
                  "fmla   v13.4s, v5.4s, v0.4s[3]           \n\t"
                  "ext    v8.16b, v5.16b, v7.16b, #4        \n\t"
                  "fmla   v12.4s, v6.4s, v1.4s[0]           \n\t"

                  "fmla   v13.4s, v8.4s, v1.4s[1]           \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[inptr3]], #32   \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[inptr3]]        \n\t"

                  // inptr3 multiply
                  "fmla   v12.4s, v5.4s, v1.4s[2]           \n\t"
                  "ext    v8.16b, v5.16b, v7.16b, #4        \n\t"
                  "fmla   v13.4s, v6.4s, v1.4s[3]           \n\t"
                  "fmla   v12.4s, v8.4s, v4.4s[0]           \n\t"

                  // store outptr
                  "fadd   v12.4s, v12.4s, v13.4s            \n\t"
                  "fmax   v12.4s, v12.4s, v16.4s            \n\t"
                  "st1   {v12.4s}, [%[outptr1]], #16        \n\t"

                  // cycle
                  "subs       %[ow_dim4], %[ow_dim4], #1    \n\t"
                  "bne        0b                            \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
                    [inptr3] "+r"(inptr3)
                  : [f1] "r"(f1), [f9] "r"(f9), [reluptr] "r"(reluptr)
                  : "memory", "v0", "v1", "v4", "v5", "v6", "v7", "v8", "v9",
                    "v12", "v13", "v16");
            }
          }
#else
          if (oh > position_h1 && oh <= position_h2) {
            int ow_dim4 = (position_w2 - position_w1 - 1) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
              asm volatile(
                  "pld        [%[f1], #256]                 \n\t"
                  "pld        [%[f9], #256]                 \n\t"
                  "pld        [%[reluptr], #128]            \n\t"

                  "vld1.32   {d0-d3}, [%[f1]]               \n\t"
                  "vld1.32   {d8[0]}, [%[f9]]               \n\t"
                  "vld1.32   {d18, d19}, [%[reluptr]]       \n\t"

                  "0:                                       \n\t"
                  // load outptr
                  "pld        [%[outptr1], #128]            \n\t"
                  "vld1.f32   {d24, d25}, [%[outptr1]]      \n\t"

                  // inptr1 multiply
                  "vld2.f32   {d10-d13}, [%[inptr1]]!       \n\t"
                  "vld2.f32   {d14, d15}, [%[inptr1]]       \n\t"

                  "vmla.f32   q12, q5, d0[0]                \n\t"
                  "vext.32    q8, q5, q7, #1                \n\t"
                  "vmul.f32   q13, q6, d0[1]                \n\t"

                  "vld2.f32   {d10-d13}, [%[inptr2]]!       \n\t"
                  "vld2.f32   {d14, d15}, [%[inptr2]]       \n\t"
                  "vmla.f32   q12, q8, d1[0]                \n\t"

                  // inptr2 multiply
                  "vmla.f32   q13, q5, d1[1]                \n\t"
                  "vext.32    q8, q5, q7, #1                \n\t"
                  "vmla.f32   q12, q6, d2[0]                \n\t"

                  "vld2.f32   {d10-d13}, [%[inptr3]]!       \n\t"
                  "vld2.f32   {d14, d15}, [%[inptr3]]       \n\t"
                  "vmla.f32   q13, q8, d2[1]                \n\t"

                  // inptr3 multiply
                  "vmla.f32   q12, q5, d3[0]                \n\t"
                  "vext.32    q8, q5, q7, #1                \n\t"
                  "vmla.f32   q13, q6, d3[1]                \n\t"
                  "vmla.f32   q12, q8, d8[0]                \n\t"

                  // store outptr
                  "vadd.f32   q12, q12, q13                 \n\t"
                  "vmax.f32   q12, q12, q9                  \n\t"
                  "vst1.f32   {d24, d25}, [%[outptr1]]!     \n\t"

                  // cycle
                  "subs       %[ow_dim4], #1                \n\t"
                  "bne        0b                            \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
                    [inptr3] "+r"(inptr3)
                  : [f1] "r"(f1), [f9] "r"(f9), [reluptr] "r"(reluptr)
                  : "memory", "q0", "q1", "q4", "q5", "q6", "q7", "q8", "q9",
                    "q12", "q13");
            }
          }
#endif  //__aarch64__
#endif  // __ARM_NEON
          for (; ow < output_width; ++ow) {
            float sum1 = 0;
#if __ARM_NEON
            float32x4_t _inptr1 = vld1q_f32(inptr1);
            float32x4_t _ffilter1 = vld1q_f32(ffilter1);
            float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);

            float32x4_t _inptr2 = vld1q_f32(inptr2);
            float32x4_t _ffilter2 = vld1q_f32(ffilter2);
            _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);

            float32x4_t _inptr3 = vld1q_f32(inptr3);
            float32x4_t _ffilter3 = vld1q_f32(ffilter3);
            _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);

            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ssss1_ssss1 = vpadd_f32(_ss1, _ss1);
            sum1 += vget_lane_f32(_ssss1_ssss1, 0);
#else
            sum1 += inptr1[0] * ffilter1[0];
            sum1 += inptr1[1] * ffilter1[1];
            sum1 += inptr1[2] * ffilter1[2];
            sum1 += inptr2[0] * ffilter2[0];
            sum1 += inptr2[1] * ffilter2[1];
            sum1 += inptr2[2] * ffilter2[2];
            sum1 += inptr3[0] * ffilter3[0];
            sum1 += inptr3[1] * ffilter3[1];
            sum1 += inptr3[2] * ffilter3[2];
#endif
            if (isnopadding) {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;

            } else if (isoddpadding_w && ow == position_w1 ||
                       ow == position_w2 && isoddpadding_w && useallInput_w ||
                       ow == position_w2 + 1 && !isoddpadding_w &&
                           !useallInput_w) {
              ffilter1--;
              ffilter2--;
              ffilter3--;

              inptr1++;
              inptr2++;
              inptr3++;

            } else if (ow < position_w1 || ow > position_w2) {
              ffilter1 -= 2;
              ffilter2 -= 2;
              ffilter3 -= 2;

            } else {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;
            }
            *outptr1 += sum1;

            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }
            outptr1++;
          }
          if (isnopadding) {
            inptr1 += left_stride + input_width;
            inptr2 += left_stride + input_width;
            inptr3 += left_stride + input_width;
          } else if (isoddpadding_h && oh == position_h1 ||
                     oh == position_h2 && isoddpadding_h && useallInput_h ||
                     oh == position_h2 + 1 && !isoddpadding_h &&
                         !useallInput_h) {
            inptr1 += 3;
            inptr2 += 3;
            inptr3 += 3;

            ffilter1 -= left_stride;
            ffilter2 -= left_stride;
            ffilter3 -= left_stride;

          } else if (oh < position_h1 || oh > position_h2) {
            inptr1 -= input_width - 3;
            inptr2 -= input_width - 3;
            inptr3 -= input_width - 3;

            ffilter1 -= 3 + 2 * padding_width + left_stride;
            ffilter2 -= 3 + 2 * padding_width + left_stride;
            ffilter3 -= 3 + 2 * padding_width + left_stride;
          } else {
            ffilter1 += 3 + 2 * padding_width - left_stride;
            ffilter2 += 3 + 2 * padding_width - left_stride;
            ffilter3 += 3 + 2 * padding_width - left_stride;

            inptr1 += input_width + 3;
            inptr2 += input_width + 3;
            inptr3 += input_width + 3;
          }
        }
        filter_data_ch += filter_channel_stride;
        input_data_ch += input_channel_stride;
      }
    }
    input_data += input_batch_stride;
    output_data += output_batch_stride;
  }
}

void SlidingwindowConv3x3s2_8channel(const framework::Tensor *input,
                                     const framework::Tensor *filter,
                                     const std::vector<int> &paddings,
                                     framework::Tensor *output,
                                     framework::Tensor *bias, bool if_bias,
                                     bool if_relu) {
  const int batch_size = input->dims()[0];
  const int input_channels = input->dims()[1];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const int output_channels = output->dims()[1];
  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  const int padding_height = paddings[0];
  const int padding_width = paddings[1];

  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();
  const float *filter_data = filter->data<float>();

  const int input_channel_stride = input_height * input_width;
  const int input_batch_stride = input_channels * input_channel_stride;
  const int output_channel_stride = output_height * output_width;
  const int output_batch_stride = output_channels * output_channel_stride;
  const int filter_channel_stride = 9;
  const int ffilter_length = (2 * padding_height + 3) * (2 * padding_width + 3);
  const int ffilter_start =
      2 * padding_height * (2 * padding_width + 3) + 2 * padding_width;
  const int ffilter_width = 3 + padding_width * 2;

  const float *bias_data;
  bool isnopadding = false;
  const bool useallInput_w = (input_width + 2 * padding_width - 3) % 2 == 0;
  const bool useallInput_h = (input_height + 2 * padding_height - 3) % 2 == 0;
  const bool isoddpadding_w = padding_width % 2 == 1;
  const bool isoddpadding_h = padding_height % 2 == 1;

  int position_w1 = padding_width >> 1;
  int position_h1 = padding_height >> 1;
  int position_w2 = output_width - position_w1 - 2;
  int position_h2 = output_height - position_h1 - 2;
  const int left_stride = input_width + 2 * padding_width - 2 * output_width;

  if (if_bias) {
    bias_data = bias->data<float>();
  }
#if __ARM_NEON
  float *outptr = output_data;
  for (int i = 0; i < output_channels; ++i) {
    float32x4_t _bias;
    if (if_bias) {
      _bias = vld1q_dup_f32(bias_data + i);
    } else {
      _bias = vdupq_n_f32(0.0);
    }
    int dim4 = output_channel_stride >> 2;
    int lef4 = output_channel_stride & 3;

    for (int j = 0; j < dim4; ++j) {
      vst1q_f32(outptr, _bias);
      outptr += 4;
    }
    if (lef4 != 0) {
      if (lef4 >= 1) {
        vst1q_lane_f32(outptr, _bias, 0);
        outptr++;
      }
      if (lef4 >= 2) {
        vst1q_lane_f32(outptr, _bias, 1);
        outptr++;
      }
      if (lef4 >= 3) {
        vst1q_lane_f32(outptr, _bias, 2);
        outptr++;
      }
    }
  }
#else
  int k = 0;
#pragma omp parallel for
  for (int i = 0; i < output_channels; ++i) {
    for (int j = 0; j < output_channel_stride; ++j) {
      if (if_bias) {
        output_data[k++] = bias_data[i];
      } else {
        output_data[k++] = 0;
      }
    }
  }
#endif

  if (padding_height == 0 && padding_width == 0) {
    isnopadding = true;
    position_w1 = -1;
    position_h1 = -1;
    position_w2 = output_width;
    position_h2 = output_height;
  }

  for (int bs = 0; bs < batch_size; ++bs) {
    int output_channels_d8 = output_channels >> 3;

#pragma omp parallel for
    for (int oc2 = 0; oc2 < output_channels_d8; ++oc2) {
      std::atomic<float> reluvalue{0};
      int oc = oc2 * 8;
      const float *f1;
      const float *inptr1, *inptr2, *inptr3;
      const float *ffilter1, *ffilter2, *ffilter3;
      const float *ffilter1_2, *ffilter2_2, *ffilter3_2;
      const float *ffilter1_3, *ffilter2_3, *ffilter3_3;
      const float *ffilter1_4, *ffilter2_4, *ffilter3_4;
      const float *ffilter1_5, *ffilter2_5, *ffilter3_5;
      const float *ffilter1_6, *ffilter2_6, *ffilter3_6;
      const float *ffilter1_7, *ffilter2_7, *ffilter3_7;
      const float *ffilter1_8, *ffilter2_8, *ffilter3_8;

      float reformedfilter_arr[76] = {0};
      float ffilterarray[ffilter_length] = {0};
      float ffilterarray_2[ffilter_length] = {0};
      float ffilterarray_3[ffilter_length] = {0};
      float ffilterarray_4[ffilter_length] = {0};
      float ffilterarray_5[ffilter_length] = {0};
      float ffilterarray_6[ffilter_length] = {0};
      float ffilterarray_7[ffilter_length] = {0};
      float ffilterarray_8[ffilter_length] = {0};

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

      filter_data_ch =
          filter_data + oc * filter_channel_stride * input_channels;
      filter_data_ch_2 =
          filter_data + (oc + 1) * filter_channel_stride * input_channels;
      filter_data_ch_3 =
          filter_data + (oc + 2) * filter_channel_stride * input_channels;
      filter_data_ch_4 =
          filter_data + (oc + 3) * filter_channel_stride * input_channels;
      filter_data_ch_5 =
          filter_data + (oc + 4) * filter_channel_stride * input_channels;
      filter_data_ch_6 =
          filter_data + (oc + 5) * filter_channel_stride * input_channels;
      filter_data_ch_7 =
          filter_data + (oc + 6) * filter_channel_stride * input_channels;
      filter_data_ch_8 =
          filter_data + (oc + 7) * filter_channel_stride * input_channels;

      input_data_ch = input_data;
      output_data_ch = output_data + oc * output_channel_stride;
      output_data_ch_2 = output_data + (oc + 1) * output_channel_stride;
      output_data_ch_3 = output_data + (oc + 2) * output_channel_stride;
      output_data_ch_4 = output_data + (oc + 3) * output_channel_stride;
      output_data_ch_5 = output_data + (oc + 4) * output_channel_stride;
      output_data_ch_6 = output_data + (oc + 5) * output_channel_stride;
      output_data_ch_7 = output_data + (oc + 6) * output_channel_stride;
      output_data_ch_8 = output_data + (oc + 7) * output_channel_stride;

      for (int ic = 0; ic < input_channels; ++ic) {
        int k = 0;
        for (int i = 0; i < 9; ++i) {
          for (int j = 0; j < 8; ++j) {
            reformedfilter_arr[k++] =
                filter_data_ch[i + input_channels * 9 * j];
          }
        }

        if (if_relu && ic == input_channels - 1) {
          reluvalue = 0;
        } else {
          reluvalue = -FLT_MAX;
        }
        reformedfilter_arr[72] = reluvalue;
        reformedfilter_arr[73] = reluvalue;
        reformedfilter_arr[74] = reluvalue;
        reformedfilter_arr[75] = reluvalue;

        f1 = reformedfilter_arr;

        if (!isnopadding) {
          ffilterarray[ffilter_length] = {0};

          for (int i = 0; i < 9; ++i) {
            int j = i / 3 * (2 * padding_width + 3) + i % 3 +
                    padding_height * 3 +
                    padding_width * (2 * padding_height + 1);
            ffilterarray[j] = filter_data_ch[i];
            ffilterarray_2[j] = filter_data_ch_2[i];
            ffilterarray_3[j] = filter_data_ch_3[i];
            ffilterarray_4[j] = filter_data_ch_4[i];
            ffilterarray_5[j] = filter_data_ch_5[i];
            ffilterarray_6[j] = filter_data_ch_6[i];
            ffilterarray_7[j] = filter_data_ch_7[i];
            ffilterarray_8[j] = filter_data_ch_8[i];
          }

          ffilter1 = ffilterarray;
          ffilter1 += ffilter_start;
          ffilter2 = ffilter1 + ffilter_width;
          ffilter3 = ffilter2 + ffilter_width;

          ffilter1_2 = ffilterarray_2;
          ffilter1_2 += ffilter_start;
          ffilter2_2 = ffilter1_2 + ffilter_width;
          ffilter3_2 = ffilter2_2 + ffilter_width;

          ffilter1_3 = ffilterarray_3;
          ffilter1_3 += ffilter_start;
          ffilter2_3 = ffilter1_3 + ffilter_width;
          ffilter3_3 = ffilter2_3 + ffilter_width;

          ffilter1_4 = ffilterarray_4;
          ffilter1_4 += ffilter_start;
          ffilter2_4 = ffilter1_4 + ffilter_width;
          ffilter3_4 = ffilter2_4 + ffilter_width;

          ffilter1_5 = ffilterarray_5;
          ffilter1_5 += ffilter_start;
          ffilter2_5 = ffilter1_5 + ffilter_width;
          ffilter3_5 = ffilter2_5 + ffilter_width;

          ffilter1_6 = ffilterarray_6;
          ffilter1_6 += ffilter_start;
          ffilter2_6 = ffilter1_6 + ffilter_width;
          ffilter3_6 = ffilter2_6 + ffilter_width;

          ffilter1_7 = ffilterarray_7;
          ffilter1_7 += ffilter_start;
          ffilter2_7 = ffilter1_7 + ffilter_width;
          ffilter3_7 = ffilter2_7 + ffilter_width;

          ffilter1_8 = ffilterarray_8;
          ffilter1_8 += ffilter_start;
          ffilter2_8 = ffilter1_8 + ffilter_width;
          ffilter3_8 = ffilter2_8 + ffilter_width;
        } else {
          ffilter1 = filter_data_ch;
          ffilter2 = ffilter1 + 3;
          ffilter3 = ffilter2 + 3;

          ffilter1_2 = filter_data_ch_2;
          ffilter2_2 = ffilter1_2 + 3;
          ffilter3_2 = ffilter2_2 + 3;

          ffilter1_3 = filter_data_ch_3;
          ffilter2_3 = ffilter1_3 + 3;
          ffilter3_3 = ffilter2_3 + 3;

          ffilter1_4 = filter_data_ch_4;
          ffilter2_4 = ffilter1_4 + 3;
          ffilter3_4 = ffilter2_4 + 3;

          ffilter1_5 = filter_data_ch_5;
          ffilter2_5 = ffilter1_5 + 3;
          ffilter3_5 = ffilter2_5 + 3;

          ffilter1_6 = filter_data_ch_6;
          ffilter2_6 = ffilter1_6 + 3;
          ffilter3_6 = ffilter2_6 + 3;

          ffilter1_7 = filter_data_ch_7;
          ffilter2_7 = ffilter1_7 + 3;
          ffilter3_7 = ffilter2_7 + 3;

          ffilter1_8 = filter_data_ch_8;
          ffilter2_8 = ffilter1_8 + 3;
          ffilter3_8 = ffilter2_8 + 3;
        }
        float *outptr1;
        float *outptr1_2;
        float *outptr1_3;
        float *outptr1_4;
        float *outptr1_5;
        float *outptr1_6;
        float *outptr1_7;
        float *outptr1_8;

        outptr1 = output_data_ch;
        outptr1_2 = output_data_ch_2;
        outptr1_3 = output_data_ch_3;
        outptr1_4 = output_data_ch_4;
        outptr1_5 = output_data_ch_5;
        outptr1_6 = output_data_ch_6;
        outptr1_7 = output_data_ch_7;
        outptr1_8 = output_data_ch_8;

        inptr1 = input_data_ch;
        inptr2 = inptr1 + input_width;
        inptr3 = inptr2 + input_width;

        int oh = 0;

        for (; oh < output_height; ++oh) {
          int ow = 0;

          for (; ow <= position_w1; ++ow) {
            float sum1 = 0;
            float sum1_2 = 0;
            float sum1_3 = 0;
            float sum1_4 = 0;
            float sum1_5 = 0;
            float sum1_6 = 0;
            float sum1_7 = 0;
            float sum1_8 = 0;
#if __ARM_NEON
            float32x4_t _inptr1 = vld1q_f32(inptr1);
            float32x4_t _ffilter1 = vld1q_f32(ffilter1);
            float32x4_t _ffilter1_2 = vld1q_f32(ffilter1_2);
            float32x4_t _ffilter1_3 = vld1q_f32(ffilter1_3);
            float32x4_t _ffilter1_4 = vld1q_f32(ffilter1_4);
            float32x4_t _ffilter1_5 = vld1q_f32(ffilter1_5);
            float32x4_t _ffilter1_6 = vld1q_f32(ffilter1_6);
            float32x4_t _ffilter1_7 = vld1q_f32(ffilter1_7);
            float32x4_t _ffilter1_8 = vld1q_f32(ffilter1_8);

            float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);
            float32x4_t _sum1_2 = vmulq_f32(_inptr1, _ffilter1_2);
            float32x4_t _sum1_3 = vmulq_f32(_inptr1, _ffilter1_3);
            float32x4_t _sum1_4 = vmulq_f32(_inptr1, _ffilter1_4);
            float32x4_t _sum1_5 = vmulq_f32(_inptr1, _ffilter1_5);
            float32x4_t _sum1_6 = vmulq_f32(_inptr1, _ffilter1_6);
            float32x4_t _sum1_7 = vmulq_f32(_inptr1, _ffilter1_7);
            float32x4_t _sum1_8 = vmulq_f32(_inptr1, _ffilter1_8);

            float32x4_t _inptr2 = vld1q_f32(inptr2);
            float32x4_t _ffilter2 = vld1q_f32(ffilter2);
            float32x4_t _ffilter2_2 = vld1q_f32(ffilter2_2);
            float32x4_t _ffilter2_3 = vld1q_f32(ffilter2_3);
            float32x4_t _ffilter2_4 = vld1q_f32(ffilter2_4);
            float32x4_t _ffilter2_5 = vld1q_f32(ffilter2_5);
            float32x4_t _ffilter2_6 = vld1q_f32(ffilter2_6);
            float32x4_t _ffilter2_7 = vld1q_f32(ffilter2_7);
            float32x4_t _ffilter2_8 = vld1q_f32(ffilter2_8);

            _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);
            _sum1_2 = vmlaq_f32(_sum1_2, _inptr2, _ffilter2_2);
            _sum1_3 = vmlaq_f32(_sum1_3, _inptr2, _ffilter2_3);
            _sum1_4 = vmlaq_f32(_sum1_4, _inptr2, _ffilter2_4);
            _sum1_5 = vmlaq_f32(_sum1_5, _inptr2, _ffilter2_5);
            _sum1_6 = vmlaq_f32(_sum1_6, _inptr2, _ffilter2_6);
            _sum1_7 = vmlaq_f32(_sum1_7, _inptr2, _ffilter2_7);
            _sum1_8 = vmlaq_f32(_sum1_8, _inptr2, _ffilter2_8);

            float32x4_t _inptr3 = vld1q_f32(inptr3);
            float32x4_t _ffilter3 = vld1q_f32(ffilter3);
            float32x4_t _ffilter3_2 = vld1q_f32(ffilter3_2);
            float32x4_t _ffilter3_3 = vld1q_f32(ffilter3_3);
            float32x4_t _ffilter3_4 = vld1q_f32(ffilter3_4);
            float32x4_t _ffilter3_5 = vld1q_f32(ffilter3_5);
            float32x4_t _ffilter3_6 = vld1q_f32(ffilter3_6);
            float32x4_t _ffilter3_7 = vld1q_f32(ffilter3_7);
            float32x4_t _ffilter3_8 = vld1q_f32(ffilter3_8);

            _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
            _sum1_2 = vmlaq_f32(_sum1_2, _inptr3, _ffilter3_2);
            _sum1_3 = vmlaq_f32(_sum1_3, _inptr3, _ffilter3_3);
            _sum1_4 = vmlaq_f32(_sum1_4, _inptr3, _ffilter3_4);
            _sum1_5 = vmlaq_f32(_sum1_5, _inptr3, _ffilter3_5);
            _sum1_6 = vmlaq_f32(_sum1_6, _inptr3, _ffilter3_6);
            _sum1_7 = vmlaq_f32(_sum1_7, _inptr3, _ffilter3_7);
            _sum1_8 = vmlaq_f32(_sum1_8, _inptr3, _ffilter3_8);

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
            sum1 += inptr1[0] * ffilter1[0];
            sum1 += inptr1[1] * ffilter1[1];
            sum1 += inptr1[2] * ffilter1[2];
            sum1 += inptr2[0] * ffilter2[0];
            sum1 += inptr2[1] * ffilter2[1];
            sum1 += inptr2[2] * ffilter2[2];
            sum1 += inptr3[0] * ffilter3[0];
            sum1 += inptr3[1] * ffilter3[1];
            sum1 += inptr3[2] * ffilter3[2];

            sum1_2 += inptr1[0] * ffilter1_2[0];
            sum1_2 += inptr1[1] * ffilter1_2[1];
            sum1_2 += inptr1[2] * ffilter1_2[2];
            sum1_2 += inptr2[0] * ffilter2_2[0];
            sum1_2 += inptr2[1] * ffilter2_2[1];
            sum1_2 += inptr2[2] * ffilter2_2[2];
            sum1_2 += inptr3[0] * ffilter3_2[0];
            sum1_2 += inptr3[1] * ffilter3_2[1];
            sum1_2 += inptr3[2] * ffilter3_2[2];

            sum1_3 += inptr1[0] * ffilter1_3[0];
            sum1_3 += inptr1[1] * ffilter1_3[1];
            sum1_3 += inptr1[2] * ffilter1_3[2];
            sum1_3 += inptr2[0] * ffilter2_3[0];
            sum1_3 += inptr2[1] * ffilter2_3[1];
            sum1_3 += inptr2[2] * ffilter2_3[2];
            sum1_3 += inptr3[0] * ffilter3_3[0];
            sum1_3 += inptr3[1] * ffilter3_3[1];
            sum1_3 += inptr3[2] * ffilter3_3[2];

            sum1_4 += inptr1[0] * ffilter1_4[0];
            sum1_4 += inptr1[1] * ffilter1_4[1];
            sum1_4 += inptr1[2] * ffilter1_4[2];
            sum1_4 += inptr2[0] * ffilter2_4[0];
            sum1_4 += inptr2[1] * ffilter2_4[1];
            sum1_4 += inptr2[2] * ffilter2_4[2];
            sum1_4 += inptr3[0] * ffilter3_4[0];
            sum1_4 += inptr3[1] * ffilter3_4[1];
            sum1_4 += inptr3[2] * ffilter3_4[2];

            sum1_5 += inptr1[0] * ffilter1_5[0];
            sum1_5 += inptr1[1] * ffilter1_5[1];
            sum1_5 += inptr1[2] * ffilter1_5[2];
            sum1_5 += inptr2[0] * ffilter2_5[0];
            sum1_5 += inptr2[1] * ffilter2_5[1];
            sum1_5 += inptr2[2] * ffilter2_5[2];
            sum1_5 += inptr3[0] * ffilter3_5[0];
            sum1_5 += inptr3[1] * ffilter3_5[1];
            sum1_5 += inptr3[2] * ffilter3_5[2];

            sum1_6 += inptr1[0] * ffilter1_6[0];
            sum1_6 += inptr1[1] * ffilter1_6[1];
            sum1_6 += inptr1[2] * ffilter1_6[2];
            sum1_6 += inptr2[0] * ffilter2_6[0];
            sum1_6 += inptr2[1] * ffilter2_6[1];
            sum1_6 += inptr2[2] * ffilter2_6[2];
            sum1_6 += inptr3[0] * ffilter3_6[0];
            sum1_6 += inptr3[1] * ffilter3_6[1];
            sum1_6 += inptr3[2] * ffilter3_6[2];

            sum1_7 += inptr1[0] * ffilter1_7[0];
            sum1_7 += inptr1[1] * ffilter1_7[1];
            sum1_7 += inptr1[2] * ffilter1_7[2];
            sum1_7 += inptr2[0] * ffilter2_7[0];
            sum1_7 += inptr2[1] * ffilter2_7[1];
            sum1_7 += inptr2[2] * ffilter2_7[2];
            sum1_7 += inptr3[0] * ffilter3_7[0];
            sum1_7 += inptr3[1] * ffilter3_7[1];
            sum1_7 += inptr3[2] * ffilter3_7[2];

            sum1_8 += inptr1[0] * ffilter1_8[0];
            sum1_8 += inptr1[1] * ffilter1_8[1];
            sum1_8 += inptr1[2] * ffilter1_8[2];
            sum1_8 += inptr2[0] * ffilter2_8[0];
            sum1_8 += inptr2[1] * ffilter2_8[1];
            sum1_8 += inptr2[2] * ffilter2_8[2];
            sum1_8 += inptr3[0] * ffilter3_8[0];
            sum1_8 += inptr3[1] * ffilter3_8[1];
            sum1_8 += inptr3[2] * ffilter3_8[2];
#endif
            if (isnopadding) {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;

            } else if (input_width > 3 &&
                       (isoddpadding_w && ow == position_w1 ||
                        ow == position_w2 && isoddpadding_w && useallInput_w ||
                        ow == position_w2 + 1 && !isoddpadding_w &&
                            !useallInput_w)) {
              ffilter1--;
              ffilter2--;
              ffilter3--;
              ffilter1_2--;
              ffilter2_2--;
              ffilter3_2--;

              ffilter1_3--;
              ffilter2_3--;
              ffilter3_3--;
              ffilter1_4--;
              ffilter2_4--;
              ffilter3_4--;

              ffilter1_5--;
              ffilter2_5--;
              ffilter3_5--;
              ffilter1_6--;
              ffilter2_6--;
              ffilter3_6--;

              ffilter1_7--;
              ffilter2_7--;
              ffilter3_7--;
              ffilter1_8--;
              ffilter2_8--;
              ffilter3_8--;

              inptr1++;
              inptr2++;
              inptr3++;

            } else if (input_width <= 3 || ow < position_w1 ||
                       ow > position_w2) {
              ffilter1 -= 2;
              ffilter2 -= 2;
              ffilter3 -= 2;
              ffilter1_2 -= 2;
              ffilter2_2 -= 2;
              ffilter3_2 -= 2;

              ffilter1_3 -= 2;
              ffilter2_3 -= 2;
              ffilter3_3 -= 2;
              ffilter1_4 -= 2;
              ffilter2_4 -= 2;
              ffilter3_4 -= 2;

              ffilter1_5 -= 2;
              ffilter2_5 -= 2;
              ffilter3_5 -= 2;
              ffilter1_6 -= 2;
              ffilter2_6 -= 2;
              ffilter3_6 -= 2;

              ffilter1_7 -= 2;
              ffilter2_7 -= 2;
              ffilter3_7 -= 2;
              ffilter1_8 -= 2;
              ffilter2_8 -= 2;
              ffilter3_8 -= 2;
            } else {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;
            }
            *outptr1 += sum1;
            *outptr1_2 += sum1_2;
            *outptr1_3 += sum1_3;
            *outptr1_4 += sum1_4;
            *outptr1_5 += sum1_5;
            *outptr1_6 += sum1_6;
            *outptr1_7 += sum1_7;
            *outptr1_8 += sum1_8;

            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }
            if (outptr1_2[0] < reluvalue) {
              *outptr1_2 = reluvalue;
            }
            if (outptr1_3[0] < reluvalue) {
              *outptr1_3 = reluvalue;
            }
            if (outptr1_4[0] < reluvalue) {
              *outptr1_4 = reluvalue;
            }
            if (outptr1_5[0] < reluvalue) {
              *outptr1_5 = reluvalue;
            }
            if (outptr1_6[0] < reluvalue) {
              *outptr1_6 = reluvalue;
            }
            if (outptr1_7[0] < reluvalue) {
              *outptr1_7 = reluvalue;
            }
            if (outptr1_8[0] < reluvalue) {
              *outptr1_8 = reluvalue;
            }

            outptr1++;
            outptr1_2++;
            outptr1_3++;
            outptr1_4++;
            outptr1_5++;
            outptr1_6++;
            outptr1_7++;
            outptr1_8++;
          }

#if __ARM_NEON
#if __aarch64__
          if (oh > position_h1 && oh <= position_h2) {
            int ow_dim4 = (position_w2 - position_w1 - 1) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
              asm volatile(

                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "prfm  pldl1keep, [%[inptr1], #288]         \n\t"

                  "ld1  {v0.4s, v1.4s}, [%[f1]], #32          \n\t"
                  "ld2   {v4.4s, v5.4s}, [%[inptr1]], #32     \n\t"
                  "ld2   {v6.4s, v7.4s}, [%[inptr1]]          \n\t"
                  "0:                                         \n\t"
                  // load outptr
                  "prfm  pldl1keep, [%[outptr1], #128]        \n\t"
                  "prfm  pldl1keep, [%[outptr1_2], #128]      \n\t"
                  "prfm  pldl1keep, [%[outptr1_3], #128]      \n\t"
                  "prfm  pldl1keep, [%[outptr1_4], #128]      \n\t"
                  "prfm  pldl1keep, [%[outptr1_5], #128]      \n\t"
                  "prfm  pldl1keep, [%[outptr1_6], #128]      \n\t"
                  "prfm  pldl1keep, [%[outptr1_7], #128]      \n\t"
                  "prfm  pldl1keep, [%[outptr1_8], #128]      \n\t"

                  "ld1   {v8.4s}, [%[outptr1]]                \n\t"
                  "ld1   {v9.4s}, [%[outptr1_2]]              \n\t"
                  "ld1   {v10.4s}, [%[outptr1_3]]             \n\t"
                  "ld1   {v11.4s}, [%[outptr1_4]]             \n\t"
                  "ld1   {v12.4s}, [%[outptr1_5]]             \n\t"
                  "ld1   {v13.4s}, [%[outptr1_6]]             \n\t"
                  "ld1   {v14.4s}, [%[outptr1_7]]             \n\t"
                  "ld1   {v15.4s}, [%[outptr1_8]]             \n\t"

                  // inptr1 multiply
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

                  "prfm  pldl1keep, [%[inptr2], #288]         \n\t"
                  "ld2    {v4.4s, v5.4s}, [%[inptr2]], #32    \n\t"
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

                  // inptr2 multiply

                  "ld2    {v6.4s, v7.4s}, [%[inptr2]]         \n\t"
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
                  "prfm  pldl1keep, [%[inptr3], #288]         \n\t"
                  "fmla   v14.4s, v5.4s, v1.4s[2]             \n\t"
                  "fmla   v15.4s, v5.4s, v1.4s[3]             \n\t"

                  "ld1  {v0.4s, v1.4s}, [%[f1]], #32          \n\t"
                  "ld2   {v4.4s, v5.4s}, [%[inptr3]], #32     \n\t"
                  "fmla    v8.4s, v7.4s, v2.4s[0]             \n\t"
                  "fmla    v9.4s, v7.4s, v2.4s[1]             \n\t"
                  "fmla   v10.4s, v7.4s, v2.4s[2]             \n\t"
                  "fmla   v11.4s, v7.4s, v2.4s[3]             \n\t"

                  "fmla   v12.4s, v7.4s, v3.4s[0]             \n\t"
                  "fmla   v13.4s, v7.4s, v3.4s[1]             \n\t"
                  "fmla   v14.4s, v7.4s, v3.4s[2]             \n\t"
                  "fmla   v15.4s, v7.4s, v3.4s[3]             \n\t"

                  // inptr3 multiply
                  "ld2   {v6.4s, v7.4s}, [%[inptr3]]          \n\t"
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

                  // store outptr
                  "prfm  pldl1keep, [%[f1], #256]             \n\t"
                  "fmax    v8.4s,  v8.4s, v2.4s               \n\t"
                  "prfm  pldl1keep, [%[inptr1], #288]         \n\t"
                  "fmax    v9.4s,  v9.4s, v2.4s               \n\t"

                  "ld1  {v0.4s, v1.4s}, [%[f1]], #32          \n\t"
                  "fmax   v10.4s, v10.4s, v2.4s               \n\t"

                  "ld2   {v4.4s, v5.4s}, [%[inptr1]], #32     \n\t"
                  "st1   {v8.4s}, [%[outptr1]], #16           \n\t"
                  "fmax   v11.4s, v11.4s, v2.4s               \n\t"
                  "st1   {v9.4s}, [%[outptr1_2]], #16         \n\t"

                  "fmax   v12.4s, v12.4s, v2.4s               \n\t"
                  "st1   {v10.4s}, [%[outptr1_3]], #16        \n\t"
                  "fmax   v13.4s, v13.4s, v2.4s               \n\t"
                  "st1   {v11.4s}, [%[outptr1_4]], #16        \n\t"

                  "fmax   v14.4s, v14.4s, v2.4s               \n\t"
                  "st1   {v12.4s}, [%[outptr1_5]], #16        \n\t"
                  "fmax   v15.4s, v15.4s, v2.4s               \n\t"
                  "st1   {v13.4s}, [%[outptr1_6]], #16        \n\t"

                  "ld2   {v6.4s, v7.4s}, [%[inptr1]]          \n\t"
                  "st1   {v14.4s}, [%[outptr1_7]], #16        \n\t"
                  "subs       %[ow_dim4], %[ow_dim4], #1      \n\t"
                  "st1   {v15.4s}, [%[outptr1_8]], #16        \n\t"

                  // cycle
                  "bne        0b                              \n\t"
                  "sub       %[f1], %[inptr1], #32            \n\t"
                  "sub       %[inptr1], %[inptr1], #32        \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [outptr1_2] "+r"(outptr1_2), [outptr1_3] "+r"(outptr1_3),
                    [outptr1_4] "+r"(outptr1_4), [outptr1_5] "+r"(outptr1_5),
                    [outptr1_6] "+r"(outptr1_6), [outptr1_7] "+r"(outptr1_7),
                    [outptr1_8] "+r"(outptr1_8), [inptr1] "+r"(inptr1),
                    [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3)
                  : [f1] "r"(f1)
                  : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                    "v8", "v9", "v12", "v13", "v14", "v15");
            }
          }
#else
          if (oh > position_h1 && oh <= position_h2) {
            int ow_dim4 = (position_w2 - position_w1 - 1) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
              asm volatile(

                  "pld        [%[f1], #256]                   \n\t"
                  "pld        [%[inptr1], #288]               \n\t"

                  "vld1.32   {d0-d3}, [%[f1]]!                \n\t"
                  "vld2.f32   {d8-d11}, [%[inptr1]]!          \n\t"
                  "vld2.f32   {d12, d13}, [%[inptr1]]         \n\t"

                  "0:                                         \n\t"
                  // load outptr
                  "pld        [%[outptr1], #128]              \n\t"
                  "pld        [%[outptr1_2], #128]            \n\t"
                  "pld        [%[outptr1_3], #128]            \n\t"
                  "pld        [%[outptr1_4], #128]            \n\t"
                  "pld        [%[outptr1_5], #128]            \n\t"
                  "pld        [%[outptr1_6], #128]            \n\t"
                  "pld        [%[outptr1_7], #128]            \n\t"
                  "pld        [%[outptr1_8], #128]            \n\t"

                  "vld1.f32   {d16, d17}, [%[outptr1]]        \n\t"
                  "vld1.f32   {d18, d19}, [%[outptr1_2]]      \n\t"
                  "vld1.f32   {d20, d21}, [%[outptr1_3]]      \n\t"
                  "vld1.f32   {d22, d23}, [%[outptr1_4]]      \n\t"
                  "vld1.f32   {d24, d25}, [%[outptr1_5]]      \n\t"
                  "vld1.f32   {d26, d27}, [%[outptr1_6]]      \n\t"
                  "vld1.f32   {d28, d29}, [%[outptr1_7]]      \n\t"
                  "vld1.f32   {d30, d31}, [%[outptr1_8]]      \n\t"

                  // inptr1 multiply
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

                  "vext.32    q7, q4, q6, #1                 \n\t"
                  "vmla.f32   q10, q5, d5[0]                  \n\t"
                  "vmla.f32   q11, q5, d5[1]                  \n\t"

                  "vmla.f32   q12, q5, d6[0]                  \n\t"
                  "vmla.f32   q13, q5, d6[1]                  \n\t"

                  "pld        [%[inptr2], #288]               \n\t"
                  "vmla.f32   q14, q5, d7[0]                  \n\t"
                  "vmla.f32   q15, q5, d7[1]                  \n\t"

                  "vld2.f32   {d8-d11}, [%[inptr2]]!          \n\t"
                  "vmla.f32   q8, q7, d0[0]                   \n\t"
                  "vmla.f32   q9, q7, d0[1]                   \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vld1.32   {d4-d7}, [%[f1]]!                \n\t"
                  "vmla.f32   q10, q7, d1[0]                  \n\t"
                  "vmla.f32   q11, q7, d1[1]                  \n\t"

                  "vld2.f32   {d12, d13}, [%[inptr2]]         \n\t"
                  "vmla.f32   q12, q7, d2[0]                  \n\t"
                  "vmla.f32   q13, q7, d2[1]                  \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vmla.f32   q14, q7, d3[0]                  \n\t"
                  "vmla.f32   q15, q7, d3[1]                  \n\t"

                  // inptr2 multiply
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

                  "pld        [%[inptr3], #288]               \n\t"
                  "vmla.f32   q14, q5, d3[0]                  \n\t"
                  "vmla.f32   q15, q5, d3[1]                  \n\t"

                  "vld2.f32   {d8-d11}, [%[inptr3]]!          \n\t"
                  "vmla.f32   q8, q7, d4[0]                   \n\t"
                  "vmla.f32   q9, q7, d4[1]                   \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vld1.32   {d0-d3}, [%[f1]]!                \n\t"
                  "vmla.f32   q10, q7, d5[0]                  \n\t"
                  "vmla.f32   q11, q7, d5[1]                  \n\t"

                  "vld2.f32   {d12, d13}, [%[inptr3]]         \n\t"
                  "vmla.f32   q12, q7, d6[0]                  \n\t"
                  "vmla.f32   q13, q7, d6[1]                  \n\t"

                  "pld        [%[f1], #256]                   \n\t"
                  "vmla.f32   q14, q7, d7[0]                  \n\t"
                  "vmla.f32   q15, q7, d7[1]                  \n\t"

                  // inptr3 multiply
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

                  // store outptr
                  "vmax.f32   q8, q8, q2                      \n\t"
                  "vmax.f32   q9, q9, q2                      \n\t"

                  "vmax.f32   q10, q10, q2                    \n\t"
                  "pld        [%[f1], #256]                   \n\t"
                  "vld1.32   {d0-d3}, [%[f1]]!                \n\t"

                  "pld        [%[inptr1], #288]               \n\t"
                  "vld2.f32   {d8-d11}, [%[inptr1]]!          \n\t"
                  "vst1.f32   {d16, d17}, [%[outptr1]]!       \n\t"

                  "vmax.f32   q11, q11, q2                    \n\t"
                  "vst1.f32   {d18, d19}, [%[outptr1_2]]!     \n\t"

                  "vmax.f32   q12, q12, q2                    \n\t"
                  "vst1.f32   {d20, d21}, [%[outptr1_3]]!     \n\t"

                  "vmax.f32   q13, q13, q2                    \n\t"
                  "vst1.f32   {d22, d23}, [%[outptr1_4]]!     \n\t"

                  "vmax.f32   q14, q14, q2                    \n\t"
                  "vst1.f32   {d24, d25}, [%[outptr1_5]]!     \n\t"

                  "vmax.f32   q15, q15, q2                    \n\t"
                  "vst1.f32   {d26, d27}, [%[outptr1_6]]!     \n\t"

                  "vld2.f32   {d12, d13}, [%[inptr1]]         \n\t"
                  "vst1.f32   {d28, d29}, [%[outptr1_7]]!     \n\t"

                  "subs       %[ow_dim4], #1                  \n\t"
                  "vst1.f32   {d30, d31}, [%[outptr1_8]]!     \n\t"

                  // cycle
                  "bne        0b                              \n\t"
                  "sub        %[f1], %[f1], #32               \n\t"
                  "sub        %[inptr1], %[inptr1], #32       \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [outptr1_2] "+r"(outptr1_2), [outptr1_3] "+r"(outptr1_3),
                    [outptr1_4] "+r"(outptr1_4), [outptr1_5] "+r"(outptr1_5),
                    [outptr1_6] "+r"(outptr1_6), [outptr1_7] "+r"(outptr1_7),
                    [outptr1_8] "+r"(outptr1_8), [inptr1] "+r"(inptr1),
                    [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3)
                  : [f1] "r"(f1)
                  : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                    "q8", "q9", "q12", "q13", "q14", "q15");
            }
          }
#endif  //__aarch64__
#endif  // __ARM_NEON

          for (; ow < output_width; ++ow) {
            float sum1 = 0;
            float sum1_2 = 0;
            float sum1_3 = 0;
            float sum1_4 = 0;
            float sum1_5 = 0;
            float sum1_6 = 0;
            float sum1_7 = 0;
            float sum1_8 = 0;
#if __ARM_NEON
            float32x4_t _inptr1 = vld1q_f32(inptr1);
            float32x4_t _ffilter1 = vld1q_f32(ffilter1);
            float32x4_t _ffilter1_2 = vld1q_f32(ffilter1_2);
            float32x4_t _ffilter1_3 = vld1q_f32(ffilter1_3);
            float32x4_t _ffilter1_4 = vld1q_f32(ffilter1_4);
            float32x4_t _ffilter1_5 = vld1q_f32(ffilter1_5);
            float32x4_t _ffilter1_6 = vld1q_f32(ffilter1_6);
            float32x4_t _ffilter1_7 = vld1q_f32(ffilter1_7);
            float32x4_t _ffilter1_8 = vld1q_f32(ffilter1_8);

            float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);
            float32x4_t _sum1_2 = vmulq_f32(_inptr1, _ffilter1_2);
            float32x4_t _sum1_3 = vmulq_f32(_inptr1, _ffilter1_3);
            float32x4_t _sum1_4 = vmulq_f32(_inptr1, _ffilter1_4);
            float32x4_t _sum1_5 = vmulq_f32(_inptr1, _ffilter1_5);
            float32x4_t _sum1_6 = vmulq_f32(_inptr1, _ffilter1_6);
            float32x4_t _sum1_7 = vmulq_f32(_inptr1, _ffilter1_7);
            float32x4_t _sum1_8 = vmulq_f32(_inptr1, _ffilter1_8);

            float32x4_t _inptr2 = vld1q_f32(inptr2);
            float32x4_t _ffilter2 = vld1q_f32(ffilter2);
            float32x4_t _ffilter2_2 = vld1q_f32(ffilter2_2);
            float32x4_t _ffilter2_3 = vld1q_f32(ffilter2_3);
            float32x4_t _ffilter2_4 = vld1q_f32(ffilter2_4);
            float32x4_t _ffilter2_5 = vld1q_f32(ffilter2_5);
            float32x4_t _ffilter2_6 = vld1q_f32(ffilter2_6);
            float32x4_t _ffilter2_7 = vld1q_f32(ffilter2_7);
            float32x4_t _ffilter2_8 = vld1q_f32(ffilter2_8);

            _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);
            _sum1_2 = vmlaq_f32(_sum1_2, _inptr2, _ffilter2_2);
            _sum1_3 = vmlaq_f32(_sum1_3, _inptr2, _ffilter2_3);
            _sum1_4 = vmlaq_f32(_sum1_4, _inptr2, _ffilter2_4);
            _sum1_5 = vmlaq_f32(_sum1_5, _inptr2, _ffilter2_5);
            _sum1_6 = vmlaq_f32(_sum1_6, _inptr2, _ffilter2_6);
            _sum1_7 = vmlaq_f32(_sum1_7, _inptr2, _ffilter2_7);
            _sum1_8 = vmlaq_f32(_sum1_8, _inptr2, _ffilter2_8);

            float32x4_t _inptr3 = vld1q_f32(inptr3);
            float32x4_t _ffilter3 = vld1q_f32(ffilter3);
            float32x4_t _ffilter3_2 = vld1q_f32(ffilter3_2);
            float32x4_t _ffilter3_3 = vld1q_f32(ffilter3_3);
            float32x4_t _ffilter3_4 = vld1q_f32(ffilter3_4);
            float32x4_t _ffilter3_5 = vld1q_f32(ffilter3_5);
            float32x4_t _ffilter3_6 = vld1q_f32(ffilter3_6);
            float32x4_t _ffilter3_7 = vld1q_f32(ffilter3_7);
            float32x4_t _ffilter3_8 = vld1q_f32(ffilter3_8);

            _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);
            _sum1_2 = vmlaq_f32(_sum1_2, _inptr3, _ffilter3_2);
            _sum1_3 = vmlaq_f32(_sum1_3, _inptr3, _ffilter3_3);
            _sum1_4 = vmlaq_f32(_sum1_4, _inptr3, _ffilter3_4);
            _sum1_5 = vmlaq_f32(_sum1_5, _inptr3, _ffilter3_5);
            _sum1_6 = vmlaq_f32(_sum1_6, _inptr3, _ffilter3_6);
            _sum1_7 = vmlaq_f32(_sum1_7, _inptr3, _ffilter3_7);
            _sum1_8 = vmlaq_f32(_sum1_8, _inptr3, _ffilter3_8);

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
            sum1 += inptr1[0] * ffilter1[0];
            sum1 += inptr1[1] * ffilter1[1];
            sum1 += inptr1[2] * ffilter1[2];
            sum1 += inptr2[0] * ffilter2[0];
            sum1 += inptr2[1] * ffilter2[1];
            sum1 += inptr2[2] * ffilter2[2];
            sum1 += inptr3[0] * ffilter3[0];
            sum1 += inptr3[1] * ffilter3[1];
            sum1 += inptr3[2] * ffilter3[2];

            sum1_2 += inptr1[0] * ffilter1_2[0];
            sum1_2 += inptr1[1] * ffilter1_2[1];
            sum1_2 += inptr1[2] * ffilter1_2[2];
            sum1_2 += inptr2[0] * ffilter2_2[0];
            sum1_2 += inptr2[1] * ffilter2_2[1];
            sum1_2 += inptr2[2] * ffilter2_2[2];
            sum1_2 += inptr3[0] * ffilter3_2[0];
            sum1_2 += inptr3[1] * ffilter3_2[1];
            sum1_2 += inptr3[2] * ffilter3_2[2];

            sum1_3 += inptr1[0] * ffilter1_3[0];
            sum1_3 += inptr1[1] * ffilter1_3[1];
            sum1_3 += inptr1[2] * ffilter1_3[2];
            sum1_3 += inptr2[0] * ffilter2_3[0];
            sum1_3 += inptr2[1] * ffilter2_3[1];
            sum1_3 += inptr2[2] * ffilter2_3[2];
            sum1_3 += inptr3[0] * ffilter3_3[0];
            sum1_3 += inptr3[1] * ffilter3_3[1];
            sum1_3 += inptr3[2] * ffilter3_3[2];

            sum1_4 += inptr1[0] * ffilter1_4[0];
            sum1_4 += inptr1[1] * ffilter1_4[1];
            sum1_4 += inptr1[2] * ffilter1_4[2];
            sum1_4 += inptr2[0] * ffilter2_4[0];
            sum1_4 += inptr2[1] * ffilter2_4[1];
            sum1_4 += inptr2[2] * ffilter2_4[2];
            sum1_4 += inptr3[0] * ffilter3_4[0];
            sum1_4 += inptr3[1] * ffilter3_4[1];
            sum1_4 += inptr3[2] * ffilter3_4[2];

            sum1_5 += inptr1[0] * ffilter1_5[0];
            sum1_5 += inptr1[1] * ffilter1_5[1];
            sum1_5 += inptr1[2] * ffilter1_5[2];
            sum1_5 += inptr2[0] * ffilter2_5[0];
            sum1_5 += inptr2[1] * ffilter2_5[1];
            sum1_5 += inptr2[2] * ffilter2_5[2];
            sum1_5 += inptr3[0] * ffilter3_5[0];
            sum1_5 += inptr3[1] * ffilter3_5[1];
            sum1_5 += inptr3[2] * ffilter3_5[2];

            sum1_6 += inptr1[0] * ffilter1_6[0];
            sum1_6 += inptr1[1] * ffilter1_6[1];
            sum1_6 += inptr1[2] * ffilter1_6[2];
            sum1_6 += inptr2[0] * ffilter2_6[0];
            sum1_6 += inptr2[1] * ffilter2_6[1];
            sum1_6 += inptr2[2] * ffilter2_6[2];
            sum1_6 += inptr3[0] * ffilter3_6[0];
            sum1_6 += inptr3[1] * ffilter3_6[1];
            sum1_6 += inptr3[2] * ffilter3_6[2];

            sum1_7 += inptr1[0] * ffilter1_7[0];
            sum1_7 += inptr1[1] * ffilter1_7[1];
            sum1_7 += inptr1[2] * ffilter1_7[2];
            sum1_7 += inptr2[0] * ffilter2_7[0];
            sum1_7 += inptr2[1] * ffilter2_7[1];
            sum1_7 += inptr2[2] * ffilter2_7[2];
            sum1_7 += inptr3[0] * ffilter3_7[0];
            sum1_7 += inptr3[1] * ffilter3_7[1];
            sum1_7 += inptr3[2] * ffilter3_7[2];

            sum1_8 += inptr1[0] * ffilter1_8[0];
            sum1_8 += inptr1[1] * ffilter1_8[1];
            sum1_8 += inptr1[2] * ffilter1_8[2];
            sum1_8 += inptr2[0] * ffilter2_8[0];
            sum1_8 += inptr2[1] * ffilter2_8[1];
            sum1_8 += inptr2[2] * ffilter2_8[2];
            sum1_8 += inptr3[0] * ffilter3_8[0];
            sum1_8 += inptr3[1] * ffilter3_8[1];
            sum1_8 += inptr3[2] * ffilter3_8[2];
#endif
            if (isnopadding) {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;
            } else if (input_width > 3 &&
                       (isoddpadding_w && ow == position_w1 ||
                        ow == position_w2 && isoddpadding_w && useallInput_w ||
                        ow == position_w2 + 1 && !isoddpadding_w &&
                            !useallInput_w)) {
              ffilter1--;
              ffilter2--;
              ffilter3--;
              ffilter1_2--;
              ffilter2_2--;
              ffilter3_2--;

              ffilter1_3--;
              ffilter2_3--;
              ffilter3_3--;
              ffilter1_4--;
              ffilter2_4--;
              ffilter3_4--;

              ffilter1_5--;
              ffilter2_5--;
              ffilter3_5--;
              ffilter1_6--;
              ffilter2_6--;
              ffilter3_6--;

              ffilter1_7--;
              ffilter2_7--;
              ffilter3_7--;
              ffilter1_8--;
              ffilter2_8--;
              ffilter3_8--;

              inptr1++;
              inptr2++;
              inptr3++;
            } else if (input_width <= 3 || ow < position_w1 ||
                       ow > position_w2) {
              ffilter1 -= 2;
              ffilter2 -= 2;
              ffilter3 -= 2;
              ffilter1_2 -= 2;
              ffilter2_2 -= 2;
              ffilter3_2 -= 2;

              ffilter1_3 -= 2;
              ffilter2_3 -= 2;
              ffilter3_3 -= 2;
              ffilter1_4 -= 2;
              ffilter2_4 -= 2;
              ffilter3_4 -= 2;

              ffilter1_5 -= 2;
              ffilter2_5 -= 2;
              ffilter3_5 -= 2;
              ffilter1_6 -= 2;
              ffilter2_6 -= 2;
              ffilter3_6 -= 2;

              ffilter1_7 -= 2;
              ffilter2_7 -= 2;
              ffilter3_7 -= 2;
              ffilter1_8 -= 2;
              ffilter2_8 -= 2;
              ffilter3_8 -= 2;
            } else {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;
            }
            *outptr1 += sum1;
            *outptr1_2 += sum1_2;
            *outptr1_3 += sum1_3;
            *outptr1_4 += sum1_4;
            *outptr1_5 += sum1_5;
            *outptr1_6 += sum1_6;
            *outptr1_7 += sum1_7;
            *outptr1_8 += sum1_8;

            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }
            if (outptr1_2[0] < reluvalue) {
              *outptr1_2 = reluvalue;
            }
            if (outptr1_3[0] < reluvalue) {
              *outptr1_3 = reluvalue;
            }
            if (outptr1_4[0] < reluvalue) {
              *outptr1_4 = reluvalue;
            }
            if (outptr1_5[0] < reluvalue) {
              *outptr1_5 = reluvalue;
            }
            if (outptr1_6[0] < reluvalue) {
              *outptr1_6 = reluvalue;
            }
            if (outptr1_7[0] < reluvalue) {
              *outptr1_7 = reluvalue;
            }
            if (outptr1_8[0] < reluvalue) {
              *outptr1_8 = reluvalue;
            }
            outptr1++;
            outptr1_2++;
            outptr1_3++;
            outptr1_4++;
            outptr1_5++;
            outptr1_6++;
            outptr1_7++;
            outptr1_8++;
          }
          if (isnopadding) {
            inptr1 += left_stride + input_width;
            inptr2 += left_stride + input_width;
            inptr3 += left_stride + input_width;

          } else if (input_height > 3 &&
                     (isoddpadding_h && oh == position_h1 ||
                      oh == position_h2 && isoddpadding_h && useallInput_h ||
                      oh == position_h2 + 1 && !isoddpadding_h &&
                          !useallInput_h)) {
            inptr1 += 3;
            inptr2 += 3;
            inptr3 += 3;

            ffilter1 -= left_stride;
            ffilter2 -= left_stride;
            ffilter3 -= left_stride;
            ffilter1_2 -= left_stride;
            ffilter2_2 -= left_stride;
            ffilter3_2 -= left_stride;

            ffilter1_3 -= left_stride;
            ffilter2_3 -= left_stride;
            ffilter3_3 -= left_stride;
            ffilter1_4 -= left_stride;
            ffilter2_4 -= left_stride;
            ffilter3_4 -= left_stride;

            ffilter1_5 -= left_stride;
            ffilter2_5 -= left_stride;
            ffilter3_5 -= left_stride;
            ffilter1_6 -= left_stride;
            ffilter2_6 -= left_stride;
            ffilter3_6 -= left_stride;

            ffilter1_7 -= left_stride;
            ffilter2_7 -= left_stride;
            ffilter3_7 -= left_stride;
            ffilter1_8 -= left_stride;
            ffilter2_8 -= left_stride;
            ffilter3_8 -= left_stride;
          } else if (input_height <= 3 || oh < position_h1 ||
                     oh > position_h2) {
            inptr1 -= input_width - 3;
            inptr2 -= input_width - 3;
            inptr3 -= input_width - 3;

            ffilter1 -= 3 + 2 * padding_width + left_stride;
            ffilter2 -= 3 + 2 * padding_width + left_stride;
            ffilter3 -= 3 + 2 * padding_width + left_stride;
            ffilter1_2 -= 3 + 2 * padding_width + left_stride;
            ffilter2_2 -= 3 + 2 * padding_width + left_stride;
            ffilter3_2 -= 3 + 2 * padding_width + left_stride;

            ffilter1_3 -= 3 + 2 * padding_width + left_stride;
            ffilter2_3 -= 3 + 2 * padding_width + left_stride;
            ffilter3_3 -= 3 + 2 * padding_width + left_stride;
            ffilter1_4 -= 3 + 2 * padding_width + left_stride;
            ffilter2_4 -= 3 + 2 * padding_width + left_stride;
            ffilter3_4 -= 3 + 2 * padding_width + left_stride;

            ffilter1_5 -= 3 + 2 * padding_width + left_stride;
            ffilter2_5 -= 3 + 2 * padding_width + left_stride;
            ffilter3_5 -= 3 + 2 * padding_width + left_stride;
            ffilter1_6 -= 3 + 2 * padding_width + left_stride;
            ffilter2_6 -= 3 + 2 * padding_width + left_stride;
            ffilter3_6 -= 3 + 2 * padding_width + left_stride;

            ffilter1_7 -= 3 + 2 * padding_width + left_stride;
            ffilter2_7 -= 3 + 2 * padding_width + left_stride;
            ffilter3_7 -= 3 + 2 * padding_width + left_stride;
            ffilter1_8 -= 3 + 2 * padding_width + left_stride;
            ffilter2_8 -= 3 + 2 * padding_width + left_stride;
            ffilter3_8 -= 3 + 2 * padding_width + left_stride;
          } else {
            ffilter1 += 3 + 2 * padding_width - left_stride;
            ffilter2 += 3 + 2 * padding_width - left_stride;
            ffilter3 += 3 + 2 * padding_width - left_stride;
            ffilter1_2 += 3 + 2 * padding_width - left_stride;
            ffilter2_2 += 3 + 2 * padding_width - left_stride;
            ffilter3_2 += 3 + 2 * padding_width - left_stride;

            ffilter1_3 += 3 + 2 * padding_width - left_stride;
            ffilter2_3 += 3 + 2 * padding_width - left_stride;
            ffilter3_3 += 3 + 2 * padding_width - left_stride;
            ffilter1_4 += 3 + 2 * padding_width - left_stride;
            ffilter2_4 += 3 + 2 * padding_width - left_stride;
            ffilter3_4 += 3 + 2 * padding_width - left_stride;

            ffilter1_5 += 3 + 2 * padding_width - left_stride;
            ffilter2_5 += 3 + 2 * padding_width - left_stride;
            ffilter3_5 += 3 + 2 * padding_width - left_stride;
            ffilter1_6 += 3 + 2 * padding_width - left_stride;
            ffilter2_6 += 3 + 2 * padding_width - left_stride;
            ffilter3_6 += 3 + 2 * padding_width - left_stride;

            ffilter1_7 += 3 + 2 * padding_width - left_stride;
            ffilter2_7 += 3 + 2 * padding_width - left_stride;
            ffilter3_7 += 3 + 2 * padding_width - left_stride;
            ffilter1_8 += 3 + 2 * padding_width - left_stride;
            ffilter2_8 += 3 + 2 * padding_width - left_stride;
            ffilter3_8 += 3 + 2 * padding_width - left_stride;

            inptr1 += input_width + 3;
            inptr2 += input_width + 3;
            inptr3 += input_width + 3;
          }
        }

        filter_data_ch += filter_channel_stride;
        filter_data_ch_2 += filter_channel_stride;
        filter_data_ch_3 += filter_channel_stride;
        filter_data_ch_4 += filter_channel_stride;
        filter_data_ch_5 += filter_channel_stride;
        filter_data_ch_6 += filter_channel_stride;
        filter_data_ch_7 += filter_channel_stride;
        filter_data_ch_8 += filter_channel_stride;
        input_data_ch += input_channel_stride;
      }
    }

    int output_channels_left = output_channels_d8 * 8;

#pragma omp parallel for
    for (int oc = output_channels_left; oc < output_channels; ++oc) {
      std::atomic<float> reluvalue{0};

      const float *reluptr;
      const float *f1, *f9;
      const float *inptr1, *inptr2, *inptr3;
      const float *ffilter1, *ffilter2, *ffilter3;
      float ffilterarray[ffilter_length] = {0};
      float *output_data_ch;
      const float *input_data_ch;
      const float *filter_data_ch;

      filter_data_ch =
          filter_data + oc * filter_channel_stride * input_channels;
      input_data_ch = input_data;
      output_data_ch = output_data + oc * output_channel_stride;

      for (int ic = 0; ic < input_channels; ++ic) {
        float reluarr[4];
        reluptr = reluarr;
        if (if_relu && ic == input_channels - 1) {
          reluvalue = 0;
        } else {
          reluvalue = -FLT_MAX;
        }
        reluarr[0] = reluvalue;
        reluarr[1] = reluvalue;
        reluarr[2] = reluvalue;
        reluarr[3] = reluvalue;

        f1 = filter_data_ch;
        f9 = f1 + 8;

        if (!isnopadding) {
          ffilterarray[ffilter_length] = {0};
          for (int i = 0; i < 9; ++i) {
            int j = i / 3 * (2 * padding_width + 3) + i % 3 +
                    padding_height * 3 +
                    padding_width * (2 * padding_height + 1);
            ffilterarray[j] = filter_data_ch[i];
          }
          ffilter1 = ffilterarray;
          ffilter1 += ffilter_start;
          ffilter2 = ffilter1 + ffilter_width;
          ffilter3 = ffilter2 + ffilter_width;
        } else {
          ffilter1 = filter_data_ch;
          ffilter2 = ffilter1 + 3;
          ffilter3 = ffilter2 + 3;
        }

        float *outptr1;
        outptr1 = output_data_ch;
        inptr1 = input_data_ch;
        inptr2 = inptr1 + input_width;
        inptr3 = inptr2 + input_width;

        int oh = 0;
        for (; oh < output_height; ++oh) {
          int ow = 0;

          for (; ow <= position_w1; ++ow) {
            float sum1 = 0;
#if __ARM_NEON
            float32x4_t _inptr1 = vld1q_f32(inptr1);
            float32x4_t _ffilter1 = vld1q_f32(ffilter1);
            float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);

            float32x4_t _inptr2 = vld1q_f32(inptr2);
            float32x4_t _ffilter2 = vld1q_f32(ffilter2);
            _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);

            float32x4_t _inptr3 = vld1q_f32(inptr3);
            float32x4_t _ffilter3 = vld1q_f32(ffilter3);
            _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);

            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ssss1_ssss1 = vpadd_f32(_ss1, _ss1);
            sum1 += vget_lane_f32(_ssss1_ssss1, 0);
#else
            sum1 += inptr1[0] * ffilter1[0];
            sum1 += inptr1[1] * ffilter1[1];
            sum1 += inptr1[2] * ffilter1[2];
            sum1 += inptr2[0] * ffilter2[0];
            sum1 += inptr2[1] * ffilter2[1];
            sum1 += inptr2[2] * ffilter2[2];
            sum1 += inptr3[0] * ffilter3[0];
            sum1 += inptr3[1] * ffilter3[1];
            sum1 += inptr3[2] * ffilter3[2];
#endif
            if (isnopadding) {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;
            } else if (input_width > 3 &&
                       (isoddpadding_w && ow == position_w1 ||
                        ow == position_w2 && isoddpadding_w && useallInput_w ||
                        ow == position_w2 + 1 && !isoddpadding_w &&
                            !useallInput_w)) {
              ffilter1--;
              ffilter2--;
              ffilter3--;
              inptr1++;
              inptr2++;
              inptr3++;

            } else if (input_width <= 3 || ow < position_w1 ||
                       ow > position_w2) {
              ffilter1 -= 2;
              ffilter2 -= 2;
              ffilter3 -= 2;
            } else {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;
            }
            *outptr1 += sum1;

            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }
            outptr1++;
          }
#if __ARM_NEON
#if __aarch64__
          if (oh > position_h1 && oh < position_h2) {
            int ow_dim4 = (position_w2 - position_w1 - 1) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
              asm volatile(
                  "prfm   pldl1keep, [%[f1], #256]            \n\t"
                  "prfm   pldl1keep, [%[f9], #256]            \n\t"
                  "prfm   pldl1keep, [%[reluptr], #256]       \n\t"

                  "ld1   {v0.4s, v1.4s}, [%[f1]]              \n\t"
                  "ld1   {v4.s}[0], [%[f9]]                   \n\t"
                  "ld1   {v16.4s}, [%[reluptr]]               \n\t"

                  "0:                                         \n\t"
                  // load outptr
                  "prfm   pldl1keep, [%[outptr1], #128]       \n\t"
                  "ld1   {v12.4s}, [%[outptr1]]               \n\t"

                  // inptr1 multiply
                  "prfm   pldl1keep, [%[inptr1], #256]        \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[inptr1]], #32     \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[inptr1]]          \n\t"

                  "fmla   v12.4s, v5.4s, v0.4s[0]             \n\t"
                  "fmla   v14.4s, v5.4s, v2.4s[0]             \n\t"

                  "ext    v8.16b, v5.16b, v7.16b, #4          \n\t"
                  "fmul   v13.4s, v6.4s, v0.4s[1]             \n\t"
                  "fmla   v12.4s, v8.4s, v0.4s[2]             \n\t"

                  "ld2   {v5.4s, v6.4s}, [%[inptr2]], #32     \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[inptr2]]          \n\t"

                  // inptr2 multiply
                  "fmla   v13.4s, v5.4s, v0.4s[3]             \n\t"
                  "ext    v8.16b, v5.16b, v7.16b, #4          \n\t"
                  "fmla   v12.4s, v6.4s, v1.4s[0]             \n\t"

                  "fmla   v13.4s, v8.4s, v1.4s[1]             \n\t"
                  "ld2   {v5.4s, v6.4s}, [%[inptr3]], #32     \n\t"
                  "ld2   {v7.4s, v8.4s}, [%[inptr3]]          \n\t"

                  // inptr3 multiply
                  "fmla   v12.4s, v5.4s, v1.4s[2]             \n\t"
                  "ext    v8.16b, v5.16b, v7.16b, #4          \n\t"

                  "fmla   v13.4s, v6.4s, v1.4s[3]             \n\t"
                  "fmla   v12.4s, v8.4s, v4.4s[0]             \n\t"

                  // store outptr
                  "fadd   v12.4s, v12.4s, v13.4s              \n\t"
                  "fmax   v12.4s, v12.4s, v16.4s              \n\t"
                  "st1   {v12.4s}, [%[outptr1]], #16          \n\t"

                  // cycle
                  "subs       %[ow_dim4], %[ow_dim4], #1      \n\t"
                  "bne        0b                              \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
                    [inptr3] "+r"(inptr3)
                  : [f1] "r"(f1), [f9] "r"(f9), [reluptr] "r"(reluptr)
                  : "memory", "v0", "v1", "v4", "v5", "v6", "v7", "v8", "v9",
                    "v12", "v13", "v16");
            }
          }
#else
          if (oh > position_h1 && oh < position_h2) {
            int ow_dim4 = (position_w2 - position_w1 - 1) >> 2;
            ow += ow_dim4 * 4;

            if (ow_dim4 > 0) {
              asm volatile(
                  "pld        [%[f1], #256]                   \n\t"
                  "pld        [%[f9], #256]                   \n\t"
                  "pld        [%[reluptr], #128]              \n\t"

                  "vld1.32   {d0-d3}, [%[f1]]                 \n\t"
                  "vld1.32   {d8[0]}, [%[f9]]                 \n\t"
                  "vld1.32   {d18, d19}, [%[reluptr]]         \n\t"

                  "pld        [%[inptr1], #256]               \n\t"
                  "vld2.f32   {d10-d13}, [%[inptr1]]!         \n\t"
                  "vld2.f32   {d14, d15}, [%[inptr1]]         \n\t"

                  "0:                                         \n\t"
                  // load outptr
                  "pld        [%[outptr1], #128]              \n\t"
                  "vld1.f32   {d24, d25}, [%[outptr1]]        \n\t"

                  // inptr1 multiply
                  "pld        [%[inptr2], #256]               \n\t"

                  "vld2.f32   {d4-d7}, [%[inptr2]]!           \n\t"

                  "vmla.f32   q12, q5, d0[0]                  \n\t"
                  "vld2.f32   {d20, d21}, [%[inptr2]]         \n\t"
                  "vext.32    q8, q5, q7, #1                  \n\t"

                  "pld        [%[inptr3], #256]               \n\t"
                  "vmul.f32   q13, q6, d0[1]                  \n\t"

                  "vld2.f32   {d10-d13}, [%[inptr3]]!         \n\t"

                  "vmul.f32   q14, q8, d1[0]                  \n\t"
                  "vld2.f32   {d14, d15}, [%[inptr3]]         \n\t"

                  // inptr2 multiply
                  "vmul.f32   q15, q2, d1[1]                  \n\t"
                  "vext.32    q8, q2, q10, #1                 \n\t"

                  "vmla.f32   q12, q3, d2[0]                  \n\t"
                  "vmla.f32   q13, q8, d2[1]                  \n\t"

                  // inptr3 multiply
                  "vmla.f32   q14, q5, d3[0]                  \n\t"
                  "vext.32    q8, q5, q7, #1                  \n\t"

                  "pld        [%[inptr1], #256]               \n\t"
                  "vmla.f32   q15, q6, d3[1]                  \n\t"

                  "vld2.f32   {d10-d13}, [%[inptr1]]!         \n\t"
                  "vmla.f32   q13, q8, d8[0]                  \n\t"

                  // store outptr
                  "vld2.f32   {d14, d15}, [%[inptr1]]         \n\t"
                  "vadd.f32   q12, q12, q13                   \n\t"
                  "subs       %[ow_dim4], #1                  \n\t"

                  "vadd.f32   q14, q14, q15                   \n\t"
                  "vadd.f32   q12, q12, q14                   \n\t"
                  "vmax.f32   q12, q12, q9                    \n\t"
                  "vst1.f32   {d24, d25}, [%[outptr1]]!       \n\t"

                  // cycle
                  "bne        0b                              \n\t"
                  "subs       %[inptr1], %[inptr1], #32       \n\t"

                  : [ow_dim4] "+r"(ow_dim4), [outptr1] "+r"(outptr1),
                    [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
                    [inptr3] "+r"(inptr3)
                  : [f1] "r"(f1), [f9] "r"(f9), [reluptr] "r"(reluptr)
                  : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                    "q8", "q9", "q10", "q12", "q13", "q14", "q15");
            }
          }
#endif  //__aarch64__
#endif  // __ARM_NEON
          outptr1 -= 4;
          outptr1 += 4;

          for (; ow < output_width; ++ow) {
            float sum1 = 0;
#if __ARM_NEON
            float32x4_t _inptr1 = vld1q_f32(inptr1);
            float32x4_t _ffilter1 = vld1q_f32(ffilter1);
            float32x4_t _sum1 = vmulq_f32(_inptr1, _ffilter1);

            float32x4_t _inptr2 = vld1q_f32(inptr2);
            float32x4_t _ffilter2 = vld1q_f32(ffilter2);
            _sum1 = vmlaq_f32(_sum1, _inptr2, _ffilter2);

            float32x4_t _inptr3 = vld1q_f32(inptr3);
            float32x4_t _ffilter3 = vld1q_f32(ffilter3);
            _sum1 = vmlaq_f32(_sum1, _inptr3, _ffilter3);

            _sum1 = vsetq_lane_f32(sum1, _sum1, 3);
            float32x2_t _ss1 =
                vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _ssss1_ssss1 = vpadd_f32(_ss1, _ss1);
            sum1 += vget_lane_f32(_ssss1_ssss1, 0);
#else
            sum1 += inptr1[0] * ffilter1[0];
            sum1 += inptr1[1] * ffilter1[1];
            sum1 += inptr1[2] * ffilter1[2];
            sum1 += inptr2[0] * ffilter2[0];
            sum1 += inptr2[1] * ffilter2[1];
            sum1 += inptr2[2] * ffilter2[2];
            sum1 += inptr3[0] * ffilter3[0];
            sum1 += inptr3[1] * ffilter3[1];
            sum1 += inptr3[2] * ffilter3[2];
#endif
            if (isnopadding) {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;
            } else if (input_width > 3 &&
                       (isoddpadding_w && ow == position_w1 ||
                        ow == position_w2 && isoddpadding_w && useallInput_w ||
                        ow == position_w2 + 1 && !isoddpadding_w &&
                            !useallInput_w)) {
              ffilter1--;
              ffilter2--;
              ffilter3--;

              inptr1++;
              inptr2++;
              inptr3++;

            } else if (input_width <= 3 || ow < position_w1 ||
                       ow > position_w2) {
              ffilter1 -= 2;
              ffilter2 -= 2;
              ffilter3 -= 2;
            } else {
              inptr1 += 2;
              inptr2 += 2;
              inptr3 += 2;
            }
            *outptr1 += sum1;

            if (outptr1[0] < reluvalue) {
              *outptr1 = reluvalue;
            }
            outptr1++;
          }
          if (isnopadding) {
            inptr1 += left_stride + input_width;
            inptr2 += left_stride + input_width;
            inptr3 += left_stride + input_width;
          } else if (input_height > 3 &&
                     (isoddpadding_h && oh == position_h1 ||
                      oh == position_h2 && isoddpadding_h && useallInput_h ||
                      oh == position_h2 + 1 && !isoddpadding_h &&
                          !useallInput_h)) {
            inptr1 += 3;
            inptr2 += 3;
            inptr3 += 3;

            ffilter1 -= left_stride;
            ffilter2 -= left_stride;
            ffilter3 -= left_stride;

          } else if (input_height <= 3 || oh < position_h1 ||
                     oh > position_h2) {
            inptr1 -= input_width - 3;
            inptr2 -= input_width - 3;
            inptr3 -= input_width - 3;

            ffilter1 -= 3 + 2 * padding_width + left_stride;
            ffilter2 -= 3 + 2 * padding_width + left_stride;
            ffilter3 -= 3 + 2 * padding_width + left_stride;
          } else {
            ffilter1 += 3 + 2 * padding_width - left_stride;
            ffilter2 += 3 + 2 * padding_width - left_stride;
            ffilter3 += 3 + 2 * padding_width - left_stride;

            inptr1 += input_width + 3;
            inptr2 += input_width + 3;
            inptr3 += input_width + 3;
          }
        }
        filter_data_ch += filter_channel_stride;
        input_data_ch += input_channel_stride;
      }
    }
    input_data += input_batch_stride;
    output_data += output_batch_stride;
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
