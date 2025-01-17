/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/backends/loongarch/math/lasx/conv_depthwise_pack8.h"
#include <vector>
#include "lite/backends/loongarch/math/common/conv_utils.h"

namespace paddle {
namespace lite {
namespace loongarch {
namespace math {

// input  [bs, ic/8, ih, iw, 8]
// filter [1,  oc/8, kh, kw, 8]
// bias   [    oc             ]
// output [bs, oc/8, oh, ow, 8]
void conv_depthwise_3x3s1_m256(lite::Tensor* input,
                               lite::Tensor* output,
                               lite::Tensor* filter,
                               lite::Tensor* bias,
                               const bool has_act,
                               const lite_api::ActivationType act_type,
                               const operators::ActivationParam act_param) {
  // input [bs, ic/8, ih, iw, 8]
  CHECK_EQ(input->dims().size(), 5UL);
  const int batch_size = input->dims()[0];
  const int channel_num = input->dims()[1];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const float* input_data = input->data<float>();

  // filter [1, oc/8, kh, kw, 8]
  CHECK_EQ(filter->dims().size(), 5UL);
  const int kernel_h = filter->dims()[2];
  const int kernel_w = filter->dims()[3];
  const float* filter_data = filter->data<float>();

  // output [bs, oc/8, oh, ow, 8]
  CHECK_EQ(output->dims().size(), 5UL);
  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  float* output_data = output->mutable_data<float>();

  const int input_group_step = input_width * 8;
  const int input_channel_step = input_height * input_width * 8;
  const int input_batch_step = channel_num * input_height * input_width * 8;

  const int filter_channel_step = kernel_h * kernel_w * 8;

  int total_count = batch_size * channel_num;

  for (int idx = 0; idx < total_count; ++idx) {
    __m256 _bias0 =
        bias ? lasx_loadu_f32(bias->data<float>() + (idx % channel_num) * 8)
             : lasx_set1_f32(0.f);

    const float* k0 = filter_data + (idx % channel_num) * filter_channel_step;

    const float* r0 = input_data + (idx / channel_num) * input_batch_step +
                      (idx % channel_num) * input_channel_step;
    const float* r1 = r0 + input_group_step;
    const float* r2 = r1 + input_group_step;

    __m256 _k00 = lasx_loadu_f32(k0);
    __m256 _k01 = lasx_loadu_f32(k0 + 8);
    __m256 _k02 = lasx_loadu_f32(k0 + 16);
    __m256 _k10 = lasx_loadu_f32(k0 + 24);
    __m256 _k11 = lasx_loadu_f32(k0 + 32);
    __m256 _k12 = lasx_loadu_f32(k0 + 40);
    __m256 _k20 = lasx_loadu_f32(k0 + 48);
    __m256 _k21 = lasx_loadu_f32(k0 + 56);
    __m256 _k22 = lasx_loadu_f32(k0 + 64);

    for (int i = 0; i < output_height; ++i) {
      int j = 0;
      for (; j + 7 < output_width; j += 8) {
        __m256 _sum0 = _bias0;

        __m256 _r00 = lasx_loadu_f32(r0);
        __m256 _r01 = lasx_loadu_f32(r0 + 8);
        __m256 _r02 = lasx_loadu_f32(r0 + 16);
        __m256 _r10 = lasx_loadu_f32(r1);
        __m256 _r11 = lasx_loadu_f32(r1 + 8);
        __m256 _r12 = lasx_loadu_f32(r1 + 16);
        __m256 _r20 = lasx_loadu_f32(r2);
        __m256 _r21 = lasx_loadu_f32(r2 + 8);
        __m256 _r22 = lasx_loadu_f32(r2 + 16);

        _sum0 = lasx_fmadd_f32(_k00, _r00, _sum0);
        _sum0 = lasx_fmadd_f32(_k01, _r01, _sum0);
        _sum0 = lasx_fmadd_f32(_k02, _r02, _sum0);
        _sum0 = lasx_fmadd_f32(_k10, _r10, _sum0);
        _sum0 = lasx_fmadd_f32(_k11, _r11, _sum0);
        _sum0 = lasx_fmadd_f32(_k12, _r12, _sum0);
        _sum0 = lasx_fmadd_f32(_k20, _r20, _sum0);
        _sum0 = lasx_fmadd_f32(_k21, _r21, _sum0);
        _sum0 = lasx_fmadd_f32(_k22, _r22, _sum0);

        if (has_act) {
          _sum0 = activation8_m256(_sum0, act_type, act_param);
        }

        lasx_storeu_f32(output_data, _sum0);

        __m256 _sum1 = _bias0;
        __m256 _r03 = lasx_loadu_f32(r0 + 24);
        __m256 _r13 = lasx_loadu_f32(r1 + 24);
        __m256 _r23 = lasx_loadu_f32(r2 + 24);

        _sum1 = lasx_fmadd_f32(_k00, _r01, _sum1);
        _sum1 = lasx_fmadd_f32(_k01, _r02, _sum1);
        _sum1 = lasx_fmadd_f32(_k02, _r03, _sum1);
        _sum1 = lasx_fmadd_f32(_k10, _r11, _sum1);
        _sum1 = lasx_fmadd_f32(_k11, _r12, _sum1);
        _sum1 = lasx_fmadd_f32(_k12, _r13, _sum1);
        _sum1 = lasx_fmadd_f32(_k20, _r21, _sum1);
        _sum1 = lasx_fmadd_f32(_k21, _r22, _sum1);
        _sum1 = lasx_fmadd_f32(_k22, _r23, _sum1);

        if (has_act) {
          _sum1 = activation8_m256(_sum1, act_type, act_param);
        }
        lasx_storeu_f32(output_data + 8, _sum1);

        __m256 _sum2 = _bias0;
        __m256 _r04 = lasx_loadu_f32(r0 + 32);
        __m256 _r14 = lasx_loadu_f32(r1 + 32);
        __m256 _r24 = lasx_loadu_f32(r2 + 32);

        _sum2 = lasx_fmadd_f32(_k00, _r02, _sum2);
        _sum2 = lasx_fmadd_f32(_k01, _r03, _sum2);
        _sum2 = lasx_fmadd_f32(_k02, _r04, _sum2);
        _sum2 = lasx_fmadd_f32(_k10, _r12, _sum2);
        _sum2 = lasx_fmadd_f32(_k11, _r13, _sum2);
        _sum2 = lasx_fmadd_f32(_k12, _r14, _sum2);
        _sum2 = lasx_fmadd_f32(_k20, _r22, _sum2);
        _sum2 = lasx_fmadd_f32(_k21, _r23, _sum2);
        _sum2 = lasx_fmadd_f32(_k22, _r24, _sum2);

        if (has_act) {
          _sum2 = activation8_m256(_sum2, act_type, act_param);
        }
        lasx_storeu_f32(output_data + 16, _sum2);

        __m256 _sum3 = _bias0;
        __m256 _r05 = lasx_loadu_f32(r0 + 40);
        __m256 _r15 = lasx_loadu_f32(r1 + 40);
        __m256 _r25 = lasx_loadu_f32(r2 + 40);

        _sum3 = lasx_fmadd_f32(_k00, _r03, _sum3);
        _sum3 = lasx_fmadd_f32(_k01, _r04, _sum3);
        _sum3 = lasx_fmadd_f32(_k02, _r05, _sum3);
        _sum3 = lasx_fmadd_f32(_k10, _r13, _sum3);
        _sum3 = lasx_fmadd_f32(_k11, _r14, _sum3);
        _sum3 = lasx_fmadd_f32(_k12, _r15, _sum3);
        _sum3 = lasx_fmadd_f32(_k20, _r23, _sum3);
        _sum3 = lasx_fmadd_f32(_k21, _r24, _sum3);
        _sum3 = lasx_fmadd_f32(_k22, _r25, _sum3);

        if (has_act) {
          _sum3 = activation8_m256(_sum3, act_type, act_param);
        }
        lasx_storeu_f32(output_data + 24, _sum3);

        __m256 _sum4 = _bias0;
        __m256 _r06 = lasx_loadu_f32(r0 + 48);
        __m256 _r16 = lasx_loadu_f32(r1 + 48);
        __m256 _r26 = lasx_loadu_f32(r2 + 48);

        _sum4 = lasx_fmadd_f32(_k00, _r04, _sum4);
        _sum4 = lasx_fmadd_f32(_k01, _r05, _sum4);
        _sum4 = lasx_fmadd_f32(_k02, _r06, _sum4);
        _sum4 = lasx_fmadd_f32(_k10, _r14, _sum4);
        _sum4 = lasx_fmadd_f32(_k11, _r15, _sum4);
        _sum4 = lasx_fmadd_f32(_k12, _r16, _sum4);
        _sum4 = lasx_fmadd_f32(_k20, _r24, _sum4);
        _sum4 = lasx_fmadd_f32(_k21, _r25, _sum4);
        _sum4 = lasx_fmadd_f32(_k22, _r26, _sum4);

        if (has_act) {
          _sum4 = activation8_m256(_sum4, act_type, act_param);
        }
        lasx_storeu_f32(output_data + 32, _sum4);

        __m256 _sum5 = _bias0;
        __m256 _r07 = lasx_loadu_f32(r0 + 56);
        __m256 _r17 = lasx_loadu_f32(r1 + 56);
        __m256 _r27 = lasx_loadu_f32(r2 + 56);

        _sum5 = lasx_fmadd_f32(_k00, _r05, _sum5);
        _sum5 = lasx_fmadd_f32(_k01, _r06, _sum5);
        _sum5 = lasx_fmadd_f32(_k02, _r07, _sum5);
        _sum5 = lasx_fmadd_f32(_k10, _r15, _sum5);
        _sum5 = lasx_fmadd_f32(_k11, _r16, _sum5);
        _sum5 = lasx_fmadd_f32(_k12, _r17, _sum5);
        _sum5 = lasx_fmadd_f32(_k20, _r25, _sum5);
        _sum5 = lasx_fmadd_f32(_k21, _r26, _sum5);
        _sum5 = lasx_fmadd_f32(_k22, _r27, _sum5);

        if (has_act) {
          _sum5 = activation8_m256(_sum5, act_type, act_param);
        }
        lasx_storeu_f32(output_data + 40, _sum5);

        __m256 _sum6 = _bias0;
        __m256 _r08 = lasx_loadu_f32(r0 + 64);
        __m256 _r18 = lasx_loadu_f32(r1 + 64);
        __m256 _r28 = lasx_loadu_f32(r2 + 64);

        _sum6 = lasx_fmadd_f32(_k00, _r06, _sum6);
        _sum6 = lasx_fmadd_f32(_k01, _r07, _sum6);
        _sum6 = lasx_fmadd_f32(_k02, _r08, _sum6);
        _sum6 = lasx_fmadd_f32(_k10, _r16, _sum6);
        _sum6 = lasx_fmadd_f32(_k11, _r17, _sum6);
        _sum6 = lasx_fmadd_f32(_k12, _r18, _sum6);
        _sum6 = lasx_fmadd_f32(_k20, _r26, _sum6);
        _sum6 = lasx_fmadd_f32(_k21, _r27, _sum6);
        _sum6 = lasx_fmadd_f32(_k22, _r28, _sum6);

        if (has_act) {
          _sum6 = activation8_m256(_sum6, act_type, act_param);
        }
        lasx_storeu_f32(output_data + 48, _sum6);

        __m256 _sum7 = _bias0;
        __m256 _r09 = lasx_loadu_f32(r0 + 72);
        __m256 _r19 = lasx_loadu_f32(r1 + 72);
        __m256 _r29 = lasx_loadu_f32(r2 + 72);

        _sum7 = lasx_fmadd_f32(_k00, _r07, _sum7);
        _sum7 = lasx_fmadd_f32(_k01, _r08, _sum7);
        _sum7 = lasx_fmadd_f32(_k02, _r09, _sum7);
        _sum7 = lasx_fmadd_f32(_k10, _r17, _sum7);
        _sum7 = lasx_fmadd_f32(_k11, _r18, _sum7);
        _sum7 = lasx_fmadd_f32(_k12, _r19, _sum7);
        _sum7 = lasx_fmadd_f32(_k20, _r27, _sum7);
        _sum7 = lasx_fmadd_f32(_k21, _r28, _sum7);
        _sum7 = lasx_fmadd_f32(_k22, _r29, _sum7);

        if (has_act) {
          _sum7 = activation8_m256(_sum7, act_type, act_param);
        }
        lasx_storeu_f32(output_data + 56, _sum7);

        r0 += 64;
        r1 += 64;
        r2 += 64;
        output_data += 64;
      }
      for (; j + 3 < output_width; j += 4) {
        __m256 _sum0 = _bias0;

        __m256 _r00 = lasx_loadu_f32(r0);
        __m256 _r01 = lasx_loadu_f32(r0 + 8);
        __m256 _r02 = lasx_loadu_f32(r0 + 16);
        __m256 _r10 = lasx_loadu_f32(r1);
        __m256 _r11 = lasx_loadu_f32(r1 + 8);
        __m256 _r12 = lasx_loadu_f32(r1 + 16);
        __m256 _r20 = lasx_loadu_f32(r2);
        __m256 _r21 = lasx_loadu_f32(r2 + 8);
        __m256 _r22 = lasx_loadu_f32(r2 + 16);

        _sum0 = lasx_fmadd_f32(_k00, _r00, _sum0);
        _sum0 = lasx_fmadd_f32(_k01, _r01, _sum0);
        _sum0 = lasx_fmadd_f32(_k02, _r02, _sum0);
        _sum0 = lasx_fmadd_f32(_k10, _r10, _sum0);
        _sum0 = lasx_fmadd_f32(_k11, _r11, _sum0);
        _sum0 = lasx_fmadd_f32(_k12, _r12, _sum0);
        _sum0 = lasx_fmadd_f32(_k20, _r20, _sum0);
        _sum0 = lasx_fmadd_f32(_k21, _r21, _sum0);
        _sum0 = lasx_fmadd_f32(_k22, _r22, _sum0);

        if (has_act) {
          _sum0 = activation8_m256(_sum0, act_type, act_param);
        }
        lasx_storeu_f32(output_data, _sum0);

        __m256 _sum1 = _bias0;
        __m256 _r03 = lasx_loadu_f32(r0 + 24);
        __m256 _r13 = lasx_loadu_f32(r1 + 24);
        __m256 _r23 = lasx_loadu_f32(r2 + 24);

        _sum1 = lasx_fmadd_f32(_k00, _r01, _sum1);
        _sum1 = lasx_fmadd_f32(_k01, _r02, _sum1);
        _sum1 = lasx_fmadd_f32(_k02, _r03, _sum1);
        _sum1 = lasx_fmadd_f32(_k10, _r11, _sum1);
        _sum1 = lasx_fmadd_f32(_k11, _r12, _sum1);
        _sum1 = lasx_fmadd_f32(_k12, _r13, _sum1);
        _sum1 = lasx_fmadd_f32(_k20, _r21, _sum1);
        _sum1 = lasx_fmadd_f32(_k21, _r22, _sum1);
        _sum1 = lasx_fmadd_f32(_k22, _r23, _sum1);

        if (has_act) {
          _sum1 = activation8_m256(_sum1, act_type, act_param);
        }
        lasx_storeu_f32(output_data + 8, _sum1);

        __m256 _sum2 = _bias0;
        __m256 _r04 = lasx_loadu_f32(r0 + 32);
        __m256 _r14 = lasx_loadu_f32(r1 + 32);
        __m256 _r24 = lasx_loadu_f32(r2 + 32);

        _sum2 = lasx_fmadd_f32(_k00, _r02, _sum2);
        _sum2 = lasx_fmadd_f32(_k01, _r03, _sum2);
        _sum2 = lasx_fmadd_f32(_k02, _r04, _sum2);
        _sum2 = lasx_fmadd_f32(_k10, _r12, _sum2);
        _sum2 = lasx_fmadd_f32(_k11, _r13, _sum2);
        _sum2 = lasx_fmadd_f32(_k12, _r14, _sum2);
        _sum2 = lasx_fmadd_f32(_k20, _r22, _sum2);
        _sum2 = lasx_fmadd_f32(_k21, _r23, _sum2);
        _sum2 = lasx_fmadd_f32(_k22, _r24, _sum2);

        if (has_act) {
          _sum2 = activation8_m256(_sum2, act_type, act_param);
        }
        lasx_storeu_f32(output_data + 16, _sum2);

        __m256 _sum3 = _bias0;
        __m256 _r05 = lasx_loadu_f32(r0 + 40);
        __m256 _r15 = lasx_loadu_f32(r1 + 40);
        __m256 _r25 = lasx_loadu_f32(r2 + 40);

        _sum3 = lasx_fmadd_f32(_k00, _r03, _sum3);
        _sum3 = lasx_fmadd_f32(_k01, _r04, _sum3);
        _sum3 = lasx_fmadd_f32(_k02, _r05, _sum3);
        _sum3 = lasx_fmadd_f32(_k10, _r13, _sum3);
        _sum3 = lasx_fmadd_f32(_k11, _r14, _sum3);
        _sum3 = lasx_fmadd_f32(_k12, _r15, _sum3);
        _sum3 = lasx_fmadd_f32(_k20, _r23, _sum3);
        _sum3 = lasx_fmadd_f32(_k21, _r24, _sum3);
        _sum3 = lasx_fmadd_f32(_k22, _r25, _sum3);

        if (has_act) {
          _sum3 = activation8_m256(_sum3, act_type, act_param);
        }
        lasx_storeu_f32(output_data + 24, _sum3);

        r0 += 32;
        r1 += 32;
        r2 += 32;
        output_data += 32;
      }
      for (; j + 1 < output_width; j += 2) {
        __m256 _sum0 = _bias0;

        __m256 _r00 = lasx_loadu_f32(r0);
        __m256 _r01 = lasx_loadu_f32(r0 + 8);
        __m256 _r02 = lasx_loadu_f32(r0 + 16);
        __m256 _r10 = lasx_loadu_f32(r1);
        __m256 _r11 = lasx_loadu_f32(r1 + 8);
        __m256 _r12 = lasx_loadu_f32(r1 + 16);
        __m256 _r20 = lasx_loadu_f32(r2);
        __m256 _r21 = lasx_loadu_f32(r2 + 8);
        __m256 _r22 = lasx_loadu_f32(r2 + 16);

        _sum0 = lasx_fmadd_f32(_k00, _r00, _sum0);
        _sum0 = lasx_fmadd_f32(_k01, _r01, _sum0);
        _sum0 = lasx_fmadd_f32(_k02, _r02, _sum0);
        _sum0 = lasx_fmadd_f32(_k10, _r10, _sum0);
        _sum0 = lasx_fmadd_f32(_k11, _r11, _sum0);
        _sum0 = lasx_fmadd_f32(_k12, _r12, _sum0);
        _sum0 = lasx_fmadd_f32(_k20, _r20, _sum0);
        _sum0 = lasx_fmadd_f32(_k21, _r21, _sum0);
        _sum0 = lasx_fmadd_f32(_k22, _r22, _sum0);

        if (has_act) {
          _sum0 = activation8_m256(_sum0, act_type, act_param);
        }
        lasx_storeu_f32(output_data, _sum0);

        __m256 _sum1 = _bias0;
        __m256 _r03 = lasx_loadu_f32(r0 + 24);
        __m256 _r13 = lasx_loadu_f32(r1 + 24);
        __m256 _r23 = lasx_loadu_f32(r2 + 24);

        _sum1 = lasx_fmadd_f32(_k00, _r01, _sum1);
        _sum1 = lasx_fmadd_f32(_k01, _r02, _sum1);
        _sum1 = lasx_fmadd_f32(_k02, _r03, _sum1);
        _sum1 = lasx_fmadd_f32(_k10, _r11, _sum1);
        _sum1 = lasx_fmadd_f32(_k11, _r12, _sum1);
        _sum1 = lasx_fmadd_f32(_k12, _r13, _sum1);
        _sum1 = lasx_fmadd_f32(_k20, _r21, _sum1);
        _sum1 = lasx_fmadd_f32(_k21, _r22, _sum1);
        _sum1 = lasx_fmadd_f32(_k22, _r23, _sum1);

        if (has_act) {
          _sum1 = activation8_m256(_sum1, act_type, act_param);
        }
        lasx_storeu_f32(output_data + 8, _sum1);

        r0 += 16;
        r1 += 16;
        r2 += 16;
        output_data += 16;
      }
      for (; j < output_width; ++j) {
        __m256 _sum0 = _bias0;

        __m256 _r00 = lasx_loadu_f32(r0);
        __m256 _r01 = lasx_loadu_f32(r0 + 8);
        __m256 _r02 = lasx_loadu_f32(r0 + 16);
        __m256 _r10 = lasx_loadu_f32(r1);
        __m256 _r11 = lasx_loadu_f32(r1 + 8);
        __m256 _r12 = lasx_loadu_f32(r1 + 16);
        __m256 _r20 = lasx_loadu_f32(r2);
        __m256 _r21 = lasx_loadu_f32(r2 + 8);
        __m256 _r22 = lasx_loadu_f32(r2 + 16);

        _sum0 = lasx_fmadd_f32(_k00, _r00, _sum0);
        _sum0 = lasx_fmadd_f32(_k01, _r01, _sum0);
        _sum0 = lasx_fmadd_f32(_k02, _r02, _sum0);
        _sum0 = lasx_fmadd_f32(_k10, _r10, _sum0);
        _sum0 = lasx_fmadd_f32(_k11, _r11, _sum0);
        _sum0 = lasx_fmadd_f32(_k12, _r12, _sum0);
        _sum0 = lasx_fmadd_f32(_k20, _r20, _sum0);
        _sum0 = lasx_fmadd_f32(_k21, _r21, _sum0);
        _sum0 = lasx_fmadd_f32(_k22, _r22, _sum0);

        if (has_act) {
          _sum0 = activation8_m256(_sum0, act_type, act_param);
        }
        lasx_storeu_f32(output_data, _sum0);

        r0 += 8;
        r1 += 8;
        r2 += 8;
        output_data += 8;
      }
      r0 += 2 * 8;
      r1 += 2 * 8;
      r2 += 2 * 8;
    }  // end of for output_height
  }    // end of for batch_size * channel_num
}

// input  [bs, ic/8, ih, iw, 8]
// filter [1,  oc/8, kh, kw, 8]
// bias   [    oc             ]
// output [bs, oc/8, oh, ow, 8]
void conv_depthwise_3x3s2_m256(lite::Tensor* input,
                               lite::Tensor* output,
                               lite::Tensor* filter,
                               lite::Tensor* bias,
                               const bool has_act,
                               const lite_api::ActivationType act_type,
                               const operators::ActivationParam act_param) {
  // input [bs, ic/8, ih, iw, 8]
  CHECK_EQ(input->dims().size(), 5UL);
  const int batch_size = input->dims()[0];
  const int channel_num = input->dims()[1];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const float* input_data = input->data<float>();

  // filter [1, oc/8, kh, kw, 8]
  CHECK_EQ(filter->dims().size(), 5UL);
  const int kernel_h = filter->dims()[2];
  const int kernel_w = filter->dims()[3];
  const float* filter_data = filter->data<float>();

  // output [bs, oc/8, oh, ow, 8]
  CHECK_EQ(output->dims().size(), 5UL);
  const int output_height = output->dims()[2];  // 2
  const int output_width = output->dims()[3];   // 2
  float* output_data = output->mutable_data<float>();

  const int input_group_step = input_width * 8;
  const int input_channel_step = input_height * input_width * 8;
  const int input_batch_step = channel_num * input_height * input_width * 8;

  const int filter_channel_step = kernel_h * kernel_w * 8;

  const int tailstep = (input_width - 2 * output_width + input_width) * 8;

  for (int bs = 0; bs < batch_size; ++bs) {
    for (int ic = 0; ic < channel_num; ++ic) {
      __m256 _bias0 = bias ? lasx_loadu_f32(bias->data<float>() + ic * 8)
                           : lasx_set1_f32(0.f);

      const float* k0 = filter_data + ic * filter_channel_step;

      const float* r0 =
          input_data + bs * input_batch_step + ic * input_channel_step;
      const float* r1 = r0 + input_group_step;
      const float* r2 = r1 + input_group_step;

      __m256 _k00 = lasx_loadu_f32(k0);
      __m256 _k01 = lasx_loadu_f32(k0 + 8);
      __m256 _k02 = lasx_loadu_f32(k0 + 16);
      __m256 _k10 = lasx_loadu_f32(k0 + 24);
      __m256 _k11 = lasx_loadu_f32(k0 + 32);
      __m256 _k12 = lasx_loadu_f32(k0 + 40);
      __m256 _k20 = lasx_loadu_f32(k0 + 48);
      __m256 _k21 = lasx_loadu_f32(k0 + 56);
      __m256 _k22 = lasx_loadu_f32(k0 + 64);

      for (int i = 0; i < output_height; ++i) {
        int j = 0;
        for (; j + 3 < output_width; j += 4) {
          __m256 _sum0 = _bias0;

          __m256 _r00 = lasx_loadu_f32(r0);
          __m256 _r01 = lasx_loadu_f32(r0 + 8);
          __m256 _r02 = lasx_loadu_f32(r0 + 16);
          __m256 _r10 = lasx_loadu_f32(r1);
          __m256 _r11 = lasx_loadu_f32(r1 + 8);
          __m256 _r12 = lasx_loadu_f32(r1 + 16);
          __m256 _r20 = lasx_loadu_f32(r2);
          __m256 _r21 = lasx_loadu_f32(r2 + 8);
          __m256 _r22 = lasx_loadu_f32(r2 + 16);

          _sum0 = lasx_fmadd_f32(_k00, _r00, _sum0);
          _sum0 = lasx_fmadd_f32(_k01, _r01, _sum0);
          _sum0 = lasx_fmadd_f32(_k02, _r02, _sum0);
          _sum0 = lasx_fmadd_f32(_k10, _r10, _sum0);
          _sum0 = lasx_fmadd_f32(_k11, _r11, _sum0);
          _sum0 = lasx_fmadd_f32(_k12, _r12, _sum0);
          _sum0 = lasx_fmadd_f32(_k20, _r20, _sum0);
          _sum0 = lasx_fmadd_f32(_k21, _r21, _sum0);
          _sum0 = lasx_fmadd_f32(_k22, _r22, _sum0);

          if (has_act) {
            _sum0 = activation8_m256(_sum0, act_type, act_param);
          }
          lasx_storeu_f32(output_data, _sum0);

          __m256 _sum1 = _bias0;
          __m256 _r03 = lasx_loadu_f32(r0 + 24);
          __m256 _r13 = lasx_loadu_f32(r1 + 24);
          __m256 _r23 = lasx_loadu_f32(r2 + 24);
          __m256 _r04 = lasx_loadu_f32(r0 + 32);
          __m256 _r14 = lasx_loadu_f32(r1 + 32);
          __m256 _r24 = lasx_loadu_f32(r2 + 32);

          _sum1 = lasx_fmadd_f32(_k00, _r02, _sum1);
          _sum1 = lasx_fmadd_f32(_k01, _r03, _sum1);
          _sum1 = lasx_fmadd_f32(_k02, _r04, _sum1);
          _sum1 = lasx_fmadd_f32(_k10, _r12, _sum1);
          _sum1 = lasx_fmadd_f32(_k11, _r13, _sum1);
          _sum1 = lasx_fmadd_f32(_k12, _r14, _sum1);
          _sum1 = lasx_fmadd_f32(_k20, _r22, _sum1);
          _sum1 = lasx_fmadd_f32(_k21, _r23, _sum1);
          _sum1 = lasx_fmadd_f32(_k22, _r24, _sum1);

          if (has_act) {
            _sum1 = activation8_m256(_sum1, act_type, act_param);
          }
          lasx_storeu_f32(output_data + 8, _sum1);

          __m256 _sum2 = _bias0;
          __m256 _r05 = lasx_loadu_f32(r0 + 40);
          __m256 _r15 = lasx_loadu_f32(r1 + 40);
          __m256 _r25 = lasx_loadu_f32(r2 + 40);
          __m256 _r06 = lasx_loadu_f32(r0 + 48);
          __m256 _r16 = lasx_loadu_f32(r1 + 48);
          __m256 _r26 = lasx_loadu_f32(r2 + 48);

          _sum2 = lasx_fmadd_f32(_k00, _r04, _sum2);
          _sum2 = lasx_fmadd_f32(_k01, _r05, _sum2);
          _sum2 = lasx_fmadd_f32(_k02, _r06, _sum2);
          _sum2 = lasx_fmadd_f32(_k10, _r14, _sum2);
          _sum2 = lasx_fmadd_f32(_k11, _r15, _sum2);
          _sum2 = lasx_fmadd_f32(_k12, _r16, _sum2);
          _sum2 = lasx_fmadd_f32(_k20, _r24, _sum2);
          _sum2 = lasx_fmadd_f32(_k21, _r25, _sum2);
          _sum2 = lasx_fmadd_f32(_k22, _r26, _sum2);

          if (has_act) {
            _sum2 = activation8_m256(_sum2, act_type, act_param);
          }
          lasx_storeu_f32(output_data + 16, _sum2);

          __m256 _sum3 = _bias0;
          __m256 _r07 = lasx_loadu_f32(r0 + 56);
          __m256 _r17 = lasx_loadu_f32(r1 + 56);
          __m256 _r27 = lasx_loadu_f32(r2 + 56);
          __m256 _r08 = lasx_loadu_f32(r0 + 64);
          __m256 _r18 = lasx_loadu_f32(r1 + 64);
          __m256 _r28 = lasx_loadu_f32(r2 + 64);

          _sum3 = lasx_fmadd_f32(_k00, _r06, _sum3);
          _sum3 = lasx_fmadd_f32(_k01, _r07, _sum3);
          _sum3 = lasx_fmadd_f32(_k02, _r08, _sum3);
          _sum3 = lasx_fmadd_f32(_k10, _r16, _sum3);
          _sum3 = lasx_fmadd_f32(_k11, _r17, _sum3);
          _sum3 = lasx_fmadd_f32(_k12, _r18, _sum3);
          _sum3 = lasx_fmadd_f32(_k20, _r26, _sum3);
          _sum3 = lasx_fmadd_f32(_k21, _r27, _sum3);
          _sum3 = lasx_fmadd_f32(_k22, _r28, _sum3);

          if (has_act) {
            _sum3 = activation8_m256(_sum3, act_type, act_param);
          }
          lasx_storeu_f32(output_data + 24, _sum3);

          r0 += 2 * 32;
          r1 += 2 * 32;
          r2 += 2 * 32;
          output_data += 32;
        }
        for (; j + 1 < output_width; j += 2) {
          __m256 _sum0 = _bias0;

          __m256 _r00 = lasx_loadu_f32(r0);
          __m256 _r01 = lasx_loadu_f32(r0 + 8);
          __m256 _r02 = lasx_loadu_f32(r0 + 16);
          __m256 _r10 = lasx_loadu_f32(r1);
          __m256 _r11 = lasx_loadu_f32(r1 + 8);
          __m256 _r12 = lasx_loadu_f32(r1 + 16);
          __m256 _r20 = lasx_loadu_f32(r2);
          __m256 _r21 = lasx_loadu_f32(r2 + 8);
          __m256 _r22 = lasx_loadu_f32(r2 + 16);

          _sum0 = lasx_fmadd_f32(_k00, _r00, _sum0);
          _sum0 = lasx_fmadd_f32(_k01, _r01, _sum0);
          _sum0 = lasx_fmadd_f32(_k02, _r02, _sum0);
          _sum0 = lasx_fmadd_f32(_k10, _r10, _sum0);
          _sum0 = lasx_fmadd_f32(_k11, _r11, _sum0);
          _sum0 = lasx_fmadd_f32(_k12, _r12, _sum0);
          _sum0 = lasx_fmadd_f32(_k20, _r20, _sum0);
          _sum0 = lasx_fmadd_f32(_k21, _r21, _sum0);
          _sum0 = lasx_fmadd_f32(_k22, _r22, _sum0);

          if (has_act) {
            _sum0 = activation8_m256(_sum0, act_type, act_param);
          }
          lasx_storeu_f32(output_data, _sum0);

          __m256 _sum1 = _bias0;
          __m256 _r03 = lasx_loadu_f32(r0 + 24);
          __m256 _r13 = lasx_loadu_f32(r1 + 24);
          __m256 _r23 = lasx_loadu_f32(r2 + 24);
          __m256 _r04 = lasx_loadu_f32(r0 + 32);
          __m256 _r14 = lasx_loadu_f32(r1 + 32);
          __m256 _r24 = lasx_loadu_f32(r2 + 32);

          _sum1 = lasx_fmadd_f32(_k00, _r02, _sum1);
          _sum1 = lasx_fmadd_f32(_k01, _r03, _sum1);
          _sum1 = lasx_fmadd_f32(_k02, _r04, _sum1);
          _sum1 = lasx_fmadd_f32(_k10, _r12, _sum1);
          _sum1 = lasx_fmadd_f32(_k11, _r13, _sum1);
          _sum1 = lasx_fmadd_f32(_k12, _r14, _sum1);
          _sum1 = lasx_fmadd_f32(_k20, _r22, _sum1);
          _sum1 = lasx_fmadd_f32(_k21, _r23, _sum1);
          _sum1 = lasx_fmadd_f32(_k22, _r24, _sum1);

          if (has_act) {
            _sum1 = activation8_m256(_sum1, act_type, act_param);
          }
          lasx_storeu_f32(output_data + 8, _sum1);

          r0 += 2 * 16;
          r1 += 2 * 16;
          r2 += 2 * 16;
          output_data += 16;
        }
        for (; j < output_width; j++) {
          __m256 _sum0 = _bias0;

          __m256 _r00 = lasx_loadu_f32(r0);
          __m256 _r01 = lasx_loadu_f32(r0 + 8);
          __m256 _r02 = lasx_loadu_f32(r0 + 16);
          __m256 _r10 = lasx_loadu_f32(r1);
          __m256 _r11 = lasx_loadu_f32(r1 + 8);
          __m256 _r12 = lasx_loadu_f32(r1 + 16);
          __m256 _r20 = lasx_loadu_f32(r2);
          __m256 _r21 = lasx_loadu_f32(r2 + 8);
          __m256 _r22 = lasx_loadu_f32(r2 + 16);

          _sum0 = lasx_fmadd_f32(_k00, _r00, _sum0);
          _sum0 = lasx_fmadd_f32(_k01, _r01, _sum0);
          _sum0 = lasx_fmadd_f32(_k02, _r02, _sum0);
          _sum0 = lasx_fmadd_f32(_k10, _r10, _sum0);
          _sum0 = lasx_fmadd_f32(_k11, _r11, _sum0);
          _sum0 = lasx_fmadd_f32(_k12, _r12, _sum0);
          _sum0 = lasx_fmadd_f32(_k20, _r20, _sum0);
          _sum0 = lasx_fmadd_f32(_k21, _r21, _sum0);
          _sum0 = lasx_fmadd_f32(_k22, _r22, _sum0);

          if (has_act) {
            _sum0 = activation8_m256(_sum0, act_type, act_param);
          }
          lasx_storeu_f32(output_data, _sum0);

          r0 += 2 * 8;
          r1 += 2 * 8;
          r2 += 2 * 8;
          output_data += 8;
        }
        r0 += tailstep;
        r1 += tailstep;
        r2 += tailstep;
      }  // end of for output_height
    }    // end of for channel_num
  }      // end of for batch_size
}

// input  [bs, ic/8, ih, iw, 8]
// filter [1,  oc/8, kh, kw, 8]
// bias   [    oc             ]
// output [bs, oc/8, oh, ow, 8]
void conv_depthwise_m256(lite::Tensor* input,
                         lite::Tensor* output,
                         lite::Tensor* filter,
                         lite::Tensor* bias,
                         const int stride_h,
                         const int stride_w,
                         const int dilation_h,
                         const int dilation_w,
                         const bool has_act,
                         const lite_api::ActivationType act_type,
                         const operators::ActivationParam act_param) {
  // input [bs, ic/8, ih, iw, 8]
  CHECK_EQ(input->dims().size(), 5UL);
  const int batch_size = input->dims()[0];
  const int channel_num = input->dims()[1];
  const int input_height = input->dims()[2];
  const int input_width = input->dims()[3];
  const float* input_data = input->data<float>();

  // filter [1, oc/8, kh, kw, 8]
  CHECK_EQ(filter->dims().size(), 5UL);
  const int kernel_h = filter->dims()[2];
  const int kernel_w = filter->dims()[3];
  const float* filter_data = filter->data<float>();

  // output [bs, oc/8, oh, ow, 8]
  CHECK_EQ(output->dims().size(), 5UL);
  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  float* output_data = output->mutable_data<float>();

  const int input_group_step = input_width * 8 * stride_h;
  const int input_channel_step = input_height * input_width * 8;
  const int input_batch_step = channel_num * input_height * input_width * 8;

  const int filter_kernel_size = kernel_h * kernel_w;
  const int filter_channel_step = kernel_h * kernel_w * 8;

  // kernel offsets
  std::vector<int> _space_ofs(filter_kernel_size);
  int* space_ofs = &_space_ofs[0];
  {
    int p1 = 0;
    int p2 = 0;
    int gap = input_width * dilation_h - kernel_w * dilation_w;
    for (int i = 0; i < kernel_h; i++) {
      for (int j = 0; j < kernel_w; j++) {
        space_ofs[p1++] = p2;
        p2 += dilation_w;
      }
      p2 += gap;
    }
  }

  for (int bs = 0; bs < batch_size; ++bs) {
    for (int ic = 0; ic < channel_num; ++ic) {
      const float* input_ptr =
          input_data + bs * input_batch_step + ic * input_channel_step;
      const float* filter_ptr = filter_data + ic * filter_channel_step;
      for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
          __m256 _sum = lasx_set1_f32(0.f);

          if (bias) {
            _sum = lasx_loadu_f32((bias->data<float>()) + ic * 8);
          }

          const float* start_ptr =
              input_ptr + i * input_group_step + j * 8 * stride_w;

          for (int k = 0; k < filter_kernel_size; k++) {
            __m256 _input = lasx_loadu_f32(start_ptr + +space_ofs[k] * 8);
            __m256 _filter = lasx_loadu_f32(filter_ptr + k * 8);
            _sum = lasx_fmadd_f32(_input, _filter, _sum);
          }

          if (has_act) {
            _sum = activation8_m256(_sum, act_type, act_param);
          }

          lasx_storeu_f32(output_data, _sum);
          output_data += 8;
        }
      }
    }
  }
}

}  // namespace math
}  // namespace loongarch
}  // namespace lite
}  // namespace paddle
