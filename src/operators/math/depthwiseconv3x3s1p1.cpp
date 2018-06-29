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

#include "operators/math/depthwiseconv3x3s1p1.h"
#include <arm_neon.h>
#include <algorithm>

namespace paddle_mobile {
namespace operators {
namespace math {

using framework::Tensor;

void DepthwiseConv3x3s1p1(const Tensor *input, Tensor filter, Tensor *output,
                          Tensor *bias, bool if_bias, Tensor *new_scale,
                          Tensor *new_bias, bool if_bn, bool if_relu) {
  const float *input_data = input->data<float>();
  const float *filter_data = filter.data<float>();
  float *output_data = output->data<float>();
  const float *bias_data = bias->data<float>();
  const float *newscale_data = new_scale->data<float>();
  const float *newbias_data = new_bias->data<float>();

  const int h = static_cast<int>(input->dims()[2]);
  const int w = static_cast<int>(input->dims()[3]);
  const int l = h;

  const int batch_size = static_cast<int>(input->dims()[0]);
  const int c = static_cast<int>(input->dims()[1]);
  const int hxw = h * w;
  float32x4_t vbias = vdupq_n_f32(0.0);
  float32x4_t vnewbias = vdupq_n_f32(0.0);
  float32x4_t vnewscale = vdupq_n_f32(1.0);
  float32x4_t vzero = vdupq_n_f32(0);

  for (int b = 0; b < batch_size; ++b) {
    const float *filter_data_tmp = filter_data;

    for (int j = 0; j < c; ++j) {
      if (if_bias) {
        vbias = vdupq_n_f32(bias_data[j]);
      }
      if (if_bn) {
        vnewbias = vdupq_n_f32(newbias_data[j]);
        vnewscale = vdupq_n_f32(newscale_data[j]);
      }
      int l_mid = l - 2;  // l=1->l_mid=-1,l=2->l_mid=0
      float w00 = filter_data_tmp[0];
      float w01 = filter_data_tmp[1];
      float w02 = filter_data_tmp[2];
      float w10 = filter_data_tmp[3];
      float w11 = filter_data_tmp[4];
      float w12 = filter_data_tmp[5];
      float w20 = filter_data_tmp[6];
      float w21 = filter_data_tmp[7];
      float w22 = filter_data_tmp[8];

      output_data[0] =(w11 * input_data[0] + w12 * input_data[1] + w21 * input_data[l] +
                                w22 * input_data[l + 1] + bias_data[j]) *
                               newscale_data[j] +
                               newbias_data[j];
      output_data[l - 1] = (w10 * input_data[l - 2] + w11 * input_data[l - 1] +
                            w20 * input_data[2 * l - 2] +
                            w21 * input_data[2 * l - 1] + bias_data[j]) *
                               newscale_data[j] +
                           newbias_data[j];

      output_data[(l - 1) * l] =
          (w01 * input_data[(l - 2) * l] + w02 * input_data[(l - 2) * l + 1] +
           w11 * input_data[(l - 1) * l] + w12 * input_data[(l - 1) * l + 1] +
           bias_data[j]) *
              newscale_data[j] +
          newbias_data[j];
      output_data[l * l - 1] = (w00 * input_data[(l - 2) * (l + 1)] +
                                w01 * input_data[(l - 2) * (l + 1) + 1] +
                                w10 * input_data[l * l - 2] +
                                w11 * input_data[l * l - 1] + bias_data[j]) *
                                   newscale_data[j] +
                               newbias_data[j];
      if(if_relu){
        output_data[0] = output_data[0] < 0 ? 0 : output_data[0];
        output_data[l-1] = output_data[l-1] < 0 ? 0 : output_data[l-1];
        output_data[(l-1)*l] = output_data[(l-1)*l] < 0 ? 0 : output_data[(l-1)*l];
        output_data[l * l - 1] = output_data[l * l - 1] < 0 ? 0 : output_data[l * l - 1];
      }
      for (int i = 1; i < l - 1; ++i) {
        output_data[i * l] =
            (w01 * input_data[i * l - l] + w02 * input_data[i * l - l + 1] +
             w11 * input_data[i * l] + w12 * input_data[i * l + 1] +
             w21 * input_data[i * l + l] + w22 * input_data[i * l + l + 1] +
             bias_data[j]) *
                newscale_data[j] +
            newbias_data[j];
        output_data[i * l + l - 1] =
            (w00 * input_data[i * l + l - 1 - l - 1] +
             w01 * input_data[i * l + l - 1 - l] +
             w10 * input_data[i * l + l - 1 - 1] +
             w11 * input_data[i * l + l - 1] +
             w20 * input_data[i * l + l - 1 + l - 1] +
             w21 * input_data[i * l + l - 1 + l] + bias_data[j]) *
                newscale_data[j] +
            newbias_data[j];
        if(if_relu){
          output_data[i * l] = output_data[i * l] < 0 ? 0 : output_data[i * l];
          output_data[i * l + l - 1] = output_data[i * l + l - 1] < 0 ? 0 : output_data[i * l + l - 1];
        }
      }

      // top 1 row and bottom 1 row
      const float *input_tmp = input_data;

      float32x4_t in0, in1, in2, in3, in4, in5, in6, in7, tmp0, tmp1, tmp2,
          tmp3, tmp4, tmp5, out0;
      in0 = vld1q_f32(input_tmp);
      in2 = vld1q_f32(input_tmp + l);
      const float *input_tmp_end = input_tmp + (l - 2) * l;
      in4 = vld1q_f32(input_tmp_end);
      in6 = vld1q_f32(input_tmp_end + l);
      int c_mid = l_mid;
      auto output_ptr = output_data + 1;
      for (; c_mid > 3; c_mid -= 4) {
        in1 = vld1q_f32(input_tmp + 4);
        in3 = vld1q_f32(input_tmp + l + 4);

        tmp0 = vextq_f32(in0, in1, 1);
        tmp1 = vextq_f32(in0, in1, 2);

        tmp2 = vextq_f32(in2, in3, 1);
        tmp3 = vextq_f32(in2, in3, 2);

        out0 = vmulq_n_f32(in0, w10);
        out0 = vmlaq_n_f32(out0, tmp0, w11);
        out0 = vmlaq_n_f32(out0, tmp1, w12);
        out0 = vmlaq_n_f32(out0, in2, w20);
        out0 = vmlaq_n_f32(out0, tmp2, w21);
        out0 = vmlaq_n_f32(out0, tmp3, w22);
        out0 = vaddq_f32(out0, vbias);
        out0 = vmlaq_f32(vnewbias, vnewscale, out0);
        if (if_relu) {
          out0 = vmaxq_f32(out0, vzero);
        }
        vst1q_f32(output_ptr, out0);

        in5 = vld1q_f32(input_tmp_end + 4);
        in7 = vld1q_f32(input_tmp_end + l + 4);

        tmp0 = vextq_f32(in4, in5, 1);
        tmp1 = vextq_f32(in4, in5, 2);
        tmp2 = vextq_f32(in6, in7, 1);
        tmp3 = vextq_f32(in6, in7, 2);

        out0 = vmulq_n_f32(in4, w00);
        out0 = vmlaq_n_f32(out0, tmp0, w01);
        out0 = vmlaq_n_f32(out0, tmp1, w02);
        out0 = vmlaq_n_f32(out0, in6, w10);
        out0 = vmlaq_n_f32(out0, tmp2, w11);
        out0 = vmlaq_n_f32(out0, tmp3, w12);
        out0 = vaddq_f32(out0, vbias);
        out0 = vmlaq_f32(vnewbias, vnewscale, out0);
        if (if_relu) {
          out0 = vmaxq_f32(out0, vzero);
        }
        vst1q_f32(output_ptr + (l - 1) * l, out0);

        // can optimize to each 8 stride.
        input_tmp += 4;
        input_tmp_end += 4;
        output_ptr += 4;
        in0 = in1;
        in2 = in3;
        in4 = in5;
        in6 = in7;
      }

      // top right pad
      float32x4_t pad0 = vdupq_n_f32(input_data[l - 1]);
      float32x4_t pad1 = vdupq_n_f32(input_data[2 * l - 1]);

      tmp0 = vextq_f32(in0, pad0, 1);
      tmp1 = vextq_f32(in0, pad0, 2);
      tmp2 = vextq_f32(in2, pad1, 1);
      tmp3 = vextq_f32(in2, pad1, 2);

      out0 = vmulq_n_f32(in0, w10);
      out0 = vmlaq_n_f32(out0, tmp0, w11);
      out0 = vmlaq_n_f32(out0, tmp1, w12);
      out0 = vmlaq_n_f32(out0, in2, w20);
      out0 = vmlaq_n_f32(out0, tmp2, w21);
      out0 = vmlaq_n_f32(out0, tmp3, w22);
      out0 = vaddq_f32(out0, vbias);
      out0 = vmlaq_f32(vnewbias, vnewscale, out0);
      if (if_relu) {
        out0 = vmaxq_f32(out0, vzero);
      }
      for (int i = 0; i < c_mid; ++i) {
        if (i == 0) {
          vst1q_lane_f32(output_ptr + i, out0, 0);
        }
        if (i == 1) {
          vst1q_lane_f32(output_ptr + i, out0, 1);
        }
        if (i == 2) {
          vst1q_lane_f32(output_ptr + i, out0, 2);
        }
      }

      // bottom right pad
      float32x4_t pad2 = vdupq_n_f32(input_data[l * l - 1 - l]);
      float32x4_t pad3 = vdupq_n_f32(input_data[l * l - 1]);

      tmp0 = vextq_f32(in4, pad2, 1);
      tmp1 = vextq_f32(in4, pad2, 2);
      tmp2 = vextq_f32(in6, pad3, 1);
      tmp3 = vextq_f32(in6, pad3, 2);

      out0 = vmulq_n_f32(in4, w00);
      out0 = vmlaq_n_f32(out0, tmp0, w01);
      out0 = vmlaq_n_f32(out0, tmp1, w02);
      out0 = vmlaq_n_f32(out0, in6, w10);
      out0 = vmlaq_n_f32(out0, tmp2, w11);
      out0 = vmlaq_n_f32(out0, tmp3, w12);
      out0 = vaddq_f32(out0, vbias);
      out0 = vmlaq_f32(vnewbias, vnewscale, out0);
      if (if_relu) {
        out0 = vmaxq_f32(out0, vzero);
      }
      for (int i = 0; i < c_mid; ++i) {
        if (i == 0) {
          vst1q_lane_f32(output_ptr + (l - 1) * l + i, out0, 0);
        }
        if (i == 1) {
          vst1q_lane_f32(output_ptr + (l - 1) * l + i, out0, 1);
        }
        if (i == 2) {
          vst1q_lane_f32(output_ptr + (l - 1) * l + i, out0, 2);
        }
      }
      // mid

      for (int i = 0; i < l - 2; ++i) {
        auto output_ptr = output_data + (i + 1) * l + 1;
        input_tmp = input_data + i * l;
        auto in0_tmp = vld1q_f32(input_tmp);
        auto in2_tmp = vld1q_f32(input_tmp + l);
        auto in4_tmp = vld1q_f32(input_tmp + l + l);
        c_mid = l_mid;
        for (; c_mid > 3; c_mid -= 4) {
          auto in1_tmp = vld1q_f32(input_tmp + 4);
          auto in3_tmp = vld1q_f32(input_tmp + l + 4);
          auto in5_tmp = vld1q_f32(input_tmp + l + l + 4);

          tmp0 = vextq_f32(in0_tmp, in1_tmp, 1);
          tmp1 = vextq_f32(in0_tmp, in1_tmp, 2);
          tmp2 = vextq_f32(in2_tmp, in3_tmp, 1);
          tmp3 = vextq_f32(in2_tmp, in3_tmp, 2);
          tmp4 = vextq_f32(in4_tmp, in5_tmp, 1);
          tmp5 = vextq_f32(in4_tmp, in5_tmp, 2);

          out0 = vmulq_n_f32(in0_tmp, w00);
          out0 = vmlaq_n_f32(out0, tmp0, w01);
          out0 = vmlaq_n_f32(out0, tmp1, w02);
          out0 = vmlaq_n_f32(out0, in2_tmp, w10);
          out0 = vmlaq_n_f32(out0, tmp2, w11);
          out0 = vmlaq_n_f32(out0, tmp3, w12);
          out0 = vmlaq_n_f32(out0, in4_tmp, w20);
          out0 = vmlaq_n_f32(out0, tmp4, w21);
          out0 = vmlaq_n_f32(out0, tmp5, w22);
          out0 = vaddq_f32(out0, vbias);
          out0 = vmlaq_f32(vnewbias, vnewscale, out0);
          if (if_relu) {
            out0 = vmaxq_f32(out0, vzero);
          }
          vst1q_f32(output_ptr, out0);

          output_ptr += 4;
          input_tmp += 4;
          in0_tmp = in1_tmp;
          in2_tmp = in3_tmp;
          in4_tmp = in5_tmp;
        }

        float32x4_t pad0 = vdupq_n_f32(input_data[i * l + l - 1]);
        float32x4_t pad1 = vdupq_n_f32(input_data[i * l + l - 1 + l]);
        float32x4_t pad2 = vdupq_n_f32(input_data[i * l + l - 1 + l + l]);

        tmp0 = vextq_f32(in0_tmp, pad0, 1);
        tmp1 = vextq_f32(in0_tmp, pad0, 2);
        tmp2 = vextq_f32(in2_tmp, pad1, 1);
        tmp3 = vextq_f32(in2_tmp, pad1, 2);
        tmp4 = vextq_f32(in4_tmp, pad2, 1);
        tmp5 = vextq_f32(in4_tmp, pad2, 2);

        out0 = vmulq_n_f32(in0_tmp, w00);
        out0 = vmlaq_n_f32(out0, tmp0, w01);
        out0 = vmlaq_n_f32(out0, tmp1, w02);
        out0 = vmlaq_n_f32(out0, in2_tmp, w10);
        out0 = vmlaq_n_f32(out0, tmp2, w11);
        out0 = vmlaq_n_f32(out0, tmp3, w12);
        out0 = vmlaq_n_f32(out0, in4_tmp, w20);
        out0 = vmlaq_n_f32(out0, tmp4, w21);
        out0 = vmlaq_n_f32(out0, tmp5, w22);
        out0 = vaddq_f32(out0, vbias);
        out0 = vmlaq_f32(vnewbias, vnewscale, out0);
        if (if_relu) {
          out0 = vmaxq_f32(out0, vzero);
        }
        for (int i = 0; i < c_mid; ++i) {
          if (i == 0) {
            vst1q_lane_f32(output_ptr + i, out0, 0);
          }
          if (i == 1) {
            vst1q_lane_f32(output_ptr + i, out0, 1);
          }
          if (i == 2) {
            vst1q_lane_f32(output_ptr + i, out0, 2);
          }
        }
      }
      output_data += hxw;
      input_data += hxw;
      filter_data_tmp += 9;
    }
  }

}
}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
