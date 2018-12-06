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

#ifdef POOL_OP
#ifdef _OPENMP
#include <omp.h>
#endif
#include "framework/tensor.h"
#include "operators/math/pool_3x3.h"
#if __ARM_NEON
#include <arm_neon.h>
#endif  // __ARM_NEON
#include <climits>
namespace paddle_mobile {
namespace operators {
namespace math {
using framework::Tensor;
using std::max;
using std::min;
using std::vector;
void Pool3x3Avgs1p1(const Tensor *input, Tensor *output) {
#if __ARM_NEON
  const int batch_size = static_cast<int>(input->dims()[0]);
  const int input_channel = static_cast<int>(input->dims()[1]);

  const int input_height = static_cast<int>(input->dims()[2]);
  const int input_width = static_cast<int>(input->dims()[3]);
  const int output_height = static_cast<int>(output->dims()[2]);
  const int output_width = static_cast<int>(output->dims()[3]);
  output->mutable_data<float>();

  const int hxw = input_height * input_width;

  const int l = input_height;

  const float coef = 1.0 / 9.0;
  const float coef1 = 1.0 / 6.0;
  const float coef2 = 1.0 / 4.0;

  float32x4_t v_coef = vdupq_n_f32(coef);
  float32x4_t v_coef1 = vdupq_n_f32(coef1);

  for (int b = 0; b < batch_size; b++) {
#pragma omp parallel for
    for (int c = 0; c < input_channel; c++) {
      const float *input_data = input->data<float>() + c * hxw;
      float *output_data = output->data<float>() + c * hxw;

      for (int i = 1; i < output_height - 1; i++) {
        float *output_ptr;
        float32x4_t in0, in1, in2, in3, in4, in5, tmp0, tmp1, tmp2, tmp3, tmp4,
            tmp5, out0;
        for (int m = 1; m < output_width - 4; m += 4) {
          output_ptr = output_data + i * output_width + m;
          in0 = vld1q_f32(input_data + (i - 1) * input_width + m - 1);
          in1 = vld1q_f32(input_data + (i - 1) * input_width + m + 3);
          in2 = vld1q_f32(input_data + i * input_width + m - 1);
          in3 = vld1q_f32(input_data + i * input_width + m + 3);
          in4 = vld1q_f32(input_data + (i + 1) * input_width + m - 1);
          in5 = vld1q_f32(input_data + (i + 1) * input_width + m + 3);

          tmp0 = vextq_f32(in0, in1, 1);
          tmp1 = vextq_f32(in0, in1, 2);
          tmp2 = vextq_f32(in2, in3, 1);
          tmp3 = vextq_f32(in2, in3, 2);
          tmp4 = vextq_f32(in4, in5, 1);
          tmp5 = vextq_f32(in4, in5, 2);

          out0 = in0;
          out0 = vaddq_f32(out0, tmp0);
          out0 = vaddq_f32(out0, tmp1);
          out0 = vaddq_f32(out0, in2);
          out0 = vaddq_f32(out0, tmp2);
          out0 = vaddq_f32(out0, tmp3);
          out0 = vaddq_f32(out0, in4);
          out0 = vaddq_f32(out0, tmp4);
          out0 = vaddq_f32(out0, tmp5);

          vst1q_f32(output_ptr, vmulq_f32(out0, v_coef));
        }
        int m;
        for (m = 1; (m + 3) < output_width - 1; m = m + 4) {
        }

        for (int j = m; j < output_width - 1; j++) {
          output_data[i * output_width + j] =
              input_data[(i - 1) * input_width + j - 1] +
              input_data[(i - 1) * input_width + j] +
              input_data[(i - 1) * input_width + j + 1] +
              input_data[(i)*input_width + j - 1] +
              input_data[(i)*input_width + j] +
              input_data[(i)*input_width + j + 1] +
              input_data[(i + 1) * input_width + j - 1] +
              input_data[(i + 1) * input_width + j] +
              input_data[(i + 1) * input_width + j + 1];
          output_data[i * output_width + j] =
              output_data[i * output_width + j] * coef;
        }
      }

      output_data[0] =
          input_data[0] + input_data[1] + input_data[l] + input_data[l + 1];
      output_data[l - 1] = input_data[l - 2] + input_data[l - 1] +
                           input_data[2 * l - 2] + input_data[2 * l - 1];
      output_data[(l - 1) * l] =
          input_data[(l - 2) * l] + input_data[(l - 2) * l + 1] +
          input_data[(l - 1) * l] + input_data[(l - 1) * l + 1];
      output_data[l * l - 1] = input_data[(l - 2) * (l + 1)] +
                               input_data[(l - 2) * (l + 1) + 1] +
                               input_data[l * l - 2] + input_data[l * l - 1];
      output_data[0] = output_data[0] * coef2;
      output_data[l - 1] = output_data[l - 1] * coef2;
      output_data[(l - 1) * l] = output_data[(l - 1) * l] * coef2;
      output_data[l * l - 1] = output_data[l * l - 1] * coef2;

      for (int i = 1; i < l - 1; ++i) {
        output_data[i * l] = input_data[i * l - l] + input_data[i * l - l + 1] +
                             input_data[i * l] + input_data[i * l + 1] +
                             input_data[i * l + l] + input_data[i * l + l + 1];

        output_data[i * l + l - 1] =
            input_data[i * l + l - 1 - l - 1] + input_data[i * l + l - 1 - l] +
            input_data[i * l + l - 1 - 1] + input_data[i * l + l - 1] +
            input_data[i * l + l - 1 + l - 1] + input_data[i * l + l - 1 + l];
        output_data[i * l] = output_data[i * l] * coef1;
        output_data[i * l + l - 1] = output_data[i * l + l - 1] * coef1;
      }

      int m;
      for (m = 1; m < output_width - 4; m += 4) {
        float *output_ptr = output_data + m;
        float32x4_t in0, in1, in2, in3, tmp0, tmp1, tmp2, tmp3, out0;
        in0 = vld1q_f32(input_data + m - 1);
        in1 = vld1q_f32(input_data + m + 3);
        in2 = vld1q_f32(input_data + input_width + m - 1);
        in3 = vld1q_f32(input_data + input_width + m + 3);
        tmp0 = vextq_f32(in0, in1, 1);
        tmp1 = vextq_f32(in0, in1, 2);
        tmp2 = vextq_f32(in2, in3, 1);
        tmp3 = vextq_f32(in2, in3, 2);
        out0 = in0;
        out0 = vaddq_f32(out0, tmp0);
        out0 = vaddq_f32(out0, tmp1);
        out0 = vaddq_f32(out0, in2);
        out0 = vaddq_f32(out0, tmp2);
        out0 = vaddq_f32(out0, tmp3);

        vst1q_f32(output_ptr, vmulq_f32(out0, v_coef1));
      }

      for (m = 1; (m + 3) < output_width - 1; m += 4) {
      }
      for (int j = m; j < output_width - 1; j++) {
        output_data[j] = input_data[j - 1] + input_data[j] + input_data[j + 1] +
                         input_data[input_width + j - 1] +
                         input_data[input_width + j] +
                         input_data[input_width + j + 1];
        output_data[j] = output_data[j] * coef1;
      }

      for (m = 1; m < output_width - 4; m += 4) {
        float *output_ptr =
            output_data + (output_height - 1) * output_width + m;

        float32x4_t in0, in1, in2, in3, tmp0, tmp1, tmp2, tmp3, out0;
        in0 = vld1q_f32(input_data + (output_height - 2) * input_width + m - 1);
        in1 = vld1q_f32(input_data + (output_height - 2) * input_width + m + 3);
        in2 = vld1q_f32(input_data + (output_height - 1) * input_width + m - 1);
        in3 = vld1q_f32(input_data + (output_height - 1) * input_width + m + 3);
        tmp0 = vextq_f32(in0, in1, 1);
        tmp1 = vextq_f32(in0, in1, 2);
        tmp2 = vextq_f32(in2, in3, 1);
        tmp3 = vextq_f32(in2, in3, 2);
        out0 = in0;
        out0 = vaddq_f32(out0, tmp0);
        out0 = vaddq_f32(out0, tmp1);
        out0 = vaddq_f32(out0, in2);
        out0 = vaddq_f32(out0, tmp2);
        out0 = vaddq_f32(out0, tmp3);

        vst1q_f32(output_ptr, vmulq_f32(out0, v_coef1));
      }
      for (m = 1; (m + 3) < output_width - 1; m = m + 4) {
      }
      for (int j = m; j < output_width - 1; j++) {
        output_data[(output_height - 1) * input_width + j] =
            input_data[(output_height - 2) * input_width + j - 1] +
            input_data[(output_height - 2) * input_width + j] +
            input_data[(output_height - 2) * input_width + j + 1] +
            input_data[(output_height - 1) * input_width + j - 1] +
            input_data[(output_height - 1) * input_width + j] +
            input_data[(output_height - 1) * input_width + j + 1];
        output_data[(output_height - 1) * output_width + j] =
            output_data[(output_height - 1) * output_width + j] * coef1;
      }
    }
  }

//  const int batch_size = input->dims()[0];
//
//  const int h_in = input->dims()[2];
//
//  const int w_in = input->dims()[3];
//
//  const int output_channels = output->dims()[1];
//
//  const int h_out = output->dims()[2];
//  const int w_out = output->dims()[3];
//  const int outputdata_channel_stride = h_out * w_out;
//  const int inputdata_channel_stride = h_in * w_in;
//  const int input_batch_stride = output_channels * inputdata_channel_stride;
//  const int output_batch_stride = output_channels *
//  outputdata_channel_stride; float *out_data = output->data<float>(); const
//  float *input_data = input->data<float>();
//
//  const float coef = 1.0 / 9.0;
//  for (int k = 0; k < batch_size; ++k) {
// #pragma omp parallel for
//    for (int c = 0; c < output_channels; ++c) {
//      const float *input_seg = input_data + c * inputdata_channel_stride;
//      float *output_seg = out_data + c * outputdata_channel_stride;
//      // four corner point
//      output_seg[0] = (input_seg[0] + input_seg[1] + input_seg[w_in] +
//                       input_seg[w_in + 1]) *
//                      coef;
//      output_seg[w_out - 1] =
//          (input_seg[w_in - 2] + input_seg[w_in - 1] + input_seg[w_in * 2 -
//          2] +
//           input_seg[2 * w_in - 1]) *
//          coef;
//      output_seg[(h_out - 1) * w_out] =
//          (input_seg[(h_in - 2) * w_in] + input_seg[(h_in - 2) * w_in + 1] +
//           input_seg[(h_in - 1) * w_in] + input_seg[(h_in - 1) * w_in + 1])
//           *
//          coef;
//      output_seg[h_out * w_out - 1] =
//          (input_seg[h_in * w_in - 1] + input_seg[h_in * w_in - 2] +
//           input_seg[(h_in - 1) * w_in - 1] +
//           input_seg[(h_in - 1) * w_in - 2]) *
//          coef;
//      // left side & right side
//      for (int i = 1; i < h_in - 1; ++i) {
//        output_seg[i * w_out] =
//            (input_seg[i * w_in - w_in] + input_seg[i * w_in - w_in + 1] +
//             input_seg[i * w_in] + input_seg[i * w_in + 1] +
//             input_seg[i * w_in + w_in] + input_seg[i * w_in + w_in + 1]) *
//            coef;
//        output_seg[i * w_out + w_out - 1] =
//            (input_seg[i * w_in - w_in + w_in - 2] +
//             input_seg[i * w_in - w_in + 1 + w_in - 2] +
//             input_seg[i * w_in + w_in - 2] +
//             input_seg[i * w_in + 1 + w_in - 2] +
//             input_seg[i * w_in + w_in + w_in - 2] +
//             input_seg[i * w_in + w_in + 1 + w_in - 2]) *
//            coef;
//      }
//      // top 1 row & bottom 1 row
//      const float *input_tmp = input_seg;
//
//      float32x4_t in0, in1, in2, in3, in4, in5, in6, in7, tmp0, tmp1, tmp2,
//          tmp3, tmp4, tmp5, sum, out0;
//      float32x4_t v_coef = vdupq_n_f32(coef);
//      in0 = vld1q_f32(input_tmp);
//      in2 = vld1q_f32(input_tmp + w_in);
//      const float *input_tmp_end = input_tmp + (h_in - 2) * w_in;
//      in4 = vld1q_f32(input_tmp_end);
//      in6 = vld1q_f32(input_tmp_end + w_in);
//      int c_mid = w_out - 2;
//      auto output_ptr = output_seg + 1;
//      for (; c_mid > 3; c_mid -= 4) {
//        in1 = vld1q_f32(input_tmp + 4);
//        in3 = vld1q_f32(input_tmp + w_in + 4);
//
//        tmp0 = vextq_f32(in0, in1, 1);
//        tmp1 = vextq_f32(in0, in1, 2);
//
//        tmp2 = vextq_f32(in2, in3, 1);
//        tmp3 = vextq_f32(in2, in3, 2);
//
//        sum = vaddq_f32(in0, tmp0);
//        sum = vaddq_f32(sum, tmp1);
//        sum = vaddq_f32(sum, in2);
//        sum = vaddq_f32(sum, tmp2);
//        sum = vaddq_f32(sum, tmp3);
//
//        vst1q_f32(output_ptr, vmulq_f32(sum, v_coef));
//
//        in5 = vld1q_f32(input_tmp_end + 4);
//        in7 = vld1q_f32(input_tmp_end + w_in + 4);
//
//        tmp0 = vextq_f32(in4, in5, 1);
//        tmp1 = vextq_f32(in4, in5, 2);
//        tmp2 = vextq_f32(in6, in7, 1);
//        tmp3 = vextq_f32(in6, in7, 2);
//
//        sum = vaddq_f32(in0, tmp0);
//        sum = vaddq_f32(sum, tmp1);
//        sum = vaddq_f32(sum, in2);
//        sum = vaddq_f32(sum, tmp2);
//        sum = vaddq_f32(sum, tmp3);
//
//        vst1q_f32(output_ptr + (h_out - 1) * w_out, vmulq_f32(sum, v_coef));
//
//        // can optimize to each 8 stride.
//        input_tmp += 4;
//        input_tmp_end += 4;
//        output_ptr += 4;
//        in0 = in1;
//        in2 = in3;
//        in4 = in5;
//        in6 = in7;
//      }
//      // top right remain
//      float32x4_t pad0 = vdupq_n_f32(input_seg[w_in - 1]);
//      float32x4_t pad1 = vdupq_n_f32(input_seg[2 * w_in - 1]);
//
//      tmp0 = vextq_f32(in0, pad0, 1);
//      tmp1 = vextq_f32(in0, pad0, 2);
//      tmp2 = vextq_f32(in2, pad1, 2);
//      tmp3 = vextq_f32(in2, pad1, 2);
//
//      sum = vaddq_f32(in0, tmp0);
//      sum = vaddq_f32(sum, tmp1);
//      sum = vaddq_f32(sum, in2);
//      sum = vaddq_f32(sum, tmp2);
//      sum = vaddq_f32(sum, tmp3);
//      out0 = vmulq_f32(sum, v_coef);
//
//      for (int i = 0; i < c_mid; ++i) {
//        if (i == 0) {
//          vst1q_lane_f32(output_ptr + i, out0, 0);
//        }
//        if (i == 1) {
//          vst1q_lane_f32(output_ptr + i, out0, 1);
//        }
//        if (i == 2) {
//          vst1q_lane_f32(output_ptr + i, out0, 2);
//        }
//      }
//
//      // bottom_right remain
//      float32x4_t pad2 = vdupq_n_f32(input_seg[(h_in - 1) * w_in - 1]);
//      float32x4_t pad3 = vdupq_n_f32(input_seg[h_in * w_in - 1]);
//
//      tmp0 = vextq_f32(in4, pad2, 1);
//      tmp1 = vextq_f32(in4, pad2, 2);
//      tmp2 = vextq_f32(in6, pad3, 2);
//      tmp3 = vextq_f32(in6, pad3, 2);
//
//      sum = vaddq_f32(in4, tmp0);
//      sum = vaddq_f32(sum, tmp1);
//      sum = vaddq_f32(sum, in6);
//      sum = vaddq_f32(sum, tmp2);
//      sum = vaddq_f32(sum, tmp3);
//      out0 = vmulq_f32(sum, v_coef);
//
//      for (int i = 0; i < c_mid; ++i) {
//        if (i == 0) {
//          vst1q_lane_f32(output_ptr + (h_out - 1) * w_out + i, out0, 0);
//        }
//        if (i == 1) {
//          vst1q_lane_f32(output_ptr + (h_out - 1) * w_out + i, out0, 1);
//        }
//        if (i == 2) {
//          vst1q_lane_f32(output_ptr + (h_out - 1) * w_out + i, out0, 2);
//        }
//      }
//      // mid
//      for (int j = 0; j < h_out - 2; ++j) {
//        output_ptr = output_seg + w_out * (j + 1) + 1;
//        input_tmp = input_seg + j * w_in;
//
//        in0 = vld1q_f32(input_tmp);
//        in2 = vld1q_f32(input_tmp + w_in);
//        in4 = vld1q_f32(input_tmp + 2 * w_in);
//        c_mid = w_out - 2;
//        for (; c_mid > 3; c_mid -= 4) {
//          in1 = vld1q_f32(input_tmp + 4);
//          in3 = vld1q_f32(input_tmp + w_in + 4);
//          in5 = vld1q_f32(input_tmp + 2 * w_in + 4);
//
//          tmp0 = vextq_f32(in0, in1, 1);
//          tmp1 = vextq_f32(in0, in1, 2);
//          tmp2 = vextq_f32(in2, in3, 1);
//          tmp3 = vextq_f32(in2, in3, 2);
//          tmp4 = vextq_f32(in4, in5, 1);
//          tmp5 = vextq_f32(in4, in5, 2);
//
//          sum = vaddq_f32(in0, tmp0);
//          sum = vaddq_f32(sum, tmp1);
//          sum = vaddq_f32(sum, in2);
//          sum = vaddq_f32(sum, tmp2);
//          sum = vaddq_f32(sum, tmp3);
//          sum = vaddq_f32(sum, in4);
//          sum = vaddq_f32(sum, tmp4);
//          sum = vaddq_f32(sum, tmp5);
//
//          out0 = vmulq_f32(sum, v_coef);
//          vst1q_f32(output_ptr, out0);
//          output_ptr += 4;
//          input_tmp += 4;
//          in0 = in1;
//          in2 = in3;
//          in4 = in5;
//        }
//        // mid remain
//        float32x4_t pad0 = vdupq_n_f32(input_seg[(j + 1) * w_in - 1]);
//        float32x4_t pad1 = vdupq_n_f32(input_seg[(j + 2) * w_in - 1]);
//        float32x4_t pad2 = vdupq_n_f32(input_seg[(j + 2) * w_in - 1]);
//
//        tmp0 = vextq_f32(in0, pad0, 1);
//        tmp1 = vextq_f32(in0, pad0, 2);
//        tmp2 = vextq_f32(in2, pad1, 1);
//        tmp3 = vextq_f32(in2, pad1, 2);
//        tmp4 = vextq_f32(in4, pad2, 1);
//        tmp5 = vextq_f32(in4, pad2, 2);
//
//        sum = vaddq_f32(in0, tmp0);
//        sum = vaddq_f32(sum, tmp1);
//        sum = vaddq_f32(sum, in2);
//        sum = vaddq_f32(sum, tmp2);
//        sum = vaddq_f32(sum, tmp3);
//        sum = vaddq_f32(sum, in4);
//        sum = vaddq_f32(sum, tmp4);
//        sum = vaddq_f32(sum, tmp5);
//        out0 = vmulq_f32(sum, v_coef);
//
//        for (int i = 0; i < c_mid; ++i) {
//          if (i == 0) {
//            vst1q_lane_f32(output_ptr + i, out0, 0);
//          }
//          if (i == 1) {
//            vst1q_lane_f32(output_ptr + i, out0, 1);
//          }
//          if (i == 2) {
//            vst1q_lane_f32(output_ptr + i, out0, 2);
//          }
//        }
//      }
//      //      input_data += inputdata_channel_stride;
//      //      out_data += outputdata_channel_stride;
//    }
//    input_data += input_batch_stride;
//    out_data += output_batch_stride;
//  }
#endif
}

void Pool3x3Maxs1p1(const Tensor *input, Tensor *output) {
#if __ARM_NEON
  const int batch_size = input->dims()[0];

  const int h_in = input->dims()[2];

  const int w_in = input->dims()[3];

  const int output_channels = output->dims()[1];

  const int h_out = output->dims()[2];
  const int w_out = output->dims()[3];
  const int outputdata_channel_stride = h_out * w_out;
  const int inputdata_channel_stride = h_in * w_in;
  const int input_batch_stride = output_channels * inputdata_channel_stride;
  const int output_batch_stride = output_channels * outputdata_channel_stride;
  float *out_data = output->mutable_data<float>();
  const float *input_data = input->data<float>();
  for (int k = 0; k < batch_size; ++k) {
#pragma omp parallel for
    for (int c = 0; c < output_channels; ++c) {
      const float *input_seg = input_data + c * inputdata_channel_stride;
      float *output_seg = out_data + c * outputdata_channel_stride;
      // four corner point
      output_seg[0] = std::max(std::max(input_seg[0], input_seg[1]),
                               std::max(input_seg[w_in], input_seg[w_in + 1]));
      output_seg[w_out - 1] =
          std::max(std::max(input_seg[w_in - 2], input_seg[w_in - 1]),
                   std::max(input_seg[w_in * 2 - 2], input_seg[2 * w_in - 1]));
      output_seg[(h_out - 1) * w_out] =
          std::max(std::max(input_seg[(h_in - 2) * w_in],
                            input_seg[(h_in - 2) * w_in + 1]),
                   std::max(input_seg[(h_in - 1) * w_in],
                            input_seg[(h_in - 1) * w_in + 1]));
      output_seg[h_out * w_out - 1] = std::max(
          std::max(input_seg[(h_in - 1) * w_in - 1],
                   input_seg[(h_in - 1) * w_in - 2]),
          std::max(input_seg[h_in * w_in - 1], input_seg[h_in * w_in - 2]));
      // left side & right side
      for (int i = 1; i < h_in - 1; ++i) {
        float max1 = std::max(input_seg[i * w_in - w_in],
                              input_seg[i * w_in - w_in + 1]);
        float max2 = std::max(input_seg[i * w_in], input_seg[i * w_in + 1]);
        float max3 = std::max(input_seg[i * w_in + w_in],
                              input_seg[i * w_in + w_in + 1]);
        output_seg[i * w_out] = std::max(std::max(max1, max2), max3);

        max1 = std::max(input_seg[i * w_in - w_in + w_in - 2],
                        input_seg[i * w_in - w_in + 1 + w_in - 2]);
        max2 = std::max(input_seg[i * w_in + w_in - 2],
                        input_seg[i * w_in + 1 + w_in - 2]);
        max3 = std::max(input_seg[i * w_in + w_in + w_in - 2],
                        input_seg[i * w_in + w_in + 1 + w_in - 2]);
        output_seg[i * w_out + w_out - 1] =
            std::max(std::max(max1, max2), max3);
      }
      // top 1 row & bottom 1 row
      const float *input_tmp = input_seg;

      float32x4_t in0, in1, in2, in3, in4, in5, in6, in7, tmp0, tmp1, tmp2,
          tmp3, tmp4, tmp5, max;
      in0 = vld1q_f32(input_tmp);
      in2 = vld1q_f32(input_tmp + w_in);
      const float *input_tmp_end = input_tmp + (h_in - 2) * w_in;
      in4 = vld1q_f32(input_tmp_end);
      in6 = vld1q_f32(input_tmp_end + w_in);
      int c_mid = w_out - 2;
      auto output_ptr = output_seg + 1;
      for (; c_mid > 3; c_mid -= 4) {
        in1 = vld1q_f32(input_tmp + 4);
        in3 = vld1q_f32(input_tmp + w_in + 4);

        tmp0 = vextq_f32(in0, in1, 1);
        tmp1 = vextq_f32(in0, in1, 2);

        tmp2 = vextq_f32(in2, in3, 1);
        tmp3 = vextq_f32(in2, in3, 2);

        max = vmaxq_f32(in0, tmp0);
        max = vmaxq_f32(max, tmp1);
        max = vmaxq_f32(max, in2);
        max = vmaxq_f32(max, tmp2);
        max = vmaxq_f32(max, tmp3);

        vst1q_f32(output_ptr, max);

        in5 = vld1q_f32(input_tmp_end + 4);
        in7 = vld1q_f32(input_tmp_end + w_in + 4);

        tmp0 = vextq_f32(in4, in5, 1);
        tmp1 = vextq_f32(in4, in5, 2);
        tmp2 = vextq_f32(in6, in7, 1);
        tmp3 = vextq_f32(in6, in7, 2);

        max = vmaxq_f32(in4, tmp0);
        max = vmaxq_f32(max, tmp1);
        max = vmaxq_f32(max, in6);
        max = vmaxq_f32(max, tmp2);
        max = vmaxq_f32(max, tmp3);

        vst1q_f32(output_ptr + (h_out - 1) * w_out, max);

        input_tmp += 4;
        input_tmp_end += 4;
        output_ptr += 4;
        in0 = in1;
        in2 = in3;
        in4 = in5;
        in6 = in7;
      }
      // top right remain
      float32x4_t pad0 = vdupq_n_f32(input_seg[w_in - 1]);
      float32x4_t pad1 = vdupq_n_f32(input_seg[2 * w_in - 1]);

      tmp0 = vextq_f32(in0, pad0, 1);
      tmp1 = vextq_f32(in0, pad0, 2);
      tmp2 = vextq_f32(in2, pad1, 1);
      tmp3 = vextq_f32(in2, pad1, 2);

      max = vmaxq_f32(in0, tmp0);
      max = vmaxq_f32(max, tmp1);
      max = vmaxq_f32(max, in2);
      max = vmaxq_f32(max, tmp2);
      max = vmaxq_f32(max, tmp3);

      for (int i = 0; i < c_mid; ++i) {
        if (i == 0) {
          vst1q_lane_f32(output_ptr + i, max, 0);
        }
        if (i == 1) {
          vst1q_lane_f32(output_ptr + i, max, 1);
        }
        if (i == 2) {
          vst1q_lane_f32(output_ptr + i, max, 2);
        }
      }

      // bottom_right remain
      float32x4_t pad2 = vdupq_n_f32(input_seg[(h_in - 1) * w_in - 1]);
      float32x4_t pad3 = vdupq_n_f32(input_seg[h_in * w_in - 1]);

      tmp0 = vextq_f32(in4, pad2, 1);
      tmp1 = vextq_f32(in4, pad2, 2);
      tmp2 = vextq_f32(in6, pad3, 1);
      tmp3 = vextq_f32(in6, pad3, 2);

      max = vmaxq_f32(in4, tmp0);
      max = vmaxq_f32(max, tmp1);
      max = vmaxq_f32(max, in6);
      max = vmaxq_f32(max, tmp2);
      max = vmaxq_f32(max, tmp3);

      for (int i = 0; i < c_mid; ++i) {
        if (i == 0) {
          vst1q_lane_f32(output_ptr + (h_out - 1) * w_out + i, max, 0);
        }
        if (i == 1) {
          vst1q_lane_f32(output_ptr + (h_out - 1) * w_out + i, max, 1);
        }
        if (i == 2) {
          vst1q_lane_f32(output_ptr + (h_out - 1) * w_out + i, max, 2);
        }
      }
      // mid
      for (int j = 0; j < h_out - 2; ++j) {
        output_ptr = output_seg + (j + 1) * w_out + 1;
        input_tmp = input_seg + j * w_in;

        in0 = vld1q_f32(input_tmp);
        in2 = vld1q_f32(input_tmp + w_in);
        in4 = vld1q_f32(input_tmp + 2 * w_in);
        c_mid = w_out - 2;
        for (; c_mid > 3; c_mid -= 4) {
          in1 = vld1q_f32(input_tmp + 4);
          in3 = vld1q_f32(input_tmp + w_in + 4);
          in5 = vld1q_f32(input_tmp + 2 * w_in + 4);

          tmp0 = vextq_f32(in0, in1, 1);
          tmp1 = vextq_f32(in0, in1, 2);
          tmp2 = vextq_f32(in2, in3, 1);
          tmp3 = vextq_f32(in2, in3, 2);
          tmp4 = vextq_f32(in4, in5, 1);
          tmp5 = vextq_f32(in4, in5, 2);

          max = vmaxq_f32(in0, tmp0);
          max = vmaxq_f32(max, tmp1);
          max = vmaxq_f32(max, in2);
          max = vmaxq_f32(max, tmp2);
          max = vmaxq_f32(max, tmp3);
          max = vmaxq_f32(max, in4);
          max = vmaxq_f32(max, tmp4);
          max = vmaxq_f32(max, tmp5);

          vst1q_f32(output_ptr, max);
          output_ptr += 4;
          input_tmp += 4;
          in0 = in1;
          in2 = in3;
          in4 = in5;
        }
        // mid remain
        float32x4_t pad0 = vdupq_n_f32(input_seg[(j + 1) * w_in - 1]);
        float32x4_t pad1 = vdupq_n_f32(input_seg[(j + 2) * w_in - 1]);
        float32x4_t pad2 = vdupq_n_f32(input_seg[(j + 3) * w_in - 1]);

        tmp0 = vextq_f32(in0, pad0, 1);
        tmp1 = vextq_f32(in0, pad0, 2);
        tmp2 = vextq_f32(in2, pad1, 1);
        tmp3 = vextq_f32(in2, pad1, 2);
        tmp4 = vextq_f32(in4, pad2, 1);
        tmp5 = vextq_f32(in4, pad2, 2);

        max = vmaxq_f32(in0, tmp0);
        max = vmaxq_f32(max, tmp1);
        max = vmaxq_f32(max, in2);
        max = vmaxq_f32(max, tmp2);
        max = vmaxq_f32(max, tmp3);
        max = vmaxq_f32(max, in4);
        max = vmaxq_f32(max, tmp4);
        max = vmaxq_f32(max, tmp5);

        for (int i = 0; i < c_mid; ++i) {
          if (i == 0) {
            vst1q_lane_f32(output_ptr + i, max, 0);
          }
          if (i == 1) {
            vst1q_lane_f32(output_ptr + i, max, 1);
          }
          if (i == 2) {
            vst1q_lane_f32(output_ptr + i, max, 2);
          }
        }
      }
      //      input_data += inputdata_channel_stride;
      //      out_data += outputdata_channel_stride;
    }
    input_data += input_batch_stride;
    out_data += output_batch_stride;
  }
#else

#endif
}

void Pool3x3Max(vector<int> strides, vector<int> paddings, const Tensor *input,
                Tensor *output) {
#if __ARM_NEON
  const int batch_size = input->dims()[0];

  const int input_height = input->dims()[2];

  const int input_width = input->dims()[3];

  const int output_channels = output->dims()[1];

  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  //  const int _kernel_size = 3;
  const int stride = strides[0];
  //  const int stride_width = strides[1];
  const int padding = paddings[0];
  //  const int padding_width = paddings[1];
  const float negative_max = -INT_MAX;
  const int input_channel_stride = input_height * input_width;
  const int output_channel_stride = output_height * output_width;

  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();

  const int input_batch_stride = output_channels * input_channel_stride;
  const int output_batch_stride = output_channels * output_channel_stride;
  const float *pos1, *output_ptr;
  int hstart, wstart, hend, wend;
  for (int i = 0; i < batch_size; ++i) {
#pragma omp parallel for
    for (int c = 0; c < output_channels; ++c) {
      const float *input_seg = input_data + c * input_channel_stride;
      float *output_seg = output_data + c * output_channel_stride;
      for (int ph = 0; ph < output_height; ph++) {
        int hstart = ph * stride - padding;
        int hend = min(hstart + 3, input_height);
        hstart = max(hstart, 0);
        for (int pw = 0; pw < output_width; pw++) {
          int wstart = pw * stride - padding;
          int wend = min(wstart + 3, input_width);
          wstart = max(wstart, 0);
          const float *pos1 = input_seg + hstart * input_width + wstart;
          const float *pos2 = input_seg + (hstart + 1) * input_width + wstart;
          const float *pos3 = input_seg + (hstart + 2) * input_width + wstart;
          output_ptr = output_seg + ph * output_width + pw;

          if (hend - hstart != 3 || wend - wstart != 3) {
            float max_value = -INT_MAX;
            for (int h = hstart; h < hend; h++) {
              for (int w = wstart; w < wend; w++) {
                float value = input_seg[h * input_width + w];
                if (value > max_value) {
                  max_value = value;
                }
              }
            }
            output_seg[ph * output_width + pw] = max_value;
          } else {
#if __aarch64__
            const float32x4_t data1 = vld1q_f32(pos1);
            const float32x4_t data2 = vld1q_f32(pos1 + input_width);
            const float32x4_t data3 = vld1q_f32(pos1 + 2 * input_width);
            const float32x4_t max_data =
                vmaxq_f32(vmaxq_f32(data1, data2), data3);
            float32x2_t res =
                vpmax_f32(vget_high_f32(vsetq_lane_f32(-INT_MAX, max_data, 3)),
                          vget_low_f32(max_data));
            res = vpmax_f32(res, res);
            output_seg[ph * output_width + pw] = vget_lane_f32(res, 0);
#else
            asm volatile(
                "vld1.32  {q1}, [%[pos1]]        \n\t"
                "vld1.32  {q2}, [%[pos2]]        \n\t"
                "vld1.32  {q3}, [%[pos3]]        \n\t"
                "vmax.f32 q1, q1, q2            \n\t"
                "vmax.f32 q2, q1, q3            \n\t"
                "vmov.f32 d5[1],  %[negative_max]         \n\t"
                "vpmax.f32  d6, d4, d5            \n\t"
                "vpmax.f32  d7, d6, d6             \n\t"
                "vst1.32 {d7[0]},[%[output_ptr]]    \n\t"
                :
                : [input_seg] "r"(input_seg), [pos1] "r"(pos1),
                  [pos2] "r"(pos2), [pos3] "r"(pos3),
                  [output_ptr] "r"(output_ptr), [negative_max] "r"(negative_max)
                : "memory", "q1", "q2", "q3", "q4");
#endif
          }
        }
      }
    }
    input_data += input_batch_stride;
    output_data += output_batch_stride;
  }
#endif
}

void Pool3x3Avg(vector<int> strides, vector<int> paddings, const Tensor *input,
                Tensor *output) {
#if __ARM_NEON
  const int batch_size = input->dims()[0];

  const int input_height = input->dims()[2];

  const int input_width = input->dims()[3];

  const int output_channels = output->dims()[1];

  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  const int stride = strides[0];
  const int padding = paddings[0];

  const int input_channel_stride = input_height * input_width;
  const int output_channel_stride = output_height * output_width;

  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();
  const float zero = 0;
  const float nine = 1.0 / 9.0;
  const float nine_ptr[] = {nine, nine};

  const int input_batch_stride = output_channels * input_channel_stride;
  const int output_batch_stride = output_channels * output_channel_stride;
  for (int i = 0; i < batch_size; ++i) {
#pragma omp parallel for
    for (int c = 0; c < output_channels; ++c) {
      const float *input_seg = input_data + c * input_channel_stride;
      float *output_seg = output_data + c * output_channel_stride;
      for (int ph = 0; ph < output_height; ph++) {
        for (int pw = 0; pw < output_width; pw++) {
          int hstart = ph * stride - padding;
          int wstart = pw * stride - padding;
          int hend = min(hstart + 3, input_height + padding);
          int wend = min(wstart + 3, input_width + padding);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          hend = min(hend, input_height);
          wend = min(wend, input_width);

          const float *pos1 = input_seg + hstart * input_width + wstart;
          const float *pos2 = input_seg + (hstart + 1) * input_width + wstart;
          const float *pos3 = input_seg + (hstart + 2) * input_width + wstart;
          float *output_ptr = output_seg + ph * output_width + pw;

          if (hend - hstart != 3 || wend - wstart != 3) {
            float sum = 0;
            for (int h = hstart; h < hend; h++) {
              for (int w = wstart; w < wend; w++) {
                sum += input_seg[h * input_width + w];
              }
            }
            output_seg[ph * output_width + pw] =
                sum / ((hend - hstart) * (wend - wstart) * 1.0);
          } else {
#if __aarch64__
#else
            asm volatile(
                "vld1.32  {q1}, [%[pos1]]        \n\t"
                "vld1.32  {q2}, [%[pos2]]        \n\t"
                "vld1.32  {q3}, [%[pos3]]        \n\t"
                "vadd.f32 q1, q1, q2            \n\t"
                "vadd.f32 q2, q1, q3            \n\t"
                "vmov.f32 d5[1],  %[zero]         \n\t"
                "vpadd.f32  d6, d4, d5            \n\t"
                "vpadd.f32  d6, d6, d6             \n\t"
                "vld1.f32 d7, [%[nine_ptr]]!        \n\t"
                "vmul.f32 d6,d7                     \n\t"
                "vst1.32 {d6[0]},[%[output_ptr]]    \n\t"
                :
                : [input_seg] "r"(input_seg), [pos1] "r"(pos1),
                  [pos2] "r"(pos2), [pos3] "r"(pos3),
                  [output_ptr] "r"(output_ptr), [zero] "r"(zero),
                  [nine_ptr] "r"(nine_ptr)
                : "memory", "r6", "q1", "q2", "q3", "q4");
#endif
            const float32x4_t data1 = vld1q_f32(pos1);
            const float32x4_t data2 = vld1q_f32(pos2);
            const float32x4_t data3 = vld1q_f32(pos3);
            const float32x4_t sum_data =
                vaddq_f32(vaddq_f32(data1, data3), data2);
            float32x2_t res =
                vpadd_f32(vget_high_f32(vsetq_lane_f32(0, sum_data, 3)),
                          vget_low_f32(sum_data));
            res = vpadd_f32(res, res);
            output_seg[ph * output_width + pw] = vget_lane_f32(res, 0) / 9.0;
          }
        }
      }
    }
    input_data += input_batch_stride;
    output_data += output_batch_stride;
  }
#else
#endif
}
}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile

#endif
