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

#include "operators/math/depthwise_conv3x3.h"
#include <vector>
#if __ARM_NEON
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {
namespace math {

void DepthwiseConv3x3(const framework::Tensor *input,
                      const std::vector<int> &strides,
                      const std::vector<int> &paddings,
                      const framework::Tensor *filter, framework::Tensor *bias,
                      framework::Tensor *output, bool if_bias) {
  const int batch_size = input->dims()[0];

  const int input_height = input->dims()[2];

  const int input_width = input->dims()[3];

  const int output_channels = output->dims()[1];

  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  const int _kernel_size = 3;
  const int stride_height = strides[0];
  const int stride_width = strides[1];
  const int padding_height = paddings[0];
  const int padding_width = paddings[1];
  const float zero = 0;
  const int input_channel_stride = input_height * input_width;
  const int output_channel_stride = output_height * output_width;
  const int filter_channel_stride = 9;

  const float *input_data = input->data<float>();
  const float *filter_data = filter->data<float>();
  if (if_bias) {
    math::expand_bias(*bias, 1, output->dims());
    output->ShareDataWith(*bias);
  }
  float *output_data = output->mutable_data<float>();

  const int input_batch_stride = output_channels * input_channel_stride;
  const int output_batch_stride = output_channels * output_channel_stride;
  const int filter_batch_stride = output_channels * output_channel_stride;
  const float *pos1, *pos2, *pos3, *filter1, *filter2, *filter3, *output_ptr;
  int hstart, wstart, hend, wend;
  float result;
  for (int i = 0; i < batch_size; ++i) {
    for (int c = 0; c < output_channels; ++c) {
      filter1 = filter_data;
      filter2 = filter1 + 3;
      filter3 = filter2 + 3;

      for (int ph = 0; ph < output_height; ph++) {
        for (int pw = 0; pw < output_width; pw++) {
          hstart = ph * stride_height - padding_height;
          wstart = pw * stride_width - padding_width;
          hend = std::min(hstart + _kernel_size, input_height + padding_height);
          wend = std::min(wstart + _kernel_size, input_width + padding_width);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          hend = std::min(hend, input_height);
          wend = std::min(wend, input_width);
          pos1 = input_data + hstart * input_width + wstart;
          pos2 = input_data + (hstart + 1) * input_width + wstart;
          pos3 = input_data + (hstart + 2) * input_width + wstart;
          output_ptr = output_data + ph * output_width + pw;

          if (hend - hstart != 3 || wend - wstart != 3) {
            result = 0;
            float fake_input[9] = {0};
            if (hstart == 0 && wstart == 0) {
              // 左上角
              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (j >= 3 - hend && k >= 3 - wend) {
                    fake_input[3 * j + k] =
                        input_data[(j - (3 - hend)) * input_width + k -
                                   (3 - wend)];
                  }
                }
              }
            } else if (hstart == 0 && wend == input_width) {
              // 右上角
              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (j >= 3 - hend && k <= input_width - wstart - 1) {
                    fake_input[3 * j + k] =
                        input_data[(j - (3 - hend)) * input_width + k + wstart];
                  }
                }
              }

            } else if (hend == input_height && wstart == 0) {
              // 左下角

              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (j <= input_height - 1 - hstart && k >= 3 - wend) {
                    fake_input[3 * j + k] =
                        input_data[(j + hstart) * input_width + k - (3 - wend)];
                  }
                }
              }
            } else if (hend == input_height && wend == input_width) {
              // 右下角
              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (j <= input_height - hstart - 1 &&
                      k <= input_width - wstart - 1) {
                    fake_input[3 * j + k] =
                        input_data[(j + hstart) * input_width + k + wstart];
                  }
                }
              }
            } else if (hstart == 0) {
              // 顶部
              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (j >= 3 - hend) {
                    fake_input[3 * j + k] =
                        input_data[(j - (3 - hend)) * input_width + k + wstart];
                  }
                }
              }

            } else if (hend == input_height) {
              // 底部
              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (j <= input_height - hstart - 1) {
                    fake_input[3 * j + k] =
                        input_data[(j + hstart) * input_width + k + wstart];
                  }
                }
              }

            } else if (wstart == 0) {
              // 左侧
              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (k >= 3 - wend) {
                    fake_input[3 * j + k] =
                        input_data[(j + hstart) * input_width +
                                   (k - (3 - wend))];
                  }
                }
              }

            } else if (wend == input_width) {
              // 右侧
              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (k <= input_width - wstart - 1) {
                    fake_input[3 * j + k] =
                        input_data[(j + hstart) * input_width + k + wstart];
                  }
                }
              }
            }
            for (int l = 0; l < 9; ++l) {
              result += fake_input[l] * filter1[l];
            }
            if (if_bias) {
              output_data[ph * output_width + pw] += result;
            } else {
              output_data[ph * output_width + pw] = result;
            }

          } else {
#if __ARM_NEON
#if __aarch64__
            const float32x4_t data1 = vld1q_f32(pos1);
            const float32x4_t data2 = vld1q_f32(pos2);
            const float32x4_t data3 = vld1q_f32(pos3);

            const float32x4_t v_filter1 = vld1q_f32(filter1);
            const float32x4_t v_filter2 = vld1q_f32(filter2);
            const float32x4_t v_filter3 = vld1q_f32(filter3);
            float32x4_t mula = vmulq_f32(data1, v_filter1);
            mula = vmlaq_f32(mula, data2, v_filter2);
            mula = vmlaq_f32(mula, data3, v_filter3);
            float32x2_t res = vpadd_f32(
                vget_high_f32(vsetq_lane_f32(0, mula, 3)), vget_low_f32(mula));
            res = vpadd_f32(res, res);
            if (if_bias) {
              output_data[ph * output_width + pw] += vget_lane_f32(res, 0);
            } else {
              output_data[ph * output_width + pw] = vget_lane_f32(res, 0);
            }
#else
            asm volatile(

                "vld1.32  {q1}, [%[pos1]]        \n\t"
                "vld1.32  {q4}, [%[filter1]]        \n\t"
                "vmov.f32 q0,    #0.0              \n\t"

                "vld1.32  {q2}, [%[pos2]]        \n\t"
                "vld1.32  {q5}, [%[filter2]]        \n\t"
                "vmla.f32 q0, q1, q4           \n\t"

                "vld1.32  {q3}, [%[pos3]]        \n\t"
                "vld1.32  {q6}, [%[filter3]]        \n\t"

                "vmla.f32 q0, q2, q5           \n\t"
                "vmla.f32 q0, q3, q6          \n\t"

                "vmov.f32 d1[1],  %[zero]         \n\t"

                "vadd.f32  d4, d0, d1           \n\t"
                "vadd.f32  s10, s8, s9            \n\t"
                "vst1.32 {d5[0]},[%[output_ptr]]    \n\t"
                :
                : [input_data] "r"(input_data), [pos1] "r"(pos1),
                  [pos2] "r"(pos2), [pos3] "r"(pos3), [filter1] "r"(filter1),
                  [filter2] "r"(filter2), [filter3] "r"(filter3),
                  [output_ptr] "r"(output_ptr), [zero] "r"(zero)
                : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6");
#endif  // __aarch64__
#else

#endif  // __ARM_NEON
          }
        }
      }
      input_data += input_channel_stride;
      output_data += output_channel_stride;
      filter_data += filter_channel_stride;
    }
    input_data += input_batch_stride;
    output_data += output_batch_stride;
  }
}

void DepthwiseConv3x3s1p1(const framework::Tensor *input,
                          const framework::Tensor *filter,
                          framework::Tensor *output, framework::Tensor *bias,
                          bool if_bias) {
#if __ARM_NEON
  const float *input_data = input->data<float>();
  const float *filter_data = filter->data<float>();
  float *output_data = output->mutable_data<float>();
  const float *bias_data;
  if (if_bias) {
    bias_data = bias->data<float>();
  }

  const int h = static_cast<int>(input->dims()[2]);
  const int w = static_cast<int>(input->dims()[3]);
  //  const int l = h;
  const int batch_size = static_cast<int>(input->dims()[0]);
  const int c = static_cast<int>(input->dims()[1]);
  const int hxw = h * w;
  float32x4_t vbias = vdupq_n_f32(0.0);
  for (int b = 0; b < batch_size; ++b) {
    const float *filter_data_tmp = filter_data;

    for (int j = 0; j < c; ++j) {
      if (if_bias) {
        vbias = vdupq_n_f32(bias_data[j]);
      }

      int w_mid = w - 2;  // l=1->l_mid=-1,l=2->l_mid=0
      float w00 = filter_data_tmp[0];
      float w01 = filter_data_tmp[1];
      float w02 = filter_data_tmp[2];
      float w10 = filter_data_tmp[3];
      float w11 = filter_data_tmp[4];
      float w12 = filter_data_tmp[5];
      float w20 = filter_data_tmp[6];
      float w21 = filter_data_tmp[7];
      float w22 = filter_data_tmp[8];

      output_data[0] = w11 * input_data[0] + w12 * input_data[1] +
                       w21 * input_data[w] + w22 * input_data[w + 1];
      output_data[w - 1] = w10 * input_data[w - 2] + w11 * input_data[w - 1] +
                           w20 * input_data[2 * w - 2] +
                           w21 * input_data[2 * w - 1];
      output_data[(h - 1) * w] =
          w01 * input_data[(h - 2) * w] + w02 * input_data[(h - 2) * w + 1] +
          w11 * input_data[(h - 1) * w] + w12 * input_data[(h - 1) * w + 1];
      output_data[h * w - 1] =
          w00 * input_data[h * w - w - 2] + w01 * input_data[h * w - w - 1] +
          w10 * input_data[h * w - 2] + w11 * input_data[h * w - 1];
      if (if_bias) {
        output_data[0] += bias_data[j];
        output_data[w - 1] += bias_data[j];
        output_data[(h - 1) * w] += bias_data[j];
        output_data[h * w - 1] += bias_data[j];
      }

      for (int i = 1; i < h - 1; ++i) {
        output_data[i * w] =
            w01 * input_data[i * w - w] + w02 * input_data[i * w - w + 1] +
            w11 * input_data[i * w] + w12 * input_data[i * w + 1] +
            w21 * input_data[i * w + w] + w22 * input_data[i * w + w + 1];

        output_data[i * w + w - 1] = w00 * input_data[i * w + w - 1 - w - 1] +
                                     w01 * input_data[i * w + w - 1 - w] +
                                     w10 * input_data[i * w + w - 1 - 1] +
                                     w11 * input_data[i * w + w - 1] +
                                     w20 * input_data[i * w + w - 1 + w - 1] +
                                     w21 * input_data[i * w + w - 1 + w];
        if (if_bias) {
          output_data[i * w] += bias_data[j];
          output_data[i * w + w - 1] += bias_data[j];
        }
      }

      // top 1 row and bottom 1 row
      const float *input_tmp = input_data;

      float32x4_t in0, in1, in2, in3, in4, in5, in6, in7, tmp0, tmp1, tmp2,
          tmp3, tmp4, tmp5, out0;
      in0 = vld1q_f32(input_tmp);
      in2 = vld1q_f32(input_tmp + w);
      const float *input_tmp_end = input_tmp + (h - 2) * w;
      in4 = vld1q_f32(input_tmp_end);
      in6 = vld1q_f32(input_tmp_end + w);
      int c_mid = w_mid;
      auto output_ptr = output_data + 1;
      for (; c_mid > 3; c_mid -= 4) {
        in1 = vld1q_f32(input_tmp + 4);
        in3 = vld1q_f32(input_tmp + w + 4);

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

        vst1q_f32(output_ptr, out0);

        in5 = vld1q_f32(input_tmp_end + 4);
        in7 = vld1q_f32(input_tmp_end + w + 4);

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

        vst1q_f32(output_ptr + (h - 1) * w, out0);

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
      float32x4_t pad0 = vdupq_n_f32(input_data[w - 1]);
      float32x4_t pad1 = vdupq_n_f32(input_data[2 * w - 1]);

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
      float32x4_t pad2 = vdupq_n_f32(input_data[h * w - 1 - w]);
      float32x4_t pad3 = vdupq_n_f32(input_data[h * w - 1]);

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

      for (int i = 0; i < c_mid; ++i) {
        if (i == 0) {
          vst1q_lane_f32(output_ptr + (h - 1) * w + i, out0, 0);
        }
        if (i == 1) {
          vst1q_lane_f32(output_ptr + (h - 1) * w + i, out0, 1);
        }
        if (i == 2) {
          vst1q_lane_f32(output_ptr + (h - 1) * w + i, out0, 2);
        }
      }
      // mid

      for (int i = 0; i < h - 2; ++i) {
        auto output_ptr = output_data + (i + 1) * w + 1;
        input_tmp = input_data + i * w;
        auto in0_tmp = vld1q_f32(input_tmp);
        auto in2_tmp = vld1q_f32(input_tmp + w);
        auto in4_tmp = vld1q_f32(input_tmp + w + w);
        c_mid = w_mid;
        for (; c_mid > 3; c_mid -= 4) {
          auto in1_tmp = vld1q_f32(input_tmp + 4);
          auto in3_tmp = vld1q_f32(input_tmp + w + 4);
          auto in5_tmp = vld1q_f32(input_tmp + w + w + 4);

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

          vst1q_f32(output_ptr, out0);

          output_ptr += 4;
          input_tmp += 4;
          in0_tmp = in1_tmp;
          in2_tmp = in3_tmp;
          in4_tmp = in5_tmp;
        }

        float32x4_t pad0 = vdupq_n_f32(input_data[i * w + w - 1]);
        float32x4_t pad1 = vdupq_n_f32(input_data[i * w + w - 1 + w]);
        float32x4_t pad2 = vdupq_n_f32(input_data[i * w + w - 1 + w + w]);

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
#endif
}

void DepthwiseConvAddBNRelu3x3s1p1(const framework::Tensor *input,
                                   const framework::Tensor *filter,
                                   framework::Tensor *output,
                                   const framework::Tensor *new_scale,
                                   const framework::Tensor *new_bias,
                                   bool if_relu) {
#if __ARM_NEON
  const float *input_data = input->data<float>();
  const float *filter_data = filter->data<float>();
  float *output_data = output->data<float>();
  const float *newscale_data = new_scale->data<float>();
  const float *newbias_data = new_bias->data<float>();

  const int batch_size = static_cast<int>(input->dims()[0]);
  const int input_channel = static_cast<int>(input->dims()[1]);

  const int input_height = static_cast<int>(input->dims()[2]);
  const int input_width = static_cast<int>(input->dims()[3]);
  const int output_height = static_cast<int>(output->dims()[2]);
  const int output_width = static_cast<int>(output->dims()[3]);

  const int hxw = input_height * input_width;

  //  const int l = input_height;
  const int h = input_height;
  const int w = input_width;
  float32x4_t vzero = vdupq_n_f32(0);

  for (int b = 0; b < batch_size; b++) {
#pragma omp parallel for
    for (int c = 0; c < input_channel; c++) {
      const float *filter_data = filter->data<float>() + c * 9;
      const float *input_data = input->data<float>() + c * hxw;
      float *output_data = output->data<float>() + c * hxw;
      float32x4_t vnewbias = vdupq_n_f32(newbias_data[c]);
      float32x4_t vnewscale = vdupq_n_f32(newscale_data[c]);

      float w00 = filter_data[0];
      float w01 = filter_data[1];
      float w02 = filter_data[2];
      float w10 = filter_data[3];
      float w11 = filter_data[4];
      float w12 = filter_data[5];
      float w20 = filter_data[6];
      float w21 = filter_data[7];
      float w22 = filter_data[8];

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

          out0 = vmulq_n_f32(in0, w00);
          out0 = vmlaq_n_f32(out0, tmp0, w01);
          out0 = vmlaq_n_f32(out0, tmp1, w02);
          out0 = vmlaq_n_f32(out0, in2, w10);
          out0 = vmlaq_n_f32(out0, tmp2, w11);
          out0 = vmlaq_n_f32(out0, tmp3, w12);
          out0 = vmlaq_n_f32(out0, in4, w20);
          out0 = vmlaq_n_f32(out0, tmp4, w21);
          out0 = vmlaq_n_f32(out0, tmp5, w22);

          out0 = vmlaq_f32(vnewbias, vnewscale, out0);
          if (if_relu) {
            out0 = vmaxq_f32(out0, vzero);
          }
          vst1q_f32(output_ptr, out0);
        }
        int m;
        for (m = 1; (m + 3) < output_width - 1; m = m + 4) {
        }

        for (int j = m; j < output_width - 1; j++) {
          output_data[i * output_width + j] =
              input_data[(i - 1) * input_width + j - 1] * w00 +
              input_data[(i - 1) * input_width + j] * w01 +
              input_data[(i - 1) * input_width + j + 1] * w02 +
              input_data[(i)*input_width + j - 1] * w10 +
              input_data[(i)*input_width + j] * w11 +
              input_data[(i)*input_width + j + 1] * w12 +
              input_data[(i + 1) * input_width + j - 1] * w20 +
              input_data[(i + 1) * input_width + j] * w21 +
              input_data[(i + 1) * input_width + j + 1] * w22;
          output_data[i * output_width + j] =
              newscale_data[c] * output_data[i * output_width + j] +
              newbias_data[c];
          if (if_relu) {
            output_data[i * output_width + j] =
                output_data[i * output_width + j] < 0
                    ? 0
                    : output_data[i * output_width + j];
          }
        }
      }

      output_data[0] = w11 * input_data[0] + w12 * input_data[1] +
                       w21 * input_data[w] + w22 * input_data[w + 1];
      output_data[w - 1] = w10 * input_data[w - 2] + w11 * input_data[w - 1] +
                           w20 * input_data[2 * w - 2] +
                           w21 * input_data[2 * w - 1];
      output_data[(h - 1) * w] =
          w01 * input_data[(h - 2) * w] + w02 * input_data[(h - 2) * w + 1] +
          w11 * input_data[(h - 1) * w] + w12 * input_data[(h - 1) * w + 1];
      output_data[h * w - 1] =
          w00 * input_data[h * w - w - 2] + w01 * input_data[h * w - w - 1] +
          w10 * input_data[h * w - 2] + w11 * input_data[h * w - 1];
      output_data[0] = output_data[0] * newscale_data[c] + newbias_data[c];
      output_data[w - 1] =
          output_data[w - 1] * newscale_data[c] + newbias_data[c];
      output_data[(h - 1) * w] =
          output_data[(h - 1) * w] * newscale_data[c] + newbias_data[c];
      output_data[h * w - 1] =
          output_data[h * w - 1] * newscale_data[c] + newbias_data[c];

      if (if_relu) {
        output_data[0] = output_data[0] < 0 ? 0 : output_data[0];
        output_data[w - 1] = output_data[w - 1] < 0 ? 0 : output_data[w - 1];
        output_data[(h - 1) * w] =
            output_data[(h - 1) * w] < 0 ? 0 : output_data[(h - 1) * w];
        output_data[h * w - 1] =
            output_data[h * w - 1] < 0 ? 0 : output_data[h * w - 1];
      }
      for (int i = 1; i < h - 1; ++i) {
        output_data[i * w] =
            w01 * input_data[i * w - w] + w02 * input_data[i * w - w + 1] +
            w11 * input_data[i * w] + w12 * input_data[i * w + 1] +
            w21 * input_data[i * w + w] + w22 * input_data[i * w + w + 1];

        output_data[i * w + w - 1] = w00 * input_data[i * w + w - 1 - w - 1] +
                                     w01 * input_data[i * w + w - 1 - w] +
                                     w10 * input_data[i * w + w - 1 - 1] +
                                     w11 * input_data[i * w + w - 1] +
                                     w20 * input_data[i * w + w - 1 + w - 1] +
                                     w21 * input_data[i * w + w - 1 + w];
        output_data[i * w] =
            output_data[i * w] * newscale_data[c] + newbias_data[c];
        output_data[i * w + w - 1] =
            output_data[i * w + w - 1] * newscale_data[c] + newbias_data[c];

        if (if_relu) {
          output_data[i * w] = output_data[i * w] < 0 ? 0 : output_data[i * w];
          output_data[i * w + w - 1] =
              output_data[i * w + w - 1] < 0 ? 0 : output_data[i * w + w - 1];
        }
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
        out0 = vmulq_n_f32(in0, w10);
        out0 = vmlaq_n_f32(out0, tmp0, w11);
        out0 = vmlaq_n_f32(out0, tmp1, w12);
        out0 = vmlaq_n_f32(out0, in2, w20);
        out0 = vmlaq_n_f32(out0, tmp2, w21);
        out0 = vmlaq_n_f32(out0, tmp3, w22);
        out0 = vmlaq_f32(vnewbias, vnewscale, out0);
        if (if_relu) {
          out0 = vmaxq_f32(out0, vzero);
        }
        vst1q_f32(output_ptr, out0);
      }

      for (m = 1; (m + 3) < output_width - 1; m += 4) {
      }
      for (int j = m; j < output_width - 1; j++) {
        output_data[j] = input_data[j - 1] * w10 + input_data[j] * w11 +
                         input_data[j + 1] * w12 +
                         input_data[input_width + j - 1] * w20 +
                         input_data[input_width + j] * w21 +
                         input_data[input_width + j + 1] * w22;
        output_data[j] = output_data[j] * newscale_data[c] + newbias_data[c];

        if (if_relu) {
          output_data[j] = output_data[j] < 0 ? 0 : output_data[j];
        }
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
        out0 = vmulq_n_f32(in0, w00);
        out0 = vmlaq_n_f32(out0, tmp0, w01);
        out0 = vmlaq_n_f32(out0, tmp1, w02);
        out0 = vmlaq_n_f32(out0, in2, w10);
        out0 = vmlaq_n_f32(out0, tmp2, w11);
        out0 = vmlaq_n_f32(out0, tmp3, w12);
        out0 = vmlaq_f32(vnewbias, vnewscale, out0);
        if (if_relu) {
          out0 = vmaxq_f32(out0, vzero);
        }
        vst1q_f32(output_ptr, out0);
      }
      for (m = 1; (m + 3) < output_width - 1; m = m + 4) {
      }
      for (int j = m; j < output_width - 1; j++) {
        output_data[(output_height - 1) * input_width + j] =
            input_data[(output_height - 2) * input_width + j - 1] * w00 +
            input_data[(output_height - 2) * input_width + j] * w01 +
            input_data[(output_height - 2) * input_width + j + 1] * w02 +
            input_data[(output_height - 1) * input_width + j - 1] * w10 +
            input_data[(output_height - 1) * input_width + j] * w11 +
            input_data[(output_height - 1) * input_width + j + 1] * w12;
        output_data[(output_height - 1) * output_width + j] =
            output_data[(output_height - 1) * output_width + j] *
                newscale_data[c] +
            newbias_data[c];

        if (if_relu) {
          output_data[(output_height - 1) * output_width + j] =
              output_data[(output_height - 1) * output_width + j] < 0
                  ? 0
                  : output_data[(output_height - 1) * output_width + j];
        }
      }
    }
  }

    /*
        const float *input_data = input->data<float>();
        const float *filter_data = filter->data<float>();
        float *output_data = output->data<float>();
        const float *newscale_data = new_scale->data<float>();
        const float *newbias_data = new_bias->data<float>();

        const int h = static_cast<int>(input->dims()[2]);
        const int w = static_cast<int>(input->dims()[3]);
//        const int l = h;

        const int batch_size = static_cast<int>(input->dims()[0]);
        const int c = static_cast<int>(input->dims()[1]);
        const int hxw = h * w;
        float32x4_t vnewbias = vdupq_n_f32(0.0);
        float32x4_t vnewscale = vdupq_n_f32(1.0);
        float32x4_t vzero = vdupq_n_f32(0);

        for (int b = 0; b < batch_size; ++b) {
          const float *filter_data_tmp = filter_data;

          for (int j = 0; j < c; ++j) {
            vnewbias = vdupq_n_f32(newbias_data[j]);
            vnewscale = vdupq_n_f32(newscale_data[j]);

            int w_mid = w - 2;  // l=1->l_mid=-1,l=2->l_mid=0
            float w00 = filter_data_tmp[0];
            float w01 = filter_data_tmp[1];
            float w02 = filter_data_tmp[2];
            float w10 = filter_data_tmp[3];
            float w11 = filter_data_tmp[4];
            float w12 = filter_data_tmp[5];
            float w20 = filter_data_tmp[6];
            float w21 = filter_data_tmp[7];
            float w22 = filter_data_tmp[8];

            output_data[0] = w11 * input_data[0] + w12 * input_data[1] +
                             w21 * input_data[w] + w22 * input_data[w + 1];

            output_data[w - 1] = w10 * input_data[w - 2] + w11 * input_data[w -
       1] + w20 * input_data[2 * w - 2] + w21 * input_data[2 * w - 1];

            output_data[(h - 1) * w] =
                w01 * input_data[(h - 2) * w] + w02 * input_data[(h - 2) * w +
       1] + w11 * input_data[(h - 1) * w] + w12 * input_data[(h - 1) * w + 1];
            output_data[h * w - 1] = w00 * input_data[h*w-w-2] +
                                     w01 * input_data[h*w-w-1] +
                                     w10 * input_data[h * w - 2] +
                                     w11 * input_data[h * w - 1];
            output_data[0] = output_data[0] * newscale_data[j] +
       newbias_data[j]; output_data[w - 1] = output_data[w - 1] *
       newscale_data[j] + newbias_data[j]; output_data[(h - 1) * w] =
                output_data[(h - 1) * w] * newscale_data[j] + newbias_data[j];
            output_data[h * w - 1] =
                output_data[h * w - 1] * newscale_data[j] + newbias_data[j];

            if (if_relu) {
              output_data[0] = output_data[0] < 0 ? 0 : output_data[0];
              output_data[w - 1] = output_data[w - 1] < 0 ? 0 : output_data[w -
       1]; output_data[(h - 1) * w] = output_data[(h - 1) * w] < 0 ? 0 :
       output_data[(h - 1) * w]; output_data[h * w - 1] = output_data[h * w - 1]
       < 0 ? 0 : output_data[h * w - 1];
            }
            for (int i = 1; i < h - 1; ++i) {
              output_data[i * w] =
                  w01 * input_data[i * w - w] + w02 * input_data[i * w - w + 1]
       + w11 * input_data[i * w] + w12 * input_data[i * w + 1] + w21 *
       input_data[i * w + w] + w22 * input_data[i * w + w + 1]; output_data[i *
       w + w - 1] = w00 * input_data[i * w + w - 1 - w - 1] + w01 * input_data[i
       * w + w - 1 - w] + w10 * input_data[i * w + w - 1 - 1] + w11 *
       input_data[i * w + w - 1] + w20 * input_data[i * w + w - 1 + w - 1] + w21
       * input_data[i * w + w - 1 + w]; output_data[i * w] = output_data[i * w]
       * newscale_data[j] + newbias_data[j]; output_data[i * w + w - 1] =
                  output_data[i * w + w - 1] * newscale_data[j] +
       newbias_data[j];

              if (if_relu) {
                output_data[i * w] = output_data[i * w] < 0 ? 0 : output_data[i
       * w]; output_data[i * w + w - 1] = output_data[i * w + w - 1] < 0 ? 0 :
       output_data[i * w + w - 1];
              }
            }

            // top 1 row and bottom 1 row
            const float *input_tmp = input_data;

            float32x4_t in0, in1, in2, in3, in4, in5, in6, in7, tmp0, tmp1,
       tmp2, tmp3, tmp4, tmp5, out0; in0 = vld1q_f32(input_tmp); in2 =
       vld1q_f32(input_tmp + w); const float *input_tmp_end = input_tmp + (h -
       2) * w; in4 = vld1q_f32(input_tmp_end); in6 = vld1q_f32(input_tmp_end +
       w); int c_mid = w_mid; auto output_ptr = output_data + 1; for (; c_mid >
       3; c_mid -= 4) { in1 = vld1q_f32(input_tmp + 4); in3 =
       vld1q_f32(input_tmp + w + 4);

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
              out0 = vmlaq_f32(vnewbias, vnewscale, out0);
              if (if_relu) {
                out0 = vmaxq_f32(out0, vzero);
              }
              vst1q_f32(output_ptr, out0);

              in5 = vld1q_f32(input_tmp_end + 4);
              in7 = vld1q_f32(input_tmp_end + w + 4);

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
              out0 = vmlaq_f32(vnewbias, vnewscale, out0);
              if (if_relu) {
                out0 = vmaxq_f32(out0, vzero);
              }
              vst1q_f32(output_ptr + (h - 1) * w, out0);

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
            float32x4_t pad0 = vdupq_n_f32(input_data[w - 1]);
            float32x4_t pad1 = vdupq_n_f32(input_data[2 * w - 1]);

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
            float32x4_t pad2 = vdupq_n_f32(input_data[h * w - 1 - w]);
            float32x4_t pad3 = vdupq_n_f32(input_data[h * w - 1]);

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
            out0 = vmlaq_f32(vnewbias, vnewscale, out0);
            if (if_relu) {
              out0 = vmaxq_f32(out0, vzero);
            }
            for (int i = 0; i < c_mid; ++i) {
              if (i == 0) {
                vst1q_lane_f32(output_ptr + (h - 1) * w + i, out0, 0);
              }
              if (i == 1) {
                vst1q_lane_f32(output_ptr + (h - 1) * w + i, out0, 1);
              }
              if (i == 2) {
                vst1q_lane_f32(output_ptr + (h - 1) * w + i, out0, 2);
              }
            }
            // mid


            for (int i = 0; i < h - 2; ++i) {
              auto output_ptr = output_data + (i + 1) * w + 1;
              input_tmp = input_data + i * w;
              auto in0_tmp = vld1q_f32(input_tmp);
              auto in2_tmp = vld1q_f32(input_tmp + w);
              auto in4_tmp = vld1q_f32(input_tmp + w + w);
              c_mid = w_mid;
              for (; c_mid > 3; c_mid -= 4) {
                auto in1_tmp = vld1q_f32(input_tmp + 4);
                auto in3_tmp = vld1q_f32(input_tmp + w + 4);
                auto in5_tmp = vld1q_f32(input_tmp + w + w + 4);

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

              float32x4_t pad0 = vdupq_n_f32(input_data[i * w + w - 1]);
              float32x4_t pad1 = vdupq_n_f32(input_data[i * w + w - 1 + w]);
              float32x4_t pad2 = vdupq_n_f32(input_data[i * w + w - 1 + w + w]);

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
    */

#endif
}

/// w!=h not fix
void DepthwiseConvAddBNRelu3x3s2p1(const framework::Tensor *input,
                                   const framework::Tensor *filter,
                                   framework::Tensor *output,
                                   const framework::Tensor *new_scale,
                                   const framework::Tensor *new_bias,
                                   bool if_relu) {
#if __ARM_NEON

  const int batch_size = input->dims()[0];

  const int input_height = input->dims()[2];

  const int input_width = input->dims()[3];

  const int output_channels = output->dims()[1];

  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  const int _kernel_size = 3;
  const int stride_height = 2;
  const int stride_width = 2;
  const int padding_height = 1;
  const int padding_width = 1;
  const float zero = 0;
  const int input_channel_stride = input_height * input_width;
  const int output_channel_stride = output_height * output_width;
  const int filter_channel_stride = 9;
  const float *newscale_data = new_scale->data<float>();
  const float *newbias_data = new_bias->data<float>();

  const float *input_data = input->data<float>();
  const float *filter_data = filter->data<float>();

  float *output_data = output->mutable_data<float>();

  const int input_batch_stride = output_channels * input_channel_stride;
  const int output_batch_stride = output_channels * output_channel_stride;
  const int filter_batch_stride = output_channels * output_channel_stride;
  const float *pos1, *pos2, *pos3, *filter1, *filter2, *filter3, *output_ptr;
  int hstart, wstart, hend, wend;
  float result;
  for (int i = 0; i < batch_size; ++i) {
    for (int c = 0; c < output_channels; ++c) {
      filter1 = filter_data;
      filter2 = filter1 + 3;
      filter3 = filter2 + 3;

      for (int ph = 0; ph < output_height; ph++) {
        for (int pw = 0; pw < output_width; pw++) {
          hstart = ph * stride_height - padding_height;
          wstart = pw * stride_width - padding_width;
          hend = std::min(hstart + _kernel_size, input_height + padding_height);
          wend = std::min(wstart + _kernel_size, input_width + padding_width);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          hend = std::min(hend, input_height);
          wend = std::min(wend, input_width);
          pos1 = input_data + hstart * input_width + wstart;
          pos2 = input_data + (hstart + 1) * input_width + wstart;
          pos3 = input_data + (hstart + 2) * input_width + wstart;
          output_ptr = output_data + ph * output_width + pw;

          if (hend - hstart != 3 || wend - wstart != 3) {
            result = 0;
            float fake_input[9] = {0};
            if (hstart == 0 && wstart == 0) {
              // 左上角
              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (j >= 3 - hend && k >= 3 - wend) {
                    fake_input[3 * j + k] =
                        input_data[(j - (3 - hend)) * input_width + k -
                                   (3 - wend)];
                  }
                }
              }
            } else if (hstart == 0 && wend == input_width) {
              // 右上角
              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (j >= 3 - hend && k <= input_width - wstart - 1) {
                    fake_input[3 * j + k] =
                        input_data[(j - (3 - hend)) * input_width + k + wstart];
                  }
                }
              }

            } else if (hend == input_height && wstart == 0) {
              // 左下角

              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (j <= input_height - 1 - hstart && k >= 3 - wend) {
                    fake_input[3 * j + k] =
                        input_data[(j + hstart) * input_width + k - (3 - wend)];
                  }
                }
              }
            } else if (hend == input_height && wend == input_width) {
              // 右下角
              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (j <= input_height - hstart - 1 &&
                      k <= input_width - wstart - 1) {
                    fake_input[3 * j + k] =
                        input_data[(j + hstart) * input_width + k + wstart];
                  }
                }
              }
            } else if (hstart == 0) {
              // 顶部
              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (j >= 3 - hend) {
                    fake_input[3 * j + k] =
                        input_data[(j - (3 - hend)) * input_width + k + wstart];
                  }
                }
              }

            } else if (hend == input_height) {
              // 底部
              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (j <= input_height - hstart - 1) {
                    fake_input[3 * j + k] =
                        input_data[(j + hstart) * input_width + k + wstart];
                  }
                }
              }

            } else if (wstart == 0) {
              // 左侧
              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (k >= 3 - wend) {
                    fake_input[3 * j + k] =
                        input_data[(j + hstart) * input_width +
                                   (k - (3 - wend))];
                  }
                }
              }

            } else if (wend == input_width) {
              // 右侧
              for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                  if (k <= input_width - wstart - 1) {
                    fake_input[3 * j + k] =
                        input_data[(j + hstart) * input_width + k + wstart];
                  }
                }
              }
            }
            for (int l = 0; l < 9; ++l) {
              result += fake_input[l] * filter1[l];
            }
            output_data[ph * output_width + pw] =
                newscale_data[c] * result + newbias_data[c];

            if (if_relu) {
              output_data[ph * output_width + pw] =
                  output_data[ph * output_width + pw] < 0
                      ? 0
                      : output_data[ph * output_width + pw];
            }
          } else {
            const float32x4_t data1 = vld1q_f32(pos1);
            const float32x4_t data2 = vld1q_f32(pos2);
            const float32x4_t data3 = vld1q_f32(pos3);

            const float32x4_t v_filter1 = vld1q_f32(filter1);
            const float32x4_t v_filter2 = vld1q_f32(filter2);
            const float32x4_t v_filter3 = vld1q_f32(filter3);
            float32x4_t mula = vmulq_f32(data1, v_filter1);
            mula = vmlaq_f32(mula, data2, v_filter2);
            mula = vmlaq_f32(mula, data3, v_filter3);
            float32x2_t res = vpadd_f32(
                vget_high_f32(vsetq_lane_f32(0, mula, 3)), vget_low_f32(mula));
            res = vpadd_f32(res, res);
            output_data[ph * output_width + pw] =
                vget_lane_f32(res, 0) * newscale_data[c] + newbias_data[c];

            if (if_relu) {
              output_data[ph * output_width + pw] =
                  output_data[ph * output_width + pw] < 0
                      ? 0
                      : output_data[ph * output_width + pw];
            }
          }
        }
      }
      input_data += input_channel_stride;
      output_data += output_channel_stride;
      filter_data += filter_channel_stride;
    }
    input_data += input_batch_stride;
    output_data += output_batch_stride;
  }
#endif
}

void DepthwiseConv3x3s2p1v2(const framework::Tensor *input,
                            const framework::Tensor *filter,
                            framework::Tensor *output, framework::Tensor bias,
                            bool if_bias) {
#if __ARM_NEON
  const float *input_data = input->data<float>();
  const float *filter_data = filter->data<float>();
  float *output_data = output->data<float>();
  const float *bias_data = bias.data<float>();

  const int in_h = static_cast<int>(input->dims()[2]);
  const int in_w = static_cast<int>(input->dims()[3]);
  const int out_h = static_cast<int>(output->dims()[2]);
  const int out_w = static_cast<int>(output->dims()[3]);
  const int out_l = out_h;
  const int in_l = in_h;
  const int inhxw = in_h * in_w;
  const int outhxw = out_h * out_w;
  /// todo : fix if_pad when w != h
  const int if_pad_r = in_w - 1 == (out_w - 1) * 2 ? 1 : 0;
  const int if_pad_b = in_h - 1 == (out_h - 1) * 2 ? 1 : 0;
  const int batch_size = static_cast<int>(input->dims()[0]);
  const int c = static_cast<int>(input->dims()[1]);
  const float *input_row_ptr;
  float *output_row_ptr;

  const int w_times = (out_w - 2) / 3;

  float32x4_t vbias = vdupq_n_f32(0.0);

  float32x4x2_t input_buff_mid{}, input_buff_bottom[w_times + 1];
  float32x4_t elewise_res0, elewise_res1, elewise_res2, res3;
  int out2in_mid;
  float32x4_t zero = vdupq_n_f32(0.0);
  for (int b = batch_size; b > 0; --b) {
    const float *filter_data_tmp = filter_data;
    for (int j = 0; j < c; ++j) {
      auto output_data_tmp = output_data + j * out_h * out_w;
      auto input_data_tmp = input_data + j * in_h * in_w;
      auto input_const = input_data_tmp;

      if (if_bias) {
        vbias = vdupq_n_f32(bias_data[j]);
      }

      float w00 = filter_data_tmp[0];
      float w01 = filter_data_tmp[1];
      float w02 = filter_data_tmp[2];
      float w10 = filter_data_tmp[3];
      float w11 = filter_data_tmp[4];
      float w12 = filter_data_tmp[5];
      float w20 = filter_data_tmp[6];
      float w21 = filter_data_tmp[7];
      float w22 = filter_data_tmp[8];

      int h_mid = 0;

      for (; h_mid < out_h - 1; h_mid++) {
        input_row_ptr = input_data_tmp + 1 + h_mid * 2 * in_w;
        output_row_ptr = output_data_tmp + 1 + h_mid * out_w;

        for (int w4 = 0; w4 < w_times + 1; w4++) {
          if (h_mid == 0) {
            elewise_res1 = zero;
            elewise_res0 = zero;
            elewise_res2 = zero;
          } else {
            elewise_res1 = vmulq_n_f32(input_buff_bottom[w4].val[1], w01);
            elewise_res0 = vmulq_n_f32(input_buff_bottom[w4].val[0], w00);
            elewise_res2 = vmulq_n_f32(input_buff_bottom[w4].val[0], w02);
          }
          input_buff_mid = vld2q_f32(input_row_ptr);
          input_buff_bottom[w4] = vld2q_f32(input_row_ptr + in_w);

          elewise_res1 = vmlaq_n_f32(elewise_res1, input_buff_mid.val[1], w11);
          elewise_res0 = vmlaq_n_f32(elewise_res0, input_buff_mid.val[0], w10);
          elewise_res2 = vmlaq_n_f32(elewise_res2, input_buff_mid.val[0], w12);

          elewise_res1 =
              vmlaq_n_f32(elewise_res1, input_buff_bottom[w4].val[1], w21);
          elewise_res0 =
              vmlaq_n_f32(elewise_res0, input_buff_bottom[w4].val[0], w20);
          elewise_res2 =
              vmlaq_n_f32(elewise_res2, input_buff_bottom[w4].val[0], w22);

          res3 = vaddq_f32(vextq_f32(elewise_res2, zero, 1),
                           vaddq_f32(elewise_res0, elewise_res1));
          res3 = vaddq_f32(res3, vbias);
          vst1q_f32(output_row_ptr, res3);

          input_row_ptr += 6;
          output_row_ptr += 3;
        }
      }
      clock();

      input_row_ptr = input_data_tmp + 1 + h_mid * 2 * in_w;
      output_row_ptr = output_data_tmp + 1 + h_mid * out_w;

      for (int w4 = 0; w4 < w_times + 1; w4++) {
        elewise_res1 = vmulq_n_f32(input_buff_bottom[w4].val[1], w01);
        elewise_res0 = vmulq_n_f32(input_buff_bottom[w4].val[0], w00);
        elewise_res2 = vmulq_n_f32(input_buff_bottom[w4].val[0], w02);

        input_buff_mid = vld2q_f32(input_row_ptr);
        input_buff_bottom[w4] = vld2q_f32(input_row_ptr + in_w);

        elewise_res1 = vmlaq_n_f32(elewise_res1, input_buff_mid.val[1], w11);
        elewise_res0 = vmlaq_n_f32(elewise_res0, input_buff_mid.val[0], w10);
        elewise_res2 = vmlaq_n_f32(elewise_res2, input_buff_mid.val[0], w12);

        if (!if_pad_b) {
          elewise_res1 =
              vmlaq_n_f32(elewise_res1, input_buff_bottom[w4].val[1], w21);
          elewise_res0 =
              vmlaq_n_f32(elewise_res0, input_buff_bottom[w4].val[0], w20);
          elewise_res2 =
              vmlaq_n_f32(elewise_res2, input_buff_bottom[w4].val[0], w22);
        }
        res3 = vaddq_f32(vextq_f32(elewise_res2, zero, 1),
                         vaddq_f32(elewise_res0, elewise_res1));
        res3 = vaddq_f32(res3, vbias);

        if ((w4 != w_times)) {
          vst1q_f32(output_row_ptr, res3);
        } else {
          if (out_w - 2 - w_times * 3 == 1) {
            vst1q_lane_f32(output_row_ptr, res3, 0);
          } else if (out_w - 2 - w_times * 3 == 2) {
            vst1q_lane_f32(output_row_ptr, res3, 0);
            vst1q_lane_f32(output_row_ptr + 1, res3, 1);
          }
        }
        input_row_ptr += 6;
        output_row_ptr += 3;
      }

      output_data_tmp[0] = input_const[0] * w11 + input_const[1] * w12 +
                           input_const[in_w] * w21 +
                           input_const[in_w + 1] * w22;

      out2in_mid = (out_w - 1) * 2;
      output_data_tmp[out_w - 1] =
          w10 * input_const[out2in_mid - 1] + w11 * input_const[out2in_mid] +
          w20 * input_const[out2in_mid + in_w - 1] +
          w21 * input_const[out2in_mid + in_w] +
          (1 - if_pad_r) * (w12 * input_const[out2in_mid + 1] +
                            w22 * input_const[out2in_mid + in_w + 1]);

      out2in_mid = (out_h - 1) * 2 * in_w;

      output_data_tmp[out_w * (out_h - 1)] =
          w01 * input_const[out2in_mid - in_w] +
          w02 * input_const[out2in_mid - in_w + 1] +
          w11 * input_const[out2in_mid] + w12 * input_const[out2in_mid + 1] +
          (1 - if_pad_b) * (w21 * input_const[out2in_mid + in_w] +
                            w22 * input_const[out2in_mid + in_w + 1]);
      out2in_mid = (out_h - 1) * 2 * in_w + (out_w - 1) * 2;

      output_data_tmp[out_h * out_w - 1] =
          w00 * input_const[out2in_mid - in_w - 1] +
          w01 * input_const[out2in_mid - in_w] +
          w10 * input_const[out2in_mid - 1] + w11 * input_const[out2in_mid] +
          (1 - if_pad_r) * (w20 * input_const[out2in_mid + in_w - 1] +
                            w21 * input_const[out2in_mid + in_w]) +
          (1 - if_pad_b) * (w02 * input_const[out2in_mid - in_w + 1] +
                            w12 * input_const[out2in_mid + 1]) +
          (1 - if_pad_r) * (1 - if_pad_b) * w22 *
              input_const[out2in_mid + in_w + 1];
      if (if_bias) {
        output_data_tmp[0] += bias_data[j];
        output_data_tmp[out_w - 1] += bias_data[j];
        output_data_tmp[out_w * (out_h - 1)] += bias_data[j];
        output_data_tmp[out_h * out_w - 1] += bias_data[j];
      }
      for (int i = 1; i < out_h - 1; i++) {
        out2in_mid = i * 2 * in_w;
        output_data_tmp[i * out_w] = w01 * input_const[out2in_mid - in_w] +
                                     w02 * input_const[out2in_mid - in_w + 1] +
                                     w11 * input_const[out2in_mid] +
                                     w12 * input_const[out2in_mid + 1] +
                                     w21 * input_const[out2in_mid + in_w] +
                                     w22 * input_const[out2in_mid + in_w + 1];

        out2in_mid = i * 2 * in_w + (out_w - 1) * 2;
        output_data_tmp[i * out_w + out_w - 1] =
            w00 * input_const[out2in_mid - in_w - 1] +
            w01 * input_const[out2in_mid - in_w] +
            w10 * input_const[out2in_mid - 1] + w11 * input_const[out2in_mid] +
            w20 * input_const[out2in_mid + in_w - 1] +
            w21 * input_const[out2in_mid + in_w] +
            (1 - if_pad_r) * (w02 * input_const[out2in_mid - in_w + 1] +
                              w12 * input_const[out2in_mid + 1] +
                              w22 * input_const[out2in_mid + in_w + 1]);
        if (if_bias) {
          output_data_tmp[i * out_w] += bias_data[j];
          output_data_tmp[i * out_w + out_w - 1] += bias_data[j];
        }
      }
      filter_data_tmp += 9;
    }
    input_data += inhxw * c;
    output_data += outhxw * c;
  }
#endif
}

void DepthwiseConvAddBNRelu3x3s2p1v2(const framework::Tensor *input,
                                     const framework::Tensor *filter,
                                     framework::Tensor *output,
                                     const framework::Tensor *new_scale,
                                     const framework::Tensor *new_bias,
                                     bool if_relu) {
#if __ARM_NEON
  // #ifdef _OPENMP
  //  const float *newscale_data = new_scale->data<float>();
  //  const float *newbias_data = new_bias->data<float>();
  //
  //  const int batch_size = static_cast<int>(input->dims()[0]);
  //  const int input_channel = static_cast<int>(input->dims()[1]);
  //
  //  const int input_height = static_cast<int>(input->dims()[2]);
  //  const int input_width = static_cast<int>(input->dims()[3]);
  //  const int output_height = static_cast<int>(output->dims()[2]);
  //  const int output_width = static_cast<int>(output->dims()[3]);
  //  const int inhxw = input_height * input_width;
  //  const int outhxw = output_height * output_width;
  //
  //  float32x4_t zero = vdupq_n_f32(0.0);
  //  for (int b = 0; b < batch_size; b++) {
  //    #pragma omp parallel for
  //    for (int c = 0; c < input_channel; c++) {
  //      const float *filter_data = filter->data<float>() + c * 9;
  //      const float *input_data = input->data<float>() + c * inhxw;
  //      float *output_data = output->data<float>() + c * outhxw;
  //      float32x4_t vnewbias = vdupq_n_f32(newbias_data[c]);
  //      float32x4_t vnewscale = vdupq_n_f32(newscale_data[c]);
  //
  //      float w00 = filter_data[0];
  //      float w01 = filter_data[1];
  //      float w02 = filter_data[2];
  //      float w10 = filter_data[3];
  //      float w11 = filter_data[4];
  //      float w12 = filter_data[5];
  //      float w20 = filter_data[6];
  //      float w21 = filter_data[7];
  //      float w22 = filter_data[8];
  //
  //      int m;
  //      for (m = 1; m < output_width - 2; m = m + 3) {
  //        float *output_ptr = output_data + m;
  //        float32x4x2_t input_buff_mid{}, input_buff_bottom{};
  //        float32x4_t in0, in1, in2, in3, tmp0, tmp1, tmp2, tmp3, out0;
  //        input_buff_mid = vld2q_f32(input_data + (2 * m - 1));
  //        input_buff_bottom = vld2q_f32(input_data + input_width + (2 * m -
  //        1));
  //
  //        in0 = input_buff_mid.val[0];
  //        tmp0 = input_buff_mid.val[1];
  //        tmp1 = vextq_f32(in0, zero, 1);
  //
  //        in2 = input_buff_bottom.val[0];
  //        tmp2 = input_buff_bottom.val[1];
  //        tmp3 = vextq_f32(in2, zero, 1);
  //
  //        out0 = vmulq_n_f32(in0, w10);
  //        out0 = vmlaq_n_f32(out0, tmp0, w11);
  //        out0 = vmlaq_n_f32(out0, tmp1, w12);
  //        out0 = vmlaq_n_f32(out0, in2, w20);
  //        out0 = vmlaq_n_f32(out0, tmp2, w21);
  //        out0 = vmlaq_n_f32(out0, tmp3, w22);
  //        out0 = vmlaq_f32(vnewbias, vnewscale, out0);
  //        if (if_relu) {
  //          out0 = vmaxq_f32(out0, zero);
  //        }
  //        vst1q_lane_f32(output_ptr, out0, 0);
  //        vst1q_lane_f32(output_ptr + 1, out0, 1);
  //        vst1q_lane_f32(output_ptr + 2, out0, 2);
  //      }
  //      for (m = 1; m < output_width - 2; m += 3) {
  //      }
  //      for (int j = m; j < output_width; j++) {
  //        output_data[j] = input_data[2 * j - 1] * w10 + input_data[2 * j] *
  //        w11 +
  //                         input_data[2 * j + 1] * w12 +
  //                         input_data[2 * j - 1 + input_width] * w20 +
  //                         input_data[2 * j + input_width] * w21 +
  //                         input_data[2 * j + 1 + input_width] * w22;
  //        output_data[j] = newscale_data[c] * output_data[j] +
  //        newbias_data[c]; if (if_relu) {
  //          output_data[j] = output_data[j] < 0 ? 0 : output_data[j];
  //        }
  //      }
  //
  //      for (int i = 1; i < output_height; i += 1) {
  //        for (int m = 1; m < output_width - 2; m += 3) {
  //          float *output_ptr = output_data + i * output_width + m;
  //          float32x4x2_t input_buff_top{}, input_buff_mid{},
  //          input_buff_bottom{}; float32x4_t in0, in1, in2, in3, in4, in5,
  //          tmp0, tmp1, tmp2, tmp3,
  //              tmp4, tmp5, out0;
  //          input_buff_top =
  //              vld2q_f32(input_data + (2 * i - 1) * input_width + (2 * m -
  //              1));
  //          input_buff_mid =
  //              vld2q_f32(input_data + (2 * i) * input_width + (2 * m - 1));
  //          input_buff_bottom =
  //              vld2q_f32(input_data + (2 * i + 1) * input_width + (2 * m -
  //              1));
  //
  //          in0 = input_buff_top.val[0];
  //          tmp0 = input_buff_top.val[1];
  //          tmp1 = vextq_f32(in0, zero, 1);
  //
  //          in2 = input_buff_mid.val[0];
  //          tmp2 = input_buff_mid.val[1];
  //          tmp3 = vextq_f32(in2, zero, 1);
  //
  //          in4 = input_buff_bottom.val[0];
  //          tmp4 = input_buff_bottom.val[1];
  //          tmp5 = vextq_f32(in4, zero, 1);
  //
  //          out0 = vmulq_n_f32(in0, w00);
  //          out0 = vmlaq_n_f32(out0, tmp0, w01);
  //          out0 = vmlaq_n_f32(out0, tmp1, w02);
  //          out0 = vmlaq_n_f32(out0, in2, w10);
  //          out0 = vmlaq_n_f32(out0, tmp2, w11);
  //          out0 = vmlaq_n_f32(out0, tmp3, w12);
  //          out0 = vmlaq_n_f32(out0, in4, w20);
  //          out0 = vmlaq_n_f32(out0, tmp4, w21);
  //          out0 = vmlaq_n_f32(out0, tmp5, w22);
  //          out0 = vmlaq_f32(vnewbias, vnewscale, out0);
  //          if (if_relu) {
  //            out0 = vmaxq_f32(out0, zero);
  //          }
  //          vst1q_lane_f32(output_ptr, out0, 0);
  //          vst1q_lane_f32(output_ptr + 1, out0, 1);
  //          vst1q_lane_f32(output_ptr + 2, out0, 2);
  //        }
  //        int m;
  //        for (m = 1; m < output_width - 2; m += 3) {
  //        }
  //        for (int j = m; j < output_width; j++) {
  //          output_data[i * output_width + j] =
  //              input_data[(2 * i - 1) * input_width + 2 * j - 1] * w00 +
  //              input_data[(2 * i - 1) * input_width + 2 * j] * w01 +
  //              input_data[(2 * i - 1) * input_width + 2 * j + 1] * w02 +
  //              input_data[(2 * i) * input_width + 2 * j - 1] * w10 +
  //              input_data[(2 * i) * input_width + 2 * j] * w11 +
  //              input_data[(2 * i) * input_width + 2 * j + 1] * w12 +
  //              input_data[(2 * i + 1) * input_width + 2 * j - 1] * w20 +
  //              input_data[(2 * i + 1) * input_width + 2 * j] * w21 +
  //              input_data[(2 * i + 1) * input_width + 2 * j + 1] * w22;
  //          output_data[i * output_width + j] =
  //              newscale_data[c] * output_data[i * output_width + j] +
  //              newbias_data[c];
  //          if (if_relu) {
  //            output_data[i * output_width + j] =
  //                output_data[i * output_width + j] < 0
  //                    ? 0
  //                    : output_data[i * output_width + j];
  //          }
  //        }
  //      }
  //      output_data[0] = input_data[0] * w11 + input_data[1] * w12 +
  //                       input_data[input_height] * w21 +
  //                       input_data[input_height + 1] * w22;
  //
  //      output_data[0] = newscale_data[c] * output_data[0] + newbias_data[c];
  //      if (if_relu) {
  //        output_data[0] = output_data[0] < 0 ? 0 : output_data[0];
  //      }
  //      for (int i = 1; i < output_height; i++) {
  //        output_data[i * output_width] =
  //            input_data[(2 * i - 1) * input_width] * w01 +
  //            input_data[(2 * i - 1) * input_width + 1] * w02 +
  //            input_data[(2 * i) * input_width] * w11 +
  //            input_data[(2 * i) * input_width + 1] * w12 +
  //            input_data[(2 * i + 1) * input_width] * w21 +
  //            input_data[(2 * i + 1) * input_width + 1] * w22;
  //
  //        output_data[i * output_width] =
  //            newscale_data[c] * output_data[i * output_width] +
  //            newbias_data[c];
  //        if (if_relu) {
  //          output_data[i * output_width] = output_data[i * output_width] < 0
  //                                              ? 0
  //                                              : output_data[i *
  //                                              output_width];
  //        }
  //      }
  //    }
  //  }
  //
  // #else

  const float *input_data = input->data<float>();
  const float *filter_data = filter->data<float>();
  float *output_data = output->data<float>();
  const float *newscale_data = new_scale->data<float>();
  const float *newbias_data = new_bias->data<float>();

  const int in_h = static_cast<int>(input->dims()[2]);
  const int in_w = static_cast<int>(input->dims()[3]);
  const int out_h = static_cast<int>(output->dims()[2]);
  const int out_w = static_cast<int>(output->dims()[3]);
  //  const int out_l = out_h;
  //  const int in_l = in_h;
  const int inhxw = in_h * in_w;
  const int outhxw = out_h * out_w;
  /// todo : fix if_pad when w != h
  const int if_pad_r = in_w - 1 == (out_w - 1) * 2 ? 1 : 0;
  const int if_pad_b = in_h - 1 == (out_h - 1) * 2 ? 1 : 0;
  const int batch_size = static_cast<int>(input->dims()[0]);
  const int c = static_cast<int>(input->dims()[1]);
  const int w_times = (out_w - 2) / 3;
  float32x4_t zero = vdupq_n_f32(0.0);
  for (int b = batch_size; b > 0; --b) {
#pragma omp parallel for
    for (int j = 0; j < c; j++) {
      const float *input_row_ptr;
      float *output_row_ptr;
      float32x4x2_t input_buff_mid{}, input_buff_bottom[w_times + 1];
      float32x4_t elewise_res0, elewise_res1, elewise_res2, res3;
      int out2in_mid;
      float32x4_t vnewbias = vdupq_n_f32(0.0);
      float32x4_t vnewscale = vdupq_n_f32(1.0);
      auto output_data_tmp = output_data + j * out_h * out_w;
      auto input_data_tmp = input_data + j * in_h * in_w;
      auto input_const = input_data_tmp;
      const float *filter_data_tmp = filter_data + 9 * j;
      vnewbias = vdupq_n_f32(newbias_data[j]);
      vnewscale = vdupq_n_f32(newscale_data[j]);

      float w00 = filter_data_tmp[0];
      float w01 = filter_data_tmp[1];
      float w02 = filter_data_tmp[2];
      float w10 = filter_data_tmp[3];
      float w11 = filter_data_tmp[4];
      float w12 = filter_data_tmp[5];
      float w20 = filter_data_tmp[6];
      float w21 = filter_data_tmp[7];
      float w22 = filter_data_tmp[8];

      int h_mid = 0;

      for (; h_mid < out_h - 1; h_mid++) {
        input_row_ptr = input_data_tmp + 1 + h_mid * 2 * in_w;
        output_row_ptr = output_data_tmp + 1 + h_mid * out_w;

        for (int w4 = 0; w4 < w_times + 1; w4++) {
          if (h_mid == 0) {
            elewise_res1 = zero;
            elewise_res0 = zero;
            elewise_res2 = zero;
          } else {
            elewise_res1 = vmulq_n_f32(input_buff_bottom[w4].val[1], w01);
            elewise_res0 = vmulq_n_f32(input_buff_bottom[w4].val[0], w00);
            elewise_res2 = vmulq_n_f32(input_buff_bottom[w4].val[0], w02);
          }
          input_buff_mid = vld2q_f32(input_row_ptr);
          input_buff_bottom[w4] = vld2q_f32(input_row_ptr + in_w);

          elewise_res1 = vmlaq_n_f32(elewise_res1, input_buff_mid.val[1], w11);
          elewise_res0 = vmlaq_n_f32(elewise_res0, input_buff_mid.val[0], w10);
          elewise_res2 = vmlaq_n_f32(elewise_res2, input_buff_mid.val[0], w12);

          elewise_res1 =
              vmlaq_n_f32(elewise_res1, input_buff_bottom[w4].val[1], w21);
          elewise_res0 =
              vmlaq_n_f32(elewise_res0, input_buff_bottom[w4].val[0], w20);
          elewise_res2 =
              vmlaq_n_f32(elewise_res2, input_buff_bottom[w4].val[0], w22);

          res3 = vaddq_f32(vextq_f32(elewise_res2, zero, 1),
                           vaddq_f32(elewise_res0, elewise_res1));
          res3 = vmlaq_f32(vnewbias, vnewscale, res3);

          if (if_relu) {
            res3 = vmaxq_f32(res3, zero);
          }
          vst1q_lane_f32(output_row_ptr, res3, 0);
          vst1q_lane_f32(output_row_ptr + 1, res3, 1);
          vst1q_lane_f32(output_row_ptr + 2, res3, 2);

          input_row_ptr += 6;
          output_row_ptr += 3;
        }
      }
      clock();

      input_row_ptr = input_data_tmp + 1 + h_mid * 2 * in_w;
      output_row_ptr = output_data_tmp + 1 + h_mid * out_w;

      for (int w4 = 0; w4 < w_times + 1; w4++) {
        elewise_res1 = vmulq_n_f32(input_buff_bottom[w4].val[1], w01);
        elewise_res0 = vmulq_n_f32(input_buff_bottom[w4].val[0], w00);
        elewise_res2 = vmulq_n_f32(input_buff_bottom[w4].val[0], w02);

        input_buff_mid = vld2q_f32(input_row_ptr);
        input_buff_bottom[w4] = vld2q_f32(input_row_ptr + in_w);

        elewise_res1 = vmlaq_n_f32(elewise_res1, input_buff_mid.val[1], w11);
        elewise_res0 = vmlaq_n_f32(elewise_res0, input_buff_mid.val[0], w10);
        elewise_res2 = vmlaq_n_f32(elewise_res2, input_buff_mid.val[0], w12);

        if (!if_pad_b) {
          elewise_res1 =
              vmlaq_n_f32(elewise_res1, input_buff_bottom[w4].val[1], w21);
          elewise_res0 =
              vmlaq_n_f32(elewise_res0, input_buff_bottom[w4].val[0], w20);
          elewise_res2 =
              vmlaq_n_f32(elewise_res2, input_buff_bottom[w4].val[0], w22);
        }
        res3 = vaddq_f32(vextq_f32(elewise_res2, zero, 1),
                         vaddq_f32(elewise_res0, elewise_res1));
        res3 = vmlaq_f32(vnewbias, vnewscale, res3);

        if (if_relu) {
          res3 = vmaxq_f32(res3, zero);
        }
        if ((w4 != w_times)) {
          vst1q_lane_f32(output_row_ptr, res3, 0);
          vst1q_lane_f32(output_row_ptr + 1, res3, 1);
          vst1q_lane_f32(output_row_ptr + 2, res3, 2);
        } else {
          if (out_w - 2 - w_times * 3 == 1) {
            vst1q_lane_f32(output_row_ptr, res3, 0);
          } else if (out_w - 2 - w_times * 3 == 2) {
            vst1q_lane_f32(output_row_ptr, res3, 0);
            vst1q_lane_f32(output_row_ptr + 1, res3, 1);
          }
        }
        input_row_ptr += 6;
        output_row_ptr += 3;
      }

      output_data_tmp[0] = input_const[0] * w11 + input_const[1] * w12 +
                           input_const[in_w] * w21 +
                           input_const[in_w + 1] * w22;

      out2in_mid = (out_w - 1) * 2;
      output_data_tmp[out_w - 1] =
          w10 * input_const[out2in_mid - 1] + w11 * input_const[out2in_mid] +
          w20 * input_const[out2in_mid + in_w - 1] +
          w21 * input_const[out2in_mid + in_w] +
          (1 - if_pad_r) * (w12 * input_const[out2in_mid + 1] +
                            w22 * input_const[out2in_mid + in_w + 1]);

      out2in_mid = (out_h - 1) * 2 * in_w;

      output_data_tmp[out_w * (out_h - 1)] =
          w01 * input_const[out2in_mid - in_w] +
          w02 * input_const[out2in_mid - in_w + 1] +
          w11 * input_const[out2in_mid] + w12 * input_const[out2in_mid + 1] +
          (1 - if_pad_b) * (w21 * input_const[out2in_mid + in_w] +
                            w22 * input_const[out2in_mid + in_w + 1]);
      out2in_mid = (out_h - 1) * 2 * in_w + (out_w - 1) * 2;

      output_data_tmp[out_h * out_w - 1] =
          w00 * input_const[out2in_mid - in_w - 1] +
          w01 * input_const[out2in_mid - in_w] +
          w10 * input_const[out2in_mid - 1] + w11 * input_const[out2in_mid] +
          (1 - if_pad_r) * (w20 * input_const[out2in_mid + in_w - 1] +
                            w21 * input_const[out2in_mid + in_w]) +
          (1 - if_pad_b) * (w02 * input_const[out2in_mid - in_w + 1] +
                            w12 * input_const[out2in_mid + 1]) +
          (1 - if_pad_r) * (1 - if_pad_b) * w22 *
              input_const[out2in_mid + in_w + 1];
      output_data_tmp[0] =
          output_data_tmp[0] * newscale_data[j] + newbias_data[j];
      output_data_tmp[out_w - 1] =
          output_data_tmp[out_w - 1] * newscale_data[j] + newbias_data[j];
      output_data_tmp[out_w * (out_h - 1)] =
          output_data_tmp[out_w * (out_h - 1)] * newscale_data[j] +
          newbias_data[j];
      output_data_tmp[out_h * out_w - 1] =
          output_data_tmp[out_h * out_w - 1] * newscale_data[j] +
          newbias_data[j];
      if (if_relu) {
        output_data_tmp[0] = output_data_tmp[0] < 0 ? 0 : output_data_tmp[0];
        output_data_tmp[out_w - 1] =
            output_data_tmp[out_w - 1] < 0 ? 0 : output_data_tmp[out_w - 1];
        output_data_tmp[out_w * (out_h - 1)] =
            output_data_tmp[out_w * (out_h - 1)] < 0
                ? 0
                : output_data_tmp[out_w * (out_h - 1)];
        output_data_tmp[out_h * out_w - 1] =
            output_data_tmp[out_h * out_w - 1] < 0
                ? 0
                : output_data_tmp[out_h * out_w - 1];
      }
      for (int i = 1; i < out_h - 1; i++) {
        out2in_mid = i * 2 * in_w;
        output_data_tmp[i * out_w] = w01 * input_const[out2in_mid - in_w] +
                                     w02 * input_const[out2in_mid - in_w + 1] +
                                     w11 * input_const[out2in_mid] +
                                     w12 * input_const[out2in_mid + 1] +
                                     w21 * input_const[out2in_mid + in_w] +
                                     w22 * input_const[out2in_mid + in_w + 1];

        out2in_mid = i * 2 * in_w + (out_w - 1) * 2;
        output_data_tmp[i * out_w + out_w - 1] =
            w00 * input_const[out2in_mid - in_w - 1] +
            w01 * input_const[out2in_mid - in_w] +
            w10 * input_const[out2in_mid - 1] + w11 * input_const[out2in_mid] +
            w20 * input_const[out2in_mid + in_w - 1] +
            w21 * input_const[out2in_mid + in_w] +
            (1 - if_pad_r) * (w02 * input_const[out2in_mid - in_w + 1] +
                              w12 * input_const[out2in_mid + 1] +
                              w22 * input_const[out2in_mid + in_w + 1]);
        output_data_tmp[i * out_w] =
            output_data_tmp[i * out_w] * newscale_data[j] + newbias_data[j];
        output_data_tmp[i * out_w + out_w - 1] =
            output_data_tmp[i * out_w + out_w - 1] * newscale_data[j] +
            newbias_data[j];
        if (if_relu) {
          output_data_tmp[i * out_w] =
              output_data_tmp[i * out_w] < 0 ? 0 : output_data_tmp[i * out_w];
          output_data_tmp[i * out_w + out_w - 1] =
              output_data_tmp[i * out_w + out_w - 1] < 0
                  ? 0
                  : output_data_tmp[i * out_w + out_w - 1];
        }
      }
    }
    input_data += inhxw * c;
    output_data += outhxw * c;
  }
// #endif
#endif
}

void DepthwiseConv3x3s2p0(const framework::Tensor *input,
                          const framework::Tensor *filter,
                          framework::Tensor *output, framework::Tensor bias,
                          bool if_bias) {
#if __ARM_NEON

  const int batch_size = static_cast<int>(input->dims()[0]);
  const int input_channel = static_cast<int>(input->dims()[1]);

  const int input_height = static_cast<int>(input->dims()[2]);
  const int input_width = static_cast<int>(input->dims()[3]);
  const int output_height = static_cast<int>(output->dims()[2]);
  const int output_width = static_cast<int>(output->dims()[3]);
  const int inhxw = input_height * input_width;
  const int outhxw = output_height * output_width;

  float32x4_t zero = vdupq_n_f32(0.0);
  for (int b = 0; b < batch_size; b++) {
#pragma omp parallel for
    for (int c = 0; c < input_channel; c++) {
      const float *filter_data = filter->data<float>() + c * 9;
      const float *input_data = input->data<float>() + c * inhxw;
      const float *bias_data = bias.data<float>() + c;
      float *output_data = output->data<float>() + c * outhxw;
      float w00 = filter_data[0];
      float w01 = filter_data[1];
      float w02 = filter_data[2];
      float w10 = filter_data[3];
      float w11 = filter_data[4];
      float w12 = filter_data[5];
      float w20 = filter_data[6];
      float w21 = filter_data[7];
      float w22 = filter_data[8];
      float32x4_t biasv = vld1q_dup_f32(bias_data);
      for (int i = 0; i < output_height; i += 1) {
        for (int m = 0; m < output_width - 2; m += 3) {
          float *output_ptr = output_data + i * output_width + m;
          float32x4x2_t input_buff_top{}, input_buff_mid{}, input_buff_bottom{};
          float32x4_t in0, in1, in2, in3, in4, in5, tmp0, tmp1, tmp2, tmp3,
              tmp4, tmp5, out0;
          input_buff_top =
              vld2q_f32(input_data + (2 * i) * input_width + (2 * m));
          input_buff_mid =
              vld2q_f32(input_data + (2 * i + 1) * input_width + (2 * m));
          input_buff_bottom =
              vld2q_f32(input_data + (2 * i + 2) * input_width + (2 * m));

          in0 = input_buff_top.val[0];
          tmp0 = input_buff_top.val[1];
          tmp1 = vextq_f32(in0, zero, 1);

          in2 = input_buff_mid.val[0];
          tmp2 = input_buff_mid.val[1];
          tmp3 = vextq_f32(in2, zero, 1);

          in4 = input_buff_bottom.val[0];
          tmp4 = input_buff_bottom.val[1];
          tmp5 = vextq_f32(in4, zero, 1);

          out0 = vmulq_n_f32(in0, w00);
          out0 = vmlaq_n_f32(out0, tmp0, w01);
          out0 = vmlaq_n_f32(out0, tmp1, w02);
          out0 = vmlaq_n_f32(out0, in2, w10);
          out0 = vmlaq_n_f32(out0, tmp2, w11);
          out0 = vmlaq_n_f32(out0, tmp3, w12);
          out0 = vmlaq_n_f32(out0, in4, w20);
          out0 = vmlaq_n_f32(out0, tmp4, w21);
          out0 = vmlaq_n_f32(out0, tmp5, w22);
          if (if_bias) {
            out0 = vaddq_f32(out0, biasv);
          }
          vst1q_lane_f32(output_ptr, out0, 0);
          vst1q_lane_f32(output_ptr + 1, out0, 1);
          vst1q_lane_f32(output_ptr + 2, out0, 2);
        }
        int m;
        for (m = 0; m < output_width - 2; m += 3) {
        }
        for (int j = m; j < output_width; j++) {
          output_data[i * output_width + j] =
              input_data[(2 * i) * input_width + 2 * j] * w00 +
              input_data[(2 * i) * input_width + 2 * j + 1] * w01 +
              input_data[(2 * i) * input_width + 2 * j + 2] * w02 +
              input_data[(2 * i + 1) * input_width + 2 * j] * w10 +
              input_data[(2 * i + 1) * input_width + 2 * j + 1] * w11 +
              input_data[(2 * i + 1) * input_width + 2 * j + 2] * w12 +
              input_data[(2 * i + 2) * input_width + 2 * j] * w20 +
              input_data[(2 * i + 2) * input_width + 2 * j + 1] * w21 +
              input_data[(2 * i + 2) * input_width + 2 * j + 2] * w22;
          if (if_bias) {
            output_data[i * output_width + j] += *bias_data;
          }
        }
      }
    }
  }

#endif
}

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
