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
#include "operators/math/depthwise_conv_3x3.h"
#include <arm_neon.h>
#include <vector>

namespace paddle_mobile {
namespace operators {
namespace math {
void DepthwiseConv3x3(const Tensor *input, vector<int> strides,
                      vector<int> paddings, const Tensor *filter, Tensor *bias,
                      Tensor *output, bool if_bias) {
#if __ARM_NEON
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
          hend = min(hstart + _kernel_size, input_height + padding_height);
          wend = min(wstart + _kernel_size, input_width + padding_width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          hend = min(hend, input_height);
          wend = min(wend, input_width);
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
#if defined(ARMV17)
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
#else
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
#endif
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

void DepthwiseConv3x3s1p1(const Tensor *input, const Tensor *filter,
                          Tensor *output, Tensor *bias, bool if_bias) {
  const float *input_data = input->data<float>();
  const float *filter_data = filter->data<float>();
  float *output_data = output->data<float>();
  const float *bias_data = bias->data<float>();

  const int h = static_cast<int>(input->dims()[2]);
  const int w = static_cast<int>(input->dims()[3]);
  const int l = h;

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

      output_data[0] = w11 * input_data[0] + w12 * input_data[1] +
                       w21 * input_data[l] + w22 * input_data[l + 1] +
                       bias_data[j];
      output_data[l - 1] = w10 * input_data[l - 2] + w11 * input_data[l - 1] +
                           w20 * input_data[2 * l - 2] +
                           w21 * input_data[2 * l - 1] + bias_data[j];
      output_data[(l - 1) * l] =
          w01 * input_data[(l - 2) * l] + w02 * input_data[(l - 2) * l + 1] +
          w11 * input_data[(l - 1) * l] + w12 * input_data[(l - 1) * l + 1] +
          bias_data[j];
      output_data[l * l - 1] = w00 * input_data[(l - 2) * (l + 1)] +
                               w01 * input_data[(l - 2) * (l + 1) + 1] +
                               w10 * input_data[l * l - 2] +
                               w11 * input_data[l * l - 1] + bias_data[j];

      for (int i = 1; i < l - 1; ++i) {
        output_data[i * l] =
            w01 * input_data[i * l - l] + w02 * input_data[i * l - l + 1] +
            w11 * input_data[i * l] + w12 * input_data[i * l + 1] +
            w21 * input_data[i * l + l] + w22 * input_data[i * l + l + 1] +
            bias_data[j];
        output_data[i * l + l - 1] = w00 * input_data[i * l + l - 1 - l - 1] +
                                     w01 * input_data[i * l + l - 1 - l] +
                                     w10 * input_data[i * l + l - 1 - 1] +
                                     w11 * input_data[i * l + l - 1] +
                                     w20 * input_data[i * l + l - 1 + l - 1] +
                                     w21 * input_data[i * l + l - 1 + l] +
                                     bias_data[j];
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
