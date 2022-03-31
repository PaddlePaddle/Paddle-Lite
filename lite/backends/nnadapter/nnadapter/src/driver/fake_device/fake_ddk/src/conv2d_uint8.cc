// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>
#include "fake_ddk/fake_ddk_pub.h"
namespace fake_ddk {
namespace nn {
int conv_uint8_8bit_mmad(fakedevice_nn_tensor_t* input_tensor,
                         fakedevice_nn_tensor_t* output_tensor,
                         fakedevice_nn_tensor_t* kernel,
                         fakedevice_nn_tensor_t* bias,
                         fakedevice_nn_conv2d_param* conv_param) {
  printf("comein conv_uint8\n ");
  int batch = input_tensor->attr->dims[0];
  int group = conv_param->group;
  int input_c = input_tensor->attr->dims[1];
  int input_h = input_tensor->attr->dims[2];
  int input_w = input_tensor->attr->dims[3];
  int output_c = output_tensor->attr->dims[1];
  int output_h = output_tensor->attr->dims[2];
  int output_w = output_tensor->attr->dims[3];
  int kernel_size = input_c * conv_param->ksize[0] * conv_param->ksize[1];
  int n, g, c, h, w, kc, kh, kw;
  int input_offset = 0;
  int kernel_offset = 0;
  int output_offset = 0;
  uint8_t* input_data = reinterpret_cast<uint8_t*>(input_tensor->data);
  uint8_t* output_data = reinterpret_cast<uint8_t*>(output_tensor->data);
  uint8_t* kernel_data = reinterpret_cast<uint8_t*>(kernel->data);
  int32_t* bias_data = NULL;
  if (bias != NULL) bias_data = reinterpret_cast<int32_t*>(bias->data);
  float input_scale = input_tensor->attr->qntParamAffineAsymmetric.scale[0];
  float kernel_scale = kernel->attr->qntParamAffineAsymmetric.scale[0];
  float output_scale = output_tensor->attr->qntParamAffineAsymmetric.scale[0];
  int32_t kernel_zero = kernel->attr->qntParamAffineAsymmetric.zero_point[0];
  int32_t input_zero =
      input_tensor->attr->qntParamAffineAsymmetric.zero_point[0];
  int32_t output_zero =
      output_tensor->attr->qntParamAffineAsymmetric.zero_point[0];
  printf(" input_scale %f\n", input_scale);
  printf(" kernel_scale %f\n", kernel_scale);
  printf(" output_scale %f\n", output_scale);
  printf(" input_zero %d\n", input_zero);
  printf(" kernel_zero %d\n", kernel_zero);
  printf(" output_zero %d\n", output_zero);
  printf(" input_c %d\n", input_c);
  printf(" output_c %d\n", output_c);
  printf(" ksize[0] %d\n", conv_param->ksize[0]);
  printf(" ksize[1] %d\n", conv_param->ksize[1]);
  printf(" pad[0] %d\n", conv_param->pad[0]);
  printf(" pad[1] %d\n", conv_param->pad[1]);
  printf(" pad[2] %d\n", conv_param->pad[2]);
  printf(" pad[3] %d\n", conv_param->pad[3]);
  printf(" stride[0] %d\n", conv_param->stride[0]);
  printf(" stride[1] %d\n", conv_param->stride[1]);
  printf(" layout %d\n", input_tensor->attr->layout);
  printf(" has BIAS %d\n", bias == NULL ? 0 : 1);
  printf(" has RELU %d\n", conv_param->has_relu);
  if (conv_param->ksize[0] == 0) conv_param->ksize[0] = 1;
  if (conv_param->ksize[1] == 0) conv_param->ksize[1] = 1;
  if (input_w == 0) input_w = 1;
  for (n = 0; n < batch; ++n) {
    for (g = 0; g < group; ++g) {
      for (c = 0; c < output_c; ++c) {
        for (h = 0; h < output_h; ++h) {
          for (w = 0; w < output_w; ++w) {
            const int h_start =
                (h * conv_param->stride[1]) - conv_param->pad[2];
            const int w_start =
                (w * conv_param->stride[0]) - conv_param->pad[0];
            int32_t total = 0;
            float total_fp32;
            if (input_tensor->attr->layout == DataLayoutType::NCHW) {
              output_offset = n * group * output_c * output_h * output_w +
                              g * output_c * output_h * output_w +
                              c * output_h * output_w + h * output_w + w;
            } else {
              output_offset = n * group * output_c * output_h * output_w +
                              h * output_w * group * output_c +
                              w * group * output_c + output_c * g + c;
            }
            for (kc = 0; kc < input_c; ++kc) {
              for (kh = 0; kh < conv_param->ksize[1]; ++kh) {
                for (kw = 0; kw < conv_param->ksize[0]; ++kw) {
                  const int cur_y = h_start + conv_param->dilation[1] * kh;
                  const int cur_x = w_start + conv_param->dilation[0] * kw;
                  // If the location is outside the bounds of the input image,
                  // use zero as a default value.
                  if ((cur_x >= 0) && (cur_x < input_w) && (cur_y >= 0) &&
                      (cur_y < input_h)) {
                    if (input_tensor->attr->layout == DataLayoutType::NCHW) {
                      input_offset = n * group * input_c * input_h * input_w +
                                     g * input_c * input_h * input_w +
                                     kc * input_h * input_w + cur_y * input_w +
                                     cur_x;
                      kernel_offset =
                          g * output_c * kernel_size + c * kernel_size +
                          kc * conv_param->ksize[1] * conv_param->ksize[0] +
                          kh * conv_param->ksize[0] + kw;
                    } else {
                      input_offset = n * group * input_c * input_h * input_w +
                                     cur_y * input_w * input_c * group +
                                     cur_x * input_c * group + g * input_c + kc;
                      kernel_offset =
                          c * group * kernel_size +
                          kh * conv_param->ksize[0] * input_c * group +
                          kw * input_c * group + g * input_c + kc;
                    }
                    total +=
                        input_data[input_offset] * kernel_data[kernel_offset];
                  }
                }
              }
            }
            if (bias != NULL) total += bias_data[output_c * g + c];

            total_fp32 = (static_cast<float>(total) - input_zero) *
                         input_scale * kernel_scale;
            if (conv_param->has_relu) {
              if (total_fp32 < 0) {
                total_fp32 = 0;
              }
            }
            int out = round(total_fp32 / output_scale) + output_zero;
            if (out > 255) out = 255;
            if (out < 0) out = 0;
            output_data[output_offset] = out;
          }
        }
      }
    }
  }
  return 0;
}

int conv_uint8_fp32_mmad(fakedevice_nn_tensor_t* input_tensor,
                         fakedevice_nn_tensor_t* output_tensor,
                         fakedevice_nn_tensor_t* kernel,
                         fakedevice_nn_tensor_t* bias,
                         fakedevice_nn_conv2d_param* conv_param) {
  printf("comein conv_uint8\n ");
  int batch = input_tensor->attr->dims[0];
  int group = conv_param->group;
  int input_c = input_tensor->attr->dims[1];
  int input_h = input_tensor->attr->dims[2];
  int input_w = input_tensor->attr->dims[3];
  int output_c = output_tensor->attr->dims[1];
  int output_h = output_tensor->attr->dims[2];
  int output_w = output_tensor->attr->dims[3];
  int kernel_size = input_c * conv_param->ksize[0] * conv_param->ksize[1];
  int n, g, c, h, w, kc, kh, kw;
  int input_offset = 0;
  int kernel_offset = 0;
  int output_offset = 0;
  uint8_t* input_data = reinterpret_cast<uint8_t*>(input_tensor->data);
  uint8_t* output_data = reinterpret_cast<uint8_t*>(output_tensor->data);
  uint8_t* kernel_data = reinterpret_cast<uint8_t*>(kernel->data);
  int32_t* bias_data = NULL;
  if (bias != NULL) bias_data = reinterpret_cast<int32_t*>(bias->data);
  float input_scale = input_tensor->attr->qntParamAffineAsymmetric.scale[0];
  float kernel_scale = kernel->attr->qntParamAffineAsymmetric.scale[0];
  float output_scale = output_tensor->attr->qntParamAffineAsymmetric.scale[0];
  int32_t kernel_zero = kernel->attr->qntParamAffineAsymmetric.zero_point[0];
  int32_t input_zero =
      input_tensor->attr->qntParamAffineAsymmetric.zero_point[0];
  int32_t output_zero =
      output_tensor->attr->qntParamAffineAsymmetric.zero_point[0];
  printf(" input_scale %f\n", input_scale);
  printf(" kernel_scale %f\n", kernel_scale);
  printf(" output_scale %f\n", output_scale);
  printf(" input_zero %d\n", input_zero);
  printf(" kernel_zero %d\n", kernel_zero);
  printf(" output_zero %d\n", output_zero);
  printf(" input_c %d\n", input_c);
  printf(" output_c %d\n", output_c);
  printf(" ksize[0] %d\n", conv_param->ksize[0]);
  printf(" ksize[1] %d\n", conv_param->ksize[1]);
  printf(" pad[0] %d\n", conv_param->pad[0]);
  printf(" pad[1] %d\n", conv_param->pad[1]);
  printf(" pad[2] %d\n", conv_param->pad[2]);
  printf(" pad[3] %d\n", conv_param->pad[3]);
  printf(" stride[0] %d\n", conv_param->stride[0]);
  printf(" stride[1] %d\n", conv_param->stride[1]);
  printf(" layout %d\n", input_tensor->attr->layout);
  printf(" has BIAS %d\n", bias == NULL ? 0 : 1);
  printf(" has RELU %d\n", conv_param->has_relu);
  /* dequant input  */
  int input_size = batch * group * input_c * input_h * input_w;
  float* input_fp32 =
      reinterpret_cast<float*>(malloc(sizeof(float) * input_size));
  for (int i = 0; i < input_size; i++)
    input_fp32[i] =
        (static_cast<float>(input_data[i]) - input_zero) * input_scale;
  /* dequant kernel  */
  int kernel_total = group * output_c * kernel_size;
  float* kernel_fp32 =
      reinterpret_cast<float*>(malloc(sizeof(float) * kernel_total));
  for (int i = 0; i < kernel_total; i++)
    kernel_fp32[i] =
        (static_cast<float>(kernel_data[i]) - kernel_zero) * kernel_scale;
  /* dequant biases  */
  int bias_size = group * output_c;
  float* bias_fp32 = NULL;
  if (bias != NULL) {
    bias_fp32 = reinterpret_cast<float*>(malloc(sizeof(float) * bias_size));
    for (int i = 0; i < bias_size; i++)
      bias_fp32[i] =
          static_cast<float>(bias_data[i]) * input_scale * kernel_scale;
  }
  if (conv_param->ksize[0] == 0) conv_param->ksize[0] = 1;
  if (conv_param->ksize[1] == 0) conv_param->ksize[1] = 1;
  if (input_w == 0) input_w = 1;
  for (n = 0; n < batch; ++n) {
    for (g = 0; g < group; ++g) {
      for (c = 0; c < output_c; ++c) {
        for (h = 0; h < output_h; ++h) {
          for (w = 0; w < output_w; ++w) {
            const int h_start =
                (h * conv_param->stride[1]) - conv_param->pad[2];
            const int w_start =
                (w * conv_param->stride[0]) - conv_param->pad[0];
            float total = 0.f;
            if (input_tensor->attr->layout == DataLayoutType::NCHW) {
              output_offset = n * group * output_c * output_h * output_w +
                              g * output_c * output_h * output_w +
                              c * output_h * output_w + h * output_w + w;
            } else {
              output_offset = n * group * output_c * output_h * output_w +
                              h * output_w * group * output_c +
                              w * group * output_c + output_c * g + c;
            }
            for (kc = 0; kc < input_c; ++kc) {
              for (kh = 0; kh < conv_param->ksize[1]; ++kh) {
                for (kw = 0; kw < conv_param->ksize[0]; ++kw) {
                  const int cur_y = h_start + conv_param->dilation[1] * kh;
                  const int cur_x = w_start + conv_param->dilation[0] * kw;
                  // If the location is outside the bounds of the input image,
                  // use zero as a default value.
                  if ((cur_x >= 0) && (cur_x < input_w) && (cur_y >= 0) &&
                      (cur_y < input_h)) {
                    if (input_tensor->attr->layout == DataLayoutType::NCHW) {
                      input_offset = n * group * input_c * input_h * input_w +
                                     g * input_c * input_h * input_w +
                                     kc * input_h * input_w + cur_y * input_w +
                                     cur_x;
                      kernel_offset =
                          g * output_c * kernel_size + c * kernel_size +
                          kc * conv_param->ksize[1] * conv_param->ksize[0] +
                          kh * conv_param->ksize[0] + kw;
                    } else {
                      input_offset = n * group * input_c * input_h * input_w +
                                     cur_y * input_w * input_c * group +
                                     cur_x * input_c * group + g * input_c + kc;
                      kernel_offset =
                          c * group * kernel_size +
                          kh * conv_param->ksize[0] * input_c * group +
                          kw * input_c * group + g * input_c + kc;
                    }
                    total +=
                        input_fp32[input_offset] * kernel_fp32[kernel_offset];
                  }
                }
              }
            }
            if (bias != NULL) total += bias_fp32[output_c * g + c];
            if (conv_param->has_relu) {
              if (total < 0) {
                total = 0;
              }
            }
            int out = round(total / output_scale) + output_zero;
            if (out > 255) out = 255;
            if (out < 0) out = 0;
            output_data[output_offset] = out;
          }
        }
      }
    }
  }

  free(input_fp32);
  free(kernel_fp32);
  if (bias != NULL) free(bias_fp32);

  return 0;
}
}  // namespace nn
}  // namespace fake_ddk
