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

#include <algorithm>
#include <utility>
#include <vector>
#include "fake_ddk/fake_ddk_pub.h"

namespace fake_ddk {
namespace nn {

std::vector<uint32_t> shape_slice(const std::vector<uint32_t>& input_shape,
                                  int start,
                                  int end) {
  int input_rank = input_shape.size();
  start = start < 0 ? 0 : (start > input_rank ? input_rank : start);
  end = end < start ? start : (end > input_rank ? input_rank : end);
  return std::vector<uint32_t>(input_shape.data() + start,
                               input_shape.data() + end);
}

int64_t shape_production(const std::vector<uint32_t>& input_shape) {
  auto input_rank = input_shape.size();
  int64_t production = 1;
  for (size_t i = 0; i < input_rank; i++) {
    auto dimension = input_shape[i];
    production *= dimension;
  }
  return production;
}

template <typename T>
static int dequantize(T* input_data,
                      const std::vector<uint32_t>& input_shape,
                      const std::pair<std::vector<float>, int>& input_scales,
                      const std::pair<std::vector<uint32_t>, int>& input_zp,
                      float* output_data) {
  if (!input_data || input_shape.empty() || input_scales.first.empty() ||
      !output_data) {
    return -1;
  }
  int quant_bits = sizeof(T) * 8;
  int dtype_max, dtype_min = 0;
  if (typeid(T) == typeid(uint8_t) || typeid(T) == typeid(uint32_t)) {
    dtype_max = static_cast<int>((1 << (quant_bits)) - 1);  // unsigned int
    dtype_min = static_cast<int>(0);
  } else {
    dtype_max = static_cast<int>((1 << (quant_bits - 1)) - 1);  // int
    dtype_min = static_cast<int>(0 - dtype_max);
  }
  auto input_rank = input_shape.size();
  auto input_count = shape_production(input_shape);
  auto scale_count = input_scales.first.size();
  auto channel_dim = input_scales.second;
  if (scale_count > 1 && channel_dim < 0) {
    return -1;
  }
  int64_t outer_count = input_count;
  int64_t inner_count = 1;
  if (scale_count > 1 && channel_dim >= 0) {
    auto channel_count = input_shape[channel_dim];
    if (channel_count != scale_count) {
      return -1;
    }
    outer_count = shape_production(shape_slice(input_shape, 0, channel_dim));
    inner_count =
        shape_production(shape_slice(input_shape, channel_dim + 1, input_rank));
  }
  for (int64_t i = 0; i < outer_count; i++) {
    for (size_t j = 0; j < scale_count; j++) {
      for (int64_t k = 0; k < inner_count; k++) {
        auto index = i * scale_count * inner_count + j * inner_count + k;
        output_data[index] =
            (static_cast<float>(std::min(
                 std::max(static_cast<int>(input_data[index]), dtype_min),
                 dtype_max)) -
             input_zp.first[j]) *
            input_scales.first[j];
      }
    }
  }
  return 0;
}

template <typename T>
static int dequantize(T* input_data,
                      const std::vector<uint32_t>& input_shape,
                      float input_scale,
                      uint32_t input_zp,
                      float* output_data) {
  return dequantize<T>(input_data,
                       input_shape,
                       std::make_pair(std::vector<float>({input_scale}), -1),
                       std::make_pair(std::vector<uint32_t>({input_zp}), -1),
                       output_data);
}

template <typename T>
static int quantize(float* input_data,
                    const std::vector<uint32_t>& input_shape,
                    const std::pair<std::vector<float>, int>& output_scales,
                    const std::pair<std::vector<uint32_t>, int>& output_zp,
                    T* output_data) {
  if (!input_data || input_shape.empty() || output_scales.first.empty() ||
      !output_data) {
    return -1;
  }
  int dtype_max, dtype_min = 0;
  int quant_bits = sizeof(T) * 8;
  if (typeid(T) == typeid(uint8_t) || typeid(T) == typeid(uint32_t)) {
    dtype_max = static_cast<int>((1 << (quant_bits)) - 1);  // unsigned int
    dtype_min = static_cast<int>(0);
  } else {
    dtype_max = static_cast<int>((1 << (quant_bits - 1)) - 1);  // int
    dtype_min = static_cast<int>(0 - dtype_max);
  }
  auto input_rank = input_shape.size();
  auto input_count = shape_production(input_shape);
  auto scale_count = output_scales.first.size();
  auto channel_dim = output_scales.second;
  if (scale_count > 1 && channel_dim < 0) {
    return -1;
  }
  int64_t outer_count = input_count;
  int64_t inner_count = 1;
  if (scale_count > 1 && channel_dim >= 0) {
    auto channel_count = input_shape[channel_dim];
    if (channel_count != scale_count) {
      return -1;
    }
    outer_count = shape_production(shape_slice(input_shape, 0, channel_dim));
    inner_count =
        shape_production(shape_slice(input_shape, channel_dim + 1, input_rank));
  }
  for (int64_t i = 0; i < outer_count; i++) {
    for (size_t j = 0; j < scale_count; j++) {
      for (int64_t k = 0; k < inner_count; k++) {
        auto index = i * scale_count * inner_count + j * inner_count + k;
        output_data[index] =
            std::min(std::max(static_cast<int>(round(input_data[index] /
                                                     output_scales.first[j]) +
                                               output_zp.first[j]),
                              dtype_min),
                     dtype_max);
      }
    }
  }
  return 0;
}

template <typename T>
static int quantize(float* input_data,
                    const std::vector<uint32_t>& input_shape,
                    float output_scale,
                    uint32_t output_zp,
                    T* output_data) {
  return quantize<T>(input_data,
                     input_shape,
                     std::make_pair(std::vector<float>({output_scale}), -1),
                     std::make_pair(std::vector<uint32_t>({output_zp}), -1),
                     output_data);
}

template <typename T>
int conv2d(T* input_data,
           const std::vector<uint32_t>& input_shape,
           T* filter_data,
           const std::vector<uint32_t>& filter_shape,
           T* bias_data,
           int pad_height_top,
           int pad_height_bottom,
           int pad_width_left,
           int pad_width_right,
           int stride_height,
           int stride_width,
           int dilation_height,
           int dilation_width,
           int group,
           bool has_relu,
           T* output_data) {
  if (!input_data || !filter_data || !output_data) {
    return -1;
  }
  auto input_rank = input_shape.size();
  auto filter_rank = filter_shape.size();
  if (input_rank != 4 || filter_rank != 4) {
    return -1;
  }
  auto batch_size = input_shape[0];
  auto input_channel_size = input_shape[1];
  auto input_height = input_shape[2];
  auto input_width = input_shape[3];
  auto output_channel_size = filter_shape[0];
  auto kernel_height = filter_shape[2];
  auto kernel_width = filter_shape[3];
  auto output_height = (input_height + (pad_height_top + pad_height_bottom) -
                        dilation_height * (kernel_height - 1) - 1) /
                           stride_height +
                       1;
  auto output_width = (input_width + (pad_width_left + pad_width_right) -
                       dilation_width * (kernel_width - 1) - 1) /
                          stride_width +
                      1;
  auto output_channel_group = output_channel_size / group;
  auto input_channel_group = input_channel_size / group;
  for (int bs = 0; bs < batch_size; bs++) {
    for (int g = 0; g < group; g++) {
      for (int oc = 0; oc < output_channel_group; oc++) {
        for (int oh = 0; oh < output_height; oh++) {
          for (int ow = 0; ow < output_width; ow++) {
            int output_index =
                bs * group * output_channel_group * output_height *
                    output_width +
                g * output_channel_group * output_height * output_width +
                oc * output_height * output_width + oh * output_width + ow;
            T output_value =
                bias_data ? bias_data[g * output_channel_group + oc] : 0;
            for (int ic = 0; ic < input_channel_group; ic++) {
              for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                  int iw =
                      ow * stride_width - pad_width_left + kw * dilation_width;
                  int ih = oh * stride_height - pad_height_top +
                           kh * dilation_height;
                  if (iw < 0 || iw >= input_width) continue;
                  if (ih < 0 || ih >= input_height) continue;
                  int input_index =
                      bs * input_channel_size * input_height * input_width +
                      g * input_channel_group * input_height * input_width +
                      ic * input_height * input_width + ih * input_width + iw;
                  int filter_index =
                      g * output_channel_group * input_channel_group *
                          kernel_height * kernel_width +
                      oc * input_channel_group * kernel_height * kernel_width +
                      ic * kernel_height * kernel_width + kh * kernel_width +
                      kw;
                  output_value +=
                      input_data[input_index] * filter_data[filter_index];
                }
              }
            }
            if (has_relu) {
              output_value = output_value > 0 ? output_value : 0;
            }
            output_data[output_index] = output_value;
          }
        }
      }
    }
  }
  return 0;
}

int conv2d(uint8_t* input_data,
           const std::vector<uint32_t>& input_shape,
           float input_scale,
           uint32_t input_zp,
           uint8_t* filter_data,
           const std::vector<uint32_t>& filter_shape,
           const std::pair<std::vector<float>, int>& filter_scales,
           const std::pair<std::vector<uint32_t>, int>& filter_zp,
           int32_t* bias_data,
           int pad_height_top,
           int pad_height_bottom,
           int pad_width_left,
           int pad_width_right,
           int stride_height,
           int stride_width,
           int dilation_height,
           int dilation_width,
           int group,
           bool has_relu,
           uint8_t* output_data,
           float output_scale,
           uint32_t output_zp) {
  if (!input_data || !filter_data) {
    return -1;
  }
  auto input_rank = input_shape.size();
  auto filter_rank = filter_shape.size();
  if (input_rank != 4 || filter_rank != 4) {
    return -1;
  }
  auto batch_size = input_shape[0];
  auto input_height = input_shape[2];
  auto input_width = input_shape[3];
  auto output_channel_size = filter_shape[0];
  auto kernel_height = filter_shape[2];
  auto kernel_width = filter_shape[3];
  auto output_height = (input_height + (pad_height_top + pad_height_bottom) -
                        dilation_height * (kernel_height - 1) - 1) /
                           stride_height +
                       1;
  auto output_width = (input_width + (pad_width_left + pad_width_right) -
                       dilation_width * (kernel_width - 1) - 1) /
                          stride_width +
                      1;
  std::vector<uint32_t> output_shape = {
      batch_size, output_channel_size, output_height, output_width};
  // Dequantize input data
  auto input_count = shape_production(input_shape);
  std::vector<float> dequantized_input_data(input_count);
  int status = dequantize(input_data,
                          input_shape,
                          input_scale,
                          input_zp,
                          dequantized_input_data.data());
  if (status) return status;
  // Dequantize filter data
  auto filter_count = shape_production(filter_shape);
  std::vector<float> dequantized_filter_data(filter_count);
  status = dequantize(filter_data,
                      filter_shape,
                      filter_scales,
                      filter_zp,
                      dequantized_filter_data.data());
  if (status) return status;
  // Dequantize bias data
  std::vector<float> dequantized_bias_data;
  if (bias_data) {
    std::vector<float> bias_scales;
    for (auto filter_scale : filter_scales.first) {
      bias_scales.push_back(input_scale * filter_scale);
    }
    dequantized_bias_data.resize(output_channel_size);
    status =
        dequantize(bias_data,
                   {output_channel_size},
                   std::make_pair(bias_scales, bias_scales.size() > 0 ? 0 : -1),
                   std::make_pair(std::vector<uint32_t>({0}), -1),
                   dequantized_bias_data.data());
    if (status) return status;
  }
  // Prepare dequantized output data
  auto output_count = shape_production(output_shape);
  std::vector<float> dequantized_output_data(output_count);
  status = conv2d<float>(
      dequantized_input_data.data(),
      input_shape,
      dequantized_filter_data.data(),
      filter_shape,
      dequantized_bias_data.size() > 0 ? dequantized_bias_data.data() : nullptr,
      pad_height_top,
      pad_height_bottom,
      pad_width_left,
      pad_width_right,
      stride_height,
      stride_width,
      dilation_height,
      dilation_width,
      group,
      has_relu,
      dequantized_output_data.data());
  if (status) return status;
  // Quantize output data
  return quantize(dequantized_output_data.data(),
                  output_shape,
                  output_scale,
                  output_zp,
                  output_data);
}

int conv2d_invoke(uint8_t* input_data,
                  const std::vector<uint32_t>& input_shape,
                  float input_scale,
                  uint32_t input_zp,
                  uint8_t* filter_data,
                  const std::vector<uint32_t>& filter_shape,
                  float filter_scale,
                  uint32_t filter_zp,
                  int32_t* bias_data,
                  int pad_height_top,
                  int pad_height_bottom,
                  int pad_width_left,
                  int pad_width_right,
                  int stride_height,
                  int stride_width,
                  int dilation_height,
                  int dilation_width,
                  int group,
                  bool has_relu,
                  uint8_t* output_data,
                  float output_scale,
                  uint32_t output_zp) {
  return conv2d(input_data,
                input_shape,
                input_scale,
                input_zp,
                filter_data,
                filter_shape,
                std::make_pair(std::vector<float>({filter_scale}), -1),
                std::make_pair(std::vector<uint32_t>({filter_zp}), -1),
                bias_data,
                pad_height_top,
                pad_height_bottom,
                pad_width_left,
                pad_width_right,
                stride_height,
                stride_width,
                dilation_height,
                dilation_width,
                group,
                has_relu,
                output_data,
                output_scale,
                output_zp);
}

int conv_uint8_compute(fakedevice_nn_tensor_t* input_tensor,
                       fakedevice_nn_tensor_t* output_tensor,
                       fakedevice_nn_tensor_t* kernel,
                       fakedevice_nn_tensor_t* bias,
                       fakedevice_nn_conv2d_param* conv_param) {
  fprintf(stderr, "fake_ddk: conv_uint8 computing\n");
  conv2d_invoke(reinterpret_cast<uint8_t*>(input_tensor->data),
                input_tensor->attr->dims,
                input_tensor->attr->qntParamAffineAsymmetric.scale[0],
                input_tensor->attr->qntParamAffineAsymmetric.zero_point[0],
                reinterpret_cast<uint8_t*>(kernel->data),
                kernel->attr->dims,
                kernel->attr->qntParamAffineAsymmetric.scale[0],
                kernel->attr->qntParamAffineAsymmetric.zero_point[0],
                reinterpret_cast<int32_t*>(bias->data),
                conv_param->pad[0],
                conv_param->pad[1],
                conv_param->pad[2],
                conv_param->pad[3],
                conv_param->stride[0],
                conv_param->stride[1],
                conv_param->dilation[0],
                conv_param->dilation[1],
                conv_param->group,
                conv_param->has_relu,
                reinterpret_cast<uint8_t*>(output_tensor->data),
                output_tensor->attr->qntParamAffineAsymmetric.scale[0],
                output_tensor->attr->qntParamAffineAsymmetric.zero_point[0]);
}

}  // namespace nn
}  // namespace fake_ddk
