/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifndef conv_process_hpp
#define conv_process_hpp

#include <string.h>
#include <cmath>
#include <vector>

#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/backends/fpga/KD/llapi/bias_scale.h"
#include "lite/backends/fpga/KD/llapi/filter.h"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/tensor.hpp"
#include "lite/backends/fpga/KD/tensor_util.hpp"

namespace paddle {
namespace zynqmp {

inline int get_aligned_filter_element_num(int chw) {
  return align_to_x(chw, FILTER_ELEMENT_ALIGNMENT);
}

inline int get_filter_num_per_div(Tensor* filter, int group_num) {
  auto chw = filter->shape().channel() * filter->shape().height() *
             filter->shape().width();
  auto num = filter->shape().num();
  int div_capacity = filter::calc_division_capacity(chw);
  return filter::calc_num_per_div(num, group_num, div_capacity);
}

inline int get_split_num(Tensor* filter) {
  auto chw = filter->shape().channel() * filter->shape().height() *
             filter->shape().width();
  auto num = filter->shape().num();
  int div_capacity = filter::calc_division_capacity(chw);
  int filter_num_alignment = filter::get_filter_num_alignment();
  int aligned_num = align_to_x(num, filter_num_alignment);
  return filter::calc_split_num(aligned_num, div_capacity);
}

inline int get_pack_num(Tensor* filter, int group_num) {
  auto chw = filter->shape().channel() * filter->shape().height() *
             filter->shape().width();
  auto num = filter->shape().num();
  int div_capacity = filter::calc_division_capacity(chw);
  int filter_num_alignment = filter::get_filter_num_alignment();
  int aligned_num_per_group = align_to_x(num / group_num, filter_num_alignment);
  return filter::calc_pack_num(aligned_num_per_group, group_num, div_capacity);
}

inline void fill_scale_bias_const(ConvParam* param_) {
  int channel = param_->output->shape().channel();
  Shape sb_shape(N, {channel});
  float* new_scale_ptr = param_->scale()->mutableData<float>(FP32, sb_shape);
  float* new_bias_ptr = param_->bias()->mutableData<float>(FP32, sb_shape);
  for (int i = 0; i < channel; i++) {
    new_scale_ptr[i] = 1.0f;
    new_bias_ptr[i] = 0.0f;
  }
  param_->scale()->flush();
  param_->bias()->flush();
}

inline void combine_bn_params(BatchnormParam* bn, ConvParam* param_) {
  int channel = param_->output->shape().channel();
  Shape sb_shape(N, {channel});
  float* new_scale_ptr = param_->scale()->mutableData<float>(FP32, sb_shape);
  float* new_bias_ptr = param_->bias()->mutableData<float>(FP32, sb_shape);
  float* bn_scale_ptr = bn->scale->data<float>();
  float* bn_bias_ptr = bn->bias->data<float>();
  float* bn_var_ptr = bn->variance->data<float>();
  float* bn_mean_ptr = bn->mean->data<float>();
  float epsilon = bn->epsilon;
  for (int i = 0; i < channel; i++) {
    float new_scale = bn_scale_ptr[i] /
                      static_cast<float>(pow((bn_var_ptr[i] + epsilon), 0.5));
    new_scale_ptr[i] = new_scale;
    new_bias_ptr[i] = bn_bias_ptr[i] + (0 - bn_mean_ptr[i]) * new_scale_ptr[i];
  }
}

inline void combine_add_bn_params(BatchnormParam* bn,
                                  Tensor* bias,
                                  ConvParam* param_) {
  int channel = param_->output->shape().channel();
  Shape sb_shape(N, {channel});
  float* new_scale_ptr = param_->scale()->mutableData<float>(FP32, sb_shape);
  float* new_bias_ptr = param_->bias()->mutableData<float>(FP32, sb_shape);
  if (bn != nullptr) {
    float* bn_scale_ptr = bn->scale->data<float>();
    float* bn_bias_ptr = bn->bias->data<float>();
    float* bn_var_ptr = bn->variance->data<float>();
    float* bn_mean_ptr = bn->mean->data<float>();
    float epsilon = bn->epsilon;
    float* bias_data = bias->data<float>();
    for (int i = 0; i < channel; i++) {
      float new_scale = bn_scale_ptr[i] /
                        static_cast<float>(pow((bn_var_ptr[i] + epsilon), 0.5));
      new_scale_ptr[i] = new_scale;
      new_bias_ptr[i] =
          bn_bias_ptr[i] + (bias_data[i] - bn_mean_ptr[i]) * new_scale_ptr[i];
    }
  } else {
    for (int i = 0; i < channel; i++) {
      new_scale_ptr[i] = 1.0f;
      new_bias_ptr[i] = 0.0f;
    }
  }
  param_->scale()->flush();
  param_->bias()->flush();
  param_->scale()->setDataLocation(CPU);
  param_->bias()->setDataLocation(CPU);
}

inline int gcd_(int a, int b) {
  while (b) {
    int temp = a;
    a = b;
    b = temp % b;
  }
  return a;
}

inline int lcm_(int a, int b) { return a * b / gcd_(a, b); }

inline void format_bias_scale_new(Tensor* bias,
                                  Tensor* scale,
                                  Tensor* scale_bias) {
  Shape& bias_shape = bias->shape();
  int channel = bias_shape.channel();
  int repeat = 1;
  int alignment = 16;
  int length = channel;

  if (channel % alignment != 0 || channel < alignment) {
    int c_lcm = lcm_(channel, alignment);
    repeat = c_lcm / (channel);
  }
  Shape shape(N, {2 * channel * repeat});
  float16* scale_bias_data = scale_bias->mutableData<float16>(FP16, shape);

  float* bias_data_float = bias->data<float>();
  float* scale_data_float = scale->data<float>();

  for (int i = 0; i < repeat; i++) {
    for (int j = 0; j < length; j++) {
      float16 value_bias = float_to_half(bias_data_float[j]);
      scale_bias_data[i * length + j] = value_bias;
    }
  }
  for (int i = 0; i < repeat; i++) {
    for (int j = 0; j < length; j++) {
      float16 value_scale = float_to_half(scale_data_float[j]);
      scale_bias_data[i * length + j + length * repeat] = value_scale;
    }
  }
}

inline void format_16_bias(Tensor* bias, Tensor* quantized_bias, int channel) {
  int repeat = 1;
  int alignment = 16;
  int length = channel;

  if (channel % alignment != 0 || channel < alignment) {
    int c_lcm = lcm_(channel, alignment);
    repeat = c_lcm / (channel);
  }
  Shape shape(N, {channel * repeat});
  float16* quantized_bias_data =
      quantized_bias->mutableData<float16>(FP16, shape);

  float* bias_data = bias->data<float>();
  // bias aligned to 16 by hw;
  for (int i = 0; i < repeat; i++) {
    for (int j = 0; j < length; j++) {
      float16 value = float_to_half(bias_data[j]);
      quantized_bias_data[i * length + j] = value;
    }
  }
  quantized_bias->flush();
}

inline void format_scale_bias(Tensor* scale,
                              Tensor* bias,
                              Tensor* filter,
                              Tensor* scale_bias,
                              int group) {
  float* scale_data = nullptr;
  float* bias_data = nullptr;
  if (scale != nullptr) {
    scale_data = scale->data<float>();
  }
  if (bias != nullptr) {
    bias_data = bias->data<float>();
  }
  int channel = filter->shape().num();
  int scale_bias_len = align_to_x(channel / group, BS_NUM_ALIGNMENT) * group;

  int c_per_group = channel / group;
  int aligned_c_per_group = align_to_x(channel / group, BS_NUM_ALIGNMENT);

  Shape bias_scale_shape(N, {2 * scale_bias_len});
  float* bs_data = scale_bias->mutableData<float>(FP32, bias_scale_shape);
  float* temp_data =
      reinterpret_cast<float*>(fpga_malloc(2 * scale_bias_len * sizeof(float)));
  memset(temp_data, 0, 2 * scale_bias_len * sizeof(float));

  std::vector<float> scales;
  if (scale_data != nullptr) {
    for (int i = 0; i < channel; ++i) {
      scales.push_back(scale_data[i]);
    }
    for (int i = 0; i < scale_bias_len - channel; i++) {
      scales.push_back(1);
    }
  } else {
    for (int i = 0; i < scale_bias_len; i++) {
      scales.push_back(1);
    }
  }

  for (int i = 0; i < scale_bias_len; ++i) {
    temp_data[i + scale_bias_len] = 1;
    temp_data[i] = 0;
  }

  for (int g = 0; g < group; g++) {
    for (int c = 0; c < c_per_group; c++) {
      int src_index = g * c_per_group + c;
      int dst_index = g * aligned_c_per_group + c;
      float scale_value = scales[src_index];
      float bias_value = bias_data == nullptr ? 0 : bias_data[src_index];
      temp_data[dst_index + scale_bias_len] = scale_value;
      temp_data[dst_index] = bias_value;
    }
  }

  bias_scale::format_bias_scale_array(
      &temp_data, scale_bias_len / group, scale_bias_len);
  memcpy(bs_data, temp_data, 2 * scale_bias_len * sizeof(float));
}

inline void format_filter(Tensor* filter,
                          Tensor* quantized_filter,
                          int group,
                          std::vector<float>& scales,  // NOLINT
                          float max) {
  float max_value = find_max(*filter);
  Shape& filter_shape = filter->shape();

  int mem_size;
  std::vector<float> max_values;
  int8_t* quantized_data = filter::format_filter(filter->data<float>(),
                                                 mem_size,
                                                 filter_shape.num(),
                                                 filter_shape.channel(),
                                                 filter_shape.height(),
                                                 filter_shape.width(),
                                                 group,
                                                 max_value,
                                                 max_values);

  float mem_factor = mem_size * 1.0f / filter->shape().numel();
  quantized_filter->setMemScale(mem_factor);

  quantized_filter->setAligned(true);
  int8_t* src = quantized_filter->mutableData<int8_t>(INT8, filter->shape());
  quantized_filter->scale()[0] = max_value / 127.0f;
  quantized_filter->scale()[1] = 127.0f / max_value;

  memcpy(src, quantized_data, mem_size);
  quantized_filter->flush();
  fpga_free(quantized_data);

  for (size_t i = 0; i < max_values.size(); i++) {
    scales.push_back(max_values[i] / max_value);
  }
}

inline void format_dw_filter(Tensor* filter,
                             Tensor* quantized_filter,
                             float* scale) {
  int num = filter->shape().num();
  int height = filter->shape().height();
  int width = filter->shape().width();
  auto memory_size = filter->shape().memorySize(sizeof(float));
  auto new_data = (float*)fpga_malloc(memory_size);  // NOLINT
  memcpy(new_data, filter->data<float>(), memory_size);

  size_t size =
      filter::format_dwconv_filter(&new_data, num, height, width, scale);
  float16* src = quantized_filter->mutableData<float16>(FP16, filter->shape());

  memcpy(src, new_data, size);
  quantized_filter->flush();

  fpga_free(new_data);
}

inline void format_fc_filter(Tensor* filter, Tensor* quantized_filter) {
  float max_value = find_max(*filter);
  Shape& filter_shape = filter->shape();
  quantized_filter->setAligned(true);
  quantized_filter->mutableData<int8_t>(INT8, filter->shape());
  quantized_filter->scale()[0] = max_value / 127.0f;
  quantized_filter->scale()[1] = 127.0f / max_value;

  size_t memory_size = filter->shape().memorySize(sizeof(float));
  auto new_data = (float*)fpga_malloc(memory_size);  // NOLINT
  memcpy(new_data, filter->data<float>(), memory_size);

  int8_t* src = quantized_filter->mutableData<int8_t>(INT8, filter->shape());
  memcpy(src, new_data, quantized_filter->shape().memorySize(sizeof(int8_t)));
  quantized_filter->flush();
  fpga_free(new_data);
}

inline void split_filter_num(const ConvParam& c_param) {
  ConvParam& param = const_cast<ConvParam&>(c_param);
  Tensor* input = param.input;
  Tensor* out = param.output;
  Tensor* filter = param.filter;
  auto channel = out->shape().channel();
  int split_num = get_split_num(param.filter);
  int filter_num_per_div = get_filter_num_per_div(filter, param.groups);

  float max = find_max(*filter);

  Shape& out_shape = out->shape();
  for (int i = 0; i < split_num; i++) {
    BasicConvParam* conv_param = new BasicConvParam();
    conv_param->output.setDataLocation(Device);
    conv_param->output.setAligned(true);

    int filter_num = filter->shape().num();
    float16* out_address = nullptr;
    float* out_scale_address = nullptr;

    ConvArgs& args = conv_param->args;

    if (split_num == 1) {
      out_address = out->data<float16>();
      out_scale_address = out->scale();
    }
    filter_num = i == split_num - 1
                     ? channel - (split_num - 1) * filter_num_per_div  // NOLINT
                     : filter_num_per_div;

    if (split_num != 1) {
      Shape shape(NHWC, {1, out_shape.height(), out_shape.width(), filter_num});
      out_address = conv_param->output.mutableData<float16>(FP16, shape);
      out_scale_address = conv_param->output.scale();
    }
    Shape f_shape(NCHW,
                  {filter_num,
                   filter->shape().channel(),
                   filter->shape().height(),
                   filter->shape().width()});

    Tensor new_filter;
    float* new_filter_data = new_filter.mutableData<float>(FP32, f_shape);
    int filter_hwc = filter->shape().height() * filter->shape().width() *
                     filter->shape().channel();

    memcpy(new_filter_data,
           filter->data<float>() + i * filter_num_per_div * filter_hwc,
           filter_num * filter_hwc * sizeof(float));
    new_filter.flush();
    conv_param->filter.mutableData<float>(FP32, f_shape);

    std::vector<float> quant_scale;
    format_filter(
        &new_filter, &(conv_param->filter), param.groups, quant_scale, max);
    conv_param->filter.setDataType(INT8);

    Tensor scale;
    Tensor bias;

    int chnnnel_start = i * filter_num_per_div;

    Shape s_shape(NC, {1, filter_num});
    float* scale_data = scale.mutableData<float>(FP32, s_shape);
    float* bias_data = bias.mutableData<float>(FP32, s_shape);
    for (int n = 0; n < filter_num; n++) {
      scale_data[n] =
          param.scale()->data<float>()[n + chnnnel_start] * quant_scale[n];
    }
    for (int n = 0; n < filter_num; n++) {
      bias_data[n] = param.bias()->data<float>()[n + chnnnel_start];
    }
    format_bias_scale_new(&bias, &scale, &conv_param->scaleBias);
    conv_param->scaleBias.flush();
    args.group_num = param.groups;
    args.sb_address = conv_param->scaleBias.data<float16>();
    args.kernel.stride_h = param.strides[1];
    args.kernel.stride_w = param.strides[0];
    args.kernel.height = new_filter.shape().height();
    args.kernel.width = new_filter.shape().width();

    args.filter_address = conv_param->filter.data<int8_t>();
    args.filter_num = filter_num;
    args.filter_scale_address = conv_param->filter.scale();
    args.image.address = input->data<void>();
    args.image.scale_address = input->scale();
    args.image.channels = input->shape().channel();
    args.image.width = input->shape().width();
    args.image.height = input->shape().height();
    args.image.pad_width = param.paddings[1];
    args.image.pad_height = param.paddings[0];
    args.dilation = param.dilations[0];

    args.output.address = out_address;
    args.output.scale_address = out_scale_address;
    param.splitParams().push_back(conv_param);
  }
}

inline void pack_channel_filter(const ConvParam& c_param) {
  ConvParam& param = const_cast<ConvParam&>(c_param);
  Tensor* input = param.input;
  Tensor* out = param.output;
  Tensor* filter = param.filter;
  int filter_num_alignment = filter::get_filter_num_alignment();
  auto filter_num = filter->shape().num();
  int pack_num = get_pack_num(param.filter, param.groups);
  int group_per_pack = (param.groups + pack_num - 1) / pack_num;
  int filter_per_group = filter_num / param.groups;
  int filter_per_pack = filter_per_group * group_per_pack;
  int channel_per_pack = filter->shape().channel() * group_per_pack;

  float max = find_max(*filter);
  Shape& out_shape = out->shape();

  for (int i = 0; i < pack_num; i++) {
    BasicConvParam* conv_param = new BasicConvParam();

    conv_param->output.setDataLocation(Device);
    conv_param->output.setAligned(true);

    float16* out_address = nullptr;
    float* out_scale_address = nullptr;

    float16* input_address = nullptr;

    ConvArgs& args = conv_param->args;

    if (pack_num == 1) {
      out_address = out->data<float16>();
      out_scale_address = out->scale();
    }

    int new_group = param.groups;
    int filter_current_pack = filter->shape().num();
    int channel_current_pack = input->shape().channel();

    new_group = i == pack_num - 1
                    ? param.groups - (pack_num - 1) * group_per_pack
                    : group_per_pack;
    filter_current_pack = new_group * filter_per_group;
    channel_current_pack = new_group * filter->shape().channel();

    if (pack_num == 1) {
      input_address = input->data<float16>();
    } else {
      Shape in_shape(NCHW,
                     {1,
                      channel_current_pack,
                      input->shape().height(),
                      input->shape().width()});
      input_address = conv_param->input.mutableData<float16>(FP16, in_shape);
    }

    if (pack_num != 1) {
      Shape shape(
          NHWC,
          {1, out_shape.height(), out_shape.width(), filter_current_pack});
      out_address = conv_param->output.mutableData<float16>(FP16, shape);
      out_scale_address = conv_param->output.scale();
    }
    Shape f_shape(NCHW,
                  {filter_current_pack,
                   filter->shape().channel(),
                   filter->shape().height(),
                   filter->shape().width()});

    Tensor new_filter;
    float* new_filter_data = new_filter.mutableData<float>(FP32, f_shape);
    int filter_hwc = filter->shape().height() * filter->shape().width() *
                     filter->shape().channel();

    memcpy(new_filter_data,
           filter->data<float>() + i * filter_per_pack * filter_hwc,
           filter_current_pack * filter_hwc * sizeof(float));
    new_filter.flush();
    conv_param->filter.mutableData<float>(FP32, f_shape);

    float mem_factor = filter_num_alignment / filter_per_pack;
    conv_param->filter.setMemScale(mem_factor);

    std::vector<float> quant_scale;
    format_filter(
        &new_filter, &(conv_param->filter), new_group, quant_scale, max);
    conv_param->filter.setDataType(INT8);

    Tensor scale;
    Tensor bias;

    int chnnnel_start = i * filter_per_pack;

    Shape s_shape(NC, {1, filter_current_pack});
    float* scale_data = scale.mutableData<float>(FP32, s_shape);
    float* bias_data = bias.mutableData<float>(FP32, s_shape);
    for (int n = 0; n < filter_current_pack; n++) {
      scale_data[n] =
          param.scale()->data<float>()[n + chnnnel_start] * quant_scale[n];
    }
    for (int n = 0; n < filter_current_pack; n++) {
      bias_data[n] = param.bias()->data<float>()[n + chnnnel_start];
    }
    format_bias_scale_new(&bias, &scale, &conv_param->scaleBias);
    conv_param->scaleBias.flush();

    args.group_num = new_group;
    args.sb_address = conv_param->scaleBias.data<float16>();
    args.kernel.stride_h = param.strides[1];
    args.kernel.stride_w = param.strides[0];
    args.kernel.height = new_filter.shape().height();
    args.kernel.width = new_filter.shape().width();

    args.filter_address = conv_param->filter.data<int8_t>();
    args.filter_num = filter_current_pack;
    args.filter_scale_address = conv_param->filter.scale();
    args.image.address = input_address;
    args.image.scale_address = input->scale();
    args.image.channels = channel_current_pack;
    args.image.width = input->shape().width();
    args.image.height = input->shape().height();
    args.image.pad_width = param.paddings[1];
    args.image.pad_height = param.paddings[0];
    args.dilation = param.dilations[0];

    args.output.address = out_address;
    args.output.scale_address = out_scale_address;
    param.splitParams().push_back(conv_param);
  }
}

inline void split_channel(const ConvParam& c_param) {
  ConvParam& param = const_cast<ConvParam&>(c_param);
  Tensor* input = param.input;
  Tensor* output = param.output;
  input->syncToCPU();

  int num = ceil(input->shape().channel() * 1.0f / 2047);
  int channel = input->shape().channel() / num;

  Shape bs_shape(N, {channel});

  float max = 1.0f;

  for (int i = 0; i < num; i++) {
    BasicConvParam* conv_param = new BasicConvParam();

    // input && output;
    Shape in_shape(
        NCHW, {1, channel, input->shape().height(), input->shape().width()});
    conv_param->input.shareDataWith(input, in_shape, channel * i);
    conv_param->output.mutableData<float16>(FP16, output->shape());

    // filter transformation;
    Shape f_shape(NCHW, {param.filter->shape().num(), channel, 1, 1});

    Tensor new_filter;

    float* dst = new_filter.mutableData<float>(FP32, f_shape);
    float* src = param.filter->data<float>() + i * channel;
    for (int n = 0; n < f_shape.num(); n++) {
      memcpy(dst, src, channel * sizeof(float));
      dst += channel;
      src += param.filter->shape().channel();
    }
    new_filter.flush();
    std::vector<float> scales;
    format_filter(
        &new_filter, &(conv_param->filter), param.groups, scales, max);

    Tensor bias;
    Tensor scale;

    float* bias_data = bias.mutableData<float>(FP32, bs_shape);
    float* scale_data = scale.mutableData<float>(FP32, bs_shape);
    for (int c = 0; c < channel; c++) {
      scale_data[c] = scales[c];
      bias_data[c] = param.bias()->data<float>()[c] / num;
    }
    scale.flush();
    bias.flush();
    format_scale_bias(&scale,
                      &bias,
                      &conv_param->filter,
                      &conv_param->scaleBias,
                      param.groups);
    conv_param->scaleBias.flush();

    ConvArgs& args = conv_param->args;
    args.group_num = param.groups;
    args.sb_address = conv_param->scaleBias.data<float>();
    args.kernel.stride_h = param.strides[1];
    args.kernel.stride_w = param.strides[0];
    args.kernel.height = new_filter.shape().height();
    args.kernel.width = new_filter.shape().width();

    args.filter_address = conv_param->filter.data<int8_t>();
    args.filter_num = f_shape.num();
    args.filter_scale_address = conv_param->filter.scale();
    args.image.address = conv_param->input.mutableData<void>();
    args.image.scale_address = conv_param->input.scale();

    args.image.channels = conv_param->input.shape().channel();
    args.image.width = conv_param->input.shape().width();
    args.image.height = conv_param->input.shape().height();
    args.image.pad_width = param.paddings[1];
    args.image.pad_height = param.paddings[0];
    args.dilation = param.dilations[0];
    args.output.address = conv_param->output.mutableData<void>();
    args.output.scale_address = conv_param->output.scale();
    param.splitParams().push_back(conv_param);
  }
}

inline int fill_split_arg(const ConvParam& c_param) {
  ConvParam& param = const_cast<ConvParam&>(c_param);
  Tensor* input = param.input;
  Tensor* output = param.output;

  if (output->shape().dimSize() == 4 && input->shape().channel() > 2047 &&
      input->shape().width() == 1) {
    split_channel(c_param);
    return 1;
  } else if (param.groups == 1) {
    split_filter_num(c_param);
    return 0;
  } else {
    pack_channel_filter(c_param);
    return 0;
  }
}

inline bool compute_conv(const ConvParam& c_conv_params) {
  ConvParam& conv_params = const_cast<ConvParam&>(c_conv_params);
  std::vector<BasicConvParam*>& params = conv_params.splitParams();
  int ret = 0;
  for (auto conv_param : params) {
    ret |= compute_fpga_conv_basic(conv_param->args);
  }
  size_t size = params.size();
  if (ret == 0 && size > 1) {
    Tensor& img = params[0]->output;
    for (int i = 0; i < 1; i++) {
      for (int i = 0; i < img.shape().numel(); i++) {
        float value = half_to_float(img.data<float16>()[i]);
      }
    }
  }
  return ret == 0;
}

inline void dwconv_split_channel(DepthwiseConvSplitParam& param) {  // NOLINT
  Tensor* input = param.input;
  Tensor* output = param.output;
  Tensor* filter = param.filter;
  input->syncToCPU();

  int h_kernel = filter->shape().height();
  int w_kernel = filter->shape().width();
  int c = input->shape().channel();
  int w = input->shape().width();
  int wc_h_kernel = w * c * h_kernel;
  int dwconv_limit = 131072;
  int num = ceil(wc_h_kernel * 1.0f / dwconv_limit);
  while (input->shape().channel() % num != 0) {
    num++;
  }
  int channel = input->shape().channel() / num;
  if (channel % 16 != 0) {
    std::cout << "input channel must div by 16" << std::endl;
    // throw -1;
  }

  Shape bs_shape(N, {channel});

  float16* output_address = nullptr;
  float16* input_address = nullptr;
  float* out_scale_address = nullptr;

  for (int i = 0; i < num; i++) {
    BasicDWConvParam* dwconv_param = new BasicDWConvParam();

    // input && output;
    Shape in_shape(
        NCHW, {1, channel, input->shape().height(), input->shape().width()});
    if (num == 1) {
      input_address = input->data<float16>();
      output_address = output->data<float16>();
      out_scale_address = output->scale();
    } else {
      input_address = dwconv_param->input.mutableData<float16>(FP16, in_shape);
      output_address =
          dwconv_param->output.mutableData<float16>(FP16, in_shape);
      out_scale_address = dwconv_param->output.scale();
    }

    // filter transformation;
    Shape f_shape(NCHW, {channel, 1, h_kernel, w_kernel});

    Tensor split_filter;
    float* split_filter_data = split_filter.mutableData<float>(FP32, f_shape);
    int filter_hwc = h_kernel * w_kernel * channel;

    memcpy(split_filter_data,
           filter->data<float>() + i * filter_hwc,
           filter_hwc * sizeof(float));
    split_filter.flush();

    Tensor split_scale;
    Tensor split_bias;
    float* scale_data = split_scale.mutableData<float>(FP32, bs_shape);
    float* bias_data = split_bias.mutableData<float>(FP32, bs_shape);
    for (int c = 0; c < channel; c++) {
      scale_data[c] = param.scale()->data<float>()[i * channel + c];
      bias_data[c] = param.bias()->data<float>()[i * channel + c];
    }
    split_bias.flush();

    Tensor quantized_filter = dwconv_param->quantizedFilter;
    Tensor quantized_bias = dwconv_param->quantizedBias;
    quantized_filter.mutableData<float16>(FP16, f_shape);
    quantized_bias.mutableData<float16>(FP16, f_shape);

    format_dw_filter(
        &split_filter, &(dwconv_param->quantizedFilter), scale_data);
    format_16_bias(&split_bias, &(dwconv_param->quantizedBias), channel);

    DWconvArgs& args = dwconv_param->args;
    args.bias_address = dwconv_param->quantizedBias.data<float16>();
    args.filter_address = dwconv_param->quantizedFilter.data<float16>();
    args.kernel.width = f_shape.height();
    args.kernel.height = f_shape.width();
    args.kernel.stride_w = param.strides[0];
    args.kernel.stride_h = param.strides[1];
    args.image.address = input_address;
    args.image.channels = channel;
    args.image.height = input->shape().height();
    args.image.width = input->shape().width();
    args.image.pad_width = param.paddings[0];
    args.image.pad_height = param.paddings[1];
    args.image.scale_address = input->scale();
    args.output.address = output_address;
    args.output.scale_address = out_scale_address;
    args.out_width = param.output->shape().width();
    args.out_height = param.output->shape().height();
    args.sub_conv_num = 1;
    param.splitParams().push_back(dwconv_param);
  }
}

}  // namespace zynqmp
}  // namespace paddle

#endif /* conv_process_hpp */
