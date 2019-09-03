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
  return filter::calc_split_num(num, div_capacity);
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
  Shape bias_scale_shape(N, {2 * channel});
  float* bs_data = scale_bias->mutableData<float>(FP32, bias_scale_shape);
  for (int i = 0; i < channel; i++) {
    float scale_value = scale_data == nullptr ? 1 : scale_data[i];
    float bias_value = bias_data == nullptr ? 0 : bias_data[i];
    bs_data[i + channel] = scale_value;
    bs_data[i] = bias_value;
  }

  int element_num_per_div = get_filter_num_per_div(filter, group);
  bias_scale::format_bias_scale_array(&bs_data, element_num_per_div, channel);
}

inline void format_filter(Tensor* filter, Tensor* quantized_filter, int group) {
  float max_value = find_max(*filter);
  Shape& filter_shape = filter->shape();
  quantized_filter->setAligned(true);
  quantized_filter->mutableData<int8_t>(INT8, filter->shape());
  quantized_filter->scale()[0] = max_value / 127.0f;
  quantized_filter->scale()[1] = 127.0f / max_value;

  auto memory_size = filter->shape().memorySize(sizeof(float));
  auto new_data = reinterpret_cast<float*>(fpga_malloc(memory_size));
  memcpy(new_data, filter->data<float>(), memory_size);
  size_t mem_size = filter::format_filter(&new_data,
                                          filter_shape.num(),
                                          filter_shape.channel(),
                                          filter_shape.height(),
                                          filter_shape.width(),
                                          group,
                                          max_value);
  int8_t* src = quantized_filter->mutableData<int8_t>(INT8, filter->shape());
  memcpy(src, new_data, mem_size);
  fpga_free(new_data);
  quantized_filter->flush();
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

  int split_num = param.groups == 1 ? get_split_num(param.filter) : 1;
  int filter_num_per_div = get_filter_num_per_div(filter, param.groups);

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
    format_filter(&new_filter, &(conv_param->filter), param.groups);

    int sb_num = 2 * align_to_x(filter_num, BS_NUM_ALIGNMENT);
    Tensor scale;
    Tensor bias;

    int chnnnel_start = i * filter_num_per_div;

    Shape s_shape(N, {filter_num});
    float* scale_data = scale.mutableData<float>(FP32, s_shape);
    float* bias_data = bias.mutableData<float>(FP32, s_shape);
    for (int n = 0; n < filter_num; n++) {
      scale_data[n] = param.scale()->data<float>()[n + chnnnel_start];
    }
    for (int n = 0; n < filter_num; n++) {
      bias_data[n] = param.bias()->data<float>()[n + chnnnel_start];
    }
    Shape sb_shape(N, {sb_num});
    format_scale_bias(&scale,
                      &bias,
                      &conv_param->filter,
                      &conv_param->scaleBias,
                      param.groups);
    conv_param->scaleBias.flush();

    args.group_num = param.groups;
    args.relu_enabled = param.relu.enabled;
    args.sb_address = conv_param->scaleBias.data<float>();
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
  std::cout << "channel::" << channel << "num::" << num << std::endl;
  Shape bs_shape(N, {channel});

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
    format_filter(&new_filter, &(conv_param->filter), param.groups);

    Tensor bias;
    Tensor scale;

    float* bias_data = bias.mutableData<float>(FP32, bs_shape);
    float* scale_data = scale.mutableData<float>(FP32, bs_shape);
    for (int c = 0; c < channel; c++) {
      scale_data[c] = 1;
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
    args.relu_enabled = param.relu.enabled;
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
  } else {
    split_filter_num(c_param);
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
        std::cout << "value:" << value << std::endl;
      }
    }
  }
  return ret == 0;
}

}  // namespace zynqmp
}  // namespace paddle
