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

#include "../float16.hpp"
#include "../llapi/bias_scale.h"
#include "../llapi/filter.h"
#include "../llapi/image.h"
#include "../tensor.hpp"

namespace paddle_mobile {
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

inline void format_scale_bias(Tensor* scale, Tensor* bias, Tensor* filter,
                              Tensor* scale_bias, int group) {
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
  size_t mem_size = filter::format_filter(
      &new_data, filter_shape.num(), filter_shape.channel(),
      filter_shape.height(), filter_shape.width(), group, max_value);
  int8_t* src = quantized_filter->mutableData<int8_t>(INT8, filter->shape());
  memcpy(src, new_data, mem_size);
  fpga_free(new_data);
  quantized_filter->flush();
}

inline void format_dw_filter(Tensor* filter, Tensor* quantized_filter,
                             float* scale) {
  int num = filter->shape().num();
  int height = filter->shape().height();
  int width = filter->shape().width();
  auto memory_size = filter->shape().memorySize(sizeof(float));
  auto new_data = (float*)fpga_malloc(memory_size);  // NOLINT
  memcpy(new_data, filter->data<float>(), memory_size);

  filter::format_dwconv_filter(&new_data, num, height, width, scale);
  float16* src = quantized_filter->mutableData<float16>(FP16, filter->shape());
  memcpy(src, new_data, quantized_filter->shape().memorySize(sizeof(float16)));
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
  filter::format_fc_filter(&new_data, filter_shape.num(),
                           filter_shape.channel(), filter_shape.height(),
                           filter_shape.width(), 1, max_value);

  int8_t* src = quantized_filter->mutableData<int8_t>(INT8, filter->shape());
  memcpy(src, new_data, quantized_filter->shape().memorySize(sizeof(int8_t)));
  quantized_filter->flush();
  fpga_free(new_data);
}

inline void fill_split_arg(const ConvParam& c_param) {
  ConvParam& param = const_cast<ConvParam&>(c_param);
  Tensor* input = param.input;
  Tensor* out = param.output;
  Tensor* filter = param.filter;
  auto channel = out->shape().channel();

  int split_num = param.groups == 1 ? get_split_num(param.filter) : 1;
  int filter_num_per_div = get_filter_num_per_div(filter, param.groups);
  int element_num = get_aligned_filter_element_num(filter->shape().channel() *
                                                   filter->shape().height() *
                                                   filter->shape().width());

  Shape& out_shape = out->shape();
  for (int i = 0; i < split_num; i++) {
    BasicConvParam* conv_param = new BasicConvParam();

    int filter_num = filter->shape().num();
    float16* out_address = nullptr;
    int8_t* filter_address = nullptr;
    float* sb_address = nullptr;
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
    Shape f_shape(NCHW, {filter_num, filter->shape().channel(),
                         filter->shape().height(), filter->shape().width()});

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
    filter_address = conv_param->filter.data<int8_t>();
    std::cout << conv_param->filter.scale()[0] << std::endl;
    args.filter_scale_address = conv_param->filter.scale();

    int sb_num = 2 * align_to_x(filter_num, BS_NUM_ALIGNMENT);
    Tensor scale;
    Tensor bias;

    int chnnnel_start = i * filter_num_per_div;

    Shape s_shape(N, {filter_num});
    float* scale_data = scale.mutableData<float>(FP32, s_shape);
    float* bias_data = bias.mutableData<float>(FP32, s_shape);
    for (int i = 0; i < filter_num; i++) {
      scale_data[i] = param.scale()->data<float>()[i + chnnnel_start];
    }
    for (int i = 0; i < filter_num; i++) {
      // bias_data[i] = 0.0f;//TODO
      bias_data[i] = param.bias()->data<float>()[i + chnnnel_start];
    }
    Shape sb_shape(N, {sb_num});
    format_scale_bias(&scale, &bias, &conv_param->filter,
                      &conv_param->scaleBias, param.groups);
    sb_address = conv_param->scaleBias.mutableData<float>(FP32, sb_shape);

    args.group_num = param.groups;
    args.relu_enabled = param.relu.enabled;
    args.sb_address = sb_address;
    args.kernel.stride_h = param.strides[1];
    args.kernel.stride_w = param.strides[0];
    args.kernel.height = new_filter.shape().height();
    args.kernel.width = new_filter.shape().width();

    args.filter_address = filter_address;
    args.filter_num = filter_num;

    args.image.address = input->data<void>();
    args.image.scale_address = input->scale();
    args.image.channels = input->shape().channel();
    args.image.width = input->shape().width();
    args.image.height = input->shape().height();
    args.image.pad_width = param.paddings[0];
    args.image.pad_height = param.paddings[1];

    args.output.address = out_address;
    args.output.scale_address = out_scale_address;
    param.splitParams().push_back(conv_param);
  }
}

inline void fill_split_arg(struct SplitConvArgs* arg, Tensor* input,
                           Tensor* out, Tensor* filter, bool relu_enabled,
                           int group_num, int stride_h, int stride_w,
                           int padding_h, int padding_w, float* bs_ptr) {
  auto input_ptr = input->data<float>();
  auto filter_ptr = filter->data<float>();
  auto out_ptr = out->data<float>();

  arg->group_num = (uint32_t)group_num;
  arg->split_num = group_num == 1 ? get_split_num(filter) : 1;
  arg->filter_num = filter->shape().num();
  arg->output.address = out_ptr;
  arg->output.scale_address = out->scale();
  arg->conv_arg =
      (ConvArgs*)fpga_malloc(arg->split_num * sizeof(ConvArgs));  // NOLINT

  memset(arg->conv_arg, 0, arg->split_num * sizeof(struct ConvArgs));

  arg->concat_arg.image_num = arg->split_num;
  arg->concat_arg.image_out = out_ptr;
  arg->concat_arg.scale_out = out->scale();
  arg->concat_arg.height = out->shape().height();
  arg->concat_arg.width = out->shape().width();

  int n = arg->split_num;
  arg->concat_arg.images_in = (half**)fpga_malloc(n * sizeof(int*));  // NOLINT
  arg->concat_arg.scales_in =
      (float**)fpga_malloc(n * sizeof(float*));  // NOLINT
  arg->concat_arg.channel_num =
      (uint32_t*)fpga_malloc(n * sizeof(uint32_t));  // NOLINT

  auto channel = out->shape().channel();
  int filter_num_per_div = get_filter_num_per_div(filter, group_num);
  int element_num = get_aligned_filter_element_num(filter->shape().channel() *
                                                   filter->shape().height() *
                                                   filter->shape().width());

  for (int i = 0; i < n; i++) {
    arg->conv_arg[i].relu_enabled = relu_enabled;
    arg->conv_arg[i].group_num = (uint32_t)group_num;
    arg->conv_arg[i].kernel.stride_h = (uint32_t)stride_h;
    arg->conv_arg[i].kernel.stride_w = (uint32_t)stride_w;
    arg->conv_arg[i].kernel.height = filter->shape().height();
    arg->conv_arg[i].kernel.width = filter->shape().width();
    arg->conv_arg[i].image.address = input_ptr;
    arg->conv_arg[i].image.channels = input->shape().channel();
    arg->conv_arg[i].image.height = input->shape().height();
    arg->conv_arg[i].image.width = input->shape().width();
    arg->conv_arg[i].image.scale_address = input->scale();
    arg->conv_arg[i].image.pad_height = (uint32_t)padding_h;
    arg->conv_arg[i].image.pad_width = (uint32_t)padding_w;
    arg->conv_arg[i].filter_scale_address = filter->scale();
    arg->conv_arg[i].filter_num = (uint32_t)(
        i == n - 1 ? channel - (n - 1) * filter_num_per_div  // NOLINT
                   : filter_num_per_div);

    size_t filter_size =
        element_num *
        align_to_x(arg->conv_arg[i].filter_num, FILTER_NUM_ALIGNMENT) *
        sizeof(int8_t);
    auto filter_head =
        &((int8_t*)filter_ptr)[i * element_num * filter_num_per_div];  // NOLINT
    arg->conv_arg[i].filter_address = fpga_malloc(filter_size);
    memcpy(arg->conv_arg[i].filter_address, filter_head, filter_size);
    fpga_flush(arg->conv_arg[i].filter_address, filter_size);

    size_t bs_size = 2 *
                     align_to_x(arg->conv_arg[i].filter_num, BS_NUM_ALIGNMENT) *
                     sizeof(float);
    auto bs_head = &bs_ptr[i * filter_num_per_div * 2];
    arg->conv_arg[i].sb_address = fpga_malloc(bs_size);
    memcpy(arg->conv_arg[i].sb_address, bs_head, bs_size);
    fpga_flush(arg->conv_arg[i].sb_address, bs_size);

    if (n > 1) {
      arg->conv_arg[i].output.scale_address =
          (float*)fpga_malloc(2 * sizeof(float));  // NOLINT
      arg->conv_arg[i].output.address = fpga_malloc(
          out->shape().height() *
          align_to_x(out->shape().width() * arg->conv_arg[i].filter_num,
                     IMAGE_ALIGNMENT) *
          sizeof(half));
    } else {
      arg->conv_arg[i].output.scale_address = out->scale();
      arg->conv_arg[i].output.address = out_ptr;
    }

    arg->concat_arg.images_in[i] =
        (half*)arg->conv_arg[i].output.address;  // NOLINT
    arg->concat_arg.scales_in[i] = arg->conv_arg[i].output.scale_address;
    arg->concat_arg.channel_num[i] = arg->conv_arg[i].filter_num;
  }
}

inline int do_concat(const struct ConcatArgs& args) {
  image::concat_images(args.images_in, args.scales_in, args.image_out,
                       args.scale_out, args.image_num, args.channel_num,
                       args.height, args.width);
  return 0;
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
    Tensor* output = conv_params.output;

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

inline bool compute_conv(const SplitConvArgs& args) {
  int ret = 0;
  int split_num = args.split_num;
  for (int i = 0; i < split_num; i++) {
    ret |= compute_fpga_conv_basic(args.conv_arg[i]);
  }

  if (split_num > 1) {
    do_concat(args.concat_arg);
  }
  return ret == 0;
}

}  // namespace zynqmp
}  // namespace paddle_mobile

#endif /* conv_process_hpp */
