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

enum DCpuConcatType {
  DISABLED = 0,
  ALIGNED = 1,
  UNALIGNED = 2,
};

const int MAX_CHANNEL = 16384;

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
  int channel = param_->filter->shape().num();
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

inline void format_dw_filter(Tensor* filter,
                             Tensor* quantized_filter,
                             Tensor* scale,
                             int dynamic_range) {
  float max_filter = find_max(*filter);
  float max_scale = 1.0f;
  float* scale_ptr = nullptr;
  if (scale != nullptr) {
    max_scale = find_max(*scale);
    scale_ptr = scale->data<float>();
  }

  int num = filter->shape().num();
  int height = filter->shape().height();
  int width = filter->shape().width();

  float quant_scale = dynamic_range / (max_filter * max_scale);
  float dequant_scale = (max_filter * max_scale) / dynamic_range;

  int mem_size;
  int16_t* quantized_data = filter::format_dwconv_filter(filter->data<float>(),
                                                         num,
                                                         height,
                                                         width,
                                                         scale_ptr,
                                                         mem_size,
                                                         quant_scale);

  float mem_factor = mem_size * 1.0f / filter->shape().numel();
  quantized_filter->setMemScale(mem_factor);

  quantized_filter->setAligned(true);
  int16_t* dst = quantized_filter->mutableData<int16_t>(INT16, filter->shape());

  quantized_filter->scale()[0] = dequant_scale;
  quantized_filter->scale()[1] = quant_scale;

  memcpy(dst, quantized_data, mem_size);
  quantized_filter->flush();
  fpga_free(quantized_data);
}

inline void config_basic_conv_input_args(BasicConvParam* conv_param,
                                         void* input_address,
                                         void* scale_address,
                                         int channels,
                                         int width,
                                         int height,
                                         int pad_h,
                                         int pad_w) {
  ConvArgs& args = conv_param->args;
  args.image.address = input_address;
  args.image.scale_address = scale_address;
  args.image.channels = channels;
  args.image.width = width;
  args.image.height = height;
  args.image.pad_width = pad_w;
  args.image.pad_height = pad_h;
}

inline void config_basic_conv_filter(BasicConvParam* conv_param,
                                     void* fllter_address,
                                     int filter_num,
                                     void* scale_address) {
  ConvArgs& args = conv_param->args;
  args.filter_address = fllter_address;
  args.filter_num = filter_num;
  args.filter_scale_address = scale_address;
}

inline void config_basic_conv_kernel(BasicConvParam* conv_param,
                                     int dilation,
                                     int groups,
                                     void* scalebias,
                                     int stride_h,
                                     int stride_w,
                                     int h,
                                     int w) {
  ConvArgs& args = conv_param->args;
  args.group_num = groups;
  args.sb_address = scalebias;
  args.kernel.stride_h = stride_h;
  args.kernel.stride_w = stride_h;
  args.kernel.height = h;
  args.kernel.width = w;
  args.dilation = dilation;
}

inline void config_basic_conv_stride_info(BasicConvParam* conv_param,
                                          bool wr_enable,
                                          bool rd_enable,
                                          int offset) {
  ConvArgs& args = conv_param->args;
  args.stride.rd_enabled = rd_enable;
  args.stride.wr_enabled = wr_enable;
  args.stride.wr_offset = offset;
}

inline void config_basic_conv_deconv_info(BasicConvParam* conv_param,
                                          bool deconv_enable,
                                          int sub_conv_number = 0,
                                          int omit_size = 0) {
  ConvArgs& args = conv_param->args;
  args.deconv.enabled = deconv_enable;
  if (deconv_enable) {
    args.deconv.sub_kernel_num = sub_conv_number;
    args.deconv.invalid_col_num = omit_size;
  }
}

inline void config_basic_conv_output(BasicConvParam* conv_param,
                                     void* out_address,
                                     float16* out_scale_address) {
  ConvArgs& args = conv_param->args;
  args.output.address = out_address;
  args.output.scale_address = out_scale_address;
}

inline DCpuConcatType split_filter_num(const ConvParam& c_param,
                                       int start_pos = 0,
                                       bool deconv = false,
                                       int kernel_num = 0,
                                       int omit_size = 0,
                                       int sub_conv_number = 0) {
  DCpuConcatType deconv_concat_type = DCpuConcatType::DISABLED;

  ConvParam& param = const_cast<ConvParam&>(c_param);
  Tensor* input = param.input;
  Tensor* out = param.output;
  Tensor* filter = param.filter;
  auto out_channel = filter->shape().num();
  int split_num = get_split_num(param.filter);
  int filter_num_per_div = get_filter_num_per_div(filter, param.groups);

  param.cpu_concat =
      out_channel % 16 != 0 && split_num > 1 && out->shape().width() != 1;

  bool deconv_out_reshape = deconv && split_num > 1;

  int dynamic_range = 127;  // int8 max value
  float16 dynamic_range_fp16 = float_to_half(dynamic_range * 1.0);
  float inv_dynamic_range = 1.0 / dynamic_range;
  float max = find_max(*filter);

  Shape& out_shape = out->shape();
  int out_w = out_shape.width();
  int out_h = out_shape.height();
  if (deconv_out_reshape) {
    // stride is 1 in this case.
    Shape in_shape = input->shape();
    out_w = input->shape().width() + 2 * param.paddings[1] -
            filter->shape().width() + 1;
    out_h = input->shape().height() + 2 * param.paddings[0] -
            filter->shape().height() + 1;
  }

  // support jump write
  int jump_out_start_offset = param.original_out_channel;
  int fuse_idx = param.fuse_idx;
  bool enable_jump = param.wd_enable;

  if (deconv) {
    // TODO(chengruichang) jump write is not support in deconv mode
    enable_jump = false;
  }

  float16* out_base_address = nullptr;
  for (int i = 0; i < split_num; i++) {
    BasicConvParam* conv_param = new BasicConvParam();
    conv_param->output.setDataLocation(Device);
    conv_param->output.setAligned(true);
    int filter_num = filter->shape().num();
    float16* out_address = nullptr;
    float16* out_scale_address = nullptr;
    ConvArgs& args = conv_param->args;

    if (i == split_num - 1) {
      filter_num = out_channel - (split_num - 1) * filter_num_per_div;
    } else {
      filter_num = filter_num_per_div;
    }
    int offset = i * filter_num_per_div;

    // jump write between op is disabled in this branch
    if (param.cpu_concat) {
      if (i == 0) {
        Shape shape(NHWC,
                    {1, out_h, out_w, filter_num_per_div * (split_num - 1)});
        out_base_address = conv_param->output.mutableData<float16>(FP16, shape);
      }
      if (i == split_num - 1) {
        Shape extra_shape(NHWC, {1, out_h, out_w, filter_num});
        out_address =
            conv_param->output.mutableData<float16>(FP16, extra_shape);
      } else {
        // base_address is allocated in first iteration(i = 0);
        out_address = out_base_address + offset;
      }
    } else if (deconv_out_reshape) {
      // deconv mode with split num > 1 and no residual channels
      // deconv mode does not support cross ops jump write
      if (i == 0) {
        Shape shape(NHWC, {1, out_h, out_w, filter_num_per_div * split_num});
        out_base_address = conv_param->output.mutableData<float16>(FP16, shape);
      }
      out_address = out_base_address + offset;
    } else if (!enable_jump) {
      // deconv mode,jump write disabled,split num is 1 or normal conv without
      // residual channels
      // for single conv op
      out_address = out->data<float16>() + offset;
    } else {
      // normal conv with jump write enabled across ops
      out_address = out->data<float16>() + offset + jump_out_start_offset;
    }

    // support deconv sub filter split num
    if (deconv && !deconv_out_reshape) {
      out_address = out_address + start_pos;
    }

    out_scale_address = &conv_param->output_max;

    Shape f_shape(NCHW,
                  {filter_num,
                   filter->shape().channel(),
                   filter->shape().height(),
                   filter->shape().width()});

    Tensor new_filter;
    float* new_filter_data = new_filter.mutableData<float>(FP32, f_shape);
    int filter_hwc = filter->shape().height() * filter->shape().width() *
                     filter->shape().channel();
    bool wd_enable = param.wd_enable;

    memcpy(new_filter_data,
           filter->data<float>() + i * filter_num_per_div * filter_hwc,
           filter_num * filter_hwc * sizeof(float));
    new_filter.flush();

    conv_param->filter.mutableData<float>(FP32, f_shape);

    std::vector<float> quant_scale;
    format_filter(
        &new_filter, &(conv_param->filter), param.groups, quant_scale, max);
    conv_param->filter.flush();
    conv_param->filter.setDataType(INT8);

    int sb_channnel_start = i * filter_num_per_div;
    Shape s_shape(NC, {1, filter_num});
    Tensor scale;
    Tensor bias;
    float* scale_data = scale.mutableData<float>(FP32, s_shape);
    float* bias_data = bias.mutableData<float>(FP32, s_shape);

    for (int n = 0; n < filter_num; n++) {
      int nn = n;
      if (deconv && !deconv_out_reshape) {
        nn = (kernel_num * omit_size + n) % filter_num;
      }
      scale_data[n] =
          param.scale()->data<float>()[n + sb_channnel_start] * quant_scale[nn];
    }
    for (int n = 0; n < filter_num; n++) {
      bias_data[n] = param.bias()->data<float>()[n + sb_channnel_start];
    }
    bias.flush();
    scale.flush();

    format_bias_scale_new(&bias, &scale, &conv_param->scaleBias);
    conv_param->scaleBias.flush();

    config_basic_conv_kernel(conv_param,
                             param.dilations[0],
                             param.groups,
                             conv_param->scaleBias.data<float16>(),
                             param.strides[1],
                             param.strides[0],
                             new_filter.shape().height(),
                             new_filter.shape().width());

    config_basic_conv_filter(conv_param,
                             conv_param->filter.data<int8_t>(),
                             filter_num,
                             conv_param->filter.scale());

    config_basic_conv_input_args(conv_param,
                                 input->data<void>(),
                                 input->max(),
                                 input->shape().channel(),
                                 input->shape().width(),
                                 input->shape().height(),
                                 param.paddings[0],
                                 param.paddings[1]);

    config_basic_conv_deconv_info(conv_param, false);
    if (enable_jump) {
      config_basic_conv_stride_info(conv_param, true, false, param.wd_offset);
    } else {
      bool wr_enable =
          (split_num != 1 && (param.cpu_concat == false || i != split_num - 1));
      int wd_offset =
          param.cpu_concat ? filter_num_per_div * (split_num - 1) : out_channel;
      config_basic_conv_stride_info(conv_param, wr_enable, false, wd_offset);
    }

    args.quant.dynamic_range =
        *(reinterpret_cast<uint16_t*>(&dynamic_range_fp16));
    args.quant.inv_dynamic_range =
        *(reinterpret_cast<uint32_t*>(&inv_dynamic_range));

    config_basic_conv_output(conv_param, out_address, out_scale_address);

    if (deconv && !deconv_out_reshape) {
      config_basic_conv_deconv_info(
          conv_param, true, sub_conv_number, omit_size);
    }

    param.splitParams().push_back(conv_param);
  }
  if (deconv && param.cpu_concat) deconv_concat_type = DCpuConcatType::ALIGNED;
  if (deconv_out_reshape) deconv_concat_type = DCpuConcatType::UNALIGNED;
  return deconv_concat_type;
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

  int dynamic_range = 127;  // int8 max value
  float16 dynamic_range_fp16 = float_to_half(dynamic_range * 1.0);
  float inv_dynamic_range = 1.0 / dynamic_range;
  float max = find_max(*filter);
  Shape& out_shape = out->shape();

  for (int i = 0; i < pack_num; i++) {
    BasicConvParam* conv_param = new BasicConvParam();

    conv_param->output.setDataLocation(Device);
    conv_param->output.setAligned(true);

    float16* out_address = nullptr;
    float16* out_scale_address = nullptr;

    float16* input_address = nullptr;

    ConvArgs& args = conv_param->args;

    if (pack_num == 1) {
      out_address = out->data<float16>();
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
    }
    out_scale_address = &conv_param->output_max;
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
    args.image.scale_address = input->max();
    args.image.channels = channel_current_pack;
    args.image.width = input->shape().width();
    args.image.height = input->shape().height();
    args.image.pad_width = param.paddings[1];
    args.image.pad_height = param.paddings[0];
    args.dilation = param.dilations[0];
    args.deconv.enabled = false;
    args.stride.rd_enabled = false;
    args.stride.wr_enabled = false;

    args.output.address = out_address;
    args.output.scale_address = out_scale_address;
    args.quant.dynamic_range =
        *(reinterpret_cast<uint16_t*>(&dynamic_range_fp16));
    args.quant.inv_dynamic_range =
        *(reinterpret_cast<uint32_t*>(&inv_dynamic_range));
    param.splitParams().push_back(conv_param);
  }
}

inline void split_channel(const ConvParam& c_param) {
  ConvParam& param = const_cast<ConvParam&>(c_param);
  Tensor* input = param.input;
  Tensor* output = param.output;
  input->syncToCPU();

  int dynamic_range = 127;  // int8 max value
  float16 dynamic_range_fp16 = float_to_half(dynamic_range * 1.0);
  float inv_dynamic_range = 1.0 / dynamic_range;

  Tensor* filter = param.filter;
  int num = ceil(input->shape().channel() * 1.0f / 2047);
  if (output->shape().dimSize() == 2) {
    num = ceil(input->shape().numel() * 1.0f / 16384);
  }
  int channel = input->shape().channel() / num;
  Shape bs_shape(NC, {1, output->shape().channel()});
  float max = find_max(*param.filter);

  for (int i = 0; i < num; i++) {
    BasicConvParam* conv_param = new BasicConvParam();
    Shape in_shape(
        NCHW, {1, channel, input->shape().height(), input->shape().width()});
    conv_param->input.mutableData<float16>(FP16, in_shape);
    conv_param->output.mutableData<float16>(FP16, output->shape());

    // filter transformation;split filters by channel;
    Shape f_shape(NCHW,
                  {filter->shape().num(),
                   channel,
                   filter->shape().height(),
                   filter->shape().width()});

    Tensor new_filter_hwc;
    auto cal_chw = [](Tensor* t) {
      Shape& s = t->shape();
      return s.channel() * s.height() * s.width();
    };

    for (int n = 0; n < f_shape.num(); n++) {
      float* dst = new_filter_hwc.mutableData<float>(FP32, f_shape) +
                   n * cal_chw(&new_filter_hwc);
      float* src = filter->data<float>() + i * channel + n * cal_chw(filter);

      for (int hw = 0; hw < f_shape.height() * f_shape.width(); hw++) {
        memcpy(dst, src, channel * sizeof(float));
        dst += channel;
        src += filter->shape().channel();
      }
    }
    Tensor new_filter;
    hwc_to_chw(&new_filter_hwc, &new_filter);
    std::vector<float> scales;
    format_filter(
        &new_filter, &(conv_param->filter), param.groups, scales, max);

    conv_param->filter.flush();

    Tensor bias;
    Tensor scale;

    int sb_channel = output->shape().channel();
    float* bias_data = bias.mutableData<float>(FP32, bs_shape);
    float* scale_data = scale.mutableData<float>(FP32, bs_shape);
    for (int c = 0; c < sb_channel; c++) {
      scale_data[c] = param.scale()->data<float>()[c] * scales[c];
      bias_data[c] = param.bias()->data<float>()[c] / num;
    }
    format_bias_scale_new(&bias, &scale, &conv_param->scaleBias);
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
    args.image.scale_address = conv_param->input.max();
    args.image.channels = conv_param->input.shape().channel();
    args.image.width = conv_param->input.shape().width();
    args.image.height = conv_param->input.shape().height();
    args.image.pad_width = param.paddings[1];
    args.image.pad_height = param.paddings[0];
    args.dilation = param.dilations[0];
    args.output.address = conv_param->output.mutableData<void>();
    args.output.scale_address = conv_param->output.max();
    args.deconv.enabled = false;
    args.stride.rd_enabled = false;
    args.stride.wr_enabled = false;
    args.quant.dynamic_range =
        *(reinterpret_cast<uint16_t*>(&dynamic_range_fp16));
    args.quant.inv_dynamic_range =
        *(reinterpret_cast<uint32_t*>(&inv_dynamic_range));
    param.splitParams().push_back(conv_param);
  }
}

inline int fill_split_arg(const ConvParam& c_param) {
  ConvParam& param = const_cast<ConvParam&>(c_param);
  Tensor* input = param.input;
  Tensor* output = param.output;

  if ((output->shape().dimSize() == 4 && input->shape().channel() > 2047 &&
       input->shape().width() == 1) ||
      (output->shape().dimSize() == 2 &&
       input->shape().numel() > MAX_CHANNEL)) {
    split_channel(c_param);
    return 1;
  } else if (param.groups == 1) {
    split_filter_num(c_param);
    return 0;
  } else {
    pack_channel_filter(c_param);
    return 2;
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

}  // namespace zynqmp
}  // namespace paddle

#endif /* conv_process_hpp */
