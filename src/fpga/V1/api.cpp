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

#include "fpga/V1/api.h"
#include "fpga/V1/bias_scale.h"
#include "fpga/V1/deconv_filter.h"
#include "fpga/V1/filter.h"
#include "fpga/V1/image.h"

namespace paddle_mobile {
namespace fpga {

int get_align_image_cw(int cw) { return align_to_x(cw, IMAGE_ALIGNMENT); }

void format_image(framework::Tensor *image_tensor) {
  auto dims = image_tensor->dims();
  auto channel = dims[1], height = dims[2], width = dims[3];
  auto data_ptr = image_tensor->data<float>();
  size_t memory_size = channel * height * width * sizeof(float);
  auto new_data = (float *)fpga_malloc(memory_size);  // NOLINT
  fpga_copy(new_data, data_ptr, memory_size);
  image::format_image(&new_data, channel, height, width);
  image_tensor->reset_data_ptr(new_data);
}

void format_fp16_ofm(framework::Tensor *ofm_tensor) {
  auto dims = ofm_tensor->dims();
  size_t memory_size = 0;
  if (dims.size() == 4) {
    auto channel = dims[1], height = dims[2], width = dims[3];
    memory_size =
        height * align_to_x(channel * width, IMAGE_ALIGNMENT) * sizeof(half);
  } else if (dims.size() == 2) {
    memory_size = align_to_x(dims[1], IMAGE_ALIGNMENT) * sizeof(half);
  } else {
    DLOG << "Wrong ofm dimension";
  }
  auto p = fpga_malloc(memory_size);
  memset(p, 0, memory_size);
  ofm_tensor->reset_data_ptr(p);
}

void format_fp32_ofm(framework::Tensor *ofm_tensor) {
  auto dims = ofm_tensor->dims();
  size_t memory_size = 0;
  if (dims.size() == 4) {
    auto channel = dims[1], height = dims[2], width = dims[3];
    memory_size =
        height * align_to_x(channel * width, IMAGE_ALIGNMENT) * sizeof(float);
  } else if (dims.size() == 2) {
    memory_size = align_to_x(dims[1], IMAGE_ALIGNMENT) * sizeof(float);
  } else {
    DLOG << "Wrong ofm dimension";
  }
  auto p = fpga_malloc(memory_size);
  memset(p, 0, memory_size);
  ofm_tensor->reset_data_ptr(p);
}

float filter_find_max(framework::Tensor *filter_tensor) {
  auto filter_ptr = filter_tensor->data<float>();
  return filter::find_max(filter_ptr, filter_tensor->numel());
}

int get_plit_num(framework::Tensor *filter_tensor) {
  auto dims = filter_tensor->dims();
  auto chw = dims[1] * dims[2] * dims[3];
  auto num = dims[0];
  int div_capacity = filter::calc_division_capacity(chw);
  return filter::calc_split_num(num, div_capacity);
}

int get_filter_num_per_div(framework::Tensor *filter_tensor, int group_num) {
  auto dims = filter_tensor->dims();
  auto chw = dims[1] * dims[2] * dims[3];
  auto num = dims[0];
  int div_capacity = filter::calc_division_capacity(chw);
  return filter::calc_num_per_div(num, group_num, div_capacity);
}

int get_aligned_filter_element_num(int chw) {
  return align_to_x(chw, FILTER_ELEMENT_ALIGNMENT);
}

int get_aligned_filter_num(int num) {
  return align_to_x(num, FILTER_NUM_ALIGNMENT);
}

void format_filter(framework::Tensor *filter_tensor, float max_value,
                   int group_num) {
  filter_tensor->scale[0] = float(max_value / 127.0);  // NOLINT
  filter_tensor->scale[1] = float(127.0 / max_value);  // NOLINT
  auto dims = filter_tensor->dims();
  auto num = dims[0], channel = dims[1], height = dims[2], width = dims[3];
  auto data_ptr = filter_tensor->data<float>();
  size_t memory_size = num * channel * height * width * sizeof(float);
  auto new_data = (float *)fpga_malloc(memory_size);  // NOLINT
  fpga_copy(new_data, data_ptr, memory_size);
  filter::format_filter(&new_data, num, channel, height, width, group_num,
                        max_value);
  filter_tensor->reset_data_ptr(new_data);
}

void format_fc_filter(framework::Tensor *filter_tensor, float max_value) {
  filter_tensor->scale[0] = float(max_value / 127.0);  // NOLINT
  filter_tensor->scale[1] = float(127.0 / max_value);  // NOLINT
  auto dims = filter_tensor->dims();
  auto num = dims[0], channel = dims[1], height = dims[2], width = dims[3];
  auto data_ptr = filter_tensor->data<float>();
  size_t memory_size = num * channel * height * width * sizeof(float);
  auto new_data = (float *)fpga_malloc(memory_size);  // NOLINT
  fpga_copy(new_data, data_ptr, memory_size);
  filter::format_fc_filter(&new_data, num, channel, height, width, 1,
                           max_value);
  filter_tensor->reset_data_ptr(new_data);
}
void format_deconv_filter(framework::Tensor *filter_tensor, float max_value,
                          int group_num, int stride) {
  filter_tensor->scale[0] = float(max_value / 127.0);  // NOLINT
  filter_tensor->scale[1] = float(127.0 / max_value);  // NOLINT
  auto dims = filter_tensor->dims();
  auto num = dims[0], channel = dims[1], height = dims[2], width = dims[3];
  auto data_ptr = filter_tensor->data<float>();
  size_t memory_size = num * channel * height * width * sizeof(float);
  auto new_data = (float *)fpga_malloc(memory_size);  // NOLINT
  memcpy(new_data, data_ptr, memory_size);

  int hw = height * width;
  deconv_filter::deconv_NC_convert(&new_data, num, channel, hw);

  num = dims[1];
  channel = dims[0];
  deconv_filter::deconv_format_filter(
      &new_data, (int)num, (int)channel,          // NOLINT
      (int)height,                                // NOLINT
      (int)width, group_num, max_value, stride);  // NOLINT

  framework::DDim dims_new =
      framework::make_ddim({num, channel, height, width});
  filter_tensor->Resize(dims_new);
  filter_tensor->reset_data_ptr(new_data);
}

void format_bias_scale_array(float **bias_scale_array,
                             int element_num_per_division, int num) {
  bias_scale::format_bias_scale_array(bias_scale_array,
                                      element_num_per_division, num);
}

void format_concat_output(framework::Tensor *out, int height, int width,
                          int image_num, uint32_t *channel_num) {
  int sum_channel = 0, sum_cw = 0;
  for (int i = 0; i < image_num; i++) {
    sum_channel += channel_num[i];
  }

  sum_cw = align_to_x(width * sum_channel, IMAGE_ALIGNMENT);
  auto data_ptr = fpga_malloc(height * sum_cw * sizeof(half));
  auto ddim = framework::make_ddim({1, sum_channel, height, width});
  out->Resize(ddim);
  out->reset_data_ptr(data_ptr);
}

void fill_split_arg(struct SplitConvArgs *arg, framework::Tensor *input,
                    framework::Tensor *out, framework::Tensor *filter,
                    bool relu_enabled, int group_num, int stride_h,
                    int stride_w, int padding_h, int padding_w, float *bs_ptr) {
  auto input_ptr = input->data<float>();
  auto filter_ptr = filter->data<float>();
  auto out_ptr = out->data<float>();

  arg->group_num = (uint32_t)group_num;
  // Either group_num or split_num = 1;
  arg->split_num = group_num == 1 ? (uint32_t)get_plit_num(filter) : 1;
  arg->filter_num = (uint32_t)filter->dims()[0];
  arg->output.address = out_ptr;
  arg->output.scale_address = out->scale;
  arg->conv_arg =
      (ConvArgs *)fpga_malloc(arg->split_num * sizeof(ConvArgs));  // NOLINT

  arg->concat_arg.image_num = arg->split_num;
  arg->concat_arg.image_out = out_ptr;
  arg->concat_arg.scale_out = out->scale;
  arg->concat_arg.height = (uint32_t)out->dims()[2];
  arg->concat_arg.width = (uint32_t)out->dims()[3];

  int n = arg->split_num;
  arg->concat_arg.images_in =
      (half **)fpga_malloc(n * sizeof(int *));  // NOLINT
  arg->concat_arg.scales_in =
      (float **)fpga_malloc(n * sizeof(float *));  // NOLINT
  arg->concat_arg.channel_num =
      (uint32_t *)fpga_malloc(n * sizeof(uint32_t));  // NOLINT

  auto channel = (int)out->dims()[1];  // NOLINT
  int filter_num_per_div = get_filter_num_per_div(filter, group_num);
  int element_num = get_aligned_filter_element_num(
      filter->dims()[1] * filter->dims()[2] * filter->dims()[3]);

  for (int i = 0; i < n; i++) {
    arg->conv_arg[i].relu_enabled = relu_enabled;
    arg->conv_arg[i].group_num = (uint32_t)group_num;
    arg->conv_arg[i].kernel.stride_h = (uint32_t)stride_h;
    arg->conv_arg[i].kernel.stride_w = (uint32_t)stride_w;
    arg->conv_arg[i].kernel.height = (uint32_t)filter->dims()[2];
    arg->conv_arg[i].kernel.width = (uint32_t)filter->dims()[3];
    arg->conv_arg[i].image.address = input_ptr;
    arg->conv_arg[i].image.channels = (uint32_t)input->dims()[1];
    arg->conv_arg[i].image.height = (uint32_t)input->dims()[2];
    arg->conv_arg[i].image.width = (uint32_t)input->dims()[3];
    arg->conv_arg[i].image.scale_address = input->scale;
    arg->conv_arg[i].image.pad_height = (uint32_t)padding_h;
    arg->conv_arg[i].image.pad_width = (uint32_t)padding_w;
    arg->conv_arg[i].filter_scale_address = filter->scale;
    //    arg->conv_arg[i].filter_address = &(
    //        (int8_t *)filter_ptr)[i * element_num * filter_num_per_div];  //
    //        NOLINT
    //    arg->conv_arg[i].sb_address = &bs_ptr[i * filter_num_per_div * 2];

    arg->conv_arg[i].filter_num = (uint32_t)(
        i == n - 1 ? channel - (n - 1) * filter_num_per_div  // NOLINT
                   : filter_num_per_div);

    size_t filter_size =
        element_num * arg->conv_arg[i].filter_num * sizeof(int8_t);
    auto filter_head =
        &((int8_t *)filter_ptr)[i * element_num * filter_num_per_div];
    arg->conv_arg[i].filter_address = fpga_malloc(filter_size);
    memcpy(arg->conv_arg[i].filter_address, filter_head, filter_size);
    fpga_flush(arg->conv_arg[i].filter_address, filter_size);

    size_t bs_size = 2 * arg->conv_arg[i].filter_num * sizeof(float);
    auto bs_head = &bs_ptr[i * filter_num_per_div * 2];
    arg->conv_arg[i].sb_address = fpga_malloc(bs_size);
    memcpy(arg->conv_arg[i].sb_address, bs_head, bs_size);
    fpga_flush(arg->conv_arg[i].sb_address, bs_size);

    if (n > 1) {
      arg->conv_arg[i].output.scale_address =
          (float *)fpga_malloc(2 * sizeof(float));  // NOLINT
      arg->conv_arg[i].output.address =
          fpga_malloc(out->dims()[2] *
                      align_to_x(out->dims()[3] * arg->conv_arg[i].filter_num,
                                 IMAGE_ALIGNMENT) *
                      sizeof(half));
    } else {
      arg->conv_arg[i].output.scale_address = out->scale;
      arg->conv_arg[i].output.address = out_ptr;
    }

    arg->concat_arg.images_in[i] =
        (half *)arg->conv_arg[i].output.address;  // NOLINT
    arg->concat_arg.scales_in[i] = arg->conv_arg[i].output.scale_address;
    arg->concat_arg.channel_num[i] = arg->conv_arg[i].filter_num;
  }
  filter->reset_data_ptr(nullptr);
  fpga_free(bs_ptr);
}
void fill_deconv_arg(struct DeconvArgs *arg, framework::Tensor *input,
                     framework::Tensor *out, framework::Tensor *filter,
                     bool relu_enabled, int group_num, int stride_h,
                     int stride_w, int padding_h, int padding_w,
                     float *bs_ptr) {
  auto input_ptr = input->data<float>();
  auto filter_ptr = filter->data<float>();
  auto out_ptr = out->data<float>();

  arg->group_num = (uint32_t)group_num;
  arg->sub_conv_num = stride_h;
  arg->filter_num = (uint32_t)filter->dims()[0];

  int sub_conv_num = arg->sub_conv_num;
  int sub_stride = 1;
  int sub_pad = deconv_filter::deconv_calc_sub_pad(filter->dims()[3], padding_w,
                                                   stride_w);
  int sub_filter_width =
      deconv_filter::deconv_get_sub_filter_axis(filter->dims()[3], stride_w);

  int sub_output_width = deconv_filter::deconv_get_sub_out_axis(
      input->dims()[3], sub_pad, sub_filter_width);
  int sub_output_height = deconv_filter::deconv_get_sub_out_axis(
      input->dims()[2], sub_pad, sub_filter_width);

  arg->sub_output_width = sub_output_width;
  arg->sub_output_height = sub_output_height;
  arg->omit_size =
      deconv_filter::deconv_get_omit(stride_w, filter->dims()[3], padding_w);
  arg->conv_args = (ConvArgs *)fpga_malloc(sub_conv_num * sizeof(ConvArgs));

  int sub_channels = (int32_t)input->dims()[1];
  int omit_size = arg->omit_size;
  int real_out_width = sub_output_width * sub_conv_num - 2 * omit_size;
  int real_out_height = sub_output_height * sub_conv_num - 2 * omit_size;
  int sub_filter_num = sub_conv_num * (arg->filter_num);

  int conv_output_size =
      (align_to_x(sub_output_width * sub_filter_num, IMAGE_ALIGNMENT)) *
      sub_output_height;
  int ouput_size = conv_output_size * sub_conv_num;

  int align_sub_filter_num = align_to_x(sub_filter_num, FILTER_NUM_ALIGNMENT);
  int align_sub_filter_count =
      align_to_x(sub_filter_width * sub_filter_width * sub_channels,
                 FILTER_ELEMENT_ALIGNMENT);
  int align_conv_sub_filter_count =
      align_sub_filter_count * align_sub_filter_num;

  for (int i = 0; i < sub_conv_num; ++i) {
    arg->conv_args[i].filter_num = (arg->sub_conv_num) * (arg->filter_num);
    arg->conv_args[i].group_num = group_num;

    arg->conv_args[i].filter_scale_address = filter->scale;
    arg->conv_args[i].relu_enabled = relu_enabled;

    arg->conv_args[i].kernel.width = sub_filter_width;
    arg->conv_args[i].kernel.height = sub_filter_width;
    arg->conv_args[i].kernel.stride_w = 1;
    arg->conv_args[i].kernel.stride_h = 1;

    // DeconvParam.conv_args[i].image.address = (void*)ptr_image;
    arg->conv_args[i].image.scale_address = input->scale;
    arg->conv_args[i].image.channels = sub_channels;
    arg->conv_args[i].image.width = (uint32_t)input->dims()[3];
    arg->conv_args[i].image.height = (uint32_t)input->dims()[2];
    arg->conv_args[i].image.pad_width = sub_pad;
    arg->conv_args[i].image.pad_height = sub_pad;
    arg->conv_args[i].image.address = input_ptr;

    arg->conv_args[i].sb_address = (void *)bs_ptr;

    char *filter_sub_space =
        (char *)fpga_malloc(align_conv_sub_filter_count * sizeof(char));
    fpga_copy(filter_sub_space,
              (char *)filter_ptr + i * align_conv_sub_filter_count,
              align_conv_sub_filter_count);
    arg->conv_args[i].filter_address = (void *)(filter_sub_space);
    fpga_flush(filter_sub_space, align_conv_sub_filter_count);

    if (sub_conv_num == 1) {
      arg->conv_args[i].output.address = out_ptr;
      arg->conv_args[i].output.scale_address = out->scale;
    } else {
      half *ptr_output = (half *)fpga_malloc(conv_output_size * sizeof(half));
      arg->conv_args[i].output.address = (void *)((half *)ptr_output);
      float *ptr_output_scale = (float *)fpga_malloc(2 * sizeof(float));
      arg->conv_args[i].output.scale_address = ptr_output_scale;
    }
  }

  arg->output.address = out_ptr;
  arg->output.scale_address = out->scale;
  // fpga_free(filter_ptr);
}
}  // namespace fpga
}  // namespace paddle_mobile
