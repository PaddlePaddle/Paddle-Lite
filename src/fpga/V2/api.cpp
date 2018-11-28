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

#include "fpga/V2/api.h"
#include "fpga/V2/bias_scale.h"
#include "fpga/V2/filter.h"
#include "fpga/V2/image.h"

namespace paddle_mobile {
namespace fpga {

void format_image(framework::Tensor *image_tensor) {
  auto dims = image_tensor->dims();
  auto channel = dims[1], height = dims[2], width = dims[3];
  auto data_ptr = image_tensor->data<float>();
  size_t memory_size = channel * height * width * sizeof(float);
  auto new_data = (float *)fpga_malloc(memory_size);  // NOLINT
  memcpy(new_data, data_ptr, memory_size);
  int aligned_channel = filter::calc_aligned_channel((int)channel);  // NOLINT
  image::format_image(&new_data, (int)channel, (int)height,          // NOLINT
                      (int)width,                                    // NOLINT
                      aligned_channel);
  image_tensor->reset_data_ptr(new_data);
}

void format_fp16_ofm(framework::Tensor *ofm_tensor, int aligned_channel) {
  auto dims = ofm_tensor->dims();
  size_t memory_size = 0;
  if (dims.size() == 4) {
    auto height = dims[2], width = dims[3];
    memory_size = (height + 1) / 2 * 2 * width * aligned_channel * sizeof(half);
  } else if (dims.size() == 2) {
    memory_size = aligned_channel * sizeof(half);
  } else {
    DLOG << "Wrong ofm dimension";
  }
  auto p = fpga_malloc(memory_size);
  memset(p, 0, memory_size);
  ofm_tensor->reset_data_ptr(p);
}

void format_fp32_ofm(framework::Tensor *ofm_tensor, int aligned_channel) {
  auto dims = ofm_tensor->dims();
  size_t memory_size = 0;
  if (dims.size() == 4) {
    auto height = dims[2], width = dims[3];
    memory_size = height * width * aligned_channel * sizeof(float);
  } else if (dims.size() == 2) {
    memory_size = aligned_channel * sizeof(float);
  } else {
    DLOG << "Wrong ofm dimension";
  }
  auto p = fpga_malloc(memory_size);
  memset(p, 0, memory_size);
  ofm_tensor->reset_data_ptr(p);
}

float filter_find_max(framework::Tensor *filter_tensor) {
  auto filter_ptr = filter_tensor->data<float>();
  return filter::find_max(filter_ptr, (int)filter_tensor->numel());  // NOLINT
}

int get_aligned_channel_num(int channel_num) {
  return filter::calc_aligned_channel(channel_num);
}

int get_aligned_filter_num(framework::Tensor *filter_tensor) {
  auto dims = filter_tensor->dims();
  return filter::calc_aligned_num((int)dims[0], (int)dims[1]);  // NOLINT
}

int get_conv_output_channel(framework::Tensor *filter_tensor) {
  int aligned_filter_num = get_aligned_filter_num(filter_tensor);
  return get_aligned_channel_num(aligned_filter_num);
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
  memcpy(new_data, data_ptr, memory_size);
  filter::format_filter(&new_data, (int)num, (int)channel,  // NOLINT
                        (int)height,                        // NOLINT
                        (int)width, group_num, max_value);  // NOLINT
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
  memcpy(new_data, data_ptr, memory_size);
  filter::format_fc_filter(&new_data, (int)num, (int)channel,  // NOLINT
                           (int)height,                        // NOLINT
                           (int)width, 1, max_value);          // NOLINT
  filter_tensor->reset_data_ptr(new_data);
}

void format_bias_scale_array(float **bias_scale_array, int filter_num,
                             int filter_channel) {
  int num_after_alignment =
      filter::calc_aligned_num(filter_channel, filter_channel);
  bias_scale::format_bias_scale_array(bias_scale_array, filter_num,
                                      num_after_alignment);
}

void format_concat_output(framework::Tensor *out, int height, int width,
                          uint32_t out_channel) {
  auto data_ptr = fpga_malloc(out_channel * height * width * sizeof(half));
  auto ddim = framework::make_ddim({1, out_channel, height, width});
  out->Resize(ddim);
  out->reset_data_ptr(data_ptr);
}

int format_conv_data(framework::Tensor *filter_tensor,
                     framework::Tensor *ofm_tensor, float **bs_ptr, int group) {
  float max_value = fpga::filter_find_max(filter_tensor);
  fpga::format_filter(filter_tensor, max_value, group);
  int aligned_num = get_aligned_filter_num(filter_tensor);
  fpga::format_bias_scale_array(bs_ptr,
                                (int)filter_tensor->dims()[0],  // NOLINT
                                aligned_num);
  int aligned_channel = fpga::get_conv_output_channel(filter_tensor);
  fpga::format_fp16_ofm(ofm_tensor, aligned_channel);
  DLOG << aligned_channel;
  return aligned_channel;
}

int format_fc_data(framework::Tensor *filter_tensor,
                   framework::Tensor *ofm_tensor, float *bs_ptr) {
  float max_value = fpga::filter_find_max(filter_tensor);
  fpga::format_fc_filter(filter_tensor, max_value);
  int aligned_num = get_aligned_filter_num(filter_tensor);
  fpga::format_bias_scale_array(&bs_ptr,
                                (int)filter_tensor->dims()[0],  // NOLINT
                                aligned_num);
  int aligned_channel = fpga::get_conv_output_channel(filter_tensor);
  fpga::format_fp16_ofm(ofm_tensor, aligned_channel);
  DLOG << aligned_channel;
  return aligned_channel;
}

void fill_split_arg(struct SplitConvArgs *arg, framework::Tensor *input,
                    framework::Tensor *out, framework::Tensor *filter,
                    bool relu_enabled, int group_num, int stride_h,
                    int stride_w, int padding_h, int padding_w, float *bs_ptr) {
  auto input_ptr = input->data<float>();
  auto filter_ptr = filter->data<float>();
  auto out_ptr = out->data<float>();

  arg->group_num = (uint32_t)group_num;
  arg->split_num = 1;
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

  for (int i = 0; i < n; i++) {
    arg->conv_arg[i].relu_enabled = relu_enabled;
    arg->conv_arg[i].sb_address = bs_ptr;
    arg->conv_arg[i].filter_address = (int8_t *)filter_ptr;  // NOLINT
    arg->conv_arg[i].filter_scale_address = filter->scale;
    arg->conv_arg[i].filter_num = arg->filter_num;
    arg->conv_arg[i].group_num = (uint32_t)group_num;

    arg->conv_arg[i].kernel.stride_h = (uint32_t)stride_h;
    arg->conv_arg[i].kernel.stride_w = (uint32_t)stride_w;
    arg->conv_arg[i].kernel.height = (uint32_t)filter->dims()[2];
    arg->conv_arg[i].kernel.width = (uint32_t)filter->dims()[3];

    arg->conv_arg[i].image.address = input_ptr;
    arg->conv_arg[i].image.scale_address = input->scale;
    arg->conv_arg[i].image.channels = (uint32_t)input->dims()[1];
    arg->conv_arg[i].image.height = (uint32_t)input->dims()[2];
    arg->conv_arg[i].image.width = (uint32_t)input->dims()[3];
    arg->conv_arg[i].image.pad_height = (uint32_t)padding_h;
    arg->conv_arg[i].image.pad_width = (uint32_t)padding_w;

    arg->conv_arg[i].output.address = out_ptr;
    arg->conv_arg[i].output.scale_address = out->scale;

    int num_after_alignment = filter::calc_aligned_num(
        (int)input->dims()[1], arg->filter_num);  // NOLINT
    arg->conv_arg[i].free_space =
        fpga_malloc(num_after_alignment * 2 * sizeof(half));
  }
}

}  // namespace fpga
}  // namespace paddle_mobile
