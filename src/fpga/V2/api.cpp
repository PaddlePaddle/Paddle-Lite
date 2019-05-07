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
#include <memory>
#include "fpga/V2/bias_scale.h"
#include "fpga/V2/deconv_filter.h"
#include "fpga/V2/filter.h"
#include "fpga/V2/image.h"

namespace paddle_mobile {
namespace fpga {

#define USE_BIAS 2

void format_image(framework::Tensor *image_tensor) {
  auto dims = image_tensor->dims();
  auto channel = dims[1], height = dims[2], width = dims[3];
  auto data_ptr = image_tensor->data<int8_t>();
  auto external_ptr = reinterpret_cast<int8_t *>(image_tensor->external_data);
  int8_t *p_data = external_ptr == nullptr ? data_ptr : external_ptr;

  image::format_image<int8_t>(&p_data, channel, height, width);
  if (p_data != data_ptr && external_ptr == nullptr) {
    image_tensor->reset_data_ptr(p_data);
  }
}

void format_ofm(framework::Tensor *ofm_tensor) {
  if (ofm_tensor->type() == type_id<float>()) {
    format_fp32_ofm(ofm_tensor);
  } else {
    format_int8_ofm(ofm_tensor);
  }
  format_int8_ofm(ofm_tensor);
}

void format_int8_ofm(framework::Tensor *ofm_tensor) {
  auto dims = ofm_tensor->dims();
  size_t memory_size = 0;
  if (dims.size() == 4) {
    auto channel = dims[1], height = dims[2], width = dims[3], num = dims[0];
    memory_size = num * height * align_to_x(channel * width, IMAGE_ALIGNMENT) *
                  sizeof(int8_t);
  } else if (dims.size() == 2) {
    memory_size = align_to_x(dims[1], IMAGE_ALIGNMENT) * sizeof(int8_t);
  } else {
    DLOG << "Wrong ofm dimension";
  }
  auto p = fpga_malloc(memory_size);
  ofm_tensor->reset_data_ptr(p);
  ofm_tensor->set_type(type_id<int8_t>().hash_code());
  ofm_tensor->fpga_data_num = memory_size / sizeof(int8_t);
  fpga::fpga_flush(p, memory_size);
}

void format_int8_ofm(framework::Tensor *ofm_tensor, framework::DDim dims) {
  size_t memory_size = 0;
  if (dims.size() == 4) {
    auto channel = dims[1], height = dims[2], width = dims[3];
    memory_size =
        height * align_to_x(channel * width, IMAGE_ALIGNMENT) * sizeof(int8_t);
  } else if (dims.size() == 2) {
    memory_size = align_to_x(dims[1], IMAGE_ALIGNMENT) * sizeof(int8_t);
  } else {
    DLOG << "Wrong ofm dimension";
  }
  auto p = fpga_malloc(memory_size);
  ofm_tensor->reset_data_ptr(p);
  ofm_tensor->set_type(type_id<int8_t>().hash_code());
  ofm_tensor->fpga_data_num = memory_size / sizeof(int8_t);
  fpga::fpga_flush(p, memory_size);
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
  ofm_tensor->reset_data_ptr(p);
  ofm_tensor->set_type(type_id<float>().hash_code());
  ofm_tensor->fpga_data_num = memory_size / sizeof(float);
  fpga::fpga_flush(p, memory_size);
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
int get_deconv_plit_num(framework::Tensor *filter_tensor, int stride) {
  auto dims = filter_tensor->dims();
  auto chw = dims[1] * dims[2] / stride * dims[3] / stride;
  auto num = dims[0] * stride;
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

int get_deconv_filter_num_per_div(framework::Tensor *filter_tensor,
                                  int group_num, int stride) {
  auto dims = filter_tensor->dims();
  auto chw = dims[1] * dims[2] / stride * dims[3] / stride;
  auto num = dims[0] * stride;
  int div_capacity = filter::calc_division_capacity(chw);
  return filter::calc_num_per_div(num, group_num, div_capacity);
}

int get_aligned_filter_element_num(int chw) {
  return align_to_x(chw, FILTER_ELEMENT_ALIGNMENT);
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
  filter_tensor->set_type(type_id<int8_t>().hash_code());
}
void format_dwconv_filter(framework::Tensor *filter_tensor, float *scale_ptr) {
  auto dims = filter_tensor->dims();
  auto num = dims[0], height = dims[2], width = dims[3];
  auto data_ptr = filter_tensor->data<float>();
  size_t memory_size = num * height * width * sizeof(float);
  auto new_data = (float *)fpga_malloc(memory_size);  // NOLINT
  fpga_copy(new_data, data_ptr, memory_size);
  filter::format_dwconv_filter(&new_data, num, height, width, scale_ptr);
  filter_tensor->reset_data_ptr(new_data);
  filter_tensor->set_type(type_id<int16_t>().hash_code());
}

void format_DWDconv_filter(framework::Tensor *filter_tensor, float *scale_ptr,
                           int stride) {
  auto dims = filter_tensor->dims();
  auto num = dims[0], height = dims[2], width = dims[3];
  auto data_ptr = filter_tensor->data<float>();
  size_t memory_size = num * height * width * sizeof(float);
  auto new_data = (float *)fpga_malloc(memory_size);  // NOLINT
  fpga_copy(new_data, data_ptr, memory_size);

  int hw = height * width;
  deconv_filter::deconv_NC_convert(&new_data, num, 1, hw);

  num = dims[1];
  int channel = dims[0];

  deconv_filter::DWDconv_format_filter(&new_data, num, channel, height, width,
                                       scale_ptr, stride);

  filter_tensor->reset_data_ptr(new_data);
  filter_tensor->set_type(type_id<int16_t>().hash_code());
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
  filter_tensor->set_type(type_id<int8_t>().hash_code());
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
  filter_tensor->set_type(type_id<int8_t>().hash_code());
}

void format_bias_scale_array(float **bias_scale_array,
                             int element_num_per_division, int num) {
  bias_scale::format_bias_scale_array(bias_scale_array,
                                      element_num_per_division, num);
}
void format_bias_array(float **bias_array, int num) {
  bias_scale::format_bias_array(bias_array, num);
}

void format_concat_output(framework::Tensor *out, int height, int width,
                          int image_num, uint32_t *channel_num) {
  int sum_channel = 0, sum_cw = 0;
  for (int i = 0; i < image_num; i++) {
    sum_channel += channel_num[i];
  }

  sum_cw = align_to_x(width * sum_channel, IMAGE_ALIGNMENT);
  auto data_ptr = fpga_malloc(height * sum_cw * sizeof(int8_t));
  auto ddim = framework::make_ddim({1, sum_channel, height, width});
  out->Resize(ddim);
  out->reset_data_ptr(data_ptr);
  out->set_type(type_id<int8_t>().hash_code());
}
void format_conv_data(framework::Tensor *filter_tensor,
                      framework::Tensor *ofm_tensor, float **bs_ptr,
                      int group) {
  float max_value = fpga::filter_find_max(filter_tensor);
  fpga::format_filter(filter_tensor, max_value, group);
  int element_num_per_div = fpga::get_filter_num_per_div(filter_tensor, group);
  fpga::format_bias_scale_array(bs_ptr, element_num_per_div,
                                ofm_tensor->dims()[1]);
  fpga::format_ofm(ofm_tensor);
}
void format_deconv_data(framework::Tensor *filter_tensor,
                        framework::Tensor *ofm_tensor, float **bs_ptr,
                        int group, int sub_conv_n) {
  int channel = ofm_tensor->dims()[1];
  float max_value = filter_find_max(filter_tensor);
  format_deconv_filter(filter_tensor, max_value, group, sub_conv_n);
  int element_num_per_div =
      get_deconv_filter_num_per_div(filter_tensor, group, sub_conv_n);
  format_bias_scale_array(bs_ptr, element_num_per_div, channel * sub_conv_n);
  format_ofm(ofm_tensor);
}

void format_dwconv_data(framework::Tensor *filter_tensor,
                        framework::Tensor *ofm_tensor, float *scale_ptr,
                        float **bias_ptr) {
  auto channel = ofm_tensor->dims()[1];
  format_dwconv_filter(filter_tensor, scale_ptr);
  format_bias_array(bias_ptr, channel);
  format_ofm(ofm_tensor);
}
void format_DWDeconv_data(framework::Tensor *filter_tensor,
                          framework::Tensor *ofm_tensor, float **bs_ptr,
                          int group, int sub_conv_n) {
  int channel = ofm_tensor->dims()[1];
  format_DWDconv_filter(
      filter_tensor,
      (reinterpret_cast<float *>(*bs_ptr) + sub_conv_n * channel), sub_conv_n);
  format_bias_array(bs_ptr, channel);
  format_ofm(ofm_tensor);
}

void expand_conv_arg(ConvArgs *arg) {
  ConvArgs args = *arg;

  auto fpga_bias_scale_len =
      align_to_x(args.filter_num / args.group_num, 8) * args.group_num;

  auto output_height =
      (args.image.height + args.image.pad_height * 2 - args.kernel.height) /
          args.kernel.stride_h +
      1;
  auto output_width =
      (args.image.width + args.image.pad_width * 2 - args.kernel.width) /
          args.kernel.stride_w +
      1;

  auto filter_per_group = args.filter_num / args.group_num;
  auto channel_per_group = args.image.channels / args.group_num;

  auto image_row_count = args.image.width * args.image.channels;
  auto image_amount_per_row = align_to_x(image_row_count, IMAGE_ALIGNMENT);
  auto image_one_pad_per_row = align_to_x(image_row_count, IMAGE_ALIGNMENT) +
                               args.image.pad_width * args.image.channels;
  auto filter_amount_all =
      align_to_x(args.kernel.height * args.kernel.width * channel_per_group,
                 FILTER_ELEMENT_ALIGNMENT);

  auto output_amount_per_row = align_to_x(
      (output_width - (args.deconv_tx_param.omit_size) * 2) * args.filter_num,
      IMAGE_ALIGNMENT);

  // find the opt partition strategy
  uint64_t res_win;
  uint64_t res_fit = 0;
  for (res_win = 1; res_win <= output_width; res_win++) {
    if ((align_to_x(
             (args.image.channels *
              (args.kernel.width + (res_win - 1) * args.kernel.stride_w)),
             IMAGE_ALIGNMENT) /
             16 +
         1) *
            args.kernel.height >
        2048) {
      break;
    }
  }

  if (res_win != output_width) {
    res_win -= 1;
  }

  if (((res_win % 2) != 0) && (res_win != 1)) {
    res_win = res_win - 1;
  }
  res_fit = res_win;

  auto block_num = (output_width + res_fit - 1) / res_fit;
  auto block_len = res_fit;
  auto block_last = output_width - res_fit * (block_num - 1);

  auto res_amount_per_row =
      (output_width - (args.deconv_tx_param.omit_size) * 2) * args.filter_num;
  auto res_amount_per_row_pad = output_amount_per_row - res_amount_per_row;

  auto image_block_amount_per_row =
      args.kernel.stride_w * res_fit * args.image.channels;
  auto filter_pad_width_mul_channel =
      args.image.pad_width * args.image.channels;
  auto image_amount_per_row_multi_win_first =
      image_amount_per_row *
      (ROW_PARALLEL_NUM * args.kernel.stride_h - args.image.pad_height);
  auto image_amount_per_row_multi_win =
      image_amount_per_row * (ROW_PARALLEL_NUM * args.kernel.stride_h);

  auto image_block_num = block_num;
  auto image_block_len =
      align_to_x((args.image.channels *
                  (args.kernel.width + (block_len - 1) * args.kernel.stride_w)),
                 IMAGE_ALIGNMENT) /
          16 +
      1;
  auto image_block_len_last =
      align_to_x(
          (args.image.channels *
           (args.kernel.width + (block_last - 1) * args.kernel.stride_w)),
          IMAGE_ALIGNMENT) /
          16 +
      1;
  auto image_win_cnt = block_len;
  auto image_win_cnt_last = block_last;
  auto res_row_data_align4_pad = res_amount_per_row_pad / 8;
  auto prog_full_cnt = 1024 / (filter_amount_all / 16 * 2) - 1;
  if (prog_full_cnt == 511) {
    prog_full_cnt--;
  }
  auto post_prog_full_cnt =
      (512 / (align_to_x(args.filter_num, 4) / 4 * 2) > 2)
          ? (512 / (align_to_x(args.filter_num, 4) / 4 * 2) - 2)
          : 0;
  // auto cmd = 0UL | (args.relu_enabled ? USE_RELU : 0) | USE_BIAS;
  auto cmd = 0UL | USE_BIAS;

  auto deconv_param = ((args.deconv_tx_param.deconv_en) << 16) |
                      ((args.deconv_tx_param.sub_conv_num) << 8) |
                      ((args.deconv_tx_param.omit_size) << 0);
  (*arg).driver.image_address_phy = vaddr_to_paddr(args.image.address);
  (*arg).driver.sb_address_phy = vaddr_to_paddr(args.sb_address);
  (*arg).driver.filter_address_phy = vaddr_to_paddr(args.filter_address);
  (*arg).driver.output_address_phy = vaddr_to_paddr(args.output.address) +
                                     args.deconv_tx_param.out_addr_offset;
  (*arg).driver.output_height = output_height;
  (*arg).driver.output_width = output_width;
  (*arg).driver.filter_per_group = filter_per_group;
  (*arg).driver.channel_per_group = channel_per_group;
  (*arg).driver.image_amount_per_row = image_amount_per_row;
  (*arg).driver.image_one_pad_per_row = image_one_pad_per_row;
  (*arg).driver.filter_amount_all = filter_amount_all;
  (*arg).driver.output_amount_per_row = output_amount_per_row;
  (*arg).driver.image_block_amount_per_row = image_block_amount_per_row;
  (*arg).driver.filter_pad_width_mul_channel = filter_pad_width_mul_channel;
  (*arg).driver.image_amount_per_row_multi_win_first =
      image_amount_per_row_multi_win_first;
  (*arg).driver.image_amount_per_row_multi_win = image_amount_per_row_multi_win;
  (*arg).driver.image_block_num = image_block_num;
  (*arg).driver.image_block_len = image_block_len;
  (*arg).driver.image_block_len_last = image_block_len_last;
  (*arg).driver.image_win_cnt = image_win_cnt;
  (*arg).driver.image_win_cnt_last = image_win_cnt_last;
  (*arg).driver.res_row_data_align4_pad = res_row_data_align4_pad;
  (*arg).driver.prog_full_cnt = prog_full_cnt;
  (*arg).driver.post_prog_full_cnt = post_prog_full_cnt;
  (*arg).driver.fpga_bias_scale_len = fpga_bias_scale_len;
  (*arg).driver.cmd = cmd;
  (*arg).driver.deconv_param = deconv_param;
}  // expand_conv_arg()

void expand_EW_arg(EWAddArgs *arg) {
  EWAddArgs args = *arg;
  uint64_t cmd = 0;
  uint64_t datalen = (uint64_t)args.image0.width *
                     (uint64_t)args.image0.height *
                     (uint64_t)args.image0.channels;
  uint64_t coefficient = (uint64_t)args.const0 << 32 | (uint64_t)args.const1;
  uint64_t image0_address_phy = vaddr_to_paddr(args.image0.address);
  uint64_t image1_address_phy = vaddr_to_paddr(args.image1.address);
  uint64_t output_address_phy = vaddr_to_paddr(args.output.address);

  uint64_t image_amount_per_row =
      align_to_x((uint64_t)args.image0.width * (uint64_t)args.image0.channels,
                 IMAGE_ALIGNMENT);
  uint64_t image_image_pixel = ((uint64_t)args.image0.channels << 32) |
                               ((uint64_t)args.image0.width << 16) |
                               (uint64_t)args.image0.height;

  (*arg).driver.image0_address_phy = image0_address_phy;
  (*arg).driver.image1_address_phy = image1_address_phy;
  (*arg).driver.datalen = datalen;
  (*arg).driver.image_image_pixel = image_image_pixel;
  (*arg).driver.image_amount_per_row = image_amount_per_row;
  (*arg).driver.output_address_phy = output_address_phy;
  (*arg).driver.coefficient = coefficient;
  (*arg).driver.cmd = cmd;
}  // expand_EW_arg

void fill_split_arg(struct SplitConvArgs *arg, framework::Tensor *input,
                    framework::Tensor *out, framework::Tensor *filter,
                    ActivationType activation_enable,
                    int16_t leaky_relu_negative_slope, int group_num,
                    int stride_h, int stride_w, int padding_h, int padding_w,
                    float *bs_ptr) {
  auto input_ptr = input->data<int8_t>();
  auto filter_ptr = filter->data<int8_t>();
  auto out_ptr = out->data<int8_t>();
  auto deleter = [](void *p) { fpga_free(p); };

  arg->group_num = (uint32_t)group_num;
  // Either group_num or split_num = 1;
  arg->split_num = group_num == 1 ? (uint32_t)get_plit_num(filter) : 1;
  arg->filter_num = (uint32_t)filter->dims()[0];
  arg->output.address = out_ptr;
  arg->output.scale_address = out->scale;
  arg->conv_arg =
      (ConvArgs *)fpga_malloc(arg->split_num * sizeof(ConvArgs));  // NOLINT

  arg->shared_conv_arg = std::shared_ptr<ConvArgs>(arg->conv_arg, deleter);

  memset(arg->conv_arg, 0, arg->split_num * sizeof(struct ConvArgs));

  arg->concat_arg.image_num = arg->split_num;
  arg->concat_arg.image_out = out_ptr;
  arg->concat_arg.scale_out = out->scale;
  arg->concat_arg.height = (uint32_t)out->dims()[2];
  arg->concat_arg.width = (uint32_t)out->dims()[3];

  int n = arg->split_num;
  arg->concat_arg.images_in =
      static_cast<int8_t **>(fpga_malloc(n * sizeof(int *)));
  arg->concat_arg.scales_in =
      static_cast<float **>(fpga_malloc(n * sizeof(float *)));
  arg->concat_arg.channel_num =
      static_cast<uint32_t *>(fpga_malloc(n * sizeof(uint32_t)));
  arg->vector_concat_space.push_back(std::shared_ptr<char>(
      reinterpret_cast<char *>(arg->concat_arg.images_in), deleter));
  arg->vector_concat_space.push_back(std::shared_ptr<char>(
      reinterpret_cast<char *>(arg->concat_arg.scales_in), deleter));
  arg->vector_concat_space.push_back(std::shared_ptr<char>(
      reinterpret_cast<char *>(arg->concat_arg.channel_num), deleter));

  auto channel = (int)out->dims()[1];  // NOLINT
  int filter_num_per_div = get_filter_num_per_div(filter, group_num);
  int element_num = get_aligned_filter_element_num(
      (int)(filter->dims()[1] * filter->dims()[2] *  // NOLINT
            filter->dims()[3]));

  for (int i = 0; i < n; i++) {
    arg->conv_arg[i].output.activation.activation_type = activation_enable;
    arg->conv_arg[i].output.activation.leaky_relu_negative_slope =
        leaky_relu_negative_slope;
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
    arg->conv_arg[i].filter_num = (uint32_t)(
        i == n - 1 ? channel - (n - 1) * filter_num_per_div  // NOLINT
                   : filter_num_per_div);

    size_t filter_size =
        element_num *
        align_to_x(arg->conv_arg[i].filter_num, FILTER_NUM_ALIGNMENT) *
        sizeof(int8_t);
    auto filter_head = &(
        (int8_t *)filter_ptr)[i * element_num * filter_num_per_div];  // NOLINT
    arg->conv_arg[i].filter_address = fpga_malloc(filter_size);
    arg->vector_conv_space.push_back(std::shared_ptr<char>(
        reinterpret_cast<char *>(arg->conv_arg[i].filter_address), deleter));
    memcpy(arg->conv_arg[i].filter_address, filter_head, filter_size);
    fpga_flush(arg->conv_arg[i].filter_address, filter_size);

    size_t bs_size = 2 *
                     align_to_x(arg->conv_arg[i].filter_num, BS_NUM_ALIGNMENT) *
                     sizeof(float);
    auto bs_head = &bs_ptr[i * filter_num_per_div * 2];
    arg->conv_arg[i].sb_address = fpga_malloc(bs_size);
    arg->vector_conv_space.push_back(std::shared_ptr<char>(
        reinterpret_cast<char *>(arg->conv_arg[i].sb_address), deleter));
    memcpy(arg->conv_arg[i].sb_address, bs_head, bs_size);
    fpga_flush(arg->conv_arg[i].sb_address, bs_size);

    if (n > 1) {
      arg->conv_arg[i].output.scale_address =
          static_cast<float *>(fpga_malloc(2 * sizeof(float)));
      arg->conv_arg[i].output.address =
          fpga_malloc(out->dims()[2] *
                      align_to_x((int)(out->dims()[3] *  // NOLINT
                                       arg->conv_arg[i].filter_num),
                                 IMAGE_ALIGNMENT) *
                      sizeof(int8_t));
      arg->vector_conv_space.push_back(std::shared_ptr<char>(
          reinterpret_cast<char *>(arg->conv_arg[i].output.scale_address),
          deleter));
      arg->vector_conv_space.push_back(std::shared_ptr<char>(
          reinterpret_cast<char *>(arg->conv_arg[i].output.address), deleter));
    } else {
      arg->conv_arg[i].output.scale_address = out->scale;
      arg->conv_arg[i].output.address = out_ptr;
    }

    arg->concat_arg.images_in[i] =
        (int8_t *)arg->conv_arg[i].output.address;  // NOLINT
    arg->concat_arg.scales_in[i] = arg->conv_arg[i].output.scale_address;
    arg->concat_arg.channel_num[i] = arg->conv_arg[i].filter_num;

    expand_conv_arg(&arg->conv_arg[i]);
  }
  filter->reset_data_ptr(nullptr);
  fpga_free(bs_ptr);
}  // fill_split_arg

void fill_deconv_arg(struct DeconvArgs *arg, framework::Tensor *input,
                     framework::Tensor *out, framework::Tensor *filter,
                     ActivationType activation_enable,
                     int16_t leaky_relu_negative_slope, int group_num,
                     int stride_h, int stride_w, int padding_h, int padding_w,
                     float *bs_ptr) {
  auto input_ptr = input->data<int8_t>();
  auto filter_ptr = filter->data<int8_t>();
  auto deleter = [](void *p) { fpga_free(p); };

  arg->group_num = (uint32_t)group_num;
  arg->sub_conv_num = (uint32_t)stride_h;
  arg->filter_num = (uint32_t)filter->dims()[0];
  uint32_t sub_conv_num = arg->sub_conv_num;
  int sub_pad =
      deconv_filter::deconv_calc_sub_pad((int)filter->dims()[3],  // NOLINT
                                         padding_w, stride_w);
  auto sub_filter_width = (uint32_t)deconv_filter::deconv_get_sub_filter_axis(
      (int)filter->dims()[3], stride_w);  // NOLINT

  auto sub_output_width = (uint32_t)deconv_filter::deconv_get_sub_out_axis(
      (int)input->dims()[3], sub_pad, sub_filter_width);  // NOLINT
  auto sub_output_height = (uint32_t)deconv_filter::deconv_get_sub_out_axis(
      (int)input->dims()[2], sub_pad, sub_filter_width);  // NOLINT

  arg->sub_output_width = (uint32_t)sub_output_width;
  arg->sub_output_height = (uint32_t)sub_output_height;
  arg->omit_size = (uint32_t)deconv_filter::deconv_get_omit(
      stride_w, (int)filter->dims()[3], padding_w);  // NOLINT

  auto sub_channels = (int)input->dims()[1];  // NOLINT
  uint32_t omit_size = arg->omit_size;
  int real_out_width = sub_output_width * sub_conv_num - 2 * omit_size;
  int sub_filter_num = sub_conv_num * (arg->filter_num);

  framework::DDim dims_out_new = framework::make_ddim(
      {1, arg->filter_num, sub_output_height * sub_conv_num, real_out_width});
  fpga::format_int8_ofm(out, dims_out_new);
  auto out_ptr = out->data<int8_t>();
  arg->output.address =
      (int8_t *)out_ptr +  // NOLINT
      omit_size * sizeof(int8_t) *
          (align_to_x(real_out_width * arg->filter_num, IMAGE_ALIGNMENT));
  arg->output.scale_address = out->scale;

  uint32_t conv_output_size =
      (align_to_x(sub_output_width * sub_filter_num, IMAGE_ALIGNMENT)) *
      sub_output_height;
  uint32_t split_num =
      group_num == 1 ? (uint32_t)get_deconv_plit_num(filter, sub_conv_num) : 1;

  for (int i = 0; i < sub_conv_num; ++i) {
    arg->split_conv_args.push_back(std::make_shared<SplitConvArgs>());
    arg->split_conv_args[i]->filter_num =
        (arg->sub_conv_num) * (arg->filter_num);
    arg->split_conv_args[i]->group_num = (uint32_t)group_num;
    arg->split_conv_args[i]->split_num = split_num;
    arg->split_conv_args[i]->concat_arg.height = sub_output_height;
    arg->split_conv_args[i]->concat_arg.width = sub_output_width;
    arg->split_conv_args[i]->concat_arg.image_num = split_num;

    arg->split_conv_args[i]->conv_arg =
        static_cast<ConvArgs *>(fpga_malloc(split_num * sizeof(ConvArgs)));
    arg->split_conv_args[i]->concat_arg.images_in =
        static_cast<int8_t **>(fpga_malloc(split_num * sizeof(int8_t *)));
    arg->split_conv_args[i]->concat_arg.scales_in =
        static_cast<float **>(fpga_malloc(split_num * sizeof(float *)));
    arg->split_conv_args[i]->concat_arg.channel_num =
        static_cast<uint32_t *>(fpga_malloc(split_num * sizeof(uint32_t)));
    arg->split_conv_args[i]->shared_conv_arg =
        std::shared_ptr<ConvArgs>(arg->split_conv_args[i]->conv_arg, deleter);
    arg->split_conv_args[i]->vector_concat_space.push_back(
        std::shared_ptr<char>(
            reinterpret_cast<char *>(
                arg->split_conv_args[i]->concat_arg.images_in),
            deleter));
    arg->split_conv_args[i]->vector_concat_space.push_back(
        std::shared_ptr<char>(
            reinterpret_cast<char *>(
                arg->split_conv_args[i]->concat_arg.scales_in),
            deleter));
    arg->split_conv_args[i]->vector_concat_space.push_back(
        std::shared_ptr<char>(
            reinterpret_cast<char *>(
                arg->split_conv_args[i]->concat_arg.channel_num),
            deleter));
  }

  auto filter_num_per_div =
      (uint32_t)get_deconv_filter_num_per_div(filter, group_num, stride_w);
  int element_num = get_aligned_filter_element_num(
      (int)(sub_channels * sub_filter_width * sub_filter_width));  // NOLINT

  int chw = sub_channels * sub_filter_width * sub_filter_width;
  int division_capacity = filter::calc_division_capacity(chw);
  int num_per_div_before_alignment =
      filter::calc_num_per_div(sub_filter_num, group_num, division_capacity);
  int num_per_div_after_alignment =
      align_to_x(num_per_div_before_alignment, FILTER_NUM_ALIGNMENT);
  int div_num = (sub_filter_num + num_per_div_before_alignment - 1) /
                num_per_div_before_alignment;
  int residual = sub_filter_num % num_per_div_before_alignment;
  int num_after_alignment = num_per_div_after_alignment *
                                ((residual == 0) ? div_num : (div_num - 1)) +
                            align_to_x(residual, FILTER_NUM_ALIGNMENT);

  int filter_sub_conv_offset = element_num * num_after_alignment;
  uint32_t out_addr_offset = 0;
  for (int i = 0; i < sub_conv_num; ++i) {
    if (sub_conv_num == 1) {
      arg->split_conv_args[i]->output.address = arg->output.address;
      arg->split_conv_args[i]->output.scale_address = arg->output.scale_address;
      out_addr_offset = 0;

    } else {
      out_addr_offset =
          sizeof(int8_t) * (sub_conv_num - 1 - i) *
          (align_to_x(real_out_width * arg->filter_num, IMAGE_ALIGNMENT));

      arg->split_conv_args[i]->output.address = out_ptr;
      arg->split_conv_args[i]->output.scale_address =
          static_cast<float *>(fpga_malloc(2 * sizeof(float)));
      arg->split_conv_args[i]->vector_conv_space.push_back(
          std::shared_ptr<char>(
              reinterpret_cast<char *>(
                  arg->split_conv_args[i]->output.scale_address),
              deleter));
    }

    for (int j = 0; j < split_num; ++j) {
      arg->split_conv_args[i]->conv_arg[j].output.activation.activation_type =
          activation_enable;
      arg->split_conv_args[i]
          ->conv_arg[j]
          .output.activation.leaky_relu_negative_slope =
          leaky_relu_negative_slope;
      arg->split_conv_args[i]->conv_arg[j].group_num = (uint32_t)group_num;

      arg->split_conv_args[i]->conv_arg[j].kernel.width =
          (uint32_t)sub_filter_width;
      arg->split_conv_args[i]->conv_arg[j].kernel.height =
          (uint32_t)sub_filter_width;
      arg->split_conv_args[i]->conv_arg[j].kernel.stride_w = 1;
      arg->split_conv_args[i]->conv_arg[j].kernel.stride_h = 1;

      arg->split_conv_args[i]->conv_arg[j].deconv_tx_param.deconv_en = 1;
      arg->split_conv_args[i]->conv_arg[j].deconv_tx_param.sub_conv_num =
          sub_conv_num;
      arg->split_conv_args[i]->conv_arg[j].deconv_tx_param.omit_size =
          omit_size;
      arg->split_conv_args[i]->conv_arg[j].deconv_tx_param.out_addr_offset =
          out_addr_offset;

      arg->split_conv_args[i]->conv_arg[j].image.scale_address = input->scale;
      arg->split_conv_args[i]->conv_arg[j].image.channels =
          (uint32_t)sub_channels;
      arg->split_conv_args[i]->conv_arg[j].image.width =
          (uint32_t)input->dims()[3];
      arg->split_conv_args[i]->conv_arg[j].image.height =
          (uint32_t)input->dims()[2];
      arg->split_conv_args[i]->conv_arg[j].image.pad_width = (uint32_t)sub_pad;
      arg->split_conv_args[i]->conv_arg[j].image.pad_height = (uint32_t)sub_pad;
      arg->split_conv_args[i]->conv_arg[j].image.address = input_ptr;

      arg->split_conv_args[i]->conv_arg[j].filter_scale_address = filter->scale;
      arg->split_conv_args[i]->conv_arg[j].filter_num =
          (uint32_t)(j == split_num - 1
                         ? sub_filter_num - (split_num - 1) * filter_num_per_div
                         : filter_num_per_div);

      size_t filter_size =
          element_num *
          align_to_x(arg->split_conv_args[i]->conv_arg[j].filter_num,
                     FILTER_NUM_ALIGNMENT) *
          sizeof(int8_t);
      auto filter_head = &((
          int8_t *)filter_ptr)[j * element_num * filter_num_per_div +  // NOLINT
                               i * filter_sub_conv_offset];
      arg->split_conv_args[i]->conv_arg[j].filter_address =
          fpga_malloc(filter_size);
      arg->split_conv_args[i]->vector_conv_space.push_back(
          std::shared_ptr<char>(
              reinterpret_cast<char *>(
                  arg->split_conv_args[i]->conv_arg[j].filter_address),
              deleter));

      memcpy(arg->split_conv_args[i]->conv_arg[j].filter_address, filter_head,
             filter_size);
      fpga_flush(arg->split_conv_args[i]->conv_arg[j].filter_address,
                 filter_size);

      size_t bs_align_num = align_to_x(
          arg->split_conv_args[i]->conv_arg[j].filter_num, BS_NUM_ALIGNMENT);
      size_t bs_size = 2 * bs_align_num * sizeof(float);
      auto bs_head = &bs_ptr[j * filter_num_per_div * 2];

      arg->split_conv_args[i]->conv_arg[j].sb_address = fpga_malloc(bs_size);
      arg->split_conv_args[i]->vector_conv_space.push_back(
          std::shared_ptr<char>(
              reinterpret_cast<char *>(
                  arg->split_conv_args[i]->conv_arg[j].sb_address),
              deleter));

      memcpy(arg->split_conv_args[i]->conv_arg[j].sb_address, bs_head, bs_size);
      fpga_flush(arg->split_conv_args[i]->conv_arg[j].sb_address, bs_size);

      if (split_num == 1) {
        arg->split_conv_args[i]->conv_arg[j].output.address =
            arg->split_conv_args[i]->output.address;
        arg->split_conv_args[i]->conv_arg[j].output.scale_address =
            arg->split_conv_args[i]->output.scale_address;
      } else {
        arg->split_conv_args[i]->conv_arg[j].output.address =
            fpga_malloc(conv_output_size * sizeof(int8_t));
        arg->split_conv_args[i]->conv_arg[j].output.scale_address =
            static_cast<float *>(fpga_malloc(2 * sizeof(float)));
        arg->split_conv_args[i]->vector_conv_space.push_back(
            std::shared_ptr<char>(
                reinterpret_cast<char *>(
                    arg->split_conv_args[i]->conv_arg[j].output.address),
                deleter));
        arg->split_conv_args[i]->vector_conv_space.push_back(
            std::shared_ptr<char>(
                reinterpret_cast<char *>(
                    arg->split_conv_args[i]->conv_arg[j].output.scale_address),
                deleter));
      }
      arg->split_conv_args[i]->concat_arg.images_in[j] = static_cast<int8_t *>(
          arg->split_conv_args[i]->conv_arg[j].output.address);
      arg->split_conv_args[i]->concat_arg.scales_in[j] =
          arg->split_conv_args[i]->conv_arg[j].output.scale_address;
      arg->split_conv_args[i]->concat_arg.channel_num[j] =
          arg->split_conv_args[i]->conv_arg[j].filter_num;

      expand_conv_arg(&(arg->split_conv_args[i]->conv_arg[j]));
    }

    arg->split_conv_args[i]->concat_arg.image_out =
        arg->split_conv_args[i]->output.address;
    arg->split_conv_args[i]->concat_arg.scale_out =
        arg->split_conv_args[i]->output.scale_address;
  }
  filter->reset_data_ptr(nullptr);
  fpga_free(bs_ptr);
}  // fill_deconv_arg

void fill_dwconv_arg(struct DWconvArgs *arg, framework::Tensor *input,
                     framework::Tensor *out, framework::Tensor *filter,
                     ActivationType activation_enable,
                     int16_t leaky_relu_negative_slope, int stride_h,
                     int stride_w, int padding_h, int padding_w,
                     float *bias_ptr) {
  auto filter_ptr = filter->data<int16_t>();
  auto input_ptr = input->data<int8_t>();
  auto output_ptr = out->mutable_data<int8_t>();
  arg->sub_conv_num = 1;
  arg->output.activation.activation_type = activation_enable;
  arg->output.activation.leaky_relu_negative_slope = leaky_relu_negative_slope;
  arg->bias_address = bias_ptr;
  arg->filter_address = filter_ptr;
  arg->kernel.height = (uint32_t)filter->dims()[2];
  arg->kernel.width = (uint32_t)filter->dims()[3];
  arg->kernel.stride_h = (uint32_t)stride_h;
  arg->kernel.stride_w = (uint32_t)stride_w;
  arg->image.address = input_ptr;
  arg->image.channels = (uint32_t)input->dims()[1];
  arg->image.height = (uint32_t)input->dims()[2];
  arg->image.width = (uint32_t)input->dims()[3];
  arg->image.pad_height = (uint32_t)padding_h;
  arg->image.pad_width = (uint32_t)padding_w;
  arg->image.scale_address = input->scale;
  arg->output.address = output_ptr;
  arg->output.scale_address = out->scale;
}  // end dwconv arg fill

void fill_DWDeconv_arg(struct DWDeconvArgs *arg, framework::Tensor *input,
                       framework::Tensor *out, framework::Tensor *filter,
                       ActivationType activation_enable,
                       int16_t leaky_relu_negative_slope, int stride_h,
                       int stride_w, int padding_h, int padding_w,
                       float *bias_ptr) {
  auto filter_ptr = filter->data<int8_t>();
  auto input_ptr = input->data<int8_t>();

  auto deleter = [](void *p) { fpga_free(p); };

  arg->group_num = (uint32_t)filter->dims()[0];
  arg->sub_conv_num = (uint32_t)stride_w;
  arg->filter_num = (uint32_t)filter->dims()[0];

  int sub_conv_num = stride_w;

  int sub_pad =
      deconv_filter::deconv_calc_sub_pad((int)filter->dims()[3],  // NOLINT
                                         padding_w, stride_w);
  auto sub_filter_width = (uint32_t)deconv_filter::deconv_get_sub_filter_axis(
      (int)filter->dims()[3], stride_w);  // NOLINT

  auto sub_output_width = (uint32_t)deconv_filter::deconv_get_sub_out_axis(
      (int)input->dims()[3], sub_pad, sub_filter_width);  // NOLINT
  auto sub_output_height = (uint32_t)deconv_filter::deconv_get_sub_out_axis(
      (int)input->dims()[2], sub_pad, sub_filter_width);  // NOLINT

  arg->sub_output_width = (uint32_t)sub_output_width;
  arg->sub_output_height = (uint32_t)sub_output_height;
  arg->omit_size = (uint32_t)deconv_filter::deconv_get_omit(
      stride_w, (int)filter->dims()[3], padding_w);  // NOLINT

  auto sub_channels = (int)input->dims()[1];  // NOLINT
  uint32_t omit_size = arg->omit_size;
  int real_out_width = sub_output_width * sub_conv_num - 2 * omit_size;
  int real_out_height = sub_output_height * sub_conv_num - 2 * omit_size;
  int sub_filter_num = sub_conv_num * (arg->filter_num);

  framework::DDim dims_out_new = framework::make_ddim(
      {1, arg->filter_num, real_out_height, real_out_width});
  fpga::format_int8_ofm(out, dims_out_new);
  auto out_ptr = out->data<int8_t>();

  arg->output.address = out_ptr;
  arg->output.scale_address = out->scale;

  int filter_offset = sub_filter_width * sub_filter_width *
                      align_to_x(sub_channels, FILTER_ELEMENT_ALIGNMENT) *
                      arg->sub_conv_num;

  for (int i = 0; i < sub_conv_num; ++i) {
    arg->dw_conv_args.push_back(std::make_shared<DWconvArgs>());

    arg->dw_conv_args[i]->sub_conv_num = sub_conv_num;
    // arg->dw_conv_args[i]->relu_enabled = relu_enabled;
    arg->dw_conv_args[i]->output.activation.activation_type = activation_enable;
    arg->dw_conv_args[i]->output.activation.leaky_relu_negative_slope =
        leaky_relu_negative_slope;
    arg->dw_conv_args[i]->bias_address = bias_ptr;

    arg->dw_conv_args[i]->filter_address =
        fpga_malloc(filter_offset * sizeof(int16_t));
    memcpy(arg->dw_conv_args[i]->filter_address,
           (reinterpret_cast<half *>(filter_ptr) + i * filter_offset),
           filter_offset * sizeof(int16_t));
    arg->vector_dw_conv_space.push_back(std::shared_ptr<char>(
        reinterpret_cast<char *>(arg->dw_conv_args[i]->filter_address),
        deleter));

    arg->dw_conv_args[i]->kernel.height = (uint32_t)sub_filter_width;
    arg->dw_conv_args[i]->kernel.width = (uint32_t)sub_filter_width;

    arg->dw_conv_args[i]->kernel.stride_h = (uint32_t)1;
    arg->dw_conv_args[i]->kernel.stride_w = (uint32_t)1;
    arg->dw_conv_args[i]->image.address = input_ptr;
    arg->dw_conv_args[i]->image.channels = (uint32_t)input->dims()[1];
    arg->dw_conv_args[i]->image.height = (uint32_t)input->dims()[2];
    arg->dw_conv_args[i]->image.width = (uint32_t)input->dims()[3];

    arg->dw_conv_args[i]->image.pad_height = sub_pad;
    arg->dw_conv_args[i]->image.pad_width = sub_pad;
    arg->dw_conv_args[i]->image.scale_address = input->scale;

    arg->dw_conv_args[i]->output.address =
        fpga_malloc(sub_output_height *
                    align_to_x(sub_output_width * sub_channels * sub_conv_num,
                               IMAGE_ALIGNMENT) *
                    sizeof(int8_t));
    arg->dw_conv_args[i]->output.scale_address =
        static_cast<float *>(fpga_malloc(2 * sizeof(float)));
    arg->vector_dw_conv_space.push_back(std::shared_ptr<char>(
        reinterpret_cast<char *>(arg->dw_conv_args[i]->output.address),
        deleter));
    arg->vector_dw_conv_space.push_back(std::shared_ptr<char>(
        reinterpret_cast<char *>(arg->dw_conv_args[i]->output.scale_address),
        deleter));
  }

  // arg->output.scale_address = out->scale;
}  // end dwconv arg fill

}  // namespace fpga
}  // namespace paddle_mobile
