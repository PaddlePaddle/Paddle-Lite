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

#define USE_RELU 1
#define USE_BIAS 2

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

void expand_conv_arg(ConvArgs *arg) {
  ConvArgs args = *arg;
  uint64_t filterlen = (uint64_t)args.kernel.width *
                       (uint64_t)args.kernel.height *
                       (uint64_t)args.image.channels;
  filterlen = align_to_x(filterlen, FILTER_ELEMENT_ALIGNMENT);
  filterlen *= align_to_x((uint64_t)args.filter_num, FILTER_NUM_ALIGNMENT);
  uint64_t fpga_bias_scale_len =
      align_to_x(args.filter_num / args.group_num, 8) * args.group_num;

  uint64_t output_height =
      (args.image.height + args.image.pad_height * 2 - args.kernel.height) /
          args.kernel.stride_h +
      1;
  uint64_t output_width =
      (args.image.width + args.image.pad_width * 2 - args.kernel.width) /
          args.kernel.stride_w +
      1;
  uint64_t output_size =
      output_height * output_width * (uint64_t)args.filter_num;

  auto filter_per_group = (uint64_t)(args.filter_num / args.group_num);
  auto channel_per_group = (uint64_t)(args.image.channels / args.group_num);

  uint64_t image_row_count = ((uint64_t)args.image.width) *
                             ((uint64_t)args.image.channels);  // without align
  uint64_t image_amount_per_row = align_to_x(image_row_count, IMAGE_ALIGNMENT);
  uint64_t image_one_pad_per_row =
      align_to_x(image_row_count, IMAGE_ALIGNMENT) +
      ((uint64_t)args.image.pad_width) * ((uint64_t)args.image.channels);
  uint64_t filter_amount_all =
      align_to_x(((uint64_t)args.kernel.height) *
                     ((uint64_t)args.kernel.width) * channel_per_group,
                 FILTER_ELEMENT_ALIGNMENT);

  uint64_t output_amount_per_row =
      align_to_x(output_width * ((uint64_t)args.filter_num), IMAGE_ALIGNMENT);

  // find the opt partition strategy
  uint64_t res_win;
  uint64_t res_fit = 0;
  for (res_win = 1; res_win <= output_width; res_win = res_win + 1) {
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

  uint64_t block_num = (output_width + res_fit - 1) / res_fit;
  uint64_t block_len = res_fit;
  uint64_t block_last = output_width - res_fit * (block_num - 1);

  uint64_t res_amount_per_row = output_width * args.filter_num;
  uint64_t res_amount_per_row_pad = output_amount_per_row - res_amount_per_row;

  uint64_t image_block_amount_per_row =
      args.kernel.stride_w * (res_fit)*args.image.channels;
  uint64_t filter_pad_width_mul_channel =
      args.image.pad_width * args.image.channels;
  uint64_t image_amount_per_row_multi_win_first =
      image_amount_per_row * (4 * args.kernel.stride_h - args.image.pad_height);
  uint64_t image_amount_per_row_multi_win =
      image_amount_per_row * (4 * args.kernel.stride_h);

  uint64_t image_block_num = block_num;
  uint64_t image_block_len =
      align_to_x((args.image.channels *
                  (args.kernel.width + (block_len - 1) * args.kernel.stride_w)),
                 IMAGE_ALIGNMENT) /
          16 +
      1;
  uint64_t image_block_len_last =
      align_to_x(
          (args.image.channels *
           (args.kernel.width + (block_last - 1) * args.kernel.stride_w)),
          IMAGE_ALIGNMENT) /
          16 +
      1;
  uint64_t image_win_cnt = block_len;
  uint64_t image_win_cnt_last = block_last;
  uint64_t res_row_data_align4_pad = res_amount_per_row_pad / 8;
  uint64_t prog_full_cnt = 2048 / (filter_amount_all / 16 * 2) - 1;
  if (prog_full_cnt == 1023) {
    prog_full_cnt--;
  }
  uint64_t post_prog_full_cnt =
      (512 / (align_to_x(args.filter_num, 4) / 4 * 2) > 2)
          ? (512 / (align_to_x(args.filter_num, 4) / 4 * 2) - 2)
          : 0;
  uint64_t cmd = 0UL | (args.relu_enabled ? USE_RELU : 0) | USE_BIAS;

  (*arg).driver.image_address_phy = vaddr_to_paddr(args.image.address);
  (*arg).driver.sb_address_phy = vaddr_to_paddr(args.sb_address);
  (*arg).driver.filter_address_phy = vaddr_to_paddr(args.filter_address);
  (*arg).driver.output_address_phy = vaddr_to_paddr(args.output.address);
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
}  // expand_conv_arg()

void expand_EW_arg(EWAddArgs *arg) {
  EWAddArgs args = *arg;
  uint64_t cmd = args.relu_enabled ? USE_RELU : 0;
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
      (int)(filter->dims()[1] * filter->dims()[2] * filter->dims()[3]));

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
    arg->conv_arg[i].filter_num = (uint32_t)(
        i == n - 1 ? channel - (n - 1) * filter_num_per_div  // NOLINT
                   : filter_num_per_div);

    size_t filter_size =
        element_num *
        align_to_x(arg->conv_arg[i].filter_num, FILTER_NUM_ALIGNMENT) *
        sizeof(int8_t);
    auto filter_head =
        &((int8_t *)filter_ptr)[i * element_num * filter_num_per_div];
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
          (float *)fpga_malloc(2 * sizeof(float));  // NOLINT
      arg->conv_arg[i].output.address = fpga_malloc(
          out->dims()[2] *
          align_to_x((int)(out->dims()[3] * arg->conv_arg[i].filter_num),
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

    expand_conv_arg(&arg->conv_arg[i]);
  }
  filter->reset_data_ptr(nullptr);
  fpga_free(bs_ptr);
}  // fill_split_arg

void fill_deconv_arg(struct DeconvArgs *arg, framework::Tensor *input,
                     framework::Tensor *out, framework::Tensor *filter,
                     bool relu_enabled, int group_num, int stride_h,
                     int stride_w, int padding_h, int padding_w,
                     float *bs_ptr) {
  auto input_ptr = input->data<float>();
  auto filter_ptr = filter->data<float>();
  auto out_ptr = out->data<float>();

  arg->group_num = (uint32_t)group_num;
  arg->sub_conv_num = (uint32_t)stride_h;
  arg->filter_num = (uint32_t)filter->dims()[0];
  int sub_conv_num = arg->sub_conv_num;
  int sub_stride = 1;
  int sub_pad = deconv_filter::deconv_calc_sub_pad((int)filter->dims()[3],
                                                   padding_w, stride_w);
  int sub_filter_width = deconv_filter::deconv_get_sub_filter_axis(
      (int)filter->dims()[3], stride_w);

  int sub_output_width = deconv_filter::deconv_get_sub_out_axis(
      (int)input->dims()[3], sub_pad, sub_filter_width);
  int sub_output_height = deconv_filter::deconv_get_sub_out_axis(
      (int)input->dims()[2], sub_pad, sub_filter_width);

  arg->sub_output_width = (uint32_t)sub_output_width;
  arg->sub_output_height = (uint32_t)sub_output_height;
  arg->omit_size = (uint32_t)deconv_filter::deconv_get_omit(
      stride_w, (int)filter->dims()[3], padding_w);
  arg->conv_args = (ConvArgs *)fpga_malloc(sub_conv_num * sizeof(ConvArgs));

  int sub_channels = (int)input->dims()[1];
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
    arg->conv_args[i].group_num = (uint32_t)group_num;

    arg->conv_args[i].filter_scale_address = filter->scale;
    arg->conv_args[i].relu_enabled = relu_enabled;

    arg->conv_args[i].kernel.width = (uint32_t)sub_filter_width;
    arg->conv_args[i].kernel.height = (uint32_t)sub_filter_width;
    arg->conv_args[i].kernel.stride_w = 1;
    arg->conv_args[i].kernel.stride_h = 1;

    // DeconvParam.conv_args[i].image.address = (void*)ptr_image;
    arg->conv_args[i].image.scale_address = input->scale;
    arg->conv_args[i].image.channels = (uint32_t)sub_channels;
    arg->conv_args[i].image.width = (uint32_t)input->dims()[3];
    arg->conv_args[i].image.height = (uint32_t)input->dims()[2];
    arg->conv_args[i].image.pad_width = (uint32_t)sub_pad;
    arg->conv_args[i].image.pad_height = (uint32_t)sub_pad;
    arg->conv_args[i].image.address = input_ptr;
    arg->conv_args[i].sb_address = (void *)bs_ptr;

    auto filter_sub_space =
        (char *)fpga_malloc(align_conv_sub_filter_count * sizeof(char));
    fpga_copy(filter_sub_space,
              (char *)filter_ptr + i * align_conv_sub_filter_count,
              (size_t)align_conv_sub_filter_count);
    arg->conv_args[i].filter_address = (void *)(filter_sub_space);
    fpga_flush(filter_sub_space, (size_t)align_conv_sub_filter_count);

    if (sub_conv_num == 1) {
      arg->conv_args[i].output.address = out_ptr;
      arg->conv_args[i].output.scale_address = out->scale;
    } else {
      auto ptr_output = (half *)fpga_malloc(conv_output_size * sizeof(half));
      arg->conv_args[i].output.address = (void *)((half *)ptr_output);
      auto ptr_output_scale = (float *)fpga_malloc(2 * sizeof(float));
      arg->conv_args[i].output.scale_address = ptr_output_scale;
    }
  }

  arg->output.address = out_ptr;
  arg->output.scale_address = out->scale;
  // fpga_free(filter_ptr);
}  // fill_deconv_arg

}  // namespace fpga
}  // namespace paddle_mobile
