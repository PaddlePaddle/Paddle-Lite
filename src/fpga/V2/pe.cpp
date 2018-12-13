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

#include "fpga/common/pe.h"
#include "fpga/V2/api.h"
#include "fpga/V2/filter.h"
#include "fpga/V2/image.h"
#include "fpga/common/config.h"
#include "fpga/common/driver.h"

using namespace std;                          // NOLINT
using namespace paddle_mobile::fpga::driver;  // NOLINT

namespace paddle_mobile {
namespace fpga {
#define MUL8(x) (x * 8)
#define BYPASS_DONE 2
#define CONV_DONE 1

static inline int get_image_out_axis(int src_len, int pad, int kernel_len,
                                     int kernel_step) {
  if (kernel_step == 0) {
    return 0;
  }
  return (src_len + 2 * pad - kernel_len) / kernel_step + 1;
}

float Findfp16Max() {
  uint16_t abs_vals[16];
  uint64_t max_fp16;

  max_fp16 = reg_readq(MUL8(49));
  abs_vals[0] = (uint16_t)(0x0000007fff & (max_fp16));        // NOLINT
  abs_vals[1] = (uint16_t)(0x0000007fff & (max_fp16 >> 16));  // NOLINT
  abs_vals[2] = (uint16_t)(0x0000007fff & (max_fp16 >> 32));  // NOLINT
  abs_vals[3] = (uint16_t)(0x0000007fff & (max_fp16 >> 48));  // NOLINT
  max_fp16 = reg_readq(MUL8(50));
  abs_vals[4] = (uint16_t)(0x0000007fff & (max_fp16));        // NOLINT
  abs_vals[5] = (uint16_t)(0x0000007fff & (max_fp16 >> 16));  // NOLINT
  abs_vals[6] = (uint16_t)(0x0000007fff & (max_fp16 >> 32));  // NOLINT
  abs_vals[7] = (uint16_t)(0x0000007fff & (max_fp16 >> 48));  // NOLINT
  max_fp16 = reg_readq(MUL8(51));
  abs_vals[8] = (uint16_t)(0x0000007fff & (max_fp16));         // NOLINT
  abs_vals[9] = (uint16_t)(0x0000007fff & (max_fp16 >> 16));   // NOLINT
  abs_vals[10] = (uint16_t)(0x0000007fff & (max_fp16 >> 32));  // NOLINT
  abs_vals[11] = (uint16_t)(0x0000007fff & (max_fp16 >> 48));  // NOLINT
  max_fp16 = reg_readq(MUL8(52));
  abs_vals[12] = (uint16_t)(0x0000007fff & (max_fp16));
  abs_vals[13] = (uint16_t)(0x0000007fff & (max_fp16 >> 16));  // NOLINT
  abs_vals[14] = (uint16_t)(0x0000007fff & (max_fp16 >> 32));  // NOLINT
  abs_vals[15] = (uint16_t)(0x0000007fff & (max_fp16 >> 48));  // NOLINT

  uint16_t tmp = 0;
  for (int i = 0; i < 16; i++) {
    if (tmp < abs_vals[i]) {
      tmp = abs_vals[i];
    }
  }
  DLOG << "max value found: " << fp16_2_fp32(tmp);
  return fp16_2_fp32(tmp) / 127.0f;
}

int ComputeFpgaConv(const struct SplitConvArgs &args) {
  ComputeBasicConv(args.conv_arg[0]);
}

int ComputeBasicConv(const struct ConvArgs &args) {
#ifdef FPGA_PRINT_MODE
  DLOG << "======Compute Basic Conv======";
  DLOG << "   relu_enabled:" << args.relu_enabled
       << "   sb_address:" << args.sb_address
       << "   filter_address:" << args.filter_address
       << "   filter_num:" << args.filter_num
       << "   group_num:" << args.group_num;
  DLOG << "   image_address:" << args.image.address
       << "   image_scale_address:" << args.image.scale_address
       << "   image_channels:" << args.image.channels
       << "   image_height:" << args.image.height
       << "   image_width:" << args.image.width
       << "   pad_height:" << args.image.pad_height
       << "   pad_width:" << args.image.pad_width;
  DLOG << "   kernel_height:" << args.kernel.height
       << "   kernel_width:" << args.kernel.width
       << "   stride_h:" << args.kernel.stride_h
       << "   stride_w:" << args.kernel.stride_w;
  DLOG << "   out_address:" << args.output.address
       << "   out_scale_address:" << args.output.scale_address;
#endif

#ifndef PADDLE_MOBILE_ZU5
  return 0;
#endif

  uint64_t ifm_pixel_num =
      ((args.image.width) * (args.image.height) * args.image.channels);
  uint64_t ifm_memory_size = ifm_pixel_num * sizeof(short);          // NOLINT
  uint64_t flt_pixel_num = (args.filter_num * (args.kernel.width) *  // NOLINT
                            (args.kernel.height) * args.image.channels);
  uint64_t filter_memory_size = flt_pixel_num * sizeof(short);  // NOLINT

  uint64_t bn_pixel_num = (args.filter_num * 2);  // NOLINT
  uint64_t bn_memory_size = bn_pixel_num * sizeof(float);

  uint64_t ofm_width =
      ((args.image.width) + 2 * args.image.pad_width - args.kernel.width) /
          (args.kernel.stride_w) +
      1;
  uint64_t ofm_height = ((args.image.height) + 2 * (args.image.pad_height) -
                         (args.kernel.height)) /
                            (args.kernel.stride_h) +
                        1;

  uint32_t filter_num = args.filter_num;
  uint32_t image_channels = args.image.channels;

  DLOG << "filter_num: " << filter_num;
  uint64_t ifm_src_paddr = vaddr_to_paddr((args.image.address));
  uint64_t flt_src_paddr = vaddr_to_paddr((args.filter_address));
  uint64_t sb_src_paddr = vaddr_to_paddr((args.free_space));
  uint64_t ifm_dst_paddr = vaddr_to_paddr((args.output.address));
  /**********BN******************/
  float image_inv_scale = (args.image.scale_address)[0];
  float filter_inv_scale = (args.filter_scale_address)[0];
  float scale_tmp = image_inv_scale * filter_inv_scale;
  int idx = 0;
  float tmp = 0.0;
  float *convert_sb_addr = (float *)(args.free_space);  // NOLINT
  for (idx = 0; idx < args.filter_num * 2; idx++) {
    if (idx % 2 == 1) {
      tmp = ((float *)(args.sb_address))[idx] * scale_tmp;  // NOLINT
    } else {
      tmp = ((float *)(args.sb_address))[idx];  // NOLINT
    }
    convert_sb_addr[idx] = tmp;  // NOLINT
  }

  fpga_flush(convert_sb_addr, args.filter_num * 2 * sizeof(float));
  reg_writeq(1, MUL8(24));
  usleep(1);
  reg_writeq(0, MUL8(24));

  reg_writeq(sb_src_paddr, MUL8(27));
  reg_writeq(0, MUL8(0));

  uint64_t bps_addr = 0x8c00000000000000;
  bps_addr += bn_memory_size;
  reg_writeq(bps_addr, MUL8(0));
  int ret = -1;
  ret = fpga_regpoll(MUL8(48), BYPASS_DONE, 0xffffff);
  if (ret) {
    DLOG << "conv bypass failed";
    return ret;
  }
  reg_readq(MUL8(63));

  /*********configuring registers*************/
  uint32_t cmd_image_vir_base_addr = (uint32_t)ifm_src_paddr;
  uint32_t cmd_filter_vir_base_addr = (uint32_t)flt_src_paddr;
  uint32_t cmd_scale_base_addr = (uint32_t)sb_src_paddr;
  uint32_t conv_ofm_addr_base = (uint32_t)ifm_dst_paddr;
  uint64_t cmd_group_num = args.group_num;
  uint64_t cmd_filter_per_group = filter_num / cmd_group_num;

  uint64_t cmd_flt_sqr_len = (args.kernel.width) * (args.kernel.height);
  uint64_t cmd_ifm_pre_row_num = 0;

  if (1 == args.image.height) {
    cmd_ifm_pre_row_num = 1;
  } else {
    cmd_ifm_pre_row_num =
        (args.kernel.height) - (args.image.pad_height) + (args.kernel.stride_h);
  }
  uint64_t cmd_flt_pre_batch_num = 1;
  uint64_t cmd_ifm_pack_num_per_row_mns1 =
      (uint64_t)(((args.image.channels) + 127) / 128) - 1;
  uint64_t cmd_bn_num = filter_num;
  uint64_t cmd_bias_num = filter_num;
  uint64_t cmd_ifm_stride_row_length = args.image.width * args.kernel.stride_h;
  uint64_t cmd_flt_pack_num_per_kernel_mns1 =
      (uint64_t)(((args.image.channels) + 127) / 128) - 1;
  uint64_t cmd_ofm_width_mns1 = (uint64_t)(
      ((args.image.width) - (args.kernel.width) + 2 * (args.image.pad_width)) /
      (args.kernel.stride_w));
  uint64_t cmd_ofm_height =
      (uint64_t)(((args.image.height) - (args.kernel.height) +
                  2 * (args.image.pad_height)) /
                 (args.kernel.stride_h)) +
      1;

  uint64_t cmd_channel_num = 0;
  uint64_t cmd_ifm_pack_len = 0;
  uint64_t cmd_channel_per_group = 0;
  uint64_t cmd_flt_batch_num_mns1 = 0;
  uint64_t cmd_flt_N_impl = 8;
  uint64_t cmd_ifm_C_impl = 16;
  uint64_t cmd_flt_pack_length = 0;
  uint64_t cmd_step_h_mul_row_byte_len = 0;
  uint64_t cmd_pad_h_mul_row_byte_len = 0;
  uint64_t cmd_ifm_pack_byte_length = 16 * ((((args.image.width) + 7) / 8) * 8);
  uint64_t row_len_align = args.image.width;
  if (image_channels > 64) {
    cmd_channel_num = (uint64_t)((((args.image.channels) + 127)) / 128) * 128;
    cmd_ifm_pack_len = 128 * (args.image.width);
    cmd_channel_per_group = 128;
    cmd_flt_batch_num_mns1 = (uint64_t)(((args.filter_num + 7)) / 8 - 1);
    cmd_flt_N_impl = 8;
    cmd_ifm_C_impl = 128;
    cmd_flt_pack_length = (args.kernel.width) * (args.kernel.height) * 128;
    cmd_step_h_mul_row_byte_len =
        (args.kernel.stride_h) * cmd_channel_num * (args.image.width);
    cmd_pad_h_mul_row_byte_len =
        (args.image.pad_height) * cmd_channel_num * (args.image.width);
    cmd_ifm_pack_byte_length = 128 * (args.image.width);
    row_len_align = args.image.width * (cmd_ifm_pack_num_per_row_mns1 + 1);
  } else if (image_channels > 32) {
    cmd_channel_num = 64;
    cmd_ifm_pack_len = 64 * (args.image.width);
    cmd_channel_per_group = 64;
    cmd_flt_batch_num_mns1 = (uint64_t)((((args.filter_num) + 15)) / 16 - 1);
    cmd_flt_N_impl = 16;
    cmd_ifm_C_impl = 64;
    cmd_flt_pack_length = (args.kernel.width) * (args.kernel.height) * 64;
    cmd_step_h_mul_row_byte_len = (args.kernel.stride_h) * cmd_channel_num *
                                  ((((args.image.width) + 1)) / 2) * 2;
    cmd_pad_h_mul_row_byte_len = (args.image.pad_height) * cmd_channel_num *
                                 ((((args.image.width) + 1)) / 2) * 2;
    cmd_ifm_pack_byte_length =
        64 * (uint64_t)((((args.image.width) + 1)) / 2) * 2;
    row_len_align = (uint64_t)((((args.image.width) + 1)) / 2);
  } else if (image_channels > 16) {
    cmd_channel_num = 32;
    cmd_ifm_pack_len = 32 * (args.image.width);
    cmd_channel_per_group = 32;
    cmd_flt_batch_num_mns1 = (uint64_t)((((args.filter_num) + 31)) / 32 - 1);
    cmd_flt_N_impl = 32;
    cmd_ifm_C_impl = 32;
    cmd_flt_pack_length = (args.kernel.width) * (args.kernel.height) * 32;
    cmd_step_h_mul_row_byte_len = (args.kernel.stride_h) * cmd_channel_num *
                                  ((((args.image.width) + 3)) / 4) * 4;
    cmd_pad_h_mul_row_byte_len = (args.image.pad_height) * cmd_channel_num *
                                 ((((args.image.width) + 3)) / 4) * 4;
    cmd_ifm_pack_byte_length =
        32 * (uint64_t)((((args.image.width) + 3)) / 4) * 4;
    row_len_align = (uint64_t)((((args.image.width) + 3)) / 4);
  } else {
    cmd_channel_num = 16;
    cmd_ifm_pack_len = 16 * (args.image.width);
    cmd_channel_per_group = 16;
    cmd_flt_batch_num_mns1 = (uint64_t)((((args.filter_num) + 63)) / 64 - 1);
    cmd_flt_N_impl = 64;
    cmd_ifm_C_impl = 16;
    cmd_flt_pack_length = (args.kernel.width) * (args.kernel.height) * 16;
    cmd_step_h_mul_row_byte_len = (args.kernel.stride_h) * cmd_channel_num *
                                  ((((args.image.width) + 7)) / 8) * 8;
    cmd_pad_h_mul_row_byte_len = (args.image.pad_height) * cmd_channel_num *
                                 ((((args.image.width) + 7)) / 8) * 8;
    cmd_ifm_pack_byte_length = 16 * ((((args.image.width) + 7)) / 8) * 8;
    row_len_align = (uint64_t)((((args.image.width) + 7)) / 8);
  }
  uint64_t cmd_flt_length =
      (args.kernel.width) * (args.kernel.height) * cmd_channel_num;
  uint64_t cmd_ifm_row_byte_length = cmd_channel_num * (args.image.width);

  uint64_t cmd_ifm_buf_col_len = 0;

  uint64_t ifm_one_batch_len =
      (1048576 / ((args.image.width) * cmd_channel_num));
  uint64_t cmd_ifm_batch_num_tmp = (uint64_t)(
      ((args.image.height) + ifm_one_batch_len - 1) / ifm_one_batch_len);
  if (1 == cmd_ifm_batch_num_tmp) {
    cmd_ifm_buf_col_len = args.image.height;
  } else {
    if (((args.image.height) / (cmd_ifm_batch_num_tmp) % 2) == 0) {
      cmd_ifm_buf_col_len = (args.image.height) / cmd_ifm_batch_num_tmp;
    } else {
      cmd_ifm_buf_col_len = (args.image.height) / cmd_ifm_batch_num_tmp - 1;
    }
  }
  uint64_t cmd_ifm_batch_num_mns1 =
      (((args.image.height) + cmd_ifm_buf_col_len - 1) / cmd_ifm_buf_col_len) -
      1;
  uint64_t cmd_flt_cycle_num_mns1 = cmd_ifm_batch_num_mns1;
  uint64_t cmd_flt_total_batch_num = filter_num / cmd_flt_N_impl;
  uint64_t cmd_ifm_buf_col_len_rem =
      (args.image.height) -
      cmd_ifm_batch_num_mns1 * cmd_ifm_buf_col_len;  //= -4;
  uint64_t cmd_flt_N_len = args.kernel.width * args.kernel.height *
                           (cmd_flt_pack_num_per_kernel_mns1 + 1);

  //-------- ofm batch number reg &&  initial URAM reading address
  // logic-----------------
  uint64_t cmd_init_raddr_cnt = 1;
  uint64_t cmd_init_raddr_flag = 0;
  int64_t cmd_init_raddr_index = -8;
  int64_t cmd_init_raddr_col_0 = -4;
  int64_t cmd_init_raddr_col_1 = -4;
  uint64_t conv_ofm_buf_col_len = 0;
  uint64_t conv_ofm_buf_col_len_rem = 0;

  if (((args.image.pad_height) % (2 * (args.kernel.stride_h))) == 0) {
    cmd_init_raddr_cnt = 0;
    cmd_init_raddr_flag = 0;
    cmd_init_raddr_index =
        0 - (int64_t)row_len_align * (((args.image.pad_height) + 1) / 2);
    cmd_init_raddr_col_0 = cmd_init_raddr_index;
    cmd_init_raddr_col_1 = cmd_init_raddr_index;
  } else if (((args.image.pad_height) -
              2 * ((args.image.pad_height) / (2 * (args.kernel.stride_h)))) <=
             (args.kernel.stride_h)) {
    cmd_init_raddr_cnt =
        (args.kernel.stride_h) -
        ((args.image.pad_height) -
         ((args.image.pad_height) / (2 * (args.kernel.stride_h))));
    cmd_init_raddr_flag = 1;
    cmd_init_raddr_index =
        0 - (int64_t)row_len_align * (int64_t)(args.image.pad_height) -
        (int64_t)row_len_align *
            ((args.image.pad_height) / (2 * args.kernel.stride_h));
    cmd_init_raddr_col_0 =
        0 - (int64_t)row_len_align * (int64_t)(args.image.pad_height) -
        (int64_t)row_len_align *
            ((args.image.pad_height) / (2 * (args.kernel.stride_h)));
    cmd_init_raddr_col_1 = 0;
  } else if (((args.image.pad_height) -
              2 * ((args.image.pad_height) / (2 * (args.kernel.stride_h)))) <=
             2 * (args.kernel.stride_h)) {
    cmd_init_raddr_cnt =
        2 * (args.kernel.stride_h) *
            (((args.image.pad_height) + 2 * (args.kernel.stride_h) - 1) /
             (2 * (args.kernel.stride_h))) -
        (args.image.pad_height);
    cmd_init_raddr_flag = 0;
    cmd_init_raddr_index =
        0 - (int64_t)row_len_align * (int64_t)(args.kernel.stride_h) *
                (((args.image.pad_height) + 2 * (args.kernel.stride_h) - 1) /
                 (2 * (args.kernel.stride_h)));
    cmd_init_raddr_col_0 =
        0 -
        (int64_t)row_len_align *
            ((args.image.pad_height) / (2 * (args.kernel.stride_h))) -
        (int64_t)row_len_align *
            (2 * (args.kernel.stride_h) *
                 (((args.image.pad_height) + 2 * (args.kernel.stride_h) - 1) /
                  (2 * (args.kernel.stride_h))) -
             (args.image.pad_height));
    cmd_init_raddr_col_1 = cmd_init_raddr_col_0;
  }

  if (cmd_ifm_batch_num_mns1 == 0) {
    if ((args.kernel.height) <= (args.kernel.stride_h)) {
      conv_ofm_buf_col_len = (args.image.height) + 2 * (args.image.pad_height) -
                             3 * (args.kernel.stride_h);
    } else {
      conv_ofm_buf_col_len = (args.image.height) + 2 * (args.image.pad_height) -
                             2 * (args.kernel.stride_h) - (args.kernel.height);
    }
    conv_ofm_buf_col_len_rem = conv_ofm_buf_col_len;
  } else {
    int N_rem = 0;
    int row_rem = 0;

    if ((args.kernel.height) <= (args.kernel.stride_h)) {
      conv_ofm_buf_col_len = cmd_ifm_buf_col_len - 3 * (args.kernel.stride_h);
      N_rem = (cmd_ifm_buf_col_len - (args.kernel.height)) /
                  (args.kernel.stride_h) +
              1;
      row_rem = cmd_ifm_buf_col_len - (args.kernel.stride_h) * N_rem;
      conv_ofm_buf_col_len_rem = cmd_ifm_buf_col_len_rem +
                                 2 * (args.image.pad_height) + row_rem -
                                 3 * (args.kernel.stride_h);
    } else {
      conv_ofm_buf_col_len = cmd_ifm_buf_col_len + 2 * (args.image.pad_height) -
                             2 * (args.kernel.stride_h) - (args.kernel.height);
      N_rem = (cmd_ifm_buf_col_len - (args.kernel.height)) /
                  (args.kernel.stride_h) +
              1;
      row_rem = cmd_ifm_buf_col_len - (args.kernel.stride_h) * N_rem;
      conv_ofm_buf_col_len_rem =
          cmd_ifm_buf_col_len_rem + (args.image.pad_height) + row_rem -
          2 * (args.kernel.stride_h) - (args.kernel.height);
    }
  }
  //-----------------------  para functions --------------------------------
  float filter_quant_scale_tmp = ((args.filter_scale_address)[1]);
  float image_quant_scale_tmp = ((args.image.scale_address)[1]);

  uint32_t cmd_filter_quant_scale =
      *(uint32_t *)(&filter_quant_scale_tmp);  // NOLINT
  uint32_t cmd_image_quant_scale =
      *(uint32_t *)(&image_quant_scale_tmp);  // NOLINT

  uint64_t wParallelsim = cmd_flt_N_impl >> 3;
  uint64_t wParallelsim_num =
      (uint64_t)(((args.filter_num) + cmd_flt_N_impl - 1) / cmd_flt_N_impl) - 1;
  uint64_t win_size = (args.kernel.width) * (args.kernel.height) *
                          (cmd_ifm_pack_num_per_row_mns1 + 1) -
                      1;
  uint64_t conv_ofm_width = (((args.image.width) - (args.kernel.width) +
                              (args.image.pad_width) + (args.image.pad_width)) /
                             (args.kernel.stride_w));
  uint64_t conv_ofm_dma_length = cmd_flt_N_impl * sizeof(short);   // NOLINT
  uint64_t conv_ofm_dma_stride = args.filter_num * sizeof(short);  // NOLINT
  uint64_t conv_ofm_height_batch_tmp =
      get_image_out_axis(args.image.height, args.image.pad_height,
                         args.kernel.height, args.kernel.stride_h);
  uint64_t conv_ofm_height_batch = (conv_ofm_height_batch_tmp + 1) / 2 - 1;
  uint64_t o_ust_rst = 0;
  uint64_t conv_ofm_dma_repeat =
      (uint64_t)(((((args.image.width) - (args.kernel.width) +
                    (args.image.pad_width) + (args.image.pad_width))) /
                  (args.kernel.stride_w)) +
                 1);
  uint64_t conv_ofm_dma_offset =
      args.filter_num * conv_ofm_dma_repeat * sizeof(short);  // NOLINT
  uint64_t conv_ofm_inter_stride = conv_ofm_dma_offset * 2;
  //----------------- register contation ------------------
  uint64_t cmd_ifm_flt_base_addr = ((uint64_t)cmd_filter_vir_base_addr << 32) |
                                   ((uint64_t)cmd_image_vir_base_addr);
  uint64_t cmd_ifm_flt_dim = ((uint64_t)(args.kernel.height) << 48) |
                             ((uint64_t)(args.kernel.width) << 32) |
                             ((uint64_t)(args.image.height) << 16) |
                             ((uint64_t)(args.image.width));
  uint64_t cmd_pad_step_size = ((uint64_t)(args.kernel.stride_h) << 48) |
                               ((uint64_t)(args.kernel.stride_w) << 32) |
                               ((uint64_t)(args.image.pad_height) << 16) |
                               ((uint64_t)(args.image.pad_width));
  uint64_t cmd_param1 = ((uint64_t)cmd_filter_per_group << 48) |
                        ((uint64_t)cmd_channel_num << 32) |
                        ((uint64_t)filter_num << 16) |
                        ((uint64_t)cmd_group_num);
  uint64_t cmd_param2 =
      ((uint64_t)cmd_flt_sqr_len << 48) | ((uint64_t)cmd_ifm_pack_len << 32) |
      ((uint64_t)cmd_ifm_pre_row_num << 16) | ((uint64_t)cmd_channel_per_group);
  uint64_t cmd_param3 = ((uint64_t)cmd_flt_batch_num_mns1 << 48) |
                        ((uint64_t)cmd_flt_total_batch_num << 32) |
                        ((uint64_t)cmd_flt_N_impl << 16) |
                        ((uint64_t)cmd_flt_pre_batch_num);
  uint64_t cmd_param4 = ((uint64_t)cmd_ifm_pack_num_per_row_mns1 << 48) |
                        ((uint64_t)cmd_bn_num << 32) |
                        ((uint64_t)cmd_bias_num << 16) |
                        ((uint64_t)cmd_flt_N_len);
  uint64_t cmd_param5 = ((uint64_t)cmd_ifm_stride_row_length << 48) |
                        ((uint64_t)cmd_flt_pack_length << 32) |
                        ((uint64_t)cmd_flt_cycle_num_mns1 << 16) |
                        ((uint64_t)cmd_flt_pack_num_per_kernel_mns1);
  uint64_t cmd_param6 = ((uint64_t)cmd_ofm_width_mns1 << 48) |
                        ((uint64_t)cmd_ifm_batch_num_mns1 << 32) |
                        ((uint64_t)cmd_ifm_buf_col_len << 16) |
                        ((uint64_t)cmd_ifm_C_impl);
  uint64_t cmd_param7 = ((uint64_t)conv_ofm_inter_stride << 32) |
                        ((uint64_t)cmd_ifm_buf_col_len_rem << 16) |
                        ((uint64_t)cmd_ofm_height);
  uint64_t cmd_param8 =
      ((uint64_t)cmd_flt_length << 32) | ((uint64_t)cmd_ifm_row_byte_length);
  uint64_t cmd_ifm_flt_quant_scale =
      (((uint64_t)cmd_filter_quant_scale) << 32) |
      ((uint64_t)cmd_image_quant_scale);
  uint64_t cmd_step_pad_mul_row_len =
      ((uint64_t)cmd_pad_h_mul_row_byte_len << 32) |
      ((uint64_t)cmd_step_h_mul_row_byte_len);
  //---- ofm paras ----
  uint64_t cmd_conv_param_reg = ((uint64_t)wParallelsim_num << 32) |
                                ((uint64_t)wParallelsim << 16) |
                                ((uint64_t)win_size);
  uint64_t cmd_ofm_addr_width_reg =
      ((uint64_t)conv_ofm_width << 32) | ((uint64_t)conv_ofm_addr_base);
  uint64_t cmd_intra_stride_atoms_reg =
      ((uint64_t)conv_ofm_dma_length << 32) | ((uint64_t)conv_ofm_dma_stride);
  uint64_t cmd_ofm_height_batch_reg =
      ((uint64_t)conv_ofm_buf_col_len_rem << 48) |
      ((uint64_t)conv_ofm_buf_col_len << 32) |
      ((uint64_t)conv_ofm_height_batch + 0x80000000);
  uint64_t cmd_user_ctrl_reg = ((uint64_t)o_ust_rst);
  uint64_t cmd_wdma_param_reg =
      ((uint64_t)(conv_ofm_dma_repeat | 0x80000000) << 32) |
      ((uint64_t)conv_ofm_dma_offset);

  uint64_t cmd_init_raddr_reg = ((cmd_init_raddr_col_1 & 0xffff) << 48) |
                                ((cmd_init_raddr_col_0 & 0xffff) << 32) |
                                (((cmd_init_raddr_index & 0xffff) << 16)) |
                                (cmd_init_raddr_flag & 0xffff) << 15 |
                                ((cmd_init_raddr_cnt & 0xffff));

  uint64_t cmd_para31 = (cmd_para31 & 0x1) | args.relu_enabled;

  DLOG << "cmd_init_raddr_col_1 = " << hex << cmd_init_raddr_col_1;

  DLOG << "cmd_init_raddr_col_0 = " << hex << cmd_init_raddr_col_0;
  DLOG << "cmd_init_raddr_index = " << hex << cmd_init_raddr_index;  //
  DLOG << "cmd_init_raddr_cnt = " << hex << cmd_init_raddr_cnt;
  DLOG << "conv_ofm_height_batch = " << conv_ofm_height_batch;

  DLOG << "cmd_ifm_flt_base_addr = " << hex << cmd_ifm_flt_base_addr;
  DLOG << "cmd_scale_base_addr = " << hex << cmd_scale_base_addr;
  DLOG << "cmd_ifm_flt_dim = " << hex << cmd_ifm_flt_dim;
  DLOG << "cmd_pad_step_size = " << hex << cmd_pad_step_size;
  DLOG << "cmd_param1 = " << hex << cmd_param1;
  DLOG << "cmd_param2 = " << hex << cmd_param2;
  DLOG << "cmd_param3 = " << hex << cmd_param3;
  DLOG << "cmd_param4 = " << hex << cmd_param4;
  DLOG << "cmd_param5 = " << hex << cmd_param5;
  DLOG << "cmd_param6 = " << hex << cmd_param6;
  DLOG << "cmd_param7 = " << hex << cmd_param7;
  DLOG << "cmd_param8 =  " << hex << cmd_param8;
  DLOG << "cmd_ifm_flt_quant_scale =  " << hex << cmd_ifm_flt_quant_scale;
  DLOG << "cmd_step_pad_mul_row_len = " << hex << cmd_step_pad_mul_row_len;
  DLOG << "cmd_ifm_pack_byte_length = " << hex << cmd_ifm_pack_byte_length;
  DLOG << "cmd_conv_param_reg = " << hex << cmd_conv_param_reg;
  DLOG << "cmd_ofm_addr_width_reg = " << hex << cmd_ofm_addr_width_reg;
  DLOG << "cmd_intra_stride_atoms_reg = " << hex << cmd_intra_stride_atoms_reg;
  DLOG << "cmd_init_raddr_reg = " << hex << cmd_init_raddr_reg;
  DLOG << "cmd_ofm_height_batch_reg = " << hex << cmd_ofm_height_batch_reg;
  DLOG << "cmd_wdma_param_reg = " << hex << cmd_wdma_param_reg;
  DLOG << "cmd_para31 = " << hex << cmd_para31;

  reg_writeq(cmd_ifm_flt_base_addr, MUL8(1));
  reg_writeq(cmd_scale_base_addr, MUL8(2));
  reg_writeq(cmd_ifm_flt_dim, MUL8(3));
  reg_writeq(cmd_pad_step_size, MUL8(4));
  reg_writeq(cmd_param1, MUL8(5));
  reg_writeq(cmd_param2, MUL8(6));
  reg_writeq(cmd_param3, MUL8(7));
  reg_writeq(cmd_param4, MUL8(8));
  reg_writeq(cmd_param5, MUL8(9));
  reg_writeq(cmd_param6, MUL8(10));
  reg_writeq(cmd_param7, MUL8(11));
  reg_writeq(cmd_param8, MUL8(12));
  reg_writeq(cmd_ifm_flt_quant_scale, MUL8(13));
  reg_writeq(cmd_step_pad_mul_row_len, MUL8(14));
  reg_writeq(cmd_ifm_pack_byte_length, MUL8(15));
  reg_writeq(cmd_conv_param_reg, MUL8(16));
  reg_writeq(cmd_ofm_addr_width_reg, MUL8(17));
  reg_writeq(cmd_intra_stride_atoms_reg, MUL8(18));

  reg_writeq(cmd_init_raddr_reg, MUL8(29));
  reg_writeq(cmd_para31, MUL8(31));

  reg_writeq(0, MUL8(19));
  reg_writeq(cmd_ofm_height_batch_reg, MUL8(19));
  reg_writeq(cmd_ofm_height_batch_reg & 0xffffffff00000000, MUL8(19));

  reg_writeq(cmd_wdma_param_reg, MUL8(25));

  reg_writeq(0, MUL8(0));
  reg_writeq(0x4000000000000000, MUL8(0));

  ret = fpga_regpoll(MUL8(48), CONV_DONE, 0xffffff);
  if (ret == -1) {
    DLOG << "fpga conv no interrupt!!";
    return ret;
  }
  reg_readq(MUL8(63));

  usleep(10);
  float scale = Findfp16Max();
  (args.output.scale_address)[0] = scale;                 // NOLINT
  (args.output.scale_address)[1] = (float)(1.0 / scale);  // NOLINT
  DLOG << "Findfp16Max scale = " << scale;
  DLOG << "ret=" << ret;
  return ret;
}

int ComputeFpgaPool(const struct PoolingArgs &args) {
#ifdef FPGA_PRINT_MODE
  DLOG << "=============ComputeFpgaPool===========";
  DLOG << "   mode:" << args.mode
       << "   kernel_reciprocal:" << fp16_2_fp32(args.kernel_reciprocal);
  DLOG << "   image_address:" << args.image.address
       << "   image_channels:" << args.image.channels
       << "   image_height:" << args.image.height
       << "   image_width:" << args.image.width
       << "   pad_height:" << args.image.pad_height
       << "   pad_width:" << args.image.pad_width;
  DLOG << "   kernel_height:" << args.kernel.height
       << "   kernel_width:" << args.kernel.width
       << "   stride_h:" << args.kernel.stride_h
       << "   stride_w:" << args.kernel.stride_w;
  DLOG << "   out_address:" << args.output.address;
#endif
#ifndef PADDLE_MOBILE_ZU5
  return 0;
#endif

  uint32_t filter_num_align = 0;
  filter_num_align = args.image.channels;

  DLOG << "______db_______: begin to set registers. ";
  uint64_t ifm_pixel_num =
      ((args.image.width) * (args.image.height) * args.image.channels);
  uint64_t ifm_memory_size = ifm_pixel_num * sizeof(short);  // NOLINT
  uint64_t flt_pixel_num = 0;
  uint64_t filter_memory_size = 0;
  //!! ???
  uint64_t bn_pixel_num = (filter_num_align * 2);
  uint64_t bn_memory_size = bn_pixel_num * sizeof(uint16_t);

  uint64_t ofm_width =
      ((args.image.width) + 2 * args.image.pad_width - args.kernel.width) /
          (args.kernel.stride_w) +
      1;
  uint64_t ofm_height = ((args.image.height) + 2 * (args.image.pad_height) -
                         (args.kernel.height)) /
                            (args.kernel.stride_h) +
                        1;

  uint32_t filter_num = filter_num_align;
  uint32_t image_channels = args.image.channels;

  uint64_t ifm_src_paddr = vaddr_to_paddr((args.image.address));
  uint64_t flt_src_paddr = 0;
  uint64_t sb_src_paddr = 0;
  uint64_t ifm_dst_paddr = vaddr_to_paddr((args.output.address));

  /**********BN******************/
  float image_inv_scale = 0;
  float filter_inv_scale = 0;
  int idx = 0;
  DLOG << "______db_______: reset registers. ";
  reg_writeq(1, MUL8(24));
  usleep(1);
  reg_writeq(0, MUL8(24));
  /*********configuring registers*************/
  uint32_t cmd_image_vir_base_addr = (uint32_t)ifm_src_paddr;
  uint32_t cmd_filter_vir_base_addr = (uint32_t)flt_src_paddr;
  uint32_t cmd_scale_base_addr = (uint32_t)sb_src_paddr;
  uint32_t conv_ofm_addr_base = (uint32_t)ifm_dst_paddr;
  uint64_t cmd_group_num = 1;  // args.group_num;
  uint64_t cmd_filter_per_group = filter_num / cmd_group_num;

  uint64_t cmd_flt_sqr_len = (args.kernel.width) * (args.kernel.height);
  uint64_t cmd_ifm_pre_row_num = args.kernel.height;
  if ((args.kernel.height == args.image.height) &&
      (0 == args.image.pad_height)) {
    cmd_ifm_pre_row_num = (args.kernel.height);
  } else {
    cmd_ifm_pre_row_num =
        (args.kernel.height) - (args.image.pad_height) + (args.kernel.stride_h);
  }
  uint64_t cmd_flt_pre_batch_num = 1;
  uint64_t cmd_ifm_pack_num_per_row_mns1 =
      (uint64_t)(((args.image.channels) + 63) / 64) - 1;
  uint64_t cmd_bn_num = filter_num;
  uint64_t cmd_bias_num = filter_num;
  uint64_t cmd_ifm_stride_row_length = args.image.width * args.kernel.stride_h;
  uint64_t cmd_flt_pack_num_per_kernel_mns1 =
      (uint64_t)(((args.image.channels) + 63) / 64) - 1;
  uint64_t cmd_ofm_width_mns1 = (uint64_t)(
      ((args.image.width) - (args.kernel.width) + 2 * (args.image.pad_width)) /
      (args.kernel.stride_w));
  uint64_t cmd_ofm_height =
      (uint64_t)(((args.image.height) - (args.kernel.height) +
                  2 * (args.image.pad_height)) /
                 (args.kernel.stride_h)) +
      1;

  uint64_t cmd_channel_num = 0;
  uint64_t cmd_ifm_pack_len = 0;
  uint64_t cmd_channel_per_group = 0;
  uint64_t cmd_flt_batch_num_mns1 = 0;
  uint64_t cmd_flt_N_impl = 8;
  uint64_t cmd_ifm_C_impl = 16;
  uint64_t cmd_flt_pack_length = 0;
  uint64_t cmd_step_h_mul_row_byte_len = 0;
  uint64_t cmd_pad_h_mul_row_byte_len = 0;
  uint64_t cmd_ifm_pack_byte_length = 16 * ((((args.image.width) + 7) / 8) * 8);
  uint64_t row_len_align = args.image.width;
  uint64_t cmd_flt_cycle_num_mns1 = 0;
  if (image_channels > 32) {
    cmd_channel_num = (uint64_t)((((args.image.channels) + 63)) / 64) * 64;
    cmd_ifm_pack_len = 64 * (args.image.width);
    cmd_channel_per_group = 64;
    cmd_flt_batch_num_mns1 = (uint64_t)(((filter_num + 7)) / 8 - 1);
    cmd_flt_N_impl = 8;
    cmd_ifm_C_impl = 64;
    cmd_flt_pack_length = (args.kernel.width) * (args.kernel.height) * 64;
    cmd_step_h_mul_row_byte_len =
        (args.kernel.stride_h) * cmd_channel_num * args.image.width;
    cmd_pad_h_mul_row_byte_len =
        (args.image.pad_height) * cmd_channel_num * args.image.width;
    cmd_ifm_pack_byte_length = 64 * args.image.width;
    row_len_align = args.image.width * (cmd_ifm_pack_num_per_row_mns1 + 1);
    cmd_flt_cycle_num_mns1 = (cmd_channel_num / 64) - 1;
  } else if (image_channels > 16) {
    cmd_channel_num = 32;
    cmd_ifm_pack_len = 32 * (args.image.width);
    cmd_channel_per_group = 32;
    cmd_flt_batch_num_mns1 = (uint64_t)((((filter_num) + 15)) / 16 - 1);
    cmd_flt_N_impl = 16;
    cmd_ifm_C_impl = 32;
    cmd_flt_pack_length = (args.kernel.width) * (args.kernel.height) * 32;
    cmd_step_h_mul_row_byte_len = (args.kernel.stride_h) * cmd_channel_num *
                                  ((((args.image.width) + 1)) / 2) * 2;
    cmd_pad_h_mul_row_byte_len = (args.image.pad_height) * cmd_channel_num *
                                 ((((args.image.width) + 1)) / 2) * 2;
    cmd_ifm_pack_byte_length =
        32 * (uint64_t)((((args.image.width) + 1)) / 2) * 2;
    row_len_align = (uint64_t)((((args.image.width) + 1)) / 2);
    cmd_flt_cycle_num_mns1 = 0;
  } else if (image_channels > 8) {
    cmd_channel_num = 16;
    cmd_ifm_pack_len = 16 * (args.image.width);
    cmd_channel_per_group = 16;
    cmd_flt_batch_num_mns1 = (uint64_t)((((filter_num) + 15)) / 16 - 1);
    cmd_flt_N_impl = 32;
    cmd_ifm_C_impl = 16;
    cmd_flt_pack_length = (args.kernel.width) * (args.kernel.height) * 16;
    cmd_step_h_mul_row_byte_len = (args.kernel.stride_h) * cmd_channel_num *
                                  ((((args.image.width) + 3)) / 4) * 4;
    cmd_pad_h_mul_row_byte_len = (args.image.pad_height) * cmd_channel_num *
                                 ((((args.image.width) + 3)) / 4) * 4;
    cmd_ifm_pack_byte_length =
        16 * (uint64_t)((((args.image.width) + 3)) / 4) * 4;
    row_len_align = (uint64_t)((((args.image.width) + 3)) / 4);
    cmd_flt_cycle_num_mns1 = 0;
  }

  cmd_flt_N_impl = 16;
  cmd_flt_batch_num_mns1 = 0;
  cmd_flt_pack_length = 64;
  uint64_t cmd_flt_N_len = 0;
  uint64_t cmd_flt_length = 64;

  uint64_t cmd_ifm_row_byte_length = cmd_channel_num * (args.image.width);

  uint64_t cmd_ifm_buf_col_len = 0;

  uint64_t ifm_one_batch_len =
      (1048576 / ((args.image.width) * cmd_channel_num));
  uint64_t cmd_ifm_batch_num_tmp = (uint64_t)(
      ((args.image.height) + ifm_one_batch_len - 1) / ifm_one_batch_len);
  if (1 == cmd_ifm_batch_num_tmp) {
    cmd_ifm_buf_col_len = args.image.height;
  } else {
    if (((args.image.height) / (cmd_ifm_batch_num_tmp) % 2) == 0) {
      cmd_ifm_buf_col_len = (args.image.height) / cmd_ifm_batch_num_tmp;
    } else {
      cmd_ifm_buf_col_len = (args.image.height) / cmd_ifm_batch_num_tmp - 1;
    }
  }
  uint64_t cmd_ifm_batch_num_mns1 =
      (((args.image.height) + cmd_ifm_buf_col_len - 1) / cmd_ifm_buf_col_len) -
      1;

  uint64_t cmd_flt_total_batch_num = 1;
  uint64_t cmd_ifm_buf_col_len_rem =
      (args.image.height) -
      cmd_ifm_batch_num_mns1 * cmd_ifm_buf_col_len;  //= -4;

  //-------- ofm batch number reg &&  initial URAM reading address
  uint64_t cmd_init_raddr_cnt = 1;
  uint64_t cmd_init_raddr_flag = 0;
  int64_t cmd_init_raddr_index = -8;
  int64_t cmd_init_raddr_col_0 = -4;
  int64_t cmd_init_raddr_col_1 = -4;
  int64_t conv_ofm_buf_col_len = 0;
  int64_t conv_ofm_buf_col_len_rem = 0;

  if (((args.image.pad_height) % (2 * (args.kernel.stride_h))) == 0) {
    cmd_init_raddr_cnt = 0;
    cmd_init_raddr_flag = 0;
    cmd_init_raddr_index =
        0 - (int64_t)row_len_align * (((args.image.pad_height) + 1) / 2);
    cmd_init_raddr_col_0 = cmd_init_raddr_index;
    cmd_init_raddr_col_1 = cmd_init_raddr_index;
  } else if (((args.image.pad_height) -
              2 * ((args.image.pad_height) / (2 * (args.kernel.stride_h)))) <=
             (args.kernel.stride_h)) {
    cmd_init_raddr_cnt =
        (args.kernel.stride_h) -
        ((args.image.pad_height) -
         ((args.image.pad_height) / (2 * (args.kernel.stride_h))));
    cmd_init_raddr_flag = 1;
    cmd_init_raddr_index =
        0 - (int64_t)row_len_align * (int64_t)(args.image.pad_height) -
        (int64_t)row_len_align *
            ((args.image.pad_height) / (2 * args.kernel.stride_h));
    cmd_init_raddr_col_0 =
        0 - (int64_t)row_len_align * (int64_t)(args.image.pad_height) -
        (int64_t)row_len_align *
            ((args.image.pad_height) / (2 * (args.kernel.stride_h)));
    cmd_init_raddr_col_1 =
        cmd_init_raddr_col_0 + args.kernel.stride_h * (int64_t)row_len_align;
  } else if (((args.image.pad_height) -
              2 * ((args.image.pad_height) / (2 * (args.kernel.stride_h)))) <=
             2 * (args.kernel.stride_h)) {
    cmd_init_raddr_cnt =
        2 * (args.kernel.stride_h) *
            (((args.image.pad_height) + 2 * (args.kernel.stride_h) - 1) /
             (2 * (args.kernel.stride_h))) -
        (args.image.pad_height);
    cmd_init_raddr_flag = 0;
    cmd_init_raddr_index =
        0 - (int64_t)row_len_align * (int64_t)(args.kernel.stride_h) *
                (((args.image.pad_height) + 2 * (args.kernel.stride_h) - 1) /
                 (2 * (args.kernel.stride_h)));
    cmd_init_raddr_col_0 =
        0 -
        (int64_t)row_len_align *
            ((args.image.pad_height) / (2 * (args.kernel.stride_h))) -
        (int64_t)row_len_align *
            (2 * (args.kernel.stride_h) *
                 (((args.image.pad_height) + 2 * (args.kernel.stride_h) - 1) /
                  (2 * (args.kernel.stride_h))) -
             (args.image.pad_height));
    cmd_init_raddr_col_1 = cmd_init_raddr_col_0;
  }

  if (cmd_ifm_batch_num_mns1 == 0) {
    if ((args.kernel.height) <= (args.kernel.stride_h)) {
      conv_ofm_buf_col_len = (args.image.height) + 2 * (args.image.pad_height) -
                             3 * (args.kernel.stride_h);
    } else {
      conv_ofm_buf_col_len = (args.image.height) + 2 * (args.image.pad_height) -
                             2 * (args.kernel.stride_h) - (args.kernel.height);
    }
    conv_ofm_buf_col_len_rem = conv_ofm_buf_col_len;
  } else {
    int N_rem = 0;
    int row_rem = 0;

    if ((args.kernel.height) <= (args.kernel.stride_h)) {
      conv_ofm_buf_col_len = cmd_ifm_buf_col_len - 3 * (args.kernel.stride_h);
      N_rem = (cmd_ifm_buf_col_len - (args.kernel.height)) /
                  (args.kernel.stride_h) +
              1;
      row_rem = cmd_ifm_buf_col_len - (args.kernel.stride_h) * N_rem;
      conv_ofm_buf_col_len_rem = cmd_ifm_buf_col_len_rem +
                                 2 * (args.image.pad_height) + row_rem -
                                 3 * (args.kernel.stride_h);
    } else {
      conv_ofm_buf_col_len = cmd_ifm_buf_col_len + 2 * (args.image.pad_height) -
                             2 * (args.kernel.stride_h) - (args.kernel.height);
      N_rem = (cmd_ifm_buf_col_len - (args.kernel.height)) /
                  (args.kernel.stride_h) +
              1;
      row_rem = cmd_ifm_buf_col_len - (args.kernel.stride_h) * N_rem;
      conv_ofm_buf_col_len_rem =
          cmd_ifm_buf_col_len_rem + (args.image.pad_height) + row_rem -
          2 * (args.kernel.stride_h) - (args.kernel.height);
    }
  }

  //-----------------------  para functions --------------------------------
  uint64_t cmd_filter_quant_scale = 0x3c00;
  uint64_t cmd_image_quant_scale = 0x3c00;
  uint64_t wParallelsim = cmd_ifm_C_impl >> 3;
  uint64_t wParallelsim_num = cmd_flt_cycle_num_mns1;
  uint64_t win_size = (args.kernel.width) * (args.kernel.height) *
                          (cmd_ifm_pack_num_per_row_mns1 + 1) -
                      1;  //
  uint64_t conv_ofm_width = (((args.image.width) - (args.kernel.width) +
                              (args.image.pad_width) + (args.image.pad_width)) /
                             (args.kernel.stride_w));
  uint64_t conv_ofm_dma_length = cmd_channel_num * sizeof(short);  // NOLINT
  uint64_t conv_ofm_dma_stride = conv_ofm_dma_length;
  uint64_t conv_ofm_height_batch_tmp =
      (args.image.height + 2 * args.image.pad_height - args.kernel.height) /
          args.kernel.stride_h +
      1;

  uint64_t conv_ofm_height_batch = (conv_ofm_height_batch_tmp + 1) / 2 - 1;
  uint64_t o_ust_rst = 0;
  uint64_t conv_ofm_dma_repeat =
      (uint64_t)(((((args.image.width) - (args.kernel.width) +
                    (args.image.pad_width) + (args.image.pad_width))) /
                  (args.kernel.stride_w)) +
                 1);
  uint64_t conv_ofm_dma_offset =
      args.image.channels * conv_ofm_dma_repeat * sizeof(short);  // NOLINT
  uint64_t conv_ofm_inter_stride = conv_ofm_dma_offset * 2;
  //----------------- register contation ------------------
  uint64_t cmd_ifm_flt_base_addr = ((uint64_t)cmd_filter_vir_base_addr << 32) |
                                   ((uint64_t)cmd_image_vir_base_addr);
  uint64_t cmd_ifm_flt_dim = ((uint64_t)(args.kernel.height) << 48) |
                             ((uint64_t)(args.kernel.width) << 32) |
                             ((uint64_t)(args.image.height) << 16) |
                             ((uint64_t)(args.image.width));
  uint64_t cmd_pad_step_size = ((uint64_t)(args.kernel.stride_h) << 48) |
                               ((uint64_t)(args.kernel.stride_w) << 32) |
                               ((uint64_t)(args.image.pad_height) << 16) |
                               ((uint64_t)(args.image.pad_width));
  uint64_t cmd_param1 = ((uint64_t)cmd_filter_per_group << 48) |
                        ((uint64_t)cmd_channel_num << 32) |
                        ((uint64_t)filter_num << 16) |
                        ((uint64_t)cmd_group_num);
  uint64_t cmd_param2 =
      ((uint64_t)cmd_flt_sqr_len << 48) | ((uint64_t)cmd_ifm_pack_len << 32) |
      ((uint64_t)cmd_ifm_pre_row_num << 16) | ((uint64_t)cmd_channel_per_group);
  uint64_t cmd_param3 = ((uint64_t)cmd_flt_batch_num_mns1 << 48) |
                        ((uint64_t)cmd_flt_total_batch_num << 32) |
                        ((uint64_t)cmd_flt_N_impl << 16) |
                        ((uint64_t)cmd_flt_pre_batch_num);
  uint64_t cmd_param4 = ((uint64_t)cmd_ifm_pack_num_per_row_mns1 << 48) |
                        ((uint64_t)cmd_bn_num << 32) |
                        ((uint64_t)cmd_bias_num << 16) |
                        ((uint64_t)cmd_flt_N_len);
  uint64_t cmd_param5 = ((uint64_t)cmd_ifm_stride_row_length << 48) |
                        ((uint64_t)cmd_flt_pack_length << 32) |
                        ((uint64_t)cmd_flt_cycle_num_mns1 << 16) |
                        ((uint64_t)cmd_flt_pack_num_per_kernel_mns1);
  uint64_t cmd_param6 = ((uint64_t)cmd_ofm_width_mns1 << 48) |
                        ((uint64_t)cmd_ifm_batch_num_mns1 << 32) |
                        ((uint64_t)cmd_ifm_buf_col_len << 16) |
                        ((uint64_t)cmd_ifm_C_impl);
  uint64_t cmd_param7 = ((uint64_t)conv_ofm_inter_stride << 32) |
                        ((uint64_t)cmd_ifm_buf_col_len_rem << 16) |
                        ((uint64_t)cmd_ofm_height);
  uint64_t cmd_param8 =
      ((uint64_t)cmd_flt_length << 32) | ((uint64_t)cmd_ifm_row_byte_length);
  uint64_t cmd_ifm_flt_quant_scale = ((uint64_t)cmd_filter_quant_scale << 32) |
                                     ((uint64_t)cmd_image_quant_scale);
  uint64_t cmd_step_pad_mul_row_len =
      ((uint64_t)cmd_pad_h_mul_row_byte_len << 32) |
      ((uint64_t)cmd_step_h_mul_row_byte_len);
  //---- ofm paras ----
  uint64_t cmd_conv_param_reg = ((uint64_t)wParallelsim_num << 32) |
                                ((uint64_t)wParallelsim << 16) |
                                ((uint64_t)win_size);
  uint64_t cmd_ofm_addr_width_reg =
      ((uint64_t)conv_ofm_width << 32) | ((uint64_t)conv_ofm_addr_base);
  uint64_t cmd_intra_stride_atoms_reg =
      ((uint64_t)conv_ofm_dma_length << 32) | ((uint64_t)conv_ofm_dma_stride);
  uint64_t cmd_ofm_height_batch_reg =
      ((uint64_t)(conv_ofm_buf_col_len_rem & 0xffff) << 48) |
      ((uint64_t)(conv_ofm_buf_col_len & 0xffff) << 32) |
      ((uint64_t)conv_ofm_height_batch + 0x80000000);
  uint64_t cmd_user_ctrl_reg = ((uint64_t)o_ust_rst);
  uint64_t cmd_wdma_param_reg =
      ((uint64_t)(conv_ofm_dma_repeat | 0x80000000) << 32) |
      ((uint64_t)conv_ofm_dma_offset);
  uint64_t cmd_init_raddr_reg = ((cmd_init_raddr_col_1 & 0xffff) << 48) |
                                ((cmd_init_raddr_col_0 & 0xffff) << 32) |
                                (((cmd_init_raddr_index & 0xffff) << 16)) |
                                (cmd_init_raddr_flag & 0xffff) << 15 |
                                ((cmd_init_raddr_cnt & 0xffff));

  DLOG << "cmd_init_raddr_col_1 = " << hex << cmd_init_raddr_col_1;

  DLOG << "cmd_init_raddr_col_0 = " << hex << cmd_init_raddr_col_0;
  DLOG << "cmd_init_raddr_index = " << hex << cmd_init_raddr_index;  //
  DLOG << "cmd_init_raddr_cnt = " << hex << cmd_init_raddr_cnt;
  DLOG << "conv_ofm_buf_col_len = " << hex << conv_ofm_buf_col_len;
  DLOG << "conv_ofm_buf_col_len_rem = " << hex << conv_ofm_buf_col_len_rem;
  DLOG << "cmd_ifm_flt_base_addr = " << hex << cmd_ifm_flt_base_addr;
  DLOG << "cmd_scale_base_addr = " << hex << cmd_scale_base_addr;
  DLOG << "cmd_ifm_flt_dim = " << hex << cmd_ifm_flt_dim;
  DLOG << "cmd_pad_step_size = " << hex << cmd_pad_step_size;
  DLOG << "cmd_param1 = " << hex << cmd_param1;
  DLOG << "cmd_param2 = " << hex << cmd_param2;
  DLOG << "cmd_param3 = " << hex << cmd_param3;
  DLOG << "cmd_param4 = " << hex << cmd_param4;
  DLOG << "cmd_param5 = " << hex << cmd_param5;
  DLOG << "cmd_param6 = " << hex << cmd_param6;
  DLOG << "cmd_param7 = " << hex << cmd_param7;
  DLOG << "cmd_param8 =  " << hex << cmd_param8;
  DLOG << "cmd_ifm_flt_quant_scale =  " << hex << cmd_ifm_flt_quant_scale;
  DLOG << "cmd_step_pad_mul_row_len = " << hex << cmd_step_pad_mul_row_len;
  DLOG << "cmd_ifm_pack_byte_length = " << hex << cmd_ifm_pack_byte_length;
  DLOG << "cmd_conv_param_reg = " << hex << cmd_conv_param_reg;
  DLOG << "cmd_ofm_addr_width_reg = " << hex << cmd_ofm_addr_width_reg;
  DLOG << "cmd_intra_stride_atoms_reg = " << hex << cmd_intra_stride_atoms_reg;
  DLOG << "cmd_init_raddr_reg = " << hex << cmd_init_raddr_reg;
  DLOG << "cmd_ofm_height_batch_reg = " << hex << cmd_ofm_height_batch_reg;
  DLOG << "cmd_wdma_param_reg = " << hex << cmd_wdma_param_reg;
  DLOG << "pooling_mode = " << hex << args.mode;

  reg_writeq(cmd_ifm_flt_base_addr, MUL8(1));
  reg_writeq(cmd_scale_base_addr, MUL8(2));
  reg_writeq(cmd_ifm_flt_dim, MUL8(3));
  reg_writeq(cmd_pad_step_size, MUL8(4));
  reg_writeq(cmd_param1, MUL8(5));
  reg_writeq(cmd_param2, MUL8(6));
  reg_writeq(cmd_param3, MUL8(7));
  reg_writeq(cmd_param4, MUL8(8));
  reg_writeq(cmd_param5, MUL8(9));
  reg_writeq(cmd_param6, MUL8(10));
  reg_writeq(cmd_param7, MUL8(11));
  reg_writeq(cmd_param8, MUL8(12));
  reg_writeq(cmd_ifm_flt_quant_scale, MUL8(13));
  reg_writeq(cmd_step_pad_mul_row_len, MUL8(14));
  reg_writeq(cmd_ifm_pack_byte_length, MUL8(15));
  reg_writeq(cmd_conv_param_reg, MUL8(16));
  reg_writeq(cmd_ofm_addr_width_reg, MUL8(17));
  reg_writeq(cmd_intra_stride_atoms_reg, MUL8(18));

  reg_writeq(cmd_init_raddr_reg, MUL8(29));

  reg_writeq(0, MUL8(19));
  reg_writeq(cmd_ofm_height_batch_reg, MUL8(19));
  reg_writeq(cmd_ofm_height_batch_reg & 0xffffffff00000000, MUL8(19));

  reg_writeq(cmd_wdma_param_reg, MUL8(25));

  /******************************************************************/
  uint64_t cmd_mult_factor = ((uint64_t)args.kernel_reciprocal) |
                             ((uint64_t)args.kernel_reciprocal << 16);
  reg_writeq(cmd_mult_factor, MUL8(30));
  /******************************************************************/

  reg_writeq(0, MUL8(0));
  if (args.mode == 0) {  // max pooling
    reg_writeq(0x2200000000000000, MUL8(0));
  } else {  // average pooling
    reg_writeq(0x2400000000000000, MUL8(0));
  }
  int ret = -1;
  ret = fpga_regpoll(MUL8(48), CONV_DONE, 0x00ffff);
  if (ret == -1) {
    DLOG << "fpga pooling no interrupt!!";
    return ret;
  }
  reg_readq(MUL8(63));
  usleep(10);
  // get max value
  float scale = Findfp16Max();
  (args.output.scale_address)[0] = scale;                 // NOLINT
  (args.output.scale_address)[1] = (float)(1.0 / scale);  // NOLINT
  DLOG << "Findfp16Max scale = " << scale;
  DLOG << "ret=" << ret;
  return ret;
}

int get_ofm_batch_size(int width, int channel) {
  int pad_channel, row_size;

  if (64 < channel) {
    pad_channel = (int)((channel + 127) / 128) * 128;  // NOLINT
  } else if (32 < channel && channel <= 64) {
    pad_channel = ((channel + 63) / (64)) * 64;
  } else if (16 < channel && channel <= 32) {
    pad_channel = ((channel + 31) / (32)) * 32;
  } else if (channel <= 16) {
    pad_channel = ((channel + 15) / (16)) * 16;
  }

  row_size = pad_channel * width;

  return row_size;
}

int ComputeFpgaEWAdd(const struct EWAddArgs &args) {
#ifdef FPGA_PRINT_MODE
  DLOG << "=============ComputeFpgaEWAdd===========";
  DLOG << "   relu_enabled:" << args.relu_enabled
       << "   const0:" << fp16_2_fp32(int16_t(args.const0))
       << "   const1:" << fp16_2_fp32(int16_t(args.const1));
  DLOG << "   image0_address:" << args.image0.address
       << "   image0_channels:" << args.image0.channels
       << "   image0_height:" << args.image0.height
       << "   image0_width:" << args.image0.width;
  DLOG << "   image1_address:" << args.image1.address
       << "   image1_channels:" << args.image1.channels
       << "   image1_height:" << args.image1.height
       << "   image1_width:" << args.image1.width;
  DLOG << "   out_address:" << args.output.address;
#endif
#ifndef PADDLE_MOBILE_ZU5
  return 0;
#endif
  uint32_t filter_num_align = args.image0.channels;

  uint32_t const_kernel_width_1 = 1;
  uint32_t const_stride_width_1 = 1;
  uint32_t const_kernel_height_2 = 2;
  uint32_t const_stride_height_2 = 2;
  uint32_t const_pad_height_0 = 0;
  uint32_t const_pad_width_0 = 0;
  uint32_t ew_image_height = args.image0.height * 2;

  DLOG << "______db_______: begin to set registers. ";
  uint64_t ifm_pixel_num =
      ((args.image0.width) * (args.image0.height) * args.image0.channels);
  uint64_t ifm_memory_size = ifm_pixel_num * sizeof(short);  // NOLINT
  uint64_t flt_pixel_num = 0;
  uint64_t filter_memory_size = 0;
  uint64_t bn_pixel_num = (filter_num_align * 2);
  uint64_t bn_memory_size = bn_pixel_num * sizeof(uint16_t);

  uint64_t ofm_width =
      ((args.image0.width) + 2 * const_pad_width_0 - const_kernel_width_1) /
          (const_stride_width_1) +
      1;
  uint64_t ofm_height =
      ((ew_image_height) + 2 * (const_pad_height_0) - (const_kernel_height_2)) /
          (const_stride_height_2) +
      1;

  uint32_t filter_num = filter_num_align;
  uint32_t image_channels = args.image0.channels;

  uint64_t ifm_src_paddr = vaddr_to_paddr((args.image0.address));
  uint64_t flt_src_paddr = vaddr_to_paddr((args.image1.address));
  uint64_t ifm_dst_paddr = vaddr_to_paddr((args.output.address));
  float image_inv_scale = 0;
  float filter_inv_scale = 0;
  int idx = 0;

  DLOG << "______db_______: reset registers. ";

  reg_writeq(1, MUL8(24));
  usleep(1);
  reg_writeq(0, MUL8(24));

  /*********configuring registers*************/
  uint32_t cmd_image_vir_base_addr = (uint32_t)ifm_src_paddr;
  uint32_t cmd_filter_vir_base_addr = (uint32_t)flt_src_paddr;
  uint32_t cmd_scale_base_addr = 0;
  uint32_t conv_ofm_addr_base = (uint32_t)ifm_dst_paddr;
  uint64_t cmd_group_num = 1;
  uint64_t cmd_filter_per_group = filter_num / cmd_group_num;

  uint64_t cmd_flt_sqr_len = (const_kernel_width_1) * (const_kernel_height_2);
  uint64_t cmd_ifm_pre_row_num = const_kernel_height_2;
  if ((const_kernel_height_2 == ew_image_height) && (0 == const_pad_height_0)) {
    cmd_ifm_pre_row_num = (const_kernel_height_2);
  } else {
    cmd_ifm_pre_row_num = (const_kernel_height_2) - (const_pad_height_0) +
                          (const_stride_height_2);
  }
  uint64_t cmd_flt_pre_batch_num = 1;
  uint64_t cmd_ifm_pack_num_per_row_mns1 =
      (uint64_t)(((args.image0.channels) + 63) / 64) - 1;
  uint64_t cmd_bn_num = filter_num;
  uint64_t cmd_bias_num = filter_num;
  uint64_t cmd_ifm_stride_row_length =
      args.image0.width * const_stride_height_2;
  uint64_t cmd_flt_pack_num_per_kernel_mns1 =
      (uint64_t)(((args.image0.channels) + 63) / 64) - 1;
  uint64_t cmd_ofm_width_mns1 = (uint64_t)(
      ((args.image0.width) - (const_kernel_width_1) + 2 * (const_pad_width_0)) /
      (const_stride_width_1));
  uint64_t cmd_ofm_height =
      (uint64_t)(((args.image0.height) * 2 - (const_kernel_height_2) +
                  2 * (const_pad_height_0)) /
                 (const_stride_height_2)) +
      1;

  uint64_t cmd_channel_num = 0;
  uint64_t cmd_ifm_pack_len = 0;
  uint64_t cmd_channel_per_group = 0;
  uint64_t cmd_flt_batch_num_mns1 = 0;
  uint64_t cmd_flt_N_impl = 8;
  uint64_t cmd_ifm_C_impl = 16;
  uint64_t cmd_flt_pack_length = 0;
  uint64_t cmd_step_h_mul_row_byte_len = 0;
  uint64_t cmd_pad_h_mul_row_byte_len = 0;
  uint64_t cmd_ifm_pack_byte_length =
      16 * ((((args.image0.width) + 7) / 8) * 8);
  uint64_t row_len_align = args.image0.width;
  uint64_t cmd_flt_cycle_num_mns1 = 0;
  if (image_channels > 32) {
    cmd_channel_num = (uint64_t)((((args.image0.channels) + 63)) / 64) * 64;
    cmd_ifm_pack_len = 64 * (args.image0.width);
    cmd_channel_per_group = 64;
    cmd_flt_batch_num_mns1 = (uint64_t)(((filter_num + 7)) / 8 - 1);
    cmd_flt_N_impl = 8;
    cmd_ifm_C_impl = 64;
    cmd_flt_pack_length = (const_kernel_width_1) * (const_kernel_height_2)*64;
    cmd_step_h_mul_row_byte_len =
        (const_stride_height_2)*cmd_channel_num * args.image0.width;
    cmd_pad_h_mul_row_byte_len =
        (const_pad_height_0)*cmd_channel_num * args.image0.width;
    cmd_ifm_pack_byte_length = 64 * args.image0.width;
    row_len_align = args.image0.width;
    cmd_flt_cycle_num_mns1 = (cmd_channel_num / 64) - 1;
  } else if (image_channels > 16) {
    cmd_channel_num = 32;
    cmd_ifm_pack_len = 32 * (args.image0.width);
    cmd_channel_per_group = 32;
    cmd_flt_batch_num_mns1 = (uint64_t)((((filter_num) + 15)) / 16 - 1);
    cmd_flt_N_impl = 16;
    cmd_ifm_C_impl = 32;
    cmd_flt_pack_length = (const_kernel_width_1) * (const_kernel_height_2)*32;
    cmd_step_h_mul_row_byte_len = (const_stride_height_2)*cmd_channel_num *
                                  ((((args.image0.width) + 1)) / 2) * 2;
    cmd_pad_h_mul_row_byte_len = (const_pad_height_0)*cmd_channel_num *
                                 ((((args.image0.width) + 1)) / 2) * 2;
    cmd_ifm_pack_byte_length =
        32 * (uint64_t)((((args.image0.width) + 1)) / 2) * 2;
    row_len_align = (uint64_t)((((args.image0.width) + 1)) / 2);
    cmd_flt_cycle_num_mns1 = 0;
  } else if (image_channels > 8) {
    cmd_channel_num = 16;
    cmd_ifm_pack_len = 16 * (args.image0.width);
    cmd_channel_per_group = 16;
    cmd_flt_batch_num_mns1 = (uint64_t)((((filter_num) + 15)) / 16 - 1);
    cmd_flt_N_impl = 32;
    cmd_ifm_C_impl = 16;
    cmd_flt_pack_length = (const_kernel_width_1) * (const_kernel_height_2)*16;
    cmd_step_h_mul_row_byte_len = (const_stride_height_2)*cmd_channel_num *
                                  ((((args.image0.width) + 3)) / 4) * 4;
    cmd_pad_h_mul_row_byte_len = (const_pad_height_0)*cmd_channel_num *
                                 ((((args.image0.width) + 3)) / 4) * 4;
    cmd_ifm_pack_byte_length =
        16 * (uint64_t)((((args.image0.width) + 3)) / 4) * 4;
    row_len_align = (uint64_t)((((args.image0.width) + 3)) / 4);
    cmd_flt_cycle_num_mns1 = 0;
  }

  cmd_flt_N_impl = 16;
  cmd_flt_batch_num_mns1 = 0;
  cmd_flt_pack_length = 64;
  uint64_t cmd_flt_N_len = 0;
  uint64_t cmd_flt_length = 64;
  uint64_t cmd_ifm_row_byte_length = cmd_channel_num * (args.image0.width);
  uint64_t cmd_ifm_buf_col_len = 0;
  uint64_t ifm_one_batch_len =
      (1048576 / ((2 * row_len_align) * cmd_channel_num));
  uint64_t cmd_ifm_batch_num_tmp = (uint64_t)(
      ((ew_image_height) + ifm_one_batch_len - 1) / ifm_one_batch_len);
  DLOG << "ifm_one_batch_len = " << hex << ifm_one_batch_len;
  DLOG << "cmd_ifm_batch_num_tmp = " << hex << cmd_ifm_batch_num_tmp;

  if (1 == cmd_ifm_batch_num_tmp) {
    cmd_ifm_buf_col_len = ew_image_height;
  } else {
    cmd_ifm_buf_col_len = ifm_one_batch_len;
  }
  uint64_t cmd_ifm_batch_num_mns1 =
      (((ew_image_height) + cmd_ifm_buf_col_len - 1) / cmd_ifm_buf_col_len) - 1;
  DLOG << "___db____ew____:cmd_ifm_batch_num_mns1 = " << hex
       << cmd_ifm_batch_num_mns1;

  uint64_t cmd_flt_total_batch_num = 1;
  uint64_t cmd_ifm_buf_col_len_rem =
      (ew_image_height)-cmd_ifm_batch_num_mns1 * cmd_ifm_buf_col_len;
  //-------- ofm batch number reg &&  initial URAM reading address
  // logic-----------------
  uint64_t cmd_init_raddr_cnt = 1;
  uint64_t cmd_init_raddr_flag = 0;
  int64_t cmd_init_raddr_index = -8;
  int64_t cmd_init_raddr_col_0 = -4;
  int64_t cmd_init_raddr_col_1 = -4;
  int64_t conv_ofm_buf_col_len = 0;
  int64_t conv_ofm_buf_col_len_rem = 0;

  if (((const_pad_height_0) % (2 * (const_stride_height_2))) == 0) {
    cmd_init_raddr_cnt = 0;
    cmd_init_raddr_flag = 0;
    cmd_init_raddr_index =
        0 - (int64_t)row_len_align * (((const_pad_height_0) + 1) / 2);
    cmd_init_raddr_col_0 = cmd_init_raddr_index;
    cmd_init_raddr_col_1 = cmd_init_raddr_index;
  } else if (((const_pad_height_0)-2 *
              ((const_pad_height_0) / (2 * (const_stride_height_2)))) <=
             (const_stride_height_2)) {
    cmd_init_raddr_cnt =
        (const_stride_height_2) -
        ((const_pad_height_0) -
         ((const_pad_height_0) / (2 * (const_stride_height_2))));
    cmd_init_raddr_flag = 1;
    cmd_init_raddr_index =
        0 - (int64_t)row_len_align * (int64_t)(const_pad_height_0) -
        (int64_t)row_len_align *
            ((const_pad_height_0) / (2 * const_stride_height_2));
    cmd_init_raddr_col_0 =
        0 - (int64_t)row_len_align * (int64_t)(const_pad_height_0) -
        (int64_t)row_len_align *
            ((const_pad_height_0) / (2 * (const_stride_height_2)));
    cmd_init_raddr_col_1 =
        cmd_init_raddr_col_0 +
        const_stride_height_2 * (int64_t)row_len_align;  // 0;
  } else if (((const_pad_height_0)-2 *
              ((const_pad_height_0) / (2 * (const_stride_height_2)))) <=
             2 * (const_stride_height_2)) {
    cmd_init_raddr_cnt =
        2 * (const_stride_height_2) *
            (((const_pad_height_0) + 2 * (const_stride_height_2)-1) /
             (2 * (const_stride_height_2))) -
        (const_pad_height_0);
    cmd_init_raddr_flag = 0;
    cmd_init_raddr_index =
        0 - (int64_t)row_len_align * (int64_t)(const_stride_height_2) *
                (((const_pad_height_0) + 2 * (const_stride_height_2)-1) /
                 (2 * (const_stride_height_2)));
    cmd_init_raddr_col_0 =
        0 -
        (int64_t)row_len_align *
            ((const_pad_height_0) / (2 * (const_stride_height_2))) -
        (int64_t)row_len_align *
            (2 * (const_stride_height_2) *
                 (((const_pad_height_0) + 2 * (const_stride_height_2)-1) /
                  (2 * (const_stride_height_2))) -
             (const_pad_height_0));
    cmd_init_raddr_col_1 = cmd_init_raddr_col_0;
  }

  if (cmd_ifm_batch_num_mns1 == 0) {
    if ((const_kernel_height_2) <= (const_stride_height_2)) {
      conv_ofm_buf_col_len = cmd_ifm_buf_col_len + 2 * (const_pad_height_0)-3 *
                                                       (const_stride_height_2);
    } else {
      conv_ofm_buf_col_len =
          cmd_ifm_buf_col_len +
          2 * (const_pad_height_0)-3 * (const_stride_height_2) -
          (const_kernel_height_2);
    }
    conv_ofm_buf_col_len_rem = conv_ofm_buf_col_len;
  } else {
    int N_rem = 0;
    int row_rem = 0;

    if ((const_kernel_height_2) <= (const_stride_height_2)) {
      conv_ofm_buf_col_len = cmd_ifm_buf_col_len - 3 * (const_stride_height_2);
      N_rem = (cmd_ifm_buf_col_len - (const_kernel_height_2)) /
                  (const_stride_height_2) +
              1;
      row_rem = cmd_ifm_buf_col_len - (const_stride_height_2)*N_rem;
      conv_ofm_buf_col_len_rem = cmd_ifm_buf_col_len_rem +
                                 2 * (const_pad_height_0) + row_rem -
                                 3 * (const_stride_height_2);
    } else {
      conv_ofm_buf_col_len =
          cmd_ifm_buf_col_len +
          2 * (const_pad_height_0)-3 * (const_stride_height_2) -
          (const_kernel_height_2);
      N_rem = (cmd_ifm_buf_col_len - (const_kernel_height_2)) /
                  (const_stride_height_2) +
              1;
      row_rem = cmd_ifm_buf_col_len - (const_stride_height_2)*N_rem;
      conv_ofm_buf_col_len_rem =
          cmd_ifm_buf_col_len_rem + (const_pad_height_0) + row_rem -
          3 * (const_stride_height_2) - (const_kernel_height_2);
    }
  }

  //*************************
  uint64_t ifm_height_raw_batch = 0;
  uint64_t cmd_ofm_height_batch_reg;
  uint64_t conv_ofm_height_batch_tmp = 0;
  uint64_t conv_ofm_height_batch[16];
  int ofm_height_norm_batch;
  int height_batch_num;

  int row_norm_size = get_ofm_batch_size(args.image0.width, cmd_channel_num);
  int ifm_norm_size =
      ew_image_height * row_norm_size * sizeof(short);  // NOLINT

  if (ifm_norm_size <= (1024 * 1024)) {
    conv_ofm_height_batch[0] =
        get_image_out_axis(ew_image_height, const_pad_height_0,
                           const_kernel_height_2, const_stride_height_2);
    height_batch_num = 0;
  } else if (row_norm_size < (1024 * 1024)) {
    // raw ifm batch ,should make ofm be 2*N
    ifm_height_raw_batch =
        (int)(((double)(1024 * 1024) - row_norm_size + 1) /  // NOLINT
              (double)(2 * row_norm_size));                  // NOLINT
    ofm_height_norm_batch = get_image_out_axis(
        ifm_height_raw_batch, 0, const_kernel_height_2, const_stride_height_2);
    if (ofm_height_norm_batch % 2 == 0) {
      ofm_height_norm_batch = ofm_height_norm_batch;
    } else {
      ofm_height_norm_batch = ofm_height_norm_batch - 1;
    }

    DLOG << "ofm_height_norm_batch = " << hex << ofm_height_norm_batch;
    int ofm_height_rems = cmd_ofm_height;
    int i = 0;
    for (i = 0; 0 < ofm_height_rems; i++) {
      if (ofm_height_norm_batch <= ofm_height_rems) {
        ofm_height_rems = ofm_height_rems - ofm_height_norm_batch;
        conv_ofm_height_batch[i] = ofm_height_norm_batch;
        DLOG << "ofm_height_norm_batch[i] = " << hex
             << conv_ofm_height_batch[i];
      } else {
        conv_ofm_height_batch[i] = ofm_height_rems;
        break;
      }
    }
    height_batch_num = i;
  }
  //*************************

  //-----------------------  para functions --------------------------------
  uint64_t cmd_filter_quant_scale = 0x3c00;
  uint64_t cmd_image_quant_scale = 0x3c00;
  uint64_t wParallelsim = cmd_ifm_C_impl >> 3;
  uint64_t wParallelsim_num = cmd_flt_cycle_num_mns1;
  uint64_t win_size = (const_kernel_width_1) * (const_kernel_height_2) *
                          (cmd_ifm_pack_num_per_row_mns1 + 1) -
                      1;  //
  uint64_t conv_ofm_width = (((args.image0.width) - (const_kernel_width_1) +
                              (const_pad_width_0) + (const_pad_width_0)) /
                             (const_stride_width_1));
  uint64_t conv_ofm_dma_length = cmd_channel_num * sizeof(short);  // NOLINT
  uint64_t conv_ofm_dma_stride = cmd_channel_num * sizeof(short);  // NOLINT
  uint64_t cmd_image_addr_low = 0;
  uint64_t cmd_image_addr_high = 0;
  uint64_t cmd_image_addr_diff = 0;

  if (cmd_filter_vir_base_addr < cmd_image_vir_base_addr) {
    cmd_image_addr_low = (uint64_t)cmd_filter_vir_base_addr;
    cmd_image_addr_high = (uint64_t)cmd_image_vir_base_addr;
  } else {
    cmd_image_addr_low = (uint64_t)cmd_image_vir_base_addr;
    cmd_image_addr_high = (uint64_t)cmd_filter_vir_base_addr;
  }

  cmd_image_addr_diff = cmd_image_addr_high - cmd_image_addr_low;
  uint64_t o_ust_rst = 0;
  uint64_t conv_ofm_dma_repeat =
      (uint64_t)(((((args.image0.width) - (const_kernel_width_1) +
                    (const_pad_width_0) + (const_pad_width_0))) /
                  (const_stride_width_1)) +
                 1);
  uint64_t conv_ofm_dma_offset =
      cmd_channel_num * conv_ofm_dma_repeat * sizeof(short);  // NOLINT
  uint64_t conv_ofm_inter_stride = conv_ofm_dma_offset * 2;
  //----------------- register contation ------------------
  uint64_t cmd_ifm_flt_base_addr =
      (cmd_image_addr_high << 32) | (cmd_image_addr_low);

  uint64_t cmd_ifm_flt_dim = ((uint64_t)(const_kernel_height_2) << 48) |
                             ((uint64_t)(const_kernel_width_1) << 32) |
                             ((uint64_t)(ew_image_height) << 16) |
                             ((uint64_t)(args.image0.width));
  uint64_t cmd_pad_step_size = ((uint64_t)(const_stride_height_2) << 48) |
                               ((uint64_t)(const_stride_width_1) << 32) |
                               ((uint64_t)(const_pad_height_0) << 16) |
                               ((uint64_t)(const_pad_width_0));
  uint64_t cmd_param1 = ((uint64_t)cmd_filter_per_group << 48) |
                        ((uint64_t)cmd_channel_num << 32) |
                        ((uint64_t)filter_num << 16) |
                        ((uint64_t)cmd_group_num);
  uint64_t cmd_param2 =
      ((uint64_t)cmd_flt_sqr_len << 48) | ((uint64_t)cmd_ifm_pack_len << 32) |
      ((uint64_t)cmd_ifm_pre_row_num << 16) | ((uint64_t)cmd_channel_per_group);
  uint64_t cmd_param3 = ((uint64_t)cmd_flt_batch_num_mns1 << 48) |
                        ((uint64_t)cmd_flt_total_batch_num << 32) |
                        ((uint64_t)cmd_flt_N_impl << 16) |
                        ((uint64_t)cmd_flt_pre_batch_num);
  uint64_t cmd_param4 = ((uint64_t)cmd_ifm_pack_num_per_row_mns1 << 48) |
                        ((uint64_t)cmd_bn_num << 32) |
                        ((uint64_t)cmd_bias_num << 16) |
                        ((uint64_t)cmd_flt_N_len);
  uint64_t cmd_param5 = ((uint64_t)cmd_ifm_stride_row_length << 48) |
                        ((uint64_t)cmd_flt_pack_length << 32) |
                        ((uint64_t)cmd_flt_cycle_num_mns1 << 16) |
                        ((uint64_t)cmd_flt_pack_num_per_kernel_mns1);
  uint64_t cmd_param6 = ((uint64_t)cmd_ofm_width_mns1 << 48) |
                        ((uint64_t)cmd_ifm_batch_num_mns1 << 32) |
                        ((uint64_t)cmd_ifm_buf_col_len << 16) |
                        ((uint64_t)cmd_ifm_C_impl);
  uint64_t cmd_param7 = ((uint64_t)conv_ofm_inter_stride << 32) |
                        ((uint64_t)cmd_ifm_buf_col_len_rem << 16) |
                        ((uint64_t)cmd_ofm_height);
  uint64_t cmd_param8 =
      ((uint64_t)cmd_flt_length << 32) | ((uint64_t)cmd_ifm_row_byte_length);
  uint64_t cmd_ifm_flt_quant_scale = ((uint64_t)cmd_filter_quant_scale << 32) |
                                     ((uint64_t)cmd_image_quant_scale);
  uint64_t cmd_step_pad_mul_row_len =
      ((uint64_t)cmd_pad_h_mul_row_byte_len << 32) |
      ((uint64_t)cmd_step_h_mul_row_byte_len);
  //---- ofm paras ----
  uint64_t cmd_conv_param_reg = ((uint64_t)wParallelsim_num << 32) |
                                ((uint64_t)wParallelsim << 16) |
                                ((uint64_t)win_size);
  uint64_t cmd_ofm_addr_width_reg =
      ((uint64_t)conv_ofm_width << 32) | ((uint64_t)conv_ofm_addr_base);
  uint64_t cmd_intra_stride_atoms_reg =
      ((uint64_t)conv_ofm_dma_length << 32) | ((uint64_t)conv_ofm_dma_stride);
  uint64_t cmd_user_ctrl_reg = ((uint64_t)o_ust_rst);
  uint64_t cmd_wdma_param_reg =
      ((uint64_t)(conv_ofm_dma_repeat | 0x80000000) << 32) |
      ((uint64_t)conv_ofm_dma_offset);
  uint64_t cmd_init_raddr_reg = ((cmd_init_raddr_col_1 & 0xffff) << 48) |
                                ((cmd_init_raddr_col_0 & 0xffff) << 32) |
                                (((cmd_init_raddr_index & 0xffff) << 16)) |
                                (cmd_init_raddr_flag & 0xffff) << 15 |
                                ((cmd_init_raddr_cnt & 0xffff));
  uint64_t cmd_mult_factor =
      ((uint64_t)args.const0) | ((uint64_t)args.const1 << 16);
  uint64_t cmd_para31 = (cmd_para31 & 0x1) | args.relu_enabled;

  DLOG << "cmd_init_raddr_col_1 = " << hex << cmd_init_raddr_col_1;
  DLOG << "cmd_init_raddr_col_0 = " << hex << cmd_init_raddr_col_0;
  DLOG << "cmd_init_raddr_index = " << hex << cmd_init_raddr_index;  //
  DLOG << "cmd_init_raddr_cnt = " << hex << cmd_init_raddr_cnt;
  DLOG << "cmd_ifm_buf_col_len = " << hex << cmd_ifm_buf_col_len;
  DLOG << "cmd_ifm_buf_col_len_rem = " << hex << cmd_ifm_buf_col_len_rem;
  DLOG << "conv_ofm_buf_col_len = " << hex << conv_ofm_buf_col_len;
  DLOG << "conv_ofm_buf_col_len_rem = " << hex << conv_ofm_buf_col_len_rem;
  DLOG << "cmd_ifm_flt_base_addr = " << hex << cmd_ifm_flt_base_addr;
  DLOG << "cmd_scale_base_addr = " << hex << cmd_scale_base_addr;
  DLOG << "cmd_ifm_flt_dim = " << hex << cmd_ifm_flt_dim;
  DLOG << "cmd_pad_step_size = " << hex << cmd_pad_step_size;
  DLOG << "cmd_param1 = " << hex << cmd_param1;
  DLOG << "cmd_param2 = " << hex << cmd_param2;
  DLOG << "cmd_param3 = " << hex << cmd_param3;
  DLOG << "cmd_param4 = " << hex << cmd_param4;
  DLOG << "cmd_param5 = " << hex << cmd_param5;
  DLOG << "cmd_param6 = " << hex << cmd_param6;
  DLOG << "cmd_param7 = " << hex << cmd_param7;
  DLOG << "cmd_param8 =  " << hex << cmd_param8;
  DLOG << "cmd_ifm_flt_quant_scale =  " << hex << cmd_ifm_flt_quant_scale;
  DLOG << "cmd_step_pad_mul_row_len = " << hex << cmd_step_pad_mul_row_len;
  DLOG << "cmd_ifm_pack_byte_length = " << hex << cmd_ifm_pack_byte_length;
  DLOG << "cmd_conv_param_reg = " << hex << cmd_conv_param_reg;
  DLOG << "cmd_ofm_addr_width_reg = " << hex << cmd_ofm_addr_width_reg;
  DLOG << "cmd_intra_stride_atoms_reg = " << hex << cmd_intra_stride_atoms_reg;
  DLOG << "cmd_init_raddr_reg = " << hex << cmd_init_raddr_reg;
  DLOG << "cmd_mult_factor = " << hex << cmd_mult_factor;
  DLOG << "cmd_wdma_param_reg = " << hex << cmd_wdma_param_reg;
  DLOG << "cmd_para31 = " << hex << cmd_para31;

  reg_writeq(cmd_ifm_flt_base_addr, MUL8(1));
  reg_writeq(cmd_scale_base_addr, MUL8(2));
  reg_writeq(cmd_ifm_flt_dim, MUL8(3));
  reg_writeq(cmd_pad_step_size, MUL8(4));
  reg_writeq(cmd_param1, MUL8(5));
  reg_writeq(cmd_param2, MUL8(6));
  reg_writeq(cmd_param3, MUL8(7));
  reg_writeq(cmd_param4, MUL8(8));
  reg_writeq(cmd_param5, MUL8(9));
  reg_writeq(cmd_param6, MUL8(10));
  reg_writeq(cmd_param7, MUL8(11));
  reg_writeq(cmd_param8, MUL8(12));
  reg_writeq(cmd_ifm_flt_quant_scale, MUL8(13));
  reg_writeq(cmd_step_pad_mul_row_len, MUL8(14));
  reg_writeq(cmd_ifm_pack_byte_length, MUL8(15));
  reg_writeq(cmd_conv_param_reg, MUL8(16));
  reg_writeq(cmd_ofm_addr_width_reg, MUL8(17));
  reg_writeq(cmd_intra_stride_atoms_reg, MUL8(18));

  reg_writeq(cmd_init_raddr_reg, MUL8(29));
  reg_writeq(cmd_para31, MUL8(31));

  reg_writeq(0, MUL8(19));
  for (int i = 0; i < height_batch_num + 1; i++) {
    conv_ofm_height_batch_tmp =
        int((conv_ofm_height_batch[i] + 1) / 2) - 1;  // NOLINT
    cmd_ofm_height_batch_reg =
        ((uint64_t)(conv_ofm_buf_col_len_rem & 0xffff) << 48) |
        ((uint64_t)(conv_ofm_buf_col_len & 0xffff) << 32) |
        ((uint64_t)conv_ofm_height_batch_tmp + 0x80000000);
    reg_writeq(cmd_ofm_height_batch_reg, MUL8(19));
    reg_writeq(cmd_ofm_height_batch_reg & 0xffffffff00000000, MUL8(19));
    usleep(1);
  }
  reg_writeq(cmd_wdma_param_reg, MUL8(25));
  DLOG << "cmd_ofm_height_batch_reg = " << hex << cmd_ofm_height_batch_reg;

  /******************************************************************/
  reg_writeq(cmd_mult_factor, MUL8(30));
  /******************************************************************/

  reg_writeq(0, MUL8(0));

  reg_writeq(0x2100000000000000, MUL8(0));

  int ret = fpga_regpoll(MUL8(48), CONV_DONE, 0xffffff);
  if (ret == -1) {
    DLOG << "fpga EW no interrupt!!";
    return ret;
  }
  reg_readq(MUL8(63));
  usleep(10);
  // get max value
  float scale = Findfp16Max();
  (args.output.scale_address)[0] = scale;                 // NOLINT
  (args.output.scale_address)[1] = (float)(1.0 / scale);  // NOLINT
  DLOG << "Findfp16Max scale = " << scale;

  DLOG << "ret=" << ret;
  return ret;
}

int PerformBypass(const struct BypassArgs &args) {
#ifdef FPGA_PRINT_MODE
  DLOG << "=============ComputeFpgaBypass===========";
  DLOG << "   input_type:" << args.input_data_type
       << "   output_type:" << args.output_data_type
       << "   input_layout_type:" << args.input_layout_type
       << "   output_layout_type:" << args.output_layout_type;
  DLOG << "   image_address:" << args.image.address
       << "   image_scale_address:" << args.image.scale_address
       << "   image_channels:" << args.image.channels
       << "   image_height:" << args.image.height
       << "   image_width:" << args.image.width
       << "   pad_height:" << args.image.pad_height
       << "   pad_width:" << args.image.pad_width;
  DLOG << "   out_address:" << args.output.address
       << "   out_scale_address:" << args.output.scale_address;
#endif
#ifndef PADDLE_MOBILE_ZU5
  return 0;
#endif

  uint64_t ifm_src_paddr = vaddr_to_paddr(args.image.address);
  uint64_t ifm_dst_paddr = vaddr_to_paddr(args.output.address);
  uint64_t bp_enable;
  int64_t length;
  uint64_t pixels;

  // fp32->fp16
  if ((args.input_data_type) && (!args.output_data_type)) {
    DLOG << "fp32-fp16";
    pixels = (args.image.channels) * (args.image.width) * (args.image.height);
    length = pixels * sizeof(float);
    bp_enable = 0x8800000000000000UL + (uint64_t)length;
  }
  // fp16->fp32
  else if ((!args.input_data_type) && (args.output_data_type)) {  // NOLINT
    DLOG << "fp16-fp32";
    pixels = filter::calc_aligned_channel((args.image.channels)) *
             (args.image.width) * (args.image.height);
    length = pixels * sizeof(short);       // NOLINT
    length = align_to_x((int)length, 64);  // NOLINT
    bp_enable = 0x8a00000000000000UL + length;
  }
  // fp16->fp16 findmax
  else if ((!args.input_data_type) && (!args.output_data_type)) {  // NOLINT
    DLOG << "16-16";
    pixels = (args.image.channels) * (args.image.width) * (args.image.height);
    length = pixels * sizeof(short);  // NOLINT
    bp_enable = 0x8900000000000000 + length;
  } else {
    return -1;
  }
  // start bypass
  reg_writeq(0, MUL8(0));
  reg_writeq(ifm_src_paddr, MUL8(27));
  reg_writeq(ifm_dst_paddr, MUL8(28));
  reg_writeq(bp_enable, MUL8(0));
  int ret = -1;
  ret = fpga_regpoll(MUL8(48), BYPASS_DONE, 0xffffff);

  if (ret != -1) {
    DLOG << "test done";
  }
  reg_readq(MUL8(63));
  usleep(10);
  // get max value
  float scale = Findfp16Max();
  args.output.scale_address[0] = scale;                 // NOLINT
  args.output.scale_address[1] = (float)(1.0 / scale);  // NOLINT
  DLOG << "ret=" << ret;
  return ret;
}

int ComputeFPGAConcat(const struct ConcatArgs &args) {
#ifdef FPGA_PRINT_MODE
  DLOG << "=============ComputeFpgaConcat===========";
  DLOG << "   Image_num: " << args.image_num

       << "   out_address:" << args.image_out
       << "   out_scale_address:" << args.scale_out
       << "   out_channel:" << args.out_channel;
  DLOG << "   image_height:" << args.height << "   image_width:" << args.width;
  for (int i = 0; i < args.image_num; i++) {
    DLOG << "   " << i << "th:        ";
    DLOG << "   channel_num:" << args.channel_num[i]
         << "   aligned_channel_num:" << args.aligned_channel_num[i]
         << "   image_address:" << args.images_in[i]
         << "   image_scale_address:" << args.scales_in[i];
  }
#endif

  image::concat_images(args.images_in, args.scales_in, args.image_out,
                       args.scale_out, args.image_num, args.channel_num,
                       args.height, args.width, args.aligned_channel_num,
                       args.out_channel);
  return 0;
}

}  // namespace fpga
}  // namespace paddle_mobile
