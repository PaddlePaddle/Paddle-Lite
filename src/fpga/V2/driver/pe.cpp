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

#include "fpga/V2/driver/pe.h"
#include "fpga/V2/config.h"
#include "fpga/V2/driver/driver.h"
#include "fpga/V2/filter.h"
#include "fpga/V2/image.h"

namespace paddle_mobile {
namespace fpga {
<<<<<<< HEAD
#define MUL8(x) (x * 8)
=======
#define MUL8(x) ((x)*8)
>>>>>>> upstream/develop
#define BYPASS_DONE 1

float Findfp16Max() {
  uint16_t abs_vals[16];
  uint64_t max_fp16;

<<<<<<< HEAD
  max_fp16 = reg_readq(MUL8(49));
=======
  max_fp16 = driver::reg_readq(MUL8(49));
>>>>>>> upstream/develop
  abs_vals[0] = (uint16_t)(0x0000007f & (max_fp16));        // NOLINT
  abs_vals[1] = (uint16_t)(0x0000007f & (max_fp16 >> 16));  // NOLINT
  abs_vals[2] = (uint16_t)(0x0000007f & (max_fp16 >> 32));  // NOLINT
  abs_vals[3] = (uint16_t)(0x0000007f & (max_fp16 >> 48));  // NOLINT
<<<<<<< HEAD
  max_fp16 = reg_readq(MUL8(50));
=======
  max_fp16 = driver::reg_readq(MUL8(50));
>>>>>>> upstream/develop
  abs_vals[4] = (uint16_t)(0x0000007f & (max_fp16));        // NOLINT
  abs_vals[5] = (uint16_t)(0x0000007f & (max_fp16 >> 16));  // NOLINT
  abs_vals[6] = (uint16_t)(0x0000007f & (max_fp16 >> 32));  // NOLINT
  abs_vals[7] = (uint16_t)(0x0000007f & (max_fp16 >> 48));  // NOLINT
<<<<<<< HEAD
  max_fp16 = reg_readq(MUL8(51));
=======
  max_fp16 = driver::reg_readq(MUL8(51));
>>>>>>> upstream/develop
  abs_vals[8] = (uint16_t)(0x0000007f & (max_fp16));         // NOLINT
  abs_vals[9] = (uint16_t)(0x0000007f & (max_fp16 >> 16));   // NOLINT
  abs_vals[10] = (uint16_t)(0x0000007f & (max_fp16 >> 32));  // NOLINT
  abs_vals[11] = (uint16_t)(0x0000007f & (max_fp16 >> 48));  // NOLINT
<<<<<<< HEAD
  max_fp16 = reg_readq(MUL8(52));
=======
  max_fp16 = driver::reg_readq(MUL8(52));
>>>>>>> upstream/develop
  abs_vals[12] = (uint16_t)(0x0000007f & (max_fp16));
  abs_vals[13] = (uint16_t)(0x0000007f & (max_fp16 >> 16));  // NOLINT
  abs_vals[14] = (uint16_t)(0x0000007f & (max_fp16 >> 32));  // NOLINT
  abs_vals[15] = (uint16_t)(0x0000007f & (max_fp16 >> 48));  // NOLINT

  uint16_t tmp = 0;
  for (int i = 0; i < 16; i++) {
    if (tmp < abs_vals[i]) {
      tmp = abs_vals[i];
    }
  }
  return fp16_2_fp32(tmp) / 127.0f;
}

int ComputeFpgaConv(const struct SplitConvArgs &args) {
<<<<<<< HEAD
  ComputeBasicConv(args.conv_args[0]);
=======
  ComputeBasicConv(args.conv_arg[0]);
>>>>>>> upstream/develop
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

  return 0;
}

int ComputeFpgaPool(const struct PoolingArgs &args) {
#ifdef FPGA_PRINT_MODE
  DLOG << "=============ComputeFpgaPool===========";
  DLOG << "   mode:" << args.mode
       << "   kernel_reciprocal:" << fp16_2_fp32(args.kernel_reciprocal);
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
  return 0;
}

int ComputeFpgaEWAdd(const struct EWAddArgs &args) {
#ifdef FPGA_PRINT_MODE
  DLOG << "=============ComputeFpgaEWAdd===========";
  DLOG << "   relu_enabled:" << args.relu_enabled
       << "   const0:" << fp16_2_fp32(int16_t(args.const0))
       << "   const1:" << fp16_2_fp32(int16_t(args.const1));
  DLOG << "   image0_address:" << args.image0.address
       << "   image0_scale_address:" << args.image0.scale_address
       << "   image0_channels:" << args.image0.channels
       << "   image0_height:" << args.image0.height
       << "   image0_width:" << args.image0.width
       << "   pad0_height:" << args.image0.pad_height
       << "   pad0_width:" << args.image0.pad_width;
  DLOG << "   image1_address:" << args.image1.address
       << "   image1_scale_address:" << args.image1.scale_address
       << "   image1_channels:" << args.image1.channels
       << "   image1_height:" << args.image1.height
       << "   image1_width:" << args.image1.width
       << "   pad1_height:" << args.image1.pad_height
       << "   pad_width:" << args.image1.pad_width;
  DLOG << "   out_address:" << args.output.address
       << "   out_scale_address:" << args.output.scale_address;
#endif
#ifndef PADDLE_MOBILE_ZU5
  return 0;
#endif
  return 0;
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

<<<<<<< HEAD
  uint64_t ifm_src_paddr = vaddr_to_paddr(args.image.address);
  uint64_t ifm_dst_paddr = vaddr_to_paddr(args.output.address);
=======
  uint64_t ifm_src_paddr = driver::vaddr_to_paddr(args.image.address);
  uint64_t ifm_dst_paddr = driver::vaddr_to_paddr(args.output.address);
>>>>>>> upstream/develop
  uint64_t bp_enable;
  int64_t length;
  uint64_t pixels;

  // fp32->fp16
  if ((args.input_data_type) && (!args.output_data_type)) {
    pixels = (args.image.channels) * (args.image.width) * (args.image.height);
    length = pixels * sizeof(float);
    bp_enable = 0x8800000000000000 + length;
  }
  // fp16->fp32
  else if ((!args.input_data_type) && (args.output_data_type)) {
    pixels = filter::calc_aligned_channel((args.image.channels)) *
             (args.image.width) * (args.image.height);
    length = pixels * sizeof(short);
    length = align_to_x((int)length, 64);  // NOLINT
    bp_enable = 0x8a00000000000000 + length;
  }
  // fp16->fp16 findmax
  else if ((!args.input_data_type) && (!args.output_data_type)) {
    pixels = (args.image.channels) * (args.image.width) * (args.image.height);
    length = pixels * sizeof(short);
    bp_enable = 0x8900000000000000 + length;
  } else {
    return -1;
  }

  // start bypass
<<<<<<< HEAD
  reg_writeq(ifm_src_paddr, MUL8(27));
  reg_writeq(ifm_dst_paddr, MUL8(28));
  reg_writeq(0, MUL8(0));
  reg_writeq(bp_enable, MUL8(0));
  // poll
  int ret = -1;
  ret = fpga_regpoll(MUL8(48), BYPASS_DONE, 0xffffffff);
  if (ret != -1) {
    // clear "irq"
    reg_readq(MUL8(63));
=======
  driver::reg_writeq(ifm_src_paddr, MUL8(27));
  driver::reg_writeq(ifm_dst_paddr, MUL8(28));
  driver::reg_writeq(0, MUL8(0));
  driver::reg_writeq(bp_enable, MUL8(0));
  // poll
  int ret = -1;
  ret = driver::fpga_regpoll(MUL8(48), BYPASS_DONE, 0xffffffff);
  if (ret != -1) {
    // clear "irq"
    driver::reg_readq(MUL8(63));
>>>>>>> upstream/develop
  }
  // get max value
  if ((!args.input_data_type) && (!args.output_data_type)) {
    float scale = Findfp16Max();
    args.output.scale_address[0] = (float)(1.0 / scale);  // NOLINT
    args.output.scale_address[1] = scale;
  }
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
