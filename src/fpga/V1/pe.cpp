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
#include "fpga/V1/filter.h"
#include "fpga/V1/image.h"
#include "fpga/common/config.h"
#include "fpga/common/driver.h"

namespace paddle_mobile {
namespace fpga {

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

  return 0;
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
                       args.height, args.width);
  return 0;
}

}  // namespace fpga
}  // namespace paddle_mobile
