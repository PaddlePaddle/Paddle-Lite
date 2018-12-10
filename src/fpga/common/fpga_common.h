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

#pragma once

#include <cstddef>
#include <cstdint>

namespace paddle_mobile {
namespace fpga {

#ifdef PADDLE_MOBILE_FPGA_V1
#define IMAGE_ALIGNMENT 16           // Aligned to 16
#define FILTER_NUM_ALIGNMENT 32      // Filter number aligned to 32
#define FILTER_ELEMENT_ALIGNMENT 16  // Filter element number aligned to 16
#define BS_NUM_ALIGNMENT 8
#endif

enum DataType {
  DATA_TYPE_FP32 = 1,
  DATA_TYPE_FP16 = 0,
};

enum LayoutType {
  LAYOUT_CHW = 1,
  LAYOUT_HWC = 0,
};

struct KernelArgs {
  uint32_t width;
  uint32_t height;
  uint32_t stride_w;
  uint32_t stride_h;
};

struct ImageInputArgs {
  void* address;         // input featuremap virtual address
  float* scale_address;  // input scale address;
  uint32_t channels;
  uint32_t width;  // featuremap width
  uint32_t height;
  uint32_t pad_width;  // padding width;
  uint32_t pad_height;
};

struct ImageOutputArgs {
  void* address;         // output result address;
  float* scale_address;  // output scale address;
  uint64_t timer_cnt;    // time counter for FPGA computation
};
#ifdef PADDLE_MOBILE_FPGA_V1
struct ConvDriverParam {
  uint64_t image_address_phy;
  uint64_t filter_address_phy;
  uint64_t sb_address_phy;
  uint64_t output_address_phy;

  uint64_t output_height;
  uint64_t output_width;
  uint64_t filter_per_group;
  uint64_t channel_per_group;

  uint64_t image_amount_per_row;
  uint64_t image_one_pad_per_row;
  uint64_t filter_amount_all;
  uint64_t output_amount_per_row;

  uint64_t image_block_amount_per_row;
  uint64_t filter_pad_width_mul_channel;
  uint64_t image_amount_per_row_multi_win_first;
  uint64_t image_amount_per_row_multi_win;
  uint64_t image_block_num;
  uint64_t image_block_len;
  uint64_t image_block_len_last;
  uint64_t image_win_cnt;
  uint64_t image_win_cnt_last;
  uint64_t res_row_data_align4_pad;
  uint64_t prog_full_cnt;
  uint64_t post_prog_full_cnt;
  uint64_t fpga_bias_scale_len;
  uint64_t cmd;
};

struct EWAddDriverParam {
  uint64_t image0_address_phy;
  uint64_t image1_address_phy;
  uint64_t datalen;
  uint64_t image_image_pixel;
  uint64_t image_amount_per_row;
  uint64_t output_address_phy;
  uint64_t coefficient;
  uint64_t cmd;
};
#endif

struct ConvArgs {
  bool relu_enabled;
  void* sb_address;  // scale and bias
  void* filter_address;
  float* filter_scale_address;
  uint32_t filter_num;
  uint32_t group_num;

  struct KernelArgs kernel;
  struct ImageInputArgs image;  // input image;
  struct ImageOutputArgs output;

#ifdef PADDLE_MOBILE_FPGA_V2
  void* free_space;  // used by FPGA logic
#endif

#ifdef PADDLE_MOBILE_FPGA_V1
  struct ConvDriverParam driver;
#endif
};

struct ConcatArgs {
  uint32_t image_num;
  int16_t** images_in;
  float** scales_in;
  void* image_out;
  float* scale_out;
  uint32_t* channel_num;
  uint32_t* aligned_channel_num;
  uint32_t out_channel;
  uint32_t height;
  uint32_t width;
};

struct SplitConvArgs {
  uint32_t split_num;
  uint32_t group_num;
  uint32_t filter_num;
  struct ImageOutputArgs output;
  struct ConvArgs* conv_arg;
  struct ConcatArgs concat_arg;
};

struct SplitArgs {
  uint32_t image_num;
  int16_t* image_in;
  float* scale_in;
  void** images_out;
  float** scales_out;
  uint32_t* out_channel_nums;
  uint32_t height;
  uint32_t width;
};

struct PoolingArgs {
  int16_t mode;  // mode: 0:max, 1:avg
  int16_t kernel_reciprocal;
  struct KernelArgs kernel;
  struct ImageInputArgs image;  // input image;
  struct ImageOutputArgs output;
};

struct EWAddArgs {
  bool relu_enabled;
  uint32_t const0;  // output0 = const0 x input0 + const1 x input1;
  uint32_t const1;
  struct ImageInputArgs image0;
  struct ImageInputArgs image1;
  struct ImageOutputArgs output;
#ifdef PADDLE_MOBILE_FPGA_V1
  struct EWAddDriverParam driver;
#endif
};

struct BypassArgs {
  enum DataType input_data_type;
  enum DataType output_data_type;
  enum LayoutType input_layout_type;
  enum LayoutType output_layout_type;
  struct ImageInputArgs image;
  struct ImageOutputArgs output;
};

struct DeconvArgs {
  uint32_t sub_conv_num;
  uint32_t group_num;
  uint32_t filter_num;
  uint32_t omit_size;
  uint32_t sub_output_width;
  uint32_t sub_output_height;
  struct ImageOutputArgs output;
  struct ConvArgs* conv_args;
};

static inline int align_to_x(int num, int x) { return (num + x - 1) / x * x; }

int16_t fp32_2_fp16(float fp32_num);
float fp16_2_fp32(int16_t fp16_num);

int open_device();
int close_device();
void* fpga_malloc(size_t size);
void fpga_free(void* ptr);
void fpga_copy(void* dest, const void* src, size_t num);
int fpga_flush(void* address, size_t size);
int fpga_invalidate(void* address, size_t size);

uint64_t vaddr_to_paddr(void* address);
void expand_conv_arg(ConvArgs* arg);
void expand_EW_arg(EWAddArgs* arg);

}  // namespace fpga
}  // namespace paddle_mobile
