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
#include <memory>
#include <vector>

#ifdef PADDLE_MOBILE_FPGA_V1
#define IMAGE_ALIGNMENT (16)           // Aligned to 16
#define FILTER_NUM_ALIGNMENT (32)      // Filter number aligned to 32
#define FILTER_ELEMENT_ALIGNMENT (16)  // Filter element number aligned to 16
#define BS_NUM_ALIGNMENT (8)
#define BIAS_NUM_ALIGNMENT (16)
#define ROW_PARALLEL_NUM (3)
#endif
#ifdef PADDLE_MOBILE_FPGA_V2
#define IMAGE_ALIGNMENT (32)           // Aligned to 32
#define FILTER_NUM_ALIGNMENT (32)      // Filter number aligned to 32
#define FILTER_ELEMENT_ALIGNMENT (16)  // Filter element number aligned to 16
#define BS_NUM_ALIGNMENT (8)
#define BIAS_SCALE_DMA_NUM (4)
#define RESULT_ALIGNMENT (32)
#define PE_COLUMN (8)
#define ROW_PARALLEL_NUM (2)
#define BIAS_NUM_ALIGNMENT (16)

#endif

namespace paddle_mobile {
namespace fpga {

enum DataType {
  DATA_TYPE_INT8 = 2,
  DATA_TYPE_FP32 = 1,
  DATA_TYPE_FP16 = 0,
};

enum LayoutType {
  LAYOUT_CHW = 1,
  LAYOUT_HWC = 0,
};

enum ActivationType {
  NONE = 0,
  LEAKYRELU = 1,
  SIGMOID = 2,
  TANH = 3,
  SOFTMAX = 4,
};

struct ActivationArgs {
  enum ActivationType activation_type = NONE;
  int16_t leaky_relu_negative_slope;
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
  struct ActivationArgs
      activation;  // To select activation and specify (Leaky)Relu parameter.
};

struct ConvDriverParam {
  uint64_t filter_per_group;
  uint64_t channel_per_group;

  uint64_t image_one_pad_per_row;
  uint64_t deconv_param;

  uint64_t col_padding_up;
  uint64_t col_padding_down;
  uint64_t row_padding_up;
  uint64_t row_padding_down;

  uint64_t image_block_amount_per_row;
  uint64_t filter_pad_width_mul_channel;
  uint64_t image_win_cnt;
  uint64_t image_win_cnt_last;
  uint64_t filter_row;
  uint64_t filter_width;
  uint64_t filter_height;
  uint64_t skip_window;
  uint64_t stride_h;
  uint64_t filter_amount_all;
  uint64_t prog_full_cnt;
  uint64_t filter_align;
  uint64_t filter_num;
  uint64_t output_width;
  uint64_t output_amount_per_row;
  uint64_t res_row_data_align4_pad;
  uint64_t cal_res_num;
  uint64_t last_cal_res_row_num;
  uint64_t post_prog_full_cnt;
  uint64_t deconv_skip_row;      // paralvl*deconv_group
  uint64_t deconv_res_skip_row;  // deconv_group * result_amount_per_row
  uint64_t deconv_ena;
  uint64_t deconv_dump;
  uint64_t output_address_phy;
  uint64_t output_height;
  uint64_t result_amount_per_row_multi_para;
  uint64_t sb_address_phy;
  uint64_t fpga_bias_scale_len;
  uint64_t filter_amount_whole;
  uint64_t filter_address_phy;
  uint64_t filters_amount_whole;
  uint64_t image_address_phy;
  uint64_t image_hight;
  uint64_t image_amount_per_row;
  uint64_t image_amount_per_row_multi_win_first;
  uint64_t image_amount_per_row_multi_win;
  uint64_t filter_pad_hight;
  uint64_t image_block_num;
  uint64_t image_block_len;
  uint64_t image_block_len_last;

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

struct DeconvTxParm {
  uint32_t omit_size;
  uint32_t sub_conv_num;
  uint32_t deconv_en;
  uint32_t out_addr_offset;
};

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

  struct DeconvTxParm deconv_tx_param;
  struct ConvDriverParam driver;
};

struct ConcatArgs {
  uint32_t image_num;
#ifdef PADDLE_MOBILE_FPGA_V2
  int8_t** images_in;
#else
  int16_t** images_in;
#endif
  float** scales_in;
  void* image_out;
  float* scale_out;
  uint32_t* channel_num;
  uint32_t* aligned_channel_num;  // Not used so far. Reserved for V2.
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
  std::shared_ptr<ConvArgs> shared_conv_arg;
  std::vector<std::shared_ptr<char>> vector_concat_space;
  std::vector<std::shared_ptr<char>> vector_conv_space;
};

struct SplitArgs {
  uint32_t image_num;
#ifdef PADDLE_MOBILE_FPGA_V2
  int8_t* image_in;
#else
  int16_t* image_in;
#endif
  float* scale_in;
  void** images_out;
  float** scales_out;
  uint32_t* out_channel_nums;
  uint32_t height;
  uint32_t width;
  std::vector<std::shared_ptr<char>> vector_split_space;
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
  struct EWAddDriverParam driver;
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
  std::vector<std::shared_ptr<SplitConvArgs>> split_conv_args;
};
struct DWconvArgs {
  uint32_t sub_conv_num;
  bool relu_enabled;
  void* bias_address;
  void* filter_address;
  struct KernelArgs kernel;
  struct ImageInputArgs image;
  struct ImageOutputArgs output;
  std::vector<std::shared_ptr<char>> vector_dwconv_space;
};

struct DWDeconvArgs {
  uint32_t sub_conv_num;
  uint32_t group_num;
  uint32_t filter_num;
  uint32_t omit_size;
  uint32_t sub_output_width;
  uint32_t sub_output_height;
  struct ImageOutputArgs output;
  std::vector<std::shared_ptr<DWconvArgs>> dw_conv_args;
  std::vector<std::shared_ptr<char>> vector_dw_conv_space;
};

static inline uint32_t align_to_x(int64_t num, int64_t x) {
  return ((uint32_t)(num + x) - 1) / (uint32_t)x * (uint32_t)x;
}

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
inline int32_t convertmantissa(int32_t i);

uint32_t paddle_mobile_version();

}  // namespace fpga
}  // namespace paddle_mobile
