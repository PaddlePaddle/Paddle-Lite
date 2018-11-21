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

struct ConvArgs {
  bool relu_enabled;
  void* sb_address;  // scale and bias
  void* filter_address;
  float* filter_scale_address;
  void* free_space;  // used by FPGA logic
  uint32_t filter_num;
  uint32_t group_num;

  struct KernelArgs kernel;
  struct ImageInputArgs image;  // input image;
  struct ImageOutputArgs output;
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
  struct ConvArgs conv_arg;
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

}  // namespace fpga
}  // namespace paddle_mobile
