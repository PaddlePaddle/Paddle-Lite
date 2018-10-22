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

#include <stdint.h>
#include <cstddef>
#include <iostream>
#include <limits>
#include "framework/tensor.h"

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

struct VersionArgs {
  void* buffer;
};

struct MemoryCopyArgs {
  void* src;
  void* dest;
  size_t size;
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
};

struct ConvArgs {
  bool relu_enabled;
  void* sb_address;  // scale and bias are interlaced;
  void* filter_address;
  float* filter_scale_address;
  uint32_t filter_num;
  uint32_t group_num;

  struct KernelArgs kernel;
  struct ImageInputArgs image;  // input image;
  struct ImageOutputArgs output;
};

struct ConcatArgs {
  uint32_t image_num;
  half** images_in;
  float** scales_in;
  void* image_out;
  float* scale_out;
  uint32_t* channel_num;
  uint32_t height;
  uint32_t width;
};

struct SplitConvArgs {
  uint32_t split_num;
  uint32_t group_num;
  uint32_t filter_num;
  struct ImageOutputArgs output;
  struct ConvArgs* conv_args;
  struct ConcatArgs concat_arg;
};

struct GroupConvArgs {
  uint32_t group_num;
  uint32_t filter_num;
  struct ImageOutputArgs output;
  struct SplitConvArgs* conv_args;
  struct ConcatArgs concat_arg;
};

struct PoolingArgs {
  int16_t mode;  // mode: 0:max, 1:avg
  half kernel_reciprocal;
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

struct FpgaRegWriteArgs {
  uint64_t address;  //
  uint64_t value;
};

struct FpgaRegReadArgs {
  uint64_t address;
  uint64_t value;
};

struct MemoryCacheArgs {
  void* address;
  size_t size;
};

#define IOCTL_FPGA_MAGIC 'FPGA'

#define IOCTL_VERSION _IOW(IOCTL_FPGA_MAGIC, 01, struct VersionArgs)

#define IOCTL_SEPARATOR_0 10

#define IOCTL_MEM_COPY _IOW(IOCTL_FPGA_MAGIC, 11, struct MemoryCopyArgs)
#define IOCTL_MEMCACHE_INVAL _IOW(IOCTL_FPGA_MAGIC, 12, struct MemoryCacheArgs)
#define IOCTL_MEMCACHE_FLUSH _IOW(IOCTL_FPGA_MAGIC, 13, struct MemoryCacheArgs)

#define IOCTL_SEPARATOR_1 20

#define IOCTL_CONFIG_CONV _IOW(IOCTL_FPGA_MAGIC, 21, struct ConvArgs)
#define IOCTL_CONFIG_POOLING _IOW(IOCTL_FPGA_MAGIC, 22, struct PoolingArgs)
#define IOCTL_CONFIG_EW _IOW(IOCTL_FPGA_MAGIC, 23, struct EWAddArgs)
#define IOCTL_CONFIG_BYPASS _IOW(IOCTL_FPGA_MAGIC, 24, struct BypassArgs)
#define IOCTL_FPGA_REG_READ _IOW(IOCTL_FPGA_MAGIC, 28, struct FpgaRegReadArgs)
#define IOCTL_FPGA_REG_WRITE _IOW(IOCTL_FPGA_MAGIC, 29, struct FpgaRegWriteArgs)

//============================== API =============================

int open_device();
int close_device();

void* fpga_malloc(size_t size);
void fpga_free(void* ptr);
void fpga_copy(void* dst, const void* src, size_t num);
int fpga_flush(void* address, size_t size);
int fpga_invalidate(void* address, size_t size);

int PerformBypass(const struct BypassArgs& args);
int ComputeFpgaConv(const struct SplitConvArgs& args);
int ComputeFpgaPool(const struct PoolingArgs& args);
int ComputeFpgaEWAdd(const struct EWAddArgs& args);
int ComputeFPGAConcat(const struct ConcatArgs& args);

static inline int align_to_x(int num, int x) { return (num + x - 1) / x * x; }

int get_align_image_cw(int cw);
void format_image(framework::Tensor* image_tensor);
void format_fp16_ofm(framework::Tensor* ofm_tensor);  // only allocate memory
void format_fp32_ofm(framework::Tensor* ofm_tensor);

float filter_find_max(framework::Tensor* filter_tensor);
int get_filter_num_per_div(framework::Tensor* filter_tensor, int group_num);
int get_plit_num(framework::Tensor* filter_tensor);
int get_aligned_filter_element_num(int chw);
int get_aligned_filter_num(int num);
void format_filter(framework::Tensor* filter_tensor, float max_value,
                   int group_num);
void format_fc_filter(framework::Tensor* filter_tensor, float max_value);
void format_bias_scale_array(float** bias_scale_array,
                             int element_num_per_division, int num);
void format_concat_output(framework::Tensor* out, int height, int width,
                          int image_num, uint32_t* channel_num);

void fill_split_arg(struct SplitConvArgs* arg, framework::Tensor* input,
                    framework::Tensor* out, framework::Tensor* filter,
                    bool relu_enabled, int group_num, int stride_h,
                    int stride_w, int padding_h, int padding_w, float* bs_ptr);

half fp32_2_fp16(float fp32_num);
float fp16_2_fp32(half fp16_num);

}  // namespace fpga
}  // namespace paddle_mobile
