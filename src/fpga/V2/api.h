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
#include "fpga/V2/driver/driver.h"
#include "fpga/V2/driver/pe.h"
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
  struct ConvArgs* conv_args;
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

int open_device();
int close_device();
void* fpga_malloc(size_t size);
void fpga_free(void* ptr);

static inline int align_to_x(int num, int x) { return (num + x - 1) / x * x; }

float filter_find_max(framework::Tensor* filter_tensor);
int get_aligned_channel_num(int channel_num);
int get_aligned_filter_num(framework::Tensor* filter_tensor);
int get_conv_output_channel(framework::Tensor* filter_tensor);

void format_image(framework::Tensor* image_tensor);
void format_fp16_ofm(framework::Tensor* ofm_tensor,
                     int aligned_channel);  // only allocate memory
void format_fp32_ofm(framework::Tensor* ofm_tensor, int aligned_channel);

void format_filter(framework::Tensor* filter_tensor, float max_value,
                   int group_num);
void format_fc_filter(framework::Tensor* filter_tensor, float max_value);
void format_bias_scale_array(float** bias_scale_array, int filter_num,
                             int filter_channel);
void format_concat_output(framework::Tensor* out, int height, int width,
                          uint32_t out_channel);
int format_conv_data(framework::Tensor* filter_tensor,
                     framework::Tensor* ofm_tensor, float* bs_ptr, int group);
int format_fc_data(framework::Tensor* filter_tensor,
                   framework::Tensor* ofm_tensor, float* bs_ptr);
void fill_split_arg(struct SplitConvArgs* arg, framework::Tensor* input,
                    framework::Tensor* out, framework::Tensor* filter,
                    bool relu_enabled, int group_num, int stride_h,
                    int stride_w, int padding_h, int padding_w, float* bs_ptr);

half fp32_2_fp16(float fp32_num);
float fp16_2_fp32(half fp16_num);

}  // namespace fpga
}  // namespace paddle_mobile
