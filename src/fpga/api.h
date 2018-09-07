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

// memory management;

namespace paddle_mobile {
namespace fpga {

int open_device();
int close_device();

void* fpga_malloc(size_t size);
void fpga_free(void* ptr);
void fpga_copy(void* dst, const void* src, size_t num);

enum DataConvertType {
  DATA_NO_CONVERT = 0,
  DATA_FP32_TO_FP16 = 1,
  DATA_FP16_TO_FP32 = 2,
};

enum LayoutConvertType {
  LAYOUT_NO_CONVERT = 0,
  LAYOUT_CHW_TO_HWC = 1,
  LAYOUT_HWC_TO_CHW = 2,
};

struct VersionArgs {
  void* buffer;
};

struct MemoryCopyArgs {
  void* src;
  void* dest;
  size_t size;
};

/**
Conv and Pooling kernel
*/
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

struct WrapperConvArgs {
  uint32_t split_num;
  uint32_t group_num;
  uint32_t filter_num;
  struct ImageOutputArgs output;
  struct ConvArgs* args;
};

struct PoolingArgs {
  struct KernelArgs kernel;
  struct ImageInputArgs image;  // input image;
  struct ImageOutputArgs output;
};

// elementwise add arguments
struct EWAddArgs {
  bool relu_enabled;

  float const0;  // output0 = const0 x input0 + const1 x input1;
  float const1;
  struct ImageInputArgs image0;
  struct ImageInputArgs image1;
  struct ImageOutputArgs output;
};

struct BypassArgs {
  enum DataConvertType convert_type;
  enum LayoutConvertType layout_type;
  struct ImageInputArgs image;
  struct ImageOutputArgs output;
};

struct FpgaRegWriteArgs {
  uint64_t address;  //
  uint64_t value;
};

#define IOCTL_FPGA_MAGIC 'FPGA'

#define IOCTL_VERSION _IOW(IOCTL_FPGA_MAGIC, 01, struct VersionArgs)

#define IOCTL_SEPARATOR_0 10

#define IOCTL_MEM_COPY _IOW(IOCTL_FPGA_MAGIC, 11, struct MemoryCopyArgs)

#define IOCTL_SEPARATOR_1 20

#define IOCTL_CONFIG_CONV _IOW(IOCTL_FPGA_MAGIC, 21, struct ConvArgs)
#define IOCTL_CONFIG_POOLING _IOW(IOCTL_FPGA_MAGIC, 22, struct PoolingArgs)
#define IOCTL_CONFIG_EW _IOW(IOCTL_FPGA_MAGIC, 23, struct EWAddArgs)
#define IOCTL_CONFIG_BYPASS _IOW(IOCTL_FPGA_MAGIC, 24, struct BypassArgs)
#define IOCTL_FPGA_REG_READ _IOW(IOCTL_FPGA_MAGIC, 28, struct FpgaRegReadArgs)
#define IOCTL_FPGA_REG_WRITE _IOW(IOCTL_FPGA_MAGIC, 29, struct FpgaRegWriteArgs)

enum FPGA_ERR_TYPE {
  ERR_IOCTL_CMD = -1,
  ERR_TIMEOUT = -2,
  ERR_COMPLETION_TIMEOUT = -3,
  ERR_INVALID_FPGA_ADDR = -4,
  ERR_NOMEM = -5,
  ERR_NO_RESERVE_MEM = -6,
  ERR_COPY_FROM_USER = -7,
  ERR_COPY_TO_USER = -8,
  ERR_DEL_TIMER = -9,
  ERR_ENABLE_MSI = -10,
  ERR_REGISTER_IRQ = -11,
  ERR_PCIE_REGISTER = -12,
  ERR_PCIE_PROBE = -13,
  ERR_REGISTER_BLOCK = -14,
  ERR_ALLOC_GENDISK = -15,
  ERR_INIT_QUEUE = -16,
  ERR_WAIT = -17,
  ERR_ECC_ERROR = -31,
  ERR_FPGA_FAIL_STOP = -64,
  ERR_FPGA_DEBUG_STOP = -113,
  DEV_TMP_UNAVAILABLE = -128
};

//============================== API =============================

int PerformBypass(const struct BypassArgs& args);
int ComputeFpgaConv(const struct WrapperConvArgs& args);
int ComputeFpgaPool(const struct PoolingArgs& args);
int ComputeFpgaEWAdd(const struct EWAddArgs& args);

static inline int align_to_x(int num, int x) { return (num + x - 1) / x * x; }
void format_image(framework::Tensor* image_tensor);
void format_ofm(framework::Tensor* ofm_tensor);  // only allocate memory
float filter_find_max(framework::Tensor* filter_tensor);
int get_element_num_per_div(framework::Tensor* filter_tensor, int group_num);
int get_plit_num(framework::Tensor* filter_tensor);
int get_aligned_filter_element_num(int chw);
int get_aligned_filter_num(int num);

void format_filter(framework::Tensor* filter_tensor, float max_value,
                   int group_num);
void format_fc_matrix(framework::Tensor* filter_tensor, float max_value,
                      int group_num, int height = 1, int width = 1);
void format_bias_scale_array(float** bias_scale_array,
                             int element_num_per_division, int num);

}  // namespace fpga
}  // namespace paddle_mobile
