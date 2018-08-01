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

// memory management;

namespace paddle_mobile {
namespace fpga {

int open_device();
int close_device();

void* fpga_malloc(size_t size);
void fpga_free(void* ptr);
void fpga_copy(void* dst, const void* src, size_t num);

struct FpgaVersionArgs {
  void* buf;
};

struct MemoryToPhysicalArgs {
  const void* src;
  uint64_t physical;
};

struct MemoryCopyArgs {
  void* src;
  void* dst;
  size_t size;
};

struct FpgaQuantArgs {
  float scale;
};

struct FpgaBNArgs {
  bool enabled = false;
  void* bias_addr;
  void* scale_addr;
};

struct FpgaKernelArgs {
  uint32_t width;
  uint32_t height;
  uint32_t stride_h;
  uint32_t stride_w;
};

struct FpgaImageArgs {
  uint32_t width;
  uint32_t height;
  uint32_t channels;
  uint32_t pad_h;
  uint32_t pad_w;
};

struct FpgaConvArgs {
  bool relu_enabled;
  struct FpgaBNArgs BNargs;
  void* image_addr;
  void* filter_addr;
  void* bias_addr;
  void* output_addr;
  float quant_scale;
  struct FpgaImageArgs image;
  uint32_t filter_num;
  uint32_t group_num;

  struct FpgaKernelArgs kernel;
};

struct FpgaPoolArgs {
  void* image_addr;
  void* output_addr;
  struct FpgaImageArgs image;
  struct FpgaKernelArgs kernel;
};

struct FpgaEWAddArgs {
  bool relu_enabled;
  void* image0_addr;
  void* image1_addr;
  void* result_addr;
  uint32_t const0;
  uint32_t const1;
  uint32_t data_len;  // aligned element count
};

int ComputeFpgaConv(struct FpgaConvArgs args);
int ComputeFpgaPool(struct FpgaPoolArgs args);
int ComputeFpgaEWAdd(struct FpgaEWAddArgs args);

#define IOCTL_FPGA_MAGIC 'CNN'
#define IOCTL_VERSION _IOW(IOCTL_FPGA_MAGIC, 1, struct FpgaVersionArgs)
#define IOCTL_GET_QUANT _IOW(IOCTL_FPGA_MAGIC, 2, struct FpgaQuantArgs)
#define IOCTL_SET_QUANT _IOW(IOCTL_FPGA_MAGIC, 3, struct FpgaQuantArgs)
#define IOCTL_MEM_COPY _IOW(IOCTL_FPGA_MAGIC, 11, struct MemoryCopyArgs)
#define IOCTL_CONFIG_CONV _IOW(IOCTL_FPGA_MAGIC, 21, struct FpgaConvArgs)
#define IOCTL_CONFIG_POOLING _IOW(IOCTL_FPGA_MAGIC, 22, struct FpgaPoolArgs)
#define IOCTL_CONFIG_EW _IOW(IOCTL_FPGA_MAGIC, 23, struct FpgaEWAddArgs)

}  // namespace fpga
}  // namespace paddle_mobile
