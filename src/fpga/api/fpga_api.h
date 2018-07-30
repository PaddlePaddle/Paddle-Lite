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
#include <iostream>
#include <limits>

// memory management;

namespace paddle {
namespace mobile {
namespace fpga {
namespace api {

int open_device();
int close_device();

void *fpga_malloc(size_t size);
void fpga_free(void *ptr);
void fpga_copy(void *dst, const void *src, size_t num);

struct FpgaVersionArgs {
  void *buf;
};

struct MemoryToPhysicalArgs {
  const void *src;
  uint64_t physical;
};

struct MemoryCopyArgs {
  void *src;
  void *dst;
  size_t size;
};

struct FpgaQuantArgs {
  float scale;
};

struct FpgaBNArgs {};

struct FpgaConvArgs {
  bool enable_BN = false;
  bool enable_Relu = false;
  struct FpgaBNParam bn_parm;
};

struct FpgaPoolArgs {
  bool enable_BN = false;
  struct FpgaBNParam bn_parm;
};

struct FpgaEWAddArgs {  // only support X + Y
  bool enable_Relu = false;
};

int ComputeFpgaConv(struct FpgaConvArgs);
int ComputeFpgaPool(struct FpgaPoolArgs);
int ComputeFpgaEWAdd(struct FpgaEWAddArgs);

#define IOCTL_FPGA_MAGIC 'FPGA'
#define IOCTL_VERSION _IOW(IOCTL_FPGA_MAGIC, 1, struct FpgaVersionArgs)
#define IOCTL_GET_QUANT _IOW(IOCTL_FPGA_MAGIC, 2, struct FpgaQuantArgs)
#define IOCTL_SET_QUANT _IOW(IOCTL_FPGA_MAGIC, 3, struct FpgaArgs)
#define IOCTL_MEM_COPY _IOW(IOCTL_FPGA_MAGIC, 11, struct MemoryCopyArgs)
#define IOCTL_MEM_TOPHY _IOW(IOCTL_FPGA_MAGIC, 12, struct MemoryToPhysicalArgs)
#define IOCTL_CONFIG_CONV _IOW(IOCTL_FPGA_MAGIC, 21, struct FpgaConvArgs)
#define IOCTL_CONFIG_POOLING _IOW(IOCTL_FPGA_MAGIC, 22, struct FpgaPoolArgs)
#define IOCTL_CONFIG_EW _IOW(IOCTL_FPGA_MAGIC, 23, struct FpgaEWAddArgs)

}  // namespace api
}  // namespace fpga
}  // namespace mobile
}  // namespace paddle
