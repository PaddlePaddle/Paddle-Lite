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

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "api.h"

#define FPGA_TEST_MODE
#ifdef FPGA_TEST_MODE
#include "common/log.h"
#endif

namespace paddle_mobile {
namespace fpga {

static int fd = -1;
static const char *device_path = "/dev/fpgadrv0";

static inline int do_ioctl(int req, const void *arg) {
#ifdef PADDLE_MOBILE_OS_LINUX
  return ioctl(req, (unsigned int64_t)arg);
#else
  return -1;
#endif
}

int open_device() {
  if (fd == -1) {
    fd = open(device_path, O_RDWR);
  }
  return fd;
}

// memory management;
void *fpga_malloc(size_t size) {
#ifdef PADDLE_MOBILE_OS_LINUX
  return reinterpret_cast<void *>(
      mmap64(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
#else
  return malloc(size);
#endif
}

void fpga_free(void *ptr) {
#ifdef PADDLE_MOBILE_OS_LINUX
  munmap(ptr, 0);
#else
  free(ptr);
#endif
}

void fpga_copy(void *dest, const void *src, size_t num) {
  memcpy(dest, src, num);
}

int ComputeFpgaConv(const struct ConvArgs &args) {
#ifdef FPGA_TEST_MODE
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

  return do_ioctl(IOCTL_CONFIG_CONV, &args);
}

int ComputeFpgaPool(const struct PoolingArgs &args) {
#ifdef FPGA_TEST_MODE
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

  return do_ioctl(IOCTL_CONFIG_POOLING, &args);
}

int ComputeFpgaEWAdd(const struct EWAddArgs &args) {
#ifdef FPGA_TEST_MODE
  DLOG << "   relu_enabled:" << args.relu_enabled << "   const0:" << args.const0
       << "   const1:" << args.const1;
  DLOG << "   image0_address:" << args.image0.address
       << "   image0_scale_address:" << args.image0.scale_address
       << "   image0_channels:" << args.image0.channels
       << "   image0_height:" << args.image0.height
       << "   image0_width:" << args.image0.width
       << "   pad0_height:" << args.image0.pad_height
       << "   pad0_width:" << args.image0.pad_width;
  DLOG << "   image1_address:" << args.image1.address
       << "   image_scale_address:" << args.image1.scale_address
       << "   image1_channels:" << args.image1.channels
       << "   image1_height:" << args.image1.height
       << "   image1_width:" << args.image1.width
       << "   pad1_height:" << args.image1.pad_height
       << "   pad_width:" << args.image1.pad_width;
  DLOG << "   out_address:" << args.output.address
       << "   out_scale_address:" << args.output.scale_address;
#endif

  return do_ioctl(IOCTL_CONFIG_EW, &args);
}
int PerformBypass(const struct BypassArgs &args) {
#ifdef FPGA_TEST_MODE
  DLOG << "   layout_type:" << args.layout_type
       << "   convert_type:" << args.convert_type;
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

  return do_ioctl(IOCTL_CONFIG_BYPASS, &args);
}

}  // namespace fpga
}  // namespace paddle_mobile
