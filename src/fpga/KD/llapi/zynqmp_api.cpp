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

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <map>
#include <utility>

#include "fpga/KD/llapi/config.h"
#include "fpga/KD/llapi/zynqmp_api.h"

namespace paddle_mobile {
namespace zynqmp {

#define PADDLE_MOBILE_OS_LINUX

static int fd = -1;
static const char *device_path = "/dev/fpgadrv0";
static std::map<void *, size_t> memory_map;

static size_t memory_size_max = 0;
static size_t memory_size = 0;

static inline int do_ioctl(uint64_t req, const void *arg) {
#ifdef PADDLE_MOBILE_OS_LINUX
  return ioctl(fd, req, arg);
#else
  return -1;
#endif
}

int open_device() {
  // std::cout << "open_device" << std::endl;
  if (fd == -1) {
    fd = open(device_path, O_RDWR);
  }
  // std::cout << "open_device fd:" << fd << std::endl;
  return fd;
}

void close_device() { close(fd); }

void reset_device() {
  FpgaResetArgs args;
  do_ioctl(IOCTL_FPGA_RESET, &args);
}

// memory management;
void *fpga_malloc(size_t size) {
// std::cout << "fpga malloc: 0x" << std::hex << size  << std::dec << "  (" <<
// size << ") - ";
#ifdef ENABLE_DEBUG
// std::cout << "fpga_malloc:" << size << std::endl;
#endif
#ifdef PADDLE_MOBILE_OS_LINUX
  void *ptr = reinterpret_cast<void *>(
      mmap64(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  if (ptr == NULL) {
    std::cout << "not enough memory !";
    exit(-1);
  }
  // std::cout << std::hex << ptr << std::dec << std::endl;
  memory_map.insert(std::make_pair(ptr, size));
  memory_size += size;
  if (memory_size > memory_size_max) {
    memory_size_max = memory_size;
  }
  return ptr;
#else
  return malloc(size);
#endif
}

size_t fpga_get_memory_size(void *ptr) { return memory_map[ptr]; }

size_t fpga_get_memory_size_max() { return memory_size_max; }

size_t fpga_diagnose_memory(int detailed) {
  size_t total = 0;
  //        size_t size = 0;
  //        int i = 0;
  auto iter = memory_map.begin();  // std::map<void *, size_t>::iterator
  while (iter != memory_map.end()) {
    total += iter->second;
    iter++;
  }
  return total;
}

void fpga_free(void *ptr) {
  size_t size = 0;
  auto iter = memory_map.find(ptr);  // std::map<void *, size_t>::iterator
  if (iter != memory_map.end()) {
    size = iter->second;
    memory_map.erase(iter);
  }

  memory_size -= size;

#ifdef PADDLE_MOBILE_OS_LINUX

  munmap(ptr, size);
#else
  free(ptr);
#endif
}

void fpga_copy(void *dst, const void *src, int size) { memcpy(dst, src, size); }

int fpga_flush(void *address, size_t size) {
  struct MemoryCacheArgs args;
  args.address = address;
  args.size = size;
  return do_ioctl(IOCTL_MEMCACHE_FLUSH, &args);
}

int fpga_invalidate(void *address, size_t size) {
  // std::cout <<
  // "=================================================================================="
  // << std::endl;
  struct MemoryCacheArgs args;
  args.address = address;
  args.size = size;
  return do_ioctl(IOCTL_MEMCACHE_INVAL, &args);
}

int invalidate_cache(void *addr, int size) {
  struct MemoryCacheArgs args;
  args.address = addr;
  args.size = size;
  return do_ioctl(IOCTL_MEMCACHE_INVAL, &args);
}

int flush_cache(void *addr, int size) {
  struct MemoryCacheArgs args;
  args.address = addr;
  args.size = size;
  return do_ioctl(IOCTL_MEMCACHE_FLUSH, &args);
}

void fpga_copy(void *dest, const void *src, size_t num) {
  memcpy(dest, src, num);
}

int fpga_reset() {
  struct FpgaResetArgs args;
  return  do_ioctl(IOCTL_FPGA_RESET, &args);
}

int ioctl_conv(const struct ConvArgs &args) {
#ifdef ENABLE_DEBUG
//        std::cout << "======Compute Basic Conv======";
//        std::cout << "   relu_enabled:" << args.relu_enabled
//       << "   sb_address:" << args.sb_address
//       << "   filter_address:" << args.filter_address
//       << "   filter_num:" << args.filter_num
//       << "   group_num:" << args.group_num;
//  std::cout << "   image_address:" << args.image.address
//       << "   image_scale_address:" << args.image.scale_address
//       << "   image_channels:" << args.image.channels
//       << "   image_height:" << args.image.height
//       << "   image_width:" << args.image.width
//       << "   pad_height:" << args.image.pad_height
//       << "   pad_width:" << args.image.pad_width;
//  std::cout << "   kernel_height:" << args.kernel.height
//       << "   kernel_width:" << args.kernel.width
//       << "   stride_h:" << args.kernel.stride_h
//       << "   stride_w:" << args.kernel.stride_w;
//  std::cout << "   out_address:" << args.output.address
//       << "   out_scale_address:" << args.output.scale_address;
//
//       float* in_scale = (float*)args.image.scale_address;
//       std::cout << "inv_scale:" << in_scale[0] << "," << in_scale[1] <<
//       std::endl;

#endif

  return do_ioctl(IOCTL_CONFIG_CONV, &args);

  // return 0;
}

int compute_fpga_conv_basic(const struct ConvArgs &args) {
#ifdef ENABLE_DEBUG

//        std::cout << "======Compute Basic Conv======";
//        std::cout << "   relu_enabled:" << args.relu_enabled
//       << "   sb_address:" << args.sb_address
//       << "   filter_address:" << args.filter_address
//       << "   filter_num:" << args.filter_num
//       << "   group_num:" << args.group_num;
//  std::cout << "   image_address:" << args.image.address
//       << "   image_scale_address:" << args.image.scale_address
//       << "   image_channels:" << args.image.channels
//       << "   image_height:" << args.image.height
//       << "   image_width:" << args.image.width
//       << "   pad_height:" << args.image.pad_height
//       << "   pad_width:" << args.image.pad_width;
//  std::cout << "   kernel_height:" << args.kernel.height
//       << "   kernel_width:" << args.kernel.width
//       << "   stride_h:" << args.kernel.stride_h
//       << "   stride_w:" << args.kernel.stride_w;
//  std::cout << "   out_address:" << args.output.address
//       << "   out_scale_address:" << args.output.scale_address;

// float *in_scale = (float *)args.image.scale_address;
//        std::cout << " scale:" << in_scale[0] << "," << in_scale[1] <<
//        std::endl;

// float *filter_scale = (float *)args.filter_scale_address;
//        std::cout << " filter scale:" << filter_scale[0] << "," <<
//        filter_scale[1] << std::endl;

#endif
  return do_ioctl(IOCTL_CONFIG_CONV, &args);
}

int compute_fpga_conv(const struct SplitConvArgs &args) {
  // return do_ioctl(IOCTL_CONFIG_CONV, &args);
  int split_num = args.split_num;
  int ret = -1;
  for (int i = 0; i < split_num; i++) {
    // ComputeBasicConv(args.conv_args[i]);
    ret = compute_fpga_conv_basic(args.conv_arg[i]);
  }

  if (split_num > 1) {
    std::cout << "Split num > 1 !!!!!!!!!!!!!!!!!!" << std::endl;
    exit(-1);
  }
  return ret;
}

int compute_fpga_pool(const struct PoolingArgs &args) {
  return do_ioctl(IOCTL_CONFIG_POOLING, &args);
}

int compute_fpga_ewadd(const struct EWAddArgs &args) {
  return do_ioctl(IOCTL_CONFIG_EW, &args);
}

int get_device_info(const struct DeviceInfo &args) {
  // DeviceInfo info;
  // struct DeviceInfo* a = &info;
  int ret = do_ioctl(IOCTL_DEVICE_INFO, &args);
  // std::cout << "a." << a->filter_cap << std::endl;
  return ret;
}

int perform_bypass(const struct BypassArgs &args) {
  int ret = -1;
  int size = args.image.channels * args.image.width * args.image.height;
  int max_size = 1 << 21;

  float times = 1.0 * size / max_size;
  int count = static_cast<int>(times);

  void *input_address = args.image.address;
  int type_size =
      args.input_data_type == DATA_TYPE_FP32 ? sizeof(float) : sizeof(int16_t);

  void *output_address = args.output.address;
  int out_type_size =
      args.output_data_type == DATA_TYPE_FP32 ? sizeof(float) : sizeof(int16_t);

  float scales[2];
  struct BypassArgs bypassArgs = args;
  bypassArgs.image.width = 1;
  bypassArgs.image.height = 1;
  bypassArgs.output.scale_address = scales;


  float scale = 0;
  for (int i = 0; i < count; ++i) {
    bypassArgs.image.channels = max_size;
    bypassArgs.image.address =
        reinterpret_cast<char *>(input_address + i * max_size * type_size);
    bypassArgs.output.address =
        reinterpret_cast<char *>(output_address + i * max_size * out_type_size);
    ret = do_ioctl(IOCTL_CONFIG_BYPASS, &bypassArgs);
    scale = std::max(scale, scales[0]);

    if (ret != 0) {
      return ret;
    }
  }

  int remainder = size - max_size * count;
  // std::cout << "remainder:" << remainder << std::endl;
  if (remainder > 0) {
    bypassArgs.image.channels = remainder;
    bypassArgs.image.address =
        reinterpret_cast<char *>(input_address + count * max_size * type_size);
    bypassArgs.output.address = reinterpret_cast<char *>(
        output_address + count * max_size * out_type_size);
    ret = do_ioctl(IOCTL_CONFIG_BYPASS, &bypassArgs);
    scale = std::max(scale, scales[0]);

  }
 
  args.output.scale_address[0] = scale;
  args.output.scale_address[1] = 1.0f / scale;
  return ret;
}

int compute_fpga_concat(const struct ConcatArgs &args) { return -1; }

int compute_fpga_scale(const struct ScaleArgs &args) {
#ifdef ENABLE_DEBUG
  std::cout << "======Compute Scale======";
  std::cout << "scale_address:" << args.scale_address << std::endl;
  std::cout << "bias_address:" << args.bias_address << std::endl;

  std::cout << "wc_alignment:" << args.wc_alignment << std::endl;
  std::cout << "channel_alignment:" << args.channel_alignment << std::endl;

  std::cout << "   image_address:" << args.image.address
            << "   image_scale_address:" << args.image.scale_address
            << "   image_channels:" << args.image.channels
            << "   image_height:" << args.image.height
            << "   image_width:" << args.image.width
            << "   pad_height:" << args.image.pad_height
            << "   pad_width:" << args.image.pad_width;

  std::cout << "   out_address:" << args.output.address
            << "   out_scale_address:" << args.output.scale_address;

#endif
  return do_ioctl(IOCTL_CONFIG_SCALE, &args);
}

int compute_fpga_dwconv(const struct DWconvArgs &args) {
#ifdef ENABLE_DEBUG
  std::cout << "======Compute Basic Conv======";
  std::cout << "   relu_enabled:" << args.relu_enabled
            << "   filter_address:" << args.filter_address;
  std::cout << "   image_address:" << args.image.address
            << "   image_scale_address:" << args.image.scale_address
            << "   image_channels:" << args.image.channels
            << "   image_height:" << args.image.height
            << "   image_width:" << args.image.width
            << "   pad_height:" << args.image.pad_height
            << "   pad_width:" << args.image.pad_width;
  std::cout << "   kernel_height:" << args.kernel.height
            << "   kernel_width:" << args.kernel.width
            << "   stride_h:" << args.kernel.stride_h
            << "   stride_w:" << args.kernel.stride_w;
  std::cout << "   out_address:" << args.output.address
            << "   out_scale_address:" << args.output.scale_address;

  // float *in_scale = (float *)args.image.scale_address;
  // std::cout << "inv_scale:" << in_scale[0] << "," << in_scale[1] <<
  // std::endl;
#endif
  return do_ioctl(IOCTL_CONFIG_DWCONV, &args);
}

int config_activation(const struct ActiveParamterArgs& args) {
    return do_ioctl(IOCTL_CONFIG_ACTIVATION_PARAMETER, &args);
}

// int config_power(const struct PowerArgs& args) {
//     return do_ioctl(IOCTL_CONFIG_POWER, &args);
// }

int config_inplace(const struct InplaceArgs &args) {
  return do_ioctl(IOCTL_CONFIG_INPLACE, &args);
}

int config_norm_param(const struct NormalizeParameterArgs &args) {
  return do_ioctl(IOCTL_CONFIG_NORMALIZE_PARAMETER, &args);
}

int compute_norm(const struct NormalizeArgs &args) {
  return do_ioctl(IOCTL_CONFIG_NORMALIZE, &args);
}

int compute_fpga_resize(const struct ResizeArgs &args) {
  return do_ioctl(IOCTL_CONFIG_RESIZE, &args);
}

int16_t fp32_2_fp16(float fp32_num) {
  unsigned long tmp = *(unsigned long *)(&fp32_num);  // NOLINT
  auto t = (int16_t)(((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) |
                     (((tmp & 0x7f800000) >> 13) - (112 << 10)));
  if (tmp & 0x1000) {
    t++;  // roundoff
  }
  return t;
}

float fp16_2_fp32(int16_t fp16_num) {
  if (0 == fp16_num) {
    return 0;
  }
  int frac = (fp16_num & 0x3ff);
  int exp = ((fp16_num & 0x7c00) >> 10) + 112;
  int s = fp16_num & 0x8000;
  int tmp = 0;
  float fp32_num = 0;
  tmp = s << 16 | exp << 23 | frac << 13;
  fp32_num = *(float *)&tmp;  // NOLINT
  return fp32_num;
}

}  // namespace zynqmp
}  // namespace paddle_mobile
