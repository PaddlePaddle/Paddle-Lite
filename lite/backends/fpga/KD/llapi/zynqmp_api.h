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

#ifndef PADDLE_LITE_SRC_FPGA_KD_ZYNQMP_API_H
#define PADDLE_LITE_SRC_FPGA_KD_ZYNQMP_API_H

#include <stdint.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "lite/backends/fpga/KD/float16.hpp"

namespace paddle {
namespace zynqmp {

typedef int16_t half;

#define IMAGE_ALIGNMENT 16           // Aligned to 16
#define FILTER_ELEMENT_ALIGNMENT 16  // Filter element number aligned to 16
#define BS_NUM_ALIGNMENT 8
#define BIAS_NUM_ALIGNMENT 16

static int POOL_CAP = 4096;

enum DDataType {
  DATA_TYPE_FP32 = 1,
  DATA_TYPE_FP16 = 0,
};

enum DLayoutType {
  LAYOUT_CHW = 1,
  LAYOUT_HWC = 0,
};

enum ActivationType {
  TYPE_NONE = 0,
  TYPE_RELU = 1,
  TYPE_RELU6 = 2,
  TYPE_LEAKY_RELU = 3,
  TYPE_SIGMOID = 4,
};

struct DeviceInfoArgs {
  uint32_t filter_cap;
  uint32_t version;
  uint16_t device_type;
  uint32_t column;
  uint32_t pool_cap;
  uint32_t reserved1;
  uint32_t reserved2;
  uint32_t reserved3;
  uint32_t reserved4;
  uint32_t reserved5;
};

struct VersionArgs {
  void* buffer;
  size_t size;
};

struct MemoryCopyArgs {
  void* src;
  void* dest;
  size_t size;
};

struct MemoryCacheArgs {
  void* address;
  size_t size;
};

struct MemoryBarrierArgs {
  uint16_t dummy;
};

struct ActivationArgs {
  enum ActivationType type;
  uint16_t leaky_relu_factor = float_to_half(0.0f);
};

struct NormalizeParameterArgs {
  uint32_t channel;
  uint32_t hight_width;
  bool enabled;
};

struct InplaceArgs {
  // bool                          findmax_restart;
  struct ActivationArgs active_param;
  struct NormalizeParameterArgs normalize_param;
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
  void* address;        // input featuremap virtual address
  void* scale_address;  // input scale address;
  uint32_t channels;
  uint32_t width;  // featuremap width
  uint32_t height;
  uint32_t pad_width;  // padding width;
  uint32_t pad_height;
};

struct ImageOutputArgs {
  void* address;           // output result address;
  float16* scale_address;  // output scale address;
};

struct DeconvArgs {
  bool enabled;
  uint16_t sub_kernel_num;   // which is the stride of deconv, means that deconv
                             // will be divided into several sub conv operation
  uint16_t invalid_col_num;  // which will be dumped in the left and right for
                             // each row directly in FPGA
};

struct StrideArgs {
  uint32_t wr_offset;
  uint32_t rd_offset;
  bool wr_enabled;
  bool rd_enabled;
};

struct QuantArgs {
  uint16_t dynamic_range;
  uint32_t inv_dynamic_range;
};

struct ConvArgs {
  void* sb_address;  // scale and bias are interlaced;
  void* filter_address;
  void* filter_scale_address;
  uint32_t filter_num;
  uint32_t group_num;
  uint32_t dilation;

  struct DeconvArgs deconv;
  struct KernelArgs kernel;
  struct StrideArgs stride;
  struct ImageInputArgs image;  // input image;
  struct ImageOutputArgs output;
  struct QuantArgs quant;
  struct InplaceArgs inplace;
};

struct DWconvArgs {
  void* bias_address;
  void* filter_address;
  void* filter_scale_address;
  struct KernelArgs kernel;
  struct ImageInputArgs image;
  struct ImageOutputArgs output;
  uint16_t out_width;
  uint16_t out_height;
  uint16_t sub_conv_num;
  uint32_t dilation;
  struct InplaceArgs inplace;
  struct QuantArgs quant;
};

struct PoolingArgs {
  uint16_t mode;
  uint32_t kernel_reciprocal;
  struct KernelArgs kernel;
  struct ImageInputArgs image;  // input image;
  struct ImageOutputArgs output;
  uint16_t out_width;
  uint16_t out_height;
  struct InplaceArgs inplace;
  struct QuantArgs quant;
};

// elementwise add arguments
struct EWAddArgs {
  uint32_t const0;  // output0 = const0 x input0 + const1 x input1;
  uint32_t const1;
  struct ImageInputArgs image0;
  struct ImageInputArgs image1;
  struct ImageOutputArgs output;
  struct InplaceArgs inplace;
  struct QuantArgs quant;
};

struct BypassArgs {
  enum DDataType input_data_type;
  enum DDataType output_data_type;
  enum DLayoutType input_layout_type;
  enum DLayoutType output_layout_type;
  struct ImageInputArgs image;
  struct ImageOutputArgs output;
  struct InplaceArgs inplace;
};

struct ScaleArgs {
  void* scale_address;
  void* bias_address;
  uint32_t wc_alignment;
  uint32_t channel_alignment;

  struct ImageInputArgs image;
  struct ImageOutputArgs output;
  struct InplaceArgs inplace;
};

struct NormalizeArgs {
  void* input_image_address;
  void* output_image_address;
  uint32_t image_width;
  uint32_t image_height;
  uint32_t image_channel;
  uint32_t* output_scale_address;
  struct InplaceArgs inplace;
};

struct PreprocessArgs {
  void* input_image_address;
  void* output_image_address;
  uint32_t input_width;
  uint32_t input_height;
  uint32_t output_width;
  uint32_t output_height;
  uint32_t height_ratio;
  uint32_t width_ratio;
  uint16_t mean0;
  uint16_t mean1;
  uint16_t mean2;
  uint16_t scale0;
  uint16_t scale1;
  uint16_t scale2;
  uint32_t rd_ring_buf_size;
  uint32_t wr_ring_buf_size;
  uint32_t vedio_in_fomat;
  uint32_t vedio_out_fomat;
  uint32_t vedio_source;
  bool mean_scale_enabled;
};

struct ResizeArgs {
  void* input_image_address;
  void* output_image_address;
  uint32_t input_width;
  uint32_t input_height;
  uint32_t image_channel;
  uint32_t output_width;
  uint32_t output_height;
  uint32_t height_ratio;
  uint32_t width_ratio;
  uint16_t* output_scale_address;
};

struct PowerParameterArgs {
  uint16_t shift;
  uint16_t scale;
  uint16_t power;
};

struct FpgaRegWriteArgs {
  uint64_t address;
  uint64_t value;
};

struct FpgaRegReadArgs {
  uint64_t address;
  uint64_t value;
};

struct FpgaResetArgs {
  uint32_t dummy;
};

#define IOCTL_FPGA_MAGIC (('F' + 'P' + 'G' + 'A') / 4)
// #define IOCTL_MEMORY_MAGIC                  (('M' + 'E' + 'M' + 'Y') / 4)

#define IOCTL_VERSION _IOW(IOCTL_FPGA_MAGIC, 01, struct VersionArgs)

#define IOCTL_SEPARATOR_0 10

#define IOCTL_MEM_COPY _IOW(IOCTL_FPGA_MAGIC, 11, struct MemoryCopyArgs)
#define IOCTL_MEMCACHE_INVAL _IOW(IOCTL_FPGA_MAGIC, 12, struct MemoryCacheArgs)
#define IOCTL_MEMCACHE_FLUSH _IOW(IOCTL_FPGA_MAGIC, 13, struct MemoryCacheArgs)
#define IOCTL_MEMORY_BARRIER \
  _IOW(IOCTL_FPGA_MAGIC, 14, struct MemoryBarrierArgs)

#define IOCTL_SEPARATOR_1 20

#define IOCTL_CONFIG_CONV _IOW(IOCTL_FPGA_MAGIC, 21, struct ConvArgs)
#define IOCTL_CONFIG_POOLING _IOW(IOCTL_FPGA_MAGIC, 22, struct PoolingArgs)
#define IOCTL_CONFIG_EW _IOW(IOCTL_FPGA_MAGIC, 23, struct EWAddArgs)
#define IOCTL_CONFIG_BYPASS _IOW(IOCTL_FPGA_MAGIC, 24, struct BypassArgs)
#define IOCTL_CONFIG_SCALE _IOW(IOCTL_FPGA_MAGIC, 25, struct ScaleArgs)
#define IOCTL_CONFIG_NORMALIZE _IOW(IOCTL_FPGA_MAGIC, 26, struct NormalizeArgs)
#define IOCTL_CONFIG_RESIZE _IOW(IOCTL_FPGA_MAGIC, 30, struct ResizeArgs)
#define IOCTL_CONFIG_DWCONV _IOW(IOCTL_FPGA_MAGIC, 31, struct DWconvArgs)

#define IOCTL_CONFIG_INPLACE _IOW(IOCTL_FPGA_MAGIC, 40, struct InplaceArgs)

#define IOCTL_FPGA_REG_READ _IOW(IOCTL_FPGA_MAGIC, 50, struct FpgaRegReadArgs)
#define IOCTL_FPGA_REG_WRITE _IOW(IOCTL_FPGA_MAGIC, 51, struct FpgaRegWriteArgs)
#define IOCTL_FPGA_RESET _IOW(IOCTL_FPGA_MAGIC, 52, struct FpgaResetArgs)

#define IOCTL_DEVICE_INFO _IOW(IOCTL_FPGA_MAGIC, 100, struct DeviceInfoArgs)

#define IOCTL_SEPARATOR_2 110

#define IOCTL_SEPARATOR_3 200
#define IOCTL_PREPROCESS _IOW(IOCTL_FPGA_MAGIC, 201, struct PreprocessArgs)

//============================== API =============================
std::ostream& operator<<(std::ostream& os, const ConvArgs& args);
std::ostream& operator<<(std::ostream& os, const DWconvArgs& args);

inline int align_to_x(int num, int x) { return (num + x - 1) / x * x; }
inline int align_to_x_floor(int num, int x) { return (num / x) * x; }

void set_pool_cap(uint32_t pool_cap);
uint32_t get_pool_cap();

int open_device();
void close_device();
void reset_device();

void* fpga_malloc(size_t size);
void fpga_free(void* ptr);
size_t fpga_get_memory_size(void* ptr);
size_t fpga_get_memory_size_max();
size_t fpga_diagnose_memory(int detailed);

void fpga_copy(void* dst, const void* src, int size);

int fpga_flush(void* address, size_t size);
int fpga_invalidate(void* address, size_t size);

int get_version(const struct VersionArgs& args);
int get_device_info(const struct DeviceInfoArgs& args);

int perform_bypass(const struct BypassArgs& args);
int compute_fpga_conv_basic(const struct ConvArgs& args);
int compute_fpga_pool(const struct PoolingArgs& args);
int compute_fpga_ewadd(const struct EWAddArgs& args);
int compute_fpga_scale(const struct ScaleArgs& args);
int compute_fpga_concat(const struct ConcatArgs& args);
int compute_fpga_resize(const struct ResizeArgs& args);

int config_global_pool(const struct GlobalPoolArgs& args);
int compute_fpga_dwconv(const struct DWconvArgs& args);
int compute_norm(const struct NormalizeArgs& args);

int flush_cache(void* addr, int size);
int invalidate_cache(void* addr, int size);

int fpga_reset();
int compute_preprocess(const struct PreprocessArgs& args);

int16_t fp32_2_fp16(float fp32_num);
float fp16_2_fp32(int16_t fp16_num);

}  // namespace zynqmp
}  // namespace paddle

#endif  // PADDLE_LITE_SRC_FPGA_KD_ZYNQMP_API_H
