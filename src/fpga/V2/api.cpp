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

#include "fpga/V2/api.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <algorithm>
#include <map>
#include "fpga/V2/bias_scale.h"
#include "fpga/V2/filter.h"
#include "fpga/V2/image.h"
#define FPGA_TEST_MODE
// #define PADDLE_MOBILE_OS_LINUX

namespace paddle_mobile {
namespace fpga {

static int fd = -1;
static const char *device_path = "/dev/fpgadrv0";
static std::map<void *, size_t> memory_map;

static inline int do_ioctl(int req, const void *arg) {
#ifdef PADDLE_MOBILE_OS_LINUX
  int result = ioctl(fd, req, (uint64_t)arg);
  PADDLE_MOBILE_ENFORCE(result == 0, "ioctl didn't return correctly");
  return result;
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
  static uint64_t counter = 0;

#ifdef PADDLE_MOBILE_OS_LINUX
  auto ptr = mmap64(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
#else
  auto ptr = malloc(size);
#endif
  counter += size;
  memory_map.insert(std::make_pair(ptr, size));
  //  DLOG << "Address: " << ptr << ", " << size << " bytes allocated. Total "
  //       << counter << " bytes";
  return ptr;
}

void fpga_free(void *ptr) {
  static uint64_t counter = 0;
  size_t size = 0;

  auto iter = memory_map.find(ptr);  // std::map<void *, size_t>::iterator
  if (iter != memory_map.end()) {
    size = iter->second;
    memory_map.erase(iter);
#ifdef PADDLE_MOBILE_OS_LINUX
    munmap(ptr, size);
#else
    free(ptr);
#endif
    counter += size;
    //    DLOG << "Address: " << ptr << ", " << size << " bytes freed. Total "
    //         << counter << " bytes";
  } else {
    DLOG << "Invalid pointer";
  }
}

void fpga_copy(void *dest, const void *src, size_t num) {
  memcpy(dest, src, num);
}

int fpga_flush(void *address, size_t size) {
  struct MemoryCacheArgs args = {nullptr};
  args.address = address;
  args.size = size;
  return do_ioctl(IOCTL_MEMCACHE_FLUSH, &args);
}

int fpga_invalidate(void *address, size_t size) {
  struct MemoryCacheArgs args = {nullptr};
  args.address = address;
  args.size = size;
  return do_ioctl(IOCTL_MEMCACHE_INVAL, &args);
}

half fp32_2_fp16(float fp32_num) {
  unsigned long tmp = *(unsigned long *)(&fp32_num);  // NOLINT
  auto t = (half)(((tmp & 0x007fffff) >> 13) | ((tmp & 0x80000000) >> 16) |
                  (((tmp & 0x7f800000) >> 13) - (112 << 10)));
  if (tmp & 0x1000) {
    t++;  // roundoff
  }
  return t;
}

float fp16_2_fp32(half fp16_num) {
  int frac = (fp16_num & 0x3ff);
  int exp = ((fp16_num & 0x7c00) >> 10) + 112;
  int s = fp16_num & 0x8000;
  int tmp = 0;
  float fp32_num;
  tmp = s << 16 | exp << 23 | frac << 13;
  fp32_num = *(float *)&tmp;  // NOLINT
  return fp32_num;
}

int ComputeBasicConv(const struct ConvArgs &args) {
#ifdef FPGA_TEST_MODE
  DLOG << "======Compute Basic Conv======";
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

int ComputeFpgaConv(const struct SplitConvArgs &args) {
  ComputeBasicConv(args.conv_args[0]);
}

int ComputeFpgaPool(const struct PoolingArgs &args) {
#ifdef FPGA_TEST_MODE
  DLOG << "=============ComputeFpgaPool===========";
  DLOG << "   mode:" << args.mode
       << "   kernel_reciprocal:" << fp16_2_fp32(args.kernel_reciprocal);
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
  DLOG << "=============ComputeFpgaEWAdd===========";
  DLOG << "   relu_enabled:" << args.relu_enabled
       << "   const0:" << fp16_2_fp32(int16_t(args.const0))
       << "   const1:" << fp16_2_fp32(int16_t(args.const1));
  DLOG << "   image0_address:" << args.image0.address
       << "   image0_scale_address:" << args.image0.scale_address
       << "   image0_channels:" << args.image0.channels
       << "   image0_height:" << args.image0.height
       << "   image0_width:" << args.image0.width
       << "   pad0_height:" << args.image0.pad_height
       << "   pad0_width:" << args.image0.pad_width;
  DLOG << "   image1_address:" << args.image1.address
       << "   image1_scale_address:" << args.image1.scale_address
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
  DLOG << "=============ComputeFpgaBypass===========";
  DLOG << "   input_type:" << args.input_data_type
       << "   output_type:" << args.output_data_type
       << "   input_layout_type:" << args.input_layout_type
       << "   output_layout_type:" << args.output_layout_type;
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

int ComputeFPGAConcat(const struct ConcatArgs &args) {
#ifdef FPGA_TEST_MODE
  DLOG << "=============ComputeFpgaConcat===========";
  DLOG << "   Image_num: " << args.image_num
       << "   out_address:" << args.image_out
       << "   out_scale_address:" << args.scale_out
       << "   out_channel:" << args.out_channel;
  DLOG << "   image_height:" << args.height << "   image_width:" << args.width;
  for (int i = 0; i < args.image_num; i++) {
    DLOG << "   " << i << "th:        ";
    DLOG << "   channel_num:" << args.channel_num[i]
         << "   aligned_channel_num:" << args.aligned_channel_num[i]
         << "   image_address:" << args.images_in[i]
         << "   image_scale_address:" << args.scales_in[i];
  }
#endif

  image::concat_images(args.images_in, args.scales_in, args.image_out,
                       args.scale_out, args.image_num, args.channel_num,
                       args.height, args.width, args.aligned_channel_num,
                       args.out_channel);
  return 0;
}

void format_image(framework::Tensor *image_tensor) {
  auto dims = image_tensor->dims();
  auto channel = dims[1], height = dims[2], width = dims[3];
  auto data_ptr = image_tensor->data<float>();
  size_t memory_size = channel * height * width * sizeof(float);
  auto new_data = (float *)fpga_malloc(memory_size);  // NOLINT
  fpga_copy(new_data, data_ptr, memory_size);
  int aligned_channel = filter::calc_aligned_channel((int)channel);  // NOLINT
  image::format_image(&new_data, (int)channel, (int)height,          // NOLINT
                      (int)width,                                    // NOLINT
                      aligned_channel);
  image_tensor->reset_data_ptr(new_data);
}

void format_fp16_ofm(framework::Tensor *ofm_tensor, int aligned_channel) {
  auto dims = ofm_tensor->dims();
  size_t memory_size = 0;
  if (dims.size() == 4) {
    auto height = dims[2], width = dims[3];
    memory_size = height * width * aligned_channel * sizeof(half);
  } else if (dims.size() == 2) {
    memory_size = aligned_channel * sizeof(half);
  } else {
    DLOG << "Wrong ofm dimension";
  }
  auto p = fpga_malloc(memory_size);
  memset(p, 0, memory_size);
  ofm_tensor->reset_data_ptr(p);
}

void format_fp32_ofm(framework::Tensor *ofm_tensor, int aligned_channel) {
  auto dims = ofm_tensor->dims();
  size_t memory_size = 0;
  if (dims.size() == 4) {
    auto height = dims[2], width = dims[3];
    memory_size = height * width * aligned_channel * sizeof(float);
  } else if (dims.size() == 2) {
    memory_size = aligned_channel * sizeof(float);
  } else {
    DLOG << "Wrong ofm dimension";
  }
  auto p = fpga_malloc(memory_size);
  memset(p, 0, memory_size);
  ofm_tensor->reset_data_ptr(p);
}

float filter_find_max(framework::Tensor *filter_tensor) {
  auto filter_ptr = filter_tensor->data<float>();
  return filter::find_max(filter_ptr, (int)filter_tensor->numel());  // NOLINT
}

int get_aligned_channel_num(int channel_num) {
  return filter::calc_aligned_channel(channel_num);
}

int get_aligned_filter_num(framework::Tensor *filter_tensor) {
  auto dims = filter_tensor->dims();
  return filter::calc_aligned_num((int)dims[0], (int)dims[1]);  // NOLINT
}

int get_conv_output_channel(framework::Tensor *filter_tensor) {
  int aligned_filter_num = get_aligned_filter_num(filter_tensor);
  return get_aligned_channel_num(aligned_filter_num);
}
void format_filter(framework::Tensor *filter_tensor, float max_value,
                   int group_num) {
  filter_tensor->scale[0] = float(max_value / 127.0);  // NOLINT
  filter_tensor->scale[1] = float(127.0 / max_value);  // NOLINT
  auto dims = filter_tensor->dims();
  auto num = dims[0], channel = dims[1], height = dims[2], width = dims[3];
  auto data_ptr = filter_tensor->data<float>();
  size_t memory_size = num * channel * height * width * sizeof(float);
  auto new_data = (float *)fpga_malloc(memory_size);  // NOLINT
  fpga_copy(new_data, data_ptr, memory_size);
  filter::format_filter(&new_data, (int)num, (int)channel,  // NOLINT
                        (int)height,                        // NOLINT
                        (int)width, group_num, max_value);  // NOLINT
  filter_tensor->reset_data_ptr(new_data);
}

void format_fc_filter(framework::Tensor *filter_tensor, float max_value) {
  filter_tensor->scale[0] = float(max_value / 127.0);  // NOLINT
  filter_tensor->scale[1] = float(127.0 / max_value);  // NOLINT
  auto dims = filter_tensor->dims();
  auto num = dims[0], channel = dims[1], height = dims[2], width = dims[3];
  auto data_ptr = filter_tensor->data<float>();
  size_t memory_size = num * channel * height * width * sizeof(float);
  auto new_data = (float *)fpga_malloc(memory_size);  // NOLINT
  fpga_copy(new_data, data_ptr, memory_size);
  filter::format_fc_filter(&new_data, (int)num, (int)channel,  // NOLINT
                           (int)height,                        // NOLINT
                           (int)width, 1, max_value);          // NOLINT
  filter_tensor->reset_data_ptr(new_data);
}

void format_bias_scale_array(float **bias_scale_array, int filter_num,
                             int filter_channel) {
  int num_after_alignment =
      filter::calc_aligned_num(filter_channel, filter_channel);
  bias_scale::format_bias_scale_array(bias_scale_array, filter_num,
                                      num_after_alignment);
}

void format_concat_output(framework::Tensor *out, int height, int width,
                          uint32_t out_channel) {
  auto data_ptr = fpga_malloc(out_channel * height * width * sizeof(half));
  auto ddim = framework::make_ddim({1, out_channel, height, width});
  out->Resize(ddim);
  out->reset_data_ptr(data_ptr);
}

int format_conv_data(framework::Tensor *filter_tensor,
                     framework::Tensor *ofm_tensor, float *bs_ptr, int group) {
  float max_value = fpga::filter_find_max(filter_tensor);
  fpga::format_filter(filter_tensor, max_value, group);
  int aligned_num = get_aligned_filter_num(filter_tensor);
  fpga::format_bias_scale_array(&bs_ptr,
                                (int)filter_tensor->dims()[0],  // NOLINT
                                aligned_num);
  int aligned_channel = fpga::get_conv_output_channel(filter_tensor);
  fpga::format_fp16_ofm(ofm_tensor, aligned_channel);
  DLOG << aligned_channel;
  return aligned_channel;
}

int format_fc_data(framework::Tensor *filter_tensor,
                   framework::Tensor *ofm_tensor, float *bs_ptr) {
  float max_value = fpga::filter_find_max(filter_tensor);
  fpga::format_fc_filter(filter_tensor, max_value);
  int aligned_num = get_aligned_filter_num(filter_tensor);
  fpga::format_bias_scale_array(&bs_ptr,
                                (int)filter_tensor->dims()[0],  // NOLINT
                                aligned_num);
  int aligned_channel = fpga::get_conv_output_channel(filter_tensor);
  fpga::format_fp16_ofm(ofm_tensor, aligned_channel);
  DLOG << aligned_channel;
  return aligned_channel;
}

void fill_split_arg(struct SplitConvArgs *arg, framework::Tensor *input,
                    framework::Tensor *out, framework::Tensor *filter,
                    bool relu_enabled, int group_num, int stride_h,
                    int stride_w, int padding_h, int padding_w, float *bs_ptr) {
  auto input_ptr = input->data<float>();
  auto filter_ptr = filter->data<float>();
  auto out_ptr = out->data<float>();

  arg->group_num = (uint32_t)group_num;
  arg->split_num = 1;
  arg->filter_num = (uint32_t)filter->dims()[0];
  arg->output.address = out_ptr;
  arg->output.scale_address = out->scale;
  arg->conv_args =
      (ConvArgs *)fpga_malloc(arg->split_num * sizeof(ConvArgs));  // NOLINT

  arg->concat_arg.image_num = arg->split_num;
  arg->concat_arg.image_out = out_ptr;
  arg->concat_arg.scale_out = out->scale;
  arg->concat_arg.height = (uint32_t)out->dims()[2];
  arg->concat_arg.width = (uint32_t)out->dims()[3];

  int n = arg->split_num;
  arg->concat_arg.images_in =
      (half **)fpga_malloc(n * sizeof(int *));  // NOLINT
  arg->concat_arg.scales_in =
      (float **)fpga_malloc(n * sizeof(float *));  // NOLINT
  arg->concat_arg.channel_num =
      (uint32_t *)fpga_malloc(n * sizeof(uint32_t));  // NOLINT

  for (int i = 0; i < n; i++) {
    arg->conv_args[i].relu_enabled = relu_enabled;
    arg->conv_args[i].sb_address = bs_ptr;
    arg->conv_args[i].filter_address = (int8_t *)filter_ptr;  // NOLINT
    arg->conv_args[i].filter_scale_address = filter->scale;
    arg->conv_args[i].filter_num = arg->filter_num;
    arg->conv_args[i].group_num = (uint32_t)group_num;

    arg->conv_args[i].kernel.stride_h = (uint32_t)stride_h;
    arg->conv_args[i].kernel.stride_w = (uint32_t)stride_w;
    arg->conv_args[i].kernel.height = (uint32_t)filter->dims()[2];
    arg->conv_args[i].kernel.width = (uint32_t)filter->dims()[3];

    arg->conv_args[i].image.address = input_ptr;
    arg->conv_args[i].image.scale_address = input->scale;
    arg->conv_args[i].image.channels = (uint32_t)input->dims()[1];
    arg->conv_args[i].image.height = (uint32_t)input->dims()[2];
    arg->conv_args[i].image.width = (uint32_t)input->dims()[3];
    arg->conv_args[i].image.pad_height = (uint32_t)padding_h;
    arg->conv_args[i].image.pad_width = (uint32_t)padding_w;

    arg->conv_args[i].output.address = out_ptr;
    arg->conv_args[i].output.scale_address = out->scale;
  }
}

}  // namespace fpga
}  // namespace paddle_mobile
