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

#include "fpga/V2/driver/pe.h"
#include "fpga/V2/fpga_common.h"
#include "framework/tensor.h"

namespace paddle_mobile {
namespace fpga {

int open_device();
int close_device();
void* fpga_malloc(size_t size);
void fpga_free(void* ptr);
void fpga_copy(void* dest, const void* src, size_t num);
int fpga_flush(void* address, size_t size);
int fpga_invalidate(void* address, size_t size);

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

}  // namespace fpga
}  // namespace paddle_mobile
