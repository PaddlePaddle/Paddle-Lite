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

#include <string>
#include "fpga/common/fpga_common.h"
#include "fpga/common/pe.h"
#include "framework/tensor.h"

namespace paddle_mobile {
namespace fpga {

void format_image(framework::Tensor* image_tensor);
void format_ofm(framework::Tensor* ofm_tensor);
void format_fp16_ofm(framework::Tensor* ofm_tensor);  // only allocate memory
void format_fp16_ofm(framework::Tensor* ofm_tensor, framework::DDim dims);
void format_fp32_ofm(framework::Tensor* ofm_tensor);

float filter_find_max(framework::Tensor* filter_tensor);
int get_filter_num_per_div(framework::Tensor* filter_tensor, int group_num);
int get_deconv_filter_num_per_div(framework::Tensor* filter_tensor,
                                  int group_num, int stride);

int get_plit_num(framework::Tensor* filter_tensor);
int get_deconv_plit_num(framework::Tensor* filter_tensor, int stride);

int get_aligned_filter_element_num(int chw);
void format_filter(framework::Tensor* filter_tensor, float max_value,
                   int group_num);
void format_fc_filter(framework::Tensor* filter_tensor, float max_value);
void format_bias_scale_array(float** bias_scale_array,
                             int element_num_per_division, int num);
void format_bias_array(float** bias_array, int num);
void format_concat_output(framework::Tensor* out, int height, int width,
                          int image_num, uint32_t* channel_num);

void fill_split_arg(struct SplitConvArgs* arg, framework::Tensor* input,
                    framework::Tensor* out, framework::Tensor* filter,
                    ActivationType activation_enable,
                    int16_t leaky_relu_negative_slope, int group_num,
                    int stride_h, int stride_w, int padding_h, int padding_w,
                    float* bs_ptr);
void fill_deconv_arg(struct DeconvArgs* arg, framework::Tensor* input,
                     framework::Tensor* out, framework::Tensor* filter,
                     ActivationType activation_enable,
                     int16_t leaky_relu_negative_slope, int group_num,
                     int stride_h, int stride_w, int padding_h, int padding_w,
                     float* bs_ptr);
void fill_dwconv_arg(struct DWconvArgs* arg, framework::Tensor* input,
                     framework::Tensor* out, framework::Tensor* filter,
                     ActivationType activation_enable,
                     int16_t leaky_relu_negative_slope, int stride_h,
                     int stride_w, int padding_h, int padding_w,
                     float* bias_ptr);
void fill_DWDeconv_arg(struct DWDeconvArgs* arg, framework::Tensor* input,
                       framework::Tensor* out, framework::Tensor* filter,
                       ActivationType activation_enable,
                       int16_t leaky_relu_negative_slope, int stride_h,
                       int stride_w, int padding_h, int padding_w,
                       float* bs_ptr);

void format_deconv_filter(framework::Tensor* filter_tensor, float max_value,
                          int group_num, int stride);
void format_dwconv_filter(framework::Tensor* filter_tensor, float* scale_ptr);
void format_conv_data(framework::Tensor* filter_tensor,
                      framework::Tensor* ofm_tensor, float** bs_ptr, int group);
void format_deconv_data(framework::Tensor* filter_tensor,
                        framework::Tensor* ofm_tensor, float** bs_ptr,
                        int group, int sub_conv_n);
void format_dwconv_data(framework::Tensor* filter_tensor,
                        framework::Tensor* ofm_tensor, float* scale_ptr,
                        float** bias_ptr);
void format_DWDeconv_data(framework::Tensor* filter_tensor,
                          framework::Tensor* ofm_tensor, float** bs_ptr,
                          int group, int sub_conv_n);

template <typename Dtype>
void savefile(std::string filename, void* buffer, int dataSize, Dtype tmp) {
  float data;
  std::ofstream out(filename.c_str());
  for (int i = 0; i < dataSize; ++i) {
    data = (((Dtype*)buffer)[i]);  // NOLINT
    out << data << std::endl;
  }
  out.close();
  return;
}

}  // namespace fpga
}  // namespace paddle_mobile
