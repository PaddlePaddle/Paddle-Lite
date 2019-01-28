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

namespace paddle_mobile {
namespace fpga {
namespace deconv_filter {

void deconv_inverse_filter(float** data_in, int num, int channel, int width,
                           int height);
int deconv_calc_sub_pad(int filter_axis, int pad, int stride);
int deconv_get_sub_filter_axis(int filter_axis, int stride);
int deconv_get_sub_out_axis(int image_axis, int sub_pad, int sub_filter_axis);
int deconv_get_omit(int stride, int filter_width, int pad);

template <typename T>
void deconv_get_sub_filter(T** data_in, int height, int width, int sub_conv_n,
                           int kernel_num, int channel);
void deconv_format_filter(float** data_in, int num, int channel, int height,
                          int width, int group_num, float max, int stride);
void deconv_NC_convert(float** filter_in, int kernel_num, int channels, int hw);
void DWDconv_format_filter(float** data_in, int num, int channel, int height,
                           int width, float* scale_ptr, int stride);

}  // namespace deconv_filter
}  // namespace fpga
}  // namespace paddle_mobile
