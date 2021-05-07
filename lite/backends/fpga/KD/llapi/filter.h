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

#include <cstdint>
#include <cstdlib>
#include <cwchar>

#include <vector>

namespace paddle {
namespace zynqmp {
namespace filter {

void set_filter_capacity(uint32_t cap);
void set_colunm(uint32_t column);
int get_filter_num_alignment();
int calc_division_capacity(int chw);
int calc_split_num(int num, int division_capacity);
int calc_division_number(int num, int group_num, int division_capacity);
int calc_num_per_div(int num, int group_num, int division_capacity);
int calc_pack_num(int num_per_group, int group, int division_capacity);

float find_max(float* data_in, int data_size);
int8_t* format_filter(float* data_in,
                      int& mem_size,  // NOLINT
                      int num,
                      int channel,
                      int height,
                      int width,
                      int group_num,
                      float max,
                      std::vector<float>& filter_max);  // NOLINT

void convert_to_hwn(int16_t** data_in, int num, int height, int width);
size_t align_element_n(int16_t** data_in, int num, int height, int width);
void quantize_to_fp16(
    float** data_in, int num, int height, int width, float* scale_ptr);
int16_t* quantize_to_int16(float* data_in,
                           int num,
                           int height,
                           int width,
                           float* scale_ptr,
                           float quant_scale);
size_t format_dwconv_filter(
    float** data_in, int num, int height, int width, float* scale_ptr);
int16_t* format_dwconv_filter(float* data_in,
                              int num,
                              int height,
                              int width,
                              float* scale_ptr,
                              int& mem_size,  // NOLINT
                              float quant_scale);

}  // namespace filter
}  // namespace zynqmp
}  // namespace paddle
