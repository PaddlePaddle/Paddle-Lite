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

#define FILTER_PARALLELISM 1024
namespace paddle_mobile {
namespace fpga {
namespace filter {

int calc_channel_parallelism(int channel);
int calc_aligned_channel(int channel);
int calc_num_parallelism(int channel);
int calc_aligned_num(int num, int channel);
int calc_aligned_total_pixel_num(int num, int channel, int height, int width);
void convert_to_hwc(float** data_in, int num, int channel, int height,
                    int width);
void format_filter(float** data_in, int num, int channel, int height, int width,
                   int group_num, float max);
void convert_fc_filter(float** data_in, int num, int chw);
void format_fc_filter(float** data_in, int num, int channel, int height,
                      int width, int group_num, float max);
float find_max(float* data_in, int data_size);
}  // namespace filter
}  // namespace fpga
}  // namespace paddle_mobile
