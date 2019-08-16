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

namespace paddle_mobile {
namespace zynqmp {
namespace image {

void convert_to_hwc(float** data_in, int channel, int height, int width);
void align_element_conv(float** data_in, int height, int cw);
void format_image(float** data_in, int channel, int height, int width);

// Concat featuremaps along channel direction
void concat_images(int16_t** images_in, float** scales_in, void* image_out,
                   float* scale_out, int image_num, uint32_t* channel_num,
                   int height, int width);

// Split featuremap along channel direction
void split_image(int16_t* image_in, const float* scale_in, void** images_out,
                 float** scales_out, int image_num,
                 const uint32_t* channel_nums, int height, int width);
}  // namespace image
}  // namespace zynqmp
}  // namespace paddle_mobile
