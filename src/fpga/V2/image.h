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

#include <memory.h>
#include <algorithm>
#include <cstdint>
#include "fpga/common/fpga_common.h"
namespace paddle_mobile {
namespace fpga {
namespace image {

void convert_to_hwc(float** data_in, int channel, int height, int width,
                    int num = 1);
void convert_to_chw(float** data_in, int channel, int height, int width,
                    int num = 1);
template <typename Dtype>
void align_element_conv(Dtype** data_in, int height, int cw);
template <typename Dtype>
void align_element_conv(Dtype** data_in, int height, int cw) {
  int h = 0;
  int align_cw = align_to_x(cw, IMAGE_ALIGNMENT);

  Dtype* data_tmp =
      (Dtype*)fpga_malloc(height * align_cw * sizeof(Dtype));  // NOLINT

  memset(data_tmp, 0, height * align_cw * sizeof(Dtype));

  for (h = 0; h < height; h++) {
    memcpy((void*)(data_tmp + h * align_cw),  // NOLINT
           (void*)(*data_in + h * cw),        // NOLINT
           cw * sizeof(Dtype));
  }

  *data_in = data_tmp;
}
template <typename T>
void format_image(T** data_in, int channel, int height, int width) {
  int cw = channel * width;
  int align_cw = align_to_x(cw, IMAGE_ALIGNMENT);
  if (align_cw != cw) {
    T* hwc_temp = *data_in;
    align_element_conv(data_in, height, channel * width);
    fpga_free(hwc_temp);
  }
  fpga_flush(*data_in,
             align_to_x(channel * width, IMAGE_ALIGNMENT) * height * sizeof(T));
}
// Concat featuremaps along channel direction
void concat_images(int8_t** images_in, float** scales_in, void* image_out,
                   float* scale_out, int image_num, uint32_t* channel_num,
                   int height, int width);

// Split featuremap along channel direction
void split_image(int8_t* image_in, void** images_out, int image_num,
                 const uint32_t* channel_nums, int height, int width);
}  // namespace image
}  // namespace fpga
}  // namespace paddle_mobile
