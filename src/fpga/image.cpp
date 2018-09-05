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

#include "image.h"
#include <memory.h>
#include "api.h"

namespace paddle_mobile {
namespace fpga {
namespace image {

void convert_to_hwc(float **data_in, int channel, int height, int width) {
  float *tmp = *data_in;
  float *data_tmp =
      (float *)fpga_malloc(channel * height * width * sizeof(float));
  int64_t amount_per_row = width * channel;
  for (int c = 0; c < channel; c++) {
    for (int h = 0; h < height; h++) {
      int64_t offset_height = h * amount_per_row;
      for (int w = 0; w < width; w++) {
        *(data_tmp + offset_height + w * channel + c) = *((*data_in)++);
      }
    }
  }
  *data_in = data_tmp;
  fpga_free(tmp);
}

void align_element_conv(float **data_in, int height, int cw) {
  int i = 0;
  int h = 0;
  int align_cw = align_to_x(cw, IMAGE_ALIGNMENT);
  if (align_cw != cw) {
    float *tmp = *data_in;
    float *data_tmp = (float *)fpga_malloc(height * align_cw * sizeof(float));

    memset(data_tmp, 0, height * align_cw * sizeof(float));

    for (h = 0; h < height; h++) {
      memcpy((void *)(data_tmp + h * align_cw), (void *)(*data_in + h * cw),
             cw * sizeof(float));
    }

    *data_in = data_tmp;
    fpga_free(tmp);
  }
}

void format_image(float **data_in, int channel, int height, int width) {
  convert_to_hwc(data_in, channel, height, width);
  align_element_conv(data_in, height, channel * width);
}

}  // namespace image
}  // namespace fpga
}  // namespace paddle_mobile
