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

#include "fpga/V2/image.h"
#include <memory.h>
#include <algorithm>
#include "fpga/V2/api.h"

namespace paddle_mobile {
namespace fpga {
namespace image {

void convert_to_hwc(float **data_in, int channel, int height, int width) {
  float *tmp = *data_in;
  float *data_tmp =
      (float *)fpga_malloc(channel * height * width * sizeof(float));  // NOLINT
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
void align_image(float **data_in, int channel, int height, int width,
                 int aligned_channel) {
  if (channel == aligned_channel) return;
  float *tmp = *data_in;
  float *new_data =
      (float *)fpga_malloc(aligned_channel * height * width *  // NOLINT
                           sizeof(float));                     // NOLINT
  memset(new_data, 0, aligned_channel * height * width * sizeof(float));

  for (int i = 0; i < height * width; i++) {
    memcpy(new_data + i * aligned_channel, tmp + i * channel,
           channel * sizeof(float));
  }
  *data_in = new_data;
  fpga_free(tmp);
}

void format_image(float **data_in, int channel, int height, int width,
                  int aligned_channel) {
  convert_to_hwc(data_in, channel, height, width);
  align_image(data_in, channel, height, width, aligned_channel);
  fpga_flush(*data_in, aligned_channel * height * width * sizeof(float));
}

void concat_images(int16_t **images_in, float **scales_in, void *image_out,
                   float *scale_out, int image_num, const uint32_t *channel_num,
                   int height, int width, const uint32_t *aligned_channel_num,
                   int out_channel) {
  int hw = height * width;
  scale_out[0] = 0.0;
  scale_out[1] = 0.0;
  for (int i = 0; i < image_num; i++) {
    scale_out[0] = std::max(*scale_out, scales_in[i][0]);
    fpga_invalidate(images_in[i],
                    height * width * aligned_channel_num[i] * sizeof(int16_t));
  }
  scale_out[1] = 1 / scale_out[0];

  for (int j = 0; j < hw; j++) {
    int tmp_channel_sum = 0;
    for (int i = 0; i < image_num; i++) {
      memcpy(
          (int16_t *)image_out + j * out_channel + tmp_channel_sum,  // NOLINT
          images_in[i] + j * aligned_channel_num[i],
          channel_num[i] * sizeof(int16_t));

      tmp_channel_sum += channel_num[i];
    }
  }
  fpga_flush(image_out, hw * out_channel * sizeof(int16_t));
}

}  // namespace image
}  // namespace fpga
}  // namespace paddle_mobile
