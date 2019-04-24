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

namespace paddle_mobile {
namespace fpga {
namespace image {

void convert_to_hwc(float **data_in, int channel, int height, int width,
                    int num) {
  float *data_tmp = reinterpret_cast<float *>(
      fpga_malloc(num * channel * height * width * sizeof(float)));
  int64_t amount_per_row = width * channel;
  for (int n = 0; n < num; n++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        int64_t offset_height = h * amount_per_row;
        for (int w = 0; w < width; w++) {
          *(data_tmp + n * channel * height * width + offset_height +
            w * channel + c) = *((*data_in)++);
        }
      }
    }
  }
  *data_in = data_tmp;
}

void convert_to_chw(float **data_in, int channel, int height, int width,
                    int num) {
  float *data_tmp =
      (float *)fpga_malloc(channel * height * width * sizeof(float));  // NOLINT
  int64_t amount_per_side = width * height;
  for (int n = 0; n < num; n++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channel; c++) {
          *(data_tmp + n * height * width * channel + c * amount_per_side +
            width * h + w) = *((*data_in)++);
        }
      }
    }
  }
  *data_in = data_tmp;
}

void concat_images(int8_t **images_in, float **scales_in, void *image_out,
                   float *scale_out, int image_num, uint32_t *channel_num,
                   int height, int width) {
  int i = 0;
  int j = 0;
  int k = 0;
  int each_out_line_channel = 0;
  int align_each_out_area_cw = 0;
  int align_each_in_area_cw = 0;
  int align_each_out_area_cw_differ = 0;
  int tmp_channel = 0;
  float Ck = 0.0f;
  float So = scale_out[0];
  auto images_in_tmp =
      (int8_t **)fpga::fpga_malloc(image_num * sizeof(int8_t *));  // NOLINT
  for (i = 0; i < image_num; i++) {
    images_in_tmp[i] = reinterpret_cast<int8_t *>(fpga::fpga_malloc(
        height * align_to_x(channel_num[i] * width, IMAGE_ALIGNMENT) *
        sizeof(int8_t)));
  }
  for (i = 0; i < image_num; i++) {
    each_out_line_channel += channel_num[i];
    float Si_k = scales_in[i][0];
    Ck = Si_k / So;
    fpga_invalidate(images_in[i],
                    height *
                        align_to_x(channel_num[i] * width, IMAGE_ALIGNMENT) *
                        sizeof(int8_t));
    for (j = 0;
         j < height * align_to_x(channel_num[i] * width, IMAGE_ALIGNMENT);
         j++) {
      images_in_tmp[i][j] = (int8_t)(images_in[i][j] * Ck + 0.5);
    }
  }
  align_each_out_area_cw =
      align_to_x(each_out_line_channel * width, IMAGE_ALIGNMENT);
  align_each_out_area_cw_differ =
      align_each_out_area_cw - each_out_line_channel * width;

  for (k = 0; k < height; k++) {
    for (j = 0; j < width; j++) {
      for (i = 0; i < image_num; i++) {
        align_each_in_area_cw =
            align_to_x(channel_num[i] * width, IMAGE_ALIGNMENT);
        memcpy(
            (int16_t *)image_out + tmp_channel +  // NOLINT
                k * align_each_out_area_cw_differ,
            images_in_tmp[i] + j * channel_num[i] + k * align_each_in_area_cw,
            channel_num[i] * sizeof(int8_t));

        tmp_channel += channel_num[i];
      }
    }
  }
  fpga_flush(image_out, height * align_each_out_area_cw * sizeof(int8_t));
}

void split_image(int8_t *image_in, const float *scale_in, void **images_out,
                 float **scales_out, int image_num,
                 const uint32_t *channel_nums, int height, int width) {
  int total_channel = 0;
  for (int i = 0; i < image_num; i++) {
    total_channel += channel_nums[i];
  }
  int element_num = height * align_to_x(width * total_channel, IMAGE_ALIGNMENT);
  fpga_invalidate(image_in, element_num * sizeof(int8_t));
  int src_offset = 0, des_offset = 0;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      src_offset = h * align_to_x(total_channel * width, IMAGE_ALIGNMENT) +
                   w * total_channel;
      for (int i = 0; i < image_num; i++) {
        des_offset = h * align_to_x(channel_nums[i] * width, IMAGE_ALIGNMENT) +
                     w * channel_nums[i];
        memcpy(reinterpret_cast<int8_t *>(images_out[i]) + des_offset,
               image_in + src_offset, channel_nums[i] * sizeof(int8_t));
        src_offset += channel_nums[i];
      }
    }
  }

  for (int i = 0; i < image_num; i++) {
    element_num = height * align_to_x(width * channel_nums[i], IMAGE_ALIGNMENT);
    fpga_flush(images_out[i], element_num * sizeof(int8_t));
  }
}

}  // namespace image
}  // namespace fpga
}  // namespace paddle_mobile
