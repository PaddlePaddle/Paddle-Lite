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

#include "fpga/V2/filter.h"
#include <memory.h>
#include <algorithm>
#include "fpga/common/fpga_common.h"
namespace paddle_mobile {
namespace fpga {
namespace filter {

int calc_channel_parallelism(int channel) {
  if (channel <= 16) {
    return 16;
  } else if (channel <= 32) {
    return 32;
  } else if (channel <= 64) {
    return 64;
  } else {
    return 128;
  }
}
int calc_aligned_channel(int channel) {
  return align_to_x(channel, calc_channel_parallelism(channel));
}

int calc_num_parallelism(int channel) {
  return FILTER_PARALLELISM / calc_channel_parallelism(channel);
}

int calc_aligned_num(int num, int channel) {
  return align_to_x(num, calc_num_parallelism(channel));
}

int calc_aligned_total_pixel_num(int num, int channel, int height, int width) {
  int aligned_channel = calc_aligned_channel(channel);
  int aligned_filter_num = calc_aligned_num(num, channel);
  return aligned_filter_num * aligned_channel * height * width;
}

void convert_to_hwc(float **data_in, int num, int channel, int height,
                    int width) {
  float *tmp = *data_in;
  int chw = channel * height * width;
  float *data_tmp = (float *)fpga_malloc(chw * num * sizeof(float));  // NOLINT
  for (int n = 0; n < num; n++) {
    int64_t amount_per_row = width * channel;
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        int64_t offset_height = h * amount_per_row;
        for (int w = 0; w < width; w++) {
          *(data_tmp + n * chw + offset_height + w * channel + c) =
              *((*data_in)++);
        }
      }
    }
  }
  *data_in = data_tmp;
  fpga_free(tmp);
}

void align_filter(float **data_in, int num, int channel, int height,
                  int width) {
  int aligned_channel = calc_aligned_channel(channel);
  int hw = height * width;
  int pixel_num = calc_aligned_total_pixel_num(num, channel, height, width);
  float *new_data = (float *)fpga_malloc(pixel_num * sizeof(float));  // NOLINT
  float *temp = *data_in;
  memset(new_data, 0, pixel_num * sizeof(float));
  for (int i = 0; i < num; i++) {
    for (int j = 0; j < hw; j++) {
      memcpy(new_data + i * aligned_channel * hw + j * aligned_channel,
             temp + i * channel * hw + j * channel, channel * sizeof(float));
    }
  }
  *data_in = new_data;
  fpga_free(temp);
}
void convert_to_fp16(float **data_in, int data_size) {
  float *tmp = *data_in;
  // half_float::half *tmp_data = (half_float::half *)fpga_malloc(data_size *
  // sizeof(half_float::half));
  int16_t *tmp_data =
      (int16_t *)fpga_malloc(data_size * sizeof(int16_t));  // NOLINT
  for (int i = 0; i < data_size; i++) {
    // tmp_data[i] = (half_float::half)((*data_in)[i]);
    tmp_data[i] = fp32_2_fp16((*data_in)[i]);
  }
  *data_in = (float *)tmp_data;  // NOLINT
  fpga_free(tmp);
}
void format_filter(float **data_in, int num, int channel, int height, int width,
                   int group_num, float max) {
  convert_to_hwc(data_in, num, channel, height, width);
  align_filter(data_in, num, channel, height, width);
  int pixel_num = calc_aligned_total_pixel_num(num, channel, height, width);
  convert_to_fp16(data_in, pixel_num);
  fpga_flush(*data_in, pixel_num * sizeof(float));
}

void convert_fc_filter(float **data_in, int num, int chw) {
  float *tmp = *data_in;
  float *data_tmp = (float *)fpga_malloc(chw * num * sizeof(float));  // NOLINT
  for (int n = 0; n < num; n++) {
    for (int c = 0; c < chw; c++) {
      data_tmp[n * chw + c] = (*data_in)[num * c + n];
    }
  }
  *data_in = data_tmp;
  fpga_free(tmp);
}

void format_fc_filter(float **data_in, int num, int channel, int height,
                      int width, int group_num, float max) {
  int chw = channel * height * width;
  convert_fc_filter(data_in, num, chw);
  align_filter(data_in, num, channel, height, width);
  int pixel_num = calc_aligned_total_pixel_num(num, channel, height, width);
  convert_to_fp16(data_in, pixel_num);
  fpga_flush(*data_in, pixel_num * sizeof(float));
}

float find_max(float *data_in, int data_size) {
  float max = 0.0;
  for (int i = 0; i < data_size; ++i) {
    float value = data_in[i];
    float abs = value > 0 ? value : -value;
    max = std::max(max, abs);
  }
  return max;
}

signed char float_to_int8(float fdata) {
  if (fdata < 0.0) {
    fdata -= 0.5;
  } else {
    fdata += 0.5;
  }
  return (signed char)fdata;
}

void quantize(float **data_in, int data_size, float max) {
  float *tmp = *data_in;
  float fix_range = 127;
  float scale = fix_range / max;

  signed char *tmp_data = (signed char *)fpga_malloc(data_size * sizeof(char));
  for (int i = 0; i < data_size; i++) {
    tmp_data[i] = float_to_int8(
        (*data_in)[i] * scale);  // (signed char)((*data_in)[i] * scale);
  }
  *data_in = (float *)tmp_data;  // NOLINT
  fpga_free(tmp);
}

}  // namespace filter
}  // namespace fpga
}  // namespace paddle_mobile
