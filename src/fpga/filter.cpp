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

#include "fpga/filter.h"
#include <memory.h>
#include <algorithm>
#include "fpga/api.h"

namespace paddle_mobile {
namespace fpga {
namespace filter {

int calc_division_capacity(int chw) {
  int n = 2048 / ((chw + 15) / 16) * 32;
  return n < 2048 ? n : 2048;
}

int calc_split_num(int num, int division_capacity) {
  return (num + division_capacity - 1) / division_capacity;
}

int calc_division_number(int num, int group_num, int division_capacity) {
  PADDLE_MOBILE_ENFORCE(num % group_num == 0,
                        "Filter number should be divisible by group number");
  int split_num = calc_split_num(num, division_capacity);
  PADDLE_MOBILE_ENFORCE(group_num == 1 || split_num == 1,
                        "Split number or group number should be 1");
  return group_num * split_num;
}

int calc_num_per_div(int num, int group_num, int division_capacity) {
  PADDLE_MOBILE_ENFORCE(num % group_num == 0,
                        "Filter number should be divisible by group number");
  int split_num = calc_split_num(num, division_capacity);
  PADDLE_MOBILE_ENFORCE(group_num == 1 || split_num == 1,
                        "Split number or group number should be 1");
  if (group_num == 1) {
    if (num > division_capacity) {
      return division_capacity;
    } else {
      return num;
    }
  } else {
    return (num + group_num - 1) / group_num;
  }
}

void convert_to_hwc(char **data_in, int num, int channel, int height,
                    int width) {
  char *tmp = *data_in;
  int chw = channel * height * width;
  char *data_tmp = (char *)fpga_malloc(chw * num * sizeof(char));  // NOLINT
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

void align_element(char **data_in, int num, int chw) {
  int i = 0;
  int j = 0;
  int align_chw = align_to_x(chw, FILTER_ELEMENT_ALIGNMENT);
  if (align_chw != chw) {
    char *tmp = *data_in;
    char *data_tmp =
        (char *)fpga_malloc(num * align_chw * sizeof(char));  // NOLINT

    memset(data_tmp, 0, num * align_chw);
    for (j = 0; j < num; j++) {
      memcpy(data_tmp + j * align_chw, (*data_in) + j * chw, chw);
    }
    *data_in = data_tmp;
    fpga_free(tmp);
  }
}

void align_num(char **data_in, int num_per_div_before_alignment, int num,
               int chw) {
  int i = 0;
  int align_chw = align_to_x(chw, FILTER_ELEMENT_ALIGNMENT);
  int num_per_div_after_alignment =
      align_to_x(num_per_div_before_alignment, FILTER_NUM_ALIGNMENT);
  if (num_per_div_after_alignment != num_per_div_before_alignment) {
    char *tmp = *data_in;
    int div_num =
        (num + num_per_div_before_alignment - 1) / num_per_div_before_alignment;
    int num_element = div_num * num_per_div_after_alignment * align_chw;
    char *data_tmp = (char *)fpga_malloc(num_element * sizeof(char));  // NOLINT

    memset(data_tmp, 0, num_element * sizeof(char));

    for (i = 0; i < div_num; i++) {
      memcpy(data_tmp + num_per_div_after_alignment * align_chw * i,
             *data_in + num_per_div_before_alignment * align_chw * i,
             num_per_div_before_alignment * align_chw);
    }

    *data_in = data_tmp;
    fpga_free(tmp);
  }
}

void reorder(char **data_in, int num_after_alignment, int chw) {
  int index = 0;
  int new_index;

  int chw_align = align_to_x(chw, FILTER_ELEMENT_ALIGNMENT);

  char *data_tmp =
      (char *)fpga_malloc(chw_align * num_after_alignment *  // NOLINT
                          sizeof(char));
  char *tmp = *data_in;
  for (index = 0; index < num_after_alignment; index++) {
    new_index = index / 32 * 32 + (index % 16 / 4 * 8) + (index % 16 % 4) +
                (index / 16 % 2 * 4);
    memcpy(data_tmp + index * chw_align, *data_in + new_index * chw_align,
           chw_align);
  }
  *data_in = data_tmp;
  fpga_free(tmp);
}

void interleave(char **data_in, int num_after_alignment, int chw) {
  int i = 0;
  int j = 0;
  int k = 0;
  int interleave_per_num = 16;

  int chw_align = align_to_x(chw, FILTER_ELEMENT_ALIGNMENT);
  char *data_tmp =
      (char *)fpga_malloc(chw_align * num_after_alignment *  // NOLINT
                          sizeof(char));
  char *tmp = *data_in;
  int interleave_num = chw_align * 2 / interleave_per_num;
  for (i = 0; i < num_after_alignment; i += 2) {
    for (j = 0, k = 0; j < interleave_num; j += 2, k++) {
      memcpy(data_tmp + i * chw_align + interleave_per_num * j,
             *data_in + i * chw_align + interleave_per_num * k,
             interleave_per_num);
      memcpy(data_tmp + i * chw_align + interleave_per_num * (j + 1),
             *data_in + (i + 1) * chw_align + interleave_per_num * k,
             interleave_per_num);
    }
  }
  *data_in = data_tmp;
  fpga_free(tmp);
}

void format_filter(float **data_in, int num, int channel, int height, int width,
                   int group_num, float max) {
  int data_size = channel * height * width * num;
  int chw = channel * height * width;

  int division_capacity = calc_division_capacity(chw);
  int num_per_div_before_alignment =
      calc_num_per_div(num, group_num, division_capacity);
  int num_per_div_after_alignment =
      align_to_x(num_per_div_before_alignment, FILTER_NUM_ALIGNMENT);
  int div_num =
      (num + num_per_div_before_alignment - 1) / num_per_div_before_alignment;
  int residual = num % num_per_div_before_alignment;
  int num_after_alignment = num_per_div_after_alignment *
                                ((residual == 0) ? div_num : (div_num - 1)) +
                            align_to_x(residual, FILTER_NUM_ALIGNMENT);
  quantize(data_in, data_size, max);
  char **quantize_data = (char **)data_in;  // NOLINT
  convert_to_hwc(quantize_data, num, channel, height, width);
  align_element(quantize_data, num, chw);
  align_num(quantize_data, num_per_div_before_alignment, num, chw);
  reorder(quantize_data, num_after_alignment, chw);
  interleave(quantize_data, num_after_alignment, chw);
  fpga_flush(*quantize_data, align_to_x(chw, FILTER_ELEMENT_ALIGNMENT) *
                                 num_after_alignment * sizeof(char));
}

void convert_fc_filter(char **data_in, int num, int chw) {
  char *tmp = *data_in;
  char *data_tmp = (char *)fpga_malloc(chw * num * sizeof(char));  // NOLINT
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
  int data_size = channel * height * width * num;
  int chw = channel * height * width;

  int division_capacity = calc_division_capacity(chw);
  int num_per_div_before_alignment =
      calc_num_per_div(num, group_num, division_capacity);
  int num_per_div_after_alignment =
      align_to_x(num_per_div_before_alignment, FILTER_NUM_ALIGNMENT);
  int div_num =
      (num + num_per_div_before_alignment - 1) / num_per_div_before_alignment;
  int num_after_alignment = num_per_div_after_alignment * div_num;

  quantize(data_in, data_size, max);

  char **quantize_data = (char **)data_in;  // NOLINT

  convert_fc_filter(quantize_data, num, chw);
  align_element(quantize_data, num, chw);
  align_num(quantize_data, num_per_div_before_alignment, num, chw);
  reorder(quantize_data, num_after_alignment, chw);
  interleave(quantize_data, num_after_alignment, chw);
  fpga_flush(*quantize_data, align_to_x(chw, FILTER_ELEMENT_ALIGNMENT) *
                                 num_after_alignment * sizeof(char));
}

}  // namespace filter
}  // namespace fpga
}  // namespace paddle_mobile
