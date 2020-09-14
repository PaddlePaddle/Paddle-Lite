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

#include "lite/backends/fpga/KD/llapi/filter.h"
#include <memory.h>
#include <algorithm>
#include <fstream>
#include <string>
#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/backends/fpga/KD/llapi/zynqmp_api.h"

namespace paddle {
namespace zynqmp {
namespace filter {

static int FILTER_SIZE = 2048;
static int COLUMN = 4;

void saveToFile(std::string name, void* data_in, int size) {
  std::ofstream ofs;
  ofs.open(name);

  int8_t* data = reinterpret_cast<int8_t*>(data_in);
  for (int i = 0; i < size; i++) {
    float value = data[i];
    ofs << value << std::endl;
  }
  ofs.close();
}

void saveFloatToFile(std::string name, float* data_in, int size) {
  std::ofstream ofs;
  ofs.open(name);

  for (int i = 0; i < size; i++) {
    float value = data_in[i];
    ofs << value << std::endl;
  }
  ofs.close();
}

void set_filter_capacity(uint32_t cap) { FILTER_SIZE = cap; }

void set_colunm(uint32_t column) { COLUMN = column; }

// replace zynqmp_api.h  #define FILTER_NUM_ALIGNMENT
int get_filter_num_alignment() { return COLUMN * 4; }

int calc_division_capacity(int chw) {
  int filter_num_alignment = get_filter_num_alignment();
  int n = FILTER_SIZE / ((chw + 15) / 16) * filter_num_alignment;
  return n < FILTER_SIZE ? n : FILTER_SIZE;
}

int calc_split_num(int num, int division_capacity) {
  return (num + division_capacity - 1) / division_capacity;
}

int calc_division_number(int num, int group_num, int division_capacity) {
  int split_num = calc_split_num(num, division_capacity);
  return group_num * split_num;
}

int calc_num_per_div(int num, int group_num, int division_capacity) {
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

int calc_pack_num(int num_per_group, int group, int division_capacity) {
  auto n = 1;
  if (num_per_group * group % division_capacity == 0) {
    n = num_per_group * group / division_capacity;
    return n;
  }

  while ((num_per_group * (group + n - 1) / n) > division_capacity) {
    n++;
  }
  return (n);
}

void convert_to_hwc(int8_t* chw_data,
                    int8_t* hwc_data,
                    int num,
                    int channel,
                    int height,
                    int width) {
  int chw = channel * height * width;
  int wc = width * channel;
  int index = 0;
  for (int n = 0; n < num; n++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          hwc_data[n * chw + h * wc + w * channel + c] = chw_data[index];
          index++;
        }
      }
    }
  }
}

float find_max(float* data_in, int data_size) {
  float max = 0.0;
  for (int i = 0; i < data_size; ++i) {
    float value = data_in[i];
    float abs = value > 0 ? value : -value;
    max = std::max(max, abs);
  }
  return max;
}

int8_t float_to_int8(float fdata) {
  if (fdata < 0.0) {
    fdata -= 0.5;
  } else {
    fdata += 0.5;
  }
  return (int8_t)fdata;
}

void quantize(float* src, int8_t* dst, int len, float max) {
  float fix_range = 127;
  float scale = fix_range / max;
  for (size_t i = 0; i < len; i++) {
    dst[i] = float_to_int8(src[i] * scale);
  }
}

bool should_align_chw(int chw) {
  int align_chw = align_to_x(chw, FILTER_ELEMENT_ALIGNMENT);
  return align_chw != chw;
}

void align_chw(int8_t* src, int8_t* dst, int num, int chw) {
  int aligned_chw = align_to_x(chw, FILTER_ELEMENT_ALIGNMENT);
  memset(dst, 0, num * aligned_chw);
  for (int j = 0; j < num; j++) {
    memcpy((dst + j * aligned_chw), (src + j * chw), chw);
  }
}

void align_num(int8_t* src,
               int8_t* dst,
               int num_per_div_before_alignment,
               int num,
               int align_chw) {
  int filter_num_alignment = get_filter_num_alignment();
  int num_per_div_after_alignment =
      align_to_x(num_per_div_before_alignment, filter_num_alignment);

  int div_num =
      (num + num_per_div_before_alignment - 1) / num_per_div_before_alignment;
  int num_element = div_num * num_per_div_after_alignment * align_chw;

  memset(dst, 0, num_element * sizeof(int8_t));
  int i = 0;
  for (i = 0; i < div_num - 1; i++) {
    memcpy(dst + num_per_div_after_alignment * align_chw * i,
           src + num_per_div_before_alignment * align_chw * i,
           num_per_div_before_alignment * align_chw);
  }

  memcpy(dst + num_per_div_after_alignment * align_chw * i,
         src + num_per_div_before_alignment * align_chw * i,
         (num - (div_num - 1) * num_per_div_before_alignment) * align_chw);
}

void reorder(int8_t* src, int8_t* dst, int num_after_alignment, int chw) {
  int index = 0;
  int new_index = 0;
  int filter_num_alignment = get_filter_num_alignment();
  int chw_align = align_to_x(chw, FILTER_ELEMENT_ALIGNMENT);
  for (index = 0; index < num_after_alignment; index++) {
    new_index = index / filter_num_alignment * filter_num_alignment +
                (index % (filter_num_alignment / 2) / 4 * 8) +
                (index % (filter_num_alignment / 2) % 4) +
                (index / (filter_num_alignment / 2) % 2 * 4);
    memcpy((dst + index * chw_align), (src + new_index * chw_align), chw_align);
  }
}

void interleave(int8_t* src, int8_t* dst, int num_after_alignment, int chw) {
  int interleave_per_num = 16;
  int chw_align = align_to_x(chw, FILTER_ELEMENT_ALIGNMENT);
  int interleave_num = chw_align * 2 / interleave_per_num;
  for (int i = 0; i < num_after_alignment; i += 2) {
    for (int j = 0, k = 0; j < interleave_num; j += 2, k++) {
      memcpy(dst + i * chw_align + interleave_per_num * j,
             src + i * chw_align + interleave_per_num * k,
             interleave_per_num);
      memcpy(dst + i * chw_align + interleave_per_num * (j + 1),
             src + (i + 1) * chw_align + interleave_per_num * k,
             interleave_per_num);
    }
  }
}

int8_t* format_filter(float* data_in,
                      int& mem_size_a,  // NOLINT
                      int num,
                      int channel,
                      int height,
                      int width,
                      int group_num,
                      float max,
                      std::vector<float>& filter_max) {  // NOLINT
  int data_size = channel * height * width * num;
  int chw = channel * height * width;

  int division_capacity = calc_division_capacity(chw);
  int filter_num_alignment = get_filter_num_alignment();
  int num_per_div_before_alignment =
      calc_num_per_div(num, group_num, division_capacity);
  int num_per_div_after_alignment =
      align_to_x(num_per_div_before_alignment, filter_num_alignment);
  int div_num =
      (num + num_per_div_before_alignment - 1) / num_per_div_before_alignment;
  int residual = num % num_per_div_before_alignment;
  int num_after_alignment = num_per_div_after_alignment *
                                ((residual == 0) ? div_num : (div_num - 1)) +
                            align_to_x(residual, filter_num_alignment);

  int8_t* quantized_data =
      reinterpret_cast<int8_t*>(fpga_malloc(data_size * sizeof(int8_t)));

  for (int n = 0; n < num; n++) {
    float* filter_start = data_in + n * chw;
    float f_max = find_max(filter_start, chw);
    int8_t* quantized_start = quantized_data + n * chw;
    quantize(filter_start, quantized_start, chw, f_max);
    filter_max.push_back(f_max);
  }

  int8_t* hwc_data =
      reinterpret_cast<int8_t*>(fpga_malloc(data_size * sizeof(int8_t)));
  convert_to_hwc(quantized_data, hwc_data, num, channel, height, width);
  fpga_free(quantized_data);

  int8_t* temp_data = hwc_data;  // NOLINT
  int chw_aligned = align_to_x(chw, FILTER_ELEMENT_ALIGNMENT);
  if (should_align_chw(chw)) {
    int8_t* hwc_aligned_data = reinterpret_cast<int8_t*>(
        fpga_malloc(num * chw_aligned * sizeof(int8_t)));
    align_chw(hwc_data, hwc_aligned_data, num, chw);

    temp_data = hwc_aligned_data;
    fpga_free(hwc_data);
  }
  if (num_after_alignment != num) {
    int filter_num_alignment = get_filter_num_alignment();
    int num_per_div_after_alignment =
        align_to_x(num_per_div_before_alignment, filter_num_alignment);

    int num_element = div_num * num_per_div_after_alignment * chw_aligned;
    int8_t* num_aligned_data =
        reinterpret_cast<int8_t*>(fpga_malloc(num_element * sizeof(int8_t)));
    align_num(temp_data,
              num_aligned_data,
              num_per_div_before_alignment,
              num,
              chw_aligned);

    fpga_free(temp_data);
    temp_data = num_aligned_data;
  }
  int8_t* aligned_data =
      reinterpret_cast<int8_t*>(fpga_malloc(num_after_alignment * chw_aligned));
  reorder(temp_data, aligned_data, num_after_alignment, chw);
  fpga_free(temp_data);
  int8_t* interleaved_data =
      reinterpret_cast<int8_t*>(fpga_malloc(num_after_alignment * chw_aligned));
  interleave(aligned_data, interleaved_data, num_after_alignment, chw);
  fpga_free(aligned_data);
  fpga_flush(interleaved_data,
             align_to_x(chw, FILTER_ELEMENT_ALIGNMENT) * num_after_alignment *
                 sizeof(char));
  mem_size_a = num_after_alignment * chw_aligned;
  return interleaved_data;
}

void convert_to_hwn(int16_t** data_in, int num, int height, int width) {
  int16_t* tmp = *data_in;
  int16_t* data_tmp =
      (int16_t*)fpga_malloc(height * width * num * sizeof(int16_t));  // NOLINT
  for (int n = 0; n < num; n++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        *(data_tmp + h * width * num + w * num + n) = *((*data_in)++);
      }
    }
  }
  *data_in = data_tmp;
  fpga_free(tmp);
}

size_t align_element_n(int16_t** data_in, int num, int height, int width) {
  int unalign_n = num;
  int align_n = align_to_x(num, FILTER_ELEMENT_ALIGNMENT);
  int num_element = height * width * align_n;
  if (unalign_n != align_n) {
    int16_t* tmp = *data_in;

    int num_element = height * width * align_n;
    int16_t* data_tmp =
        (int16_t*)fpga_malloc(num_element * sizeof(int16_t));  // NOLINT

    memset(data_tmp, 0, num_element * sizeof(int16_t));
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int offset_unalign = h * width * unalign_n + w * unalign_n;
        int offset_align = h * width * align_n + w * align_n;
        for (int n = 0; n < unalign_n; n++) {
          data_tmp[offset_align + n] = *((*data_in) + offset_unalign + n);
        }
      }
    }
    *data_in = data_tmp;
    fpga_free(tmp);
  }
  return num_element * sizeof(int16_t);
}

void to_fp16(float* src,
             float16* dst,
             int num,
             int height,
             int width,
             float* scale_ptr) {
  int size = num * height * width;
  for (int n = 0; n < num; n++) {
    float scale_val = scale_ptr[n];
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int index = n * height * width + h * width + w;
        float value = src[index] * scale_val;
        dst[index] = float_to_half(value);
      }
    }
  }
  fpga_flush(dst, size * sizeof(int16_t));
}

void quantize_to_fp16(
    float** data_in, int num, int height, int width, float* scale_ptr) {
  float* tmp = *data_in;
  int size = num * height * width;

  float16* tmp_data = (float16*)fpga_malloc(size * sizeof(float16));  // NOLINT
  for (int n = 0; n < num; n++) {
    float scale_val = scale_ptr[n];
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int index = n * height * width + h * width + w;
        float value = tmp[index] * scale_val;
        tmp_data[index] = float_to_half(value);
      }
    }
  }
  fpga_flush(tmp_data, size * sizeof(int16_t));
  *data_in = (float*)tmp_data;  // NOLINT
  fpga_free(tmp);
}
size_t format_dwconv_filter(
    float** data_in, int num, int height, int width, float* scale_ptr) {
  quantize_to_fp16(data_in, num, height, width, scale_ptr);
  int16_t** quantize_data = reinterpret_cast<int16_t**>(data_in);
  convert_to_hwn(quantize_data, num, height, width);
  size_t size = align_element_n(quantize_data, num, height, width);
  fpga_flush(*quantize_data,
             align_to_x(num, FILTER_ELEMENT_ALIGNMENT) * height * width *
                 sizeof(int16_t));
  return size;
}

}  // namespace filter
}  // namespace zynqmp
}  // namespace paddle
