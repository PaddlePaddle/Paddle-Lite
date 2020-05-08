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

#include <memory.h>

#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/backends/fpga/KD/llapi/bias_scale.h"
#include "lite/backends/fpga/KD/llapi/zynqmp_api.h"

namespace paddle {
namespace zynqmp {
namespace bias_scale {

void align_element(float **data_in, int num_per_div_before_alignment, int num) {
  int copynum = 0;
  float *ptr_unaligned = *data_in;
  int div_num =
      (num + num_per_div_before_alignment - 1) / num_per_div_before_alignment;
  int num_per_div_after_alignment =
      align_to_x(num_per_div_before_alignment, BS_NUM_ALIGNMENT);
  int num_element =
      2 * div_num * num_per_div_after_alignment;  // including bias & scale
  float *ptr_aligned =
      (float *)fpga_malloc(num_element * sizeof(float));  // NOLINT

  memset(ptr_aligned, 0, num_element * sizeof(float));
  for (int i = 0; i < div_num; i++) {
    if (i == div_num - 1) {
      copynum = (num_per_div_after_alignment * div_num > num)
                    ? (num % num_per_div_after_alignment)
                    : (num_per_div_before_alignment);
    } else {
      copynum = num_per_div_before_alignment;
    }

    memcpy(ptr_aligned + i * num_per_div_after_alignment,
           ptr_unaligned + num_per_div_before_alignment * i,
           copynum * sizeof(float));
    memcpy(ptr_aligned + (div_num + i) * num_per_div_after_alignment,
           ptr_unaligned + num_per_div_before_alignment * i + num,
           copynum * sizeof(float));
  }
  fpga_free(ptr_unaligned);
  *data_in = ptr_aligned;
}

size_t interleave(float **data_in, int num_after_alignment) {
  float *ptr_uninterleaved = *data_in;
  float *ptr_interleaved =
      (float *)fpga_malloc(2 * num_after_alignment * sizeof(float));  // NOLINT
  int num = num_after_alignment / 4;
  for (int i = 0; i < num; i++) {
    memcpy(
        ptr_interleaved + 8 * i, ptr_uninterleaved + 4 * i, 4 * sizeof(float));
    memcpy(ptr_interleaved + 8 * i + 4,
           ptr_uninterleaved + num_after_alignment + 4 * i,
           4 * sizeof(float));
  }

  fpga_free(ptr_uninterleaved);
  *data_in = ptr_interleaved;
  return 2 * num_after_alignment * sizeof(float);
}

void format_bias_scale_array(float **bias_scale_array,
                             int element_num_per_division,
                             int num) {
  align_element(bias_scale_array, element_num_per_division, num);
  int div_num = (num + element_num_per_division - 1) / element_num_per_division;
  int element_num_after_division =
      align_to_x(element_num_per_division, BS_NUM_ALIGNMENT);
  size_t mem =
      interleave(bias_scale_array, div_num * element_num_after_division);
  fpga_flush(*bias_scale_array, mem);
}
void format_bias_array(float **bias_array, int num) {
  float *ptr_unaligned = *bias_array;
  int num_before_align = num;
  int num_after_align = align_to_x(num_before_align, BIAS_NUM_ALIGNMENT);
  int16_t *ptr_aligned =
      (int16_t *)fpga_malloc(num_after_align * sizeof(int16_t));  // NOLINT

  memset(ptr_aligned, 0, num_after_align * sizeof(int16_t));
  for (int i = 0; i < num_before_align; i++) {
    float value = ptr_aligned[i];
    ptr_aligned[i] = fp32_2_fp16(ptr_unaligned[i]);
  }
  *bias_array = (float *)ptr_aligned;  // NOLINT
  fpga_free(ptr_unaligned);
}

}  // namespace bias_scale
}  // namespace zynqmp
}  // namespace paddle
