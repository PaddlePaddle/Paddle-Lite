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

#include "bias_scale.h"
#include <memory.h>
#include "api.h"

namespace paddle_mobile {
namespace fpga {
namespace bias_scale {

void align_element(float **data_in, int num_per_div_before_alignment, int num) {
  float *ptr_unaligned = *data_in;
  int div_num =
      (num + num_per_div_before_alignment - 1) / num_per_div_before_alignment;
  int num_per_div_after_alignment =
      align_to_x(num_per_div_before_alignment, BS_NUM_ALIGNMENT);
  int num_element =
      2 * div_num * num_per_div_after_alignment;  // including bias & scale
  float *ptr_aligned = (float *)fpga_malloc(num_element * sizeof(float));

  memset(ptr_aligned, 0, num_element * sizeof(float));

  for (int i = 0; i < div_num; i++) {
    memcpy(ptr_aligned + i * num_per_div_after_alignment, ptr_unaligned,
           num_per_div_before_alignment * sizeof(float));
  }

  fpga_free(ptr_unaligned);
  *data_in = ptr_aligned;
}

void interleave(float **data_in, int num_after_alignment) {
  // num_after_alignment: number of bias after alignment

  float *ptr_uninterleaved = *data_in;
  float *ptr_interleaved =
      (float *)fpga_malloc(2 * num_after_alignment * sizeof(float));
  int num = num_after_alignment / 4;
  for (int i = 0; i < num; i++) {
    memcpy(ptr_interleaved + 8 * i, ptr_uninterleaved + 4 * i,
           4 * sizeof(float));
    memcpy(ptr_interleaved + 8 * i + 4,
           ptr_uninterleaved + num_after_alignment * sizeof(float) + 4 * i,
           4 * sizeof(float));
  }

  fpga_free(ptr_uninterleaved);
  *data_in = ptr_interleaved;
}

}  // namespace bias_scale
}  // namespace fpga
}  // namespace paddle_mobile
