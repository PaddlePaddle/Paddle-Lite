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

#include "fpga/V2/bias_scale.h"
#include <memory.h>
#include "fpga/common/fpga_common.h"

namespace paddle_mobile {
namespace fpga {
namespace bias_scale {

void align_element(float **data_in, int num, int num_after_alignment) {
  float *ptr_unaligned = *data_in;
  int total_element = 2 * num_after_alignment;  // including bias & scale
  float *ptr_aligned =
      (float *)fpga_malloc(total_element * sizeof(float));  // NOLINT
  memset(ptr_aligned, 0, total_element * sizeof(float));

  for (int i = 0; i < num; i++) {
    ptr_aligned[i * 2 + 0] = ptr_unaligned[i];
    ptr_aligned[i * 2 + 1] = ptr_unaligned[i + num];
  }

  fpga_free(ptr_unaligned);
  *data_in = ptr_aligned;
}

void format_bias_scale_array(float **data_in, int num,
                             int num_after_alignment) {
  align_element(data_in, num, num_after_alignment);
  fpga_flush(*data_in, 2 * num_after_alignment * sizeof(float));
}

}  // namespace bias_scale
}  // namespace fpga
}  // namespace paddle_mobile
