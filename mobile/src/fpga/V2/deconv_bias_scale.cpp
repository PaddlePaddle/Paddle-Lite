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

#include "fpga/V2/deconv_bias_scale.h"
// #include "deconv_bias_scale.h"
#include "fpga/V2/bias_scale.h"
// #include "bias_scale.h"
// #include <memory.h>

#include "fpga/V2/api.h"
// #include "fpga_api.h"
namespace paddle_mobile {
namespace fpga {
namespace deconv_bias_scale {

void deconv_bias_scale_expand(float** bias_scale_array, int num,
                              int sub_conv_n) {
  int sub_num = num * sub_conv_n;
  float* ptr_tmp = *bias_scale_array;
  float* ptr_bias_scale_expand =
      reinterpret_cast<float*>(fpga_malloc(sizeof(float) * sub_num * 2));
  int scale_base_offset = sub_num;
  for (int i = 0; i < sub_conv_n; ++i) {
    int offset = num * i;
    // copy bias
    fpga_copy(ptr_bias_scale_expand + offset, ptr_tmp, num * sizeof(float));
    // copy scale
    fpga_copy(ptr_bias_scale_expand + scale_base_offset + offset, ptr_tmp + num,
              num * sizeof(float));
  }
  *bias_scale_array = ptr_bias_scale_expand;
  fpga_free(ptr_tmp);
}

}  // namespace deconv_bias_scale
}  // namespace fpga
}  // namespace paddle_mobile
