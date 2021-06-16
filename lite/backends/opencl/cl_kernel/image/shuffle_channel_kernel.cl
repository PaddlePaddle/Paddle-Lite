/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cl_common.h>

__kernel void shuffle_channel(__read_only image2d_t input,
                              __write_only image2d_t output,
                              __private const int group,
                              __private const int group_size,
                              __private const int channels,
                              __private const int out_W) {
  const int w_idx = get_global_id(0);
  const int c4_idx = get_global_id(1);
  const int nh_idx = get_global_id(2);

  int2 output_pos;
  output_pos.x = c4_idx * out_W + w_idx;
  output_pos.y = nh_idx;

  CL_DTYPE4 output_data;
  for (int i = 0; i < 4; i++) {
    int outc_idx = (c4_idx << 2) + i;
    if (outc_idx >= channels) {
      break;
    }
    int inc_idx = outc_idx % group * group_size + outc_idx / group;
    int inc4_idx = inc_idx >> 2;
    int2 input_pos;
    input_pos.x = inc4_idx * out_W + w_idx;
    input_pos.y = nh_idx;
    CL_DTYPE4 input_data;
    input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);
    CL_DTYPE value;
    int sub_idx = inc_idx % 4;
    if (sub_idx == 0) {
      value = input_data.x;
    } else if (sub_idx == 1) {
      value = input_data.y;
    } else if (sub_idx == 2) {
      value = input_data.z;
    } else if (sub_idx == 3) {
      value = input_data.w;
    }
    if (i == 0) {
      output_data.x = value;
    } else if (i == 1) {
      output_data.y = value;
    } else if (i == 2) {
      output_data.z = value;
    } else if (i == 3) {
      output_data.w = value;
    }
  }
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, output_data);
}
