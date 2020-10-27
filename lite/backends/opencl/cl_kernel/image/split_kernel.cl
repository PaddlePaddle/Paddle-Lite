/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

__kernel void split2(__read_only image2d_t input,
                    __write_only image2d_t output0,
                    __write_only image2d_t output1,
                    __private const int flag,
                    __private const int out0_dims_axis,
                    __private const int in_dims_second,
                    __private const int in_dims_last,
                    __private const int width) {

  const int width_idx = get_global_id(0);
  const int channel_blk_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;


  int2 in_pos = (int2)(channel_blk_idx * in_dims_last + width_idx, hb_idx);
  int2 out_pos;
  CL_DTYPE4 in_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, in_pos);
  CL_DTYPE4 out_data;
  int c;

  if (flag == 1) {
    for (int i = 0; i < 4; i++) {
      c = channel_blk_idx * 4 + i;
      if (c >= in_dims_second) break;

      int c_out;
      CL_DTYPE4 in_data;
      if (c < out0_dims_axis) {
          out_pos = in_pos;
      } else {
          c_out = c - out0_dims_axis;
          out_pos = (int2)((c_out / 4) * in_dims_last + width_idx, hb_idx);
      }

      int offset = c % 4;
      CL_DTYPE val;
      if (offset == 0) {
          val = in_data.x;
      } else if (offset == 1) {
          val = in_data.y;
      } else if (offset == 2) {
          val = in_data.z;
      } else if (offset == 3) {
          val = in_data.w;
      }

      if (i == 0) {
        out_data.x = val;
      } else if (i == 1) {
        out_data.y = val;
      } else if (i == 2) {
        out_data.z = val;
      } else if (i == 3) {
        out_data.w = val;
      }
    }

    if (c < out0_dims_axis) {
        WRITE_IMG_TYPE(CL_DTYPE_CHAR, output0, out_pos, out_data);
    } else {
        WRITE_IMG_TYPE(CL_DTYPE_CHAR, output1, out_pos, out_data);
    }
  } else if (flag == 2) {
      printf("not imple in split kernel\n");
  }
}
