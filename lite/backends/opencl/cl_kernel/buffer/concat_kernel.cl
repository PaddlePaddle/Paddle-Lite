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

#include <cl_common.h>

__kernel void concat2(__global const CL_DTYPE* x_data0,
                      __global const CL_DTYPE* x_data1,
                      __global CL_DTYPE* out_data,
                      int size,
                      int axis_size,
                      int pre_size,
                      int post_size,
                      int total,
                      int total0,
                      int total1) {
  const int index = get_global_id(0);
  if (index < size) {
    for (int i = 0; i < pre_size; i++) {
      int offset_out = index * post_size + i * total;
      int offset_in = index * post_size + i * total0;
      // memcpy(out_data + offset_out, x_data0 + offset_in, post_size);
      __global CL_DTYPE* dst = (__global CL_DTYPE*)(out_data + offset_out);
      __global CL_DTYPE* src = (__global CL_DTYPE*)(x_data0 + offset_in);
      for (int k = 0; k < post_size; k++) {
        *dst++ = *src++;
      }
    }
  } else if (index < axis_size) {
    for (int i = 0; i < pre_size; i++) {
      int offset_out = index * post_size + i * total;
      int offset_in = index * post_size + i * total1;
      // memcpy(out_data + offset_out, x_data1 + offset_in, post_size);
      __global CL_DTYPE* dst = (__global CL_DTYPE*)(out_data + offset_out);
      __global CL_DTYPE* src = (__global CL_DTYPE*)(x_data1 + offset_in);
      for (int k = 0; k < post_size; k++) {
        *dst++ = *src++;
      }
    }
  }
}

/*
__kernel void concat_mul_buffer(
                     __global const CL_DTYPE* x_data,
                     __global CL_DTYPE* out_data,
                     int axis_size,
                     int pre_size,
                     int post_size,
                     int start,
                     int total,
                     int total0) {
    const int index = get_global_id(0); // [0, axis_size)
    if (index < axis_size) {
        for (int i = 0; i < pre_size; i++) {
            int offset_out = (start + index) * post_size + i * total;
            int offset_in = index * post_size + i * total0;
            // memcpy(out_data + offset_out, x_data + offset_in, post_size);
            __global CL_DTYPE* dst = (__global CL_DTYPE*)(out_data +
offset_out);
            __global CL_DTYPE* src = (__global CL_DTYPE*)(x_data + offset_in);
            for (int k = 0; k < post_size; k++) {
                *dst++ = *src++;
            }
        }
    }
}
*/

__kernel void concat_mul_buffer(__global const CL_DTYPE* x_data,
                                __global CL_DTYPE* out_data,
                                int start,
                                int total,
                                int total0) {
  const int post_idx = get_global_id(0);  // [0, post_size)
  const int axis_idx = get_global_id(1);  // [0, axis_size)
  const int pre_idx = get_global_id(2);   // [0, pre_size)
  const int post_size = get_global_size(0);

  int offset_out = (start + axis_idx) * post_size + pre_idx * total;
  int offset_in = axis_idx * post_size + pre_idx * total0;
  int pos_out = offset_out + post_idx;
  int pos_in = offset_in + post_idx;

  out_data[pos_out] = x_data[pos_in];
}
