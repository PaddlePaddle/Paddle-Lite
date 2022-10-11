/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

__kernel void argmax_buffer(__global const CL_DTYPE* src,
                            __global CL_DTYPE* dst,
                            __private const int outer_size,
                            __private const int axis_size,
                            __private const int inner_size) {
  const int inner_id = get_global_id(0);
  const int outer_id = get_global_id(1);

  size_t offset = outer_id * outer_size + inner_id * inner_size;
  const CL_DTYPE* input_ptr = src + offset;
  CL_DTYPE max_val = input_ptr[0];
  CL_DTYPE max_idx = 0;
  for (int i = 0; i < axis_size; ++i) {
    CL_DTYPE cur_val = input_ptr[i * inner_size];
    if (cur_val > max_val) {
      max_val = cur_val;
      max_idx = i;
    }
  }
  dst[offset] = max_idx;
}

__kernel void argmax_out_int32(__global const CL_DTYPE* src,
                               __global int* dst,
                               __private const int outer_size,
                               __private const int axis_size,
                               __private const int inner_size) {
  const int inner_id = get_global_id(0);
  const int outer_id = get_global_id(1);

  size_t offset = outer_id * inner_size * axis_size + inner_id;
  const CL_DTYPE* input_ptr = src + offset;
  CL_DTYPE max_val = input_ptr[0];
  int max_idx = 0;
  for (int i = 0; i < axis_size; ++i) {
    CL_DTYPE cur_val = input_ptr[i * inner_size];
    if (cur_val > max_val) {
      max_val = cur_val;
      max_idx = i;
    }
  }
  dst[outer_id * inner_size + inner_id] = max_idx;
}

__kernel void argmax_out_int64(__global const CL_DTYPE* src,
                               __global long* dst,
                               __private const int outer_size,
                               __private const int axis_size,
                               __private const int inner_size) {
  const int inner_id = get_global_id(0);
  const int outer_id = get_global_id(1);

  size_t offset = outer_id * inner_size * axis_size + inner_id;
  const CL_DTYPE* input_ptr = src + offset;
  CL_DTYPE max_val = input_ptr[0];
  long max_idx = 0;
  for (int i = 0; i < axis_size; ++i) {
    CL_DTYPE cur_val = input_ptr[i * inner_size];
    if (cur_val > max_val) {
      max_val = cur_val;
      max_idx = i;
    }
  }
  dst[outer_id * inner_size + inner_id] = max_idx;
}
