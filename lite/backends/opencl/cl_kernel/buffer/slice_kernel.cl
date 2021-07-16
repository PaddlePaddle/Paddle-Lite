/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

__kernel void slice(__global const DTYPE* in,
                    __global DTYPE* out,
                    __global const int* src_step,
                    __global const int* dst_step,
                    __global const int* real_starts,
                    const int dim_size,
                    const int out_num) {
  const int dst_id = get_global_id(0);  // [0, out_num)

  if (dst_id >= out_num) {
    return;
  }

  int src_id = 0;
  int index_id = dst_id;
  for (int j = 0; j < dim_size; j++) {
    int cur_id = index_id / dst_step[j];
    index_id = index_id % dst_step[j];
    src_id += (cur_id + real_starts[j]) * src_step[j];
  }
  out[dst_id] = in[src_id];
}
