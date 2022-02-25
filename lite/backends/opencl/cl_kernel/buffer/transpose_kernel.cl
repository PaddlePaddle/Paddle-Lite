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

__kernel void transpose_general_buffer(__global const CL_DTYPE* src,
                                       __global CL_DTYPE* dst,
                                       __global const int* out_idxs,
                                       __private const int out_tensor_c,
                                       __private const int out_tensor_h,
                                       __private const int out_tensor_w,
                                       __private const int out_tensor_hw) {
  int hidx = get_global_id(0);   // [0, h) columns of dst
  int widx = get_global_id(1);   // [0, w) rows of dst
  int chidx = get_global_id(2);  // [0, ch) channels of dst

  // idx = chidx * out_tensor_hw + hidx * out_tensor_w + widx
  const int idx = mad((CL_DTYPE)chidx,
                      (CL_DTYPE)out_tensor_hw,
                      (CL_DTYPE)(mul24(hidx, out_tensor_w) + widx));

  dst[out_idxs[idx]] = src[idx];
}
