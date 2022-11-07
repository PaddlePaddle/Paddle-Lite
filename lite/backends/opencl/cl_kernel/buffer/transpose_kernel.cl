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
  int idx0 = get_global_id(0);   // [0, nh) columns of dst
  int widx = get_global_id(1);   // [0, w) rows of dst
  int chidx = get_global_id(2);  // [0, ch) channels of dst

  int nidx = idx0 / out_tensor_h;
  int hidx = idx0 % out_tensor_h;

  const int idx = nidx * out_tensor_c * out_tensor_hw +
                  mad((CL_DTYPE)chidx,
                      (CL_DTYPE)out_tensor_hw,
                      (CL_DTYPE)(mul24(hidx, out_tensor_w) + widx));

  dst[out_idxs[idx]] = src[idx];
}

__kernel void transpose_0213_buffer(__global const CL_DTYPE* src,
                                    __global CL_DTYPE* dst,
                                    __global const int* out_idxs,
                                    __private const int out_tensor_c,
                                    __private const int out_tensor_h,
                                    __private const int out_tensor_w,
                                    __private const int out_tensor_hw) {
  int idx0 = get_global_id(0);       // [0, nh) columns of dst src_c
  int widx = get_global_id(1) << 3;  // [0, w) rows of dst
  int chidx = get_global_id(2);      // [0, ch) channels of dst src_h

  int nidx = idx0 / out_tensor_h;
  int hidx = idx0 % out_tensor_h;

  const int idx = nidx * out_tensor_c * out_tensor_hw +
                  hidx * out_tensor_c * out_tensor_w + chidx * out_tensor_w;
  const int dst_idx = nidx * out_tensor_c * out_tensor_hw +
                      chidx * out_tensor_w * out_tensor_h + hidx * out_tensor_w;
  if (widx + 8 < out_tensor_w) {
    CL_DTYPE8 src_w8 = vload8(0, src + idx + widx);
    vstore8(src_w8, 0, dst + dst_idx + widx);
  } else {
    for (int i = widx; i < out_tensor_w; i++) {
      dst[dst_idx + i] = src[idx + i];
    }
  }
}