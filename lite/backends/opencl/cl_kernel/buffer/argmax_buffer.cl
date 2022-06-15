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

__kernel void argmax_w(__global const CL_DTYPE* src,
                       __global CL_DTYPE* dst,
                       __private const int out_n,
                       __private const int out_c,
                       __private const int out_h,
                       __private const int in_w) {
  const int bh = get_global_id(0);
  const int c = get_global_id(1);
  const int w = get_global_id(2);

  const int idx_n = bh / out_h;
  const int idx_h = bh % out_h;

  CL_DTYPE cur_data = (CL_DTYPE)(DATAINIT);
  CL_DTYPE max_data = (CL_DTYPE)(DATAINIT);
  CL_DTYPE cur_idx = (CL_DTYPE)(DATAINIT);
  CL_DTYPE max_idx = (CL_DTYPE)(DATAINIT);

  FLAG_TYPE flag_v = (FLAG_TYPE)(0);

  int index_src =
      idx_n * out_c * out_h * in_w + c * out_h * in_w + idx_h * in_w;
  int index_dst = idx_n * out_c * out_h + c * out_h + idx_h;
  for (unsigned short i = 0; i < in_w; i++) {
    cur_data = src[index_src + i];
    cur_idx = (CL_DTYPE)(i);
    flag_v = isgreaterequal(cur_data, max_data);
    max_data = select(max_data, cur_data, flag_v);
    max_idx = select(max_idx, cur_idx, flag_v);
  }

  dst[index_dst] = max_idx;
}
