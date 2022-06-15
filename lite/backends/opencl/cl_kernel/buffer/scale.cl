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

__kernel void scale_buffer(__global const CL_DTYPE* src,
                           __global CL_DTYPE* dst,
                           __private float scale,
                           __private float bias,
                           __private float alpha,
                           __private const int out_n,
                           __private const int out_c,
                           __private const int out_h,
                           __private const int out_w) {
  const int bh = get_global_id(0);
  const int c = get_global_id(1);
  const int w = get_global_id(2) << 3;

  const int idx_n = bh / out_h;
  const int idx_h = bh % out_h;

  int index = idx_n * out_c * out_h * out_w + c * out_h * out_w + idx_h * out_w;
  index += (w + 8) > out_w ? (out_w - 8) : w;
  CL_DTYPE8 src_w8 = vload8(0, src + index);
  CL_DTYPE8 scale_v = src_w8 * (CL_DTYPE8)(CONVERT_TYPE_TO(scale, CL_DTYPE));
  CL_DTYPE8 bias_v = scale_v + (CL_DTYPE8)(CONVERT_TYPE_TO(bias, CL_DTYPE));
  vstore8(bias_v, 0, dst + index);
}
