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

__kernel void fp32_fp16_buffer(__global const float* src,
                               __global half* dst,
                               __private const int count) {
  const int idx = get_global_id(0) << 3;
  if (idx > count) {
    return;
  }

  if (count < 8) {
    for (int i = 0; i < count; i++) {
      dst[i] = convert_half(src[i]);
    }
  } else {
    const int index = (idx + 8) > count ? (count - 8) : idx;
    float8 src_v8 = vload8(0, src + index);
    half8 out_v8 = convert_half8(src_v8);
    vstore8(out_v8, 0, dst + index);
  }
}

__kernel void fp16_fp32_buffer(__global const half* src,
                               __global float* dst,
                               __private const int count) {
  const int idx = get_global_id(0) << 3;
  if (idx > count) {
    return;
  }

  if (count < 8) {
    for (int i = 0; i < count; i++) {
      dst[i] = convert_float(src[i]);
    }
  } else {
    const int index = (idx + 8) > count ? (count - 8) : idx;
    half8 src_v8 = vload8(0, src + index);
    float8 out_v8 = convert_float8(src_v8);
    vstore8(out_v8, 0, dst + index);
  }
}
