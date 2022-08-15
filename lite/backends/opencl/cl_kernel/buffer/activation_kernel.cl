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

__kernel void relu(__global const CL_DTYPE* x_data,
                   const int count,
                   __global CL_DTYPE* out_data) {
  const int idx = get_global_id(0) << 3;

  const int index = (idx + 8) > count ? (count - 8) : idx;
  CL_DTYPE8 in = vload8(0, x_data + index);
  in = max((CL_DTYPE8)(0.0f), in);
  vstore8(in, 0, out_data + index);
}

__kernel void relu6(__global const CL_DTYPE* x_data,
                    const int count,
                    __global CL_DTYPE* out_data) {
  const int idx = get_global_id(0) << 3;

  const int index = (idx + 8) > count ? (count - 8) : idx;
  CL_DTYPE8 in = vload8(0, x_data + index);
  in = clamp(in, (CL_DTYPE8)(0.0f), (CL_DTYPE)6);
  vstore8(in, 0, out_data + index);
}

__kernel void tanh_act(__global const CL_DTYPE* x_data,
                       const int count,
                       __global CL_DTYPE* out_data) {
  const int idx = get_global_id(0) << 3;

  const int index = (idx + 8) > count ? (count - 8) : idx;
  CL_DTYPE8 in = vload8(0, x_data + index);
  in = (exp(in) - exp(-in)) / (exp(in) + exp(-in));
  vstore8(in, 0, out_data + index);
}

__kernel void gelu(__global const CL_DTYPE* x_data,
                   const int count,
                   __global CL_DTYPE* out_data) {
  const int idx = get_global_id(0) << 3;

  const int index = (idx + 8) > count ? (count - 8) : idx;
  CL_DTYPE8 in = vload8(0, x_data + index);
  in = (CL_DTYPE8)0.5f * in * ((CL_DTYPE8)1.0f + erf(in / (CL_DTYPE8)1.41421f));
  vstore8(in, 0, out_data + index);
}

__kernel void sigmoid(__global const CL_DTYPE* x_data,
                      const int count,
                      __global CL_DTYPE* out_data) {
  const int idx = get_global_id(0) << 3;

  const int index = (idx + 8) > count ? (count - 8) : idx;
  CL_DTYPE8 in = vload8(0, x_data + index);
  in = (CL_DTYPE8)1 / ((CL_DTYPE8)1 + exp(-in));
  vstore8(in, 0, out_data + index);
}
