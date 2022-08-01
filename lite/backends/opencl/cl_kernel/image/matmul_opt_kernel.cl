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
#ifdef ADRENO_HIGH
#define LOC_MEM_SIZE 8
#else
#define LOC_MEM_SIZE 4
#endif

__kernel void matmul(__read_only image2d_t input,
                     __write_only image2d_t output,
#ifdef USE_IMAGE_Y
                     __read_only image2d_t weights,
#else
                     __global CL_COMPUTE_DTYPE16 *weights,
#endif
                     int M,
                     int k_blks,
                     int n_blks,
                     float scale) {
  int out_n = get_global_id(0);  // m
  int out_c = get_global_id(2);  // n
  int3 tid = (int3)(get_local_id(2), get_local_id(1), get_local_id(0));

  CL_COMPUTE_DTYPE4 s = (CL_COMPUTE_DTYPE4)(0.0f);
  if (out_n >= M) return;

  if (out_c < n_blks) {
    for (int c = tid.y; c < k_blks; c += 4) {
      CL_COMPUTE_DTYPE4 v = READ_IMG_TYPE(
          CL_COMPUTE_DTYPE_CHAR, input, SAMPLER, (int2)(c, out_n));
#ifdef USE_IMAGE_Y
      CL_COMPUTE_DTYPE4 w0 = READ_IMG_TYPE(
          CL_COMPUTE_DTYPE_CHAR, weights, SAMPLER, (int2)(out_c * 4 + 0, c));
      CL_COMPUTE_DTYPE4 w1 = READ_IMG_TYPE(
          CL_COMPUTE_DTYPE_CHAR, weights, SAMPLER, (int2)(out_c * 4 + 1, c));
      CL_COMPUTE_DTYPE4 w2 = READ_IMG_TYPE(
          CL_COMPUTE_DTYPE_CHAR, weights, SAMPLER, (int2)(out_c * 4 + 2, c));
      CL_COMPUTE_DTYPE4 w3 = READ_IMG_TYPE(
          CL_COMPUTE_DTYPE_CHAR, weights, SAMPLER, (int2)(out_c * 4 + 3, c));

      s += v.x * w0;
      s += v.y * w1;
      s += v.z * w2;
      s += v.w * w3;
#else
      CL_COMPUTE_DTYPE16 w = weights[c * n_blks + out_c];
      CL_COMPUTE_DTYPE4 partial = v.x * w.s0123;
      partial += v.y * w.s4567;
      partial += v.z * w.s89ab;
      partial += v.w * w.scdef;
      s += partial;
#endif
    }
  }
  __local CL_COMPUTE_DTYPE4 temp[LOC_MEM_SIZE][4][16];
  temp[tid.z][tid.y][tid.x] = s;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (out_c >= n_blks) return;

  if (tid.y == 0) {
    s += temp[tid.z][1][tid.x];
    s += temp[tid.z][2][tid.x];
    s += temp[tid.z][3][tid.x];

    int2 output_pos0 = (int2)(out_c, out_n);
    CL_COMPUTE_DTYPE4 output0 = s;

    CL_DTYPE4 out0;
    out0.x = CONVERT_TYPE_TO(output0.x, CL_DTYPE);
    out0.y = CONVERT_TYPE_TO(output0.y, CL_DTYPE);
    out0.z = CONVERT_TYPE_TO(output0.z, CL_DTYPE);
    out0.w = CONVERT_TYPE_TO(output0.w, CL_DTYPE);

    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   output,
                   output_pos0,
                   out0 * CONVERT_TYPE_TO(scale, CL_DTYPE));
  }
}
