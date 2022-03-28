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

__kernel void matmul_transpose_x(__read_only image2d_t input,
                                 __write_only image2d_t output,
                                 __global const CL_COMPUTE_DTYPE16 *weights,
                                 int M,
                                 int k_blks,
                                 int n_blks,
                                 float scale) {
  int out_n = get_global_id(2);  // m
  int out_c = get_global_id(0);  // n
  int2 tid = (int2)(get_local_id(0), get_local_id(1));

  CL_COMPUTE_DTYPE4 s0 = (CL_COMPUTE_DTYPE4)(0.0f);
  CL_COMPUTE_DTYPE4 s1 = (CL_COMPUTE_DTYPE4)(0.0f);
  CL_COMPUTE_DTYPE4 s2 = (CL_COMPUTE_DTYPE4)(0.0f);
  CL_COMPUTE_DTYPE4 s3 = (CL_COMPUTE_DTYPE4)(0.0f);
  if (out_n >= M) return;

  if (out_c < n_blks) {
    for (int c = tid.y; c < k_blks; c += 4) {
      CL_COMPUTE_DTYPE16 w = weights[c * n_blks + out_c];

      CL_COMPUTE_DTYPE4 v0 = READ_IMG_TYPE(
          CL_COMPUTE_DTYPE_CHAR, input, SAMPLER, (int2)(out_n, c * 4 + 0));
      s0 += v0.x * w.s0123;
      s1 += v0.y * w.s0123;
      s2 += v0.z * w.s0123;
      s3 += v0.w * w.s0123;
      CL_COMPUTE_DTYPE4 v1 = READ_IMG_TYPE(
          CL_COMPUTE_DTYPE_CHAR, input, SAMPLER, (int2)(out_n, c * 4 + 1));
      s0 += v1.x * w.s4567;
      s1 += v1.y * w.s4567;
      s2 += v1.z * w.s4567;
      s3 += v1.w * w.s4567;
      CL_COMPUTE_DTYPE4 v2 = READ_IMG_TYPE(
          CL_COMPUTE_DTYPE_CHAR, input, SAMPLER, (int2)(out_n, c * 4 + 2));
      s0 += v2.x * w.s89ab;
      s1 += v2.y * w.s89ab;
      s2 += v2.z * w.s89ab;
      s3 += v2.w * w.s89ab;
      CL_COMPUTE_DTYPE4 v3 = READ_IMG_TYPE(
          CL_COMPUTE_DTYPE_CHAR, input, SAMPLER, (int2)(out_n, c * 4 + 3));
      s0 += v3.x * w.scdef;
      s1 += v3.y * w.scdef;
      s2 += v3.z * w.scdef;
      s3 += v3.w * w.scdef;
    }
  }
  __local CL_COMPUTE_DTYPE4 temp[32][16];
  temp[tid.x][tid.y * 4 + 0] = s0;
  temp[tid.x][tid.y * 4 + 1] = s1;
  temp[tid.x][tid.y * 4 + 2] = s2;
  temp[tid.x][tid.y * 4 + 3] = s3;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (out_c >= n_blks) return;

  if (tid.y == 0) {
    s0 += temp[tid.x][4];
    s0 += temp[tid.x][8];
    s0 += temp[tid.x][12];

    s1 += temp[tid.x][5];
    s1 += temp[tid.x][9];
    s1 += temp[tid.x][13];

    s2 += temp[tid.x][6];
    s2 += temp[tid.x][10];
    s2 += temp[tid.x][14];

    s3 += temp[tid.x][7];
    s3 += temp[tid.x][11];
    s3 += temp[tid.x][15];

    CL_COMPUTE_DTYPE4 output0 = s0;
    CL_COMPUTE_DTYPE4 output1 = s1;
    CL_COMPUTE_DTYPE4 output2 = s2;
    CL_COMPUTE_DTYPE4 output3 = s3;

    CL_DTYPE4 out0, out1, out2, out3;
    out0.x = CONVERT_TYPE_TO(output0.x, CL_DTYPE);
    out0.y = CONVERT_TYPE_TO(output0.y, CL_DTYPE);
    out0.z = CONVERT_TYPE_TO(output0.z, CL_DTYPE);
    out0.w = CONVERT_TYPE_TO(output0.w, CL_DTYPE);

    out1.x = CONVERT_TYPE_TO(output1.x, CL_DTYPE);
    out1.y = CONVERT_TYPE_TO(output1.y, CL_DTYPE);
    out1.z = CONVERT_TYPE_TO(output1.z, CL_DTYPE);
    out1.w = CONVERT_TYPE_TO(output1.w, CL_DTYPE);

    out2.x = CONVERT_TYPE_TO(output2.x, CL_DTYPE);
    out2.y = CONVERT_TYPE_TO(output2.y, CL_DTYPE);
    out2.z = CONVERT_TYPE_TO(output2.z, CL_DTYPE);
    out2.w = CONVERT_TYPE_TO(output2.w, CL_DTYPE);

    out3.x = CONVERT_TYPE_TO(output3.x, CL_DTYPE);
    out3.y = CONVERT_TYPE_TO(output3.y, CL_DTYPE);
    out3.z = CONVERT_TYPE_TO(output3.z, CL_DTYPE);
    out3.w = CONVERT_TYPE_TO(output3.w, CL_DTYPE);

    int2 out_pos0 = (int2)(out_c, out_n * 4 + 0);
    int2 out_pos1 = (int2)(out_c, out_n * 4 + 1);
    int2 out_pos2 = (int2)(out_c, out_n * 4 + 2);
    int2 out_pos3 = (int2)(out_c, out_n * 4 + 3);

    CL_DTYPE s = CONVERT_TYPE_TO(scale, CL_DTYPE);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos0, out0 * s);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos1, out1 * s);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos2, out2 * s);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos3, out3 * s);
  }
}

__kernel void matmul_highdim_transpose_x(
    __read_only image2d_t input,
    __write_only image2d_t output,
    __global const CL_COMPUTE_DTYPE4 *weights,
    int M,
    int K,
    int out_w,
    int out_img_width,
    float scale) {
  int out_n = get_global_id(2);  // h * N
  int out_c = get_global_id(0);  // n
  int out_cblks = get_global_id(1);

  if (out_c >= out_w) return;

  CL_COMPUTE_DTYPE4 s0 = (CL_COMPUTE_DTYPE4)(0.0f);

  int out_n_id = out_n / M;
  int in_k_id = out_n % M;

  for (int c = 0; c < K; c++) {
    CL_COMPUTE_DTYPE4 v0 =
        READ_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR,
                      input,
                      SAMPLER,
                      (int2)(out_cblks * M + in_k_id, out_n_id * K + c));
    CL_COMPUTE_DTYPE4 w0 =
        weights[(out_n_id * K + c) * out_img_width + out_cblks * out_w + out_c];

    s0 = mad(v0, w0, s0);
  }

  CL_COMPUTE_DTYPE4 output0 = s0;
  CL_DTYPE4 out0;
  out0.x = CONVERT_TYPE_TO(output0.x, CL_DTYPE);
  out0.y = CONVERT_TYPE_TO(output0.y, CL_DTYPE);
  out0.z = CONVERT_TYPE_TO(output0.z, CL_DTYPE);
  out0.w = CONVERT_TYPE_TO(output0.w, CL_DTYPE);
  int2 output_pos0 = (int2)(out_cblks * out_w + out_c, out_n);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output,
                 output_pos0,
                 out0 * CONVERT_TYPE_TO(scale, CL_DTYPE));
}

__kernel void matmul_highdimxtranspose_ydim2(
    __read_only image2d_t input,
    __write_only image2d_t output,
    __global const CL_COMPUTE_DTYPE4 *weights,
    int M,
    int K,
    int out_w,
    int out_img_width,
    float scale) {
  int out_n = get_global_id(2);  // w * N
  int out_c = get_global_id(0);  // n
  int cblk_id = get_global_id(1);

  if (out_c >= out_w) return;

  CL_COMPUTE_DTYPE4 s0 = (CL_COMPUTE_DTYPE4)(0.0f);

  int h_id = out_n % M;
  int nblk_id = out_n / M;
  for (int k = 0; k < K; k++) {
    CL_COMPUTE_DTYPE4 w0 = weights[k * out_w + out_c];
    CL_COMPUTE_DTYPE4 v0 =
        READ_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR,
                      input,
                      SAMPLER,
                      (int2)(cblk_id * M + h_id, nblk_id * K + k));
    s0 += w0.x * v0;
  }

  CL_COMPUTE_DTYPE4 output0 = s0;
  CL_DTYPE4 out0;
  out0.x = CONVERT_TYPE_TO(output0.x, CL_DTYPE);
  out0.y = CONVERT_TYPE_TO(output0.y, CL_DTYPE);
  out0.z = CONVERT_TYPE_TO(output0.z, CL_DTYPE);
  out0.w = CONVERT_TYPE_TO(output0.w, CL_DTYPE);
  int2 output_pos0 = (int2)(cblk_id * out_w + out_c, out_n);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output,
                 output_pos0,
                 out0 * CONVERT_TYPE_TO(scale, CL_DTYPE));
}
