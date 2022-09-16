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

__kernel void matmul_highdim(__read_only image2d_t input,
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

  int out_N = out_n / M;
  for (int c = 0; c < K; c++) {
    CL_COMPUTE_DTYPE4 w0 =
        weights[(out_N * K + c) * out_img_width + out_cblks * out_w + out_c];
    CL_COMPUTE_DTYPE4 v0 = READ_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR,
                                         input,
                                         SAMPLER,
                                         (int2)(out_cblks * K + c, out_n));

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

__kernel void matmul_xdim4_ydim1(__read_only image2d_t input,
                                 __write_only image2d_t output,
                                 __global const CL_COMPUTE_DTYPE4 *weights,
                                 int M,
                                 int C,
                                 int H,
                                 int W,
                                 float scale) {
  int nblk_id = get_global_id(2);  // n
  int h_id = get_global_id(0);     // h --> c
  int cblk_id = get_global_id(1);  // cblk_id

  if (h_id >= H) return;

  CL_COMPUTE_DTYPE4 s0 = (CL_COMPUTE_DTYPE4)(0.0f);
  CL_COMPUTE_DTYPE4 s1 = (CL_COMPUTE_DTYPE4)(0.0f);
  CL_COMPUTE_DTYPE4 s2 = (CL_COMPUTE_DTYPE4)(0.0f);
  CL_COMPUTE_DTYPE4 s3 = (CL_COMPUTE_DTYPE4)(0.0f);

  for (int w = 0; w < W; ++w) {
    CL_COMPUTE_DTYPE4 w0 = weights[w];
    CL_COMPUTE_DTYPE4 v0 =
        READ_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR,
                      input,
                      SAMPLER,
                      (int2)(cblk_id * W + w, (nblk_id * 4 + 0) * H + h_id));
    CL_COMPUTE_DTYPE4 v1 =
        READ_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR,
                      input,
                      SAMPLER,
                      (int2)(cblk_id * W + w, (nblk_id * 4 + 1) * H + h_id));
    CL_COMPUTE_DTYPE4 v2 =
        READ_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR,
                      input,
                      SAMPLER,
                      (int2)(cblk_id * W + w, (nblk_id * 4 + 2) * H + h_id));
    CL_COMPUTE_DTYPE4 v3 =
        READ_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR,
                      input,
                      SAMPLER,
                      (int2)(cblk_id * W + w, (nblk_id * 4 + 3) * H + h_id));

    s0 += w0.x * v0;
    s1 += w0.x * v1;
    s2 += w0.x * v2;
    s3 += w0.x * v3;
  }

  CL_COMPUTE_DTYPE4 output0 = (CL_COMPUTE_DTYPE4)(s0.x, s1.x, s2.x, s3.x);
  CL_COMPUTE_DTYPE4 output1 = (CL_COMPUTE_DTYPE4)(s0.y, s1.y, s2.y, s3.y);
  CL_COMPUTE_DTYPE4 output2 = (CL_COMPUTE_DTYPE4)(s0.z, s1.z, s2.z, s3.z);
  CL_COMPUTE_DTYPE4 output3 = (CL_COMPUTE_DTYPE4)(s0.w, s1.w, s2.w, s3.w);

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

  int2 out_pos0 = (int2)(nblk_id * 4 + h_id, cblk_id * 4 + 0);
  int2 out_pos1 = (int2)(nblk_id * 4 + h_id, cblk_id * 4 + 1);
  int2 out_pos2 = (int2)(nblk_id * 4 + h_id, cblk_id * 4 + 2);
  int2 out_pos3 = (int2)(nblk_id * 4 + h_id, cblk_id * 4 + 3);

  CL_DTYPE s = CONVERT_TYPE_TO(scale, CL_DTYPE);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos0, out0 * s);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos1, out1 * s);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos2, out2 * s);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos3, out3 * s);
}

__kernel void matmul_xdim3_ydim1(__read_only image2d_t input,
                                 __write_only image2d_t output,
                                 __global const CL_COMPUTE_DTYPE4 *weights,
                                 int M,
                                 int C,
                                 int H,
                                 int W,
                                 float scale) {
  int hblk_id = get_global_id(0);
  int cblk_id = get_global_id(1);

  CL_COMPUTE_DTYPE4 s0 = (CL_COMPUTE_DTYPE4)(0.0f);
  CL_COMPUTE_DTYPE4 s1 = (CL_COMPUTE_DTYPE4)(0.0f);
  CL_COMPUTE_DTYPE4 s2 = (CL_COMPUTE_DTYPE4)(0.0f);
  CL_COMPUTE_DTYPE4 s3 = (CL_COMPUTE_DTYPE4)(0.0f);

  for (int w = 0; w < W; ++w) {
    CL_COMPUTE_DTYPE4 w0 = weights[w];
    CL_COMPUTE_DTYPE4 v0 = READ_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR,
                                         input,
                                         SAMPLER,
                                         (int2)(cblk_id * W + w, hblk_id * 4));
    CL_COMPUTE_DTYPE4 v1 =
        READ_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR,
                      input,
                      SAMPLER,
                      (int2)(cblk_id * W + w, hblk_id * 4 + 1));
    CL_COMPUTE_DTYPE4 v2 =
        READ_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR,
                      input,
                      SAMPLER,
                      (int2)(cblk_id * W + w, hblk_id * 4 + 2));
    CL_COMPUTE_DTYPE4 v3 =
        READ_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR,
                      input,
                      SAMPLER,
                      (int2)(cblk_id * W + w, hblk_id * 4 + 3));

    s0 += w0.x * v0;
    s1 += w0.x * v1;
    s2 += w0.x * v2;
    s3 += w0.x * v3;
  }

  CL_COMPUTE_DTYPE4 output0 = (CL_COMPUTE_DTYPE4)(s0.x, s1.x, s2.x, s3.x);
  CL_COMPUTE_DTYPE4 output1 = (CL_COMPUTE_DTYPE4)(s0.y, s1.y, s2.y, s3.y);
  CL_COMPUTE_DTYPE4 output2 = (CL_COMPUTE_DTYPE4)(s0.z, s1.z, s2.z, s3.z);
  CL_COMPUTE_DTYPE4 output3 = (CL_COMPUTE_DTYPE4)(s0.w, s1.w, s2.w, s3.w);

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

  int2 out_pos0 = (int2)(hblk_id, cblk_id * 4 + 0);
  int2 out_pos1 = (int2)(hblk_id, cblk_id * 4 + 1);
  int2 out_pos2 = (int2)(hblk_id, cblk_id * 4 + 2);
  int2 out_pos3 = (int2)(hblk_id, cblk_id * 4 + 3);

  CL_DTYPE s = CONVERT_TYPE_TO(scale, CL_DTYPE);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos0, out0 * s);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos1, out1 * s);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos2, out2 * s);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos3, out3 * s);
}

__kernel void matmul_xdim2_ydim1(__read_only image2d_t input,
                                 __write_only image2d_t output,
                                 __global const CL_COMPUTE_DTYPE4 *weights,
                                 int M,
                                 int C,
                                 int H,
                                 int W,
                                 float scale) {
  int hblk_id = get_global_id(0);

  CL_COMPUTE_DTYPE s0 = (CL_COMPUTE_DTYPE)(0.0f);
  CL_COMPUTE_DTYPE s1 = (CL_COMPUTE_DTYPE)(0.0f);
  CL_COMPUTE_DTYPE s2 = (CL_COMPUTE_DTYPE)(0.0f);
  CL_COMPUTE_DTYPE s3 = (CL_COMPUTE_DTYPE)(0.0f);

  for (int w = 0; w < (W + 3) / 4; ++w) {
    CL_COMPUTE_DTYPE4 w0 = weights[w];
    CL_COMPUTE_DTYPE4 v0 = READ_IMG_TYPE(
        CL_COMPUTE_DTYPE_CHAR, input, SAMPLER, (int2)(w, hblk_id * 4));
    CL_COMPUTE_DTYPE4 v1 = READ_IMG_TYPE(
        CL_COMPUTE_DTYPE_CHAR, input, SAMPLER, (int2)(w, hblk_id * 4 + 1));
    CL_COMPUTE_DTYPE4 v2 = READ_IMG_TYPE(
        CL_COMPUTE_DTYPE_CHAR, input, SAMPLER, (int2)(w, hblk_id * 4 + 2));
    CL_COMPUTE_DTYPE4 v3 = READ_IMG_TYPE(
        CL_COMPUTE_DTYPE_CHAR, input, SAMPLER, (int2)(w, hblk_id * 4 + 3));
    s0 += dot(v0, w0);
    s1 += dot(v1, w0);
    s2 += dot(v2, w0);
    s3 += dot(v3, w0);
  }

  CL_COMPUTE_DTYPE4 output0 = (CL_COMPUTE_DTYPE4)(s0, s1, s2, s3);
  CL_DTYPE4 zero_v4 = (CL_DTYPE4)0;
  CL_DTYPE4 out0 = zero_v4;
  out0.x = CONVERT_TYPE_TO(output0.x, CL_DTYPE);
  out0.y = CONVERT_TYPE_TO(output0.y, CL_DTYPE);
  out0.z = CONVERT_TYPE_TO(output0.z, CL_DTYPE);
  out0.w = CONVERT_TYPE_TO(output0.w, CL_DTYPE);

  int2 out_pos0 = (int2)(hblk_id, 0);
  CL_DTYPE s = CONVERT_TYPE_TO(scale, CL_DTYPE);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos0, out0 * s);
}

__kernel void matmul_highdimx_ydim2(__read_only image2d_t input,
                                    __write_only image2d_t output,
                                    __global const CL_COMPUTE_DTYPE4 *weights,
                                    int M,
                                    int K,
                                    int out_w,
                                    int out_img_width,
                                    float scale) {
  int out_n = get_global_id(2);  // h * N
  int out_c = get_global_id(0);  // n
  int cblk_id = get_global_id(1);

  if (out_c >= out_w) return;

  CL_COMPUTE_DTYPE4 s0 = (CL_COMPUTE_DTYPE4)(0.0f);

  for (int k = 0; k < K; k++) {
    CL_COMPUTE_DTYPE4 w0 = weights[k * out_w + out_c];
    CL_COMPUTE_DTYPE4 v0 = READ_IMG_TYPE(
        CL_COMPUTE_DTYPE_CHAR, input, SAMPLER, (int2)(cblk_id * K + k, out_n));
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
