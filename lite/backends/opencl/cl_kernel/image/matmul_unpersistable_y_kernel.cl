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

__kernel void matmul(__read_only image2d_t input,
                     __write_only image2d_t output,
                     __read_only image2d_t weights,
                     int shared_dim,
                     int out_width,
                     int out_height,
                     float scale) {
  int out_c = get_global_id(0);
  int out_w = get_global_id(1);
  int out_nh = get_global_id(2);

  int out_h = out_nh % out_height;
  int out_n = out_nh / out_height;

  CL_COMPUTE_DTYPE4 output0 = (CL_COMPUTE_DTYPE4)(0.0f);
  for (int w = 0; w < shared_dim; ++w) {
    CL_COMPUTE_DTYPE4 v0 =
        READ_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR,
                      input,
                      SAMPLER,
                      (int2)(out_c * shared_dim + w, out_nh));
    CL_COMPUTE_DTYPE4 w0 = READ_IMG_TYPE(
        CL_COMPUTE_DTYPE_CHAR,
        weights,
        SAMPLER,
        (int2)(out_c * out_width + out_w, out_n * shared_dim + w));
    output0 = mad(v0, w0, output0);
  }

  CL_DTYPE4 out0;
  out0.x = CONVERT_TYPE_TO(output0.x, CL_DTYPE);
  out0.y = CONVERT_TYPE_TO(output0.y, CL_DTYPE);
  out0.z = CONVERT_TYPE_TO(output0.z, CL_DTYPE);
  out0.w = CONVERT_TYPE_TO(output0.w, CL_DTYPE);

  int2 out_pos0 = (int2)(out_c * out_width + out_w, out_nh);

  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR, output, out_pos0, out0 * CONVERT_TYPE_TO(scale, CL_DTYPE));
}

__kernel void matmul_ytranspose(__read_only image2d_t input,
                                __write_only image2d_t output,
                                __read_only image2d_t weights,
                                int shared_dim,
                                int out_width,
                                int out_height,
                                float scale) {
  int out_c = get_global_id(0);
  int out_w = get_global_id(1);
  int out_nh = get_global_id(2);

  int out_h = out_nh % out_height;
  int out_n = out_nh / out_height;

  CL_COMPUTE_DTYPE4 output0 = (CL_COMPUTE_DTYPE4)(0.0f);
  for (int w = 0; w < shared_dim; ++w) {
    CL_COMPUTE_DTYPE4 v0 =
        READ_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR,
                      input,
                      SAMPLER,
                      (int2)(out_c * shared_dim + w, out_nh));
    CL_COMPUTE_DTYPE4 w0 = READ_IMG_TYPE(
        CL_COMPUTE_DTYPE_CHAR,
        weights,
        SAMPLER,
        (int2)(out_c * shared_dim + w, out_n * out_width + out_w));
    output0 = mad(v0, w0, output0);
  }

  CL_DTYPE4 out0;
  out0.x = CONVERT_TYPE_TO(output0.x, CL_DTYPE);
  out0.y = CONVERT_TYPE_TO(output0.y, CL_DTYPE);
  out0.z = CONVERT_TYPE_TO(output0.z, CL_DTYPE);
  out0.w = CONVERT_TYPE_TO(output0.w, CL_DTYPE);

  int2 out_pos0 = (int2)(out_c * out_width + out_w, out_nh);

  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR, output, out_pos0, out0 * CONVERT_TYPE_TO(scale, CL_DTYPE));
}

__kernel void matmul_xtranspose(__read_only image2d_t input,
                                __write_only image2d_t output,
                                __read_only image2d_t weights,
                                int shared_dim,
                                int out_width,
                                int out_height,
                                float scale) {
  int out_c = get_global_id(0);
  int out_w = get_global_id(1);
  int out_nh = get_global_id(2);

  int out_h = out_nh % out_height;
  int out_n = out_nh / out_height;

  CL_COMPUTE_DTYPE4 output0 = (CL_COMPUTE_DTYPE4)(0.0f);
  for (int w = 0; w < shared_dim; ++w) {
    CL_COMPUTE_DTYPE4 v0 = READ_IMG_TYPE(
        CL_COMPUTE_DTYPE_CHAR,
        input,
        SAMPLER,
        (int2)(out_c * out_height + out_h, out_n * shared_dim + w));
    CL_COMPUTE_DTYPE4 w0 = READ_IMG_TYPE(
        CL_COMPUTE_DTYPE_CHAR,
        weights,
        SAMPLER,
        (int2)(out_c * out_width + out_w, out_n * shared_dim + w));
    output0 = mad(v0, w0, output0);
  }

  CL_DTYPE4 out0;
  out0.x = CONVERT_TYPE_TO(output0.x, CL_DTYPE);
  out0.y = CONVERT_TYPE_TO(output0.y, CL_DTYPE);
  out0.z = CONVERT_TYPE_TO(output0.z, CL_DTYPE);
  out0.w = CONVERT_TYPE_TO(output0.w, CL_DTYPE);

  int2 out_pos0 = (int2)(out_c * out_width + out_w, out_nh);

  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR, output, out_pos0, out0 * CONVERT_TYPE_TO(scale, CL_DTYPE));
}

__kernel void matmul_xytranspose(__read_only image2d_t input,
                                 __write_only image2d_t output,
                                 __read_only image2d_t weights,
                                 int shared_dim,
                                 int out_width,
                                 int out_height,
                                 float scale) {
  int out_c = get_global_id(0);
  int out_w = get_global_id(1);
  int out_nh = get_global_id(2);

  int out_h = out_nh % out_height;
  int out_n = out_nh / out_height;

  CL_COMPUTE_DTYPE4 output0 = (CL_COMPUTE_DTYPE4)(0.0f);
  for (int w = 0; w < shared_dim; ++w) {
    CL_COMPUTE_DTYPE4 v0 = READ_IMG_TYPE(
        CL_COMPUTE_DTYPE_CHAR,
        input,
        SAMPLER,
        (int2)(out_c * out_height + out_h, out_n * shared_dim + w));
    CL_COMPUTE_DTYPE4 w0 = READ_IMG_TYPE(
        CL_COMPUTE_DTYPE_CHAR,
        weights,
        SAMPLER,
        (int2)(out_c * shared_dim + w, out_n * out_width + out_w));
    output0 = mad(v0, w0, output0);
  }

  CL_DTYPE4 out0;
  out0.x = CONVERT_TYPE_TO(output0.x, CL_DTYPE);
  out0.y = CONVERT_TYPE_TO(output0.y, CL_DTYPE);
  out0.z = CONVERT_TYPE_TO(output0.z, CL_DTYPE);
  out0.w = CONVERT_TYPE_TO(output0.w, CL_DTYPE);

  int2 out_pos0 = (int2)(out_c * out_width + out_w, out_nh);

  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR, output, out_pos0, out0 * CONVERT_TYPE_TO(scale, CL_DTYPE));
}
