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

inline CL_DTYPE compute_reshape_output(__read_only image2d_t input_image,
                                       int out_n,
                                       int out_c,
                                       int out_h,
                                       int out_w,
                                       int in_Stride0,
                                       int in_Stride1,
                                       int in_Stride2,
                                       int out_Stride0,
                                       int out_Stride1,
                                       int out_Stride2,
                                       int in_W,
                                       int in_H) {
  int count =
      out_n * out_Stride2 + out_c * out_Stride1 + out_h * out_Stride0 + out_w;
  int in_n = count / in_Stride2;
  count = count % in_Stride2;
  int in_c = count / in_Stride1;
  int in_h = (count % in_Stride1) / in_Stride0;
  int in_w = (count % in_Stride1) % in_Stride0;

  int2 input_pos;
  input_pos.x = (in_c / 4) * in_W + in_w;
  input_pos.y = in_n * in_H + in_h;
  CL_DTYPE4 input =
      READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, input_pos);
  CL_DTYPE out;
  if (in_c % 4 == 0) {
    out = input.x;
  } else if (in_c % 4 == 1) {
    out = input.y;
  } else if (in_c % 4 == 2) {
    out = input.z;
  } else {
    out = input.w;
  }
  return out;
}

__kernel void reshape(__read_only image2d_t input_image,
                      __write_only image2d_t output_image,
                      __private const int out_C,
                      __private const int out_H,
                      __private const int out_W,
                      __private const int in_W,
                      __private const int in_H,
                      __private const int in_Stride0,
                      __private const int in_Stride1,
                      __private const int in_Stride2,
                      __private const int out_Stride0,
                      __private const int out_Stride1,
                      __private const int out_Stride2) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const int out_n = out_nh / out_H;
  const int out_h = out_nh % out_H;

  const int out_c0 = out_c * 4;
  const int out_c1 = out_c * 4 + 1;
  const int out_c2 = out_c * 4 + 2;
  const int out_c3 = out_c * 4 + 3;

  int2 output_pos;
  output_pos.x = out_c * out_W + out_w;
  output_pos.y = out_nh;
  CL_DTYPE4 output = (CL_DTYPE4)(0.0f);

  output.x = compute_reshape_output(input_image,
                                    out_n,
                                    out_c0,
                                    out_h,
                                    out_w,
                                    in_Stride0,
                                    in_Stride1,
                                    in_Stride2,
                                    out_Stride0,
                                    out_Stride1,
                                    out_Stride2,
                                    in_W,
                                    in_H);

  if (out_C - out_c * 4 >= 2) {
    output.y = compute_reshape_output(input_image,
                                      out_n,
                                      out_c1,
                                      out_h,
                                      out_w,
                                      in_Stride0,
                                      in_Stride1,
                                      in_Stride2,
                                      out_Stride0,
                                      out_Stride1,
                                      out_Stride2,
                                      in_W,
                                      in_H);
  }
  if (out_C - out_c * 4 >= 3) {
    output.z = compute_reshape_output(input_image,
                                      out_n,
                                      out_c2,
                                      out_h,
                                      out_w,
                                      in_Stride0,
                                      in_Stride1,
                                      in_Stride2,
                                      out_Stride0,
                                      out_Stride1,
                                      out_Stride2,
                                      in_W,
                                      in_H);
  }
  if (out_C - out_c * 4 >= 4) {
    output.w = compute_reshape_output(input_image,
                                      out_n,
                                      out_c3,
                                      out_h,
                                      out_w,
                                      in_Stride0,
                                      in_Stride1,
                                      in_Stride2,
                                      out_Stride0,
                                      out_Stride1,
                                      out_Stride2,
                                      in_W,
                                      in_H);
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output);
}
