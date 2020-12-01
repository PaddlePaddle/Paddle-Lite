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

__kernel void expend_c1(__private const int OUT_C,
                        __private const int OUT_W,
                        __private const int OUT_NH,

                        __private const int IN_C,
                        __private const int IN_W,
                        __private const int IN_NH,

                        __private const int input_width,  /* of one block */
                        __private const int input_height, /* of one block */
                        __private const int output_width,
                        __private const int output_height,

                        __read_only image2d_t input,
                        __write_only image2d_t output,
                        __private const int n_times,
                        __private const int c_times,
                        __private const int h_times,
                        __private const int w_times) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  if (out_c >= OUT_C || out_w >= OUT_W || out_nh >= OUT_NH) {
    return;
  }

  const int out_n = out_nh / output_height;
  const int out_h = out_nh % output_height;
  const int in_c = 0;
  const int in_w = out_w / w_times;
  const int in_h = out_h / h_times;
  const int in_n = out_n / n_times;
  const int in_nh = in_n * input_height + in_h;

  int2 output_pos = (int2)(out_c * OUT_W + out_w, out_nh);
  int2 input_pos = (int2)(in_w, in_nh);

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);
  in.y = 0;
  in.z = 0;
  in.w = 0;
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, in);
}

__kernel void expend_c2(__private const int OUT_C,
                        __private const int OUT_W,
                        __private const int OUT_NH,

                        __private const int IN_C,
                        __private const int IN_W,
                        __private const int IN_NH,

                        __private const int input_width,  /* of one block */
                        __private const int input_height, /* of one block */
                        __private const int output_width,
                        __private const int output_height,

                        __read_only image2d_t input,
                        __write_only image2d_t output,
                        __private const int n_times,
                        __private const int c_times,
                        __private const int h_times,
                        __private const int w_times) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  if (out_c >= OUT_C || out_w >= OUT_W || out_nh >= OUT_NH) {
    return;
  }

  const int out_n = out_nh / output_height;
  const int out_h = out_nh % output_height;
  const int in_c = 0;
  const int in_w = out_w / w_times;
  const int in_h = out_h / h_times;
  const int in_n = out_n / n_times;
  const int in_nh = in_n * input_height + in_h;

  int2 output_pos = (int2)(out_c * OUT_W + out_w, out_nh);
  int2 input_pos = (int2)(in_w, in_nh);

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);
  in.z = 0;
  in.w = 0;
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, in);
}

__kernel void expend_c3(__private const int OUT_C,
                        __private const int OUT_W,
                        __private const int OUT_NH,

                        __private const int IN_C,
                        __private const int IN_W,
                        __private const int IN_NH,

                        __private const int input_width,  /* of one block */
                        __private const int input_height, /* of one block */
                        __private const int output_width,
                        __private const int output_height,

                        __read_only image2d_t input,
                        __write_only image2d_t output,
                        __private const int n_times,
                        __private const int c_times,
                        __private const int h_times,
                        __private const int w_times) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  if (out_c >= OUT_C || out_w >= OUT_W || out_nh >= OUT_NH) {
    return;
  }

  const int out_n = out_nh / output_height;
  const int out_h = out_nh % output_height;
  const int in_c = 0;
  const int in_w = out_w / w_times;
  const int in_h = out_h / h_times;
  const int in_n = out_n / n_times;
  const int in_nh = in_n * input_height + in_h;

  int2 output_pos = (int2)(out_c * OUT_W + out_w, out_nh);
  int2 input_pos = (int2)(in_w, in_nh);

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);
  in.w = 0;
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, in);
}
__kernel void expend_c4(__private const int OUT_C,
                        __private const int OUT_W,
                        __private const int OUT_NH,

                        __private const int IN_C,
                        __private const int IN_W,
                        __private const int IN_NH,

                        __private const int input_width,  /* of one block */
                        __private const int input_height, /* of one block */
                        __private const int output_width,
                        __private const int output_height,

                        __read_only image2d_t input,
                        __write_only image2d_t output,
                        __private const int n_times,
                        __private const int c_times,
                        __private const int h_times,
                        __private const int w_times) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  if (out_c >= OUT_C || out_w >= OUT_W || out_nh >= OUT_NH) {
    return;
  }

  const int out_n = out_nh / output_height;
  const int out_h = out_nh % output_height;
  const int in_c = 0;
  const int in_w = out_w / w_times;
  const int in_h = out_h / h_times;
  const int in_n = out_n / n_times;
  const int in_nh = in_n * input_height + in_h;

  int2 output_pos = (int2)(out_c * OUT_W + out_w, out_nh);
  int2 input_pos = (int2)(in_w, in_nh);

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, in);
}

__kernel void expend_cn(__private const int OUT_C,
                        __private const int OUT_W,
                        __private const int OUT_NH,

                        __private const int IN_C,
                        __private const int IN_W,
                        __private const int IN_NH,

                        __private const int input_width,  /* of one block */
                        __private const int input_height, /* of one block */
                        __private const int output_width,
                        __private const int output_height,

                        __read_only image2d_t input,
                        __write_only image2d_t output,
                        __private const int n_times,
                        __private const int c_times,
                        __private const int h_times,
                        __private const int w_times) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  if (out_c >= OUT_C || out_w >= OUT_W || out_nh >= OUT_NH) {
    return;
  }

  const int out_n = out_nh / output_height;
  const int out_h = out_nh % output_height;
  const int in_c = out_c;
  const int in_w = out_w / w_times;
  const int in_h = out_h / h_times;
  const int in_n = out_n / n_times;
  const int in_nh = in_n * input_height + in_h;

  int2 output_pos = (int2)(out_c * OUT_W + out_w, out_nh);
  int2 input_pos = (int2)(in_c * IN_W + in_w, in_nh);

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, in);
}
