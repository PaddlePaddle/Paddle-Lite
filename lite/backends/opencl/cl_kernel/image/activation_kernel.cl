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

__kernel void relu(__read_only image2d_t input, __write_only image2d_t output) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  in = max((CL_DTYPE4)(0.0f), in);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}

__kernel void abs_act(__read_only image2d_t input,
                      __write_only image2d_t output) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  in = fabs(in);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}

__kernel void log_act(__read_only image2d_t input,
                      __write_only image2d_t output) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  in = log(in);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}
__kernel void relu6(__read_only image2d_t input,
                    __write_only image2d_t output) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  in = max((CL_DTYPE4)(0.0f), in);
  in = min((CL_DTYPE4)(6.0f), in);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}

__kernel void sigmoid(__read_only image2d_t input,
                      __write_only image2d_t output) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  CL_DTYPE4 out;

  out.x = (CL_DTYPE)(1.0f / (1.0f + pow(2.71828182f, -1.0f * (float)(in.x))));
  out.y = (CL_DTYPE)(1.0f / (1.0f + pow(2.71828182f, -1.0f * (float)(in.y))));
  out.z = (CL_DTYPE)(1.0f / (1.0f + pow(2.71828182f, -1.0f * (float)(in.z))));
  out.w = (CL_DTYPE)(1.0f / (1.0f + pow(2.71828182f, -1.0f * (float)(in.w))));

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), out);
}

__kernel void hard_sigmoid(__read_only image2d_t input,
                           __write_only image2d_t output,
                           __private const float value_offset,
                           __private const float scale) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  CL_DTYPE4 out = clamp(in * (CL_DTYPE4)(scale) + (CL_DTYPE4)(value_offset),
                        (CL_DTYPE4)(0.0),
                        (CL_DTYPE4)(1.0));

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), out);
}

__kernel void leaky_relu(__read_only image2d_t input,
                         __write_only image2d_t output,
                         __private const float scale) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  CL_DTYPE4 s_val = CONVERT_TYPE_TO(scale, CL_DTYPE) * in;
  if (in.x < 0.0f) {
    in.x = s_val.x;
  }
  if (in.y < 0.0f) {
    in.y = s_val.y;
  }
  if (in.z < 0.0f) {
    in.z = s_val.z;
  }
  if (in.w < 0.0f) {
    in.w = s_val.w;
  }
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}

__kernel void prelu_channel(__read_only image2d_t input,
                            __write_only image2d_t output,
                            int width,
                            __read_only image2d_t alpha) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int c_idx = x / width;
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  CL_DTYPE4 v_alpha =
      READ_IMG_TYPE(CL_DTYPE_CHAR, alpha, SAMPLER, (int2)(c_idx, 0));
  in = select(in, in * v_alpha, in < ((CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f)));
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}

__kernel void prelu_element(__read_only image2d_t input,
                            __write_only image2d_t output,
                            int height,
                            __read_only image2d_t alpha) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int h_idx = y % height;

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  CL_DTYPE4 v_alpha =
      READ_IMG_TYPE(CL_DTYPE_CHAR, alpha, SAMPLER, (int2)(x, h_idx));
  in = select(in, in * v_alpha, in < ((CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f)));
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}

__kernel void tanh_act(__read_only image2d_t input,
                       __write_only image2d_t output) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  CL_DTYPE4 out = (exp(in) - exp(-in)) / (exp(in) + exp(-in));
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), out);
}

__kernel void exp_act(__read_only image2d_t input,
                      __write_only image2d_t output) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  CL_DTYPE4 out = exp(in);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), out);
}

__kernel void swish(__read_only image2d_t input,
                    __write_only image2d_t output,
                    __private const float scale) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  CL_DTYPE4 out = in / (1 + exp(-(CL_DTYPE)scale * in));
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), out);
}

__kernel void hard_swish(__read_only image2d_t input,
                         __write_only image2d_t output,
                         __private const float threshold,
                         __private const float scale,
                         __private const float offset) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  CL_DTYPE4 in0 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));

  CL_DTYPE4 in = in0 + (CL_DTYPE4)(offset, offset, offset, offset);
  in = max((CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f), in);
  in = min((CL_DTYPE4)(threshold, threshold, threshold, threshold), in);
  in = in0 * in / (CL_DTYPE4)(scale, scale, scale, scale);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}

__kernel void sqrt_func(__read_only image2d_t input,
                        __write_only image2d_t output) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  in.x = sqrt(in.x);
  in.y = sqrt(in.y);
  in.z = sqrt(in.z);
  in.w = sqrt(in.w);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}

__kernel void rsqrt_func(__read_only image2d_t input,
                         __write_only image2d_t output) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  in.x = rsqrt(in.x);
  in.y = rsqrt(in.y);
  in.z = rsqrt(in.z);
  in.w = rsqrt(in.w);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}

__kernel void square_func(__read_only image2d_t input,
                          __write_only image2d_t output) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  in.x = pown(in.x, 2);
  in.y = pown(in.y, 2);
  in.z = pown(in.z, 2);
  in.w = pown(in.w, 2);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}

__kernel void gelu(__read_only image2d_t input, __write_only image2d_t output) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));

  const float4 in_f32 = convert_float4(in);
  in.x = (CL_DTYPE)(0.5f * in_f32.x * (1.0f + erf(in_f32.x / 1.41421f)));
  in.y = (CL_DTYPE)(0.5f * in_f32.y * (1.0f + erf(in_f32.y / 1.41421f)));
  in.z = (CL_DTYPE)(0.5f * in_f32.z * (1.0f + erf(in_f32.z / 1.41421f)));
  in.w = (CL_DTYPE)(0.5f * in_f32.w * (1.0f + erf(in_f32.w / 1.41421f)));

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}
