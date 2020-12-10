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

// function name "sin" is not allowed by adreno
__kernel void trigonometric_sin(__read_only image2d_t input,
                                __write_only image2d_t output) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  in.x = native_sin(in.x);
  in.y = native_sin(in.y);
  in.z = native_sin(in.z);
  in.w = native_sin(in.w);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}

__kernel void trigonometric_cos(__read_only image2d_t input,
                                __write_only image2d_t output) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  in.x = native_cos(in.x);
  in.y = native_cos(in.y);
  in.z = native_cos(in.z);
  in.w = native_cos(in.w);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}

__kernel void trigonometric_tan(__read_only image2d_t input,
                                __write_only image2d_t output) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  in.x = native_tan(in.x);
  in.y = native_tan(in.y);
  in.z = native_tan(in.z);
  in.w = native_tan(in.w);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}

__kernel void trigonometric_atan(__read_only image2d_t input,
                                 __write_only image2d_t output) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  in.x = atan(in.x);
  in.y = atan(in.y);
  in.z = atan(in.z);
  in.w = atan(in.w);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}

__kernel void trigonometric_asin(__read_only image2d_t input,
                                 __write_only image2d_t output) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  in.x = asin(in.x);
  in.y = asin(in.y);
  in.z = asin(in.z);
  in.w = asin(in.w);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}

__kernel void trigonometric_acos(__read_only image2d_t input,
                                 __write_only image2d_t output) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x, y));
  in.x = acos(in.x);
  in.y = acos(in.y);
  in.z = acos(in.z);
  in.w = acos(in.w);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}
