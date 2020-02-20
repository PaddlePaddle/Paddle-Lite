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

__kernel void elementwise_mul(__global image2d_t input,
                              __global image2d_t bias,
                              __write_only image2d_t outputImage) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int2 coords;
  coords.x = x;
  coords.y = y;
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);
  CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords);
  CL_DTYPE4 output = in * biase;
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}


__kernel void channel_mul_d1(__read_only image2d_t input,
                             __read_only image2d_t bias,
                             __write_only image2d_t outputImage,
							 int w) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int2 coords;
  coords.x = x;
  coords.y = y;

  int2 coords_bias;
  coords_bias.x = x % w;
  coords_bias.y = 0;

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);
  CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords_bias);
  CL_DTYPE4 output = in * (CL_DTYPE4)(biase.x);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}


// #define DEBUG
__kernel void channel_mul_d2_nc(__read_only image2d_t input,
                                __read_only image2d_t bias,
                                __write_only image2d_t outputImage,
	   						    int w) {
  int x = get_global_id(0);
  int y = get_global_id(1);

#ifdef DEBUG
  printf("x:%d y:%d\n", x, y);
#endif

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int2 coords;
  coords.x = x;
  coords.y = y;
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);

  int2 coords_bias0 = (int2)(x / w * 4, 0);
  int2 coords_bias1 = (int2)(x / w * 4 + 1, 0);
  int2 coords_bias2 = (int2)(x / w * 4 + 2, 0);
  int2 coords_bias3 = (int2)(x / w * 4 + 3, 0);

  CL_DTYPE4 b0 = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords_bias0);
  CL_DTYPE4 b1 = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords_bias1);
  CL_DTYPE4 b2 = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords_bias2);
  CL_DTYPE4 b3 = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords_bias3);

  CL_DTYPE4 biase = {b0.x, b1.x, b2.x, b3.x};
  CL_DTYPE4 output = mad(in, biase, 0);

#ifdef DEBUG
  if (x == 0 && y == 0) {
    printf("w:%d\n", w);

    printf("biase:%.1f %.1f %.1f %.1f\n", biase.x, biase.y, biase.z, biase.w);
    printf("output:%.1f %.1f %.1f %.1f\n", output.x, output.y, output.z, output.w);

    coords.x = 0;
    coords.y = 0;
    in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);
    printf("in(%d,%d):%.2f %.2f %.2f %.2f\n", coords.x, coords.y, in.x, in.y, in.z, in.w);
    coords.x = 0;
    coords.y = 1;
    in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);
    printf("in(%d,%d):%.2f %.2f %.2f %.2f\n", coords.x, coords.y, in.x, in.y, in.z, in.w);
    coords.x = 1;
    coords.y = 0;
    in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);
    printf("in(%d,%d):%.2f %.2f %.2f %.2f\n", coords.x, coords.y, in.x, in.y, in.z, in.w);
    coords.x = 1;
    coords.y = 1;
    in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);
    printf("in(%d,%d):%.2f %.2f %.2f %.2f\n", coords.x, coords.y, in.x, in.y, in.z, in.w);

    coords_bias.x = 0;
    coords_bias.y = 0;
    biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords_bias);
    printf("biase(%d,%d):%.2f %.2f %.2f %.2f\n", coords_bias.x, coords_bias.y, biase.x, biase.y, biase.z, biase.w);
    coords_bias.x = 1;
    coords_bias.y = 0;
    biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords_bias);
    printf("biase(%d,%d):%.2f %.2f %.2f %.2f\n", coords_bias.x, coords_bias.y, biase.x, biase.y, biase.z, biase.w);
    coords_bias.x = 2;
    coords_bias.y = 0;
    biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords_bias);
    printf("biase(%d,%d):%.2f %.2f %.2f %.2f\n", coords_bias.x, coords_bias.y, biase.x, biase.y, biase.z, biase.w);
  }
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}


__kernel void channel_mul_d2_hw(__read_only image2d_t input,
                                __read_only image2d_t bias,
                                __write_only image2d_t outputImage,
                                int w,
                                int h) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int2 coords;
  coords.x = x;
  coords.y = y;

  int2 coords_bias;
  coords_bias.x = x % w;
  coords_bias.y = y % h;

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);
  CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords_bias);
  CL_DTYPE4 output = in * (CL_DTYPE4)(biase.x);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}


__kernel void channel_mul_d4(__read_only image2d_t input,
                             __read_only image2d_t bias,
                             __write_only image2d_t outputImage,
							 int w) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int2 coords;
  coords.x = x;
  coords.y = y;

  int2 coords_bias;
  coords_bias.x = x / w;
  coords_bias.y = 0;

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);
  CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords_bias);
  CL_DTYPE4 output = in * biase;

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}

