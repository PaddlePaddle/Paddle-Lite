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

__kernel void elementwise_mul(__read_only image2d_t input,
                              __read_only image2d_t bias,
                              __write_only image2d_t outputImage) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 coords;
  coords.x = x;
  coords.y = y;

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coords);
  CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, coords);

#ifdef FUSE_SCALE
  CL_DTYPE4 output =
      fuse_scale(in * biase, SCALE_SLOPE, SCALE_BIAS, SCALE_ALPHA);
#else
  CL_DTYPE4 output = in * biase;
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}

__kernel void channel_mul(__read_only image2d_t input,
                          __read_only image2d_t bias,
                          __write_only image2d_t outputImage,
                          int w) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 coords;
  coords.x = x;
  coords.y = y;

  int2 coords_bias;
  coords_bias.x = x / w;
  coords_bias.y = 0;

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coords);
  CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, coords_bias);

#ifdef FUSE_SCALE
  CL_DTYPE4 output =
      fuse_scale(in * biase, SCALE_SLOPE, SCALE_BIAS, SCALE_ALPHA);
#else
  CL_DTYPE4 output = in * biase;
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}

__kernel void channel_mul_d1(__read_only image2d_t input,
                             __read_only image2d_t bias,
                             __write_only image2d_t outputImage,
                             int x_w,
                             int opt) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 coords;
  coords.x = x;
  coords.y = y;

  int2 coords_bias;
  coords_bias.x = (opt == 1) ? 0 : (x % x_w);
  coords_bias.y = 0;

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coords);
  CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, coords_bias);
  CL_DTYPE4 output = in * (CL_DTYPE4)(biase.x);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}

// etc : 1 1 1 72
// run time Y  [value,0,0,0] * 72
__kernel void channel_mul_d2(__read_only image2d_t input,
                             __read_only image2d_t bias,
                             __write_only image2d_t outputImage,
                             int w) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 coords;
  coords.x = x;
  coords.y = y;

  int2 coords_bias0;
  int2 coords_bias1;
  int2 coords_bias2;
  int2 coords_bias3;
  /*  if (x == 0 && y == 0) {
      CL_DTYPE4 b = (CL_DTYPE4){0, 0, 0, 0};
  #define PPI(j, k)                                                          \
    b = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2){j, k}); \
    printf("bias(%d,%d)={ %f , %f , %f , %f }\n ", j, k, convert_float(b.x), \
           convert_float(b.y), convert_float(b.z), convert_float(b.w));
      for (int i = 0; i < 73; ++i) {
        PPI(i, 0);
      }
  #undef PPI
    }*/
  coords_bias0.x = x / w * 4;
  coords_bias0.y = 0;
  coords_bias1.x = x / w * 4 + 1;
  coords_bias1.y = 0;
  coords_bias2.x = x / w * 4 + 2;
  coords_bias2.y = 0;
  coords_bias3.x = x / w * 4 + 3;
  coords_bias3.y = 0;
  CL_DTYPE4 biase0 = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, coords_bias0);
  CL_DTYPE4 biase1 = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, coords_bias1);
  CL_DTYPE4 biase2 = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, coords_bias2);
  CL_DTYPE4 biase3 = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, coords_bias3);
  /*  if (x == 0 && y == 0) {
      printf("bias0={ %f , %f , %f , %f }\n ",
             convert_float(biase0.x), convert_float(biase0.y),
             convert_float(biase0.z), convert_float(biase0.w));
      printf("bias1={ %f , %f , %f , %f }\n ",
             convert_float(biase1.x), convert_float(biase1.y),
             convert_float(biase1.z), convert_float(biase1.w));
      printf("bias2={ %f , %f , %f , %f }\n ",
             convert_float(biase2.x), convert_float(biase2.y),
             convert_float(biase2.z), convert_float(biase2.w));
      printf("bias3={ %f , %f , %f , %f }\n ",
             convert_float(biase3.x), convert_float(biase3.y),
             convert_float(biase3.z), convert_float(biase3.w));
    }*/
  CL_DTYPE4 biase = {biase0.x, biase1.x, biase2.x, biase3.x};
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coords);

#ifdef FUSE_SCALE
  CL_DTYPE4 output =
      fuse_scale(mad(in, biase, 0), SCALE_SLOPE, SCALE_BIAS, SCALE_ALPHA);
#else
  CL_DTYPE4 output = mad(in, biase, 0);
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}

// c 1 1
__kernel void channel_mul_d3(__read_only image2d_t input,
                             __read_only image2d_t bias,
                             __write_only image2d_t outputImage,
                             int w) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 coords;
  coords.x = x;
  coords.y = y;

  int2 coords_bias;
  coords_bias.x = x / w;
  coords_bias.y = 0;

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coords);
  CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, coords_bias);

#ifdef FUSE_SCALE
  CL_DTYPE4 output =
      fuse_scale(in * biase, SCALE_SLOPE, SCALE_BIAS, SCALE_ALPHA);
#else
  CL_DTYPE4 output = in * biase;
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}

__kernel void channel_mul_d4(__read_only image2d_t input,
                             __read_only image2d_t bias,
                             __write_only image2d_t outputImage,
                             int w) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 coords;
  coords.x = x;
  coords.y = y;

  int2 coords_bias;
  coords_bias.x = x / w;
  coords_bias.y = 0;

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coords);
  CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, coords_bias);

#ifdef FUSE_SCALE
  CL_DTYPE4 output =
      fuse_scale(in * biase, SCALE_SLOPE, SCALE_BIAS, SCALE_ALPHA);
#else
  CL_DTYPE4 output = in * biase;
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}
