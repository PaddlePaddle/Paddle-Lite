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

__kernel void elementwise_add(__read_only image2d_t input,
                              __read_only image2d_t bias,
                              __write_only image2d_t outputImage,
                              int h,
                              int w) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 coords;
  coords.x = x;
  coords.y = y;

#ifdef BROADCAST
  int c_blk = x / w;
  int n_blk = y / h;
  CL_DTYPE4 in =
      READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(c_blk, n_blk));
#else
  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coords);
#endif
  CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, coords);
  CL_DTYPE4 alpha;
  CL_DTYPE4 output = activation_type4(in + biase, alpha);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}

__kernel void elementwise_add_n1h1w1(__read_only image2d_t input,
                                     __read_only image2d_t bias,
                                     __write_only image2d_t outputImage,
                                     int h,
                                     int w) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 coords;
  coords.x = x;
  coords.y = y;

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coords);

  CL_DTYPE4 cur_bias =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(x / w, 0));
  CL_DTYPE4 alpha;
  CL_DTYPE4 output = activation_type4(in + cur_bias, alpha);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}

__kernel void channel_add(__read_only image2d_t input,
                          __read_only image2d_t bias,
                          __write_only image2d_t outputImage,
                          int w,
                          int opt) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 coords;
  coords.x = x;
  coords.y = y;

  int2 coords_bias;
  coords_bias.x = (opt == 1) ? 0 : x % w;
  coords_bias.y = 0;

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coords);
  CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, coords_bias);
  CL_DTYPE4 output = in + (CL_DTYPE4)(biase.x);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}

__kernel void width_add(__read_only image2d_t input,
                        __read_only image2d_t bias,
                        __write_only image2d_t outputImage,
                        int w) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 coords;
  coords.x = x;
  coords.y = y;

  int2 coords_bias;
  coords_bias.x = x % w;
  coords_bias.y = 0;

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coords);
  CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, coords_bias);
  CL_DTYPE4 output;

  output.x = in.x + biase.x;
  output.y = in.y + biase.x;
  output.z = in.z + biase.x;
  output.w = in.w + biase.x;

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}
