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
                              __write_only image2d_t outputImage) {
     int x = get_global_id(0);
     int y = get_global_id(1);

     const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

     int2 coords;
     coords.x = x;
     coords.y = y;

     CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);
     CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords);
     CL_DTYPE4 output = activation_type4(in + biase);

     WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage,coords,output);
 }

__kernel void channel_add(__read_only image2d_t input,
                          __read_only image2d_t bias,
                          __write_only image2d_t outputImage,
                          int w) {
     int x = get_global_id(0);
     int y = get_global_id(1);

     const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
     int2 coords;
     coords.x = x;
     coords.y = y;

     int2 coords_bias;
     coords_bias.x = x % w;
     coords_bias.y = 0;

     CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);
     CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords_bias);
     CL_DTYPE4 output = in + (CL_DTYPE4)(biase.x);

#if 0 // enable to check input/output/bias
     printf("x:%d, y:%d\n", x, y);

     if (x == 0 && y == 0) {
         printf("w:%d\n", w);

         coords_bias.x = 0;
         coords_bias.y = 0;
         biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords_bias);
         printf("+ bias(%d,%d):%.2f %.2f %.2f %.2f\n", coords_bias.x, coords_bias.y, biase.x, biase.y, biase.z, biase.w);

         coords_bias.x = 1;
         coords_bias.y = 0;
         biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords_bias);
         printf("+ bias(%d,%d):%.2f %.2f %.2f %.2f\n", coords_bias.x, coords_bias.y, biase.x, biase.y, biase.z, biase.w);

         coords.x = 0;
         coords.y = 0;
         in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);
         printf(">>> input(%d,%d):%.2f %.2f %.2f %.2f\n", coords.x, coords.y, in.x, in.y, in.z, in.w);

         coords.x = 0;
         coords.y = 1;
         in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);
         printf(">>> input(%d,%d):%.2f %.2f %.2f %.2f\n", coords.x, coords.y, in.x, in.y, in.z, in.w);

         coords.x = 1;
         coords.y = 0;
         in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);
         printf(">>> input(%d,%d):%.2f %.2f %.2f %.2f\n", coords.x, coords.y, in.x, in.y, in.z, in.w);

         coords.x = 1;
         coords.y = 1;
         in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);
         printf(">>> input(%d,%d):%.2f %.2f %.2f %.2f\n", coords.x, coords.y, in.x, in.y, in.z, in.w);
     }
     printf("==> in(%d,%d):%.2f %.2f %.2f %.2f bias(%d,%d).x:%.2f %.2f %.2f out(%d,%d):%.2f %.2f %.2f %.2f\n",
            coords.x, coords.y, in.x, in.y, in.z, in.w,
            coords_bias.x, coords_bias.y, biase.x, biase.x, biase.x, biase.x,
            coords.x, coords.y, output.x, output.y, output.z, output.w);
#endif

     WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
 }

__kernel void width_add(__read_only image2d_t input,
                        __read_only image2d_t bias,
                        __write_only image2d_t outputImage,
                        int w) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int2 coords;
  coords.x = x;
  coords.y = y;

  int2 coords_bias;
  coords_bias.x = x % w;
  coords_bias.y = 0;

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, coords);
  CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, coords_bias);
  CL_DTYPE4 output;

  output.x = in.x + biase.x;
  output.y = in.y + biase.x;
  output.z = in.z + biase.x;
  output.w = in.w + biase.x;

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}
