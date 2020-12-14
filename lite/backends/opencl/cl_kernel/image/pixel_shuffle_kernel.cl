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
__kernel void pixel_shuffle(__read_only image2d_t input_image,
                            __write_only image2d_t output_image,
                            __private const int in_N,
                            __private const int in_C,
                            __private const int in_H,
                            __private const int in_W,
                            __private const int out_N,
                            __private const int out_C,
                            __private const int out_H,
                            __private const int out_W,
                            __private const int upscale_factor) {
  const int out_c4 = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_h = out_nh % out_H;
  int out_n = out_nh / out_H;

  int in_h = out_h / upscale_factor;
  int in_w = out_w / upscale_factor;
  int in_nh = out_n * in_H + in_h;

  CL_DTYPE4 res;
  int out_c;
  int in_c;
  CL_DTYPE4 in;
  int2 in_pos;

  out_c = out_c4 * 4 + 0;
  in_c = out_c * upscale_factor * upscale_factor +
         (out_h % upscale_factor) * upscale_factor + (out_w % upscale_factor);
  in_pos.x = (in_c / 4) * in_W + in_w;
  in_pos.y = in_nh;
  in = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, in_pos);
  if (in_c % 4 == 0) {
    res.x = in.x;
  } else if (in_c % 4 == 1) {
    res.x = in.y;
  } else if (in_c % 4 == 2) {
    res.x = in.z;
  } else if (in_c % 4 == 3) {
    res.x = in.w;
  }

  out_c = out_c4 * 4 + 1;
  in_c = out_c * upscale_factor * upscale_factor +
         (out_h % upscale_factor) * upscale_factor + (out_w % upscale_factor);
  in_pos.x = (in_c / 4) * in_W + in_w;
  in_pos.y = in_nh;
  in = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, in_pos);
  if (in_c % 4 == 0) {
    res.y = in.x;
  } else if (in_c % 4 == 1) {
    res.y = in.y;
  } else if (in_c % 4 == 2) {
    res.y = in.z;
  } else if (in_c % 4 == 3) {
    res.y = in.w;
  }

  out_c = out_c4 * 4 + 2;
  in_c = out_c * upscale_factor * upscale_factor +
         (out_h % upscale_factor) * upscale_factor + (out_w % upscale_factor);
  in_pos.x = (in_c / 4) * in_W + in_w;
  in_pos.y = in_nh;
  in = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, in_pos);
  if (in_c % 4 == 0) {
    res.z = in.x;
  } else if (in_c % 4 == 1) {
    res.z = in.y;
  } else if (in_c % 4 == 2) {
    res.z = in.z;
  } else if (in_c % 4 == 3) {
    res.z = in.w;
  }

  out_c = out_c4 * 4 + 3;
  in_c = out_c * upscale_factor * upscale_factor +
         (out_h % upscale_factor) * upscale_factor + (out_w % upscale_factor);
  in_pos.x = (in_c / 4) * in_W + in_w;
  in_pos.y = in_nh;
  in = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, in_pos);
  if (in_c % 4 == 0) {
    res.w = in.x;
  } else if (in_c % 4 == 1) {
    res.w = in.y;
  } else if (in_c % 4 == 2) {
    res.w = in.z;
  } else if (in_c % 4 == 3) {
    res.w = in.w;
  }

  int2 out_pos;
  out_pos.x = out_c4 * out_W + out_w;
  out_pos.y = out_nh;
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, out_pos, res);
}
