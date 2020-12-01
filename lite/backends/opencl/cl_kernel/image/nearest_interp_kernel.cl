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


__kernel void nearest_interp(__read_only image2d_t input,
                             __write_only image2d_t output,
                             __private const float scale_h,
                             __private const float scale_w,
                             __private const int in_dims_h,
                             __private const int out_dims_h,
                             __private const int in_dims_w,
                             __private const int out_dims_w) {

  const int c = get_global_id(0);
  const int w = get_global_id(1);
  const int nh = get_global_id(2);

  int2 output_pos;
  output_pos.x = c * out_dims_w + w;
  output_pos.y = nh;

  int out_n = nh / out_dims_h;
  int out_h = nh % out_dims_h;

  int2 input_pos;
  input_pos.x = c * in_dims_w + w / scale_w;
  input_pos.y = out_n * in_dims_h + out_h / scale_h;

  CL_DTYPE4 input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(input_pos.x, input_pos.y));

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(output_pos.x , output_pos.y), input_data);
}
