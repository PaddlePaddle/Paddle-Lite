/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

__kernel void slice(__read_only image2d_t input,
                    __write_only image2d_t output,
                    __private const int start,
                    __private const int end,
                    __private const int dims_w) {
  const int c = get_global_id(0);
  const int w = get_global_id(1);
  const int nh = get_global_id(2);

  int2 output_pos;
  output_pos.x = c * dims_w + w;
  output_pos.y = nh;

  int2 input_pos;
  CL_DTYPE4 input_data;
  CL_DTYPE4 output_data;

  if (start % 4 == 0) {
    input_pos.x = (4 * c + start) / 4 * dims_w + w;
    input_pos.y = nh;
    input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);
    output_data = input_data;
  } else if (start % 4 == 1) {
    input_pos.x = (4 * c + start) / 4 * dims_w + w;
    input_pos.y = nh;
    input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);
    output_data.x = input_data.y;
    output_data.y = input_data.z;
    output_data.z = input_data.w;
    input_pos.x = input_pos.x + dims_w;
    input_pos.y = nh;
    input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);
    output_data.w = input_data.x;
  } else if (start % 4 == 2) {
    input_pos.x = (4 * c + start) / 4 * dims_w + w;
    input_pos.y = nh;
    input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);
    output_data.x = input_data.z;
    output_data.y = input_data.w;
    input_pos.x = input_pos.x + dims_w;
    input_pos.y = nh;
    input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);
    output_data.z = input_data.x;
    output_data.w = input_data.y;
  } else if (start % 4 == 3) {
    input_pos.x = (4 * c + start) / 4 * dims_w + w;
    input_pos.y = nh;
    input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);
    output_data.x = input_data.w;
    input_pos.x = input_pos.x + dims_w;
    input_pos.y = nh;
    input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, input_pos);
    output_data.y = input_data.x;
    output_data.z = input_data.y;
    output_data.w = input_data.z;
  }
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, output_data);
}
