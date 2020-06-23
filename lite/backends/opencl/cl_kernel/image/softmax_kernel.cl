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

__kernel void softmax(__read_only image2d_t input_image,
                      __write_only image2d_t output_image,
                      __private const int out_W) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const int in_c = out_c;
  const int in_w = out_w;
  const int in_nh = out_nh;

  int2 input_pos;
  int2 output_pos;

  input_pos.x = in_c * out_W + in_w;
  input_pos.y = in_nh;

  output_pos.x = out_c * out_W + out_w;
  output_pos.y = out_nh;

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  CL_DTYPE4 input_max = 0.0f;
  CL_DTYPE4 input_tmp;
  for (int i = 0; i < out_W; i++) {
    input_tmp = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input_image, sampler, (int2)(in_c * out_W + i, in_nh));
    input_max = max(input_max, input_tmp);
  }

  CL_DTYPE4 sum = (CL_DTYPE4)0.0f;
  for (int i = 0; i < out_W; i++) {
    input_tmp = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input_image, sampler, (int2)(in_c * out_W + i, in_nh));
    sum += exp(input_tmp - input_max);
  }

  CL_DTYPE4 input =
      READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler, input_pos);
  CL_DTYPE4 output = exp(input - input_max) / sum;
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output);
}
