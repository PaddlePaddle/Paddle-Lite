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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void scale(__read_only image2d_t input,
                    __write_only image2d_t output,
                    __private float scale,
                    __private float bias,
                    __private int out_width){

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;

  int pos_x = mad24(out_c, out_width, out_w);
  half4 in = read_imageh(input, sampler, (int2)(pos_x, out_nh));
  in = convert_half(scale) * in + convert_half(bias);
  write_imageh(output, (int2)(pos_x, out_nh), in);
}
