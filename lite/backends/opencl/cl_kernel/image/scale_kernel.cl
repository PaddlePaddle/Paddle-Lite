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

__kernel void scale(__read_only image2d_t input,
                    __write_only image2d_t output,
                    __private float scale,
                    __private float bias){

  const int x = get_global_id(0); // image_width
  const int y = get_global_id(1); // image_height

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x, y));
  in = convert_float(scale) * in + convert_float(bias);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), in);
}
