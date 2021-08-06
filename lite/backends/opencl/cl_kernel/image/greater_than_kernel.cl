/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

__kernel void greater_than(__read_only image2d_t input_x,
                           __private float input_y,
                           __write_only image2d_t output) {
  const int x = get_global_id(0);  // image_width
  const int y = get_global_id(1);  // image_height
  CL_DTYPE4 in_y = (CL_DTYPE4)((CL_DTYPE)(input_y),
                               (CL_DTYPE)(input_y),
                               (CL_DTYPE)(input_y),
                               (CL_DTYPE)(input_y));
  CL_DTYPE4 in_x = READ_IMG_TYPE(CL_DTYPE_CHAR, input_x, SAMPLER, (int2)(x, y));
  CL_DTYPE4 ones = (CL_DTYPE4)(1.0);
  CL_DTYPE4 out;
#ifdef CL_DTYPE_half
  short4 is_greater = in_x > in_y;
  out = as_half4(as_short4(ones) & is_greater);
#else
  int4 is_greater = in_x > in_y;
  out = as_float4(as_int4(ones) & is_greater);
#endif
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), out);
}
