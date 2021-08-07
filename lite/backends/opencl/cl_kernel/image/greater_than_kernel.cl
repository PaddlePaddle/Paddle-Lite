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
  CL_DTYPE4 in_x = READ_IMG_TYPE(CL_DTYPE_CHAR, input_x, SAMPLER, (int2)(x, y));
  CL_DTYPE4 out;
  if (in_x.x > ((CL_DTYPE)(input_y))) {
    out.x = ((CL_DTYPE)(1.0));
  } else {
    out.x = ((CL_DTYPE)(0.0));
  }
  if (in_x.y > ((CL_DTYPE)(input_y))) {
    out.y = ((CL_DTYPE)(1.0));
  } else {
    out.y = ((CL_DTYPE)(0.0));
  }
  if (in_x.z > ((CL_DTYPE)(input_y))) {
    out.z = ((CL_DTYPE)(1.0));
  } else {
    out.z = ((CL_DTYPE)(0.0));
  }
  if (in_x.w > ((CL_DTYPE)(input_y))) {
    out.w = ((CL_DTYPE)(1.0));
  } else {
    out.w = ((CL_DTYPE)(0.0));
  }
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(x, y), out);
}
