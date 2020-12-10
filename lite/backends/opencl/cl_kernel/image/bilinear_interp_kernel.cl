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


__kernel void bilinear_interp(__read_only image2d_t input,
                             __write_only image2d_t output,
                             __private const float scale_h,
                             __private const float scale_w,
                             __private const float align_delta,
                             __private const int in_dims_h,
                             __private const int in_dims_w,
                             __private const int out_dims_h,
                             __private const int out_dims_w){
    const int c = get_global_id(0);
    const int w = get_global_id(1);
    const int nh = get_global_id(2);

    int2 output_pos;
    output_pos.x = c * out_dims_w + w;
    output_pos.y = nh;

    // calculate center pixel's pos
    int out_n = nh / out_dims_h;
    int out_h = nh % out_dims_h;
    float center_w = (w + align_delta)  * scale_w - align_delta;
    float center_h = (out_h + align_delta) * scale_h - align_delta;

    int floor_w = (int)center_w;
    int floor_h = (int)center_h;
    int ceil_w = floor_w + 1;
    int ceil_h = floor_h + 1;
    if (floor_w < 0){
        floor_w = 0;
    }
    if (floor_h < 0){
        floor_h = 0;
    }
    if (ceil_w > in_dims_w - 1) {
        ceil_w = in_dims_w - 1;
    }
    if (ceil_h > in_dims_h - 1) {
        ceil_h = in_dims_h- 1;
    }
    CL_DTYPE wight0_w = center_w - floor_w;
    CL_DTYPE wight0_h = center_h - floor_h;
    CL_DTYPE wight1_w = 1.0 - wight0_w;
    CL_DTYPE wight1_h = 1.0 - wight0_h;

    // get left up pixel data
    int2 left_up;
    left_up.x = c * in_dims_w + floor_w;
    left_up.y = out_n * in_dims_h + ceil_h;
    CL_DTYPE4 left_up_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, left_up);


    // get left down pixel data
    int2 left_down;
    left_down.x = c * in_dims_w + floor_w;
    left_down.y = out_n * in_dims_h + floor_h;
    CL_DTYPE4 left_down_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, left_down);

    // get right up pixel data
    int2 right_up;
    right_up.x = c * in_dims_w + ceil_w;
    right_up.y = out_n * in_dims_h + ceil_h;
    CL_DTYPE4 right_up_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, right_up);

    // get right down pixel's data
    int2 right_down;
    right_down.x = c * in_dims_w + ceil_w;
    right_down.y = out_n * in_dims_h + floor_h;
    CL_DTYPE4 right_down_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, right_down);

    // calculate output data
    CL_DTYPE4 out = (left_down_data * wight1_w + right_down_data * wight0_w) * wight1_h
            + (left_up_data * wight1_w + right_up_data * wight0_w) * wight0_h;

    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, out);
}
