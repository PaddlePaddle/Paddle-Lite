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

#include "cl_common.h"

__kernel void conv_transpose(__private const int input_c_block,
                                              __private const int input_width,/* of one block */
                                              __private const int input_height,/* of one block */
                                              __private const int output_width,
                                              __private const int output_height,
                                              __read_only image2d_t input_image,
                                              __read_only image2d_t filter,
                                              __write_only image2d_t output_image) {

    const int out_c = get_global_id(0);
    const int in_w = get_global_id(1);
    const int in_nh = get_global_id(2);
    const int n = in_nh / input_height;
    const int h = in_nh % input_height;

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                              CLK_ADDRESS_CLAMP          |
                              CLK_FILTER_NEAREST;

    half4 input1, input2, input3, input4;
    half4 output1 = 0.0f, output2 = 0.0f, output3 = 0.0f, output4 = 0.0f;
    half4 w = 0.0f;
    int2 pos_in;
    for (int i = 0; i < input_c_block; i += 1) {
      pos_in = (int2)(mad24(i, input_width, in_w), in_nh);
      input1 = select(read_imageh(input_image, sampler,
                                               (int2)(pos_in.x, pos_in.y)),
                                               (half4)(0.0f),
                                               (ushort4)((in_w < 0 || h < 0 || in_w >= input_width || h >= input_height) << 15));
      input2 = select(read_imageh(input_image, sampler,
                                                     (int2)(pos_in.x + 1, pos_in.y)),
                                                     (half4)(0.0f),
                                                     (ushort4)((in_w + 1 < 0 || h < 0 || in_w + 1 >= input_width || h >= input_height) << 15));
      input3 = select(read_imageh(input_image, sampler,
                                                     (int2)(pos_in.x, pos_in.y + 1)),
                                                     (half4)(0.0f),
                                                     (ushort4)((in_w < 0 || h + 1 < 0 || in_w >= input_width || h + 1 >= input_height) << 15));
      input4 = select(read_imageh(input_image, sampler,
                                                     (int2)(pos_in.x + 1, pos_in.y + 1)),
                                                     (half4)(0.0f),
                                                     (ushort4)((in_w + 1 < 0 || h + 1 < 0 || in_w + 1 >= input_width || h + 1 >= input_height) << 15));

      int wx = i * 3;
      int wy = out_c * 4 * 3;
      w = read_imageh(filter, sampler, (int2)(wx, wy));
      output4.x += dot(input4, w);
      w = read_imageh(filter, sampler, (int2)(wx + 1, wy));
      output3.x += dot(input3, w);
      w = read_imageh(filter, sampler, (int2)(wx + 2, wy));
      output4.x += dot(input3, w);
      w = read_imageh(filter, sampler, (int2)(wx, wy + 1));
      output2.x += dot(input2, w);
      w = read_imageh(filter, sampler, (int2)(wx + 1, wy + 1));
      output1.x += dot(input1, w);
      w = read_imageh(filter, sampler, (int2)(wx + 2, wy + 1));
      output2.x += dot(input1, w);
      w = read_imageh(filter, sampler, (int2)(wx, wy + 2));
      output4.x += dot(input2, w);
      w = read_imageh(filter, sampler, (int2)(wx + 1, wy + 2));
      output3.x += dot(input1, w);
      w = read_imageh(filter, sampler, (int2)(wx + 2, wy + 2));
      output4.x += dot(input1, w);

      wy = (out_c * 4 + 1) * 3;
      w = read_imageh(filter, sampler, (int2)(wx, wy));
      output4.y += dot(input4, w);
      w = read_imageh(filter, sampler, (int2)(wx + 1, wy));
      output3.y += dot(input3, w);
      w = read_imageh(filter, sampler, (int2)(wx + 2, wy));
      output4.y += dot(input3, w);
      w = read_imageh(filter, sampler, (int2)(wx, wy + 1));
      output2.y += dot(input2, w);
      w = read_imageh(filter, sampler, (int2)(wx + 1, wy + 1));
      output1.y += dot(input1, w);
      w = read_imageh(filter, sampler, (int2)(wx + 2, wy + 1));
      output2.y += dot(input1, w);
      w = read_imageh(filter, sampler, (int2)(wx, wy + 2));
      output4.y += dot(input2, w);
      w = read_imageh(filter, sampler, (int2)(wx + 1, wy + 2));
      output3.y += dot(input1, w);
      w = read_imageh(filter, sampler, (int2)(wx + 2, wy + 2));
      output4.y += dot(input1, w);

      wy = (out_c * 4 + 2) * 3;
      w = read_imageh(filter, sampler, (int2)(wx, wy));
      output4.z += dot(input4, w);
      w = read_imageh(filter, sampler, (int2)(wx + 1, wy));
      output3.z += dot(input3, w);
      w = read_imageh(filter, sampler, (int2)(wx + 2, wy));
      output4.z += dot(input3, w);
      w = read_imageh(filter, sampler, (int2)(wx, wy + 1));
      output2.z += dot(input2, w);
      w = read_imageh(filter, sampler, (int2)(wx + 1, wy + 1));
      output1.z += dot(input1, w);
      w = read_imageh(filter, sampler, (int2)(wx + 2, wy + 1));
      output2.z += dot(input1, w);
      w = read_imageh(filter, sampler, (int2)(wx, wy + 2));
      output4.z += dot(input2, w);
      w = read_imageh(filter, sampler, (int2)(wx + 1, wy + 2));
      output3.z += dot(input1, w);
      w = read_imageh(filter, sampler, (int2)(wx + 2, wy + 2));
      output4.z += dot(input1, w);

      wy = (out_c * 4 + 3) * 3;
      w = read_imageh(filter, sampler, (int2)(wx, wy));
      output4.w += dot(input4, w);
      w = read_imageh(filter, sampler, (int2)(wx + 1, wy));
      output3.w += dot(input3, w);
      w = read_imageh(filter, sampler, (int2)(wx + 2, wy));
      output4.w += dot(input3, w);
      w = read_imageh(filter, sampler, (int2)(wx, wy + 1));
      output2.w += dot(input2, w);
      w = read_imageh(filter, sampler, (int2)(wx + 1, wy + 1));
      output1.w += dot(input1, w);
      w = read_imageh(filter, sampler, (int2)(wx + 2, wy + 1));
      output2.w += dot(input1, w);
      w = read_imageh(filter, sampler, (int2)(wx, wy + 2));
      output4.w += dot(input2, w);
      w = read_imageh(filter, sampler, (int2)(wx + 1, wy + 2));
      output3.w += dot(input1, w);
      w = read_imageh(filter, sampler, (int2)(wx + 2, wy + 2));
      output4.w += dot(input1, w);
    }

    int2 pos_out = (int2)(out_c * output_width + 2 * in_w, n * output_height + 2 * h);
    write_imageh(output_image, pos_out, output1);
    write_imageh(output_image, (int2)(pos_out.x + 1, pos_out.y), output2);
    write_imageh(output_image, (int2)(pos_out.x, pos_out.y + 1), output3);
    write_imageh(output_image, (int2)(pos_out.x + 1, pos_out.y + 1), output4);
}