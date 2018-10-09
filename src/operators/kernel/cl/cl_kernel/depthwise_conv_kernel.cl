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

/*

__kernel void depth_conv_3x3(__private const int global_size_dim0,
                                              __private const int global_size_dim1,
                                              __private const int global_size_dim2,
                                              __read_only image2d_t input,
                                              __read_only image2d_t filter,
                                              __read_only image2d_t bias,
                                              __write_only image2d_t output_image,
                                              __private const int stride,
                                              __private const int offset,
                                              __private const int input_c,
                                              __private const int dilation,
                                              __private const int input_width,/* of one block */
                                              __private const int input_height, /* of one block */
                                              __private const int output_width,
                                              __private const int output_height) {

    const int out_c = get_global_id(0);
    const int out_w = get_global_id(1);
    const int out_nh = get_global_id(2);

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                              CLK_ADDRESS_CLAMP          |
                              CLK_FILTER_NEAREST;

    const int batch_index = out_nh / output_height;

    const int out_nh_in_one_batch = out_nh % output_height;

    const uint kernelHXW = 1;

    int2 stride_xy = int2(stride, stride);
    int2 ouput_pos_in_one_block = int2(out_w, out_nh_in_one_batch);

    int2 in_pos_in_one_block = ouput_pos_in_one_block * stride_xy + int2(offset, offset);

    half4 output = read_imageh(bias, sampler, int2(out_c, 0));

    int2 pos_in_input_block = int2(out_c * input_width, batch_index * input_height);

    int weight_x_to = out_c * 3;

    half4 inputs[9];

    inputs[0] = select(read_imageh(input, sampler, int2(pos_in_input_block.x + in_pos_in_one_block.x - 1, pos_in_input_block.y + in_pos_in_one_block.y - 1)),
                       0.0,
                       in_pos_in_one_block.x - 1 < 0 || in_pos_in_one_block.y - 1 < 0 || in_pos_in_one_block.x - 1 >= input_width || in_pos_in_one_block.y - 1 >= input_height);

    inputs[1] = select(read_imageh(input, sampler, int2(pos_in_input_block.x + in_pos_in_one_block.x, pos_in_input_block.y + in_pos_in_one_block.y - 1)),
                       0.0,
                       n_pos_in_one_block.x < 0 || in_pos_in_one_block.y - 1 < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y - 1 >= input_height);

    inputs[2] = select(read_imageh(input, sampler, int2(pos_in_input_block.x + in_pos_in_one_block.x + 1, pos_in_input_block.y + in_pos_in_one_block.y - 1)),
                       0.0,
                       in_pos_in_one_block.x + 1 < 0 || in_pos_in_one_block.y - 1 < 0 || in_pos_in_one_block.x + 1 >= input_width || in_pos_in_one_block.y - 1 >= input_height);
    inputs[3] = select(read_imageh(input, sampler, int2(pos_in_input_block.x + in_pos_in_one_block.x - 1, pos_in_input_block.y + in_pos_in_one_block.y)),
                       0.0,
                       in_pos_in_one_block.x - 1 < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x - 1 >= input_width || in_pos_in_one_block.y >= input_height);
    inputs[4] = select(read_imageh(input, sampler, int2(pos_in_input_block.x + in_pos_in_one_block.x, pos_in_input_block.y + in_pos_in_one_block.y)),
                       0.0,
                       in_pos_in_one_block.x < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y >= input_height);

    inputs[5] = select(read_imageh(input, sampler, int2(pos_in_input_block.x + in_pos_in_one_block.x + 1, pos_in_input_block.y + in_pos_in_one_block.y)),
                       0.0,
                       in_pos_in_one_block.x + 1 < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x + 1 >= input_width || in_pos_in_one_block.y >= input_height);

    inputs[6] = select(read_imageh(input, sampler, int2(pos_in_input_block.x + in_pos_in_one_block.x - 1, pos_in_input_block.y + in_pos_in_one_block.y + 1)),
                       0.0,
                       in_pos_in_one_block.x - 1 < 0 || in_pos_in_one_block.y + 1 < 0 || in_pos_in_one_block.x - 1 >= input_width || in_pos_in_one_block.y + 1 >= input_height);

    inputs[7] = select(read_imageh(input, sampler, int2(pos_in_input_block.x + in_pos_in_one_block.x, pos_in_input_block.y + in_pos_in_one_block.y + 1)),
                       0.0,
                       in_pos_in_one_block.x < 0 || in_pos_in_one_block.y + 1 < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y + 1 >= input_height);

    inputs[8] = select(read_imageh(input, sampler, int2(pos_in_input_block.x + in_pos_in_one_block.x + 1, pos_in_input_block.y + in_pos_in_one_block.y + 1)),
                       0.0,
                       in_pos_in_one_block.x + 1 < 0 || in_pos_in_one_block.y + 1 < 0 || in_pos_in_one_block.x + 1 >= input_width || in_pos_in_one_block.y + 1 >= input_height);

    for (int j = 0; j < 9; ++j) {
        half4 input = inputs[j];
        half4 weight = read_imageh(filter, sampler, int2(weight_x_to + j % 3, j / 3));
        output.x += input.x * weight.x;
        output.y += input.y * weight.y;
        output.z += input.z * weight.z;
        output.w += input.w * weight.w;
    }

    #if defined(RELU)
      output = activation(output);
    #endif

      int2 output_pos(out_c * global_size_dim1 + out_w, out_nh);
      write_imageh(output_image, output_pos, output);
}

*/