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
conv
conv_bn
conv_add
conv_relu
conv_bn_relu
conv_add_relu
conv_add_bn_relu
*/

#include "cl_common.h"

__kernel void conv_3x3(__private const int global_size_dim0,
                                              __private const int global_size_dim1,
                                              __private const int global_size_dim2,
                                              __read_only image2d_t input_image,
                                              __read_only image2d_t filter,

#ifdef BIASE
                                              __read_only image2d_t bias,
#endif

#ifdef BATCH_NORM
                                              __read_only image2d_t new_scale,
                                              __read_only image2d_t new_biase,
#endif

                                              __write_only image2d_t output_image,
                                              __private const int stride,
                                              __private const int offset,
                                              __private const int input_c,
                                              __private const int dilation,
                                              __private const int input_width,/* of one block */
                                              __private const int input_height,/* of one block */
                                              __private const int output_width,
                                              __private const int output_height) {

    const int out_c = get_global_id(0);
    const int out_w = get_global_id(1);
    const int out_nh = get_global_id(2);

    if (out_c >= global_size_dim0 ||
        out_w >= global_size_dim1 ||
        out_nh >= global_size_dim2) {
        return;
    }


    int2 stride_xy;
    stride_xy.x = stride;
    stride_xy.y = stride;

    int2 ouput_pos_in_one_block;
    ouput_pos_in_one_block.x = out_w;
    ouput_pos_in_one_block.y = out_nh;


    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                              CLK_ADDRESS_CLAMP          |
                              CLK_FILTER_NEAREST;

    int2 in_pos_in_one_block;
    in_pos_in_one_block.x = ouput_pos_in_one_block.x * stride + offset;
    in_pos_in_one_block.y = ouput_pos_in_one_block.y * stride + offset;

#ifdef BIASE
    half4 output = read_imageh(bias, sampler, (int2)(out_c, 0));
#else
    half4 output = 0.0f;
#endif

   half4 input[9];

   for (int i = 0; i < input_c; ++i) {
        int2 pos_in = (int2)(i * input_width + in_pos_in_one_block.x, in_pos_in_one_block.y);
        input[0] = select(read_imageh(input_image, sampler,
                            (int2)(pos_in.x - dilation, pos_in.y - dilation)),
                            (half4)(0.0f),
                            (ushort4)((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y - dilation < 0 || in_pos_in_one_block.x - dilation >= input_width || in_pos_in_one_block.y - dilation >= input_height) << 15));

        input[1] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x, pos_in.y - dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y - dilation < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y - dilation >= input_height) << 15));

        input[2] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x + dilation, pos_in.y - dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y - dilation < 0 || in_pos_in_one_block.x + dilation >= input_width || in_pos_in_one_block.y - dilation >= input_height) << 15));

        input[3] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x - dilation, pos_in.y)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x - dilation >= input_width || in_pos_in_one_block.y >= input_height) << 15));

        input[4] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x, pos_in.y)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y >= input_height) << 15));

        input[5] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x + dilation, pos_in.y)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x + dilation >= input_width || in_pos_in_one_block.y >= input_height) << 15));

        input[6] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x - dilation, pos_in.y + dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y + dilation < 0 || in_pos_in_one_block.x - dilation >= input_width || in_pos_in_one_block.y + dilation >= input_height) << 15));

        input[7] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x, pos_in.y + dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y + dilation < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y + dilation >= input_height) << 15));

        input[8] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x + dilation, pos_in.y + dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y + dilation < 0 || in_pos_in_one_block.x + dilation >= input_width || in_pos_in_one_block.y + dilation >= input_height) << 15));


/*
        for (int j = 0; j < 9; ++j) {
            int2 pos_of_weight;
            pos_of_weight.x = i * 3 + j % 3;
            pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
            half4 weight_x = read_imageh(filter, sampler, pos_of_weight);
            output.x += dot(input[j], weight_x);

            pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
            half4 weight_y = read_imageh(filter, sampler, pos_of_weight);
            output.y += dot(input[j], weight_y);

            pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
            half4 weight_z = read_imageh(filter, sampler, pos_of_weight);
            output.z += dot(input[j], weight_z);

            pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
            half4 weight_w = read_imageh(filter, sampler, pos_of_weight);
            output.w += dot(input[j], weight_w);
        }
*/
            int j = 0;
            int2 pos_of_weight;
            pos_of_weight.x = i * 3 + j % 3;
            pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
            half4 weight_x = read_imageh(filter, sampler, pos_of_weight);
            output.x += dot(input[j], weight_x);

            pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
            half4 weight_y = read_imageh(filter, sampler, pos_of_weight);
            output.y += dot(input[j], weight_y);

            pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
            half4 weight_z = read_imageh(filter, sampler, pos_of_weight);
            output.z += dot(input[j], weight_z);

            pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
            half4 weight_w = read_imageh(filter, sampler, pos_of_weight);
            output.w += dot(input[j], weight_w);

            j = 1;
            pos_of_weight.x = i * 3 + j % 3;
            pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
            weight_x = read_imageh(filter, sampler, pos_of_weight);
            output.x += dot(input[j], weight_x);

            pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
            weight_y = read_imageh(filter, sampler, pos_of_weight);
            output.y += dot(input[j], weight_y);

            pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
            weight_z = read_imageh(filter, sampler, pos_of_weight);
            output.z += dot(input[j], weight_z);

            pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
            weight_w = read_imageh(filter, sampler, pos_of_weight);
            output.w += dot(input[j], weight_w);

            j = 2;
            pos_of_weight.x = i * 3 + j % 3;
            pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
            weight_x = read_imageh(filter, sampler, pos_of_weight);
            output.x += dot(input[j], weight_x);

            pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
            weight_y = read_imageh(filter, sampler, pos_of_weight);
            output.y += dot(input[j], weight_y);

            pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
            weight_z = read_imageh(filter, sampler, pos_of_weight);
            output.z += dot(input[j], weight_z);

            pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
            weight_w = read_imageh(filter, sampler, pos_of_weight);
            output.w += dot(input[j], weight_w);

            j = 3;
            pos_of_weight.x = i * 3 + j % 3;
            pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
            weight_x = read_imageh(filter, sampler, pos_of_weight);
            output.x += dot(input[j], weight_x);

            pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
            weight_y = read_imageh(filter, sampler, pos_of_weight);
            output.y += dot(input[j], weight_y);

            pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
            weight_z = read_imageh(filter, sampler, pos_of_weight);
            output.z += dot(input[j], weight_z);

            pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
            weight_w = read_imageh(filter, sampler, pos_of_weight);
            output.w += dot(input[j], weight_w);

            j = 4;
            pos_of_weight.x = i * 3 + j % 3;
            pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
            weight_x = read_imageh(filter, sampler, pos_of_weight);
            output.x += dot(input[j], weight_x);

            pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
            weight_y = read_imageh(filter, sampler, pos_of_weight);
            output.y += dot(input[j], weight_y);

            pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
            weight_z = read_imageh(filter, sampler, pos_of_weight);
            output.z += dot(input[j], weight_z);

            pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
            weight_w = read_imageh(filter, sampler, pos_of_weight);
            output.w += dot(input[j], weight_w);

            j = 5;
            pos_of_weight.x = i * 3 + j % 3;
            pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
            weight_x = read_imageh(filter, sampler, pos_of_weight);
            output.x += dot(input[j], weight_x);

            pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
            weight_y = read_imageh(filter, sampler, pos_of_weight);
            output.y += dot(input[j], weight_y);

            pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
            weight_z = read_imageh(filter, sampler, pos_of_weight);
            output.z += dot(input[j], weight_z);

            pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
            weight_w = read_imageh(filter, sampler, pos_of_weight);
            output.w += dot(input[j], weight_w);

           j = 6;
           pos_of_weight.x = i * 3 + j % 3;
           pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
           weight_x = read_imageh(filter, sampler, pos_of_weight);
           output.x += dot(input[j], weight_x);

           pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
           weight_y = read_imageh(filter, sampler, pos_of_weight);
           output.y += dot(input[j], weight_y);

           pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
           weight_z = read_imageh(filter, sampler, pos_of_weight);
           output.z += dot(input[j], weight_z);

           pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
           weight_w = read_imageh(filter, sampler, pos_of_weight);
           output.w += dot(input[j], weight_w);

           j = 7;
           pos_of_weight.x = i * 3 + j % 3;
           pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
           weight_x = read_imageh(filter, sampler, pos_of_weight);
           output.x += dot(input[j], weight_x);

           pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
           weight_y = read_imageh(filter, sampler, pos_of_weight);
           output.y += dot(input[j], weight_y);

           pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
           weight_z = read_imageh(filter, sampler, pos_of_weight);
           output.z += dot(input[j], weight_z);

           pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
           weight_w = read_imageh(filter, sampler, pos_of_weight);
           output.w += dot(input[j], weight_w);

           j = 8;
           pos_of_weight.x = i * 3 + j % 3;
           pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
           weight_x = read_imageh(filter, sampler, pos_of_weight);
           output.x += dot(input[j], weight_x);

           pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
           weight_y = read_imageh(filter, sampler, pos_of_weight);
           output.y += dot(input[j], weight_y);

           pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
           weight_z = read_imageh(filter, sampler, pos_of_weight);
           output.z += dot(input[j], weight_z);

           pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
           weight_w = read_imageh(filter, sampler, pos_of_weight);
           output.w += dot(input[j], weight_w);

    }

#ifdef BATCH_NORM
    output = output * read_imageh(new_scale, sampler, (int2)(out_c, 0)) + read_imageh(new_biase, sampler, (int2)(out_c, 0));
#endif

#ifdef RELU
    output = activation(output);
#endif

    write_imageh(output_image, (int2)(out_c * global_size_dim1 + out_w, out_nh), output);
}




__kernel void depth_conv_3x3(__private const int global_size_dim0,
                                              __private const int global_size_dim1,
                                              __private const int global_size_dim2,
                                              __read_only image2d_t input,
                                              __read_only image2d_t filter,
#ifdef BIASE
                                              __read_only image2d_t bias,
#endif
#ifdef BATCH_NORM
                                              __read_only image2d_t new_scale,
                                              __read_only image2d_t new_biase,
#endif
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

    int2 output_pos = (int2)(out_c * global_size_dim1 + out_w, out_nh);


    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                              CLK_ADDRESS_CLAMP          |
                              CLK_FILTER_NEAREST;

    const int batch_index = out_nh / output_height;

    const int out_nh_in_one_batch = out_nh % output_height;


    int2 stride_xy = (int2)(stride, stride);
    int2 ouput_pos_in_one_block = (int2)(out_w, out_nh_in_one_batch);

    int2 in_pos_in_one_block = ouput_pos_in_one_block * stride_xy + (int2)(offset, offset);

#ifdef BIASE
    half4 output = read_imageh(bias, sampler, (int2)(out_c, 0));
#else
    half4 output = 0.0f;
#endif

    const int filter_width = 3;
    const int filter_height = 3;

    int2 pos_in_input_block = (int2)(out_c * input_width, batch_index * input_height);

    int2 pos_in_filter_block = (int2)(out_c * filter_width, batch_index * filter_height);

    int filter_x = pos_in_filter_block.x ;
    int filter_y = pos_in_filter_block.y ;

    half4 inputs[9];

        inputs[0] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x - 1, pos_in_input_block.y + in_pos_in_one_block.y - 1)),
                           (half4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x - 1 < 0 || in_pos_in_one_block.y - 1 < 0 || in_pos_in_one_block.x - 1 >= input_width || in_pos_in_one_block.y - 1 >= input_height) << 15));

        inputs[1] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x, pos_in_input_block.y + in_pos_in_one_block.y - 1)),
                           (half4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y - 1 < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y - 1 >= input_height) << 15));

        inputs[2] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x + 1, pos_in_input_block.y + in_pos_in_one_block.y - 1)),
                           (half4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x + 1 < 0 || in_pos_in_one_block.y - 1 < 0 || in_pos_in_one_block.x + 1 >= input_width || in_pos_in_one_block.y - 1 >= input_height) << 15));

        inputs[3] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x - 1, pos_in_input_block.y + in_pos_in_one_block.y)),
                           (half4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x - 1 < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x - 1 >= input_width || in_pos_in_one_block.y >= input_height) << 15));
        /*
        if (output_pos.x == 112 && output_pos.y == 0) {
              half4 input1 = inputs[3];
              float4 in = (float4)(input1.x, input1.y, input1.z, input1.w);
              printf(" input4 3 - %v4hlf \n", in);
              printf(" --- %d ---\n", in_pos_in_one_block.x - 1);
        }
        */


        inputs[4] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x, pos_in_input_block.y + in_pos_in_one_block.y)),
                           (half4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y >= input_height) << 15));

        inputs[5] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x + 1, pos_in_input_block.y + in_pos_in_one_block.y)),
                           (half4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x + 1 < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x + 1 >= input_width || in_pos_in_one_block.y >= input_height) << 15));

        inputs[6] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x - 1, pos_in_input_block.y + in_pos_in_one_block.y + 1)),
                           (half4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x - 1 < 0 || in_pos_in_one_block.y + 1 < 0 || in_pos_in_one_block.x - 1 >= input_width || in_pos_in_one_block.y + 1 >= input_height) << 15));

        inputs[7] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x, pos_in_input_block.y + in_pos_in_one_block.y + 1)),
                           (half4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y + 1 < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y + 1 >= input_height) << 15));

        inputs[8] = select(read_imageh(input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x + 1, pos_in_input_block.y + in_pos_in_one_block.y + 1)),
                           (half4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x + 1 < 0 || in_pos_in_one_block.y + 1 < 0 || in_pos_in_one_block.x + 1 >= input_width || in_pos_in_one_block.y + 1 >= input_height) << 15));

    half4 filters[9];
    filters[0] =  read_imageh(filter, sampler,(int2)(filter_x,filter_y));
    filters[1] =  read_imageh(filter, sampler,(int2)(filter_x + 1,filter_y));
    filters[2] =  read_imageh(filter, sampler,(int2)(filter_x + 2,filter_y));
    filters[3] =  read_imageh(filter, sampler,(int2)(filter_x,filter_y + 1));
    filters[4] =  read_imageh(filter, sampler,(int2)(filter_x + 1,filter_y + 1));
    filters[5] =  read_imageh(filter, sampler,(int2)(filter_x + 2,filter_y + 1));
    filters[6] =  read_imageh(filter, sampler,(int2)(filter_x,filter_y + 2));
    filters[7] =  read_imageh(filter, sampler,(int2)(filter_x + 1,filter_y + 2));
    filters[8] =  read_imageh(filter, sampler,(int2)(filter_x + 2,filter_y + 2));

    for(int i = 0 ;i < 9 ; i++){
     output += inputs[i] * filters[i];
    }
#ifdef BATCH_NORM
    output = output * read_imageh(new_scale, sampler, (int2)(out_c, 0)) + read_imageh(new_biase, sampler, (int2)(out_c, 0));
#endif

#ifdef RELU
    output = activation(output);
#endif


    /*

    if (output_pos.x == 112 && output_pos.y == 0) {

        for (int i = 0; i < 9; ++i) {
            half4 input1 = inputs[i];
            float4 in = (float4)(input1.x, input1.y, input1.z, input1.w);
            printf(" input4 %d - %v4hlf \n", i, in);
        }

        float4 out = (float4)(output.x, output.y, output.z, output.w);
        printf(" depth wise output output4 = %v4hlf \n", out);
        printf(" pos_in_input_block -x %d \n ", pos_in_input_block.x);
        printf(" pos_in_input_block -y %d \n ", pos_in_input_block.y);
        printf(" in_pos_in_one_block - x %d \n", in_pos_in_one_block.x);
        printf(" in_pos_in_one_block - y %d \n", in_pos_in_one_block.y);
    }

    */

    write_imageh(output_image, output_pos, output);

}


__kernel void conv_1x1(__private const int global_size_dim0,
                       __private const int global_size_dim1,
                       __private const int global_size_dim2,
                       __read_only image2d_t input_image,
                       __read_only image2d_t filter,
#ifdef BIASE
                       __read_only image2d_t bias,
#endif
#ifdef BATCH_NORM
                       __read_only image2d_t new_scale,
                       __read_only image2d_t new_biase,
#endif
                       __write_only image2d_t output_image,
                       __private const int stride,
                       __private const int offset,
                       __private const int input_c,
                       __private const int dilation,
                       __private const int input_width,/* of one block */
                       __private const int input_height,/* of one block */
                       __private const int output_width,
                       __private const int output_height) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                           CLK_ADDRESS_CLAMP         |
                           CLK_FILTER_NEAREST;

  const uint kernelHXW = 1;
  int2 stride_xy = (int2)(stride, stride);
  int2 ouput_pos_in_one_block = (int2)(out_w, out_nh);
  int2 in_pos_in_one_block = ouput_pos_in_one_block * stride_xy + (int2)(offset, offset);

#ifdef BIASE
    half4 output = read_imageh(bias, sampler, (int2)(out_c, 0));
#else
    half4 output = 0.0f;
#endif

   for (int i = 0; i < input_c; ++i) {
        int2 pos_in = (int2)(i * input_width + in_pos_in_one_block.x, in_pos_in_one_block.y);
        half4 input = read_imageh(input_image, sampler, pos_in);

        half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 0));
        half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 1));
        half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 2));
        half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 3));
/*
        output.x = dot(input, weight0);
        output.y = dot(input, weight1);
        output.z = dot(input, weight2);
        output.w = dot(input, weight3);
*/

        output = mad(input.x, weight0, output);
        output = mad(input.y, weight1, output);
        output = mad(input.z, weight2, output);
        output = mad(input.w, weight3, output);

   }

#ifdef BATCH_NORM
    output = output * read_imageh(new_scale, sampler, (int2)(out_c, 0)) + read_imageh(new_biase, sampler, (int2)(out_c, 0));
#endif

#ifdef RELU
  output = activation(output);
#endif

  int2 output_pos = (int2)(out_c * global_size_dim1 + out_w, out_nh);
  write_imageh(output_image, output_pos, output);
}

__kernel void conv_1x1_spl(
    __private const int global_size_dim0, __private const int global_size_dim1,
    __private const int global_size_dim2, __read_only image2d_t input_image,
    __read_only image2d_t filter,
#ifdef BIASE
    __read_only image2d_t bias,
#endif
#ifdef BATCH_NORM
    __read_only image2d_t new_scale, __read_only image2d_t new_biase,
#endif
    __write_only image2d_t output_image, __private const int stride,
    __private const int offset, __private const int input_c,
    __private const int dilation,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __private const int old_w
    ) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_w0 = out_w;
  int out_w1 = out_w + global_size_dim1;
  int out_w2 = out_w + global_size_dim1 * 2;
  int out_w3 = out_w + global_size_dim1 * 3;

//  int out_w1 = out_w + global_size_dim1;
//  int out_w2 = out_w + global_size_dim1 * 2;
//  int out_w3 = out_w + global_size_dim1 * 3;

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int2 stride_xy = (int2)(stride, stride);

  int2 ouput_pos_in_one_block0 = (int2)(out_w0, out_nh);
  int2 in_pos_in_one_block0 =
      ouput_pos_in_one_block0 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block1 = (int2)(out_w1, out_nh);
  int2 in_pos_in_one_block1 =
      ouput_pos_in_one_block1 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block2 = (int2)(out_w2, out_nh);
  int2 in_pos_in_one_block2 =
      ouput_pos_in_one_block2 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block3 = (int2)(out_w3, out_nh);
  int2 in_pos_in_one_block3 =
      ouput_pos_in_one_block3 * stride_xy + (int2)(offset, offset);

#ifdef BIASE
      half4 output0= read_imageh(bias, sampler, (int2)(out_c, 0));
      half4 output1 = read_imageh(bias, sampler, (int2)(out_c, 0));
      half4 output2 = read_imageh(bias, sampler, (int2)(out_c, 0));
      half4 output3 = read_imageh(bias, sampler, (int2)(out_c, 0));
//  half4 output0 = 0.0f;
//  half4 output1 = 0.0f;
//  half4 output2 = 0.0f;
//  half4 output3 = 0.0f;

#else
  half4 output0 = 0.0f;
  half4 output1 = 0.0f;
  half4 output2 = 0.0f;
  half4 output3 = 0.0f;
#endif
  for (int i = 0; i < input_c; ++i) {
    // ------------0---------------
    int2 pos_in = (int2)(i * input_width + in_pos_in_one_block0.x, in_pos_in_one_block0.y);
    half4 input0 = read_imageh(input_image, sampler, pos_in);

    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 0));
    half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 1));
    half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 2));
    half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 3));

    output0 = mad(input0.x, weight0, output0);
    output0 = mad(input0.y, weight1, output0);
    output0 = mad(input0.z, weight2, output0);
    output0 = mad(input0.w, weight3, output0);

    // -------------1--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block1.x, in_pos_in_one_block1.y);
    half4 input1 = read_imageh(input_image, sampler, pos_in);
    //
    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
    //    * 4 + 3));

    output1 = mad(input1.x, weight0, output1);
    output1 = mad(input1.y, weight1, output1);
    output1 = mad(input1.z, weight2, output1);
    output1 = mad(input1.w, weight3, output1);

    // -------------2--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block2.x, in_pos_in_one_block2.y);
    half4 input2 = read_imageh(input_image, sampler, pos_in);

    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
    //    * 4 + 3));

    output2 = mad(input2.x, weight0, output2);
    output2 = mad(input2.y, weight1, output2);
    output2 = mad(input2.z, weight2, output2);
    output2 = mad(input2.w, weight3, output2);

    // -------------3--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block3.x, in_pos_in_one_block3.y);
    half4 input3 = read_imageh(input_image, sampler, pos_in);

    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
    //    * 4 + 3));

    output3 = mad(input3.x, weight0, output3);
    output3 = mad(input3.y, weight1, output3);
    output3 = mad(input3.z, weight2, output3);
    output3 = mad(input3.w, weight3, output3);
  }

#ifdef BATCH_NORM
    output0 = output0 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output1 = output1 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output2 = output2 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output3 = output3 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));
#endif

#ifdef RELU
     output0 = activation(output0);
     output1 = activation(output1);
     output2 = activation(output2);
     output3 = activation(output3);
#endif
  int outpos_main = mul24(out_c , old_w);
  int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);

  if (out_w0 < old_w) {
    write_imageh(output_image, output_pos0, output0);
  }
  int2 output_pos1 = (int2)(outpos_main + out_w1, out_nh);
  if (out_w1 < old_w){
    write_imageh(output_image, output_pos1, output1);
  }

  int2 output_pos2 = (int2)(outpos_main + out_w2, out_nh);
  if (out_w2 < old_w){
    write_imageh(output_image, output_pos2, output2);
  }

  int2 output_pos3 = (int2)(outpos_main + out_w3, out_nh);
  if (out_w3 < old_w){
    write_imageh(output_image, output_pos3, output3);
  }
}

__kernel void conv_1x1_spl2(
    __private const int global_size_dim0, __private const int global_size_dim1,
    __private const int global_size_dim2, __read_only image2d_t input_image,
    __read_only image2d_t filter,
#ifdef BIASE
    __read_only image2d_t bias,
#endif
#ifdef BATCH_NORM
    __read_only image2d_t new_scale, __read_only image2d_t new_biase,
#endif
    __write_only image2d_t output_image, __private const int stride,
    __private const int offset, __private const int input_c,
    __private const int dilation,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __private const int old_w
) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_w0 = out_w;
  int out_w1 = out_w + global_size_dim1;
  int out_w2 = out_w + global_size_dim1 * 2;
  int out_w3 = out_w + global_size_dim1 * 3;
  int out_w4 = out_w + global_size_dim1 * 4;
  int out_w5 = out_w + global_size_dim1 * 5;
  int out_w6 = out_w + global_size_dim1 * 6;
  int out_w7 = out_w + global_size_dim1 * 7;

//  int out_w1 = out_w + global_size_dim1;
//  int out_w2 = out_w + global_size_dim1 * 2;
//  int out_w3 = out_w + global_size_dim1 * 3;

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int2 stride_xy = (int2)(stride, stride);

  int2 ouput_pos_in_one_block0 = (int2)(out_w0, out_nh);
  int2 in_pos_in_one_block0 =
      ouput_pos_in_one_block0 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block1 = (int2)(out_w1, out_nh);
  int2 in_pos_in_one_block1 =
      ouput_pos_in_one_block1 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block2 = (int2)(out_w2, out_nh);
  int2 in_pos_in_one_block2 =
      ouput_pos_in_one_block2 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block3 = (int2)(out_w3, out_nh);
  int2 in_pos_in_one_block3 =
      ouput_pos_in_one_block3 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block4 = (int2)(out_w4, out_nh);
  int2 in_pos_in_one_block4 =
      ouput_pos_in_one_block4 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block5 = (int2)(out_w5, out_nh);
  int2 in_pos_in_one_block5 =
      ouput_pos_in_one_block5 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block6 = (int2)(out_w6, out_nh);
  int2 in_pos_in_one_block6 =
      ouput_pos_in_one_block6 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block7 = (int2)(out_w7, out_nh);
  int2 in_pos_in_one_block7 =
      ouput_pos_in_one_block7 * stride_xy + (int2)(offset, offset);

#ifdef BIASE
  half4 output0 = read_imageh(bias, sampler, (int2)(out_c, 0));
  half4 output1 = read_imageh(bias, sampler, (int2)(out_c, 0));
  half4 output2 = read_imageh(bias, sampler, (int2)(out_c, 0));
  half4 output3 = read_imageh(bias, sampler, (int2)(out_c, 0));
  half4 output4 = read_imageh(bias, sampler, (int2)(out_c, 0));
  half4 output5 = read_imageh(bias, sampler, (int2)(out_c, 0));
  half4 output6 = read_imageh(bias, sampler, (int2)(out_c, 0));
  half4 output7 = read_imageh(bias, sampler, (int2)(out_c, 0));
//  half4 output0 = 0.0f;
//  half4 output1 = 0.0f;
//  half4 output2 = 0.0f;
//  half4 output3 = 0.0f;

#else
  half4 output0 = 0.0f;
  half4 output1 = 0.0f;
  half4 output2 = 0.0f;
  half4 output3 = 0.0f;
  half4 output4 = 0.0f;
  half4 output5 = 0.0f;
  half4 output6 = 0.0f;
  half4 output7 = 0.0f;
#endif
  for (int i = 0; i < input_c; ++i) {
    // ------------0---------------
    int2 pos_in = (int2)(i * input_width + in_pos_in_one_block0.x, in_pos_in_one_block0.y);
    half4 input0 = read_imageh(input_image, sampler, pos_in);

    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 0));
    half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 1));
    half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 2));
    half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 3));

    output0 = mad(input0.x, weight0, output0);
    output0 = mad(input0.y, weight1, output0);
    output0 = mad(input0.z, weight2, output0);
    output0 = mad(input0.w, weight3, output0);

    // -------------1--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block1.x, in_pos_in_one_block1.y);
    half4 input1 = read_imageh(input_image, sampler, pos_in);
    //
    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
    //    * 4 + 3));

    output1 = mad(input1.x, weight0, output1);
    output1 = mad(input1.y, weight1, output1);
    output1 = mad(input1.z, weight2, output1);
    output1 = mad(input1.w, weight3, output1);

    // -------------2--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block2.x, in_pos_in_one_block2.y);
    half4 input2 = read_imageh(input_image, sampler, pos_in);

    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
    //    * 4 + 3));

    output2 = mad(input2.x, weight0, output2);
    output2 = mad(input2.y, weight1, output2);
    output2 = mad(input2.z, weight2, output2);
    output2 = mad(input2.w, weight3, output2);

    // -------------3--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block3.x, in_pos_in_one_block3.y);
    half4 input3 = read_imageh(input_image, sampler, pos_in);

    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
    //    * 4 + 3));

    output3 = mad(input3.x, weight0, output3);
    output3 = mad(input3.y, weight1, output3);
    output3 = mad(input3.z, weight2, output3);
    output3 = mad(input3.w, weight3, output3);


    // -------------4--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block4.x, in_pos_in_one_block4.y);
    half4 input4 = read_imageh(input_image, sampler, pos_in);

    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
    //    * 4 + 3));

    output4 = mad(input4.x, weight0, output4);
    output4 = mad(input4.y, weight1, output4);
    output4 = mad(input4.z, weight2, output4);
    output4 = mad(input4.w, weight3, output4);



    // -------------5--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block5.x, in_pos_in_one_block5.y);
    half4 input5 = read_imageh(input_image, sampler, pos_in);

    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
    //    * 4 + 3));

    output5= mad(input5.x, weight0, output5);
    output5 = mad(input5.y, weight1, output5);
    output5 = mad(input5.z, weight2, output5);
    output5 = mad(input5.w, weight3, output5);


    // -------------6--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block6.x, in_pos_in_one_block6.y);
    half4 input6 = read_imageh(input_image, sampler, pos_in);

    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
    //    * 4 + 3));

    output6 = mad(input6.x, weight0, output6);
    output6 = mad(input6.y, weight1, output6);
    output6 = mad(input6.z, weight2, output6);
    output6 = mad(input6.w, weight3, output6);


    // -------------7--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block7.x, in_pos_in_one_block7.y);
    half4 input7 = read_imageh(input_image, sampler, pos_in);

    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
    //    * 4 + 3));

    output7 = mad(input7.x, weight0, output7);
    output7 = mad(input7.y, weight1, output7);
    output7 = mad(input7.z, weight2, output7);
    output7 = mad(input7.w, weight3, output7);
  }

#ifdef BATCH_NORM
    output0 = output0 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output1 = output1 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output2 = output2 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output3 = output3 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output4 = output4 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output5 = output5 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output6 = output6 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output7 = output7 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

#endif

#ifdef RELU
     output0 = activation(output0);
     output1 = activation(output1);
     output2 = activation(output2);
     output3 = activation(output3);
     output4 = activation(output4);
     output5 = activation(output5);
     output6 = activation(output6);
     output7 = activation(output7);
#endif
  int outpos_main = mul24(out_c , old_w);
  int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);

  if (out_w0 < old_w) {
    write_imageh(output_image, output_pos0, output0);
  }
  int2 output_pos1 = (int2)(outpos_main + out_w1, out_nh);
  if (out_w1 < old_w){
    write_imageh(output_image, output_pos1, output1);
  }

  int2 output_pos2 = (int2)(outpos_main + out_w2, out_nh);
  if (out_w2 < old_w){
    write_imageh(output_image, output_pos2, output2);
  }

  int2 output_pos3 = (int2)(outpos_main + out_w3, out_nh);
  if (out_w3 < old_w){
    write_imageh(output_image, output_pos3, output3);
  }

  int2 output_pos4 = (int2)(outpos_main + out_w4, out_nh);
  if (out_w4 < old_w){
    write_imageh(output_image, output_pos4, output4);
  }

  int2 output_pos5 = (int2)(outpos_main + out_w5, out_nh);
  if (out_w5 < old_w){
    write_imageh(output_image, output_pos5, output5);

  }
  int2 output_pos6 = (int2)(outpos_main + out_w6, out_nh);
  if (out_w6 < old_w){
    write_imageh(output_image, output_pos6, output6);
  }

  int2 output_pos7 = (int2)(outpos_main + out_w7, out_nh);
  if (out_w7 < old_w){
    write_imageh(output_image, output_pos7, output7);
  }

}
__kernel void conv_1x1_spl3(
    __private const int global_size_dim0, __private const int global_size_dim1,
    __private const int global_size_dim2, __read_only image2d_t input_image,
    __read_only image2d_t filter,
#ifdef BIASE
    __read_only image2d_t bias,
#endif
#ifdef BATCH_NORM
    __read_only image2d_t new_scale, __read_only image2d_t new_biase,
#endif
    __write_only image2d_t output_image, __private const int stride,
    __private const int offset, __private const int input_c,
    __private const int dilation,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __private const int old_w
) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_w0 = out_w;
  int out_w1 = out_w + global_size_dim1;
  int out_w2 = out_w + global_size_dim1 * 2;
//  int out_w3 = out_w + global_size_dim1 * 3;
//  int out_w4 = out_w + global_size_dim1 * 4;
//  int out_w5 = out_w + global_size_dim1 * 5;
//  int out_w6 = out_w + global_size_dim1 * 6;
//  int out_w7 = out_w + global_size_dim1 * 7;

//  int out_w1 = out_w + global_size_dim1;
//  int out_w2 = out_w + global_size_dim1 * 2;
//  int out_w3 = out_w + global_size_dim1 * 3;

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int2 stride_xy = (int2)(stride, stride);

  int2 ouput_pos_in_one_block0 = (int2)(out_w0, out_nh);
  int2 in_pos_in_one_block0 =
      ouput_pos_in_one_block0 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block1 = (int2)(out_w1, out_nh);
  int2 in_pos_in_one_block1 =
      ouput_pos_in_one_block1 * stride_xy + (int2)(offset, offset);

//  int2 ouput_pos_in_one_block2 = (int2)(out_w2, out_nh);
//  int2 in_pos_in_one_block2 =
//      ouput_pos_in_one_block2 * stride_xy + (int2)(offset, offset);
//
//  int2 ouput_pos_in_one_block3 = (int2)(out_w3, out_nh);
//  int2 in_pos_in_one_block3 =
//      ouput_pos_in_one_block3 * stride_xy + (int2)(offset, offset);
//
//  int2 ouput_pos_in_one_block4 = (int2)(out_w4, out_nh);
//  int2 in_pos_in_one_block4 =
//      ouput_pos_in_one_block4 * stride_xy + (int2)(offset, offset);
//
//  int2 ouput_pos_in_one_block5 = (int2)(out_w5, out_nh);
//  int2 in_pos_in_one_block5 =
//      ouput_pos_in_one_block5 * stride_xy + (int2)(offset, offset);
//
//  int2 ouput_pos_in_one_block6 = (int2)(out_w6, out_nh);
//  int2 in_pos_in_one_block6 =
//      ouput_pos_in_one_block6 * stride_xy + (int2)(offset, offset);
//
//  int2 ouput_pos_in_one_block7 = (int2)(out_w7, out_nh);
//  int2 in_pos_in_one_block7 =
//      ouput_pos_in_one_block7 * stride_xy + (int2)(offset, offset);

#ifdef BIASE
  half4 output0 = read_imageh(bias, sampler, (int2)(out_c, 0));
  half4 output1 = read_imageh(bias, sampler, (int2)(out_c, 0));
//  half4 output2 = read_imageh(bias, sampler, (int2)(out_c, 0));
//  half4 output3 = read_imageh(bias, sampler, (int2)(out_c, 0));
//  half4 output4 = read_imageh(bias, sampler, (int2)(out_c, 0));
//  half4 output5 = read_imageh(bias, sampler, (int2)(out_c, 0));
//  half4 output6 = read_imageh(bias, sampler, (int2)(out_c, 0));
//  half4 output7 = read_imageh(bias, sampler, (int2)(out_c, 0));
//  half4 output0 = 0.0f;
//  half4 output1 = 0.0f;
//  half4 output2 = 0.0f;
//  half4 output3 = 0.0f;

#else
  half4 output0 = 0.0f;
  half4 output1 = 0.0f;
//  half4 output2 = 0.0f;
//  half4 output3 = 0.0f;
//  half4 output4 = 0.0f;
//  half4 output5 = 0.0f;
//  half4 output6 = 0.0f;
//  half4 output7 = 0.0f;
#endif
  for (int i = 0; i < input_c; ++i) {
    // ------------0---------------
    int2 pos_in = (int2)(i * input_width + in_pos_in_one_block0.x, in_pos_in_one_block0.y);
    half4 input0 = read_imageh(input_image, sampler, pos_in);

    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 0));
    half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 1));
    half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 2));
    half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 3));

    output0 = mad(input0.x, weight0, output0);
    output0 = mad(input0.y, weight1, output0);
    output0 = mad(input0.z, weight2, output0);
    output0 = mad(input0.w, weight3, output0);

    // -------------1--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block1.x, in_pos_in_one_block1.y);
    half4 input1 = read_imageh(input_image, sampler, pos_in);
    //
    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
    //    * 4 + 3));

    output1 = mad(input1.x, weight0, output1);
    output1 = mad(input1.y, weight1, output1);
    output1 = mad(input1.z, weight2, output1);
    output1 = mad(input1.w, weight3, output1);
//
//    // -------------2--------------
//    pos_in = (int2)(i * input_width + in_pos_in_one_block2.x, in_pos_in_one_block2.y);
//    half4 input2 = read_imageh(input_image, sampler, pos_in);
//
//    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
//    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
//    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
//    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
//    //    * 4 + 3));
//
//    output2 = mad(input2.x, weight0, output2);
//    output2 = mad(input2.y, weight1, output2);
//    output2 = mad(input2.z, weight2, output2);
//    output2 = mad(input2.w, weight3, output2);
//
//    // -------------3--------------
//    pos_in = (int2)(i * input_width + in_pos_in_one_block3.x, in_pos_in_one_block3.y);
//    half4 input3 = read_imageh(input_image, sampler, pos_in);
//
//    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
//    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
//    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
//    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
//    //    * 4 + 3));
//
//    output3 = mad(input3.x, weight0, output3);
//    output3 = mad(input3.y, weight1, output3);
//    output3 = mad(input3.z, weight2, output3);
//    output3 = mad(input3.w, weight3, output3);
//
//
//    // -------------4--------------
//    pos_in = (int2)(i * input_width + in_pos_in_one_block4.x, in_pos_in_one_block4.y);
//    half4 input4 = read_imageh(input_image, sampler, pos_in);
//
//    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
//    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
//    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
//    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
//    //    * 4 + 3));
//
//    output4 = mad(input4.x, weight0, output4);
//    output4 = mad(input4.y, weight1, output4);
//    output4 = mad(input4.z, weight2, output4);
//    output4 = mad(input4.w, weight3, output4);
//
//
//
//    // -------------5--------------
//    pos_in = (int2)(i * input_width + in_pos_in_one_block5.x, in_pos_in_one_block5.y);
//    half4 input5 = read_imageh(input_image, sampler, pos_in);
//
//    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
//    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
//    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
//    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
//    //    * 4 + 3));
//
//    output5= mad(input5.x, weight0, output5);
//    output5 = mad(input5.y, weight1, output5);
//    output5 = mad(input5.z, weight2, output5);
//    output5 = mad(input5.w, weight3, output5);
//
//
//    // -------------6--------------
//    pos_in = (int2)(i * input_width + in_pos_in_one_block6.x, in_pos_in_one_block6.y);
//    half4 input6 = read_imageh(input_image, sampler, pos_in);
//
//    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
//    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
//    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
//    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
//    //    * 4 + 3));
//
//    output6 = mad(input6.x, weight0, output6);
//    output6 = mad(input6.y, weight1, output6);
//    output6 = mad(input6.z, weight2, output6);
//    output6 = mad(input6.w, weight3, output6);
//
//
//    // -------------7--------------
//    pos_in = (int2)(i * input_width + in_pos_in_one_block7.x, in_pos_in_one_block7.y);
//    half4 input7 = read_imageh(input_image, sampler, pos_in);
//
//    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
//    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
//    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
//    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
//    //    * 4 + 3));
//
//    output7 = mad(input7.x, weight0, output7);
//    output7 = mad(input7.y, weight1, output7);
//    output7 = mad(input7.z, weight2, output7);
//    output7 = mad(input7.w, weight3, output7);
  }

#ifdef BATCH_NORM
  output0 = output0 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output1 = output1 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));
//
//    output2 = output2 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
//          read_imageh(new_biase, sampler, (int2)(out_c, 0));
//
//    output3 = output3 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
//          read_imageh(new_biase, sampler, (int2)(out_c, 0));
//
//    output4 = output4 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
//          read_imageh(new_biase, sampler, (int2)(out_c, 0));
//
//    output5 = output5 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
//          read_imageh(new_biase, sampler, (int2)(out_c, 0));
//
//    output6 = output6 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
//          read_imageh(new_biase, sampler, (int2)(out_c, 0));
//
//    output7 = output7 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
//          read_imageh(new_biase, sampler, (int2)(out_c, 0));

#endif

#ifdef RELU
  output0 = activation(output0);
     output1 = activation(output1);
//     output2 = activation(output2);
//     output3 = activation(output3);
//     output4 = activation(output4);
//     output5 = activation(output5);
//     output6 = activation(output6);
//     output7 = activation(output7);
#endif
  int outpos_main = mul24(out_c , old_w);
  int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);

  if (out_w0 < old_w) {
    write_imageh(output_image, output_pos0, output0);
  }
  int2 output_pos1 = (int2)(outpos_main + out_w1, out_nh);
  if (out_w1 < old_w){
    write_imageh(output_image, output_pos1, output1);
  }
//
//  int2 output_pos2 = (int2)(outpos_main + out_w2, out_nh);
//  if (out_w2 < old_w){
//    write_imageh(output_image, output_pos2, output2);
//  }
//
//  int2 output_pos3 = (int2)(outpos_main + out_w3, out_nh);
//  if (out_w3 < old_w){
//    write_imageh(output_image, output_pos3, output3);
//  }
//
//  int2 output_pos4 = (int2)(outpos_main + out_w4, out_nh);
//  if (out_w4 < old_w){
//    write_imageh(output_image, output_pos4, output4);
//  }
//
//  int2 output_pos5 = (int2)(outpos_main + out_w5, out_nh);
//  if (out_w5 < old_w){
//    write_imageh(output_image, output_pos5, output5);
//
//  }
//  int2 output_pos6 = (int2)(outpos_main + out_w6, out_nh);
//  if (out_w6 < old_w){
//    write_imageh(output_image, output_pos6, output6);
//  }
//
//  int2 output_pos7 = (int2)(outpos_main + out_w7, out_nh);
//  if (out_w7 < old_w){
//    write_imageh(output_image, output_pos7, output7);
//  }

}
//__kernel void conv_1x1_c(
//    __private const int global_size_dim0,
//    __private const int global_size_dim1,
//    __private const int global_size_dim2,
//    __read_only image2d_t input_image,
//    __read_only image2d_t filter,
//#ifdef BIASE
//    __read_only image2d_t bias,
//#endif
//#ifdef BATCH_NORM
//    __read_only image2d_t new_scale,
//    __read_only image2d_t new_biase,
//#endif
//    __write_only image2d_t output_image,
//    __private const int stride,
//    __private const int offset,
//    __private const int input_c,
//    __private const int dilation,
//    __private const int input_width,  /* of one block */
//    __private const int input_height, /* of one block */
//    __private const int output_width,
//    __private const int output_height,
//    __private const int old_w) {
//
//  const int out_c = get_global_id(0);
//  const int out_w = get_global_id(1);
//  const int out_nh = get_global_id(2);
//
//  const sampler_t sampler =
//      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
//  const int2 stride_xy = (int2)(stride, stride);
//
//  for (int i = 0; i < input_c; ++i) {
//    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 0));
//    half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 1));
//    half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 2));
//    half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 3));
//
//#pragma unroll
//  for (int j = 0; j < 4; ++j) {
//    int out_w0 = out_w + global_size_dim1 * j;
//    int2 ouput_pos_in_one_block0 = (int2)(out_w0, out_nh);
//    int2 in_pos_in_one_block0 = ouput_pos_in_one_block0 * stride_xy + (int2)(offset, offset);
//
//#ifdef BIASE
//    half4 output0 = read_imageh(bias, sampler, (int2)(out_c, 0));
//#else
//    half4 output0 = 0.0f;
//#endif
//      int2 pos_in = (int2)(i * input_width + in_pos_in_one_block0.x, in_pos_in_one_block0.y);
//      half4 input0 = read_imageh(input_image, sampler, pos_in);
//
//      output0 = mad(input0.x, weight0, output0);
//      output0 = mad(input0.y, weight1, output0);
//      output0 = mad(input0.z, weight2, output0);
//      output0 = mad(input0.w, weight3, output0);
//
//#ifdef BATCH_NORM
//      output0 = output0 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) + read_imageh(new_biase, sampler, (int2)(out_c, 0));
//#endif
//
//#ifdef RELU
//      output0 = activation(output0);
//#endif
//      int outpos_main = mul24(out_c, old_w);
//      int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);
//
//      if (out_w0 < old_w) {
//        write_imageh(output_image, output_pos0, output0);
//      }
//    }
//  }
//}

/*

__kernel void conv_1x1_4(__private const int global_size_dim0,
                       __private const int global_size_dim1,
                       __private const int global_size_dim2,
                       __read_only image2d_t input_image,
                       __read_only image2d_t filter,
#ifdef BIASE
                       __read_only image2d_t bias,
#endif
#ifdef BATCH_NORM
                       __read_only image2d_t new_scale,
                       __read_only image2d_t new_biase,
#endif
                       __write_only image2d_t output_image,
                       __private const int stride,
                       __private const int offset,
                       __private const int input_c,
                       __private const int dilation,
                       __private const int input_width,
                       __private const int input_height,
                       __private const int output_width,
                       __private const int output_height) {
  const int out_c = get_global_id(0) * 4;
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                           CLK_ADDRESS_CLAMP         |
                           CLK_FILTER_NEAREST;

  int2 stride_xy = (int2)(stride, stride);
  int2 ouput_pos_in_one_block = (int2)(out_w, out_nh);
  int2 in_pos_in_one_block = ouput_pos_in_one_block * stride_xy + (int2)(offset, offset);

#ifdef BIASE
    half4 output0 = read_imageh(bias, sampler, (int2)(out_c, 0));
    half4 output1 = read_imageh(bias, sampler, (int2)(out_c + 1, 0));
    half4 output2 = read_imageh(bias, sampler, (int2)(out_c + 2, 0));
    half4 output3 = read_imageh(bias, sampler, (int2)(out_c + 3, 0));
#else
    half4 output0 = 0.0f;
    half4 output1 = 0.0f;
    half4 output2 = 0.0f;
    half4 output3 = 0.0f;
#endif

   for (int i = 0; i < input_c; ++i) {
        int2 pos_in = (int2)(i * input_width + in_pos_in_one_block.x, in_pos_in_one_block.y);
        half4 input = read_imageh(input_image, sampler, pos_in);

        half4 weight0_0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 0));
        half4 weight0_1 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 1));
        half4 weight0_2 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 2));
        half4 weight0_3 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 3));

        output0 = mad(input.x, weight0_0, output0);
        output0 = mad(input.y, weight0_1, output0);
        output0 = mad(input.z, weight0_2, output0);
        output0 = mad(input.w, weight0_3, output0);

        half4 weight1_0 = read_imageh(filter, sampler, (int2)(out_c + 1, i * 4 + 0));
        half4 weight1_1 = read_imageh(filter, sampler, (int2)(out_c + 1, i * 4 + 1));
        half4 weight1_2 = read_imageh(filter, sampler, (int2)(out_c + 1, i * 4 + 2));
        half4 weight1_3 = read_imageh(filter, sampler, (int2)(out_c + 1, i * 4 + 3));

        output1 = mad(input.x, weight1_0, output1);
        output1 = mad(input.y, weight1_1, output1);
        output1 = mad(input.z, weight1_2, output1);
        output1 = mad(input.w, weight1_3, output1);

        half4 weight2_0 = read_imageh(filter, sampler, (int2)(out_c + 2, i * 4 + 0));
        half4 weight2_1 = read_imageh(filter, sampler, (int2)(out_c + 2, i * 4 + 1));
        half4 weight2_2 = read_imageh(filter, sampler, (int2)(out_c + 2, i * 4 + 2));
        half4 weight2_3 = read_imageh(filter, sampler, (int2)(out_c + 2, i * 4 + 3));

        output2 = mad(input.x, weight2_0, output2);
        output2 = mad(input.y, weight2_1, output2);
        output2 = mad(input.z, weight2_2, output2);
        output2 = mad(input.w, weight2_3, output2);

        half4 weight3_0 = read_imageh(filter, sampler, (int2)(out_c + 3, i * 4 + 0));
        half4 weight3_1 = read_imageh(filter, sampler, (int2)(out_c + 3, i * 4 + 1));
        half4 weight3_2 = read_imageh(filter, sampler, (int2)(out_c + 3, i * 4 + 2));
        half4 weight3_3 = read_imageh(filter, sampler, (int2)(out_c + 3, i * 4 + 3));

        output3 = mad(input.x, weight3_0, output3);
        output3 = mad(input.y, weight3_1, output3);
        output3 = mad(input.z, weight3_2, output3);
        output3 = mad(input.w, weight3_3, output3);

   }

#ifdef BATCH_NORM
    output0 = output0 * read_imageh(new_scale, sampler, (int2)(out_c + 0, 0)) + read_imageh(new_biase, sampler, (int2)(out_c + 0, 0));

    output1 = output1 * read_imageh(new_scale, sampler, (int2)(out_c + 1, 0)) + read_imageh(new_biase, sampler, (int2)(out_c + 1, 0));

    output2 = output2 * read_imageh(new_scale, sampler, (int2)(out_c + 2, 0)) + read_imageh(new_biase, sampler, (int2)(out_c + 2, 0));

    output3 = output3 * read_imageh(new_scale, sampler, (int2)(out_c + 3, 0)) + read_imageh(new_biase, sampler, (int2)(out_c + 3, 0));

#endif

#ifdef RELU
  output0 = activation(output0);
  output1 = activation(output1);
  output2 = activation(output2);
  output3 = activation(output3);
#endif

  int2 output_pos0 = (int2)(out_c * global_size_dim1 + out_w, out_nh);
  write_imageh(output_image, output_pos0, output0);


  int2 output_pos1 = (int2)((out_c + 1) * global_size_dim1 + out_w, out_nh);
  write_imageh(output_image, output_pos1, output1);


  int2 output_pos2 = (int2)((out_c + 2) * global_size_dim1 + out_w, out_nh);
  write_imageh(output_image, output_pos2, output2);


  int2 output_pos3 = (int2)((out_c + 3) * global_size_dim1 + out_w, out_nh);
  write_imageh(output_image, output_pos3, output3);
}

*/

__kernel void conv_7x7(__private const int global_size_dim0,
                                              __private const int global_size_dim1,
                                              __private const int global_size_dim2,
                                              __read_only image2d_t input_image,
                                              __read_only image2d_t filter_image,

#ifdef BIASE
                                              __read_only image2d_t bias,
#endif

#ifdef BATCH_NORM
                                              __read_only image2d_t new_scale,
                                              __read_only image2d_t new_biase,
#endif

                                              __write_only image2d_t output_image,
                                              __private const int stride,
                                              __private const int offset,
                                              __private const int input_c,
                                              __private const int dilation,
                                              __private const int input_width,/* of one block */
                                              __private const int input_height,/* of one block */
                                              __private const int output_width,
                                              __private const int output_height) {

    const int out_c = get_global_id(0);
    const int out_w = get_global_id(1);
    const int out_nh = get_global_id(2);

    if (out_c >= global_size_dim0 ||
        out_w >= global_size_dim1 ||
        out_nh >= global_size_dim2) {
        return;
    }
    const filter_n0 = 4 * out_c + 0;
    const filter_n1 = 4 * out_c + 1;
    const filter_n2 = 4 * out_c + 2;
    const filter_n3 = 4 * out_c + 3;

    int2 stride_xy;
    stride_xy.x = stride;
    stride_xy.y = stride;

    int2 ouput_pos_in_one_block;
    ouput_pos_in_one_block.x = out_w;
    ouput_pos_in_one_block.y = out_nh;


    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                              CLK_ADDRESS_CLAMP          |
                              CLK_FILTER_NEAREST;

    int2 in_pos_in_one_block;
    in_pos_in_one_block.x = ouput_pos_in_one_block.x * stride + offset;
    in_pos_in_one_block.y = ouput_pos_in_one_block.y * stride + offset;

#ifdef BIASE
    half4 output = read_imageh(bias, sampler, (int2)(out_c, 0));
#else
    half4 output = 0.0f;
#endif

   half4 input;
   half4 filter[4];
   int2 filter_pos0;
   int2 filter_pos1;
   int2 filter_pos2;
   int2 filter_pos3;
   for (int i = 0; i < input_c; ++i) {
   int2 pos_in = (int2)(i * input_width + in_pos_in_one_block.x, in_pos_in_one_block.y);
        for(int j = 0; j < 7; j++){
         for(int k = 0; k < 7; k++){
          input  =  select(read_imageh(input_image, sampler,
                                (int2)(pos_in.x + (j - 3) * dilation, pos_in.y +  (k - 3) * dilation)),
                                (half4)(0.0f),
                                (ushort4)((in_pos_in_one_block.x + (j - 3) * dilation < 0 || in_pos_in_one_block.y + (k - 3) * dilation < 0 || in_pos_in_one_block.x + (j - 3) * dilation >= input_width || in_pos_in_one_block.y + (k - 3) * dilation >= input_height) << 15));
         int filter_h = k;
         int filter_w = j;
         int filter_c = i;

         filter_pos0.x = filter_c * 7 + filter_w;
         filter_pos0.y = filter_n0 * 7 + filter_h;

         filter_pos1.x = filter_c * 7 + filter_w;
         filter_pos1.y = filter_n1 * 7 + filter_h;

         filter_pos2.x = filter_c * 7 + filter_w;
         filter_pos2.y = filter_n2 * 7 + filter_h;

         filter_pos3.x = filter_c * 7 + filter_w;
         filter_pos3.y = filter_n3 * 7 + filter_h;

         filter[0] =  read_imageh(filter_image, sampler, filter_pos0);
         filter[1] =  read_imageh(filter_image, sampler, filter_pos1);
         filter[2] =  read_imageh(filter_image, sampler, filter_pos2);
         filter[3] =  read_imageh(filter_image, sampler, filter_pos3);

         output.x += dot(input, filter[0]);
         output.y += dot(input, filter[1]);
         output.z += dot(input, filter[2]);
         output.w += dot(input, filter[3]);
         }
        }
    }

#ifdef BATCH_NORM
    output = output * read_imageh(new_scale, sampler, (int2)(out_c, 0)) + read_imageh(new_biase, sampler, (int2)(out_c, 0));
#endif

#ifdef RELU
    output = activation(output);
#endif

    write_imageh(output_image, (int2)(out_c * global_size_dim1 + out_w, out_nh), output);
}

__kernel void conv_5x5(__private const int global_size_dim0,
                                              __private const int global_size_dim1,
                                              __private const int global_size_dim2,
                                              __read_only image2d_t input_image,
                                              __read_only image2d_t filter_image,

#ifdef BIASE
                                              __read_only image2d_t bias,
#endif

#ifdef BATCH_NORM
                                              __read_only image2d_t new_scale,
                                              __read_only image2d_t new_biase,
#endif

                                              __write_only image2d_t output_image,
                                              __private const int stride,
                                              __private const int offset,
                                              __private const int input_c,
                                              __private const int dilation,
                                              __private const int input_width,/* of one block */
                                              __private const int input_height,/* of one block */
                                              __private const int output_width,
                                              __private const int output_height) {

    const int out_c = get_global_id(0);
    const int out_w = get_global_id(1);
    const int out_nh = get_global_id(2);

    if (out_c >= global_size_dim0 ||
        out_w >= global_size_dim1 ||
        out_nh >= global_size_dim2) {
        return;
    }
    const filter_n0 = 4 * out_c + 0;
    const filter_n1 = 4 * out_c + 1;
    const filter_n2 = 4 * out_c + 2;
    const filter_n3 = 4 * out_c + 3;

    int2 stride_xy;
    stride_xy.x = stride;
    stride_xy.y = stride;

    int2 ouput_pos_in_one_block;
    ouput_pos_in_one_block.x = out_w;
    ouput_pos_in_one_block.y = out_nh;


    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                              CLK_ADDRESS_CLAMP          |
                              CLK_FILTER_NEAREST;

    int2 in_pos_in_one_block;
    in_pos_in_one_block.x = ouput_pos_in_one_block.x * stride + offset;
    in_pos_in_one_block.y = ouput_pos_in_one_block.y * stride + offset;

#ifdef BIASE
    half4 output = read_imageh(bias, sampler, (int2)(out_c, 0));
#else
    half4 output = 0.0f;
#endif

   half4 input;
   half4 filter[4];
   int2 filter_pos0;
   int2 filter_pos1;
   int2 filter_pos2;
   int2 filter_pos3;
   for (int i = 0; i < input_c; ++i) {
   int2 pos_in = (int2)(i * input_width + in_pos_in_one_block.x, in_pos_in_one_block.y);
        for(int j = 0; j < 5; j++){
         for(int k = 0; k < 5; k++){
          input  =  select(read_imageh(input_image, sampler,
                                (int2)(pos_in.x + (j - 2) * dilation, pos_in.y +  (k - 2) * dilation)),
                                (half4)(0.0f),
                                (ushort4)((in_pos_in_one_block.x + (j - 2) * dilation < 0 || in_pos_in_one_block.y + (k - 2) * dilation < 0 || in_pos_in_one_block.x + (j - 2) * dilation >= input_width || in_pos_in_one_block.y + (k - 2) * dilation >= input_height) << 15));
         int filter_h = k;
         int filter_w = j;
         int filter_c = i;

         filter_pos0.x = filter_c * 5 + filter_w;
         filter_pos0.y = filter_n0 * 5 + filter_h;

         filter_pos1.x = filter_c * 5 + filter_w;
         filter_pos1.y = filter_n1 * 5 + filter_h;

         filter_pos2.x = filter_c * 5 + filter_w;
         filter_pos2.y = filter_n2 * 5 + filter_h;

         filter_pos3.x = filter_c * 5 + filter_w;
         filter_pos3.y = filter_n3 * 5 + filter_h;

         filter[0] =  read_imageh(filter_image, sampler, filter_pos0);
         filter[1] =  read_imageh(filter_image, sampler, filter_pos1);
         filter[2] =  read_imageh(filter_image, sampler, filter_pos2);
         filter[3] =  read_imageh(filter_image, sampler, filter_pos3);

         output.x += dot(input, filter[0]);
         output.y += dot(input, filter[1]);
         output.z += dot(input, filter[2]);
         output.w += dot(input, filter[3]);
         }
        }
    }

#ifdef BATCH_NORM
    output = output * read_imageh(new_scale, sampler, (int2)(out_c, 0)) + read_imageh(new_biase, sampler, (int2)(out_c, 0));
#endif

#ifdef RELU
    output = activation(output);
#endif

    write_imageh(output_image, (int2)(out_c * global_size_dim1 + out_w, out_nh), output);
}

__kernel void convBNAdd_3x3(__private const int global_size_dim0,
                                              __private const int global_size_dim1,
                                              __private const int global_size_dim2,
                                              __read_only image2d_t input_image,
                                              __read_only image2d_t filter,

#ifdef BIASE
                                              __read_only image2d_t bias,
#endif

#ifdef BATCH_NORM
                                              __read_only image2d_t new_scale,
                                              __read_only image2d_t new_biase,
#endif

                                              __write_only image2d_t output_image,
                                              __private const int stride,
                                              __private const int offset,
                                              __private const int input_c,
                                              __private const int dilation,
                                              __private const int input_width,/* of one block */
                                              __private const int input_height,/* of one block */
                                              __private const int output_width,
                                              __private const int output_height) {

    const int out_c = get_global_id(0);
    const int out_w = get_global_id(1);
    const int out_nh = get_global_id(2);

    if (out_c >= global_size_dim0 ||
        out_w >= global_size_dim1 ||
        out_nh >= global_size_dim2) {
        return;
    }


    int2 stride_xy;
    stride_xy.x = stride;
    stride_xy.y = stride;

    int2 ouput_pos_in_one_block;
    ouput_pos_in_one_block.x = out_w;
    ouput_pos_in_one_block.y = out_nh;


    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                              CLK_ADDRESS_CLAMP          |
                              CLK_FILTER_NEAREST;

    int2 in_pos_in_one_block;
    in_pos_in_one_block.x = ouput_pos_in_one_block.x * stride + offset;
    in_pos_in_one_block.y = ouput_pos_in_one_block.y * stride + offset;


    half4 output = (half4)0.0f;

   half4 input[9];

   for (int i = 0; i < input_c; ++i) {
        int2 pos_in = (int2)(i * input_width + in_pos_in_one_block.x, in_pos_in_one_block.y);
        input[0] = select(read_imageh(input_image, sampler,
                            (int2)(pos_in.x - dilation, pos_in.y - dilation)),
                            (half4)(0.0f),
                            (ushort4)((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y - dilation < 0 || in_pos_in_one_block.x - dilation >= input_width || in_pos_in_one_block.y - dilation >= input_height) << 15));

        input[1] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x, pos_in.y - dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y - dilation < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y - dilation >= input_height) << 15));

        input[2] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x + dilation, pos_in.y - dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y - dilation < 0 || in_pos_in_one_block.x + dilation >= input_width || in_pos_in_one_block.y - dilation >= input_height) << 15));

        input[3] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x - dilation, pos_in.y)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x - dilation >= input_width || in_pos_in_one_block.y >= input_height) << 15));

        input[4] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x, pos_in.y)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y >= input_height) << 15));

        input[5] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x + dilation, pos_in.y)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x + dilation >= input_width || in_pos_in_one_block.y >= input_height) << 15));

        input[6] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x - dilation, pos_in.y + dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y + dilation < 0 || in_pos_in_one_block.x - dilation >= input_width || in_pos_in_one_block.y + dilation >= input_height) << 15));

        input[7] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x, pos_in.y + dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y + dilation < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y + dilation >= input_height) << 15));

        input[8] = select(read_imageh(input_image, sampler,
                          (int2)(pos_in.x + dilation, pos_in.y + dilation)),
                          (half4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y + dilation < 0 || in_pos_in_one_block.x + dilation >= input_width || in_pos_in_one_block.y + dilation >= input_height) << 15));


/*
        for (int j = 0; j < 9; ++j) {
            int2 pos_of_weight;
            pos_of_weight.x = i * 3 + j % 3;
            pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
            half4 weight_x = read_imageh(filter, sampler, pos_of_weight);
            output.x += dot(input[j], weight_x);

            pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
            half4 weight_y = read_imageh(filter, sampler, pos_of_weight);
            output.y += dot(input[j], weight_y);

            pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
            half4 weight_z = read_imageh(filter, sampler, pos_of_weight);
            output.z += dot(input[j], weight_z);

            pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
            half4 weight_w = read_imageh(filter, sampler, pos_of_weight);
            output.w += dot(input[j], weight_w);
        }
*/
            int j = 0;
            int2 pos_of_weight;
            pos_of_weight.x = i * 3 + j % 3;
            pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
            half4 weight_x = read_imageh(filter, sampler, pos_of_weight);
            output.x += dot(input[j], weight_x);

            pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
            half4 weight_y = read_imageh(filter, sampler, pos_of_weight);
            output.y += dot(input[j], weight_y);

            pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
            half4 weight_z = read_imageh(filter, sampler, pos_of_weight);
            output.z += dot(input[j], weight_z);

            pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
            half4 weight_w = read_imageh(filter, sampler, pos_of_weight);
            output.w += dot(input[j], weight_w);

            j = 1;
            pos_of_weight.x = i * 3 + j % 3;
            pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
            weight_x = read_imageh(filter, sampler, pos_of_weight);
            output.x += dot(input[j], weight_x);

            pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
            weight_y = read_imageh(filter, sampler, pos_of_weight);
            output.y += dot(input[j], weight_y);

            pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
            weight_z = read_imageh(filter, sampler, pos_of_weight);
            output.z += dot(input[j], weight_z);

            pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
            weight_w = read_imageh(filter, sampler, pos_of_weight);
            output.w += dot(input[j], weight_w);

            j = 2;
            pos_of_weight.x = i * 3 + j % 3;
            pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
            weight_x = read_imageh(filter, sampler, pos_of_weight);
            output.x += dot(input[j], weight_x);

            pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
            weight_y = read_imageh(filter, sampler, pos_of_weight);
            output.y += dot(input[j], weight_y);

            pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
            weight_z = read_imageh(filter, sampler, pos_of_weight);
            output.z += dot(input[j], weight_z);

            pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
            weight_w = read_imageh(filter, sampler, pos_of_weight);
            output.w += dot(input[j], weight_w);

            j = 3;
            pos_of_weight.x = i * 3 + j % 3;
            pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
            weight_x = read_imageh(filter, sampler, pos_of_weight);
            output.x += dot(input[j], weight_x);

            pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
            weight_y = read_imageh(filter, sampler, pos_of_weight);
            output.y += dot(input[j], weight_y);

            pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
            weight_z = read_imageh(filter, sampler, pos_of_weight);
            output.z += dot(input[j], weight_z);

            pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
            weight_w = read_imageh(filter, sampler, pos_of_weight);
            output.w += dot(input[j], weight_w);

            j = 4;
            pos_of_weight.x = i * 3 + j % 3;
            pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
            weight_x = read_imageh(filter, sampler, pos_of_weight);
            output.x += dot(input[j], weight_x);

            pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
            weight_y = read_imageh(filter, sampler, pos_of_weight);
            output.y += dot(input[j], weight_y);

            pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
            weight_z = read_imageh(filter, sampler, pos_of_weight);
            output.z += dot(input[j], weight_z);

            pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
            weight_w = read_imageh(filter, sampler, pos_of_weight);
            output.w += dot(input[j], weight_w);

            j = 5;
            pos_of_weight.x = i * 3 + j % 3;
            pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
            weight_x = read_imageh(filter, sampler, pos_of_weight);
            output.x += dot(input[j], weight_x);

            pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
            weight_y = read_imageh(filter, sampler, pos_of_weight);
            output.y += dot(input[j], weight_y);

            pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
            weight_z = read_imageh(filter, sampler, pos_of_weight);
            output.z += dot(input[j], weight_z);

            pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
            weight_w = read_imageh(filter, sampler, pos_of_weight);
            output.w += dot(input[j], weight_w);

           j = 6;
           pos_of_weight.x = i * 3 + j % 3;
           pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
           weight_x = read_imageh(filter, sampler, pos_of_weight);
           output.x += dot(input[j], weight_x);

           pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
           weight_y = read_imageh(filter, sampler, pos_of_weight);
           output.y += dot(input[j], weight_y);

           pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
           weight_z = read_imageh(filter, sampler, pos_of_weight);
           output.z += dot(input[j], weight_z);

           pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
           weight_w = read_imageh(filter, sampler, pos_of_weight);
           output.w += dot(input[j], weight_w);

           j = 7;
           pos_of_weight.x = i * 3 + j % 3;
           pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
           weight_x = read_imageh(filter, sampler, pos_of_weight);
           output.x += dot(input[j], weight_x);

           pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
           weight_y = read_imageh(filter, sampler, pos_of_weight);
           output.y += dot(input[j], weight_y);

           pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
           weight_z = read_imageh(filter, sampler, pos_of_weight);
           output.z += dot(input[j], weight_z);

           pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
           weight_w = read_imageh(filter, sampler, pos_of_weight);
           output.w += dot(input[j], weight_w);

           j = 8;
           pos_of_weight.x = i * 3 + j % 3;
           pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
           weight_x = read_imageh(filter, sampler, pos_of_weight);
           output.x += dot(input[j], weight_x);

           pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
           weight_y = read_imageh(filter, sampler, pos_of_weight);
           output.y += dot(input[j], weight_y);

           pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
           weight_z = read_imageh(filter, sampler, pos_of_weight);
           output.z += dot(input[j], weight_z);

           pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
           weight_w = read_imageh(filter, sampler, pos_of_weight);
           output.w += dot(input[j], weight_w);

    }

#ifdef BATCH_NORM
    output = output * read_imageh(new_scale, sampler, (int2)(out_c, 0)) + read_imageh(new_biase, sampler, (int2)(out_c, 0));
#endif

#ifdef BIASE
    output += read_imageh(bias, sampler, (int2)(out_c * global_size_dim1 + out_w, out_nh));
#endif

#ifdef RELU
    output = activation(output);
#endif

    write_imageh(output_image, (int2)(out_c * global_size_dim1 + out_w, out_nh), output);
}

__kernel void convBNAdd_1x1(__private const int global_size_dim0,
                       __private const int global_size_dim1,
                       __private const int global_size_dim2,
                       __read_only image2d_t input_image,
                       __read_only image2d_t filter,
#ifdef BIASE
                       __read_only image2d_t bias,
#endif
#ifdef BATCH_NORM
                       __read_only image2d_t new_scale,
                       __read_only image2d_t new_biase,
#endif
                       __write_only image2d_t output_image,
                       __private const int stride,
                       __private const int offset,
                       __private const int input_c,
                       __private const int dilation,
                       __private const int input_width,/* of one block */
                       __private const int input_height,/* of one block */
                       __private const int output_width,
                       __private const int output_height) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                           CLK_ADDRESS_CLAMP         |
                           CLK_FILTER_NEAREST;

  const uint kernelHXW = 1;
  int2 stride_xy = (int2)(stride, stride);
  int2 ouput_pos_in_one_block = (int2)(out_w, out_nh);
  int2 in_pos_in_one_block = ouput_pos_in_one_block * stride_xy + (int2)(offset, offset);


  half4 output = 0.0f;

   for (int i = 0; i < input_c; ++i) {
        int2 pos_in = (int2)(i * input_width + in_pos_in_one_block.x, in_pos_in_one_block.y);
        half4 input = read_imageh(input_image, sampler, pos_in);

        half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 0));
        half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 1));
        half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 2));
        half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 3));
/*
        output.x = dot(input, weight0);
        output.y = dot(input, weight1);
        output.z = dot(input, weight2);
        output.w = dot(input, weight3);
*/

        output = mad(input.x, weight0, output);
        output = mad(input.y, weight1, output);
        output = mad(input.z, weight2, output);
        output = mad(input.w, weight3, output);

   }

#ifdef BATCH_NORM
    output = output * read_imageh(new_scale, sampler, (int2)(out_c, 0)) + read_imageh(new_biase, sampler, (int2)(out_c, 0));
#endif

#ifdef BIASE
   output += read_imageh(bias, sampler, (int2)(out_c * global_size_dim1 + out_w, out_nh));
#endif

#ifdef RELU
  output = activation(output);
#endif

  int2 output_pos = (int2)(out_c * global_size_dim1 + out_w, out_nh);
  write_imageh(output_image, output_pos, output);
}

__kernel void convBNAdd_1x1_spl(
        __private const int global_size_dim0, __private const int global_size_dim1,
        __private const int global_size_dim2, __read_only image2d_t input_image,
        __read_only image2d_t filter,
#ifdef BIASE
        __read_only image2d_t bias,
#endif
#ifdef BATCH_NORM
        __read_only image2d_t new_scale, __read_only image2d_t new_biase,
#endif
        __write_only image2d_t output_image, __private const int stride,
        __private const int offset, __private const int input_c,
        __private const int dilation,
        __private const int input_width,  /* of one block */
        __private const int input_height, /* of one block */
        __private const int output_width,
        __private const int output_height,
        __private const int old_w
) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int out_w0 = out_w;
  int out_w1 = out_w + global_size_dim1;
  int out_w2 = out_w + global_size_dim1 * 2;
  int out_w3 = out_w + global_size_dim1 * 3;

//  int out_w1 = out_w + global_size_dim1;
//  int out_w2 = out_w + global_size_dim1 * 2;
//  int out_w3 = out_w + global_size_dim1 * 3;

  const sampler_t sampler =
          CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int2 stride_xy = (int2)(stride, stride);

  int2 ouput_pos_in_one_block0 = (int2)(out_w0, out_nh);
  int2 in_pos_in_one_block0 =
          ouput_pos_in_one_block0 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block1 = (int2)(out_w1, out_nh);
  int2 in_pos_in_one_block1 =
          ouput_pos_in_one_block1 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block2 = (int2)(out_w2, out_nh);
  int2 in_pos_in_one_block2 =
          ouput_pos_in_one_block2 * stride_xy + (int2)(offset, offset);

  int2 ouput_pos_in_one_block3 = (int2)(out_w3, out_nh);
  int2 in_pos_in_one_block3 =
          ouput_pos_in_one_block3 * stride_xy + (int2)(offset, offset);

  
  half4 output0 = 0.0f;
  half4 output1 = 0.0f;
  half4 output2 = 0.0f;
  half4 output3 = 0.0f;

  for (int i = 0; i < input_c; ++i) {
    // ------------0---------------
    int2 pos_in = (int2)(i * input_width + in_pos_in_one_block0.x, in_pos_in_one_block0.y);
    half4 input0 = read_imageh(input_image, sampler, pos_in);

    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 0));
    half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 1));
    half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 2));
    half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i * 4 + 3));

    output0 = mad(input0.x, weight0, output0);
    output0 = mad(input0.y, weight1, output0);
    output0 = mad(input0.z, weight2, output0);
    output0 = mad(input0.w, weight3, output0);

    // -------------1--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block1.x, in_pos_in_one_block1.y);
    half4 input1 = read_imageh(input_image, sampler, pos_in);
    //
    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
    //    * 4 + 3));

    output1 = mad(input1.x, weight0, output1);
    output1 = mad(input1.y, weight1, output1);
    output1 = mad(input1.z, weight2, output1);
    output1 = mad(input1.w, weight3, output1);

    // -------------2--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block2.x, in_pos_in_one_block2.y);
    half4 input2 = read_imageh(input_image, sampler, pos_in);

    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
    //    * 4 + 3));

    output2 = mad(input2.x, weight0, output2);
    output2 = mad(input2.y, weight1, output2);
    output2 = mad(input2.z, weight2, output2);
    output2 = mad(input2.w, weight3, output2);

    // -------------3--------------
    pos_in = (int2)(i * input_width + in_pos_in_one_block3.x, in_pos_in_one_block3.y);
    half4 input3 = read_imageh(input_image, sampler, pos_in);

    //    half4 weight0 = read_imageh(filter, sampler, (int2)(out_c, i * 4 +
    //    0)); half4 weight1 = read_imageh(filter, sampler, (int2)(out_c, i * 4
    //    + 1)); half4 weight2 = read_imageh(filter, sampler, (int2)(out_c, i *
    //    4 + 2)); half4 weight3 = read_imageh(filter, sampler, (int2)(out_c, i
    //    * 4 + 3));

    output3 = mad(input3.x, weight0, output3);
    output3 = mad(input3.y, weight1, output3);
    output3 = mad(input3.z, weight2, output3);
    output3 = mad(input3.w, weight3, output3);
  }

#ifdef BATCH_NORM
  output0 = output0 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output1 = output1 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output2 = output2 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));

    output3 = output3 * read_imageh(new_scale, sampler, (int2)(out_c, 0)) +
          read_imageh(new_biase, sampler, (int2)(out_c, 0));
#endif

#ifdef BIASE
  output0= read_imageh(bias, sampler, (int2)(out_c, 0));
  output1 = read_imageh(bias, sampler, (int2)(out_c, 0));
  output2 = read_imageh(bias, sampler, (int2)(out_c, 0));
  output3 = read_imageh(bias, sampler, (int2)(out_c, 0));
#endif
      
#ifdef RELU
  output0 = activation(output0);
  output1 = activation(output1);
  output2 = activation(output2);
  output3 = activation(output3);
#endif
  int outpos_main = mul24(out_c , old_w);
  int2 output_pos0 = (int2)(outpos_main + out_w0, out_nh);

  if (out_w0 < old_w) {
    write_imageh(output_image, output_pos0, output0);
  }
  int2 output_pos1 = (int2)(outpos_main + out_w1, out_nh);
  if (out_w1 < old_w){
    write_imageh(output_image, output_pos1, output1);
  }

  int2 output_pos2 = (int2)(outpos_main + out_w2, out_nh);
  if (out_w2 < old_w){
    write_imageh(output_image, output_pos2, output2);
  }

  int2 output_pos3 = (int2)(outpos_main + out_w3, out_nh);
  if (out_w3 < old_w){
    write_imageh(output_image, output_pos3, output3);
  }
}







