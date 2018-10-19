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


#define BIASE
#define BATCH_NORM
#define RELU

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline half4 activation(half4 in
#ifdef PRELU
                        ,
                        half4 prelu_alpha
#endif
) {
  half4 output;
#ifdef PRELU
  output = select(prelu_alpha * in, in, in >= (half4)0.0);
#endif

#ifdef RELU
  output = fmax(in, (half4)(0.0f));
#endif
  return output;
}

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
        printf(" out of range ");
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

        for (int j = 0; j < 9; ++j) {
            int2 fuck;
            fuck.x = i * 3 + j % 3;
            fuck.y = out_c * 4 * 3 + 0 * 3 + j / 3;
            half4 weight_x = read_imageh(filter, sampler, fuck);
            output.x += dot(input[j], weight_x);

            fuck.y = out_c * 4 * 3 + 1 * 3 + j / 3;
            half4 weight_y = read_imageh(filter, sampler, fuck);
            output.y += dot(input[j], weight_y);

            fuck.y = out_c * 4 * 3 + 2 * 3 + j / 3;
            half4 weight_z = read_imageh(filter, sampler, fuck);
            output.z += dot(input[j], weight_z);

            fuck.y = out_c * 4 * 3 + 3 * 3 + j / 3;
            half4 weight_w = read_imageh(filter, sampler, fuck);
            output.w += dot(input[j], weight_w);
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

    int2 pos_in_input_block = (int2)(out_c * input_width, batch_index * input_height);
    int weight_y_to = out_c * 12;

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

    for (int j = 0; j < 9; ++j) {
        half4 input = inputs[j];
        half4 weight0 = read_imageh(filter, sampler, (int2)(j % 3, weight_y_to + j / 3));
        half4 weight1 = read_imageh(filter, sampler, (int2)(j % 3, weight_y_to + 3 + j / 3));
        half4 weight2 = read_imageh(filter, sampler, (int2)(j % 3, weight_y_to + 6 + j / 3));
        half4 weight3 = read_imageh(filter, sampler, (int2)(j % 3, weight_y_to + 9 + j / 3));
        output.x += input.x * weight0.x;
        output.y += input.y * weight1.x;
        output.z += input.z * weight2.x;
        output.w += input.w * weight3.x;
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

  int out_c_p = 0, out_w_p = 0, out_nh_p = 0;

/*
  if (out_c == out_c_p && out_w == out_w_p && out_nh == out_nh_p) {
        float4 out = (float4)(output.x, output.y, output.z, output.w);
        printf(" after bias output4 = %v4hlf \n", out);

  }

*/

   for (int i = 0; i < input_c; ++i) {
        int2 pos_in = (int2)(i * input_width + in_pos_in_one_block.x, in_pos_in_one_block.y);
        half4 input = read_imageh(input_image, sampler, pos_in);

        half4 weight_x = read_imageh(filter, sampler, (int2)(i, out_c * 4 + 0));
        output.x += dot(input, weight_x);

        half4 weight_y = read_imageh(filter, sampler, (int2)(i, out_c * 4 + 1));
        output.y += dot(input, weight_y);

        half4 weight_z = read_imageh(filter, sampler, (int2)(i, out_c * 4 + 2));
        output.z += dot(input, weight_z);

        half4 weight_w = read_imageh(filter, sampler, (int2)(i, out_c * 4 + 3));
        output.w += dot(input, weight_w);
/*
        if (out_c == out_c_p && out_w == out_w_p && out_nh == out_nh_p) {
            printf("x - %d \n", pos_in.x);

            printf("y - %d \n", pos_in.y);

            float4 in = (float4)(input.x, input.y, input.z, input.w);
            printf("input4 = %v4hlf \n", in);

            float4 w = (float4)(weight_x.x, weight_x.y, weight_x.z, weight_x.w);
            printf("weight4 = %v4hlf \n", w);

        }
*/
  }
/*
  if (out_c == out_c_p && out_w == out_w_p && out_nh == out_nh_p) {
        float4 out = (float4)(output.x, output.y, output.z, output.w);
        printf("output4 = %v4hlf \n", out);

  }

*/

#ifdef BATCH_NORM
    output = output * read_imageh(new_scale, sampler, (int2)(out_c, 0)) + read_imageh(new_biase, sampler, (int2)(out_c, 0));
#endif

/*
  if (out_c == out_c_p && out_w == out_w_p && out_nh == out_nh_p) {
        float4 out = (float4)(output.x, output.y, output.z, output.w);
        printf(" after batch output4 = %v4hlf \n", out);

  }

*/

#ifdef RELU
  output = activation(output);
#endif

/*
  if (out_c == out_c_p && out_w == out_w_p && out_nh == out_nh_p) {
        float4 out = (float4)(output.x, output.y, output.z, output.w);
        printf(" after relu output4 = %v4hlf \n", out);

  }

*/

  int2 output_pos = (int2)(out_c * global_size_dim1 + out_w, out_nh);
  write_imageh(output_image, output_pos, output);
}
