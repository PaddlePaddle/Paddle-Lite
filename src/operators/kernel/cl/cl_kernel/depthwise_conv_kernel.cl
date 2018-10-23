#define BIASE
#define BATCH_NORM
#define RELU
#include "cl_common.h"

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

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
                                              __private const int output_height,
                                              __private const int filter_width,
                                              __private const int filter_height) {

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