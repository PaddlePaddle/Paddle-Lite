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

__kernel void depth_conv2d_3x3(__private const int global_size_dim0,
                                              __private const int global_size_dim1,
                                              __private const int global_size_dim2,
                                              __read_only image2d_t input,
                                              __read_only image2d_t filter,
#if defined(BIASE_CH) || defined(BIASE_ELE)
                                              __read_only image2d_t bias,
#endif
#ifdef BATCH_NORM
                                              __read_only image2d_t new_scale,
                                              __read_only image2d_t new_biase,
#endif
                                              __write_only image2d_t output_image,
                                              __private const int stride,
                                              __private const int offset,
                                              __private const int dilation,
                                              __private const int input_c,
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

#ifdef BIASE_CH
    CL_DTYPE4 output = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, (int2)(out_c, 0));
#elif defined(BIASE_ELE)
    CL_DTYPE4 output = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, output_pos);
#else
    CL_DTYPE4 output = 0.0f;
#endif

    const int filter_width = 3;
    const int filter_height = 3;

    int2 pos_in_input_block = (int2)(out_c * input_width, batch_index * input_height);

    int2 pos_in_filter_block = (int2)(out_c * filter_width, batch_index * filter_height);

    int filter_x = pos_in_filter_block.x ;
    int filter_y = pos_in_filter_block.y ;

    CL_DTYPE4 inputs[9];

        inputs[0] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x - 1, pos_in_input_block.y + in_pos_in_one_block.y - 1)),
                           (CL_DTYPE4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x - 1 < 0 || in_pos_in_one_block.y - 1 < 0 || in_pos_in_one_block.x - 1 >= input_width || in_pos_in_one_block.y - 1 >= input_height) << 15));

        inputs[1] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x, pos_in_input_block.y + in_pos_in_one_block.y - 1)),
                           (CL_DTYPE4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y - 1 < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y - 1 >= input_height) << 15));

        inputs[2] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x + 1, pos_in_input_block.y + in_pos_in_one_block.y - 1)),
                           (CL_DTYPE4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x + 1 < 0 || in_pos_in_one_block.y - 1 < 0 || in_pos_in_one_block.x + 1 >= input_width || in_pos_in_one_block.y - 1 >= input_height) << 15));

        inputs[3] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x - 1, pos_in_input_block.y + in_pos_in_one_block.y)),
                           (CL_DTYPE4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x - 1 < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x - 1 >= input_width || in_pos_in_one_block.y >= input_height) << 15));
        /*
        if (output_pos.x == 112 && output_pos.y == 0) {
              CL_DTYPE4 input1 = inputs[3];
              float4 in = (float4)(input1.x, input1.y, input1.z, input1.w);
              printf(" input4 3 - %v4hlf \n", in);
              printf(" --- %d ---\n", in_pos_in_one_block.x - 1);
        }
        */


        inputs[4] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x, pos_in_input_block.y + in_pos_in_one_block.y)),
                           (CL_DTYPE4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y >= input_height) << 15));

        inputs[5] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x + 1, pos_in_input_block.y + in_pos_in_one_block.y)),
                           (CL_DTYPE4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x + 1 < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x + 1 >= input_width || in_pos_in_one_block.y >= input_height) << 15));

        inputs[6] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x - 1, pos_in_input_block.y + in_pos_in_one_block.y + 1)),
                           (CL_DTYPE4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x - 1 < 0 || in_pos_in_one_block.y + 1 < 0 || in_pos_in_one_block.x - 1 >= input_width || in_pos_in_one_block.y + 1 >= input_height) << 15));

        inputs[7] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x, pos_in_input_block.y + in_pos_in_one_block.y + 1)),
                           (CL_DTYPE4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y + 1 < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y + 1 >= input_height) << 15));

        inputs[8] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(pos_in_input_block.x + in_pos_in_one_block.x + 1, pos_in_input_block.y + in_pos_in_one_block.y + 1)),
                           (CL_DTYPE4)(0.0f),
                           (ushort4)((in_pos_in_one_block.x + 1 < 0 || in_pos_in_one_block.y + 1 < 0 || in_pos_in_one_block.x + 1 >= input_width || in_pos_in_one_block.y + 1 >= input_height) << 15));

    CL_DTYPE4 filters[9];
    filters[0] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x,filter_y));
    filters[1] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x + 1,filter_y));
    filters[2] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x + 2,filter_y));
    filters[3] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x,filter_y + 1));
    filters[4] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x + 1,filter_y + 1));
    filters[5] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x + 2,filter_y + 1));
    filters[6] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x,filter_y + 2));
    filters[7] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x + 1,filter_y + 2));
    filters[8] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x + 2,filter_y + 2));

    for(int i = 0 ;i < 9 ; i++){
     output += inputs[i] * filters[i];
    }
#ifdef BATCH_NORM
    output = output * READ_IMG_TYPE(CL_DTYPE_CHAR, new_scale, sampler, (int2)(out_c, 0)) + READ_IMG_TYPE(CL_DTYPE_CHAR, new_biase, sampler, (int2)(out_c, 0));
#endif

#ifdef RELU
    output = activation_type4(output);
#endif


    /*

    if (output_pos.x == 112 && output_pos.y == 0) {

        for (int i = 0; i < 9; ++i) {
            CL_DTYPE4 input1 = inputs[i];
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

    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output);

}



__kernel void depth_conv2d_3x3s1(__private const int ou_ch_blk,
                                              __private const int ou_w_blk,
                                              __private const int ou_nh,
                                              __read_only image2d_t input,
                                              __read_only image2d_t filter,
#if defined(BIASE_CH) || defined(BIASE_ELE)
                                              __read_only image2d_t bias,
#endif
#ifdef BATCH_NORM
                                              __read_only image2d_t new_scale,
                                              __read_only image2d_t new_biase,
#endif
                                              __write_only image2d_t output_image,
                                              __private const int stride,
                                              __private const int pad,
                                              __private const int dilation,
                                              __private const int in_ch,
                                              __private const int in_w,/* of one block */
                                              __private const int in_h, /* of one block */
                                              __private const int ou_w,
                                              __private const int ou_h) {

    const int ou_ch_blk_id = get_global_id(0);
    const int ou_w_blk_id = get_global_id(1);
    const int ou_nh_id = get_global_id(2);
    const int w_blk_size = 2;

    const int batch_id = ou_nh_id / ou_h;
    int ou_col_id = ou_w_blk_id * w_blk_size;
    int ou_row_id = ou_nh_id % ou_h;
    int ou_x = mad24(ou_ch_blk_id, ou_w, ou_col_id);

    // input pos in one block and on batch
    int col_id = ou_col_id - pad;
    int row_id = ou_row_id - pad;

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                              CLK_ADDRESS_CLAMP          |
                              CLK_FILTER_NEAREST;

#ifdef BIASE_CH
    CL_DTYPE4 output[2];
    output[0] = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, (int2)(ou_ch_blk_id, 0));
    output[1] = output[0];
#elif defined(BIASE_ELE)
    CL_DTYPE4 output[2];
    output[0] = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, (int2)(ou_x, ou_nh_id));
    if (ou_col_id + 1 < ou_w) {
        output[1] = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, (int2)(ou_x + 1, ou_nh_id));
    }
#else
    CL_DTYPE4 output[2] = {0.0f};
#endif

    CL_DTYPE4 inputs[12];

    int filter_x = ou_ch_blk_id * 3;
    int filter_y = 0;
    CL_DTYPE4 filters[9];
    filters[0] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x,filter_y));
    filters[1] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x + 1,filter_y));
    filters[2] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x + 2,filter_y));

    int in_x = mad24(ou_ch_blk_id, in_w, col_id);
    int in_y = mad24(batch_id, in_h, row_id);

    int y0 = select(in_y, -1, row_id < 0 || row_id >= in_h);
    int x0 = select(in_x, -1, col_id < 0 || col_id >= in_w);
    inputs[0] = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x0, y0));
    int x1 = select(in_x + 1, -1, col_id + 1 < 0 || col_id + 1 >= in_w);
    inputs[1] = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x1, y0));
    int x2 = select(in_x + 2, -1, col_id + 2 < 0 || col_id + 2 >= in_w);
    inputs[2] = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x2, y0));
    int x3 = select(in_x + 3, -1, col_id + 3 < 0 || col_id + 3 >= in_w);
    inputs[3] = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x3, y0));

    output[0] = mad(inputs[0], filters[0], output[0]);
    output[1] = mad(inputs[1], filters[0], output[1]);

    output[0] = mad(inputs[1], filters[1], output[0]);
    output[1] = mad(inputs[2], filters[1], output[1]);

    output[0] = mad(inputs[2], filters[2], output[0]);
    output[1] = mad(inputs[3], filters[2], output[1]);


    filters[3] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x,filter_y + 1));
    filters[4] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x + 1,filter_y + 1));
    filters[5] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x + 2,filter_y + 1));


    int y1 = select(in_y + 1, -1, row_id + 1 < 0 || row_id + 1 >= in_h);
    inputs[4] = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x0, y1));
    inputs[5] = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x1, y1));
    inputs[6] = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x2, y1));
    inputs[7] = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x3, y1));


    output[0] = mad(inputs[4], filters[3], output[0]);
    output[1] = mad(inputs[5], filters[3], output[1]);

    output[0] = mad(inputs[5], filters[4], output[0]);
    output[1] = mad(inputs[6], filters[4], output[1]);

    output[0] = mad(inputs[6], filters[5], output[0]);
    output[1] = mad(inputs[7], filters[5], output[1]);


    filters[6] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x,filter_y + 2));
    filters[7] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x + 1,filter_y + 2));
    filters[8] =  READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler,(int2)(filter_x + 2,filter_y + 2));

    int y2 = select(in_y + 2, -1, row_id + 2 < 0 || row_id + 2 >= in_h);
    inputs[8] = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x0, y2));
    inputs[9] = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x1, y2));
    inputs[10] = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x2, y2));
    inputs[11] = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, (int2)(x3, y2));


    output[0] = mad(inputs[8], filters[6], output[0]);
    output[1] = mad(inputs[9], filters[6], output[1]);

    output[0] = mad(inputs[9], filters[7], output[0]);
    output[1] = mad(inputs[10], filters[7], output[1]);

    output[0] = mad(inputs[10], filters[8], output[0]);
    output[1] = mad(inputs[11], filters[8], output[1]);
#ifdef BATCH_NORM
    CL_DTYPE4 scale = READ_IMG_TYPE(CL_DTYPE_CHAR, new_scale, sampler, (int2)(ou_ch_blk_id, 0));
    CL_DTYPE4 biase = READ_IMG_TYPE(CL_DTYPE_CHAR, new_biase, sampler, (int2)(ou_ch_blk_id, 0));
    output[0] = mad(scale, output[0], biase);
    if (ou_col_id + 1 < ou_w) {
        output[1] = mad(scale, output[1], biase);
    }
#endif

#ifdef RELU
    output[0] = activation_type4(output[0]);
    output[1] = activation_type4(output[1]);
#endif

    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, (int2)(ou_x, ou_nh_id), output[0]);
    if (ou_col_id + 1 < ou_w) {
        WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, (int2)(ou_x + 1, ou_nh_id), output[1]);
    }

}

