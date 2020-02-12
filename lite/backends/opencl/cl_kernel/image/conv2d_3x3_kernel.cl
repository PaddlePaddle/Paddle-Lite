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

#define DEBUG
__kernel void conv2d_3x3(__private const int global_size_dim0,
                         __private const int global_size_dim1,
                         __private const int global_size_dim2,
                         __read_only image2d_t input_image,
                         __read_only image2d_t filter,
#if defined(BIASE_CH) || defined(BIASE_ELE)
                         __read_only image2d_t bias,
#endif
                         __write_only image2d_t output_image,
                         __private const int stride,
                         __private const int offset,
                         __private const int input_c,
                         __private const int dilation,
                         __private const int input_width,/* of one block */
                         __private const int input_height,/* of one block */
                         __private const int output_width,
                         __private const int output_height,
                         __private const int output_c,
                         __private const int filter_channel,
						 __private const int filter_width,
						 __private const int filter_height,
                         __private const int group) {

    const int out_c = get_global_id(0);
    const int out_w = get_global_id(1);
    const int out_nh = get_global_id(2);

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                              CLK_ADDRESS_CLAMP          |
                              CLK_FILTER_NEAREST;


#ifdef DEBUG
    if (out_c == 0 && out_w == 0 && out_nh == 0) {
      printf("kkkkkkkkkkkkkkkkkkkkkkkkk\n");
      printf("global_size_dim0:%d\n", global_size_dim0); 
      printf("global_size_dim1:%d\n", global_size_dim1);
      printf("global_size_dim2:%d\n", global_size_dim2);

      printf("stride:%d\n", stride);
      printf("offset:%d\n", offset);
      printf("input_c:%d\n", input_c);
      printf("dilation:%d\n", dilation);
      printf("input_width:%d\n", input_width);
      printf("input_height:%d\n", input_height);
      printf("output_width:%d\n", output_width);
      printf("output_height:%d\n", output_height);
      printf("output_c:%d\n", output_c);
      printf("filter_channel:%d\n", filter_channel);
      printf("filter_height:%d\n", filter_height);
      printf("filter_width:%d\n", filter_width);
      printf("group:%d\n", group);

     // filter
     printf("================================== filter ==============================\n");
     int2 print_pos;
     CL_DTYPE4 ff;
     print_pos.x = 0;
     print_pos.y = 0;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);
     print_pos.x = 0;
     print_pos.y = 1;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);
     print_pos.x = 0;
     print_pos.y = 2;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);
     print_pos.x = 0;
     print_pos.y = 3;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);
     print_pos.x = 0;
     print_pos.y = 4;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);
     print_pos.x = 0;
     print_pos.y = 5;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);

     print_pos.x = 1;
     print_pos.y = 0;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);
     print_pos.x = 1;
     print_pos.y = 1;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);
     print_pos.x = 1;
     print_pos.y = 2;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);
     print_pos.x = 1;
     print_pos.y = 3;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);
     print_pos.x = 1;
     print_pos.y = 4;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);
     print_pos.x = 1;
     print_pos.y = 5;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);

     print_pos.x = 2;
     print_pos.y = 0;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);
     print_pos.x = 2;
     print_pos.y = 1;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);
     print_pos.x = 2;
     print_pos.y = 2;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);
     print_pos.x = 2;
     print_pos.y = 3;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);
     print_pos.x = 2;
     print_pos.y = 4;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);
     print_pos.x = 2;
     print_pos.y = 5;
     ff = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tfilter(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ff.x, ff.y, ff.z, ff.w);



     // 00,01,02
     printf("======================================= input ======================================\n");
     CL_DTYPE4 ii;
     print_pos.x = 0;
     print_pos.y = 0;
     ii = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tinput(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ii.x, ii.y, ii.z, ii.w);
     print_pos.x = 0;
     print_pos.y = 1;
     ii = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tinput(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ii.x, ii.y, ii.z, ii.w);
     print_pos.x = 0;
     print_pos.y = 2;
     ii = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tinput(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ii.x, ii.y, ii.z, ii.w);
     // 10,11,12
     print_pos.x = 1;
     print_pos.y = 0;
     ii = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tinput(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ii.x, ii.y, ii.z, ii.w);
     print_pos.x = 1;
     print_pos.y = 1;
     ii = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tinput(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ii.x, ii.y, ii.z, ii.w);
     print_pos.x = 1;
     print_pos.y = 2;
     ii = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tinput(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ii.x, ii.y, ii.z, ii.w);
     // 20,21,22
     print_pos.x = 2;
     print_pos.y = 0;
     ii = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tinput(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ii.x, ii.y, ii.z, ii.w);
     print_pos.x = 2;
     print_pos.y = 1;
     ii = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tinput(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ii.x, ii.y, ii.z, ii.w);
     print_pos.x = 2;
     print_pos.y = 2;
     ii = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler, print_pos);
     printf("out[c|w|nh]:%d %d %d\tinput(%d.%d): %f %f %f %f\n", out_c, out_w, out_nh, print_pos.x, print_pos.y, ii.x, ii.y, ii.z, ii.w);

#if 0
     printf("input[0]:%.0f %.0f %.0f %.0f\n", input[0].x, input[0].y, input[0].z, input[0].w);
     printf("input[1]:%.0f %.0f %.0f %.0f\n", input[1].x, input[1].y, input[1].z, input[1].w);
     printf("input[2]:%.0f %.0f %.0f %.0f\n", input[2].x, input[2].y, input[2].z, input[2].w);
     printf("input[3]:%.0f %.0f %.0f %.0f\n", input[3].x, input[3].y, input[3].z, input[3].w);
     printf("input[4]:%.0f %.0f %.0f %.0f\n", input[4].x, input[4].y, input[4].z, input[4].w);
     printf("input[5]:%.0f %.0f %.0f %.0f\n", input[5].x, input[5].y, input[5].z, input[5].w);
     printf("input[6]:%.0f %.0f %.0f %.0f\n", input[6].x, input[6].y, input[6].z, input[6].w);
     printf("input[7]:%.0f %.0f %.0f %.0f\n", input[7].x, input[7].y, input[7].z, input[7].w);
     printf("input[8]:%.0f %.0f %.0f %.0f\n", input[8].x, input[8].y, input[8].z, input[8].w);
#endif

/*
global_size_dim0:1
global_size_dim1:2
global_size_dim2:2
stride:2
offset:0
input_c:1
dilation:1
input_width:3
input_height:3
output_width:2
output_height:2
output_c:1
filter_channel:1
group:1
*/
    }
#endif

    int2 output_pos = (int2)(out_c * global_size_dim1 + out_w, out_nh);

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

    int2 in_pos_in_one_block;
    in_pos_in_one_block.x = ouput_pos_in_one_block.x * stride + offset;
    in_pos_in_one_block.y = ouput_pos_in_one_block.y * stride + offset;

#ifdef BIASE_CH
    CL_DTYPE4 output = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, (int2)(out_c, 0));
#elif defined(BIASE_ELE)
    CL_DTYPE4 output = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, output_pos);
#else
    CL_DTYPE4 output = 0.0f;
#endif

    CL_DTYPE4 input[9]; // 3x3 region of input
    if (group == 1) {
        for (int i = 0; i < input_c; ++i) { // each run for 3x3
            int2 pos_in = (int2)(i * input_width + in_pos_in_one_block.x, in_pos_in_one_block.y);
            input[0] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                                (int2)(pos_in.x - dilation, pos_in.y - dilation)),
                                (CL_DTYPE4)(0.0f),
                                (ushort4)((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y - dilation < 0 || in_pos_in_one_block.x - dilation >= input_width || in_pos_in_one_block.y - dilation >= input_height) << 15));

            input[1] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                              (int2)(pos_in.x, pos_in.y - dilation)),
                              (CL_DTYPE4)(0.0f),
                              (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y - dilation < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y - dilation >= input_height) << 15));

            input[2] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                              (int2)(pos_in.x + dilation, pos_in.y - dilation)),
                              (CL_DTYPE4)(0.0f),
                              (ushort4)((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y - dilation < 0 || in_pos_in_one_block.x + dilation >= input_width || in_pos_in_one_block.y - dilation >= input_height) << 15));

            input[3] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                              (int2)(pos_in.x - dilation, pos_in.y)),
                              (CL_DTYPE4)(0.0f),
                              (ushort4)((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x - dilation >= input_width || in_pos_in_one_block.y >= input_height) << 15));

            input[4] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                              (int2)(pos_in.x, pos_in.y)),
                              (CL_DTYPE4)(0.0f),
                              (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y >= input_height) << 15));

            input[5] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                              (int2)(pos_in.x + dilation, pos_in.y)),
                              (CL_DTYPE4)(0.0f),
                              (ushort4)((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y < 0 || in_pos_in_one_block.x + dilation >= input_width || in_pos_in_one_block.y >= input_height) << 15));

            input[6] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                              (int2)(pos_in.x - dilation, pos_in.y + dilation)),
                              (CL_DTYPE4)(0.0f),
                              (ushort4)((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y + dilation < 0 || in_pos_in_one_block.x - dilation >= input_width || in_pos_in_one_block.y + dilation >= input_height) << 15));

            input[7] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                              (int2)(pos_in.x, pos_in.y + dilation)),
                              (CL_DTYPE4)(0.0f),
                              (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y + dilation < 0 || in_pos_in_one_block.x >= input_width || in_pos_in_one_block.y + dilation >= input_height) << 15));

            input[8] = select(READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                              (int2)(pos_in.x + dilation, pos_in.y + dilation)),
                              (CL_DTYPE4)(0.0f),
                              (ushort4)((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y + dilation < 0 || in_pos_in_one_block.x + dilation >= input_width || in_pos_in_one_block.y + dilation >= input_height) << 15));

                int j = 0;
                int2 pos_of_weight;
                pos_of_weight.x = i * 3 + j % 3;
                pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
                //pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
                CL_DTYPE4 weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.x += dot(input[j], weight_x);

                pos_of_weight.y += 3;
                // pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               CL_DTYPE4 weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.y += dot(input[j], weight_y);

                pos_of_weight.y += 3;
                // pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               CL_DTYPE4 weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.z += dot(input[j], weight_z);

                pos_of_weight.y += 3;
                // pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               CL_DTYPE4 weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.w += dot(input[j], weight_w);
#ifdef DEBUG
    if (out_c == 0 && out_w == 0 && out_nh == 0) {
                printf("---- j:%d ----\n", j);
                printf("000 input[%d]:%.0f %.0f %.0f %.0f\n", j, input[j].x, input[j].y, input[j].z, input[j].w);
                printf("000 filter(%d,%d):weight_x:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 3 * 3,
				       weight_x.x, weight_x.y, weight_x.z, weight_x.w);
                printf("000 filter(%d,%d):weight_y:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 2 * 3,
                       weight_y.x, weight_y.y, weight_y.z, weight_y.w);
                printf("000 filter(%d,%d):weight_z:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 1 * 3,
                       weight_z.x, weight_z.y, weight_z.z, weight_z.w);
                printf("000 filter(%d,%d):weight_w:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 0 * 3,
                       weight_w.x, weight_w.y, weight_w.z, weight_w.w);
                printf("000 output:%.0f %.0f %.0f %.0f\n", output.x, output.y, output.z, output.w);
    }
#endif


                j = 1;
                pos_of_weight.x = i * 3 + j % 3;
                pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
                // pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.x += dot(input[j], weight_x);

                pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.y += dot(input[j], weight_y);

                pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
               //   pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
              weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.z += dot(input[j], weight_z);

                pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.w += dot(input[j], weight_w);
#ifdef DEBUG
    if (out_c == 0 && out_w == 0 && out_nh == 0) {
                printf("---- j:%d ----\n", j);
                printf("111 input[%d]:%.0f %.0f %.0f %.0f\n", j, input[j].x, input[j].y, input[j].z, input[j].w);
                printf("filter(%d,%d):weight_x:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 3 * 3,
				       weight_x.x, weight_x.y, weight_x.z, weight_x.w);
                printf("111 filter(%d,%d):weight_y:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 2 * 3,
                       weight_y.x, weight_y.y, weight_y.z, weight_y.w);
                printf("111 filter(%d,%d):weight_z:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 1 * 3,
                       weight_z.x, weight_z.y, weight_z.z, weight_z.w);
                printf("111 filter(%d,%d):weight_w:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 0 * 3,
                       weight_w.x, weight_w.y, weight_w.z, weight_w.w);
                printf("111 output:%.0f %.0f %.0f %.0f\n", output.x, output.y, output.z, output.w);
    }
#endif



                j = 2;
                pos_of_weight.x = i * 3 + j % 3;
                pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
                //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
              weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.x += dot(input[j], weight_x);

                pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.y += dot(input[j], weight_y);

                pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
               //   pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
              weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.z += dot(input[j], weight_z);

                pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
               //    pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
             weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.w += dot(input[j], weight_w);
#ifdef DEBUG
    if (out_c == 0 && out_w == 0 && out_nh == 0) {
                printf("---- j:%d ----\n", j);
                printf("222 input[%d]:%.0f %.0f %.0f %.0f\n", j, input[j].x, input[j].y, input[j].z, input[j].w);
                printf("222 filter(%d,%d):weight_x:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 3 * 3,
				       weight_x.x, weight_x.y, weight_x.z, weight_x.w);
                printf("222 filter(%d,%d):weight_y:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 2 * 3,
                       weight_y.x, weight_y.y, weight_y.z, weight_y.w);
                printf("222 filter(%d,%d):weight_z:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 1 * 3,
                       weight_z.x, weight_z.y, weight_z.z, weight_z.w);
                printf("222 filter(%d,%d):weight_w:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 0 * 3,
                       weight_w.x, weight_w.y, weight_w.z, weight_w.w);
                printf("222 output:%.0f %.0f %.0f %.0f\n", output.x, output.y, output.z, output.w);
    }
#endif



                j = 3;
                pos_of_weight.x = i * 3 + j % 3;
                pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.x += dot(input[j], weight_x);

                pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.y += dot(input[j], weight_y);

                pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.z += dot(input[j], weight_z);

                pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.w += dot(input[j], weight_w);

#ifdef DEBUG
    if (out_c == 0 && out_w == 0 && out_nh == 0) {
                printf("---- j:%d ----\n", j);
                printf("333 input[%d]:%.0f %.0f %.0f %.0f\n", j, input[j].x, input[j].y, input[j].z, input[j].w);
                printf("333 filter(%d,%d):weight_x:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 3 * 3,
				       weight_x.x, weight_x.y, weight_x.z, weight_x.w);
                printf("333 filter(%d,%d):weight_y:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 2 * 3,
                       weight_y.x, weight_y.y, weight_y.z, weight_y.w);
                printf("333 filter(%d,%d):weight_z:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 1 * 3,
                       weight_z.x, weight_z.y, weight_z.z, weight_z.w);
                printf("333 filter(%d,%d):weight_w:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 0 * 3,
                       weight_w.x, weight_w.y, weight_w.z, weight_w.w);
                printf("333 output:%.0f %.0f %.0f %.0f\n", output.x, output.y, output.z, output.w);
    }
#endif

                j = 4;
                pos_of_weight.x = i * 3 + j % 3;
                pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.x += dot(input[j], weight_x);

                pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.y += dot(input[j], weight_y);

                pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
               //   pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
              weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.z += dot(input[j], weight_z);

                pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.w += dot(input[j], weight_w);

#ifdef DEBUG
    if (out_c == 0 && out_w == 0 && out_nh == 0) {
                printf("---- j:%d ----\n", j);
                printf("444 input[%d]:%.0f %.0f %.0f %.0f\n", j, input[j].x, input[j].y, input[j].z, input[j].w);
                printf("444 filter(%d,%d):weight_x:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 3 * 3,
				       weight_x.x, weight_x.y, weight_x.z, weight_x.w);
                printf("444 filter(%d,%d):weight_y:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 2 * 3,
                       weight_y.x, weight_y.y, weight_y.z, weight_y.w);
                printf("444 filter(%d,%d):weight_z:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 1 * 3,
                       weight_z.x, weight_z.y, weight_z.z, weight_z.w);
                printf("444 filter(%d,%d):weight_w:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 0 * 3,
                       weight_w.x, weight_w.y, weight_w.z, weight_w.w);
                printf("444 output:%.0f %.0f %.0f %.0f\n", output.x, output.y, output.z, output.w);
    }
#endif

                j = 5;
                pos_of_weight.x = i * 3 + j % 3;
                pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.x += dot(input[j], weight_x);

                pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.y += dot(input[j], weight_y);

                pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.z += dot(input[j], weight_z);

                pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
               weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
                output.w += dot(input[j], weight_w);

#ifdef DEBUG
    if (out_c == 0 && out_w == 0 && out_nh == 0) {
                printf("---- j:%d ----\n", j);
                printf("555 input[%d]:%.0f %.0f %.0f %.0f\n", j, input[j].x, input[j].y, input[j].z, input[j].w);
                printf("555 filter(%d,%d):weight_x:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 3 * 3,
				       weight_x.x, weight_x.y, weight_x.z, weight_x.w);
                printf("555 filter(%d,%d):weight_y:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 2 * 3,
                       weight_y.x, weight_y.y, weight_y.z, weight_y.w);
                printf("555 filter(%d,%d):weight_z:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 1 * 3,
                       weight_z.x, weight_z.y, weight_z.z, weight_z.w);
                printf("555 filter(%d,%d):weight_w:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 0 * 3,
                       weight_w.x, weight_w.y, weight_w.z, weight_w.w);
                printf("555 output:%.0f %.0f %.0f %.0f\n", output.x, output.y, output.z, output.w);
    }
#endif

               j = 6;
               pos_of_weight.x = i * 3 + j % 3;
               pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1 : pos_of_weight.y;
              weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
               output.x += dot(input[j], weight_x);

               pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
              //   pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
              weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
               output.y += dot(input[j], weight_y);

               pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
              //   pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
              weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
               output.z += dot(input[j], weight_z);

               pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
              weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
               output.w += dot(input[j], weight_w);

#ifdef DEBUG
    if (out_c == 0 && out_w == 0 && out_nh == 0) {
                printf("---- j:%d ----\n", j);
                printf("666 input[%d]:%.0f %.0f %.0f %.0f\n", j, input[j].x, input[j].y, input[j].z, input[j].w);
                printf("666 filter(%d,%d):weight_x:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 3 * 3,
				       weight_x.x, weight_x.y, weight_x.z, weight_x.w);
                printf("666 filter(%d,%d):weight_y:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 2 * 3,
                       weight_y.x, weight_y.y, weight_y.z, weight_y.w);
                printf("666 filter(%d,%d):weight_z:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 1 * 3,
                       weight_z.x, weight_z.y, weight_z.z, weight_z.w);
                printf("666 filter(%d,%d):weight_w:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 0 * 3,
                       weight_w.x, weight_w.y, weight_w.z, weight_w.w);
                printf("666 output:%.0f %.0f %.0f %.0f\n", output.x, output.y, output.z, output.w);
    }
#endif

               j = 7;
               pos_of_weight.x = i * 3 + j % 3;
               pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1 : pos_of_weight.y;
              weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
               output.x += dot(input[j], weight_x);

               pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
              weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
               output.y += dot(input[j], weight_y);

               pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
              weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
               output.z += dot(input[j], weight_z);

               pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
              weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
               output.w += dot(input[j], weight_w);

#ifdef DEBUG
    if (out_c == 0 && out_w == 0 && out_nh == 0) {
                printf("---- j:%d ----\n", j);
                printf("777 input[%d]:%.0f %.0f %.0f %.0f\n", j, input[j].x, input[j].y, input[j].z, input[j].w);
                printf("777 filter(%d,%d):weight_x:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 3 * 3,
				       weight_x.x, weight_x.y, weight_x.z, weight_x.w);
                printf("777 filter(%d,%d):weight_y:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 2 * 3,
                       weight_y.x, weight_y.y, weight_y.z, weight_y.w);
                printf("777 filter(%d,%d):weight_z:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 1 * 3,
                       weight_z.x, weight_z.y, weight_z.z, weight_z.w);
                printf("777 filter(%d,%d):weight_w:%.0f %.0f %.0f %.0f\n", pos_of_weight.x, pos_of_weight.y - 0 * 3,
                       weight_w.x, weight_w.y, weight_w.z, weight_w.w);
                printf("777 output:%.0f %.0f %.0f %.0f\n", output.x, output.y, output.z, output.w);
    }
#endif

               j = 8;
               pos_of_weight.x = i * 3 + j % 3;
               pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1 : pos_of_weight.y;
              weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
               output.x += dot(input[j], weight_x);

               pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
              weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
               output.y += dot(input[j], weight_y);

               pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
              weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
               output.z += dot(input[j], weight_z);

               pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
               //  pos_of_weight.y = pos_of_weight.y >= filter_channel ? filter_channel - 1: pos_of_weight.y;
              weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
               output.w += dot(input[j], weight_w);

#ifdef DEBUG
    if (out_c == 0 && out_w == 0 && out_nh == 0) {
                printf("---- j:%d ----\n", j);
                printf("888 input[%d]:%.0f %.0f %.0f %.0f\n", j, input[j].x, input[j].y, input[j].z, input[j].w);
                //printf("weight_x:%.0f %.0f %.0f %.0f\n", weight_x.x, weight_x.y, weight_x.z, weight_x.w);
                //printf("output:%.0f %.0f %.0f %.0f\n", output.x, output.y, output.z, output.w);
    }
               printf("===> out[%d %d]:%.2f %.2f %.2f %.2f\n", pos_of_weight.x, pos_of_weight.y, output.x, output.x, output.z, output.w);
#endif
        }
    } else { // group != 1
      for (int i = 0; i < 4; i++) {
        int used_input_channel_num =
          (out_c * 4 + i) / (output_c / group) * filter_channel;
        for (int f_c = 0; f_c < filter_channel; ++f_c) {
          int input_c = used_input_channel_num + f_c;
          int input_block = input_c / 4;
          int2 pos_in = (int2)(input_block * input_width + in_pos_in_one_block.x,
                               in_pos_in_one_block.y);
          input[0] = select(
              READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                          (int2)(pos_in.x - dilation, pos_in.y - dilation)),
              (CL_DTYPE4)(0.0f),
              (ushort4)((in_pos_in_one_block.x - dilation < 0 ||
                         in_pos_in_one_block.y - dilation < 0 ||
                         in_pos_in_one_block.x - dilation >= input_width ||
                         in_pos_in_one_block.y - dilation >= input_height)
                        << 15));
          input[1] =
              select(READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                                 (int2)(pos_in.x, pos_in.y - dilation)),
                     (CL_DTYPE4)(0.0f),
                     (ushort4)((in_pos_in_one_block.x < 0 ||
                                in_pos_in_one_block.y - dilation < 0 ||
                                in_pos_in_one_block.x >= input_width ||
                                in_pos_in_one_block.y - dilation >= input_height)
                               << 15));
          input[2] = select(
              READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                          (int2)(pos_in.x + dilation, pos_in.y - dilation)),
                          (CL_DTYPE4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x + dilation < 0 ||
                         in_pos_in_one_block.y - dilation < 0 ||
                         in_pos_in_one_block.x + dilation >= input_width ||
                         in_pos_in_one_block.y - dilation >= input_height)
                        << 15));
          input[3] = select(
              READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                          (int2)(pos_in.x - dilation, pos_in.y)),
                          (CL_DTYPE4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x - dilation < 0 ||
                                     in_pos_in_one_block.y < 0 ||
                                     in_pos_in_one_block.x - dilation >= input_width ||
                                     in_pos_in_one_block.y >= input_height)
                                    << 15));
          input[4] = select(
              READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler, (int2)(pos_in.x, pos_in.y)),
                          (CL_DTYPE4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y < 0 ||
                                     in_pos_in_one_block.x >= input_width ||
                                     in_pos_in_one_block.y >= input_height)
                                     << 15));
          input[5] =
            select(READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                               (int2)(pos_in.x + dilation, pos_in.y)),
                   (CL_DTYPE4)(0.0f),
                   (ushort4)((in_pos_in_one_block.x + dilation < 0 ||
                              in_pos_in_one_block.y < 0 ||
                              in_pos_in_one_block.x + dilation >= input_width ||
                              in_pos_in_one_block.y >= input_height)
                             << 15));
          input[6] = select(
              READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                          (int2)(pos_in.x - dilation, pos_in.y + dilation)),
                          (CL_DTYPE4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x - dilation < 0 ||
                                     in_pos_in_one_block.y + dilation < 0 ||
                                     in_pos_in_one_block.x - dilation >= input_width ||
                                     in_pos_in_one_block.y + dilation >= input_height)
                                     << 15));
          input[7] =
              select(READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                                 (int2)(pos_in.x, pos_in.y + dilation)),
                     (CL_DTYPE4)(0.0f),
                     (ushort4)((in_pos_in_one_block.x < 0 ||
                                in_pos_in_one_block.y + dilation < 0 ||
                                in_pos_in_one_block.x >= input_width ||
                                in_pos_in_one_block.y + dilation >= input_height)
                                 << 15));
          input[8] = select(
              READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,
                          (int2)(pos_in.x + dilation, pos_in.y + dilation)),
                          (CL_DTYPE4)(0.0f),
                          (ushort4)((in_pos_in_one_block.x + dilation < 0 ||
                                     in_pos_in_one_block.y + dilation < 0 ||
                                     in_pos_in_one_block.x + dilation >= input_width ||
                                     in_pos_in_one_block.y + dilation >= input_height)
                                      << 15));

          CL_DTYPE tmp_out = 0;
          for (int j = 0; j < 9; j++) {
            int2 pos_of_weight;
            pos_of_weight.x = (f_c / 4) * 3 + j % 3;
            pos_of_weight.y = out_c * 4 * 3 + i * 3 + j / 3;
            CL_DTYPE4 weight = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);

            int f_c_offset = f_c % 4;
            CL_DTYPE f_value;
            if (f_c_offset == 0) {
              f_value = weight.x;
            } else if (f_c_offset == 1) {
              f_value = weight.y;
            } else if (f_c_offset == 2) {
              f_value = weight.z;
            } else if (f_c_offset == 3) {
              f_value = weight.w;
            }

            int input_c_offset = input_c % 4;
            CL_DTYPE input_value;
            if (input_c_offset == 0) {
              input_value = input[j].x;
            } else if (input_c_offset == 1) {
              input_value = input[j].y;
            } else if (input_c_offset == 2) {
              input_value = input[j].z;
            } else if (input_c_offset == 3) {
              input_value = input[j].w;
            }
            tmp_out += f_value * input_value;
          }

          if (i == 0) {
            output.x += tmp_out;
          } else if (i == 1) {
            output.y += tmp_out;
          } else if (i == 2) {
            output.z += tmp_out;
          } else if (i == 3) {
            output.w += tmp_out;
          }
        }
      }
    }

#ifdef RELU
	output = activation_type4(output);
#endif

    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output);

//#ifdef DEBUG
	printf("output(%d, %d):%f %f %f %f\n", output_pos.x, output_pos.y, output.x, output.y, output.z, output.w);
//#endif
}

#undef DEBUG
