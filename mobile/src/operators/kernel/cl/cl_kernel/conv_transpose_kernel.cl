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

__kernel void conv_transpose_b(__private const int input_c_block,
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

__kernel void depthwise_transpose(__private const int item_ch,
                               __private const int item_w,
                               __private const int item_h,
                               __read_only image2d_t input_image,
                               __read_only image2d_t filter_image,
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
                               __private const int in_w,
                               __private const int in_h,
                               __private const int out_w,
                               __private const int out_h,
                               __private const int filter_w,
                               __private const int filter_h) {

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                              CLK_ADDRESS_CLAMP          |
                              CLK_FILTER_NEAREST;

    // item_id
    const int item_ch_id = get_global_id(0);
    const int item_w_id = get_global_id(1);
    const int item_h_id = get_global_id(2);

    // out_id
    int out_b_id = item_h_id / out_h;
    int out_w_id_per_ch_blk = item_w_id;
    int out_h_id_per_batch = item_h_id % out_h;
    int out_w_id = item_ch_id * out_w + out_w_id_per_ch_blk;

    // in_id
    int in_w_id_per_ch_blk = (out_w_id_per_ch_blk + pad - filter_w + stride) / stride;
    in_w_id_per_ch_blk = in_w_id_per_ch_blk > 0 ? in_w_id_per_ch_blk : 0;
    int in_h_id_per_batch = (out_h_id_per_batch + pad - filter_h + stride) / stride;
    in_h_id_per_batch = in_h_id_per_batch > 0 ? in_h_id_per_batch : 0;

    // filter_id
    int align_w_i = out_w_id_per_ch_blk + pad - filter_w + 1;
    int align_w = align_w_i % stride > 0 ?
                  align_w_i % stride - stride : align_w_i % stride;
    int filter_w_id_per_ch_blk = out_w_id_per_ch_blk + pad < filter_w ? out_w_id_per_ch_blk + pad : filter_w + align_w - 1;

    int align_h_i = out_h_id_per_batch + pad - filter_h + 1;
    int align_h = align_h_i % stride > 0 ?
                  align_h_i % stride - stride : align_h_i % stride;
    int filter_h_id = out_h_id_per_batch + pad < filter_h ? out_h_id_per_batch + pad : filter_h + align_h - 1;

#ifdef BIASE_CH
    half4 output;
    output = read_imageh(bias, sampler, (int2)(item_ch_id, 0));
#elif defined(BIASE_ELE)
    half4 output;
    output = read_imageh(bias, sampler, (int2)(out_w_id, item_h_id));
#else
    half4 output = 0.0f;
#endif
    half4 filter = 0.0f;
    half4 input = 0.0f;
    for (int h = filter_h_id; h >= 0; h -= stride) {
        int in_h_id = select(out_b_id * in_h + in_h_id_per_batch, -1,
                             in_h_id_per_batch < 0 || in_h_id_per_batch >= in_h);
        for (int w = filter_w_id_per_ch_blk; w >= 0; w -= stride) {
            int in_w_id = select(item_ch_id * in_w + in_w_id_per_ch_blk, -1,
                                 in_w_id_per_ch_blk < 0 || in_w_id_per_ch_blk >= in_w);
            int filter_w_id = item_ch_id * filter_w + w;
            input = read_imageh(input_image, sampler, (int2)(in_w_id, in_h_id));
            filter = read_imageh(filter_image, sampler, (int2)(filter_w_id, h));

            output = mad(input, filter, output);
            in_w_id_per_ch_blk++;
        }
        in_h_id_per_batch++;
    }

#ifdef BATCH_NORM
    half4 scale = read_imageh(new_scale, sampler, (int2)(item_ch_id, 0));
    half4 biase = read_imageh(new_biase, sampler, (int2)(item_ch_id, 0));
    output = mad(scale, output, biase);
#endif

#ifdef RELU
    output = activation(output);
#endif

    write_imageh(output_image, (int2)(out_w_id, item_h_id), output);
}


/* batch == 1 pad(output) == 1 out_w % 2 == 0 */
__kernel void conv_transpose3x3s2(__private const int item_ch,
                                  __private const int item_w,
                                  __private const int item_h,
                                  __read_only image2d_t input_image,
                                  __read_only image2d_t filter_image,
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
                                  __private const int in_w,
                                  __private const int in_h,
                                  __private const int out_w,
                                  __private const int out_h,
                                  __private const int filter_w,
                                  __private const int filter_h) {

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                              CLK_ADDRESS_CLAMP          |
                              CLK_FILTER_NEAREST;
    // item_id
    const int item_ch_id = get_global_id(0);
    const int item_w_id = get_global_id(1);
    const int item_h_id = get_global_id(2);

    // out_id
    int out_w_id_per_ch_blk = item_w_id / 2 * 10 + item_w_id % 2;
    int out_h_id = item_h_id;
    int out_w_id0 = item_ch_id * out_w + out_w_id_per_ch_blk;
    int out_w_id1 = out_w_id0 + 2;
    int out_w_id2 = out_w_id1 + 2;
    int out_w_id3 = out_w_id2 + 2;
    int out_w_id4 = out_w_id3 + 2;

    // in_id
    int in_w_id_per_ch_blk = (out_w_id_per_ch_blk) / 2;
    in_w_id_per_ch_blk = in_w_id_per_ch_blk > 0 ? in_w_id_per_ch_blk : 0;
    int in_h_id_per_batch = (out_h_id) / 2;
    in_h_id_per_batch = in_h_id_per_batch > 0 ? in_h_id_per_batch : 0;

    // filter_id
    int align_w_i = out_w_id_per_ch_blk - 1;
    int align_w = align_w_i % 2 > 0 ?
                  align_w_i % 2 - 2 : align_w_i % 2;
    int filter_w_id_per_ch_blk = out_w_id_per_ch_blk + 1 < 3 ? out_w_id_per_ch_blk + 1 : 2 + align_w;

    int align_h_i = out_h_id - 1;
    int align_h = align_h_i % 2 > 0 ?
                  align_h_i % 2 - 2 : align_h_i % 2;
    int filter_h_id_per_out_ch = out_h_id + 1 < 3 ? out_h_id + 1 : 2 + align_h;

#ifdef BIASE_CH
    half4 output[5];
    output[0] = read_imageh(bias, sampler, (int2)(item_ch_id, 0));
    output[1] = output[0];
    output[2] = output[0];
    output[3] = output[0];
    output[4] = output[0];

#elif defined(BIASE_ELE)
    half4 output[5];
    output[0] = read_imageh(bias, sampler, (int2)(out_w_id0, item_h_id));
    if (out_w_id_per_ch_blk + 2 < out_w) {
        output[1] = read_imageh(bias, sampler, (int2)(out_w_id1, item_h_id));
    }
    if (out_w_id_per_ch_blk + 4 < out_w) {
        output[2] = read_imageh(bias, sampler, (int2)(out_w_id2, item_h_id));
    }
    if (out_w_id_per_ch_blk + 6 < out_w) {
        output[3] = read_imageh(bias, sampler, (int2)(out_w_id3, item_h_id));
    }
    if (out_w_id_per_ch_blk + 8 < out_w) {
        output[4] = read_imageh(bias, sampler, (int2)(out_w_id4, item_h_id));
    }

#else
    half4 output[5] = {0.0f};
#endif
    half4 filter[4] = {0.0f};
    half4 filter_trans[4] = {0.0f};

    half4 input[5] = {0.0f};
    for (int ch = 0; ch < (in_ch + 3) / 4; ch++) {
        int filter_w_id = ch * 3;
        int h_idx = 0;
        for (int h = filter_h_id_per_out_ch; h >= 0; h -= 2) {
            int in_h_id = select(in_h_id_per_batch + h_idx, -1,
                                 in_h_id_per_batch + h_idx < 0 || in_h_id_per_batch + h_idx >= in_h);
            int filter_h_id = item_ch_id * 12 + h;
            int w_idx = 0;
            for (int w = filter_w_id_per_ch_blk; w >= 0; w -= 2) {
                int in_w_id0 = select(ch * in_w + in_w_id_per_ch_blk + w_idx, -1,
                                     in_w_id_per_ch_blk + w_idx < 0 || in_w_id_per_ch_blk + w_idx >= in_w);
                int in_w_id1 = select(ch * in_w + in_w_id_per_ch_blk + 1 + w_idx, -1,
                                     in_w_id_per_ch_blk + 1 + w_idx < 0 || in_w_id_per_ch_blk + 1 + w_idx >= in_w);
                int in_w_id2 = select(ch * in_w + in_w_id_per_ch_blk + 2 + w_idx, -1,
                                     in_w_id_per_ch_blk + 2 + w_idx < 0 || in_w_id_per_ch_blk + 2 + w_idx >= in_w);
                int in_w_id3 = select(ch * in_w + in_w_id_per_ch_blk + 3 + w_idx, -1,
                                     in_w_id_per_ch_blk + 3 + w_idx < 0 || in_w_id_per_ch_blk + 3 + w_idx >= in_w);
                int in_w_id4 = select(ch * in_w + in_w_id_per_ch_blk + 4 + w_idx, -1,
                                     in_w_id_per_ch_blk + 4 + w_idx < 0 || in_w_id_per_ch_blk + 4 + w_idx >= in_w);

                input[0] = read_imageh(input_image, sampler, (int2)(in_w_id0, in_h_id));
                input[1] = read_imageh(input_image, sampler, (int2)(in_w_id1, in_h_id));
                input[2] = read_imageh(input_image, sampler, (int2)(in_w_id2, in_h_id));
                input[3] = read_imageh(input_image, sampler, (int2)(in_w_id3, in_h_id));
                input[4] = read_imageh(input_image, sampler, (int2)(in_w_id4, in_h_id));

                filter[0] = read_imageh(filter_image, sampler, (int2)(filter_w_id + w, filter_h_id));      // in_ch:0-3,out_ch:0
                filter[1] = read_imageh(filter_image, sampler, (int2)(filter_w_id + w, filter_h_id + 3));  // in_ch:0-3,out_ch:1
                filter[2] = read_imageh(filter_image, sampler, (int2)(filter_w_id + w, filter_h_id + 6));  // in_ch:0-3,out_ch:2
                filter[3] = read_imageh(filter_image, sampler, (int2)(filter_w_id + w, filter_h_id + 9));  // in_ch:0-3,out_ch:3

                filter_trans[0] = (half4)(filter[0].x, filter[1].x, filter[2].x, filter[3].x);             // in_ch:0,out_ch:0-3
                filter_trans[1] = (half4)(filter[0].y, filter[1].y, filter[2].y, filter[3].y);             // in_ch:1,out_ch:0-3
                filter_trans[2] = (half4)(filter[0].z, filter[1].z, filter[2].z, filter[3].z);             // in_ch:2,out_ch:0-3
                filter_trans[3] = (half4)(filter[0].w, filter[1].w, filter[2].w, filter[3].w);             // in_ch:3,out_ch:0-3

                output[0] = mad(input[0].x, filter_trans[0], output[0]);
                output[0] = mad(input[0].y, filter_trans[1], output[0]);
                output[0] = mad(input[0].z, filter_trans[2], output[0]);
                output[0] = mad(input[0].w, filter_trans[3], output[0]);

                output[1] = mad(input[1].x, filter_trans[0], output[1]);
                output[1] = mad(input[1].y, filter_trans[1], output[1]);
                output[1] = mad(input[1].z, filter_trans[2], output[1]);
                output[1] = mad(input[1].w, filter_trans[3], output[1]);

                output[2] = mad(input[2].x, filter_trans[0], output[2]);
                output[2] = mad(input[2].y, filter_trans[1], output[2]);
                output[2] = mad(input[2].z, filter_trans[2], output[2]);
                output[2] = mad(input[2].w, filter_trans[3], output[2]);

                output[3] = mad(input[3].x, filter_trans[0], output[3]);
                output[3] = mad(input[3].y, filter_trans[1], output[3]);
                output[3] = mad(input[3].z, filter_trans[2], output[3]);
                output[3] = mad(input[3].w, filter_trans[3], output[3]);

                output[4] = mad(input[4].x, filter_trans[0], output[4]);
                output[4] = mad(input[4].y, filter_trans[1], output[4]);
                output[4] = mad(input[4].z, filter_trans[2], output[4]);
                output[4] = mad(input[4].w, filter_trans[3], output[4]);
                w_idx++;
            }
            h_idx++;
        }
    }
#ifdef BATCH_NORM
    half4 scale = read_imageh(new_scale, sampler, (int2)(item_ch_id, 0));
    half4 biase = read_imageh(new_biase, sampler, (int2)(item_ch_id, 0));
    output[0] = mad(scale, output[0], biase);
    if (out_w_id_per_ch_blk + 2 < out_w) {
        output[1] = mad(scale, output[1], biase);
    }
    if (out_w_id_per_ch_blk + 4 < out_w) {
        output[2] = mad(scale, output[2], biase);
    }
    if (out_w_id_per_ch_blk + 6 < out_w) {
        output[3] = mad(scale, output[3], biase);
    }
    if (out_w_id_per_ch_blk + 8 < out_w) {
        output[4] = mad(scale, output[4], biase);
    }
#endif

#ifdef RELU
    output[0] = activation(output[0]);
    output[1] = activation(output[1]);
    output[2] = activation(output[2]);
    output[3] = activation(output[3]);
    output[4] = activation(output[4]);

#endif

    write_imageh(output_image, (int2)(out_w_id0, item_h_id), output[0]);

    if (out_w_id_per_ch_blk + 2 < out_w) {
        write_imageh(output_image, (int2)(out_w_id1, item_h_id), output[1]);
    }
    if (out_w_id_per_ch_blk + 4 < out_w) {
        write_imageh(output_image, (int2)(out_w_id2, item_h_id), output[2]);
    }
    if (out_w_id_per_ch_blk + 6 < out_w) {
        write_imageh(output_image, (int2)(out_w_id3, item_h_id), output[3]);
    }
    if (out_w_id_per_ch_blk + 8 < out_w) {
        write_imageh(output_image, (int2)(out_w_id4, item_h_id), output[4]);
    }
}

__kernel void conv_transpose(__private const int item_ch,
                                  __private const int item_w,
                                  __private const int item_h,
                                  __read_only image2d_t input_image,
                                  __read_only image2d_t filter_image,
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
                                  __private const int in_w,
                                  __private const int in_h,
                                  __private const int out_w,
                                  __private const int out_h,
                                  __private const int filter_w,
                                  __private const int filter_h) {

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                              CLK_ADDRESS_CLAMP          |
                              CLK_FILTER_NEAREST;
    // item_id
    const int item_ch_id = get_global_id(0);
    const int item_w_id = get_global_id(1);
    const int item_h_id = get_global_id(2);

    // out_id
    int out_b_id = item_h_id / out_h;
    int out_w_id_per_ch_blk = item_w_id;
    int out_h_id_per_batch = item_h_id % out_h;
    int out_w_id = item_ch_id * out_w + out_w_id_per_ch_blk;

    // in_id
    int in_w_id_per_ch_blk = (out_w_id_per_ch_blk + pad - filter_w + stride) / stride;
    in_w_id_per_ch_blk = in_w_id_per_ch_blk > 0 ? in_w_id_per_ch_blk : 0;
    int in_h_id_per_batch = (out_h_id_per_batch + pad - filter_h + stride) / stride;
    in_h_id_per_batch = in_h_id_per_batch > 0 ? in_h_id_per_batch : 0;

    // filter_id
    int align_w_i = out_w_id_per_ch_blk + pad - filter_w + 1;
    int align_w = align_w_i % stride > 0 ?
                  align_w_i % stride - stride : align_w_i % stride;
    int filter_w_id_per_ch_blk = out_w_id_per_ch_blk + pad < filter_w ? out_w_id_per_ch_blk + pad : filter_w + align_w - 1;

    int align_h_i = out_h_id_per_batch + pad - filter_h + 1;
    int align_h = align_h_i % stride > 0 ?
                  align_h_i % stride - stride : align_h_i % stride;
    int filter_h_id_per_out_ch = out_h_id_per_batch + pad < filter_h ? out_h_id_per_batch + pad : filter_h + align_h - 1;

#ifdef BIASE_CH
    half4 output;
    output = read_imageh(bias, sampler, (int2)(item_ch_id, 0));
#elif defined(BIASE_ELE)
    half4 output;
    output = read_imageh(bias, sampler, (int2)(out_w_id, item_h_id));
#else
    half4 output = 0.0f;
#endif
    half4 filter[4] = {0.0f};
    half4 filter_trans[4] = {0.0f};

    half4 input = 0.0f;
    for (int ch = 0; ch < (in_ch + 3) / 4; ch++) {
        int filter_w_id = ch * filter_w;
        int h_idx = 0;
        for (int h = filter_h_id_per_out_ch; h >= 0; h -= stride) {
            int in_h_id = select(in_h_id_per_batch + h_idx, -1,
                                 in_h_id_per_batch + h_idx < 0 || in_h_id_per_batch + h_idx >= in_h);
            int filter_h_id = item_ch_id * filter_h * 4 + h;
            int w_idx = 0;
            for (int w = filter_w_id_per_ch_blk; w >= 0; w -= stride) {
                int in_w_id = select(ch * in_w + in_w_id_per_ch_blk + w_idx, -1,
                                     in_w_id_per_ch_blk + w_idx < 0 || in_w_id_per_ch_blk + w_idx >= in_w);
                input = read_imageh(input_image, sampler, (int2)(in_w_id, in_h_id));
                filter[0] = read_imageh(filter_image, sampler, (int2)(filter_w_id + w, filter_h_id));                 // in_ch:0-3,out_ch:0
                filter[1] = read_imageh(filter_image, sampler, (int2)(filter_w_id + w, filter_h_id + filter_h));      // in_ch:0-3,out_ch:1
                filter[2] = read_imageh(filter_image, sampler, (int2)(filter_w_id + w, filter_h_id + 2 * filter_h));  // in_ch:0-3,out_ch:2
                filter[3] = read_imageh(filter_image, sampler, (int2)(filter_w_id + w, filter_h_id + 3 * filter_h));  // in_ch:0-3,out_ch:3

                filter_trans[0] = (half4)(filter[0].x, filter[1].x, filter[2].x, filter[3].x);             // in_ch:0,out_ch:0-3
                filter_trans[1] = (half4)(filter[0].y, filter[1].y, filter[2].y, filter[3].y);             // in_ch:1,out_ch:0-3
                filter_trans[2] = (half4)(filter[0].z, filter[1].z, filter[2].z, filter[3].z);             // in_ch:2,out_ch:0-3
                filter_trans[3] = (half4)(filter[0].w, filter[1].w, filter[2].w, filter[3].w);             // in_ch:3,out_ch:0-3

                output = mad(input.x, filter_trans[0], output);
                output = mad(input.y, filter_trans[1], output);
                output = mad(input.z, filter_trans[2], output);
                output = mad(input.w, filter_trans[3], output);
                w_idx++;
            }
            h_idx++;
        }
    }
#ifdef BATCH_NORM
    half4 scale = read_imageh(new_scale, sampler, (int2)(item_ch_id, 0));
    half4 biase = read_imageh(new_biase, sampler, (int2)(item_ch_id, 0));
    output = mad(scale, output, biase);
#endif

#ifdef RELU
    output = activation(output);
#endif
    write_imageh(output_image, (int2)(out_w_id, item_h_id), output);
}

