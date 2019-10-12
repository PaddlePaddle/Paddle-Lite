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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void feed(__global float *in,
                   __write_only image2d_t output_image,
                   __private const int out_H,
                   __private const int out_W,
                   __private const int out_C,
                   __private const int Stride0,
                   __private const int Stride1,
                   __private const int Stride2){

            const int out_c = get_global_id(0);
            const int out_w = get_global_id(1);
            const int out_nh = get_global_id(2);
            const int out_n = out_nh/out_H;
            const int out_h = out_nh%out_H;

            const int in_n = out_n;
            const int in_c0 = out_c * 4 + 0;
            const int in_c1 = out_c * 4 + 1;
            const int in_c2 = out_c * 4 + 2;
            const int in_c3 = out_c * 4 + 3;
            const int in_h = out_h;
            const int in_w = out_w;


            int input_pos0 = in_n * Stride2 + in_c0 * Stride1 + in_h * Stride0 + in_w;
            int input_pos1 = in_n * Stride2 + in_c1 * Stride1 + in_h * Stride0 + in_w;
            int input_pos2 = in_n * Stride2 + in_c2 * Stride1 + in_h * Stride0 + in_w;
            int input_pos3 = in_n * Stride2 + in_c3 * Stride1 + in_h * Stride0 + in_w;

            int2 output_pos;
            output_pos.x = out_c * out_W + out_w;
            output_pos.y = out_nh;

            half4 output = (half4)0.0f;
            output.x = convert_half(in[input_pos0]);
            if(out_C - 4 * out_c>=2){
             output.y = convert_half(in[input_pos1]);
            }
            if(out_C - 4 * out_c>=3){
            output.z = convert_half(in[input_pos2]);
            }
            if(out_C - 4 * out_c>=4){
             output.w = convert_half(in[input_pos3]);
            }
            write_imageh(output_image, output_pos, output);

 }

__kernel void feed_with_pre(__global uchar *in,
    __write_only image2d_t output_image,
    __private const int out_H,
    __private const int out_W,
    __private const int out_C,
    __private const int Stride0,
    __private const int Stride1,
    __private const int Stride2){

    const int out_c = get_global_id(0);
    const int out_w = get_global_id(1);
    const int out_nh = get_global_id(2);
    const int out_n = out_nh/out_H;
    const int out_h = out_nh%out_H;

    const int in_n = out_n;
    const int in_c0 = out_c * 4 + 0;
    const int in_c1 = out_c * 4 + 1;
    const int in_c2 = out_c * 4 + 2;
    const int in_c3 = out_c * 4 + 3;
    const int in_h = out_h;
    const int in_w = out_w;


    int input_pos0 = in_n * Stride2 + in_c0 * Stride1 + in_h * Stride0 + in_w;
    int input_pos1 = in_n * Stride2 + in_c1 * Stride1 + in_h * Stride0 + in_w;
    int input_pos2 = in_n * Stride2 + in_c2 * Stride1 + in_h * Stride0 + in_w;
    int input_pos3 = in_n * Stride2 + in_c3 * Stride1 + in_h * Stride0 + in_w;

    int2 output_pos;
    output_pos.x = out_c * out_W + out_w;
    output_pos.y = out_nh;

    half4 output = (half4)0.0f;
    output.x = convert_half(in[input_pos0]) / 255;
    if(out_C - 4 * out_c>=2){
        output.y = convert_half(in[input_pos1]) / 255;
    }
    if(out_C - 4 * out_c>=3){
        output.z = convert_half(in[input_pos2]) / 255;
    }
    if(out_C - 4 * out_c>=4){
        output.w = convert_half(in[input_pos3]) / 255;
    }
    write_imageh(output_image, output_pos, output);

}
