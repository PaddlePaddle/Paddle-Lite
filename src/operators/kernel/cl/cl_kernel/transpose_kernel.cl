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
__kernel void transpose_4d( __read_only image2d_t input_image,
                            __write_only image2d_t output_image,
                            __private const int out_C,
                            __private const int out_H,
                            __private const int out_W,
                            __private const int in_W
                           ){
                           const int out_c = get_global_id(0);
                           const int out_w = get_global_id(1);
                           const int out_nh = get_global_id(2);
                           const int out_n =  1;
                           const int out_h = out_nh%out_H;
                           const int out_c0 = out_c * 4;
                           const int out_c1 = out_c * 4 + 1;
                           const int out_c2 = out_c * 4+ 2;
                           const int out_c3 = out_c * 4+ 3;

                           const int in_n = out_n;
                           const int in_c = out_w / 4;
                           const int in_h0 = out_c0;
                           const int in_h1 = out_c1;
                           const int in_h2 = out_c2;
                           const int in_h3 = out_c3;
                           const int in_w = out_h;

                           int2 output_pos;
                           output_pos.x = out_c * out_W + out_w;
                           output_pos.y = out_nh;

                           int2 input_pos0;
                           int2 input_pos1;
                           int2 input_pos2;
                           int2 input_pos3;

                           input_pos0.x = in_W * in_c + in_w;
                           input_pos0.y = in_n * in_h0;

                           input_pos1.x = in_W * in_c + in_w;
                           input_pos1.y = in_n * in_h1;

                           input_pos2.x = in_W * in_c + in_w;
                           input_pos2.y = in_n * in_h2;

                           input_pos3.x = in_W * in_c + in_w;
                           input_pos3.y = in_n * in_h3;

                           const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                                                         CLK_ADDRESS_CLAMP      |
                                                         CLK_FILTER_NEAREST;

                           half4 input0;
                           half4 input1;
                           half4 input2;
                           half4 input3;
                           half4 output;
                           input0 = read_imageh(input_image, sampler,input_pos0);

                            if(out_w%4==0){
                            output.x = input0.x;
                           }else if(out_w%4==1){
                            output.x = input0.y;
                           }else if(out_w%4==2){
                            output.x = input0.z;
                           }else{
                            output.x = input0.w;
                           }
                           if(out_C - out_c * 4>=2){
                            input1 = read_imageh(input_image, sampler,input_pos1);
                            if(out_w%4==0){
                            output.y = input1.x;
                            }else if(out_w%4==1){
                             output.y = input1.y;
                            }else if(out_w%4==2){
                             output.y = input1.z;
                            }else{
                             output.y = input1.w;
                            }

                           }else{
                           output.y = 0.0f;
                           }

                           if(out_C - out_c * 4>=3){
                            input2 = read_imageh(input_image, sampler,input_pos2);

                          if(out_w%4==0){
                            output.z = input2.x;
                            }else if(out_w%4==1){
                             output.z = input2.y;
                            }else if(out_w%4==2){
                             output.z = input2.z;
                            }else{
                             output.z = input2.w;
                            }
                           }else{
                            output.z = 0.0f;
                           }

                           if(out_C - out_c * 4>=4){
                            input3 = read_imageh(input_image, sampler,input_pos3);
                           if(out_w%4==0){
                            output.w = input3.x;
                            }else if(out_w%4==1){
                             output.w = input3.y;
                            }else if(out_w%4==2){
                             output.w = input3.z;
                            }else{
                             output.w = input3.w;
                            }
                           }else{
                           output.w = 0.0f;
                           }
                           write_imageh(output_image, output_pos, output);
}

__kernel void transpose( __read_only image2d_t input_image,
                            __write_only image2d_t output_image,
                            __private const int out_C,
                            __private const int out_H,
                            __private const int out_W,
                            __private const int in_W
                           ){
                           const int out_c = get_global_id(0);
                           const int out_w = get_global_id(1);
                           const int out_nh = get_global_id(2);
                           const int out_n =  1;
                           const int out_h = out_nh%out_H;

                           const int in_n = 1;
                           const int in_c = out_c;
                           const int in_w = out_h;
                           const int in_h = out_w;

                           int2 input_pos;
                           int2 output_pos;

                           input_pos.x = in_c * in_W + in_w;
                           input_pos.y = in_n * in_h;

                           output_pos.x = out_c * out_W + out_w;
                           output_pos.y = out_n * out_h;

                           const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                                                      CLK_ADDRESS_CLAMP      |
                                                      CLK_FILTER_NEAREST;

                           half4 input;
                           half4 output;
                           input = read_imageh(input_image, sampler,input_pos);

                           output = input;
                           write_imageh(output_image, output_pos, output);

}