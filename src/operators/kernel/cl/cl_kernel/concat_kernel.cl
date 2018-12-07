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

__kernel void concatByC0(__read_only image2d_t input_image,
                      __write_only image2d_t output_image,
                      __private const int out_W) {

                      const int in_c = get_global_id(0);
                      const int in_w = get_global_id(1);
                      const int in_nh = get_global_id(2);

                      int2 input_pos ;
                      input_pos.x = in_c * out_W + in_w;
                      input_pos.y = in_nh;
                      const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                                                CLK_ADDRESS_CLAMP |
                                                CLK_FILTER_NEAREST;
                      half4 input;
                      input = read_imageh(input_image, sampler,input_pos);

                      write_imageh(output_image, input_pos, input);

}

__kernel void concatByC(__read_only image2d_t input_image1,
                      __read_only image2d_t input_image2,
                      __write_only image2d_t output_image,
                      __private const int out_C,
                      __private const int out_H,
                      __private const int out_W,
                      __private const int out_C_Start,
                      __private const int in_W,
                      __private const int in_H,
                      __private const int in_C1,
                      __private const int in_C2) {

                       const int in_c = get_global_id(0);
                       const int in_w = get_global_id(1);
                       const int in_nh = get_global_id(2);
                       int out_c1 = (out_C_Start + 3)/4 -1 + in_c;

                       int out_c2 = out_c1 + 1;

                       int2 output_pos1;
                       int2 output_pos2;

                       output_pos1.x = out_c1 * out_W + in_w;
                       output_pos1.y = in_nh;

                       output_pos2.x = out_c2 * out_W + in_w;
                       output_pos2.y = in_nh;

                       int2 input_pos1;
                       if(in_c==0){
                        input_pos1.x = ((in_C1 + 3)/4-1) * in_W + in_w;
                       }else{
                        input_pos1.x = (in_c - 1) * in_W + in_w;
                       }

                       input_pos1.y = in_nh;

                       int2 input_pos2;
                       input_pos2.x = in_c * in_W + in_w;
                       input_pos2.y = in_nh;

                       half4 output1;
                       half4 output2;
                       half4 input1;
                       half4 input2;
                       const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                                                   CLK_ADDRESS_CLAMP |
                                                   CLK_FILTER_NEAREST;
                       if(in_c==0){
                       input1 = read_imageh(input_image1, sampler,input_pos1);

                       }else {
                       input1 = read_imageh(input_image2, sampler,input_pos1);
                       }
                       input2 = read_imageh(input_image2, sampler,input_pos2);
                       output1 = input1;

                       if(out_C_Start%4==0){
                        output2 = input2;

                       }else if(out_C_Start%4==1){
                        output1.y = input2.x;
                        output1.z = input2.y;
                        output1.w = input2.z;
                        output2.x = input2.w;
                        output2.y = 0.0f;
                        output2.z = 0.0f;
                        output2.w = 0.0f;

                       }else if(out_C_Start%4==2){
                       output1.z = input2.x;
                       output1.w = input2.y;
                       output2.x = input2.z;
                       output2.y = input2.w;
                       output2.z = 0.0f;
                       output2.w = 0.0f;

                       }else if(out_C_Start%4==3){
                       output1.w = input2.x;
                       output2.x = input2.y;
                       output2.y = input2.z;
                       output2.z = input2.w;
                       output2.w = 0.0f;
                       }
                       write_imageh(output_image, output_pos1, output1);
                       write_imageh(output_image, output_pos2, output2);
}


__kernel void concatByH(__read_only image2d_t input_image,
                      __write_only image2d_t output_image,
                      __private const int out_W,
                      __private const int out_H_Start) {

                      const int in_c = get_global_id(0);
                      const int in_w = get_global_id(1);
                      const int in_nh = get_global_id(2);

                      int2 input_pos;
                      input_pos.x = in_c * out_W + in_w;
                      input_pos.y = in_nh;

                      const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                                                CLK_ADDRESS_CLAMP |
                                                CLK_FILTER_NEAREST;
                      half4 input;
                      input = read_imageh(input_image, sampler,input_pos);

                      int2 output_pos;
                      output_pos.x = input_pos.x;
                      output_pos.y = out_H_Start + input_pos.y;

                      write_imageh(output_image, output_pos, input);

}


