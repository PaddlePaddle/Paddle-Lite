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


__kernel void concatByCWith2Inputs(__read_only image2d_t input_image_0,
                    __read_only image2d_t input_image_1,
                    __private const int C_0,
                    __private const int C_1,
                    __write_only image2d_t output_image,
                    __private const int out_C,
                    __private const int out_W) {
//                      const int in_c = get_global_id(0);
//                      const int in_w = get_global_id(1);
//                      const int in_nh = get_global_id(2);
//
//                      int2 input_pos ;
//                      input_pos.x = in_c * out_W + in_w;
//                      input_pos.y = in_nh;
//                      const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
//                                                CLK_ADDRESS_CLAMP |
//                                                CLK_FILTER_NEAREST;
//                      half4 input;
//                      input = read_imageh(input_image, sampler,input_pos);
//
//                      write_imageh(output_image, input_pos, input);
}

__kernel void concatByCWith3Inputs(__read_only image2d_t input_image_0,
                    __read_only image2d_t input_image_1,
                    __read_only image2d_t input_image_2,
                    __private const int C_0,
                    __private const int C_1,
                    __private const int C_2,
                    __write_only image2d_t output_image,
                    __private const int out_C,
                    __private const int out_W) {
//                      const int in_c = get_global_id(0);
//                      const int in_w = get_global_id(1);
//                      const int in_nh = get_global_id(2);
//
//                      int2 input_pos ;
//                      input_pos.x = in_c * out_W + in_w;
//                      input_pos.y = in_nh;
//                      const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
//                                                CLK_ADDRESS_CLAMP |
//                                                CLK_FILTER_NEAREST;
//                      half4 input;
//                      input = read_imageh(input_image, sampler,input_pos);
//
//                      write_imageh(output_image, input_pos, input);
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


