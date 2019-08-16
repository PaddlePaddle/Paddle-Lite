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

__kernel void lrn(__read_only image2d_t input_image,
                        __write_only image2d_t output_image,
                        __private const int out_C,
                        __private const int out_W,
                        __private const int n,
                        __private const float k,
                        __private const float alpha,
                        __private const float beta){

                        const int out_c = get_global_id(0);
                        const int out_w = get_global_id(1);
                        const int out_nh = get_global_id(2);

                        const int out_c0 = out_c * 4;
                        const int out_c1 = out_c * 4 + 1;
                        const int out_c2 = out_c * 4+ 2;
                        const int out_c3 = out_c * 4+ 3;
                        const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                                                    CLK_ADDRESS_CLAMP |
                                                    CLK_FILTER_NEAREST;

                        const int start = -(n-1)/2;
                        const end = start + n;
                        float sqr_sum0 = 0.0f;
                        float sqr_sum1 = 0.0f;
                        float sqr_sum2 = 0.0f;
                        float sqr_sum3 = 0.0f;
                        int input_c0,input_c1,input_c2,input_c3;
                        int2 input_pos0,input_pos1,input_pos2,input_pos3;
                        float4 input0,input1,input2,input3;
                        for(int i = start; i < end ;i++){
                         if(out_c0 + i>=0&&out_c0 + i<out_C){
                          input_c0 = (out_c0 + i)/4;
                          input_pos0.x = input_c0 * out_W + out_w;
                          input_pos0.y = out_nh;
                          input0 = convert_float4(read_imageh(input_image, sampler,input_pos0));
                          if((out_c0 + i)%4 == 0){
                           sqr_sum0 += input0.x * input0.x;
                          }else if((out_c0 + i)%4 == 1){
                           sqr_sum0 += input0.y * input0.y;
                          }else if((out_c0 + i)%4 == 2){
                           sqr_sum0 += input0.z * input0.z;
                          }else{
                           sqr_sum0 += input0.w * input0.w;
                          }
                         }

                       if(out_c1 + i>=0&&out_c1 + i<out_C){
                          input_c1 = (out_c1 + i)/4;
                          input_pos1.x = input_c1 * out_W + out_w;
                          input_pos1.y = out_nh;
                          input1 = convert_float4(read_imageh(input_image, sampler,input_pos1));
                          if((out_c1 + i)%4 == 0){
                           sqr_sum1 += input1.x * input1.x;
                          }else if((out_c1 + i)%4 == 1){
                           sqr_sum1 += input1.y * input1.y;
                          }else if((out_c1 + i)%4 == 2){
                           sqr_sum1 += input1.z * input1.z;
                          }else{
                           sqr_sum1 += input1.w * input1.w;
                          }
                         }


                         if(out_c2 + i>=0&&out_c2 + i<out_C){
                          input_c2 = (out_c2 + i)/4;
                          input_pos2.x = input_c2 * out_W + out_w;
                          input_pos2.y = out_nh;
                          input2 = convert_float4(read_imageh(input_image, sampler,input_pos2));
                          if((out_c2 + i)%4 == 0){
                           sqr_sum2 += input2.x * input2.x;
                          }else if((out_c2 + i)%4 == 1){
                           sqr_sum2 += input2.y * input2.y;
                          }else if((out_c2 + i)%4 == 2){
                           sqr_sum2 += input2.z * input2.z;
                          }else{
                           sqr_sum2 += input2.w * input2.w;
                          }
                         }

                         if(out_c3 + i>=0&&out_c3 + i<out_C){
                          input_c3 = (out_c3 + i)/4;
                          input_pos3.x = input_c3 * out_W + out_w;
                          input_pos3.y = out_nh;
                          input3 = convert_float4(read_imageh(input_image, sampler,input_pos3));
                          if((out_c3 + i)%4 == 0){
                           sqr_sum3 += input3.x * input3.x;
                          }else if((out_c3 + i)%4 == 1){
                           sqr_sum3 += input3.y * input3.y;
                          }else if((out_c3 + i)%4 == 2){
                           sqr_sum3 += input3.z * input3.z;
                          }else{
                           sqr_sum3 += input3.w * input3.w;
                          }
                         }

                        }

                        float4 output = (float4)0.0f;
                        float4 input;
                        int2 output_pos;
                        output_pos.x = out_c * out_W + out_w;
                        output_pos.y = out_nh;
                        input = convert_float4(read_imageh(input_image, sampler,output_pos));

                        output.x = input.x / (pow(k + alpha * (sqr_sum0),beta));

                        if(out_C - 4 * out_c>=2){
                        output.y = input.y / (pow(k + alpha * (sqr_sum1),beta));
                        }
                        if(out_C - 4 * out_c>=3){
                        output.z = input.z / (pow(k + alpha * (sqr_sum2),beta));
                        }
                        if(out_C - 4 * out_c>=4){
                        output.w = input.w / (pow(k + alpha * (sqr_sum3),beta));
                        }
                        half4 tmp = convert_half4(output);
                        write_imageh(output_image, output_pos, tmp);

}