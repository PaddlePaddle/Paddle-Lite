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
                      const int out_c = get_global_id(0);
                      const int out_w = get_global_id(1);
                      const int out_nh = get_global_id(2);

                      const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                                                CLK_ADDRESS_CLAMP |
                                                CLK_FILTER_NEAREST;

                      int2 output_pos;
                      output_pos.x = out_c * out_W + out_w;
                      output_pos.y = out_nh;
                      half4 output_data;

                      for (int i = 0; i < 4; i++) {
                        int c = out_c * 4 + i;
                        if (c >= out_C) {
                            break;
                        }
                        int c_in;
                        half4 input_data;
                        if (c < C_0) {
                          c_in = c;
                          int2 input_pos;
                          input_pos.x = (c_in / 4) * out_W + out_w;
                          input_pos.y = out_nh;
                          input_data = read_imageh(input_image_0, sampler, input_pos);
                        } else {
                          c_in = c - C_0;
                          int2 input_pos;
                          input_pos.x = (c_in / 4) * out_W + out_w;
                          input_pos.y = out_nh;
                          input_data = read_imageh(input_image_1, sampler, input_pos);
                        }
                        int value_offset = c_in % 4;
                        float value;
                        if (value_offset == 0) {
                          value = input_data.x;
                        } else if (value_offset == 1) {
                          value = input_data.y;
                        } else if (value_offset == 2) {
                          value = input_data.z;
                        } else if (value_offset == 3) {
                          value = input_data.w;
                        }
                        if (i == 0) {
                          output_data.x = value;
                        } else if (i == 1) {
                          output_data.y = value;
                        } else if (i == 2) {
                          output_data.z = value;
                        } else if (i == 3) {
                          output_data.w = value;
                        }
                      }
                      write_imageh(output_image, output_pos, output_data);
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
                      const int out_c = get_global_id(0);
                      const int out_w = get_global_id(1);
                      const int out_nh = get_global_id(2);

                      const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                                                CLK_ADDRESS_CLAMP |
                                                CLK_FILTER_NEAREST;

                      int2 output_pos;
                      output_pos.x = out_c * out_W + out_w;
                      output_pos.y = out_nh;
                      half4 output_data;

                      for (int i = 0; i < 4; i++) {
                        int c = out_c * 4 + i;
                        if (c >= out_C) {
                            break;
                        }
                        int c_in;
                        half4 input_data;
                        if (c < C_0) {
                          c_in = c;
                          int2 input_pos;
                          input_pos.x = (c_in / 4) * out_W + out_w;
                          input_pos.y = out_nh;
                          input_data = read_imageh(input_image_0, sampler, input_pos);
                        } else if (c < C_0 + C_1) {
                          c_in = c - C_0;
                          int2 input_pos;
                          input_pos.x = (c_in / 4) * out_W + out_w;
                          input_pos.y = out_nh;
                          input_data = read_imageh(input_image_1, sampler, input_pos);
                        } else {
                          c_in = c - C_0 - C_1;
                          int2 input_pos;
                          input_pos.x = (c_in / 4) * out_W + out_w;
                          input_pos.y = out_nh;
                          input_data = read_imageh(input_image_2, sampler, input_pos);
                        }
                        int value_offset = c_in % 4;
                        float value;
                        if (value_offset == 0) {
                          value = input_data.x;
                        } else if (value_offset == 1) {
                          value = input_data.y;
                        } else if (value_offset == 2) {
                          value = input_data.z;
                        } else if (value_offset == 3) {
                          value = input_data.w;
                        }
                        if (i == 0) {
                          output_data.x = value;
                        } else if (i == 1) {
                          output_data.y = value;
                        } else if (i == 2) {
                          output_data.z = value;
                        } else if (i == 3) {
                          output_data.w = value;
                        }
                      }
                      write_imageh(output_image, output_pos, output_data);
}


__kernel void concatByCWith4Inputs(__read_only image2d_t input_image_0,
                    __read_only image2d_t input_image_1,
                    __read_only image2d_t input_image_2,
                    __read_only image2d_t input_image_3,
                    __private const int C_0,
                    __private const int C_1,
                    __private const int C_2,
                    __private const int C_3,
                    __write_only image2d_t output_image,
                    __private const int out_C,
                    __private const int out_W) {
                      const int out_c = get_global_id(0);
                      const int out_w = get_global_id(1);
                      const int out_nh = get_global_id(2);

                      const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                                                CLK_ADDRESS_CLAMP |
                                                CLK_FILTER_NEAREST;

                      int2 output_pos;
                      output_pos.x = out_c * out_W + out_w;
                      output_pos.y = out_nh;
                      half4 output_data;

                      for (int i = 0; i < 4; i++) {
                        int c = out_c * 4 + i;
                        if (c >= out_C) {
                            break;
                        }
                        int c_in;
                        half4 input_data;
                        if (c < C_0) {
                          c_in = c;
                          int2 input_pos;
                          input_pos.x = (c_in / 4) * out_W + out_w;
                          input_pos.y = out_nh;
                          input_data = read_imageh(input_image_0, sampler, input_pos);
                        } else if (c < C_0 + C_1) {
                          c_in = c - C_0;
                          int2 input_pos;
                          input_pos.x = (c_in / 4) * out_W + out_w;
                          input_pos.y = out_nh;
                          input_data = read_imageh(input_image_1, sampler, input_pos);
                        } else if (c < C_0 + C_1 + C_2) {
                          c_in = c - C_0 - C_1;
                          int2 input_pos;
                          input_pos.x = (c_in / 4) * out_W + out_w;
                          input_pos.y = out_nh;
                          input_data = read_imageh(input_image_2, sampler, input_pos);
                        }else if (c < C_0 + C_1 + C_2 + C_3){
                          c_in = c - C_0 - C_1 - C_2;
                          int2 input_pos;
                          input_pos.x = (c_in / 4) * out_W + out_w;
                          input_pos.y = out_nh;
                          input_data = read_imageh(input_image_3, sampler, input_pos);
                        }
                        int value_offset = c_in % 4;
                        float value;
                        if (value_offset == 0) {
                          value = input_data.x;
                        } else if (value_offset == 1) {
                          value = input_data.y;
                        } else if (value_offset == 2) {
                          value = input_data.z;
                        } else if (value_offset == 3) {
                          value = input_data.w;
                        }
                        if (i == 0) {
                          output_data.x = value;
                        } else if (i == 1) {
                          output_data.y = value;
                        } else if (i == 2) {
                          output_data.z = value;
                        } else if (i == 3) {
                          output_data.w = value;
                        }
                      }
                      write_imageh(output_image, output_pos, output_data);
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

__kernel void concatByW(__read_only image2d_t input_image,
                      __write_only image2d_t output_image,
                      __private const int in_W,
                      __private const int pre_Width,
                      __private const int out_Width) {

                      const int in_c = get_global_id(0);
                      const int in_w = get_global_id(1);
                      const int in_nh = get_global_id(2);

                      int2 input_pos;
                      input_pos.x = in_c * in_W + in_w;
                      input_pos.y = in_nh;

                      const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                                                CLK_ADDRESS_CLAMP |
                                                CLK_FILTER_NEAREST;
                      half4 input;
                      input = read_imageh(input_image, sampler,input_pos);

                      int2 output_pos;
                      output_pos.x = input_pos.x + pre_Width + out_Width * in_c;
                      output_pos.y = input_pos.y;
                      write_imageh(output_image, output_pos, input);

}




