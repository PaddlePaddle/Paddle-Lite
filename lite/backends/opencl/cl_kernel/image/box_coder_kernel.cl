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

__kernel void decode_center_size(__read_only image2d_t prior_box_image,
                                __read_only image2d_t prior_box_var_image,
                                __read_only image2d_t target_box_image,
                                __write_only image2d_t output_image,
                                __private const int out_C,
                                __private const int out_H){
                        const int out_c = get_global_id(0);
                        const int out_nh = get_global_id(1);
                        const int out_h = out_nh % out_H;
                        const int out_n =  1;

                        const int prior_box_n = 1;
                        const int prior_box_c = 0;
                        const int prior_box_h = out_h;

                        const int prior_box_var_n = 1;
                        const int prior_box_var_c = 0;
                        const int prior_box_var_h = out_h;

                        const int target_box_n = 1;
                        const int target_box_c = out_c;
                        const int target_box_h = out_h;

                        int2  prior_box_pos;
                        int2  prior_box_var_pos;
                        int2  target_box_pos;
                        int2  output_pos;

                        prior_box_pos.x = prior_box_c * 4;
                        prior_box_pos.y = prior_box_n * prior_box_h;

                        prior_box_var_pos.x = prior_box_var_c * 4;
                        prior_box_var_pos.y = prior_box_var_n * prior_box_var_h;

                        target_box_pos.x = target_box_c * 4;
                        target_box_pos.y = target_box_n * target_box_h;

                        output_pos.x = out_c * 4;
                        output_pos.y = out_n * out_h;

                        CL_DTYPE4 prior_box_input[4];
                        CL_DTYPE4 prior_box_var_input[4];
                        CL_DTYPE4 target_box_input[4];

                        prior_box_input[0] = READ_IMG_TYPE(CL_DTYPE_CHAR, prior_box_image, SAMPLER,
                                            (int2)(prior_box_pos.x + 0, prior_box_pos.y));
                        prior_box_input[1] = READ_IMG_TYPE(CL_DTYPE_CHAR, prior_box_image, SAMPLER,
                                            (int2)(prior_box_pos.x + 1, prior_box_pos.y));
                        prior_box_input[2] = READ_IMG_TYPE(CL_DTYPE_CHAR, prior_box_image, SAMPLER,
                                            (int2)(prior_box_pos.x + 2, prior_box_pos.y));
                        prior_box_input[3] = READ_IMG_TYPE(CL_DTYPE_CHAR, prior_box_image, SAMPLER,
                                            (int2)(prior_box_pos.x + 3, prior_box_pos.y));

                        prior_box_var_input[0] = READ_IMG_TYPE(CL_DTYPE_CHAR, prior_box_var_image, SAMPLER,
                                                (int2)(prior_box_var_pos.x + 0, prior_box_var_pos.y));
                        prior_box_var_input[1] = READ_IMG_TYPE(CL_DTYPE_CHAR, prior_box_var_image, SAMPLER,
                                                (int2)(prior_box_var_pos.x + 1, prior_box_var_pos.y));
                        prior_box_var_input[2] = READ_IMG_TYPE(CL_DTYPE_CHAR, prior_box_var_image, SAMPLER,
                                                (int2)(prior_box_var_pos.x + 2, prior_box_var_pos.y));
                        prior_box_var_input[3] = READ_IMG_TYPE(CL_DTYPE_CHAR, prior_box_var_image, SAMPLER, 
                                                (int2)(prior_box_var_pos.x + 3, prior_box_var_pos.y));

                        target_box_input[0] = READ_IMG_TYPE(CL_DTYPE_CHAR, target_box_image, SAMPLER,
                                            (int2)(target_box_pos.x + 0,target_box_pos.y));
                        target_box_input[1] = READ_IMG_TYPE(CL_DTYPE_CHAR, target_box_image, SAMPLER,
                                            (int2)(target_box_pos.x + 1, target_box_pos.y));
                        target_box_input[2] = READ_IMG_TYPE(CL_DTYPE_CHAR, target_box_image, SAMPLER,
                                            (int2)(target_box_pos.x + 2, target_box_pos.y));
                        target_box_input[3] = READ_IMG_TYPE(CL_DTYPE_CHAR, target_box_image, SAMPLER,
                                            (int2)(target_box_pos.x + 3, target_box_pos.y));

                        CL_DTYPE prior_box_width = prior_box_input[2].x - prior_box_input[0].x;
                        CL_DTYPE prior_box_height = prior_box_input[3].x - prior_box_input[1].x;
                        CL_DTYPE prior_box_center_x = (prior_box_input[2].x + prior_box_input[0].x)/(CL_DTYPE)2;
                        CL_DTYPE prior_box_center_y = (prior_box_input[3].x + prior_box_input[1].x)/(CL_DTYPE)2;

                        CL_DTYPE4 target_box_center_x;
                        CL_DTYPE4 target_box_center_y;
                        CL_DTYPE4 target_box_width;
                        CL_DTYPE4 target_box_height;
                        CL_DTYPE4 output[4];

                        output[0] = 0.0f;
                        output[1] = 0.0f;
                        output[2] = 0.0f;
                        output[3] = 0.0f;

                        target_box_center_x.x = prior_box_var_input[0].x * target_box_input[0].x * prior_box_width + prior_box_center_x;
                        target_box_center_y.x = prior_box_var_input[1].x * target_box_input[1].x * prior_box_height + prior_box_center_y;
                        target_box_width.x = exp(prior_box_var_input[2].x * target_box_input[2].x) * prior_box_width;
                        target_box_height.x = exp(prior_box_var_input[3].x * target_box_input[3].x) * prior_box_height;

                        output[0].x = target_box_center_x.x - target_box_width.x/(half)2;
                        output[1].x = target_box_center_y.x - target_box_height.x/(half)2;
                        output[2].x = target_box_center_x.x + target_box_width.x/(half)2;
                        output[3].x = target_box_center_y.x + target_box_height.x/(half)2;

                        if(out_C - out_c * 4 >= 2){
                            target_box_center_x.y = prior_box_var_input[0].x * target_box_input[0].y * prior_box_width + prior_box_center_x;
                            target_box_center_y.y = prior_box_var_input[1].x * target_box_input[1].y * prior_box_height + prior_box_center_y;
                            target_box_width.y = exp(prior_box_var_input[2].x * target_box_input[2].y) * prior_box_width;
                            target_box_height.y = exp(prior_box_var_input[3].x * target_box_input[3].y) * prior_box_height;
                            output[0].y = target_box_center_x.y - target_box_width.y/(half)2;
                            output[1].y = target_box_center_y.y - target_box_height.y/(half)2;
                            output[2].y = target_box_center_x.y + target_box_width.y/(half)2;
                            output[3].y = target_box_center_y.y + target_box_height.y/(half)2;
                        }
                        if(out_C - out_c * 4 >= 3){
                            target_box_center_x.z = prior_box_var_input[0].x * target_box_input[0].z * prior_box_width + prior_box_center_x;
                            target_box_center_y.z = prior_box_var_input[1].x * target_box_input[1].z * prior_box_height + prior_box_center_y;
                            target_box_width.z = exp(prior_box_var_input[2].x * target_box_input[2].z) * prior_box_width;
                            target_box_height.z = exp(prior_box_var_input[3].x * target_box_input[3].z) * prior_box_height;
                            output[0].z = target_box_center_x.z - target_box_width.z/(half)2;
                            output[1].z = target_box_center_y.z - target_box_height.z/(half)2;
                            output[2].z = target_box_center_x.z + target_box_width.z/(half)2;
                            output[3].z = target_box_center_y.z + target_box_height.z/(half)2;
                        }
                        if(out_C - out_c * 4 >= 4){
                            target_box_center_x.w = prior_box_var_input[0].x * target_box_input[0].w * prior_box_width + prior_box_center_x;
                            target_box_center_y.w = prior_box_var_input[1].x * target_box_input[1].w * prior_box_height + prior_box_center_y;
                            target_box_width.w = exp(prior_box_var_input[2].x * target_box_input[2].w) * prior_box_width;
                            target_box_height.w = exp(prior_box_var_input[3].x * target_box_input[3].w) * prior_box_height;
                            output[0].w = target_box_center_x.w - target_box_width.w/(half)2;
                            output[1].w = target_box_center_y.w - target_box_height.w/(half)2;
                            output[2].w = target_box_center_x.w + target_box_width.w/(half)2;
                            output[3].w = target_box_center_y.w + target_box_height.w/(half)2;
                        }

                        WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, (int2)(output_pos.x + 0, output_pos.y), output[0]);
                        WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, (int2)(output_pos.x + 1, output_pos.y), output[1]);
                        WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, (int2)(output_pos.x + 2, output_pos.y), output[2]);
                        WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, (int2)(output_pos.x + 3, output_pos.y), output[3]);
}
