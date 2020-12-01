/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

__kernel void dropout(__read_only image2d_t input_image,
                      __write_only image2d_t output_image,
                      __private const int out_W,
                      __private const float dropoutPro) {

                       const int out_c = get_global_id(0);
                       const int out_w = get_global_id(1);
                       const int out_nh = get_global_id(2);

                       int2 output_pos = {out_c * out_W + out_w, out_nh};

                       CL_DTYPE4 input = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER, output_pos);
                       CL_DTYPE4 dropout = (CL_DTYPE4)(1 - dropoutPro);
                       CL_DTYPE4 output =  dropout * input;

                       WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output);
}


