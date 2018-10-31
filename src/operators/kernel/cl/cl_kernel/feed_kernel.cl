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
__kernel void feed(__global float *in, __write_only image2d_t outputImage,int h,int w)
 {
        int i = get_global_id(0);
        int j = get_global_id(1);
        half4 pixel;
        pixel.x = convert_half(in[(i * w + j)]);
        pixel.y = convert_half(in[h * w + (i * w + j)]);
        pixel.z = convert_half(in[2 * h * w + (i * w + j)]);
        pixel.w = 0.0;
        int2 coords;
        coords.x = j;
        coords.y = i;

        write_imageh(outputImage,coords,pixel);
 }
