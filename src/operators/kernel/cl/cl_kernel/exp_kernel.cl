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
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

__kernel void exp_impl(__read_only image2d_t input, __write_only image2d_t output) {
   const int x = get_global_id(0);
   const int y = get_global_id(1);

   const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                             CLK_ADDRESS_CLAMP |
                             CLK_FILTER_NEAREST;

   half4 in = read_imageh(input, sampler, (int2)(x, y));
   half4 out;
   out.x = pow(2.71828182, (float)(in.x));
   out.y = pow(2.71828182, (float)(in.y));
   out.z = pow(2.71828182, (float)(in.z));
   out.w = pow(2.71828182, (float)(in.w));
   write_imageh(output, (int2)(x, y), out);
}
