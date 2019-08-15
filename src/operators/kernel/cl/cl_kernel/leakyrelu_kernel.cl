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
__kernel void leakyrelu(__read_only image2d_t input,
    __write_only image2d_t output, __private const float alpha, __private const int dims_w) {
    const int c = get_global_id(0);
    const int w = get_global_id(1);
    const int nh = get_global_id(2);
    int2 input_pos;
    input_pos.x = c * dims_w + w;
    input_pos.y = nh;

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;

    half4 in = read_imageh(input, sampler, (int2)(input_pos.x, input_pos.y));

    half4 output_data;
    output_data.x = max((float)(in.x), (float)(alpha * (in.x)));
    output_data.y = max((float)(in.y), (float)(alpha * (in.y)));
    output_data.z = max((float)(in.z), (float)(alpha * (in.z)));
    output_data.w = max((float)(in.w), (float)(alpha * (in.w)));

    write_imageh(output, (int2)(input_pos.x, input_pos.y), output_data);
}
