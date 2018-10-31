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

__kernel void softmax(__read_only image2d_t input_image,
                      __write_only image2d_t output_image,
                      __private const int group
                      ) {
    const int out_c = get_global_id(0);   //  block index
    const int out_w = get_global_id(1);   // index in one block
    const int out_nh = get_global_id(2);


  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;

  half maxv = 0.0f;
  for (int i = 0; i < group; ++i) {
    half4 temp = read_imageh(input_image, sampler, (int2)(i, 0));
    maxv = max(maxv, max(temp.x, max(temp.y, max(temp.z, temp.w))));
  }


  half4 rsum = (half4)(0.0f);
  for (int i = 0; i < group; ++i) {
    half4 r = read_imageh(input_image, sampler, (int2)(i, 0));
    rsum += convert_half4(exp(convert_float4(r - maxv)));
  }

  float sum = rsum.x + rsum.y + rsum.z + rsum.w;

  half4 rr = read_imageh(input_image, sampler, (int2)(out_w, out_nh));
  half4 result = convert_half4(exp(convert_float4(rr - maxv)) / sum);
  write_imageh(output_image, (int2)(out_w, out_nh), result);
}

/*

__kernel void softmax(__read_only image2d_t input,
                      __write_only image2d_t output,
                      __private const int d0,
                      __private const int d1,
                      __private const int d2,
                      __private const int d3) {
  const int z = get_global_id(0);
  const int x = get_global_id(1);
  const int y = get_global_id(2);
  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;
  half4 cv = read_imageh(input, sampler, (int2)(x, y));
  half4 maxv = cv;
  for (int i = 0; i < d3; i++) {
    half4 temp = read_imageh(input, sampler, (int2)(z * d3 + i, y));
    maxv = max(maxv, temp);
  }
  half4 sum = (half4)0.0f;
  // half4 x = = (half4)0.0f;
  for (int i = 0; i < d3; i++) {
    half4 temp = read_imageh(input, sampler, (int2)(z * d3 + i, y));
    sum += exp(temp - maxv);
  }
  half4 r = exp(cv - maxv) / sum;

  write_imageh(output, (int2)(z * d3 + x, y), r);
}

*/
