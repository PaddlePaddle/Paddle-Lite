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
                      __private const int out_W
                      ) {
    const int out_c = get_global_id(0);   //  block index
    const int out_w = get_global_id(1);   // index in one block
    const int out_nh = get_global_id(2);

    const int in_c = out_c;
    const int in_w = out_w;
    const int in_nh = out_nh;

    int2 input_pos;
    int2 output_pos;

    input_pos.x = in_c * out_W + in_w;
    input_pos.y = in_nh;

    output_pos.x = out_c * out_W + out_w;
    output_pos.y = out_nh;

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                                CLK_ADDRESS_CLAMP |
                                CLK_FILTER_NEAREST;

    half4 input_max = 0.0f;
    half4 input_tmp;
    for(int i=0;i<out_W;i++){
     input_tmp = read_imageh(input_image, sampler,(int2)(in_c * out_W + i,in_nh));
     input_max = max(input_max,input_tmp);
    }

    half4 sum = (half4)0.0f;
    for(int i=0;i<out_W;i++){
        input_tmp = read_imageh(input_image, sampler,(int2)(in_c * out_W + i,in_nh));
        sum += exp(input_tmp - input_max);
       }

       half4 input = read_imageh(input_image, sampler,input_pos);
       half4 output = exp(input - input_max)/sum;
       write_imageh(output_image, output_pos, output);

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
