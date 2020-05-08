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

__kernel void fetch(__private const int in_height,
                    __private const int in_width,
                    __read_only image2d_t input,
                    __global float* out,
                    __private const int size_ch,
                    __private const int size_block,
                    __private const int size_batch,
                    __private const int C) {
  const int in_c = get_global_id(0);
  const int in_w = get_global_id(1);
  const int in_nh = get_global_id(2);
  const int in_n = in_nh / in_height;
  const int in_h = in_nh % in_height;

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  const int pos_x = mad24(in_c, in_width, in_w);
  half4 in = read_imageh(input, sampler, (int2)(pos_x, in_nh));

  const int index = in_n * size_batch + in_c * size_block + in_h * in_width + in_w;
  out[index] = convert_float(in.x);
  if(C - 4 * in_c>=2){
   out[index + size_ch] = convert_float(in.y);
  }
  if(C - 4 * in_c>=3){
  out[index + size_ch * 2] = convert_float(in.z);
  }

  if(C - 4 * in_c>=4){
   out[index + size_ch * 3] = convert_float(in.w);
  }

}

__kernel void fetch_2d(__private const int in_height,
                       __private const int in_width,
                       __read_only image2d_t input,
                       __global float* out) {
  const int in_w = get_global_id(1);
  const int in_h = get_global_id(2);

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  half4 in = read_imageh(input, sampler, (int2)(in_w, in_h));

  const int index = (in_h * in_width + in_w) * 4;
  out[index] = convert_float(in.x);
  out[index + 1] = convert_float(in.y);
  out[index + 2] = convert_float(in.z);
  out[index + 3] = convert_float(in.w);
}

__kernel void fetch_with_post(__private const int in_height,
    __private const int in_width,
    __read_only image2d_t input,
    __global uchar* out,
    __private const int size_ch,
    __private const int size_block,
    __private const int size_batch,
    __private const int C) {
  const int in_c = get_global_id(0);
  const int in_w = get_global_id(1);
  const int in_nh = get_global_id(2);
  const int in_n = in_nh / in_height;
  const int in_h = in_nh % in_height;

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  const int pos_x = mad24(in_c, in_width, in_w);
  half4 in = read_imageh(input, sampler, (int2)(pos_x, in_nh));

  const int index = in_n * size_batch + in_c * size_block + in_h * in_width + in_w;
  out[index] = convert_uchar_sat(in.x * 255);
  if(C - 4 * in_c>=2){
    out[index + size_ch] = convert_uchar_sat(in.y * 255);
  }
  if(C - 4 * in_c>=3){
    out[index + size_ch * 2] = convert_uchar_sat(in.z * 255);
  }

  if(C - 4 * in_c>=4){
    out[index + size_ch * 3] = convert_uchar_sat(in.w * 255);
  }

}
