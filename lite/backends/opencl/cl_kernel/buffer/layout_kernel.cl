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

__kernel void ImgHwcToBufferChw(__read_only image2d_t img_data, const int img_h, const int img_w,
                                __global CL_DTYPE* nchw_data, const int n, const int c, const int h, const int w) {
  const int w_idx = get_global_id(0);
  const int h_idx = get_global_id(1);

  if ((w_idx >= img_w) || h_idx >= img_h) {
    return;
  }
  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;
  CL_DTYPE4 in = read_imagef(img_data, sampler, (int2)(w_idx, h_idx));

}


__kernel void BufferChwToImgHwc(__global const CL_DTYPE* nchw_data, const int n, const int c, const int h, const int w,
                                __write_only image2d_t img_data, const int img_h, const int img_w) {
  const int w_idx = get_global_id(0);
  const int h_idx = get_global_id(1);

  if ((w_idx >= img_w) || h_idx >= img_h) {
    return;
  }
  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;
  write_imagef(img_data, (int2)(w_idx, h_idx), (CL_DTYPE4)(1));
}


// buffer -> image2d
__kernel void buffer_to_image2d(__global CL_DTYPE *in,
                                __write_only image2d_t output_image,
                                __private const int out_H,
                                __private const int out_W,
                                __private const int out_C,
                                __private const int Stride0,
                                __private const int Stride1,
                                __private const int Stride2) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_H;
  const int out_h = out_nh % out_H;

  const int in_n = out_n;
  const int in_c0 = out_c * 4 + 0;
  const int in_c1 = out_c * 4 + 1;
  const int in_c2 = out_c * 4 + 2;
  const int in_c3 = out_c * 4 + 3;
  const int in_h = out_h;
  const int in_w = out_w;

  int input_pos0 = in_n * Stride2 + in_c0 * Stride1 + in_h * Stride0 + in_w;
  int input_pos1 = in_n * Stride2 + in_c1 * Stride1 + in_h * Stride0 + in_w;
  int input_pos2 = in_n * Stride2 + in_c2 * Stride1 + in_h * Stride0 + in_w;
  int input_pos3 = in_n * Stride2 + in_c3 * Stride1 + in_h * Stride0 + in_w;

  int2 output_pos;
  output_pos.x = out_c * out_W + out_w;
  output_pos.y = out_nh;

  CL_DTYPE4 output = (CL_DTYPE4)0.0f;
  output.x = convert_float(in[input_pos0]);
  if(out_C - 4 * out_c >= 2){
    output.y = convert_float(in[input_pos1]);
  }
  if(out_C - 4 * out_c >= 3){
    output.z = convert_float(in[input_pos2]);
  }
  if(out_C - 4 * out_c >= 4){
    output.w = convert_float(in[input_pos3]);
  }
  write_imagef(output_image, output_pos, output);

#if 0
  // print output_image with another image pointer `output_image_` with __read_only def
  const sampler_t sampler =
    CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  CL_DTYPE4 out_read = read_imagef(output_image_, sampler, output_pos);
  CL_DTYPE4 out_read_v = read_imagef(output_image_, sampler, (int2)(output_pos.y, output_pos.x));
  if (output_pos.x < 10) {
    printf("b->img | out_c:%d,out_w:%d,out_nc:%d | in_pos:(%d,%d,%d,%d) | in:%.1f %.1f %.1f %.1f | out_pos:(%d,%d) | write_out: %.1f %.1f %.1f %.1f | read_out_image %.1f %.1f %.1f %.1f | out_pos_rev:(%d,%d) | read_out_image_rev: %.1f %.1f %.1f %.1f\n",
           out_c, out_w, out_nh,
           input_pos0, input_pos1, input_pos2, input_pos3,
           in[input_pos0], in[input_pos1], in[input_pos2], in[input_pos3],
           output_pos.x, output_pos.y,
           output.x, output.y, output.z, output.w,
           out_read.x, out_read.y, out_read.z, out_read.w,

           output_pos.y, output_pos.x,
           out_read_v.x, out_read_v.y, out_read_v.z, out_read_v.w);
           }
           #endif
}

// image2d -> buffer
__kernel void image2d_to_buffer(__read_only image2d_t input,
                                __private const int in_width,
                                __private const int in_height,
                                __global CL_DTYPE* out,
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
  CL_DTYPE4 in = read_imagef(input, sampler, (int2)(pos_x, in_nh));

  const int index = in_n * size_batch + in_c * size_block + in_h * in_width + in_w;
  out[index] = convert_float(in.x);
  if (C - 4 * in_c >= 2) {
    out[index + size_ch] = convert_float(in.y);
  }
  if(C - 4 * in_c >= 3) {
    out[index + size_ch * 2] = convert_float(in.z);
  }
  if(C - 4 * in_c >= 4) {
    out[index + size_ch * 3] = convert_float(in.w);
  }
}

// image2d -> buffer
__kernel void image2d_to_buffer_2d(__private const int in_height,
                                   __private const int in_width,
                                   __read_only image2d_t input,
                                   __global CL_DTYPE* out) {
  const int in_w = get_global_id(1);
  const int in_h = get_global_id(2);

  const sampler_t sampler =
    CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  CL_DTYPE4 in = read_imagef(input, sampler, (int2)(in_w, in_h));

  const int index = (in_h * in_width + in_w) * 4;
  out[index] = convert_float(in.x);
  out[index + 1] = convert_float(in.y);
  out[index + 2] = convert_float(in.z);
  out[index + 3] = convert_float(in.w);
}
