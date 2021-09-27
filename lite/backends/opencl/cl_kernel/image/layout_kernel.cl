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

// #define DEBUG
////////////////////////////////////////////////////////
// buffer -> image2d
////////////////////////////////////////////////////////
__kernel void buffer_to_image2d(__global CL_DTYPE* in,
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

  CL_COMPUTE_DTYPE4 output = (CL_COMPUTE_DTYPE4)(0.f, 0.f, 0.f, 0.f);
  output.x = CONVERT_TYPE_TO(in[input_pos0], CL_COMPUTE_DTYPE);

  if (out_C - 4 * out_c >= 2) {
    output.y = CONVERT_TYPE_TO(in[input_pos1], CL_COMPUTE_DTYPE);
  }
  if (out_C - 4 * out_c >= 3) {
    output.z = CONVERT_TYPE_TO(in[input_pos2], CL_COMPUTE_DTYPE);
  }
  if (out_C - 4 * out_c >= 4) {
    output.w = CONVERT_TYPE_TO(in[input_pos3], CL_COMPUTE_DTYPE);
  }

#ifdef DEBUG
  if (out_w > 2045) {
    printf(
        "out_w:%d, out_C - 4 * out_c:%d, input[pos0~pos3]:%.2f %.2f %.2f "
        "%.2f\n",
        out_w,
        out_C - 4 * out_c,
        (float)(in[input_pos0]),
        (float)(in[input_pos1]),
        (float)(in[input_pos2]),
        (float)(in[input_pos3]));
    printf("buffer2image ===> %d,%d,%d, out(%d,%d): %.2f %.2f %.2f %.2f \n",
           out_c,
           out_w,
           out_nh,
           output_pos.x,
           output_pos.y,
           (float)(output.x),
           (float)(output.y),
           (float)(output.z),
           (float)(output.w));
  }
#endif

  WRITE_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR, output_image, output_pos, output);
}

////////////////////////////////////////////////////////
// image2d -> buffer
////////////////////////////////////////////////////////
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

  const int pos_x = mad24(in_c, in_width, in_w);
  CL_COMPUTE_DTYPE4 in = READ_IMG_TYPE(
      CL_COMPUTE_DTYPE_CHAR, input, SAMPLER, (int2)(pos_x, in_nh));

#ifdef DEBUG
  if (in_w > 2045) {
    printf("image2buffer ===> %d,%d,%d, in(%d,%d): %.2f %.2f %.2f %.2f \n",
           in_c,
           in_w,
           in_nh,
           pos_x,
           in_nh,
           (float)(in.x),
           (float)(in.y),
           (float)(in.z),
           (float)(in.w));
  }
#endif

  const int index =
      in_n * size_batch + in_c * size_block + in_h * in_width + in_w;
  out[index] = CONVERT_TYPE_TO(in.x, CL_DTYPE);
  if (C - 4 * in_c >= 2) {
    out[index + size_ch] = CONVERT_TYPE_TO(in.y, CL_DTYPE);
  }
  if (C - 4 * in_c >= 3) {
    out[index + size_ch * 2] = CONVERT_TYPE_TO(in.z, CL_DTYPE);
  }
  if (C - 4 * in_c >= 4) {
    out[index + size_ch * 3] = CONVERT_TYPE_TO(in.w, CL_DTYPE);
  }
}

#if 0  // NOTE(ysh329): keep, un-used from paddle-mobile
////////////////////////////////////////////////////////
// buffer -> image2d_nw
////////////////////////////////////////////////////////
__kernel void buffer_to_image2d_nw(__global CL_DTYPE* in,
                                   __write_only image2d_t output_image,
                                   __private const int out_H,
                                   __private const int out_W,
                                   __private const int out_N,
                                   __private const int Stride0,
                                   __private const int Stride1,
                                   __private const int Stride2) {
  const int out_n = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_ch = get_global_id(2);

  const int out_c = out_ch / out_H;
  const int out_h = out_ch % out_H;

  const int in_c = out_c; //  index of c in h direction

  const int in_n0 = out_n * 4 + 0;
  const int in_n1 = out_n * 4 + 1;
  const int in_n2 = out_n * 4 + 2;
  const int in_n3 = out_n * 4 + 3;

  const int in_h = out_h;
  const int in_w = out_w;

  int input_pos0 = in_n0 * Stride2 + in_c * Stride1 + in_h * Stride0 + in_w;
  int input_pos1 = in_n1 * Stride2 + in_c * Stride1 + in_h * Stride0 + in_w;
  int input_pos2 = in_n2 * Stride2 + in_c * Stride1 + in_h * Stride0 + in_w;
  int input_pos3 = in_n3 * Stride2 + in_c * Stride1 + in_h * Stride0 + in_w;

  int2 output_pos;
  output_pos.x = out_n * out_W + out_w;
  output_pos.y = out_ch;

  CL_DTYPE4 output = (CL_DTYPE4)0.0f;
  output.x = CONVERT_TYPE_TO(CL_DTYPE, in[input_pos0]);
  if (out_N - 4 * out_n >= 2) {
    output.y = CONVERT_TYPE_TO(CL_DTYPE, in[input_pos1]);
  }
  if (out_N - 4 * out_n >= 3) {
    output.z = CONVERT_TYPE_TO(CL_DTYPE, in[input_pos2]);
  }
  if (out_N - 4 * out_n >= 4) {
    output.w = CONVERT_TYPE_TO(CL_DTYPE, in[input_pos3]);
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output);
}
#endif

#if 0  // NOTE(ysh329): keep, un-used from paddle-mobile
// image2d -> buffer
__kernel void image2d_to_buffer_2d(__private const int in_height,
                                   __private const int in_width,
                                   __read_only image2d_t input,
                                   __global CL_DTYPE* out) {
  const int in_w = get_global_id(1);
  const int in_h = get_global_id(2);

  CL_DTYPE4 in = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_w, in_h));

  const int index = (in_h * in_width + in_w) * 4;
  out[index] = CONVERT_TYPE_TO(CL_DTYPE, in.x);
  out[index + 1] = CONVERT_TYPE_TO(CL_DTYPE, in.y);
  out[index + 2] = CONVERT_TYPE_TO(CL_DTYPE, in.z);
  out[index + 3] = CONVERT_TYPE_TO(CL_DTYPE, in.w);
}
#endif

////////////////////////////////////////////////////////
// buffer -> image2d (divide by 255 to normalize)
////////////////////////////////////////////////////////
__kernel void buffer_to_image2d_with_pre255(__global uchar* in,
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

  CL_COMPUTE_DTYPE4 output = (CL_COMPUTE_DTYPE4)0.0f;
  output.x = CONVERT_TYPE_TO(in[input_pos0], CL_COMPUTE_DTYPE) / 255;
  if (out_C - 4 * out_c >= 2) {
    output.y = CONVERT_TYPE_TO(in[input_pos1], CL_COMPUTE_DTYPE) / 255;
  }
  if (out_C - 4 * out_c >= 3) {
    output.z = CONVERT_TYPE_TO(in[input_pos2], CL_COMPUTE_DTYPE) / 255;
  }
  if (out_C - 4 * out_c >= 4) {
    output.w = CONVERT_TYPE_TO(in[input_pos3], CL_COMPUTE_DTYPE) / 255;
  }
  WRITE_IMG_TYPE(CL_COMPUTE_DTYPE_CHAR, output_image, output_pos, output);
}

////////////////////////////////////////////////////////
// image2d -> buffer (multiply by 255 to de-normalize)
////////////////////////////////////////////////////////
__kernel void image2d_to_buffer_with_post255(__read_only image2d_t input,
                                             __private const int in_width,
                                             __private const int in_height,
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

  const int pos_x = mad24(in_c, in_width, in_w);
  CL_COMPUTE_DTYPE4 in =
      READ_IMG_TYPE(
          CL_COMPUTE_DTYPE_CHAR, input, SAMPLER, (int2)(pos_x, in_nh)) *
      255;

#ifdef DEBUG
  printf("in_c:%d, in_w:%d, in_nh:%d ===> in(%d,%d): %.2f %.2f %.2f %.2f\n",
         in_c,
         in_w,
         in_nh,
         pos_x,
         in_nh,
         in.x,
         in.y,
         in.z,
         in.w);
#endif

  const int index =
      in_n * size_batch + in_c * size_block + in_h * in_width + in_w;
  out[index] = convert_uchar_sat(in.x);
  if (C - 4 * in_c >= 2) {
    out[index + size_ch] = convert_uchar_sat(in.y);
  }
  if (C - 4 * in_c >= 3) {
    out[index + size_ch * 2] = convert_uchar_sat(in.z);
  }
  if (C - 4 * in_c >= 4) {
    out[index + size_ch * 3] = convert_uchar_sat(in.w);
  }
}
