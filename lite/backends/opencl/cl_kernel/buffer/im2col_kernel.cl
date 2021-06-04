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
#define CL_DTYPE float

#include <cl_common.h>

__kernel void im2col(__global const CL_DTYPE* data_im,
                     const int img_offset,
                     const int col_chw,
                     const int height,
                     const int width,
                     const int kernel_h,
                     const int kernel_w,
                     const int pad_h,
                     const int pad_w,
                     const int stride_h,
                     const int stride_w,
                     const int dilation_h,
                     const int dilation_w,
                     const int height_col,
                     const int width_col,
                     __global CL_DTYPE* col_data,
                     const int col_offset) {
  int index = get_global_id(0);  // [0, col_chw)

  data_im = data_im + img_offset;
  col_data = col_data + col_offset;

  if (index < col_chw) {
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;

    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;

    __global CL_DTYPE* col_data_ptr = col_data;
    col_data_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    __global const CL_DTYPE* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;

    int dh = 0;
    for (int i = 0; i < kernel_h; ++i) {
      int dw = 0;
      for (int j = 0; j < kernel_w; ++j) {
        int h = h_in + dh;
        int w = w_in + dw;
        *col_data_ptr = (h >= 0 && w >= 0 && h < height && w < width)
                            ? data_im_ptr[dh * width + dw]
                            : 0;
        col_data_ptr += height_col * width_col;
        dw += dilation_w;
      }
      dh += dilation_h;
    }
  }
}
