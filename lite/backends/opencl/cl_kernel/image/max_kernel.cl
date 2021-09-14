/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

__kernel void max_n(__read_only image2d_t input,
                    __write_only image2d_t output,
                    __private const int4 in_nchw,
                    __private const int c4_n,
                    __private const int c4_r,
                    __private const int cw4,
                    __private const int axis_n) {
  const int cw = get_global_id(0);
  const int bh = get_global_id(1);

  CL_DTYPE4 cur_data;
  CL_DTYPE4 max_data = (CL_DTYPE4)(DATAINIT);

  for (unsigned short i = 0; i < in_nchw.x; i++) {
    cur_data = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw, in_nchw.z * i + bh));
    max_data = fmax(max_data, cur_data);
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(cw, bh), max_data);
}

__kernel void max_c(__read_only image2d_t input,
                    __write_only image2d_t output,
                    __private const int4 in_nchw,
                    __private const int c4_n,
                    __private const int c4_r,
                    __private const int cw4,
                    __private const int axis_n) {
  const int cw = get_global_id(0);
  const int bh = get_global_id(1);

  CL_DTYPE4 cur_data;
  CL_DTYPE4 max_data = (CL_DTYPE4)(DATAINIT);

  for (unsigned short i = 0; i < c4_n; i++) {
    cur_data = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_nchw.w * i + cw, bh));
    max_data = fmax(max_data, cur_data);
  }

  if (c4_r != 0) {
    cur_data =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw4 + cw, bh));
  }

  if (c4_r >= 1) {
    max_data.x = fmax(max_data.x, cur_data.x);
  }
  if (c4_r >= 2) {
    max_data.y = fmax(max_data.y, cur_data.y);
  }
  if (c4_r == 3) {
    max_data.z = fmax(max_data.z, cur_data.z);
  }

  max_data.x = fmax(max_data.x, max_data.y);
  max_data.x = fmax(max_data.x, max_data.z);
  max_data.x = fmax(max_data.x, max_data.w);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(cw, bh), max_data);
}

__kernel void max_h(__read_only image2d_t input,
                    __write_only image2d_t output,
                    __private const int4 in_nchw,
                    __private const int c4_n,
                    __private const int c4_r,
                    __private const int cw4,
                    __private const int axis_n) {
  const int cw = get_global_id(0);
  const int bh = get_global_id(1);

  CL_DTYPE4 cur_data;
  CL_DTYPE4 max_data = (CL_DTYPE4)(DATAINIT);

  for (unsigned short i = 0; i < in_nchw.z; i++) {
    cur_data = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw, in_nchw.z * bh + i));

    max_data = fmax(max_data, cur_data);
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(cw, bh), max_data);
}

__kernel void max_w(__read_only image2d_t input,
                    __write_only image2d_t output,
                    __private const int4 in_nchw,
                    __private const int c4_n,
                    __private const int c4_r,
                    __private const int cw4,
                    __private const int axis_n) {
  const int cw = get_global_id(0);
  const int bh = get_global_id(1);

  CL_DTYPE4 cur_data;
  CL_DTYPE4 max_data = (CL_DTYPE4)(DATAINIT);

  for (unsigned short i = 0; i < in_nchw.w; i++) {
    cur_data = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw * in_nchw.w + i, bh));

    max_data = fmax(max_data, cur_data);
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(cw, bh), max_data);
}

__kernel void max_multi_axis(__read_only image2d_t input,
                             __write_only image2d_t output,
                             __private const int4 in_nchw,
                             __private const int c4_n,
                             __private const int c4_r,
                             __private const int cw4,
                             __private const int axis_n,
                             __private const int4 axis_nhwc) {
  const int cw = get_global_id(0);
  const int bh = get_global_id(1);

  int n_reduce_len = select(1, in_nchw.x, axis_nhwc.x);
  int h_reduce_len = select(1, in_nchw.z, axis_nhwc.y);
  int w_reduce_len = select(1, in_nchw.w, axis_nhwc.z);

  CL_DTYPE4 cur_data;
  CL_DTYPE4 max_data = (CL_DTYPE4)(DATAINIT);

  for (unsigned short n = 0; n < n_reduce_len; n++) {
    for (unsigned short h = 0; h < h_reduce_len; h++) {
      int img_h_idx = in_nchw.z * n + h + bh * h_reduce_len;
      for (unsigned short w = 0; w < w_reduce_len; w++) {
        for (unsigned short c_4 = 0; c_4 < select(1, c4_n, axis_nhwc.w);
             c_4++) {
          cur_data = READ_IMG_TYPE(
              CL_DTYPE_CHAR,
              input,
              SAMPLER,
              (int2)(in_nchw.w * c_4 + w + cw * w_reduce_len, img_h_idx));
          max_data = fmax(max_data, cur_data);
        }

        if (axis_nhwc.w) {
          if (c4_r == 1) {
            cur_data = READ_IMG_TYPE(
                CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw4 + w + cw, img_h_idx));
            max_data.x = fmax(max_data.x, cur_data.x);
          } else if (c4_r == 2) {
            cur_data = READ_IMG_TYPE(
                CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw4 + w + cw, img_h_idx));
            max_data.x = fmax(max_data.x, cur_data.x);
            max_data.y = fmax(max_data.y, cur_data.y);
          } else if (c4_r == 3) {
            cur_data = READ_IMG_TYPE(
                CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw4 + w + cw, img_h_idx));
            max_data.x = fmax(max_data.x, cur_data.x);
            max_data.y = fmax(max_data.y, cur_data.y);
            max_data.z = fmax(max_data.z, cur_data.z);
          }
        }
      }
    }
  }

  if (axis_nhwc.w) {
    max_data.x = fmax(max_data.x, max_data.y);
    max_data.x = fmax(max_data.x, max_data.z);
    max_data.x = fmax(max_data.x, max_data.w);
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(cw, bh), max_data);
}
