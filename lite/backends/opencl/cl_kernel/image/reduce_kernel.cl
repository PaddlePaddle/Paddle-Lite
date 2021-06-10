/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

__kernel void reduce_n(__read_only image2d_t input,
                       __write_only image2d_t output,
                       __private const int4 in_nchw,
                       __private const int c4_n,
                       __private const int c4_r,
                       __private const int cw4,
                       __private const int axis_n) {
  const int cw = get_global_id(0);
  const int bh = get_global_id(1);

  CL_DTYPE4 t;
  CL_DTYPE4 r = (CL_DTYPE4)(DATAINIT);

  for (unsigned short i = 0; i < in_nchw.x; i++) {
    t = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw, in_nchw.z * i + bh));
    OPERATOR(r, t)
  }
  r = POSTOPERATOR(r);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(cw, bh), r);
}

__kernel void reduce_c(__read_only image2d_t input,
                       __write_only image2d_t output,
                       __private const int4 in_nchw,
                       __private const int c4_n,
                       __private const int c4_r,
                       __private const int cw4,
                       __private const int axis_n) {
  const int cw = get_global_id(0);
  const int bh = get_global_id(1);

  CL_DTYPE4 t;
  CL_DTYPE4 r = (CL_DTYPE4)(DATAINIT);

  for (unsigned short i = 0; i < c4_n; i++) {
    t = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_nchw.z * i + cw, bh));
    OPERATOR(r, t)
  }
  if (c4_r == 1) {
    t = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw4 + cw, bh));
    OPERATOR(r.x, t.x)
  } else if (c4_r == 2) {
    t = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw4 + cw, bh));
    OPERATOR(r.x, t.x)
    OPERATOR(r.y, t.y)
  } else if (c4_r == 3) {
    t = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw4 + cw, bh));
    OPERATOR(r.x, t.x)
    OPERATOR(r.y, t.y)
    OPERATOR(r.z, t.z)
  }

  r.x = INNEROPERATOR(r);
  r = POSTOPERATOR(r);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(cw, bh), r);
}

__kernel void reduce_h(__read_only image2d_t input,
                       __write_only image2d_t output,
                       __private const int4 in_nchw,
                       __private const int c4_n,
                       __private const int c4_r,
                       __private const int cw4,
                       __private const int axis_n) {
  const int cw = get_global_id(0);
  const int bh = get_global_id(1);

  CL_DTYPE4 t;
  CL_DTYPE4 r = (CL_DTYPE4)(DATAINIT);

  for (unsigned short i = 0; i < in_nchw.z; i++) {
    t = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw, in_nchw.z * bh + i));
    OPERATOR(r, t)
  }
  r = POSTOPERATOR(r);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(cw, bh), r);
}

__kernel void reduce_w(__read_only image2d_t input,
                       __write_only image2d_t output,
                       __private const int4 in_nchw,
                       __private const int c4_n,
                       __private const int c4_r,
                       __private const int cw4,
                       __private const int axis_n) {
  const int cw = get_global_id(0);
  const int bh = get_global_id(1);

  CL_DTYPE4 t;
  CL_DTYPE4 r = (CL_DTYPE4)(DATAINIT);

  for (unsigned short i = 0; i < in_nchw.w; i++) {
    t = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw * in_nchw.w + i, bh));
    OPERATOR(r, t)
  }
  r = POSTOPERATOR(r);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(cw, bh), r);
}

__kernel void reduce_multi_axis(__read_only image2d_t input,
                                __write_only image2d_t output,
                                __private const int4 in_nchw,
                                __private const int c4_n,
                                __private const int c4_r,
                                __private const int cw4,
                                __private const int axis_n,
                                __private const int4 axis_nhwc) {
  const int cw = get_global_id(0);
  const int bh = get_global_id(1);

  CL_DTYPE4 t;
  CL_DTYPE4 r = (CL_DTYPE4)(DATAINIT);
  int n_reduce_len = select(1, in_nchw.x, axis_nhwc.x);
  int h_reduce_len = select(1, in_nchw.z, axis_nhwc.y);
  int w_reduce_len = select(1, in_nchw.w, axis_nhwc.z);

  for (unsigned short n = 0; n < n_reduce_len; n++) {
    for (unsigned short h = 0; h < h_reduce_len; h++) {
      int img_h_idx = in_nchw.z * n + h + bh * h_reduce_len;
      for (unsigned short w = 0; w < w_reduce_len; w++) {
        for (unsigned short c_4 = 0; c_4 < select(1, c4_n, axis_nhwc.w);
             c_4++) {
          t = READ_IMG_TYPE(
              CL_DTYPE_CHAR,
              input,
              SAMPLER,
              (int2)(in_nchw.w * c_4 + w + cw * w_reduce_len, img_h_idx));
          OPERATOR(r, t)
        }

        if (axis_nhwc.w) {
          if (c4_r == 1) {
            t = READ_IMG_TYPE(
                CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw4 + w + cw, img_h_idx));
            OPERATOR(r.x, t.x)
          } else if (c4_r == 2) {
            t = READ_IMG_TYPE(
                CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw4 + w + cw, img_h_idx));
            OPERATOR(r.x, t.x)
            OPERATOR(r.y, t.y)
          } else if (c4_r == 3) {
            t = READ_IMG_TYPE(
                CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw4 + w + cw, img_h_idx));
            OPERATOR(r.x, t.x)
            OPERATOR(r.y, t.y)
            OPERATOR(r.z, t.z)
          }
        }
      }
    }
  }

  if (axis_nhwc.w) {
    r.x = INNEROPERATOR(r);
  }
  r = POSTOPERATOR(r);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(cw, bh), r);
}
