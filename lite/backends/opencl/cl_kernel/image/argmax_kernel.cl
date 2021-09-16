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

__kernel void argmax_n(__read_only image2d_t input,
                       __write_only image2d_t output,
                       __private const int4 in_nchw,
                       __private const int c4_n,
                       __private const int c4_r,
                       __private const int cw4) {
  const int cw = get_global_id(0);
  const int bh = get_global_id(1);

  CL_DTYPE4 cur_data = (CL_DTYPE4)(DATAINIT);
  CL_DTYPE4 max_data = (CL_DTYPE4)(DATAINIT);
  CL_DTYPE4 cur_idx = (CL_DTYPE4)(DATAINIT);
  CL_DTYPE4 max_idx = (CL_DTYPE4)(DATAINIT);

  FLAG_TYPE4 flag_v = (FLAG_TYPE4)(0);

  for (unsigned short i = 0; i < in_nchw.x; i++) {
    cur_data = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw, in_nchw.z * i + bh));

    cur_idx = (CL_DTYPE4)(i);
    flag_v = isgreaterequal(cur_data, max_data);
    max_data = select(max_data, cur_data, flag_v);
    max_idx = select(max_idx, cur_idx, flag_v);
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(cw, bh), max_idx);
}

__kernel void argmax_c(__read_only image2d_t input,
                       __write_only image2d_t output,
                       __private const int4 in_nchw,
                       __private const int c4_n,
                       __private const int c4_r,
                       __private const int cw4) {
  const int cw = get_global_id(0);
  const int bh = get_global_id(1);

  CL_DTYPE4 cur_data = (CL_DTYPE4)(DATAINIT);
  CL_DTYPE4 max_data = (CL_DTYPE4)(DATAINIT);
  CL_DTYPE4 cur_idx = (CL_DTYPE4)(DATAINIT);
  CL_DTYPE4 max_idx = (CL_DTYPE4)(DATAINIT);

  FLAG_TYPE4 flag_v = (FLAG_TYPE4)(0);

  for (unsigned short i = 0; i < c4_n; i++) {
    cur_data = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_nchw.w * i + cw, bh));
    cur_idx = (CL_DTYPE4)(i << 2, (i << 2) + 1, (i << 2) + 2, (i << 2) + 3);
    flag_v = isgreaterequal(cur_data, max_data);
    max_data = select(max_data, cur_data, flag_v);
    max_idx = select(max_idx, cur_idx, flag_v);
  }

  if (c4_r != 0) {
    cur_data =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw4 + cw, bh));
  }

  if (c4_r >= 1) {
    cur_idx.x = (c4_n << 2);
    flag_v.x = isgreaterequal(cur_data.x, max_data.x);
    max_data.x = select(max_data.x, cur_data.x, flag_v.x);
    max_idx.x = select(max_idx.x, cur_idx.x, flag_v.x);
  }
  if (c4_r >= 2) {
    cur_idx.y = (c4_n << 2) + 1;
    flag_v.y = isgreaterequal(cur_data.y, max_data.y);
    max_data.y = select(max_data.y, cur_data.y, flag_v.y);
    max_idx.y = select(max_idx.y, cur_idx.y, flag_v.y);
  }
  if (c4_r == 3) {
    cur_idx.z = (c4_n << 2) + 2;
    flag_v.z = isgreaterequal(cur_data.z, max_data.z);
    max_data.z = select(max_data.z, cur_data.z, flag_v.z);
    max_idx.z = select(max_idx.z, cur_idx.z, flag_v.z);
  }

  if (max_data.y > max_data.x) {
    max_data.x = max_data.y;
    max_idx.x = max_idx.y;
  }

  if (max_data.z > max_data.x) {
    max_data.x = max_data.z;
    max_idx.x = max_idx.z;
  }

  if (max_data.w > max_data.x) {
    max_data.x = max_data.w;
    max_idx.x = max_idx.w;
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(cw, bh), max_idx);
}

__kernel void argmax_h(__read_only image2d_t input,
                       __write_only image2d_t output,
                       __private const int4 in_nchw,
                       __private const int c4_n,
                       __private const int c4_r,
                       __private const int cw4) {
  const int cw = get_global_id(0);
  const int bh = get_global_id(1);

  CL_DTYPE4 cur_data = (CL_DTYPE4)(DATAINIT);
  CL_DTYPE4 max_data = (CL_DTYPE4)(DATAINIT);
  CL_DTYPE4 cur_idx = (CL_DTYPE4)(DATAINIT);
  CL_DTYPE4 max_idx = (CL_DTYPE4)(DATAINIT);

  FLAG_TYPE4 flag_v = (FLAG_TYPE4)(0);

  for (unsigned short i = 0; i < in_nchw.z; i++) {
    cur_data = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw, in_nchw.z * bh + i));

    cur_idx = (CL_DTYPE4)(i);
    flag_v = isgreaterequal(cur_data, max_data);
    max_data = select(max_data, cur_data, flag_v);
    max_idx = select(max_idx, cur_idx, flag_v);
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(cw, bh), max_idx);
}

__kernel void argmax_w(__read_only image2d_t input,
                       __write_only image2d_t output,
                       __private const int4 in_nchw,
                       __private const int c4_n,
                       __private const int c4_r,
                       __private const int cw4) {
  const int cw = get_global_id(0);
  const int bh = get_global_id(1);

  CL_DTYPE4 cur_data = (CL_DTYPE4)(DATAINIT);
  CL_DTYPE4 max_data = (CL_DTYPE4)(DATAINIT);
  CL_DTYPE4 cur_idx = (CL_DTYPE4)(DATAINIT);
  CL_DTYPE4 max_idx = (CL_DTYPE4)(DATAINIT);

  FLAG_TYPE4 flag_v = (FLAG_TYPE4)(0);

  for (unsigned short i = 0; i < in_nchw.w; i++) {
    cur_data = READ_IMG_TYPE(
        CL_DTYPE_CHAR, input, SAMPLER, (int2)(cw * in_nchw.w + i, bh));

    cur_idx = (CL_DTYPE4)(i);
    flag_v = isgreaterequal(cur_data, max_data);
    max_data = select(max_data, cur_data, flag_v);
    max_idx = select(max_idx, cur_idx, flag_v);
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(cw, bh), max_idx);
}
