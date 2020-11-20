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

__kernel void SplitBatch(__read_only image2d_t input,
                         __write_only image2d_t output0,
                         __write_only image2d_t output1,
                         __private const int axis,
                         __private const int out0_dims_axis,
                         __private const int in_dims_second,
                         __private const int in_dims_last,
                         __private const int width) {
  const int channel_blk_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);

  const int2 in_pos = (int2)(channel_blk_idx * in_dims_last + width_idx, hb_idx);
  const CL_DTYPE4 in_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, in_pos);
  const int n = hb_idx / width;

  int2 out_pos;
  if (n < out0_dims_axis) {
    out_pos = in_pos;
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output0, out_pos, in_data);
  } else {
    out_pos.y = hb_idx - out0_dims_axis;
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output1, out_pos, in_data);
  }
}

__kernel void SplitChannel(__read_only image2d_t input,
                           __write_only image2d_t output0,
                           __write_only image2d_t output1,
                           __private const int axis,
                           __private const int out0_dims_axis,
                           __private const int in_dims_second,
                           __private const int in_dims_last,
                           __private const int width) {
  const int channel_blk_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);

  const int2 in_pos = (int2)(channel_blk_idx * in_dims_last + width_idx, hb_idx);
  const CL_DTYPE4 in_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, in_pos);
  const int c = channel_blk_idx * 4;

  // write all data to output0 directly
  if (c < out0_dims_axis) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output0, in_pos, in_data);
  }

  // deal with the last channel of output0
  int channel_offset = out0_dims_axis % 4;
  if (channel_blk_idx == out0_dims_axis / 4) { // only the last channel_blk of output0 hits this
    if (channel_offset != 0) {
      CL_DTYPE4 out0_last_val;
      if (channel_offset == 1) {
        out0_last_val = (CL_DTYPE4)(in_data.x, 0, 0, 0);
      } else if (channel_offset == 2) {
        out0_last_val = (CL_DTYPE4)(in_data.x, in_data.y, 0, 0);
      } else if (channel_offset == 3) {
        out0_last_val = (CL_DTYPE4)(in_data.x, in_data.y, in_data.z, 0);
      }
      WRITE_IMG_TYPE(CL_DTYPE_CHAR, output0, in_pos, out0_last_val);
    }
  }

  // deal with output1
  if (c + 4 >= out0_dims_axis) { // only theads for output1 hit this
    const int2 out_pos = (int2)((channel_blk_idx - out0_dims_axis / 4) * in_dims_last + width_idx, hb_idx);
    if (channel_offset == 0) { // write all data to output1 directly
      WRITE_IMG_TYPE(CL_DTYPE_CHAR, output1, out_pos, in_data);
    } else {
      CL_DTYPE4 combined_val;
      CL_DTYPE4 latter = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_pos.x + in_dims_last, in_pos.y));
      if (channel_offset == 1) {
        combined_val = (CL_DTYPE4)(in_data.y, in_data.z, in_data.w, latter.x);
      } else if (channel_offset == 2) {
        combined_val = (CL_DTYPE4)(in_data.z, in_data.w, latter.x, latter.y);
      } else if (channel_offset == 3) {
        combined_val = (CL_DTYPE4)(in_data.w, latter.x, latter.y, latter.z);
      }
      WRITE_IMG_TYPE(CL_DTYPE_CHAR, output1, out_pos, combined_val);
    }
  }
}

__kernel void SplitHeight(__read_only image2d_t input,
                          __write_only image2d_t output0,
                          __write_only image2d_t output1,
                          __private const int axis,
                          __private const int out0_dims_axis,
                          __private const int in_dims_second,
                          __private const int in_dims_last,
                          __private const int width) {
  const int channel_blk_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);

  const int2 in_pos = (int2)(channel_blk_idx * in_dims_last + width_idx, hb_idx);
  const CL_DTYPE4 in_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, in_pos);
  const int h = hb_idx % width;

  int2 out_pos;
  if (h < out0_dims_axis) {
    out_pos = in_pos;
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output0, out_pos, in_data);
  } else {
    out_pos.y = hb_idx - out0_dims_axis;
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output1, out_pos, in_data);
  }
}

__kernel void SplitWidth(__read_only image2d_t input,
                         __write_only image2d_t output0,
                         __write_only image2d_t output1,
                         __private const int axis,
                         __private const int out0_dims_axis,
                         __private const int in_dims_second,
                         __private const int in_dims_last,
                         __private const int width) {
  const int channel_blk_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);

  const int2 in_pos = (int2)(channel_blk_idx * in_dims_last + width_idx, hb_idx);
  const CL_DTYPE4 in_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, in_pos);

  int2 out_pos;
  if (width_idx < out0_dims_axis) {
    out_pos = in_pos;
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output0, out_pos, in_data);
  } else {
    out_pos.x = width_idx - out0_dims_axis;
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output1, out_pos, in_data);
  }
}
