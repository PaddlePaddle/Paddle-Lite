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

/***************************************************************************
 * For case: Axis N/H/W or Axis C that all input channels is aligned: Start
 ***************************************************************************/
#define CHECK_IDX                                               \
  int c_blk_idx = get_global_id(0);                             \
  int w_idx = get_global_id(1);                                 \
  int nh_idx = get_global_id(2);                                \
  if (c_blk_idx >= output_shape.y || w_idx >= output_shape.w || \
      nh_idx >= output_shape.x * output_shape.z) {              \
    return;                                                     \
  }                                                             \
  CL_DTYPE4 result;

// axis = 1
#define DOConcat2InputAxis1                                            \
  int boundary0 = input_shape0.y;             /* C_blk0 */             \
  int boundary1 = boundary0 + input_shape1.y; /* C_blk0 + C_blk1 */    \
  int2 input_pos;                                                      \
  input_pos.y = nh_idx;                                                \
  if (c_blk_idx < boundary0) {                                         \
    input_pos.x = c_blk_idx * input_shape0.w + w_idx;                  \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input0, SAMPLER, input_pos); \
  } else if (c_blk_idx < boundary1) {                                  \
    input_pos.x = (c_blk_idx - boundary0) * input_shape1.w + w_idx;    \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input1, SAMPLER, input_pos); \
  }

#define WRITE_IMG_DATA                                               \
  WRITE_IMG_TYPE(CL_DTYPE_CHAR,                                      \
                 output,                                             \
                 (int2)(c_blk_idx * output_shape.w + w_idx, nh_idx), \
                 result);

#define CONCAT2(Inputnum, Axis)                                       \
  __kernel void Concat##Inputnum##Axis(__read_only image2d_t input0,  \
                                       __read_only image2d_t input1,  \
                                       __write_only image2d_t output, \
                                       int4 input_shape0,             \
                                       int4 input_shape1,             \
                                       int4 output_shape) {           \
    CHECK_IDX                                                         \
    DOConcat##Inputnum##Axis WRITE_IMG_DATA                           \
  }

// axis = 1
CONCAT2(2Input, Axis1)

__kernel void Concat2InputAxis1Unalign(__read_only image2d_t input0,
                                       __read_only image2d_t input1,
                                       __write_only image2d_t output,
                                       __private const int in0_dims_axis,
                                       __private const int out_dims_last) {
  const int channel_blk_idx = get_global_id(0);
  const int width_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);
  const int c = channel_blk_idx * 4;

  // write all input0 data to output directly
  if (c < in0_dims_axis) {
    const int2 in_pos =
        (int2)(channel_blk_idx * out_dims_last + width_idx, hb_idx);
    const CL_DTYPE4 in_data =
        READ_IMG_TYPE(CL_DTYPE_CHAR, input0, SAMPLER, in_pos);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos, in_data);
  }

  // deal with the gap
  int channel_offset = in0_dims_axis % 4;
  if (channel_blk_idx ==
      in0_dims_axis / 4) {  // only the last channel_blk of input0 hits this
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
  if (c + 4 >= in0_dims_axis) {  // only theads for output1 hit this
    const int2 out_pos = (int2)(
        (channel_blk_idx - out0_dims_axis / 4) * in_dims_last + width_idx,
        hb_idx);
    if (channel_offset == 0) {  // write all data to output1 directly
      WRITE_IMG_TYPE(CL_DTYPE_CHAR, output1, out_pos, in_data);
    } else {
      CL_DTYPE4 combined_val;
      CL_DTYPE4 latter =
          READ_IMG_TYPE(CL_DTYPE_CHAR,
                        input,
                        SAMPLER,
                        (int2)(in_pos.x + in_dims_last, in_pos.y));
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

__kernel void concat2(__read_only image2d_t input0,
                      __read_only image2d_t input1,
                      __write_only image2d_t output,
                      int flag,
                      int C_0,    // input0_axis_dims
                      int out_C,  // output_tensor_c
                      int out_W,  // output_tensor_w
                      int width) {
  const int out_w = get_global_id(0);   // image_width w
  const int out_c = get_global_id(1);   // image_width (c+3)/4
  const int out_nh = get_global_id(2);  // image_height nxh

  if (flag == 1) {  // by channel
    int c_in = out_c;
    int2 output_pos;
    output_pos.x = out_c * out_W + out_w;
    output_pos.y = out_nh;
    CL_DTYPE4 output_data;
    for (int i = 0; i < 4; i++) {
      int c = out_c * 4 + i;
      if (c >= out_C) {
        break;
      }
      int c_in;
      CL_DTYPE4 input_data;
      if (c < C_0) {
        c_in = c;
        int2 input_pos;
        input_pos.x = (c_in / 4) * out_W + out_w;
        input_pos.y = out_nh;
        input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input0, SAMPLER, input_pos);
      } else {
        c_in = c - C_0;
        int2 input_pos;
        input_pos.x = (c_in / 4) * out_W + out_w;
        input_pos.y = out_nh;
        input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input1, SAMPLER, input_pos);
      }
      int value_offset = c_in % 4;
      CL_DTYPE value;
      if (value_offset == 0) {
        value = input_data.x;
      } else if (value_offset == 1) {
        value = input_data.y;
      } else if (value_offset == 2) {
        value = input_data.z;
      } else if (value_offset == 3) {
        value = input_data.w;
      }
      if (i == 0) {
        output_data.x = value;
      } else if (i == 1) {
        output_data.y = value;
      } else if (i == 2) {
        output_data.z = value;
      } else if (i == 3) {
        output_data.w = value;
      }
    }
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, output_data);
  } else if (flag == 2) {  // by height,  width == n
    int2 input_pos;
    input_pos.x = out_c * out_W + out_w;
    int h = out_nh / width;
    CL_DTYPE4 input;
    if (h < C_0) {
      input_pos.y = out_nh;
      input = READ_IMG_TYPE(CL_DTYPE_CHAR, input0, SAMPLER, input_pos);
    } else {
      input_pos.y = (h - C_0) * width;
      input = READ_IMG_TYPE(CL_DTYPE_CHAR, input1, SAMPLER, input_pos);
    }
    int2 output_pos;
    output_pos.x = out_c * out_W + out_w;
    output_pos.y = out_nh;
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, input);
  } else if (flag == 3) {  // by width, width == C
    int2 input_pos;
    input_pos.y = out_nh;
    CL_DTYPE4 input;
    if (out_w < C_0) {
      input_pos.x = out_c * out_W + out_w;
      input = READ_IMG_TYPE(CL_DTYPE_CHAR, input0, SAMPLER, input_pos);
    } else {
      input_pos.x = out_c * out_W + (out_w - C_0);
      input = READ_IMG_TYPE(CL_DTYPE_CHAR, input1, SAMPLER, input_pos);
    }
    int2 output_pos;
    output_pos.x = out_c * out_W + out_w;
    output_pos.y = out_nh;
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, input);
  }
}
