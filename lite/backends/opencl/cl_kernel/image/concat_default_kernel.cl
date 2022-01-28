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

__kernel void Concat2InputAxis1Common(__read_only image2d_t input0,
                                      __read_only image2d_t input1,
                                      __write_only image2d_t output,
                                      __private const int in0_dims_axis,
                                      __private const int out_dims_last) {
  const int width_idx = get_global_id(0);
  const int channel_blk_idx = get_global_id(1);
  const int hb_idx = get_global_id(2);
  const int c = channel_blk_idx * 4;

  const int2 pos = (int2)(channel_blk_idx * out_dims_last + width_idx, hb_idx);
  CL_DTYPE4 in0_data = (CL_DTYPE4)0;

  // write all input0 data to output directly
  if (c < in0_dims_axis) {
    in0_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input0, SAMPLER, pos);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, pos, in0_data);
  }

  // deal with output1
  int channel_remain = in0_dims_axis % 4;
  if (c + channel_remain >= in0_dims_axis) {
    // only theads for output1 hit this
    const int2 in1_pos = (int2)(
        (channel_blk_idx - in0_dims_axis / 4) * out_dims_last + width_idx,
        hb_idx);
    CL_DTYPE4 in1_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input1, SAMPLER, in1_pos);
    if (channel_remain == 0) {
      // write all input1 data to output directly
      WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, pos, in1_data);
    } else {
      CL_DTYPE4 remain, combined_val;
      if (in1_pos.x - out_dims_last < 0) {
        // combine input0 & input1
        remain = in0_data;
        if (channel_remain == 1) {
          combined_val =
              (CL_DTYPE4)(remain.x, in1_data.x, in1_data.y, in1_data.z);
        } else if (channel_remain == 2) {
          combined_val =
              (CL_DTYPE4)(remain.x, remain.y, in1_data.x, in1_data.y);
        } else if (channel_remain == 3) {
          combined_val = (CL_DTYPE4)(remain.x, remain.y, remain.z, in1_data.x);
        }
      } else {
        // only deal with input1
        remain = READ_IMG_TYPE(CL_DTYPE_CHAR,
                               input1,
                               SAMPLER,
                               (int2)(in1_pos.x - out_dims_last, in1_pos.y));
        if (channel_remain == 1) {
          combined_val =
              (CL_DTYPE4)(remain.w, in1_data.x, in1_data.y, in1_data.z);
        } else if (channel_remain == 2) {
          combined_val =
              (CL_DTYPE4)(remain.z, remain.w, in1_data.x, in1_data.y);
        } else if (channel_remain == 3) {
          combined_val = (CL_DTYPE4)(remain.y, remain.z, remain.w, in1_data.x);
        }
      }
      WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, pos, combined_val);
    }
  }
}
