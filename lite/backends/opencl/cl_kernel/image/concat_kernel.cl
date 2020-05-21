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

__kernel void concat2(__read_only image2d_t input0,
                      __read_only image2d_t input1,
                      __write_only image2d_t output,
                      int flag, int C_0, int out_C, int out_W, int width) {
  const int out_w = get_global_id(0); // image_width cxw/4
  const int out_c = get_global_id(1); // image_width cxw/4
  const int out_nh = get_global_id(2); // image_height nxh

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;
  if (flag == 1){ // by channel
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
        input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input0, sampler, input_pos);
      } else {
        c_in = c - C_0;
        int2 input_pos;
        input_pos.x = (c_in / 4) * out_W + out_w;
        input_pos.y = out_nh;
        input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input1, sampler, input_pos);
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
  }else if (flag == 2){ // by height,  width == n
    int2 input_pos;
    input_pos.x = out_c * out_W + out_w;
    int h = out_nh / width;
    CL_DTYPE4 input;
    if (h < C_0){
      input_pos.y = out_nh;
      input = READ_IMG_TYPE(CL_DTYPE_CHAR, input0, sampler, input_pos);
    }else{
      input_pos.y = (h - C_0) * width;
      input = READ_IMG_TYPE(CL_DTYPE_CHAR, input1, sampler, input_pos);
    }
    int2 output_pos;
    output_pos.x = out_c * out_W + out_w;
    output_pos.y = out_nh;
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, input);
  }else if (flag == 3){ // by width, width == C
    int2 input_pos;
    input_pos.y = out_nh;
    CL_DTYPE4 input;
    if (out_w < C_0){
      input_pos.x = out_c * out_W + out_w;
      input = READ_IMG_TYPE(CL_DTYPE_CHAR, input0, sampler, input_pos);
    }else{
      input_pos.x = out_c * out_W + (out_w - C_0);
      input = READ_IMG_TYPE(CL_DTYPE_CHAR, input1, sampler, input_pos);
    }
    int2 output_pos;
    output_pos.x = out_c * out_W + out_w;
    output_pos.y = out_nh;
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, input);
  }
}

__kernel void concat_mul(__read_only image2d_t input,
                         __write_only image2d_t output,
                         int flag, int C_0, int out_C, int out_W, int in_W, int width) {
  const int in_w = get_global_id(0); // image_width cxw/4
  const int in_c = get_global_id(1); // image_width cxw/4
  const int in_nh = get_global_id(2); // image_height nxh

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;
  int2 input_pos;
  int2 output_pos;
  input_pos.x = in_c * in_W + in_w;
  input_pos.y = in_nh;
  CL_DTYPE4 input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, input_pos);
  if (flag == 1){ // by channel
    CL_DTYPE4 output_data;
    for (int i = 0; i < 4; i++) {
      int c_out = C_0 + in_c * 4 + i;
      if (c_out >= out_C) {
        break;
      }
      int2 output_pos;
      output_pos.x = (c_out / 4) * in_W + in_w;
      output_pos.y = in_nh;
      CL_DTYPE val;
      if (i == 0) {
        val = input_data.x;
      } else if (i == 1) {
        val = input_data.y;
      } else if (i == 2) {
        val = input_data.z;
      } else if (i == 3) {
        val = input_data.w;
      }
      if (c_out % 4 == 0){
        output_data.x = val;
      }else if (c_out % 4 == 1){
        output_data.y = val;
      }else if (c_out % 4 == 2){
        output_data.z = val;
      }else if (c_out % 4 == 3){
        output_data.w = val;
      }
      WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, output_data);
    }
  }else if (flag == 2){ // by height, width == n
    int2 output_pos;
    output_pos.x = in_c * in_W + in_w;
    output_pos.y = in_nh + C_0 * width;
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, input_data);
  }else if (flag == 3){ // by width, width == C
    int2 output_pos;
    output_pos.y = in_nh;
    output_pos.x = in_c * out_W + (in_w + C_0);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, input_data);
  }
}
