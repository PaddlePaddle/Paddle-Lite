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

// deprecated
__kernel void concatByCWith2Inputs(
                    __write_only image2d_t output_image,
                    __private const int output_tensor_c,
                    __private const int output_tensor_w,
                    __read_only image2d_t input0_image,
                    __private const int input0_tensor_c,
                    __read_only image2d_t input1_image,
                    __private const int input1_tensor_c) {
  const int out_c = get_global_id(0);   // [0, (output_tensor_c + 3) / 4)
  const int out_w = get_global_id(1);   // [0, output_tensor_w)
  const int out_nh = get_global_id(2);  // [0, output_tensor_n * output_tensor_h)

  int2 output_pos;
  output_pos.x = out_c * output_tensor_w + out_w;
  output_pos.y = out_nh;
  CL_DTYPE4 output_data;

  for (int i = 0; i < 4; i++) {
    int c = out_c * 4 + i;
    if (c >= output_tensor_c) {
        break;
    }
    int c_in;
    CL_DTYPE4 input_data;
    if (c < input0_tensor_c) {
      c_in = c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input0_image, SAMPLER, input_pos);
    } else {
      c_in = c - input0_tensor_c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input1_image, SAMPLER, input_pos);
    }
    int value_offset = c_in % 4;
    float value;
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
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output_data);
}


__kernel void concatByCWith3Inputs(
                    __write_only image2d_t output_image,
                    __private const int output_tensor_c,
                    __private const int output_tensor_w,
                    __read_only image2d_t input0_image,
                    __private const int input0_tensor_c,
                    __read_only image2d_t input1_image,
                    __private const int input1_tensor_c,
                    __read_only image2d_t input2_image,
                    __private const int input2_tensor_c) {
  const int out_c = get_global_id(0);   // [0, (output_tensor_c + 3) / 4)
  const int out_w = get_global_id(1);   // [0, output_tensor_w)
  const int out_nh = get_global_id(2);  // [0, output_tensor_n * output_tensor_h)

  int2 output_pos;
  output_pos.x = out_c * output_tensor_w + out_w;
  output_pos.y = out_nh;
  CL_DTYPE4 output_data;

  for (int i = 0; i < 4; i++) {
    int c = out_c * 4 + i;
    if (c >= output_tensor_c) {
        break;
    }
    int c_in;
    CL_DTYPE4 input_data;
    if (c < input0_tensor_c) {
      c_in = c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input0_image, SAMPLER, input_pos);
    } else if (c < input0_tensor_c + input1_tensor_c) {
      c_in = c - input0_tensor_c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input1_image, SAMPLER, input_pos);
    } else {
      c_in = c - input0_tensor_c - input1_tensor_c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input2_image, SAMPLER, input_pos);
    }
    int value_offset = c_in % 4;
    float value;
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
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output_data);
}


__kernel void concatByCWith4Inputs(
                    __write_only image2d_t output_image,
                    __private const int output_tensor_c,
                    __private const int output_tensor_w,
                    __read_only image2d_t input0_image,
                    __private const int input0_tensor_c,
                    __read_only image2d_t input1_image,
                    __private const int input1_tensor_c,
                    __read_only image2d_t input2_image,
                    __private const int input2_tensor_c,
                    __read_only image2d_t input3_image,
                    __private const int input3_tensor_c) {
  const int out_c = get_global_id(0);   // [0, (output_tensor_c + 3) / 4)
  const int out_w = get_global_id(1);   // [0, output_tensor_w)
  const int out_nh = get_global_id(2);  // [0, output_tensor_n * output_tensor_h)

  int2 output_pos;
  output_pos.x = out_c * output_tensor_w + out_w;
  output_pos.y = out_nh;
  CL_DTYPE4 output_data;

  for (int i = 0; i < 4; i++) {
    int c = out_c * 4 + i;
    if (c >= output_tensor_c) {
        break;
    }
    int c_in;
    CL_DTYPE4 input_data;
    if (c < input0_tensor_c) {
      c_in = c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input0_image, SAMPLER, input_pos);
    } else if (c < input0_tensor_c + input1_tensor_c) {
      c_in = c - input0_tensor_c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input1_image, SAMPLER, input_pos);
    } else if (c < input0_tensor_c + input1_tensor_c + input2_tensor_c) {
      c_in = c - input0_tensor_c - input1_tensor_c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input2_image, SAMPLER, input_pos);
    }else if (c < input0_tensor_c + input1_tensor_c + input2_tensor_c + input3_tensor_c){
      c_in = c - input0_tensor_c - input1_tensor_c - input2_tensor_c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input3_image, SAMPLER, input_pos);
    }
    int value_offset = c_in % 4;
    float value;
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
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output_data);
}


// deprecated
__kernel void concatByH(__read_only image2d_t input_image,
                        __write_only image2d_t output_image,
                        __private const int out_W,
                        __private const int out_H_Start) {

  const int in_c = get_global_id(0);
  const int in_w = get_global_id(1);
  const int in_nh = get_global_id(2);

  int2 input_pos;
  input_pos.x = in_c * out_W + in_w;
  input_pos.y = in_nh;

  CL_DTYPE4 input;
  input = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER,input_pos);

  int2 output_pos;
  output_pos.x = input_pos.x;
  output_pos.y = out_H_Start + input_pos.y;

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, input);

}


// deprecated
__kernel void concatByW(__read_only image2d_t input_image,
                        __write_only image2d_t output_image,
                        __private const int in_W,
                        __private const int pre_Width,
                        __private const int out_Width) {

  const int in_c = get_global_id(0);
  const int in_w = get_global_id(1);
  const int in_nh = get_global_id(2);

  int2 input_pos;
  input_pos.x = in_c * in_W + in_w;
  input_pos.y = in_nh;

  CL_DTYPE4 input;
  input = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, SAMPLER,input_pos);

  int2 output_pos;
  output_pos.x = input_pos.x + pre_Width + out_Width * in_c;
  output_pos.y = input_pos.y;
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, input);
}


__kernel void concat2(__read_only image2d_t input0,
                      __read_only image2d_t input1,
                      __write_only image2d_t output,
                      int flag, int C_0, int out_C, int out_W, int width) {
  const int out_w = get_global_id(0); // image_width cxw/4
  const int out_c = get_global_id(1); // image_width cxw/4
  const int out_nh = get_global_id(2); // image_height nxh

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
  }else if (flag == 2){ // by height,  width == n
    int2 input_pos;
    input_pos.x = out_c * out_W + out_w;
    int h = out_nh / width;
    CL_DTYPE4 input;
    if (h < C_0){
      input_pos.y = out_nh;
      input = READ_IMG_TYPE(CL_DTYPE_CHAR, input0, SAMPLER, input_pos);
    }else{
      input_pos.y = (h - C_0) * width;
      input = READ_IMG_TYPE(CL_DTYPE_CHAR, input1, SAMPLER, input_pos);
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
      input = READ_IMG_TYPE(CL_DTYPE_CHAR, input0, SAMPLER, input_pos);
    }else{
      input_pos.x = out_c * out_W + (out_w - C_0);
      input = READ_IMG_TYPE(CL_DTYPE_CHAR, input1, SAMPLER, input_pos);
    }
    int2 output_pos;
    output_pos.x = out_c * out_W + out_w;
    output_pos.y = out_nh;
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, input);
  }
}
