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
                    __private const int out_C,
                    __private const int out_W,
                    __read_only image2d_t input_image_0,
                    __private const int C_0,
                    __read_only image2d_t input_image_1,
                    __private const int C_1) {
		    
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
        CLK_ADDRESS_CLAMP |
        CLK_FILTER_NEAREST;

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
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image_0, sampler, input_pos);
    } else {
      c_in = c - C_0;
      int2 input_pos;
      input_pos.x = (c_in / 4) * out_W + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image_1, sampler, input_pos);
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
__kernel void concatByCWith3Inputs(
                    __write_only image2d_t output_image,
                    __private const int out_C,
                    __private const int out_W,
                    __read_only image2d_t input_image_0,
                    __private const int C_0,
                    __read_only image2d_t input_image_1,
                    __private const int C_1,
                    __read_only image2d_t input_image_2,
                    __private const int C_2) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
        CLK_ADDRESS_CLAMP |
        CLK_FILTER_NEAREST;

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
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image_0, sampler, input_pos);
    } else if (c < C_0 + C_1) {
      c_in = c - C_0;
      int2 input_pos;
      input_pos.x = (c_in / 4) * out_W + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image_1, sampler, input_pos);
    } else {
      c_in = c - C_0 - C_1;
      int2 input_pos;
      input_pos.x = (c_in / 4) * out_W + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image_2, sampler, input_pos);
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
__kernel void concatByCWith4Inputs(
                    __write_only image2d_t output_image,
                    __private const int out_C,
                    __private const int out_W,
                    __read_only image2d_t input_image_0,
                    __private const int C_0,
                    __read_only image2d_t input_image_1,
                    __private const int C_1,
                    __read_only image2d_t input_image_2,
                    __private const int C_2,
                    __read_only image2d_t input_image_3,
                    __private const int C_3) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
        CLK_ADDRESS_CLAMP |
        CLK_FILTER_NEAREST;

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
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image_0, sampler, input_pos);
    } else if (c < C_0 + C_1) {
      c_in = c - C_0;
      int2 input_pos;
      input_pos.x = (c_in / 4) * out_W + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image_1, sampler, input_pos);
    } else if (c < C_0 + C_1 + C_2) {
      c_in = c - C_0 - C_1;
      int2 input_pos;
      input_pos.x = (c_in / 4) * out_W + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image_2, sampler, input_pos);
    }else if (c < C_0 + C_1 + C_2 + C_3){
      c_in = c - C_0 - C_1 - C_2;
      int2 input_pos;
      input_pos.x = (c_in / 4) * out_W + out_w;
      input_pos.y = out_nh;
      input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image_3, sampler, input_pos);
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

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
        CLK_ADDRESS_CLAMP |
        CLK_FILTER_NEAREST;
  CL_DTYPE4 input;
  input = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,input_pos);

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

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
        CLK_ADDRESS_CLAMP |
        CLK_FILTER_NEAREST;
  CL_DTYPE4 input;
  input = READ_IMG_TYPE(CL_DTYPE_CHAR, input_image, sampler,input_pos);

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


// #define DEBUG_CONCAT_MUL
__kernel void concat_mul(__read_only image2d_t input,
                         __write_only image2d_t output,
                         __read_only image2d_t output_r,
                         int flag, 
                         int cur_axis_start_idx,
                         int output_tensor_c,
                         int output_tensor_w,
                         int input_tensor_w,
                         int width) {
  const int in_w = get_global_id(0);   // [0, input_tensor_w)           // image_width cxw/4
  const int in_c = get_global_id(1);   // [0, (input_tensor_c+3) / 4)   // image_width cxw/4
  const int in_nh = get_global_id(2);  // [0, input_image_h)            // image_height nxh

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                            CLK_ADDRESS_CLAMP |
                            CLK_FILTER_NEAREST;
  int2 input_pos;
  int2 output_pos;
  input_pos.x = in_c * input_tensor_w + in_w;
  input_pos.y = in_nh;
  CL_DTYPE4 input_data = READ_IMG_TYPE(CL_DTYPE_CHAR, input, sampler, input_pos);

  if (flag == 1){ // concat by channel, example: 3 inputs, each is [1,3,15,15], output is [1,9,15,15]
    CL_DTYPE4 output_data;
    for (int i = 0; i < 4; i++) {
      int c_out = cur_axis_start_idx + in_c * 4 + i;
      if (c_out >= output_tensor_c) {
        break;
      }

      int2 output_pos;
      output_pos.x = (c_out / 4) * input_tensor_w + in_w;
      output_pos.y = in_nh;

      CL_DTYPE val;
      val = (i == 0) ? input_data.x : val;
      val = (i == 1) ? input_data.y : val;
      val = (i == 2) ? input_data.z : val;
      val = (i == 3) ? input_data.w : val;

#ifdef DEBUG_CONCAT_MUL //  used for debug
      CL_DTYPE4 output_data_before_post = output_data;
#endif

      // note1: fix overwrite 0 on previous elements, check if previous output exists value
      CL_DTYPE4 output_data_read = READ_IMG_TYPE(CL_DTYPE_CHAR, output_r, sampler, output_pos);
      CL_DTYPE4 fabs_output_data_read = fabs(output_data_read);
      output_data.x = (fabs_output_data_read.x < 1e-5) ? output_data.x : output_data_read.x;
      output_data.y = (fabs_output_data_read.y < 1e-5) ? output_data.y : output_data_read.y;
      output_data.z = (fabs_output_data_read.z < 1e-5) ? output_data.z : output_data_read.z;
      output_data.w = (fabs_output_data_read.w < 1e-5) ? output_data.w : output_data_read.w;

      // note2: fix error write for elements after 0 value
      if (fabs(output_data.x) < 1e-5) {
        output_data.y = 0;
	output_data.z = 0;
	output_data.w = 0;
      } else if (fabs(output_data.y) < 1e-5) {
        output_data.z = 0;
	output_data.w = 0;
      } else if (fabs(output_data.z) < 1e-5) {
        output_data.w = 0;
      }

      // note3: set val to output_data, if set previous before note1 may fail to write value to output_data, strange!
      output_data.x = (c_out % 4 == 0) ? val : output_data.x;
      output_data.y = (c_out % 4 == 1) ? val : output_data.y;
      output_data.z = (c_out % 4 == 2) ? val : output_data.z;
      output_data.w = (c_out % 4 == 3) ? val : output_data.w;

#ifdef DEBUG_CONCAT_MUL //  used for debug
  if (output_pos.x == 29 && output_pos.y == 14) {
      printf("->>>>>>>>>>>>>>> cur_axis_start_idx:%d, i:%d, output_data_before_post:(%d,%d): %.0f,%.0f,%.0f,%.0f, output_data:%.0f,%.0f,%.0f,%.0f, output_read:%.0f,%.0f,%.0f,%.0f, fabs_output_read:%.0f,%.0f,%.0f,%.0f, input(%d,%d): %.0f,%.0f,%.0f,%.0f, val:%.0f, c_out:%d, c_cout\%4:%d\n",
             cur_axis_start_idx, i,
             (int)output_pos.x, (int)output_pos.y,
             (float)output_data_before_post.x, (float)output_data_before_post.y, (float)output_data_before_post.z, (float)output_data_before_post.w,
             (float)output_data.x, (float)output_data.y, (float)output_data.z, (float)output_data.w,
             (float)output_data_read.x, (float)output_data_read.y, (float)output_data_read.z, (float)output_data_read.w,
             (float)fabs_output_data_read.x, (float)fabs_output_data_read.y, (float)fabs_output_data_read.z, (float)fabs_output_data_read.w,

	     (int)input_pos.x, (int)input_pos.y,
	     (float)input_data.x, (float)input_data.y, (float)input_data.z, (float)input_data.w,
	     (float)val,
	     (int)c_out, (int)(c_out % 4));
  }
#endif

      if (output_pos.x == 0 && output_pos.y == 0) {
        printf("<<<<<<<<<<<<<< i:%d, input(%d,%d):%.0f,%.0f,%.0f,%.0f, output_data(%d,%d):%.0f,%.0f,%.0f,%.0f\n",
	(int)i,
        (int)input_pos.x, (int)input_pos.y, (float)input_data.x, (float)input_data.y, (float)input_data.z, (float)input_data.w,
        (int)output_pos.x, (int)output_pos.y, (float)output_data.x, (float)output_data.y, (float)output_data.z, (float)output_data.w);
      }
      WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, output_data);
    }
  }else if (flag == 2){ // by height, width == n
    int2 output_pos;
    output_pos.x = in_c * input_tensor_w + in_w;
    output_pos.y = in_nh + cur_axis_start_idx * width;
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, input_data);
  }else if (flag == 3){ // by width, width == C
    int2 output_pos;
    output_pos.y = in_nh;
    output_pos.x = in_c * output_tensor_w + (in_w + cur_axis_start_idx);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, output_pos, input_data);
  }
}
