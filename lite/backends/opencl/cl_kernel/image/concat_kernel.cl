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

// axis = 0
// Calling enqueueCopyImage directly is also OK but may be slower than kernel
// impl.
#define DOConcat2InputAxis0                                            \
  int n_idx = nh_idx / output_shape.z;                                 \
  int h_idx = nh_idx % output_shape.z;                                 \
  int boundary0 = input_shape0.x;             /* N0 */                 \
  int boundary1 = boundary0 + input_shape1.x; /* N0 + N1 */            \
  int2 input_pos;                                                      \
  input_pos.x = c_blk_idx * input_shape0.w + w_idx;                    \
  if (n_idx < boundary0) {                                             \
    input_pos.y = n_idx * input_shape0.z + h_idx;                      \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input0, SAMPLER, input_pos); \
  } else if (n_idx < boundary1) {                                      \
    input_pos.y = (n_idx - boundary0) * input_shape1.z + h_idx;        \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input1, SAMPLER, input_pos); \
  }

#define DOConcat3InputAxis0                                            \
  DOConcat2InputAxis0;                                                 \
  int boundary2 = boundary1 + input_shape2.x;                          \
  if (n_idx >= boundary1 && n_idx < boundary2) {                       \
    input_pos.y = (n_idx - boundary1) * input_shape2.z + h_idx;        \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input2, SAMPLER, input_pos); \
  }

#define DOConcat4InputAxis0                                            \
  DOConcat3InputAxis0;                                                 \
  int boundary3 = boundary2 + input_shape3.x;                          \
  if (n_idx >= boundary2 && n_idx < boundary3) {                       \
    input_pos.y = (n_idx - boundary2) * input_shape3.z + h_idx;        \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input3, SAMPLER, input_pos); \
  }

#define DOConcat5InputAxis0                                            \
  DOConcat4InputAxis0;                                                 \
  int boundary4 = boundary3 + input_shape4.x;                          \
  if (n_idx >= boundary3 && n_idx < boundary4) {                       \
    input_pos.y = (n_idx - boundary3) * input_shape4.z + h_idx;        \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input4, SAMPLER, input_pos); \
  }

#define DOConcat6InputAxis0                                            \
  DOConcat5InputAxis0;                                                 \
  int boundary5 = boundary4 + input_shape5.x;                          \
  if (n_idx >= boundary4 && n_idx < boundary5) {                       \
    input_pos.y = (n_idx - boundary4) * input_shape5.z + h_idx;        \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input5, SAMPLER, input_pos); \
  }

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

#define DOConcat3InputAxis1                                            \
  DOConcat2InputAxis1;                                                 \
  int boundary2 = boundary1 + input_shape2.y;                          \
  if (c_blk_idx >= boundary1 && c_blk_idx < boundary2) {               \
    input_pos.x = (c_blk_idx - boundary1) * input_shape2.w + w_idx;    \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input2, SAMPLER, input_pos); \
  }

#define DOConcat4InputAxis1                                            \
  DOConcat3InputAxis1;                                                 \
  int boundary3 = boundary2 + input_shape3.y;                          \
  if (c_blk_idx >= boundary2 && c_blk_idx < boundary3) {               \
    input_pos.x = (c_blk_idx - boundary2) * input_shape3.w + w_idx;    \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input3, SAMPLER, input_pos); \
  }

#define DOConcat5InputAxis1                                            \
  DOConcat4InputAxis1;                                                 \
  int boundary4 = boundary3 + input_shape4.y;                          \
  if (c_blk_idx >= boundary3 && c_blk_idx < boundary4) {               \
    input_pos.x = (c_blk_idx - boundary3) * input_shape4.w + w_idx;    \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input4, SAMPLER, input_pos); \
  }

#define DOConcat6InputAxis1                                            \
  DOConcat5InputAxis1;                                                 \
  int boundary5 = boundary4 + input_shape5.y;                          \
  if (c_blk_idx >= boundary4 && c_blk_idx < boundary5) {               \
    input_pos.x = (c_blk_idx - boundary4) * input_shape5.w + w_idx;    \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input5, SAMPLER, input_pos); \
  }

// axis = 2
#define DOConcat2InputAxis2                                            \
  int n_idx = nh_idx / output_shape.z;                                 \
  int h_idx = nh_idx % output_shape.z;                                 \
  int boundary0 = input_shape0.z;             /* H0 */                 \
  int boundary1 = boundary0 + input_shape1.z; /* H0 + H1 */            \
  int2 input_pos;                                                      \
  input_pos.x = c_blk_idx * input_shape0.w + w_idx;                    \
  if (h_idx < boundary0) {                                             \
    input_pos.y = n_idx * input_shape0.z + h_idx;                      \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input0, SAMPLER, input_pos); \
  } else if (h_idx < boundary1) {                                      \
    input_pos.y = n_idx * input_shape1.z + h_idx - boundary0;          \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input1, SAMPLER, input_pos); \
  }

#define DOConcat3InputAxis2                                            \
  DOConcat2InputAxis2;                                                 \
  int boundary2 = boundary1 + input_shape2.z;                          \
  if (h_idx >= boundary1 && h_idx < boundary2) {                       \
    input_pos.y = n_idx * input_shape2.z + h_idx - boundary1;          \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input2, SAMPLER, input_pos); \
  }

#define DOConcat4InputAxis2                                            \
  DOConcat3InputAxis2;                                                 \
  int boundary3 = boundary2 + input_shape3.z;                          \
  if (h_idx >= boundary2 && h_idx < boundary3) {                       \
    input_pos.y = n_idx * input_shape3.z + h_idx - boundary2;          \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input3, SAMPLER, input_pos); \
  }

#define DOConcat5InputAxis2                                            \
  DOConcat4InputAxis2;                                                 \
  int boundary4 = boundary3 + input_shape4.z;                          \
  if (h_idx >= boundary3 && h_idx < boundary4) {                       \
    input_pos.y = n_idx * input_shape4.z + h_idx - boundary3;          \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input4, SAMPLER, input_pos); \
  }

#define DOConcat6InputAxis2                                            \
  DOConcat5InputAxis2;                                                 \
  int boundary5 = boundary4 + input_shape5.z;                          \
  if (h_idx >= boundary4 && h_idx < boundary5) {                       \
    input_pos.y = n_idx * input_shape5.z + h_idx - boundary4;          \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input5, SAMPLER, input_pos); \
  }

// axis = 3
#define DOConcat2InputAxis3                                            \
  int boundary0 = input_shape0.w;             /* W0 */                 \
  int boundary1 = boundary0 + input_shape1.w; /* W0 + W1 */            \
  int2 input_pos;                                                      \
  input_pos.y = nh_idx;                                                \
  if (w_idx < boundary0) {                                             \
    input_pos.x = c_blk_idx * input_shape0.w + w_idx;                  \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input0, SAMPLER, input_pos); \
  } else if (w_idx < boundary1) {                                      \
    input_pos.x = c_blk_idx * input_shape1.w + w_idx - boundary0;      \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input1, SAMPLER, input_pos); \
  }

#define DOConcat3InputAxis3                                            \
  DOConcat2InputAxis3;                                                 \
  int boundary2 = boundary1 + input_shape2.w;                          \
  if (w_idx >= boundary1 && w_idx < boundary2) {                       \
    input_pos.x = c_blk_idx * input_shape2.w + w_idx - boundary1;      \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input2, SAMPLER, input_pos); \
  }

#define DOConcat4InputAxis3                                            \
  DOConcat3InputAxis3;                                                 \
  int boundary3 = boundary2 + input_shape3.w;                          \
  if (w_idx >= boundary2 && w_idx < boundary3) {                       \
    input_pos.x = c_blk_idx * input_shape3.w + w_idx - boundary2;      \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input3, SAMPLER, input_pos); \
  }

#define DOConcat5InputAxis3                                            \
  DOConcat4InputAxis3;                                                 \
  int boundary4 = boundary3 + input_shape4.w;                          \
  if (w_idx >= boundary3 && w_idx < boundary4) {                       \
    input_pos.x = c_blk_idx * input_shape4.w + w_idx - boundary3;      \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input4, SAMPLER, input_pos); \
  }

#define DOConcat6InputAxis3                                            \
  DOConcat5InputAxis3;                                                 \
  int boundary5 = boundary4 + input_shape5.w;                          \
  if (w_idx >= boundary4 && w_idx < boundary5) {                       \
    input_pos.x = c_blk_idx * input_shape5.w + w_idx - boundary4;      \
    result = READ_IMG_TYPE(CL_DTYPE_CHAR, input5, SAMPLER, input_pos); \
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

#define CONCAT3(Inputnum, Axis)                                       \
  __kernel void Concat##Inputnum##Axis(__read_only image2d_t input0,  \
                                       __read_only image2d_t input1,  \
                                       __read_only image2d_t input2,  \
                                       __write_only image2d_t output, \
                                       int4 input_shape0,             \
                                       int4 input_shape1,             \
                                       int4 input_shape2,             \
                                       int4 output_shape) {           \
    CHECK_IDX                                                         \
    DOConcat##Inputnum##Axis WRITE_IMG_DATA                           \
  }

#define CONCAT4(Inputnum, Axis)                                       \
  __kernel void Concat##Inputnum##Axis(__read_only image2d_t input0,  \
                                       __read_only image2d_t input1,  \
                                       __read_only image2d_t input2,  \
                                       __read_only image2d_t input3,  \
                                       __write_only image2d_t output, \
                                       int4 input_shape0,             \
                                       int4 input_shape1,             \
                                       int4 input_shape2,             \
                                       int4 input_shape3,             \
                                       int4 output_shape) {           \
    CHECK_IDX                                                         \
    DOConcat##Inputnum##Axis WRITE_IMG_DATA                           \
  }

#define CONCAT5(Inputnum, Axis)                                       \
  __kernel void Concat##Inputnum##Axis(__read_only image2d_t input0,  \
                                       __read_only image2d_t input1,  \
                                       __read_only image2d_t input2,  \
                                       __read_only image2d_t input3,  \
                                       __read_only image2d_t input4,  \
                                       __write_only image2d_t output, \
                                       int4 input_shape0,             \
                                       int4 input_shape1,             \
                                       int4 input_shape2,             \
                                       int4 input_shape3,             \
                                       int4 input_shape4,             \
                                       int4 output_shape) {           \
    CHECK_IDX                                                         \
    DOConcat##Inputnum##Axis WRITE_IMG_DATA                           \
  }

#define CONCAT6(Inputnum, Axis)                                       \
  __kernel void Concat##Inputnum##Axis(__read_only image2d_t input0,  \
                                       __read_only image2d_t input1,  \
                                       __read_only image2d_t input2,  \
                                       __read_only image2d_t input3,  \
                                       __read_only image2d_t input4,  \
                                       __read_only image2d_t input5,  \
                                       __write_only image2d_t output, \
                                       int4 input_shape0,             \
                                       int4 input_shape1,             \
                                       int4 input_shape2,             \
                                       int4 input_shape3,             \
                                       int4 input_shape4,             \
                                       int4 input_shape5,             \
                                       int4 output_shape) {           \
    CHECK_IDX                                                         \
    DOConcat##Inputnum##Axis WRITE_IMG_DATA                           \
  }

// axis = 0
CONCAT2(2Input, Axis0)
CONCAT3(3Input, Axis0)
CONCAT4(4Input, Axis0)
CONCAT5(5Input, Axis0)
CONCAT6(6Input, Axis0)
// axis = 1
CONCAT3(3Input, Axis1)
CONCAT4(4Input, Axis1)
CONCAT5(5Input, Axis1)
CONCAT6(6Input, Axis1)
// axis = 2
CONCAT2(2Input, Axis2)
CONCAT3(3Input, Axis2)
CONCAT4(4Input, Axis2)
CONCAT5(5Input, Axis2)
CONCAT6(6Input, Axis2)
// axis = 3
CONCAT2(2Input, Axis3)
CONCAT3(3Input, Axis3)
CONCAT4(4Input, Axis3)
CONCAT5(5Input, Axis3)
CONCAT6(6Input, Axis3)
/*************************************************************************
 * For case: Axis N/H/W or Axis C that all input channels is aligned: End
 *************************************************************************/

__kernel void concatByCWith3Inputs(__write_only image2d_t output_image,
                                   __private const int output_tensor_c,
                                   __private const int output_tensor_w,
                                   __read_only image2d_t input0_image,
                                   __private const int input0_tensor_c,
                                   __read_only image2d_t input1_image,
                                   __private const int input1_tensor_c,
                                   __read_only image2d_t input2_image,
                                   __private const int input2_tensor_c) {
  const int out_c = get_global_id(0);  // [0, (output_tensor_c + 3) / 4)
  const int out_w = get_global_id(1);  // [0, output_tensor_w)
  const int out_nh =
      get_global_id(2);  // [0, output_tensor_n * output_tensor_h)

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
      input_data =
          READ_IMG_TYPE(CL_DTYPE_CHAR, input0_image, SAMPLER, input_pos);
    } else if (c < input0_tensor_c + input1_tensor_c) {
      c_in = c - input0_tensor_c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data =
          READ_IMG_TYPE(CL_DTYPE_CHAR, input1_image, SAMPLER, input_pos);
    } else {
      c_in = c - input0_tensor_c - input1_tensor_c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data =
          READ_IMG_TYPE(CL_DTYPE_CHAR, input2_image, SAMPLER, input_pos);
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

__kernel void concatByCWith4Inputs(__write_only image2d_t output_image,
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
  const int out_c = get_global_id(0);  // [0, (output_tensor_c + 3) / 4)
  const int out_w = get_global_id(1);  // [0, output_tensor_w)
  const int out_nh =
      get_global_id(2);  // [0, output_tensor_n * output_tensor_h)

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
      input_data =
          READ_IMG_TYPE(CL_DTYPE_CHAR, input0_image, SAMPLER, input_pos);
    } else if (c < input0_tensor_c + input1_tensor_c) {
      c_in = c - input0_tensor_c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data =
          READ_IMG_TYPE(CL_DTYPE_CHAR, input1_image, SAMPLER, input_pos);
    } else if (c < input0_tensor_c + input1_tensor_c + input2_tensor_c) {
      c_in = c - input0_tensor_c - input1_tensor_c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data =
          READ_IMG_TYPE(CL_DTYPE_CHAR, input2_image, SAMPLER, input_pos);
    } else if (c < input0_tensor_c + input1_tensor_c + input2_tensor_c +
                       input3_tensor_c) {
      c_in = c - input0_tensor_c - input1_tensor_c - input2_tensor_c;
      int2 input_pos;
      input_pos.x = (c_in / 4) * output_tensor_w + out_w;
      input_pos.y = out_nh;
      input_data =
          READ_IMG_TYPE(CL_DTYPE_CHAR, input3_image, SAMPLER, input_pos);
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

__kernel void concat2(__read_only image2d_t input0,
                      __read_only image2d_t input1,
                      __write_only image2d_t output,
                      int flag,
                      int C_0,
                      int out_C,
                      int out_W,
                      int width) {
  const int out_w = get_global_id(0);   // image_width cxw/4
  const int out_c = get_global_id(1);   // image_width cxw/4
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
