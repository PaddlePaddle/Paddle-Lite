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

__kernel void elementwise_compute(__read_only image2d_t input,
                                  __read_only image2d_t bias,
                                  __write_only image2d_t outputImage) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int2 coords;
  coords.x = x;
  coords.y = y;

  CL_DTYPE4 in_x = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, coords);
  CL_DTYPE4 in_y = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, coords);

  CL_DTYPE4 output = OPERATOR(in_x, in_y);
#ifdef FUSE_SCALE
  output = fuse_scale(output, SCALE_SLOPE, SCALE_BIAS, SCALE_ALPHA);
#endif

#if defined(RELU) || defined(RELU6) || defined(GELU)
  CL_DTYPE4 alpha;
  output = activation_type4(output, alpha);
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}

__kernel void broadcast_elementwise_basic(
    __read_only image2d_t input_x,
    __read_only image2d_t input_y,
    __write_only image2d_t output_image,
    __private const int4 input_nhwc4,
    __private const int4 bias_nhwc4,
    __private const int4 output_nhwc4,
    __private const int inputx_broadcast_c_flag,
    __private const int inputy_broadcast_c_flag,
    __private const int image_folder_flag_x,
    __private const int image_folder_flag_y,
    __private const int bias_width) {
  int idwc4 = get_global_id(0);
  int idbh = get_global_id(1);

  if (idwc4 >= output_nhwc4.w * output_nhwc4.z ||
      idbh >= output_nhwc4.x * output_nhwc4.y) {
    return;
  }

  int4 id_shape;
  id_shape.w = idwc4 / output_nhwc4.z;  // c4
  id_shape.z = idwc4 % output_nhwc4.z;  // w
  id_shape.y = idbh % output_nhwc4.y;   // h
  id_shape.x = idbh / output_nhwc4.y;   // n

  int4 v_zero = (int4)(0);
  int4 flag_v = (int4)(0);
  flag_v = isless(convert_float4(id_shape), convert_float4(input_nhwc4));
  int4 idx_shape = select(v_zero, id_shape, flag_v);

  int2 cur_index = (int2)(idx_shape.w * input_nhwc4.z + idx_shape.z,
                          idx_shape.x * input_nhwc4.y + idx_shape.y);
  CL_DTYPE4 in_x = READ_IMG_TYPE(CL_DTYPE_CHAR, input_x, SAMPLER, cur_index);

  /***************************get y data*******************************/
  flag_v = isless(convert_float4(id_shape), convert_float4(bias_nhwc4));
  int4 idy_shape = select(v_zero, id_shape, flag_v);

  cur_index = (int2)(idy_shape.w * bias_nhwc4.z + idy_shape.z,
                     idy_shape.x * bias_nhwc4.y + idy_shape.y);
  CL_DTYPE4 in_y = READ_IMG_TYPE(CL_DTYPE_CHAR, input_y, SAMPLER, cur_index);

  in_x = SELECT(in_x, (CL_DTYPE4)(in_x.x), inputx_broadcast_c_flag);
  in_y = SELECT(in_y, (CL_DTYPE4)(in_y.x), inputy_broadcast_c_flag);

  CL_DTYPE4 output = OPERATOR(in_x, in_y);
#ifdef FUSE_SCALE
  output = fuse_scale(output, SCALE_SLOPE, SCALE_BIAS, SCALE_ALPHA);
#endif

#if defined(RELU) || defined(RELU6) || defined(GELU)
  CL_DTYPE4 alpha;
  output = activation_type4(output, alpha);
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, (int2)(idwc4, idbh), output);
}
