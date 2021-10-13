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

#ifdef RELU
  CL_DTYPE4 alpha;
  output = activation_type4(output, alpha);
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, outputImage, coords, output);
}

__kernel void broadcast_compute(__read_only image2d_t input_x,
                                __read_only image2d_t input_y,
                                __write_only image2d_t output_image,
                                __private const int4 input_nhwc4,
                                __private const int4 bias_nhwc4,
                                __private const int4 output_nhwc4,
                                __private const int inputx_broadcast_c_flag,
                                __private const int inputy_broadcast_c_flag,
                                __private const int image_folder_flag,
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

  flag_v = isless(convert_float4(id_shape), convert_float4(bias_nhwc4));
  int4 idy_shape = select(v_zero, id_shape, flag_v);

  cur_index = (int2)(idy_shape.w * bias_nhwc4.z + idy_shape.z,
                     idy_shape.x * bias_nhwc4.y + idy_shape.y);
  CL_DTYPE4 in_y = READ_IMG_TYPE(CL_DTYPE_CHAR, input_y, SAMPLER, cur_index);

  //w -> n
  if (image_folder_flag == 1){
    CL_DTYPE4 in0 = 0.f;
    CL_DTYPE4 in1 = 0.f;
    CL_DTYPE4 in2 = 0.f;
    CL_DTYPE4 in3 = 0.f;
  
    in0 = READ_IMG_TYPE(CL_DTYPE_CHAR, input_y, SAMPLER, (int2)(cur_index.x * 4, cur_index.y));

    if (cur_index.x * 4 + 1 < bias_width) {
      in1 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input_y, SAMPLER, (int2)(cur_index.x * 4 + 1, cur_index.y));
    }
    if (cur_index.x * 4 + 2 < bias_width) {
      in2 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input_y, SAMPLER, (int2)(cur_index.x * 4 + 2, cur_index.y));
    }
    if (cur_index.x * 4 + 3 < bias_width) {
      in3 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input_y, SAMPLER, (int2)(cur_index.x * 4 + 3, cur_index.y));
    }

    // if (cur_index.x == 0 && cur_index.y == 0){
    //   printf("in in_y: %f %f %f %f\n", in_y.x, in_y.y, in_y.z, in_y.w);
    //   printf("in0.x: %f\n", in0.x);
    //   printf("in1.x: %f\n", in1.x);
    //   printf("in2.x: %f\n", in2.x);
    //   printf("in3.x: %f\n", in3.x);
    // }

    in_y = (CL_DTYPE4)(in0.x, in1.x, in2.x, in3.x);
  }

  //w -> h
  if (image_folder_flag == 2) {
    in_y = READ_IMG_TYPE(CL_DTYPE_CHAR, input_y, SAMPLER, (int2)(cur_index.y, cur_index.x));
  }

  //hw -> ch
  if (image_folder_flag == 3){
    CL_DTYPE4 in0 = 0.f;
    CL_DTYPE4 in1 = 0.f;
    CL_DTYPE4 in2 = 0.f;
    CL_DTYPE4 in3 = 0.f;
  
    in0 = READ_IMG_TYPE(CL_DTYPE_CHAR, input_y, SAMPLER, (int2)(cur_index.y, cur_index.x * 4));

    if (cur_index.x * 4 + 1 < bias_nhwc4.z * bias_nhwc4.w * 4) {
      in1 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input_y, SAMPLER, (int2)(cur_index.y, cur_index.x * 4 + 1));
    }
    if (cur_index.x * 4 + 2 < bias_nhwc4.z * bias_nhwc4.w * 4) {
      in2 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input_y, SAMPLER, (int2)(cur_index.y, cur_index.x * 4 + 2));
    }
    if (cur_index.x * 4 + 3 < bias_nhwc4.z * bias_nhwc4.w * 4) {
      in3 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input_y, SAMPLER, (int2)(cur_index.y, cur_index.x * 4 + 3));
    }

    // if (idwc4 == 1 && idbh == 2){
    //   printf("3333in in_y: %f %f %f %f\n", in_y.x, in_y.y, in_y.z, in_y.w);
    //   printf("in0.x: %f\n", in0.x);
    //   printf("in1.x: %f\n", in1.x);
    //   printf("in2.x: %f\n", in2.x);
    //   printf("in3.x: %f\n", in3.x);
    //   printf("3333cur_index: %d %d\n", cur_index.x, cur_index.y);
    // }

    in_y = (CL_DTYPE4)(in0.x, in1.x, in2.x, in3.x);
  }

  //chw -> nch
  if (image_folder_flag == 4){
    int tmp_c4 = idy_shape.x / 4; //n;
    int tmp_h  = idy_shape.w * 4; //c4 * 4;
    int tmp_w  = idy_shape.y;

    cur_index = (int2)(tmp_c4 * 1 + tmp_w,
                      idy_shape.x * bias_nhwc4.y + idy_shape.y);

    CL_DTYPE4 in0 = 0.f;
    CL_DTYPE4 in1 = 0.f;
    CL_DTYPE4 in2 = 0.f;
    CL_DTYPE4 in3 = 0.f;
  
    in0 = READ_IMG_TYPE(CL_DTYPE_CHAR, input_y, SAMPLER, (int2)(tmp_c4 * bias_nhwc4.y + tmp_w, tmp_h));

    if (cur_index.x + 1 < bias_nhwc4.x * bias_nhwc4.y) {
      in1 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input_y, SAMPLER, (int2)(tmp_c4 * bias_nhwc4.y + tmp_w, tmp_h + 1));
    }
    if (cur_index.x + 2 < bias_nhwc4.x * bias_nhwc4.y) {
      in2 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input_y, SAMPLER, (int2)(tmp_c4 * bias_nhwc4.y + tmp_w, tmp_h + 2));
    }
    if (cur_index.x + 3 < bias_nhwc4.x * bias_nhwc4.y) {
      in3 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input_y, SAMPLER, (int2)(tmp_c4 * bias_nhwc4.y + tmp_w, tmp_h + 3));
    }

    if (idy_shape.x % 4 == 0){
      in_y = (CL_DTYPE4)(in0.x, in1.x, in2.x, in3.x);
    }

    if (idy_shape.x % 4 == 1){
      in_y = (CL_DTYPE4)(in0.y, in1.y, in2.y, in3.y);
    }

    if (idy_shape.x % 4 == 2){
      in_y = (CL_DTYPE4)(in0.z, in1.z, in2.z, in3.z);
    }

    if (idy_shape.x % 4 == 3){
      in_y = (CL_DTYPE4)(in0.w, in1.w, in2.w, in3.w);
    }

    // if (idwc4 == 1 && idbh == 0){
    //   printf("4444in in_y: %f %f %f %f\n", in_y.x, in_y.y, in_y.z, in_y.w);
    //   printf("in0.x: %f\n", in0.x);
    //   printf("in1.x: %f\n", in1.x);
    //   printf("in2.x: %f\n", in2.x);
    //   printf("in3.x: %f\n", in3.x);
    // }

    // if (idwc4 == 0 && idbh == 1){
    //   printf("4444in in_y: %f %f %f %f\n", in_y.x, in_y.y, in_y.z, in_y.w);
    //   printf("in0.x: %f\n", in0.x);
    //   printf("in1.x: %f\n", in1.x);
    //   printf("in2.x: %f\n", in2.x);
    //   printf("in3.x: %f\n", in3.x);
    // }

  }

    // if (idwc4 == 1 && idbh == 2){
    //   printf("21cur_index: %d %d\n", cur_index.x, cur_index.y);
    //   printf("11in_y: %f %f %f %f\n", in_y.x, in_y.y, in_y.z, in_y.w);
    //   printf("11in_x: %f %f %f %f\n", in_x.x, in_x.y, in_x.z, in_x.w);
    // }

    // if (idwc4 == 1 && idbh == 2){
    //   // CL_DTYPE4 in_y_tmp = READ_IMG_TYPE(CL_DTYPE_CHAR, input_y, SAMPLER, (int2)(1, 0));
    //   // printf("00in_y: %f %f %f %f\n", in_y_tmp.x, in_y_tmp.y, in_y_tmp.z, in_y_tmp.w);
    //   // printf("00cur_index: %d %d\n", cur_index.x, cur_index.y);
    //   printf("n h w c4 00in_y: %d %d %d %d\n", id_shape.x, id_shape.y, id_shape.z, id_shape.w);
    //   printf("00in_y: %f %f %f %f\n", in_y.x, in_y.y, in_y.z, in_y.w);
    //   printf("00in_x: %f %f %f %f\n", in_x.x, in_x.y, in_x.z, in_x.w);
    // }

    // if (idwc4 == 1 && idbh == 0){
    //   printf("01cur_index: %d %d\n", cur_index.x, cur_index.y);
    //   printf("01in_y: %f %f %f %f\n", in_y.x, in_y.y, in_y.z, in_y.w);
    //   printf("01in_x: %f %f %f %f\n", in_x.x, in_x.y, in_x.z, in_x.w);
    // }

    // if (idwc4 == 2 && idbh == 0){
    //   printf("02cur_index: %d %d\n", cur_index.x, cur_index.y);
    //   printf("02in_y: %f %f %f %f\n", in_y.x, in_y.y, in_y.z, in_y.w);
    //   printf("02in_x: %f %f %f %f\n", in_x.x, in_x.y, in_x.z, in_x.w);
    // }

    // if (idwc4 == 0 && idbh == 2){
    //   printf("20cur_index: %d %d\n", cur_index.x, cur_index.y);
    //   printf("10in_y: %f %f %f %f\n", in_y.x, in_y.y, in_y.z, in_y.w);
    //   printf("10in_x: %f %f %f %f\n", in_x.x, in_x.y, in_x.z, in_x.w);
    // }

    // if (idwc4 == 1 && idbh == 2){
    //   printf("21cur_index: %d %d\n", cur_index.x, cur_index.y);
    //   printf("11in_y: %f %f %f %f\n", in_y.x, in_y.y, in_y.z, in_y.w);
    //   printf("11in_x: %f %f %f %f\n", in_x.x, in_x.y, in_x.z, in_x.w);
    // }

    // if (idwc4 == 2 && idbh == 2){
    //   printf("22cur_index: %d %d\n", cur_index.x, cur_index.y);
    //   printf("12in_y: %f %f %f %f\n", in_y.x, in_y.y, in_y.z, in_y.w);
    //   printf("12in_x: %f %f %f %f\n", in_x.x, in_x.y, in_x.z, in_x.w);
    // }

  in_x = SELECT(in_x, (CL_DTYPE4)(in_x.x), inputx_broadcast_c_flag);
  in_y = SELECT(in_y, (CL_DTYPE4)(in_y.x), inputy_broadcast_c_flag);

  CL_DTYPE4 output = OPERATOR(in_x, in_y);
#ifdef FUSE_SCALE
  output = fuse_scale(output, SCALE_SLOPE, SCALE_BIAS, SCALE_ALPHA);
#endif

#ifdef RELU
  CL_DTYPE4 alpha;
  output = activation_type4(output, alpha);
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, (int2)(idwc4, idbh), output);
}