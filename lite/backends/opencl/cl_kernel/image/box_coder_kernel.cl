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

__kernel void encode_center_size(__read_only image2d_t prior_box_image,
                                 __read_only image2d_t target_box_image,
                                 __write_only image2d_t output_image,
                                 __private const int out_C,
                                 __private const int out_H,
                                 __private const int normalized,
                                 __private const float4 variance
#ifdef PRIOR_BOX_VAR
                                 ,
                                 __read_only image2d_t prior_box_var_image
#endif
                                 ) {
  const int out_c = get_global_id(0);
  const int out_nh = get_global_id(1);
  const int out_h = out_nh % out_H;
  const int out_n = 1;

  const int prior_box_n = 1;
  const int prior_box_c = 0;
  const int prior_box_h = out_h;

  const int prior_box_var_n = 1;
  const int prior_box_var_c = 0;
  const int prior_box_var_h = out_h;

  const int target_box_n = 1;
  const int target_box_c = out_c;
  const int target_box_h = out_h;

  int2 prior_box_pos;
  int2 prior_box_var_pos;
  int2 target_box_pos;
  int2 output_pos;

  CL_DTYPE norm_value = (normalized == 0) ? (CL_DTYPE)(1.f) : (CL_DTYPE)(0.f);

  prior_box_pos.x = prior_box_c * 4;
  prior_box_pos.y = prior_box_n * prior_box_h;

  prior_box_var_pos.x = prior_box_var_c * 4;
  prior_box_var_pos.y = prior_box_var_n * prior_box_var_h;

  target_box_pos.x = 0;
  target_box_pos.y = out_c * 4;

  output_pos.x = out_c * 4;
  output_pos.y = out_n * out_h;

  CL_DTYPE4 prior_box_input[4];
  CL_DTYPE prior_box_var_input_0_x, prior_box_var_input_1_x,
      prior_box_var_input_2_x, prior_box_var_input_3_x;
  CL_DTYPE4 target_box_input[4];

  prior_box_input[0] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_image,
                    SAMPLER,
                    (int2)(prior_box_pos.x + 0, prior_box_pos.y));
  prior_box_input[1] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_image,
                    SAMPLER,
                    (int2)(prior_box_pos.x + 1, prior_box_pos.y));
  prior_box_input[2] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_image,
                    SAMPLER,
                    (int2)(prior_box_pos.x + 2, prior_box_pos.y));
  prior_box_input[3] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_image,
                    SAMPLER,
                    (int2)(prior_box_pos.x + 3, prior_box_pos.y));
  prior_box_var_input_0_x = variance.x;
  prior_box_var_input_1_x = variance.y;
  prior_box_var_input_2_x = variance.z;
  prior_box_var_input_3_x = variance.w;
#ifdef PRIOR_BOX_VAR
  prior_box_var_input_0_x =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_var_image,
                    SAMPLER,
                    (int2)(prior_box_var_pos.x + 0, prior_box_var_pos.y))
          .x;
  prior_box_var_input_1_x =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_var_image,
                    SAMPLER,
                    (int2)(prior_box_var_pos.x + 1, prior_box_var_pos.y))
          .x;
  prior_box_var_input_2_x =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_var_image,
                    SAMPLER,
                    (int2)(prior_box_var_pos.x + 2, prior_box_var_pos.y))
          .x;
  prior_box_var_input_3_x =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_var_image,
                    SAMPLER,
                    (int2)(prior_box_var_pos.x + 3, prior_box_var_pos.y))
          .x;
#endif
  target_box_input[0] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    target_box_image,
                    SAMPLER,
                    (int2)(target_box_pos.x + 0, target_box_pos.y));
  target_box_input[1] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    target_box_image,
                    SAMPLER,
                    (int2)(target_box_pos.x + 1, target_box_pos.y));
  target_box_input[2] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    target_box_image,
                    SAMPLER,
                    (int2)(target_box_pos.x + 2, target_box_pos.y));
  target_box_input[3] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    target_box_image,
                    SAMPLER,
                    (int2)(target_box_pos.x + 3, target_box_pos.y));

  CL_DTYPE prior_box_width =
      prior_box_input[2].x - prior_box_input[0].x + norm_value;
  CL_DTYPE prior_box_height =
      prior_box_input[3].x - prior_box_input[1].x + norm_value;
  CL_DTYPE prior_box_center_x =
      prior_box_input[0].x + (CL_DTYPE)0.5 * prior_box_width;
  CL_DTYPE prior_box_center_y =
      prior_box_input[1].x + (CL_DTYPE)0.5 * prior_box_height;

  CL_DTYPE4 target_box_center_x;
  CL_DTYPE4 target_box_center_y;
  CL_DTYPE4 target_box_width;
  CL_DTYPE4 target_box_height;
  CL_DTYPE4 output[4];

  output[0] = 0.0f;
  output[1] = 0.0f;
  output[2] = 0.0f;
  output[3] = 0.0f;

  target_box_center_x.x = (target_box_input[0].x + target_box_input[2].x) / 2;
  target_box_center_y.x = (target_box_input[1].x + target_box_input[3].x) / 2;
  target_box_width.x =
      target_box_input[2].x - target_box_input[0].x + norm_value;
  target_box_height.x =
      target_box_input[3].x - target_box_input[1].x + norm_value;

  output[0].x =
      ((target_box_center_x.x - prior_box_center_x) / prior_box_width) /
      prior_box_var_input_0_x;
  output[1].x =
      ((target_box_center_y.x - prior_box_center_y) / prior_box_height) /
      prior_box_var_input_1_x;
  output[2].x =
      log(fabs(target_box_width.x / prior_box_width)) / prior_box_var_input_2_x;
  output[3].x = log(fabs(target_box_height.x / prior_box_height)) /
                prior_box_var_input_3_x;

  if (out_C - out_c * 4 >= 2) {
    target_box_input[0] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      target_box_image,
                      SAMPLER,
                      (int2)(target_box_pos.x + 0, target_box_pos.y + 1));
    target_box_input[1] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      target_box_image,
                      SAMPLER,
                      (int2)(target_box_pos.x + 1, target_box_pos.y + 1));
    target_box_input[2] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      target_box_image,
                      SAMPLER,
                      (int2)(target_box_pos.x + 2, target_box_pos.y + 1));
    target_box_input[3] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      target_box_image,
                      SAMPLER,
                      (int2)(target_box_pos.x + 3, target_box_pos.y + 1));

    target_box_center_x.y = (target_box_input[0].x + target_box_input[2].x) / 2;
    target_box_center_y.y = (target_box_input[1].x + target_box_input[3].x) / 2;
    target_box_width.y =
        target_box_input[2].x - target_box_input[0].x + norm_value;
    target_box_height.y =
        target_box_input[3].x - target_box_input[1].x + norm_value;

    output[0].y =
        ((target_box_center_x.y - prior_box_center_x) / prior_box_width) /
        prior_box_var_input_0_x;
    output[1].y =
        ((target_box_center_y.y - prior_box_center_y) / prior_box_height) /
        prior_box_var_input_1_x;
    output[2].y = log(fabs(target_box_width.y / prior_box_width)) /
                  prior_box_var_input_2_x;
    output[3].y = log(fabs(target_box_height.y / prior_box_height)) /
                  prior_box_var_input_3_x;
  }

  if (out_C - out_c * 4 >= 3) {
    target_box_input[0] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      target_box_image,
                      SAMPLER,
                      (int2)(target_box_pos.x + 0, target_box_pos.y + 2));
    target_box_input[1] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      target_box_image,
                      SAMPLER,
                      (int2)(target_box_pos.x + 1, target_box_pos.y + 2));
    target_box_input[2] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      target_box_image,
                      SAMPLER,
                      (int2)(target_box_pos.x + 2, target_box_pos.y + 2));
    target_box_input[3] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      target_box_image,
                      SAMPLER,
                      (int2)(target_box_pos.x + 3, target_box_pos.y + 2));

    target_box_center_x.z = (target_box_input[0].x + target_box_input[2].x) / 2;
    target_box_center_y.z = (target_box_input[1].x + target_box_input[3].x) / 2;
    target_box_width.z =
        target_box_input[2].x - target_box_input[0].x + norm_value;
    target_box_height.z =
        target_box_input[3].x - target_box_input[1].x + norm_value;

    output[0].z =
        ((target_box_center_x.z - prior_box_center_x) / prior_box_width) /
        prior_box_var_input_0_x;
    output[1].z =
        ((target_box_center_y.z - prior_box_center_y) / prior_box_height) /
        prior_box_var_input_1_x;
    output[2].z = log(fabs(target_box_width.z / prior_box_width)) /
                  prior_box_var_input_2_x;
    output[3].z = log(fabs(target_box_height.z / prior_box_height)) /
                  prior_box_var_input_3_x;
  }

  if (out_C - out_c * 4 >= 4) {
    target_box_input[0] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      target_box_image,
                      SAMPLER,
                      (int2)(target_box_pos.x + 0, target_box_pos.y + 3));
    target_box_input[1] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      target_box_image,
                      SAMPLER,
                      (int2)(target_box_pos.x + 1, target_box_pos.y + 3));
    target_box_input[2] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      target_box_image,
                      SAMPLER,
                      (int2)(target_box_pos.x + 2, target_box_pos.y + 3));
    target_box_input[3] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      target_box_image,
                      SAMPLER,
                      (int2)(target_box_pos.x + 3, target_box_pos.y + 3));
    target_box_center_x.w = (target_box_input[0].x + target_box_input[2].x) / 2;
    target_box_center_y.w = (target_box_input[1].x + target_box_input[3].x) / 2;
    target_box_width.w =
        target_box_input[2].x - target_box_input[0].x + norm_value;
    target_box_height.w =
        target_box_input[3].x - target_box_input[1].x + norm_value;

    output[0].w =
        ((target_box_center_x.w - prior_box_center_x) / prior_box_width) /
        prior_box_var_input_0_x;
    output[1].w =
        ((target_box_center_y.w - prior_box_center_y) / prior_box_height) /
        prior_box_var_input_1_x;
    output[2].w = log(fabs(target_box_width.w / prior_box_width)) /
                  prior_box_var_input_2_x;
    output[3].w = log(fabs(target_box_height.w / prior_box_height)) /
                  prior_box_var_input_3_x;
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output_image,
                 (int2)(output_pos.x + 0, output_pos.y),
                 output[0]);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output_image,
                 (int2)(output_pos.x + 1, output_pos.y),
                 output[1]);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output_image,
                 (int2)(output_pos.x + 2, output_pos.y),
                 output[2]);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output_image,
                 (int2)(output_pos.x + 3, output_pos.y),
                 output[3]);
}

__kernel void decode_center_size_axis0(__read_only image2d_t prior_box_image,
                                       __read_only image2d_t target_box_image,
                                       __write_only image2d_t output_image,
                                       __private const int out_C,
                                       __private const int out_H,
                                       __private const int normalized,
                                       __private const float4 variance
#ifdef PRIOR_BOX_VAR
                                       ,
                                       __read_only image2d_t prior_box_var_image
#endif
                                       ) {
  const int out_c = get_global_id(0);
  const int out_nh = get_global_id(1);
  const int out_h = out_nh % out_H;
  const int out_n = 1;

  const int prior_box_n = 1;
  const int prior_box_c = 0;
  const int prior_box_h = out_h;

  const int prior_box_var_n = 1;
  const int prior_box_var_c = 0;
  const int prior_box_var_h = out_h;

  const int target_box_n = 1;
  const int target_box_c = out_c;
  const int target_box_h = out_h;

  int2 prior_box_pos;
  int2 prior_box_var_pos;
  int2 target_box_pos;
  int2 output_pos;

  CL_DTYPE norm_value = (normalized == 0) ? (CL_DTYPE)(1.f) : (CL_DTYPE)(0.f);

  prior_box_pos.x = prior_box_c * 4;
  prior_box_pos.y = prior_box_n * prior_box_h;

  prior_box_var_pos.x = prior_box_var_c * 4;
  prior_box_var_pos.y = prior_box_var_n * prior_box_var_h;

  target_box_pos.x = target_box_c * 4;
  target_box_pos.y = target_box_n * target_box_h;

  output_pos.x = out_c * 4;
  output_pos.y = out_n * out_h;

  CL_DTYPE4 prior_box_input[4];
  CL_DTYPE prior_box_var_input_0_x, prior_box_var_input_1_x,
      prior_box_var_input_2_x, prior_box_var_input_3_x;
  CL_DTYPE4 target_box_input[4];

  prior_box_input[0] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_image,
                    SAMPLER,
                    (int2)(prior_box_pos.x + 0, prior_box_pos.y));
  prior_box_input[1] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_image,
                    SAMPLER,
                    (int2)(prior_box_pos.x + 1, prior_box_pos.y));
  prior_box_input[2] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_image,
                    SAMPLER,
                    (int2)(prior_box_pos.x + 2, prior_box_pos.y));
  prior_box_input[3] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_image,
                    SAMPLER,
                    (int2)(prior_box_pos.x + 3, prior_box_pos.y));

  prior_box_var_input_0_x = variance.x;
  prior_box_var_input_1_x = variance.y;
  prior_box_var_input_2_x = variance.z;
  prior_box_var_input_3_x = variance.w;
#ifdef PRIOR_BOX_VAR
  prior_box_var_input_0_x =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_var_image,
                    SAMPLER,
                    (int2)(prior_box_var_pos.x + 0, prior_box_var_pos.y))
          .x;
  prior_box_var_input_1_x =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_var_image,
                    SAMPLER,
                    (int2)(prior_box_var_pos.x + 1, prior_box_var_pos.y))
          .x;
  prior_box_var_input_2_x =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_var_image,
                    SAMPLER,
                    (int2)(prior_box_var_pos.x + 2, prior_box_var_pos.y))
          .x;
  prior_box_var_input_3_x =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_var_image,
                    SAMPLER,
                    (int2)(prior_box_var_pos.x + 3, prior_box_var_pos.y))
          .x;
#endif

  target_box_input[0] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    target_box_image,
                    SAMPLER,
                    (int2)(target_box_pos.x + 0, target_box_pos.y));
  target_box_input[1] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    target_box_image,
                    SAMPLER,
                    (int2)(target_box_pos.x + 1, target_box_pos.y));
  target_box_input[2] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    target_box_image,
                    SAMPLER,
                    (int2)(target_box_pos.x + 2, target_box_pos.y));
  target_box_input[3] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    target_box_image,
                    SAMPLER,
                    (int2)(target_box_pos.x + 3, target_box_pos.y));

  CL_DTYPE prior_box_width =
      prior_box_input[2].x - prior_box_input[0].x + norm_value;
  CL_DTYPE prior_box_height =
      prior_box_input[3].x - prior_box_input[1].x + norm_value;
  CL_DTYPE prior_box_center_x =
      prior_box_input[0].x + (CL_DTYPE)0.5 * prior_box_width;
  CL_DTYPE prior_box_center_y =
      prior_box_input[1].x + (CL_DTYPE)0.5 * prior_box_height;

  CL_DTYPE4 target_box_center_x;
  CL_DTYPE4 target_box_center_y;
  CL_DTYPE4 target_box_width;
  CL_DTYPE4 target_box_height;
  CL_DTYPE4 output[4];

  output[0] = 0.0f;
  output[1] = 0.0f;
  output[2] = 0.0f;
  output[3] = 0.0f;

  target_box_center_x.x =
      prior_box_var_input_0_x * target_box_input[0].x * prior_box_width +
      prior_box_center_x;
  target_box_center_y.x =
      prior_box_var_input_1_x * target_box_input[1].x * prior_box_height +
      prior_box_center_y;
  target_box_width.x =
      exp(prior_box_var_input_2_x * target_box_input[2].x) * prior_box_width;
  target_box_height.x =
      exp(prior_box_var_input_3_x * target_box_input[3].x) * prior_box_height;

  output[0].x = target_box_center_x.x - target_box_width.x / (half)2;
  output[1].x = target_box_center_y.x - target_box_height.x / (half)2;
  output[2].x =
      target_box_center_x.x + target_box_width.x / (half)2 - norm_value;
  output[3].x =
      target_box_center_y.x + target_box_height.x / (half)2 - norm_value;

  if (out_C - out_c * 4 >= 2) {
    target_box_center_x.y =
        prior_box_var_input_0_x * target_box_input[0].y * prior_box_width +
        prior_box_center_x;
    target_box_center_y.y =
        prior_box_var_input_1_x * target_box_input[1].y * prior_box_height +
        prior_box_center_y;
    target_box_width.y =
        exp(prior_box_var_input_2_x * target_box_input[2].y) * prior_box_width;
    target_box_height.y =
        exp(prior_box_var_input_3_x * target_box_input[3].y) * prior_box_height;
    output[0].y = target_box_center_x.y - target_box_width.y / (half)2;
    output[1].y = target_box_center_y.y - target_box_height.y / (half)2;
    output[2].y =
        target_box_center_x.y + target_box_width.y / (half)2 - norm_value;
    output[3].y =
        target_box_center_y.y + target_box_height.y / (half)2 - norm_value;
  }
  if (out_C - out_c * 4 >= 3) {
    target_box_center_x.z =
        prior_box_var_input_0_x * target_box_input[0].z * prior_box_width +
        prior_box_center_x;
    target_box_center_y.z =
        prior_box_var_input_1_x * target_box_input[1].z * prior_box_height +
        prior_box_center_y;
    target_box_width.z =
        exp(prior_box_var_input_2_x * target_box_input[2].z) * prior_box_width;
    target_box_height.z =
        exp(prior_box_var_input_3_x * target_box_input[3].z) * prior_box_height;
    output[0].z = target_box_center_x.z - target_box_width.z / (half)2;
    output[1].z = target_box_center_y.z - target_box_height.z / (half)2;
    output[2].z =
        target_box_center_x.z + target_box_width.z / (half)2 - norm_value;
    output[3].z =
        target_box_center_y.z + target_box_height.z / (half)2 - norm_value;
  }
  if (out_C - out_c * 4 >= 4) {
    target_box_center_x.w =
        prior_box_var_input_0_x * target_box_input[0].w * prior_box_width +
        prior_box_center_x;
    target_box_center_y.w =
        prior_box_var_input_1_x * target_box_input[1].w * prior_box_height +
        prior_box_center_y;
    target_box_width.w =
        exp(prior_box_var_input_2_x * target_box_input[2].w) * prior_box_width;
    target_box_height.w =
        exp(prior_box_var_input_3_x * target_box_input[3].w) * prior_box_height;
    output[0].w = target_box_center_x.w - target_box_width.w / (half)2;
    output[1].w = target_box_center_y.w - target_box_height.w / (half)2;
    output[2].w =
        target_box_center_x.w + target_box_width.w / (half)2 - norm_value;
    output[3].w =
        target_box_center_y.w + target_box_height.w / (half)2 - norm_value;
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output_image,
                 (int2)(output_pos.x + 0, output_pos.y),
                 output[0]);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output_image,
                 (int2)(output_pos.x + 1, output_pos.y),
                 output[1]);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output_image,
                 (int2)(output_pos.x + 2, output_pos.y),
                 output[2]);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output_image,
                 (int2)(output_pos.x + 3, output_pos.y),
                 output[3]);
}

__kernel void decode_center_size_axis1(__read_only image2d_t prior_box_image,
                                       __read_only image2d_t target_box_image,
                                       __write_only image2d_t output_image,
                                       __private const int out_C,
                                       __private const int out_H,
                                       __private const int normalized,
                                       __private const float4 variance
#ifdef PRIOR_BOX_VAR
                                       ,
                                       __read_only image2d_t prior_box_var_image
#endif
                                       ) {
  const int out_c = get_global_id(0);
  const int out_nh = get_global_id(1);
  const int out_h = out_nh % out_H;
  const int out_n = 1;

  const int prior_box_n = 1;
  const int prior_box_c = 0;
  const int prior_box_h = out_h;

  const int prior_box_var_n = 1;
  const int prior_box_var_c = 0;
  const int prior_box_var_h = out_h;

  const int target_box_n = 1;
  const int target_box_c = out_c;
  const int target_box_h = out_h;

  int2 prior_box_pos;
  int2 prior_box_var_pos;
  int2 target_box_pos;
  int2 output_pos;

  CL_DTYPE norm_value = (normalized == 0) ? (CL_DTYPE)(1.f) : (CL_DTYPE)(0.f);

  prior_box_pos.x = prior_box_c * 4;
  prior_box_pos.y = target_box_c * 4;

  prior_box_var_pos.x = prior_box_var_c * 4;
  prior_box_var_pos.y = target_box_c * 4;

  target_box_pos.x = target_box_c * 4;
  target_box_pos.y = target_box_n * target_box_h;

  output_pos.x = out_c * 4;
  output_pos.y = out_n * out_h;

  CL_DTYPE4 prior_box_input[4];
  CL_DTYPE prior_box_var_input_0_x, prior_box_var_input_1_x,
      prior_box_var_input_2_x, prior_box_var_input_3_x;
  CL_DTYPE4 target_box_input[4];
  prior_box_var_input_0_x = variance.x;
  prior_box_var_input_1_x = variance.y;
  prior_box_var_input_2_x = variance.z;
  prior_box_var_input_3_x = variance.w;

  prior_box_input[0] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_image,
                    SAMPLER,
                    (int2)(prior_box_pos.x + 0, prior_box_pos.y));
  prior_box_input[1] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_image,
                    SAMPLER,
                    (int2)(prior_box_pos.x + 1, prior_box_pos.y));
  prior_box_input[2] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_image,
                    SAMPLER,
                    (int2)(prior_box_pos.x + 2, prior_box_pos.y));
  prior_box_input[3] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_image,
                    SAMPLER,
                    (int2)(prior_box_pos.x + 3, prior_box_pos.y));
#ifdef PRIOR_BOX_VAR
  prior_box_var_input_0_x =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_var_image,
                    SAMPLER,
                    (int2)(prior_box_var_pos.x + 0, prior_box_var_pos.y))
          .x;
  prior_box_var_input_1_x =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_var_image,
                    SAMPLER,
                    (int2)(prior_box_var_pos.x + 1, prior_box_var_pos.y))
          .x;
  prior_box_var_input_2_x =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_var_image,
                    SAMPLER,
                    (int2)(prior_box_var_pos.x + 2, prior_box_var_pos.y))
          .x;
  prior_box_var_input_3_x =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    prior_box_var_image,
                    SAMPLER,
                    (int2)(prior_box_var_pos.x + 3, prior_box_var_pos.y))
          .x;
#endif
  target_box_input[0] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    target_box_image,
                    SAMPLER,
                    (int2)(target_box_pos.x + 0, target_box_pos.y));
  target_box_input[1] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    target_box_image,
                    SAMPLER,
                    (int2)(target_box_pos.x + 1, target_box_pos.y));
  target_box_input[2] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    target_box_image,
                    SAMPLER,
                    (int2)(target_box_pos.x + 2, target_box_pos.y));
  target_box_input[3] =
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    target_box_image,
                    SAMPLER,
                    (int2)(target_box_pos.x + 3, target_box_pos.y));

  CL_DTYPE prior_box_width =
      prior_box_input[2].x - prior_box_input[0].x + norm_value;
  CL_DTYPE prior_box_height =
      prior_box_input[3].x - prior_box_input[1].x + norm_value;
  CL_DTYPE prior_box_center_x =
      prior_box_input[0].x + (CL_DTYPE)0.5 * prior_box_width;
  CL_DTYPE prior_box_center_y =
      prior_box_input[1].x + (CL_DTYPE)0.5 * prior_box_height;

  CL_DTYPE4 target_box_center_x;
  CL_DTYPE4 target_box_center_y;
  CL_DTYPE4 target_box_width;
  CL_DTYPE4 target_box_height;
  CL_DTYPE4 output[4];

  output[0] = 0.0f;
  output[1] = 0.0f;
  output[2] = 0.0f;
  output[3] = 0.0f;

  target_box_center_x.x =
      prior_box_var_input_0_x * target_box_input[0].x * prior_box_width +
      prior_box_center_x;
  target_box_center_y.x =
      prior_box_var_input_1_x * target_box_input[1].x * prior_box_height +
      prior_box_center_y;
  target_box_width.x =
      exp(prior_box_var_input_2_x * target_box_input[2].x) * prior_box_width;
  target_box_height.x =
      exp(prior_box_var_input_3_x * target_box_input[3].x) * prior_box_height;

  output[0].x = target_box_center_x.x - target_box_width.x / (half)2;
  output[1].x = target_box_center_y.x - target_box_height.x / (half)2;
  output[2].x =
      target_box_center_x.x + target_box_width.x / (half)2 - norm_value;
  output[3].x =
      target_box_center_y.x + target_box_height.x / (half)2 - norm_value;

  if (out_C - out_c * 4 >= 2) {
    CL_DTYPE4 prior_box_input_1[4];
    prior_box_input_1[0] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_image,
                      SAMPLER,
                      (int2)(prior_box_pos.x + 0, prior_box_pos.y + 1));
    prior_box_input_1[1] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_image,
                      SAMPLER,
                      (int2)(prior_box_pos.x + 1, prior_box_pos.y + 1));
    prior_box_input_1[2] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_image,
                      SAMPLER,
                      (int2)(prior_box_pos.x + 2, prior_box_pos.y + 1));
    prior_box_input_1[3] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_image,
                      SAMPLER,
                      (int2)(prior_box_pos.x + 3, prior_box_pos.y + 1));
#ifdef PRIOR_BOX_VAR
    prior_box_var_input_0_x =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_var_image,
                      SAMPLER,
                      (int2)(prior_box_var_pos.x + 0, prior_box_var_pos.y + 1))
            .x;
    prior_box_var_input_1_x =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_var_image,
                      SAMPLER,
                      (int2)(prior_box_var_pos.x + 1, prior_box_var_pos.y + 1))
            .x;
    prior_box_var_input_2_x =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_var_image,
                      SAMPLER,
                      (int2)(prior_box_var_pos.x + 2, prior_box_var_pos.y + 1))
            .x;
    prior_box_var_input_3_x =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_var_image,
                      SAMPLER,
                      (int2)(prior_box_var_pos.x + 3, prior_box_var_pos.y + 1))
            .x;
#endif
    CL_DTYPE prior_box_width_1 =
        prior_box_input_1[2].x - prior_box_input_1[0].x + norm_value;
    CL_DTYPE prior_box_height_1 =
        prior_box_input_1[3].x - prior_box_input_1[1].x + norm_value;
    CL_DTYPE prior_box_center_x_1 =
        prior_box_input_1[0].x + (CL_DTYPE)0.5 * prior_box_width_1;
    CL_DTYPE prior_box_center_y_1 =
        prior_box_input_1[1].x + (CL_DTYPE)0.5 * prior_box_height_1;

    target_box_center_x.y =
        prior_box_var_input_0_x * target_box_input[0].y * prior_box_width_1 +
        prior_box_center_x_1;
    target_box_center_y.y =
        prior_box_var_input_1_x * target_box_input[1].y * prior_box_height_1 +
        prior_box_center_y_1;
    target_box_width.y = exp(prior_box_var_input_2_x * target_box_input[2].y) *
                         prior_box_width_1;
    target_box_height.y = exp(prior_box_var_input_3_x * target_box_input[3].y) *
                          prior_box_height_1;
    output[0].y = target_box_center_x.y - target_box_width.y / (half)2;
    output[1].y = target_box_center_y.y - target_box_height.y / (half)2;
    output[2].y =
        target_box_center_x.y + target_box_width.y / (half)2 - norm_value;
    output[3].y =
        target_box_center_y.y + target_box_height.y / (half)2 - norm_value;
  }
  if (out_C - out_c * 4 >= 3) {
    CL_DTYPE4 prior_box_input_2[4];
    prior_box_input_2[0] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_image,
                      SAMPLER,
                      (int2)(prior_box_pos.x + 0, prior_box_pos.y + 2));
    prior_box_input_2[1] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_image,
                      SAMPLER,
                      (int2)(prior_box_pos.x + 1, prior_box_pos.y + 2));
    prior_box_input_2[2] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_image,
                      SAMPLER,
                      (int2)(prior_box_pos.x + 2, prior_box_pos.y + 2));
    prior_box_input_2[3] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_image,
                      SAMPLER,
                      (int2)(prior_box_pos.x + 3, prior_box_pos.y + 2));
#ifdef PRIOR_BOX_VAR
    prior_box_var_input_0_x =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_var_image,
                      SAMPLER,
                      (int2)(prior_box_var_pos.x + 0, prior_box_var_pos.y + 2))
            .x;
    prior_box_var_input_1_x =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_var_image,
                      SAMPLER,
                      (int2)(prior_box_var_pos.x + 1, prior_box_var_pos.y + 2))
            .x;
    prior_box_var_input_2_x =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_var_image,
                      SAMPLER,
                      (int2)(prior_box_var_pos.x + 2, prior_box_var_pos.y + 2))
            .x;
    prior_box_var_input_3_x =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_var_image,
                      SAMPLER,
                      (int2)(prior_box_var_pos.x + 3, prior_box_var_pos.y + 2))
            .x;
#endif
    CL_DTYPE prior_box_width_2 =
        prior_box_input_2[2].x - prior_box_input_2[0].x + norm_value;
    CL_DTYPE prior_box_height_2 =
        prior_box_input_2[3].x - prior_box_input_2[1].x + norm_value;
    CL_DTYPE prior_box_center_x_2 =
        prior_box_input_2[0].x + (CL_DTYPE)0.5 * prior_box_width_2;
    CL_DTYPE prior_box_center_y_2 =
        prior_box_input_2[1].x + (CL_DTYPE)0.5 * prior_box_height_2;

    target_box_center_x.z =
        prior_box_var_input_0_x * target_box_input[0].z * prior_box_width_2 +
        prior_box_center_x_2;
    target_box_center_y.z =
        prior_box_var_input_1_x * target_box_input[1].z * prior_box_height_2 +
        prior_box_center_y_2;
    target_box_width.z = exp(prior_box_var_input_2_x * target_box_input[2].z) *
                         prior_box_width_2;
    target_box_height.z = exp(prior_box_var_input_3_x * target_box_input[3].z) *
                          prior_box_height_2;

    output[0].z = target_box_center_x.z - target_box_width.z / (half)2;
    output[1].z = target_box_center_y.z - target_box_height.z / (half)2;
    output[2].z =
        target_box_center_x.z + target_box_width.z / (half)2 - norm_value;
    output[3].z =
        target_box_center_y.z + target_box_height.z / (half)2 - norm_value;
  }
  if (out_C - out_c * 4 >= 4) {
    CL_DTYPE4 prior_box_input_3[4];
    CL_DTYPE4 prior_box_var_input_3[4];
    prior_box_input_3[0] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_image,
                      SAMPLER,
                      (int2)(prior_box_pos.x + 0, prior_box_pos.y + 3));
    prior_box_input_3[1] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_image,
                      SAMPLER,
                      (int2)(prior_box_pos.x + 1, prior_box_pos.y + 3));
    prior_box_input_3[2] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_image,
                      SAMPLER,
                      (int2)(prior_box_pos.x + 2, prior_box_pos.y + 3));
    prior_box_input_3[3] =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_image,
                      SAMPLER,
                      (int2)(prior_box_pos.x + 3, prior_box_pos.y + 3));
#ifdef PRIOR_BOX_VAR
    prior_box_var_input_0_x =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_var_image,
                      SAMPLER,
                      (int2)(prior_box_var_pos.x + 0, prior_box_var_pos.y + 3))
            .x;
    prior_box_var_input_1_x =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_var_image,
                      SAMPLER,
                      (int2)(prior_box_var_pos.x + 1, prior_box_var_pos.y + 3))
            .x;
    prior_box_var_input_2_x =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_var_image,
                      SAMPLER,
                      (int2)(prior_box_var_pos.x + 2, prior_box_var_pos.y + 3))
            .x;
    prior_box_var_input_3_x =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prior_box_var_image,
                      SAMPLER,
                      (int2)(prior_box_var_pos.x + 3, prior_box_var_pos.y + 3))
            .x;
#endif
    CL_DTYPE prior_box_width_3 =
        prior_box_input_3[2].x - prior_box_input_3[0].x + norm_value;
    CL_DTYPE prior_box_height_3 =
        prior_box_input_3[3].x - prior_box_input_3[1].x + norm_value;
    CL_DTYPE prior_box_center_x_3 =
        prior_box_input_3[0].x + (CL_DTYPE)0.5 * prior_box_width_3;
    CL_DTYPE prior_box_center_y_3 =
        prior_box_input_3[1].x + (CL_DTYPE)0.5 * prior_box_height_3;

    target_box_center_x.w =
        prior_box_var_input_0_x * target_box_input[0].w * prior_box_width_3 +
        prior_box_center_x_3;
    target_box_center_y.w =
        prior_box_var_input_1_x * target_box_input[1].w * prior_box_height_3 +
        prior_box_center_y_3;
    target_box_width.w = exp(prior_box_var_input_2_x * target_box_input[2].w) *
                         prior_box_width_3;
    target_box_height.w = exp(prior_box_var_input_3_x * target_box_input[3].w) *
                          prior_box_height_3;
    output[0].w = target_box_center_x.w - target_box_width.w / (half)2;
    output[1].w = target_box_center_y.w - target_box_height.w / (half)2;
    output[2].w =
        target_box_center_x.w + target_box_width.w / (half)2 - norm_value;
    output[3].w =
        target_box_center_y.w + target_box_height.w / (half)2 - norm_value;
  }

  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output_image,
                 (int2)(output_pos.x + 0, output_pos.y),
                 output[0]);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output_image,
                 (int2)(output_pos.x + 1, output_pos.y),
                 output[1]);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output_image,
                 (int2)(output_pos.x + 2, output_pos.y),
                 output[2]);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                 output_image,
                 (int2)(output_pos.x + 3, output_pos.y),
                 output[3]);
}
