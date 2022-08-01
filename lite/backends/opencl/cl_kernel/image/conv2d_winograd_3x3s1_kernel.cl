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
__kernel void transform_from_input(__read_only image2d_t input,
                                   __write_only image2d_t matrix_v,
                                   __private const int in_height,
                                   __private const int in_width,
                                   __private const int in_channel,
                                   __private const int round_h,
                                   __private const int round_w,
                                   __private const int pad,
                                   __private const int global_size_dim0,
                                   __private const int global_size_dim1) {
  const int output_cw_idx = get_global_id(0);  // c/4 w/2
  const int output_bh_idx = get_global_id(1);  // b h/2
  if (output_cw_idx >= global_size_dim0 || output_bh_idx >= global_size_dim1) {
    return;
  }
  const int c_block_idx = output_cw_idx / round_w;
  const int w_block_idx = output_cw_idx - mul24(c_block_idx, round_w);
  const int batch = output_bh_idx / round_h;
  const int h_block_idx = output_bh_idx - mul24(batch, round_h);

  const int width_start_idx = (w_block_idx << 1) - pad;
  const int height_start_idx = (h_block_idx << 1) - pad;

  const int4 width_idx = (int4)(width_start_idx) + (int4)(0, 1, 2, 3);
  const int4 height_idx = (int4)(height_start_idx) + (int4)(0, 1, 2, 3);

  int4 in_wc_idx = mad24((int4)(c_block_idx), (int4)(in_width), width_idx);
  int4 in_bh_idx = mad24((int4)(batch), (int4)(in_height), height_idx);

  in_wc_idx = select(in_wc_idx,
                     (int4)(-1),
                     width_idx < (int4)(0) || width_idx >= (int4)(in_width));
  in_bh_idx = select(in_bh_idx,
                     (int4)(-1),
                     height_idx < (int4)(0) || height_idx >= (int4)(in_height));

  CL_DTYPE4 in00 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s0, in_bh_idx.s0));
  CL_DTYPE4 in10 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s1, in_bh_idx.s0));
  CL_DTYPE4 in20 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s2, in_bh_idx.s0));
  CL_DTYPE4 in30 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s3, in_bh_idx.s0));

  CL_DTYPE4 in01 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s0, in_bh_idx.s1));
  CL_DTYPE4 in11 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s1, in_bh_idx.s1));
  CL_DTYPE4 in21 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s2, in_bh_idx.s1));
  CL_DTYPE4 in31 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s3, in_bh_idx.s1));

  CL_DTYPE4 in02 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s0, in_bh_idx.s2));
  CL_DTYPE4 in12 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s1, in_bh_idx.s2));
  CL_DTYPE4 in22 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s2, in_bh_idx.s2));
  CL_DTYPE4 in32 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s3, in_bh_idx.s2));

  CL_DTYPE4 in03 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s0, in_bh_idx.s3));
  CL_DTYPE4 in13 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s1, in_bh_idx.s3));
  CL_DTYPE4 in23 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s2, in_bh_idx.s3));
  CL_DTYPE4 in33 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_wc_idx.s3, in_bh_idx.s3));

  CL_DTYPE4 v00 = in00 - in02;
  CL_DTYPE4 v10 = in10 - in12;
  CL_DTYPE4 v20 = in20 - in22;
  CL_DTYPE4 v30 = in30 - in32;

  CL_DTYPE4 v01 = (CL_DTYPE)0.5f * in01 + (CL_DTYPE)0.5f * in02;
  CL_DTYPE4 v11 = (CL_DTYPE)0.5f * in11 + (CL_DTYPE)0.5f * in12;
  CL_DTYPE4 v21 = (CL_DTYPE)0.5f * in21 + (CL_DTYPE)0.5f * in22;
  CL_DTYPE4 v31 = (CL_DTYPE)0.5f * in31 + (CL_DTYPE)0.5f * in32;

  CL_DTYPE4 v02 = -(CL_DTYPE)0.5f * in01 + (CL_DTYPE)0.5f * in02;
  CL_DTYPE4 v12 = -(CL_DTYPE)0.5f * in11 + (CL_DTYPE)0.5f * in12;
  CL_DTYPE4 v22 = -(CL_DTYPE)0.5f * in21 + (CL_DTYPE)0.5f * in22;
  CL_DTYPE4 v32 = -(CL_DTYPE)0.5f * in31 + (CL_DTYPE)0.5f * in32;

  CL_DTYPE4 v03 = -in01 + in03;
  CL_DTYPE4 v13 = -in11 + in13;
  CL_DTYPE4 v23 = -in21 + in23;
  CL_DTYPE4 v33 = -in31 + in33;

  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR, matrix_v, (int2)(output_cw_idx, output_bh_idx), v00 - v20);
  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_v,
      (int2)(output_cw_idx, mad24(1, global_size_dim1, output_bh_idx)),
      (CL_DTYPE)0.5f * v10 + (CL_DTYPE)0.5f * v20);
  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_v,
      (int2)(output_cw_idx, mad24(2, global_size_dim1, output_bh_idx)),
      -(CL_DTYPE)0.5f * v10 + (CL_DTYPE)0.5f * v20);
  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_v,
      (int2)(output_cw_idx, mad24(3, global_size_dim1, output_bh_idx)),
      -v10 + v30);

  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_v,
      (int2)(output_cw_idx, mad24(4, global_size_dim1, output_bh_idx)),
      v01 - v21);
  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_v,
      (int2)(output_cw_idx, mad24(5, global_size_dim1, output_bh_idx)),
      (CL_DTYPE)0.5f * v11 + (CL_DTYPE)0.5f * v21);
  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_v,
      (int2)(output_cw_idx, mad24(6, global_size_dim1, output_bh_idx)),
      -(CL_DTYPE)0.5f * v11 + (CL_DTYPE)0.5f * v21);
  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_v,
      (int2)(output_cw_idx, mad24(7, global_size_dim1, output_bh_idx)),
      -v11 + v31);

  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_v,
      (int2)(output_cw_idx, mad24(8, global_size_dim1, output_bh_idx)),
      v02 - v22);
  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_v,
      (int2)(output_cw_idx, mad24(9, global_size_dim1, output_bh_idx)),
      (CL_DTYPE)0.5f * v12 + (CL_DTYPE)0.5f * v22);
  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_v,
      (int2)(output_cw_idx, mad24(10, global_size_dim1, output_bh_idx)),
      -(CL_DTYPE)0.5f * v12 + (CL_DTYPE)0.5f * v22);
  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_v,
      (int2)(output_cw_idx, mad24(11, global_size_dim1, output_bh_idx)),
      -v12 + v32);

  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_v,
      (int2)(output_cw_idx, mad24(12, global_size_dim1, output_bh_idx)),
      v03 - v23);
  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_v,
      (int2)(output_cw_idx, mad24(13, global_size_dim1, output_bh_idx)),
      (CL_DTYPE)0.5f * v13 + (CL_DTYPE)0.5f * v23);
  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_v,
      (int2)(output_cw_idx, mad24(14, global_size_dim1, output_bh_idx)),
      -(CL_DTYPE)0.5f * v13 + (CL_DTYPE)0.5f * v23);
  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_v,
      (int2)(output_cw_idx, mad24(15, global_size_dim1, output_bh_idx)),
      -v13 + v33);
}

__kernel void matrix_inner_product(__read_only image2d_t matrix_v,
                                   __read_only image2d_t matrix_u,
                                   __write_only image2d_t matrix_m,
                                   __private const int round_w,
                                   __private const int round_4x4_w,
                                   __private const int batch_round_h,
                                   __private const int out_channel_block,
                                   __private const int in_channel_block,
                                   __private const int global_size_dim0,
                                   __private const int global_size_dim1) {
  const int output_cw_block_idx = get_global_id(0);  // c/4  w/2/4
  const int output_16_bh_idx = get_global_id(1);     // 16 b h/2

  if (output_cw_block_idx >= global_size_dim0 ||
      output_16_bh_idx >= global_size_dim1) {
    return;
  }
  const int c_block_idx = output_cw_block_idx / round_4x4_w;
  const int w_block_idx = output_cw_block_idx - mul24(c_block_idx, round_4x4_w);
  const int4 w_idx = (int4)(w_block_idx << 2) + (int4)(0, 1, 2, 3);

  const int alpha = output_16_bh_idx / batch_round_h;
  const int u_bh_idx = mul24(alpha, out_channel_block) + c_block_idx;

  CL_DTYPE4 m0 = (CL_DTYPE4)(0);
  CL_DTYPE4 m1 = (CL_DTYPE4)(0);
  CL_DTYPE4 m2 = (CL_DTYPE4)(0);
  CL_DTYPE4 m3 = (CL_DTYPE4)(0);

  for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block;
       ++input_c_block_idx) {
    const int4 input_c_idx =
        (int4)(input_c_block_idx << 2) + (int4)(0, 1, 2, 3);
    int4 v_cw_idx =
        select(mad24((int4)(input_c_block_idx), (int4)(round_w), w_idx),
               (int4)(-1),
               w_idx >= (int4)(round_w));
    CL_DTYPE4 v_in0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                    matrix_v,
                                    SAMPLER,
                                    (int2)(v_cw_idx.s0, output_16_bh_idx));
    CL_DTYPE4 v_in1 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                    matrix_v,
                                    SAMPLER,
                                    (int2)(v_cw_idx.s1, output_16_bh_idx));
    CL_DTYPE4 v_in2 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                    matrix_v,
                                    SAMPLER,
                                    (int2)(v_cw_idx.s2, output_16_bh_idx));
    CL_DTYPE4 v_in3 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                    matrix_v,
                                    SAMPLER,
                                    (int2)(v_cw_idx.s3, output_16_bh_idx));
    CL_DTYPE4 u_in0 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, matrix_u, SAMPLER, (int2)(input_c_idx.s0, u_bh_idx));
    CL_DTYPE4 u_in1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, matrix_u, SAMPLER, (int2)(input_c_idx.s1, u_bh_idx));
    CL_DTYPE4 u_in2 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, matrix_u, SAMPLER, (int2)(input_c_idx.s2, u_bh_idx));
    CL_DTYPE4 u_in3 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, matrix_u, SAMPLER, (int2)(input_c_idx.s3, u_bh_idx));

    m0 = mad(v_in0.s0, u_in0, m0);
    m0 = mad(v_in0.s1, u_in1, m0);
    m0 = mad(v_in0.s2, u_in2, m0);
    m0 = mad(v_in0.s3, u_in3, m0);

    m1 = mad(v_in1.s0, u_in0, m1);
    m1 = mad(v_in1.s1, u_in1, m1);
    m1 = mad(v_in1.s2, u_in2, m1);
    m1 = mad(v_in1.s3, u_in3, m1);

    m2 = mad(v_in2.s0, u_in0, m2);
    m2 = mad(v_in2.s1, u_in1, m2);
    m2 = mad(v_in2.s2, u_in2, m2);
    m2 = mad(v_in2.s3, u_in3, m2);

    m3 = mad(v_in3.s0, u_in0, m3);
    m3 = mad(v_in3.s1, u_in1, m3);
    m3 = mad(v_in3.s2, u_in2, m3);
    m3 = mad(v_in3.s3, u_in3, m3);
  }

  const int output_cw_idx = mad24(c_block_idx, round_w, w_idx.s0);
  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR, matrix_m, (int2)(output_cw_idx, output_16_bh_idx), m0);

  if (w_idx.s1 < round_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   matrix_m,
                   (int2)(output_cw_idx + 1, output_16_bh_idx),
                   m1);
  }

  if (w_idx.s2 < round_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   matrix_m,
                   (int2)(output_cw_idx + 2, output_16_bh_idx),
                   m2);
  }

  if (w_idx.s3 < round_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   matrix_m,
                   (int2)(output_cw_idx + 3, output_16_bh_idx),
                   m3);
  }
}

__kernel void transform_to_output(__read_only image2d_t matrix_m,
                                  __read_only image2d_t bias,
                                  __write_only image2d_t output,
                                  __private const int round_w,
                                  __private const int round_h,
                                  __private const int out_width,
                                  __private const int out_height,
                                  __private const int global_size_dim0,
                                  __private const int global_size_dim1,
                                  __read_only image2d_t prelu_alpha) {
  const int output_cw_idx = get_global_id(0);  // c/4 w/2
  const int output_bh_idx = get_global_id(1);  // b h/2
  if (output_cw_idx >= global_size_dim0 || output_bh_idx >= global_size_dim1) {
    return;
  }
  const int c_block_idx = output_cw_idx / round_w;
  const int w_block_idx = output_cw_idx - mul24(c_block_idx, round_w);
  const int batch = output_bh_idx / round_h;
  const int h_block_idx = output_bh_idx - mul24(batch, round_h);

#ifdef BIASE_CH
  CL_DTYPE4 output0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(c_block_idx, 0));
  CL_DTYPE4 output1 = output0;
  CL_DTYPE4 output2 = output0;
  CL_DTYPE4 output3 = output0;
#else
  CL_DTYPE4 output0 = 0.0f;
  CL_DTYPE4 output1 = 0.0f;
  CL_DTYPE4 output2 = 0.0f;
  CL_DTYPE4 output3 = 0.0f;
#endif

  CL_DTYPE4 m00 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, matrix_m, SAMPLER, (int2)(output_cw_idx, output_bh_idx));
  CL_DTYPE4 m10 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(1, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m20 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(2, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m30 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(3, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m01 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(4, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m11 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(5, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m21 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(6, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m31 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(7, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m02 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(8, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m12 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(9, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m22 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(10, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m32 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(11, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m03 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(12, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m13 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(13, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m23 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(14, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m33 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(15, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 out00 = m00 + m01 + m02;
  CL_DTYPE4 out10 = m10 + m11 + m12;
  CL_DTYPE4 out20 = m20 + m21 + m22;
  CL_DTYPE4 out30 = m30 + m31 + m32;
  CL_DTYPE4 out01 = m01 - m02 + m03;
  CL_DTYPE4 out11 = m11 - m12 + m13;
  CL_DTYPE4 out21 = m21 - m22 + m23;
  CL_DTYPE4 out31 = m31 - m32 + m33;
  int2 ow = (int2)(w_block_idx << 1) + (int2)(0, 1);
  int2 oh = (int2)(h_block_idx << 1) + (int2)(0, 1);
  int2 ox = mad24((int2)(c_block_idx), (int2)(out_width), ow);
  int2 oy = mad24((int2)(batch), (int2)(out_height), oh);

  output0 = output0 + out00 + out10 + out20;
  output1 = output1 + out10 - out20 + out30;
  output2 = output2 + out01 + out11 + out21;
  output3 = output3 + out11 - out21 + out31;

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(c_block_idx, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  alpha0 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(ox.s0, oy.s0 % out_height));
  if (ow.s1 < out_width && oh.s0 < out_height) {
    alpha1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(ox.s1, oy.s0 % out_height));
  }
  if (ow.s0 < out_width && oh.s1 < out_height) {
    alpha2 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(ox.s0, oy.s1 % out_height));
  }
  if (ow.s1 < out_width && oh.s1 < out_height) {
    alpha3 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(ox.s1, oy.s1 % out_height));
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#endif
  output0 = activation_type4(output0, alpha0);
  output1 = activation_type4(output1, alpha1);
  output2 = activation_type4(output2, alpha2);
  output3 = activation_type4(output3, alpha3);

#ifdef SCALE_ACTIVATION
  output0 = fuse_scale(output0, 1.f, 0.f, 0.f);
  output1 = fuse_scale(output1, 1.f, 0.f, 0.f);
  output2 = fuse_scale(output2, 1.f, 0.f, 0.f);
  output3 = fuse_scale(output3, 1.f, 0.f, 0.f);
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(ox.s0, oy.s0), output0);
  if (ow.s1 < out_width && oh.s0 < out_height) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(ox.s1, oy.s0), output1);
  }
  if (ow.s0 < out_width && oh.s1 < out_height) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(ox.s0, oy.s1), output2);
  }
  if (ow.s1 < out_width && oh.s1 < out_height) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(ox.s1, oy.s1), output3);
  }
}

__kernel void matrix_inner_product_mali(__read_only image2d_t matrix_v,
                                        __global CL_DTYPE4 *filter_u,
                                        __write_only image2d_t matrix_m,
                                        __private const int round_w,
                                        __private const int round_4x4_w,
                                        __private const int batch_round_h,
                                        __private const int out_channel_block,
                                        __private const int in_channel_block,
                                        __private const int global_size_dim0,
                                        __private const int global_size_dim1) {
  const int output_cw_block_idx = get_global_id(0);  // c/4  w/2/4
  const int output_16_bh_idx = get_global_id(1);     // 16 b h/2

  if (output_cw_block_idx >= global_size_dim0 ||
      output_16_bh_idx >= global_size_dim1) {
    return;
  }
  const int c_block_idx = output_cw_block_idx / round_4x4_w;
  const int w_block_idx = output_cw_block_idx - mul24(c_block_idx, round_4x4_w);
  const int4 w_idx = (int4)(w_block_idx << 2) + (int4)(0, 1, 2, 3);

  const int alpha = output_16_bh_idx / batch_round_h;
  const int u_bh_idx = mul24(alpha, out_channel_block) + c_block_idx;

  CL_DTYPE4 m0 = (CL_DTYPE4)(0);
  CL_DTYPE4 m1 = (CL_DTYPE4)(0);
  CL_DTYPE4 m2 = (CL_DTYPE4)(0);
  CL_DTYPE4 m3 = (CL_DTYPE4)(0);

  __global CL_DTYPE4 *filter_u_ptr = filter_u + in_channel_block * 4 * u_bh_idx;
  for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block;
       ++input_c_block_idx) {
    const int4 input_c_idx =
        (int4)(input_c_block_idx << 2) + (int4)(0, 1, 2, 3);
    int4 v_cw_idx =
        select(mad24((int4)(input_c_block_idx), (int4)(round_w), w_idx),
               (int4)(-1),
               w_idx >= (int4)(round_w));
    CL_DTYPE4 v_in0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                    matrix_v,
                                    SAMPLER,
                                    (int2)(v_cw_idx.s0, output_16_bh_idx));
    CL_DTYPE4 v_in1 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                    matrix_v,
                                    SAMPLER,
                                    (int2)(v_cw_idx.s1, output_16_bh_idx));
    CL_DTYPE4 v_in2 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                    matrix_v,
                                    SAMPLER,
                                    (int2)(v_cw_idx.s2, output_16_bh_idx));
    CL_DTYPE4 v_in3 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                    matrix_v,
                                    SAMPLER,
                                    (int2)(v_cw_idx.s3, output_16_bh_idx));

    m0 = mad(v_in0.s0, filter_u_ptr[0], m0);
    m0 = mad(v_in0.s1, filter_u_ptr[1], m0);
    m0 = mad(v_in0.s2, filter_u_ptr[2], m0);
    m0 = mad(v_in0.s3, filter_u_ptr[3], m0);

    m1 = mad(v_in1.s0, filter_u_ptr[0], m1);
    m1 = mad(v_in1.s1, filter_u_ptr[1], m1);
    m1 = mad(v_in1.s2, filter_u_ptr[2], m1);
    m1 = mad(v_in1.s3, filter_u_ptr[3], m1);

    m2 = mad(v_in2.s0, filter_u_ptr[0], m2);
    m2 = mad(v_in2.s1, filter_u_ptr[1], m2);
    m2 = mad(v_in2.s2, filter_u_ptr[2], m2);
    m2 = mad(v_in2.s3, filter_u_ptr[3], m2);

    m3 = mad(v_in3.s0, filter_u_ptr[0], m3);
    m3 = mad(v_in3.s1, filter_u_ptr[1], m3);
    m3 = mad(v_in3.s2, filter_u_ptr[2], m3);
    m3 = mad(v_in3.s3, filter_u_ptr[3], m3);
    filter_u_ptr += 4;
  }

  const int output_cw_idx = mad24(c_block_idx, round_w, w_idx.s0);
  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR, matrix_m, (int2)(output_cw_idx, output_16_bh_idx), m0);

  if (w_idx.s1 < round_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   matrix_m,
                   (int2)(output_cw_idx + 1, output_16_bh_idx),
                   m1);
  }

  if (w_idx.s2 < round_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   matrix_m,
                   (int2)(output_cw_idx + 2, output_16_bh_idx),
                   m2);
  }

  if (w_idx.s3 < round_w) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR,
                   matrix_m,
                   (int2)(output_cw_idx + 3, output_16_bh_idx),
                   m3);
  }
}

__kernel void transform_to_output_mali(__read_only image2d_t matrix_m,
                                       __global CL_DTYPE4 *bias,
                                       __write_only image2d_t output,
                                       __private const int round_w,
                                       __private const int round_h,
                                       __private const int out_width,
                                       __private const int out_height,
                                       __private const int global_size_dim0,
                                       __private const int global_size_dim1,
                                       __read_only image2d_t prelu_alpha) {
  const int output_cw_idx = get_global_id(0);  // c/4 w/2
  const int output_bh_idx = get_global_id(1);  // b h/2
  if (output_cw_idx >= global_size_dim0 || output_bh_idx >= global_size_dim1) {
    return;
  }
  const int c_block_idx = output_cw_idx / round_w;
  const int w_block_idx = output_cw_idx - mul24(c_block_idx, round_w);
  const int batch = output_bh_idx / round_h;
  const int h_block_idx = output_bh_idx - mul24(batch, round_h);

#ifdef BIASE_CH
  CL_DTYPE4 output0 = (bias + c_block_idx)[0];
  CL_DTYPE4 output1 = output0;
  CL_DTYPE4 output2 = output0;
  CL_DTYPE4 output3 = output0;
#else
  CL_DTYPE4 output0 = 0.0f;
  CL_DTYPE4 output1 = 0.0f;
  CL_DTYPE4 output2 = 0.0f;
  CL_DTYPE4 output3 = 0.0f;
#endif

  CL_DTYPE4 m00 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, matrix_m, SAMPLER, (int2)(output_cw_idx, output_bh_idx));
  CL_DTYPE4 m10 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(1, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m20 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(2, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m30 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(3, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m01 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(4, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m11 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(5, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m21 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(6, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m31 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(7, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m02 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(8, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m12 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(9, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m22 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(10, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m32 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(11, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m03 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(12, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m13 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(13, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m23 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(14, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 m33 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      matrix_m,
      SAMPLER,
      (int2)(output_cw_idx, mad24(15, global_size_dim1, output_bh_idx)));
  CL_DTYPE4 out00 = m00 + m01 + m02;
  CL_DTYPE4 out10 = m10 + m11 + m12;
  CL_DTYPE4 out20 = m20 + m21 + m22;
  CL_DTYPE4 out30 = m30 + m31 + m32;
  CL_DTYPE4 out01 = m01 - m02 + m03;
  CL_DTYPE4 out11 = m11 - m12 + m13;
  CL_DTYPE4 out21 = m21 - m22 + m23;
  CL_DTYPE4 out31 = m31 - m32 + m33;
  int2 ow = (int2)(w_block_idx << 1) + (int2)(0, 1);
  int2 oh = (int2)(h_block_idx << 1) + (int2)(0, 1);
  int2 ox = mad24((int2)(c_block_idx), (int2)(out_width), ow);
  int2 oy = mad24((int2)(batch), (int2)(out_height), oh);

  output0 = output0 + out00 + out10 + out20;
  output1 = output1 + out10 - out20 + out30;
  output2 = output2 + out01 + out11 + out21;
  output3 = output3 + out11 - out21 + out31;

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(c_block_idx, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  alpha0 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(ox.s0, oy.s0 % out_height));
  if (ow.s1 < out_width && oh.s0 < out_height) {
    alpha1 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(ox.s1, oy.s0 % out_height));
  }
  if (ow.s0 < out_width && oh.s1 < out_height) {
    alpha2 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(ox.s0, oy.s1 % out_height));
  }
  if (ow.s1 < out_width && oh.s1 < out_height) {
    alpha3 = READ_IMG_TYPE(
        CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(ox.s1, oy.s1 % out_height));
  }
//}
#elif defined(PRELU_ALL)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#endif
  output0 = activation_type4(output0, alpha0);
  output1 = activation_type4(output1, alpha1);
  output2 = activation_type4(output2, alpha2);
  output3 = activation_type4(output3, alpha3);

#ifdef SCALE_ACTIVATION
  output0 = fuse_scale(output0, 1.f, 0.f, 0.f);
  output1 = fuse_scale(output1, 1.f, 0.f, 0.f);
  output2 = fuse_scale(output2, 1.f, 0.f, 0.f);
  output3 = fuse_scale(output3, 1.f, 0.f, 0.f);
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(ox.s0, oy.s0), output0);
  if (ow.s1 < out_width && oh.s0 < out_height) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(ox.s1, oy.s0), output1);
  }
  if (ow.s0 < out_width && oh.s1 < out_height) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(ox.s0, oy.s1), output2);
  }
  if (ow.s1 < out_width && oh.s1 < out_height) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(ox.s1, oy.s1), output3);
  }
}