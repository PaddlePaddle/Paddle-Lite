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

__kernel void depth_conv2d_3x3(
    __private const int global_size_dim0,
    __private const int global_size_dim1,
    __private const int global_size_dim2,
    __read_only image2d_t input,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
    __write_only image2d_t output_image,
    __private const int stride_h,
    __private const int stride_w,
    __private const int offset,
    __private const int dilation,
    __private const int input_c,
    __private const int input_width,  /* of one block */
    __private const int input_height, /* of one block */
    __private const int output_width,
    __private const int output_height,
    __read_only image2d_t prelu_alpha) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  if (out_c >= global_size_dim0 || out_w >= global_size_dim1 ||
      out_nh >= global_size_dim2) {
    return;
  }

  int2 output_pos = (int2)(out_c * global_size_dim1 + out_w, out_nh);
  const int batch_index = out_nh / output_height;
  const int out_nh_in_one_batch = out_nh % output_height;

  int2 stride_xy = (int2)(stride_w, stride_h);
  int2 ouput_pos_in_one_block = (int2)(out_w, out_nh_in_one_batch);

  int2 in_pos_in_one_block =
      ouput_pos_in_one_block * stride_xy +
      (int2)(offset + dilation - 1, offset + dilation - 1);

#ifdef BIASE_CH
  CL_DTYPE4 output =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
#elif defined(BIASE_ELE)
  CL_DTYPE4 output = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, output_pos);
#else
  CL_DTYPE4 output = 0.0f;
#endif

  const int filter_width = 3;
  const int filter_height = 3;

  int2 pos_in_input_block =
      (int2)(out_c * input_width, batch_index * input_height);

  int2 pos_in_filter_block =
      (int2)(out_c * filter_width, batch_index * filter_height);

  int filter_x = pos_in_filter_block.x;
  int filter_y = pos_in_filter_block.y;
  CL_DTYPE4 inputs[9];

  inputs[0] = SELECT(
      READ_IMG_TYPE(
          CL_DTYPE_CHAR,
          input,
          SAMPLER,
          (int2)(pos_in_input_block.x + in_pos_in_one_block.x - dilation,
                 pos_in_input_block.y + in_pos_in_one_block.y - dilation)),
      (CL_DTYPE4)(0.0f),
      ((in_pos_in_one_block.x - dilation < 0 ||
        in_pos_in_one_block.y - dilation < 0 ||
        in_pos_in_one_block.x - dilation >= input_width ||
        in_pos_in_one_block.y - dilation >= input_height)));

  inputs[1] = SELECT(
      READ_IMG_TYPE(
          CL_DTYPE_CHAR,
          input,
          SAMPLER,
          (int2)(pos_in_input_block.x + in_pos_in_one_block.x,
                 pos_in_input_block.y + in_pos_in_one_block.y - dilation)),
      (CL_DTYPE4)(0.0f),
      ((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y - dilation < 0 ||
        in_pos_in_one_block.x >= input_width ||
        in_pos_in_one_block.y - dilation >= input_height)));

  inputs[2] = SELECT(
      READ_IMG_TYPE(
          CL_DTYPE_CHAR,
          input,
          SAMPLER,
          (int2)(pos_in_input_block.x + in_pos_in_one_block.x + dilation,
                 pos_in_input_block.y + in_pos_in_one_block.y - dilation)),
      (CL_DTYPE4)(0.0f),
      ((in_pos_in_one_block.x + dilation < 0 ||
        in_pos_in_one_block.y - dilation < 0 ||
        in_pos_in_one_block.x + dilation >= input_width ||
        in_pos_in_one_block.y - dilation >= input_height)));

  inputs[3] = SELECT(
      READ_IMG_TYPE(
          CL_DTYPE_CHAR,
          input,
          SAMPLER,
          (int2)(pos_in_input_block.x + in_pos_in_one_block.x - dilation,
                 pos_in_input_block.y + in_pos_in_one_block.y)),
      (CL_DTYPE4)(0.0f),
      ((in_pos_in_one_block.x - dilation < 0 || in_pos_in_one_block.y < 0 ||
        in_pos_in_one_block.x - dilation >= input_width ||
        in_pos_in_one_block.y >= input_height)));

  inputs[4] = SELECT(
      READ_IMG_TYPE(CL_DTYPE_CHAR,
                    input,
                    SAMPLER,
                    (int2)(pos_in_input_block.x + in_pos_in_one_block.x,
                           pos_in_input_block.y + in_pos_in_one_block.y)),
      (CL_DTYPE4)(0.0f),
      ((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y < 0 ||
        in_pos_in_one_block.x >= input_width ||
        in_pos_in_one_block.y >= input_height)));

  inputs[5] = SELECT(
      READ_IMG_TYPE(
          CL_DTYPE_CHAR,
          input,
          SAMPLER,
          (int2)(pos_in_input_block.x + in_pos_in_one_block.x + dilation,
                 pos_in_input_block.y + in_pos_in_one_block.y)),
      (CL_DTYPE4)(0.0f),
      ((in_pos_in_one_block.x + dilation < 0 || in_pos_in_one_block.y < 0 ||
        in_pos_in_one_block.x + dilation >= input_width ||
        in_pos_in_one_block.y >= input_height)));

  inputs[6] = SELECT(
      READ_IMG_TYPE(
          CL_DTYPE_CHAR,
          input,
          SAMPLER,
          (int2)(pos_in_input_block.x + in_pos_in_one_block.x - dilation,
                 pos_in_input_block.y + in_pos_in_one_block.y + dilation)),
      (CL_DTYPE4)(0.0f),
      ((in_pos_in_one_block.x - dilation < 0 ||
        in_pos_in_one_block.y + dilation < 0 ||
        in_pos_in_one_block.x - dilation >= input_width ||
        in_pos_in_one_block.y + dilation >= input_height)));

  inputs[7] = SELECT(
      READ_IMG_TYPE(
          CL_DTYPE_CHAR,
          input,
          SAMPLER,
          (int2)(pos_in_input_block.x + in_pos_in_one_block.x,
                 pos_in_input_block.y + in_pos_in_one_block.y + dilation)),
      (CL_DTYPE4)(0.0f),
      ((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y + dilation < 0 ||
        in_pos_in_one_block.x >= input_width ||
        in_pos_in_one_block.y + dilation >= input_height)));

  inputs[8] = SELECT(
      READ_IMG_TYPE(
          CL_DTYPE_CHAR,
          input,
          SAMPLER,
          (int2)(pos_in_input_block.x + in_pos_in_one_block.x + dilation,
                 pos_in_input_block.y + in_pos_in_one_block.y + dilation)),
      (CL_DTYPE4)(0.0f),
      ((in_pos_in_one_block.x + dilation < 0 ||
        in_pos_in_one_block.y + dilation < 0 ||
        in_pos_in_one_block.x + dilation >= input_width ||
        in_pos_in_one_block.y + dilation >= input_height)));

  CL_DTYPE4 filters_0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(filter_x, filter_y));
  CL_DTYPE4 filters_1 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, filter, SAMPLER, (int2)(filter_x + 1, filter_y));
  CL_DTYPE4 filters_2 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, filter, SAMPLER, (int2)(filter_x + 2, filter_y));
  CL_DTYPE4 filters_3 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, filter, SAMPLER, (int2)(filter_x, filter_y + 1));
  CL_DTYPE4 filters_4 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, filter, SAMPLER, (int2)(filter_x + 1, filter_y + 1));
  CL_DTYPE4 filters_5 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, filter, SAMPLER, (int2)(filter_x + 2, filter_y + 1));
  CL_DTYPE4 filters_6 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, filter, SAMPLER, (int2)(filter_x, filter_y + 2));
  CL_DTYPE4 filters_7 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, filter, SAMPLER, (int2)(filter_x + 1, filter_y + 2));
  CL_DTYPE4 filters_8 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, filter, SAMPLER, (int2)(filter_x + 2, filter_y + 2));

  output += inputs[0] * filters_0;
  output += inputs[1] * filters_1;
  output += inputs[2] * filters_2;
  output += inputs[3] * filters_3;
  output += inputs[4] * filters_4;
  output += inputs[5] * filters_5;
  output += inputs[6] * filters_6;
  output += inputs[7] * filters_7;
  output += inputs[8] * filters_8;

  CL_DTYPE4 alpha0;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
//}
#elif defined(PRELU_ELE)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, output_pos);
//}
#else                     //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
//}
#endif
  output = activation_type4(output, alpha0);

#ifdef SCALE_ACTIVATION
  output = fuse_scale(output, 1.f, 0.f, 0.f);
#endif

  /*

  if (output_pos.x == 0 && output_pos.y == 0) {

      for (int i = 0; i < 9; ++i) {
          CL_DTYPE4 input1 = inputs[i];
          float4 in = (float4)(input1.x, input1.y, input1.z, input1.w);
          printf(" input4[%d]: %v4hlf \n", i, in);
      }
      for (int i = 0; i < 9; ++i) {
          CL_DTYPE4 filters1 = filters[i];
          float4 f = (float4)(filters1.x, filters1.y, filters1.z, filters1.w);
          printf(" weights4[%d]: %v4hlf \n", i, f);
      }
      float4 out = (float4)(output.x, output.y, output.z, output.w);
      printf(" depth wise output output4 = %v4hlf \n", out);
      printf(" pos_in_input_block -x %d \n ", pos_in_input_block.x);
      printf(" pos_in_input_block -y %d \n ", pos_in_input_block.y);
      printf(" in_pos_in_one_block - x %d \n", in_pos_in_one_block.x);
      printf(" in_pos_in_one_block - y %d \n", in_pos_in_one_block.y);
  }

  */

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output);
}

__kernel void depth_conv2d_3x3s1(__private const int ou_ch_blk,
                                 __private const int ou_w_blk,
                                 __private const int ou_nh,
                                 __read_only image2d_t input,
                                 __read_only image2d_t filter,
                                 __read_only image2d_t bias,
                                 __write_only image2d_t output_image,
                                 __private const int stride,
                                 __private const int pad,
                                 __private const int dilation,
                                 __private const int in_ch,
                                 __private const int in_w, /* of one block */
                                 __private const int in_h, /* of one block */
                                 __private const int ou_w,
                                 __private const int ou_h,
                                 __read_only image2d_t prelu_alpha) {
  const int ou_ch_id = get_global_id(0);
  const int ou_w_id = 2 * get_global_id(1);
  const int ou_nh_id = 2 * get_global_id(2);

  int ou_x = mad24(ou_ch_id, ou_w, ou_w_id);
  if (get_global_id(0) >= ou_ch_blk || get_global_id(1) >= ou_w_blk ||
      get_global_id(2) >= ou_nh) {
    return;
  }
#ifdef BIASE_CH
  CL_DTYPE4 output[4];
  output[0] = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(ou_ch_id, 0));
  output[1] = output[0];
  output[2] = output[0];
  output[3] = output[0];
#else
  CL_DTYPE4 output[4] = {0.0f};
#endif

  CL_DTYPE4 input00, input10, input20, input30;
  CL_DTYPE4 input01, input11, input21, input31;
  CL_DTYPE4 input02, input12, input22, input32;
  CL_DTYPE4 input03, input13, input23, input33;

  CL_DTYPE4 filter0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(0, ou_ch_id));
  CL_DTYPE4 filter1 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(1, ou_ch_id));
  CL_DTYPE4 filter2 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(2, ou_ch_id));
  CL_DTYPE4 filter3 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(3, ou_ch_id));
  CL_DTYPE4 filter4 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(4, ou_ch_id));
  CL_DTYPE4 filter5 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(5, ou_ch_id));
  CL_DTYPE4 filter6 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(6, ou_ch_id));
  CL_DTYPE4 filter7 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(7, ou_ch_id));
  CL_DTYPE4 filter8 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(8, ou_ch_id));

  int y0 = select(
      (ou_nh_id - pad), -1, (ou_nh_id - pad) < 0 || (ou_nh_id - pad) >= in_h);
  int y1 = select((ou_nh_id - pad + 1),
                  -1,
                  (ou_nh_id - pad + 1) < 0 || (ou_nh_id - pad + 1) >= in_h);
  int y2 = select((ou_nh_id - pad + 2),
                  -1,
                  (ou_nh_id - pad + 2) < 0 || (ou_nh_id - pad + 2) >= in_h);
  int y3 = select((ou_nh_id - pad + 3),
                  -1,
                  (ou_nh_id - pad + 3) < 0 || (ou_nh_id - pad + 3) >= in_h);
  int x0 = select(ou_ch_id * ou_w + (ou_w_id - pad),
                  -1,
                  (ou_w_id - pad) < 0 || (ou_w_id - pad) >= in_w);
  int x1 = select(ou_ch_id * ou_w + (ou_w_id - pad + 1),
                  -1,
                  (ou_w_id - pad + 1) < 0 || (ou_w_id - pad + 1) >= in_w);
  int x2 = select(ou_ch_id * ou_w + (ou_w_id - pad + 2),
                  -1,
                  (ou_w_id - pad + 2) < 0 || (ou_w_id - pad + 2) >= in_w);
  int x3 = select(ou_ch_id * ou_w + (ou_w_id - pad + 3),
                  -1,
                  (ou_w_id - pad + 3) < 0 || (ou_w_id - pad + 3) >= in_w);

  input00 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x0, y0));
  input10 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x1, y0));
  input20 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x2, y0));
  input30 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x3, y0));
  input01 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x0, y1));
  input11 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x1, y1));
  input21 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x2, y1));
  input31 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x3, y1));
  input02 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x0, y2));
  input12 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x1, y2));
  input22 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x2, y2));
  input32 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x3, y2));
  input03 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x0, y3));
  input13 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x1, y3));
  input23 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x2, y3));
  input33 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(x3, y3));

  output[0] += (filter0 * input00 + filter1 * input10 + filter2 * input20);
  output[1] += (filter0 * input10 + filter1 * input20 + filter2 * input30);
  output[2] += (filter0 * input01 + filter1 * input11 + filter2 * input21);
  output[3] += (filter0 * input11 + filter1 * input21 + filter2 * input31);
  output[0] += (filter3 * input01 + filter4 * input11 + filter5 * input21);
  output[1] += (filter3 * input11 + filter4 * input21 + filter5 * input31);
  output[0] += (filter6 * input02 + filter7 * input12 + filter8 * input22);
  output[1] += (filter6 * input12 + filter7 * input22 + filter8 * input32);
  output[2] += (filter3 * input02 + filter4 * input12 + filter5 * input22);
  output[3] += (filter3 * input12 + filter4 * input22 + filter5 * input32);
  output[2] += (filter6 * input03 + filter7 * input13 + filter8 * input23);
  output[3] += (filter6 * input13 + filter7 * input23 + filter8 * input33);

  CL_DTYPE4 alpha[4];
#ifdef PRELU_CH  //{
  alpha[0] = READ_IMG_TYPE(
      CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(ou_ch_blk_id, 0));
  alpha[1] = alpha[0];
  alpha[2] = alpha[0];
  alpha[3] = alpha[0]
//}
#elif defined(PRELU_ELE)  //{
  alpha[0] = READ_IMG_TYPE(
      CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(ou_x, ou_nh_id));
  if (ou_w_id + 1 < ou_w) {
    alpha[1] = READ_IMG_TYPE(
        CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(ou_x + 1, ou_nh_id));
  }
  if (ou_nh_id + 1 < ou_h) {
    alpha[2] = READ_IMG_TYPE(
        CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(ou_x, ou_nh_id + 1));
  }
  if (ou_nh_id + 1 < ou_h && ou_w_id + 1 < ou_w) {
    alpha[3] = READ_IMG_TYPE(
        CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(ou_x + 1, ou_nh_id + 1));
  }
//}
#else                     //{
  alpha[0] = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha[0].y = alpha[0].x;
  alpha[0].z = alpha[0].x;
  alpha[0].w = alpha[0].x;
  alpha[1] = alpha[0];
  alpha[2] = alpha[0];
  alpha[3] = alpha[0];
//}
#endif
      output[0] = activation_type4(output[0], alpha[0]);
  output[1] = activation_type4(output[1], alpha[1]);
  output[2] = activation_type4(output[2], alpha[2]);
  output[3] = activation_type4(output[3], alpha[3]);

#ifdef SCALE_ACTIVATION
  output[0] = fuse_scale(output[0], 1.f, 0.f, 0.f);
  output[1] = fuse_scale(output[1], 1.f, 0.f, 0.f);
  output[2] = fuse_scale(output[2], 1.f, 0.f, 0.f);
  output[3] = fuse_scale(output[3], 1.f, 0.f, 0.f);
#endif

  WRITE_IMG_TYPE(
      CL_DTYPE_CHAR, output_image, (int2)(ou_x, ou_nh_id), output[0]);
  if (ou_w_id + 1 < ou_w) {
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output_image, (int2)(ou_x + 1, ou_nh_id), output[1]);
  }
  if (ou_nh_id + 1 < ou_h) {
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output_image, (int2)(ou_x, ou_nh_id + 1), output[2]);
  }
  if (ou_w_id + 1 < ou_w && ou_nh_id + 1 < ou_h) {
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output_image, (int2)(ou_x + 1, ou_nh_id + 1), output[3]);
  }
}
