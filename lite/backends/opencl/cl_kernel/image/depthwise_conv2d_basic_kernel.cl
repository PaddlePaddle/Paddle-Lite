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

__kernel void depth_conv2d_common(
    __private const int global_size_dim0,  // (out_c + 1) / 4
    __private const int global_size_dim1,  // (out_w + 1) / 4
    __private const int global_size_dim2,  // out_n * out_h
    __read_only image2d_t input,
    __read_only image2d_t filter,
    __read_only image2d_t bias,
    __write_only image2d_t output,
    __private const int stride_w,
    __private const int stride_h,
    __private const int pad_h,
    __private const int pad_w,
    __private const int dilation_w,
    __private const int dilation_h,
    __private const int input_width,
    __private const int input_height,
    __private const int output_width,
    __private const int output_height,
    __private const int filter_width,
    __private const int filter_height,
    __read_only image2d_t prelu_alpha) {
  const int out_c_blk = get_global_id(0);  // [0, (C+3)/4)
  const int out_w_blk = get_global_id(1);  // [0, (W+3)/4)
  const int out_nh = get_global_id(2);     // [0, N*H)

  if (out_c_blk >= global_size_dim0 || out_w_blk >= global_size_dim1 ||
      out_nh >= global_size_dim2) {
    return;
  }

  int2 out_pos = (int2)(out_c_blk * global_size_dim1 + out_w_blk, out_nh);

#ifdef BIASE_CH
  CL_DTYPE4 out0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c_blk, 0));
  CL_DTYPE4 out1 = out0;
  CL_DTYPE4 out2 = out0;
  CL_DTYPE4 out3 = out0;
#elif defined(BIASE_ELE)
  CL_DTYPE4 out0 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_pos.x + 0, out_pos.y));
  CL_DTYPE4 out1 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_pos.x + 1, out_pos.y));
  CL_DTYPE4 out2 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_pos.x + 2, out_pos.y));
  CL_DTYPE4 out3 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_pos.x + 3, out_pos.y));
#else
  CL_DTYPE4 out0 = 0;
  CL_DTYPE4 out1 = 0;
  CL_DTYPE4 out2 = 0;
  CL_DTYPE4 out3 = 0;
#endif

  const int in_w_offset0 = mad24(
      out_w_blk, stride_w << 2, -pad_w);  // out_w_blk * stride_w * 4 - pad_w
  const int in_w_offset1 = in_w_offset0 + stride_w;
  const int in_w_offset2 = in_w_offset1 + stride_w;
  const int in_w_offset3 = in_w_offset2 + stride_w;

  int in_h_idx = mad24(
      out_nh % output_height, stride_h, -pad_h);  // out_nh % output_height *
                                                  // stride_h - pad_h. height
                                                  // index of one input feature
                                                  // map
  const int batch_idx = out_nh / output_height;
  const int in_c_blk = out_c_blk;

  const int in_x_base = mul24(in_c_blk, input_width);  // in_c_blk * input_width
  for (int kh = 0; kh < filter_height; kh++) {
    int in_pos_y = select(in_h_idx + batch_idx * input_height,
                          -1,
                          (in_h_idx < 0 || in_h_idx >= input_height));
    in_h_idx += dilation_h;
    for (int kw = 0; kw < filter_width; kw++) {
      CL_DTYPE4 in0, in1, in2, in3;

      int base = mul24(kw, dilation_w);
      int in_w_idx =
          in_w_offset0 + base;  // width index of one input feature map
      int in_pos_x = select(
          in_x_base + in_w_idx, -1, (in_w_idx < 0 || in_w_idx >= input_width));
      in0 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_pos_x, in_pos_y));

      in_w_idx = in_w_offset1 + base;
      in_pos_x = select(
          in_x_base + in_w_idx, -1, (in_w_idx < 0 || in_w_idx >= input_width));
      in1 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_pos_x, in_pos_y));

      in_w_idx = in_w_offset2 + base;
      in_pos_x = select(
          in_x_base + in_w_idx, -1, (in_w_idx < 0 || in_w_idx >= input_width));
      in2 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_pos_x, in_pos_y));

      in_w_idx = in_w_offset3 + base;
      in_pos_x = select(
          in_x_base + in_w_idx, -1, (in_w_idx < 0 || in_w_idx >= input_width));
      in3 = READ_IMG_TYPE(
          CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_pos_x, in_pos_y));

      int filter_idx = mad24(kh, filter_width, kw);
      CL_DTYPE4 weights = READ_IMG_TYPE(
          CL_DTYPE_CHAR, filter, SAMPLER, (int2)(filter_idx, in_c_blk));

      out0 = mad(in0, weights, out0);
      out1 = mad(in1, weights, out1);
      out2 = mad(in2, weights, out2);
      out3 = mad(in3, weights, out3);
    }
  }

  const int out_w_blk4 = out_w_blk << 2;  // [0, W)
  const int remain = output_width - out_w_blk4;
  const int out_pos_x = mad24(out_c_blk, output_width, out_w_blk4);

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3;
#ifdef PRELU_CH  //{
  alpha0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c_blk, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                         prelu_alpha,
                         SAMPLER,
                         (int2)(out_pos_x, out_nh % output_height));
  if (out_w_blk4 + 1 < output_width) {
    alpha1 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_pos_x + 1, out_nh % output_height));
  }
  if (out_w_blk4 + 2 < output_width) {
    alpha2 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_pos_x + 2, out_nh % output_height));
  }
  if (out_w_blk4 + 3 < output_width) {
    alpha3 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                           prelu_alpha,
                           SAMPLER,
                           (int2)(out_pos_x + 3, out_nh % output_height));
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
  out0 = activation_type4(out0, alpha0);
  out1 = activation_type4(out1, alpha1);
  out2 = activation_type4(out2, alpha2);
  out3 = activation_type4(out3, alpha3);

#ifdef SCALE_ACTIVATION
  out0 = fuse_scale(out0, 1.f, 0.f, 0.f);
  out1 = fuse_scale(out1, 1.f, 0.f, 0.f);
  out2 = fuse_scale(out2, 1.f, 0.f, 0.f);
  out3 = fuse_scale(out3, 1.f, 0.f, 0.f);
#endif

  if (remain >= 4) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_pos_x, out_nh), out0);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_pos_x + 1, out_nh), out1);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_pos_x + 2, out_nh), out2);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_pos_x + 3, out_nh), out3);
  } else if (remain == 3) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_pos_x, out_nh), out0);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_pos_x + 1, out_nh), out1);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_pos_x + 2, out_nh), out2);
  } else if (remain == 2) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_pos_x, out_nh), out0);
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_pos_x + 1, out_nh), out1);
  } else if (remain == 1) {
    WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(out_pos_x, out_nh), out0);
  }
}

__kernel void depth_conv2d(__private const int global_size_dim0,
                           __private const int global_size_dim1,
                           __private const int global_size_dim2,
                           __read_only image2d_t input,
                           __read_only image2d_t filter,
                           __read_only image2d_t bias,
                           __write_only image2d_t output_image,
                           __private const int stride_w,
                           __private const int stride_h,
                           __private const int offset_w,
                           __private const int offset_h,
                           __private const int dilation_w,
                           __private const int dilation_h,
                           __private const int input_width,
                           __private const int input_height,
                           __private const int output_width,
                           __private const int output_height,
                           __private const int filter_width,
                           __private const int filter_height,
                           __read_only image2d_t prelu_alpha
#ifdef ELT_FUSE
                           ,
                           __read_only image2d_t second_input_image
#endif
                           ) {
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
      ouput_pos_in_one_block * stride_xy + (int2)(offset_w, offset_h);
#ifdef BIASE_CH
  CL_DTYPE4 output =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c, 0));
#elif defined(BIASE_ELE)
  CL_DTYPE4 output = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, output_pos);
#else
  CL_DTYPE4 output = 0.0f;
#endif

  int2 pos_in_input_block =
      (int2)(out_c * input_width, batch_index * input_height);
  int2 pos_in_filter_block = (int2)(out_c * filter_width, 0);
  int filter_x = pos_in_filter_block.x;
  int filter_y = pos_in_filter_block.y;
  int input_x_base = pos_in_input_block.x + in_pos_in_one_block.x;
  int input_y_base = pos_in_input_block.y + in_pos_in_one_block.y;

  int2 align = {filter_width / 2, filter_height / 2};
  for (int fy = 0; fy < filter_height; ++fy) {
    int y_off = fy * dilation_h - align.y;
    for (int fx = 0; fx < filter_width; ++fx) {
      int x_off = fx * dilation_w - align.x;

      CL_DTYPE4 in = SELECT(
          READ_IMG_TYPE(CL_DTYPE_CHAR,
                        input,
                        SAMPLER,
                        (int2)(input_x_base + x_off, input_y_base + y_off)),
          (CL_DTYPE4)(0.0f),
          ((in_pos_in_one_block.x + x_off < 0 ||
            in_pos_in_one_block.y + y_off < 0 ||
            in_pos_in_one_block.x + x_off >= input_width ||
            in_pos_in_one_block.y + y_off >= input_height)));

      CL_DTYPE4 f = READ_IMG_TYPE(
          CL_DTYPE_CHAR, filter, SAMPLER, (int2)(filter_x + fx, filter_y + fy));
      output += in * f;
    }
  }

  CL_DTYPE4 alpha0;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c, 0));
//}
#elif defined(PRELU_ELE)  //{
  alpha0 = READ_IMG_TYPE(
      CL_DTYPE_CHAR,
      prelu_alpha,
      SAMPLER,
      (int2)(out_c * global_size_dim1 + out_w, out_nh % output_height));
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

#ifdef ELT_FUSE
  elt_fuse_func_wrapper(second_input_image, output_pos, &output);
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output);
}
