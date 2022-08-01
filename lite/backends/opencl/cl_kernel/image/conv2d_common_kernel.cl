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

__kernel void conv2d_common(__private const int global_size_dim0,
                            __private const int global_size_dim1,
                            __private const int global_size_dim2,
                            __read_only image2d_t input,
                            __read_only image2d_t filter,
                            __read_only image2d_t bias,
                            __write_only image2d_t output,
                            __private const int input_width,
                            __private const int input_height,
                            __private const int in_channel_block_length,
                            __private const int output_width,
                            __private const int output_height,
                            __private const int kernel_width,
                            __private const int kernel_height,
                            __private const int stride_width,
                            __private const int stride_height,
                            __private const int padding_width,
                            __private const int padding_height,
                            __private const int dilation_width,
                            __private const int dilation_height,
                            __read_only image2d_t prelu_alpha) {
  const int out_channel_block_idx = get_global_id(0);
  const int out_width_block_idx = get_global_id(1);
  const int output_bh_idx = get_global_id(2);

  if (out_channel_block_idx >= global_size_dim0 ||
      out_width_block_idx >= global_size_dim1 ||
      output_bh_idx >= global_size_dim2) {
    return;
  }

  int out_w_base_id = out_channel_block_idx * output_width;
  int out_w_id0 = out_width_block_idx;
  int out_w_id1 = out_w_id0 + global_size_dim1;
  int out_w_id2 = out_w_id1 + global_size_dim1;
  int out_w_id3 = out_w_id2 + global_size_dim1;
#ifdef BIASE_CH
  CL_DTYPE4 out_base = READ_IMG_TYPE(
      CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_channel_block_idx, 0));
  CL_DTYPE4 out0 = out_base;
  CL_DTYPE4 out1 = out0;
  CL_DTYPE4 out2 = out0;
  CL_DTYPE4 out3 = out0;
#elif defined(BIASE_ELE)
  CL_DTYPE4 out0, out1, out2, out3;
  out0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                       bias,
                       SAMPLER,
                       (int2)(out_w_base_id + out_w_id0, output_bh_idx));
  if (out_w_id1 < output_width) {
    out1 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                         bias,
                         SAMPLER,
                         (int2)(out_w_base_id + out_w_id1, output_bh_idx));
  }
  if (out_w_id2 < output_width) {
    out2 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                         bias,
                         SAMPLER,
                         (int2)(out_w_base_id + out_w_id2, output_bh_idx));
  }
  if (out_w_id3 < output_width) {
    out3 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                         bias,
                         SAMPLER,
                         (int2)(out_w_base_id + out_w_id3, output_bh_idx));
  }
#else
  CL_DTYPE4 out_base = (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f);
  CL_DTYPE4 out0 = out_base;
  CL_DTYPE4 out1 = out0;
  CL_DTYPE4 out2 = out0;
  CL_DTYPE4 out3 = out0;
#endif

  int in_width0 = mad24(out_width_block_idx, stride_width << 2, -padding_width);
  int in_width1 = in_width0 + stride_width;
  int in_width2 = in_width0 + stride_width * 2;
  int in_width3 = in_width0 + stride_width * 3;

  const int height_start =
      mad24((output_bh_idx % output_height), stride_height, -padding_height);
  int in_height_start =
      mad24(select(0,
                   (-height_start + dilation_height - 1) / dilation_height,
                   height_start < 0),
            dilation_height,
            height_start);
  int in_height_end =
      min(mad24(kernel_height, dilation_height, height_start), input_height);

  const int batch_idx = mul24((output_bh_idx / output_height), input_height);
  const int filter_h_idx =
      mul24(out_channel_block_idx, mul24(kernel_width, kernel_height)) +
      mul24(select(0,
                   (-height_start + dilation_height - 1) / dilation_height,
                   height_start < 0),
            kernel_width);
  CL_DTYPE4 in0, in1, in2, in3;
  CL_DTYPE4 filter0, filter1, filter2, filter3;
  for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length;
       ++input_c_block_idx) {
    const int in_idx = mul24(input_c_block_idx, input_width);
    int filter_x_idx = input_c_block_idx << 2;
    int filter_y_idx = filter_h_idx;
    for (int iy = in_height_start; iy < in_height_end; iy += dilation_height) {
      int in_hb_value = iy + batch_idx;
      for (int w = 0; w < kernel_width; w++) {
        int input_w_base = mul24(w, dilation_width);

        int in_width_value0 = in_width0 + input_w_base;
        in_width_value0 =
            select(in_idx + in_width_value0,
                   -1,
                   (in_width_value0 < 0 || in_width_value0 >= input_width));
        in0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                            input,
                            SAMPLER,
                            (int2)(in_width_value0, in_hb_value));

        int in_width_value1 = in_width1 + input_w_base;
        in_width_value1 =
            select(in_idx + in_width_value1,
                   -1,
                   (in_width_value1 < 0 || in_width_value1 >= input_width));
        in1 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                            input,
                            SAMPLER,
                            (int2)(in_width_value1, in_hb_value));

        int in_width_value2 = in_width2 + input_w_base;
        in_width_value2 =
            select(in_idx + in_width_value2,
                   -1,
                   (in_width_value2 < 0 || in_width_value2 >= input_width));
        in2 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                            input,
                            SAMPLER,
                            (int2)(in_width_value2, in_hb_value));

        int in_width_value3 = in_width3 + input_w_base;
        in_width_value3 =
            select(in_idx + in_width_value3,
                   -1,
                   (in_width_value3 < 0 || in_width_value3 >= input_width));
        in3 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                            input,
                            SAMPLER,
                            (int2)(in_width_value3, in_hb_value));

        filter0 = READ_IMG_TYPE(
            CL_DTYPE_CHAR, filter, SAMPLER, (int2)(filter_x_idx, filter_y_idx));
        filter1 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                filter,
                                SAMPLER,
                                (int2)(filter_x_idx + 1, filter_y_idx));
        filter2 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                filter,
                                SAMPLER,
                                (int2)(filter_x_idx + 2, filter_y_idx));
        filter3 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                filter,
                                SAMPLER,
                                (int2)(filter_x_idx + 3, filter_y_idx++));

        out0 = mad(in0.x, filter0, out0);
        out0 = mad(in0.y, filter1, out0);
        out0 = mad(in0.z, filter2, out0);
        out0 = mad(in0.w, filter3, out0);

        out1 = mad(in1.x, filter0, out1);
        out1 = mad(in1.y, filter1, out1);
        out1 = mad(in1.z, filter2, out1);
        out1 = mad(in1.w, filter3, out1);

        out2 = mad(in2.x, filter0, out2);
        out2 = mad(in2.y, filter1, out2);
        out2 = mad(in2.z, filter2, out2);
        out2 = mad(in2.w, filter3, out2);

        out3 = mad(in3.x, filter0, out3);
        out3 = mad(in3.y, filter1, out3);
        out3 = mad(in3.z, filter2, out3);
        out3 = mad(in3.w, filter3, out3);
      }
    }
  }
  const int out_x_base = mul24(out_channel_block_idx, output_width);
  int out_x_idx = out_width_block_idx << 2;

  const int remain = output_width - out_x_idx;
  int output_w_idx = out_x_base + out_x_idx;

  CL_DTYPE4 alpha0, alpha1, alpha2, alpha3;
#ifdef PRELU_CH  //{
  alpha0 = READ_IMG_TYPE(
      CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_channel_block_idx, 0));
  alpha1 = alpha0;
  alpha2 = alpha0;
  alpha3 = alpha0;
//}
#elif defined(PRELU_ELE)  //{
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                         prelu_alpha,
                         SAMPLER,
                         (int2)(output_w_idx, output_bh_idx % output_height));
  if (out_x_idx + 1 < output_width) {
    alpha1 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prelu_alpha,
                      SAMPLER,
                      (int2)(output_w_idx + 1, output_bh_idx % output_height));
  }
  if (out_x_idx + 2 < output_width) {
    alpha2 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prelu_alpha,
                      SAMPLER,
                      (int2)(output_w_idx + 2, output_bh_idx % output_height));
  }
  if (out_x_idx + 3 < output_width) {
    alpha3 =
        READ_IMG_TYPE(CL_DTYPE_CHAR,
                      prelu_alpha,
                      SAMPLER,
                      (int2)(output_w_idx + 3, output_bh_idx % output_height));
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
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx, output_bh_idx), out0);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx + 1, output_bh_idx), out1);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx + 2, output_bh_idx), out2);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx + 3, output_bh_idx), out3);
  } else if (remain == 3) {
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx, output_bh_idx), out0);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx + 1, output_bh_idx), out1);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx + 2, output_bh_idx), out2);
  } else if (remain == 2) {
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx, output_bh_idx), out0);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx + 1, output_bh_idx), out1);
  } else if (remain == 1) {
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx, output_bh_idx), out0);
  }
}

__kernel void conv2d_common_mul_group(__private const int global_size_dim0,
                                      __private const int global_size_dim1,
                                      __private const int global_size_dim2,
                                      __read_only image2d_t input,
                                      __read_only image2d_t filter,
                                      __write_only image2d_t output,
                                      __private const int input_c,
                                      __private const int input_height,
                                      __private const int input_width,
                                      __private const int output_c,
                                      __private const int output_height,
                                      __private const int output_width,
                                      __private const int kernel_width,
                                      __private const int kernel_height,
                                      __private const int stride_width,
                                      __private const int stride_height,
                                      __private const int padding_width,
                                      __private const int padding_height,
                                      __private const int dilation_width,
                                      __private const int dilation_height,
                                      __private const int group) {
  const int out_channel_block_idx = get_global_id(0);
  const int out_width_block_idx = get_global_id(1);
  const int output_bh_idx = get_global_id(2);
  const int output_single_group_lenth_c4 = (output_c / group + 3) / 4;
  const int group_id = out_channel_block_idx / output_single_group_lenth_c4;
  const int single_group_line_id =
      out_channel_block_idx % output_single_group_lenth_c4;
  const int input_group_stride = (input_c / group + 3) / 4 * input_width;
  const int in_single_group_length_c4 = (input_c / group + 3) / 4;

  if (out_channel_block_idx >= global_size_dim0 ||
      out_width_block_idx >= global_size_dim1 ||
      output_bh_idx >= global_size_dim2) {
    return;
  }

  CL_DTYPE4 out_base = (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f);
  CL_DTYPE4 out0 = out_base;
  CL_DTYPE4 out1 = out0;
  CL_DTYPE4 out2 = out0;
  CL_DTYPE4 out3 = out0;

  int in_width0 = mad24(out_width_block_idx, stride_width << 2, -padding_width);
  int in_width1 = in_width0 + stride_width;
  int in_width2 = in_width0 + stride_width * 2;
  int in_width3 = in_width0 + stride_width * 3;

  const int height_start =
      mad24((output_bh_idx % output_height), stride_height, -padding_height);
  int in_height_start =
      mad24(select(0,
                   (-height_start + dilation_height - 1) / dilation_height,
                   height_start < 0),
            dilation_height,
            height_start);
  int in_height_end =
      min(mad24(kernel_height, dilation_height, height_start), input_height);

  const int batch_idx = mul24((output_bh_idx / output_height), input_height);
  const int filter_group_stride =
      output_single_group_lenth_c4 * mul24(kernel_width, kernel_height);
  const int filter_h_idx =
      filter_group_stride * group_id +
      single_group_line_id * mul24(kernel_width, kernel_height) +
      mul24(select(0,
                   (-height_start + dilation_height - 1) / dilation_height,
                   height_start < 0),
            kernel_width);

  CL_DTYPE4 in0, in1, in2, in3;
  CL_DTYPE4 filter0, filter1, filter2, filter3;
  for (int input_c_block_idx = 0; input_c_block_idx < in_single_group_length_c4;
       ++input_c_block_idx) {
    const int in_idx =
        group_id * input_group_stride + mul24(input_c_block_idx, input_width);
    int filter_x_idx = input_c_block_idx << 2;
    int filter_y_idx = filter_h_idx;

    for (int iy = in_height_start; iy < in_height_end; iy += dilation_height) {
      int in_hb_value = iy + batch_idx;
      for (int w = 0; w < kernel_width; w++) {
        int input_w_base = mul24(w, dilation_width);

        int in_width_value0 = in_width0 + input_w_base;
        in_width_value0 =
            select(in_idx + in_width_value0,
                   -1,
                   (in_width_value0 < 0 || in_width_value0 >= input_width));
        in0 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                            input,
                            SAMPLER,
                            (int2)(in_width_value0, in_hb_value));

        int in_width_value1 = in_width1 + input_w_base;
        in_width_value1 =
            select(in_idx + in_width_value1,
                   -1,
                   (in_width_value1 < 0 || in_width_value1 >= input_width));
        in1 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                            input,
                            SAMPLER,
                            (int2)(in_width_value1, in_hb_value));

        int in_width_value2 = in_width2 + input_w_base;
        in_width_value2 =
            select(in_idx + in_width_value2,
                   -1,
                   (in_width_value2 < 0 || in_width_value2 >= input_width));
        in2 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                            input,
                            SAMPLER,
                            (int2)(in_width_value2, in_hb_value));

        int in_width_value3 = in_width3 + input_w_base;
        in_width_value3 =
            select(in_idx + in_width_value3,
                   -1,
                   (in_width_value3 < 0 || in_width_value3 >= input_width));
        in3 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                            input,
                            SAMPLER,
                            (int2)(in_width_value3, in_hb_value));

        filter0 = READ_IMG_TYPE(
            CL_DTYPE_CHAR, filter, SAMPLER, (int2)(filter_x_idx, filter_y_idx));
        filter1 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                filter,
                                SAMPLER,
                                (int2)(filter_x_idx + 1, filter_y_idx));
        filter2 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                filter,
                                SAMPLER,
                                (int2)(filter_x_idx + 2, filter_y_idx));
        filter3 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                                filter,
                                SAMPLER,
                                (int2)(filter_x_idx + 3, filter_y_idx++));

        out0 = mad(in0.x, filter0, out0);
        out0 = mad(in0.y, filter1, out0);
        out0 = mad(in0.z, filter2, out0);
        out0 = mad(in0.w, filter3, out0);

        out1 = mad(in1.x, filter0, out1);
        out1 = mad(in1.y, filter1, out1);
        out1 = mad(in1.z, filter2, out1);
        out1 = mad(in1.w, filter3, out1);

        out2 = mad(in2.x, filter0, out2);
        out2 = mad(in2.y, filter1, out2);
        out2 = mad(in2.z, filter2, out2);
        out2 = mad(in2.w, filter3, out2);

        out3 = mad(in3.x, filter0, out3);
        out3 = mad(in3.y, filter1, out3);
        out3 = mad(in3.z, filter2, out3);
        out3 = mad(in3.w, filter3, out3);
      }
    }
  }
  const int out_x_base = mul24(out_channel_block_idx, output_width);
  int out_x_idx = out_width_block_idx << 2;

  const int remain = output_width - out_x_idx;
  int output_w_idx = out_x_base + out_x_idx;
  if (remain >= 4) {
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx, output_bh_idx), out0);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx + 1, output_bh_idx), out1);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx + 2, output_bh_idx), out2);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx + 3, output_bh_idx), out3);
  } else if (remain == 3) {
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx, output_bh_idx), out0);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx + 1, output_bh_idx), out1);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx + 2, output_bh_idx), out2);
  } else if (remain == 2) {
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx, output_bh_idx), out0);
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx + 1, output_bh_idx), out1);
  } else if (remain == 1) {
    WRITE_IMG_TYPE(
        CL_DTYPE_CHAR, output, (int2)(output_w_idx, output_bh_idx), out0);
  }
}

__kernel void mul_groups_fill0(__read_only image2d_t input,
                               __write_only image2d_t output,
                               __private const int out_img_w,
                               __private const int in_c,
                               __private const int in_w,
                               __private const int groups) {
  const int pos_x = get_global_id(0);
  const int pos_y = get_global_id(1);

  const int w_id = pos_x % out_img_w;
  const int in_single_group_len = in_c / groups;
  const int out_c4_id = pos_x / out_img_w;
  const int single_group_c4_len = (in_c / groups + 3) / 4;
  const int group_id = out_c4_id / single_group_c4_len;
  const int single_group_c_id = out_c4_id % single_group_c4_len * 4;
  const int in_c_id_0 = group_id * (in_c / groups) + single_group_c_id;

  CL_DTYPE4 in0_4 = 0.f;
  CL_DTYPE4 in1_4 = 0.f;
  CL_DTYPE4 in2_4 = 0.f;
  CL_DTYPE4 in3_4 = 0.f;
  CL_DTYPE in0 = 0.f;
  CL_DTYPE in1 = 0.f;
  CL_DTYPE in2 = 0.f;
  CL_DTYPE in3 = 0.f;
  if (single_group_c_id < in_single_group_len) {
    int remain = (in_c_id_0) % 4;
    in0_4 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                          input,
                          SAMPLER,
                          (int2)((in_c_id_0) / 4 * in_w + w_id, pos_y));
    in0 = remain == 0
              ? in0_4.s0
              : (remain == 1 ? in0_4.s1 : (remain == 2 ? in0_4.s2 : in0_4.s3));
  }
  if (single_group_c_id + 1 < in_single_group_len) {
    int remain = (in_c_id_0 + 1) % 4;
    in1_4 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                          input,
                          SAMPLER,
                          (int2)((in_c_id_0 + 1) / 4 * in_w + w_id, pos_y));
    in1 = remain == 0
              ? in1_4.s0
              : (remain == 1 ? in1_4.s1 : (remain == 2 ? in1_4.s2 : in1_4.s3));
  }
  if (single_group_c_id + 2 < in_single_group_len) {
    int remain = (in_c_id_0 + 2) % 4;
    in2_4 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                          input,
                          SAMPLER,
                          (int2)((in_c_id_0 + 2) / 4 * in_w + w_id, pos_y));
    in2 = remain == 0
              ? in2_4.s0
              : (remain == 1 ? in2_4.s1 : (remain == 2 ? in2_4.s2 : in2_4.s3));
  }
  if (single_group_c_id + 3 < in_single_group_len) {
    int remain = (in_c_id_0 + 3) % 4;
    in3_4 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                          input,
                          SAMPLER,
                          (int2)((in_c_id_0 + 3) / 4 * in_w + w_id, pos_y));
    in3 = remain == 0
              ? in3_4.s0
              : (remain == 1 ? in3_4.s1 : (remain == 2 ? in3_4.s2 : in3_4.s3));
  }

  CL_DTYPE4 out = (CL_DTYPE4)(in0, in1, in2, in3);
  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(pos_x, pos_y), out);
}

__kernel void mul_groups_cut0_bias(__read_only image2d_t input,
                                   __read_only image2d_t bias,
                                   __write_only image2d_t output,
                                   __private const int out_img_w,
                                   __private const int out_c,
                                   __private const int out_c_align_group4,
                                   __private const int groups,
                                   __read_only image2d_t prelu_alpha
#ifdef ELT_FUSE
                                   ,
                                   __read_only image2d_t second_input_image
#endif
                                   ) {
  const int pos_x = get_global_id(0);
  const int pos_y = get_global_id(1);

  const int w_id = pos_x % out_img_w;
  const int in_single_group_len = out_c / groups;
  const int out_c4_id = pos_x / out_img_w;
  const int out_c_id0 = out_c4_id * 4;
  const int out_c_id1 = out_c4_id * 4 + 1;
  const int out_c_id2 = out_c4_id * 4 + 2;
  const int out_c_id3 = out_c4_id * 4 + 3;

  const int single_group_c4_len = (out_c / groups + 3) / 4;

  const int group_id0 = out_c_id0 / in_single_group_len;
  const int group_id1 = out_c_id1 / in_single_group_len;
  const int group_id2 = out_c_id2 / in_single_group_len;
  const int group_id3 = out_c_id3 / in_single_group_len;

  const int in_c_id_0 =
      group_id0 * single_group_c4_len * 4 + out_c_id0 % in_single_group_len;
  const int in_c_id_1 =
      group_id1 * single_group_c4_len * 4 + out_c_id1 % in_single_group_len;
  const int in_c_id_2 =
      group_id2 * single_group_c4_len * 4 + out_c_id2 % in_single_group_len;
  const int in_c_id_3 =
      group_id3 * single_group_c4_len * 4 + out_c_id3 % in_single_group_len;

  CL_DTYPE4 in0_4 = 0.f;
  CL_DTYPE4 in1_4 = 0.f;
  CL_DTYPE4 in2_4 = 0.f;
  CL_DTYPE4 in3_4 = 0.f;
  CL_DTYPE in0 = 0.f;
  CL_DTYPE in1 = 0.f;
  CL_DTYPE in2 = 0.f;
  CL_DTYPE in3 = 0.f;
  if (in_c_id_0 < out_c_align_group4) {
    int remain = (in_c_id_0) % 4;
    in0_4 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                          input,
                          SAMPLER,
                          (int2)((in_c_id_0) / 4 * out_img_w + w_id, pos_y));
    in0 = remain == 0
              ? in0_4.s0
              : (remain == 1 ? in0_4.s1 : (remain == 2 ? in0_4.s2 : in0_4.s3));
  }
  if (in_c_id_1 < out_c_align_group4) {
    int remain = (in_c_id_1) % 4;
    in1_4 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                          input,
                          SAMPLER,
                          (int2)((in_c_id_1) / 4 * out_img_w + w_id, pos_y));
    in1 = remain == 0
              ? in1_4.s0
              : (remain == 1 ? in1_4.s1 : (remain == 2 ? in1_4.s2 : in1_4.s3));
  }
  if (in_c_id_2 < out_c_align_group4) {
    int remain = (in_c_id_2) % 4;
    in2_4 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                          input,
                          SAMPLER,
                          (int2)((in_c_id_2) / 4 * out_img_w + w_id, pos_y));
    in2 = remain == 0
              ? in2_4.s0
              : (remain == 1 ? in2_4.s1 : (remain == 2 ? in2_4.s2 : in2_4.s3));
  }
  if (in_c_id_3 < out_c_align_group4) {
    int remain = (in_c_id_3) % 4;
    in3_4 = READ_IMG_TYPE(CL_DTYPE_CHAR,
                          input,
                          SAMPLER,
                          (int2)((in_c_id_3) / 4 * out_img_w + w_id, pos_y));
    in3 = remain == 0
              ? in3_4.s0
              : (remain == 1 ? in3_4.s1 : (remain == 2 ? in3_4.s2 : in3_4.s3));
  }

  CL_DTYPE4 out = (CL_DTYPE4)(in0, in1, in2, in3);

// add bias
#ifdef BIASE_CH
  out = out + READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c4_id, 0));
#elif defined(BIASE_ELE)
  out = out + READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(pos_x, pos_y));
#endif

  // fuse activation
  CL_DTYPE4 alpha0;
#ifdef PRELU_CH
  alpha0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(out_c4_id, 0));
#elif defined(PRELU_ELE)
  alpha0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(pos_x, pos_y));
#elif defined(PRELU_ALL)
  alpha0 = READ_IMG_TYPE(CL_DTYPE_CHAR, prelu_alpha, SAMPLER, (int2)(0, 0));
  alpha0.y = alpha0.x;
  alpha0.z = alpha0.x;
  alpha0.w = alpha0.x;
#endif
  out = activation_type4(out, alpha0);

#ifdef SCALE_ACTIVATION
  out = fuse_scale(out, 1.f, 0.f, 0.f);
#endif

#ifdef ELT_FUSE
  elt_fuse_func_wrapper(second_input_image, (int2)(pos_x, pos_y), &out);
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, (int2)(pos_x, pos_y), out);
}
