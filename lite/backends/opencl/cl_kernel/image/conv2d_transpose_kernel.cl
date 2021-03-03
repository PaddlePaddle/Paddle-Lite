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

#include "cl_common.h"


__kernel void conv2d_transpose(__private const int global_size_dim0, // (out_c + 1) / 4
                               __private const int global_size_dim1, // out_w
                               __private const int global_size_dim2, // out_n * out_h
                               __read_only image2d_t input,
                               __read_only image2d_t filter,
                               __read_only image2d_t bias,
                               __write_only image2d_t output,
                               __private const int2 input_shape,
                               __private const int2 output_shape,
                               __private const int2 stride_shape,
                               __private const int2 align_shape,
                               __private const int2 padding_shape,
                               __private const int2 kernel_shape,
                               __private const int kernel_size,
                               __private const int input_c_blks) {

  const int out_c_blk_idx = get_global_id(0); // [0, (C+3)/4)
  const int out_w_idx = get_global_id(1); // [0, W)
  const int out_nh_idx = get_global_id(2); // [0, N*H)

  CL_DTYPE4 d = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(0, 0));
  float4 f = (float4)(d.x, d.y, d.z, d.w);
  if (out_c_blk_idx == 0 && out_w_idx == 1 && out_nh_idx == 1) {
    printf("in conv2d_transpose cl kernel\n");
    // printf("%v4hlf \n", f);
    printf("%d\n", input_c_blks);
  }

  if (out_c_blk_idx >= global_size_dim0 || out_w_idx >= global_size_dim1
      || out_nh_idx >= global_size_dim2) {
    return;
  }

  int2 out_pos = (int2)(out_c_blk_idx * global_size_dim1 + out_w_idx, out_nh_idx);

#ifdef BIASE_CH
  CL_DTYPE4 out0 =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, SAMPLER, (int2)(out_c_blk_idx, 0));
#else
  CL_DTYPE4 out0 = 0.f;
#endif

  const int out_n_idx = out_nh_idx / output_shape.y;
  const int out_h_idx = out_nh_idx % output_shape.y;

  int kernel_start_x = max(0, (out_w_idx + align_shape.x) / stride_shape.x);
  int kernel_start_y = max(0, (out_h_idx + align_shape.y) / stride_shape.y);
  int deal_kernel_width  = kernel_shape.x - mad24(kernel_start_x, stride_shape.x, padding_shape.x) + out_w_idx - 1;
  int deal_kernel_height = kernel_shape.y - mad24(kernel_start_y, stride_shape.y, padding_shape.y) + out_h_idx - 1;

  int kernel_x_0, kernel_x_1, kernel_x_2, kernel_x_3, kernel_y;
  CL_DTYPE4 in0;
  CL_DTYPE4 weights0, weights1, weights2, weights3;
  for (int ic = 0; ic < input_c_blks; ic++) { // [0, (in_c+3)/4)
    kernel_x_0 = ic << 2;
    kernel_x_1 = kernel_x_0 + 1;
    kernel_x_2 = kernel_x_0 + 2;
    kernel_x_3 = kernel_x_0 + 3;
    for (int k_y = deal_kernel_height, idx_h = kernel_start_y; k_y >= 0; k_y -= stride_shape.y, idx_h++) {
      int in_idy = mad24(out_n_idx, input_shape.y, idx_h); // out_n_idx * input_image2d_height + idx_h, height idx of input image2d
      int in_nh_value = select(in_idy, -1, idx_h < 0 || idx_h >= input_shape.y); // height idx of input image2d
      int in_width0 = kernel_start_x;
      for (int k_x = deal_kernel_width; k_x >= 0; k_x -= stride_shape.x) {
        kernel_y = mad24(k_y, kernel_shape.x, k_x); // k_y * k_w + k_x
        kernel_y = mad24(out_c_blk_idx, kernel_size, kernel_y); // todo: tmp = out_c_blk_idx * kernel_size
        weights0 = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(kernel_x_0, kernel_y));
        weights1 = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(kernel_x_1, kernel_y));
        weights2 = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(kernel_x_2, kernel_y));
        weights3 = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, SAMPLER, (int2)(kernel_x_3, kernel_y));

        if (out_c_blk_idx == 0 && out_w_idx == 1 && out_nh_idx == 1 && ic == 0 && k_x == 0 && k_y == 0) {
        //   printf("wo=%v4hlf \n", weights0);
        //   printf("w1=%v4hlf \n", weights1);
        //   printf("w2=%v4hlf \n", weights2);
        //   printf("w3=%v4hlf \n", weights3);
        }

        int in_idx = mul24(ic, input_shape.x); // ic * input_image2d_width
        int in_width_value0 = in_width0;
        in_width_value0 =
                select(in_idx + in_width_value0, -1, (in_width_value0 < 0 || in_width_value0 >= input_shape.x));
        in0 = READ_IMG_TYPE(CL_DTYPE_CHAR, input, SAMPLER, (int2)(in_width_value0, in_nh_value));

        out0 = mad(in0.x, weights0, out0);
        out0 = mad(in0.y, weights1, out0);
        out0 = mad(in0.z, weights2, out0);
        out0 = mad(in0.w, weights3, out0);
        in_width0++;
      }
    }
  }

  out0 = activation_type4(out0);

#ifdef SCALE_ACTIVATION
  out0 = fuse_scale(out0, 1.f, 0.f, 0.f);
#endif

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output, out_pos, out0);
}

/*
__kernel void conv2d_transpose(__private const int item_ch,
                                  __private const int item_w,
                                  __private const int item_h,
                                  __read_only image2d_t input_image,
                                  __read_only image2d_t filter_image,
#if defined(BIASE_CH) || defined(BIASE_ELE)
                                  __read_only image2d_t bias,
#endif
                                  __write_only image2d_t output_image,
                                  __private const int stride,
                                  __private const int pad,
                                  __private const int dilation,
                                  __private const int in_ch,
                                  __private const int in_w,
                                  __private const int in_h,
                                  __private const int out_w,
                                  __private const int out_h,
                                  __private const int filter_w,
                                  __private const int filter_h) {

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
                              CLK_ADDRESS_CLAMP          |
                              CLK_FILTER_NEAREST;
    // item_id
    const int item_ch_id = get_global_id(0);
    const int item_w_id = get_global_id(1);
    const int item_h_id = get_global_id(2);

    // out_id
    int out_b_id = item_h_id / out_h; // 输出 batch idx
    int out_w_id_per_ch_blk = item_w_id; // 输出 width idx
    int out_h_id_per_batch = item_h_id % out_h; // 输出 height idx
    int out_w_id = item_ch_id * out_w + out_w_id_per_ch_blk; // OpenCL image width idx

    // in_id
    int in_w_id_per_ch_blk = (out_w_id_per_ch_blk + pad - filter_w + stride) / stride;
    in_w_id_per_ch_blk = in_w_id_per_ch_blk > 0 ? in_w_id_per_ch_blk : 0;
    int in_h_id_per_batch = (out_h_id_per_batch + pad - filter_h + stride) / stride;
    in_h_id_per_batch = in_h_id_per_batch > 0 ? in_h_id_per_batch : 0;

    // filter_id
    int align_w_i = out_w_id_per_ch_blk + pad - filter_w + 1;
    int align_w = align_w_i % stride > 0 ?
                  align_w_i % stride - stride : align_w_i % stride;
    int filter_w_id_per_ch_blk = out_w_id_per_ch_blk + pad < filter_w ? out_w_id_per_ch_blk + pad : filter_w + align_w - 1;

    int align_h_i = out_h_id_per_batch + pad - filter_h + 1;
    int align_h = align_h_i % stride > 0 ?
                  align_h_i % stride - stride : align_h_i % stride;
    int filter_h_id_per_out_ch = out_h_id_per_batch + pad < filter_h ? out_h_id_per_batch + pad : filter_h + align_h - 1;

#ifdef BIASE_CH
    half4 output;
    output = read_imageh(bias, sampler, (int2)(item_ch_id, 0));
#elif defined(BIASE_ELE)
    half4 output;
    output = read_imageh(bias, sampler, (int2)(out_w_id, item_h_id));
#else
    half4 output = 0.0f;
#endif
    half4 filter[4] = {0.0f};
    half4 filter_trans[4] = {0.0f};

    half4 input = 0.0f;
    for (int ch = 0; ch < (in_ch + 3) / 4; ch++) {
        int filter_w_id = ch * filter_w;
        int h_idx = 0;
        for (int h = filter_h_id_per_out_ch; h >= 0; h -= stride) {
            int in_h_id = select(in_h_id_per_batch + h_idx, -1,
                                 in_h_id_per_batch + h_idx < 0 || in_h_id_per_batch + h_idx >= in_h);
            int filter_h_id = item_ch_id * filter_h * 4 + h;
            int w_idx = 0;
            for (int w = filter_w_id_per_ch_blk; w >= 0; w -= stride) {
                int in_w_id = select(ch * in_w + in_w_id_per_ch_blk + w_idx, -1,
                                     in_w_id_per_ch_blk + w_idx < 0 || in_w_id_per_ch_blk + w_idx >= in_w);
                input = read_imageh(input_image, sampler, (int2)(in_w_id, in_h_id));
                filter[0] = read_imageh(filter_image, sampler, (int2)(filter_w_id + w, filter_h_id));                 // in_ch:0-3,out_ch:0
                filter[1] = read_imageh(filter_image, sampler, (int2)(filter_w_id + w, filter_h_id + filter_h));      // in_ch:0-3,out_ch:1
                filter[2] = read_imageh(filter_image, sampler, (int2)(filter_w_id + w, filter_h_id + 2 * filter_h));  // in_ch:0-3,out_ch:2
                filter[3] = read_imageh(filter_image, sampler, (int2)(filter_w_id + w, filter_h_id + 3 * filter_h));  // in_ch:0-3,out_ch:3

                filter_trans[0] = (half4)(filter[0].x, filter[1].x, filter[2].x, filter[3].x);             // in_ch:0,out_ch:0-3
                filter_trans[1] = (half4)(filter[0].y, filter[1].y, filter[2].y, filter[3].y);             // in_ch:1,out_ch:0-3
                filter_trans[2] = (half4)(filter[0].z, filter[1].z, filter[2].z, filter[3].z);             // in_ch:2,out_ch:0-3
                filter_trans[3] = (half4)(filter[0].w, filter[1].w, filter[2].w, filter[3].w);             // in_ch:3,out_ch:0-3

                output = mad(input.x, filter_trans[0], output);
                output = mad(input.y, filter_trans[1], output);
                output = mad(input.z, filter_trans[2], output);
                output = mad(input.w, filter_trans[3], output);
                w_idx++;
            }
            h_idx++;
        }
    }
#ifdef RELU
    output = activation(output);
#endif
    write_imageh(output_image, (int2)(out_w_id, item_h_id), output);
}
*/