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

__kernel void depth_conv2d(__private const int global_size_dim0,
                           __private const int global_size_dim1,
                           __private const int global_size_dim2,
                           __read_only image2d_t input,
                           __read_only image2d_t filter,
                           __read_only image2d_t bias,
#ifdef BATCH_NORM
                           __read_only image2d_t new_scale,
                           __read_only image2d_t new_biase,
#endif
                           __write_only image2d_t output_image,
                           __private const int stride,
                           __private const int offset,
                           __private const int input_c,
                           __private const int dilation,
                           __private const int input_width,  /* of one block */
                           __private const int input_height, /* of one block */
                           __private const int output_width,
                           __private const int output_height,
                           __private const int filter_width,
                           __private const int filter_height) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  int2 output_pos = (int2)(out_c * global_size_dim1 + out_w, out_nh);
  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  const int batch_index = out_nh / output_height;
  const int out_nh_in_one_batch = out_nh % output_height;
  int2 stride_xy = (int2)(stride, stride);
  int2 ouput_pos_in_one_block = (int2)(out_w, out_nh_in_one_batch);
  int2 in_pos_in_one_block =
      ouput_pos_in_one_block * stride_xy + (int2)(offset, offset);
#ifdef BIASE_CH
  CL_DTYPE4 output =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, (int2)(out_c, 0));
#elif defined(BIASE_ELE)
  CL_DTYPE4 output = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, output_pos);
#else
  CL_DTYPE4 output = 0.0f;
#endif

  int2 pos_in_input_block =
      (int2)(out_c * input_width, batch_index * input_height);
  int2 pos_in_filter_block =
      (int2)(out_c * filter_width, batch_index * filter_height);
  int filter_x = pos_in_filter_block.x;
  int filter_y = pos_in_filter_block.y;
  int input_x_base = pos_in_input_block.x + in_pos_in_one_block.x;
  int input_y_base = pos_in_input_block.y + in_pos_in_one_block.y;
  int2 align = {filter_width / 2, filter_height / 2};
  for (int fy = 0; fy < filter_height; ++fy) {
    for (int fx = 0; fx < filter_width; ++fx) {
      int x_off = fx - align.x;
      int y_off = fy - align.y;
      CL_DTYPE4 in = SELECT(
          READ_IMG_TYPE(CL_DTYPE_CHAR,
                        input,
                        sampler,
                        (int2)(input_x_base + x_off, input_y_base + y_off)),
          (CL_DTYPE4)(0.0f),
          ((in_pos_in_one_block.x + x_off < 0 ||
                     in_pos_in_one_block.y + y_off < 0 ||
                     in_pos_in_one_block.x + x_off >= input_width ||
                     in_pos_in_one_block.y + y_off >= input_height)
                    ));
      CL_DTYPE4 f = READ_IMG_TYPE(
          CL_DTYPE_CHAR, filter, sampler, (int2)(filter_x + fx, filter_y + fy));
      output += in * f;
    }
  }
#ifdef BATCH_NORM
  output = output * READ_IMG_TYPE(
                        CL_DTYPE_CHAR, new_scale, sampler, (int2)(out_c, 0)) +
           READ_IMG_TYPE(CL_DTYPE_CHAR, new_biase, sampler, (int2)(out_c, 0));
#endif

  output = activation_type4(output);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output);
}
