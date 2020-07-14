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

__kernel void conv2d_3x3(__private const int global_size_dim0,
                         __private const int global_size_dim1,
                         __private const int global_size_dim2,
                         __read_only image2d_t input_image,
                         __read_only image2d_t filter,
                         __read_only image2d_t bias,
                         __write_only image2d_t output_image,
                         __private const int stride,
                         __private const int offset,
                         __private const int input_c,
                         __private const int dilation,
                         __private const int input_width,  /* of one block */
                         __private const int input_height, /* of one block */
                         __private const int output_width,
                         __private const int output_height,
                         __private const int output_c,
                         __private const int filter_channel,
                         __private const int filter_width,
                         __private const int filter_height,
                         __private const int group,
                         __private const int input_tensor_c

) {

  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int2 output_pos = (int2)(out_c * global_size_dim1 + out_w, out_nh);

  if (out_c >= global_size_dim0 || out_w >= global_size_dim1 ||
      out_nh >= global_size_dim2) {
    return;
  }

  int2 stride_xy;
  stride_xy.x = stride;
  stride_xy.y = stride;

  int2 ouput_pos_in_one_block;
  ouput_pos_in_one_block.x = out_w;
  ouput_pos_in_one_block.y = out_nh;

  int2 in_pos_in_one_block;
  in_pos_in_one_block.x = ouput_pos_in_one_block.x * stride + offset;
  in_pos_in_one_block.y = ouput_pos_in_one_block.y * stride + offset;

#ifdef BIASE_CH
  CL_DTYPE4 output =
      READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, (int2)(out_c, 0));
#elif defined(BIASE_ELE)
  CL_DTYPE4 output = READ_IMG_TYPE(CL_DTYPE_CHAR, bias, sampler, output_pos);
#else
  CL_DTYPE4 output = (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f);
#endif

  CL_DTYPE4 input[9];  // 3x3 region of input
  if (group == 1) {
    for (int i = 0; i < input_c; ++i) {  // each run for 3x3
      int2 pos_in = (int2)(i * input_width + in_pos_in_one_block.x,
                           in_pos_in_one_block.y);

      input[0] = select(
          READ_IMG_TYPE(CL_DTYPE_CHAR,
                        input_image,
                        sampler,
                        (int2)(pos_in.x - dilation, pos_in.y - dilation)),
          (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
          (ushort4)((in_pos_in_one_block.x - dilation < 0 ||
                     in_pos_in_one_block.y - dilation < 0 ||
                     in_pos_in_one_block.x - dilation >= input_width ||
                     in_pos_in_one_block.y - dilation >= input_height)
                    << 15));

      input[1] =
          select(READ_IMG_TYPE(CL_DTYPE_CHAR,
                               input_image,
                               sampler,
                               (int2)(pos_in.x, pos_in.y - dilation)),
                 (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
                 (ushort4)((in_pos_in_one_block.x < 0 ||
                            in_pos_in_one_block.y - dilation < 0 ||
                            in_pos_in_one_block.x >= input_width ||
                            in_pos_in_one_block.y - dilation >= input_height)
                           << 15));

      input[2] = select(
          READ_IMG_TYPE(CL_DTYPE_CHAR,
                        input_image,
                        sampler,
                        (int2)(pos_in.x + dilation, pos_in.y - dilation)),
          (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
          (ushort4)((in_pos_in_one_block.x + dilation < 0 ||
                     in_pos_in_one_block.y - dilation < 0 ||
                     in_pos_in_one_block.x + dilation >= input_width ||
                     in_pos_in_one_block.y - dilation >= input_height)
                    << 15));

      input[3] =
          select(READ_IMG_TYPE(CL_DTYPE_CHAR,
                               input_image,
                               sampler,
                               (int2)(pos_in.x - dilation, pos_in.y)),
                 (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
                 (ushort4)((in_pos_in_one_block.x - dilation < 0 ||
                            in_pos_in_one_block.y < 0 ||
                            in_pos_in_one_block.x - dilation >= input_width ||
                            in_pos_in_one_block.y >= input_height)
                           << 15));

      input[4] = select(
          READ_IMG_TYPE(
              CL_DTYPE_CHAR, input_image, sampler, (int2)(pos_in.x, pos_in.y)),
          (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
          (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y < 0 ||
                     in_pos_in_one_block.x >= input_width ||
                     in_pos_in_one_block.y >= input_height)
                    << 15));

      input[5] =
          select(READ_IMG_TYPE(CL_DTYPE_CHAR,
                               input_image,
                               sampler,
                               (int2)(pos_in.x + dilation, pos_in.y)),
                 (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
                 (ushort4)((in_pos_in_one_block.x + dilation < 0 ||
                            in_pos_in_one_block.y < 0 ||
                            in_pos_in_one_block.x + dilation >= input_width ||
                            in_pos_in_one_block.y >= input_height)
                           << 15));

      input[6] = select(
          READ_IMG_TYPE(CL_DTYPE_CHAR,
                        input_image,
                        sampler,
                        (int2)(pos_in.x - dilation, pos_in.y + dilation)),
          (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
          (ushort4)((in_pos_in_one_block.x - dilation < 0 ||
                     in_pos_in_one_block.y + dilation < 0 ||
                     in_pos_in_one_block.x - dilation >= input_width ||
                     in_pos_in_one_block.y + dilation >= input_height)
                    << 15));

      input[7] =
          select(READ_IMG_TYPE(CL_DTYPE_CHAR,
                               input_image,
                               sampler,
                               (int2)(pos_in.x, pos_in.y + dilation)),
                 (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
                 (ushort4)((in_pos_in_one_block.x < 0 ||
                            in_pos_in_one_block.y + dilation < 0 ||
                            in_pos_in_one_block.x >= input_width ||
                            in_pos_in_one_block.y + dilation >= input_height)
                           << 15));

      input[8] = select(
          READ_IMG_TYPE(CL_DTYPE_CHAR,
                        input_image,
                        sampler,
                        (int2)(pos_in.x + dilation, pos_in.y + dilation)),
          (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
          (ushort4)((in_pos_in_one_block.x + dilation < 0 ||
                     in_pos_in_one_block.y + dilation < 0 ||
                     in_pos_in_one_block.x + dilation >= input_width ||
                     in_pos_in_one_block.y + dilation >= input_height)
                    << 15));

      if (i == input_c - 1) {
        int c_shr = input_tensor_c % 4;
        if (c_shr == 1) {
          for (int k = 0; k < 9; k++) {
            input[k].y = (half)0.f;
            input[k].z = (half)0.f;
            input[k].w = (half)0.f;
          }
        } else if (c_shr == 2) {
          for (int k = 0; k < 9; k++) {
            input[k].z = (half)0.f;
            input[k].w = (half)0.f;
          }
        } else if (c_shr == 3) {
          for (int k = 0; k < 9; k++) {
            input[k].w = (half)0.f;
          }
        } else if (c_shr == 0) {
        }
      }

      int j = 0;
      int2 pos_of_weight;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      CL_DTYPE4 weight_x =
          READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y += 3;
      CL_DTYPE4 weight_y =
          READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y += 3;
      CL_DTYPE4 weight_z =
          READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y += 3;
      CL_DTYPE4 weight_w =
          READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 1;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 2;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 3;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 4;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 5;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 6;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 7;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);

      j = 8;
      pos_of_weight.x = i * 3 + j % 3;
      pos_of_weight.y = out_c * 4 * 3 + 0 * 3 + j / 3;
      weight_x = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.x += dot(input[j], weight_x);

      pos_of_weight.y = out_c * 4 * 3 + 1 * 3 + j / 3;
      weight_y = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.y += dot(input[j], weight_y);

      pos_of_weight.y = out_c * 4 * 3 + 2 * 3 + j / 3;
      weight_z = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.z += dot(input[j], weight_z);

      pos_of_weight.y = out_c * 4 * 3 + 3 * 3 + j / 3;
      weight_w = READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);
      output.w += dot(input[j], weight_w);
    }
  } else {  // group != 1
    for (int i = 0; i < 4; i++) {
      int used_input_channel_num =
          (out_c * 4 + i) / (output_c / group) * filter_channel;
      for (int f_c = 0; f_c < filter_channel; ++f_c) {
        int input_c = used_input_channel_num + f_c;
        int input_block = input_c / 4;
        int2 pos_in = (int2)(input_block * input_width + in_pos_in_one_block.x,
                             in_pos_in_one_block.y);
        input[0] = select(
            READ_IMG_TYPE(CL_DTYPE_CHAR,
                          input_image,
                          sampler,
                          (int2)(pos_in.x - dilation, pos_in.y - dilation)),
            (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
            (ushort4)((in_pos_in_one_block.x - dilation < 0 ||
                       in_pos_in_one_block.y - dilation < 0 ||
                       in_pos_in_one_block.x - dilation >= input_width ||
                       in_pos_in_one_block.y - dilation >= input_height)
                      << 15));
        input[1] =
            select(READ_IMG_TYPE(CL_DTYPE_CHAR,
                                 input_image,
                                 sampler,
                                 (int2)(pos_in.x, pos_in.y - dilation)),
                   (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
                   (ushort4)((in_pos_in_one_block.x < 0 ||
                              in_pos_in_one_block.y - dilation < 0 ||
                              in_pos_in_one_block.x >= input_width ||
                              in_pos_in_one_block.y - dilation >= input_height)
                             << 15));
        input[2] = select(
            READ_IMG_TYPE(CL_DTYPE_CHAR,
                          input_image,
                          sampler,
                          (int2)(pos_in.x + dilation, pos_in.y - dilation)),
            (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
            (ushort4)((in_pos_in_one_block.x + dilation < 0 ||
                       in_pos_in_one_block.y - dilation < 0 ||
                       in_pos_in_one_block.x + dilation >= input_width ||
                       in_pos_in_one_block.y - dilation >= input_height)
                      << 15));
        input[3] =
            select(READ_IMG_TYPE(CL_DTYPE_CHAR,
                                 input_image,
                                 sampler,
                                 (int2)(pos_in.x - dilation, pos_in.y)),
                   (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
                   (ushort4)((in_pos_in_one_block.x - dilation < 0 ||
                              in_pos_in_one_block.y < 0 ||
                              in_pos_in_one_block.x - dilation >= input_width ||
                              in_pos_in_one_block.y >= input_height)
                             << 15));
        input[4] = select(
            READ_IMG_TYPE(CL_DTYPE_CHAR,
                          input_image,
                          sampler,
                          (int2)(pos_in.x, pos_in.y)),
            (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
            (ushort4)((in_pos_in_one_block.x < 0 || in_pos_in_one_block.y < 0 ||
                       in_pos_in_one_block.x >= input_width ||
                       in_pos_in_one_block.y >= input_height)
                      << 15));
        input[5] =
            select(READ_IMG_TYPE(CL_DTYPE_CHAR,
                                 input_image,
                                 sampler,
                                 (int2)(pos_in.x + dilation, pos_in.y)),
                   (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
                   (ushort4)((in_pos_in_one_block.x + dilation < 0 ||
                              in_pos_in_one_block.y < 0 ||
                              in_pos_in_one_block.x + dilation >= input_width ||
                              in_pos_in_one_block.y >= input_height)
                             << 15));
        input[6] = select(
            READ_IMG_TYPE(CL_DTYPE_CHAR,
                          input_image,
                          sampler,
                          (int2)(pos_in.x - dilation, pos_in.y + dilation)),
            (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
            (ushort4)((in_pos_in_one_block.x - dilation < 0 ||
                       in_pos_in_one_block.y + dilation < 0 ||
                       in_pos_in_one_block.x - dilation >= input_width ||
                       in_pos_in_one_block.y + dilation >= input_height)
                      << 15));
        input[7] =
            select(READ_IMG_TYPE(CL_DTYPE_CHAR,
                                 input_image,
                                 sampler,
                                 (int2)(pos_in.x, pos_in.y + dilation)),
                   (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
                   (ushort4)((in_pos_in_one_block.x < 0 ||
                              in_pos_in_one_block.y + dilation < 0 ||
                              in_pos_in_one_block.x >= input_width ||
                              in_pos_in_one_block.y + dilation >= input_height)
                             << 15));
        input[8] = select(
            READ_IMG_TYPE(CL_DTYPE_CHAR,
                          input_image,
                          sampler,
                          (int2)(pos_in.x + dilation, pos_in.y + dilation)),
            (CL_DTYPE4)(0.0f, 0.0f, 0.0f, 0.0f),
            (ushort4)((in_pos_in_one_block.x + dilation < 0 ||
                       in_pos_in_one_block.y + dilation < 0 ||
                       in_pos_in_one_block.x + dilation >= input_width ||
                       in_pos_in_one_block.y + dilation >= input_height)
                      << 15));

        CL_DTYPE tmp_out = 0;
        for (int j = 0; j < 9; j++) {
          int2 pos_of_weight;
          pos_of_weight.x = (f_c / 4) * 3 + j % 3;
          pos_of_weight.y = out_c * 4 * 3 + i * 3 + j / 3;
          CL_DTYPE4 weight =
              READ_IMG_TYPE(CL_DTYPE_CHAR, filter, sampler, pos_of_weight);

          int f_c_offset = f_c % 4;
          CL_DTYPE f_value;
          if (f_c_offset == 0) {
            f_value = weight.x;
          } else if (f_c_offset == 1) {
            f_value = weight.y;
          } else if (f_c_offset == 2) {
            f_value = weight.z;
          } else if (f_c_offset == 3) {
            f_value = weight.w;
          }

          int input_c_offset = input_c % 4;
          CL_DTYPE input_value;
          if (input_c_offset == 0) {
            input_value = input[j].x;
          } else if (input_c_offset == 1) {
            input_value = input[j].y;
          } else if (input_c_offset == 2) {
            input_value = input[j].z;
          } else if (input_c_offset == 3) {
            input_value = input[j].w;
          }
          tmp_out += f_value * input_value;
        }

        if (i == 0) {
          output.x += tmp_out;
        } else if (i == 1) {
          output.y += tmp_out;
        } else if (i == 2) {
          output.z += tmp_out;
        } else if (i == 3) {
          output.w += tmp_out;
        }
      }
    }
  }

  output = activation_type4(output);

  WRITE_IMG_TYPE(CL_DTYPE_CHAR, output_image, output_pos, output);
}
